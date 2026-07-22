"""Loss 优化策略（Part4K1 Task 6.7）。

参考 GPT_teacher-3.37M-cn（不停压低 Loss）的实践经验，封装一组 loss 优化
策略，可集成到 :class:`verse_torch.training.Trainer` 训练循环：

- **梯度裁剪**（grad_clip）：已有，``Trainer`` 内置。
- **LR warmup + cosine + ReduceLROnPlateau 组合**：已有 scheduler，
  :class:`LossOptimizer` 提供组合调用入口。
- **loss plateau 重走** :meth:`LossOptimizer.maybe_rollback`：
  - 检测：连续 ``patience`` 步 val_loss 未下降
  - 触发：回退 best_state_dict + LR × 0.3 + 重置 Adam 动量（m/v 清零）+ 继续
- **NaN/Inf 检测** :meth:`LossOptimizer.check_loss_finite`：
  loss 为 NaN/Inf 时跳过该 batch（不更新参数），记录警告。

设计目标
========
1. **零侵入**：``LossOptimizer`` 不接管训练循环，仅作为"钩子集合"，
   由 ``Trainer`` 在合适时机调用对应方法。
2. **可独立使用**：``LossOptimizer`` 不依赖任何具体 Trainer 类，
   只需要 ``model`` / ``optimizer`` / ``val_loss 历史``。
3. **CPU-first**：与 VerseTorch 保持一致，无 GPU 依赖。
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _is_finite(x) -> bool:
    """判断标量是否有限（非 NaN / 非 Inf）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def reset_adam_momentum(optimizer) -> int:
    """重置 AdamW / NAdamW 优化器的一阶/二阶动量（m/v 清零）。

    用于 loss plateau 重走：清空历史动量后，优化器从当前梯度重新累积
    动量，避免旧动量把参数拉回 plateau 区间。

    Args:
        optimizer: ``verse_torch.optim.AdamW`` / ``NAdamW`` 或兼容优化器
    Returns:
        重置的参数张量数（成功重置 m/v 的参数个数）；-1 表示优化器无 state
    """
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict) or not state:
        return -1
    n_reset = 0
    for key, st in state.items():
        if not isinstance(st, dict):
            continue
        changed = False
        for buf_name in ("m", "v"):
            buf = st.get(buf_name)
            if buf is None:
                continue
            # NumPy 路径
            if isinstance(buf, np.ndarray):
                st[buf_name] = np.zeros_like(buf)
                changed = True
            else:
                # torch 路径（GPU 后端）：尝试 zeros_like
                try:
                    import torch as _torch
                    if isinstance(buf, _torch.Tensor):
                        st[buf_name] = _torch.zeros_like(buf)
                        changed = True
                except Exception:
                    pass
        if changed:
            n_reset += 1
    # 重置 Adam 步数计数器 t（让 bias correction 从头开始）
    if hasattr(optimizer, "t"):
        try:
            optimizer.t = 0
        except Exception:
            pass
    return n_reset


def scale_optimizer_lr(optimizer, factor: float) -> float:
    """把优化器所有 param_group 的 lr 乘以 ``factor``，返回缩放后的最大 lr。

    Args:
        optimizer: 任意 ``verse_torch.optim.Optimizer``
        factor: 缩放系数（0 < factor < 1 表示衰减）
    Returns:
        缩放后所有 param_group 中的最大 lr
    """
    factor = float(factor)
    max_lr = 0.0
    for group in optimizer.param_groups:
        old_lr = float(group.get("lr", 0.0))
        new_lr = old_lr * factor
        group["lr"] = new_lr
        max_lr = max(max_lr, new_lr)
    # 同步 optimizer.lr 顶层属性（verse_torch Optimizer 兼容）
    if hasattr(optimizer, "lr"):
        try:
            optimizer.lr = float(getattr(optimizer, "lr", 0.0)) * factor
            max_lr = max(max_lr, optimizer.lr)
        except Exception:
            pass
    return max_lr


# ---------------------------------------------------------------------------
# LossOptimizer：loss 优化策略集合
# ---------------------------------------------------------------------------


class LossOptimizer:
    """Loss 优化策略集合（plateau 重走 + NaN/Inf 跳过 + LR 衰减组合）。

    使用方式（在 Trainer 训练循环中）::

        loss_opt = LossOptimizer(model, optimizer, patience=5, rollback_factor=0.3)

        for step in range(max_steps):
            loss = compute_loss(batch)
            # 1. NaN/Inf 检测
            if not loss_opt.check_loss_finite(loss, step):
                continue  # 跳过该 batch
            loss.backward()
            optimizer.step()
            if step % eval_interval == 0:
                val_loss = evaluate()
                # 2. plateau 重走检测
                loss_opt.maybe_rollback(val_loss, step, best_state_dict)

    Args:
        model: 训练模型（需实现 ``state_dict`` / ``load_state_dict`` / ``parameters``）
        optimizer: 优化器（需有 ``state`` / ``param_groups``）
        patience: val_loss 连续多少步未下降触发 plateau 重走（默认 5）
        rollback_factor: 触发重走时 LR 的缩放系数（默认 0.3）
        min_lr: LR 衰减下界（默认 1e-7，低于此值不再衰减）
        min_delta: val_loss 改善阈值（默认 1e-4，相对改善量）
        max_rollbacks: 整个训练过程允许的最大重走次数（默认 3，
            避免无限重走）
        verbose: 是否打印调试信息（默认 True）
    """

    def __init__(
        self,
        model,
        optimizer,
        patience: int = 5,
        rollback_factor: float = 0.3,
        min_lr: float = 1e-7,
        min_delta: float = 1e-4,
        max_rollbacks: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.patience = int(patience)
        self.rollback_factor = float(rollback_factor)
        self.min_lr = float(min_lr)
        self.min_delta = float(min_delta)
        self.max_rollbacks = int(max_rollbacks)
        self.verbose = bool(verbose)

        # plateau 追踪
        self._best_val_loss: float = float("inf")
        self._bad_epochs: int = 0
        self._rollback_count: int = 0
        # NaN/Inf 跳过统计
        self._nan_skip_count: int = 0
        # val_loss 历史（供外部诊断）
        self.val_loss_history: List[float] = []
        # rollback 事件历史（供测试验证）
        self.rollback_history: List[dict] = []

    # ------------------------------------------------------------------
    # NaN/Inf 检测
    # ------------------------------------------------------------------

    def check_loss_finite(self, loss, step: int = -1) -> bool:
        """检测 loss 是否有限；非有限时记录警告并返回 False。

        调用方应在 ``loss.backward()`` 之前调用本方法；返回 False 时
        应 ``continue`` 跳过该 batch（不更新参数）。

        Args:
            loss: 标量 loss（Tensor / float / np.ndarray 均可）
            step: 当前训练步数（仅用于日志）
        Returns:
            True 表示 loss 有限，可继续 backward；False 表示 NaN/Inf，应跳过
        """
        if _is_finite(loss):
            return True
        self._nan_skip_count += 1
        try:
            v = float(loss)
        except Exception:
            v = float("nan")
        if self.verbose:
            print(
                f"[LossOptimizer] step={step} loss={v!r} 非有限，跳过该 batch "
                f"(累计跳过 {self._nan_skip_count} 次)",
                flush=True,
            )
        return False

    @property
    def nan_skip_count(self) -> int:
        """累计跳过的 NaN/Inf batch 数。"""
        return self._nan_skip_count

    # ------------------------------------------------------------------
    # loss plateau 重走
    # ------------------------------------------------------------------

    def maybe_rollback(
        self,
        val_loss: float,
        step: int = -1,
        best_state_dict: Optional[dict] = None,
    ) -> bool:
        """检测 loss plateau 并在触发时执行重走。

        触发条件：``val_loss`` 连续 ``patience`` 次未显著下降
        （``improved = val_loss < best * (1 - min_delta)``）。

        触发动作：
        1. 回退到 ``best_state_dict``（若提供）
        2. LR × ``rollback_factor``
        3. 重置 Adam 动量（m/v 清零）
        4. 重置 ``_bad_epochs`` 计数

        Args:
            val_loss: 当前步的 val_loss
            step: 当前训练步数（仅用于日志）
            best_state_dict: 最佳模型 state_dict；None 时尝试从 model 读取
        Returns:
            True 表示触发了重走；False 表示未触发
        """
        self.val_loss_history.append(float(val_loss))

        # 改善判定
        if val_loss < self._best_val_loss * (1.0 - self.min_delta):
            self._best_val_loss = float(val_loss)
            self._bad_epochs = 0
            return False

        self._bad_epochs += 1
        if self._bad_epochs < self.patience:
            return False
        if self._rollback_count >= self.max_rollbacks:
            if self.verbose and self._bad_epochs == self.patience:
                print(
                    f"[LossOptimizer] step={step} 达到 plateau 但已超过 "
                    f"max_rollbacks={self.max_rollbacks}，不再重走",
                    flush=True,
                )
            # 重置计数避免重复打印
            self._bad_epochs = 0
            return False

        # 触发重走
        self._rollback_count += 1
        rollback_id = self._rollback_count

        # 1. 回退 best_state_dict
        sd = best_state_dict
        if sd is None and hasattr(self.model, "state_dict"):
            # 没有外部传入 best，用当前 model state 兜底
            sd = self.model.state_dict()
        if sd is not None and hasattr(self.model, "load_state_dict"):
            try:
                self.model.load_state_dict(copy.deepcopy(sd))
                if self.verbose:
                    print(
                        f"[LossOptimizer] step={step} plateau 重走 #{rollback_id}："
                        f"已回退到 best_state_dict",
                        flush=True,
                    )
            except Exception as e:
                if self.verbose:
                    print(
                        f"[LossOptimizer] step={step} 回退 best_state_dict 失败：{e}",
                        flush=True,
                    )

        # 2. LR × rollback_factor
        new_lr = scale_optimizer_lr(self.optimizer, self.rollback_factor)
        if new_lr < self.min_lr:
            # 抬回 min_lr，避免完全停滞
            for group in self.optimizer.param_groups:
                group["lr"] = self.min_lr
            if hasattr(self.optimizer, "lr"):
                self.optimizer.lr = self.min_lr
            new_lr = self.min_lr

        # 3. 重置 Adam 动量
        n_reset = reset_adam_momentum(self.optimizer)

        # 4. 重置计数
        self._bad_epochs = 0

        if self.verbose:
            print(
                f"[LossOptimizer] step={step} plateau 重走 #{rollback_id} 完成："
                f"LR={new_lr:.6e}，重置 {n_reset} 个参数的 m/v 动量",
                flush=True,
            )

        self.rollback_history.append({
            "step": int(step),
            "rollback_id": rollback_id,
            "val_loss": float(val_loss),
            "best_val_loss": float(self._best_val_loss),
            "new_lr": float(new_lr),
            "n_params_reset": int(n_reset),
        })
        return True

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def best_val_loss(self) -> float:
        """历史最佳 val_loss。"""
        return self._best_val_loss

    @property
    def rollback_count(self) -> int:
        """已触发的重走次数。"""
        return self._rollback_count

    def reset(self) -> None:
        """重置内部状态（开始新一段训练时调用）。"""
        self._best_val_loss = float("inf")
        self._bad_epochs = 0
        self._rollback_count = 0
        self._nan_skip_count = 0
        self.val_loss_history = []
        self.rollback_history = []


# ---------------------------------------------------------------------------
# _rollback_and_perturb：独立函数版本（便于测试单独调用）
# ---------------------------------------------------------------------------


def _rollback_and_perturb(
    trainer,
    optimizer,
    best_state_dict: dict,
    lr_factor: float = 0.3,
) -> int:
    """回退 best_state_dict + LR × factor + 重置 Adam 动量。

    独立函数版本，不依赖 :class:`LossOptimizer` 实例，便于在自定义训练
    循环中单独调用，或测试时直接验证三步动作。

    Args:
        trainer: 训练器（或模型），需实现 ``load_state_dict``
        optimizer: 优化器
        best_state_dict: 要回退到的 state dict
        lr_factor: LR 缩放系数（默认 0.3）
    Returns:
        重置的参数张量数（``reset_adam_momentum`` 返回值）
    """
    model = getattr(trainer, "model", trainer)
    if model is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(copy.deepcopy(best_state_dict))
    new_lr = scale_optimizer_lr(optimizer, lr_factor)
    n_reset = reset_adam_momentum(optimizer)
    print(
        f"[_rollback_and_perturb] 已回退 best_state_dict + LR×{lr_factor} "
        f"= {new_lr:.6e}，重置 {n_reset} 个参数的 m/v",
        flush=True,
    )
    return n_reset


__all__ = [
    "LossOptimizer",
    "reset_adam_momentum",
    "scale_optimizer_lr",
    "_rollback_and_perturb",
]
