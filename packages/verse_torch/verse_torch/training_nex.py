"""VerseNex 训练工具链（Part4）。

本模块在 :mod:`verse_torch.training` 的基础上提供针对 VerseNex 原生架构
（``CometSparkNexLM`` / ``CometSparkLM(arch="verse_nex")``）的训练器：

- :class:`VerseNexTrainer`：aux_loss-aware 预训练 / 续训训练器
  - 自动检测模型是否提供 ``forward_with_aux``，若是则把 MoD aux_loss
    合并入总 loss（``loss = cross_entropy + aux_loss_weight * aux``）。
  - 兼容标准 ``model(x) -> logits`` 接口（transformer / hybrid 架构同样可用）。
- :class:`LoRATrainer`：LoRA-aware 训练器
  - 自动调用 :func:`verse_torch.compress.lora_only` 包装模型，
    仅训练 A/B 矩阵，base 全部冻结。
  - 训练结束后支持 :meth:`merge_lora` 把 ΔW 合并回 base。
- :class:`SFTTrainer`：监督微调训练器
  - 支持 chat 数组数据格式 ``{"messages": [{"role","content"}, ...]}``
  - 仅对 assistant 回复 token 计算 loss（user/system token 被 ``ignore_index`` 屏蔽）
- :class:`DPOTrainer`：Direct Preference Optimization 训练器
  - 偏好对数据 ``{"prompt","chosen","rejected"}``
  - 冻结 reference model 计算 log_prob，DPO loss = -log σ(β·(Δchosen - Δrejected))

设计目标：
1. **零侵入**：不修改现有 :class:`verse_torch.training.Trainer`，
   保持 transformer / hybrid 架构训练路径完全不变。
2. **可组合**：所有训练器复用 :class:`EarlyStopping` /
   :class:`CheckpointManager` / :func:`clip_grad_norm` / :func:`plot_loss_curve`
   等成熟组件。
3. **CPU-first**：与 VerseTorch 保持一致，无 GPU 依赖。

仅依赖 NumPy + Python 标准库 + VerseTorch 现有组件。
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np

from .tensor import Tensor, no_grad
from .optim import Optimizer, AdamW
from .losses import cross_entropy
from .training import (
    EarlyStopping,
    GradientAccumulator,
    CheckpointManager,
    clip_grad_norm,
    cross_entropy_loss,
    plot_loss_curve,
    _as_tensor,
    _scalar,
    _cfg_get,
    _format_eta,
    _NoOpPBar,
)
from .device import empty_cache, is_cpu_device

try:  # tqdm 可选依赖
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    _tqdm = None


# ---------------------------------------------------------------------------
# 辅助：检测模型是否支持 forward_with_aux
# ---------------------------------------------------------------------------


def _model_has_aux(model) -> bool:
    """检测模型是否提供 ``forward_with_aux`` 方法。

    支持两种层级：
    - 模型本身实现：``model.forward_with_aux(x) -> (logits, aux)``
    - 内部 net 实现：``model.net.forward_with_aux(x) -> (logits, aux)``
      （典型场景：``CometSparkLM`` 包装了 ``CometSparkNexLM``）
    """
    if hasattr(model, "forward_with_aux") and callable(getattr(model, "forward_with_aux")):
        return True
    net = getattr(model, "net", None)
    if net is not None and hasattr(net, "forward_with_aux") and callable(getattr(net, "forward_with_aux")):
        return True
    return False


def _count_mod_layers(model) -> int:
    """统计模型中实际的 MoD 层数（Part5K1.1：MoD 显示稳定性修复）。

    优先从 ``layer_pattern`` 统计 ``"mod"`` 条目；其次遍历 ``blocks`` 的
    ``layer_kind``；再退化为遍历子模块统计 ``MoDLayer`` 实例。返回 0 表示
    该模型无 MoD 层（无论 ``forward_with_aux`` 是否存在）。

    这解决了旧版"声明启用 aux_loss 路径但日志无 aux="的语义不一致：``_model_has_aux``
    只检查方法是否存在（CometSparkNexLM 永远有该方法），不区分有无 MoD 层。
    """
    # 1. layer_pattern（CometSparkNexLM / CometSparkV05LM.net 上）
    for obj in (model, getattr(model, "net", None)):
        if obj is None:
            continue
        pattern = getattr(obj, "layer_pattern", None)
        if pattern:
            try:
                return int(sum(1 for k in pattern if str(k) == "mod"))
            except Exception:
                pass
    # 2. blocks 的 layer_kind
    for obj in (model, getattr(model, "net", None)):
        if obj is None:
            continue
        blocks = getattr(obj, "blocks", None)
        if blocks is not None:
            try:
                cnt = sum(
                    1 for b in blocks if getattr(b, "layer_kind", "") == "mod"
                )
                if cnt > 0:
                    return int(cnt)
                # blocks 存在但无 mod，返回 0（明确无 MoD）
                if hasattr(obj, "layer_pattern") or hasattr(obj, "blocks"):
                    return 0
            except Exception:
                pass
    # 3. 退化：遍历子模块统计 MoDLayer 实例
    try:
        named_modules = getattr(model, "named_modules", None)
        if callable(named_modules):
            cnt = 0
            for _, m in named_modules():
                if type(m).__name__ == "MoDLayer":
                    cnt += 1
            return cnt
    except Exception:
        pass
    return 0


def _call_forward_with_aux(model, x) -> tuple:
    """调用模型的 ``forward_with_aux``，自动选择 model 或 model.net 层级。

    返回 ``(logits, aux_loss)``。
    """
    if hasattr(model, "forward_with_aux") and callable(getattr(model, "forward_with_aux")):
        return model.forward_with_aux(x)
    # 回退到 model.net
    return model.net.forward_with_aux(x)


def _get_aux_loss_weight(model, default: float = 0.01) -> float:
    """从模型读取 aux_loss_weight。

    优先从 ``model.config.aux_loss_weight`` 读取（CometSparkLM 路径）；
    否则从 ``model.net.aux_loss_weight`` 读取（CometSparkNexLM 路径）；
    都没有则返回 default。
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "aux_loss_weight"):
        return float(getattr(cfg, "aux_loss_weight"))
    net = getattr(model, "net", None)
    if net is not None and hasattr(net, "aux_loss_weight"):
        return float(getattr(net, "aux_loss_weight"))
    return float(default)


# ---------------------------------------------------------------------------
# VerseNexTrainer：aux_loss-aware 预训练 / 续训训练器
# ---------------------------------------------------------------------------


class VerseNexTrainer:
    """VerseNex 原生架构训练器（aux_loss-aware）。

    与 :class:`verse_torch.training.Trainer` 的关键区别：
    - 当模型提供 ``forward_with_aux`` 时，loss = ``cross_entropy(logits, y)
      + aux_loss_weight * aux``；否则回退到标准 ``cross_entropy`` 路径。
    - evaluate() 同样使用 forward_with_aux 计算 val loss（aux 不计入 val loss，
      仅用于训练时的负载均衡）。

    兼容性：
    - 接受任意 ``nn.Module``（包括 ``CometSparkLM`` / ``CometSparkNexLM`` /
      ``TransformerLM``）。无 forward_with_aux 时退化为标准训练器。
    - 配置项与 :class:`Trainer` 完全一致（max_steps / eval_interval / patience /
      save_dir / grad_accum / grad_clip / label_smoothing / 等）。

    Args:
        model: ``nn.Module``，实现 ``forward(x) -> logits`` 或
            ``forward_with_aux(x) -> (logits, aux)``
        train_loader: 可迭代对象，每次返回 ``(x, y)``
        val_loader: 可迭代对象，每次返回 ``(x, y)``
        optimizer: ``optim.Optimizer`` 实例
        scheduler: 可选的学习率调度器
        cfg: dict 或 dataclass，配置项同 :class:`Trainer`，额外支持：
            - aux_loss_weight: float，覆盖模型默认的 aux_loss 权重
              （None 表示用模型自带的 aux_loss_weight）
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer: Optimizer,
        scheduler=None,
        cfg=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg if cfg is not None else {}

        self.max_steps = int(_cfg_get(cfg, "max_steps", 100))
        self.eval_interval = int(_cfg_get(cfg, "eval_interval", 10))
        self.patience = int(_cfg_get(cfg, "patience", 10))
        self.save_dir = str(_cfg_get(cfg, "save_dir", "./checkpoints"))
        self.grad_accum_n = max(1, int(_cfg_get(cfg, "grad_accum", 1)))
        self.log_interval = int(_cfg_get(cfg, "log_interval", 10))
        self.loss_rate_window = int(_cfg_get(cfg, "loss_rate_window", 50))

        # 精度优化：梯度裁剪 / 标签平滑
        self.grad_clip = float(_cfg_get(cfg, "grad_clip", 0.0))
        self.label_smoothing = float(_cfg_get(cfg, "label_smoothing", 0.0))
        # 训练 UX
        self.enable_progress_bar = bool(_cfg_get(cfg, "enable_progress_bar", True))
        self.realtime_plot = bool(_cfg_get(cfg, "realtime_plot", True))
        self.eta_window = int(_cfg_get(cfg, "eta_window", 20))
        # Part4K2 Task 7.2: 输出控制
        self.quiet = bool(_cfg_get(cfg, "quiet", False))
        self.verbose = bool(_cfg_get(cfg, "verbose", False))
        # Part4K2 Task 7.4: 1B 模型 GPU 显存定期清理（CPU 时 no-op，默认 0=关闭）
        self.empty_cache_interval = int(_cfg_get(cfg, "empty_cache_interval", 0))
        # 设备字符串（用于 empty_cache；默认 cpu）
        self.device = str(_cfg_get(cfg, "device", "cpu"))

        # aux_loss 配置：None 表示用模型自带的 aux_loss_weight
        aux_w = _cfg_get(cfg, "aux_loss_weight", None)
        if aux_w is None:
            self.aux_loss_weight = _get_aux_loss_weight(model, default=0.01)
        else:
            self.aux_loss_weight = float(aux_w)

        # 子控制器
        self.early_stopping = EarlyStopping(self.patience)
        self.grad_accum = GradientAccumulator(
            micro_batch=1, effective_batch=self.grad_accum_n
        )
        self.checkpoint = CheckpointManager(self.save_dir)

        # 是否启用 aux 路径
        self.use_aux = _model_has_aux(model)
        # Part5K1.1：一次性探测实际 MoD 层数，用于确定性地控制 aux 显示
        # （旧版用运行时 aux_val>0 判断，NaN 或 0 时会隐藏 aux，造成
        # "有时显示有 MoD，有时无 MoD"的假象；改为配置驱动，显示稳定）
        self._n_mod_layers = _count_mod_layers(model)
        self._has_mod_layers = self._n_mod_layers > 0
        if self.use_aux:
            if self._has_mod_layers:
                print(
                    f"[VerseNexTrainer] 检测到 forward_with_aux + {self._n_mod_layers} 个 "
                    f"MoD 层，启用 aux_loss 路径 (aux_loss_weight={self.aux_loss_weight})",
                    flush=True,
                )
            else:
                # 有 forward_with_aux 方法但无 MoD 层：aux 恒为 0，不显示 aux
                print(
                    "[VerseNexTrainer] 检测到 forward_with_aux，但模型无 MoD 层"
                    "（layer_pattern 全为 trisparse），aux_loss 恒为 0，"
                    "训练日志不显示 aux（这是预期行为，非 bug）",
                    flush=True,
                )
        else:
            print(
                "[VerseNexTrainer] 模型未提供 forward_with_aux，"
                "退化为标准 cross_entropy 训练路径",
                flush=True,
            )

        # 训练历史
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.aux_losses: list[float] = []  # 仅 use_aux 时填充
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # loss 计算
    # ------------------------------------------------------------------

    def _compute_loss(self, x: Tensor, y: Tensor) -> tuple:
        """前向 + loss 计算，返回 ``(total_loss, ce_loss, aux_loss)``。

        - use_aux=True：调用 forward_with_aux，total = ce + aux_w * aux
        - use_aux=False：调用 forward，total = ce，aux = 0
        """
        if self.use_aux:
            logits, aux = _call_forward_with_aux(self.model, x)
            ce = cross_entropy(logits, y, label_smoothing=self.label_smoothing)
            if isinstance(aux, Tensor):
                aux_scalar_loss = aux * self.aux_loss_weight
                total = ce + aux_scalar_loss
                aux_val = float(aux.data.item()) if aux.data.ndim == 0 else float(aux.data.sum())
            else:
                # aux 可能是 0（无 MoD 层）或数值
                aux_val = float(aux)
                aux_scalar_loss = Tensor(np.array(aux_val * self.aux_loss_weight,
                                                   dtype=np.float32))
                total = ce + aux_scalar_loss
            return total, float(_scalar(ce)), aux_val
        # 标准路径
        logits = self.model(x)
        ce = cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        return ce, float(_scalar(ce)), 0.0

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        """在 val_loader 上计算平均 cross_entropy loss（不含 aux）。"""
        total_loss = 0.0
        n_batches = 0
        with no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                x, y = batch
                x = _as_tensor(x)
                y = _as_tensor(y)
                if self.use_aux:
                    logits, _ = _call_forward_with_aux(self.model, x)
                else:
                    logits = self.model(x)
                loss = cross_entropy(logits, y)
                total_loss += _scalar(loss)
                n_batches += 1
        if n_batches == 0:
            return float("nan")
        return total_loss / n_batches

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _make_state(self, step: int, val_loss: float) -> dict:
        """构造保存到 checkpoint 的 state 字典。"""
        return {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "val_loss": float(val_loss),
            "train_loss": float(self.train_losses[-1]) if self.train_losses else float("nan"),
            "aux_loss": float(self.aux_losses[-1]) if self.aux_losses else 0.0,
        }

    def fit(self):
        """主训练循环。返回 ``(train_losses, val_losses)``。"""
        train_iter = itertools.cycle(self.train_loader)

        # Part4K2 Task 7.2: quiet 模式下关闭进度条
        use_tqdm = self.enable_progress_bar and _HAS_TQDM and not self.quiet
        if use_tqdm:
            pbar = _tqdm(range(self.max_steps), desc="train_nex",
                         unit="step", dynamic_ncols=True)
        else:
            pbar = _NoOpPBar(range(self.max_steps))

        t_start = time.time()
        step_times: deque = deque(maxlen=max(self.eta_window, 1))
        last_log_step = -1
        best_step = -1

        for step in pbar:
            t_step = time.time()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if batch is None:
                continue
            x, y = batch
            x = _as_tensor(x)
            y = _as_tensor(y)

            loss, ce_val, aux_val = self._compute_loss(x, y)
            loss.backward()

            self.grad_accum.step()
            if self.grad_accum.should_step():
                if self.grad_clip > 0:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(ce_val)
            self.aux_losses.append(aux_val)
            step_times.append(time.time() - t_step)

            # Part4K2 Task 7.4: 1B 模型 GPU 显存定期清理（CPU 时 no-op）
            if (
                self.empty_cache_interval > 0
                and step > 0
                and step % self.empty_cache_interval == 0
            ):
                try:
                    empty_cache(self.device)
                except Exception:
                    pass

            # 定期评估 + checkpoint + early stop
            if self.eval_interval > 0 and step % self.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                self.early_stopping(val_loss)

                state = self._make_state(step, val_loss)
                self.checkpoint.save_last(state)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = float(val_loss)
                    best_step = step
                    self.checkpoint.save_best(state)

                # 实时 loss 图
                if self.realtime_plot:
                    curve_path = os.path.join(self.save_dir, "loss_curve.png")
                    try:
                        plot_loss_curve(
                            self.train_losses, self.val_losses,
                            curve_path, eval_interval=self.eval_interval,
                        )
                    except Exception:
                        pass

                if self.early_stopping.should_stop:
                    last_log_step = step
                    break

            # 进度条后缀
            lr_now = getattr(self.optimizer, "lr", None)
            if use_tqdm:
                postfix = {"loss": f"{ce_val:.4f}"}
                # Part5K1.1：基于 _has_mod_layers 确定性显示 aux（不再用 aux_val>0）
                if self.use_aux and self._has_mod_layers:
                    # NaN 时显示 nan（信息性），不再隐藏
                    postfix["aux"] = f"{aux_val:.4f}" if math.isfinite(aux_val) else "nan"
                if self.val_losses:
                    postfix["val"] = f"{self.val_losses[-1]:.4f}"
                if lr_now is not None:
                    postfix["lr"] = f"{lr_now:.2e}"
                postfix["best"] = f"{self.best_val_loss:.4f}"
                try:
                    pbar.set_postfix(postfix)
                except Exception:
                    pass

            # 无 tqdm 时打印日志
            # Part4K2 Task 7.2: quiet 模式下跳过中间日志打印
            if (
                not use_tqdm
                and not self.quiet
                and self.log_interval > 0
                and (step % self.log_interval == 0 or step == self.max_steps - 1)
                and step != last_log_step
            ):
                last_log_step = step
                msg = f"[step {step:>6d}/{self.max_steps}] train_loss={ce_val:.6f}"
                # Part5K1.1：基于 _has_mod_layers 确定性显示 aux
                if self.use_aux and self._has_mod_layers:
                    if math.isfinite(aux_val):
                        msg += f" aux={aux_val:.6f}"
                    else:
                        msg += " aux=nan"
                if self.val_losses:
                    msg += f" val_loss={self.val_losses[-1]:.6f}"
                if lr_now is not None:
                    msg += f" lr={lr_now:.6e}"
                if step_times and step < self.max_steps - 1:
                    avg_dt = float(np.mean(list(step_times)))
                    eta = _format_eta(avg_dt * (self.max_steps - step - 1))
                    msg += f" eta={eta}"
                print(msg, flush=True)

        pbar.close()

        # 训练摘要
        # Part4K2 Task 7.2: quiet 模式下只打印简短结果
        wall = time.time() - t_start
        n_done = len(self.train_losses)
        avg_step = wall / n_done if n_done > 0 else 0.0
        if self.quiet:
            print(
                f"[train_nex] done best_val={self.best_val_loss:.4f} "
                f"steps={n_done} wall={wall:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[train_nex] done steps={n_done}/{self.max_steps} wall={wall:.2f}s "
                f"avg_step={avg_step:.3f}s best_val={self.best_val_loss:.4f}"
                + (f" best@step={best_step}" if best_step >= 0 else ""),
                flush=True,
            )

        self._save_history()
        return self.train_losses, self.val_losses

    def _save_history(self) -> None:
        """保存 loss 历史与曲线图（含 aux_losses）。"""
        os.makedirs(self.save_dir, exist_ok=True)
        history = {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "aux_losses": list(self.aux_losses),
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "best_val_loss": self.best_val_loss,
            "aux_loss_weight": self.aux_loss_weight,
        }
        with open(os.path.join(self.save_dir, "loss_history.json"), "w",
                  encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # 纯文本列表
        for name, lst in [("train_losses", self.train_losses),
                          ("val_losses", self.val_losses),
                          ("aux_losses", self.aux_losses)]:
            with open(os.path.join(self.save_dir, f"{name}.txt"), "w",
                      encoding="utf-8") as f:
                for v in lst:
                    f.write(f"{float(v):.6f}\n")

        # 曲线图
        curve_path = os.path.join(self.save_dir, "loss_curve.png")
        actual = plot_loss_curve(
            self.train_losses, self.val_losses,
            curve_path, eval_interval=self.eval_interval,
        )
        if actual != curve_path:
            print(f"[VerseNexTrainer] 注意：loss 曲线降级保存到 {actual}",
                  flush=True)


# ---------------------------------------------------------------------------
# NoOpPBar：training.py 中是模块私有类，这里复制一份避免 import 私有名
# ---------------------------------------------------------------------------


class _NoOpPBar:
    """无 tqdm 时的进度条占位。"""

    def __init__(self, iterable):
        self._it = iterable

    def __iter__(self):
        return iter(self._it)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# LoRATrainer：LoRA-aware 训练器
# ---------------------------------------------------------------------------


class LoRATrainer(VerseNexTrainer):
    """LoRA-aware 训练器：自动包装模型为 LoRA + 仅训练 A/B 矩阵。

    工作流程：
    1. ``__init__`` 时调用 :func:`verse_torch.compress.lora_only` 把模型中
       所有 ``Linear`` / ``QLinear`` 包装为 :class:`LoRALinear`。
       原模型参数全部冻结（``requires_grad=False``），仅新增的 A/B 可训练。
    2. 训练循环复用 :class:`VerseNexTrainer`，但 optimizer 只接收 LoRA 参数。
    3. 训练结束后调用 :meth:`merge_lora` 把 ΔW 合并回 base，返回新模型。

    Args:
        model: ``nn.Module``，原模型（通常已加载预训练权重）
        train_loader / val_loader / optimizer / scheduler / cfg:
            同 :class:`VerseNexTrainer`，但 optimizer 若为 None 则自动
            基于 LoRA 参数构建 AdamW。
        lora_r: int，LoRA 秩（默认 8）
        lora_alpha: float，LoRA 缩放因子（默认 16）
        merge_after: bool，是否在 fit 结束后自动 merge LoRA 到 base
            （默认 False，需要用户显式调用 :meth:`merge_lora`）

    注意：
    - LoRA 包装是 **原地修改**：``model`` 自身会被包装，原结构不可恢复。
      若需保留原模型，请在传入前 ``copy.deepcopy``。
    - ``forward_with_aux`` 在 LoRA 包装后仍然可用（LoRALinear 是 Module 子类，
      递归包装不破坏顶层 forward_with_aux 方法）。
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        cfg=None,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        merge_after: bool = False,
    ):
        # 1. LoRA 包装（原地）
        from .compress import lora_only, LoRALinear
        print(f"[LoRATrainer] 包装模型为 LoRA (r={lora_r}, alpha={lora_alpha})",
              flush=True)
        n_before = sum(int(np.prod(p.data.shape)) for p in model.parameters())
        lora_only(model, r=lora_r, alpha=lora_alpha)
        # 统计可训练参数
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_after = sum(int(np.prod(p.data.shape)) for p in trainable)
        print(
            f"[LoRATrainer] 参数变化: total={n_before} → trainable={n_after} "
            f"({n_after / max(1, n_before) * 100:.2f}% 可训练)",
            flush=True,
        )

        # 2. 若 optimizer 未提供，自动基于 LoRA 参数构建 AdamW
        if optimizer is None:
            lr = float(_cfg_get(cfg, "lr", 1e-3))
            weight_decay = float(_cfg_get(cfg, "weight_decay", 0.0))
            optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
            print(f"[LoRATrainer] 自动构建 AdamW (lr={lr}, wd={weight_decay}) "
                  f"for {len(trainable)} LoRA 参数张量", flush=True)

        # 3. 调用父类初始化（含 forward_with_aux 检测、子控制器等）
        super().__init__(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, cfg=cfg,
        )
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.merge_after = bool(merge_after)
        # 记录 LoRA 层引用，便于后续 merge
        self._lora_layers: list = []
        for m in model.modules():
            if isinstance(m, LoRALinear):
                self._lora_layers.append(m)
        print(f"[LoRATrainer] 共发现 {len(self._lora_layers)} 个 LoRALinear 层",
              flush=True)

    def fit(self):
        """训练 + 可选 merge。"""
        result = super().fit()
        if self.merge_after:
            print("[LoRATrainer] merge_after=True，自动合并 LoRA 到 base",
                  flush=True)
            self.merge_lora()
        return result

    def merge_lora(self):
        """把所有 LoRALinear 的 ΔW 合并回 base.weight，并把 LoRALinear 替换为 Linear。

        合并后模型结构与 LoRA 包装前一致，可直接用于推理 / 保存。
        """
        from .compress import LoRALinear
        from . import vnn as nn

        # 递归遍历 _modules，把每个 LoRALinear 替换为 merge 后的 Linear
        def _replace_in_module(parent):
            for name, child in list(parent._modules.items()):
                if isinstance(child, LoRALinear):
                    merged = child.merge()  # 返回新的 nn.Linear
                    parent._modules[name] = merged
                    # 同步 __dict__（nn.Module.__setattr__ 走 else 分支）
                    object.__setattr__(parent, name, merged)
                else:
                    _replace_in_module(child)

        _replace_in_module(self.model)
        # 清空 LoRA 层引用
        n_merged = len(self._lora_layers)
        self._lora_layers = []
        print(f"[LoRATrainer] 已合并 {n_merged} 个 LoRA 层，模型恢复为标准结构",
              flush=True)


# ---------------------------------------------------------------------------
# SFTTrainer：监督微调训练器（chat 数据格式）
# ---------------------------------------------------------------------------


def _messages_to_tokens(messages, tokenizer, *,
                        system_prefix: str = "<|system|>",
                        user_prefix: str = "<|user|>",
                        assistant_prefix: str = "<|assistant|>",
                        eos_token: str = "<|endoftext|>") -> tuple:
    """把 chat messages 数组转为 ``(token_ids, loss_mask)``。

    约定：仅 assistant 的 token 参与 loss 计算（mask=1），
    system / user / 角色前缀 token 的 mask=0。

    Args:
        messages: list[dict]，每个 dict 含 ``role`` 与 ``content``
        tokenizer: 实现 ``encode(str) -> list[int]`` 与 ``decode`` 接口
        system_prefix / user_prefix / assistant_prefix: 角色标记
        eos_token: 序列结束标记

    Returns:
        token_ids: list[int]，完整 token 序列
        loss_mask: list[int]，与 token_ids 等长，1 表示该位置参与 loss
    """
    token_ids: list = []
    loss_mask: list = []

    def _append_text(text: str, trainable: bool):
        ids = tokenizer.encode(text)
        token_ids.extend(ids)
        loss_mask.extend([1 if trainable else 0] * len(ids))

    def _append_role(prefix: str):
        ids = tokenizer.encode(prefix)
        token_ids.extend(ids)
        loss_mask.extend([0] * len(ids))

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            _append_role(system_prefix)
            _append_text(content, trainable=False)
        elif role == "user":
            _append_role(user_prefix)
            _append_text(content, trainable=False)
        elif role == "assistant":
            _append_role(assistant_prefix)
            _append_text(content, trainable=True)
            _append_text(eos_token, trainable=True)
        else:
            # 未知角色：当作 system 处理（不参与 loss）
            _append_role(system_prefix)
            _append_text(content, trainable=False)

    return token_ids, loss_mask


def _build_sft_sample(messages, tokenizer, seq_len: int,
                      ignore_index: int = -100) -> tuple:
    """从 messages 构造一个 SFT 训练样本 ``(input_ids, labels)``。

    - input_ids: ndarray, shape (seq_len,)
    - labels: ndarray, shape (seq_len,)，非参与位置填 ``ignore_index``

    若 token 总数 < seq_len，左侧 pad 0；若 > seq_len，截断右侧。
    """
    token_ids, loss_mask = _messages_to_tokens(messages, tokenizer)
    # 截断或 pad 到 seq_len
    if len(token_ids) >= seq_len:
        token_ids = token_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
    else:
        pad_len = seq_len - len(token_ids)
        # 左 pad（与训练时 attention mask 对齐：右侧是有效 token）
        token_ids = [0] * pad_len + token_ids
        loss_mask = [0] * pad_len + loss_mask

    input_ids = np.asarray(token_ids, dtype=np.int64)
    labels = np.where(np.asarray(loss_mask, dtype=np.int64) == 1,
                      input_ids, ignore_index)
    return input_ids, labels


class SFTDataset:
    """SFT 数据集：从 jsonl 读取 chat 数据，构造 ``(input_ids, labels)`` 样本。

    jsonl 每行格式::

        {"messages": [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."},
                       {"role": "assistant", "content": "..."}]}

    也支持单条文本字段 ``text``（退化为标准 LM 训练，所有 token 都参与 loss）。
    """

    def __init__(self, tokenizer, jsonl_path: str, seq_len: int = 512,
                 ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.ignore_index = int(ignore_index)
        self.samples: list = []
        self._load(jsonl_path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" in obj:
                    input_ids, labels = _build_sft_sample(
                        obj["messages"], self.tokenizer,
                        seq_len=self.seq_len, ignore_index=self.ignore_index,
                    )
                elif "text" in obj:
                    # 退化为标准 LM：所有 token 参与 loss
                    ids = self.tokenizer.encode(obj["text"])
                    if len(ids) >= self.seq_len:
                        ids = ids[:self.seq_len]
                    else:
                        pad = self.seq_len - len(ids)
                        ids = [0] * pad + ids
                    input_ids = np.asarray(ids, dtype=np.int64)
                    labels = input_ids.copy()
                else:
                    continue
                self.samples.append((input_ids, labels))
        print(f"[SFTDataset] loaded {len(self.samples)} samples from {path}",
              flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _sft_collate(batch):
    """SFT batch collate：把 ``[(input_ids, labels), ...]`` stack 成 (B, T)。

    返回 ``(x, y)``，两者均为 ``np.ndarray``，shape ``(B, T)``。
    """
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    x_out = np.stack(xs, axis=0).astype(np.int64)
    y_out = np.stack(ys, axis=0).astype(np.int64)
    return x_out, y_out


class SFTTrainer(VerseNexTrainer):
    """监督微调训练器（SFT）。

    基于 :class:`VerseNexTrainer`，关键区别：
    - labels 中的 ``ignore_index`` 位置（默认 -100）不参与 loss
      （由 :func:`cross_entropy` 的 ``ignore_index`` 参数屏蔽）
    - 适合 chat 数据格式，仅对 assistant 回复 token 计算 loss

    数据加载：
    - 推荐使用 :class:`SFTDataset` + :class:`verse_torch.training.BatchLoader`
      （``collate_fn=_sft_collate``）构造 loader 后传入。

    Args:
        model / train_loader / val_loader / optimizer / scheduler / cfg:
            同 :class:`VerseNexTrainer`
        ignore_index: int，labels 中标记"不参与 loss"的值（默认 -100）
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer: Optimizer,
        scheduler=None,
        cfg=None,
        ignore_index: int = -100,
    ):
        super().__init__(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, cfg=cfg,
        )
        self.ignore_index = int(ignore_index)
        # SFT 期望 labels 是带 ignore_index 的 Tensor
        print(f"[SFTTrainer] ignore_index={self.ignore_index}", flush=True)

    def _compute_loss(self, x: Tensor, y: Tensor) -> tuple:
        """SFT loss：使用 ignore_index 屏蔽非 assistant token。"""
        if self.use_aux:
            logits, aux = _call_forward_with_aux(self.model, x)
            ce = cross_entropy(logits, y, ignore_index=self.ignore_index,
                               label_smoothing=self.label_smoothing)
            if isinstance(aux, Tensor):
                total = ce + aux * self.aux_loss_weight
                aux_val = float(aux.data.item()) if aux.data.ndim == 0 else float(aux.data.sum())
            else:
                aux_val = float(aux)
                total = ce + Tensor(np.array(aux_val * self.aux_loss_weight,
                                              dtype=np.float32))
            return total, float(_scalar(ce)), aux_val
        logits = self.model(x)
        ce = cross_entropy(logits, y, ignore_index=self.ignore_index,
                           label_smoothing=self.label_smoothing)
        return ce, float(_scalar(ce)), 0.0

    def evaluate(self) -> float:
        """SFT evaluate：使用 ignore_index 屏蔽非 assistant token。"""
        total_loss = 0.0
        n_batches = 0
        with no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                x, y = batch
                x = _as_tensor(x)
                y = _as_tensor(y)
                if self.use_aux:
                    logits, _ = _call_forward_with_aux(self.model, x)
                else:
                    logits = self.model(x)
                loss = cross_entropy(logits, y, ignore_index=self.ignore_index)
                total_loss += _scalar(loss)
                n_batches += 1
        if n_batches == 0:
            return float("nan")
        return total_loss / n_batches


# ---------------------------------------------------------------------------
# DPOTrainer：Direct Preference Optimization
# ---------------------------------------------------------------------------


def _log_probs_from_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """从 logits 计算每个位置的 log P(label)，返回与 labels 同形状的 Tensor。

    logits: (B, T, V)
    labels: (B, T) int

    Returns:
        log_probs: (B, T)
    """
    # log_softmax(logits, dim=-1) → (B, T, V)
    log_probs_full = logits.log_softmax(dim=-1)
    # gather: 取出 labels 对应位置的 log_prob
    # labels shape (B, T) → expand 到 (B, T, 1) → gather → (B, T, 1) → squeeze
    B, T, V = log_probs_full.shape
    labels_int = labels.data.astype(np.int64)
    # 用 numpy gather（避免 Tensor.__getitem__ 复杂语义）
    # log_probs_full.data: (B, T, V)
    # 对每个 (b, t)，取 log_probs_full.data[b, t, labels_int[b, t]]
    idx = labels_int[:, :, None]  # (B, T, 1)
    gathered = np.take_along_axis(log_probs_full.data, idx, axis=-1)  # (B, T, 1)
    gathered = gathered.squeeze(-1)  # (B, T)
    return Tensor(gathered, requires_grad=log_probs_full.requires_grad)


def _sum_log_probs_for_response(log_probs: Tensor,
                                 response_mask: Tensor) -> Tensor:
    """对每个样本，对 response 部分的 log_probs 求和。

    log_probs: (B, T)
    response_mask: (B, T) float（1 表示 response token，0 表示 prompt / pad）

    Returns:
        sum_log_probs: (B,)
    """
    masked = log_probs * response_mask
    return masked.sum(dim=-1)  # (B,)


class DPODataset:
    """DPO 偏好对数据集。

    jsonl 每行格式::

        {"prompt": "1+1=",
         "chosen": "2",
         "rejected": "3"}

    每条样本构造为：
    - chosen_input_ids  = prompt + chosen + eos
    - rejected_input_ids = prompt + rejected + eos
    - chosen_mask        = 0..0 (prompt len) + 1..1 (chosen + eos len)
    - rejected_mask      = 0..0 (prompt len) + 1..1 (rejected + eos len)

    所有序列 pad / truncate 到 seq_len。
    """

    def __init__(self, tokenizer, jsonl_path: str, seq_len: int = 256,
                 eos_token: str = "<|endoftext|>"):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.eos_token = eos_token
        self.samples: list = []
        self._load(jsonl_path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = obj.get("prompt", "")
                chosen = obj.get("chosen", "")
                rejected = obj.get("rejected", "")
                if not prompt or not chosen or not rejected:
                    continue
                self.samples.append((prompt, chosen, rejected))
        print(f"[DPODataset] loaded {len(self.samples)} preference pairs from {path}",
              flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        prompt, chosen, rejected = self.samples[idx]
        eos_ids = self.tokenizer.encode(self.eos_token)

        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen) + eos_ids
        rejected_ids = self.tokenizer.encode(rejected) + eos_ids

        chosen_seq = prompt_ids + chosen_ids
        chosen_mask = [0] * len(prompt_ids) + [1] * len(chosen_ids)
        rejected_seq = prompt_ids + rejected_ids
        rejected_mask = [0] * len(prompt_ids) + [1] * len(rejected_ids)

        # 截断 + pad 到 seq_len
        def _pad(seq, mask):
            if len(seq) >= self.seq_len:
                seq = seq[:self.seq_len]
                mask = mask[:self.seq_len]
            else:
                pad = self.seq_len - len(seq)
                seq = seq + [0] * pad
                mask = mask + [0] * pad
            return np.asarray(seq, dtype=np.int64), np.asarray(mask, dtype=np.float32)

        chosen_seq, chosen_mask = _pad(chosen_seq, chosen_mask)
        rejected_seq, rejected_mask = _pad(rejected_seq, rejected_mask)

        return {
            "chosen_input_ids": chosen_seq,
            "chosen_labels": chosen_seq.copy(),  # 用作 gather 索引
            "chosen_mask": chosen_mask,
            "rejected_input_ids": rejected_seq,
            "rejected_labels": rejected_seq.copy(),
            "rejected_mask": rejected_mask,
        }


def _dpo_collate(batch):
    """DPO batch collate：把 list[dict] → dict[str, ndarray]。"""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = np.stack([b[k] for b in batch], axis=0)
    return out


def _dpo_loss(policy_chosen_logps: Tensor,
              policy_rejected_logps: Tensor,
              ref_chosen_logps: Tensor,
              ref_rejected_logps: Tensor,
              beta: float = 0.1) -> Tensor:
    """DPO loss = -mean(log σ(β·((π_chosen - π_rejected) - (ref_chosen - ref_rejected)))).

    Returns:
        scalar Tensor
    """
    # π - ref
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps
    # logits = β·((π_c - π_r) - (ref_c - ref_r))
    logits = (chosen_logratio - rejected_logratio) * float(beta)
    # -log σ(logits) = log(1 + exp(-logits)) = softplus(-logits)
    # numerically stable: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    x = -logits
    # 用 Tensor 实现 softplus
    # x: (B,)
    abs_x = x * Tensor(np.array([1.0 if v >= 0 else -1.0 for v in x.data],
                                 dtype=np.float32))  # |x|
    # max(x, 0)
    pos_mask = (x.data >= 0).astype(np.float32)
    max_part = x * Tensor(pos_mask)
    # log(1 + exp(-|x|))
    exp_neg_abs = (abs_x * Tensor(np.array([-1.0], dtype=np.float32))).exp()
    log_part = (exp_neg_abs + Tensor(np.array([1.0], dtype=np.float32))).log()
    softplus = max_part + log_part
    return softplus.mean()


class DPOTrainer:
    """Direct Preference Optimization 训练器。

    DPO 通过偏好对数据直接优化策略，无需显式 reward model：
        L = -E[log σ(β·((π_c - π_r) - (ref_c - ref_r)))]

    训练流程：
    1. ``__init__`` 时深拷贝 reference model（冻结），用于计算 ref log_probs
    2. 每个 batch：
       a. policy model 前向 chosen / rejected，得到 policy log_probs
       b. reference model（no_grad）前向 chosen / rejected，得到 ref log_probs
       c. 计算 DPO loss，反向，optimizer.step
    3. 可选：每隔 N 步评估（在 val set 上计算 accuracy = mean(π_c - π_r > 0)）

    Args:
        model: policy 模型（可训练），需实现 ``forward(x) -> logits``
        ref_model: reference 模型（冻结）；若 None 则深拷贝 policy 作为 ref
        train_loader / val_loader: 可迭代对象，每次返回 DPO batch dict
            （由 :func:`_dpo_collate` 构造）
        optimizer: ``optim.Optimizer`` 实例（仅优化 policy 参数）
        scheduler: 可选学习率调度器
        cfg: dict，配置项（max_steps / eval_interval / patience / save_dir 等），
            额外支持：
            - beta: float，DPO 温度（默认 0.1）
            - ref_free: bool，是否启用参考自由版（cDPO / SimPO 风格，
              default False）
        beta: float，DPO 温度（默认 0.1，可被 cfg.beta 覆盖）
    """

    def __init__(
        self,
        model,
        ref_model: Optional = None,
        train_loader=None,
        val_loader=None,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        cfg=None,
        beta: float = 0.1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.cfg = cfg if cfg is not None else {}

        # reference model
        if ref_model is None:
            print("[DPOTrainer] ref_model=None，深拷贝 policy 作为 reference",
                  flush=True)
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model
        # 冻结 ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # optimizer
        if optimizer is None:
            lr = float(_cfg_get(cfg, "lr", 5e-6))
            weight_decay = float(_cfg_get(cfg, "weight_decay", 0.0))
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=weight_decay,
            )
            print(f"[DPOTrainer] 自动构建 AdamW (lr={lr}, wd={weight_decay})",
                  flush=True)
        self.optimizer = optimizer

        # 配置
        self.max_steps = int(_cfg_get(cfg, "max_steps", 100))
        self.eval_interval = int(_cfg_get(cfg, "eval_interval", 10))
        self.patience = int(_cfg_get(cfg, "patience", 10))
        self.save_dir = str(_cfg_get(cfg, "save_dir", "./checkpoints_dpo"))
        self.grad_accum_n = max(1, int(_cfg_get(cfg, "grad_accum", 1)))
        self.log_interval = int(_cfg_get(cfg, "log_interval", 10))
        self.grad_clip = float(_cfg_get(cfg, "grad_clip", 0.0))
        self.enable_progress_bar = bool(_cfg_get(cfg, "enable_progress_bar", True))
        self.eta_window = int(_cfg_get(cfg, "eta_window", 20))
        self.beta = float(_cfg_get(cfg, "beta", beta))

        # 子控制器
        self.early_stopping = EarlyStopping(self.patience)
        self.grad_accum = GradientAccumulator(
            micro_batch=1, effective_batch=self.grad_accum_n,
        )
        self.checkpoint = CheckpointManager(self.save_dir)

        # 历史
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # 前向：计算 chosen / rejected log_probs
    # ------------------------------------------------------------------

    def _compute_logps(self, model, input_ids, labels, mask) -> Tensor:
        """计算单个 model 在 (input_ids, labels, mask) 上的 sum log_probs。

        Returns:
            (B,) Tensor
        """
        x = _as_tensor(input_ids)
        # 前向：若 model 支持 forward_with_aux，调用并取 logits
        if _model_has_aux(model):
            logits, _ = _call_forward_with_aux(model, x)
        else:
            logits = model(x)
        # logits: (B, T, V)
        log_probs = _log_probs_from_logits(logits, _as_tensor(labels))  # (B, T)
        mask_t = _as_tensor(mask)
        return _sum_log_probs_for_response(log_probs, mask_t)  # (B,)

    def _compute_dpo_loss(self, batch) -> tuple:
        """计算 DPO loss，返回 ``(loss, accuracy)``。"""
        # policy 前向（可训练）
        policy_chosen_logps = self._compute_logps(
            self.model, batch["chosen_input_ids"],
            batch["chosen_labels"], batch["chosen_mask"],
        )
        policy_rejected_logps = self._compute_logps(
            self.model, batch["rejected_input_ids"],
            batch["rejected_labels"], batch["rejected_mask"],
        )
        # reference 前向（冻结，no_grad）
        with no_grad():
            ref_chosen_logps = self._compute_logps(
                self.ref_model, batch["chosen_input_ids"],
                batch["chosen_labels"], batch["chosen_mask"],
            )
            ref_rejected_logps = self._compute_logps(
                self.ref_model, batch["rejected_input_ids"],
                batch["rejected_labels"], batch["rejected_mask"],
            )
        # DPO loss
        loss = _dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.beta,
        )
        # accuracy: π_c - π_r > 0 的比例（policy 是否更偏好 chosen）
        # detach 后用 numpy 计算
        pc_np = policy_chosen_logps.data
        pr_np = policy_rejected_logps.data
        accuracy = float(np.mean(pc_np > pr_np))
        return loss, accuracy

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple:
        """在 val set 上计算平均 DPO loss 与 accuracy。

        Returns:
            (val_loss, val_accuracy)
        """
        if self.val_loader is None:
            return float("nan"), 0.0
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        with no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                loss, acc = self._compute_dpo_loss(batch)
                total_loss += float(_scalar(loss))
                total_acc += acc
                n_batches += 1
        if n_batches == 0:
            return float("nan"), 0.0
        return total_loss / n_batches, total_acc / n_batches

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self):
        """DPO 主训练循环。返回 ``(train_losses, val_losses, val_accuracies)``。"""
        if self.train_loader is None:
            raise ValueError("DPOTrainer.fit 需要 train_loader")

        train_iter = itertools.cycle(self.train_loader)
        use_tqdm = self.enable_progress_bar and _HAS_TQDM
        if use_tqdm:
            pbar = _tqdm(range(self.max_steps), desc="train_dpo",
                         unit="step", dynamic_ncols=True)
        else:
            pbar = _NoOpPBar(range(self.max_steps))

        t_start = time.time()
        step_times: deque = deque(maxlen=max(self.eta_window, 1))
        last_log_step = -1
        best_step = -1

        for step in pbar:
            t_step = time.time()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if batch is None:
                continue

            loss, acc = self._compute_dpo_loss(batch)
            loss.backward()

            self.grad_accum.step()
            if self.grad_accum.should_step():
                if self.grad_clip > 0:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            loss_val = float(_scalar(loss))
            self.train_losses.append(loss_val)
            step_times.append(time.time() - t_step)

            # 定期评估
            if self.eval_interval > 0 and step % self.eval_interval == 0:
                val_loss, val_acc = self.evaluate()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                self.early_stopping(val_loss)

                state = {
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "train_loss": float(loss_val),
                }
                self.checkpoint.save_last(state)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = float(val_loss)
                    best_step = step
                    self.checkpoint.save_best(state)

                if self.early_stopping.should_stop:
                    last_log_step = step
                    break

            # 进度条
            lr_now = getattr(self.optimizer, "lr", None)
            if use_tqdm:
                postfix = {"loss": f"{loss_val:.4f}", "acc": f"{acc:.3f}"}
                if self.val_losses:
                    postfix["val"] = f"{self.val_losses[-1]:.4f}"
                    postfix["val_acc"] = f"{self.val_accuracies[-1]:.3f}"
                if lr_now is not None:
                    postfix["lr"] = f"{lr_now:.2e}"
                postfix["best"] = f"{self.best_val_loss:.4f}"
                try:
                    pbar.set_postfix(postfix)
                except Exception:
                    pass

            # 无 tqdm 时打印
            if (
                not use_tqdm
                and self.log_interval > 0
                and (step % self.log_interval == 0 or step == self.max_steps - 1)
                and step != last_log_step
            ):
                last_log_step = step
                msg = (f"[dpo {step:>6d}/{self.max_steps}] "
                       f"loss={loss_val:.6f} acc={acc:.3f}")
                if self.val_losses:
                    msg += (f" val_loss={self.val_losses[-1]:.6f} "
                            f"val_acc={self.val_accuracies[-1]:.3f}")
                if lr_now is not None:
                    msg += f" lr={lr_now:.6e}"
                if step_times and step < self.max_steps - 1:
                    avg_dt = float(np.mean(list(step_times)))
                    eta = _format_eta(avg_dt * (self.max_steps - step - 1))
                    msg += f" eta={eta}"
                print(msg, flush=True)

        pbar.close()

        wall = time.time() - t_start
        n_done = len(self.train_losses)
        avg_step = wall / n_done if n_done > 0 else 0.0
        print(
            f"[dpo] done steps={n_done}/{self.max_steps} wall={wall:.2f}s "
            f"avg_step={avg_step:.3f}s best_val={self.best_val_loss:.4f}"
            + (f" best@step={best_step}" if best_step >= 0 else ""),
            flush=True,
        )

        self._save_history()
        return self.train_losses, self.val_losses, self.val_accuracies

    def _save_history(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        history = {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "val_accuracies": list(self.val_accuracies),
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "best_val_loss": self.best_val_loss,
            "beta": self.beta,
        }
        with open(os.path.join(self.save_dir, "dpo_history.json"), "w",
                  encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # DPO loss 曲线（含 accuracy）
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(10, 6))
            train_x = list(range(len(self.train_losses)))
            ax1.plot(train_x, self.train_losses, color="blue",
                     linestyle="-", linewidth=1.0, label="dpo train loss")
            if self.val_losses:
                eval_interval = max(1, self.eval_interval)
                val_x = [i * eval_interval for i in range(len(self.val_losses))]
                ax1.plot(val_x, self.val_losses, color="orange",
                         linestyle="--", linewidth=2.5, marker="o",
                         markersize=8, label=f"dpo val loss (every {eval_interval})")
            ax1.set_xlabel("step")
            ax1.set_ylabel("dpo loss")
            ax1.legend(loc="upper left")
            ax1.grid(True)
            # 第二 y 轴：accuracy
            if self.val_accuracies:
                ax2 = ax1.twinx()
                ax2.plot(val_x, self.val_accuracies, color="green",
                         linestyle=":", linewidth=2.0, marker="s",
                         markersize=6, label="val accuracy")
                ax2.set_ylabel("accuracy")
                ax2.set_ylim(0, 1)
                ax2.legend(loc="upper right")
            fig.tight_layout()
            curve_path = os.path.join(self.save_dir, "dpo_curve.png")
            fig.savefig(curve_path, dpi=100)
            plt.close(fig)
            print(f"[dpo] loss/acc 曲线已保存到: {curve_path}", flush=True)
        except ImportError:
            print("[dpo] matplotlib 未安装，跳过曲线图绘制", flush=True)


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


__all__ = [
    "VerseNexTrainer",
    "LoRATrainer",
    "SFTTrainer",
    "DPOTrainer",
    "SFTDataset",
    "DPODataset",
    "_sft_collate",
    "_dpo_collate",
    "_messages_to_tokens",
    "_build_sft_sample",
    "_log_probs_from_logits",
    "_sum_log_probs_for_response",
    "_dpo_loss",
]
