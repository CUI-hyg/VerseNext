"""VerseTorch: 训练基础设施模块。

提供训练循环常用的工具与高层 API：
- ``cross_entropy_loss``: 支持 (B, T, V) / (N, V) 与 ignore_index 的交叉熵
- ``EarlyStopping``: 早停控制器
- ``GradientAccumulator``: 梯度累积控制器
- ``CheckpointManager``: best/last 检查点持久化
- ``compute_loss_rate``: 滑动窗口 loss 下降率
- ``plot_loss_curve``: matplotlib 可选 + ASCII fallback 的 loss 曲线绘制
- ``Trainer``: 端到端训练循环

仅依赖 NumPy + Python 标准库（pickle / json / math / itertools / os）。
matplotlib 为可选依赖，缺失时自动降级为 ASCII 输出。
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .tensor import Tensor, no_grad, _is_torch_data
from .optim import Optimizer, LambdaLR, warmup_cosine_lr  # noqa: F401  重新导出方便用户
from .device import (
    has_torch,
    get_torch_module,
    _parse_device,
    is_cpu_device,
    DEFAULT_DEVICE,
    empty_cache,
    auto_tune_threads,
)

# 模块级缓存 torch 模块（无 torch 时为 None）
_TORCH = get_torch_module()


def _get_autocast(device=None, enabled: bool = True):
    """获取 autocast 上下文管理器（无 torch 或 CPU 时返回 no-op contextmanager）。

    GPU/NPU 时返回 ``torch.autocast``（fp16 默认，NPU 同样支持）；
    CPU / 无 torch / enabled=False 时为 no-op。

    Args:
        device: 设备字符串；``None`` / ``"cpu"`` 时返回 no-op。
        enabled: 是否启用 autocast；``False`` 时返回 no-op。

    Returns:
        上下文管理器（``contextlib.contextmanager`` 或自定义 no-op）
    """
    if not enabled or _TORCH is None:
        from contextlib import nullcontext
        return nullcontext()
    dev_type = _parse_device(device)
    if dev_type == "cpu":
        from contextlib import nullcontext
        return nullcontext()
    # GPU/NPU：委托 backend_torch.autocast
    from .backend_torch import autocast as _autocast
    return _autocast(device=device, enabled=enabled)


# ---------------------------------------------------------------------------
# GradScaler：FP16 梯度缩放（PyTorch 可用时用 torch.cuda.amp.GradScaler，
# 不可用时 no-op）
# ---------------------------------------------------------------------------


class GradScaler:
    """梯度缩放器（FP16 防梯度下溢）兼容接口。

    - PyTorch 可用且 device 为 CUDA：包装 ``torch.cuda.amp.GradScaler``，
      ``scale(loss)`` 缩放 loss 防止 FP16 梯度下溢，
      ``step(optimizer)`` 反缩放梯度并调用 ``optimizer.step()``，
      ``update()`` 动态调整 scale 因子。
    - 无 PyTorch / CPU 设备：所有方法为 no-op（``scale`` 直接返回原 loss，
      ``step`` 直接调用 ``optimizer.step()``，``update`` no-op）。

    这样上层训练代码可以用统一接口处理 GPU/CPU，不需要在调用点做 device 分支。

    Args:
        init_scale: 初始 scale（默认 2**16 = 65536）
        growth_factor: scale 增长因子（默认 2.0）
        backoff_factor: scale 回退因子（默认 0.5）
        growth_interval: growth 间隔步数（默认 2000）
        enabled: 是否启用梯度缩放（False 时所有方法 no-op）
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self._enabled = bool(enabled) and _TORCH is not None
        if self._enabled:
            try:
                self._scaler = _TORCH.cuda.amp.GradScaler(
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                    enabled=enabled,
                )
            except Exception:
                # torch.cuda.amp.GradScaler 在某些版本签名不同，退化为 no-op
                self._enabled = False
                self._scaler = None
        else:
            self._scaler = None

    @property
    def is_enabled(self) -> bool:
        """是否真正启用梯度缩放（CPU / 无 torch 时为 False）。"""
        return self._enabled and self._scaler is not None

    def scale(self, loss):
        """缩放 loss（无 torch / CPU 时原样返回）。"""
        if not self.is_enabled:
            return loss
        return self._scaler.scale(loss)

    def step(self, optimizer, *args, **kwargs):
        """反缩放梯度并执行 optimizer.step（无 torch / CPU 时直接 step）。"""
        if not self.is_enabled:
            return optimizer.step(*args, **kwargs)
        return self._scaler.step(optimizer, *args, **kwargs)

    def update(self, new_scale: float = None) -> None:
        """更新 scale 因子（无 torch / CPU 时 no-op）。"""
        if not self.is_enabled:
            return
        if new_scale is not None:
            self._scaler.update(new_scale)
        else:
            self._scaler.update()

    def unscale_(self, optimizer) -> None:
        """反缩放梯度（无 torch / CPU 时 no-op）。"""
        if not self.is_enabled:
            return
        try:
            self._scaler.unscale_(optimizer)
        except Exception:
            pass

    def get_scale(self) -> float:
        """获取当前 scale 因子（无 torch / CPU 时返回 1.0）。"""
        if not self.is_enabled:
            return 1.0
        try:
            return float(self._scaler.get_scale())
        except Exception:
            return 1.0

    def state_dict(self) -> dict:
        """返回 scaler 状态（无 torch / CPU 时返回空 dict）。"""
        if not self.is_enabled:
            return {}
        try:
            return self._scaler.state_dict()
        except Exception:
            return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """加载 scaler 状态（无 torch / CPU 时 no-op）。"""
        if not self.is_enabled or not state_dict:
            return
        try:
            self._scaler.load_state_dict(state_dict)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# activation checkpointing：前向不保存中间激活，反向重新计算
# ---------------------------------------------------------------------------


def activation_checkpoint(module, *args, **kwargs):
    """激活检查点：前向时不保存中间激活，反向时重新计算，节省显存。

    - PyTorch 可用且 module 为 ``torch.nn.Module``（或前向返回 torch.Tensor）：
      委托 ``torch.utils.checkpoint.checkpoint``，前向不保存中间激活，
      反向时重新计算。需 ``module`` 在 torch 路径上工作（通常配合 GPU 训练）。
    - 无 PyTorch / CPU / module 非 torch：直接 ``module(*args, **kwargs)`` 前向
      （no-op 降级，不节省显存但保证正确性）。

    Args:
        module: 可调用对象（通常是 ``torch.nn.Module`` 或 ``verse_torch.nn.Module``）
        *args: 透传给 module 的位置参数
        **kwargs: 透传给 module 的关键字参数

    Returns:
        module 前向输出
    """
    if _TORCH is None:
        # 无 torch：直接前向（CPU 路径不需要 checkpoint）
        return module(*args, **kwargs)
    try:
        from torch.utils.checkpoint import checkpoint as _torch_ckpt
    except Exception:
        return module(*args, **kwargs)
    # 检测 module 是否为 torch.nn.Module（torch 路径前向）
    is_torch_module = isinstance(module, _TORCH.nn.Module)
    # 进一步检测第一个位置参数是否为 torch.Tensor
    is_torch_input = len(args) > 0 and isinstance(args[0], _TORCH.Tensor)
    if not (is_torch_module or is_torch_input):
        # 非 torch 路径（自研 NumPy autograd）：直接前向降级
        return module(*args, **kwargs)
    try:
        # use_reentrant=False 是 PyTorch >= 1.10 推荐（更稳定）
        # 但部分版本不支持此参数，先尝试带，失败回退不带
        try:
            return _torch_ckpt(
                module, *args, use_reentrant=False, **kwargs
            )
        except TypeError:
            return _torch_ckpt(module, *args, **kwargs)
    except Exception:
        # checkpoint 失败（如输入需 grad 等）：降级为直接前向，保证训练不中断
        return module(*args, **kwargs)

# tqdm 为可选依赖：缺失时降级为无进度条的普通迭代器
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover - 环境差异
    _HAS_TQDM = False
    _tqdm = None


class _NoOpPBar:
    """无 tqdm 时的进度条占位：所有方法为 no-op，仅迭代底层 iterable。"""

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
# Part4K2 Task 7.1: ParallelTrainer 外层进度条（chunk 级别）
# ---------------------------------------------------------------------------


class _ChunkPBar:
    """ParallelTrainer 外层进度条（chunk 级别）。

    - tqdm 可用且未静默时用 ``tqdm`` 显示进度条；
    - tqdm 不可用时降级为简洁打印（每完成一个 chunk 打印一行）；
    - ``quiet=True`` 时完全静默（所有方法 no-op）。

    进度条格式示例::

        Parallel Training: 67%|████████░░░░| 2/3 chunks [02:30<01:15, 75.0s/chunk]
          | chunk_2: step 50/100, loss=3.45

    Args:
        total: 总 chunk 数
        quiet: 是否静默（True 时所有方法 no-op）
        desc: 进度条描述（默认 "Parallel Training"）
    """

    def __init__(self, total: int, quiet: bool = False,
                 desc: str = "Parallel Training"):
        self.total = int(total)
        self.quiet = bool(quiet)
        self.n = 0
        self.desc = desc
        self._tqdm = None
        self._t0 = time.time()
        # Part4K2.5 Task 6 修复：非 tty 环境降级为简洁打印
        # （CI / 重定向场景下 tqdm 会输出大量 \r 垃圾字符）
        # 仅在 tqdm 可用且未静默且 stderr 是 tty 时才用 tqdm
        if not self.quiet and _HAS_TQDM:
            try:
                import sys as _sys
                is_tty = (
                    hasattr(_sys.stderr, "isatty")
                    and _sys.stderr.isatty()
                )
                if is_tty:
                    self._tqdm = _tqdm(
                        total=self.total,
                        desc=self.desc,
                        unit="chunk",
                        dynamic_ncols=True,
                    )
            except Exception:
                self._tqdm = None

    def update(self, n: int = 1, postfix: Optional[dict] = None) -> None:
        """更新进度条。

        Args:
            n: 已完成的 chunk 数增量
            postfix: 附加信息字典（如 ``{"chunk": "2/3", "loss": 3.45}``），
                tqdm 可用时显示在进度条后缀；降级打印时拼接到行尾
        """
        self.n += n
        if self._tqdm is not None:
            try:
                self._tqdm.update(n)
                if postfix is not None:
                    self._tqdm.set_postfix(postfix)
            except Exception:
                pass
        elif not self.quiet:
            # 降级打印：每完成一个 chunk 打印一行
            pct = 100.0 * self.n / max(1, self.total)
            elapsed = time.time() - self._t0
            msg = f"[{self.desc}] {self.n}/{self.total} chunks ({pct:.0f}%) " \
                  f"elapsed={_format_eta(elapsed)}"
            if postfix:
                parts = [f"{k}={v}" for k, v in postfix.items()]
                msg += " | " + " ".join(parts)
            print(msg, flush=True)

    def close(self) -> None:
        """关闭进度条。"""
        if self._tqdm is not None:
            try:
                self._tqdm.close()
            except Exception:
                pass
            self._tqdm = None


class _SubsetDataset:
    """数据集子集包装器（用于 ParallelTrainer round_robin 策略）。

    Part4K2 Task 7.5：round_robin 模式下把数据集按索引轮询分配到不同 chunk，
    每个 chunk 只训练属于自己的数据子集，避免 chunk 间数据重复。

    Args:
        dataset: 原始数据集（需实现 ``__len__`` 与 ``__getitem__``）
        indices: 子集索引列表
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.dataset[int(self.indices[i])]


def clip_grad_norm(params, max_norm: float) -> float:
    """裁剪梯度总范数到 ``max_norm``（in-place 修改 ``p.grad``）。

    返回裁剪前的梯度总范数。``max_norm <= 0`` 时不裁剪。

    Args:
        params: 可迭代的参数（Tensor），仅处理 ``p.grad is not None`` 的参数
        max_norm: 梯度总范数上界

    Returns:
        裁剪前的总范数（float）
    """
    if max_norm is None or max_norm <= 0:
        return 0.0
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = float(math.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total_norm


def _format_eta(seconds: float) -> str:
    """把秒数格式化为人类可读的 ETA 字符串。"""
    if seconds is None or seconds < 0 or math.isnan(seconds):
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Task 2.1: cross_entropy_loss
# ---------------------------------------------------------------------------


def cross_entropy_loss(logits: Tensor, targets, ignore_index: int = -100,
                       label_smoothing: float = 0.0) -> Tensor:
    """交叉熵损失，支持 (B, T, V) / (N, V) 形状与 ignore_index 屏蔽。

    Task 7.4: 合并重复实现，本函数改为委托给
    :func:`verse_torch.losses.cross_entropy`，两个 API 入口共用同一实现，
    保持向后兼容（用户惯用 ``training.cross_entropy_loss`` 仍可用）。

    Args:
        logits: 形状 (B, T, V) 或 (N, V) 的未归一化预测
        targets: 形状 (B, T) 或 (N,) 的整型类别索引，
                 可以是 Tensor / np.ndarray / list
        ignore_index: 待忽略的标签值（默认 -100），不参与 loss 计算；
            传 ``None`` 表示不做屏蔽（与 ``losses.cross_entropy`` 行为一致）
        label_smoothing: 标签平滑系数（默认 0.0 关闭）。>0 时将 hard target
            与均匀分布混合，``loss = (1-ε)·CE_hard + ε·CE_uniform``，
            起到正则化、缓解过拟合的作用。

    Returns:
        标量 Tensor，支持 backward
    """
    # 延迟导入避免循环依赖（losses.py 不依赖 training.py，可安全 import）
    from .losses import cross_entropy
    return cross_entropy(logits, targets, ignore_index=ignore_index,
                         label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Task 2.2: EarlyStopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """早停控制器。

    当验证集 loss 连续 ``patience`` 次未出现显著下降（> ``min_delta``）时触发停止。

    Args:
        patience: 容忍的未改善轮数
        min_delta: 视为"显著改善"的最小降幅

    用法:
        >>> es = EarlyStopping(patience=5, min_delta=1e-4)
        >>> for val_loss in val_losses:
        ...     if es(val_loss):
        ...         break
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience 必须为正整数")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """传入当前 val_loss，返回是否应该停止训练。"""
        if val_loss < self.best_loss - self.min_delta:
            # 有显著改善，重置计数
            self.best_loss = float(val_loss)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        """重置内部状态。"""
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False


# ---------------------------------------------------------------------------
# Task 2.3: GradientAccumulator
# ---------------------------------------------------------------------------


class GradientAccumulator:
    """梯度累积控制器。

    通过 ``micro_batch`` 与 ``effective_batch`` 计算累积步数：
        accum_steps = effective_batch // micro_batch

    每 ``accum_steps`` 次 ``step()`` 后，``should_step()`` 返回 True 并自动重置计数。

    Args:
        micro_batch: 单次前向的 batch 大小
        effective_batch: 期望的有效 batch 大小（必须为 micro_batch 的整数倍）

    用法:
        >>> ga = GradientAccumulator(micro_batch=4, effective_batch=16)
        >>> for x, y in loader:
        ...     loss = model(x).loss(y)
        ...     loss.backward()
        ...     ga.step()
        ...     if ga.should_step():
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """

    def __init__(self, micro_batch: int, effective_batch: int):
        if micro_batch <= 0 or effective_batch <= 0:
            raise ValueError("micro_batch 和 effective_batch 必须为正整数")
        if effective_batch % micro_batch != 0:
            raise ValueError(
                f"effective_batch({effective_batch}) 必须是 micro_batch({micro_batch}) 的整数倍"
            )
        self.micro_batch = int(micro_batch)
        self.effective_batch = int(effective_batch)
        self.accum_steps = effective_batch // micro_batch
        self.counter = 0

    def step(self) -> None:
        """记录一次反向（counter++）。"""
        self.counter += 1

    def should_step(self) -> bool:
        """检查是否应该执行 optimizer.step。若 True 则自动重置 counter。"""
        if self.counter >= self.accum_steps:
            self.counter = 0
            return True
        return False

    def reset(self) -> None:
        """重置计数。"""
        self.counter = 0


# ---------------------------------------------------------------------------
# Task 2.4: CheckpointManager
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """递归把 Tensor / np.ndarray 等转为 pickle 友好的形式。"""
    if isinstance(obj, Tensor):
        return {"__tensor__": True, "data": obj.data, "requires_grad": obj.requires_grad}
    if isinstance(obj, np.ndarray):
        return obj  # pickle 原生支持 ndarray
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(obj: Any) -> Any:
    """递归把序列化形式还原（必要时把 dict 还原成 Tensor）。"""
    if isinstance(obj, dict):
        if obj.get("__tensor__") is True:
            return Tensor(obj["data"], requires_grad=bool(obj.get("requires_grad", False)))
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    return obj


class CheckpointManager:
    """检查点管理器：保存/加载 best 与 last 模型状态。

    Args:
        save_dir: 保存目录
        best_path: 自定义 best 文件路径（默认 save_dir/best.pt）
        last_path: 自定义 last 文件路径（默认 save_dir/last.pt）

    用法:
        >>> ckpt = CheckpointManager("./checkpoints")
        >>> ckpt.save_best({"model": model.state_dict(), "val_loss": 0.5})
        >>> state = ckpt.load_best()
    """

    def __init__(
        self,
        save_dir,
        best_path: Optional[os.PathLike] = None,
        last_path: Optional[os.PathLike] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = Path(best_path) if best_path is not None else self.save_dir / "best.pt"
        self.last_path = Path(last_path) if last_path is not None else self.save_dir / "last.pt"

    def save_best(self, state: dict) -> None:
        """保存最佳模型状态到 best.pt。"""
        payload = _to_serializable(state)
        with open(self.best_path, "wb") as f:
            pickle.dump(payload, f)

    def save_last(self, state: dict) -> None:
        """保存最近一次检查点到 last.pt。"""
        payload = _to_serializable(state)
        with open(self.last_path, "wb") as f:
            pickle.dump(payload, f)

    def load_best(self) -> dict:
        """从 best.pt 读取并返回状态字典。"""
        with open(self.best_path, "rb") as f:
            payload = pickle.load(f)
        return _from_serializable(payload)

    def load_last(self) -> dict:
        """从 last.pt 读取并返回状态字典。"""
        with open(self.last_path, "rb") as f:
            payload = pickle.load(f)
        return _from_serializable(payload)


# ---------------------------------------------------------------------------
# Task 2.6: compute_loss_rate
# ---------------------------------------------------------------------------


def compute_loss_rate(loss_window, window: int = 50, min_delta: float = 1e-4) -> float:
    """滑动窗口 loss 下降率。

    返回 ``(avg_first_half - avg_second_half) / avg_first_half``。
    若数据量不足 ``window`` 或 ``avg_first_half < min_delta``，返回 0.0。

    Args:
        loss_window: 最近若干步的 loss 列表
        window: 滑动窗口大小（取 loss_window 的最后 window 个）
        min_delta: 平均值低于此值视为已收敛，返回 0.0
    """
    n = len(loss_window)
    if n < window:
        return 0.0
    recent = list(loss_window[-window:])
    mid = window // 2
    first_half = recent[:mid]
    second_half = recent[mid:]
    if len(first_half) == 0 or len(second_half) == 0:
        return 0.0
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    if avg_first < min_delta:
        return 0.0
    return (avg_first - avg_second) / avg_first


# ---------------------------------------------------------------------------
# Task 2.7: plot_loss_curve
# ---------------------------------------------------------------------------


def _plot_ascii(
    train_losses,
    val_losses,
    save_path,
    eval_interval: int = 1,
    width: int = 80,
    height: int = 20,
) -> None:
    """ASCII fallback：在终端宽度 80 字符内绘制两条曲线。

    增强点（Task 8.2）：
    - val 点用独立符号 ``V`` 绘制，**后于** train 写入画布，因此即使位置重叠也会覆盖 ``T``，
      确保 val 点在密集 train 曲线中仍然可见。
    - 重叠位置（既有 T 又有 V）改用 ``*`` 标记，让用户一眼看出 val 与 train 在何处交汇。
    - 画布下方附加 val 数值表，列出每个 eval step 对应的 val loss，避免 val 点在网格中被忽略。
    """
    # 收集所有非空 loss 用于确定 y 轴范围
    all_vals = list(train_losses) + list(val_losses)
    if not all_vals:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("(no loss data)\n")
        return

    y_min = float(min(all_vals))
    y_max = float(max(all_vals))
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    n_train = len(train_losses)
    n_val = len(val_losses)

    # 构造画布
    canvas = [[" "] * width for _ in range(height)]

    def put_curve(values, n_total, char, step_fn=None):
        """在画布上绘制一条曲线。

        Args:
            values: loss 值列表
            n_total: 用于 x 坐标映射的总步数（通常是 train 的步数）
            char: 绘制字符（'T' 或 'V'）
            step_fn: 可选函数，把 value 的索引映射到实际 step 位置
                （val 用 ``lambda i: i * eval_interval`` 对齐到 train 的 step）；
                None 时用索引本身作为 step（train 行为）
        """
        n_v = len(values)
        if n_v == 0:
            return
        for i, v in enumerate(values):
            # x 映射到 [0, width-1]
            step = step_fn(i) if step_fn is not None else i
            if n_total <= 1:
                x = 0
            else:
                x = int(step * (width - 1) / max(1, n_total - 1))
            # 限制 x 在画布范围内（val 的 step 可能超出 train 范围）
            x = max(0, min(x, width - 1))
            # y 映射到 [0, height-1]（注意翻转：高 loss 在顶部）
            yf = (float(v) - y_min) / (y_max - y_min)
            yf = max(0.0, min(1.0, yf))
            y = height - 1 - int(round(yf * (height - 1)))
            if 0 <= y < height and 0 <= x < width:
                # 若该位置已有 T，则用 * 表示 val 与 train 重叠
                if char == "V" and canvas[y][x] == "T":
                    canvas[y][x] = "*"
                else:
                    # val 后绘制，自然覆盖 T（确保 V 可见）
                    canvas[y][x] = char

    # 先绘制 train（T），再绘制 val（V）——val 后绘制保证 V 在重叠处可见
    # val 的 x 坐标基于 eval_interval 对齐到 train 的 step，确保位置准确
    if eval_interval < 1:
        eval_interval = 1
    put_curve(train_losses, n_train, "T")
    put_curve(
        val_losses, n_train, "V",
        step_fn=lambda i: min(i * eval_interval, max(0, n_train - 1)),
    )

    # 写入文件
    lines = []
    lines.append(f"Loss Curve (ASCII)  range=[{y_min:.4f}, {y_max:.4f}]")
    lines.append(f"T=train  V=val  *=overlap  (width={width}, height={height})")
    lines.append("+" + "-" * width + "+")
    for row in canvas:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")
    lines.append(f"train_steps={n_train}  val_steps={n_val}  eval_interval={eval_interval}")

    # 附加 val 数值表（让 val 数据即使在密集 train 中也能被精确读出）
    if n_val > 0:
        lines.append("")
        lines.append("val_losses detail:")
        for i, v in enumerate(val_losses):
            # val 的 step 与 plot_loss_curve 中 val_x 保持一致
            step = i * eval_interval if eval_interval > 0 else i
            lines.append(f"  [step {step:>6d}] val_loss={float(v):.6f}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _print_val_info(val_losses, val_x=None) -> None:
    """打印 val_losses 摘要信息，明确告知用户数据存在。

    格式：``[info] val_losses: N points, best=X.XXXX at step M``

    Args:
        val_losses: 验证 loss 列表
        val_x: 与 val_losses 对应的 step 坐标列表；若为 None 则用索引 * 1
    """
    n = len(val_losses)
    if n == 0:
        print("[info] val_losses: 0 points", flush=True)
        return
    best_idx = int(min(range(n), key=lambda i: val_losses[i]))
    best_val = float(val_losses[best_idx])
    if val_x is not None and 0 <= best_idx < len(val_x):
        best_step = int(val_x[best_idx])
    else:
        best_step = best_idx
    print(
        f"[info] val_losses: {n} points, best={best_val:.4f} at step {best_step}",
        flush=True,
    )


def plot_loss_curve(
    train_losses,
    val_losses,
    save_path,
    eval_interval: int = 1,
) -> str:
    """绘制 loss 曲线并保存到 save_path。

    优先使用 matplotlib 绘制 PNG（蓝色实线 train + 橙色加粗带 marker 虚线 val + legend + grid + title）。
    matplotlib 不可用时降级为 ASCII 文本图（保存到 save_path 改后缀 .txt），
    ASCII 模式下 val 点用独立符号 ``V`` 绘制且优先级高于 ``T``，避免被 train 覆盖。

    Args:
        train_losses: 训练 loss 列表
        val_losses: 验证 loss 列表
        save_path: 保存路径（推荐 .png）
        eval_interval: 验证频率（用于对齐 val_x 坐标）

    Returns:
        实际写入的文件路径
    """
    save_path = str(save_path)
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # 仅捕获 ImportError，其他异常向上抛
        # 降级为 ASCII，但明确告知用户
        print(
            "[plot_loss_curve] 警告：matplotlib 未安装，降级为 ASCII 文本图。"
            "安装 matplotlib 可获得 PNG 图：pip install matplotlib",
            flush=True,
        )
        # 降级 ASCII
        if save_path.lower().endswith(".png"):
            txt_path = save_path[:-4] + ".txt"
        elif save_path.lower().endswith((".jpg", ".jpeg", ".svg", ".pdf")):
            # 去掉扩展名后加 .txt
            import os.path as _osp
            txt_path = _osp.splitext(save_path)[0] + ".txt"
        else:
            txt_path = save_path + ".txt"
        _plot_ascii(train_losses, val_losses, txt_path, eval_interval=eval_interval)
        # ASCII 模式也打印 val 信息（best step 与 matplotlib 分支一致）
        if eval_interval < 1:
            eval_interval = 1
        n_train_ascii = len(train_losses)
        if n_train_ascii > 0:
            val_x_ascii = [
                min(i * eval_interval, n_train_ascii - 1)
                for i in range(len(val_losses))
            ]
        else:
            val_x_ascii = list(range(len(val_losses)))
        _print_val_info(val_losses, val_x_ascii)
        print(f"[plot_loss_curve] loss 曲线已保存到: {txt_path}", flush=True)
        return txt_path

    # matplotlib 可用分支
    fig, ax = plt.subplots(figsize=(10, 6))
    train_x = list(range(len(train_losses)))
    # val 的 x 坐标按 eval_interval 对齐
    if eval_interval < 1:
        eval_interval = 1
    n_train_mpl = len(train_losses)
    if n_train_mpl > 0:
        # 每个 val 点对齐到 i*eval_interval，超出 train 范围时截断到最后一个 step
        val_x = [
            min(i * eval_interval, n_train_mpl - 1)
            for i in range(len(val_losses))
        ]
    else:
        val_x = list(range(len(val_losses)))

    ax.plot(train_x, train_losses, color="blue", linestyle="-", linewidth=1.0, label="train")
    if val_losses:
        # matplotlib 模式增强：val 点用显著 marker + 加粗线条 + 醒目橙色
        # 图例标注 "val (every N steps)"，让用户明确知道数据存在
        ax.plot(
            val_x,
            val_losses,
            color="orange",
            linestyle="--",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=f"val (every {eval_interval} steps)",
        )
    # Part4K2.5 Task 5：显式设置 y 轴范围，避免 loss 全 0（或全相等）时
    # matplotlib 自动缩放导致曲线不可见（与 ASCII 路径的兜底逻辑一致）。
    # 过滤 NaN / Inf，避免 set_ylim 抛 ValueError（训练异常时 loss 可能为 inf）
    all_vals_mpl = [float(v) for v in list(train_losses) + list(val_losses)
                    if v is not None and np.isfinite(float(v))]
    if all_vals_mpl:
        y_min_mpl = float(min(all_vals_mpl))
        y_max_mpl = float(max(all_vals_mpl))
        if y_max_mpl - y_min_mpl < 1e-12:
            y_max_mpl = y_min_mpl + 1.0
        ax.set_ylim(y_min_mpl, y_max_mpl)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    _print_val_info(val_losses, val_x)
    print(f"[plot_loss_curve] loss 曲线已保存到: {save_path}", flush=True)
    return save_path


# ---------------------------------------------------------------------------
# Task 2.8: Trainer
# ---------------------------------------------------------------------------


def _cfg_get(cfg, key, default):
    """从 dict 或 dataclass 中读取配置项。"""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_tensor(x) -> Tensor:
    """把 x 转为 Tensor。若 x 已是 Tensor 则原样返回。"""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, np.ndarray):
        return Tensor(x)
    return Tensor(x)


def _scalar(x) -> float:
    """把 Tensor / np scalar / float 转为 Python float。"""
    if isinstance(x, Tensor):
        return float(x.data.item()) if x.data.ndim == 0 else float(x.data.sum())
    if isinstance(x, np.ndarray):
        return float(x.item()) if x.ndim == 0 else float(x.sum())
    return float(x)


class Trainer:
    """端到端训练器。

    Args:
        model: ``nn.Module``，需实现 ``forward(x) -> logits``
        train_loader: 可迭代对象，每次返回 ``(x, y)``（Tensor 或 np.ndarray 均可）
        val_loader: 可迭代对象，每次返回 ``(x, y)``
        optimizer: ``optim.Optimizer`` 实例
        scheduler: 可选的学习率调度器
        cfg: dict 或 dataclass，包含：
            - max_steps: 最大训练步数
            - eval_interval: 评估频率
            - patience: 早停容忍轮数
            - save_dir: 检查点保存目录
            - grad_accum: 梯度累积步数（默认 1）
            - log_interval: 日志打印间隔（默认 10）
            - loss_rate_window: loss 下降率滑动窗口（默认 50）
            - autocast: 是否启用混合精度（默认 False；GPU 时启用 fp16）
        device: 目标设备字符串（``"cpu"`` / ``"cuda"`` / ``"npu"`` / ...），
            ``None`` 等价于 ``"cpu"``。传入非 CPU 设备时，``__init__`` 会
            自动把 model 迁移到该设备，并在前向时启用 autocast（若 cfg.autocast=True）。
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer: Optimizer,
        scheduler=None,
        cfg=None,
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg if cfg is not None else {}

        # 设备：None / "cpu" 走 NumPy 路径；GPU/NPU 走 torch 委托路径
        self.device = str(device) if device is not None else DEFAULT_DEVICE
        # 非 CPU 设备：迁移 model（若 model 实现 .to(device)）
        if not is_cpu_device(self.device):
            if hasattr(self.model, "to"):
                try:
                    self.model.to(self.device)
                except Exception as e:
                    print(f"[Trainer] 警告：迁移模型到 {self.device} 失败：{e}", flush=True)

        self.max_steps = int(_cfg_get(cfg, "max_steps", 100))
        self.eval_interval = int(_cfg_get(cfg, "eval_interval", 10))
        self.patience = int(_cfg_get(cfg, "patience", 10))
        self.save_dir = str(_cfg_get(cfg, "save_dir", "./checkpoints"))
        self.grad_accum_n = int(_cfg_get(cfg, "grad_accum", 1))
        self.log_interval = int(_cfg_get(cfg, "log_interval", 10))
        self.loss_rate_window = int(_cfg_get(cfg, "loss_rate_window", 50))

        # 精度优化：梯度裁剪 / 标签平滑
        self.grad_clip = float(_cfg_get(cfg, "grad_clip", 0.0))
        self.label_smoothing = float(_cfg_get(cfg, "label_smoothing", 0.0))
        # 训练 UX：tqdm 进度条 / 实时 loss 图 / ETA 滑动窗口
        self.enable_progress_bar = bool(_cfg_get(cfg, "enable_progress_bar", True))
        self.realtime_plot = bool(_cfg_get(cfg, "realtime_plot", True))
        self.eta_window = int(_cfg_get(cfg, "eta_window", 20))
        # Part4K2 Task 7.2: 输出控制
        # quiet=True：只打印最终结果（关闭进度条 + 关闭中间日志）
        # verbose=True：打印详细日志（保留进度条 + 额外信息）
        self.quiet = bool(_cfg_get(cfg, "quiet", False))
        self.verbose = bool(_cfg_get(cfg, "verbose", False))
        # 混合精度：GPU 时启用 autocast（需显式开启或 device 非 CPU 时自动启用）
        self.use_autocast = bool(_cfg_get(cfg, "autocast", False)) or not is_cpu_device(self.device)

        # Part4K2 Task 5.2: 梯度缩放（GPU 时启用，CPU no-op）
        # 显式 cfg 传入 grad_scaler；默认 GPU + autocast 时自动启用
        self.use_grad_scaler = bool(_cfg_get(cfg, "grad_scaler", False)) or (
            self.use_autocast and not is_cpu_device(self.device)
        )
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)

        # Part4K2 Task 5.2: 显存清理间隔（默认 100 步；CPU 时 no-op）
        self.empty_cache_interval = int(_cfg_get(cfg, "empty_cache_interval", 100))
        # Part4K2 Task 5.2: 梯度累积步数（accumulation_steps 是 grad_accum 的别名，
        # 优先使用 grad_accum 保持向后兼容）
        grad_accum_cfg = _cfg_get(cfg, "grad_accum", None)
        if grad_accum_cfg is None:
            grad_accum_cfg = _cfg_get(cfg, "accumulation_steps", 1)
        self.grad_accum_n = int(grad_accum_cfg)

        # Part4K2 Task 5.3: CPU BLAS 线程自动调优（Trainer 初始化时调用一次）
        try:
            auto_tune_threads()
        except Exception:
            pass

        # 子控制器
        self.early_stopping = EarlyStopping(self.patience)
        # 梯度累积：micro_batch=1, effective_batch=grad_accum_n
        # 这里采用步数累积语义（每 N 次反向 step 一次）
        if self.grad_accum_n < 1:
            self.grad_accum_n = 1
        self.grad_accum = GradientAccumulator(
            micro_batch=1, effective_batch=self.grad_accum_n
        )
        self.checkpoint = CheckpointManager(self.save_dir)

        # 训练历史
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")

    def _make_state(self, step: int, val_loss: float) -> dict:
        """构造保存到 checkpoint 的 state 字典。"""
        return {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "val_loss": float(val_loss),
            "train_loss": float(self.train_losses[-1]) if self.train_losses else float("nan"),
        }

    def evaluate(self) -> float:
        """在 val_loader 上计算平均 loss（no_grad 上下文）。"""
        total_loss = 0.0
        n_batches = 0
        with no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                x, y = batch
                x = _as_tensor(x)
                y = _as_tensor(y)
                # 非 CPU 设备：迁移输入到目标 device
                # Part4K2 Task 5.4: non_blocking=True 与 pin_memory 配合异步传输
                if not is_cpu_device(self.device):
                    if hasattr(x, "to"):
                        try:
                            x = x.to(self.device, non_blocking=True)
                        except TypeError:
                            x = x.to(self.device)
                    if hasattr(y, "to"):
                        try:
                            y = y.to(self.device, non_blocking=True)
                        except TypeError:
                            y = y.to(self.device)
                with _get_autocast(self.device, enabled=self.use_autocast):
                    logits = self.model(x)
                    loss = cross_entropy_loss(logits, y)
                total_loss += _scalar(loss)
                n_batches += 1
        if n_batches == 0:
            return float("nan")
        return total_loss / n_batches

    def fit(self):
        """主训练循环。返回 (train_losses, val_losses)。

        增强点（参考 GPT_teacher-3.37M-cn）：
        - tqdm 进度条（可选）：显示 step/total、it/s、ETA，后缀含 loss/lr/best_val
        - 梯度裁剪（grad_clip>0）：在 optimizer.step 前裁剪梯度总范数，稳定训练
        - 标签平滑（label_smoothing>0）：cross_entropy 混合均匀分布，缓解过拟合
        - 实时 loss 图（realtime_plot）：每次 eval_interval 刷新 loss_curve 文件
        - ETA 时间估算：无 tqdm 时用滑动窗口平均步耗时估算剩余时间
        """
        # 用 itertools.cycle 循环遍历 train_loader
        # 注意：若 train_loader 是生成器（一次性），cycle 会缓存全部数据
        train_iter = itertools.cycle(self.train_loader)

        # 进度条：tqdm 可用且启用时用真进度条，否则用 no-op 占位
        # Part4K2 Task 7.2: quiet 模式下关闭进度条
        use_tqdm = self.enable_progress_bar and _HAS_TQDM and not self.quiet
        if use_tqdm:
            pbar = _tqdm(
                range(self.max_steps),
                desc="train",
                unit="step",
                dynamic_ncols=True,
            )
        else:
            pbar = _NoOpPBar(range(self.max_steps))

        t_train_start = time.time()
        step_times: deque = deque(maxlen=max(self.eta_window, 1))
        last_log_step = -1
        best_step = -1

        for step in pbar:
            t_step = time.time()
            try:
                batch = next(train_iter)
            except StopIteration:
                # 空的 loader
                break
            if batch is None:
                continue
            x, y = batch
            x = _as_tensor(x)
            y = _as_tensor(y)
            # 非 CPU 设备：迁移输入到目标 device
            # Part4K2 Task 5.4: non_blocking=True 与 pin_memory 配合实现异步传输
            if not is_cpu_device(self.device):
                if hasattr(x, "to"):
                    try:
                        x = x.to(self.device, non_blocking=True)
                    except TypeError:
                        # 自研 Tensor.to 不接受 non_blocking，回退
                        x = x.to(self.device)
                if hasattr(y, "to"):
                    try:
                        y = y.to(self.device, non_blocking=True)
                    except TypeError:
                        y = y.to(self.device)

            # 混合精度前向（autocast 在 CPU 时为 no-op）
            with _get_autocast(self.device, enabled=self.use_autocast):
                logits = self.model(x)
                loss = cross_entropy_loss(
                    logits, y, label_smoothing=self.label_smoothing
                )
            # Part4K2 Task 5.1: GradScaler 缩放 loss（CPU 时 no-op，原样 backward）
            scaled_loss = self.grad_scaler.scale(loss)
            scaled_loss.backward()

            self.grad_accum.step()
            if self.grad_accum.should_step():
                # 梯度裁剪：在 optimizer.step 前裁剪累积梯度的总范数
                if self.grad_clip > 0:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)
                # Part4K2 Task 5.1: scaler.step（GPU 时反缩放梯度后 step；CPU 时直接 step）
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            loss_val = _scalar(loss)
            self.train_losses.append(loss_val)
            step_times.append(time.time() - t_step)

            # Part4K2 Task 5.2: 每 N 步清理显存（CPU 时 no-op）
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

                # 计算 loss 下降率（用于诊断）
                _ = compute_loss_rate(
                    self.train_losses, window=self.loss_rate_window
                )

                # 实时 loss 图：每次评估后刷新曲线文件，便于训练中查看进度
                if self.realtime_plot:
                    curve_path = os.path.join(self.save_dir, "loss_curve.png")
                    try:
                        plot_loss_curve(
                            self.train_losses,
                            self.val_losses,
                            curve_path,
                            eval_interval=self.eval_interval,
                        )
                    except Exception:
                        pass  # 实时绘图失败不影响训练

                if self.early_stopping.should_stop:
                    last_log_step = step
                    break

            # 更新进度条后缀
            lr_now = getattr(self.optimizer, "lr", None)
            if use_tqdm:
                postfix = {"loss": f"{loss_val:.4f}"}
                if self.val_losses:
                    postfix["val"] = f"{self.val_losses[-1]:.4f}"
                if lr_now is not None:
                    postfix["lr"] = f"{lr_now:.2e}"
                postfix["best"] = f"{self.best_val_loss:.4f}"
                try:
                    pbar.set_postfix(postfix)
                except Exception:
                    pass

            # 无 tqdm 时：保留 log_interval 打印（含 ETA），用于 CI / 无 TTY 场景
            # Part4K2 Task 7.2: quiet 模式下跳过中间日志打印
            if (
                not use_tqdm
                and not self.quiet
                and self.log_interval > 0
                and (step % self.log_interval == 0 or step == self.max_steps - 1)
                and step != last_log_step
            ):
                last_log_step = step
                msg = f"[step {step:>6d}/{self.max_steps}] train_loss={loss_val:.6f}"
                if self.val_losses:
                    msg += f" val_loss={self.val_losses[-1]:.6f}"
                if lr_now is not None:
                    msg += f" lr={lr_now:.6e}"
                # ETA：基于滑动窗口平均步耗时估算
                if step_times and step < self.max_steps - 1:
                    avg_dt = float(np.mean(list(step_times)))
                    eta = _format_eta(avg_dt * (self.max_steps - step - 1))
                    msg += f" eta={eta}"
                print(msg, flush=True)

        pbar.close()

        # 训练摘要
        # Part4K2 Task 7.2: quiet 模式下只打印简短结果
        wall = time.time() - t_train_start
        n_done = len(self.train_losses)
        avg_step = wall / n_done if n_done > 0 else 0.0
        if self.quiet:
            print(
                f"[train] done best_val={self.best_val_loss:.4f} "
                f"steps={n_done} wall={wall:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[train] done steps={n_done}/{self.max_steps} wall={wall:.2f}s "
                f"avg_step={avg_step:.3f}s best_val={self.best_val_loss:.4f}"
                + (f" best@step={best_step}" if best_step >= 0 else ""),
                flush=True,
            )

        # 训练结束：保存 loss_history.json + loss_curve 图
        self._save_history()
        return self.train_losses, self.val_losses

    def _save_history(self) -> None:
        """保存 loss 历史与曲线图。

        额外输出 ``val_losses.txt`` / ``train_losses.txt`` 纯文本列表（每行一个值），
        方便用户用 grep / awk 等命令行工具直接读取，避免依赖 JSON 解析。
        """
        os.makedirs(self.save_dir, exist_ok=True)
        # initial_loss / final_loss：训练第一个与最后一个 step 的 loss
        initial_loss = float(self.train_losses[0]) if self.train_losses else float("nan")
        final_loss = float(self.train_losses[-1]) if self.train_losses else float("nan")
        history = {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "best_val_loss": self.best_val_loss,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        }
        with open(os.path.join(self.save_dir, "loss_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # Task 8.3: 额外生成 val_losses.txt 与 train_losses.txt 纯文本列表
        with open(os.path.join(self.save_dir, "val_losses.txt"), "w", encoding="utf-8") as f:
            for v in self.val_losses:
                f.write(f"{float(v):.6f}\n")
        with open(os.path.join(self.save_dir, "train_losses.txt"), "w", encoding="utf-8") as f:
            for v in self.train_losses:
                f.write(f"{float(v):.6f}\n")

        # 画曲线图
        curve_path = os.path.join(self.save_dir, "loss_curve.png")
        actual_curve_path = plot_loss_curve(
            self.train_losses,
            self.val_losses,
            curve_path,
            eval_interval=self.eval_interval,
        )
        if actual_curve_path != curve_path:
            print(
                f"[Trainer] 注意：loss 曲线降级保存到 {actual_curve_path}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Task 6.2: inference —— 批量推理生成
    # ------------------------------------------------------------------

    def inference(
        self,
        prompts,
        temperature: float = 1.0,
        top_k=None,
        top_p=None,
        max_tokens: int = 30,
    ):
        """模型推理：批量生成（同时支持字符串 prompt 与 token ID 序列）。

        支持三种输入模式：
        1. **字符串 prompt + tokenizer**：先用 ``tokenizer.encode`` 把字符串
           转为 token ID 序列，调用 ``model.generate`` 生成，再用
           ``tokenizer.decode`` 把结果转回字符串。
        2. **字符串 prompt + 无 tokenizer**：原样把字符串传给
           ``model.generate``（由模型自己处理编码），返回值转为字符串。
        3. **token ID 序列**（向后兼容）：直接传给 ``model.generate``，
           返回 ``list[list[int]]``（每条 prompt 对应的完整 ID 序列）。

        若模型未实现 ``generate``，抛 ``NotImplementedError``。

        Args:
            prompts: 字符串列表（需 tokenizer 或模型自带编码）或
                     token ID 序列列表（list / np.ndarray / Tensor 均可）
            temperature: 采样温度；1.0 等价 greedy，>1 增加随机性，<1 收敛
            top_k: top-k 采样；None 表示不限制
            top_p: nucleus sampling 阈值 (0,1)；None 表示不限制
            max_tokens: 每条 prompt 生成的最大 token 数

        Returns:
            - 字符串 prompt：返回 ``list[str]``（每条 prompt 对应的生成文本）
            - token ID 序列 prompt：返回 ``list[list[int]]``（每条 prompt
              对应的完整 token ID 序列，含原始 prompt + 新生成部分）
        """
        # 模型必须实现 generate
        if not (hasattr(self.model, "generate") and callable(self.model.generate)):
            raise NotImplementedError(
                "Trainer.inference 需要模型实现 generate 方法；"
                f"当前模型 {type(self.model).__name__} 未提供。"
            )

        # 可选 tokenizer：若 Trainer 实例上挂了 tokenizer 属性则使用
        tokenizer = getattr(self, "tokenizer", None)

        results = []
        with no_grad():
            if hasattr(self.model, "eval"):
                try:
                    self.model.eval()
                except Exception:
                    pass
            for prompt in prompts:
                is_str = isinstance(prompt, str)

                if is_str:
                    if tokenizer is not None:
                        # 字符串 + tokenizer：encode → generate → decode
                        input_ids = self._tokenizer_encode(tokenizer, prompt)
                        generated = self._call_generate(
                            input_ids, temperature, top_k, top_p, max_tokens
                        )
                        gen_ids = self._extract_ids(generated)
                        result = tokenizer.decode(gen_ids)
                    else:
                        # 字符串 + 无 tokenizer：原样传给 generate
                        generated = self._call_generate(
                            prompt, temperature, top_k, top_p, max_tokens
                        )
                        # 模型可能返回字符串或 ndarray/list/Tensor
                        if isinstance(generated, str):
                            result = generated
                        else:
                            arr = (generated.data if isinstance(generated, Tensor)
                                   else np.asarray(generated))
                            result = str(arr.reshape(-1).tolist())
                    results.append(result)
                else:
                    # token ID 序列（向后兼容）
                    if isinstance(prompt, Tensor):
                        ids_np = prompt.data.reshape(-1).astype(np.int64)
                    else:
                        ids_np = np.asarray(prompt).reshape(-1).astype(np.int64)
                    idx_2d = ids_np[None, :]  # (1, T)
                    generated = self._call_generate(
                        idx_2d, temperature, top_k, top_p, max_tokens
                    )
                    if isinstance(generated, Tensor):
                        gen_ids = generated.data.reshape(-1).tolist()
                    else:
                        gen_ids = np.asarray(generated).reshape(-1).tolist()
                    results.append([int(x) for x in gen_ids])
        return results

    @staticmethod
    def _tokenizer_encode(tokenizer, text):
        """调用 tokenizer.encode，兼容是否接受 ``add_special_tokens`` 参数。"""
        try:
            return tokenizer.encode(text, add_special_tokens=True)
        except TypeError:
            return tokenizer.encode(text)

    @staticmethod
    def _extract_ids(generated):
        """把 generate 的返回值统一转为 1D int 列表（供 tokenizer.decode）。"""
        if isinstance(generated, Tensor):
            arr = generated.data
        else:
            arr = np.asarray(generated)
        return [int(x) for x in arr.reshape(-1).tolist()]

    def _call_generate(self, idx_2d, temperature, top_k, top_p, max_tokens):
        """调用模型 generate，兼容是否支持 top_p 参数。"""
        try:
            return self.model.generate(
                idx_2d,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=top_k,
                top_p=top_p,
            )
        except TypeError:
            # 模型 generate 不支持 top_p，降级调用
            return self.model.generate(
                idx_2d,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=top_k,
            )

    def _generate_loop(self, ids, temperature, top_k, top_p, max_tokens):
        """手动 token-by-token 生成（当模型无 generate 时使用）。"""
        ids = list(ids)
        for _ in range(int(max_tokens)):
            x = Tensor(np.asarray([ids], dtype=np.int64))
            logits = self.model(x)
            logits_np = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
            # 取最后一个时间步的 logits，兼容 (B, T, V) / (B, V) / (V,)
            while logits_np.ndim > 1:
                if logits_np.shape[0] == 1:
                    logits_np = logits_np[0]
                else:
                    logits_np = logits_np[-1]
            next_id = self._sample_from_logits(
                logits_np.reshape(-1), temperature, top_k, top_p
            )
            ids.append(int(next_id))
        return ids

    @staticmethod
    def _sample_from_logits(logits, temperature, top_k, top_p):
        """从 logits 采样单个 token，支持 temperature / top_k / top_p。"""
        # 温度 ≤ 0 等价 greedy
        if temperature is None or temperature <= 0:
            return int(np.argmax(logits))
        scaled = logits / max(float(temperature), 1e-8)
        # 数值稳定：减去最大值后再 softmax
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        s = probs.sum()
        if s <= 0:
            return int(np.argmax(logits))
        probs = probs / s
        # top-k 截断
        if top_k is not None and top_k > 0:
            k = min(int(top_k), len(probs))
            top_idx = np.argpartition(probs, -k)[-k:]
            mask = np.zeros_like(probs)
            mask[top_idx] = 1.0
            probs = probs * mask
            probs = probs / probs.sum()
        # top-p (nucleus) 截断
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = int(np.searchsorted(cumsum, float(top_p))) + 1
            cutoff = min(cutoff, len(probs))
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = 1.0
            probs = probs * mask
            probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Task 3.6: BatchLoader —— 对齐 torch.utils.data.DataLoader 接口
# ---------------------------------------------------------------------------


def _default_collate(batch):
    """默认 collate 函数：把 ``[(x1, y1), (x2, y2), ...]`` 拼成 ``([X], [Y])``。

    约定每个 sample 是 ``(x, y)`` 元组（或可索引对象），沿第 0 维 stack。
    若 ``x`` / ``y`` 是 ``Tensor``，返回 stacked ``Tensor``；否则返回 ndarray。
    """
    if not isinstance(batch, (list, tuple)) or len(batch) == 0:
        return batch

    # 检查是否是 (x, y) 元组列表
    first = batch[0]
    if isinstance(first, (list, tuple)) and len(first) == 2:
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        # 优先用 Tensor（保持 requires_grad 信息）
        if isinstance(xs[0], Tensor):
            x_out = Tensor(np.stack([x.data if isinstance(x, Tensor) else x
                                     for x in xs], axis=0),
                           requires_grad=False)
        else:
            x_out = np.stack(xs, axis=0)
        if isinstance(ys[0], Tensor):
            y_out = Tensor(np.stack([y.data if isinstance(y, Tensor) else y
                                     for y in ys], axis=0),
                           requires_grad=False)
        else:
            y_out = np.stack(ys, axis=0)
        return x_out, y_out
    # 不是 (x, y) 元组：直接 stack
    if isinstance(first, Tensor):
        return Tensor(np.stack([b.data if isinstance(b, Tensor) else b for b in batch], axis=0),
                      requires_grad=False)
    return np.stack(batch, axis=0)


class BatchLoader:
    """对齐 ``torch.utils.data.DataLoader`` 接口的批量加载器。

    CPU-only 实现：``num_workers`` / ``pin_memory`` / ``persistent_workers``
    为占位参数（保留以匹配 PyTorch API 但忽略实际行为），实际单线程同步迭代。

    Args:
        dataset: 可索引数据集（实现 ``__getitem__`` 与 ``__len__``），
                 或一个 ``[(x, y), ...]`` 列表
        batch_size: 批量大小
        shuffle: 是否每轮打乱顺序
        collate_fn: 自定义 collate 函数，签名 ``(list[sample]) -> batch``；
                    默认 ``_default_collate`` 处理 ``(x, y)`` 元组
        drop_last: 是否丢弃最后不足 ``batch_size`` 的 batch
        seed: 随机种子（默认 0；设为 None 表示不固定）
        num_workers: 占位参数（CPU-only，默认 0，忽略）
        pin_memory: 占位参数（CPU-only，默认 False，忽略）
        persistent_workers: 占位参数（默认 False，忽略）

    用法:
        >>> loader = BatchLoader(dataset, batch_size=8, shuffle=True)
        >>> for x, y in loader:
        ...     ...

    或与 ``Trainer`` 配合（``Trainer`` 期望 loader 可迭代返回 ``(x, y)``）。
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = True,
                 collate_fn=None, drop_last: bool = False, seed: int = 0,
                 num_workers: int = 0, pin_memory: bool = False,
                 persistent_workers: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.collate_fn = collate_fn if collate_fn is not None else _default_collate
        self.drop_last = bool(drop_last)
        self.seed = seed
        # 占位参数（CPU-only 实现忽略，但保留以匹配 PyTorch API）
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        # 内部 RNG 状态
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def __len__(self) -> int:
        """返回一个 epoch 内的 batch 数。"""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """迭代一个 epoch 的所有 batch。"""
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)

        # 按 batch_size 切片
        for i in range(0, n, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                # drop_last 且最后一个 batch 不满：丢弃
                break
            # 收集 batch
            batch = [self.dataset[int(idx)] for idx in batch_indices]
            yield self.collate_fn(batch)


__all__ = [
    "cross_entropy_loss",
    "EarlyStopping",
    "GradientAccumulator",
    "CheckpointManager",
    "compute_loss_rate",
    "plot_loss_curve",
    "Trainer",
    "BatchLoader",
    "clip_grad_norm",
    "ParallelTrainer",
    "DistributedTrainer",
    # Part4K2 Task 5: 资源利用优化
    "GradScaler",
    "activation_checkpoint",
    # 重新导出 optim 中新增项，方便用户从 training 一次性导入
    "LambdaLR",
    "warmup_cosine_lr",
]


# ---------------------------------------------------------------------------
# Task 5: ParallelTrainer —— 并行训练器（CPU 串行实现，接口对齐并行）
# ---------------------------------------------------------------------------


class ParallelTrainer:
    """并行训练器（CPU 串行实现，接口对齐并行）。

    把 ``max_steps`` 拆成 N 个 chunk，每个 chunk 用一个独立 ``Trainer`` 实例训练，
    训练完后按 ``train_loss + val_loss`` 排序（**差的前、好的后**）串行重训，
    最后整体 fine-tune 若干步。

    关键 BUG 修复：每个 chunk 完成后基于**完整** val 数据集更新 ``val_loss``
    （旧实现只用 batch 局部 val，存在不可比与方差大的漏洞）。

    Args:
        model: 模型（需实现 ``state_dict`` / ``load_state_dict`` / ``parameters``）
        train_dataset: 训练数据集（实现 ``__getitem__`` 与 ``__len__``）
        val_dataset: 验证数据集（**完整！**用于 ``val_loss`` 更新）
        optimizer_cls: 优化器类（默认在 ``fit`` 时回退到 ``AdamW``）
        optimizer_kwargs: 优化器额外参数（如 ``betas`` / ``eps`` / ``weight_decay``）
        cfg: 训练配置 dict，包含：
            - parallel_chunks: int (N, 默认 4)
            - max_steps: int (默认 200)
            - batch_size: int (默认 8)
            - lr: float (默认 0.003)
            - warmup: int (默认 20)
            - eval_interval: int (默认 20)
            - grad_clip: float (默认 0.0)
            - label_smoothing: float (默认 0.0)
            - merge_finetune_steps: int (默认 max_steps // 10)
            - seed: int (默认 42)
            - patience: int (默认 10，传入 chunk Trainer 用于早停)
        loss_fn: 损失函数（默认 ``cross_entropy_loss``）
        collate_fn: 批处理函数（默认 ``_default_collate``）
        checkpoint_mgr: ``CheckpointManager``（可选；若提供则在 fit 结束保存 best）

    用法:
        >>> trainer = ParallelTrainer(model, train_ds, val_ds, cfg={"parallel_chunks": 4, "max_steps": 200})
        >>> trainer.fit()
    """

    def __init__(self, model, train_dataset, val_dataset,
                 optimizer_cls=None, optimizer_kwargs=None,
                 cfg=None, loss_fn=None, collate_fn=None,
                 checkpoint_mgr=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer_cls = optimizer_cls  # 默认在 fit 时回退到 AdamW
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.cfg = cfg or {}
        self.loss_fn = loss_fn  # 默认 cross_entropy_loss
        self.collate_fn = collate_fn  # 默认 _default_collate
        self.checkpoint_mgr = checkpoint_mgr

        self.parallel_chunks = int(self.cfg.get("parallel_chunks", 4))
        if self.parallel_chunks < 1:
            self.parallel_chunks = 1
        self.max_steps = int(self.cfg.get("max_steps", 200))
        self.batch_size = int(self.cfg.get("batch_size", 8))
        self.lr = float(self.cfg.get("lr", 0.003))
        self.warmup = int(self.cfg.get("warmup", 20))
        self.eval_interval = int(self.cfg.get("eval_interval", 20))
        self.grad_clip = float(self.cfg.get("grad_clip", 0.0))
        self.label_smoothing = float(self.cfg.get("label_smoothing", 0.0))
        self.merge_finetune_steps = int(self.cfg.get(
            "merge_finetune_steps", max(1, self.max_steps // 10)))
        self.seed = int(self.cfg.get("seed", 42))
        # chunk Trainer 的早停耐心（设大一些避免 chunk 内过早停止）
        self.patience = int(self.cfg.get("patience", 10))

        # Part4K2 Task 7.2: 输出控制（quiet/verbose）
        # quiet=True：只打印最终结果；verbose=True：打印详细日志
        # 两者都为 False 时为标准模式（进度条 + 关键指标）
        self.quiet = bool(self.cfg.get("quiet", False))
        self.verbose = bool(self.cfg.get("verbose", False))
        # Part4K2 Task 7.1: 外层进度条开关（chunk 级别）
        self.enable_progress_bar = bool(self.cfg.get("enable_progress_bar", True))
        # Part4K2 Task 7.5: 并行训练数据分配策略
        # sequential（默认）：每个 chunk 用完整 train_dataset
        # round_robin：数据集按索引轮询分配，chunk 间数据不重复
        self.parallel_strategy = str(self.cfg.get("parallel_strategy", "sequential"))
        if self.parallel_strategy not in ("sequential", "round_robin"):
            self.parallel_strategy = "sequential"

        # 训练历史
        self.history = {"train_loss": [], "val_loss": [], "steps": []}
        self.chunk_stats = []  # 每个 chunk 的统计（按完成顺序）
        self.chunk_steps_list = []  # 拆分后的步数列表（fit 后填充）
        self.best_val_loss = float("inf")
        self.best_state_dict = None

        # Part4 P10: 检测模型是否支持 forward_with_aux，启用 aux_loss 路径
        # use_aux=True 时 _train_chunk 会用 VerseNexTrainer 而非 Trainer，
        # _eval_full_val 会调用 forward_with_aux 取 logits 避免 (logits, aux)
        # tuple 破坏 loss_fn。
        from .training_nex import _model_has_aux, _get_aux_loss_weight
        self.use_aux = _model_has_aux(model)
        if self.use_aux:
            self.aux_loss_weight = _get_aux_loss_weight(model, default=0.01)
            print(
                f"[ParallelTrainer] 检测到 forward_with_aux，启用 aux_loss 路径 "
                f"(aux_loss_weight={self.aux_loss_weight})",
                flush=True,
            )
        else:
            self.aux_loss_weight = 0.0

    # ------------------------------------------------------------------
    # 公开辅助：步数拆分（便于测试验证）
    # ------------------------------------------------------------------

    def _split_steps(self):
        """把 ``max_steps`` 拆成 ``parallel_chunks`` 份，余数均摊到前几个 chunk。

        返回 ``list[int]``：
        - 若 ``max_steps >= parallel_chunks``：长度 = ``parallel_chunks``，
          每个元素 >= 1，和 = ``max_steps``。
        - 若 ``max_steps < parallel_chunks``：过滤掉 0 步 chunk，
          长度 = ``max_steps``（每个 chunk 1 步），和 = ``max_steps``。
        """
        n = self.parallel_chunks
        base = self.max_steps // n
        remainder = self.max_steps % n
        steps_list = [base] * n
        for i in range(remainder):
            steps_list[i] += 1
        # 过滤 0 步 chunk（max_steps < parallel_chunks 时部分 chunk 为 0）
        # 当 max_steps >= parallel_chunks 时，base >= 1，所有 chunk 至少 1 步
        steps_list = [s for s in steps_list if s > 0]
        return steps_list

    # ------------------------------------------------------------------
    # Part4K2 Task 7: 辅助方法（参数统计 / 设备信息 / 数据切分 / 输出控制）
    # ------------------------------------------------------------------

    def _count_params(self) -> int:
        """统计模型可训练参数量。"""
        try:
            if hasattr(self.model, "parameters"):
                return sum(
                    int(np.prod(p.data.shape))
                    for p in self.model.parameters()
                    if getattr(p, "requires_grad", True)
                )
        except Exception:
            pass
        return 0

    def _get_arch(self) -> str:
        """获取模型架构名（用于训练开始时打印）。"""
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            arch = getattr(cfg, "arch", None)
            if arch:
                return str(arch)
        return type(self.model).__name__

    def _get_device(self) -> str:
        """获取模型当前设备信息。"""
        # 优先用 device_info()（CometSparkV05LM 提供）
        if hasattr(self.model, "device_info") and callable(self.model.device_info):
            try:
                info = self.model.device_info()
                if isinstance(info, str):
                    return info
                if isinstance(info, dict):
                    return str(info.get("device", "cpu"))
            except Exception:
                pass
        # 兜底：检测 model.net.device 或 model.device
        for attr in ("device", "_device"):
            dev = getattr(self.model, attr, None)
            if dev is not None:
                return str(dev)
            net = getattr(self.model, "net", None)
            if net is not None:
                dev = getattr(net, attr, None)
                if dev is not None:
                    return str(dev)
        return "cpu"

    def _print_model_info(self) -> None:
        """训练开始时打印模型信息（参数量、架构、设备）。

        受 ``quiet`` 控制：quiet=True 时不打印。
        """
        if self.quiet:
            return
        n_params = self._count_params()
        arch = self._get_arch()
        device = self._get_device()
        # 参数量人类可读格式
        if n_params >= 1_000_000_000:
            params_str = f"{n_params / 1e9:.2f}B"
        elif n_params >= 1_000_000:
            params_str = f"{n_params / 1e6:.2f}M"
        elif n_params >= 1_000:
            params_str = f"{n_params / 1e3:.1f}K"
        else:
            params_str = str(n_params)
        print(
            f"[parallel] 模型: arch={arch} params={params_str} ({n_params:,}) "
            f"device={device} | chunks={self.parallel_chunks} "
            f"max_steps={self.max_steps} strategy={self.parallel_strategy}",
            flush=True,
        )

    def _print_summary(self, wall_time: float) -> None:
        """训练结束时打印总结（最佳 val_loss、总步数、训练时间、loss 趋势）。

        受 ``quiet`` 控制：quiet=True 时只打印一行最简结果。
        """
        n_chunks = len(self.chunk_stats)
        total_steps = sum(s.get("steps", 0) for s in self.chunk_stats)
        # loss 趋势：比较第一个和最后一个 chunk 的 train_loss
        trend = "N/A"
        if len(self.chunk_stats) >= 2:
            first = self.chunk_stats[0].get("train_loss", float("nan"))
            last = self.chunk_stats[-1].get("train_loss", float("nan"))
            if not (math.isnan(first) or math.isnan(last)):
                if last < first:
                    trend = f"↓ {first:.4f}→{last:.4f}"
                else:
                    trend = f"↑ {first:.4f}→{last:.4f}"

        if self.quiet:
            # quiet 模式：只打印最终结果
            print(
                f"[parallel] done best_val={self.best_val_loss:.4f} "
                f"steps={total_steps} wall={wall_time:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[parallel] 训练完成 best_val_loss={self.best_val_loss:.4f} "
                f"chunks={n_chunks} steps={total_steps} wall={wall_time:.1f}s "
                f"trend={trend}",
                flush=True,
            )

    def _split_dataset_round_robin(self, dataset, n_chunks: int, chunk_id: int):
        """round_robin 策略：把数据集按索引轮询分配到不同 chunk。

        Part4K2 Task 7.5：chunk 间数据不重复，每个 chunk 训练不同的数据子集。
        分配规则：``indices[i]`` 属于 ``chunk_id`` 当且仅当
        ``i % n_chunks == chunk_id``。

        Args:
            dataset: 原始数据集
            n_chunks: 总 chunk 数
            chunk_id: 当前 chunk 编号（0-based）

        Returns:
            :class:`_SubsetDataset` 包装的子集
        """
        n = len(dataset)
        chunk_indices = [i for i in range(n) if i % n_chunks == chunk_id]
        return _SubsetDataset(dataset, chunk_indices)

    # ------------------------------------------------------------------
    # 关键 BUG 修复：基于完整 val 数据集评估
    # ------------------------------------------------------------------

    def _eval_full_val(self, model) -> float:
        """【BUG 修复】基于完整 val 数据集计算平均 val_loss。

        旧实现只用一个 batch 估算 val_loss，存在严重漏洞：
        - 不同 chunk 用的 batch 可能不同，val_loss 不可比
        - 单 batch 估算方差大，无法准确反映模型质量

        本方法跑完整 val 数据集，返回平均 val_loss。
        若 ``val_dataset`` 为空，返回 ``inf``。

        Part4 P10: 当 ``self.use_aux=True`` 时调用 ``forward_with_aux`` 取
        logits，避免 ``model(x)`` 返回 ``(logits, aux)`` tuple 破坏 loss_fn。
        """
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return float("inf")

        loss_fn = self.loss_fn if self.loss_fn is not None else cross_entropy_loss

        # 延迟导入 aux 辅助函数（仅 use_aux 时）
        if self.use_aux:
            from .training_nex import _call_forward_with_aux

        total_loss = 0.0
        n_batches = 0
        # 用 BatchLoader 跑完整 val（shuffle=False 保证可复现）
        val_loader = BatchLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=self.collate_fn, seed=self.seed,
        )
        with no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x_batch, y_batch = batch
                x = _as_tensor(x_batch)
                y = _as_tensor(y_batch)
                if self.use_aux:
                    logits, _ = _call_forward_with_aux(model, x)
                else:
                    logits = model(x)
                loss = loss_fn(logits, y, label_smoothing=self.label_smoothing)
                total_loss += _scalar(loss)
                n_batches += 1
        if n_batches == 0:
            return float("inf")
        return total_loss / n_batches

    # ------------------------------------------------------------------
    # 单个 chunk 训练
    # ------------------------------------------------------------------

    def _train_chunk(self, model, train_dataset, chunk_steps, chunk_id):
        """训练单个 chunk，返回 ``(model, train_loss, val_loss)``。

        - 用 ``BatchLoader`` 包装 ``train_dataset`` / ``val_dataset`` 后传入 ``Trainer``
          （修复参考实现中错误使用 ``train_dataset=`` 关键字的 Bug）。
        - 用 ``tempfile.TemporaryDirectory`` 作为 chunk 的 ``save_dir``，
          避免 chunk 内部的 checkpoint/loss 文件污染用户目录。
        - 关闭 tqdm 进度条与实时绘图，降低 IO 噪音。
        - chunk 训练结束后调用 ``_eval_full_val`` 计算可比较的 val_loss。

        Part4K2.5 Task 6 修复：
        - chunk 开始时备份模型状态，确保 chunk 训练后状态可追踪
        - 每个 chunk 创建独立的优化器实例，避免优化器状态跨 chunk 泄漏
        - chunk 训练后模型状态为该 chunk 的最终状态（由调用方决定是否恢复）
        """
        if chunk_steps <= 0:
            # 0 步 chunk：仅评估当前模型
            val_loss = self._eval_full_val(model)
            return model, float("inf"), val_loss

        # chunk 配置：覆盖 max_steps/warmup/eval_interval，关闭进度条与实时绘图
        chunk_cfg = dict(self.cfg)
        chunk_cfg["max_steps"] = chunk_steps
        chunk_cfg["warmup"] = min(self.warmup, max(1, chunk_steps // 4))
        chunk_cfg["eval_interval"] = min(self.eval_interval, max(1, chunk_steps // 2))
        chunk_cfg["patience"] = max(self.patience, chunk_steps + 1)  # chunk 内不早停
        chunk_cfg["grad_clip"] = self.grad_clip
        chunk_cfg["label_smoothing"] = self.label_smoothing
        chunk_cfg["enable_progress_bar"] = False
        chunk_cfg["realtime_plot"] = False
        chunk_cfg["log_interval"] = max(chunk_steps + 1, 1000)  # 静默
        chunk_cfg["loss_rate_window"] = min(50, max(10, chunk_steps // 4))

        # Part4 P10: use_aux 时把 aux_loss_weight 写入 chunk_cfg，
        # VerseNexTrainer 会从 cfg 读取以覆盖模型默认值（保持一致）
        if self.use_aux:
            chunk_cfg["aux_loss_weight"] = self.aux_loss_weight

        # 每个 chunk 创建全新的优化器实例，确保优化器状态不跨 chunk 泄漏
        optimizer_cls = self.optimizer_cls
        if optimizer_cls is None:
            from .optim import AdamW
            optimizer_cls = AdamW
        optimizer = optimizer_cls(
            model.parameters(), lr=self.lr, **self.optimizer_kwargs)

        # 用 BatchLoader 包装 dataset，对齐 Trainer 接口
        # 注意：chunk_id 在重训阶段为 -(idx+1)、finetune 阶段为 -999，
        # 直接相加会导致 seed 为负数（RandomState 要求 [0, 2**32-1]），
        # 用 abs(chunk_id)+1 偏移确保非负且各 chunk 间 seed 互不相同。
        chunk_seed = int(self.seed) + abs(int(chunk_id)) + 1
        chunk_seed = chunk_seed % (2**32 - 1)
        train_loader = BatchLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.collate_fn, seed=chunk_seed)
        val_loader = BatchLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.collate_fn, seed=self.seed)

        loss_fn = self.loss_fn if self.loss_fn is not None else cross_entropy_loss

        # chunk 临时保存目录（自动清理）
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"verse_chunk_{chunk_id}_") as tmp_dir:
            chunk_cfg["save_dir"] = tmp_dir
            # Part4 P10: use_aux 时使用 VerseNexTrainer（aux_loss-aware），
            # 否则保持原 Trainer 行为（transformer arch）
            if self.use_aux:
                from .training_nex import VerseNexTrainer
                chunk_trainer = VerseNexTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=None,
                    cfg=chunk_cfg,
                )
            else:
                chunk_trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=None,
                    cfg=chunk_cfg,
                )
            chunk_trainer.fit()
            # 关键：用完整 val 数据集计算 val_loss（修复 BUG）
            val_loss = self._eval_full_val(model)
            train_loss = (chunk_trainer.train_losses[-1]
                          if chunk_trainer.train_losses else float("inf"))
        # chunk_trainer 和 optimizer 在 with 块结束后被 GC，
        # 确保优化器状态不泄漏到下一个 chunk
        return model, train_loss, val_loss

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def fit(self):
        """并行训练主流程。

        1. 拆分 ``max_steps`` 为 N 个 chunk
        2. 每个 chunk 独立训练（CPU 串行，避免 GIL 竞争），每个 chunk 结束后用
           完整 val 数据集评估 val_loss
        3. 按 ``train_loss + val_loss`` 排序（差前好后）
        4. 串行重训（差的部分先收敛、好的部分微调），每次重训后更新 best
        5. 整体 fine-tune（``merge_finetune_steps`` 步）
        6. 加载最佳状态到 ``self.model``，若提供 ``checkpoint_mgr`` 则保存

        Part4K2 Task 7.1：添加外层进度条（chunk 级别），chunk 内部不开 tqdm
        （保持现有行为，避免嵌套进度条）。tqdm 不可用时降级为简洁打印。

        Part4K2 Task 7.2：输出控制
        - ``quiet=True``：只打印最终结果
        - ``verbose=True``：打印详细日志（包括每个 chunk 的详细信息）
        - 默认（两者都 False）：进度条 + 关键指标

        Part4K2 Task 7.5：``parallel_strategy``
        - ``sequential``（默认）：每个 chunk 用完整 train_dataset
        - ``round_robin``：数据集按索引轮询分配，chunk 间数据不重复

        Returns:
            ``self.history`` dict，含 ``train_loss`` / ``val_loss`` / ``steps`` 三个列表
        """
        import copy

        t_fit_start = time.time()

        # Part4K2 Task 7.2: 训练开始打印模型信息（除非 quiet）
        self._print_model_info()

        # 1. 拆分步数
        chunk_steps_list = self._split_steps()
        self.chunk_steps_list = list(chunk_steps_list)
        actual_chunks = len(chunk_steps_list)
        if self.verbose:
            print(f"[parallel] 步数拆分: {chunk_steps_list} "
                  f"merge_ft={self.merge_finetune_steps}", flush=True)

        # 备份原始模型状态（每个 chunk 都从同一状态出发）
        original_state = None
        if hasattr(self.model, "state_dict"):
            original_state = copy.deepcopy(self.model.state_dict())

        # Part4K2 Task 7.1: 创建外层进度条
        # 总阶段数 = Phase 1 (actual_chunks) + Phase 2 (actual_chunks 重训)
        #           + Phase 3 (0 or 1 finetune)
        total_phases = actual_chunks * 2 + (1 if self.merge_finetune_steps > 0 else 0)
        outer_pbar = _ChunkPBar(
            total=total_phases, quiet=self.quiet, desc="Parallel Training")

        # 2. 训练每个 chunk（从原始状态出发，独立训练）
        chunk_results = []
        for i in range(actual_chunks):
            # 重置模型到原始状态
            if original_state is not None and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(copy.deepcopy(original_state))

            # Part4K2 Task 7.5: round_robin 模式下使用数据子集
            if self.parallel_strategy == "round_robin":
                chunk_dataset = self._split_dataset_round_robin(
                    self.train_dataset, actual_chunks, i)
            else:
                chunk_dataset = self.train_dataset

            if self.verbose:
                ds_info = ""
                if self.parallel_strategy == "round_robin":
                    ds_info = f" (subset={len(chunk_dataset)})"
                print(f"[parallel] chunk {i+1}/{actual_chunks} "
                      f"训练 {chunk_steps_list[i]} 步{ds_info}...", flush=True)

            model, train_loss, val_loss = self._train_chunk(
                self.model, chunk_dataset, chunk_steps_list[i], i)

            stat = {
                "chunk_id": i,
                "steps": chunk_steps_list[i],
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "model_state": (copy.deepcopy(model.state_dict())
                                if hasattr(model, "state_dict") else None),
            }
            chunk_results.append(stat)
            self.chunk_stats.append(stat)

            # Part4K2 Task 7.1: 更新外层进度条
            outer_pbar.update(
                n=1,
                postfix={
                    "chunk": f"{i+1}/{actual_chunks}",
                    "loss": f"{train_loss:.4f}",
                    "val": f"{val_loss:.4f}",
                },
            )

            if self.verbose:
                print(f"[parallel] chunk {i+1} 完成: "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
                      flush=True)

            # 更新 best（chunk 阶段也记录）
            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                if hasattr(self.model, "state_dict"):
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())

        # 3. 按 train_loss + val_loss 排序（差前好后：loss 大的在前）
        chunk_results.sort(
            key=lambda x: x["train_loss"] + x["val_loss"], reverse=True)
        if self.verbose:
            print(f"[parallel] chunk 排序（差前好后）: "
                  f"{[r['chunk_id'] for r in chunk_results]}", flush=True)

        # 4. 串行重训（差前好后）
        if original_state is not None and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(copy.deepcopy(original_state))

        for idx, result in enumerate(chunk_results):
            # Part4K2.5 Task 6 修复：chunk_steps < 4 时跳过 Phase 2 重训
            # （步数太少重训意义不大，且 chunk_steps // 4 == 0 会导致步数为 0）
            retrain_steps = result["steps"] // 4
            if retrain_steps <= 0:
                # chunk_steps < 4：跳过重训，但仍 update 进度条以保持 total 一致
                if self.verbose:
                    print(f"[parallel] 跳过 chunk {result['chunk_id']} 重训"
                          f"（steps={result['steps']} < 4）", flush=True)
                outer_pbar.update(
                    n=1,
                    postfix={
                        "retrain": f"chunk_{result['chunk_id']}(skip)",
                        "val": "N/A",
                    },
                )
                continue

            if self.verbose:
                print(f"[parallel] 串行重训 chunk {result['chunk_id']} "
                      f"({idx+1}/{len(chunk_results)})...", flush=True)
            # 加载该 chunk 的最佳状态作为重训起点
            if result["model_state"] is not None and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(copy.deepcopy(result["model_state"]))
            # Part4K2.5 Task 6 修复：retrain_steps = chunk_steps // 4（>= 1，因为
            # chunk_steps >= 4 时 result["steps"] // 4 >= 1）
            # _train_chunk 内部会创建新的优化器，确保 Phase 2 不复用旧优化器状态

            # Part4K2 Task 7.5: round_robin 重训也用对应 chunk 的数据子集
            if self.parallel_strategy == "round_robin":
                chunk_dataset = self._split_dataset_round_robin(
                    self.train_dataset, actual_chunks, result["chunk_id"])
            else:
                chunk_dataset = self.train_dataset

            self._train_chunk(
                self.model, chunk_dataset, retrain_steps, -(idx + 1))
            # 更新 val_loss 与 best
            val_loss = self._eval_full_val(self.model)

            # Part4K2 Task 7.1: 更新外层进度条
            outer_pbar.update(
                n=1,
                postfix={
                    "retrain": f"chunk_{result['chunk_id']}",
                    "val": f"{val_loss:.4f}",
                },
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                if hasattr(self.model, "state_dict"):
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                if self.verbose:
                    print(f"[parallel] 新最佳 val_loss={val_loss:.4f}", flush=True)

        # 5. 整体 fine-tune
        if self.merge_finetune_steps > 0:
            if self.verbose:
                print(f"[parallel] 整体 fine-tune {self.merge_finetune_steps} 步...",
                      flush=True)
            if self.best_state_dict is not None and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(copy.deepcopy(self.best_state_dict))
            self._train_chunk(
                self.model, self.train_dataset,
                self.merge_finetune_steps, -999)
            val_loss = self._eval_full_val(self.model)

            # Part4K2 Task 7.1: 更新外层进度条
            outer_pbar.update(
                n=1,
                postfix={"finetune": "done", "val": f"{val_loss:.4f}"},
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                if hasattr(self.model, "state_dict"):
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())

        # 关闭外层进度条
        outer_pbar.close()

        # 6. 加载最佳状态到 model
        if self.best_state_dict is not None and hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(copy.deepcopy(self.best_state_dict))

        # 7. 保存 checkpoint（若提供了 CheckpointManager）
        if self.checkpoint_mgr is not None and self.best_state_dict is not None:
            try:
                # CheckpointManager.save_best 期望 state dict（任意结构）
                self.checkpoint_mgr.save_best({
                    "model_state_dict": self.best_state_dict,
                    "val_loss": float(self.best_val_loss),
                })
            except Exception as e:
                if not self.quiet:
                    print(f"[parallel] 警告：保存 checkpoint 失败：{e}", flush=True)

        # 8. 汇总 history（取每个 chunk 的最终 train/val loss）
        for stat in self.chunk_stats:
            self.history["train_loss"].append(stat["train_loss"])
            self.history["val_loss"].append(stat["val_loss"])
            self.history["steps"].append(stat["steps"])

        # Part4K2 Task 7.2: 训练结束打印总结
        wall_time = time.time() - t_fit_start
        self._print_summary(wall_time)
        return self.history


# ---------------------------------------------------------------------------
# Task 1.7: DistributedTrainer —— 多卡数据并行训练器（占位接口）
# ---------------------------------------------------------------------------


class DistributedTrainer:
    """多卡数据并行训练器（占位接口，API 预留）。

    本类为分布式训练预留统一 API。当前实现为**单进程串行回退**：
    - 若可用 PyTorch 分布式（``torch.distributed``），后续可扩展为真正的
      DDP（DistributedDataParallel）多卡训练。
    - 当前版本在单卡 / CPU 上行为与 ``Trainer`` 一致，仅添加了 rank/world_size
      等分布式元信息与屏障同步接口。

    设计目标
    ========
    - **API 对齐 PyTorch DDP**：``world_size`` / ``rank`` / ``local_rank`` /
      ``barrier()`` / ``all_reduce()`` 等接口，便于后续无缝迁移。
    - **优雅降级**：无 torch.distributed 时自动回退到单进程，不报错。
    - **数据并行预留**：``DistributedSampler`` 风格的数据分片接口预留
      （当前实现为全量数据，不做分片）。

    Args:
        model: ``nn.Module`` 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 可选学习率调度器
        cfg: 训练配置 dict（同 ``Trainer``，额外支持 ``world_size`` / ``rank``）
        device: 设备字符串（``"cpu"`` / ``"cuda:0"`` / ``"npu:0"`` ...）
        world_size: 世界大小（进程数），默认 1
        rank: 当前进程 rank，默认 0
        local_rank: 本机 local rank，默认 0
        backend: 分布式后端（``"nccl"`` / ``"gloo"`` / ``"hccl"``），默认 None

    用法:
        >>> trainer = DistributedTrainer(model, train_loader, val_loader, opt,
        ...                              device="cuda:0", world_size=1, rank=0)
        >>> trainer.fit()

    注意:
        当前版本为占位实现，不启动真正的多进程。真正的 DDP 训练需配合
        ``torchrun`` / ``torch.distributed.init_process_group`` 使用，
        将在后续版本中实现。
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer: Optimizer,
        scheduler=None,
        cfg=None,
        device=None,
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        backend: str = None,
    ):
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.backend = backend
        self.device = str(device) if device is not None else DEFAULT_DEVICE
        # 检测 torch.distributed 可用性
        self._dist_available = has_torch() and _TORCH is not None and _TORCH.distributed.is_available()
        self._dist_initialized = False
        # 内部委托一个 Trainer 实例处理实际训练逻辑
        self._trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            device=self.device,
        )
        # 暴露内部 trainer 的属性以便外部访问
        self.model = self._trainer.model
        self.optimizer = self._trainer.optimizer

    def init_process_group(self, backend: str = None, init_method: str = None):
        """初始化分布式进程组（占位接口）。

        若 PyTorch 分布式可用且 ``world_size > 1``，调用
        ``torch.distributed.init_process_group``；否则跳过（单进程回退）。

        Args:
            backend: 后端类型（``"nccl"`` / ``"gloo"`` / ``"hccl"``）
            init_method: 初始化方法（默认 ``"env://"``）
        """
        if not self._dist_available or self.world_size <= 1:
            if self.rank == 0:
                print("[DistributedTrainer] 单进程模式（world_size=1），"
                      "跳过 init_process_group", flush=True)
            return
        if backend is not None:
            self.backend = backend
        if self.backend is None:
            self.backend = "nccl" if "cuda" in self.device else "gloo"
        if init_method is None:
            init_method = "env://"
        try:
            _TORCH.distributed.init_process_group(
                backend=self.backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank,
            )
            self._dist_initialized = True
            if self.rank == 0:
                print(f"[DistributedTrainer] 初始化进程组 backend={self.backend} "
                      f"world_size={self.world_size} rank={self.rank}", flush=True)
        except Exception as e:
            print(f"[DistributedTrainer] 警告：init_process_group 失败：{e}",
                  flush=True)

    def barrier(self):
        """分布式屏障同步（单进程时为 no-op）。"""
        if self._dist_initialized:
            _TORCH.distributed.barrier()

    def all_reduce(self, tensor, op=None):
        """All-reduce 聚合（单进程时直接返回原 tensor）。

        Args:
            tensor: 待聚合的 Tensor（或标量）
            op: 规约操作（默认 ``SUM``）
        """
        if not self._dist_initialized:
            return tensor
        if op is None:
            op = _TORCH.distributed.ReduceOp.SUM
        # 把 Tensor 的 data 转成 torch.Tensor 进行 all_reduce
        if isinstance(tensor, Tensor):
            t_data = tensor.data
            if _is_torch_data(t_data):
                _TORCH.distributed.all_reduce(t_data, op=op)
            return tensor
        # 标量 / ndarray：包装成 torch.Tensor 再 reduce
        if _TORCH is not None:
            t = _TORCH.tensor(float(tensor))
            _TORCH.distributed.all_reduce(t, op=op)
            return float(t.item()) / self.world_size
        return tensor

    def fit(self):
        """主训练循环（委托内部 Trainer，训练前后加 barrier 同步）。"""
        self.barrier()
        if self.rank == 0:
            print(f"[DistributedTrainer] 开始训练 world_size={self.world_size} "
                  f"rank={self.rank} device={self.device}", flush=True)
        result = self._trainer.fit()
        self.barrier()
        if self.rank == 0:
            print("[DistributedTrainer] 训练完成", flush=True)
        return result

    def evaluate(self):
        """评估（委托内部 Trainer）。"""
        return self._trainer.evaluate()

    @property
    def is_main_process(self) -> bool:
        """当前是否为主进程（rank == 0）。"""
        return self.rank == 0

    def destroy_process_group(self):
        """销毁分布式进程组（单进程时为 no-op）。"""
        if self._dist_initialized:
            try:
                _TORCH.distributed.destroy_process_group()
                self._dist_initialized = False
            except Exception:
                pass
