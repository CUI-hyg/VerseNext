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

from .tensor import Tensor, no_grad
from .optim import Optimizer, LambdaLR, warmup_cosine_lr  # noqa: F401  重新导出方便用户

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

    内部实现：log_softmax + NLL，对 ``ignore_index`` 位置不计入 loss 与梯度。

    Args:
        logits: 形状 (B, T, V) 或 (N, V) 的未归一化预测
        targets: 形状 (B, T) 或 (N,) 的整型类别索引，
                 可以是 Tensor / np.ndarray / list
        ignore_index: 待忽略的标签值（默认 -100），不参与 loss 计算
        label_smoothing: 标签平滑系数（默认 0.0 关闭）。>0 时将 hard target
            与均匀分布混合，``loss = (1-ε)·CE_hard + ε·CE_uniform``，
            起到正则化、缓解过拟合的作用。

    Returns:
        标量 Tensor，支持 backward
    """
    if not isinstance(logits, Tensor):
        logits = Tensor(logits, requires_grad=True)

    # 把 targets 转为 int64 ndarray
    if isinstance(targets, Tensor):
        targets_np = targets.data
    else:
        targets_np = np.asarray(targets)
    targets_np = targets_np.astype(np.int64)

    # 自动 reshape 为 (N, V) / (N,)
    if logits.ndim > 2:
        V = logits.shape[-1]
        logits = logits.reshape(-1, V)
        targets_np = targets_np.reshape(-1)

    N = logits.shape[0]
    # log_softmax 沿最后一维
    log_probs = logits.log_softmax(dim=-1)  # (N, V)

    # 计算 ignore_index mask（转为整数索引以便 __getitem__ 反向稳定）
    mask = (targets_np != ignore_index)  # (N,) bool
    valid_idx = np.where(mask)[0]  # (n_valid,) int
    n_valid = int(valid_idx.shape[0])

    if n_valid == 0:
        # 所有位置都被忽略：返回 0 标量但保持计算图连接，避免 backward 报错
        return log_probs.sum() * 0.0

    # 选取有效样本（用整数索引，反向 add.at 行为更明确）
    valid_log_probs = log_probs[valid_idx]  # (n_valid, V)
    valid_targets = targets_np[valid_idx]  # (n_valid,)

    # 选取每个样本对应类别的 log_prob
    selected = valid_log_probs[np.arange(n_valid), valid_targets]  # (n_valid,)

    # 标签平滑：loss = (1-ε)·CE_hard + ε·CE_uniform
    if label_smoothing is not None and label_smoothing > 0.0:
        hard_loss = -selected.mean()
        # 均匀分布部分：所有类别 log_prob 的平均（等价于 -mean(sum_V log_probs / V)）
        uniform_loss = -valid_log_probs.mean()
        loss = (1.0 - label_smoothing) * hard_loss + label_smoothing * uniform_loss
    else:
        # 负平均
        loss = -selected.mean()
    return loss


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

    def put_curve(values, n_total, char):
        n_v = len(values)
        if n_v == 0:
            return
        for i, v in enumerate(values):
            # x 映射到 [0, width-1]
            if n_total <= 1:
                x = 0
            else:
                x = int(i * (width - 1) / max(1, n_total - 1))
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
    put_curve(train_losses, n_train, "T")
    put_curve(val_losses, max(n_train, n_val), "V")

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
    except Exception:
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
        val_x_ascii = [i * eval_interval for i in range(len(val_losses))]
        if val_x_ascii and val_x_ascii[-1] >= len(train_losses):
            val_x_ascii = [min(x, len(train_losses) - 1) for x in val_x_ascii]
        _print_val_info(val_losses, val_x_ascii)
        return txt_path

    # matplotlib 可用分支
    fig, ax = plt.subplots(figsize=(10, 6))
    train_x = list(range(len(train_losses)))
    # val 的 x 坐标按 eval_interval 对齐（不超过 train_x 范围）
    if eval_interval < 1:
        eval_interval = 1
    val_x = [i * eval_interval for i in range(len(val_losses))]
    if val_x and val_x[-1] >= len(train_losses):
        # 防止超出，按比例缩放到 train 范围内
        val_x = [min(x, len(train_losses) - 1) for x in val_x]

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
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    _print_val_info(val_losses, val_x)
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
        use_tqdm = self.enable_progress_bar and _HAS_TQDM
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

            logits = self.model(x)
            loss = cross_entropy_loss(
                logits, y, label_smoothing=self.label_smoothing
            )
            loss.backward()

            self.grad_accum.step()
            if self.grad_accum.should_step():
                # 梯度裁剪：在 optimizer.step 前裁剪累积梯度的总范数
                if self.grad_clip > 0:
                    clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            loss_val = _scalar(loss)
            self.train_losses.append(loss_val)
            step_times.append(time.time() - t_step)

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
            if (
                not use_tqdm
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
        wall = time.time() - t_train_start
        n_done = len(self.train_losses)
        avg_step = wall / n_done if n_done > 0 else 0.0
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
        history = {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "best_val_loss": self.best_val_loss,
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
        plot_loss_curve(
            self.train_losses,
            self.val_losses,
            curve_path,
            eval_interval=self.eval_interval,
        )


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
    # 重新导出 optim 中新增项，方便用户从 training 一次性导入
    "LambdaLR",
    "warmup_cosine_lr",
]
