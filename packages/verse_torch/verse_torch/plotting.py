"""训练曲线绘制与 loss 收敛评估（plot_loss_curve / compute_loss_rate）。

Part1 Task2 拆分自 training.py：matplotlib 可选 + ASCII 降级的绘图逻辑，
training.py 顶部 re-export 以保持 ``from verse_torch.training import ...`` 兼容。
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["compute_loss_rate", "plot_loss_curve"]


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


def _compute_loss_x(train_losses, val_losses, eval_interval: int = 1):
    """计算 train_x 和 val_x 坐标列表。

    Part4K2.6 Task 1 修复：解决 val_loss 曲线画反的 bug。

    旧实现中 val_x 用 ``min(i * eval_interval, n_train - 1)`` 截断，导致：
    - ParallelTrainer per-chunk 场景（train/val 等长）下多个 val 点堆叠在
     最后一个 step，曲线变形（即"画反"）。
    - per-step 场景下 val_x 超出 train 范围时被截断，同样导致堆叠。

    新策略：
    - ``len(val_losses) == len(train_losses)``：1:1 对齐（ParallelTrainer
      per-chunk 或 eval_interval=1），val_x = train_x = [0, 1, ..., n-1]。
    - ``len(val_losses) < len(train_losses)`` 且 ``eval_interval > 1``：
      per-step 对齐，val_x = [0, eval_interval, 2*eval_interval, ...]，
      **不截断**（允许超出 train 范围，matplotlib 自动扩展 x 轴）。
    - 其他情况（空数据等）：用索引作为 x 坐标。

    Args:
        train_losses: 训练 loss 列表
        val_losses: 验证 loss 列表
        eval_interval: 验证频率（<1 时视为 1）

    Returns:
        ``(train_x, val_x)`` 两个 list[int]
    """
    if eval_interval < 1:
        eval_interval = 1
    n_train = len(train_losses)
    n_val = len(val_losses)

    if n_train == 0:
        # 无训练数据：val 用索引作为 x
        train_x = []
        val_x = list(range(n_val))
    elif n_val == n_train:
        # 1:1 对齐（ParallelTrainer per-chunk 或 eval_interval=1）
        train_x = list(range(n_train))
        val_x = list(range(n_val))
    elif n_val > 0 and eval_interval > 1:
        # per-step：val 对齐到 i*eval_interval，不截断
        train_x = list(range(n_train))
        val_x = [i * eval_interval for i in range(n_val)]
    else:
        train_x = list(range(n_train))
        val_x = list(range(n_val))

    return train_x, val_x


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

    # Part4K2.6 Task 1: 智能 val_x 计算，修复 val_loss 曲线画反的 bug
    if eval_interval < 1:
        eval_interval = 1
    train_x, val_x = _compute_loss_x(train_losses, val_losses, eval_interval)

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
    # Part4K2.6 Task 1: val 的 x 坐标用 _compute_loss_x 计算（不截断），
    # x 轴范围覆盖 train 和 val 的最大 step，避免 val 点堆叠
    x_max_step = max(n_train - 1, val_x[-1] if val_x else 0)
    n_total = x_max_step + 1
    put_curve(train_losses, n_total, "T")
    put_curve(
        val_losses, n_total, "V",
        step_fn=lambda i: val_x[i] if i < len(val_x) else i,
    )

    # 写入文件
    lines = []
    lines.append(f"Loss Curve (ASCII)  range=[{y_min:.4f}, {y_max:.4f}]")
    lines.append(f"T=train  V=val  *=overlap  (width={width}, height={height})")

    # Part4K2.6 Task 1: 添加 y 轴刻度标签（y_max / y_mid / y_min）
    # 标签宽度 label_w 字符，加上 " |" 共 label_w+2 字符（不超过 10）
    label_w = 8

    def _fmt_y_label(val):
        """格式化 y 轴数值标签，右对齐到 label_w 字符。"""
        s = f"{val:.4f}"
        if len(s) > label_w:
            # 数值过大时用科学计数法
            s = f"{val:.2e}"
        return s.rjust(label_w)

    y_mid_val = (y_max + y_min) / 2.0
    mid_row = height // 2
    # 顶部边框
    lines.append(" " * label_w + " +" + "-" * width + "+")
    for r in range(height):
        if r == 0:
            label = _fmt_y_label(y_max)
        elif r == height - 1:
            label = _fmt_y_label(y_min)
        elif r == mid_row:
            label = _fmt_y_label(y_mid_val)
        else:
            label = " " * label_w
        lines.append(f"{label} |" + "".join(canvas[r]) + "|")
    # 底部边框
    lines.append(" " * label_w + " +" + "-" * width + "+")
    lines.append(" " * label_w + "   step →")
    lines.append(f"train_steps={n_train}  val_steps={n_val}  eval_interval={eval_interval}")

    # 附加 val 数值表（让 val 数据即使在密集 train 中也能被精确读出）
    if n_val > 0:
        lines.append("")
        lines.append("val_losses detail:")
        for i, v in enumerate(val_losses):
            # Part4K2.6 Task 1: val 的 step 与 _compute_loss_x 计算的 val_x 保持一致
            step = val_x[i] if i < len(val_x) else i
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
        # Part4K2.6 Task 1: 用 _compute_loss_x 统一计算 val_x（不截断）
        if eval_interval < 1:
            eval_interval = 1
        _, val_x_ascii = _compute_loss_x(train_losses, val_losses, eval_interval)
        _print_val_info(val_losses, val_x_ascii)
        print(f"[plot_loss_curve] loss 曲线已保存到: {txt_path}", flush=True)
        return txt_path

    # matplotlib 可用分支
    fig, ax = plt.subplots(figsize=(10, 6))
    # Part4K2.6 Task 1: 用 _compute_loss_x 统一计算 train_x / val_x（不截断）
    if eval_interval < 1:
        eval_interval = 1
    train_x, val_x = _compute_loss_x(train_losses, val_losses, eval_interval)

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
    # Part4K2.6 Task 1: y 轴标签明确标注 "lower is better"，避免方向歧义
    ax.set_ylabel("loss (lower is better)")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True)
    # Part4K2.6 Task 1: 标注 best_val_loss 的位置（红色箭头）
    if val_losses and all_vals_mpl:
        best_idx = int(min(range(len(val_losses)), key=lambda i: float(val_losses[i])))
        best_val = float(val_losses[best_idx])
        best_x = val_x[best_idx] if best_idx < len(val_x) else best_idx
        # 箭头从上方指向 best 点
        offset = (y_max_mpl - y_min_mpl) * 0.12
        ax.annotate(
            f"best={best_val:.4f}",
            xy=(best_x, best_val),
            xytext=(best_x, best_val + offset),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9,
            color="red",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    _print_val_info(val_losses, val_x)
    print(f"[plot_loss_curve] loss 曲线已保存到: {save_path}", flush=True)
    return save_path
