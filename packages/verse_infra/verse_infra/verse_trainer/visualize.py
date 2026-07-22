"""loss 历史可视化：调用 verse_torch.training.plot_loss_curve + ASCII fallback。

从 ``data/demo/train/visualize.py`` 迁入。matplotlib 不可用时
plot_loss_curve 内部自动降级为 ASCII 文本图。
"""

from __future__ import annotations

import json
import os

from verse_torch.training import plot_loss_curve


def visualize(
    loss_history_path: str,
    save_path: str = "loss_curve.png",
) -> str:
    """读取 loss_history.json 并绘制曲线。

    额外打印统计摘要：平均 loss、最佳 loss、loss 下降率等，
    便于在不打开图片的情况下快速了解训练效果。

    Args:
        loss_history_path: loss_history.json 路径
        save_path: 输出图片路径（推荐 .png；无 matplotlib 时自动改 .txt）
    Returns:
        实际写入的文件路径
    """
    with open(loss_history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    train_losses = hist.get("train_losses", [])
    val_losses = hist.get("val_losses", [])
    eval_interval = int(hist.get("eval_interval", 1))

    actual = plot_loss_curve(
        train_losses,
        val_losses,
        save_path,
        eval_interval=eval_interval,
    )

    # 统计摘要
    n_train = len(train_losses)
    n_val = len(val_losses)
    summary = (
        f"[visualize] train_steps={n_train} val_steps={n_val} "
        f"eval_interval={eval_interval}"
    )
    if train_losses:
        initial_loss = float(train_losses[0])
        final_loss = float(train_losses[-1])
        avg_loss = sum(float(v) for v in train_losses) / n_train
        min_loss = float(min(train_losses))
        summary += (
            f" initial_loss={initial_loss:.4f}"
            f" final_loss={final_loss:.4f}"
            f" avg_loss={avg_loss:.4f}"
            f" min_loss={min_loss:.4f}"
        )
        # loss 下降率（百分比）：initial -> final 的下降比例
        if initial_loss > 0:
            decline_rate = (initial_loss - final_loss) / initial_loss * 100.0
            summary += f" decline_rate={decline_rate:.1f}%"
    if val_losses:
        # 优先用 loss_history.json 中记录的 best_val_loss，否则从 val_losses 计算
        best_val = hist.get("best_val_loss", None)
        if best_val is None:
            best_val = float(min(val_losses))
        summary += f" best_val_loss={float(best_val):.4f}"
    print(summary, flush=True)
    print(f"[visualize] loss 曲线已保存到: {actual}", flush=True)
    return actual


__all__ = ["visualize"]
