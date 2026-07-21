"""loss 历史可视化：调用 verse_torch.training.plot_loss_curve + ASCII fallback。

matplotlib 不可用时 plot_loss_curve 内部自动降级为 ASCII 文本图。
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

    # 调用 verse_torch 训练模块的 plot_loss_curve
    # 若 matplotlib 不可用会自动降级为 ASCII（保存为 .txt）
    actual = plot_loss_curve(
        train_losses,
        val_losses,
        save_path,
        eval_interval=eval_interval,
    )

    # 额外打印简短摘要到 stdout
    n_train = len(train_losses)
    n_val = len(val_losses)
    summary = (
        f"[visualize] train_steps={n_train} val_steps={n_val} "
        f"eval_interval={eval_interval}"
    )
    if train_losses:
        summary += (
            f" initial_loss={train_losses[0]:.4f}"
            f" final_loss={train_losses[-1]:.4f}"
        )
    if val_losses:
        summary += f" best_val_loss={hist.get('best_val_loss', float('nan')):.4f}"
    print(summary, flush=True)
    print(f"[visualize] loss 曲线已保存到: {actual}", flush=True)
    return actual


__all__ = ["visualize"]
