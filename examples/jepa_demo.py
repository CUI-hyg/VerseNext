"""Task 4.10: JEPA 简化 demo.

在 8x8 色块网格（即 8 个 4x4 色块组成的 8x8 图像）上训练 I-JEPA 50 步。
显示 loss 下降，并将 loss 曲线保存到 /workspace/verse_data/experiments/jepa_demo/

运行：
    cd /workspace && PYTHONPATH=packages/verse_torch:packages/verse_awm \
        python3 examples/jepa_demo.py
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np

# 让 examples/ 目录能 import verse_torch / verse_awm
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_awm"))

from verse_torch import Tensor, optim
from verse_awm import IJEPA, update_target_encoder, ema_decay_schedule


# ---------------------------------------------------------------------------
# 合成数据：8x8 色块网格
# ---------------------------------------------------------------------------


def make_color_block_dataset(n_samples: int = 200, img_size: int = 8,
                             patch_size: int = 4, seed: int = 42):
    """生成 8x8 色块网格数据.

    每张图像是一个 2x2 网格的 4x4 色块（共 4 个色块），颜色随机。
    shape: (N, 3, 8, 8) float32 in [0, 1]
    """
    rng = np.random.default_rng(seed)
    n_blocks_per_side = img_size // patch_size  # 2
    images = np.zeros((n_samples, 3, img_size, img_size), dtype=np.float32)
    for i in range(n_samples):
        for by in range(n_blocks_per_side):
            for bx in range(n_blocks_per_side):
                color = rng.uniform(0.1, 0.9, size=(3,)).astype(np.float32)
                images[i, :,
                       by * patch_size:(by + 1) * patch_size,
                       bx * patch_size:(bx + 1) * patch_size] = color[:, None, None]
    return images


# ---------------------------------------------------------------------------
# 训练流程
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Task 4.10: I-JEPA Demo (8x8 色块网格)")
    print("=" * 60)

    # 配置
    n_steps = 50
    batch_size = 16
    lr = 1e-3
    out_dir = _REPO_ROOT / "verse_data" / "experiments" / "jepa_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # 1. 生成数据
    print("\n[1] 生成 8x8 色块网格数据...")
    images = make_color_block_dataset(n_samples=200, img_size=8, patch_size=4, seed=42)
    print(f"  数据 shape: {images.shape}, 范围 [{images.min():.3f}, {images.max():.3f}]")

    # 2. 构建模型（极小，因为图像只有 8x8）
    print("\n[2] 构建 IJEPA 模型...")
    model = IJEPA(
        img_size=8, patch_size=4, in_channels=3,
        embed_dim=32, depth=2, n_heads=2,
        predictor_depth=2,
    )
    n_params = sum(p.data.size for p in model.parameters())
    print(f"  参数量: {n_params}")

    # 3. 训练 50 步
    print(f"\n[3] 训练 {n_steps} 步...")
    params = (
        list(model.context_encoder.parameters())
        + list(model.predictor.parameters())
        + [model.pos_embed]
        + list(model.patch_embed.parameters())
    )
    opt = optim.Adam(params, lr=lr)
    rng = np.random.default_rng(0)
    N = images.shape[0]

    losses = []
    ema_decays = []
    t0 = time.time()
    for step in range(n_steps):
        # 采样 batch
        idx = rng.integers(0, N, size=batch_size)
        batch = images[idx]
        batch_t = Tensor(batch.astype(np.float32), requires_grad=False)

        opt.zero_grad()
        loss, metrics = model(batch_t, n_targets=4, rng=rng)
        loss.backward()
        opt.step()

        decay = ema_decay_schedule(step, n_steps)
        update_target_encoder(model.context_encoder, model.target_encoder, decay)
        losses.append(float(loss.data))
        ema_decays.append(decay)

        if step % 5 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            print(f"  step {step:3d}/{n_steps}: loss={loss.data:.4f} "
                  f"decay={decay:.4f} t={elapsed:.1f}s")

    # 4. 输出 loss 曲线
    print("\n[4] loss 曲线总结：")
    print(f"  初始 loss: {losses[0]:.4f}")
    print(f"  终值 loss: {losses[-1]:.4f}")
    print(f"  最小 loss: {min(losses):.4f} (step {int(np.argmin(losses))})")
    print(f"  下降量: {losses[0] - losses[-1]:.4f}")
    print(f"  下降比例: {(losses[0] - losses[-1]) / max(losses[0], 1e-8) * 100:.1f}%")

    # 5. 保存 loss 曲线到 stdout（ASCII）
    print("\n[5] loss 曲线（ASCII）：")
    max_loss = max(losses)
    min_loss = min(losses)
    width = 40
    for i, l in enumerate(losses):
        if i % 2 == 0 or i == n_steps - 1:
            # 归一化到 [0, width]
            if max_loss > min_loss:
                norm = int((l - min_loss) / (max_loss - min_loss) * width)
            else:
                norm = 0
            bar = "#" * norm + "." * (width - norm)
            print(f"  step {i:3d} |{bar}| {l:.4f}")

    # 6. 保存到文件
    log = {
        "losses": losses,
        "ema_decays": ema_decays,
        "config": {
            "img_size": 8, "patch_size": 4, "in_channels": 3,
            "embed_dim": 32, "depth": 2, "n_heads": 2,
            "predictor_depth": 2,
            "n_steps": n_steps, "batch_size": batch_size, "lr": lr,
        },
        "summary": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "loss_drop": losses[0] - losses[-1],
        },
    }
    log_path = out_dir / "loss_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[6] loss 日志已保存到: {log_path}")

    # 7. 判定
    loss_drop = losses[0] - losses[-1]
    passed = loss_drop > 0.05
    print("\n" + "=" * 60)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"  loss_drop={loss_drop:.4f} (>{0.05}? {passed})")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
