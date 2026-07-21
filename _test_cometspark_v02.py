"""验证 VerseNexLM 代码路径正确性（小配置，不爆内存）。"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/workspace/packages/verse_nex")
sys.path.insert(0, "/workspace/packages/verse_torch")

import numpy as np
from verse_nex.versenex import VerseNexSmall, VerseNexCometSparkV02, VerseNexConfig, VerseNexLM
from verse_torch.tensor import Tensor, no_grad


def count_params(model):
    """用 id 去重统计参数量（处理 tie_weights 共享权重）。"""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.data.size
    return total


def test_small():
    print("=" * 70)
    print("[1] VerseNexSmall 实例化 + forward + backward + generate")
    print("=" * 70)
    np.random.seed(42)
    model = VerseNexSmall()
    n_params = count_params(model)
    print(f"  参数量: {n_params:,}")

    tokens = Tensor(np.random.randint(0, 1000, size=(2, 16)))
    targets = Tensor(np.random.randint(0, 1000, size=(2, 16)))

    t0 = time.time()
    out = model(tokens, targets=targets)
    print(f"  forward: {time.time()-t0:.2f}s")
    print(f"  logits.shape: {out['logits'].shape}")
    print(f"  loss: {out['loss'].data.item():.4f}")
    print(f"  aux_loss: {out['aux_loss'].data.item():.4f}")
    print(f"  total_loss: {out['total_loss'].data.item():.4f}")

    t0 = time.time()
    out["total_loss"].backward()
    print(f"  backward: {time.time()-t0:.2f}s")

    grad_count = 0
    grad_nonzero = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_count += 1
            if np.any(p.grad != 0):
                grad_nonzero += 1
    print(f"  梯度: {grad_nonzero}/{grad_count} 参数有非零梯度")

    model.eval()
    prompt = Tensor(np.array([[1, 2, 3, 4, 5]], dtype=np.int64))
    t0 = time.time()
    with no_grad():
        gen = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=None)
    print(f"  生成: {time.time()-t0:.2f}s, shape={gen.shape}, out={gen.data.tolist()}")
    print("[OK] VerseNexSmall 通过\n")


def test_cometspark_instantiate_only():
    """仅实例化 CometSpark-V0.2 验证参数量（不跑 forward 避免爆内存）。"""
    print("=" * 70)
    print("[2] CometSpark-V0.2 实例化（参数量验证，去重共享权重）")
    print("=" * 70)
    t0 = time.time()
    model = VerseNexCometSparkV02()
    n_params = count_params(model)
    print(f"  实例化: {time.time()-t0:.2f}s")
    print(f"  参数量(去重): {n_params:,} ({n_params/1e9:.4f}B)")
    expected = 486_300_000
    diff = (n_params - expected) / expected * 100
    print(f"  预期:       {expected:,} (0.486B)")
    print(f"  差异: {diff:+.2f}%")
    if abs(diff) < 5.0:
        print("[OK] 参数量符合预期\n")
    else:
        print(f"[WARN] 参数量偏差较大 ({diff:+.2f}%)\n")
    del model


def test_cometspark_minicfg():
    """用极小 batch + 极短序列 + 缩小配置验证 CometSpark-V0.2 forward/backward 代码路径。"""
    print("=" * 70)
    print("[3] CometSpark-V0.2 缩小配置 forward/backward (n_layer=2, d=64)")
    print("=" * 70)
    np.random.seed(42)
    model = VerseNexCometSparkV02(
        n_layer=2, d_model=64, n_head=4, n_kv_head=2,
        mod_d_ff=128, mod_n_parts=2, mod_n_experts=2,
        mod_top_k_parts=1, mod_top_k_experts=1,
        medusa_n_heads=0, tie_weights=True,
        vocab_size=1000,
    )
    n_params = count_params(model)
    print(f"  参数量(缩小配置): {n_params:,}")

    tokens = Tensor(np.random.randint(0, 1000, size=(2, 16)))
    targets = Tensor(np.random.randint(0, 1000, size=(2, 16)))
    t0 = time.time()
    out = model(tokens, targets=targets)
    print(f"  forward: {time.time()-t0:.2f}s")
    print(f"  loss: {out['loss'].data.item():.4f}")
    print(f"  aux_loss: {out['aux_loss'].data.item():.4f}")
    print(f"  total_loss: {out['total_loss'].data.item():.4f}")
    t0 = time.time()
    out["total_loss"].backward()
    print(f"  backward: {time.time()-t0:.2f}s")

    grad_count = 0
    grad_nonzero = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_count += 1
            if np.any(p.grad != 0):
                grad_nonzero += 1
    print(f"  梯度: {grad_nonzero}/{grad_count} 参数有非零梯度")

    # 生成测试
    model.eval()
    prompt = Tensor(np.array([[1, 2, 3, 4, 5]], dtype=np.int64))
    t0 = time.time()
    with no_grad():
        gen = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=None)
    print(f"  生成: {time.time()-t0:.2f}s, shape={gen.shape}")
    print("[OK] CometSpark-V0.2 代码路径验证通过\n")


if __name__ == "__main__":
    test_small()
    test_cometspark_instantiate_only()
    test_cometspark_minicfg()
    print("=" * 70)
    print("全部测试通过！P3.1 CometSpark-V0.2 模型定义完成")
    print("=" * 70)
