"""P2.3 验证：VerseNexLM forward + loss + backward + generate。"""
from __future__ import annotations

import sys

sys.path.insert(0, "/workspace/packages/verse_nex")
sys.path.insert(0, "/workspace/packages/verse_torch")

import numpy as np
from verse_torch.tensor import Tensor
from verse_nex.versenex import VerseNexSmall, VerseNexConfig, VerseNexLM


def test_forward_backward():
    print("\n[1] VerseNexSmall forward + backward + loss...")
    np.random.seed(42)
    model = VerseNexSmall(vocab_size=100)
    print(f"  参数量: {sum(p.data.size for p in model.parameters())}")

    B, T = 2, 8
    tokens = Tensor(np.random.randint(0, 100, (B, T)).astype(np.int64))
    targets = Tensor(np.random.randint(0, 100, (B, T)).astype(np.int64))

    result = model(tokens, targets=targets)
    print(f"  logits shape: {result['logits'].shape}")
    print(f"  aux_logits count: {len(result['aux_logits'])}")
    print(f"  aux_loss: {float(result['aux_loss'].data):.4f}")
    print(f"  main loss: {float(result['loss'].data):.4f}")
    print(f"  medusa_loss: {float(result['medusa_loss'].data):.4f}")
    print(f"  total_loss: {float(result['total_loss'].data):.4f}")

    assert result["logits"].shape == (B, T, 100)
    assert len(result["aux_logits"]) == 2  # medusa_n_heads=2
    assert result["total_loss"].requires_grad

    # backward
    result["total_loss"].backward()
    # 检查梯度
    embed_grad = model.token_embed.weight.grad
    assert embed_grad is not None, "token_embed.weight.grad 为 None"
    print(f"  embed grad norm: {np.linalg.norm(embed_grad):.4f}")
    print("  OK")


def test_generate():
    print("\n[2] VerseNexLM generate (贪婪)...")
    np.random.seed(42)
    model = VerseNexSmall(vocab_size=100)

    B, T = 1, 4
    tokens = Tensor(np.random.randint(0, 100, (B, T)).astype(np.int64))
    print(f"  input tokens: {tokens.data.tolist()}")

    generated = model.generate(tokens, max_new_tokens=5, temperature=0.0)
    print(f"  generated: {generated.tolist()}")
    assert generated.shape == (B, T + 5)
    print("  OK")


def test_kv_cache_generate():
    print("\n[3] VerseNexLM generate with KV cache（增量推理）...")
    np.random.seed(42)
    model = VerseNexSmall(vocab_size=100)

    B, T = 1, 4
    tokens = Tensor(np.random.randint(0, 100, (B, T)).astype(np.int64))
    generated = model.generate(tokens, max_new_tokens=3, temperature=0.0)
    print(f"  KV cache generate: {generated.shape}, tokens={generated.tolist()}")
    assert generated.shape == (B, T + 3)
    print("  OK")


if __name__ == "__main__":
    test_forward_backward()
    test_generate()
    test_kv_cache_generate()
    print("\n" + "=" * 60)
    print("P2.3 验证全部通过 ✓")
    print("=" * 60)
