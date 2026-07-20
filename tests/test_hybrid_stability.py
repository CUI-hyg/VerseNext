"""测试：Hybrid 模式数值稳定性 (Task 2 / Task 3.9).

验证 Mamba-2 / RWKV-7 / HybridLM 在长序列下不出现 NaN / Inf。

背景：
    修复前，hybrid 模式（SSM + Sparse Attention）在 seq_len >= 64 时
    np.exp(log_decay) 数值溢出为 NaN，迫使 CometSpark 退化为纯 transformer。

修复策略：
    1. log_decay 被 clip 到 [-50, 0]
    2. A_log 参数化约束为 A = -softplus(A_log) - 1e-4（严格负、有限）
    3. dt 加上界约束 dt = softplus(dt_raw).clamp(0, 10)
    4. recurrent 路径 A_bar = exp(dt * A) 同样 clip

测试内容：
    - parallel / recurrent 两种模式在不同 seq_len (64, 128, 256) 下无 NaN / Inf
    - parallel 与 recurrent 输出数值一致（atol=1e-3）
    - 模拟 A_log 异常大时仍然不溢出（防御性测试）
    - HybridLM 完整 forward 在 seq_len=128 下无 NaN / Inf

运行：
    python -m pytest tests/test_hybrid_stability.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 确保 verse_torch 与 verse_nex 可导入
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "verse_nex"))

from verse_torch import Tensor, no_grad
from verse_nex import Mamba2Block, RWKV7Block, HybridLM


# ---------------------------------------------------------------------------
# 通用断言工具
# ---------------------------------------------------------------------------


def assert_no_nan_inf(arr: np.ndarray, name: str = "output"):
    """断言数组中无 NaN / Inf。"""
    assert not np.any(np.isnan(arr)), \
        f"{name} contains NaN (count={np.sum(np.isnan(arr))})"
    assert not np.any(np.isinf(arr)), \
        f"{name} contains Inf (count={np.sum(np.isinf(arr))})"


# ---------------------------------------------------------------------------
# Mamba2Block：parallel 模式数值稳定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_mamba2_parallel_no_nan_inf(seq_len):
    """Mamba2Block parallel 模式在不同 seq_len 下无 NaN / Inf。"""
    np.random.seed(42)
    B, D = 2, 64
    model = Mamba2Block(dim=D, d_state=32, n_heads=8, expand=2)
    x = Tensor(np.random.randn(B, seq_len, D).astype(np.float32))

    with no_grad():
        out = model.forward_parallel(x)

    assert out.shape == (B, seq_len, D), f"unexpected shape: {out.shape}"
    assert_no_nan_inf(out.data, name=f"mamba2 parallel seq_len={seq_len}")


# ---------------------------------------------------------------------------
# Mamba2Block：recurrent 模式数值稳定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_mamba2_recurrent_no_nan_inf(seq_len):
    """Mamba2Block recurrent 模式逐 token 解码无 NaN / Inf。"""
    np.random.seed(42)
    B, D = 2, 64
    model = Mamba2Block(dim=D, d_state=32, n_heads=8, expand=2)
    x_data = np.random.randn(B, seq_len, D).astype(np.float32)

    with no_grad():
        state = None
        outs = []
        for t in range(seq_len):
            x_t = Tensor(x_data[:, t:t + 1, :])
            out, state = model.forward_recurrent(x_t, state)
            outs.append(out.data)
        out_arr = np.concatenate(outs, axis=1)

    assert out_arr.shape == (B, seq_len, D)
    assert_no_nan_inf(out_arr, name=f"mamba2 recurrent seq_len={seq_len}")


# ---------------------------------------------------------------------------
# Mamba2Block：parallel 与 recurrent 数值一致性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [64, 128])
def test_mamba2_parallel_recurrent_consistency(seq_len):
    """parallel 与 recurrent 输出应吻合到 atol=1e-3。

    这是 spec 要求的数值一致性约束（float32 下吻合到 1e-3）。
    修复 clip + A_log 约束 + dt 约束后两种模式仍应一致。
    """
    np.random.seed(0)
    B, D = 1, 64
    model = Mamba2Block(dim=D, d_state=32, n_heads=8, expand=2)
    x_data = np.random.randn(B, seq_len, D).astype(np.float32)

    with no_grad():
        # parallel
        out_parallel = model.forward_parallel(Tensor(x_data)).data

        # recurrent
        state = None
        outs = []
        for t in range(seq_len):
            x_t = Tensor(x_data[:, t:t + 1, :])
            out, state = model.forward_recurrent(x_t, state)
            outs.append(out.data)
        out_recurrent = np.concatenate(outs, axis=1)

    assert_no_nan_inf(out_parallel, "parallel")
    assert_no_nan_inf(out_recurrent, "recurrent")
    assert np.allclose(out_parallel, out_recurrent, atol=1e-3), \
        f"parallel/recurrent mismatch at seq_len={seq_len}: " \
        f"max_diff={np.max(np.abs(out_parallel - out_recurrent))}"


# ---------------------------------------------------------------------------
# Mamba2Block：模拟 A_log 异常大的鲁棒性
# ---------------------------------------------------------------------------


def test_mamba2_extreme_a_log_stability():
    """模拟训练中 A_log 学到异常大正值时仍然不溢出。

    修复前：A = -exp(A_log)，A_log 大正 → exp 溢出为 inf → A = -inf
    修复后：A = -softplus(A_log) - 1e-4，A_log 大正 → softplus ≈ A_log，A 仍有限
    """
    np.random.seed(7)
    B, T, D = 1, 128, 64
    model = Mamba2Block(dim=D, d_state=32, n_heads=8, expand=2)

    # 故意把 A_log 设为很大的正值（模拟训练异常）
    with no_grad():
        model.A_log.data = np.full((8,), 100.0, dtype=np.float32)

    x = Tensor(np.random.randn(B, T, D).astype(np.float32))
    with no_grad():
        out = model.forward_parallel(x)

    assert_no_nan_inf(out.data, name="mamba2 parallel with extreme A_log=100")

    # recurrent 模式也应稳定
    with no_grad():
        state = None
        outs = []
        for t in range(T):
            x_t = Tensor(x.data[:, t:t + 1, :])
            out, state = model.forward_recurrent(x_t, state)
            outs.append(out.data)
        out_recurrent = np.concatenate(outs, axis=1)
    assert_no_nan_inf(out_recurrent, name="mamba2 recurrent with extreme A_log=100")


# ---------------------------------------------------------------------------
# RWKV7Block：parallel 模式数值稳定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_rwkv7_parallel_no_nan_inf(seq_len):
    """RWKV7Block parallel 模式在不同 seq_len 下无 NaN / Inf。"""
    np.random.seed(42)
    B, D = 2, 64
    model = RWKV7Block(dim=D, n_head=8)
    x = Tensor(np.random.randn(B, seq_len, D).astype(np.float32))

    with no_grad():
        out = model.forward_parallel(x)

    assert out.shape == (B, seq_len, D), f"unexpected shape: {out.shape}"
    assert_no_nan_inf(out.data, name=f"rwkv7 parallel seq_len={seq_len}")


# ---------------------------------------------------------------------------
# RWKV7Block：recurrent 模式数值稳定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [64, 128])
def test_rwkv7_recurrent_no_nan_inf(seq_len):
    """RWKV7Block recurrent 模式逐 token 解码无 NaN / Inf。"""
    np.random.seed(42)
    B, D = 2, 64
    model = RWKV7Block(dim=D, n_head=8)
    x_data = np.random.randn(B, seq_len, D).astype(np.float32)

    with no_grad():
        state = None
        outs = []
        for t in range(seq_len):
            x_t = Tensor(x_data[:, t:t + 1, :])
            out, state = model.forward_recurrent(x_t, state)
            outs.append(out.data)
        out_arr = np.concatenate(outs, axis=1)

    assert out_arr.shape == (B, seq_len, D)
    assert_no_nan_inf(out_arr, name=f"rwkv7 recurrent seq_len={seq_len}")


# ---------------------------------------------------------------------------
# HybridLM：完整模型 forward 数值稳定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_hybrid_lm_parallel_no_nan_inf(seq_len):
    """HybridLM（SSM + Sparse Attention 混合）parallel 模式无 NaN / Inf。"""
    np.random.seed(42)
    vocab_size = 32
    model = HybridLM(
        vocab_size=vocab_size,
        dim=64,
        n_layers=4,
        sparse_ratio=0.25,
        ssm_kind="mamba2",
        ssm_kwargs={"n_heads": 4, "d_state": 32, "d_conv": 4, "expand": 2},
        sparse_kwargs={
            "n_heads": 4,
            "chunk_size": 16,
            "n_sliding_chunks": 2,
            "topk_chunks": 2,
        },
        sparse_placement="spread",
        tie_weights=True,
    )

    input_ids = Tensor(np.random.randint(0, vocab_size, size=(2, seq_len)).astype(np.int64))
    with no_grad():
        logits = model.forward_parallel(input_ids)

    assert logits.shape == (2, seq_len, vocab_size), f"unexpected shape: {logits.shape}"
    assert_no_nan_inf(logits.data, name=f"hybrid lm parallel seq_len={seq_len}")


def test_hybrid_lm_parallel_recurrent_consistency():
    """HybridLM parallel 与 recurrent 输出在 mamba2 SSM 层下应吻合。

    注意：sparse attention 层在 recurrent 模式下需要 kv_cache，
    因此这里使用 sparse_ratio=0（纯 SSM）以验证一致性。
    """
    np.random.seed(0)
    seq_len = 64
    vocab_size = 16
    model = HybridLM(
        vocab_size=vocab_size,
        dim=64,
        n_layers=2,
        sparse_ratio=0.0,  # 纯 SSM
        ssm_kind="mamba2",
        ssm_kwargs={"n_heads": 4, "d_state": 32, "d_conv": 4, "expand": 2},
        sparse_kwargs={},
        sparse_placement="spread",
        tie_weights=True,
    )

    input_ids_data = np.random.randint(0, vocab_size, size=(1, seq_len)).astype(np.int64)

    with no_grad():
        out_parallel = model.forward_parallel(Tensor(input_ids_data)).data

        states = None
        outs = []
        for t in range(seq_len):
            tok = Tensor(input_ids_data[:, t:t + 1])
            logits, states = model.forward_recurrent(tok, states)
            outs.append(logits.data)
        out_recurrent = np.concatenate(outs, axis=1)

    assert_no_nan_inf(out_parallel, "hybrid parallel")
    assert_no_nan_inf(out_recurrent, "hybrid recurrent")
    assert np.allclose(out_parallel, out_recurrent, atol=1e-3), \
        f"hybrid parallel/recurrent mismatch: " \
        f"max_diff={np.max(np.abs(out_parallel - out_recurrent))}"


# ---------------------------------------------------------------------------
# 防御性：极小 dt / 极大 dt 输入下的鲁棒性
# ---------------------------------------------------------------------------


def test_mamba2_extreme_input_stability():
    """测试极端输入（极大/极小）下不出现 NaN / Inf。"""
    np.random.seed(123)
    B, T, D = 1, 128, 64
    model = Mamba2Block(dim=D, d_state=32, n_heads=8, expand=2)

    # 极大输入（可能使 dt_raw 极大）
    x_large = Tensor(np.full((B, T, D), 100.0, dtype=np.float32))
    with no_grad():
        out_large = model.forward_parallel(x_large)
    assert_no_nan_inf(out_large.data, name="mamba2 parallel with large input=100")

    # 极小（负）输入
    x_small = Tensor(np.full((B, T, D), -100.0, dtype=np.float32))
    with no_grad():
        out_small = model.forward_parallel(x_small)
    assert_no_nan_inf(out_small.data, name="mamba2 parallel with small input=-100")


if __name__ == "__main__":
    # 直接运行：python tests/test_hybrid_stability.py
    sys.exit(pytest.main([__file__, "-v"]))
