"""Part1：verse_rs Rust 内核（parallel.rs 编译产物）数值一致性测试。

验证内容：
1. ``verse_rs.batched_matmul`` 与 ``np.matmul`` 数值一致（3Dx2D / 3Dx3D / 2Dx2D）。
2. 非连续内存输入（转置 / 步进视图）正确。
3. bias 广播与 numpy 一致。
4. ``parallel_matmul`` 集成：f32 走 Rust、float64/异常降级回 multiprocessing。
5. ParallelLinear 前向 + 反向梯度与父类 Linear 一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

import verse_torch  # noqa: F401
from verse_torch import Tensor  # noqa: E402
from verse_torch.parallel import parallel_matmul  # noqa: E402

try:
    from verse_torch import verse_rs

    HAS_RS = True
except ImportError:  # pragma: no cover - .so 未构建时跳过 Rust 侧用例
    HAS_RS = False

pytestmark = pytest.mark.skipif(
    not HAS_RS, reason="verse_rs.so 未构建（源码树/纯净环境）"
)


# ---------------------------------------------------------------------------
# 1. Rust 内核数值一致性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((4, 8, 16), (16, 5)),      # 3D x 2D
        ((4, 8, 16), (4, 16, 5)),   # 3D x 3D
        ((1, 8, 16), (16, 5)),      # batch=1
        ((8, 16), (16, 5)),         # 2D x 2D
        ((3, 5, 7), (7, 1)),        # N=1
    ],
)
def test_batched_matmul_matches_numpy(a_shape, b_shape):
    np.random.seed(0)
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    r = verse_rs.batched_matmul(a, b)
    assert np.allclose(r, np.matmul(a, b), atol=1e-4), (
        f"shape {a_shape} x {b_shape} 数值不一致"
    )
    assert r.shape == np.matmul(a, b).shape


def test_batched_matmul_bias():
    np.random.seed(0)
    a = np.random.randn(4, 8, 16).astype(np.float32)
    b = np.random.randn(16, 5).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    r = verse_rs.batched_matmul(a, b, bias=bias)
    assert np.allclose(r, np.matmul(a, b) + bias, atol=1e-4)


def test_non_contiguous_input():
    np.random.seed(0)
    base = np.random.randn(8, 32, 64).astype(np.float32)
    b = np.random.randn(64, 16).astype(np.float32)
    # 步进视图（非 C 连续）
    a_strided = base[:, :, ::2]
    r = verse_rs.batched_matmul(a_strided, b[:32])
    ref = np.matmul(a_strided, b[:32])
    assert np.allclose(r, ref, atol=1e-4)
    # Fortran 连续
    a_f = np.asfortranarray(base[:, :, :32])
    r2 = verse_rs.batched_matmul(a_f, b[:32])
    assert np.allclose(r2, np.matmul(a_f, b[:32]), atol=1e-4)


def test_default_threads_positive():
    assert verse_rs.default_threads() >= 1


def test_dim_mismatch_raises():
    a = np.zeros((3, 4, 5), np.float32)
    b = np.zeros((6, 7), np.float32)
    with pytest.raises(ValueError):
        verse_rs.batched_matmul(a, b)


# ---------------------------------------------------------------------------
# 2. parallel_matmul 集成与降级
# ---------------------------------------------------------------------------


def test_parallel_matmul_rust_path_f32():
    np.random.seed(0)
    a = np.random.randn(6, 8, 16).astype(np.float32)
    b = np.random.randn(16, 4).astype(np.float32)
    r = parallel_matmul(a, b, n_workers=2)
    assert np.allclose(r, np.matmul(a, b), atol=1e-4)
    assert r.shape == (6, 8, 4)


def test_parallel_matmul_f64_falls_back():
    np.random.seed(0)
    a = np.random.randn(6, 8, 16)
    b = np.random.randn(16, 4)
    r = parallel_matmul(a, b, n_workers=2)
    assert np.allclose(r, np.matmul(a, b), atol=1e-12)


def test_parallel_matmul_rust_disabled_falls_back(monkeypatch):
    np.random.seed(0)
    a = np.random.randn(6, 8, 16).astype(np.float32)
    b = np.random.randn(16, 4).astype(np.float32)
    import verse_torch.parallel as P

    monkeypatch.setattr(P, "_VERSE_RS", None)
    r = parallel_matmul(a, b, n_workers=2)
    assert np.allclose(r, np.matmul(a, b), atol=1e-4)


def test_parallel_matmul_tensor_wrapping():
    np.random.seed(0)
    a = np.random.randn(6, 8, 16).astype(np.float32)
    b = np.random.randn(16, 4).astype(np.float32)
    r = parallel_matmul(Tensor(a), b, n_workers=2)
    assert isinstance(r, Tensor)
    assert np.allclose(r.data, np.matmul(a, b), atol=1e-4)


# ---------------------------------------------------------------------------
# 3. ParallelLinear（Rust 路径）前向 + 反向
# ---------------------------------------------------------------------------


def test_parallel_linear_forward_backward():
    """Rust 路径下 ParallelLinear 前向 + 反向与 numpy 参考一致。"""
    from verse_torch.parallel import ParallelLinear

    np.random.seed(0)
    d_in, d_out, batch = 8, 4, 32
    layer = ParallelLinear(d_in, d_out, n_workers=2, batch_threshold=4)
    x = Tensor(np.random.randn(batch, d_in).astype(np.float32), requires_grad=True)
    out = layer(x)
    assert out.shape == (batch, d_out)
    assert np.allclose(out.data, x.data @ layer.weight.data.T + layer.bias.data, atol=1e-4)
    # 反向
    g = Tensor(np.random.randn(batch, d_out).astype(np.float32))
    out.backward(g)
    assert x.grad is not None
    assert np.allclose(x.grad, g.data @ layer.weight.data, atol=1e-3)
    assert layer.weight.grad is not None
    assert np.allclose(layer.weight.grad, g.data.T @ x.data, atol=1e-3)
    assert layer.bias.grad is not None
    assert np.allclose(layer.bias.grad, g.data.sum(axis=0), atol=1e-3)
