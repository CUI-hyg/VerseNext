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


# ---------------------------------------------------------------------------
# 4. 优化器 Rust 内核集成（与 NumPy 降级路径多步对拍）
# ---------------------------------------------------------------------------


def _run_optim(opt_factory, steps=4, rs_enabled=True):
    """以固定种子跑 N 步优化器，返回 (param_list, state_list)。"""
    rng = np.random.default_rng(42)
    shapes = [(8,), (4, 6), (2, 3, 4)]
    params = [
        Tensor(rng.standard_normal(s).astype(np.float32), requires_grad=True)
        for s in shapes
    ]
    import verse_torch.optim as O

    old = O._VERSE_RS
    try:
        O._VERSE_RS = verse_rs if rs_enabled else None
        opt = opt_factory(params)
        for _ in range(steps):
            for p in params:
                p.grad = rng.standard_normal(p.shape).astype(np.float32)
            opt.step()
    finally:
        O._VERSE_RS = old
    return [p.data.copy() for p in params]


@pytest.mark.parametrize(
    "opt_factory",
    [
        lambda ps: verse_torch.optim.SGD(ps, lr=0.01, momentum=0.9),
        lambda ps: verse_torch.optim.SGD(ps, lr=0.01, momentum=0.9, weight_decay=0.01, nesterov=True),
        lambda ps: verse_torch.optim.SGD(ps, lr=0.01, momentum=0.9, dampening=0.1, weight_decay=0.01),
        lambda ps: verse_torch.optim.Adam(ps, lr=1e-3, weight_decay=0.01),
        lambda ps: verse_torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.01),
        lambda ps: verse_torch.optim.NAdamW(ps, lr=1e-3, weight_decay=0.01),
        lambda ps: verse_torch.optim.RMSProp(ps, lr=1e-2, momentum=0.9, centered=True),
        lambda ps: verse_torch.optim.RMSProp(ps, lr=1e-2, momentum=0.9),
        lambda ps: verse_torch.optim.RMSProp(ps, lr=1e-2, centered=True),
    ],
)
def test_optimizer_rust_matches_numpy(opt_factory):
    """Rust 内核路径与 NumPy 降级路径多步更新完全一致。"""
    rs = _run_optim(opt_factory, rs_enabled=True)
    np_ = _run_optim(opt_factory, rs_enabled=False)
    for p_rs, p_np in zip(rs, np_):
        assert np.allclose(p_rs, p_np, atol=1e-6), f"param mismatch: {p_rs} vs {p_np}"


def _run_lion(rs_enabled):
    """固定种子跑 4 步 Lion，返回参数列表。"""
    from verse_torch.optim_extras import Lion
    import verse_torch.optim_extras as OE
    import verse_torch.optim as O

    rng = np.random.default_rng(7)
    shapes = [(5,), (3, 4)]
    params = [
        Tensor(rng.standard_normal(s).astype(np.float32), requires_grad=True)
        for s in shapes
    ]
    old_o, old_e = O._VERSE_RS, OE._VERSE_RS
    try:
        O._VERSE_RS = verse_rs if rs_enabled else None
        OE._VERSE_RS = verse_rs if rs_enabled else None
        opt = Lion(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)
        for _ in range(4):
            for p in params:
                p.grad = rng.standard_normal(p.shape).astype(np.float32)
            opt.step()
    finally:
        O._VERSE_RS, OE._VERSE_RS = old_o, old_e
    return [p.data.copy() for p in params]


def test_lion_rust_matches_numpy():
    """Lion（optim_extras）Rust 路径与 NumPy 降级路径一致。"""
    for p_rs, p_np in zip(_run_lion(True), _run_lion(False)):
        assert np.allclose(p_rs, p_np, atol=1e-6)


def test_optimizer_f64_falls_back():
    """float64 参数自动降级 NumPy 路径（不报错且语义一致）。"""
    import verse_torch.optim as O

    def run(rs_enabled):
        rng = np.random.default_rng(3)
        p = Tensor(rng.standard_normal((4, 6)), requires_grad=True)  # float64
        old = O._VERSE_RS
        try:
            O._VERSE_RS = verse_rs if rs_enabled else None
            opt = O.AdamW([p], lr=1e-3, weight_decay=0.01)
            for _ in range(2):
                p.grad = rng.standard_normal((4, 6))
                opt.step()
        finally:
            O._VERSE_RS = old
        return p.data.copy()

    res = run(True)
    ref = run(False)
    assert res.dtype == np.float64
    assert np.allclose(res, ref, atol=1e-12)
