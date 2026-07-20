"""Task 6.1: VerseTorch 算子单元测试 + 有限差分梯度检查.

覆盖范围：
- 元素级：add, sub, mul, div, pow, exp, log, relu, gelu, sigmoid, tanh, neg
- shape：reshape, transpose, permute, slice, expand, view, flatten
- reduction：sum, mean, max, min, argmax, var
- matmul（含 batched matmul，2D×2D, 3D×3D, 广播 batched matmul）
- broadcasting：所有元素级算子在 broadcast 形状下的反向梯度

每个算子检查：
- 正向：与 NumPy 直接计算结果一致（atol=1e-6）
- 反向：有限差分梯度检查（相对误差 ≤ 1e-4）

运行方式：
    python3 tests/test_unit_operators.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

# 正向：绝对误差阈值
ATOL_FORWARD = 1e-6
# 反向：相对误差阈值（数值梯度 vs 解析梯度）
RTOL_GRAD = 1e-4
# 有限差分步长
EPS = 1e-5
# 梯度检查使用的 dtype（float64 以提升数值精度）
GRAD_DTYPE = np.float64

# 统计
_PASS = 0
_FAIL = 0
_SKIP = 0
_FAILURES: list[str] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    """记录一次检查结果。"""
    global _PASS, _FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"{name}: {detail}")
    msg = f"  [{status}] {name}"
    if detail and not passed:
        msg += f"  -> {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# 有限差分梯度计算
# ---------------------------------------------------------------------------


def numeric_grad(f, x: Tensor, eps: float = EPS) -> np.ndarray:
    """对函数 f(x) -> scalar 求 x 的数值梯度（中心差分）。

    f 应返回标量 Tensor 或可调用 .item() 的对象。
    直接修改 x.numpy()（即 x.data）来扰动输入。

    Args:
        f: 输入 Tensor，输出标量 Tensor 的函数
        x: 输入 Tensor（requires_grad=True，float64 推荐）
        eps: 扰动步长
    Returns:
        与 x.shape 相同的 np.ndarray 数值梯度
    """
    grad = np.zeros_like(x.numpy())
    it = np.nditer(x.numpy(), flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x.numpy()[idx])

        x.numpy()[idx] = orig + eps
        out_plus = f(x)
        f_plus = float(out_plus.item())

        x.numpy()[idx] = orig - eps
        out_minus = f(x)
        f_minus = float(out_minus.item())

        x.numpy()[idx] = orig  # 恢复
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def analytic_grad(f, x: Tensor) -> np.ndarray:
    """计算解析梯度：构建计算图，反向传播，返回 x.grad。

    Args:
        f: 输入 Tensor，输出标量 Tensor 的函数
        x: 输入 Tensor（requires_grad=True）
    Returns:
        与 x.shape 相同的 np.ndarray 解析梯度
    """
    # 每次调用都重置 grad 与计算图
    x.grad = None
    y = f(x)
    # 反向传播：标量输出默认 grad=1
    y.backward()
    return np.array(x.grad, copy=True)


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个数组的最大相对误差。

    使用 CS231n 风格公式：
        rel = |a - b| / max(|a| + |b|, 1e-8)

    当 a、b 均接近 0 时，分母退化到 1e-8，避免假阳性。
    """
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-8)
    return float(np.max(diff / denom))


def check_forward(name: str, verse_fn, numpy_fn, x_data, *args, **kwargs):
    """检查正向：verse_fn(x, *args) 与 numpy_fn(x.data, *args) 一致。

    atol = ATOL_FORWARD
    """
    try:
        x = Tensor(x_data, requires_grad=False)
        y_verse = verse_fn(x, *args, **kwargs)
        y_np = numpy_fn(x_data, *args, **kwargs)
        if isinstance(y_verse, Tensor):
            y_verse_arr = y_verse.numpy()
        else:
            y_verse_arr = np.asarray(y_verse)
        max_diff = float(np.max(np.abs(y_verse_arr - y_np)))
        passed = max_diff <= ATOL_FORWARD
        _record(name + " [forward]", passed,
                f"max_diff={max_diff:.3e}" if not passed else "")
    except Exception as e:
        _record(name + " [forward]", False, f"exception: {e!r}")


def check_grad(name: str, f, x_data):
    """检查反向：解析梯度 vs 数值梯度，相对误差 ≤ RTOL_GRAD。

    Args:
        name: 测试名
        f: 输入 Tensor -> 标量 Tensor 的函数
        x_data: numpy 数组（推荐 float64）
    """
    try:
        x = Tensor(np.array(x_data, copy=True), requires_grad=True)
        ag = analytic_grad(f, x)
        # 数值梯度：用同一个 x（已恢复 grad=None），重新构建一份用于扰动
        x2 = Tensor(np.array(x_data, copy=True), requires_grad=True)
        ng = numeric_grad(f, x2, eps=EPS)
        err = rel_error(ag, ng)
        passed = err <= RTOL_GRAD
        _record(name + " [grad]", passed,
                f"rel_err={err:.3e}" if not passed else "")
    except Exception as e:
        _record(name + " [grad]", False, f"exception: {e!r}")


# ---------------------------------------------------------------------------
# 测试数据生成
# ---------------------------------------------------------------------------


def randn(*shape, dtype=GRAD_DTYPE, seed=None) -> np.ndarray:
    """生成标准正态分布数组。"""
    if seed is not None:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(dtype)
    return np.random.standard_normal(shape).astype(dtype)


def randn_positive(*shape, dtype=GRAD_DTYPE, seed=None) -> np.ndarray:
    """生成正数数组（用于 log / pow 等需要正输入的算子）。"""
    arr = randn(*shape, dtype=dtype, seed=seed)
    return np.abs(arr) + 0.5  # 保证 > 0.5


# ---------------------------------------------------------------------------
# 元素级算子测试
# ---------------------------------------------------------------------------


def test_elementwise():
    print("\n=== Elementwise Operators ===")
    np.random.seed(42)

    # add
    x = randn(4, 5, seed=1)
    y = randn(4, 5, seed=2)
    check_forward("add", lambda a, b: a + b, lambda a, b: a + b, x, y)
    check_grad("add", lambda a: (a + Tensor(y)).sum(), x)

    # sub
    check_forward("sub", lambda a, b: a - b, lambda a, b: a - b, x, y)
    check_grad("sub", lambda a: (a - Tensor(y)).sum(), x)

    # mul
    check_forward("mul", lambda a, b: a * b, lambda a, b: a * b, x, y)
    check_grad("mul", lambda a: (a * Tensor(y)).sum(), x)

    # div
    y_safe = randn_positive(4, 5, seed=2)  # 避免除零
    check_forward("div", lambda a, b: a / b, lambda a, b: a / b, x, y_safe)
    check_grad("div", lambda a: (a / Tensor(y_safe)).sum(), x)

    # pow (scalar power)
    check_forward("pow_scalar", lambda a, p: a ** p, lambda a, p: a ** p,
                  randn_positive(4, 5, seed=1), 3.0)
    check_grad("pow_scalar", lambda a: (a ** 3.0).sum(), randn_positive(4, 5, seed=1))

    # pow (tensor power): a ** b
    a_pos = randn_positive(4, 5, seed=3)
    b_small = randn(4, 5, seed=4) * 0.5 + 1.5  # 1.0~2.0 范围
    # 注意：check_forward 的第二个参数会原样传给 verse_fn 与 numpy_fn。
    # verse_fn 内 a ** b_small 时，Tensor.__pow__ 检测到非 Tensor 标量会走标量路径，
    # 但 b_small 是 ndarray 会触发 float() 错误。所以这里把 b 包装成 Tensor。
    check_forward("pow_tensor",
                  lambda a, b: a ** Tensor(b), lambda a, b: a ** b,
                  a_pos, b_small)
    check_grad("pow_tensor", lambda a: (a ** Tensor(b_small)).sum(), a_pos)

    # exp
    x_small = randn(4, 5, seed=5) * 0.5  # 限制范围避免数值过大
    check_forward("exp", lambda a: a.exp(), lambda a: np.exp(a), x_small)
    check_grad("exp", lambda a: a.exp().sum(), x_small)

    # log
    x_pos = randn_positive(4, 5, seed=6)
    check_forward("log", lambda a: a.log(), lambda a: np.log(a), x_pos)
    check_grad("log", lambda a: a.log().sum(), x_pos)

    # relu
    x_relu = randn(4, 5, seed=7)
    check_forward("relu", lambda a: a.relu(), lambda a: np.maximum(a, 0), x_relu)
    # relu 梯度在 0 处为 subgradient，避免取到 0：用偏移
    x_relu_grad = randn(4, 5, seed=7) + 0.01  # 几乎不会刚好为 0
    check_grad("relu", lambda a: a.relu().sum(), x_relu_grad)

    # gelu
    x_gelu = randn(4, 5, seed=8) * 0.8
    c = np.sqrt(2.0 / np.pi)
    inner_np = c * (x_gelu + 0.044715 * x_gelu ** 3)
    gelu_np = 0.5 * x_gelu * (1.0 + np.tanh(inner_np))
    check_forward("gelu", lambda a: a.gelu(), lambda a: gelu_np, x_gelu)
    check_grad("gelu", lambda a: a.gelu().sum(), x_gelu)

    # sigmoid
    x_sig = randn(4, 5, seed=9) * 0.8
    sig_np = 1.0 / (1.0 + np.exp(-x_sig))
    check_forward("sigmoid", lambda a: a.sigmoid(), lambda a: sig_np, x_sig)
    check_grad("sigmoid", lambda a: a.sigmoid().sum(), x_sig)

    # tanh
    x_tanh = randn(4, 5, seed=10) * 0.8
    check_forward("tanh", lambda a: a.tanh(), lambda a: np.tanh(a), x_tanh)
    check_grad("tanh", lambda a: a.tanh().sum(), x_tanh)

    # neg
    check_forward("neg", lambda a: -a, lambda a: -a, x)
    check_grad("neg", lambda a: (-a).sum(), x)


# ---------------------------------------------------------------------------
# Broadcasting 反向梯度测试
# ---------------------------------------------------------------------------


def test_broadcasting_grad():
    print("\n=== Broadcasting Gradient ===")

    # (3,) + (2, 3) -> (2, 3)
    x1 = randn(3, seed=11)
    x2 = randn(2, 3, seed=12)
    check_grad("broadcast_add_row", lambda a: (a + Tensor(x2)).sum(), x1)
    check_grad("broadcast_add_col", lambda a: (Tensor(x2) + a).sum(), x1)

    # (2, 1) * (1, 3) -> (2, 3)
    x_row = randn(2, 1, seed=13)
    x_col = randn(1, 3, seed=14)
    check_grad("broadcast_mul_row", lambda a: (a * Tensor(x_col)).sum(), x_row)
    check_grad("broadcast_mul_col", lambda a: (Tensor(x_row) * a).sum(), x_col)

    # (4,) + (3, 4) -> (3, 4)
    xv = randn(4, seed=15)
    xm = randn(3, 4, seed=16)
    check_grad("broadcast_sub_vec", lambda a: (a - Tensor(xm)).sum(), xv)
    check_grad("broadcast_sub_mat", lambda a: (Tensor(xv) - a).sum(), xm)

    # (5,) / (2, 5) -> (2, 5)
    xv5 = randn(5, seed=17)
    xm25 = randn_positive(2, 5, seed=18)  # 避免除零
    check_grad("broadcast_div_vec", lambda a: (a / Tensor(xm25)).sum(), xv5)
    check_grad("broadcast_div_mat", lambda a: (Tensor(xv5) / a).sum(), xm25)

    # (2, 3, 4) + (4,) -> (2, 3, 4)
    x3d = randn(2, 3, 4, seed=19)
    x1d = randn(4, seed=20)
    check_grad("broadcast_add_3d", lambda a: (a + Tensor(x1d)).sum(), x3d)
    check_grad("broadcast_add_1d", lambda a: (Tensor(x3d) + a).sum(), x1d)

    # scalar + tensor (Python scalar 算子)
    xs = randn(3, 4, seed=21)
    check_grad("broadcast_add_scalar", lambda a: (a + 2.5).sum(), xs)
    check_grad("broadcast_radd_scalar", lambda a: (2.5 + a).sum(), xs)
    check_grad("broadcast_mul_scalar", lambda a: (a * 1.5).sum(), xs)
    check_grad("broadcast_rmul_scalar", lambda a: (1.5 * a).sum(), xs)
    check_grad("broadcast_div_scalar", lambda a: (a / 2.0).sum(), xs)
    check_grad("broadcast_rdiv_scalar", lambda a: (2.0 / a).sum(),
               randn_positive(3, 4, seed=21))


# ---------------------------------------------------------------------------
# Shape 算子测试
# ---------------------------------------------------------------------------


def test_shape_ops():
    print("\n=== Shape Operators ===")
    np.random.seed(42)

    # reshape
    x = randn(2, 3, 4, seed=1)
    check_forward("reshape", lambda a, s: a.reshape(s), lambda a, s: a.reshape(s),
                  x, (6, 4))
    check_grad("reshape", lambda a: a.reshape(6, 4).sum(), x)

    # view (alias for reshape)
    check_forward("view", lambda a, s: a.view(s), lambda a, s: a.reshape(s),
                  x, (24,))
    check_grad("view", lambda a: a.view(24).sum(), x)

    # transpose (full)
    check_forward("transpose_full", lambda a: a.transpose(),
                  lambda a: a.T, x)
    check_grad("transpose_full", lambda a: a.transpose().sum(), x)

    # transpose (swap axes)
    check_forward("transpose_swap", lambda a, d0, d1: a.transpose(d0, d1),
                  lambda a, d0, d1: np.swapaxes(a, d0, d1),
                  x, 0, 2)
    check_grad("transpose_swap", lambda a: a.transpose(0, 2).sum(), x)

    # permute
    check_forward("permute", lambda a, p: a.permute(p),
                  lambda a, p: np.transpose(a, p), x, (1, 2, 0))
    check_grad("permute", lambda a: a.permute(1, 2, 0).sum(), x)

    # slice (basic int + slice)
    check_forward("slice_basic", lambda a: a[1:3, :2, 1:3],
                  lambda a: a[1:3, :2, 1:3], x)
    check_grad("slice_basic", lambda a: a[1:3, :2, 1:3].sum(), x)

    # slice (integer index along axis 0)
    check_forward("slice_int", lambda a: a[1], lambda a: a[1], x)
    check_grad("slice_int", lambda a: a[1].sum(), x)

    # slice (advanced indexing) - x.shape = (2, 3, 4)，axis 0 仅允许索引 0/1
    idx = np.array([0, 1, 0])  # 允许重复索引
    check_forward("slice_adv", lambda a: a[idx], lambda a: a[idx], x)
    check_grad("slice_adv", lambda a: a[idx].sum(), x)

    # expand
    x1 = randn(1, 3, seed=2)
    check_forward("expand", lambda a, s: a.expand(s),
                  lambda a, s: np.broadcast_to(a, s), x1, (4, 3))
    check_grad("expand", lambda a: a.expand(4, 3).sum(), x1)

    # flatten
    check_forward("flatten", lambda a: a.flatten(), lambda a: a.flatten(), x)
    check_grad("flatten", lambda a: a.flatten().sum(), x)

    # flatten with start_dim
    check_forward("flatten_start", lambda a: a.flatten(1), lambda a: a.reshape(a.shape[0], -1), x)
    check_grad("flatten_start", lambda a: a.flatten(1).sum(), x)

    # squeeze / unsqueeze
    x_sq = randn(1, 3, 1, 4, seed=3)
    check_forward("squeeze", lambda a: a.squeeze(),
                  lambda a: np.squeeze(a), x_sq)
    check_grad("squeeze", lambda a: a.squeeze().sum(), x_sq)

    x_unsq = randn(3, 4, seed=4)
    check_forward("unsqueeze", lambda a, d: a.unsqueeze(d),
                  lambda a, d: np.expand_dims(a, d), x_unsq, 0)
    check_grad("unsqueeze", lambda a: a.unsqueeze(0).sum(), x_unsq)


# ---------------------------------------------------------------------------
# Reduction 算子测试
# ---------------------------------------------------------------------------


def test_reduction_ops():
    print("\n=== Reduction Operators ===")
    np.random.seed(42)

    x = randn(3, 4, seed=1)

    # sum (all)
    check_forward("sum_all", lambda a: a.sum(), lambda a: a.sum(), x)
    check_grad("sum_all", lambda a: a.sum(), x)

    # sum (dim)
    check_forward("sum_dim", lambda a, d: a.sum(d), lambda a, d: a.sum(axis=d), x, 1)
    check_grad("sum_dim", lambda a: a.sum(1).sum(), x)

    # sum (dim, keepdim)
    check_forward("sum_keepdim", lambda a, d: a.sum(d, keepdim=True),
                  lambda a, d: a.sum(axis=d, keepdims=True), x, 1)
    check_grad("sum_keepdim", lambda a: a.sum(1, keepdim=True).sum(), x)

    # mean (all)
    check_forward("mean_all", lambda a: a.mean(), lambda a: a.mean(), x)
    check_grad("mean_all", lambda a: a.mean(), x)

    # mean (dim)
    check_forward("mean_dim", lambda a, d: a.mean(d), lambda a, d: a.mean(axis=d), x, 0)
    check_grad("mean_dim", lambda a: a.mean(0).sum(), x)

    # max (all)
    check_forward("max_all", lambda a: a.max(), lambda a: a.max(), x)
    check_grad("max_all", lambda a: a.max(), x)

    # max (dim) - 注意 max 沿 dim 返回值（不含 indices）
    check_forward("max_dim", lambda a, d: a.max(d), lambda a, d: a.max(axis=d), x, 1)
    check_grad("max_dim", lambda a: a.max(1).sum(), x)

    # min (all)
    check_forward("min_all", lambda a: a.min(), lambda a: a.min(), x)
    check_grad("min_all", lambda a: a.min(), x)

    # min (dim)
    check_forward("min_dim", lambda a, d: a.min(d), lambda a, d: a.min(axis=d), x, 0)
    check_grad("min_dim", lambda a: a.min(0).sum(), x)

    # argmax - 不可微，只检查正向
    check_forward("argmax_all", lambda a: a.argmax(),
                  lambda a: np.argmax(a), x)
    check_forward("argmax_dim", lambda a, d: a.argmax(d),
                  lambda a, d: np.argmax(a, axis=d), x, 1)

    # var (all) - verse_torch 默认 unbiased=True (ddof=1)
    check_forward("var_all", lambda a: a.var(),
                  lambda a: a.var(ddof=1), x)
    check_grad("var_all", lambda a: a.var(), x)

    # var (dim)
    check_forward("var_dim", lambda a, d: a.var(d), lambda a, d: a.var(axis=d, ddof=1), x, 1)
    check_grad("var_dim", lambda a: a.var(1).sum(), x)


# ---------------------------------------------------------------------------
# Matmul 测试
# ---------------------------------------------------------------------------


def test_matmul():
    print("\n=== Matmul Operators ===")
    np.random.seed(42)

    # 2D x 2D
    a = randn(3, 4, seed=1)
    b = randn(4, 5, seed=2)
    check_forward("matmul_2d", lambda x, y: x @ y, lambda x, y: x @ y, a, b)
    check_grad("matmul_2d_a", lambda x: (x @ Tensor(b)).sum(), a)
    check_grad("matmul_2d_b", lambda y: (Tensor(a) @ y).sum(), b)

    # 3D x 3D (batched)
    a3 = randn(2, 3, 4, seed=3)
    b3 = randn(2, 4, 5, seed=4)
    check_forward("matmul_3d", lambda x, y: x @ y, lambda x, y: x @ y, a3, b3)
    check_grad("matmul_3d_a", lambda x: (x @ Tensor(b3)).sum(), a3)
    check_grad("matmul_3d_b", lambda y: (Tensor(a3) @ y).sum(), b3)

    # 广播 batched: (B, M, K) @ (K, N) -> (B, M, N)
    a_b = randn(2, 3, 4, seed=5)
    b_2d = randn(4, 5, seed=6)
    check_forward("matmul_broadcast", lambda x, y: x @ y,
                  lambda x, y: x @ y, a_b, b_2d)
    check_grad("matmul_broadcast_a", lambda x: (x @ Tensor(b_2d)).sum(), a_b)
    check_grad("matmul_broadcast_b", lambda y: (Tensor(a_b) @ y).sum(), b_2d)

    # 1D x 1D (dot product)
    a1 = randn(5, seed=7)
    b1 = randn(5, seed=8)
    check_forward("matmul_1d", lambda x, y: x @ y, lambda x, y: x @ y, a1, b1)
    check_grad("matmul_1d_a", lambda x: (x @ Tensor(b1)).sum(), a1)
    check_grad("matmul_1d_b", lambda y: (Tensor(a1) @ y).sum(), b1)

    # matmul() method alias
    check_forward("matmul_method", lambda x, y: x.matmul(y),
                  lambda x, y: x @ y, a, b)


# ---------------------------------------------------------------------------
# 损失函数 / Softmax 等额外检查（顺带）
# ---------------------------------------------------------------------------


def test_extras():
    print("\n=== Extra: softmax / log_softmax ===")
    np.random.seed(42)

    x = randn(3, 4, seed=1)

    # softmax
    def _softmax_np(a, dim=-1):
        a = a - a.max(axis=dim, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=dim, keepdims=True)
    check_forward("softmax", lambda a: a.softmax(-1),
                  lambda a: _softmax_np(a, -1), x)
    # softmax.sum() 的梯度恒为 0（softmax 沿 dim 求和=1 是常数），
    # 用非均匀权重 w 让梯度非平凡。
    w = randn(3, 4, seed=30)
    check_grad("softmax", lambda a: (a.softmax(-1) * Tensor(w)).sum(), x)

    # log_softmax
    def _log_softmax_np(a, dim=-1):
        m = a.max(axis=dim, keepdims=True)
        shifted = a - m
        return shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
    check_forward("log_softmax", lambda a: a.log_softmax(-1),
                  lambda a: _log_softmax_np(a, -1), x)
    check_grad("log_softmax", lambda a: a.log_softmax(-1).sum(), x)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Task 6.1: VerseTorch Unit Operator Tests + Gradient Checks")
    print(f"  forward atol = {ATOL_FORWARD}")
    print(f"  grad rtol    = {RTOL_GRAD}")
    print(f"  fd eps       = {EPS}")
    print(f"  grad dtype   = {GRAD_DTYPE}")
    print("=" * 72)

    t0 = time.time()
    test_elementwise()
    test_broadcasting_grad()
    test_shape_ops()
    test_reduction_ops()
    test_matmul()
    test_extras()
    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print(f"Summary: PASS={_PASS}  FAIL={_FAIL}  SKIP={_SKIP}  "
          f"elapsed={elapsed:.1f}s")
    if _FAIL > 0:
        print("Failures:")
        for fmsg in _FAILURES:
            print(f"  - {fmsg}")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
