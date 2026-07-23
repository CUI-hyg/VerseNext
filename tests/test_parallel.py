"""verse_torch.parallel 单元测试（Task 4.4）。

覆盖：
1. parallel_matmul 与 np.matmul 数值一致（多 shape 组合）
2. ParallelLinear forward 与 nn.Linear forward 数值一致
3. ParallelLinear 反向梯度与 nn.Linear 反向梯度数值一致（max diff < 1e-6）
4. parallel_map 与 map 结果一致
5. 基准测试（手动 timeit）：batch=64, M=N=K=256 上对比 parallel_matmul vs np.matmul

运行方式：
    python3 -m pytest tests/test_parallel.py -v
    python3 tests/test_parallel.py           # 也可作为脚本运行
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor
from verse_torch.vnn import Linear
from verse_torch.parallel import parallel_matmul, ParallelLinear, parallel_map


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

ATOL = 1e-6          # 数值一致性阈值
N_WORKERS = 2        # 测试用 worker 数（保持小，避免 CI 资源占用）
BATCH_THRESHOLD = 16  # ParallelLinear 阈值
SEED = 42


def _randn(*shape, dtype=np.float32):
    """固定 seed 的 randn 辅助函数。"""
    return np.random.randn(*shape).astype(dtype)


# ---------------------------------------------------------------------------
# Task 4.1: parallel_matmul 数值一致性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_A,shape_B", [
    ((16, 8, 4), (4, 6)),       # 3D x 2D
    ((16, 8, 4), (16, 4, 6)),   # 3D x 3D
    ((8, 4), (4, 6)),           # 2D x 2D
    ((1, 8, 4), (4, 6)),        # batch=1（短路路径）
    ((16, 1, 4), (4, 6)),       # M=1
    ((16, 8, 4), (4, 1)),       # N=1
    ((32, 16, 8), (8, 4)),      # 较大 batch
    ((4, 8, 4), (4, 6)),        # 小 batch（< n_workers）
])
def test_parallel_matmul_numerical(shape_A, shape_B):
    """parallel_matmul 与 np.matmul 数值一致。"""
    np.random.seed(SEED)
    A = _randn(*shape_A)
    B = _randn(*shape_B)

    expected = np.matmul(A, B)
    actual = parallel_matmul(A, B, n_workers=N_WORKERS)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert max_diff < ATOL, f"shape A={shape_A}, B={shape_B}, max diff={max_diff}"


def test_parallel_matmul_tensor_input():
    """parallel_matmul 接受 Tensor 输入并返回 Tensor。"""
    np.random.seed(SEED)
    A = Tensor(_randn(16, 8, 4))
    B = Tensor(_randn(4, 6))

    expected = np.matmul(A.data, B.data)
    actual = parallel_matmul(A, B, n_workers=N_WORKERS)

    assert isinstance(actual, Tensor)
    assert actual.shape == expected.shape
    max_diff = float(np.max(np.abs(actual.data - expected)))
    assert max_diff < ATOL


def test_parallel_matmul_mixed_input():
    """parallel_matmul 支持 Tensor + ndarray 混合输入。"""
    np.random.seed(SEED)
    A = Tensor(_randn(16, 8, 4))
    B = _randn(4, 6)

    expected = np.matmul(A.data, B)
    actual = parallel_matmul(A, B, n_workers=N_WORKERS)

    assert isinstance(actual, Tensor)
    max_diff = float(np.max(np.abs(actual.data - expected)))
    assert max_diff < ATOL


def test_parallel_matmul_default_workers():
    """默认 n_workers 不报错。"""
    np.random.seed(SEED)
    A = _randn(16, 8, 4)
    B = _randn(4, 6)

    actual = parallel_matmul(A, B)
    expected = np.matmul(A, B)
    assert np.allclose(actual, expected, atol=ATOL)


def test_parallel_matmul_invalid_shape():
    """无效 shape 应抛出 ValueError。"""
    A = _randn(16, 8, 4, 2)  # 4D 不支持
    B = _randn(4, 6)
    with pytest.raises(ValueError):
        parallel_matmul(A, B, n_workers=N_WORKERS)


# ---------------------------------------------------------------------------
# Task 4.2: ParallelLinear forward & backward
# ---------------------------------------------------------------------------


def _make_paired_linear(d_in, d_out, batch_threshold=BATCH_THRESHOLD, bias=True):
    """创建一对共享权重的 Linear 与 ParallelLinear。"""
    np.random.seed(SEED)
    ref = Linear(d_in, d_out, bias=bias)
    par = ParallelLinear(d_in, d_out, bias=bias,
                         n_workers=N_WORKERS,
                         batch_threshold=batch_threshold)
    # 共享权重（确保前向数值可比）
    par.weight = ref.weight
    if bias:
        par.bias = ref.bias
    return ref, par


def test_parallel_linear_forward_consistency():
    """ParallelLinear forward 与 nn.Linear forward 数值一致（batch >= threshold）。"""
    np.random.seed(SEED)
    d_in, d_out, batch = 8, 4, 32
    ref, par = _make_paired_linear(d_in, d_out)

    x = Tensor(_randn(batch, d_in))
    ref_out = ref(x)
    par_out = par(x)

    assert par_out.shape == ref_out.shape
    max_diff = float(np.max(np.abs(par_out.data - ref_out.data)))
    assert max_diff < ATOL, f"forward max diff: {max_diff}"


def test_parallel_linear_forward_small_batch():
    """batch < threshold 时走父类 forward（数值一致）。"""
    np.random.seed(SEED)
    d_in, d_out = 8, 4
    ref, par = _make_paired_linear(d_in, d_out)

    x = Tensor(_randn(8, d_in))  # batch=8 < 16
    ref_out = ref(x)
    par_out = par(x)

    max_diff = float(np.max(np.abs(par_out.data - ref_out.data)))
    assert max_diff < ATOL


def test_parallel_linear_backward_grad():
    """ParallelLinear 反向梯度与 nn.Linear 反向梯度数值一致。"""
    np.random.seed(SEED)
    d_in, d_out, batch = 8, 4, 32
    ref, par = _make_paired_linear(d_in, d_out)

    x_data = _randn(batch, d_in)
    x_ref = Tensor(x_data.copy(), requires_grad=True)
    x_par = Tensor(x_data.copy(), requires_grad=True)

    # 前向
    ref_out = ref(x_ref)
    par_out = par(x_par)
    assert par_out.requires_grad == ref_out.requires_grad

    # 用相同的上游梯度做反向
    grad_data = _randn(*ref_out.shape)
    ref_out.backward(Tensor(grad_data))
    par_out.backward(Tensor(grad_data))

    # x.grad
    x_diff = float(np.max(np.abs(x_par.grad - x_ref.grad)))
    assert x_diff < ATOL, f"x.grad max diff: {x_diff}"

    # W.grad
    w_diff = float(np.max(np.abs(par.weight.grad - ref.weight.grad)))
    assert w_diff < ATOL, f"W.grad max diff: {w_diff}"

    # b.grad
    b_diff = float(np.max(np.abs(par.bias.grad - ref.bias.grad)))
    assert b_diff < ATOL, f"b.grad max diff: {b_diff}"


def test_parallel_linear_no_bias():
    """ParallelLinear bias=False 时正常工作。"""
    np.random.seed(SEED)
    d_in, d_out, batch = 8, 4, 32
    ref, par = _make_paired_linear(d_in, d_out, bias=False)

    x_data = _randn(batch, d_in)
    x_ref = Tensor(x_data.copy(), requires_grad=True)
    x_par = Tensor(x_data.copy(), requires_grad=True)

    ref_out = ref(x_ref)
    par_out = par(x_par)

    assert par.bias is None
    max_diff = float(np.max(np.abs(par_out.data - ref_out.data)))
    assert max_diff < ATOL

    grad_data = _randn(*ref_out.shape)
    ref_out.backward(Tensor(grad_data))
    par_out.backward(Tensor(grad_data))

    x_diff = float(np.max(np.abs(x_par.grad - x_ref.grad)))
    assert x_diff < ATOL
    w_diff = float(np.max(np.abs(par.weight.grad - ref.weight.grad)))
    assert w_diff < ATOL


def test_parallel_linear_3d_input():
    """ParallelLinear 接受 3D 输入 (B, T, D)。"""
    np.random.seed(SEED)
    d_in, d_out = 8, 4
    batch, seq = 4, 8  # total first-dim = 32 >= threshold

    ref, par = _make_paired_linear(d_in, d_out)

    x = Tensor(_randn(batch, seq, d_in))

    ref_out = ref(x)
    par_out = par(x)

    assert par_out.shape == ref_out.shape
    max_diff = float(np.max(np.abs(par_out.data - ref_out.data)))
    assert max_diff < ATOL, f"3D input max diff: {max_diff}"


def test_parallel_linear_3d_backward():
    """ParallelLinear 3D 输入反向梯度与父类一致。"""
    np.random.seed(SEED)
    d_in, d_out = 8, 4
    batch, seq = 4, 8

    ref, par = _make_paired_linear(d_in, d_out)

    x_data = _randn(batch, seq, d_in)
    x_ref = Tensor(x_data.copy(), requires_grad=True)
    x_par = Tensor(x_data.copy(), requires_grad=True)

    ref_out = ref(x_ref)
    par_out = par(x_par)

    grad_data = _randn(*ref_out.shape)
    ref_out.backward(Tensor(grad_data))
    par_out.backward(Tensor(grad_data))

    x_diff = float(np.max(np.abs(x_par.grad - x_ref.grad)))
    assert x_diff < ATOL, f"3D x.grad max diff: {x_diff}"
    w_diff = float(np.max(np.abs(par.weight.grad - ref.weight.grad)))
    assert w_diff < ATOL, f"3D W.grad max diff: {w_diff}"


def test_parallel_linear_in_no_grad():
    """no_grad 上下文中 ParallelLinear 不构建计算图。"""
    from verse_torch import no_grad
    np.random.seed(SEED)
    d_in, d_out, batch = 8, 4, 32
    ref, par = _make_paired_linear(d_in, d_out)

    x = Tensor(_randn(batch, d_in))

    with no_grad():
        ref_out = ref(x)
        par_out = par(x)

    assert not ref_out.requires_grad
    assert not par_out.requires_grad
    max_diff = float(np.max(np.abs(par_out.data - ref_out.data)))
    assert max_diff < ATOL


def test_parallel_linear_extra_repr():
    """extra_repr 包含并行参数。"""
    par = ParallelLinear(8, 4, n_workers=4, batch_threshold=8)
    s = par.extra_repr()
    assert "n_workers=4" in s
    assert "batch_threshold=8" in s


# ---------------------------------------------------------------------------
# Task 4.3: parallel_map
# ---------------------------------------------------------------------------


def _square(x):
    return x * x


def _double(x):
    return x * 2


def test_parallel_map_basic():
    """parallel_map 与 map 结果一致。"""
    items = list(range(20))
    expected = list(map(_square, items))
    actual = parallel_map(_square, items, n_workers=N_WORKERS)
    assert actual == expected


def test_parallel_map_short_circuit():
    """iterable 长度 < 2 时串行（不启动 Pool）。"""
    assert parallel_map(_square, [5], n_workers=4) == [25]
    assert parallel_map(_square, [], n_workers=4) == []


def test_parallel_map_single_worker():
    """n_workers=1 时串行。"""
    items = list(range(20))
    expected = [_square(x) for x in items]
    actual = parallel_map(_square, items, n_workers=1)
    assert actual == expected


def test_parallel_map_default_workers():
    """默认 n_workers 不报错。"""
    items = list(range(10))
    expected = list(map(_double, items))
    actual = parallel_map(_double, items)
    assert actual == expected


def test_parallel_map_preserves_order():
    """结果顺序与输入一致（即使 worker 并行执行）。"""
    items = list(range(50))
    actual = parallel_map(_double, items, n_workers=4)
    expected = [x * 2 for x in items]
    assert actual == expected


def test_parallel_map_with_strings():
    """parallel_map 支持非数值类型。"""
    items = ["hello", "world", "foo", "bar"]
    expected = [s.upper() for s in items]
    actual = parallel_map(str.upper, items, n_workers=2)
    assert actual == expected


# ---------------------------------------------------------------------------
# Task 4.4: 基准测试（手动 timeit）
# ---------------------------------------------------------------------------

# 默认 CI 跳过基准测试（通过环境变量 BENCHMARK=1 显式启用）
_BENCHMARK_ENABLED = os.environ.get("BENCHMARK", "0") == "1"
benchmark_skip = pytest.mark.skipif(
    not _BENCHMARK_ENABLED,
    reason="set BENCHMARK=1 to run performance benchmarks",
)


@benchmark_skip
def test_benchmark_parallel_matmul():
    """在 batch=64, M=N=K=256 上对比 parallel_matmul vs np.matmul wall-clock。

    通过环境变量 BENCHMARK=1 启用：
        BENCHMARK=1 python -m pytest tests/test_parallel.py::test_benchmark_parallel_matmul -v -s
    """
    np.random.seed(SEED)
    batch, M, K, N = 64, 256, 256, 256
    A = _randn(batch, M, K)
    B = _randn(K, N)

    n_workers = max(1, (os.cpu_count() or 2) // 2)
    n_iter = 5
    warmup = 2

    # warmup
    for _ in range(warmup):
        _ = np.matmul(A, B)
        _ = parallel_matmul(A, B, n_workers=n_workers)

    # 串行
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = np.matmul(A, B)
    serial_ms = (time.perf_counter() - t0) * 1000 / n_iter

    # 并行
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = parallel_matmul(A, B, n_workers=n_workers)
    parallel_ms = (time.perf_counter() - t0) * 1000 / n_iter

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else 0.0
    print(f"\n[benchmark] batch={batch}, M={M}, K={K}, N={N}")
    print(f"[benchmark] CPU cores: {os.cpu_count()}, n_workers: {n_workers}")
    print(f"[benchmark] serial:   {serial_ms:.2f} ms")
    print(f"[benchmark] parallel: {parallel_ms:.2f} ms")
    print(f"[benchmark] speedup:  {speedup:.2f}x")

    # 至少不报错（speedup 不强制 > 1，受 BLAS 与 IPC 开销影响）
    assert serial_ms > 0
    assert parallel_ms > 0


# ---------------------------------------------------------------------------
# 作为脚本运行：执行单元测试 + 基准测试并生成报告
# ---------------------------------------------------------------------------


def _run_benchmark_to_report():
    """运行基准并写入 docs/benchmarks/parallel_benchmark.md。"""
    np.random.seed(SEED)
    batch, M, K, N = 64, 256, 256, 256
    A = _randn(batch, M, K)
    B = _randn(K, N)

    cpu_count = os.cpu_count() or 2
    default_n_workers = max(1, cpu_count // 2)
    # 显式覆盖：至少用 2 个 worker 才能真正触发 multiprocessing 路径
    n_workers = max(2, default_n_workers)
    n_iter = 10
    warmup = 3

    # warmup
    for _ in range(warmup):
        _ = np.matmul(A, B)
        _ = parallel_matmul(A, B, n_workers=n_workers)

    # 串行 baseline
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = np.matmul(A, B)
    serial_ms = (time.perf_counter() - t0) * 1000 / n_iter

    # 并行（主测）：n_workers = max(2, cpu_count // 2)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = parallel_matmul(A, B, n_workers=n_workers)
    parallel_ms = (time.perf_counter() - t0) * 1000 / n_iter

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else 0.0

    # 不同 n_workers 对比
    worker_results = []
    for nw in sorted(set([1, 2, 4, default_n_workers])):
        # warmup
        for _ in range(2):
            _ = parallel_matmul(A, B, n_workers=nw)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = parallel_matmul(A, B, n_workers=nw)
        ms = (time.perf_counter() - t0) * 1000 / n_iter
        sp = serial_ms / ms if ms > 0 else 0.0
        worker_results.append((nw, ms, sp))

    # 数值一致性验证
    diff = float(np.max(np.abs(parallel_matmul(A, B, n_workers=n_workers) - np.matmul(A, B))))

    # 写报告
    report_dir = _REPO_ROOT / "docs" / "benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "parallel_benchmark.md"

    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    worker_rows = "\n".join(
        f"| {nw} | {ms:.2f} | {sp:.2f}x |" for nw, ms, sp in worker_results
    )

    content = f"""# VerseTorch.parallel CPU 并行计算基准测试报告

> 自动生成自 `tests/test_parallel.py`，时间 {now}

## 1. 测试配置

- 矩阵形状: A ({batch}, {M}, {K}) × B ({K}, {N}) → C ({batch}, {M}, {N})
- 计时迭代次数: {n_iter}（预热 {warmup} 次）
- CPU 核数: {cpu_count}
- 默认 n_workers: {default_n_workers}（= max(1, cpu_count // 2)）
- 基准 n_workers: {n_workers}（= max(2, cpu_count // 2)，强制触发并行路径）
- Python: {sys.version.split()[0]}, NumPy: {np.__version__}

## 2. 性能对比（主测）

| 方式 | 每次 wall-clock (ms) | 相对加速比 |
|------|----------------------|------------|
| np.matmul（串行，BLAS 多线程） | {serial_ms:.2f} | 1.00x |
| parallel_matmul（multiprocessing, n_workers={n_workers}） | {parallel_ms:.2f} | {speedup:.2f}x |

## 3. 不同 n_workers 对比

| n_workers | 每次 wall-clock (ms) | 相对加速比 |
|-----------|----------------------|------------|
{worker_rows}

> n_workers=1 等价于串行 np.matmul（短路路径，无 IPC 开销）。

## 4. 数值一致性

- max|parallel - serial| = {diff:.2e}
- 阈值 1e-6：{"PASS" if diff < ATOL else "FAIL"}

## 5. 备注

- np.matmul 本身已通过底层 BLAS（如 OpenBLAS/MKL）多线程并行计算单个 matmul
- parallel_matmul 在 **batch 维度**切片到不同进程，绕过 GIL
- 当 batch 较小或矩阵较大时，IPC（进程间通信）开销可能抵消并行收益
- 适用场景：batch >= 16 且每个 batch slice 的 matmul 已饱和单核 BLAS
- 受限于测试环境 CPU 与 BLAS 实现，加速比可能 ≤ 1（不强制 > 1）
- 在多核 CPU（>=4 核）环境下加速比应更明显
"""
    report_path.write_text(content, encoding="utf-8")
    print(f"\n[benchmark] 报告已写入: {report_path}")
    print(f"[benchmark] serial_ms={serial_ms:.2f}, parallel_ms={parallel_ms:.2f}, speedup={speedup:.2f}x")
    return serial_ms, parallel_ms, speedup


if __name__ == "__main__":
    # 作为脚本运行：执行 pytest 单元测试 + 生成基准报告
    print("=" * 70)
    print("Running unit tests via pytest...")
    print("=" * 70)
    import subprocess
    ret = subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v", "-x", "--no-header"],
        cwd=str(_REPO_ROOT),
    )
    if ret != 0:
        print("Unit tests FAILED, skip benchmark.")
        sys.exit(ret)

    print()
    print("=" * 70)
    print("Running benchmark and generating report...")
    print("=" * 70)
    _run_benchmark_to_report()
