# VerseTorch.parallel CPU 并行计算基准测试报告

> 自动生成自 `tests/test_parallel.py`，时间 2026-07-20 13:32:33

## 1. 测试配置

- 矩阵形状: A (64, 256, 256) × B (256, 256) → C (64, 256, 256)
- 计时迭代次数: 10（预热 3 次）
- CPU 核数: 3
- 默认 n_workers: 1（= max(1, cpu_count // 2)）
- 基准 n_workers: 2（= max(2, cpu_count // 2)，强制触发并行路径）
- Python: 3.14.4, NumPy: 2.5.1

## 2. 性能对比（主测）

| 方式 | 每次 wall-clock (ms) | 相对加速比 |
|------|----------------------|------------|
| np.matmul（串行，BLAS 多线程） | 18.37 | 1.00x |
| parallel_matmul（multiprocessing, n_workers=2） | 105.02 | 0.17x |

## 3. 不同 n_workers 对比

| n_workers | 每次 wall-clock (ms) | 相对加速比 |
|-----------|----------------------|------------|
| 1 | 18.77 | 0.98x |
| 2 | 103.11 | 0.18x |
| 4 | 116.71 | 0.16x |

> n_workers=1 等价于串行 np.matmul（短路路径，无 IPC 开销）。

## 4. 数值一致性

- max|parallel - serial| = 0.00e+00
- 阈值 1e-6：PASS

## 5. 备注

- np.matmul 本身已通过底层 BLAS（如 OpenBLAS/MKL）多线程并行计算单个 matmul
- parallel_matmul 在 **batch 维度**切片到不同进程，绕过 GIL
- 当 batch 较小或矩阵较大时，IPC（进程间通信）开销可能抵消并行收益
- 适用场景：batch >= 16 且每个 batch slice 的 matmul 已饱和单核 BLAS
- 受限于测试环境 CPU 与 BLAS 实现，加速比可能 ≤ 1（不强制 > 1）
- 在多核 CPU（>=4 核）环境下加速比应更明显
