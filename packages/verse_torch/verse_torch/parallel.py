"""verse_torch.parallel: CPU 并行计算模块（multiprocessing 后端）。

提供：
- parallel_matmul(A, B, n_workers=None): 批量矩阵乘法并行计算
- ParallelLinear(d_in, d_out, n_workers=None, batch_threshold=16): 并行全连接层
- parallel_map(fn, iterable, n_workers=None): 通用并行 map

设计要点：
- 仅使用 NumPy + 标准库（multiprocessing/os），不依赖 torch/tensorflow/jax
- 默认 n_workers = max(1, os.cpu_count() // 2)
- Linux 默认 fork 启动方式，子进程继承父进程内存
- 测试环境如 Pool 启动失败，自动降级为串行
- ParallelLinear 仅前向并行；反向通过手动构建 _backward 闭包走标准 autograd
"""

from __future__ import annotations

import os
import pickle
import multiprocessing as mp
from typing import Callable, Iterable, Optional

import numpy as np

from .tensor import Tensor, is_grad_enabled
from . import vnn as nn


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _default_n_workers() -> int:
    """默认 worker 数：CPU 核数的一半（至少 1）。"""
    cpu = os.cpu_count() or 2
    return max(1, cpu // 2)


def _split_batch(batch: int, n_workers: int):
    """把 batch 维度切成 n_workers 个近似等分的 (start, end) 区间。"""
    if n_workers > batch:
        n_workers = batch
    if n_workers < 1:
        n_workers = 1
    chunk_size = (batch + n_workers - 1) // n_workers
    chunks = []
    for i in range(0, batch, chunk_size):
        end = min(i + chunk_size, batch)
        chunks.append((i, end))
    return chunks


# ---------------------------------------------------------------------------
# Task 4.1: parallel_matmul
# ---------------------------------------------------------------------------


# 全局共享矩阵：fork 模式下子进程继承，避免每个 task 重复 pickle 大数组
# 每次 parallel_matmul 调用前设置，调用后清空
_SHARED_B: Optional[np.ndarray] = None


def _matmul_worker_shared(A_chunk):
    """worker 任务：返回 A_chunk @ _SHARED_B（B 通过 fork 继承，无需 pickle）。"""
    return A_chunk @ _SHARED_B


def _matmul_worker(args):
    """单个 worker 的 matmul 任务：返回 A_chunk @ B（向后兼容签名）。"""
    A_chunk, B = args
    return A_chunk @ B


def parallel_matmul(A, B, n_workers: Optional[int] = None):
    """批量矩阵乘法的并行实现。

    支持的 shape 组合：
        A: (B, M, K) 或 (M, K)
        B: (K, N) 或 (B, K, N)
    输出：
        - 若 A 是 3D：返回 (B, M, N)
        - 若 A 是 2D：返回 (M, N)

    参数:
        A: Tensor 或 np.ndarray
        B: Tensor 或 np.ndarray
        n_workers: 进程数；None 则用 max(1, cpu_count // 2)

    返回:
        与输入中第一个 Tensor 同类型；若输入均为 ndarray 则返回 ndarray
    """
    a_is_tensor = isinstance(A, Tensor)
    b_is_tensor = isinstance(B, Tensor)
    A_np = A.data if a_is_tensor else np.asarray(A)
    B_np = B.data if b_is_tensor else np.asarray(B)

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))

    if A_np.ndim == 2:
        # 2D x 任意：直接 np.matmul（底层已 BLAS 多线程并行）
        result = np.matmul(A_np, B_np)
    elif A_np.ndim == 3:
        batch = A_np.shape[0]
        if B_np.ndim == 2:
            # (B, M, K) x (K, N) -> (B, M, N)
            if batch <= 1 or n_workers == 1:
                result = np.matmul(A_np, B_np)
            else:
                result = _parallel_batched_matmul(A_np, B_np, n_workers, batch)
        elif B_np.ndim == 3:
            # (B, M, K) x (B, K, N) -> (B, M, N)
            if batch <= 1 or n_workers == 1:
                result = np.matmul(A_np, B_np)
            else:
                result = _parallel_batched_matmul_paired(A_np, B_np, n_workers, batch)
        else:
            raise ValueError(f"Unsupported B.ndim={B_np.ndim} for A.ndim=3")
    else:
        raise ValueError(f"Unsupported A.ndim={A_np.ndim}; expected 2 or 3")

    # 包装回原类型：若任一输入是 Tensor，返回 Tensor
    if a_is_tensor or b_is_tensor:
        return Tensor(result)
    return result


def _parallel_batched_matmul(A: np.ndarray, B: np.ndarray, n_workers: int, batch: int) -> np.ndarray:
    """(B, M, K) x (K, N) -> (B, M, N)，按 batch 切片并行。

    B 是共享的 2D 矩阵，通过全局变量在 fork 模式下让子进程继承，避免每个 task pickle。
    """
    global _SHARED_B
    chunks = _split_batch(batch, n_workers)
    A_chunks = [A[s:e] for s, e in chunks]

    _SHARED_B = B  # fork 模式下子进程继承此引用
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_matmul_worker_shared, A_chunks)
    except (OSError, RuntimeError, ValueError):
        # 受限环境降级为串行（例如 CI 不允许 spawn 子进程）
        results = [_matmul_worker_shared(chunk) for chunk in A_chunks]
    finally:
        _SHARED_B = None

    return np.concatenate(results, axis=0)


def _parallel_batched_matmul_paired(A: np.ndarray, B: np.ndarray, n_workers: int, batch: int) -> np.ndarray:
    """(B, M, K) x (B, K, N) -> (B, M, N)，A 与 B 沿 batch 同步切片。

    由于 B 也是 3D，无法走全局共享路径（每个 chunk 的 B 不同），使用 args 元组传递。
    """
    chunks = _split_batch(batch, n_workers)
    args = [(A[s:e], B[s:e]) for s, e in chunks]

    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_matmul_worker, args)
    except (OSError, RuntimeError, ValueError):
        results = [_matmul_worker(arg) for arg in args]

    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Task 4.3: parallel_map
# ---------------------------------------------------------------------------


def _map_worker(args):
    """worker 调用 fn(item)。"""
    fn, item = args
    return fn(item)


def parallel_map(fn: Callable, iterable: Iterable, n_workers: Optional[int] = None) -> list:
    """通用并行 map，类似 multiprocessing.Pool.map。

    参数:
        fn: 可调用对象（推荐顶层函数以保证 picklable；lambda 在某些环境下不可 pickle）
        iterable: 可迭代对象
        n_workers: 进程数；None 则用 max(1, cpu_count // 2)

    返回:
        list，按原顺序

    注意:
        - 若 iterable 长度 < 2 或 n_workers==1，则串行
        - 若 fn 不可 pickle（如 __main__ 中的 lambda）或 Pool 启动失败，自动降级为串行
    """
    items = list(iterable)
    n = len(items)

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))

    # 短路：长度过小或单 worker，直接串行
    if n < 2 or n_workers == 1:
        return [fn(x) for x in items]

    args = [(fn, x) for x in items]

    try:
        with mp.Pool(processes=n_workers) as pool:
            return pool.map(_map_worker, args)
    except (OSError, RuntimeError, ValueError, pickle.PicklingError, AttributeError):
        # 受限环境 / fn 不可 pickle / Pool 启动失败 -> 降级为串行
        return [fn(x) for x in items]


# ---------------------------------------------------------------------------
# Task 4.2: ParallelLinear
# ---------------------------------------------------------------------------


class ParallelLinear(nn.Linear):
    """并行全连接层：在 batch 维度并行计算 x @ W^T + b。

    Args:
        in_features: 输入维度
        out_features: 输出维度
        bias: 是否使用偏置
        n_workers: 进程数；None 则用 max(1, cpu_count // 2)
        batch_threshold: batch 大小阈值；当 x.shape[0] >= 该值时启用并行

    前向:
        - 若 x.shape[0] < batch_threshold：走父类 forward（单进程）
        - 否则用 parallel_matmul 在 batch 维度并行计算

    反向:
        - 手动构建 _backward 闭包，逻辑与父类 Linear 一致（数值 1e-6 内一致）
        - 仅前向并行；反向走标准 autograd（通过 parameters() 返回的 Tensor 自带 _backward）
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 n_workers: Optional[int] = None, batch_threshold: int = 16):
        super().__init__(in_features, out_features, bias=bias)
        # n_workers=None 表示在调用时使用默认值
        self.n_workers = n_workers
        self.batch_threshold = int(batch_threshold)

    def forward(self, x: Tensor) -> Tensor:
        # 短路：维度不足或 batch 太小，走父类（保留父类 autograd）
        if x.data.ndim < 2 or x.shape[0] < self.batch_threshold:
            return super().forward(x)

        x_data = x.data
        W = self.weight       # (d_out, d_in)
        b = self.bias         # (d_out,) 或 None
        W_data = W.data
        # W^T: (d_in, d_out)；2D 矩阵共享内存的转置视图
        W_T = W_data.T

        # 并行计算 x @ W^T（沿 batch 维度切片）
        out_data = parallel_matmul(x_data, W_T, n_workers=self.n_workers)
        if isinstance(out_data, Tensor):
            out_data = out_data.data
        if b is not None:
            out_data = out_data + b.data

        # 决定是否构建 autograd 节点
        requires_grad = is_grad_enabled() and (
            x.requires_grad
            or W.requires_grad
            or (b is not None and b.requires_grad)
        )

        children = []
        if x.requires_grad:
            children.append(x)
        if W.requires_grad:
            children.append(W)
        if b is not None and b.requires_grad:
            children.append(b)

        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            _children=tuple(children),
            _op="parallel_linear",
        )

        if requires_grad:
            # 手动构建 _backward 闭包，复制父类 Linear 的反向逻辑
            # 前向：out = x @ W^T + b
            # 反向：
            #   dx = g @ W          (W shape (d_out, d_in)，g shape 同 out)
            #   dW = g_flat^T @ x_flat  (sum over batch/seq dims, shape (d_out, d_in))
            #   db = sum over leading dims of g  (shape (d_out,))
            def _backward():
                g = out.grad
                # dx = g @ W
                if x.requires_grad:
                    dx = g @ W_data
                    x._accumulate_grad(dx)
                # dW = g^T @ x（处理 batch/seq 维度）
                if W.requires_grad:
                    if g.ndim > 2:
                        # 3D+ 输入：把前 N-1 维 flatten 后求和
                        g_flat = g.reshape(-1, g.shape[-1])
                        x_flat = x_data.reshape(-1, x_data.shape[-1])
                        dW = g_flat.T @ x_flat
                    elif g.ndim == 2:
                        dW = g.T @ x_data
                    else:
                        # g 是 1D（不太可能，保底）
                        dW = np.outer(g, x_data)
                    W._accumulate_grad(dW)
                # db = sum over 所有非最后一维
                if b is not None and b.requires_grad:
                    if g.ndim > 1:
                        db = g.sum(axis=tuple(range(g.ndim - 1)))
                    else:
                        db = g
                    b._accumulate_grad(db)

            out._backward = _backward

        return out

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"n_workers={self.n_workers}, batch_threshold={self.batch_threshold}"
        )
