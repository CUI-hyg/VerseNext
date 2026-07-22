"""通用工具（Part4K1 Task 8.5）。

提供：
- ``set_seed``：设置 Python random / NumPy 随机种子。
- ``num_threads``：限制 NumPy BLAS / OpenMP 线程数（CPU 利用率优化）。
- ``ensure_dir``：确保目录存在。
- ``get_device``：返回当前可用设备名。
- ``load_qwen_tokenizer``：加载 Qwen3.5-35B-A3B tokenizer（graceful skip 无网络）。

CPU 利用率优化（Task 8.7）：
- ``num_threads`` 设置 OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS 等
  环境变量，并尝试用 ``threadpoolctl`` 在运行时限制。
- ``optimize_cpu_for_training``：一键设置 BLAS 线程 + numba 线程 + 数据加载
  预取，适合 CPU 训练场景。
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """设置 Python random / NumPy 的随机种子以保证可复现。"""
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def num_threads(n: int) -> None:
    """限制 NumPy 底层 BLAS / OpenMP 线程数。

    在多进程并行场景下避免线程争抢 CPU。设为 0 或负数时不做改动。

    Args:
        n: 期望的线程数
    """
    if n <= 0:
        return
    n_str = str(int(n))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = n_str
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        threadpool_limits(limits=int(n))
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则递归创建。"""
    os.makedirs(path, exist_ok=True)
    return path


def get_device() -> str:
    """返回当前可用设备名。

    优先返回 ``"cuda"``（若 PyTorch 可用且有 GPU），否则 ``"cpu"``。
    """
    try:
        from verse_torch import has_torch
        if has_torch():
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda"
    except Exception:
        pass
    return "cpu"


def load_qwen_tokenizer(
    repo: str = "Qwen/Qwen3.5-35B-A3B",
    cache_dir: str | None = None,
):
    """加载 Qwen tokenizer（graceful skip 无网络情况）。

    Args:
        repo: HuggingFace repo id，默认 ``"Qwen/Qwen3.5-35B-A3B"``。
        cache_dir: 本地缓存目录；None 时使用临时目录。

    Returns:
        :class:`BPETokenizer` 实例（vocab 248320）。

    Raises:
        RuntimeError: 网络不可用 / 下载失败时抛出明确错误（不会卡住）。
        调用方应 try/except 并 graceful skip。
    """
    # 路径自举：确保 verse_infra 可被 import
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(os.path.dirname(_here))
    for _dep in ("verse_infra",):
        _p = os.path.join(_repo_root, "packages", _dep)
        if os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
    from verse_infra.verse_tokenizer import BPETokenizer
    return BPETokenizer.from_pretrained(repo, cache_dir=cache_dir)


def optimize_cpu_for_training(n_threads: int = 4) -> None:
    """一键 CPU 训练优化（Task 8.7）。

    - 设置 BLAS / OpenMP 线程数（``OMP_NUM_THREADS`` 等）
    - 尝试启用 numba 线程（若 numba 可用）
    - 提示多线程数据加载（由 BatchLoader 的 num_workers 控制，此处仅环境变量）

    Args:
        n_threads: BLAS 线程数（默认 4，适合 4 核 CPU）
    """
    num_threads(n_threads)
    # numba 线程（若安装）
    try:
        import numba  # type: ignore
        numba.set_num_threads(int(n_threads))
    except Exception:
        pass
    # 多线程数据加载提示（BatchLoader 内部支持）
    os.environ.setdefault("SPARK_DATA_NUM_WORKERS", str(max(1, n_threads // 2)))


__all__ = [
    "set_seed",
    "num_threads",
    "ensure_dir",
    "get_device",
    "load_qwen_tokenizer",
    "optimize_cpu_for_training",
]
