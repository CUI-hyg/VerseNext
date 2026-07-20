"""通用工具：随机种子 / 线程数 / 目录创建 / 设备信息。

仅依赖 Python 标准库 + NumPy。
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """设置 Python random / NumPy 的随机种子以保证可复现。

    Args:
        seed: 随机种子（建议 0 <= seed < 2**32）
    """
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    # Python 3.11+ 可设置 hash 种子，避免 dict 顺序抖动
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
    # 不同 BLAS 后端读取的环境变量名不同，全部覆盖一遍
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = n_str
    # 如果 NumPy 安装了 threadpool_ctl，则尝试在运行时限制
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        threadpool_limits(limits=int(n))
    except Exception:
        # 没有该包也没关系，环境变量已经设置
        pass


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则递归创建。

    Args:
        path: 目录路径
    Returns:
        入参 path（便于链式调用）
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_device() -> str:
    """返回当前可用设备名。PoC 阶段仅支持 CPU。"""
    return "cpu"


__all__ = ["set_seed", "num_threads", "ensure_dir", "get_device"]
