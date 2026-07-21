"""Numba 可选加速统一入口。

集中管理 numba 的导入与可用性检测，避免分散在多个模块中。
其他模块应从这里导入 ``_HAS_NUMBA`` 与 ``numba`` 本身，而不是各自 try/except。

设计:
- numba 为可选依赖：不可用时 ``_HAS_NUMBA = False``，``numba`` 为 ``None``，
  调用方应回退到纯 NumPy 实现
- 导入失败不抛异常，仅设置标志位，避免影响 import 时的副作用
"""

from __future__ import annotations

try:
    import numba as numba  # type: ignore[import]
    _HAS_NUMBA = True
except ImportError:
    numba = None  # type: ignore[assignment]
    _HAS_NUMBA = False


__all__ = ["numba", "_HAS_NUMBA"]
