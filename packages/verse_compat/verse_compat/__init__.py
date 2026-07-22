"""verse_compat 兼容 shim —— 已迁入 verse_infra.verse_compat。

旧路径仍可用（带 DeprecationWarning），推荐改用::

    from verse_infra.verse_compat import load_hf_state_dict
"""

import os as _os
import sys as _sys
import warnings

# 路径自举：确保 verse_infra 可被导入
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PACKAGES_DIR = _os.path.dirname(_os.path.dirname(_THIS_DIR))  # → packages/
_VERSE_INFRA_PATH = _os.path.join(_PACKAGES_DIR, "verse_infra")
if _os.path.isdir(_VERSE_INFRA_PATH) and _VERSE_INFRA_PATH not in _sys.path:
    _sys.path.insert(0, _VERSE_INFRA_PATH)

warnings.warn(
    "verse_compat 已迁入 verse_infra.verse_compat，"
    "请改用 from verse_infra.verse_compat import ...",
    DeprecationWarning,
    stacklevel=2,
)

from verse_infra.verse_compat import *  # noqa: F401,E402,F403
from verse_infra.verse_compat import (  # noqa: F401,E402  显式重导出常用 API
    load_hf_state_dict, Tensor, nn, optim, losses, Linear, Embedding,
    LayerNorm, RMSNorm, Dropout, Module, Sequential, ModuleList,
    SGD, Adam, AdamW, cross_entropy, mse_loss, no_grad, enable_grad,
    set_grad_enabled, is_grad_enabled, tensor, zeros, ones, randn, rand,
    arange, full, empty, eye, softmax, sigmoid, relu, gelu, tanh, exp,
    log, sqrt, matmul, cat, stack, float16, float32, float64, bfloat16,
    int8, int16, int32, int64, uint8,
)
from verse_infra.verse_compat import __version__  # noqa: F401,E402


def __getattr__(name):
    """延迟重导出子模块。"""
    try:
        _mod = __import__(f"verse_infra.verse_compat.{name}", fromlist=[name])
        globals()[name] = _mod
        return _mod
    except ImportError:
        raise AttributeError(f"module 'verse_compat' has no attribute {name!r}")
