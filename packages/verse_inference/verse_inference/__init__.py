"""verse_inference 兼容 shim —— 已迁入 verse_infra.verse_inference。

旧路径仍可用（带 DeprecationWarning），推荐改用::

    from verse_infra.verse_inference import ModelLoader, StreamingGenerator
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
    "verse_inference 已迁入 verse_infra.verse_inference，"
    "请改用 from verse_infra.verse_inference import ...",
    DeprecationWarning,
    stacklevel=2,
)

from verse_infra.verse_inference import *  # noqa: F401,E402,F403
from verse_infra.verse_inference import (  # noqa: F401,E402  显式重导出常用 API
    ModelLoader, StateCache, Sampler, GreedySampler, StreamingGenerator,
)
from verse_infra.verse_inference import __version__  # noqa: F401,E402


def __getattr__(name):
    """延迟重导出子模块（如 verse_inference.server → verse_infra.verse_inference.server）。"""
    try:
        _mod = __import__(f"verse_infra.verse_inference.{name}", fromlist=[name])
        globals()[name] = _mod
        return _mod
    except ImportError:
        raise AttributeError(f"module 'verse_inference' has no attribute {name!r}")
