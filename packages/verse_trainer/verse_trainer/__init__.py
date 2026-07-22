"""verse_trainer 兼容 shim —— 已迁入 verse_infra.verse_trainer。

旧路径仍可用（带 DeprecationWarning），推荐改用::

    from verse_infra.verse_trainer import train, RLTrainer
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
    "verse_trainer 已迁入 verse_infra.verse_trainer，"
    "请改用 from verse_infra.verse_trainer import ...",
    DeprecationWarning,
    stacklevel=2,
)

from verse_infra.verse_trainer import *  # noqa: F401,E402,F403
from verse_infra.verse_trainer import (  # noqa: F401,E402  显式重导出常用 API
    CachedDataset, TextDataset, SingleSampleDataset, BatchLoader,
    collate_fn, load_jsonl, train, ParallelTrainerSafe, _safe_chunk_run,
    install_signal_handlers, reset_shutdown_flag, is_shutdown_requested,
    ChunkOOMError, evaluate, visualize, LossOptimizer, RLTrainer,
)
from verse_infra.verse_trainer import __version__  # noqa: F401,E402


def __getattr__(name):
    """延迟重导出子模块（如 verse_trainer.cli → verse_infra.verse_trainer.cli）。"""
    try:
        _mod = __import__(f"verse_infra.verse_trainer.{name}", fromlist=[name])
        globals()[name] = _mod
        return _mod
    except ImportError:
        raise AttributeError(f"module 'verse_trainer' has no attribute {name!r}")
