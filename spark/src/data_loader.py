"""数据加载器（Part4K1 Task 8.5）。

本模块是 ``spark`` 的数据加载入口，**直接委托给**
``verse_infra.verse_trainer.data``（已迁移的 CachedDataset / TextDataset /
BatchLoader / collate_fn / load_jsonl）。不重造数据加载逻辑。

用法：
    from spark.src.data_loader import CachedDataset, BatchLoader, collate_fn
"""

from __future__ import annotations

# 路径自举：确保 verse_infra 可被 import
import os as _os
import sys as _sys
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
# spark/src/ → spark/ → /workspace
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_THIS_DIR))
for _dep in ("verse_infra", "verse_torch", "verse_nex"):
    _dep_path = _os.path.join(_REPO_ROOT, "packages", _dep)
    if _os.path.isdir(_dep_path) and _dep_path not in _sys.path:
        _sys.path.insert(0, _dep_path)

# 直接重导出 verse_trainer.data 的公共 API（不重造）
# 注意：_detect_format 虽以 _ 开头，但是 data_loader 单元测试需要直接调用它，
# 因此在此显式重导出（verse_trainer.data 的 __all__ 未包含它）。
from verse_infra.verse_trainer.data import (  # noqa: E402
    CachedDataset,
    TextDataset,
    SingleSampleDataset,
    BatchLoader,
    collate_fn,
    load_jsonl,
    _detect_format,
)


__all__ = [
    "CachedDataset",
    "TextDataset",
    "SingleSampleDataset",
    "BatchLoader",
    "collate_fn",
    "load_jsonl",
    "_detect_format",
]
