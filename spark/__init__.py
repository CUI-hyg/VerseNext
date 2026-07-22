"""CometSpark V0.5-1B 顶层包（Part4K1 Task 8）。

提供：
- :mod:`spark.model`：模型定义（CometSparkV05LM + CometSparkV05Config + 工厂）。
- :mod:`spark.src`：训练 / 评估 / 数据 / 工具（委托 verse_infra.verse_trainer）。
- :mod:`spark.config`：YAML 配置文件目录。

路径自举
--------
本包位于 ``<repo>/spark/``，不在 ``packages/`` 下，因此需要在导入时
把 ``<repo>/packages/verse_torch`` / ``verse_nex`` / ``verse_infra`` 注入
``sys.path``，以便 ``from spark.model.model import CometSparkV05LM`` 能
正确解析 ``verse_torch`` / ``verse_nex`` 依赖。
"""

from __future__ import annotations

import os as _os
import sys as _sys

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_THIS_DIR)
for _dep in ("verse_torch", "verse_nex", "verse_infra"):
    _dep_path = _os.path.join(_REPO_ROOT, "packages", _dep)
    if _os.path.isdir(_dep_path) and _dep_path not in _sys.path:
        _sys.path.insert(0, _dep_path)

__version__ = "0.5.0"

__all__ = ["model", "src"]
