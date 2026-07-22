"""评估入口（Part4K1 Task 8.5）。

本模块是 ``spark`` 的评估入口，**直接委托给**
``verse_infra.verse_trainer.evaluate``。不重造评估逻辑。

支持：
- 加载 checkpoint → 生成示例文本
- ``--score`` 模式：用 :class:`verse_torch.scoring.ScoringEvaluator` 打分
- 5 条预设 prompt（中英混合 + 数字序列）
"""

from __future__ import annotations

import os as _os
import sys as _sys
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_THIS_DIR))
for _dep in ("verse_infra", "verse_torch", "verse_nex"):
    _dep_path = _os.path.join(_REPO_ROOT, "packages", _dep)
    if _os.path.isdir(_dep_path) and _dep_path not in _sys.path:
        _sys.path.insert(0, _dep_path)

from verse_infra.verse_trainer import evaluate  # noqa: E402


__all__ = ["evaluate"]
