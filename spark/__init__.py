"""CometSpark 顶层包（Part5K1 Task 9 双模型并行升级 / Part5K1.1 目录优化）。

提供：
- :mod:`spark.small`：Small 模型（0.06zB 目标，VMPC-small 预设）。
- :mod:`spark.mate`：Mate 模型（0.2zB 旗舰，VMPC-mate 预设）。
- :mod:`spark.src`：基类 + 训练 / 评估 / 数据 / 工具（委托 verse_infra.verse_trainer）。
  Part5K1.1：原 ``spark.model`` 下的基类配置与模型已迁移到 ``spark.src``：
  - :mod:`spark.src.base_config`：``CometSparkV05Config`` + YAML 工具。
  - :mod:`spark.src.base_model`：``CometSparkV05LM`` + 工厂函数。

路径自举
--------
本包位于 ``<repo>/spark/``，不在 ``packages/`` 下。路径设置统一委托
:mod:`spark._bootstrap`，子模块无需重复 ``sys.path.insert``。
"""

from __future__ import annotations

# 统一路径引导：自动将 verse_torch / verse_nex / verse_infra / spark / data
# 注入 sys.path（幂等，多次调用安全）。
from spark._bootstrap import ensure_paths

ensure_paths()

__version__ = "0.5.0"

# ---------------------------------------------------------------------------
# Part5K1 Task 9：导出双模型（small / mate）
# ---------------------------------------------------------------------------
# 双模型均基于 verse_nex.CometSparkNexLM，通过 config 传入 VMPC 预设。
# 顶层导出便于 `from spark import CometSparkSmall, CometSparkMate` 使用。
from .small.model import (
    CometSparkSmallLM,
    CometSparkSmallConfig,
    CometSparkSmall,
)
from .mate.model import (
    CometSparkMateLM,
    CometSparkMateConfig,
    CometSparkMate,
)

__all__ = [
    # 双模型子包
    "small",
    "mate",
    # 基类子包（Part5K1.1：原 spark.model 迁移到 spark.src）
    "src",
    # 双模型顶层导出
    "CometSparkSmallLM",
    "CometSparkSmallConfig",
    "CometSparkSmall",
    "CometSparkMateLM",
    "CometSparkMateConfig",
    "CometSparkMate",
]
