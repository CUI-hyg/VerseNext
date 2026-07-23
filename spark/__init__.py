"""CometSpark 顶层包（Part5K1 Task 9 双模型并行升级）。

提供：
- :mod:`spark.small`：Small 模型（0.06zB 目标，VMPC-small 预设）。
- :mod:`spark.mate`：Mate 模型（0.2zB 旗舰，VMPC-mate 预设）。
- :mod:`spark.model`：旧 V05 模型（兼容保留，Task 11 清理）。
- :mod:`spark.src`：训练 / 评估 / 数据 / 工具（委托 verse_infra.verse_trainer）。
- :mod:`spark.config`：旧 YAML 配置文件目录（兼容保留，Task 11 清理）。

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
    # 旧子包（兼容保留，Task 11 清理）
    "model",
    "src",
    # 双模型顶层导出
    "CometSparkSmallLM",
    "CometSparkSmallConfig",
    "CometSparkSmall",
    "CometSparkMateLM",
    "CometSparkMateConfig",
    "CometSparkMate",
]
