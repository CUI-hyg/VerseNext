"""CometSpark V0.5-1B 顶层包（Part4K1 Task 8）。

提供：
- :mod:`spark.model`：模型定义（CometSparkV05LM + CometSparkV05Config + 工厂）。
- :mod:`spark.src`：训练 / 评估 / 数据 / 工具（委托 verse_infra.verse_trainer）。
- :mod:`spark.config`：YAML 配置文件目录。

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

__all__ = ["model", "src"]
