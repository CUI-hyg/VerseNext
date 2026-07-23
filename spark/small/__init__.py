"""CometSpark Small 子包（Part5K1 Task 9）。

双模型并行中的 small 档：0.06zB 小模型（VMPC-small 预设）。

提供：
- :mod:`spark.small.model`：模型定义（CometSparkSmallLM + CometSparkSmallConfig + 工厂）。
- :mod:`spark.small.config`：YAML 配置文件目录（cometspark_small.yml）。

路径自举由上层 :mod:`spark._bootstrap` 统一处理，子模块无需重复 sys.path 设置。
"""

from __future__ import annotations

__all__ = ["model"]
