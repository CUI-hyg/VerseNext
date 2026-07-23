"""CometSpark Mate 子包（Part5K1 Task 9）。

双模型并行中的 mate 档：0.2zB 旗舰模型（VMPC-mate 预设）。

提供：
- :mod:`spark.mate.model`：模型定义（CometSparkMateLM + CometSparkMateConfig + 工厂）。
- :mod:`spark.mate.config`：YAML 配置文件目录（cometspark_mate.yml）。

路径自举由上层 :mod:`spark._bootstrap` 统一处理，子模块无需重复 sys.path 设置。
"""

from __future__ import annotations

__all__ = ["model"]
