"""CometSpark V0.5-1B 模型包（Part4K1 Task 8）。

导出：
- :class:`CometSparkV05Config`：1B 配置 dataclass。
- :class:`CometSparkV05LM`：基于 ``verse_nex.CometSparkNexLM`` 的顶层 LM。
- :func:`CometSparkV05`：≈1B 参数工厂。
- :func:`CometSparkV05Small`：调试小配置工厂。
"""

from .config import CometSparkV05Config
from .model import CometSparkV05LM, CometSparkV05, CometSparkV05Small

__all__ = [
    "CometSparkV05Config",
    "CometSparkV05LM",
    "CometSparkV05",
    "CometSparkV05Small",
]
