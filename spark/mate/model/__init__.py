"""CometSpark Mate 模型包（Part5K1 Task 9.4）。

导出：
- :class:`CometSparkMateConfig`：Mate 配置 dataclass（含 VMPC-mate 字段）。
- :class:`CometSparkMateLM`：基于 ``verse_nex.CometSparkNexLM`` 的 Mate LM。
- :func:`CometSparkMate`：0.2zB 旗舰工厂（VMPC-mate 预设）。
"""

from .config import CometSparkMateConfig
from .model import CometSparkMateLM, CometSparkMate

__all__ = [
    "CometSparkMateConfig",
    "CometSparkMateLM",
    "CometSparkMate",
]
