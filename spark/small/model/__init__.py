"""CometSpark Small 模型包（Part5K1 Task 9.3）。

导出：
- :class:`CometSparkSmallConfig`：Small 配置 dataclass（含 VMPC-small 字段）。
- :class:`CometSparkSmallLM`：基于 ``verse_nex.CometSparkNexLM`` 的 Small LM。
- :func:`CometSparkSmall`：0.06zB 目标工厂（VMPC-small 预设）。
"""

from .config import CometSparkSmallConfig
from .model import CometSparkSmallLM, CometSparkSmall

__all__ = [
    "CometSparkSmallConfig",
    "CometSparkSmallLM",
    "CometSparkSmall",
]
