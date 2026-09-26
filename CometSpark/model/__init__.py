"""CometSpark 模型包。

导入本包即完成 ``cometspark`` 模型在 verse registry 中的注册。
"""

from .cometspark import (
    MODEL_NAME,
    ROOT_DIR,
    VERSION_FILE,
    CometSparkConfig,
    CometSparkTransformer,
    __version__,
)

__all__ = [
    "MODEL_NAME",
    "ROOT_DIR",
    "VERSION_FILE",
    "CometSparkConfig",
    "CometSparkTransformer",
    "__version__",
]
