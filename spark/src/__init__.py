"""spark.src 包：训练 / 评估 / 数据加载 / 通用工具 + 基类（Part5K1.1 目录优化）。

本包是 ``spark`` 的 src 层，**全面接入新框架**（verse_infra.verse_trainer /
verse_tokenizer），不重造底层逻辑。

Part5K1.1 目录优化：原 ``spark/model/`` 下的基类配置与模型已迁移到本包：
- :mod:`spark.src.base_config`：``CometSparkV05Config`` + YAML 工具函数。
- :mod:`spark.src.base_model`：``CometSparkV05LM`` + 工厂函数。

子模块：
- :mod:`spark.src.data_loader`：数据加载（委托 verse_trainer.data）。
- :mod:`spark.src.trainer`：训练入口（委托 verse_trainer）。
- :mod:`spark.src.evaluate`：评估入口（委托 verse_trainer.evaluate）。
- :mod:`spark.src.utils`：通用工具（set_seed / num_threads / Qwen tokenizer）。
"""

# Part5K1.1：基类配置与模型（从 spark/model/ 迁移到 spark/src/）
from .base_config import (
    CometSparkV05Config,
    load_full_config,
    save_full_config,
    _dump_yaml,
)
from .base_model import (
    CometSparkV05LM,
    CometSparkV05,
    CometSparkV05Small,
)

from . import data_loader
from . import trainer
from . import evaluate
from . import utils

__all__ = [
    # 基类配置与模型（Part5K1.1 迁移）
    "CometSparkV05Config",
    "CometSparkV05LM",
    "CometSparkV05",
    "CometSparkV05Small",
    "load_full_config",
    "save_full_config",
    # 子模块
    "data_loader",
    "trainer",
    "evaluate",
    "utils",
]
