"""spark.src 包：训练 / 评估 / 数据加载 / 通用工具 + 共享模型基类（Part5K1.1）。

本包是 ``spark`` 的 src 层，**全面接入新框架**（verse_infra.verse_trainer /
verse_tokenizer），不重造底层逻辑。

Part5K1.1 目录重构后，本包新增"共享模型基类"职责：
- :mod:`spark.src.config`：``CometSparkV05Config``（含 VMPC V2.0 + MoD V1.2 字段）。
- :mod:`spark.src.model`：``CometSparkV05LM`` + ``CometSparkV05`` / ``CometSparkV05Small``
  工厂。``spark.small`` / ``spark.mate`` 的 LM 子类均继承自此处的 ``CometSparkV05LM``。

子模块：
- :mod:`spark.src.config`：共享模型配置基类（Part5K1.1 从 ``spark/model/`` 迁入）。
- :mod:`spark.src.model`：共享 LM 基类 + 工厂（Part5K1.1 从 ``spark/model/`` 迁入）。
- :mod:`spark.src.data_loader`：数据加载（委托 verse_trainer.data）。
- :mod:`spark.src.trainer`：训练入口（委托 verse_trainer）。
- :mod:`spark.src.evaluate`：评估入口（委托 verse_trainer.evaluate）。
- :mod:`spark.src.utils`：通用工具（set_seed / num_threads / Qwen tokenizer）。
"""

from . import config
from . import model
from . import data_loader
from . import trainer
from . import evaluate
from . import utils

# 顶层导出共享基类（便于 ``from spark.src import CometSparkV05LM`` 使用）
from .config import CometSparkV05Config
from .model import CometSparkV05LM, CometSparkV05, CometSparkV05Small

__all__ = [
    "config",
    "model",
    "data_loader",
    "trainer",
    "evaluate",
    "utils",
    # 共享基类导出
    "CometSparkV05Config",
    "CometSparkV05LM",
    "CometSparkV05",
    "CometSparkV05Small",
]
