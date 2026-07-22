"""spark.src 包：训练 / 评估 / 数据加载 / 通用工具（Part4K1 Task 8.5）。

本包是 ``spark`` 的 src 层，**全面接入新框架**（verse_infra.verse_trainer /
verse_tokenizer），不重造底层逻辑。

子模块：
- :mod:`spark.src.data_loader`：数据加载（委托 verse_trainer.data）。
- :mod:`spark.src.trainer`：训练入口（委托 verse_trainer）。
- :mod:`spark.src.evaluate`：评估入口（委托 verse_trainer.evaluate）。
- :mod:`spark.src.utils`：通用工具（set_seed / num_threads / Qwen tokenizer）。
"""

from . import data_loader
from . import trainer
from . import evaluate
from . import utils

__all__ = ["data_loader", "trainer", "evaluate", "utils"]
