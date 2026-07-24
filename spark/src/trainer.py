"""训练入口（Part4K1 Task 8.5）。

本模块是 ``spark`` 的训练入口，**直接委托给**
``verse_infra.verse_trainer``（VerseTrainer / ParallelTrainerSafe /
LossOptimizer / RLTrainer）。不重造训练逻辑。

支持：
- 预训练：``train(config_path, ...)``
- 并行训练：``parallel_chunks > 1`` 自动走 ParallelTrainerSafe
- 断点续训：``resume=True``
- 单样本 / 单文件模式
- VerseNex 原生 arch（``forward_with_aux`` + MoD aux loss）

CLI 入口（推荐）：
    verse-train --config spark/small/config/cometspark_small.yml --device cpu
    verse-finetune --config spark/mate/config/cometspark_mate.yml --method lora
    verse-posttrain --config spark/mate/config/cometspark_mate.yml --rl nexrl
    verse-eval --config spark/small/config/cometspark_small.yml --score
"""

from __future__ import annotations

# 路径引导：统一委托 spark._bootstrap（幂等，自动注入 verse_infra 等）
import spark._bootstrap  # noqa: F401 — 副作用导入：设置 sys.path

from verse_infra.verse_trainer import (  # noqa: E402
    train,
    ParallelTrainerSafe,
    CachedDataset,
    LossOptimizer,
    RLTrainer,
    install_signal_handlers,
    reset_shutdown_flag,
    is_shutdown_requested,
)


def build_model_from_config(config_dict: dict, vocab_size: int):
    """从配置 dict 构建 CometSparkV05LM。

    Args:
        config_dict: model 段配置 dict（来自 config.yml 的 model 段）。
        config_dict 会与 vocab_size 合并后构造 ``CometSparkV05Config``。
        vocab_size: 实际词表大小（由 tokenizer 决定，覆盖 config 的 vocab_size）。

    Returns:
        (model, config)：model 是 :class:`CometSparkV05LM` 实例，
        config 是 :class:`CometSparkV05Config` 实例。
    """
    # 延迟导入避免循环
    # Part5K1.1：基类已从 spark.model.* 迁移到 spark.src.base_*
    from spark.src.base_config import CometSparkV05Config
    from spark.src.base_model import CometSparkV05LM

    merged = dict(config_dict)
    merged["vocab_size"] = vocab_size
    config = CometSparkV05Config.from_dict(merged)
    model = CometSparkV05LM(config)
    return model, config


__all__ = [
    "train",
    "ParallelTrainerSafe",
    "CachedDataset",
    "LossOptimizer",
    "RLTrainer",
    "install_signal_handlers",
    "reset_shutdown_flag",
    "is_shutdown_requested",
    "build_model_from_config",
]
