"""VerseNext 训练优化包。"""

from verse_trainer.trainer import Trainer, TrainerConfig
from verse_trainer.optim import build_optimizer, build_lr_scheduler, WarmupCosineScheduler
from verse_trainer.data import TokenBatchIterator, DataConfig, fingerprint_source
from verse_trainer.plan import (
    ChainPlan,
    StageSpec,
    TrainStage,
    chain_fingerprint,
    check_chain_resume,
    load_chain_plan,
    stage_bounds,
)
from verse_trainer.sft import SFTDataset, SFTConfig, load_conversations
from verse_trainer.data_formats import (
    detect_format,
    extract_text,
    is_pretokenized,
    load_texts_auto,
    load_token_stream,
)
from verse_trainer.inference import Generator, GenerateConfig, dynamic_quantize, optimize_cpu
from verse_trainer.resources import (
    MemoryGuard,
    ResourceConfig,
    apply_resource_limits,
    available_cpu_count,
)

__all__ = [
    "Trainer",
    "TrainerConfig",
    "build_optimizer",
    "build_lr_scheduler",
    "WarmupCosineScheduler",
    "TokenBatchIterator",
    "DataConfig",
    "fingerprint_source",
    "ChainPlan",
    "StageSpec",
    "TrainStage",
    "chain_fingerprint",
    "check_chain_resume",
    "load_chain_plan",
    "stage_bounds",
    "SFTDataset",
    "SFTConfig",
    "load_conversations",
    "detect_format",
    "extract_text",
    "is_pretokenized",
    "load_texts_auto",
    "load_token_stream",
    "Generator",
    "GenerateConfig",
    "dynamic_quantize",
    "optimize_cpu",
    "MemoryGuard",
    "ResourceConfig",
    "apply_resource_limits",
    "available_cpu_count",
]
__version__ = "0.1.0"
