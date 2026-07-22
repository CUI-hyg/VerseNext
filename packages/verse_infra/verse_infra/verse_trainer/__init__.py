"""VerseTrainer: 独立训练包（Part4K1 Task 6）。

从 ``data/demo/`` 剥离并升级，统一封装预训练 / 微调 / 后训练 / 评估 / 分词
入口。本包只做"训练入口与流程编排"，底层训练器（Trainer / ParallelTrainer /
VerseNexTrainer / LoRATrainer / SFTTrainer / DPOTrainer / NexTrainer）全部
复用 ``verse_torch`` / ``verse_nex.nexrl`` 的成熟实现。

公共 API
========
- :class:`CachedDataset`：首次扫描缓存为 ``.npz``，后续启动毫秒级加载，
  支持流式 lazy load（大数据集不全量入内存）。
- :class:`LossOptimizer`：loss 优化策略（plateau 重走 + NaN/Inf 跳过 +
  LR 衰减组合）。
- :class:`RLTrainer`：包装 :class:`verse_nex.nexrl.NexTrainer`，
  对接 ``verse-posttrain --rl nexrl``。
- :func:`train` / :func:`evaluate` / :func:`visualize`：从 ``data/demo``
  迁入并升级的训练/评估/可视化入口。
- :func:`main`：CLI 分发入口（``verse-train`` / ``verse-finetune`` /
  ``verse-posttrain`` / ``verse-eval`` / ``verse-tokenize``）。

依赖
====
- ``verse_torch``（Trainer / ParallelTrainer / CheckpointManager / 优化器 /
  损失函数 / 调度器）
- ``verse_nex``（VerseNexLM 策略网络 + ``nexrl.NexTrainer``）
- ``verse_tokenizer``（BPE / Byte / WordPiece / Qwen）

可通过 ``pip install -e packages/verse_trainer`` 安装，或直接在 sys.path
注入 ``packages/verse_torch`` / ``packages/verse_nex`` /
``packages/verse_tokenizer`` 后 ``import verse_trainer``。
"""

from __future__ import annotations

# 路径引导：verse_torch / verse_nex 的 sys.path 注入已由上层
# verse_infra/__init__.py（调用 spark._bootstrap.ensure_paths 或内联回退）统一处理，
# 本子包无需重复 sys.path.insert。


# 公共 API（延迟导入避免循环依赖）
from .data import (
    CachedDataset,
    TextDataset,
    SingleSampleDataset,
    BatchLoader,
    collate_fn,
    load_jsonl,
)
from .loss_optim import LossOptimizer
from .trainer import (
    train,
    ParallelTrainerSafe,
    _safe_chunk_run,
    install_signal_handlers,
    reset_shutdown_flag,
    is_shutdown_requested,
    ChunkOOMError,
)
from .evaluate import evaluate
from .visualize import visualize
from .rl_trainer import RLTrainer

# VerseTrainer 门面别名：升级后的主训练器 ParallelTrainerSafe（_safe_chunk_run +
# 信号处理 + OOM 兜底 + 断点续训），作为本包对外统一训练入口名。
VerseTrainer = ParallelTrainerSafe

__version__ = "0.1.0"

__all__ = [
    # 数据
    "CachedDataset",
    "TextDataset",
    "SingleSampleDataset",
    "BatchLoader",
    "collate_fn",
    "load_jsonl",
    # 训练
    "train",
    "ParallelTrainerSafe",
    "VerseTrainer",
    "_safe_chunk_run",
    "install_signal_handlers",
    "reset_shutdown_flag",
    "is_shutdown_requested",
    "ChunkOOMError",
    # 评估
    "evaluate",
    "visualize",
    # Loss 优化
    "LossOptimizer",
    # RL 后训练
    "RLTrainer",
]
