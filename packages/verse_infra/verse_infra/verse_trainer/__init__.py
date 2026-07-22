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

# 路径自举：verse_trainer 现为 verse_infra 子包，需向上回溯到 packages/ 目录
# 确保 verse_torch / verse_nex（保持独立的底层后端）可被定位。
# verse_tokenizer 已是 verse_infra 兄弟子包，无需额外 sys.path 注入。
import os as _os
import sys as _sys

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
# packages/verse_infra/verse_infra/verse_trainer/ → packages/
_PACKAGES_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_THIS_DIR)))
for _dep in ("verse_torch", "verse_nex"):
    _dep_path = _os.path.join(_PACKAGES_DIR, _dep)
    if _os.path.isdir(_dep_path) and _dep_path not in _sys.path:
        _sys.path.insert(0, _dep_path)


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
