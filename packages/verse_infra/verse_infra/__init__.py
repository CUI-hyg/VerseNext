"""VerseInfra: 总包，聚合 verse_tokenizer / verse_compat / verse_inference / verse_trainer。

子模块
======
- :mod:`verse_infra.verse_tokenizer` — BPE / Unigram / WordPiece / Qwen 分词器
- :mod:`verse_infra.verse_compat` — HuggingFace / PyTorch 兼容适配器
- :mod:`verse_infra.verse_inference` — 模型加载 / 状态缓存 / 流式生成
- :mod:`verse_infra.verse_trainer` — 预训练 / 微调 / 后训练 / 评估 CLI

导入路径迁移指南
----------------
旧路径（带 DeprecationWarning，仍可用）::

    from verse_tokenizer import BPETokenizer       # 仍可用，会发警告

新路径（推荐）::

    from verse_infra.verse_tokenizer import BPETokenizer

便捷重导出::

    from verse_infra import BPETokenizer, ModelLoader, train, RLTrainer

设计说明
--------
- ``verse_torch`` / ``verse_nex`` 保持独立未并入 VerseInfra（它们是底层后端）。
- 便捷重导出使用 ``__getattr__`` 延迟导入，避免导入 ``verse_infra`` 时强制加载
  所有子包（verse_trainer 较重，仅在真正访问时才加载）。
"""

from __future__ import annotations

import os as _os
import sys as _sys

# 路径自举：确保 verse_torch / verse_nex 可被导入（它们保持独立未并入 verse_infra）
# 本文件位于 packages/verse_infra/verse_infra/__init__.py，需向上回溯 2 级到 packages/
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PACKAGES_DIR = _os.path.dirname(_os.path.dirname(_THIS_DIR))  # packages/
for _dep in ("verse_torch", "verse_nex"):
    _dep_path = _os.path.join(_PACKAGES_DIR, _dep)
    if _os.path.isdir(_dep_path) and _dep_path not in _sys.path:
        _sys.path.insert(0, _dep_path)

__version__ = "0.1.0"

# 子模块列表（可通过 from verse_infra import verse_tokenizer 访问）
_SUBMODULES = ("verse_tokenizer", "verse_compat", "verse_inference", "verse_trainer")

# 便捷重导出的公共 API 名称（用于 from verse_infra import * 和文档）
__all__ = [
    # 子模块
    "verse_tokenizer", "verse_compat", "verse_inference", "verse_trainer",
    # verse_tokenizer 公共 API
    "BaseTokenizer", "BPETokenizer", "CharTokenizer", "ByteTokenizer",
    "WordPieceTokenizer", "SentencePieceUnigramTokenizer", "VerseTokenizer",
    "QwenTokenizer", "load_tokenizer", "NexTokenizerWrapper",
    "nfkc_normalize", "pre_tokenize", "trim_to_utf8_boundary",
    "render_chat", "render_prompt", "split_prompt_completion",
    "SpecialTokens", "QWEN_IM_START", "QWEN_IM_END", "QWEN_ENDOFTEXT",
    "render_chat_qwen", "render_prompt_qwen", "split_prompt_completion_qwen",
    # verse_compat 公共 API
    "load_hf_state_dict", "Tensor", "nn", "optim", "losses",
    "Linear", "Embedding", "LayerNorm", "RMSNorm", "Module",
    "SGD", "Adam", "AdamW", "cross_entropy", "mse_loss",
    "no_grad", "enable_grad", "set_grad_enabled",
    "tensor", "zeros", "ones", "randn", "rand", "arange",
    "softmax", "sigmoid", "relu", "gelu", "tanh", "exp", "log", "sqrt",
    "matmul", "cat", "stack",
    # verse_inference 公共 API
    "ModelLoader", "StateCache", "Sampler", "GreedySampler", "StreamingGenerator",
    # verse_trainer 公共 API
    "CachedDataset", "TextDataset", "SingleSampleDataset", "BatchLoader",
    "collate_fn", "load_jsonl",
    "train", "ParallelTrainerSafe", "VerseTrainer", "ChunkOOMError",
    "evaluate", "visualize", "LossOptimizer", "RLTrainer",
]


def __getattr__(name):
    """延迟导入子模块公共 API，避免导入 verse_infra 时强制加载所有子包。

    首次访问某个名称时，按子模块顺序查找并缓存到 globals()，
    后续直接从 ``__dict__`` 取值。
    """
    # 先检查是否是子模块名
    if name in _SUBMODULES:
        _mod = __import__(f"verse_infra.{name}", fromlist=[name])
        globals()[name] = _mod
        return _mod

    # 按子包顺序查找公共 API
    for _sub in _SUBMODULES:
        try:
            _mod = __import__(f"verse_infra.{_sub}", fromlist=[name])
        except ImportError:
            continue
        if hasattr(_mod, name):
            _val = getattr(_mod, name)
            globals()[name] = _val  # 缓存，后续直接从 __dict__ 取
            return _val

    raise AttributeError(f"module 'verse_infra' has no attribute {name!r}")


def __dir__():
    """tab 补全支持。"""
    return sorted(list(globals().keys()) + list(__all__))
