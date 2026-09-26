"""VerseNext 模型架构包。

除模型本身外，还提供两个横切子系统（供 trainer / 推理 / CometSpark 复用）：

- :mod:`verse_nn.devices` —— 设备抽象与资源锁（CPU/CUDA/ROCm/CANN NPU）；
- :mod:`verse_nn.kernels` —— 自研融合算子与多后端派发（含 CPU int8 量化）。
"""

from verse_nn.transformer import (
    VerseTransformer,
    VerseTransformerConfig,
    count_parameters,
    _apply_repetition_penalty,
    _apply_top_p,
    _ban_repeat_ngrams,
)
from verse_nn.blocks import TransformerBlock, RMSNorm, SwiGLU
from verse_nn.attention import (
    CausalSelfAttention,
    DSAttention,
    KDAAttention,
    build_attention,
)
from verse_nn.kv_cache import StaticKVCache, RecurrentStateCache
from verse_nn.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    mark_only_lora_trainable,
    lora_state_dict,
    load_lora_state_dict,
    merge_lora,
)
from verse_nn import devices, kernels

__all__ = [
    "VerseTransformer",
    "VerseTransformerConfig",
    "count_parameters",
    "_apply_repetition_penalty",
    "_apply_top_p",
    "_ban_repeat_ngrams",
    "TransformerBlock",
    "RMSNorm",
    "SwiGLU",
    "CausalSelfAttention",
    "DSAttention",
    "KDAAttention",
    "build_attention",
    "StaticKVCache",
    "RecurrentStateCache",
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "mark_only_lora_trainable",
    "lora_state_dict",
    "load_lora_state_dict",
    "merge_lora",
    "devices",
    "kernels",
]
__version__ = "0.4.0"
