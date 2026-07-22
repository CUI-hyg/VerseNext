"""VerseNex: Transformer-alternative architectures (Mamba-2, RWKV-7, Linear Attention, Hybrid).

提供线性复杂度的 Transformer 替代架构库，主推：
- Mamba-2 SSM (selective state space model with SSD)
- RWKV-7 (time mixing + channel mixing)
- Linear Attention (RetNet 风格 retention + chunkwise)
- Sparse Attention (top-k chunk sparse)
- Hybrid Block/LM (SSM + Sparse Attention 混合)
- 位置编码 (RoPE, ALiBi, NoPE)

Part4 新增（VerseNex 原生架构）：
- TriSparseAttention: 三路并行稀疏注意力（SWA + Global + ALiBi）
- MoD: 多稠密分区架构（5 DensePart × 8 Experts × top-3 双层门控）

所有模块支持：
- parallel 模式（训练，可微，整序列并行）
- recurrent 模式（推理，常数内存，单步递推）
- 数值一致：parallel 与 recurrent 输出在 float32 下吻合到 1e-3
"""

from .positional import RoPE, ALiBi, NoPE
from .linear_attention import RetNet
from .mamba2 import Mamba2Block
from .rwkv7 import RWKV7TimeMix, RWKV7ChannelMix, RWKV7Block
from .sparse_attention import TopKChunkSparseAttention
from .hybrid import HybridBlock, HybridLM
# Part4 新增
from .tri_sparse_attn import TriSparseAttention
from .moe import Router, Expert, DensePart, MoDLayer
from .cometspark import (
    VerseNexBlock,
    CometSparkNexLM,
    CometSparkV02,
    VerseNexLM,
)

# Part4K1 Task 4: NexRL 强化学习算法包
from . import nexrl
from .nexrl import (
    NexAgent,
    NexEnv,
    NexState,
    NexAction,
    NexReward,
    ChatEnv,
    MathEnv,
    CodeEnv,
    ActionSampler,
    ExplorationSchedule,
    repeat_penalty,
    RewardNormalizer,
    RewardShaper,
    Rollout,
    ParallelRolloutCollector,
    NexTrainer,
)
# Part4K1 Task 3.2: 分离式并行预测（speculative decoding 风格）
from .speculative import SpeculativeDecoder
# Part4K1 Task 3.3: 并行 KV cache
from .kv_cache_parallel import ParallelKVCache

# Part4K1 SubTask 2.1: VerseNexAttention 品牌别名
# TriSparseAttention 是 VerseNex 原生注意力，暴露 VerseNexAttention 作为品牌统一入口。
VerseNexAttention = TriSparseAttention

__version__ = "0.3.0"

__all__ = [
    # 位置编码
    "RoPE",
    "ALiBi",
    "NoPE",
    # Linear Attention
    "RetNet",
    # Mamba-2
    "Mamba2Block",
    # RWKV-7
    "RWKV7TimeMix",
    "RWKV7ChannelMix",
    "RWKV7Block",
    # Sparse Attention
    "TopKChunkSparseAttention",
    # Hybrid
    "HybridBlock",
    "HybridLM",
    # Part4: TriSparseAttention（三路并行稀疏注意力）
    "TriSparseAttention",
    # Part4: MoD（多稠密分区架构）
    "Router",
    "Expert",
    "DensePart",
    "MoDLayer",
    # Part4: CometSparkNexLM（VerseNex 原生顶层架构）
    "VerseNexBlock",
    "CometSparkNexLM",
    "CometSparkV02",
    # Part4K1 SubTask 2.1: VerseNex 品牌统一入口
    "VerseNexLM",
    "VerseNexAttention",
    # Part4K1 Task 3.2: 分离式并行预测
    "SpeculativeDecoder",
    # Part4K1 Task 3.3: 并行 KV cache
    "ParallelKVCache",
    # Part4K1 Task 4: NexRL 强化学习算法
    "nexrl",
    "NexAgent",
    "NexEnv",
    "NexState",
    "NexAction",
    "NexReward",
    "ChatEnv",
    "MathEnv",
    "CodeEnv",
    "ActionSampler",
    "ExplorationSchedule",
    "repeat_penalty",
    "RewardNormalizer",
    "RewardShaper",
    "Rollout",
    "ParallelRolloutCollector",
    "NexTrainer",
]
