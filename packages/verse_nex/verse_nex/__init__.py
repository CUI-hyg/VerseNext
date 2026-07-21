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
from .cometspark import VerseNexBlock, CometSparkNexLM, CometSparkV02

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
]
