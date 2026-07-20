"""VerseNex: Transformer-alternative architectures (Mamba-2, RWKV-7, Linear Attention, Hybrid).

提供线性复杂度的 Transformer 替代架构库，主推：
- Mamba-2 SSM (selective state space model with SSD)
- RWKV-7 (time mixing + channel mixing)
- Linear Attention (RetNet 风格 retention + chunkwise)
- Sparse Attention (top-k chunk sparse)
- Hybrid Block/LM (SSM + Sparse Attention 混合)
- 位置编码 (RoPE, ALiBi, NoPE)

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

__version__ = "0.1.0"

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
]
