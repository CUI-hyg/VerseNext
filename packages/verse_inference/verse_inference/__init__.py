"""VerseInference: Model loading, state caching, streaming generation.

提供：
- ``ModelLoader``: 从 HF repo 或本地路径加载预训练 LM 到 VerseNex + VerseTorch
- ``StateCache``: Mamba/RWKV 的递归状态缓存
- ``Sampler`` / ``GreedySampler``: Token 采样器（temperature / top_k / top_p）
- ``StreamingGenerator``: 流式生成器（逐步产生 token）
"""

from .model_loader import ModelLoader
from .cache import StateCache
from .sampler import Sampler, GreedySampler
from .generator import StreamingGenerator

__version__ = "0.1.0"

__all__ = [
    "ModelLoader",
    "StateCache",
    "Sampler",
    "GreedySampler",
    "StreamingGenerator",
]
