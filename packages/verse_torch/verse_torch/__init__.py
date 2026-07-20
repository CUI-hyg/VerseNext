"""VerseTorch: Pure-Python/NumPy tensor & autograd engine (CPU-first).

提供 PyTorch 风格的 API：
    >>> import numpy as np
    >>> from verse_torch import Tensor
    >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> y.backward()
    >>> x.grad  # 应为 [2.0, 4.0, 6.0]
"""

from .tensor import (
    Tensor,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
    unbroadcast,
)

# Parameter 别名（与 PyTorch 习惯一致：Parameter = Tensor，通过 requires_grad=True 标识）
Parameter = Tensor
from . import nn
from . import optim
from . import losses
from . import training
from . import quantize
from . import parallel
from . import compress
from .parallel import (
    parallel_matmul,
    ParallelLinear,
    parallel_map,
)
from .quantize import (
    quantize_int8,
    dequantize_int8,
    quantize_int4,
    dequantize_int4,
    matmul_int4,
    quantize_ternary,
    dequantize_ternary,
    matmul_ternary,
    QuantizedLinear,
)
from .nn import (
    Module,
    Linear,
    Embedding,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
    ModuleList,
    SwiGLUMLP,
    GQASelfAttention,
    TransformerBlock,
    TransformerLM,
    repeat_kv,
    kaiming_uniform_,
    xavier_uniform_,
    normal_,
    zeros_,
    ones_,
    uniform_,
)
from .optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LambdaLR,
    warmup_cosine_lr,
)
from .losses import (
    cross_entropy,
    nll_loss,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    mse_loss,
    l1_loss,
    kl_div_loss,
)
from .training import (
    cross_entropy_loss,
    EarlyStopping,
    GradientAccumulator,
    CheckpointManager,
    compute_loss_rate,
    plot_loss_curve,
    Trainer,
)
from .compress import (
    OutlierSafePruner,
    LoRALinear,
    KnowledgeDistiller,
    QLinear,
    compress_pipeline,
    prune_only,
    quantize_only,
    lora_only,
    ternary_only,
    distill_only,
    count_parameters as compress_count_parameters,
    count_nonzero_params,
    compute_compressed_bits,
)

__version__ = "0.1.0"

__all__ = [
    # Tensor 核心
    "Tensor",
    "Parameter",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "unbroadcast",
    # 子模块
    "nn",
    "optim",
    "losses",
    "training",
    "quantize",
    "parallel",
    # nn 类
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
    "ModuleList",
    "SwiGLUMLP",
    "GQASelfAttention",
    "TransformerBlock",
    "TransformerLM",
    "repeat_kv",
    "kaiming_uniform_",
    "xavier_uniform_",
    "normal_",
    "zeros_",
    "ones_",
    "uniform_",
    # optim 类
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "LRScheduler",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "LambdaLR",
    "warmup_cosine_lr",
    # losses 函数
    "cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "mse_loss",
    "l1_loss",
    "kl_div_loss",
    # training 模块
    "cross_entropy_loss",
    "EarlyStopping",
    "GradientAccumulator",
    "CheckpointManager",
    "compute_loss_rate",
    "plot_loss_curve",
    "Trainer",
    # quantize 函数
    "quantize_int8",
    "dequantize_int8",
    "quantize_int4",
    "dequantize_int4",
    "matmul_int4",
    "quantize_ternary",
    "dequantize_ternary",
    "matmul_ternary",
    "QuantizedLinear",
    # parallel 函数/类
    "parallel_matmul",
    "ParallelLinear",
    "parallel_map",
    # compress 函数/类
    "compress",
    "OutlierSafePruner",
    "LoRALinear",
    "KnowledgeDistiller",
    "QLinear",
    "compress_pipeline",
    "prune_only",
    "quantize_only",
    "lora_only",
    "ternary_only",
    "distill_only",
    "compress_count_parameters",
    "count_nonzero_params",
    "compute_compressed_bits",
]
