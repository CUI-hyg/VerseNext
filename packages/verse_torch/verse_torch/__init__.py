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
from . import nn
from . import optim
from . import losses
from . import quantize
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

__version__ = "0.1.0"

__all__ = [
    # Tensor 核心
    "Tensor",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "unbroadcast",
    # 子模块
    "nn",
    "optim",
    "losses",
    "quantize",
    # nn 类
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
    "ModuleList",
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
    # losses 函数
    "cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "mse_loss",
    "l1_loss",
    "kl_div_loss",
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
]
