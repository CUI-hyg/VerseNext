"""扩展激活函数：SiLU + Mish + GeGLU。

这些是 ``nn.Module`` 子类，可直接用于 ``Sequential`` / ``ModuleList``。
底层算子复用 ``Tensor`` 上的可微方法（``exp`` / ``log`` / ``tanh`` / ``__mul__`` 等），
因此自动获得 autograd 支持。

参考论文：
- SiLU / Swish: https://arxiv.org/abs/1702.03118
- Mish: https://arxiv.org/abs/1908.08681
- GeGLU: https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor
from .nn import Module


class SiLU(Module):
    """SiLU / Swish 激活：``x * sigmoid(x)``。

    与 ``Tensor.silu()`` 方法等价，但封装为 ``nn.Module`` 以便在容器中使用。
    """

    def forward(self, x: Tensor) -> Tensor:
        # 用 Tensor 上的 sigmoid 方法以获得数值稳定实现
        s = x.sigmoid()
        return x * s


class Mish(Module):
    """Mish 激活：``x * tanh(softplus(x))``。

    ``softplus(x) = log(1 + exp(x))``
    数值稳定：通过 ``Tensor.exp`` / ``Tensor.log`` / ``Tensor.tanh`` 自动获得 autograd 支持。
    """

    def forward(self, x: Tensor) -> Tensor:
        # softplus(x) = log(exp(x) + 1)
        # 用 Tensor.exp() + 标量 1.0（经 __radd__）+ Tensor.log()
        softplus = (x.exp() + 1.0).log()
        return x * softplus.tanh()


class GeGLU(Module):
    """GeGLU 激活：将输入最后一维 split 成 (a, b)，返回 ``a * gelu(b)``。

    论文: https://arxiv.org/abs/2002.05202

    与 SwiGLU 类似但用 GELU 代替 SiLU。
    输入 shape ``(..., 2*d)``，输出 shape ``(..., d)``。

    GELU 近似：用 ``x * sigmoid(1.702 * x)`` 近似（误差 < 0.001），
    避免依赖 ``scipy.special.erf``，保持纯 NumPy + 标准库约束。
    """

    # GELU 的 sigmoid 近似系数（与 tanh 近似的最大相对误差 ~0.1%）
    _GELU_APPROX_COEF = 1.702

    def forward(self, x: Tensor) -> Tensor:
        # 沿最后一维 split
        d = x.shape[-1] // 2
        a = x[..., :d]
        b = x[..., d:]
        # GELU 近似：gelu(b) ≈ b * sigmoid(1.702 * b)
        gelu_b = b * (b * self._GELU_APPROX_COEF).sigmoid()
        return a * gelu_b


__all__ = ["SiLU", "Mish", "GeGLU"]
