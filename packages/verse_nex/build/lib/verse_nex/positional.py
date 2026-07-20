"""VerseNex: 位置编码（Task 3.1）。

提供三种位置编码：
- RoPE (Rotary Position Embedding)：旋转式位置编码，对 query/key 应用，
  保持内积的相对位置性质。论文：https://arxiv.org/abs/2104.09864
- ALiBi (Attention with Linear Biases)：不显式编码位置，而在 attention score
  上加上一个与相对距离成正比的负偏置。论文：https://arxiv.org/abs/2108.12409
- NoPE：空实现，用于无需位置编码的场景（如因果线性注意力）。

设计要点：
- 预计算 cos/sin 表与 bias 矩阵，避免每步重算。
- forward 接受 VerseTorch Tensor，返回同形状 Tensor。
- 当不需要梯度（如推理）时，直接对 .data 操作以减少开销。
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor


# ---------------------------------------------------------------------------
# Task 3.1: RoPE
# ---------------------------------------------------------------------------


class RoPE:
    """Rotary Position Embedding.

    对 head_dim 维向量做旋转：把 head_dim 拆为两半，
    x = [x1, x2]，rotate_half(x) = [-x2, x1]，
    则 x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)。

    theta_i = base^(-2i/d)  for i in [0, d/2)
    即频率从 1 到 1/base^(d/2 - 1)，对应波长从 2π 到 2π * base^(d/2 - 1)。

    forward 输入支持两种 shape：
        (batch, seq, n_heads, head_dim)  -> 沿 seq 维应用
        (batch, seq, head_dim)           -> 沿 seq 维应用
    """

    def __init__(self, head_dim: int, max_seq_len: int = 32768, base: float = 10000.0):
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 频率向量 theta_i = base^(-2i/d), i in [0, d/2)
        half = head_dim // 2
        i = np.arange(half, dtype=np.float32)
        inv_freq = 1.0 / (base ** (2.0 * i / head_dim))  # (half,)
        self.inv_freq = inv_freq

        # 预计算 [0, max_seq_len) 的 cos/sin 表，形状 (max_seq_len, head_dim)
        positions = np.arange(max_seq_len, dtype=np.float32)  # (T,)
        # angles[t, i] = t * theta_i
        angles = np.outer(positions, inv_freq)  # (T, half)
        # 拼成 head_dim：cos/sin 直接 repeat（与 rotate_half 一致）
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (T, head_dim)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
        # 转为 (1, T, 1, head_dim)，便于广播
        self._cos = cos.reshape(1, max_seq_len, 1, head_dim)
        self._sin = sin.reshape(1, max_seq_len, 1, head_dim)

    @staticmethod
    def rotate_half(x: np.ndarray) -> np.ndarray:
        """rotate_half(x) = concat(-x_half2, x_half1)。"""
        half = x.shape[-1] // 2
        return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    def __call__(self, x: Tensor, seq_dim: int = 1) -> Tensor:
        """应用 RoPE。

        Args:
            x: shape (batch, seq, n_heads, head_dim) 或 (batch, seq, head_dim)
            seq_dim: 序列所在轴（默认 1）

        Returns:
            同形状 Tensor。
        """
        seq_len = x.shape[seq_dim]
        if seq_len > self.max_seq_len:
            # 动态扩展表（罕见路径，避免重复构造）
            self.max_seq_len = seq_len
            positions = np.arange(seq_len, dtype=np.float32)
            angles = np.outer(positions, self.inv_freq)
            cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
            sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
            self._cos = cos.reshape(1, seq_len, 1, self.head_dim)
            self._sin = sin.reshape(1, seq_len, 1, self.head_dim)

        # 切片到当前 seq_len
        cos = self._cos[:, :seq_len]  # (1, T, 1, head_dim)
        sin = self._sin[:, :seq_len]

        x_data = x.data
        # 根据输入 ndim 调整广播 shape
        if x_data.ndim == 4:
            # (B, T, H, D)
            cos_b = cos  # (1, T, 1, D)
            sin_b = sin
        elif x_data.ndim == 3:
            # (B, T, D)
            cos_b = cos.reshape(1, seq_len, self.head_dim)
            sin_b = sin.reshape(1, seq_len, self.head_dim)
        else:
            raise ValueError(f"RoPE expects 3D/4D input, got {x_data.ndim}D")

        rotated = x_data * cos_b + self.rotate_half(x_data) * sin_b

        # 保持计算图：x 是输入 Tensor，rotated 与 x 同 dtype
        requires_grad = x.requires_grad
        out = Tensor(rotated, requires_grad=requires_grad, _children=(x,) if requires_grad else (), _op="rope")

        if requires_grad:
            def _backward():
                # dx = d_rotated * cos + rotate_half(d_rotated) * sin
                #   注意：rotate_half 的雅可比是正交的（转置即 rotate_half 的反向）
                #   实际上 rotate_half(rotate_half(x)) = -x，所以 d/dx[rotate_half(x)] @ v = rotate_half(v)
                #   因此 dx = grad * cos + rotate_half(grad) * sin
                grad = out.grad
                g = grad * cos_b + self.rotate_half(grad) * sin_b
                x._accumulate_grad(g)

            out._backward = _backward
        return out


# ---------------------------------------------------------------------------
# Task 3.1: ALiBi
# ---------------------------------------------------------------------------


class ALiBi:
    """Attention with Linear Biases.

    不显式编码位置；在 attention score 上加 bias：
        score[i, j] += -|i - j| * slope[h]

    slope per head 是几何级数。原始论文对 8 个 head 用
        1/2^1, 1/2^2, ..., 1/2^8
    对一般 n_heads 用 1/2^(8/n) 的几何级数。
    """

    def __init__(self, n_heads: int, max_seq_len: int = 8192):
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        # slope[h] = 1 / 2^( (h+1) * 8 / n_heads )  近似原论文公式
        # 更通用的做法：取最接近 2^(-8/n) 的几何级数
        # 这里用 base = 2^(-8 / n_heads)
        if n_heads <= 8:
            # 直接用 1/2^(h+1)
            slopes = 1.0 / (2.0 ** (np.arange(1, n_heads + 1, dtype=np.float32)))
        else:
            # 通用版：几何级数
            base = 2.0 ** (-8.0 / n_heads)
            slopes = base ** np.arange(1, n_heads + 1, dtype=np.float32)
        self.slopes = slopes  # (n_heads,)

        # 预计算 bias：bias[h, i, j] = -|i-j| * slope[h]
        # 为节省内存，只缓存 base matrix (T, T)，slope 在调用时 broadcast
        positions = np.arange(max_seq_len, dtype=np.float32)
        # 相对距离矩阵 |i-j|
        rel = np.abs(positions[:, None] - positions[None, :])  # (T, T)
        self._rel = rel  # (T, T)

    def get_bias(self, seq_len: int) -> np.ndarray:
        """返回 (n_heads, seq_len, seq_len) 的 bias 矩阵。

        bias[h, i, j] = -|i-j| * slope[h]
        """
        if seq_len > self.max_seq_len:
            # 动态扩展
            self.max_seq_len = seq_len
            positions = np.arange(seq_len, dtype=np.float32)
            self._rel = np.abs(positions[:, None] - positions[None, :])
        rel = self._rel[:seq_len, :seq_len]  # (T, T)
        bias = -rel[None, :, :] * self.slopes[:, None, None]  # (H, T, T)
        return bias.astype(np.float32)

    def __call__(self, x: Tensor) -> Tensor:
        """ALiBi 不修改输入，而是返回 bias 矩阵供 attention 使用。

        这里返回当前序列长度的 bias，shape (n_heads, seq, seq)。
        """
        seq_len = x.shape[1] if x.ndim >= 3 else x.shape[0]
        bias = self.get_bias(seq_len)
        return Tensor(bias, requires_grad=False, _op="alibi_bias")


# ---------------------------------------------------------------------------
# Task 3.1: NoPE
# ---------------------------------------------------------------------------


class NoPE:
    """No Position Embedding. 直接返回输入。"""

    def __call__(self, x: Tensor, seq_dim: int = 1) -> Tensor:
        return x


__all__ = ["RoPE", "ALiBi", "NoPE"]
