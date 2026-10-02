"""RoPE 旋转位置编码。

相比可学习位置编码：外推性更好、无额外参数，是现代 LLM（LLaMA/Qwen 等）的主流选择。
"""

from __future__ import annotations

import torch


class RotaryEmbedding(torch.nn.Module):
    """旋转位置编码（RoPE）。

    Args:
        dim: 每个注意力头的 head_dim（应用 RoPE 的维度）。
        base: 频率基数，默认 10000。
        max_seq_len: 预计算 cos/sin 的最大序列长度（超过会动态重算）。
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # (offset, end, dtype, device, 缓冲区指纹) -> (cos, sin) 的 dtype/切片缓存。
        # cos/sin 表构建后不可变，同一切片在不同层、不同 forward 里完全一致，缓存
        # 可省掉每层一次 fp32->bf16 的 cast 与分配（24 层 × 每次 forward 共 48 次）。
        self._cast_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len
        self._cast_cache.clear()  # 表已重建，旧缓存作废

    def forward(
        self, x: torch.Tensor, seq_len: int, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (cos, sin)，形状 (seq_len, dim)，dtype 与 x 一致。

        Args:
            seq_len: 本次前向的 token 数。
            offset: 位置偏移（KV 缓存解码时 = 已缓存的 token 数）。
        """
        end = offset + seq_len
        if end > self.max_seq_len:
            self._build_cache(end)
        cos_c, sin_c = self.cos_cached, self.sin_cached
        # 缓冲区指纹（device+dtype+data_ptr）确保模块被 ``.to(...)`` 搬走后
        # 不会命中旧缓存；``_build_cache`` 会主动清空。
        key = (
            offset, end, x.dtype,
            str(cos_c.device), cos_c.dtype, cos_c.data_ptr(), sin_c.data_ptr(),
        )
        hit = self._cast_cache.get(key)
        if hit is not None:
            return hit
        cos = cos_c[offset:end].to(dtype=x.dtype)
        sin = sin_c[offset:end].to(dtype=x.dtype)
        if len(self._cast_cache) >= 16:
            self._cast_cache.clear()
        self._cast_cache[key] = (cos, sin)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 q/k 应用 RoPE。

    q, k: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim) -> 广播为 (1, 1, seq_len, head_dim)
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
