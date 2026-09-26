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
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

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
        cos = self.cos_cached[offset:end].to(dtype=x.dtype)
        sin = self.sin_cached[offset:end].to(dtype=x.dtype)
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
