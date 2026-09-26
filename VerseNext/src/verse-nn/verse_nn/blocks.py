"""Transformer Block：pre-norm 残差结构 + SwiGLU FFN。

架构要点（对齐现代 LLM 实践，性能优于原始 post-norm/GELU 方案）：
- Pre-Norm：训练更深网络更稳定。
- SwiGLU：SiLU 门控线性单元（gate/up 融合单 GEMM），同等参数量下效果优于 GELU-MLP。
- RMSNorm：比 LayerNorm 少一次均值统计，更快。
- 注意力可插拔：sdpa / dsa / kda，经 ``build_attention`` 工厂构建。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from verse_nn import kernels
from verse_nn.attention import build_attention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.to(dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN：``(SiLU(x @ W_gate) * (x @ W_up)) @ W_down``。

    gate/up 融合为单次 GEMM（``gate_up_proj``，输出 2×hidden 后切分），
    减少 kernel 启动与内存往返。隐藏层默认 ``~8/3 * d_model`` 对齐 256。
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.hidden_dim = hidden_dim
        self.gate_up_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        # 走融合算子派发：CUDA/ROCm/NPU 命中自研 kernel，CPU 用 ATen 优化路径
        return self.dropout(self.down_proj(kernels.swiglu(gate, up)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer Block。

    Args:
        d_model: 隐藏维度。
        num_heads: Q 头数。
        num_kv_heads: KV 头数（GQA），None 表示与 num_heads 相同。
        ffn_hidden_dim: FFN 隐藏维度，None 自动取 8/3*d_model 对齐 256。
        dropout: 残差/attention dropout。
        attn_type: 注意力类型（sdpa / dsa / kda），透传其余 kwargs 给注意力工厂。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attn_type: str = "sdpa",
        **attn_kwargs,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = build_attention(
            attn_type, d_model, num_heads, num_kv_heads, dropout, **attn_kwargs
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden_dim, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        layer_cache=None,
        past_kv=None,
        use_cache: bool = False,
    ):
        use_cache_path = layer_cache is not None or past_kv is not None or use_cache
        if use_cache_path:
            attn_out, present = self.attn(
                self.attn_norm(x), layer_cache=layer_cache, past_kv=past_kv, use_cache=True
            )
            x = x + self.resid_dropout(attn_out)
            x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
            return x, present
        x = x + self.resid_dropout(self.attn(self.attn_norm(x)))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x
