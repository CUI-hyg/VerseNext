"""算子参考实现（reference）：纯 PyTorch、数学精确、无任何融合优化。

用途
----
1. **正确性基准**：所有后端（CPU 优化实现 / CUDA / ROCm / NPU）都必须与
   本模块输出在容差内一致；测试直接对拍。
2. **最终回退**：设备不支持自研算子时，派发层落到这里，保证功能永远可用
   （只是慢）。因此本模块**绝不依赖任何扩展**，只用 torch 原生算子。

约定：所有函数都是纯函数（无状态），输入张量不被原地修改。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "rms_norm",
    "add_rms_norm",
    "swiglu",
    "apply_rope",
    "flash_attention",
    "chunked_cross_entropy",
    "kda_recurrent",
    "kda_chunkwise",
]


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm：``x / sqrt(mean(x^2) + eps) * weight``。

    与 ``blocks.RMSNorm`` 数值口径一致（fp32 统计后转回原 dtype），保证
    启用融合算子不会改变训练结果。
    """
    dtype = x.dtype
    xf = x.float()
    norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return norm.to(dtype) * weight


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """残差相加 + RMSNorm 融合。

    Returns:
        (out, residual_out)：``residual_out = x + residual``，
        ``out = rms_norm(residual_out)``。返回残差是因为后续层还需要它。
    """
    residual_out = residual + x
    return rms_norm(residual_out, weight, eps), residual_out


# ---------------------------------------------------------------------------
# 激活
# ---------------------------------------------------------------------------

def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU 激活：``silu(gate) * up``。"""
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """旋转位置编码，语义与 ``rope.apply_rope`` 严格一致（rotate_half 风格）。

    仓库内 ``RotaryEmbedding`` 返回的 cos/sin 形状为 ``(S, D)``，且
    ``cos[..., :D/2] == cos[..., D/2:]``（``_build_cache`` 里 freqs 复制了两份）。
    因此这里只用前一半，既兼容 ``(S, D)`` 也兼容 ``(S, D/2)`` 的入参：

        out = [x1 * c - x2 * s,  x1 * s + x2 * c]
        x1 = x[..., :D/2], x2 = x[..., D/2:]

    Args:
        q, k: (B, H, S, D)。
        cos, sin: (S, D) 或 (S, D/2)，广播到 (1, 1, S, ·)。
    """
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=k.dtype)
    # (S, ·) -> (1, 1, S, ·) 以广播到 batch/head
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    half = q.shape[-1] // 2
    c = cos[..., :half]
    s = sin[..., :half]

    def _rotate(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * c - t2 * s, t1 * s + t2 * c], dim=-1)

    return _rotate(q), _rotate(k)


# ---------------------------------------------------------------------------
# 注意力
# ---------------------------------------------------------------------------

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """标准缩放点积注意力（参考实现走 PyTorch SDPA）。

    自研 flash kernel 需与此对拍；这里不自己写在线 softmax，直接复用
    SDPA 的数学实现作为基准。
    """
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=scale, dropout_p=dropout_p
    )


# ---------------------------------------------------------------------------
# 融合 lm_head + 交叉熵
# ---------------------------------------------------------------------------

def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """融合 lm_head 投影 + 交叉熵（按序列维分块，避免物化完整 logits）。

    与 ``VerseTransformer._chunked_cross_entropy`` 等价：按有效 token 数
    加权平均，``reduction="sum"`` 累加后除以计数。

    Args:
        hidden: (B, S, D) 最后一层隐状态。
        weight: (V, D) lm_head 权重（通常与 embedding 绑定）。
        targets: (B, S) 目标 id，``ignore_index`` 位置不计入。
        chunk_size: 分块长度；<=0 表示整段一次算。
    """
    seq_len = hidden.size(1)
    chunk = chunk_size if chunk_size > 0 else seq_len
    total: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        logits = (hidden[:, start:end] @ weight.t()).float()
        tgt = targets[:, start:end].reshape(-1).to(logits.device)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt,
            ignore_index=ignore_index, reduction="sum",
        )
        count = (tgt != ignore_index).sum()
        total = loss if total is None else total + loss
        valid = count if valid is None else valid + count
    if total is None:
        return hidden.new_zeros(())
    return total / valid.clamp(min=1)


# ---------------------------------------------------------------------------
# KDA（Kimi Delta Attention）
# ---------------------------------------------------------------------------

def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA 逐步递归（定义式，O(S) 循环，仅作正确性基准）。

    Args:
        q, k, v: (B, H, S, D)。
        gate: (B, H, S, D) 逐通道门控（sigmoid 后）。
        beta: (B, H, S, 1) 写入强度。
        state: (B, H, D, D) 初始状态；None 表示从零开始。

    Returns:
        (out, state)：(B, H, S, D) 输出与末态 (B, H, D, D)。
    """
    bsz, heads, seq_len, dim = q.shape
    if state is None:
        S = torch.zeros(bsz, heads, dim, dim, dtype=q.dtype, device=q.device)
    else:
        S = state
    outs = []
    for t in range(seq_len):
        gt = gate[:, :, t].unsqueeze(-1)            # (b,h,d,1)
        kt = k[:, :, t].unsqueeze(-1)               # (b,h,d,1)
        vt = v[:, :, t].unsqueeze(-1)               # (b,h,d,1)
        bt = beta[:, :, t].unsqueeze(-1)            # (b,h,1,1)
        S = S * gt
        S = S - bt * (S @ kt) @ kt.transpose(-1, -2) + bt * (vt @ kt.transpose(-1, -2))
        o = torch.matmul(S.transpose(-1, -2), q[:, :, t].unsqueeze(-1))[..., 0]
        outs.append(o)
    return torch.stack(outs, dim=2), S


def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA chunkwise 并行参考实现（WY 三角表示，与 attention.KDAAttention 等价）。

    该实现即 ``KDAAttention._chunkwise_forward`` 的独立可调用版本，供自研
    kernel 对拍。逻辑上应满足 ``kda_chunkwise ≈ kda_recurrent``。
    """
    bsz, heads, seq_len, dim = q.shape
    C = chunk_size
    pad = (-seq_len) % C
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        gate = F.pad(gate, (0, 0, 0, pad), value=1.0)
        beta = F.pad(beta, (0, 0, 0, pad))
    S = state if state is not None else torch.zeros(
        bsz, heads, dim, dim, dtype=q.dtype, device=q.device
    )
    outs = []
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    for c0 in range(0, seq_len + pad, C):
        qc = q[:, :, c0:c0 + C]
        kc = k[:, :, c0:c0 + C]
        vc = v[:, :, c0:c0 + C]
        gc = gate[:, :, c0:c0 + C]
        bc = beta[:, :, c0:c0 + C]                            # (b,h,C,1)
        bct = bc.transpose(-1, -2)                            # (b,h,1,C)
        log_lam = torch.cumsum(torch.log(gc.clamp_min(1e-6)), dim=2)
        lam = torch.exp(log_lam)
        lam_inv = torch.exp((-log_lam).clamp(max=80.0))
        qn = qc * lam
        vt = bc * vc * lam_inv                                # Ṽ_t = β_t v_t ⊘ Λ_t
        U = (kc @ kc.transpose(-1, -2)).triu(1)
        Mtri = U * bc + eye
        RHS = S @ kc.transpose(-1, -2) + vt.transpose(-1, -2) @ U
        M = torch.linalg.solve_triangular(Mtri, RHS, upper=True, left=False)
        coef = torch.tril(qn @ vt.transpose(-1, -2) - (qn @ M) * bct, diagonal=0)
        outs.append(qn @ S + coef @ kc)
        W = vt - (M * bct).transpose(-1, -2)
        S = (S + W.transpose(-1, -2) @ kc) * lam[:, :, -1:, :].transpose(-1, -2)
    return torch.cat(outs, dim=2)[:, :, :seq_len], S
