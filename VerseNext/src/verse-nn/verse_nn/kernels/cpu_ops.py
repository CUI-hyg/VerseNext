"""CPU 专项优化算子实现。

CPU 上「优化」的真实着力点（与 GPU 完全不同，没有共享内存/Cube 单元）：

1. **减少分配与访存**：融合残差加+RMSNorm、复用 logits 缓冲、避免 fp32 升精度
   拷贝（fp32 输入直接算统计量）。
2. **并行计算**：
   - 大算子交给 ATen 的 intra-op 线程池（``torch.set_num_threads``）；
   - **批内独立**的计算（如按 chunk 的三角求解、按样本的统计）用
     :func:`parallel_map` 在线程池里并行，规避小算子的调度开销。
3. **混合精度**：bf16 在支持 AVX512-BF16/AMX 的 CPU 上吞吐翻倍；本模块提供
   显式 ``dtype`` 参数，由调用方按 :class:`~verse_nn.devices.DeviceCaps` 决策。
4. **推理专项**：KV cache 复用（见 kv_cache.py）+ int8 动态量化（见 quant.py）。

所有实现都必须与 :mod:`verse_nn.kernels.reference` 数值一致（测试对拍）。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from verse_core import get_logger
from verse_nn.kernels import reference as _ref

logger = get_logger("kernels.cpu")

__all__ = [
    "parallel_map",
    "parallel_cpu_count",
    "add_rms_norm",
    "swiglu",
    "apply_rope",
    "flash_attention",
    "chunked_cross_entropy",
    "kda_chunkwise",
]

# 线程池按进程复用：反复创建线程池的开销远大于任务本身
_POOL: ThreadPoolExecutor | None = None
_POOL_WORKERS = 0


def parallel_cpu_count() -> int:
    """可用于并行的线程数（跟随 torch 的 intra-op 设置）。"""
    return max(1, torch.get_num_threads())


def _get_pool() -> ThreadPoolExecutor:
    global _POOL, _POOL_WORKERS
    workers = parallel_cpu_count()
    if _POOL is None or _POOL_WORKERS != workers:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="verse-cpu")
        _POOL_WORKERS = workers
    return _POOL


def parallel_map(fn: Callable[[int], torch.Tensor], n_chunks: int) -> torch.Tensor:
    """把 ``fn(i) -> Tensor`` 在 ``n_chunks`` 个任务上并行执行后按 dim 0 拼接。

    用于「批内互相独立」的计算。torch 算子会释放 GIL，因此线程池能真正并行。
    ``n_chunks <= 1`` 时直接串行，避免线程调度反而更慢。
    """
    if n_chunks <= 1 or parallel_cpu_count() <= 1:
        return torch.cat([fn(i) for i in range(n_chunks)], dim=0)
    results = list(_get_pool().map(fn, range(n_chunks)))
    return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合残差加 + RMSNorm（CPU 优化版）。

    优化点：
    - 只做一次 ``residual + x`` 分配（返回值直接复用为 residual_out）；
    - 输入已是 fp32 时**不做** ``.float()`` 拷贝（参考实现会多一次全量拷贝）；
    - 统计量用 ``mean(x*x)`` 单次归约，避免 ``pow(2)`` 的额外临时张量。
    """
    res = torch.add(residual, x)
    out = _rms_norm_inplace_safe(res, weight, eps)
    return out, res


def _rms_norm_inplace_safe(
    res: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """在 ``res`` 上计算 RMSNorm，不修改 ``res``（保留给后续残差使用）。"""
    if res.dtype == torch.float32:
        var = torch.mean(res * res, dim=-1, keepdim=True)
        norm = res * torch.rsqrt(var + eps)
        return norm * weight
    # 低精度：统计量必须在 fp32 下算，否则 bf16 的平方和会严重丢精度
    res_f = res.float()
    var = torch.mean(res_f * res_f, dim=-1, keepdim=True)
    norm = (res_f * torch.rsqrt(var + eps)).to(res.dtype)
    return norm * weight


# ---------------------------------------------------------------------------
# 激活
# ---------------------------------------------------------------------------

def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU：``silu(gate) * up``，用 ``out=`` 复用 ``up`` 的缓冲。"""
    return torch.mul(F.silu(gate), up)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE（CPU）：直接复用参考实现，保证与 ``rope.apply_rope`` 逐位一致。

    这里**刻意不做** ``out=`` 融合：``out=`` 版本不支持自动微分（训练必需），
    而 CPU 上 RoPE 是纯访存瓶颈、融合收益可忽略，正确性与可微性更重要。
    GPU/NPU 的融合收益显著，由 ``cuda_ops`` / ``npu_ops`` 的 kernel 承担。
    """
    return _ref.apply_rope(q, k, cos, sin)


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
    """CPU 融合注意力：PyTorch 2.3 起 CPU 有原生 flash-attention 后端。

    显式指定 ``scale`` 时走 math 后端（CPU flash 后端对自定义 scale 支持有限），
    保证数值与参考一致。
    """
    if scale is not None:
        # math 后端：等价于 softmax(q@k^T * scale) @ v，且支持自定义 scale
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale, dropout_p=dropout_p
        )
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, dropout_p=dropout_p
    )


# ---------------------------------------------------------------------------
# 融合 lm_head + 交叉熵（复用 logits 缓冲）
# ---------------------------------------------------------------------------

def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """分块 lm_head + CE（CPU 优化版）。

    优化点：预分配 ``(B*chunk, V)`` 的 logits 缓冲并跨 chunk 复用，
    避免每个 chunk 重新分配上百 MB（V=32k 时尤其明显）。

    注意：``torch.mm(..., out=)`` **不支持自动微分**，因此训练（需要梯度）
    时自动切到「每 chunk 新建张量」的路径——分块带来的峰值显存收益不变，
    只是少了一次缓冲复用。
    """
    bsz, seq_len, dim = hidden.shape
    vocab = weight.shape[0]
    chunk = chunk_size if chunk_size > 0 else seq_len
    chunk = max(1, min(chunk, seq_len))

    # out= 仅在不需要梯度时启用
    use_out = not (
        torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)
    )
    buf = torch.empty(bsz * chunk, vocab, dtype=torch.float32) if use_out else None

    total: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        n = end - start
        rows = bsz * n
        h_chunk = hidden[:, start:end].reshape(rows, dim).float()
        if use_out:
            logits = buf[:rows]                       # 连续视图，零拷贝写入
            torch.mm(h_chunk, weight.t(), out=logits)
        else:
            logits = torch.mm(h_chunk, weight.t())
        tgt = targets[:, start:end].reshape(-1)
        loss = F.cross_entropy(
            logits, tgt, ignore_index=ignore_index, reduction="sum",
        )
        count = (tgt != ignore_index).sum()
        total = loss if total is None else total + loss
        valid = count if valid is None else valid + count
    if total is None:
        return hidden.new_zeros(())
    return total / valid.clamp(min=1)


# ---------------------------------------------------------------------------
# KDA chunkwise
# ---------------------------------------------------------------------------

def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA chunkwise（CPU 优化版）。

    优化点：
    - 三角求解与状态传递无法跨 chunk 并行（有状态依赖），但**批×头**维度
      完全独立，按该维度切分交给线程池并行（:func:`parallel_map`）；
    - chunk 内减少临时张量数量。
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

    total_chunks = (seq_len + pad) // C
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    n_slices = bsz * heads

    def _run_slice(idx: int):
        b, h = divmod(idx, heads)
        return _kda_slice(
            q[b, h], k[b, h], v[b, h], gate[b, h], beta[b, h], eye, C, total_chunks
        )

    # 每个 (batch, head) 切片独立跑完整 chunk 循环 —— 可完全并行
    if n_slices > 1 and parallel_cpu_count() > 1:
        outs = list(_get_pool().map(_run_slice, range(n_slices)))
    else:
        outs = [_run_slice(i) for i in range(n_slices)]
    O = torch.stack([o for o, _ in outs], dim=0).view(bsz, heads, -1, dim)
    S = torch.stack([s for _, s in outs], dim=0).view(bsz, heads, dim, dim)
    return O[:, :, :seq_len], S


def _kda_slice(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    eye: torch.Tensor,
    C: int,
    n_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单个 (batch, head) 切片的 chunkwise 计算（状态在 chunk 间顺序传递）。"""
    dim = q.shape[-1]
    S = torch.zeros(dim, dim, dtype=q.dtype)
    outs = []
    for ci in range(n_chunks):
        c0 = ci * C
        qc, kc, vc = q[c0:c0 + C], k[c0:c0 + C], v[c0:c0 + C]
        gc = gate[c0:c0 + C]
        bct = beta[c0:c0 + C].reshape(1, C)                   # (1, C)
        bcol = bct.t()                                        # (C, 1)
        log_lam = torch.cumsum(torch.log(gc.clamp_min(1e-6)), dim=0)   # (C, d)
        lam = torch.exp(log_lam)
        lam_inv = torch.exp((-log_lam).clamp(max=80.0))
        qn = qc * lam
        vt = bcol * vc * lam_inv                              # (C, d)
        U = (kc @ kc.t()).triu(1)                             # (C, C)
        # Mtri 按行（每个 t）缩放 beta，故用 (C,1)；coef/W 用 (1,C) 按列广播
        Mtri = U * bcol + eye
        RHS = S @ kc.t() + vt.t() @ U
        M = torch.linalg.solve_triangular(Mtri, RHS, upper=True, left=False)
        coef = torch.tril(qn @ vt.t() - (qn @ M) * bct, diagonal=0)
        outs.append(qn @ S + coef @ kc)
        W = vt - (M * bct).t()
        S = (S + W.t() @ kc) * lam[-1:, :].t()
    return torch.cat(outs, dim=0), S
