"""CUDA / ROCm 自研算子封装。

设计要点
--------
- **前向走自研 kernel**（``torch.ops.verse_nn.*_fwd``），拿到最高吞吐；
- **反向默认用参考实现的 autograd**：把参考前向在 ``enable_grad`` 下重放一遍
  取梯度。这样**梯度数学上严格正确**，不会因为手写反向有 bug 而静默训坏；
  若扩展提供了原生 ``*_bwd``，可后续接入替换（见 ``_NATIVE_BWD``）。
- **不可用即返回 None**，由派发层回退——本模块任何函数都不得抛异常中断训练。

CUDA 与 ROCm 共用本模块：HIP 版由 hipify 从同一份源码生成，算子名一致，
运行时靠 ``torch.version.hip`` 区分（仅影响日志与分块启发式）。
"""

from __future__ import annotations

import torch

from verse_core import get_logger
from verse_nn.kernels import reference as _ref
from verse_nn.kernels._ext import get_ops

logger = get_logger("kernels.cuda")

__all__ = [
    "available",
    "backend_name",
    "add_rms_norm",
    "swiglu",
    "apply_rope",
    "flash_attention",
    "chunked_cross_entropy",
    "kda_chunkwise",
]

#: 已知提供原生融合反向的算子名（占位：当前统一走参考重放反向）
_NATIVE_BWD: frozenset[str] = frozenset()


def available() -> bool:
    """CUDA/ROCm 自研算子是否可用。"""
    return get_ops() is not None


def backend_name() -> str:
    """当前 CUDA 系后端名：``rocm`` 或 ``cuda``。"""
    return "rocm" if getattr(torch.version, "hip", None) is not None else "cuda"


# ---------------------------------------------------------------------------
# autograd 封装
# ---------------------------------------------------------------------------

class _RefBackward(torch.autograd.Function):
    """前向用自研算子、反向用参考实现重放的通用 autograd 包装。

    ``reference`` / ``forward_op`` 是普通可调用对象（非张量），通过 ``forward``
    的位置参数传入而不进 ``save_for_backward``；张量参数按原顺序保存，反向
    重建计算图后求梯度，因此返回的梯度顺序与张量入参严格一致。

    所有本模块的算子入参均为张量，故不需要处理非张量占位。
    """

    @staticmethod
    def forward(ctx, reference, forward_op, *args):  # type: ignore[override]
        ctx.reference = reference
        ctx.forward_op = forward_op
        ctx.save_for_backward(*args)
        with torch.no_grad():
            return forward_op(*args)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        with torch.enable_grad():
            detached = [t.detach().requires_grad_(True) for t in ctx.saved_tensors]
            outputs = ctx.reference(*detached)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            grads = torch.autograd.grad(
                outputs,
                detached,
                grad_outputs=list(grad_outputs),
                allow_unused=True,
                materialize_grads=True,
            )
        # 两个前导参数（reference, forward_op）无梯度
        return (None, None, *grads)


def _apply(forward_op, reference, *args):
    """统一入口：需要梯度时走 autograd 包装，否则直接调用（推理更快）。"""
    needs_grad = torch.is_grad_enabled() and any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    )
    if needs_grad:
        return _RefBackward.apply(reference, forward_op, *args)
    return forward_op(*args)


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """融合残差加 + RMSNorm。返回 None 表示扩展不可用。"""
    ns = get_ops()
    if ns is None:
        return None
    try:
        return _apply(
            lambda a, b, w: ns.add_rms_norm_fwd(a, b, w, eps),
            lambda a, b, w: _ref.add_rms_norm(a, b, w, eps),
            x, residual, weight,
        )
    except Exception as exc:  # pragma: no cover - 设备相关
        logger.warning("add_rms_norm 自研算子执行失败，回退: %s", exc)
        return None


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor | None:
    """融合 SwiGLU 激活。"""
    ns = get_ops()
    if ns is None:
        return None
    try:
        return _apply(
            lambda g, u: ns.swiglu_fwd(g, u),
            lambda g, u: _ref.swiglu(g, u),
            gate, up,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("swiglu 自研算子执行失败，回退: %s", exc)
        return None


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """融合 RoPE（旋转半维，GPT-NeoX 风格）。"""
    ns = get_ops()
    if ns is None:
        return None
    try:
        return _apply(
            lambda a, b, c, s: ns.rope_fwd(a, b, c, s),
            lambda a, b, c, s: _ref.apply_rope(a, b, c, s),
            q, k, cos, sin,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("rope 自研算子执行失败，回退: %s", exc)
        return None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor | None:
    """融合注意力（分块在线 softmax，消除 O(S²) 中间矩阵）。"""
    ns = get_ops()
    if ns is None or dropout_p != 0.0:
        # dropout 需要保存 mask，交给 SDPA 处理更稳妥
        return None
    try:
        s = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
        return _apply(
            lambda a, b, c: ns.flash_attn_fwd(a, b, c, is_causal, s),
            lambda a, b, c: _ref.flash_attention(a, b, c, is_causal=is_causal, scale=s),
            q, k, v,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("flash_attn 自研算子执行失败，回退: %s", exc)
        return None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor | None:
    """融合 lm_head + 分块交叉熵（不物化完整 logits）。"""
    ns = get_ops()
    if ns is None:
        return None
    try:
        return _apply(
            lambda h, w, t: ns.chunked_ce_fwd(h, w, t, chunk_size, ignore_index),
            lambda h, w, t: _ref.chunked_cross_entropy(
                h, w, t, chunk_size=chunk_size, ignore_index=ignore_index
            ),
            hidden, weight, targets,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("chunked_ce 自研算子执行失败，回退: %s", exc)
        return None


def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """KDA chunkwise 融合算子（三角求解 + 状态传递）。"""
    ns = get_ops()
    if ns is None or not hasattr(ns, "kda_chunk_fwd"):
        return None
    try:
        return _apply(
            lambda a, b, c, g, be: ns.kda_chunk_fwd(a, b, c, g, be, chunk_size),
            lambda a, b, c, g, be: _ref.kda_chunkwise(
                a, b, c, g, be, chunk_size=chunk_size
            ),
            q, k, v, gate, beta,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("kda_chunk 自研算子执行失败，回退: %s", exc)
        return None
