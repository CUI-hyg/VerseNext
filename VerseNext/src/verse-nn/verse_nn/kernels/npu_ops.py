"""华为昇腾 CANN NPU 算子封装。

昇腾上的算子来源分两层，本模块按「优先用 CANN 原生融合算子，其次用自研
AscendC 算子，最后回退参考实现」的顺序尝试：

1. **CANN 原生融合算子**（经 ``torch_npu`` 暴露）——这是昇腾上性能最好的路径，
   由 CANN 图编译器做算子融合与 Cube/Vector 流水：
   ``npu_rms_norm`` / ``npu_swiglu`` / ``npu_rotary_mul`` /
   ``npu_fusion_attention``（训练）/ ``npu_fused_infer_attention_score``（推理）。
2. **自研 AscendC 算子**（``torch.ops.verse_nn.*_fwd``，dispatch key
   ``PrivateUse1``，见 ``csrc/ascend``）——
   覆盖 CANN 未直接暴露的算子（分块 CE、KDA chunkwise）。
3. **参考实现**——派发层兜底。

所有候选调用都包在 try/except 里：CANN 各版本 API 签名差异较大，
任何一个候选失败都只是「换下一个」，绝不中断训练。
"""

from __future__ import annotations

from typing import Callable

import torch

from verse_core import get_logger
from verse_nn.kernels import reference as _ref

logger = get_logger("kernels.npu")

__all__ = [
    "available",
    "npu_available",
    "add_rms_norm",
    "swiglu",
    "apply_rope",
    "flash_attention",
    "chunked_cross_entropy",
    "kda_chunkwise",
]


def npu_available() -> bool:
    """torch_npu 是否可用。"""
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def available() -> bool:
    """NPU 侧是否至少有一条可用算子路径（CANN 或自研）。"""
    return npu_available()


def _npu():
    import torch_npu  # noqa: F401

    return torch.npu


def _custom_ns():
    """自研 AscendC 算子命名空间（未编译返回 None）。"""
    try:
        from verse_nn.kernels._ext import get_ops

        return get_ops("npu")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# autograd：前向走设备算子，反向用参考实现重放（与 CUDA 侧同构）
# ---------------------------------------------------------------------------

class _RefBackward(torch.autograd.Function):
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
                outputs, detached, grad_outputs=list(grad_outputs),
                allow_unused=True, materialize_grads=True,
            )
        return (None, None, *grads)


def _try(
    candidates: list[tuple[str, Callable]],
    reference: Callable,
    *args,
):
    """依次尝试候选实现；需要梯度时自动套 autograd 包装。

    返回首个成功且非 None 的结果；全部失败返回 None（由派发层回退）。
    """
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    need_grad = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    last_err: Exception | None = None
    for name, fn in candidates:
        try:
            out = _RefBackward.apply(reference, fn, *args) if need_grad else fn(*args)
            if out is not None:
                return out
        except Exception as exc:  # pragma: no cover - 版本/设备相关
            last_err = exc
            logger.debug("NPU 候选 %s 失败: %s", name, exc)
    if last_err is not None:
        logger.debug("NPU 全部候选失败，回退参考实现: %s", last_err)
    return None


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """残差加 + RMSNorm。优先 CANN ``npu_rms_norm``，其次自研 AscendC。"""
    if not npu_available():
        return None

    def _cann(a, b, w):
        res = a + b
        out = _npu().npu_rms_norm(res, w, eps)
        y = out[0] if isinstance(out, (tuple, list)) else out
        return y, res

    def _ascendc(a, b, w):
        ns = _custom_ns()
        if ns is None or not hasattr(ns, "add_rms_norm_fwd"):
            return None
        return ns.add_rms_norm_fwd(a, b, w, eps)

    return _try(
        [("cann", _cann), ("ascendc", _ascendc)],
        lambda a, b, w: _ref.add_rms_norm(a, b, w, eps),
        x, residual, weight,
    )


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor | None:
    """SwiGLU 激活。CANN ``npu_swiglu`` 需要 [gate, up] 拼接输入。"""
    if not npu_available():
        return None

    def _cann(g, u):
        return _npu().npu_swiglu(torch.cat([g, u], dim=-1), dim=-1)

    def _ascendc(g, u):
        ns = _custom_ns()
        if ns is None or not hasattr(ns, "swiglu_fwd"):
            return None
        return ns.swiglu_fwd(g, u)

    return _try([("cann", _cann), ("ascendc", _ascendc)], _ref.swiglu, gate, up)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """RoPE。优先 CANN ``npu_rotary_mul``（各版本签名差异大，失败即换路）。"""
    if not npu_available():
        return None

    def _cann(a, b, c, s):
        npu = _npu()
        return npu.npu_rotary_mul(a, c, s), npu.npu_rotary_mul(b, c, s)

    def _ascendc(a, b, c, s):
        ns = _custom_ns()
        if ns is None or not hasattr(ns, "rope_fwd"):
            return None
        return ns.rope_fwd(a, b, c, s)

    return _try([("cann", _cann), ("ascendc", _ascendc)], _ref.apply_rope, q, k, cos, sin)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor | None:
    """融合注意力。训练用 ``npu_fusion_attention``，推理用 ``npu_fused_infer_attention_score``。"""
    if not npu_available():
        return None
    s = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
    head_num = q.shape[1]
    sparse_mode = 2 if is_causal else 0

    def _cann_train(a, b, c):
        out = _npu().npu_fusion_attention(
            a, b, c, head_num,
            input_layout="BNSD",
            scale=s,
            keep_prob=1.0 - dropout_p,
            sparse_mode=sparse_mode,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def _cann_infer(a, b, c):
        out = _npu().npu_fused_infer_attention_score(
            a, b, c, num_heads=head_num,
            input_layout="BNSD",
            scale=s,
            sparse_mode=sparse_mode,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def _ascendc(a, b, c):
        ns = _custom_ns()
        if ns is None or not hasattr(ns, "flash_attn_fwd"):
            return None
        return ns.flash_attn_fwd(a, b, c, is_causal, s)

    return _try(
        [("cann_train", _cann_train), ("cann_infer", _cann_infer), ("ascendc", _ascendc)],
        lambda a, b, c: _ref.flash_attention(a, b, c, is_causal=is_causal, scale=s),
        q, k, v,
    )


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor | None:
    """分块 lm_head + CE。CANN 无直接对应算子，走自研 AscendC。"""
    if not npu_available():
        return None
    ns = _custom_ns()
    if ns is None or not hasattr(ns, "chunked_ce_fwd"):
        return None

    def _ascendc(h, w, t):
        return ns.chunked_ce_fwd(h, w, t, chunk_size, ignore_index)

    return _try(
        [("ascendc", _ascendc)],
        lambda h, w, t: _ref.chunked_cross_entropy(
            h, w, t, chunk_size=chunk_size, ignore_index=ignore_index
        ),
        hidden, weight, targets,
    )


def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """KDA chunkwise。CANN 无对应算子，走自研 AscendC。"""
    if not npu_available():
        return None
    ns = _custom_ns()
    if ns is None or not hasattr(ns, "kda_chunk_fwd"):
        return None

    def _ascendc(a, b, c, g, be):
        return ns.kda_chunk_fwd(a, b, c, g, be, chunk_size)

    return _try(
        [("ascendc", _ascendc)],
        lambda a, b, c, g, be: _ref.kda_chunkwise(a, b, c, g, be, chunk_size=chunk_size),
        q, k, v, gate, beta,
    )
