"""算子派发：自研扩展 → torch 内建融合 → 参考实现。

三级回退
--------
1. **自研扩展**（CUDA/ROCm 的 ``.so``、NPU 的 CANN/AscendC）—— 追求最高吞吐；
2. **torch 内建融合**（如 ``F.scaled_dot_product_attention``、CPU 上的
   ``_int_mm``）—— 无需编译，跨平台可用；
3. **参考实现**（:mod:`verse_nn.kernels.reference`）—— 永远可用的正确性兜底。

任何一级失败都静默降级（记 debug 日志），**绝不因算子问题中断训练**。
这是「性能优化不能牺牲可用性」的工程底线。

全局开关见 :class:`KernelPolicy`；训练脚本可据此做 A/B 基准对比。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import torch
import torch.nn.functional as F

from verse_core import get_logger
from verse_nn.kernels import reference as _ref

logger = get_logger("kernels.dispatch")

__all__ = [
    "KernelPolicy",
    "get_policy",
    "set_policy",
    "backend_of",
    "add_rms_norm",
    "swiglu",
    "apply_rope",
    "flash_attention",
    "chunked_cross_entropy",
    "kda_chunkwise",
    "available_ops",
]


@dataclass(frozen=True)
class KernelPolicy:
    """算子派发策略（不可变，用 :func:`set_policy` 整体替换）。

    Args:
        enable_fused: 总开关。False 时全部走参考实现（做正确性/性能对照用）。
        enable_custom_kernels: 是否允许使用自研扩展。
        enable_flash_attn: 是否允许使用融合注意力。
        enable_quant: 是否允许 CPU int8 量化推理。
        ce_chunk_size: 融合 CE 的分块长度，<=0 交给各实现自定。
    """

    enable_fused: bool = True
    enable_custom_kernels: bool = True
    enable_flash_attn: bool = True
    enable_quant: bool = False
    ce_chunk_size: int = 0


_POLICY = KernelPolicy()


def get_policy() -> KernelPolicy:
    return _POLICY


def set_policy(**overrides) -> KernelPolicy:
    """更新全局策略（返回新策略）。"""
    global _POLICY
    _POLICY = replace(_POLICY, **overrides)
    logger.info("算子策略已更新: %s", _POLICY)
    return _POLICY


@lru_cache(maxsize=16)
def backend_of(device_type: str) -> str:
    """把 ``torch.device.type`` 映射为后端名（结果缓存，避免热路径开销）。"""
    if device_type == "cuda":
        return "rocm" if getattr(torch.version, "hip", None) is not None else "cuda"
    if device_type in ("npu", "xpu", "mps"):
        return device_type
    return "cpu"


def _tensor_backend(*tensors: torch.Tensor) -> str:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return backend_of(t.device.type)
    return "cpu"


# ---------------------------------------------------------------------------
# 算子派发
# ---------------------------------------------------------------------------

def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """残差加 + RMSNorm（融合派发）。"""
    if _POLICY.enable_fused:
        backend = _tensor_backend(x)
        if _POLICY.enable_custom_kernels:
            if backend in ("cuda", "rocm"):
                from verse_nn.kernels import cuda_ops

                out = cuda_ops.add_rms_norm(x, residual, weight, eps)
                if out is not None:
                    return out
            elif backend == "npu":
                from verse_nn.kernels import npu_ops

                out = npu_ops.add_rms_norm(x, residual, weight, eps)
                if out is not None:
                    return out
        if backend == "cpu":
            from verse_nn.kernels import cpu_ops

            return cpu_ops.add_rms_norm(x, residual, weight, eps)
    return _ref.add_rms_norm(x, residual, weight, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU（融合派发）。"""
    if _POLICY.enable_fused:
        backend = _tensor_backend(gate)
        if _POLICY.enable_custom_kernels:
            if backend in ("cuda", "rocm"):
                from verse_nn.kernels import cuda_ops

                out = cuda_ops.swiglu(gate, up)
                if out is not None:
                    return out
            elif backend == "npu":
                from verse_nn.kernels import npu_ops

                out = npu_ops.swiglu(gate, up)
                if out is not None:
                    return out
        if backend == "cpu":
            from verse_nn.kernels import cpu_ops

            return cpu_ops.swiglu(gate, up)
    return _ref.swiglu(gate, up)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE（融合派发）。"""
    if _POLICY.enable_fused:
        backend = _tensor_backend(q)
        if _POLICY.enable_custom_kernels:
            if backend in ("cuda", "rocm"):
                from verse_nn.kernels import cuda_ops

                out = cuda_ops.apply_rope(q, k, cos, sin)
                if out is not None:
                    return out
            elif backend == "npu":
                from verse_nn.kernels import npu_ops

                out = npu_ops.apply_rope(q, k, cos, sin)
                if out is not None:
                    return out
        if backend == "cpu":
            from verse_nn.kernels import cpu_ops

            return cpu_ops.apply_rope(q, k, cos, sin)
    return _ref.apply_rope(q, k, cos, sin)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    scale: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """融合注意力（自研 kernel → SDPA → 参考）。"""
    if _POLICY.enable_fused and _POLICY.enable_flash_attn and _POLICY.enable_custom_kernels:
        backend = _tensor_backend(q)
        if backend in ("cuda", "rocm"):
            from verse_nn.kernels import cuda_ops

            out = cuda_ops.flash_attention(
                q, k, v, is_causal=is_causal, scale=scale, dropout_p=dropout_p
            )
            if out is not None:
                return out
        elif backend == "npu":
            from verse_nn.kernels import npu_ops

            out = npu_ops.flash_attention(
                q, k, v, is_causal=is_causal, scale=scale, dropout_p=dropout_p
            )
            if out is not None:
                return out
    # 二级：torch 内建 SDPA（CPU 上也有 flash 后端）
    return _ref.flash_attention(
        q, k, v, is_causal=is_causal, scale=scale, dropout_p=dropout_p
    )


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """融合 lm_head + 分块 CE。"""
    chunk = _POLICY.ce_chunk_size if chunk_size is None else chunk_size
    if _POLICY.enable_fused and _POLICY.enable_custom_kernels:
        backend = _tensor_backend(hidden)
        if backend in ("cuda", "rocm"):
            from verse_nn.kernels import cuda_ops

            out = cuda_ops.chunked_cross_entropy(
                hidden, weight, targets, chunk_size=chunk, ignore_index=ignore_index
            )
            if out is not None:
                return out
        elif backend == "npu":
            from verse_nn.kernels import npu_ops

            out = npu_ops.chunked_cross_entropy(
                hidden, weight, targets, chunk_size=chunk, ignore_index=ignore_index
            )
            if out is not None:
                return out
        elif backend == "cpu":
            from verse_nn.kernels import cpu_ops

            return cpu_ops.chunked_cross_entropy(
                hidden, weight, targets, chunk_size=chunk, ignore_index=ignore_index
            )
    return _ref.chunked_cross_entropy(
        hidden, weight, targets, chunk_size=chunk, ignore_index=ignore_index
    )


def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA chunkwise（融合派发）。"""
    if _POLICY.enable_fused and _POLICY.enable_custom_kernels:
        backend = _tensor_backend(q)
        if backend in ("cuda", "rocm"):
            from verse_nn.kernels import cuda_ops

            out = cuda_ops.kda_chunkwise(
                q, k, v, gate, beta, chunk_size=chunk_size
            )
            if out is not None:
                return out
        elif backend == "npu":
            from verse_nn.kernels import npu_ops

            out = npu_ops.kda_chunkwise(q, k, v, gate, beta, chunk_size=chunk_size)
            if out is not None:
                return out
        elif backend == "cpu":
            from verse_nn.kernels import cpu_ops

            return cpu_ops.kda_chunkwise(q, k, v, gate, beta, chunk_size=chunk_size)
    return _ref.kda_chunkwise(q, k, v, gate, beta, chunk_size=chunk_size)


def available_ops() -> dict:
    """各后端当前实际生效的算子实现（诊断/基准用）。"""
    from verse_nn.kernels._ext import extension_available

    return {
        "policy": {
            "fused": _POLICY.enable_fused,
            "custom_kernels": _POLICY.enable_custom_kernels,
            "flash_attn": _POLICY.enable_flash_attn,
            "quant": _POLICY.enable_quant,
        },
        "custom_ext": {
            backend: extension_available(backend)
            for backend in ("cpu", "cuda", "rocm", "npu")
        },
        "impl": {
            "cpu": "cpu_ops (ATen 优化 + 线程池)",
            "cuda": "cuda_ops (.cu)" if extension_available("cuda") else "reference",
            "rocm": "cuda_ops (hipify)" if extension_available("rocm") else "reference",
            "npu": "npu_ops (CANN/AscendC)" if extension_available("npu") else "reference",
        },
    }
