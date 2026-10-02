"""华为昇腾 CANN NPU 算子封装。

昇腾上的算子来源分两层，本模块按「优先用 CANN 原生融合算子，其次用自研
AscendC 算子，最后回退参考实现」的顺序尝试：

1. **CANN 原生融合算子**（经 ``torch_npu`` 暴露）——这是昇腾上性能最好的路径，
   由 CANN 图编译器做算子融合与 Cube/Vector 流水：

   ===========================  ==========================================
   本模块算子                   昇腾原生算子
   ===========================  ==========================================
   ``add_rms_norm``             ``torch_npu.npu_add_rms_norm``（残差加与
                                RMSNorm 一步融合，返回 y/rstd/residual_out）
   ``swiglu``                   ``torch_npu.npu_swiglu``（沿最后一维切分）
   ``apply_rope``               ``torch_npu.npu_rotary_mul``（half 模式）
   ``flash_attention``          ``torch_npu.npu_fusion_attention``（训练，
                                显式因果掩码 + sparse_mode=0）
   ===========================  ==========================================

   注意：这些算子在 ``torch_npu`` **模块级**命名空间（``torch_npu.npu_xxx``），
   而不是 ``torch.npu.npu_xxx``——后者不存在，早期实现曾因此静默全量回退。

2. **自研 AscendC 算子**（``torch.ops.verse_nn.*_fwd``，dispatch key
   ``PrivateUse1``，见 ``csrc/ascend``）——目前实现了 ``add_rms_norm_fwd`` 与
   ``swiglu_fwd``。**只有 schema、没有 ``m.impl`` 的算子绝不能派发**：torch_npu
   的 ``VariableFallbackKernel`` 会把张量搬回 CPU，而它们又没有 CPU kernel，
   结果每步白付一次失败的派发 + 异常。故用 :func:`_has_npu_kernel` 逐个确认
   （见下）。
3. **参考实现**——派发层兜底。

两处与「CANN 优先」不同的例外：

- ``swiglu`` **自研 AscendC 优先**：CANN ``npu_swiglu`` 只吃拼接张量，调用方
  得先 ``torch.cat``。实测 910_9362 上 verse 74.5µs vs CANN(含 cat) 182.1µs。
- ``chunked_cross_entropy`` / ``kda_chunkwise`` 目前**没有 NPU 实现**，
  ``_custom_op`` 返回 None，由派发层用设备上的参考实现（不再 CPU 往返）。

所有候选调用都包在 try/except 里：CANN 各版本 API 签名差异较大，
任何一个候选失败都只是「换下一个」，绝不中断训练。
"""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=1)
def npu_available() -> bool:
    """torch_npu 是否可用（结果缓存）。

    每个算子入口都会调它，而一个 transformer 层就要调 4~6 次；``torch_npu``
    的可用性在一个进程内不会变，缓存掉 ``is_available()`` 的 C 调用与 import
    查找能省下可观的 Python 侧开销。
    """
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def available() -> bool:
    """NPU 侧是否至少有一条可用算子路径（CANN 或自研）。"""
    return npu_available()


@functools.lru_cache(maxsize=1)
def _torch_npu():
    """返回 ``torch_npu`` 模块（原生融合算子都在这个模块级命名空间下）。"""
    import torch_npu

    return torch_npu


@functools.lru_cache(maxsize=1)
def _custom_ns():
    """自研 AscendC 算子命名空间（未编译返回 None）。"""
    try:
        from verse_nn.kernels._ext import get_ops

        return get_ops("npu")
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _has_npu_kernel(op_name: str) -> bool:
    """``verse_nn::<op_name>`` 是否真的注册了 PrivateUse1（NPU）实现。

    自研算子的 schema 由 ``TORCH_LIBRARY`` 统一 ``m.def``，但**实现**是按需
    ``m.impl`` 的（见 ``csrc/ascend/bindings_npu.cpp``）。只有 schema、没有
    PrivateUse1 实现的算子，在 NPU 张量上会被 torch_npu 的
    ``VariableFallbackKernel`` 接住——把张量搬回 CPU 算完再搬回来。这在训练
    热路径上是灾难性的（每个 step 两次设备往返 + 同步），且完全静默，只在日志
    里留一行 warning。

    因此派发层必须先确认实现存在，否则宁可返回 None 让上层用设备上的参考实现。
    """
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                f"verse_nn::{op_name}", "PrivateUse1"
            )
        )
    except Exception:  # pragma: no cover - 版本差异
        return False


def _custom_op(op_name: str):
    """取自研算子；未编译或**未挂 NPU 实现**时返回 None。"""
    ns = _custom_ns()
    if ns is None or not _has_npu_kernel(op_name):
        return None
    return getattr(ns, op_name, None)


def _clear_caches() -> None:
    """清空模块级缓存（测试/热更新 torch_npu 环境后调用）。"""
    npu_available.cache_clear()
    _torch_npu.cache_clear()
    _custom_ns.cache_clear()
    _has_npu_kernel.cache_clear()
    _CAUSAL_MASK_CACHE.clear()
    _ROPE_COEFF_CACHE.clear()


# 因果掩码缓存：``npu_fusion_attention`` 需要显式上三角 bool 掩码，训练时每个
# 层、每次 forward 都会重建一张 (S, S) 张量（S=4096 时 16MB）。掩码只读且完全
# 由 (seq_len, kv_len) 决定，按设备缓存可省掉每次的分配与 H2D/建图开销。
_CAUSAL_MASK_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_CAUSAL_MASK_CACHE_MAX = 16


def _causal_mask(seq_len: int, kv_len: int, device) -> torch.Tensor:
    """``True`` 表示屏蔽的上三角掩码（与 ``SDPA(is_causal=True)`` 对拍一致）。"""
    key = (seq_len, kv_len, str(device))
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = torch.triu(
        torch.ones(seq_len, kv_len, dtype=torch.bool, device=device), diagonal=1
    )
    if len(_CAUSAL_MASK_CACHE) >= _CAUSAL_MASK_CACHE_MAX:
        _CAUSAL_MASK_CACHE.clear()
    _CAUSAL_MASK_CACHE[key] = mask
    return mask


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
    """残差加 + RMSNorm。优先 CANN ``npu_add_rms_norm``，其次自研 AscendC。

    ``npu_add_rms_norm`` 一次完成 ``res = x + residual`` 与 ``y = rms_norm(res)``，
    返回 ``(y, rstd, res)``；本模块只取 ``(y, res)`` 以匹配参考实现的契约。
    """
    if not npu_available():
        return None

    def _cann(a, b, w):
        out = _torch_npu().npu_add_rms_norm(a, b, w, float(eps))
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            return None  # 该 CANN 版本未返回 residual_out，交回退
        return out[0], out[-1]

    def _ascendc(a, b, w):
        fn = _custom_op("add_rms_norm_fwd")
        return None if fn is None else fn(a, b, w, float(eps))

    return _try(
        [("cann", _cann), ("ascendc", _ascendc)],
        lambda a, b, w: _ref.add_rms_norm(a, b, w, eps),
        x, residual, weight,
    )


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor | None:
    """SwiGLU 激活。

    优先级与其余算子**相反**：自研 AscendC 在前、CANN 在后。原因是 CANN 的
    ``npu_swiglu`` 只接受单个拼接张量，调用方必须先 ``torch.cat([gate, up])``
    多一次显存往返——本机实测（910_9362）verse 23.9µs vs npu_swiglu(含 cat)
    46.1µs，cat 本身就占 18.0µs。自研算子直接吃两个输入，反而更快。
    """
    if not npu_available():
        return None

    def _ascendc(g, u):
        fn = _custom_op("swiglu_fwd")
        return None if fn is None else fn(g, u)

    def _cann(g, u):
        return _torch_npu().npu_swiglu(torch.cat([g, u], dim=-1), dim=-1)

    return _try([("ascendc", _ascendc), ("cann", _cann)], _ref.swiglu, gate, up)


# 系数变换缓存：``RotaryEmbedding`` 会返回同一个 cos/sin 张量对象（内部缓存），
# 同一对象在同一 dtype/device/head_dim 下的变换结果恒定。键含 ``id`` 与
# ``_version``，并把原张量一并持有（防止 ``id`` 被回收后复用导致误命中）。
_ROPE_COEFF_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ROPE_COEFF_CACHE_MAX = 32


def _rope_coeff(c: torch.Tensor, dtype: torch.dtype, device, head_dim: int) -> torch.Tensor:
    """把 cos/sin 规整成 ``npu_rotary_mul`` 需要的 ``(1, 1, S, D)`` 形状。

    仓库的 ``RotaryEmbedding`` 返回 ``(S, D)``（且两半相同）；参考实现也兼容
    ``(S, D/2)`` 的紧凑形式。``npu_rotary_mul`` 的 half 模式要求系数与输入同宽，
    因此紧凑形式需沿最后一维复制一份。

    变换本身可能含一次 ``.to(dtype)``（cos 是 fp32，而 autocast 下 q 是 bf16），
    逐层重复执行会白白多出 2×层数 次 cast/分配，故按张量身份缓存。
    """
    key = (id(c), c._version, dtype, str(device), head_dim)
    hit = _ROPE_COEFF_CACHE.get(key)
    if hit is not None:
        return hit[1]
    out = c.to(dtype=dtype, device=device)
    half = head_dim // 2
    if out.shape[-1] == half and half != head_dim:
        out = torch.cat([out, out], dim=-1)
    while out.dim() < 4:
        out = out.unsqueeze(0)
    out = out.contiguous()
    if len(_ROPE_COEFF_CACHE) >= _ROPE_COEFF_CACHE_MAX:
        _ROPE_COEFF_CACHE.clear()
    _ROPE_COEFF_CACHE[key] = (c, out)  # 持有 c，保证 id 不被复用
    return out


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """RoPE。优先 CANN ``npu_rotary_mul``（half 模式），其次自研 AscendC。"""
    if not npu_available():
        return None

    def _cann(a, b, c, s):
        tn = _torch_npu()
        head_dim = a.shape[-1]
        cc = _rope_coeff(c, a.dtype, a.device, head_dim)
        ss = _rope_coeff(s, a.dtype, a.device, head_dim)
        return tn.npu_rotary_mul(a, cc, ss, "half"), tn.npu_rotary_mul(b, cc, ss, "half")

    def _ascendc(a, b, c, s):
        fn = _custom_op("rope_fwd")
        return None if fn is None else fn(a, b, c, s)

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
    """融合注意力。训练/预填走 CANN ``npu_fusion_attention``。

    约定（与 ``attention._attention`` 一致）：``is_causal=True`` 只用于
    ``seq_len == kv_len`` 的等长因果场景。带偏移的因果掩码（解码续写）在调用方
    就走 SDPA，因此这里遇到 ``seq_len != kv_len`` 的因果请求直接放弃，交由回退。

    CANN 的 ``sparse_mode=2/3`` 在本版本上与显式掩码配合会报错，故统一用
    ``sparse_mode=0`` + 显式上三角 bool 掩码（``True`` 表示屏蔽），数值与
    ``F.scaled_dot_product_attention(is_causal=True)`` 对拍一致。
    """
    if not npu_available():
        return None
    s = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
    head_num = q.shape[1]
    seq_len = q.shape[2]
    kv_len = k.shape[2]
    if is_causal and seq_len != kv_len:
        return None

    def _cann(a, b, c):
        tn = _torch_npu()
        mask = _causal_mask(seq_len, kv_len, a.device) if is_causal else None
        out = tn.npu_fusion_attention(
            a, b, c, head_num,
            input_layout="BNSD",
            atten_mask=mask,
            scale=s,
            keep_prob=1.0 - float(dropout_p),
            sparse_mode=0,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def _ascendc(a, b, c):
        fn = _custom_op("flash_attn_fwd")
        return None if fn is None else fn(a, b, c, is_causal, s)

    return _try(
        [("cann", _cann), ("ascendc", _ascendc)],
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
    """分块 lm_head + CE。CANN 无直接对应算子，走自研 AscendC。

    当前 AscendC 构建只实现了 ``add_rms_norm_fwd`` / ``swiglu_fwd``，
    ``chunked_ce_fwd`` 只有 schema 没有 NPU 实现——若不显式跳过，torch_npu 会
    静默把 hidden/weight/targets 搬回 CPU 算完再搬回来（每 step 两次往返）。
    这里直接返回 None，让派发层用设备上的分块参考实现。
    """
    if not npu_available():
        return None
    fn = _custom_op("chunked_ce_fwd")
    if fn is None:
        return None

    def _ascendc(h, w, t):
        return fn(h, w, t, chunk_size, ignore_index)

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
    """KDA chunkwise。CANN 无对应算子，走自研 AscendC。

    同 :func:`chunked_cross_entropy`：``kda_chunk_fwd`` 尚无 NPU 实现，
    必须显式跳过以免被 VariableFallbackKernel 搬到 CPU 上算。
    """
    if not npu_available():
        return None
    fn = _custom_op("kda_chunk_fwd")
    if fn is None:
        return None

    def _ascendc(a, b, c, g, be):
        return fn(a, b, c, g, be, chunk_size)

    return _try(
        [("ascendc", _ascendc)],
        lambda a, b, c, g, be: _ref.kda_chunkwise(a, b, c, g, be, chunk_size=chunk_size),
        q, k, v, gate, beta,
    )
