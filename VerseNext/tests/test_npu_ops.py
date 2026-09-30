"""昇腾 NPU 算子测试：CANN 原生融合算子 + 自研 AscendC 算子。

这些测试只在真实 NPU 上跑；无 NPU（或未装 torch_npu）时整个模块 skip，
保证 CPU 机器上 ``pytest tests/`` 依然全绿。

为什么要**直接**测 ``torch_npu.npu_*`` / ``torch.ops.verse_nn.*``，而不是只测
``verse_nn.kernels.npu_ops`` 的封装？
因为 ``npu_ops`` 的每个算子都是「CANN 原生 -> 自研 AscendC -> 参考实现」三级
降级，任何一级失败都会**静默**回退。历史上就出过「算子名写错（``torch.npu``
而非 ``torch_npu``）导致全部静默回退到参考实现、训练照跑但一点没加速」的事故。
所以这里分两层验证：

1. ``test_cann_*``：直接调用原生算子，证明这一级**真的能用**；
2. ``test_*_uses_device_path_*``：把参考实现打成会抛异常的桩，若仍能拿到正确
   结果，说明结果确实来自设备算子而非静默回退。
"""

from __future__ import annotations

import pytest
import torch

from verse_nn.kernels import npu_ops
from verse_nn.kernels import reference as R

torch_npu = pytest.importorskip("torch_npu", reason="未安装 torch_npu")

if not torch.npu.is_available():
    pytest.skip("当前环境没有可用的昇腾 NPU", allow_module_level=True)

DEV = "npu:0"

# fp16 在 NPU 上按 fp32 累加，误差主要是 fp16 表示精度（~1e-3 量级）
_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=2e-2, atol=2e-2),
}


def _randn(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(*shape, dtype=dtype) * scale).to(DEV)


# ---------------------------------------------------------------------------
# 1) CANN 原生融合算子：证明这一级不是摆设
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cann_add_rms_norm(dtype):
    x = _randn(2, 8, 128, dtype=dtype)
    residual = _randn(2, 8, 128, dtype=dtype)
    weight = _randn(128, dtype=dtype)

    out = torch_npu.npu_add_rms_norm(x, residual, weight, 1e-6)
    assert isinstance(out, (tuple, list)) and len(out) >= 3, (
        "npu_add_rms_norm 应返回 (y, rstd, residual_out)"
    )
    y, res = out[0], out[-1]
    y_ref, res_ref = R.add_rms_norm(x, residual, weight)
    assert torch.allclose(y.float(), y_ref.float(), **_TOL[dtype])
    assert torch.allclose(res.float(), res_ref.float(), **_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cann_swiglu(dtype):
    gate = _randn(4, 16, 256, dtype=dtype)
    up = _randn(4, 16, 256, dtype=dtype)
    got = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)
    assert torch.allclose(got.float(), R.swiglu(gate, up).float(), **_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cann_rotary_mul(dtype):
    from verse_nn.rope import RotaryEmbedding

    B, H, S, D = 1, 4, 32, 64
    q = _randn(B, H, S, D, dtype=dtype)
    k = _randn(B, H, S, D, dtype=dtype)
    # RotaryEmbedding 返回 CPU 张量，搬到设备后再喂给两侧实现
    cos, sin = RotaryEmbedding(D)(q.cpu(), S)
    cos, sin = cos.to(DEV), sin.to(DEV)

    got_q, got_k = npu_ops.apply_rope(q, k, cos, sin)
    assert got_q is not None and got_k is not None, "设备算子路径不可用"
    q_ref, k_ref = R.apply_rope(q, k, cos, sin)
    assert torch.allclose(got_q.float(), q_ref.float(), **_TOL[dtype])
    assert torch.allclose(got_k.float(), k_ref.float(), **_TOL[dtype])


def test_cann_fusion_attention_causal():
    B, H, S, D = 1, 4, 64, 64
    q, k, v = _randn(B, H, S, D), _randn(B, H, S, D), _randn(B, H, S, D)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=DEV), diagonal=1)
    out = torch_npu.npu_fusion_attention(
        q, k, v, H, input_layout="BNSD", atten_mask=mask,
        scale=D ** -0.5, keep_prob=1.0, sparse_mode=0,
    )
    out = out[0] if isinstance(out, (tuple, list)) else out
    ref = R.flash_attention(q, k, v, is_causal=True)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 2) 派发层：结果必须来自设备算子，而不是静默回退参考实现
# ---------------------------------------------------------------------------

def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("回退到了参考实现——设备算子没有真正生效")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_rms_norm_uses_device_path(dtype, monkeypatch):
    x = _randn(2, 8, 128, dtype=dtype)
    residual = _randn(2, 8, 128, dtype=dtype)
    weight = _randn(128, dtype=dtype)
    # 先算基准，再打桩——否则桩会连测试自己的基准一起打掉。
    y_ref, res_ref = R.add_rms_norm(x, residual, weight)

    monkeypatch.setattr(npu_ops._ref, "add_rms_norm", _fail_if_called)
    got = npu_ops.add_rms_norm(x, residual, weight)
    assert got is not None, "设备算子路径不可用"
    y, res = got
    assert torch.allclose(y.float(), y_ref.float(), **_TOL[dtype])
    assert torch.allclose(res.float(), res_ref.float(), **_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu_uses_device_path(dtype, monkeypatch):
    gate = _randn(2, 8, 128, dtype=dtype)
    up = _randn(2, 8, 128, dtype=dtype)
    expect = R.swiglu(gate, up)

    monkeypatch.setattr(npu_ops._ref, "swiglu", _fail_if_called)
    got = npu_ops.swiglu(gate, up)
    assert got is not None, "设备算子路径不可用"
    assert torch.allclose(got.float(), expect.float(), **_TOL[dtype])


def test_flash_attention_uses_device_path(monkeypatch):
    B, H, S, D = 1, 4, 64, 64
    q, k, v = _randn(B, H, S, D), _randn(B, H, S, D), _randn(B, H, S, D)
    ref = R.flash_attention(q, k, v, is_causal=True)

    monkeypatch.setattr(npu_ops._ref, "flash_attention", _fail_if_called)
    got = npu_ops.flash_attention(q, k, v, is_causal=True)
    assert got is not None, "设备算子路径不可用"
    assert torch.allclose(got.float(), ref.float(), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 3) 自研 AscendC 算子（torch.ops.verse_nn.*）：未编译则 skip
# ---------------------------------------------------------------------------

def _custom_ops():
    from verse_nn.kernels._ext import get_ops, extension_available

    if not extension_available("npu"):
        pytest.skip("自研 AscendC 扩展未编译（kernels/_build/npu/verse_nn_kernels.so）")
    ns = get_ops("npu")
    if ns is None or not hasattr(ns, "add_rms_norm_fwd"):
        pytest.skip("自研 AscendC 扩展未注册 add_rms_norm_fwd")
    return ns


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(2, 8, 128), (4, 16, 256), (1, 3, 64)])
def test_ascendc_add_rms_norm_fwd(dtype, shape):
    ns = _custom_ops()
    x = _randn(*shape, dtype=dtype)
    residual = _randn(*shape, dtype=dtype)
    weight = _randn(shape[-1], dtype=dtype)
    y, res = ns.add_rms_norm_fwd(x, residual, weight, 1e-6)
    y_ref, res_ref = R.add_rms_norm(x, residual, weight)
    assert torch.allclose(y.float(), y_ref.float(), **_TOL[dtype])
    assert torch.allclose(res.float(), res_ref.float(), **_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(2, 8, 128), (5, 7, 33), (1, 1, 100)])
def test_ascendc_swiglu_fwd(dtype, shape):
    """(5, 7, 33) 这类非 32B 对齐的尾块用于覆盖 DataCopyPad 的边界处理。"""
    ns = _custom_ops()
    gate = _randn(*shape, dtype=dtype)
    up = _randn(*shape, dtype=dtype)
    got = ns.swiglu_fwd(gate, up)
    assert torch.allclose(got.float(), R.swiglu(gate, up).float(), **_TOL[dtype])
