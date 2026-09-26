"""算子层测试：参考实现正确性、CPU 优化实现对齐、梯度、派发回退。

覆盖三层：
1. :mod:`verse_nn.kernels.reference` —— 数学基准（且必须与仓库原实现一致）；
2. :mod:`verse_nn.kernels.cpu_ops` —— CPU 优化实现必须与基准数值对齐；
3. :mod:`verse_nn.kernels._dispatch` —— 策略开关与降级行为。
"""

from __future__ import annotations

import pytest
import torch

from verse_nn import kernels
from verse_nn.kernels import reference as R
from verse_nn.kernels import cpu_ops
from verse_nn.kernels._dispatch import KernelPolicy, get_policy, set_policy
from verse_nn.rope import RotaryEmbedding
from verse_nn.rope import apply_rope as repo_rope

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 与仓库原实现的一致性（最重要：融合不能改变语义）
# ---------------------------------------------------------------------------

def test_reference_rope_matches_repo_exactly():
    B, H, S, D = 2, 4, 16, 32
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    cos, sin = RotaryEmbedding(D)(q, S)          # 仓库真实形状 (S, D)
    rq, rk = repo_rope(q, k, cos, sin)
    mq, mk = R.apply_rope(q, k, cos, sin)
    assert torch.equal(mq, rq) and torch.equal(mk, rk)


def test_reference_rope_accepts_compact_cos_sin():
    B, H, S, D = 1, 2, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    cos, sin = RotaryEmbedding(D)(q, S)
    rq, rk = repo_rope(q, k, cos, sin)
    mq, mk = R.apply_rope(q, k, cos[:, : D // 2], sin[:, : D // 2])
    assert torch.allclose(mq, rq, atol=1e-6) and torch.allclose(mk, rk, atol=1e-6)


def test_rms_norm_matches_blocks_module():
    from verse_nn.blocks import RMSNorm

    torch.manual_seed(1)
    x = torch.randn(3, 5, 32)
    module = RMSNorm(32)
    assert torch.allclose(R.rms_norm(x, module.weight), module(x), atol=1e-6)


def test_chunked_ce_matches_plain_cross_entropy():
    torch.manual_seed(2)
    B, S, D, V = 2, 9, 16, 40
    hidden = torch.randn(B, S, D)
    weight = torch.randn(V, D)
    targets = torch.randint(0, V, (B, S))
    targets[0, 3] = -100                       # 含 ignore 位置
    logits = (hidden @ weight.t()).float()
    expect = torch.nn.functional.cross_entropy(
        logits.view(-1, V), targets.reshape(-1), ignore_index=-100
    )
    for chunk in (0, 2, 4, 9):
        got = R.chunked_cross_entropy(hidden, weight, targets, chunk_size=chunk)
        assert torch.allclose(got, expect, atol=1e-5), f"chunk={chunk}"


# ---------------------------------------------------------------------------
# CPU 优化实现 vs 参考实现
# ---------------------------------------------------------------------------

def test_cpu_add_rms_norm_matches_reference():
    torch.manual_seed(3)
    x = torch.randn(4, 6, 32)
    residual = torch.randn(4, 6, 32)
    weight = torch.randn(32)
    for dtype in (torch.float32, torch.bfloat16):
        a = cpu_ops.add_rms_norm(x.to(dtype), residual.to(dtype), weight.to(dtype))
        b = R.add_rms_norm(x.to(dtype), residual.to(dtype), weight.to(dtype))
        tol = 1e-6 if dtype is torch.float32 else 2e-2
        assert torch.allclose(a[0].float(), b[0].float(), atol=tol)
        assert torch.allclose(a[1].float(), b[1].float(), atol=tol)


def test_cpu_swiglu_matches_reference():
    torch.manual_seed(4)
    gate = torch.randn(2, 8, 64)
    up = torch.randn(2, 8, 64)
    assert torch.allclose(cpu_ops.swiglu(gate, up), R.swiglu(gate, up), atol=1e-6)


def test_cpu_flash_attention_matches_reference():
    torch.manual_seed(5)
    q = torch.randn(2, 3, 8, 16)
    k = torch.randn(2, 3, 8, 16)
    v = torch.randn(2, 3, 8, 16)
    got = cpu_ops.flash_attention(q, k, v, is_causal=True)
    expect = R.flash_attention(q, k, v, is_causal=True)
    assert torch.allclose(got, expect, atol=1e-6)
    # 自定义 scale 路径
    got2 = cpu_ops.flash_attention(q, k, v, is_causal=True, scale=0.5)
    expect2 = R.flash_attention(q, k, v, is_causal=True, scale=0.5)
    assert torch.allclose(got2, expect2, atol=1e-6)


def test_cpu_chunked_ce_matches_reference():
    torch.manual_seed(6)
    hidden = torch.randn(2, 7, 16)
    weight = torch.randn(32, 16)
    targets = torch.randint(0, 32, (2, 7))
    for chunk in (0, 3, 7):
        got = cpu_ops.chunked_cross_entropy(hidden, weight, targets, chunk_size=chunk)
        expect = R.chunked_cross_entropy(hidden, weight, targets, chunk_size=chunk)
        assert torch.allclose(got, expect, atol=1e-6), f"chunk={chunk}"


@pytest.mark.parametrize(
    "shape,chunk",
    [((2, 4, 16, 32), 8), ((1, 2, 64, 16), 16), ((3, 2, 50, 8), 8)],
)
def test_cpu_kda_matches_reference_and_recurrence(shape, chunk):
    B, H, S, D = shape
    torch.manual_seed(7)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    gate = torch.sigmoid(torch.randn(B, H, S, D))
    beta = torch.sigmoid(torch.randn(B, H, S, 1))

    o1, s1 = cpu_ops.kda_chunkwise(q, k, v, gate, beta, chunk_size=chunk)
    o2, s2 = R.kda_chunkwise(q, k, v, gate, beta, chunk_size=chunk)
    assert torch.allclose(o1, o2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)

    # chunkwise 是递归定义的等价改写，误差应在数值范围内
    o_ref, s_ref = R.kda_recurrent(q, k, v, gate, beta)
    assert torch.allclose(o2, o_ref, atol=1e-2, rtol=1e-2)


def test_kda_matches_attention_module():
    """kernels.reference.kda_chunkwise 必须与 attention.KDAAttention 内部实现一致。"""
    from verse_nn.attention import KDAAttention

    torch.manual_seed(8)
    B, H, S, D, C = 2, 2, 32, 16, 8
    att = KDAAttention(d_model=H * D, num_heads=H, chunk_size=C)
    att.eval()
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    gate = torch.sigmoid(torch.randn(B, H, S, D))
    beta = torch.sigmoid(torch.randn(B, H, S, 1))
    state = torch.zeros(B, H, D, D)
    with torch.no_grad():
        expect, _ = att._chunkwise_forward(q, k, v, gate, beta, state)
    got, _ = R.kda_chunkwise(q, k, v, gate, beta, chunk_size=C)
    assert torch.allclose(got, expect, atol=1e-5)


# ---------------------------------------------------------------------------
# 梯度
# ---------------------------------------------------------------------------

def test_add_rms_norm_gradients_flow():
    x = torch.randn(2, 4, 16, requires_grad=True)
    residual = torch.randn(2, 4, 16, requires_grad=True)
    weight = torch.randn(16, requires_grad=True)
    out, res_out = kernels.add_rms_norm(x, residual, weight)
    (out.sum() + res_out.sum()).backward()
    for t in (x, residual, weight):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_swiglu_and_ce_gradients_flow():
    gate = torch.randn(2, 4, 8, requires_grad=True)
    up = torch.randn(2, 4, 8, requires_grad=True)
    kernels.swiglu(gate, up).sum().backward()
    assert gate.grad is not None and up.grad is not None

    hidden = torch.randn(2, 5, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    targets = torch.randint(0, 16, (2, 5))
    kernels.chunked_cross_entropy(hidden, weight, targets, chunk_size=2).backward()
    assert hidden.grad is not None and weight.grad is not None


def test_flash_attention_gradients_flow():
    q = torch.randn(1, 2, 6, 8, requires_grad=True)
    k = torch.randn(1, 2, 6, 8, requires_grad=True)
    v = torch.randn(1, 2, 6, 8, requires_grad=True)
    kernels.flash_attention(q, k, v, is_causal=True).sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_kda_gradients_flow():
    B, H, S, D = 1, 2, 8, 8
    q = torch.randn(B, H, S, D, requires_grad=True)
    k = torch.randn(B, H, S, D, requires_grad=True)
    v = torch.randn(B, H, S, D, requires_grad=True)
    gate = torch.sigmoid(torch.randn(B, H, S, D)).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, H, S, 1)).detach().requires_grad_(True)
    out, state = kernels.kda_chunkwise(q, k, v, gate, beta, chunk_size=4)
    (out.sum() + state.sum()).backward()
    for t in (q, k, v, gate, beta):
        assert t.grad is not None


# ---------------------------------------------------------------------------
# 派发与策略
# ---------------------------------------------------------------------------

def test_policy_disable_falls_back_to_reference():
    old = get_policy()
    try:
        set_policy(enable_fused=False)
        x = torch.randn(2, 3, 8)
        residual = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        got = kernels.add_rms_norm(x, residual, weight)
        expect = R.add_rms_norm(x, residual, weight)
        assert torch.equal(got[0], expect[0]) and torch.equal(got[1], expect[1])
    finally:
        set_policy(**old.__dict__)


def test_policy_ce_chunk_size_applied():
    old = get_policy()
    try:
        set_policy(ce_chunk_size=3)
        assert get_policy().ce_chunk_size == 3
    finally:
        set_policy(**old.__dict__)


def test_backend_of_maps_device_types():
    from verse_nn.kernels._dispatch import backend_of

    assert backend_of("cpu") == "cpu"
    assert backend_of("npu") == "npu"
    assert backend_of("mps") == "mps"
    # cuda 视 torch 构建而定（CUDA -> "cuda"，ROCm -> "rocm"）
    assert backend_of("cuda") in ("cuda", "rocm")


def test_available_ops_reports_structure():
    info = kernels.available_ops()
    assert set(info) == {"policy", "custom_ext", "impl"}
    assert info["policy"]["fused"] is True
    assert "cpu" in info["custom_ext"]


def test_dispatch_on_cpu_uses_cpu_ops():
    """CPU 上融合开启时，结果应与 cpu_ops 完全一致（证明派发到了 CPU 实现）。"""
    torch.manual_seed(9)
    x = torch.randn(2, 4, 8)
    residual = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    got = kernels.add_rms_norm(x, residual, weight)
    expect = cpu_ops.add_rms_norm(x, residual, weight)
    assert torch.equal(got[0], expect[0])


def test_unknown_device_type_falls_back_to_cpu_ops():
    from verse_nn.kernels._dispatch import backend_of

    assert backend_of("meta") == "cpu"
    assert backend_of("weird") == "cpu"


def test_kernel_policy_is_frozen():
    p = KernelPolicy()
    with pytest.raises(Exception):
        p.enable_fused = False       # frozen dataclass
