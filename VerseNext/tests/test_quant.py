"""CPU int8 量化测试：数值正确性、层替换范围、回退行为。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from verse_nn.kernels import quant


def test_int8_gemm_available_returns_bool():
    assert isinstance(quant.int8_gemm_available(), bool)


def test_int8_linear_close_to_fp32():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False)
    x = torch.randn(16, 64)
    expect = linear(x)
    q = quant.Int8Linear.from_float(linear)
    got = q(x)
    assert got.shape == expect.shape
    # int8 量化误差应在合理范围内（相对误差 < 10%）
    rel = (got - expect).norm() / expect.norm()
    assert rel < 0.1, f"相对误差过大: {rel}"


def test_int8_linear_preserves_bias():
    torch.manual_seed(1)
    linear = nn.Linear(32, 16, bias=True)
    q = quant.Int8Linear.from_float(linear)
    assert q.bias is not None
    x = torch.randn(4, 32)
    assert torch.allclose(q(x), linear(x), atol=0.2)


def test_int8_linear_per_channel_vs_global():
    torch.manual_seed(2)
    linear = nn.Linear(64, 32, bias=False)
    x = torch.randn(8, 64)
    expect = linear(x)
    per_ch = quant.Int8Linear.from_float(linear, per_channel=True)(x)
    glob = quant.Int8Linear.from_float(linear, per_channel=False)(x)
    # per-channel 通常误差更小
    err_per = (per_ch - expect).norm()
    err_glob = (glob - expect).norm()
    assert err_per <= err_glob * 1.5


def test_int8_linear_dequantize_shapes():
    linear = nn.Linear(16, 8, bias=False)
    q = quant.Int8Linear.from_float(linear)
    assert q.dequantize_weight().shape == (8, 16)
    assert q.weight_q.dtype == torch.int8


def test_int8_linear_handles_small_batch():
    """M < 8 时不满足 _int_mm 约束，必须走 fp32 回退且不报错。"""
    linear = nn.Linear(32, 16, bias=False)
    q = quant.Int8Linear.from_float(linear)
    out = q(torch.randn(1, 32))
    assert out.shape == (1, 16) and torch.isfinite(out).all()


def test_quantize_model_int8_replaces_and_skips():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(32, 16)
            self.fc = nn.Linear(16, 16, bias=False)
            self.lm_head = nn.Linear(16, 32, bias=False)

        def forward(self, ids):
            return self.lm_head(self.fc(self.token_emb(ids)))

    model = Tiny()
    quant.quantize_model_int8(model)
    assert isinstance(model.fc, quant.Int8Linear)
    # 默认跳过 lm_head / embedding
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, quant.Int8Linear)
    out = model(torch.randint(0, 32, (2, 5)))
    assert out.shape == (2, 5, 32)


def test_quantize_dynamic_int8_runs():
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    q = quant.quantize_dynamic_int8(model)
    out = q(torch.randn(4, 16))
    assert out.shape == (4, 8)


def test_quantize_linear_helper():
    linear = nn.Linear(8, 4, bias=False)
    assert isinstance(quant.quantize_linear(linear), quant.Int8Linear)


def test_inference_dynamic_quantize_api_still_works():
    """向后兼容：verse_trainer.inference.dynamic_quantize 签名与行为保持。"""
    from verse_trainer.inference import dynamic_quantize

    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    q = dynamic_quantize(model)
    assert q(torch.randn(2, 16)).shape == (2, 8)


def test_inference_per_channel_mode():
    from verse_trainer.inference import dynamic_quantize

    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    q = dynamic_quantize(model, mode="per_channel")
    assert q(torch.randn(2, 16)).shape == (2, 8)


def test_quantize_on_cuda_skipped_in_generator():
    """Generator 在非 CPU 设备上应跳过量化（此处用 CPU 验证不报错）。"""
    from verse_nn.transformer import VerseTransformer, VerseTransformerConfig
    from verse_trainer.inference import GenerateConfig, Generator

    model = VerseTransformer(
        VerseTransformerConfig(vocab_size=64, d_model=32, num_layers=1, num_heads=4)
    )
    cfg = GenerateConfig(quantize=True, device="cpu", max_new_tokens=2)
    gen = Generator(model, None, None, cfg)
    ids = torch.randint(0, 64, (1, 4))
    out = gen.generate_ids(ids)
    assert out.shape[0] == 1 and out.shape[1] >= 4
