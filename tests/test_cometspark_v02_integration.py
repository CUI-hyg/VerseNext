"""测试 CometSparkLM 集成 arch='verse_nex'（Part4 端到端）。

验证：
- CometSparkLM 支持 arch='verse_nex' 构造
- forward / forward_recurrent / generate 接口与 transformer/hybrid 一致
- CometSparkV02Small / CometSparkV02 工厂函数
- save / load 单文件 + save_pretrained / from_pretrained 目录模式
- config.yml 持久化 roundtrip（含 layer_pattern 等 Part4 字段）
- 与 verse_tokenizer 集成（VerseTokenizer）
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_tokenizer"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
# 添加 data/demo 让 model 包可导入
sys.path.insert(0, os.path.join(_WORKSPACE, "data", "demo"))

import pytest

from verse_torch import Tensor, no_grad
from model.model import (
    CometSparkLM, CometSparkSmall,
    CometSparkV02Small, CometSparkV02,
)
from model.config import CometSparkConfig


# ---------------------------------------------------------------------------
# 构造与 arch 校验
# ---------------------------------------------------------------------------


def test_construct_verse_nex_arch():
    """CometSparkLM 支持 arch='verse_nex'。"""
    config = CometSparkConfig(
        arch="verse_nex",
        vocab_size=64,
        n_layer=3,
        n_head=4,
        n_embd=32,
        n_kv_head=2,
        tie_weights=True,
        num_dense_parts=2,
        num_experts_per_part=2,
        top_k=1,
        window_size=8,
        num_global_tokens=4,
        max_position_embeddings=64,
        seq_len=32,
    )
    model = CometSparkLM(config)
    assert model.config.arch == "verse_nex"
    # 内部 net 应是 CometSparkNexLM
    from verse_nex.cometspark import CometSparkNexLM
    assert isinstance(model.net, CometSparkNexLM)


def test_invalid_arch_rejected():
    """非法 arch 应抛 ValueError。"""
    config = CometSparkConfig(arch="invalid")
    with pytest.raises(ValueError):
        CometSparkLM(config)


# ---------------------------------------------------------------------------
# forward / generate 接口
# ---------------------------------------------------------------------------


def test_forward_shape():
    """forward 输出 (B, T, vocab)。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    model.eval()
    idx = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with no_grad():
        logits = model(Tensor(idx))
    assert logits.data.shape == (1, 4, 64)


def test_forward_recurrent_shape():
    """forward_recurrent 输出 (B, 1, vocab)。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    model.eval()
    idx = np.array([[5]], dtype=np.int64)
    with no_grad():
        logits, states = model.forward_recurrent(Tensor(idx), states=None)
    assert logits.data.shape == (1, 1, 64)
    assert isinstance(states, list)


def test_generate_greedy():
    """greedy 生成走 CometSparkNexLM.generate 路径。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    model.eval()
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=4, temperature=1.0)
    assert out.shape == (1, 7)
    assert out.dtype == np.int64


def test_generate_sampling():
    """采样生成走 CometSparkLM._generate_with_logits。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    model.eval()
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=3, temperature=0.8, top_k=5)
    assert out.shape == (1, 6)
    assert out.dtype == np.int64


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def test_v02_small_factory():
    """CometSparkV02Small 工厂：~0.5M 参数。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    n = model.count_parameters()
    assert n > 0
    # 应该比较小（< 1M）
    assert n < 1_000_000


def test_v02_factory_param_budget():
    """CometSparkV02 工厂：参数量 ≈ 0.5B（用小 vocab 测试 backbone）。"""
    model = CometSparkV02(
        vocab_size=1024,  # 小 vocab 节省测试内存
        dim=384,
        n_layer=32,
        n_head=8,
        n_kv_head=4,
    )
    n_params = model.count_parameters()
    # 去掉 Embedding 后的 backbone + 真实 V0.2 Embedding
    backbone = n_params - 1024 * 384
    v02_total = backbone + 151936 * 384
    assert 4e8 < v02_total < 6e8, (
        f"V0.2 参数量预算不符: {v02_total / 1e8:.2f}B (预期 0.4B - 0.6B)"
    )


# ---------------------------------------------------------------------------
# save / load / save_pretrained / from_pretrained
# ---------------------------------------------------------------------------


def test_save_load_single_file():
    """save → load 单文件 roundtrip。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    model.eval()
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    with no_grad():
        out_before = model(Tensor(idx)).data.copy()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        # 新模型 load（同 config）
        config = model.config
        model2 = CometSparkLM(config)
        model2.load(path)
        model2.eval()
        with no_grad():
            out_after = model2(Tensor(idx)).data
    assert np.allclose(out_before, out_after, atol=1e-5)


def test_save_pretrained_from_pretrained_dir():
    """save_pretrained → from_pretrained 目录模式 roundtrip。"""
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    model.eval()
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    with no_grad():
        out_before = model(Tensor(idx)).data.copy()

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        assert os.path.exists(os.path.join(d, "config.yml"))
        assert os.path.exists(os.path.join(d, "model.pt"))
        model2 = CometSparkLM.from_pretrained(d)
        model2.eval()
        with no_grad():
            out_after = model2(Tensor(idx)).data
    assert np.allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------------
# config.yml 持久化
# ---------------------------------------------------------------------------


def test_config_yaml_roundtrip():
    """CometSparkConfig.to_yaml → from_yaml roundtrip 保留 Part4 字段。"""
    config = CometSparkConfig(
        arch="verse_nex",
        vocab_size=64,
        n_layer=3,
        n_head=4,
        n_embd=32,
        n_kv_head=2,
        tie_weights=True,
        num_dense_parts=2,
        num_experts_per_part=2,
        top_k=1,
        window_size=8,
        num_global_tokens=4,
        max_position_embeddings=64,
        seq_len=32,
        use_alibi=True,
        use_rope=False,
        aux_loss_weight=0.01,
    )
    with tempfile.TemporaryDirectory() as d:
        cfg_path = os.path.join(d, "config.yml")
        config.to_yaml(cfg_path)
        loaded = CometSparkConfig.from_yaml(cfg_path)
    assert loaded.arch == "verse_nex"
    assert loaded.vocab_size == 64
    assert loaded.n_layer == 3
    assert loaded.n_head == 4
    assert loaded.n_embd == 32
    assert loaded.n_kv_head == 2
    assert loaded.tie_weights is True
    assert loaded.num_dense_parts == 2
    assert loaded.num_experts_per_part == 2
    assert loaded.top_k == 1
    assert loaded.window_size == 8
    assert loaded.num_global_tokens == 4
    assert loaded.use_alibi is True
    assert loaded.use_rope is False
    assert loaded.aux_loss_weight == 0.01


# ---------------------------------------------------------------------------
# 兼容性：transformer/hybrid arch 不受影响
# ---------------------------------------------------------------------------


def test_transformer_arch_still_works():
    """确保 transformer arch 未被破坏。"""
    model = CometSparkSmall()
    model.eval()
    idx = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with no_grad():
        logits = model(Tensor(idx))
    assert logits.data.shape == (1, 4, 256)


def test_existing_yaml_config_loads():
    """现有 config.yml（不含 Part4 字段）应正常加载。"""
    # 模拟一个老配置（只有 model 段基础字段）
    yaml_text = """model:
  arch: transformer
  vocab_size: 256
  n_layer: 2
  n_head: 4
  n_embd: 64
  seq_len: 64
  n_kv_head: 2
  tie_weights: true
"""
    with tempfile.TemporaryDirectory() as d:
        cfg_path = os.path.join(d, "config.yml")
        with open(cfg_path, "w") as f:
            f.write(yaml_text)
        config = CometSparkConfig.from_yaml(cfg_path)
    assert config.arch == "transformer"
    assert config.vocab_size == 256
    # Part4 字段应使用默认值
    assert config.num_dense_parts == 5
    assert config.num_experts_per_part == 8
    assert config.top_k == 3
    assert config.window_size == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
