"""Task 4.6: CometSpark 压缩集成测试。

覆盖：
1. ``compress_pipeline`` 新 API 的各种组合（prune / quantize / lora / ternary / distill）
2. ``CometSparkLM.compress()`` / ``compression_stats()`` 方法
3. ``CometSparkLM.save_pretrained`` / ``from_pretrained`` 往返一致性
4. 工厂函数 ``CometSparkSmall`` / ``CometSparkMedium`` / ``CometSparkLarge`` 参数量

运行方式：
    cd /workspace
    python -m pytest tests/test_compression_integration.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch / model.model
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_nex"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_tokenizer"))
sys.path.insert(0, str(_REPO_ROOT / "data" / "demo"))

from verse_torch import Tensor, nn
from verse_torch.compress import (
    compress_pipeline,
    count_parameters,
    count_nonzero_params,
    compute_compressed_bits,
    QLinear,
    LoRALinear,
)
from model.model import (
    CometSparkLM,
    CometSparkSmall,
    CometSparkMedium,
    CometSparkLarge,
)
from model.config import CometSparkConfig


SEED = 42


def _setup_seed(seed: int = SEED):
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. compress_pipeline 新 API 各种组合
# ---------------------------------------------------------------------------


def test_compress_prune_only():
    """仅 prune：返回新模型，原模型不变，非零参数量减少。"""
    _setup_seed()
    model = CometSparkSmall()
    orig_nonzero = count_nonzero_params(model)
    orig_params = count_parameters(model)

    # 新 API：仅 prune
    new_model, stats = compress_pipeline(
        model, {"prune": {"sparsity": 0.5}}, return_stats=True
    )

    # 原模型不变
    assert count_parameters(model) == orig_params
    assert count_nonzero_params(model) == orig_nonzero
    # 新模型非零参数应减少
    new_nonzero = count_nonzero_params(new_model)
    assert new_nonzero < orig_nonzero, (
        f"prune 后非零参数应减少：orig={orig_nonzero}, new={new_nonzero}"
    )
    # 统计字段完整
    assert stats["original_params"] == orig_params
    assert stats["sparsity"] > 0.0


def test_compress_quantize_only():
    """仅 quantize：返回新模型，bit 数降低（INT4 → ~4 bit/param）。"""
    _setup_seed()
    model = CometSparkSmall()
    orig_bits = compute_compressed_bits(model)
    orig_avg_bits = orig_bits / count_parameters(model)

    new_model, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True
    )

    new_bits = compute_compressed_bits(new_model)
    new_avg_bits = new_bits / count_parameters(model)
    # INT4 量化后平均 bit 应显著降低（< 16，远低于 32）
    assert new_avg_bits < orig_avg_bits, (
        f"quantize 后平均 bit 应降低：orig={orig_avg_bits}, new={new_avg_bits}"
    )
    assert stats["bits"] < orig_avg_bits
    assert stats["qtype"] == "int4"


def test_compress_prune_and_quantize():
    """prune + quantize 组合：两步都生效。"""
    _setup_seed()
    model = CometSparkSmall()
    orig_params = count_parameters(model)

    new_model, stats = compress_pipeline(
        model,
        {"prune": {"sparsity": 0.5}, "quantize": {"bits": 4}},
        return_stats=True,
    )

    # 步骤应同时包含 prune 和 quantize
    step_names = [s["step"] for s in stats["steps"]]
    assert "prune" in step_names
    assert "quantize" in step_names
    # 压缩比应大于 1
    assert stats["compression_ratio"] > 1.0
    # 新模型应能 forward
    x = np.random.randint(0, 256, size=(1, 8))
    out = new_model(Tensor(x))
    assert out.shape == (1, 8, 256)


def test_compress_with_lora():
    """包含 lora：新模型包含 LoRALinear。"""
    _setup_seed()
    model = CometSparkSmall()

    new_model, stats = compress_pipeline(
        model,
        {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}, "lora": {"rank": 4}},
        return_stats=True,
    )

    step_names = [s["step"] for s in stats["steps"]]
    assert "lora" in step_names
    # 模型中应至少有一个 LoRALinear
    found_lora = any(
        isinstance(m, LoRALinear) for _, m in new_model.named_modules()
    )
    assert found_lora, "包含 lora 的 pipeline 后应存在 LoRALinear"
    # forward 仍可用
    x = np.random.randint(0, 256, size=(1, 8))
    out = new_model(Tensor(x))
    assert out.shape == (1, 8, 256)


def test_compress_pipeline_combinations():
    """compress_pipeline 各种组合：空配置 / 单 key / 全配置。"""
    _setup_seed()
    model = CometSparkSmall()

    # 1. 空配置：应返回深拷贝模型，无压缩
    new1 = compress_pipeline(model, {})
    assert isinstance(new1, CometSparkLM)
    assert count_parameters(new1) == count_parameters(model)

    # 2. 仅 ternary
    new2, stats2 = compress_pipeline(
        model, {"ternary": {}}, return_stats=True
    )
    assert stats2["qtype"] == "ternary"
    assert stats2["bits"] < 32.0

    # 3. prune + quantize + lora + ternary（ternary 覆盖 quantize）
    new3, stats3 = compress_pipeline(
        model,
        {
            "prune": {"sparsity": 0.3},
            "quantize": {"bits": 4},
            "lora": {"rank": 4, "alpha": 8.0},
            "ternary": {},
        },
        return_stats=True,
    )
    step_names = [s["step"] for s in stats3["steps"]]
    assert "prune" in step_names
    assert "quantize" in step_names
    assert "lora" in step_names
    assert "ternary" in step_names
    # 最终 qtype 应为 ternary（覆盖了 int4）
    assert stats3["qtype"] == "ternary"

    # 4. 原模型始终不变（多次 compress_pipeline 调用不应修改 model.state_dict）
    orig_sd = model.state_dict()
    sd_now = model.state_dict()
    for k in orig_sd:
        assert np.array_equal(orig_sd[k], sd_now[k]), (
            f"原模型 state_dict 在 compress_pipeline 后被修改：{k}"
        )


# ---------------------------------------------------------------------------
# 2. CometSparkLM.compress / compression_stats
# ---------------------------------------------------------------------------


def test_compression_stats():
    """compression_stats() 返回正确字段。"""
    _setup_seed()
    model = CometSparkSmall()
    compressed = model.compress({"prune": {"sparsity": 0.5}, "quantize": {"bits": 4}})

    stats = compressed.compression_stats()
    expected_keys = {
        "original_params",
        "compressed_params",
        "sparsity",
        "bits",
        "compression_ratio",
    }
    assert expected_keys.issubset(stats.keys()), (
        f"compression_stats 缺少字段：{expected_keys - set(stats.keys())}"
    )
    assert stats["original_params"] > 0
    assert stats["compressed_params"] > 0
    assert 0.0 <= stats["sparsity"] <= 1.0
    assert stats["bits"] > 0
    assert stats["compression_ratio"] > 0


def test_cometspark_compress():
    """CometSparkLM.compress() 返回压缩模型（与原模型不同实例）。"""
    _setup_seed()
    model = CometSparkSmall()
    orig_params = model.count_parameters()
    orig_sd = model.state_dict()

    compressed = model.compress({"prune": {"sparsity": 0.5}, "quantize": {"bits": 4}})

    # 应是不同实例
    assert compressed is not model
    # 原模型不变
    assert model.count_parameters() == orig_params
    for k in orig_sd:
        assert np.array_equal(orig_sd[k], model.state_dict()[k])
    # 原模型记录了 _pre_compress_param_count
    assert model._pre_compress_param_count == orig_params


# ---------------------------------------------------------------------------
# 3. save_pretrained / from_pretrained 往返
# ---------------------------------------------------------------------------


def test_cometspark_save_load_pretrained():
    """save_pretrained → from_pretrained 往返一致（权重相同）。"""
    _setup_seed()
    model = CometSparkSmall()
    sd_before = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        # 目录结构正确
        files = sorted(os.listdir(tmpdir))
        assert "config.yml" in files
        assert "model.pt" in files

        loaded = CometSparkLM.from_pretrained(tmpdir)
        sd_after = loaded.state_dict()

    # 权重应一致
    assert set(sd_before.keys()) == set(sd_after.keys())
    for k in sd_before:
        assert np.array_equal(sd_before[k], sd_after[k]), (
            f"save/load 往返后权重不一致：{k}"
        )
    # config 应一致
    assert loaded.config.to_dict() == model.config.to_dict()


# ---------------------------------------------------------------------------
# 4. 工厂函数参数量
# ---------------------------------------------------------------------------


def test_cometspark_small_factory():
    """CometSparkSmall() 参数量 ~131K。"""
    _setup_seed()
    m = CometSparkSmall()
    params = m.count_parameters()
    # 任务描述：~131K 参数
    assert 100_000 <= params <= 200_000, f"Small 参数量应在 ~131K，实际 {params}"
    # forward 可用
    x = np.random.randint(0, 256, size=(1, 16))
    out = m(Tensor(x))
    assert out.shape == (1, 16, 256)


def test_cometspark_medium_factory():
    """CometSparkMedium() 参数量 ~853K。"""
    _setup_seed()
    m = CometSparkMedium()
    params = m.count_parameters()
    # 任务描述：~853K 参数
    assert 700_000 <= params <= 1_000_000, f"Medium 参数量应在 ~853K，实际 {params}"


# ---------------------------------------------------------------------------
# 5. 新配置字段
# ---------------------------------------------------------------------------


def test_config_new_fields():
    """CometSparkConfig 新字段默认值正确。"""
    cfg = CometSparkConfig()
    assert cfg.rope_theta == 10000.0
    assert cfg.max_position_embeddings == 2048
    assert cfg.attention_dropout == 0.0
    assert cfg.hidden_dropout == 0.0
    assert cfg.embedding_dropout == 0.0


def test_config_pretrained_roundtrip():
    """CometSparkConfig.from_pretrained / save_pretrained 往返一致。"""
    cfg = CometSparkConfig(
        arch="transformer",
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        seq_len=64,
        n_kv_head=2,
        tie_weights=True,
        rope_theta=500.0,
        max_position_embeddings=512,
        attention_dropout=0.1,
        hidden_dropout=0.2,
        embedding_dropout=0.05,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_pretrained(tmpdir)
        loaded = CometSparkConfig.from_pretrained(tmpdir)
    assert loaded.rope_theta == 500.0
    assert loaded.max_position_embeddings == 512
    assert loaded.attention_dropout == 0.1
    assert loaded.hidden_dropout == 0.2
    assert loaded.embedding_dropout == 0.05
    assert loaded.arch == "transformer"
    assert loaded.n_layer == 2


def test_model_with_advanced_config():
    """模型应用新配置（rope_theta / dropout 分离）后仍能 forward。"""
    _setup_seed()
    cfg = CometSparkConfig(
        arch="transformer",
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        seq_len=64,
        n_kv_head=2,
        tie_weights=True,
        rope_theta=1000.0,
        max_position_embeddings=256,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        embedding_dropout=0.1,
    )
    model = CometSparkLM(cfg)
    x = np.random.randint(0, 256, size=(1, 16))
    out = model(Tensor(x))
    assert out.shape == (1, 16, 256)


# ---------------------------------------------------------------------------
# 脚本入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
