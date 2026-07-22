"""Task 4.6: CometSpark 压缩集成测试（Part4K1 Task 8.9 迁移到 spark/model）。

覆盖：
1. ``compress_pipeline`` 新 API 的各种组合（prune / quantize / lora / ternary / distill）
2. ``CometSparkV05LM.compress()`` / ``compression_stats()`` 方法
3. ``CometSparkV05LM.save_pretrained`` / ``from_pretrained`` 往返一致性
4. 工厂函数 ``CometSparkV05Small`` 参数量

Part4K1 Task 8.9: 模型从 data/demo 迁移到 spark/model。
- compress_pipeline 需要 Module 参数，用 model.net（CometSparkNexLM = Module 子类）
- model.compress() 返回新的 CometSparkV05LM 实例
- model.forward(x) 替代 model(Tensor(x))（V05LM 不是 Module）

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

# 让 tests/ 目录能 import verse_torch / spark.model
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_nex"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verse_torch import Tensor, nn
from verse_torch.compress import (
    compress_pipeline,
    count_parameters,
    count_nonzero_params,
    compute_compressed_bits,
    QLinear,
    LoRALinear,
)
# Part4K1 Task 8.9: 从 spark/model 导入（替代 data/demo/model）
from spark.model.model import CometSparkV05LM, CometSparkV05Small
from spark.model.config import CometSparkV05Config


SEED = 42


def _setup_seed(seed: int = SEED):
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. compress_pipeline 新 API 各种组合
# ---------------------------------------------------------------------------


def test_compress_prune_only():
    """仅 prune：返回新模型，原模型不变，非零参数量减少。"""
    _setup_seed()
    model = CometSparkV05Small()
    # compress_pipeline 需要 Module 参数，用 model.net
    orig_nonzero = count_nonzero_params(model.net)
    orig_params = count_parameters(model.net)

    # 新 API：仅 prune
    new_model, stats = compress_pipeline(
        model.net, {"prune": {"sparsity": 0.5}}, return_stats=True
    )

    # 原模型不变
    assert count_parameters(model.net) == orig_params
    assert count_nonzero_params(model.net) == orig_nonzero
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
    model = CometSparkV05Small()
    orig_bits = compute_compressed_bits(model.net)
    orig_avg_bits = orig_bits / count_parameters(model.net)

    new_model, stats = compress_pipeline(
        model.net, {"quantize": {"bits": 4}}, return_stats=True
    )

    new_bits = compute_compressed_bits(new_model)
    new_avg_bits = new_bits / count_parameters(model.net)
    # INT4 量化后平均 bit 应显著降低（< 16，远低于 32）
    assert new_avg_bits < orig_avg_bits, (
        f"quantize 后平均 bit 应降低：orig={orig_avg_bits}, new={new_avg_bits}"
    )
    assert stats["bits"] < orig_avg_bits
    assert stats["qtype"] == "int4"


def test_compress_prune_and_quantize():
    """prune + quantize 组合：两步都生效。"""
    _setup_seed()
    model = CometSparkV05Small()
    orig_params = count_parameters(model.net)

    new_model, stats = compress_pipeline(
        model.net,
        {"prune": {"sparsity": 0.5}, "quantize": {"bits": 4}},
        return_stats=True,
    )

    # 步骤应同时包含 prune 和 quantize
    step_names = [s["step"] for s in stats["steps"]]
    assert "prune" in step_names
    assert "quantize" in step_names
    # 压缩比应大于 1
    assert stats["compression_ratio"] > 1.0
    # 新模型应能 forward（new_model 是 Module，可 __call__）
    x = np.random.randint(0, 256, size=(1, 8))
    out = new_model(Tensor(x))
    assert out.shape == (1, 8, 256)


def test_compress_with_lora():
    """包含 lora：新模型包含 LoRALinear。"""
    _setup_seed()
    model = CometSparkV05Small()

    new_model, stats = compress_pipeline(
        model.net,
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
    model = CometSparkV05Small()

    # 1. 空配置：应返回深拷贝模型，无压缩
    new1 = compress_pipeline(model.net, {})
    from verse_nex.cometspark import CometSparkNexLM
    assert isinstance(new1, CometSparkNexLM)
    assert count_parameters(new1) == count_parameters(model.net)

    # 2. 仅 ternary
    new2, stats2 = compress_pipeline(
        model.net, {"ternary": {}}, return_stats=True
    )
    assert stats2["qtype"] == "ternary"
    assert stats2["bits"] < 32.0

    # 3. prune + quantize + lora + ternary（ternary 覆盖 quantize）
    new3, stats3 = compress_pipeline(
        model.net,
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
# 2. CometSparkV05LM.compress / compression_stats
# ---------------------------------------------------------------------------


def test_compression_stats():
    """compression_stats() 返回正确字段。"""
    _setup_seed()
    model = CometSparkV05Small()
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
    """CometSparkV05LM.compress() 返回压缩模型（与原模型不同实例）。"""
    _setup_seed()
    model = CometSparkV05Small()
    orig_params = model.count_parameters()
    orig_sd = model.state_dict()

    compressed = model.compress({"prune": {"sparsity": 0.5}, "quantize": {"bits": 4}})

    # 应是不同实例
    assert compressed is not model
    # 原模型不变
    assert model.count_parameters() == orig_params
    for k in orig_sd:
        assert np.array_equal(orig_sd[k], model.state_dict()[k])
    # 压缩后模型应是 CometSparkV05LM
    assert isinstance(compressed, CometSparkV05LM)


# ---------------------------------------------------------------------------
# 3. save_pretrained / from_pretrained 往返
# ---------------------------------------------------------------------------


def test_cometspark_save_load_pretrained():
    """save_pretrained → from_pretrained 往返一致（权重相同）。"""
    _setup_seed()
    model = CometSparkV05Small()
    sd_before = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        # 目录结构正确
        files = sorted(os.listdir(tmpdir))
        assert "config.yml" in files
        assert "model.pt" in files

        loaded = CometSparkV05LM.from_pretrained(tmpdir)
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
    """CometSparkV05Small() 参数量 ~194K（Part4K1 Task 8.9 迁移后）。"""
    _setup_seed()
    m = CometSparkV05Small()
    params = m.count_parameters()
    # V05Small 有 MoD 层（mod_every=2, n_layer=2 → 1 mod + 1 trisparse）
    assert 100_000 <= params <= 400_000, f"Small 参数量应在 ~194K，实际 {params}"
    # forward 可用（用 forward 而非 __call__，V05LM 不是 Module）
    x = np.random.randint(0, 256, size=(1, 16))
    out = m.forward(x)
    assert out.shape == (1, 16, 256)


# ---------------------------------------------------------------------------
# 5. 新配置字段
# ---------------------------------------------------------------------------


def test_config_new_fields():
    """CometSparkV05Config 新字段默认值正确。"""
    cfg = CometSparkV05Config()
    assert cfg.rope_theta == 10000.0
    assert cfg.max_position_embeddings == 4096
    assert cfg.embedding_scale is True
    assert cfg.temperature_scaling == 1.0
    assert cfg.init_std == 0.02
    assert cfg.tie_weights is True
    assert cfg.arch == "versenex"


def test_config_pretrained_roundtrip():
    """CometSparkV05Config.from_pretrained / save_pretrained 往返一致。"""
    cfg = CometSparkV05Config(
        arch="versenex",
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        seq_len=64,
        n_kv_head=2,
        tie_weights=True,
        rope_theta=500.0,
        max_position_embeddings=512,
        embedding_scale=False,
        temperature_scaling=0.8,
        init_std=0.01,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_pretrained(tmpdir)
        loaded = CometSparkV05Config.from_pretrained(tmpdir)
    assert loaded.rope_theta == 500.0
    assert loaded.max_position_embeddings == 512
    assert loaded.embedding_scale is False
    assert loaded.temperature_scaling == 0.8
    assert loaded.init_std == 0.01
    assert loaded.arch == "versenex"
    assert loaded.n_layer == 2


def test_model_with_advanced_config():
    """模型应用新配置（rope_theta / embedding_scale）后仍能 forward。"""
    _setup_seed()
    cfg = CometSparkV05Config(
        arch="versenex",
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        seq_len=64,
        n_kv_head=2,
        tie_weights=True,
        rope_theta=1000.0,
        max_position_embeddings=256,
        embedding_scale=True,
        temperature_scaling=1.0,
        init_std=0.02,
    )
    model = CometSparkV05LM(cfg)
    x = np.random.randint(0, 256, size=(1, 16))
    out = model.forward(x)
    assert out.shape == (1, 16, 256)


# ---------------------------------------------------------------------------
# 脚本入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
