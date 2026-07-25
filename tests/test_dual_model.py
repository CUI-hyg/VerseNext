"""Part5K1 Task 9：双模型并行（small / mate）测试。

覆盖 SubTask 9.8 的全部测试用例：
1. Small 模型构建：``CometSparkSmall()`` 返回 ``CometSparkSmallLM``，参数量 < 1M
2. Mate 模型构建：``CometSparkMate()`` 返回 ``CometSparkMateLM``（小尺寸测试配置）
3. VMPC 预设：small 的 ``vmpc_profile == "small"``，mate 的 == "mate"
4. 配置加载：``CometSparkSmallConfig.from_yaml`` / ``CometSparkMateConfig.from_yaml`` 成功
5. 前向传播：small 模型 forward 不报错（小尺寸输入）
6. 导入路径：``from spark.small.model import ...`` / ``from spark.mate.model import ...`` 可用
7. VMPC 压缩：small 模型用 ``vmpc_compress`` 压缩不报错

运行方式：
    cd /workspace && python -m pytest tests/test_dual_model.py -x -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path 注入（与 test_cometspark_v05.py 一致）
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SEED = 42


# ---------------------------------------------------------------------------
# 6. 导入路径（放最前，确保导入可用再做其他测试）
# ---------------------------------------------------------------------------


def test_import_small_model():
    """from spark.small.model import CometSparkSmallLM 可用。"""
    from spark.small.model import CometSparkSmallLM
    assert CometSparkSmallLM is not None
    assert CometSparkSmallLM.__name__ == "CometSparkSmallLM"


def test_import_mate_model():
    """from spark.mate.model import CometSparkMateLM 可用。"""
    from spark.mate.model import CometSparkMateLM
    assert CometSparkMateLM is not None
    assert CometSparkMateLM.__name__ == "CometSparkMateLM"


def test_import_small_factory():
    """from spark.small.model import CometSparkSmall 工厂函数可用。"""
    from spark.small.model import CometSparkSmall
    assert callable(CometSparkSmall)


def test_import_mate_factory():
    """from spark.mate.model import CometSparkMate 工厂函数可用。"""
    from spark.mate.model import CometSparkMate
    assert callable(CometSparkMate)


def test_top_level_exports():
    """spark 顶层导出双模型符号。"""
    import spark
    assert hasattr(spark, "CometSparkSmallLM")
    assert hasattr(spark, "CometSparkSmall")
    assert hasattr(spark, "CometSparkSmallConfig")
    assert hasattr(spark, "CometSparkMateLM")
    assert hasattr(spark, "CometSparkMate")
    assert hasattr(spark, "CometSparkMateConfig")


# ---------------------------------------------------------------------------
# 1. Small 模型构建 + 参数量预算
# ---------------------------------------------------------------------------


class TestSmallModelConstruction:
    """Small 模型构建与参数量验证。"""

    def test_small_factory_returns_small_lm(self):
        """CometSparkSmall() 返回 CometSparkSmallLM 实例。"""
        from spark.small.model import CometSparkSmall, CometSparkSmallLM

        model = CometSparkSmall()
        assert isinstance(model, CometSparkSmallLM)

    def test_small_factory_param_count_under_1m(self):
        """Small 模型参数量 < 1M（小尺寸配置）。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        params = model.count_parameters()
        assert params < 1_000_000, (
            f"Small 参数量应 < 1M，实际 {params}"
        )
        # 极小配置下应在 100K-400K 区间
        assert params > 0

    def test_small_config_defaults(self):
        """Small 配置默认值正确（极小配置）。"""
        from spark.small.model import CometSparkSmallConfig

        cfg = CometSparkSmallConfig()
        assert cfg.vocab_size == 256
        assert cfg.n_embd == 64
        assert cfg.n_layer == 2
        assert cfg.n_head == 4
        assert cfg.n_kv_head == 2
        assert cfg.mod_every == 2
        assert cfg.num_dense_parts == 2
        assert cfg.num_experts_per_part == 2
        assert cfg.top_k == 1
        # small 用更高 init_std
        assert cfg.init_std == 0.04
        assert cfg.arch == "versenex"

    def test_small_model_has_cometspark_nex_lm_net(self):
        """Small 模型内部 net 是 verse_nex.CometSparkNexLM（基于 NexLM）。"""
        from spark.small.model import CometSparkSmall
        from verse_nex.cometspark import CometSparkNexLM

        model = CometSparkSmall()
        assert isinstance(model.net, CometSparkNexLM)


# ---------------------------------------------------------------------------
# 2. Mate 模型构建（小尺寸测试配置避免 1B OOM）
# ---------------------------------------------------------------------------


class TestMateModelConstruction:
    """Mate 模型构建（用小尺寸测试配置避免构建真实 1B 模型）。"""

    @staticmethod
    def _build_small_mate():
        """用小尺寸配置构造 Mate 模型（避免 OOM）。"""
        from spark.mate.model import CometSparkMate

        return CometSparkMate(
            vocab_size=256,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            seq_len=64,
            max_position_embeddings=256,
            window_size=32,
            num_global_tokens=4,
            use_alibi=True,
            use_rope=False,
        )

    def test_mate_factory_returns_mate_lm(self):
        """CometSparkMate() 返回 CometSparkMateLM 实例。"""
        from spark.mate.model import CometSparkMateLM

        model = self._build_small_mate()
        assert isinstance(model, CometSparkMateLM)

    def test_mate_factory_param_count_under_1m(self):
        """Mate 模型（小尺寸测试配置）参数量 < 1M。"""
        model = self._build_small_mate()
        params = model.count_parameters()
        assert params < 1_000_000, (
            f"Mate 测试配置参数量应 < 1M，实际 {params}"
        )
        assert params > 0

    def test_mate_config_defaults(self):
        """Mate 配置默认值正确（旗舰配置，与 V05 1B 一致）。"""
        from spark.mate.model import CometSparkMateConfig

        cfg = CometSparkMateConfig()
        assert cfg.vocab_size == 248320
        assert cfg.n_embd == 1024
        assert cfg.n_layer == 20
        assert cfg.n_head == 16
        assert cfg.n_kv_head == 8
        assert cfg.mod_every == 4
        assert cfg.num_dense_parts == 4
        assert cfg.num_experts_per_part == 4
        assert cfg.top_k == 2
        # mate 用中等 init_std
        assert cfg.init_std == 0.02
        assert cfg.arch == "versenex"

    def test_mate_model_has_cometspark_nex_lm_net(self):
        """Mate 模型内部 net 是 verse_nex.CometSparkNexLM（基于 NexLM）。"""
        from verse_nex.cometspark import CometSparkNexLM

        model = self._build_small_mate()
        assert isinstance(model.net, CometSparkNexLM)


# ---------------------------------------------------------------------------
# 3. VMPC 预设验证
# ---------------------------------------------------------------------------


class TestVMPCPresets:
    """VMPC 预设字段验证。"""

    def test_small_vmpc_profile(self):
        """Small 配置的 vmpc_profile == 'small'。"""
        from spark.small.model import CometSparkSmall

        model = CometSparkSmall()
        assert model.config.vmpc_profile == "small"
        # small 预设：ternary + 高稀疏 + 无 lora + 无蒸馏
        assert model.config.vmpc_quantize_dtype == "ternary"
        assert model.config.vmpc_prune_sparsity == 0.5
        assert model.config.vmpc_use_lora is False
        assert model.config.vmpc_distill is False

    def test_mate_vmpc_profile(self):
        """Mate 配置的 vmpc_profile == 'mate'。"""
        from spark.mate.model import CometSparkMate

        # 用小尺寸配置避免 OOM
        model = CometSparkMate(
            vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_kv_head=2,
            seq_len=64, max_position_embeddings=256,
            window_size=32, num_global_tokens=4,
            use_alibi=True, use_rope=False,
        )
        assert model.config.vmpc_profile == "mate"
        # mate 预设：int4 + 中稀疏 + lora + 蒸馏
        assert model.config.vmpc_quantize_dtype == "int4"
        assert model.config.vmpc_prune_sparsity == 0.3
        assert model.config.vmpc_use_lora is True
        assert model.config.vmpc_distill is True

    def test_small_checkpoint_save_dir(self):
        """Small 配置的 checkpoint_save_dir == 'mf_small'。"""
        from spark.small.model import CometSparkSmallConfig

        cfg = CometSparkSmallConfig()
        assert cfg.checkpoint_save_dir == "mf_small"

    def test_mate_checkpoint_save_dir(self):
        """Mate 配置的 checkpoint_save_dir == 'mf_mate'。"""
        from spark.mate.model import CometSparkMateConfig

        cfg = CometSparkMateConfig()
        assert cfg.checkpoint_save_dir == "mf_mate"


# ---------------------------------------------------------------------------
# 4. 配置加载（from_yaml）
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """YAML 配置加载测试。"""

    def test_small_config_from_yaml(self):
        """CometSparkSmallConfig.from_yaml 加载 cometspark_small.yml 成功。"""
        from spark.small.model import CometSparkSmallConfig

        yml_path = str(_REPO_ROOT / "spark" / "small" / "config" / "cometspark_small.yml")
        assert os.path.exists(yml_path), f"配置文件不存在：{yml_path}"

        cfg = CometSparkSmallConfig.from_yaml(yml_path)
        # 架构字段
        assert cfg.arch == "versenex"
        assert cfg.vocab_size == 256
        assert cfg.n_embd == 64
        assert cfg.n_layer == 2
        assert cfg.mod_every == 2
        assert cfg.num_dense_parts == 2
        assert cfg.num_experts_per_part == 2
        assert cfg.top_k == 1
        assert cfg.init_std == 0.04
        # VMPC 段
        assert cfg.vmpc_profile == "small"
        assert cfg.vmpc_prune_sparsity == 0.5
        assert cfg.vmpc_quantize_dtype == "ternary"
        assert cfg.vmpc_use_lora is False
        assert cfg.vmpc_distill is False
        # checkpoint 段
        assert cfg.checkpoint_save_dir == "mf_small"

    def test_mate_config_from_yaml(self):
        """CometSparkMateConfig.from_yaml 加载 cometspark_mate.yml 成功。"""
        from spark.mate.model import CometSparkMateConfig

        yml_path = str(_REPO_ROOT / "spark" / "mate" / "config" / "cometspark_mate.yml")
        assert os.path.exists(yml_path), f"配置文件不存在：{yml_path}"

        cfg = CometSparkMateConfig.from_yaml(yml_path)
        # 架构字段
        assert cfg.arch == "versenex"
        assert cfg.vocab_size == 248320
        assert cfg.n_embd == 1024
        assert cfg.n_layer == 20
        assert cfg.mod_every == 4
        assert cfg.num_dense_parts == 4
        assert cfg.num_experts_per_part == 4
        assert cfg.top_k == 2
        assert cfg.init_std == 0.02
        # VMPC 段
        assert cfg.vmpc_profile == "mate"
        assert cfg.vmpc_prune_sparsity == 0.3
        assert cfg.vmpc_quantize_dtype == "int4"
        assert cfg.vmpc_use_lora is True
        assert cfg.vmpc_distill is True
        # checkpoint 段
        assert cfg.checkpoint_save_dir == "mf_mate"

    def test_small_config_roundtrip(self):
        """Small 配置 to_dict → from_dict roundtrip 保持 VMPC 字段。"""
        from spark.small.model import CometSparkSmallConfig

        cfg = CometSparkSmallConfig()
        d = cfg.to_dict()
        # VMPC 字段在 dict 中
        assert d["vmpc_profile"] == "small"
        assert d["vmpc_prune_sparsity"] == 0.5
        assert d["checkpoint_save_dir"] == "mf_small"
        # roundtrip
        cfg2 = CometSparkSmallConfig.from_dict(d)
        assert cfg2.vmpc_profile == "small"
        assert cfg2.vmpc_prune_sparsity == 0.5
        assert cfg2.checkpoint_save_dir == "mf_small"

    def test_mate_config_roundtrip(self):
        """Mate 配置 to_dict → from_dict roundtrip 保持 VMPC 字段。"""
        from spark.mate.model import CometSparkMateConfig

        cfg = CometSparkMateConfig()
        d = cfg.to_dict()
        assert d["vmpc_profile"] == "mate"
        assert d["vmpc_use_lora"] is True
        assert d["checkpoint_save_dir"] == "mf_mate"
        cfg2 = CometSparkMateConfig.from_dict(d)
        assert cfg2.vmpc_profile == "mate"
        assert cfg2.vmpc_use_lora is True
        assert cfg2.checkpoint_save_dir == "mf_mate"


# ---------------------------------------------------------------------------
# 5. 前向传播
# ---------------------------------------------------------------------------


class TestForwardPass:
    """前向传播测试。"""

    def test_small_forward_does_not_error(self):
        """Small 模型 forward 不报错（小尺寸输入）。"""
        from spark.small.model import CometSparkSmall

        np.random.seed(SEED)
        model = CometSparkSmall()
        idx = np.random.randint(0, 256, size=(1, 8))

        logits = model.forward(idx)
        # logits shape 应为 (B, T, vocab_size)
        assert logits.data.shape == (1, 8, 256)

    def test_small_forward_with_aux(self):
        """Small 模型 forward_with_aux 不报错。"""
        from spark.small.model import CometSparkSmall

        np.random.seed(SEED)
        model = CometSparkSmall()
        idx = np.random.randint(0, 256, size=(1, 8))

        logits, aux = model.forward_with_aux(idx)
        assert logits.data.shape == (1, 8, 256)

    def test_small_forward_recurrent(self):
        """Small 模型 forward_recurrent 不报错（单步推理）。"""
        from spark.small.model import CometSparkSmall

        np.random.seed(SEED)
        model = CometSparkSmall()
        idx = np.random.randint(0, 256, size=(1, 1))

        logits, states = model.forward_recurrent(idx, states=None)
        assert logits.data.shape == (1, 1, 256)
        assert len(states) == model.config.n_layer

    def test_mate_forward_does_not_error(self):
        """Mate 模型（小尺寸配置）forward 不报错。"""
        from spark.mate.model import CometSparkMate

        np.random.seed(SEED)
        model = CometSparkMate(
            vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_kv_head=2,
            seq_len=64, max_position_embeddings=256,
            window_size=32, num_global_tokens=4,
            use_alibi=True, use_rope=False,
        )
        idx = np.random.randint(0, 256, size=(1, 8))

        logits = model.forward(idx)
        assert logits.data.shape == (1, 8, 256)


# ---------------------------------------------------------------------------
# 7. VMPC 压缩
# ---------------------------------------------------------------------------


class TestVMPCCompress:
    """VMPC 压缩测试。"""

    def test_vmpc_compress_small_on_net(self):
        """vmpc_compress(model.net, profile='small') 不报错。"""
        from spark.small.model import CometSparkSmall
        from verse_torch.vmpc import vmpc_compress

        model = CometSparkSmall()
        compressed_net = vmpc_compress(model.net, profile="small")
        assert compressed_net is not None
        assert compressed_net is not model.net  # 不修改原模型

    def test_vmpc_compress_mate_on_net(self):
        """vmpc_compress(model.net, profile='mate') 不报错（小尺寸 mate）。"""
        from spark.mate.model import CometSparkMate
        from verse_torch.vmpc import vmpc_compress

        model = CometSparkMate(
            vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_kv_head=2,
            seq_len=64, max_position_embeddings=256,
            window_size=32, num_global_tokens=4,
            use_alibi=True, use_rope=False,
        )
        compressed_net = vmpc_compress(model.net, profile="mate")
        assert compressed_net is not None
        assert compressed_net is not model.net

    def test_small_vmpc_compress_model_method(self):
        """Small 模型 vmpc_compress_model() 方法返回新 CometSparkSmallLM 实例。"""
        from spark.small.model import CometSparkSmall, CometSparkSmallLM

        model = CometSparkSmall()
        compressed_model = model.vmpc_compress_model()
        assert isinstance(compressed_model, CometSparkSmallLM)
        assert compressed_model is not model  # 不修改原模型

    def test_mate_vmpc_compress_model_method(self):
        """Mate 模型 vmpc_compress_model() 方法返回新 CometSparkMateLM 实例。"""
        from spark.mate.model import CometSparkMate, CometSparkMateLM

        model = CometSparkMate(
            vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_kv_head=2,
            seq_len=64, max_position_embeddings=256,
            window_size=32, num_global_tokens=4,
            use_alibi=True, use_rope=False,
        )
        compressed_model = model.vmpc_compress_model()
        assert isinstance(compressed_model, CometSparkMateLM)
        assert compressed_model is not model


# ---------------------------------------------------------------------------
# 附加：双模型差异验证
# ---------------------------------------------------------------------------


class TestDualModelDifferences:
    """验证 small / mate 双模型的主要差异。"""

    def test_small_and_mate_have_different_defaults(self):
        """Small 和 Mate 配置默认值不同（vocab/dim/layer/expert）。"""
        from spark.small.model import CometSparkSmallConfig
        from spark.mate.model import CometSparkMateConfig

        small_cfg = CometSparkSmallConfig()
        mate_cfg = CometSparkMateConfig()

        # 架构规模差异
        assert small_cfg.vocab_size != mate_cfg.vocab_size
        assert small_cfg.n_embd < mate_cfg.n_embd
        assert small_cfg.n_layer < mate_cfg.n_layer
        assert small_cfg.num_experts_per_part <= mate_cfg.num_experts_per_part

        # init_std 差异（small 更高）
        assert small_cfg.init_std > mate_cfg.init_std

        # VMPC 预设差异
        assert small_cfg.vmpc_profile == "small"
        assert mate_cfg.vmpc_profile == "mate"
        assert small_cfg.vmpc_prune_sparsity > mate_cfg.vmpc_prune_sparsity
        assert small_cfg.vmpc_use_lora is False
        assert mate_cfg.vmpc_use_lora is True

    def test_small_and_mate_checkpoint_dirs_differ(self):
        """Small 和 Mate 的 checkpoint 目录不同。"""
        from spark.small.model import CometSparkSmallConfig
        from spark.mate.model import CometSparkMateConfig

        assert CometSparkSmallConfig().checkpoint_save_dir == "mf_small"
        assert CometSparkMateConfig().checkpoint_save_dir == "mf_mate"

    def test_both_inherit_from_v05_lm(self):
        """Small 和 Mate 都继承自 CometSparkV05LM（基于 CometSparkNexLM）。"""
        from spark.src.base_model import CometSparkV05LM
        from spark.small.model import CometSparkSmallLM
        from spark.mate.model import CometSparkMateLM

        assert issubclass(CometSparkSmallLM, CometSparkV05LM)
        assert issubclass(CometSparkMateLM, CometSparkV05LM)
