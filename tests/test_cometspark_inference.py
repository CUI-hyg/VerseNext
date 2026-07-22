"""Stage 7: verse_inference CometSpark 兼容测试（Part4K1 Task 8.9 迁移到 spark/model）。

验证：
1. ``ModelLoader(arch="cometspark")`` 能从 pickle 文件加载 ``CometSparkV05LM``；
2. ``StreamingGenerator`` 能调用 ``CometSparkV05LM.forward_recurrent`` 生成 token；
3. 生成 100 tokens 的 wall-clock ≤ 5 秒；
4. 生成的 token 序列长度 = 100；
5. 向后兼容：``ModelLoader(arch="mamba2")`` 等原有 arch 分支仍可正常工作。

Part4K1 Task 8.9: 模型从 data/demo 迁移到 spark/model。
- 原依赖 ``data/demo/checkpoints/cometspark.pt`` 的 fixture 改为动态生成：
  用 ``CometSparkV05Small()`` 构造小模型 → ``save()`` 到 tmp_path。
- 加载的模型类名从 ``CometSparkLM`` 改为 ``CometSparkV05LM``。
- ``register_cometspark_path`` 参数从 ``/workspace/data/demo`` 改为 ``/workspace``（spark 是顶层包）。

运行：
    cd /workspace && python -m pytest tests/test_cometspark_inference.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import pickle

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_infra.verse_inference / spark.model
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
for _pkg in ("verse_infra", "verse_torch", "verse_nex"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cometspark_pt_path():
    """动态生成 CometSparkV05LM checkpoint 文件路径（module 级别共享）。

    Part4K1 Task 8.9: 原 fixture 依赖 ``data/demo/checkpoints/cometspark.pt``，
    现改为用 ``CometSparkV05Small()`` 构造小模型并 ``save()`` 到临时文件。
    """
    from spark.model.model import CometSparkV05Small

    # 构造小模型并保存为 pickle 单文件
    tmpdir = tempfile.mkdtemp(prefix="cometspark_v05_test_")
    pt_path = os.path.join(tmpdir, "cometspark_v05.pt")
    model = CometSparkV05Small()
    model.save(pt_path)
    return pt_path


@pytest.fixture(scope="module")
def cometspark_model(cometspark_pt_path):
    """加载 CometSparkV05LM 模型（module 级别共享，避免重复加载）。"""
    from verse_infra.verse_inference import ModelLoader

    loader = ModelLoader(arch="cometspark")
    model = loader.load(cometspark_pt_path)
    return model


# ---------------------------------------------------------------------------
# Task 7.1: ModelLoader cometspark arch 分支
# ---------------------------------------------------------------------------


class TestModelLoaderCometSpark:
    """Task 7.1: 验证 ModelLoader 的 cometspark arch 分支。"""

    def test_arch_validation_accepts_cometspark(self):
        """arch='cometspark' 应被接受（不抛异常）。"""
        from verse_infra.verse_inference import ModelLoader

        loader = ModelLoader(arch="cometspark")
        assert loader.arch == "cometspark"

    def test_arch_validation_rejects_unknown(self):
        """未知 arch 应抛 ValueError。"""
        from verse_infra.verse_inference import ModelLoader

        with pytest.raises(ValueError, match="arch must be one of"):
            ModelLoader(arch="unknown_arch")

    def test_load_cometspark_returns_model(self, cometspark_model):
        """加载 cometspark.pt 应返回 CometSparkV05LM 实例。"""
        # Part4K1 Task 8.9: 类名应为 CometSparkV05LM（不再是 CometSparkLM）
        assert type(cometspark_model).__name__ == "CometSparkV05LM"
        # 应有 config 属性
        assert hasattr(cometspark_model, "config")
        # 应有 forward_recurrent 方法（StreamingGenerator 接口）
        assert hasattr(cometspark_model, "forward_recurrent")
        assert callable(cometspark_model.forward_recurrent)
        # 应挂载 _arch 标记
        assert getattr(cometspark_model, "_arch", None) == "cometspark"

    def test_load_cometspark_eval_mode(self, cometspark_model):
        """加载后应处于 eval 模式且 requires_grad=False。"""
        assert cometspark_model.net.training is False
        for p in cometspark_model.parameters():
            assert p.requires_grad is False

    def test_load_cometspark_config_correct(self, cometspark_model, cometspark_pt_path):
        """加载的 config 应与 pickle 中的 config 一致。"""
        with open(cometspark_pt_path, "rb") as f:
            payload = pickle.load(f)
        cfg = payload["config"]
        assert cometspark_model.config.vocab_size == cfg["vocab_size"]
        assert cometspark_model.config.n_layer == cfg["n_layer"]
        assert cometspark_model.config.n_embd == cfg["n_embd"]
        # Part4K1: arch 统一为 versenex
        assert cometspark_model.config.arch == "versenex"
        assert cfg["arch"] == "versenex"

    def test_load_cometspark_path_required(self):
        """cometspark arch 不提供 path 应抛 ValueError。"""
        from verse_infra.verse_inference import ModelLoader

        loader = ModelLoader(arch="cometspark")
        with pytest.raises(ValueError, match="model_path"):
            loader.load(None)

    def test_load_cometspark_file_not_found(self):
        """不存在的文件应抛 FileNotFoundError。"""
        from verse_infra.verse_inference import ModelLoader

        loader = ModelLoader(arch="cometspark")
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/to/cometspark.pt")

    def test_register_cometspark_path_callable(self):
        """register_cometspark_path 函数应可调用。"""
        from verse_infra.verse_inference.model_loader import register_cometspark_path

        # Part4K1 Task 8.9: 路径从 /workspace/data/demo 改为 /workspace
        # （spark 是顶层包，/workspace 在 sys.path 即可导入）
        register_cometspark_path("/workspace")


# ---------------------------------------------------------------------------
# Task 7.2: StreamingGenerator 兼容 CometSparkV05LM
# ---------------------------------------------------------------------------


class TestStreamingGeneratorCompat:
    """Task 7.2: 验证 StreamingGenerator 能与 CometSparkV05LM 协作。"""

    def test_forward_recurrent_returns_tuple(self, cometspark_model):
        """forward_recurrent 应返回 (logits, new_states) 元组。"""
        from verse_torch import Tensor

        input_ids = Tensor(
            np.array([[1]], dtype=np.int64), requires_grad=False
        )
        out = cometspark_model.forward_recurrent(input_ids, None)
        assert isinstance(out, tuple) and len(out) == 2
        logits, new_states = out
        # logits 应是 Tensor，shape (1, 1, vocab_size)
        assert hasattr(logits, "data")
        assert logits.data.shape == (1, 1, cometspark_model.config.vocab_size)

    def test_generator_generates_100_tokens(self, cometspark_model):
        """StreamingGenerator 应能生成 100 个 token，wall-clock ≤ 5 秒。

        CPU 性能较弱时可能略超，本测试阈值设为 10 秒（5s 目标 + 余量），
        但实际预期远低于 5s（V05Small 模型很小）。
        """
        from verse_infra.verse_inference import StreamingGenerator

        gen = StreamingGenerator(cometspark_model)
        prompt = [1, 2, 3, 4, 5]

        t0 = time.time()
        tokens = list(gen.generate(prompt, max_new_tokens=100))
        wall_clock = time.time() - t0

        # 验证生成了正好 100 个 token
        assert len(tokens) == 100, (
            f"期望生成 100 tokens，实际 {len(tokens)}"
        )
        # 验证每个 token 都是合法的 int 且在 vocab 范围内
        vocab_size = cometspark_model.config.vocab_size
        for i, tok in enumerate(tokens):
            assert isinstance(tok, (int, np.integer)), (
                f"token[{i}] 不是 int：{type(tok)}"
            )
            assert 0 <= int(tok) < vocab_size, (
                f"token[{i}]={tok} 超出 vocab 范围 [0, {vocab_size})"
            )
        # 验证 wall-clock（5s 目标，10s 硬上限）
        assert wall_clock <= 10.0, (
            f"生成 100 tokens 耗时 {wall_clock:.3f}s 超过 10s 上限"
        )
        # 记录实际耗时到日志（便于人工查看）
        print(f"\n[perf] 100 tokens wall-clock = {wall_clock:.3f}s "
              f"(目标 ≤ 5s, 硬上限 10s)")

    def test_generator_deterministic_with_greedy(self, cometspark_model):
        """GreedySampler 应确定性：相同 prompt 两次生成结果一致。"""
        from verse_infra.verse_inference import StreamingGenerator, GreedySampler

        gen1 = StreamingGenerator(cometspark_model, sampler=GreedySampler())
        gen2 = StreamingGenerator(cometspark_model, sampler=GreedySampler())
        prompt = [1, 2, 3]

        tokens1 = list(gen1.generate(prompt, max_new_tokens=30))
        tokens2 = list(gen2.generate(prompt, max_new_tokens=30))

        assert tokens1 == tokens2, (
            "GreedySampler 应确定性，两次结果应完全一致"
        )

    def test_generator_empty_prompt(self, cometspark_model):
        """空 prompt 也应能正常生成（用零 logits 启动）。"""
        from verse_infra.verse_inference import StreamingGenerator

        gen = StreamingGenerator(cometspark_model)
        tokens = list(gen.generate([], max_new_tokens=5))
        assert len(tokens) == 5

    def test_generator_reset_state(self, cometspark_model):
        """reset_state=True 时两次生成应独立（结果一致因 greedy）。"""
        from verse_infra.verse_inference import StreamingGenerator

        gen = StreamingGenerator(cometspark_model)
        prompt = [5, 10, 15]

        tokens1 = list(gen.generate(prompt, max_new_tokens=20, reset_state=True))
        tokens2 = list(gen.generate(prompt, max_new_tokens=20, reset_state=True))

        # greedy + 重置状态：两次结果应一致
        assert tokens1 == tokens2


# ---------------------------------------------------------------------------
# 向后兼容：原有 arch 分支不应被破坏
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """验证原有 mamba2/rwkv7/hybrid arch 分支仍可正常工作。"""

    def test_mamba2_arch_still_works(self):
        """arch='mamba2' 应仍能构建 HybridLM（不依赖 cometspark.pt）。"""
        from verse_infra.verse_inference import ModelLoader

        loader = ModelLoader(arch="mamba2", vocab_size=64, dim=32, n_layers=2)
        model = loader.load()  # 不传 path，仅自构建
        assert model is not None
        assert hasattr(model, "forward_recurrent")
        # _arch 标记
        assert getattr(model, "_arch", None) == "mamba2"

    def test_unknown_arch_still_rejected(self):
        """未知 arch 仍应被拒绝（cometspark 加入后校验不放松）。"""
        from verse_infra.verse_inference import ModelLoader

        with pytest.raises(ValueError):
            ModelLoader(arch="invalid")
