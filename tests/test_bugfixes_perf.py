"""Part4K2.5 Task 5：Bug 修复 + 性能优化测试.

覆盖：
1. compute_loss_rate step=0 / 空 window / 全 0 不除零
2. plot_loss_curve 全 0 loss 不报错（matplotlib + ASCII 双路径）
3. CachedDataset / load_jsonl 空行处理
4. BatchLoader __len__ 返回 0 时 tqdm 不报错
5. CometSparkNexLM._generate_recurrent 的 eos_id 正确传递
6. compress_pipeline version 版本号比较（"1.30"/"1.3.0" → v13）
7. 性能优化正确性（cross_entropy np.asarray 路径一致性）
8. evaluate.py _EVAL_MAX_SAFE_LIMIT 安全上限
9. CometSparkV05LM save/load pathlib 路径处理

运行方式：
    cd /workspace && python -m pytest tests/test_bugfixes_perf.py -x -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path 注入
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. compute_loss_rate 边界保护（step=0 / 空 window / 全 0）
# ---------------------------------------------------------------------------


class TestComputeLossRate:
    """compute_loss_rate 在各种边界条件下不除零。"""

    def test_empty_window(self):
        """step=0 时 loss_window 为空，应返回 0.0。"""
        from verse_torch.training import compute_loss_rate
        assert compute_loss_rate([]) == 0.0

    def test_short_window(self):
        """数据量不足 window 时返回 0.0。"""
        from verse_torch.training import compute_loss_rate
        assert compute_loss_rate([1.0, 2.0], window=50) == 0.0

    def test_all_zeros(self):
        """全 0 loss（avg_first < min_delta）应返回 0.0，不除零。"""
        from verse_torch.training import compute_loss_rate
        result = compute_loss_rate([0.0] * 100, window=50)
        assert result == 0.0

    def test_normal_decreasing(self):
        """正常下降 loss 应返回正的下降率。"""
        from verse_torch.training import compute_loss_rate
        losses = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = compute_loss_rate(losses, window=10)
        assert result > 0.0

    def test_window_equal_to_data(self):
        """数据量刚好等于 window 时不报错。"""
        from verse_torch.training import compute_loss_rate
        losses = [5.0] * 25 + [3.0] * 25
        result = compute_loss_rate(losses, window=50)
        assert result > 0.0


# ---------------------------------------------------------------------------
# 2. plot_loss_curve 全 0 loss 不报错
# ---------------------------------------------------------------------------


class TestPlotLossCurveZeroLoss:
    """plot_loss_curve 在 loss 全 0 时不报错且生成有效文件。"""

    def test_all_zero_matplotlib(self, tmp_path):
        """全 0 loss + matplotlib 可用：PNG 文件正常生成。"""
        from verse_torch.training import plot_loss_curve
        save_path = str(tmp_path / "loss_zero.png")
        result = plot_loss_curve([0.0, 0.0, 0.0, 0.0, 0.0], [], save_path)
        assert os.path.exists(result)
        # 文件非空
        assert os.path.getsize(result) > 0

    def test_all_zero_ascii(self, tmp_path):
        """全 0 loss + ASCII 降级：TXT 文件正常生成。"""
        from verse_torch.training import _plot_ascii
        txt_path = str(tmp_path / "loss_zero.txt")
        _plot_ascii([0.0, 0.0, 0.0], [0.0, 0.0], txt_path, eval_interval=1)
        assert os.path.exists(txt_path)
        content = Path(txt_path).read_text(encoding="utf-8")
        # ASCII 路径有兜底：y_max = y_min + 1.0
        assert "range=" in content

    def test_all_equal_nonzero(self, tmp_path):
        """全相等非零 loss（如全 3.14）也不报错。"""
        from verse_torch.training import plot_loss_curve
        save_path = str(tmp_path / "loss_equal.png")
        val = 3.14
        result = plot_loss_curve([val] * 10, [val] * 3, save_path, eval_interval=3)
        assert os.path.exists(result)

    def test_empty_losses(self, tmp_path):
        """空 loss 列表也不报错。"""
        from verse_torch.training import plot_loss_curve
        save_path = str(tmp_path / "loss_empty.png")
        result = plot_loss_curve([], [], save_path)
        assert os.path.exists(result)

    def test_inf_loss_no_error(self, tmp_path):
        """loss 含 inf / NaN 时 matplotlib 路径不报错（过滤异常值后设 y 轴）。"""
        from verse_torch.training import plot_loss_curve
        save_path = str(tmp_path / "loss_inf.png")
        # 混合正常值与 inf（模拟训练异常时的 loss）
        result = plot_loss_curve([1.0, 2.0, float('inf'), 1.5], [], save_path)
        assert os.path.exists(result)

    def test_all_inf_loss_no_error(self, tmp_path):
        """全 inf loss 时 matplotlib 路径不报错（跳过 set_ylim）。"""
        from verse_torch.training import plot_loss_curve
        save_path = str(tmp_path / "loss_all_inf.png")
        result = plot_loss_curve([float('inf'), float('inf')], [], save_path)
        assert os.path.exists(result)


# ---------------------------------------------------------------------------
# 3. load_jsonl 空行处理
# ---------------------------------------------------------------------------


class TestLoadJsonlEmptyLines:
    """load_jsonl 正确跳过空行。"""

    def test_skip_empty_lines(self, tmp_path):
        """JSONL 文件中的空行（含纯空白行）应被跳过。"""
        from verse_infra.verse_trainer.data import load_jsonl
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(
            '{"text": "hello"}\n'
            '\n'
            '  \n'
            '{"text": "world"}\n'
            '\n'
            '{"text": "foo"}\n',
            encoding="utf-8",
        )
        items = load_jsonl(str(jsonl_path))
        assert len(items) == 3
        assert items[0]["text"] == "hello"
        assert items[1]["text"] == "world"
        assert items[2]["text"] == "foo"

    def test_all_empty_lines(self, tmp_path):
        """全空行文件返回空列表，不报错。"""
        from verse_infra.verse_trainer.data import load_jsonl
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("\n\n  \n\n", encoding="utf-8")
        items = load_jsonl(str(jsonl_path))
        assert items == []

    def test_no_trailing_newline(self, tmp_path):
        """无尾换行的文件也能正确解析。"""
        from verse_infra.verse_trainer.data import load_jsonl
        jsonl_path = tmp_path / "no_trail.jsonl"
        jsonl_path.write_text('{"a": 1}\n{"b": 2}', encoding="utf-8")
        items = load_jsonl(str(jsonl_path))
        assert len(items) == 2


# ---------------------------------------------------------------------------
# 4. BatchLoader __len__ 返回 0 时 tqdm 不报错
# ---------------------------------------------------------------------------


class TestBatchLoaderEmpty:
    """BatchLoader 在空数据集时不报错，且与 tqdm 兼容。"""

    def test_len_returns_zero(self):
        """空数据集时 __len__ 返回 0。"""
        from verse_torch.training import BatchLoader
        loader = BatchLoader([], batch_size=4, shuffle=False)
        assert len(loader) == 0

    def test_iter_empty_no_error(self):
        """空数据集迭代不报错，yield 0 个 batch。"""
        from verse_torch.training import BatchLoader
        loader = BatchLoader([], batch_size=4, shuffle=False)
        batches = list(loader)
        assert batches == []

    def test_tqdm_with_zero_total(self):
        """tqdm total=0 不报错（验证 tqdm 兼容性）。"""
        try:
            from tqdm.auto import tqdm
        except ImportError:
            pytest.skip("tqdm 不可用")
        # tqdm 对 total=0 有良好容错
        pbar = tqdm(total=0, disable=True)
        pbar.close()
        # 迭代空 loader + tqdm 也不报错
        from verse_torch.training import BatchLoader
        loader = BatchLoader([], batch_size=4, shuffle=False)
        for _ in tqdm(loader, disable=True):
            pass

    def test_data_infra_batch_loader_empty(self):
        """verse_infra 的 BatchLoader 空数据集也不报错。"""
        from verse_infra.verse_trainer.data import BatchLoader as InfraBatchLoader
        loader = InfraBatchLoader([], batch_size=4, shuffle=False)
        assert len(loader) == 0
        assert list(loader) == []


# ---------------------------------------------------------------------------
# 5. CometSparkNexLM._generate_recurrent 的 eos_id 正确传递
# ---------------------------------------------------------------------------


class _MockRecurrentModel:
    """可控 mock 模型：forward_recurrent 在第 N 步生成 EOS。

    用于测试 CometSparkNexLM._generate_recurrent 的 eos_id 传递逻辑。
    """

    def __init__(self, vocab_size: int = 16, eos_id: int = 0, eos_after: int = 5):
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.eos_after = eos_after
        self._call_count = 0

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def forward_recurrent(self, input_ids, states=None):
        """单步递推：在第 eos_after 次后输出 eos_id。"""
        from verse_torch import Tensor
        self._call_count += 1
        B = 1
        V = self.vocab_size
        if self.eos_after is not None and self._call_count >= self.eos_after:
            next_tok = self.eos_id
        else:
            next_tok = (self.eos_id + 1) % self.vocab_size
            if next_tok == self.eos_id:
                next_tok = (self.eos_id + 2) % self.vocab_size
        logits_np = np.zeros((B, 1, V), dtype=np.float32)
        logits_np[0, 0, next_tok] = 100.0
        return Tensor(logits_np, requires_grad=False), states


class TestEosIdPassing:
    """验证 eos_id 在 _generate_recurrent 中正确传递并触发提前停止。"""

    def _build_mock_model(self, eos_id=0, eos_after=5):
        """构造 tiny CometSparkNexLM 并替换 forward_recurrent 为 mock。"""
        from verse_nex.cometspark import CometSparkNexLM
        model = CometSparkNexLM(
            vocab_size=16, dim=8, n_layer=1, n_head=2,
            layer_pattern=["trisparse"],
            max_seq_len=32,
            num_dense_parts=2, num_experts_per_part=2, top_k=1,
        )
        mock = _MockRecurrentModel(vocab_size=16, eos_id=eos_id, eos_after=eos_after)
        model.forward_recurrent = mock.forward_recurrent
        return model

    def test_eos_id_stops_generation(self):
        """eos_id 不为 None 时，生成到 EOS 应提前停止（不达到 max_new_tokens）。"""
        model = self._build_mock_model(eos_id=0, eos_after=5)
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        # max_new_tokens=100 + eos_id=0：mock 在第 5 次 forward_recurrent 输出 eos
        # 前 3 次 prompt 预热，第 4-5 次生成 token，第 5 次生成 eos → 提前停止
        out = model.generate(prompt, max_new_tokens=100, eos_id=0)
        n_generated = out.shape[1] - prompt.shape[1]
        # 远小于 100（eos 提前停止），加上 generate 强制追加的 1 个 eos
        assert n_generated < 100, (
            f"eos_id 未生效：生成了 {n_generated} 个 token，应在 EOS 处提前停止"
        )

    def test_eos_id_none_no_early_stop(self):
        """eos_id=None 时不提前停止，generate 强制追加 eos 列。"""
        model = self._build_mock_model(eos_id=0, eos_after=5)
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        # eos_id=None：generate 不会追加 eos，也不会提前停止
        out = model.generate(prompt, max_new_tokens=5, eos_id=None)
        # 恰好 5 个 token（无追加）
        assert out.shape[1] == prompt.shape[1] + 5

    def test_recurrent_path_uses_eos_id(self):
        """直接调用 _generate_recurrent 验证 eos_id 参数被使用。"""
        model = self._build_mock_model(eos_id=0, eos_after=5)
        prompt = np.array([[1, 2]], dtype=np.int64)
        # eos_id=None 时生成恰好 max_new_tokens 个（不提前停止）
        out_no_eos = model._generate_recurrent(prompt, max_new_tokens=10, eos_id=None)
        assert out_no_eos.shape[1] == prompt.shape[1] + 10
        # eos_id=0 时提前停止（mock 在第 5 次 forward_recurrent 输出 eos）
        # prompt 预热 2 次，第 3-5 次生成，第 5 次生成 eos → 停止
        model.forward_recurrent.__self__._call_count = 0  # 重置计数
        out_with_eos = model._generate_recurrent(prompt, max_new_tokens=100, eos_id=0)
        assert out_with_eos.shape[1] < prompt.shape[1] + 100, (
            "eos_id 未生效：_generate_recurrent 应在 EOS 处提前停止"
        )


# ---------------------------------------------------------------------------
# 6. compress_pipeline version 版本号比较
# ---------------------------------------------------------------------------


class TestCompressPipelineVersion:
    """compress_pipeline 的 version 参数用版本号比较而非字符串比较。"""

    def _build_small_model(self):
        """构造小模型用于压缩测试。"""
        import warnings
        from verse_torch import nn
        np.random.seed(42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return nn.TransformerLM(
                vocab_size=32, n_layer=1, n_head=2,
                n_embd=16, seq_len=16, dropout=0.0, tie_weights=True,
            )

    def test_version_1_3_goes_v13(self):
        """version='1.3' 走 v13 分支。"""
        from verse_torch.compress import compress_pipeline
        model = self._build_small_model()
        _, stats = compress_pipeline(
            model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
            return_stats=True, version="1.3",
        )
        assert stats.get("version") == "1.3"

    def test_version_1_30_goes_v13(self):
        """version='1.30' 应走 v13 分支（修复前会走 v2）。"""
        from verse_torch.compress import compress_pipeline
        model = self._build_small_model()
        _, stats = compress_pipeline(
            model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
            return_stats=True, version="1.30",
        )
        assert stats.get("version") == "1.3", (
            f"version='1.30' 应走 v13 分支，实际 version={stats.get('version')}"
        )

    def test_version_1_3_0_goes_v13(self):
        """version='1.3.0' 应走 v13 分支（修复前会走 v2）。"""
        from verse_torch.compress import compress_pipeline
        model = self._build_small_model()
        _, stats = compress_pipeline(
            model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
            return_stats=True, version="1.3.0",
        )
        assert stats.get("version") == "1.3", (
            f"version='1.3.0' 应走 v13 分支，实际 version={stats.get('version')}"
        )

    def test_version_1_2_goes_v2(self):
        """version='1.2' 走 v2 分支（stats 无 version 字段）。"""
        from verse_torch.compress import compress_pipeline
        model = self._build_small_model()
        _, stats = compress_pipeline(
            model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
            return_stats=True, version="1.2",
        )
        assert "version" not in stats, (
            f"version='1.2' 应走 v2 分支（无 version 字段），实际有 version={stats.get('version')}"
        )

    def test_version_1_0_goes_v2(self):
        """version='1.0' 走 v2 分支。"""
        from verse_torch.compress import compress_pipeline
        model = self._build_small_model()
        _, stats = compress_pipeline(
            model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
            return_stats=True, version="1.0",
        )
        assert "version" not in stats

    def test_parse_version_tuple(self):
        """_parse_version_tuple 辅助函数正确解析版本字符串。"""
        from verse_torch.compress import _parse_version_tuple
        assert _parse_version_tuple("1.3") == (1, 3)
        assert _parse_version_tuple("1.30") == (1, 30)
        assert _parse_version_tuple("1.3.0") == (1, 3, 0)
        assert _parse_version_tuple("1.2") == (1, 2)
        assert _parse_version_tuple("2.0") == (2, 0)
        assert _parse_version_tuple(1.3) == (1, 3)  # float 也兼容


# ---------------------------------------------------------------------------
# 7. 性能优化正确性：cross_entropy np.asarray 路径一致性
# ---------------------------------------------------------------------------


class TestCrossEntropyConsistency:
    """cross_entropy 在不同 targets 输入类型下结果一致（np.asarray 优化不改变结果）。"""

    def test_targets_list_vs_ndarray(self):
        """targets 为 list 与 ndarray 时 loss 值一致。"""
        from verse_torch import Tensor
        from verse_torch.losses import cross_entropy
        logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        targets_list = [0, 3, 7, 9]
        targets_ndarray = np.array(targets_list, dtype=np.int64)
        loss1 = cross_entropy(logits, targets_list)
        loss2 = cross_entropy(logits, targets_ndarray)
        assert abs(float(loss1.data) - float(loss2.data)) < 1e-6

    def test_targets_tensor_vs_ndarray(self):
        """targets 为 Tensor 与 ndarray 时 loss 值一致。"""
        from verse_torch import Tensor
        from verse_torch.losses import cross_entropy
        logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        targets_np = np.array([1, 2, 3, 4], dtype=np.int64)
        targets_tensor = Tensor(targets_np)
        loss1 = cross_entropy(logits, targets_np)
        loss2 = cross_entropy(logits, targets_tensor)
        assert abs(float(loss1.data) - float(loss2.data)) < 1e-6

    def test_3d_logits_reshape(self):
        """3D logits (B, T, V) 自动 reshape 为 2D 后结果正确。"""
        from verse_torch import Tensor
        from verse_torch.losses import cross_entropy
        B, T, V = 2, 3, 5
        logits_3d = Tensor(np.random.randn(B, T, V).astype(np.float32), requires_grad=True)
        logits_2d = Tensor(logits_3d.data.reshape(-1, V).copy(), requires_grad=True)
        targets = np.random.randint(0, V, size=(B, T), dtype=np.int64)
        targets_flat = targets.reshape(-1)
        loss_3d = cross_entropy(logits_3d, targets)
        loss_2d = cross_entropy(logits_2d, targets_flat)
        assert abs(float(loss_3d.data) - float(loss_2d.data)) < 1e-6

    def test_ignore_index_mask(self):
        """ignore_index 正确屏蔽不参与 loss 计算的位置。"""
        from verse_torch import Tensor
        from verse_torch.losses import cross_entropy
        logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        targets = np.array([0, -100, 5, -100], dtype=np.int64)
        loss = cross_entropy(logits, targets, ignore_index=-100)
        # 不应报错，且 loss 为有限值
        assert np.isfinite(float(loss.data))

    def test_backward_works(self):
        """cross_entropy 反向传播正常工作（np.asarray 不破坏梯度流）。"""
        from verse_torch import Tensor
        from verse_torch.losses import cross_entropy
        logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        targets = [0, 3, 7, 9]
        loss = cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.data.shape


# ---------------------------------------------------------------------------
# 8. evaluate.py _EVAL_MAX_SAFE_LIMIT 安全上限
# ---------------------------------------------------------------------------


class TestEvalMaxSafeLimit:
    """验证 evaluate.py 的评估安全上限常量。"""

    def test_constant_value(self):
        """_EVAL_MAX_SAFE_LIMIT 应为 256（远小于模型默认 100K）。"""
        from verse_infra.verse_trainer.evaluate import _EVAL_MAX_SAFE_LIMIT
        assert _EVAL_MAX_SAFE_LIMIT == 256
        assert _EVAL_MAX_SAFE_LIMIT < 100_000

    def test_constant_is_int(self):
        """_EVAL_MAX_SAFE_LIMIT 应为整数类型。"""
        from verse_infra.verse_trainer.evaluate import _EVAL_MAX_SAFE_LIMIT
        assert isinstance(_EVAL_MAX_SAFE_LIMIT, int)


# ---------------------------------------------------------------------------
# 9. CometSparkV05LM save/load pathlib 路径处理
# ---------------------------------------------------------------------------


class TestCometSparkV05SaveLoad:
    """CometSparkV05LM save/load 用 pathlib 处理路径，跨平台兼容。"""

    def test_save_load_roundtrip(self, tmp_path):
        """save → load roundtrip 后 state_dict 一致。"""
        from spark.model.model import CometSparkV05Small
        model = CometSparkV05Small(vocab_size=32, n_embd=16, n_layer=1)
        sd_before = {k: v.copy() for k, v in model.state_dict().items()}

        save_path = str(tmp_path / "subdir" / "model.pt")
        model.save(save_path)
        assert os.path.exists(save_path)

        # 加载到新模型
        model2 = CometSparkV05Small(vocab_size=32, n_embd=16, n_layer=1)
        model2.load(save_path)
        sd_after = model2.state_dict()
        for k in sd_before:
            assert k in sd_after, f"load 后缺少 key: {k}"
            np.testing.assert_allclose(
                sd_before[k], sd_after[k], atol=1e-6,
                err_msg=f"key={k} 的值在 save/load 后不一致",
            )

    def test_save_no_directory_component(self, tmp_path):
        """save 到无目录组件的文件名（如 'model.pt'）不报错。"""
        from spark.model.model import CometSparkV05Small
        model = CometSparkV05Small(vocab_size=32, n_embd=16, n_layer=1)
        # 切换到临时目录
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            model.save("model.pt")
            assert os.path.exists("model.pt")
        finally:
            os.chdir(original_cwd)

    def test_save_creates_nested_dirs(self, tmp_path):
        """save 自动创建嵌套目录。"""
        from spark.model.model import CometSparkV05Small
        model = CometSparkV05Small(vocab_size=32, n_embd=16, n_layer=1)
        nested_path = str(tmp_path / "a" / "b" / "c" / "model.pt")
        model.save(nested_path)
        assert os.path.exists(nested_path)
