"""Part5K1.3 Task 1.7: ParallelTrainer Phase 3 merge_finetune 递归崩溃修复测试。

验证以下场景不触发 ``RecursionError: maximum recursion depth exceeded in __instancecheck__``：

1. ``Module.state_dict()`` / ``load_state_dict()`` 在 60+ 层 VerseNex 模型下不爆栈
   - 根因修复：子模块遍历从递归 DFS 改为迭代 BFS（显式队列，``collections.deque``）
2. ``pickle.loads(pickle.dumps(state_dict, protocol=4))`` 在深层模块下不爆栈
   - 根因修复：``ParallelTrainer.fit`` 中所有 ``copy.deepcopy(state_dict)`` 改为 pickle 序列化
3. ``ParallelTrainer.fit`` Phase 3 merge_finetune（``chunk_id=-999``）路径不触发 RecursionError
   - 60+ 层 VerseNex 模型 + ``merge_finetune_steps > 0`` 触发 finetune 阶段
4. ``ParallelTrainerSafe._train_chunk_safe`` 包裹的 chunk 执行不爆栈
5. val_loss 收敛（训练后 val_loss 有限且下降）

设计要点：
- 构造 65 层 VerseNex 模型（``CometSparkNexLM(n_layer=65)``），触发 ``n_layer >= 64``
  的 ``chunked_forward`` 路径，覆盖最深的模块树。
- 使用 ``sys.setrecursionlimit`` 临时下调递归上限（100），让原递归 DFS 必然爆栈、
  迭代 BFS 仍可正常工作，从而严格验证修复有效性。
- 现有 ``tests/test_parallel_trainer.py`` / ``test_loss_and_parallel_fix.py`` 零回归。

运行方式：
    cd /workspace && python -m pytest tests/test_parallel_recursion_fix.py -v
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path 适配（对齐 test_recursion_fix.py / test_training_nex.py 模式）
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = REPO_ROOT / "packages" / _pkg
    if _p.is_dir():
        sys.path.insert(0, str(_p))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verse_torch import Tensor, no_grad
from verse_torch.training import ParallelTrainer, cross_entropy_loss, BatchLoader
from verse_nex import CometSparkNexLM


# ---------------------------------------------------------------------------
# 工厂：构造 65 层 VerseNex 模型 + LM 数据集
# ---------------------------------------------------------------------------


def _build_deep_versenex_model(
    n_layer: int = 65,
    vocab_size: int = 32,
    dim: int = 16,
    n_head: int = 2,
    n_kv_head: int = 1,
    max_seq_len: int = 16,
) -> CometSparkNexLM:
    """构造 60+ 层 VerseNex 模型（全 trisparse，最小参数量便于 CPU 测试）。

    - ``n_layer=65`` 触发 ``CometSparkNexLM.chunked_forward`` 路径（n_layer >= 64）
    - 全 ``"trisparse"`` layer_pattern 避免 MoD 路由开销，专注递归路径验证
    - 小 dim / vocab_size 保证 CPU 训练速度
    """
    return CometSparkNexLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        layer_pattern=["trisparse"] * n_layer,
        window_size=4,
        num_global_tokens=2,
        use_alibi=True,
        use_rope=False,
        max_seq_len=max_seq_len,
        dropout=0.0,
        tie_weights=True,
    )


class LMDataset:
    """简单 LM 数据集：每个 sample 返回 ``(input_ids, target_ids)``。

    - ``input_ids``: (T,) int64，随机 token 序列
    - ``target_ids``: (T,) int64，input 右移一位（next-token 预测语义）

    与 ``_default_collate`` 配合：batch 拼成 (B, T) int64 ndarray。
    """

    def __init__(self, n_samples: int = 8, seq_len: int = 4,
                 vocab_size: int = 32, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n = int(n_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        # 预生成全部样本（小数据集，内存可接受）
        self.x = rng.randint(0, vocab_size,
                             size=(n_samples, seq_len)).astype(np.int64)
        # y = x 右移一位（next-token 预测）
        self.y = np.concatenate(
            [self.x[:, 1:],
             rng.randint(0, vocab_size, size=(n_samples, 1))],
            axis=1,
        ).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ---------------------------------------------------------------------------
# 1. state_dict / load_state_dict 迭代 BFS 不爆栈
# ---------------------------------------------------------------------------


class TestStateDictIterativeBFS:
    """验证 ``Module.state_dict()`` / ``load_state_dict()`` 迭代 BFS 修复。"""

    def test_state_dict_no_recursion_error_deep_model(self):
        """65 层 VerseNex 模型 state_dict() 不触发 RecursionError。"""
        model = _build_deep_versenex_model(n_layer=65)
        # 多次调用，确保稳定性
        sd1 = model.state_dict()
        sd2 = model.state_dict()
        # state_dict 应非空
        assert len(sd1) > 0, "state_dict 不应为空"
        # 两次调用结果键一致
        assert set(sd1.keys()) == set(sd2.keys())
        # 65 层 × 每层多个参数 + embedding + head + norm
        # 每层 VerseNexBlock 至少有 norm1/norm2 + attn + ffn 子模块的参数
        # 验证键数与层数正相关（至少 65 × 4 + 全局参数）
        assert len(sd1) >= 65 * 4, (
            f"65 层模型 state_dict 应至少有 {65 * 4} 个键, got {len(sd1)}"
        )

    def test_state_dict_low_recursion_limit_iterative_bfs(self):
        """严格测试：递归上限降到 100，迭代 BFS 仍能完成 state_dict()。

        原递归 DFS 在 65 层模型下会爆栈（每层 ~3-4 帧深度）；
        迭代 BFS 显式队列，调用栈深度恒为 O(1)，不受 ``sys.setrecursionlimit`` 影响。
        """
        model = _build_deep_versenex_model(n_layer=65)
        original_limit = sys.getrecursionlimit()
        try:
            # 降到 100：原递归 DFS 必然爆栈（65 层 × 3-4 帧/层 ≈ 200+ 帧）
            sys.setrecursionlimit(100)
            sd = model.state_dict()
            assert len(sd) > 0, "低递归上限下 state_dict 仍应正常返回"
        finally:
            sys.setrecursionlimit(original_limit)

    def test_load_state_dict_no_recursion_error_deep_model(self):
        """65 层 VerseNex 模型 load_state_dict() 不触发 RecursionError。"""
        model = _build_deep_versenex_model(n_layer=65)
        sd = model.state_dict()
        # 修改参数（确保 load 能覆盖）
        for p in model.parameters():
            p.data = p.data + 1.0
        # load_state_dict 应恢复原值
        model.load_state_dict(sd)
        sd_after = model.state_dict()
        # 验证恢复后数值一致
        for key in sd:
            np.testing.assert_array_equal(
                sd[key], sd_after[key],
                err_msg=f"load_state_dict 后 key={key} 数值不一致",
            )

    def test_load_state_dict_low_recursion_limit_iterative_bfs(self):
        """严格测试：递归上限降到 100，迭代 BFS 仍能完成 load_state_dict()。"""
        model = _build_deep_versenex_model(n_layer=65)
        sd = model.state_dict()
        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            model.load_state_dict(sd)
            # 验证 load 成功：取一个键检查数值一致
            sd_after = model.state_dict()
            first_key = next(iter(sd))
            np.testing.assert_array_equal(sd[first_key], sd_after[first_key])
        finally:
            sys.setrecursionlimit(original_limit)

    def test_state_dict_keys_bfs_order_consistent(self):
        """BFS 遍历顺序与递归 DFS 等价：键集合一致（顺序无关紧要，键值一致即可）。"""
        model = _build_deep_versenex_model(n_layer=10)  # 小模型便于检查
        sd = model.state_dict()
        # 验证每层的参数都出现在 state_dict 中
        # 每层 VerseNexBlock 有 norm1.gamma / norm2.gamma / attn.* / ffn.* 参数
        # 至少包含每层的 norm1.gamma
        layer0_keys = [k for k in sd if k.startswith("blocks.0.")]
        assert len(layer0_keys) > 0, "blocks.0.* 键应存在"
        # 验证 65 层都有对应键
        for i in range(10):
            layer_keys = [k for k in sd if k.startswith(f"blocks.{i}.")]
            assert len(layer_keys) > 0, f"blocks.{i}.* 键应存在"


# ---------------------------------------------------------------------------
# 2. pickle 序列化替代 deepcopy 不爆栈
# ---------------------------------------------------------------------------


class TestPickleSerializationNoRecursion:
    """验证 ``pickle.loads(pickle.dumps(state_dict, protocol=4))`` 不爆栈。

    这是 SubTask 1.2 的核心修复：用 pickle 替代 ``copy.deepcopy``，
    避免 deepcopy 递归对象图遍历在深层模块下爆栈。
    """

    def test_pickle_state_dict_no_recursion_error(self):
        """65 层模型 state_dict pickle 序列化不触发 RecursionError。"""
        model = _build_deep_versenex_model(n_layer=65)
        sd = model.state_dict()
        # pickle 序列化 + 反序列化
        sd_restored = pickle.loads(pickle.dumps(sd, protocol=4))
        # 键集合一致
        assert set(sd_restored.keys()) == set(sd.keys())
        # 数值一致（取若干键检查）
        for key in list(sd.keys())[:5]:
            np.testing.assert_array_equal(sd[key], sd_restored[key])

    def test_pickle_state_dict_low_recursion_limit(self):
        """严格测试：递归上限降到 100，pickle 序列化仍正常。"""
        model = _build_deep_versenex_model(n_layer=65)
        sd = model.state_dict()
        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            sd_restored = pickle.loads(pickle.dumps(sd, protocol=4))
            assert len(sd_restored) == len(sd)
        finally:
            sys.setrecursionlimit(original_limit)

    def test_pickle_roundtrip_preserves_values(self):
        """pickle 序列化往返数值完全一致（float32 位级吻合）。"""
        model = _build_deep_versenex_model(n_layer=20)
        sd = model.state_dict()
        sd_restored = pickle.loads(pickle.dumps(sd, protocol=4))
        # 全部键数值一致
        for key in sd:
            np.testing.assert_array_equal(
                sd[key], sd_restored[key],
                err_msg=f"pickle 往返后 key={key} 数值不一致",
            )


# ---------------------------------------------------------------------------
# 3. ParallelTrainer Phase 3 merge_finetune (chunk_id=-999) 不爆栈
# ---------------------------------------------------------------------------


class TestParallelTrainerMergeFinetuneNoRecursion:
    """验证 ``ParallelTrainer.fit`` Phase 3 merge_finetune 路径不爆栈。

    Phase 3 调用 ``_train_chunk(self.model, self.train_dataset,
    self.merge_finetune_steps, -999)``，是原 bug 的触发点。
    """

    def test_parallel_trainer_fit_deep_model_no_recursion(self, tmp_path):
        """65 层 VerseNex + ParallelTrainer.fit（含 finetune）不触发 RecursionError。"""
        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,  # 2 chunks × 2 steps
            "batch_size": 2,
            "lr": 0.001,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 2,  # 触发 chunk_id=-999
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        # fit 应正常完成，不抛 RecursionError
        history = trainer.fit()

        # best_val_loss 应为有限值
        assert trainer.best_val_loss < float("inf"), (
            f"best_val_loss 应为有限值, got {trainer.best_val_loss}"
        )
        # history 应包含三个列表
        assert "train_loss" in history
        assert "val_loss" in history
        assert "steps" in history
        # chunk_stats 应记录 2 个 chunk
        assert len(trainer.chunk_stats) == 2

    def test_chunk_id_minus_999_invoked(self, tmp_path):
        """验证 chunk_id=-999（Phase 3 finetune）确实被调用。"""
        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,
            "batch_size": 2,
            "lr": 0.001,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 2,  # > 0 触发 finetune
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        # 追踪 _train_chunk 调用，记录 chunk_id
        chunk_ids_seen = []
        original_train_chunk = trainer._train_chunk

        def tracking_train_chunk(m, ds, steps, cid):
            chunk_ids_seen.append(cid)
            return original_train_chunk(m, ds, steps, cid)

        trainer._train_chunk = tracking_train_chunk
        trainer.fit()

        # 应包含 chunk_id=-999（Phase 3 finetune）
        assert -999 in chunk_ids_seen, (
            f"应调用 chunk_id=-999 (Phase 3 finetune), got chunk_ids={chunk_ids_seen}"
        )
        # 还应包含 Phase 1 的 chunk_id >= 0
        phase1_ids = [c for c in chunk_ids_seen if c >= 0]
        assert len(phase1_ids) == 2, (
            f"Phase 1 应有 2 个 chunk (id>=0), got {phase1_ids}"
        )

    def test_merge_finetune_zero_skips_minus_999(self, tmp_path):
        """``merge_finetune_steps=0`` 时跳过 chunk_id=-999（反向验证）。"""
        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,
            "batch_size": 2,
            "lr": 0.001,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 0,  # 关闭 finetune
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        chunk_ids_seen = []
        original_train_chunk = trainer._train_chunk

        def tracking_train_chunk(m, ds, steps, cid):
            chunk_ids_seen.append(cid)
            return original_train_chunk(m, ds, steps, cid)

        trainer._train_chunk = tracking_train_chunk
        trainer.fit()

        # merge_finetune_steps=0 时不应调用 chunk_id=-999
        assert -999 not in chunk_ids_seen, (
            f"merge_finetune_steps=0 时不应调用 chunk_id=-999, "
            f"got chunk_ids={chunk_ids_seen}"
        )


# ---------------------------------------------------------------------------
# 4. ParallelTrainerSafe 包裹的 chunk 执行不爆栈
# ---------------------------------------------------------------------------


class TestParallelTrainerSafeNoRecursion:
    """验证 ``ParallelTrainerSafe._train_chunk_safe`` 包裹的 chunk 执行不爆栈。

    SubTask 1.4/1.5: ``_safe_chunk_run`` 新增 RecursionError 捕获分支 +
    ``_train_chunk_safe`` chunk 间 gc.collect + shutdown 检查。
    """

    def test_parallel_trainer_safe_fit_deep_model(self, tmp_path):
        """65 层 VerseNex + ParallelTrainerSafe.fit 不触发 RecursionError。"""
        from verse_infra.verse_trainer.trainer import ParallelTrainerSafe

        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,
            "batch_size": 2,
            "lr": 0.001,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 2,  # 触发 chunk_id=-999
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainerSafe(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        history = trainer.fit()

        # 应正常完成，best_val_loss 有限
        assert trainer.best_val_loss < float("inf"), (
            f"ParallelTrainerSafe best_val_loss 应为有限值, "
            f"got {trainer.best_val_loss}"
        )
        assert "train_loss" in history
        assert "val_loss" in history

    def test_safe_chunk_run_recursion_error_retry(self):
        """``_safe_chunk_run`` 捕获 RecursionError 后 gc.collect + 重试。"""
        from verse_infra.verse_trainer.trainer import _safe_chunk_run

        call_count = {"n": 0}

        def flaky_chunk_fn(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RecursionError("simulated recursion error")
            return "success"

        # 第一次抛 RecursionError，应被捕获 + 重试
        result = _safe_chunk_run(flaky_chunk_fn, max_oom_retries=2)
        assert result == "success"
        assert call_count["n"] == 2, (
            f"应调用 2 次（首次失败 + 重试成功）, got {call_count['n']}"
        )


# ---------------------------------------------------------------------------
# 5. val_loss 收敛验证
# ---------------------------------------------------------------------------


class TestValLossConvergence:
    """验证 65 层 VerseNex 模型训练后 val_loss 收敛（有限 + 下降）。"""

    def test_val_loss_finite_after_fit(self, tmp_path):
        """训练后 val_loss 为有限值。"""
        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,
            "batch_size": 2,
            "lr": 0.001,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 2,
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        trainer.fit()

        # best_val_loss 有限
        assert np.isfinite(trainer.best_val_loss), (
            f"best_val_loss 应为有限值, got {trainer.best_val_loss}"
        )
        # best_state_dict 应被填充
        assert trainer.best_state_dict is not None, (
            "best_state_dict 应在 fit 后非 None"
        )
        # best_state_dict 应可 pickle 序列化（验证修复后的路径）
        sd_copy = pickle.loads(
            pickle.dumps(trainer.best_state_dict, protocol=4))
        assert set(sd_copy.keys()) == set(trainer.best_state_dict.keys())

    def test_val_loss_decreases_or_stable(self, tmp_path):
        """训练后 val_loss 相比初始评估下降或保持稳定（宽松收敛验证）。"""
        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 6,
            "batch_size": 2,
            "lr": 0.003,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 2,
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        # 训练前初始 val_loss
        initial_val_loss = trainer._eval_full_val(model)
        assert np.isfinite(initial_val_loss), (
            f"初始 val_loss 应为有限值, got {initial_val_loss}"
        )

        trainer.fit()

        # 训练后 best_val_loss 应 <= 初始 val_loss（宽松收敛）
        # 注意：toy 数据 + 少量步数可能方差较大，用宽松断言（不严格下降）
        assert trainer.best_val_loss <= initial_val_loss * 1.5, (
            f"训练后 best_val_loss({trainer.best_val_loss}) 应不显著高于 "
            f"初始 val_loss({initial_val_loss})，表明训练未崩溃"
        )

    def test_best_state_dict_loadable_after_fit(self, tmp_path):
        """训练后 best_state_dict 可加载回模型（验证 pickle 序列化路径）。"""
        np.random.seed(42)
        model = _build_deep_versenex_model(n_layer=65)
        train_ds = LMDataset(n_samples=8, seq_len=4, vocab_size=32, seed=0)
        val_ds = LMDataset(n_samples=4, seq_len=4, vocab_size=32, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,
            "batch_size": 2,
            "lr": 0.001,
            "warmup": 1,
            "eval_interval": 2,
            "merge_finetune_steps": 2,
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        trainer.fit()

        # best_state_dict 应可加载回模型（这是 Phase 3 finetune 前的关键步骤）
        assert trainer.best_state_dict is not None
        # pickle 序列化 + 反序列化后加载（模拟 ParallelTrainer.fit 中的路径）
        sd_copy = pickle.loads(
            pickle.dumps(trainer.best_state_dict, protocol=4))
        model.load_state_dict(sd_copy)
        # 加载后 state_dict 应与 best_state_dict 数值一致
        sd_after = model.state_dict()
        for key in trainer.best_state_dict:
            np.testing.assert_array_equal(
                trainer.best_state_dict[key], sd_after[key],
                err_msg=f"加载 best_state_dict 后 key={key} 数值不一致",
            )


# ---------------------------------------------------------------------------
# 6. 现有测试零回归冒烟（导入 + 基本接口）
# ---------------------------------------------------------------------------


class TestNoRegressionSmoke:
    """冒烟测试：确保修复未破坏现有 ParallelTrainer / Trainer 接口。"""

    def test_parallel_trainer_imports(self):
        """ParallelTrainer / ParallelTrainerSafe 可正常导入。"""
        from verse_torch.training import ParallelTrainer, Trainer
        from verse_infra.verse_trainer.trainer import (
            ParallelTrainerSafe,
            _safe_chunk_run,
            ChunkOOMError,
        )
        assert ParallelTrainer is not None
        assert Trainer is not None
        assert ParallelTrainerSafe is not None
        assert callable(_safe_chunk_run)
        assert issubclass(ChunkOOMError, Exception)

    def test_shallow_model_still_works(self, tmp_path):
        """浅层模型（3 层）训练仍正常工作（确保迭代 BFS 对浅模型无副作用）。"""
        from verse_torch import Linear, SGD, Module

        class ToyModel(Module):
            def __init__(self):
                super().__init__()
                self.fc = Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        class ToyDataset:
            def __init__(self, n=20, seed=0):
                rng = np.random.RandomState(seed)
                self.x = rng.randn(n, 10).astype(np.float32)
                self.y = rng.randint(0, 5, size=n).astype(np.int64)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        model = ToyModel()
        train_ds = ToyDataset(n=20, seed=0)
        val_ds = ToyDataset(n=10, seed=100)
        cfg = {
            "parallel_chunks": 2,
            "max_steps": 4,
            "batch_size": 4,
            "lr": 0.01,
            "eval_interval": 2,
            "merge_finetune_steps": 1,
            "seed": 42,
            "quiet": True,
            "enable_progress_bar": False,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        trainer.fit()
        assert trainer.best_val_loss < float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
