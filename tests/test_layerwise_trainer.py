"""Part4K2 Task 4: 智能分区训练（LayerWiseTrainer）单元测试。

覆盖：
1. LayerWiseTrainer 初始化（有 .blocks 属性的模型）
2. 层分组正确（partition_size=2，6 层 → 3 组）
3. 卸载/加载往返（卸载后加载，参数数值一致）
4. 训练一组后参数更新（block 参数发生变化）
5. 合并分片后模型完整（所有 block 参数恢复）
6. 内存监控触发卸载（mock get_memory_info 返回高值）
7. 统一实体（训练前后模型对象 id 不变）
8. 小模型端到端训练（4 层 × partition_size=2，训练 10 步，验证 loss 下降）
9. CLI --partition-training 选项（argparse 解析正确）

运行方式：
    cd /workspace && python -m pytest tests/test_layerwise_trainer.py -x -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# PYTHONPATH 适配：让 tests/ 能 import verse_torch / verse_nex / verse_infra
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verse_torch import Tensor, Linear, Embedding, RMSNorm, ModuleList, Module
from verse_torch.layerwise_trainer import LayerWiseTrainer
from verse_torch.training import cross_entropy_loss, BatchLoader


# ---------------------------------------------------------------------------
# Toy 模型：带 .blocks 属性（模拟 CometSparkNexLM 结构）
# ---------------------------------------------------------------------------


class ToyBlock(Module):
    """简单 transformer block：Pre-Norm + 残差。

    结构（模拟 VerseNexBlock）：
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Linear(dim, dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ToyBlockModel(Module):
    """带 .blocks 的语言模型（模拟 CometSparkNexLM）。

    forward(idx) → logits (B, T, vocab)
    """

    def __init__(self, vocab_size: int = 20, dim: int = 8, n_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tok_emb = Embedding(vocab_size, dim)
        self.blocks = ModuleList([ToyBlock(dim) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.head = Linear(dim, vocab_size, bias=False)

    def forward(self, idx) -> Tensor:
        x = self.tok_emb(idx)  # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)  # (B, T, V)
        return logits


# ---------------------------------------------------------------------------
# Toy 数据集
# ---------------------------------------------------------------------------


class ToySeqDataset:
    """简单序列数据集：每个样本是随机 token 序列。

    返回 (x, y) 元组：
    - x = tokens[:-1]  (T-1,)  输入
    - y = tokens[1:]   (T-1,)  目标（shift by 1）
    """

    def __init__(self, n=40, seq_len=8, vocab_size=20, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokens = rng.randint(0, vocab_size, size=(n, seq_len)).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        toks = self.tokens[i]
        x = toks[:-1]  # (T-1,)
        y = toks[1:]   # (T-1,)
        return x, y


def _make_loader(vocab_size=20, dim=8, n_layers=4, n=40, seq_len=8,
                batch_size=4, seed=0):
    """构造 toy 模型 + train_loader + val_loader。"""
    model = ToyBlockModel(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
    train_ds = ToySeqDataset(n=n, seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    val_ds = ToySeqDataset(n=n // 2, seq_len=seq_len, vocab_size=vocab_size,
                           seed=seed + 100)
    train_loader = BatchLoader(train_ds, batch_size=batch_size, shuffle=True,
                               seed=seed)
    val_loader = BatchLoader(val_ds, batch_size=batch_size, shuffle=False)
    return model, train_loader, val_loader


# ===========================================================================
# 测试 1：LayerWiseTrainer 初始化
# ===========================================================================


class TestLayerWiseTrainerInit:
    """LayerWiseTrainer 初始化与基本属性。"""

    def test_init_with_blocks_model(self, tmp_path):
        """有 .blocks 属性的模型应正常初始化。"""
        model, train_loader, _ = _make_loader(n_layers=4)
        trainer = LayerWiseTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
        )
        assert trainer.model is model
        assert trainer.partition_size == 2
        assert trainer.n_partitions == 2  # 4 层 / 2 = 2 组
        assert len(trainer.partitions) == 2
        assert trainer.partitions[0] == [0, 1]
        assert trainer.partitions[1] == [2, 3]
        assert trainer.best_val_loss == float("inf")
        assert os.path.isdir(str(tmp_path))
        trainer.cleanup()

    def test_init_without_blocks_raises(self, tmp_path):
        """无 .blocks 属性的模型应抛 ValueError。"""
        class NoBlocksModel(Module):
            def __init__(self):
                super().__init__()
                self.fc = Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        model = NoBlocksModel()
        with pytest.raises(ValueError, match="blocks"):
            LayerWiseTrainer(
                model=model, config={"lr": 1e-3},
                offload_dir=str(tmp_path),
            )


# ===========================================================================
# 测试 2：层分组正确
# ===========================================================================


class TestPartitionLayers:
    """层分组逻辑正确性。"""

    def test_partition_size_2_six_layers_three_groups(self, tmp_path):
        """partition_size=2，6 层 → 3 组 [[0,1],[2,3],[4,5]]。"""
        model, _, _ = _make_loader(n_layers=6)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        assert trainer.n_partitions == 3
        assert trainer.partitions == [[0, 1], [2, 3], [4, 5]]
        trainer.cleanup()

    def test_partition_size_3_four_layers_two_groups(self, tmp_path):
        """partition_size=3，4 层 → 2 组 [[0,1,2],[3]]（不整除时余数成末组）。"""
        model, _, _ = _make_loader(n_layers=4)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=3, offload_dir=str(tmp_path),
        )
        assert trainer.n_partitions == 2
        assert trainer.partitions == [[0, 1, 2], [3]]
        trainer.cleanup()

    def test_partition_size_larger_than_layers(self, tmp_path):
        """partition_size=10，4 层 → 1 组 [[0,1,2,3]]。"""
        model, _, _ = _make_loader(n_layers=4)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=10, offload_dir=str(tmp_path),
        )
        assert trainer.n_partitions == 1
        assert trainer.partitions == [[0, 1, 2, 3]]
        trainer.cleanup()


# ===========================================================================
# 测试 3：卸载/加载往返（无损）
# ===========================================================================


class TestOffloadLoadRoundtrip:
    """卸载到硬盘再加载回来，参数数值应完全一致。"""

    def test_offload_then_load_identical(self, tmp_path):
        """卸载 partition 0 后再加载，blocks.0/1 参数与原始一致。"""
        model, _, _ = _make_loader(n_layers=4, seed=42)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        # 记录原始 state_dict（仅 partition 0 的 block 参数）
        orig_sd = trainer._get_partition_state(0)
        assert len(orig_sd) > 0
        # 深拷贝原始值（防止引用被修改）
        orig_values = {k: v.copy() for k, v in orig_sd.items()}

        # 卸载 partition 0 到硬盘
        path = trainer._offload_partition(0)
        assert os.path.exists(path)
        assert 0 in trainer._offloaded

        # 加载回来
        loaded_sd = trainer._load_partition(0)
        assert set(loaded_sd.keys()) == set(orig_sd.keys())

        # 数值应完全一致（无损往返）
        for k in orig_values:
            np.testing.assert_array_equal(loaded_sd[k], orig_values[k])
        trainer.cleanup()

    def test_offload_creates_vn_file(self, tmp_path):
        """卸载应生成 .vn 文件。"""
        model, _, _ = _make_loader(n_layers=4, seed=1)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        path = trainer._offload_partition(1)
        assert path.endswith(".vn")
        assert os.path.exists(path)
        trainer.cleanup()


# ===========================================================================
# 测试 4：训练一组后参数更新
# ===========================================================================


class TestTrainPartitionUpdates:
    """训练一组 partition 后，该组 block 参数应发生变化。"""

    def test_train_partition_0_updates_blocks(self, tmp_path):
        """训练 partition 0 后，blocks.0/1 参数变化，blocks.2/3 不变。"""
        model, train_loader, _ = _make_loader(n_layers=4, seed=7)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        # 记录训练前参数
        sd_before = model.state_dict()
        block0_before = sd_before["blocks.0.attn.weight"].copy()
        block2_before = sd_before["blocks.2.attn.weight"].copy()

        # 训练 partition 0（3 步）
        trainer._train_partition(0, train_loader, max_steps=3)
        assert len(trainer.train_losses) == 3
        assert 0 in trainer._trained

        # 训练后参数
        sd_after = model.state_dict()
        block0_after = sd_after["blocks.0.attn.weight"]
        block2_after = sd_after["blocks.2.attn.weight"]

        # partition 0 的 block 0 参数应变化（被训练）
        assert not np.allclose(block0_before, block0_after), \
            "训练后 block 0 参数应发生变化"
        # partition 1 的 block 2 参数应不变（被冻结）
        np.testing.assert_array_equal(block2_before, block2_after)
        trainer.cleanup()

    def test_set_trainable_freezes_other_partitions(self, tmp_path):
        """_set_trainable 应冻结非当前组的 block 参数。"""
        model, _, _ = _make_loader(n_layers=4)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        trainer._set_trainable(0)
        for name, p in model.named_parameters():
            if name.startswith("blocks.0.") or name.startswith("blocks.1."):
                assert p.requires_grad is True, f"{name} 应可训练"
            elif name.startswith("blocks.2.") or name.startswith("blocks.3."):
                assert p.requires_grad is False, f"{name} 应被冻结"
            else:
                # tok_emb / norm / head 始终可训练
                assert p.requires_grad is True, f"{name} 应可训练"
        trainer.cleanup()


# ===========================================================================
# 测试 5：合并分片后模型完整
# ===========================================================================


class TestMergePartitions:
    """合并所有分片后，模型应恢复完整状态。"""

    def test_merge_restores_all_blocks(self, tmp_path):
        """训练+卸载所有组后合并，所有 block 参数恢复到训练后值。"""
        model, train_loader, _ = _make_loader(n_layers=4, seed=3)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        # 训练所有组（每组 2 步）
        for i in range(trainer.n_partitions):
            trainer._train_partition(i, train_loader, max_steps=2)
            trainer._offload_partition(i)

        # 合并前记录硬盘上的值
        assert len(trainer._offloaded) == 2
        # 合并
        trainer._merge_partitions()

        # 合并后模型应包含所有 block 的参数
        sd = model.state_dict()
        for i in range(4):
            assert f"blocks.{i}.attn.weight" in sd
            assert f"blocks.{i}.ffn.weight" in sd
        # requires_grad 应被恢复
        for name, p in model.named_parameters():
            assert name in trainer._orig_requires_grad
        trainer.cleanup()

    def test_merge_after_offload_values_match(self, tmp_path):
        """卸载后模型参数（未置零时）应与硬盘分片一致；合并后保持一致。"""
        model, train_loader, _ = _make_loader(n_layers=4, seed=5)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        # 训练 partition 0 并卸载
        trainer._train_partition(0, train_loader, max_steps=2)
        sd_after_train = {k: v.copy() for k, v in trainer._get_partition_state(0).items()}
        trainer._offload_partition(0)

        # 合并（加载回内存）
        trainer._load_partition(0)
        sd_after_merge = trainer._get_partition_state(0)
        for k in sd_after_train:
            np.testing.assert_array_equal(sd_after_merge[k], sd_after_train[k])
        trainer.cleanup()


# ===========================================================================
# 测试 6：内存监控触发卸载
# ===========================================================================


class TestMemoryMonitor:
    """内存监控：超阈值时触发卸载已训练的非当前组。"""

    def test_check_memory_triggers_offload(self, tmp_path, monkeypatch):
        """mock get_memory_info 返回高内存使用，_check_memory 应触发卸载。"""
        model, train_loader, _ = _make_loader(n_layers=4, seed=9)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
            memory_threshold_mb=10,  # 低阈值，容易触发
        )
        # 训练 partition 0（标记为已训练）
        trainer._current_partition = 1  # 当前在训练 partition 1
        trainer._trained.add(0)
        trainer._offloaded.discard(0)  # 确保 0 尚未卸载

        # mock get_memory_info 返回高内存（100MB > 10MB 阈值）
        fake_info = {"total": 1000 * 1024 * 1024, "used": 100 * 1024 * 1024,
                     "free": 900 * 1024 * 1024}
        import verse_torch.layerwise_trainer as lw_mod
        monkeypatch.setattr(lw_mod, "get_memory_info",
                            lambda device="cpu": fake_info)

        triggered = trainer._check_memory()
        assert triggered is True
        assert trainer._memory_high_triggered is True
        # partition 0 应已被卸载
        assert 0 in trainer._offloaded
        # 卸载文件应存在
        path = os.path.join(str(tmp_path), "partition_0.vn")
        assert os.path.exists(path)
        trainer.cleanup()

    def test_check_memory_no_trigger_below_threshold(self, tmp_path, monkeypatch):
        """内存使用低于阈值时不应触发卸载。"""
        model, _, _ = _make_loader(n_layers=4)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
            memory_threshold_mb=10000,  # 高阈值，不会触发
        )
        trainer._current_partition = 1
        trainer._trained.add(0)

        fake_info = {"total": 1000 * 1024 * 1024, "used": 50 * 1024 * 1024,
                     "free": 950 * 1024 * 1024}
        import verse_torch.layerwise_trainer as lw_mod
        monkeypatch.setattr(lw_mod, "get_memory_info",
                            lambda device="cpu": fake_info)

        triggered = trainer._check_memory()
        assert triggered is False
        assert 0 not in trainer._offloaded
        trainer.cleanup()


# ===========================================================================
# 测试 7：统一实体（训练前后模型对象不变）
# ===========================================================================


class TestUnifiedEntity:
    """训练前后模型对象 id 应保持不变（统一实体）。"""

    def test_model_object_unchanged_after_fit(self, tmp_path):
        """fit 完成后，trainer.model 与原始 model 是同一对象。"""
        model, train_loader, val_loader = _make_loader(n_layers=4, seed=11)
        model_id_before = id(model)

        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0,
                                 "eval_interval": None, "finetune_steps": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        trainer.fit(train_loader, val_loader, max_steps=8)

        # 模型对象应不变
        assert id(trainer.model) == model_id_before
        assert trainer.model is model
        trainer.cleanup()

    def test_model_object_unchanged_during_partition_training(self, tmp_path):
        """逐组训练过程中，模型对象始终不变。"""
        model, train_loader, _ = _make_loader(n_layers=4, seed=13)
        model_id_before = id(model)

        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        for i in range(trainer.n_partitions):
            trainer._train_partition(i, train_loader, max_steps=2)
            trainer._offload_partition(i)
            assert id(trainer.model) == model_id_before

        trainer._merge_partitions()
        assert id(trainer.model) == model_id_before
        trainer.cleanup()


# ===========================================================================
# 测试 8：小模型端到端训练（loss 下降）
# ===========================================================================


class TestEndToEndTraining:
    """小模型端到端：4 层 × partition_size=2，训练 10 步，loss 应下降。"""

    def test_loss_decreases_after_fit(self, tmp_path):
        """fit 完成后，最终 loss 应低于初始 loss。"""
        model, train_loader, val_loader = _make_loader(
            vocab_size=20, dim=8, n_layers=4, n=40, seq_len=8,
            batch_size=4, seed=21,
        )
        trainer = LayerWiseTrainer(
            model=model,
            config={
                "lr": 5e-3,
                "log_interval": 0,
                "eval_interval": None,
                "finetune_steps": 2,
                "label_smoothing": 0.0,
            },
            partition_size=2,
            offload_dir=str(tmp_path),
        )
        train_losses, val_losses = trainer.fit(
            train_loader, val_loader, max_steps=12,
        )
        # 应有训练 loss 记录
        assert len(train_losses) > 0
        # 最终 loss 应低于初始 loss（允许容差，训练应整体下降）
        n = len(train_losses)
        first_avg = float(np.mean(train_losses[:max(1, n // 4)]))
        last_avg = float(np.mean(train_losses[-max(1, n // 4):]))
        assert last_avg < first_avg, (
            f"loss 应下降：first_avg={first_avg:.4f} >= last_avg={last_avg:.4f}"
        )
        trainer.cleanup()

    def test_fit_returns_losses_lists(self, tmp_path):
        """fit 应返回 (train_losses, val_losses) 两个列表。"""
        model, train_loader, val_loader = _make_loader(n_layers=4, seed=23)
        trainer = LayerWiseTrainer(
            model=model, config={"lr": 1e-3, "log_interval": 0,
                                 "eval_interval": None, "finetune_steps": 0},
            partition_size=2, offload_dir=str(tmp_path),
        )
        train_losses, val_losses = trainer.fit(train_loader, val_loader,
                                               max_steps=8)
        assert isinstance(train_losses, list)
        assert isinstance(val_losses, list)
        assert len(train_losses) > 0
        trainer.cleanup()


# ===========================================================================
# 测试 9：CLI --partition-training 选项
# ===========================================================================


class TestCLIPartitionTraining:
    """CLI --partition-training / --partition-size / --offload-dir 选项解析。"""

    def test_parser_has_partition_training_flag(self):
        """_build_train_parser 应包含 --partition-training 选项。"""
        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args(["--partition-training"])
        assert args.partition_training is True
        # 默认 partition_size
        assert args.partition_size == 2
        # 默认 offload_dir
        assert args.offload_dir is None

    def test_parser_partition_size_and_offload_dir(self):
        """--partition-size 与 --offload-dir 应正确解析。"""
        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args([
            "--partition-training",
            "--partition-size", "4",
            "--offload-dir", "/tmp/test_offload",
        ])
        assert args.partition_training is True
        assert args.partition_size == 4
        assert args.offload_dir == "/tmp/test_offload"

    def test_parser_default_no_partition_training(self):
        """不传 --partition-training 时默认为 False。"""
        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args([])
        assert args.partition_training is False
