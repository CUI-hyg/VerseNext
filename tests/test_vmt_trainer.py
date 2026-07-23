"""Part5K1 Task 8: VMT 完整智能分区训练（VMTTrainer）单元测试。

覆盖三档策略（unload / freeze / optimize）：
1. 三档分配：显式策略解析正确
2. auto 策略：自动分配覆盖全部层
3. freeze 反量化误差：freeze → unfreeze 后参数误差 ≤ 1e-3
4. optimize 前向：与原逐块前向数值一致（1e-3）
5. 统一实体：训练前后模型对象 id 不变
6. 端到端 fit：小模型 + 小数据 + fit(max_steps=5) 不报错，loss 有变化
7. 策略校验：非法档名 / 区间重叠抛 ValueError
8. LayerWiseTrainer 仍可用（向后兼容）

运行方式：
    cd /workspace && python -m pytest tests/test_vmt_trainer.py -x -q
"""

from __future__ import annotations

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
from verse_torch.layerwise_trainer import VMTTrainer, LayerWiseTrainer
from verse_torch.training import cross_entropy_loss, BatchLoader


# ---------------------------------------------------------------------------
# Toy 模型：带 .blocks 属性（模拟 CometSparkNexLM 结构）
# ---------------------------------------------------------------------------


class ToyBlock(Module):
    """简单 transformer block：Pre-Norm + 残差。"""

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
    """带 .blocks 的语言模型（模拟 CometSparkNexLM）。"""

    def __init__(self, vocab_size: int = 64, dim: int = 32, n_layers: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tok_emb = Embedding(vocab_size, dim)
        self.blocks = ModuleList([ToyBlock(dim) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.head = Linear(dim, vocab_size, bias=False)

    def forward(self, idx) -> Tensor:
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# Toy 数据集
# ---------------------------------------------------------------------------


class ToySeqDataset:
    """简单序列数据集：每个样本是随机 token 序列。"""

    def __init__(self, n=40, seq_len=8, vocab_size=64, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokens = rng.randint(0, vocab_size, size=(n, seq_len)).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        toks = self.tokens[i]
        x = toks[:-1]
        y = toks[1:]
        return x, y


def _make_loader(vocab_size=64, dim=32, n_layers=6, n=40, seq_len=8,
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
# 测试 1：三档分配（显式策略）
# ===========================================================================


class TestTierAssignmentExplicit:
    """显式策略解析：layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload。"""

    def test_explicit_strategy_parsed_correctly(self, tmp_path):
        """显式策略应解析为 {(0,2):freeze, (2,4):optimize, (4,6):unload}。"""
        model, _, _ = _make_loader(n_layers=6)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload",
        )
        ta = trainer._tier_assignments
        assert ta[(0, 2)] == "freeze"
        assert ta[(2, 4)] == "optimize"
        assert ta[(4, 6)] == "unload"
        assert len(ta) == 3
        trainer.cleanup()

    def test_partition_tier_mapping(self, tmp_path):
        """分区 tier 映射：partition 0→freeze, 1→optimize, 2→unload。"""
        model, _, _ = _make_loader(n_layers=6)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload",
        )
        assert trainer._partition_tier(0) == "freeze"
        assert trainer._partition_tier(1) == "optimize"
        assert trainer._partition_tier(2) == "unload"
        trainer.cleanup()

    def test_open_ended_range(self, tmp_path):
        """layers[4:]=unload 末尾省略表示到末层。"""
        model, _, _ = _make_loader(n_layers=6)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:]=unload",
        )
        assert trainer._tier_assignments[(4, 6)] == "unload"
        trainer.cleanup()


# ===========================================================================
# 测试 2：auto 策略
# ===========================================================================


class TestAutoStrategy:
    """auto 策略：自动分配三档，覆盖全部层。"""

    def test_auto_covers_all_layers(self, tmp_path):
        """auto 策略应覆盖 [0, n_layer) 全部层。"""
        model, _, _ = _make_loader(n_layers=6)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="auto",
        )
        ta = trainer._tier_assignments
        # 验证区间并集覆盖 [0, 6)
        covered = []
        for (s, e) in ta:
            covered.extend(range(s, e))
        assert sorted(covered) == list(range(6))
        # 验证档名合法
        for tier in ta.values():
            assert tier in ("freeze", "optimize", "unload")
        trainer.cleanup()

    def test_auto_six_layers_three_tiers(self, tmp_path):
        """auto 策略 6 层：前 1/3 freeze, 中 1/3 optimize, 后 1/3 unload。"""
        model, _, _ = _make_loader(n_layers=6)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="auto",
        )
        ta = trainer._tier_assignments
        # n=6, n3=2: freeze[0:2], optimize[2:4], unload[4:6]
        assert ta[(0, 2)] == "freeze"
        assert ta[(2, 4)] == "optimize"
        assert ta[(4, 6)] == "unload"
        trainer.cleanup()

    def test_auto_small_model_all_optimize(self, tmp_path):
        """层数 ≤ 2 时 auto 全部分配为 optimize。"""
        model, _, _ = _make_loader(n_layers=2)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=1,
            offload_dir=str(tmp_path),
            vmt_strategy="auto",
        )
        ta = trainer._tier_assignments
        assert ta[(0, 2)] == "optimize"
        trainer.cleanup()


# ===========================================================================
# 测试 3：freeze 反量化误差
# ===========================================================================


class TestFreezeUnfreeze:
    """freeze → unfreeze 后参数误差 ≤ 1e-3。"""

    def test_freeze_unfreeze_param_error(self, tmp_path):
        """对一组层 freeze 后 unfreeze，参数误差 ≤ 1e-3。"""
        model, _, _ = _make_loader(n_layers=4, seed=42)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize",
        )
        # 记录原始参数
        orig_sd = {k: v.copy() for k, v in model.state_dict().items()}

        # freeze blocks[0:2]
        trainer._freeze_partition(0, 2)
        # 验证 requires_grad=False
        for name, p in model.named_parameters():
            if name.startswith("blocks.0.") or name.startswith("blocks.1."):
                assert p.requires_grad is False, f"{name} 应被冻结"

        # unfreeze
        trainer._unfreeze_partition(0, 2)

        # 验证参数恢复（误差 ≤ 1e-3）
        after_sd = model.state_dict()
        for k, v_orig in orig_sd.items():
            if k.startswith("blocks.0.") or k.startswith("blocks.1."):
                v_after = after_sd[k]
                max_err = float(np.max(np.abs(v_after - v_orig)))
                assert max_err <= 1e-3, (
                    f"{k} 反量化误差 {max_err} > 1e-3"
                )
        trainer.cleanup()

    def test_freeze_quantizes_weights(self, tmp_path):
        """freeze 后权重应被 INT4 量化（与原始不完全一致）。"""
        model, _, _ = _make_loader(n_layers=4, seed=7)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize",
        )
        orig_w = model.blocks[0].attn.weight.data.copy()

        trainer._freeze_partition(0, 2)
        frozen_w = model.blocks[0].attn.weight.data.copy()
        # INT4 量化后权重应与原始有差异（除非权重恰好全在量化点上）
        assert not np.allclose(orig_w, frozen_w), \
            "freeze 后权重应被 INT4 量化（与原始有差异）"

        trainer._unfreeze_partition(0, 2)
        restored_w = model.blocks[0].attn.weight.data.copy()
        # unfreeze 后应精确恢复
        np.testing.assert_array_equal(orig_w, restored_w)
        trainer.cleanup()

    def test_freeze_backup_cleanup(self, tmp_path):
        """unfreeze 后 _freeze_backups 应清除该区间。"""
        model, _, _ = _make_loader(n_layers=4, seed=3)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize",
        )
        trainer._freeze_partition(0, 2)
        assert (0, 2) in trainer._frozen
        assert (0, 2) in trainer._freeze_backups
        trainer._unfreeze_partition(0, 2)
        assert (0, 2) not in trainer._frozen
        assert (0, 2) not in trainer._freeze_backups
        trainer.cleanup()


# ===========================================================================
# 测试 4：optimize 前向数值一致
# ===========================================================================


class TestOptimizeForward:
    """optimize 档前向与原逐块前向数值一致。"""

    def test_optimize_forward_matches_sequential(self, tmp_path):
        """_optimize_partition_forward 与逐块前向数值一致（1e-5）。"""
        model, _, _ = _make_loader(n_layers=6, seed=11)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload",
        )
        # 构造输入
        rng = np.random.RandomState(123)
        idx = rng.randint(0, 64, size=(2, 4)).astype(np.int64)
        idx_t = Tensor(idx)

        # 原前向：tok_emb → 逐块 → norm → head
        emb = model.tok_emb(idx_t)
        h_ref = emb
        for block in model.blocks:
            h_ref = block(h_ref)
        h_ref = model.norm(h_ref)
        logits_ref = model.head(h_ref)

        # optimize 前向：tok_emb → _optimize_partition_forward(全部层) → norm → head
        emb2 = model.tok_emb(idx_t)
        n = len(model.blocks)
        h_opt = trainer._optimize_partition_forward(emb2, 0, n)
        h_opt = model.norm(h_opt)
        logits_opt = model.head(h_opt)

        # 数值应一致（回退到逐块前向，float32 严格相等）
        np.testing.assert_allclose(
            logits_opt.data, logits_ref.data, atol=1e-5,
            err_msg="optimize 前向与原前向数值不一致"
        )
        trainer.cleanup()

    def test_full_optimize_forward_matches_model_forward(self, tmp_path):
        """_full_optimize_forward 与 model.forward 数值一致（1e-5）。"""
        model, _, _ = _make_loader(n_layers=6, seed=13)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload",
        )
        rng = np.random.RandomState(456)
        idx = rng.randint(0, 64, size=(3, 5)).astype(np.int64)

        logits_ref = model(idx)
        logits_opt = trainer._full_optimize_forward(idx)

        np.testing.assert_allclose(
            logits_opt.data, logits_ref.data, atol=1e-5,
            err_msg="_full_optimize_forward 与 model.forward 数值不一致"
        )
        trainer.cleanup()


# ===========================================================================
# 测试 5：统一实体（训练前后模型对象不变）
# ===========================================================================


class TestUnifiedEntity:
    """训练前后模型对象 id 应保持不变。"""

    def test_model_id_unchanged_after_fit(self, tmp_path):
        """fit 完成后，trainer.model 与原始 model 是同一对象。"""
        model, train_loader, val_loader = _make_loader(n_layers=6, seed=21)
        model_id_before = id(model)

        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0,
                    "eval_interval": None, "finetune_steps": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload",
        )
        trainer.fit(train_loader, val_loader, max_steps=6)

        assert id(trainer.model) == model_id_before
        assert trainer.model is model
        trainer.cleanup()

    def test_model_id_unchanged_during_freeze_unfreeze(self, tmp_path):
        """freeze / unfreeze 过程中模型对象不变。"""
        model, _, _ = _make_loader(n_layers=4, seed=23)
        model_id_before = id(model)

        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize",
        )
        trainer._freeze_partition(0, 2)
        assert id(trainer.model) == model_id_before
        trainer._unfreeze_partition(0, 2)
        assert id(trainer.model) == model_id_before
        trainer.cleanup()


# ===========================================================================
# 测试 6：端到端 fit
# ===========================================================================


class TestEndToEndFit:
    """小模型端到端 VMT 训练。"""

    def test_fit_runs_without_error(self, tmp_path):
        """VMTTrainer.fit(max_steps=6) 不报错，返回 loss 列表。"""
        model, train_loader, val_loader = _make_loader(
            vocab_size=64, dim=32, n_layers=6, n=40, seq_len=8,
            batch_size=4, seed=31,
        )
        trainer = VMTTrainer(
            model=model,
            config={
                "lr": 5e-3,
                "log_interval": 0,
                "eval_interval": None,
                "finetune_steps": 0,
            },
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize, layers[4:6]=unload",
        )
        train_losses, val_losses = trainer.fit(train_loader, val_loader, max_steps=6)
        assert isinstance(train_losses, list)
        assert len(train_losses) > 0
        # freeze 档不训练，optimize(2步) + unload(2步) + optimize(2步) ... 取决于分配
        # n_active=2, steps_per=3, 共 6 步
        assert len(train_losses) == 6
        trainer.cleanup()

    def test_fit_loss_changes(self, tmp_path):
        """fit 后 loss 应有变化（非全部相同）。"""
        model, train_loader, val_loader = _make_loader(
            vocab_size=64, dim=32, n_layers=6, n=40, seq_len=8,
            batch_size=4, seed=33,
        )
        trainer = VMTTrainer(
            model=model,
            config={
                "lr": 1e-2,
                "log_interval": 0,
                "eval_interval": None,
                "finetune_steps": 0,
            },
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=optimize, layers[2:4]=optimize, layers[4:6]=unload",
        )
        train_losses, _ = trainer.fit(train_loader, val_loader, max_steps=8)
        # loss 不应全部相同（训练应有变化）
        assert len(set(round(l, 6) for l in train_losses)) > 1, \
            f"loss 应有变化，但全部为 {train_losses}"
        trainer.cleanup()

    def test_fit_all_optimize_strategy(self, tmp_path):
        """全部 optimize 档策略：fit 不报错。"""
        model, train_loader, val_loader = _make_loader(n_layers=4, seed=35)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0,
                    "eval_interval": None, "finetune_steps": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:4]=optimize",
        )
        train_losses, _ = trainer.fit(train_loader, val_loader, max_steps=4)
        assert len(train_losses) == 4
        trainer.cleanup()

    def test_fit_with_micro_batch(self, tmp_path):
        """optimize 档梯度累积：micro_batch_size < batch_size 时不报错。"""
        model, train_loader, val_loader = _make_loader(
            vocab_size=64, dim=32, n_layers=6, n=40, seq_len=8,
            batch_size=4, seed=37,
        )
        trainer = VMTTrainer(
            model=model,
            config={
                "lr": 1e-3,
                "log_interval": 0,
                "eval_interval": None,
                "finetune_steps": 0,
                "micro_batch_size": 2,  # batch=4, micro=2 → 累积 2 微批
            },
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="layers[0:2]=optimize, layers[2:4]=optimize, layers[4:6]=unload",
        )
        train_losses, _ = trainer.fit(train_loader, val_loader, max_steps=4)
        assert len(train_losses) > 0
        trainer.cleanup()


# ===========================================================================
# 测试 7：策略校验
# ===========================================================================


class TestStrategyValidation:
    """策略校验：非法档名 / 区间重叠 / 不连续 / 未覆盖 抛 ValueError。"""

    def test_invalid_tier_name_raises(self, tmp_path):
        """非法档名应抛 ValueError。"""
        model, _, _ = _make_loader(n_layers=6)
        with pytest.raises(ValueError, match="非法 VMT 档名"):
            VMTTrainer(
                model=model,
                config={"lr": 1e-3, "log_interval": 0},
                partition_size=2,
                offload_dir=str(tmp_path),
                vmt_strategy="layers[0:2]=bad, layers[2:6]=optimize",
            )

    def test_overlapping_ranges_raises(self, tmp_path):
        """区间重叠应抛 ValueError。"""
        model, _, _ = _make_loader(n_layers=6)
        with pytest.raises(ValueError, match="重叠"):
            VMTTrainer(
                model=model,
                config={"lr": 1e-3, "log_interval": 0},
                partition_size=2,
                offload_dir=str(tmp_path),
                vmt_strategy="layers[0:4]=freeze, layers[2:6]=optimize",
            )

    def test_non_contiguous_ranges_raises(self, tmp_path):
        """区间不连续应抛 ValueError。"""
        model, _, _ = _make_loader(n_layers=6)
        with pytest.raises(ValueError, match="不连续"):
            VMTTrainer(
                model=model,
                config={"lr": 1e-3, "log_interval": 0},
                partition_size=2,
                offload_dir=str(tmp_path),
                vmt_strategy="layers[0:2]=freeze, layers[3:6]=optimize",
            )

    def test_incomplete_coverage_raises(self, tmp_path):
        """区间未覆盖全部层应抛 ValueError。"""
        model, _, _ = _make_loader(n_layers=6)
        with pytest.raises(ValueError, match="未覆盖"):
            VMTTrainer(
                model=model,
                config={"lr": 1e-3, "log_interval": 0},
                partition_size=2,
                offload_dir=str(tmp_path),
                vmt_strategy="layers[0:2]=freeze, layers[2:4]=optimize",
            )

    def test_unparseable_strategy_raises(self, tmp_path):
        """无法解析的策略字符串应抛 ValueError。"""
        model, _, _ = _make_loader(n_layers=6)
        with pytest.raises(ValueError, match="无法解析"):
            VMTTrainer(
                model=model,
                config={"lr": 1e-3, "log_interval": 0},
                partition_size=2,
                offload_dir=str(tmp_path),
                vmt_strategy="this is not a valid strategy",
            )

    def test_invalid_range_raises(self, tmp_path):
        """层范围非法（start >= end）应抛 ValueError。"""
        model, _, _ = _make_loader(n_layers=6)
        with pytest.raises(ValueError, match="范围非法"):
            VMTTrainer(
                model=model,
                config={"lr": 1e-3, "log_interval": 0},
                partition_size=2,
                offload_dir=str(tmp_path),
                vmt_strategy="layers[3:2]=freeze, layers[0:3]=optimize, layers[3:6]=unload",
            )


# ===========================================================================
# 测试 8：LayerWiseTrainer 仍可用（向后兼容）
# ===========================================================================


class TestLayerWiseTrainerBackwardCompat:
    """LayerWiseTrainer 仍可独立使用（向后兼容）。"""

    def test_layerwise_trainer_still_works(self, tmp_path):
        """LayerWiseTrainer(model, config) 不报错，可正常初始化。"""
        model, _, _ = _make_loader(n_layers=4)
        trainer = LayerWiseTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
        )
        assert trainer.partition_size == 2
        assert trainer.n_partitions == 2
        # LayerWiseTrainer 不应有 VMT 专属属性
        assert not hasattr(trainer, "vmt_strategy")
        assert not hasattr(trainer, "_tier_assignments")
        trainer.cleanup()

    def test_layerwise_trainer_fit_still_works(self, tmp_path):
        """LayerWiseTrainer.fit 不报错（向后兼容）。"""
        model, train_loader, val_loader = _make_loader(n_layers=4, seed=41)
        trainer = LayerWiseTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0,
                    "eval_interval": None, "finetune_steps": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
        )
        train_losses, _ = trainer.fit(train_loader, val_loader, max_steps=4)
        assert len(train_losses) > 0
        trainer.cleanup()

    def test_vmt_is_subclass_of_layerwise(self, tmp_path):
        """VMTTrainer 是 LayerWiseTrainer 的子类。"""
        assert issubclass(VMTTrainer, LayerWiseTrainer)
        model, _, _ = _make_loader(n_layers=4)
        trainer = VMTTrainer(
            model=model,
            config={"lr": 1e-3, "log_interval": 0},
            partition_size=2,
            offload_dir=str(tmp_path),
            vmt_strategy="auto",
        )
        assert isinstance(trainer, LayerWiseTrainer)
        trainer.cleanup()
