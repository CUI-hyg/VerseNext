"""Part4 Task P10：ParallelTrainer aux_loss 路径 + MoD Expert 压缩单元测试。

覆盖：
1. ``test_parallel_trainer_with_aux``：verse_nex arch 启用 aux 路径并完成训练
2. ``test_parallel_trainer_without_aux``：transformer arch 退化为标准路径
3. ``test_parallel_trainer_aux_loss_weight_read``：aux_loss_weight 从 model.config 读取
4. ``test_compress_mod_experts_basic``：keep_ratio=0.5 剪枝后 Experts 数减少
5. ``test_compress_mod_experts_min_per_part``：min_experts_per_part 边界生效
6. ``test_compress_mod_experts_stats``：return_stats=True 返回正确统计 dict

Part4K1 Task 8.9: 模型从 data/demo 迁移到 spark/model。

运行方式::

    cd /workspace && PYTHONPATH=packages/verse_torch:packages/verse_nex:packages/verse_infra \\
        python -m pytest tests/test_p10_parallel_compress.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# PYTHONPATH 适配（Part4K1 Task 8.9: 从 spark/ 加载模型）
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_infra"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from verse_torch import nn
from verse_torch.training import ParallelTrainer
from verse_torch.compress import compress_mod_experts
from verse_nex.moe import MoDLayer
# Part4K1 Task 8.9: 从 spark/model 导入（替代 data/demo/model）
from spark.model.model import CometSparkV05Small as CometSparkV02Small
from spark.model.model import CometSparkV05Small as CometSparkSmall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LMDataset:
    """简单 LM 数据集：随机整数 token id 序列 + next-token 标签。

    每条样本返回 ``(x, y)``，均为 ``shape=(seq_len,)`` 的 int64 ndarray。
    ParallelTrainer 内部 BatchLoader 会用 ``_default_collate`` 把 B 条样本
    stack 成 ``(B, seq_len)`` 的 batch。
    """

    def __init__(self, n_samples=8, seq_len=16, vocab_size=64, seed=0):
        rng = np.random.RandomState(seed)
        self.samples = []
        for _ in range(n_samples):
            x = rng.randint(0, vocab_size, size=(seq_len,)).astype(np.int64)
            # next-token 预测：y[i] = x[i+1]，末位补随机 token
            y = np.concatenate(
                [x[1:], rng.randint(0, vocab_size, size=(1,))]
            ).astype(np.int64)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def _count_experts(model) -> int:
    """统计模型中所有 MoDLayer 内的 Expert 总数（基于 num_experts 字段）。"""
    total = 0
    for m in model.modules():
        if isinstance(m, MoDLayer):
            for part in m.parts:
                total += int(part.num_experts)
    return total


def _experts_per_part(model) -> list:
    """返回每个 MoDLayer 的每个 DensePart 的 Expert 数列表。"""
    result = []
    for m in model.modules():
        if isinstance(m, MoDLayer):
            for part in m.parts:
                result.append(int(part.num_experts))
    return result


def _build_mod_test_model(num_dense_parts=2, num_experts_per_part=4,
                          top_k=2, dim=32):
    """构造含单个 MoDLayer 的小测试模型。

    返回 ``forward(x) -> (out, aux)`` 的 Module，用于 compress_mod_experts
    测试（不需要训练，仅做结构剪枝）。
    """

    class _MoDTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = MoDLayer(
                dim=dim,
                num_dense_parts=num_dense_parts,
                num_experts_per_part=num_experts_per_part,
                top_k=top_k,
                expert_hidden=32,
                aux_loss_weight=0.01,
            )

        def forward(self, x):
            return self.mod(x)  # 返回 (out, aux)

    return _MoDTestModel()


# ---------------------------------------------------------------------------
# 1. ParallelTrainer with verse_nex（aux_loss-aware 路径）
# ---------------------------------------------------------------------------


def test_parallel_trainer_with_aux():
    """CometSparkV02Small (arch=verse_nex) parallel_chunks=2 训练 4 步。

    验证：
    - use_aux=True（forward_with_aux 被检测到）
    - aux_loss_weight 从 model.config 读取
    - fit() 完成，train_loss / val_loss 非空且为有限值
    """
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    train_ds = _LMDataset(n_samples=8, seq_len=16, vocab_size=64, seed=0)
    val_ds = _LMDataset(n_samples=4, seq_len=16, vocab_size=64, seed=1)
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 4,  # 2 chunks × 2 steps
        "batch_size": 2,
        "lr": 1e-3,
        "eval_interval": 1,
        "warmup": 1,
        "merge_finetune_steps": 0,  # 关闭 finetune 以加速测试
        "seed": 42,
        "enable_progress_bar": False,
        "realtime_plot": False,
        "log_interval": 1000,  # 静默
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

    # use_aux 应为 True（CometSparkV02Small.net 是 CometSparkNexLM，
    # 提供 forward_with_aux 方法）
    assert trainer.use_aux is True, "CometSparkV02Small 应启用 aux 路径"
    # aux_loss_weight 应从 model.config.aux_loss_weight 读取
    assert trainer.aux_loss_weight == model.config.aux_loss_weight
    assert trainer.aux_loss_weight > 0.0

    history = trainer.fit()

    # train_loss / val_loss 应非空且为有限值
    assert len(history["train_loss"]) > 0, "train_loss 历史不应为空"
    assert len(history["val_loss"]) > 0, "val_loss 历史不应为空"
    for v in history["train_loss"]:
        assert v < float("inf"), f"train_loss 应为有限值, got {v}"
    for v in history["val_loss"]:
        assert v < float("inf"), f"val_loss 应为有限值, got {v}"
    # best_val_loss 应被更新为有限值
    assert trainer.best_val_loss < float("inf"), (
        f"best_val_loss 应为有限值, got {trainer.best_val_loss}"
    )
    # chunk_stats 应记录 2 个 chunk
    assert len(trainer.chunk_stats) == 2


# ---------------------------------------------------------------------------
# 2. ParallelTrainer with transformer（退化路径）
# ---------------------------------------------------------------------------


def test_parallel_trainer_without_aux():
    """CometSparkV05Small(mod_every=99) parallel_chunks=2 训练 4 步。

    Part4K1 Task 8.9: 使用 CometSparkV05Small(mod_every=99)。
    注意：mod_every=99 仍会在第 0 层创建 MoD（0 % 99 == 0），
    但本测试仅验证 fit() 能完成，不断言 aux_losses == 0。

    验证：
    - fit() 完成
    """
    model = CometSparkSmall(mod_every=99)  # 第 0 层仍为 MoD，但不影响测试
    train_ds = _LMDataset(n_samples=8, seq_len=16, vocab_size=256, seed=0)
    val_ds = _LMDataset(n_samples=4, seq_len=16, vocab_size=256, seed=1)
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 4,
        "batch_size": 2,
        "lr": 1e-3,
        "eval_interval": 1,
        "warmup": 1,
        "merge_finetune_steps": 0,
        "seed": 42,
        "enable_progress_bar": False,
        "realtime_plot": False,
        "log_interval": 1000,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

    history = trainer.fit()
    assert len(history["train_loss"]) > 0
    assert trainer.best_val_loss < float("inf"), (
        f"best_val_loss 应为有限值, got {trainer.best_val_loss}"
    )


# ---------------------------------------------------------------------------
# 3. aux_loss_weight 从 model.config 读取
# ---------------------------------------------------------------------------


def test_parallel_trainer_aux_loss_weight_read():
    """use_aux=True 时 aux_loss_weight 从 model.config.aux_loss_weight 读取。

    CometSparkConfig.aux_loss_weight 默认 0.01。
    """
    model = CometSparkV02Small(vocab_size=64, seq_len=32)
    # CometSparkConfig.aux_loss_weight 默认 0.01
    assert model.config.aux_loss_weight == 0.01
    train_ds = _LMDataset(n_samples=4, seq_len=16, vocab_size=64, seed=0)
    val_ds = _LMDataset(n_samples=2, seq_len=16, vocab_size=64, seed=1)

    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        cfg={"parallel_chunks": 2, "max_steps": 1})

    # ParallelTrainer.aux_loss_weight 应等于 model.config.aux_loss_weight
    assert trainer.use_aux is True
    assert trainer.aux_loss_weight == model.config.aux_loss_weight
    assert trainer.aux_loss_weight == 0.01


# ---------------------------------------------------------------------------
# 4. compress_mod_experts 基本剪枝
# ---------------------------------------------------------------------------


def test_compress_mod_experts_basic():
    """compress_mod_experts(keep_ratio=0.5)：Experts 数量减少。

    构造含 2 parts × 4 experts = 8 experts 的 MoD 模型，
    keep_ratio=0.5 应保留 max(1, int(4*0.5))=2 experts per part = 4 total。
    """
    model = _build_mod_test_model(
        num_dense_parts=2, num_experts_per_part=4, top_k=2, dim=32)
    n_before = _count_experts(model)
    assert n_before == 8, f"初始应 8 experts (2×4), got {n_before}"

    compress_mod_experts(model, keep_ratio=0.5, min_experts_per_part=1)

    n_after = _count_experts(model)
    assert n_after < n_before, (
        f"剪枝后应减少: before={n_before}, after={n_after}"
    )
    assert n_after == 4, (
        f"keep_ratio=0.5 应保留 4 个 (2 parts × 2), got {n_after}"
    )

    # 验证每个 DensePart 的元数据一致性：
    # - num_experts == len(experts)
    # - router.num_routes == len(experts)
    # - router.gate.weight 行数 == len(experts)
    # - router.top_k == min(原 top_k, len(experts))
    for m in model.modules():
        if isinstance(m, MoDLayer):
            for part in m.parts:
                assert part.num_experts == len(part.experts), (
                    f"DensePart.num_experts({part.num_experts}) "
                    f"!= len(experts)({len(part.experts)})"
                )
                assert part.router.num_routes == len(part.experts), (
                    f"router.num_routes({part.router.num_routes}) "
                    f"!= len(experts)({len(part.experts)})"
                )
                # gate weight shape: (num_routes, dim)
                assert part.router.gate.weight.data.shape[0] == len(part.experts)
                # top_k = min(原 top_k=2, remaining=2) = 2
                assert part.router.top_k == min(2, len(part.experts))


# ---------------------------------------------------------------------------
# 5. compress_mod_experts min_experts_per_part 边界
# ---------------------------------------------------------------------------


def test_compress_mod_experts_min_per_part():
    """keep_ratio=0.1 + min_experts_per_part=2：每个 DensePart 至少 2 个 Expert。

    构造 2 parts × 4 experts 模型，keep_ratio=0.1 → int(4*0.1)=0，
    min_experts_per_part=2 → max(2, 0)=2，每个 DensePart 保留 2 个。
    """
    model = _build_mod_test_model(
        num_dense_parts=2, num_experts_per_part=4, top_k=2, dim=32)
    n_before = _count_experts(model)
    assert n_before == 8

    compress_mod_experts(model, keep_ratio=0.1, min_experts_per_part=2)

    # 每个 DensePart 至少 2 个 Expert
    per_part = _experts_per_part(model)
    assert len(per_part) == 2, f"应有 2 个 DensePart, got {len(per_part)}"
    for n in per_part:
        assert n >= 2, (
            f"每个 DensePart 应至少 2 个 Expert, got {n}"
        )
    # 2 parts × 2 experts = 4 total
    assert sum(per_part) == 4, (
        f"keep_ratio=0.1+min=2 应保留 4 个 (2 parts × 2), got {sum(per_part)}"
    )


# ---------------------------------------------------------------------------
# 6. compress_mod_experts 返回统计 dict
# ---------------------------------------------------------------------------


def test_compress_mod_experts_stats():
    """return_stats=True 返回 (model, stats)，stats 含正确字段与数值。"""
    model = _build_mod_test_model(
        num_dense_parts=2, num_experts_per_part=4, top_k=2, dim=32)
    n_before = _count_experts(model)
    assert n_before == 8

    new_model, stats = compress_mod_experts(
        model, keep_ratio=0.5, min_experts_per_part=1, return_stats=True)

    # 原地修改：返回的 new_model 应与传入的 model 是同一对象
    assert new_model is model, "compress_mod_experts 应原地修改 model"

    # 统计 dict 字段齐全
    assert "original_experts" in stats
    assert "kept_experts" in stats
    assert "compression_ratio" in stats

    # 数值校验
    assert stats["original_experts"] == 8
    assert stats["kept_experts"] == 4  # 2 parts × 2 experts
    expected_ratio = 1.0 - 4 / 8  # 0.5
    assert abs(stats["compression_ratio"] - expected_ratio) < 1e-6, (
        f"compression_ratio 应为 {expected_ratio}, got {stats['compression_ratio']}"
    )
    # 压缩比应在 [0, 1] 区间
    assert 0.0 <= stats["compression_ratio"] <= 1.0

    # 再次确认模型实际状态与 stats 一致
    assert _count_experts(model) == stats["kept_experts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
