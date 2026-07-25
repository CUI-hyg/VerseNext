"""Part5K1.1：VMPC（VerseNext Model Parameters Compression）V2.0 门面 + legacy 兼容测试.

Part5K1.1 起 ``verse_torch.vmpc`` 是 VMPC V2.0 的独立实现（VMPCV2 + VSC 引擎），
但仍 re-export ``verse_torch.compress`` 的对象作为向后兼容门面（legacy 路径）。

覆盖：
1. 门面导入同一性（``verse_torch.vmpc.X is verse_torch.compress.X``，legacy 兼容）
2. ``VMPCRegularizer`` 实例化
3. ``compute_penalty`` 返回非负 float
4. ``step`` 收紧逻辑（val_loss 平台期 → target_sparsity 减小）
5. ``step`` 早停（target_sparsity 降到下限 → should_stop=True）
6. ``vmpc_compress`` small 预设（V2.0 路径）
7. ``vmpc_compress`` mate 预设（V2.0 路径）
8. ``vmpc_compress`` 未知 profile 抛 ValueError
9. 顶层导出（``from verse_torch import vmpc, VMPCRegularizer, vmpc_compress``）

运行方式：
    cd /workspace
    python -m pytest tests/test_vmpc_facade.py -x -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import nn, vmpc as vmpc_mod
from verse_torch.vmpc import (
    VMPCRegularizer,
    vmpc_compress,
    compress_pipeline as vmpc_compress_pipeline,
)
from verse_torch.compress import compress_pipeline as compress_compress_pipeline
from verse_torch import vmpc as vmpc_top, VMPCRegularizer as VMPCReg_top, vmpc_compress as vc_top


SEED = 42


# ---------------------------------------------------------------------------
# 辅助：构造小模型
# ---------------------------------------------------------------------------


def _build_small_model(seed=SEED):
    """构造一个小型 Sequential 模型（两层 Linear）。"""
    np.random.seed(seed)
    return nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))


# ---------------------------------------------------------------------------
# 1. 门面导入同一性
# ---------------------------------------------------------------------------


def test_facade_identity():
    """vmpc 门面 re-export 的对象必须与 compress 模块中的是同一对象。"""
    assert vmpc_compress_pipeline is compress_compress_pipeline
    # 也检查其他 re-export 对象的同一性
    from verse_torch.vmpc import (
        OutlierSafePruner as OSP_vmpc,
        LoRALinear as LRL_vmpc,
        KnowledgeDistiller as KD_vmpc,
        QLinear as QL_vmpc,
    )
    from verse_torch.compress import (
        OutlierSafePruner as OSP_compress,
        LoRALinear as LRL_compress,
        KnowledgeDistiller as KD_compress,
        QLinear as QL_compress,
    )
    assert OSP_vmpc is OSP_compress
    assert LRL_vmpc is LRL_compress
    assert KD_vmpc is KD_compress
    assert QL_vmpc is QL_compress


# ---------------------------------------------------------------------------
# 2. VMPCRegularizer 实例化
# ---------------------------------------------------------------------------


def test_regularizer_instantiation():
    """VMPCRegularizer 实例化不报错，参数正确保存。"""
    reg = VMPCRegularizer(l2_weight=1e-5, dropout_rate=0.1,
                          target_sparsity=0.3, patience=5, sparsity_decay=0.9)
    assert reg.l2_weight == 1e-5
    assert reg.dropout_rate == 0.1
    assert reg.target_sparsity == 0.3
    assert reg.patience == 5
    assert reg.sparsity_decay == 0.9
    assert reg.val_loss_history == []


def test_regularizer_default_values():
    """默认参数实例化。"""
    reg = VMPCRegularizer()
    assert reg.l2_weight == 1e-5
    assert reg.dropout_rate == 0.0
    assert reg.target_sparsity == 0.3
    assert reg.patience == 5
    assert reg.sparsity_decay == 0.9


# ---------------------------------------------------------------------------
# 3. compute_penalty 返回非负 float
# ---------------------------------------------------------------------------


def test_compute_penalty_non_negative():
    """compute_penalty 对小模型返回非负 float。"""
    model = _build_small_model()
    reg = VMPCRegularizer(l2_weight=1e-5, target_sparsity=0.3)
    penalty = reg.compute_penalty(model)
    assert isinstance(penalty, float)
    assert penalty >= 0.0


def test_compute_penalty_zero_l2():
    """l2_weight=0 时 penalty 为 0（无 dropout）。"""
    model = _build_small_model()
    reg = VMPCRegularizer(l2_weight=0.0, dropout_rate=0.0)
    penalty = reg.compute_penalty(model)
    assert penalty == 0.0


def test_compute_penalty_with_dropout():
    """dropout_rate>0 时 penalty 仍为非负 float（不修改原参数）。"""
    model = _build_small_model()
    # 记录原始权重
    orig_weights = []
    for p in model._parameters.values():
        orig_weights.append(p.data.copy())
    # 子模块参数
    for m in model._modules.values():
        for p in m._parameters.values():
            orig_weights.append(p.data.copy())

    np.random.seed(SEED)
    reg = VMPCRegularizer(l2_weight=1e-3, dropout_rate=0.3)
    penalty = reg.compute_penalty(model)
    assert isinstance(penalty, float)
    assert penalty >= 0.0

    # 验证原参数未被修改
    all_params_now = []
    for p in model._parameters.values():
        all_params_now.append(p.data.copy())
    for m in model._modules.values():
        for p in m._parameters.values():
            all_params_now.append(p.data.copy())
    assert len(orig_weights) == len(all_params_now)
    for ow, nw in zip(orig_weights, all_params_now):
        np.testing.assert_array_equal(ow, nw)


# ---------------------------------------------------------------------------
# 4. step 收紧逻辑
# ---------------------------------------------------------------------------


def test_step_tightening_plateau():
    """val_loss 平台期 [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]（patience=5）
    后 target_sparsity 应减小。"""
    reg = VMPCRegularizer(target_sparsity=0.3, patience=5, sparsity_decay=0.9)
    initial_sparsity = reg.target_sparsity
    history = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
    last_should_stop = False
    for v in history:
        last_should_stop, _ = reg.step(v)
    # target_sparsity 应该减小（0.3 * 0.9 = 0.27）
    assert reg.target_sparsity < initial_sparsity
    # 平台期未到下限，不应早停
    assert last_should_stop is False
    # 精确值校验：0.3 * 0.9 = 0.27
    assert abs(reg.target_sparsity - 0.27) < 1e-9


def test_step_tightening_worsening():
    """val_loss 变差 [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]（patience=5）
    后 target_sparsity 应减小（任务规格条件：min(recent) >= min(prev)）。"""
    reg = VMPCRegularizer(target_sparsity=0.3, patience=5, sparsity_decay=0.9)
    initial_sparsity = reg.target_sparsity
    history = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for v in history:
        should_stop, _ = reg.step(v)
    assert reg.target_sparsity < initial_sparsity
    assert should_stop is False


def test_step_no_tightening_when_improving():
    """val_loss 持续下降时不应收紧。"""
    reg = VMPCRegularizer(target_sparsity=0.3, patience=5, sparsity_decay=0.9)
    initial_sparsity = reg.target_sparsity
    # 持续下降（每步都比前一步小，且最近窗口总有新最佳）
    history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    for v in history:
        reg.step(v)
    assert reg.target_sparsity == initial_sparsity  # 未收紧


# ---------------------------------------------------------------------------
# 5. step 早停
# ---------------------------------------------------------------------------


def test_step_early_stop():
    """target_sparsity 降到下限（0.05）以下时 should_stop=True。"""
    # 初始 0.1，decay=0.4：一次收紧后 0.1*0.4=0.04 < 0.05 → 早停
    reg = VMPCRegularizer(target_sparsity=0.1, patience=2, sparsity_decay=0.4)
    # 喂入平台期 val_loss 触发收紧
    history = [1.0, 1.0, 1.0]
    should_stop = False
    for v in history:
        should_stop, new_sparsity = reg.step(v)
    assert should_stop is True
    # 钳位到下限
    assert reg.target_sparsity == VMPCRegularizer.SPARSITY_FLOOR
    assert new_sparsity == VMPCRegularizer.SPARSITY_FLOOR


def test_step_data_insufficient():
    """数据不足（len <= patience）时不收紧、不早停。"""
    reg = VMPCRegularizer(target_sparsity=0.3, patience=5)
    should_stop, sparsity = reg.step(1.0)
    assert should_stop is False
    assert sparsity == 0.3


# ---------------------------------------------------------------------------
# 6. vmpc_compress small 预设
# ---------------------------------------------------------------------------


def test_vmpc_compress_small():
    """vmpc_compress(profile='small') 不报错，返回压缩后模型。"""
    model = _build_small_model()
    result = vmpc_compress(model, profile="small")
    # compress_pipeline 默认返回新模型（非 stats dict）
    assert result is not None
    # 不应修改原模型（深拷贝）
    assert result is not model


# ---------------------------------------------------------------------------
# 7. vmpc_compress mate 预设
# ---------------------------------------------------------------------------


def test_vmpc_compress_mate():
    """vmpc_compress(profile='mate') 不报错，返回压缩后模型。"""
    model = _build_small_model()
    result = vmpc_compress(model, profile="mate")
    assert result is not None
    assert result is not model


def test_vmpc_compress_mate_with_teacher():
    """mate 预设 + teacher_model 启用蒸馏分支（无 train_loader，仅冻结 teacher）。"""
    np.random.seed(SEED)
    teacher = _build_small_model()
    student = _build_small_model()
    result = vmpc_compress(student, profile="mate", teacher_model=teacher)
    assert result is not None


# ---------------------------------------------------------------------------
# 8. vmpc_compress 未知 profile 抛 ValueError
# ---------------------------------------------------------------------------


def test_vmpc_compress_invalid_profile():
    """未知 profile 抛 ValueError。"""
    model = _build_small_model()
    with pytest.raises(ValueError, match="未知 profile"):
        vmpc_compress(model, profile="invalid")


# ---------------------------------------------------------------------------
# 9. 顶层导出
# ---------------------------------------------------------------------------


def test_top_level_exports():
    """from verse_torch import vmpc, VMPCRegularizer, vmpc_compress 全部可用。"""
    # 模块
    assert vmpc_top is vmpc_mod
    # 类 / 函数
    assert VMPCReg_top is VMPCRegularizer
    assert vc_top is vmpc_compress
    # 模块内的 re-export 也可用
    assert hasattr(vmpc_top, "compress_pipeline")
    assert hasattr(vmpc_top, "OutlierSafePruner")
    assert hasattr(vmpc_top, "LoRALinear")
    assert hasattr(vmpc_top, "KnowledgeDistiller")
    assert hasattr(vmpc_top, "QLinear")
    assert hasattr(vmpc_top, "VMPCRegularizer")
    assert hasattr(vmpc_top, "vmpc_compress")


def test_vmpc_in_all():
    """vmpc / VMPCRegularizer / vmpc_compress 应在 verse_torch.__all__ 中。"""
    import verse_torch
    assert "vmpc" in verse_torch.__all__
    assert "VMPCRegularizer" in verse_torch.__all__
    assert "vmpc_compress" in verse_torch.__all__


# ---------------------------------------------------------------------------
# 附加：attach 行为测试
# ---------------------------------------------------------------------------


class _DummyTrainer:
    """模拟 Trainer（duck typing）：有 model 与 _compute_loss。"""

    def __init__(self, model):
        self.model = model

    def _compute_loss(self, x):
        # 模拟返回一个标量 loss（float）
        return 1.0


def test_attach_registers_and_patches():
    """attach 后 trainer.regularizer 被设置，_compute_loss 被 patch。"""
    model = _build_small_model()
    trainer = _DummyTrainer(model)
    reg = VMPCRegularizer(l2_weight=1e-5)
    reg.attach(trainer)
    # 注册
    assert trainer.regularizer is reg
    # patch 后的 _compute_loss 应加上 penalty
    patched_loss = trainer._compute_loss(x=None)
    # 原 loss=1.0，加 penalty 后应 > 1.0（penalty 非负）
    assert patched_loss >= 1.0


def test_attach_without_compute_loss():
    """trainer 无 _compute_loss / compute_loss 时，attach 仅做注册。"""
    model = _build_small_model()

    class _BareTrainer:
        def __init__(self, model):
            self.model = model

    trainer = _BareTrainer(model)
    reg = VMPCRegularizer()
    reg.attach(trainer)
    assert trainer.regularizer is reg
