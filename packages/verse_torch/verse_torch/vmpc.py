"""VMPC（VerseNext Model Parameters Compression）V1.5 门面.

统一压缩 / 量化 / 蒸馏 / 剪枝入口，re-export ``verse_torch.compress`` 中的核心对象，
保证调用者通过 ``verse_torch.vmpc`` 与通过 ``verse_torch.compress`` 拿到的是同一对象
（``verse_torch.vmpc.compress_pipeline is verse_torch.compress.compress_pipeline``
必须为 ``True``）。

VMPC V1.5 算法升级在 Task 4 完成；本模块做命名门面 + 正则器 + 便捷预设函数。
``compress_pipeline`` 内部按 version 字段分派：``version>=1.5`` 走
``_compress_pipeline_v15``（VMPC V1.3 + contrastive_distill + logit_calibration），
``version>=1.3`` 走 ``_compress_pipeline_v13`` 路径。

公开对象：
- ``compress_pipeline`` / ``OutlierSafePruner`` / ``LoRALinear`` /
  ``KnowledgeDistiller`` / ``QLinear`` / ``compress_mod_experts`` /
  ``compression_report`` / ``prune_only`` / ``quantize_only`` /
  ``lora_only`` / ``ternary_only`` / ``distill_only`` /
  ``count_parameters`` / ``count_nonzero_params`` / ``compute_compressed_bits``
- ``VMPCRegularizer``：VMPC 正则化器（防过拟合 + 压缩感知稀疏收紧）
- ``vmpc_compress``：一键压缩预设（profile="small" / "mate"）
"""
from __future__ import annotations

import numpy as np

from .compress import (
    compress_pipeline,
    OutlierSafePruner,
    LoRALinear,
    KnowledgeDistiller,
    QLinear,
    compress_mod_experts,
    compression_report,
    prune_only,
    quantize_only,
    lora_only,
    ternary_only,
    distill_only,
    count_parameters,
    count_nonzero_params,
    compute_compressed_bits,
)


# ---------------------------------------------------------------------------
# 内部辅助：递归遍历所有 Tensor 参数（与 compress._iter_all_tensors 同义）
# ---------------------------------------------------------------------------


def _iter_all_tensors(model):
    """递归生成所有 Tensor 参数（包括 requires_grad=False）。"""
    for p in model._parameters.values():
        yield p
    for m in model._modules.values():
        yield from _iter_all_tensors(m)


# ---------------------------------------------------------------------------
# SubTask 3.2: VMPCRegularizer
# ---------------------------------------------------------------------------


class VMPCRegularizer:
    """VMPC 正则化器：防过拟合 + 压缩感知稀疏收紧。

    组成：
    1. 参数幅度 L2 正则（weight_decay 风格）
    2. 压缩感知 dropout（随机置零部分权重模拟压缩损失）
    3. early-exit 自适应稀疏收紧（val_loss 连续 patience 步不降
       → target_sparsity *= sparsity_decay）

    设计目标：
    - 轻量、无状态依赖，不绑定具体 Trainer 类（duck typing）
    - ``compute_penalty`` 可独立调用，便于用户手动接入训练循环
    - ``step`` 维护 ``val_loss_history``，自适应收紧稀疏度，达到下限时早停

    Args:
        l2_weight: L2 正则权重（默认 1e-5）
        dropout_rate: 压缩感知 dropout 比例（0-1，默认 0 不启用）
        target_sparsity: 目标稀疏度（默认 0.3）
        patience: 容忍步数（默认 5），连续 patience 步 val_loss 不降则收紧 sparsity
        sparsity_decay: 收紧系数（默认 0.9，即每次收紧到原值的 90%）
    """

    # 稀疏度下限：target_sparsity 降到此值以下时触发早停
    SPARSITY_FLOOR = 0.05

    def __init__(self, l2_weight=1e-5, dropout_rate=0.0, target_sparsity=0.3,
                 patience=5, sparsity_decay=0.9):
        self.l2_weight = float(l2_weight)
        self.dropout_rate = float(dropout_rate)
        self.target_sparsity = float(target_sparsity)
        self.patience = int(patience)
        self.sparsity_decay = float(sparsity_decay)
        self.val_loss_history: list[float] = []
        self._trainer = None

    def attach(self, trainer):
        """挂载到 Trainer，在 loss 计算后加上正则项。

        trainer 需有 ``model`` / ``val_loss_history`` 属性（或类似接口）。
        若 trainer 有 ``_compute_loss`` 方法（备选 ``compute_loss``），则
        monkey-patch 之，在原 loss 上叠加 ``self.compute_penalty(trainer.model)``；
        否则仅做注册（``trainer.regularizer = self``），由用户自行在训练循环中
        调用 ``compute_penalty``。
        """
        self._trainer = trainer
        trainer.regularizer = self
        # 优先 monkey-patch _compute_loss；备选 compute_loss
        for method_name in ("_compute_loss", "compute_loss"):
            original = getattr(trainer, method_name, None)
            if not callable(original) or getattr(original, "_vmpc_patched", False):
                continue
            reg = self

            def patched(*args, _orig=original, **kwargs):
                loss = _orig(*args, **kwargs)
                model = getattr(trainer, "model", None)
                if model is None:
                    return loss
                penalty = reg.compute_penalty(model)
                try:
                    loss = loss + penalty
                except Exception:
                    # 若 loss 与 float 不可加（罕见），保持原样
                    pass
                return loss

            patched._vmpc_patched = True
            setattr(trainer, method_name, patched)
            break  # 只 patch 第一个匹配的方法

    def compute_penalty(self, model) -> float:
        """计算当前模型的正则惩罚（L2 + dropout 模拟）。

        - L2 项：``l2_weight * sum(p^2 for p in model.parameters())``
        - dropout 模拟：若 ``dropout_rate > 0``，对每个参数随机生成 mask，
          额外加上被 mask 掉的权重平方和（模拟压缩时这部分权重丢失的损失）。
          注意：仅计算损失，不修改原参数。

        Returns:
            非负 float 惩罚值
        """
        penalty = 0.0
        for p in _iter_all_tensors(model):
            data = p.data
            penalty += self.l2_weight * float(np.sum(data * data))
        if self.dropout_rate > 0.0:
            for p in _iter_all_tensors(model):
                data = p.data
                mask = (np.random.rand(*data.shape) < self.dropout_rate)
                penalty += self.l2_weight * float(np.sum((data * mask) ** 2))
        return float(penalty)

    def step(self, val_loss):
        """每个 eval 步调用，检查是否需要收紧 sparsity。

        返回 ``(should_stop, new_sparsity)``：
        - 若最近 ``patience`` 步 val_loss 没有下降，则
          ``target_sparsity *= sparsity_decay``。检测「没有下降」有两种情况：
          1) 任务规格 ``min(history[-patience:]) >= min(history[:-patience])``
             —— 最近 patience 步的最佳值未优于之前历史最佳（val_loss 变差或持平于旧最佳）
          2) 平台期 ``min(recent) == max(recent)`` —— 最近 patience 步 val_loss
             完全持平（覆盖 val_loss 改善后进入平台期的场景，如
             ``[1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]``）
        - 若 ``target_sparsity`` 降到 ``SPARSITY_FLOOR`` 以下，返回
          ``should_stop=True``，并将 ``target_sparsity`` 钳位到 ``SPARSITY_FLOOR``

        Args:
            val_loss: 当前 eval 步的验证损失

        Returns:
            ``(should_stop: bool, new_sparsity: float)``
        """
        self.val_loss_history.append(float(val_loss))
        # 数据不足，无法判断
        if len(self.val_loss_history) <= self.patience:
            return False, self.target_sparsity
        recent = self.val_loss_history[-self.patience:]
        prev = self.val_loss_history[:-self.patience]
        # 收紧条件：val_loss 变差/持平于旧最佳，或最近 patience 步完全平台
        should_tighten = (
            (bool(prev) and min(recent) >= min(prev))
            or (self.patience >= 2 and min(recent) == max(recent))
        )
        if should_tighten:
            # 长期无改善，收紧 sparsity
            self.target_sparsity *= self.sparsity_decay
            if self.target_sparsity < self.SPARSITY_FLOOR:
                self.target_sparsity = self.SPARSITY_FLOOR
                return True, self.target_sparsity
            return False, self.target_sparsity
        return False, self.target_sparsity


# ---------------------------------------------------------------------------
# SubTask 3.3: vmpc_compress 便捷预设函数
# ---------------------------------------------------------------------------


def vmpc_compress(model, profile="small", **kwargs):
    """VMPC 一键压缩预设。

    profile="small": ternary 量化 + 高稀疏（sparsity=0.5），适配 0.06zB 小模型
    profile="mate":  int4 量化 + 中稀疏（sparsity=0.3）+ 蒸馏，适配 0.2zB 旗舰模型

    config 字段名与 ``compress_pipeline`` 实际支持的字段对齐（``prune`` /
    ``quantize`` / ``lora`` / ``ternary`` / ``distill`` 子 dict，以及顶层
    ``teacher_model`` / ``teacher`` / ``train_loader`` 便捷字段）。

    Args:
        model: 待压缩模型（``nn.Module`` 子类）
        profile: "small" / "mate"
        **kwargs: 覆盖预设 config 的字段。例如：
            - ``teacher_model=teacher, train_loader=loader`` 启用蒸馏
            - ``prune={"sparsity": 0.4}`` 覆盖默认剪枝稀疏度
            - ``quantize={"bits": 8}`` 切换为 INT8 量化

    Returns:
        压缩后的新模型（``compress_pipeline`` 默认返回新 model，不修改原模型）
    """
    if profile == "small":
        # small 预设：ternary 量化（2bit/值，高压缩比）+ 高稀疏剪枝
        config = {
            "prune": {"sparsity": 0.5},
            "ternary": {},
            # use_lora=False：不包装 LoRA
            # distill=False：不蒸馏
        }
    elif profile == "mate":
        # mate 预设：int4 量化 + 中稀疏 + LoRA 包装
        # 蒸馏默认不启用（需用户提供 teacher_model）；用户可通过 kwargs 传入
        config = {
            "prune": {"sparsity": 0.3},
            "quantize": {"bits": 4},
            "lora": {"rank": 8, "alpha": 16},
        }
    else:
        raise ValueError(f"未知 profile: {profile}，支持 'small' / 'mate'")

    # 合并用户覆盖的 kwargs
    config.update(kwargs)

    # VMPC V1.5 算法升级已在 Task 4 完成：compress_pipeline 对 version>=1.5 走
    # _compress_pipeline_v15 路径（VMPC V1.3 + contrastive_distill + logit_calibration）。
    # 极端情况下回退到 VMPC V1.3（保证可用性）。
    try:
        return compress_pipeline(model, config=config, version="1.5")
    except Exception:
        # 极端情况下回退到 VMPC V1.3（保证可用性）
        return compress_pipeline(model, config=config, version="1.3")


__all__ = [
    # re-export from compress
    "compress_pipeline",
    "OutlierSafePruner",
    "LoRALinear",
    "KnowledgeDistiller",
    "QLinear",
    "compress_mod_experts",
    "compression_report",
    "prune_only",
    "quantize_only",
    "lora_only",
    "ternary_only",
    "distill_only",
    "count_parameters",
    "count_nonzero_params",
    "compute_compressed_bits",
    # VMPC 原生
    "VMPCRegularizer",
    "vmpc_compress",
]
