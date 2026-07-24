"""VMPC (VerseNext Model Parameters Compression) V2.0 门面.

Part5K1.1 全面重写：**抛弃 V1.5 全部接口**，基于三大支柱重建：

    VMPC V2.0 = VN 格式文件 + 传统技术 + VSC

核心设计
--------
VMPC **绝对不等于** 单纯的量化、剪枝等传统技术。V2.0 明确：

1. **VN 格式文件** (.vn): 所有模型文件的默认格式（高吞吐、高速度、方便压缩）。
   当 ``use_vmpc=True`` 时强制使用 .vn，不可替换。
2. **传统技术** (compress.py / quantize.py): 量化、剪枝、蒸馏等作为 VSC 物理压缩
   层的手段，由 VSC 引擎调度，不直接暴露给用户。
3. **VSC** (verse_torch.vsc): VerseNext 空间压缩技术，从三维空间角度
   （存储/算力/时间）对模型进行特别压缩，保持「速度快、能力强、占用小」。

物理压缩占 40%，专项算法优化+训练占 60%。

配置融合
--------
所有配置中通过 ``use_vmpc`` 统一开关（默认开启）。参数分为：

- ``legacy``: 传统技术直通参数（use_vmpc=False 时生效）
- ``vmpc``: V2 专属参数（use_vmpc=True 时生效，含 VSC 配置）

当 ``use_vmpc=True`` 时：
- 所有模型文件必须使用 ``*.vn`` 格式
- 压缩走 VSC 引擎（三维空间压缩）
- 训练/微调/推理路径自动接入 VMPC 优化

独立组件 API
------------
VMPC 作为独立组件提供 API，贯穿训练/微调/推理但不与具体架构强绑定：

    from verse_torch.vmpc import VMPCConfig, VMPCV2, vmpc_compress

    # 1. 配置
    config = VMPCConfig(use_vmpc=True, profile="small")

    # 2. 压缩
    engine = VMPCV2(config)
    compressed, stats = engine.compress(model)

    # 3. 训练补偿（恢复能力）
    engine.compensate(compressed, train_fn, train_data)

    # 4. 保存为 .vn 格式（use_vmpc=True 时强制）
    engine.save(compressed, "model.vn")

保留组件
--------
``VMPCRegularizer`` 保留：它是训练时正则化器（防过拟合 + 稀疏收紧），
与 V1.5 压缩 pipeline 无关，在 V2.0 中继续作为训练辅助工具。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from .tensor import Tensor


__all__ = [
    # V2.0 核心
    "VMPCConfig",
    "VMPCV2",
    "VMPCStats",
    "vmpc_compress",
    "VMPC_PROFILE_SMALL",
    "VMPC_PROFILE_MATE",
    # 训练辅助（保留）
    "VMPCRegularizer",
    # 传统技术 re-export（供 legacy 模式使用）
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
]


# ---------------------------------------------------------------------------
# 预设 profile
# ---------------------------------------------------------------------------

# small 预设：0.06zB 目标，高压缩比（ternary + 高稀疏）
VMPC_PROFILE_SMALL = {
    "target_ratio": 0.06,       # 压到原大小 6%
    "quantize_bits": 2,         # ternary 量化
    "target_sparsity": 0.5,     # 高稀疏
    "storage_weight": 0.5,      # 存储优先（小模型受存储约束）
    "compute_weight": 0.3,
    "time_weight": 0.2,
}

# mate 预设：0.2zB 目标，中压缩比 + 蒸馏（int4 + 中稀疏）
VMPC_PROFILE_MATE = {
    "target_ratio": 0.20,       # 压到原大小 20%
    "quantize_bits": 4,         # int4 量化
    "target_sparsity": 0.3,     # 中稀疏
    "storage_weight": 0.3,
    "compute_weight": 0.4,      # 算力优先（旗舰模型受算力约束）
    "time_weight": 0.3,
}


# ---------------------------------------------------------------------------
# VMPCConfig: V2.0 配置
# ---------------------------------------------------------------------------


@dataclass
class VMPCConfig:
    """VMPC V2.0 配置。

    统一管理是否使用 VMPC（默认开启）+ legacy/vmpc 参数分离。

    Attributes:
        use_vmpc: 是否启用 VMPC V2.0（默认 True）。启用时：
            - 所有模型文件必须使用 .vn 格式
            - 压缩走 VSC 引擎（三维空间压缩）
            - 训练/推理路径自动接入 VMPC 优化
            False 时走 legacy 模式（传统技术直通）。
        profile: 预设名（"small" / "mate" / None）。指定后自动填充 vmpc 参数。
        target_ratio: 目标压缩比（0.1 = 压到 10%）
        quantize_bits: 量化 bit 数（4=int4, 2=ternary, 8=int8）
        target_sparsity: 目标稀疏度
        storage_weight: 存储维度权重
        compute_weight: 算力维度权重
        time_weight: 时间维度权重
        force_vn_format: use_vmpc=True 时是否强制 .vn 格式（默认 True）
        enable_compensation: 是否启用训练补偿（恢复压缩损失的能力）
        compensation_steps: 补偿训练步数
        legacy: legacy 模式参数（use_vmpc=False 时生效），传递给 compress_pipeline
        vmpc: V2 专属参数（覆盖 profile 预设）
    """

    use_vmpc: bool = True
    profile: Optional[str] = None
    target_ratio: float = 0.1
    quantize_bits: int = 4
    target_sparsity: float = 0.3
    storage_weight: float = 0.4
    compute_weight: float = 0.4
    time_weight: float = 0.2
    force_vn_format: bool = True
    enable_compensation: bool = True
    compensation_steps: int = 100
    legacy: dict = field(default_factory=dict)
    vmpc: dict = field(default_factory=dict)

    def __post_init__(self):
        """根据 profile 预设填充参数（vmpc 字段覆盖预设）。"""
        if self.profile is not None:
            preset = _get_profile_preset(self.profile)
            for k, v in preset.items():
                if not hasattr(self, k):
                    continue
                # vmpc 字段不覆盖已显式设置的属性
                if k in self.vmpc:
                    continue
                setattr(self, k, v)
        # vmpc 字段覆盖（最高优先级）
        for k, v in self.vmpc.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self) -> dict:
        """序列化为 dict（用于 config.yml 持久化）。"""
        return {
            "use_vmpc": self.use_vmpc,
            "profile": self.profile,
            "target_ratio": self.target_ratio,
            "quantize_bits": self.quantize_bits,
            "target_sparsity": self.target_sparsity,
            "storage_weight": self.storage_weight,
            "compute_weight": self.compute_weight,
            "time_weight": self.time_weight,
            "force_vn_format": self.force_vn_format,
            "enable_compensation": self.enable_compensation,
            "compensation_steps": self.compensation_steps,
            "legacy": dict(self.legacy),
            "vmpc": dict(self.vmpc),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VMPCConfig":
        """从 dict 反序列化。"""
        return cls(
            use_vmpc=d.get("use_vmpc", True),
            profile=d.get("profile"),
            target_ratio=d.get("target_ratio", 0.1),
            quantize_bits=d.get("quantize_bits", 4),
            target_sparsity=d.get("target_sparsity", 0.3),
            storage_weight=d.get("storage_weight", 0.4),
            compute_weight=d.get("compute_weight", 0.4),
            time_weight=d.get("time_weight", 0.2),
            force_vn_format=d.get("force_vn_format", True),
            enable_compensation=d.get("enable_compensation", True),
            compensation_steps=d.get("compensation_steps", 100),
            legacy=d.get("legacy", {}),
            vmpc=d.get("vmpc", {}),
        )

    def summary(self) -> dict:
        """返回摘要 dict。"""
        return self.to_dict()


def _get_profile_preset(profile: str) -> dict:
    """获取 profile 预设参数。"""
    if profile == "small":
        return dict(VMPC_PROFILE_SMALL)
    if profile == "mate":
        return dict(VMPC_PROFILE_MATE)
    raise ValueError(f"未知 profile: {profile}，支持 'small' / 'mate'")


# ---------------------------------------------------------------------------
# VMPCStats: V2.0 压缩统计
# ---------------------------------------------------------------------------


@dataclass
class VMPCStats:
    """VMPC V2.0 压缩结果统计。

    封装 VSC 统计 + VMPC 特有信息。

    Attributes:
        vsc_stats: VSC 引擎的详细统计（含三维压缩比）
        use_vmpc: 是否使用了 VMPC V2.0
        profile: 使用的预设名
        vn_format: 是否使用了 .vn 格式
        compensated: 是否执行了训练补偿
        compensation_result: 补偿训练结果
    """

    vsc_stats: Optional[Any] = None
    use_vmpc: bool = True
    profile: Optional[str] = None
    vn_format: bool = True
    compensated: bool = False
    compensation_result: Optional[dict] = None

    def summary(self) -> dict:
        """返回可序列化的摘要 dict。"""
        return {
            "use_vmpc": self.use_vmpc,
            "profile": self.profile,
            "vn_format": self.vn_format,
            "compensated": self.compensated,
            "compensation_result": self.compensation_result,
            "vsc": self.vsc_stats.summary() if self.vsc_stats else None,
        }


# ---------------------------------------------------------------------------
# VMPCV2: V2.0 压缩引擎
# ---------------------------------------------------------------------------


class VMPCV2:
    """VMPC V2.0 压缩引擎：协调 VN 格式 + 传统技术 + VSC。

    作为独立组件提供 API，贯穿训练/微调/推理但不与具体架构强绑定。

    用法::

        config = VMPCConfig(use_vmpc=True, profile="small")
        engine = VMPCV2(config)
        compressed, stats = engine.compress(model)
        engine.compensate(compressed, train_fn, train_data)
        engine.save(compressed, "model.vn")

    Args:
        config: :class:`VMPCConfig` 配置
    """

    def __init__(self, config: VMPCConfig):
        self.config = config

    def compress(self, model, return_stats: bool = True):
        """执行 VMPC V2.0 压缩。

        根据 ``config.use_vmpc`` 选择路径：
        - ``use_vmpc=True``: 走 VSC 引擎（三维空间压缩）
        - ``use_vmpc=False``: 走 legacy 模式（compress_pipeline 传统技术直通）

        Args:
            model: 待压缩模型
            return_stats: 是否返回统计信息

        Returns:
            压缩后的模型。``return_stats=True`` 时返回 ``(model, stats)``。
        """
        if not self.config.use_vmpc:
            return self._compress_legacy(model, return_stats)
        return self._compress_v2(model, return_stats)

    def _compress_v2(self, model, return_stats: bool):
        """V2.0 路径：VSC 引擎压缩。"""
        from .vsc import VSCProfile, VSCPlan, VSCEngine

        # 1. 分析模型三维空间占用
        profile = VSCProfile.analyze(model)

        # 2. 生成 VSC 压缩计划
        plan = VSCPlan.create(
            profile,
            target_ratio=self.config.target_ratio,
            storage_weight=self.config.storage_weight,
            compute_weight=self.config.compute_weight,
            time_weight=self.config.time_weight,
            quantize_bits=self.config.quantize_bits,
            target_sparsity=self.config.target_sparsity,
        )

        # 3. 执行 VSC 压缩
        engine = VSCEngine(plan)
        compressed, vsc_stats = engine.apply(model, return_stats=True)

        # 4. 构建 VMPC 统计
        stats = VMPCStats(
            vsc_stats=vsc_stats,
            use_vmpc=True,
            profile=self.config.profile,
            vn_format=self.config.force_vn_format,
            compensated=False,
        )

        if return_stats:
            return compressed, stats
        return compressed

    def _compress_legacy(self, model, return_stats: bool):
        """Legacy 路径：传统技术直通（use_vmpc=False）。

        Part5K1.1：VMPC V2.0 全面抛弃 V1.5 技术栈。``use_vmpc=False`` 时
        走 :func:`verse_torch.compress.compress_pipeline` 传统压缩管线
        （prune / quantize / lora / ternary / distill），作为独立组件提供，
        不再携带 VMPC V1.5 标签。
        """
        from .compress import compress_pipeline

        legacy_config = dict(self.config.legacy) if self.config.legacy else {}
        # legacy 模式：传统技术直通（不传 version，使用 compress_pipeline 默认行为）
        result = compress_pipeline(model, config=legacy_config)

        stats = VMPCStats(
            vsc_stats=None,
            use_vmpc=False,
            profile=self.config.profile,
            vn_format=False,
            compensated=False,
        )

        if return_stats:
            return result, stats
        return result

    def compensate(
        self,
        model,
        train_fn: Callable,
        train_data: Any = None,
        steps: Optional[int] = None,
    ) -> dict:
        """训练补偿：恢复因压缩损失的能力。

        VMPC V2.0 核心设计：算法优化（60%）中包含训练补偿，
        确保压缩后模型能力恢复到接近原模型水平。

        仅在 ``config.enable_compensation=True`` 时执行。

        Args:
            model: 压缩后的模型
            train_fn: 训练函数 ``train_fn(model, data, steps) -> loss_history``
            train_data: 训练数据
            steps: 补偿步数；None 则用 config.compensation_steps

        Returns:
            补偿统计 dict ``{"final_loss": float, "recovered": float}``
        """
        if not self.config.enable_compensation:
            return {"skipped": True, "reason": "compensation disabled"}

        if steps is None:
            steps = self.config.compensation_steps

        if not self.config.use_vmpc:
            # legacy 模式不执行 VSC 补偿
            return {"skipped": True, "reason": "legacy mode"}

        # VMPC V2.0 训练补偿：直接调用 train_fn 恢复能力
        # （算法优化 60% 的核心环节，与 VSCEngine.compensate 同语义）
        try:
            loss_history = train_fn(model, train_data, steps)
            final_loss = loss_history[-1] if loss_history else 0.0
            initial_loss = loss_history[0] if loss_history else 0.0
            recovered = (initial_loss - final_loss) / max(initial_loss, 1e-8)
            return {"final_loss": final_loss, "recovered": recovered}
        except Exception as e:
            return {"final_loss": -1.0, "recovered": 0.0, "error": str(e)}

    def save(self, model, path: str, config: Any = None) -> str:
        """保存模型到文件。

        ``use_vmpc=True`` 时强制使用 .vn 格式（``force_vn_format=True``）。
        ``use_vmpc=False`` 时按路径后缀决定格式。

        Args:
            model: 待保存模型
            path: 目标路径
            config: 模型配置（dict 或带 to_dict() 的对象）

        Returns:
            实际保存的文件路径
        """
        force_vn = self.config.use_vmpc and self.config.force_vn_format

        if force_vn and not path.lower().endswith(".vn"):
            # 强制 .vn 格式：自动改后缀
            base, _ = os.path.splitext(path)
            path = base + ".vn"

        if path.lower().endswith(".vn"):
            return self._save_vn(model, path, config)
        # legacy .pt 格式
        return self._save_pt(model, path, config)

    def _save_vn(self, model, path: str, config: Any) -> str:
        """保存为 .vn 格式。"""
        from .vn_format import VNFileWriter

        # 提取 state_dict
        sd = self._get_state_dict(model)
        arch = getattr(model, "arch", "versenex")

        # 构建 compression_info（VMPC V2.0 元数据）
        compression_info = {
            "vmpc_version": "2.0",
            "use_vmpc": self.config.use_vmpc,
            "profile": self.config.profile,
            "target_ratio": self.config.target_ratio,
            "quantize_bits": self.config.quantize_bits,
            "target_sparsity": self.config.target_sparsity,
        }

        writer = VNFileWriter(
            path,
            arch=arch,
            config=config,
            compression_info=compression_info,
        )
        try:
            writer.write_weights(sd)
            writer.close()
        except Exception:
            writer.close()
            raise
        return path

    def _save_pt(self, model, path: str, config: Any) -> str:
        """保存为 .pt 格式（legacy）。"""
        import pickle

        sd = self._get_state_dict(model)
        payload = {
            "arch": getattr(model, "arch", "versenex"),
            "config": config if isinstance(config, dict) else (
                config.to_dict() if hasattr(config, "to_dict") else {}
            ),
            "state_dict": sd,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    def _get_state_dict(self, model) -> dict:
        """提取模型的 state_dict。"""
        if hasattr(model, "state_dict"):
            sd = model.state_dict()
            return {k: np.asarray(v) for k, v in sd.items()}
        # 兜底：直接遍历参数
        result = {}
        for name, p in _iter_named_tensors(model):
            result[name] = np.asarray(p.data)
        return result


# ---------------------------------------------------------------------------
# VMPCRegularizer: 训练时正则化器（与 VSC 引擎独立，可独立挂载到任意 Trainer）
# ---------------------------------------------------------------------------


def _iter_all_tensors(model):
    """递归生成所有 Tensor 参数。"""
    for p in model._parameters.values():
        yield p
    for m in model._modules.values():
        yield from _iter_all_tensors(m)


def _iter_named_tensors(model, prefix: str = ""):
    """递归生成 (name, Tensor) 对。"""
    for name, p in model._parameters.items():
        yield f"{prefix}{name}", p
    for name, m in model._modules.items():
        yield from _iter_named_tensors(m, prefix=f"{prefix}{name}.")


class VMPCRegularizer:
    """VMPC 训练时正则化器：防过拟合 + 压缩感知稀疏收紧。

    在 V2.0 中保留：它是训练辅助工具，与压缩 pipeline 无关。
    可独立挂载到任意 Trainer，在 loss 上叠加正则项。

    组成：
    1. 参数幅度 L2 正则（weight_decay 风格）
    2. 压缩感知 dropout（随机置零部分权重模拟压缩损失）
    3. early-exit 自适应稀疏收紧（val_loss 连续 patience 步不降
       → target_sparsity *= sparsity_decay）

    Args:
        l2_weight: L2 正则权重（默认 1e-5）
        dropout_rate: 压缩感知 dropout 比例（0-1，默认 0 不启用）
        target_sparsity: 目标稀疏度（默认 0.3）
        patience: 容忍步数（默认 5）
        sparsity_decay: 收紧系数（默认 0.9）
    """

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
        """挂载到 Trainer，在 loss 计算后加上正则项。"""
        self._trainer = trainer
        trainer.regularizer = self
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
                    pass
                return loss

            patched._vmpc_patched = True
            setattr(trainer, method_name, patched)
            break

    def compute_penalty(self, model) -> float:
        """计算当前模型的正则惩罚（L2 + dropout 模拟）。"""
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

        Returns:
            ``(should_stop, new_sparsity)``
        """
        self.val_loss_history.append(float(val_loss))
        if len(self.val_loss_history) <= self.patience:
            return False, self.target_sparsity
        recent = self.val_loss_history[-self.patience:]
        prev = self.val_loss_history[:-self.patience]
        should_tighten = (
            (bool(prev) and min(recent) >= min(prev))
            or (self.patience >= 2 and min(recent) == max(recent))
        )
        if should_tighten:
            self.target_sparsity *= self.sparsity_decay
            if self.target_sparsity < self.SPARSITY_FLOOR:
                self.target_sparsity = self.SPARSITY_FLOOR
                return True, self.target_sparsity
            return False, self.target_sparsity
        return False, self.target_sparsity


# ---------------------------------------------------------------------------
# 传统技术 re-export（供 legacy 模式 + VSC 物理压缩层使用）
# ---------------------------------------------------------------------------

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
# 便捷函数：vmpc_compress（V2.0）
# ---------------------------------------------------------------------------


def vmpc_compress(
    model,
    profile: str = "small",
    use_vmpc: bool = True,
    compensate_fn: Optional[Callable] = None,
    compensate_data: Any = None,
    **kwargs,
):
    """VMPC V2.0 一键压缩。

    Args:
        model: 待压缩模型
        profile: 预设名（"small" / "mate"）
        use_vmpc: 是否启用 VMPC V2.0（True=VSC引擎，False=legacy传统技术）
        compensate_fn: 训练补偿函数；None 则跳过补偿
        compensate_data: 补偿训练数据
        **kwargs: 覆盖 VMPCConfig 的字段

    Returns:
        ``(compressed_model, stats)`` 元组

    用法::

        # V2.0 压缩（默认）
        model, stats = vmpc_compress(model, profile="small")

        # legacy 模式（传统技术直通）
        model, stats = vmpc_compress(model, profile="small", use_vmpc=False)
    """
    config = VMPCConfig(
        use_vmpc=use_vmpc,
        profile=profile,
        vmpc=kwargs,
    )
    engine = VMPCV2(config)
    compressed, stats = engine.compress(model, return_stats=True)

    # 训练补偿
    if compensate_fn is not None and use_vmpc:
        comp_result = engine.compensate(
            compressed, compensate_fn, compensate_data,
            steps=kwargs.get("compensation_steps", 100),
        )
        stats.compensated = True
        stats.compensation_result = comp_result

    return compressed, stats
