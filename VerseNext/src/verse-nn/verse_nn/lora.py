"""LoRA（Low-Rank Adaptation）微调支持。

原理：冻结原权重 W，训练低秩增量 ``ΔW = (B @ A) * scale``，
``forward = W x + (B @ A) x * scale``。仅训练 A/B（参数量通常 <1%），
适配器可独立保存/加载/合并。

用法::

    from verse_nn.lora import apply_lora, mark_only_lora_trainable, lora_state_dict

    model = VerseTransformer(cfg)
    apply_lora(model, LoRAConfig(r=8, lora_alpha=16))
    mark_only_lora_trainable(model)   # 冻结主干
    # ... 训练 ...
    torch.save(lora_state_dict(model), "adapter.pt")   # 只存适配器
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from verse_core.config import BaseConfig


@dataclass
class LoRAConfig(BaseConfig):
    """LoRA 超参。"""

    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: list[str] = None  # type: ignore[assignment]
    """目标模块名的正则列表；None 表示默认作用于注意力投影。"""

    def validate(self) -> None:
        if self.r <= 0:
            raise ValueError("LoRA rank r 必须为正数")
        if self.target_modules is None:
            # 覆盖融合前后的投影命名
            self.target_modules = ["q_proj", "kv_proj", "o_proj", "qkv_proj", "gate_up_proj"]

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r


class LoRALinear(nn.Module):
    """包装 ``nn.Linear``，追加低秩旁路。

    原权重保持冻结；训练/合并均不改动原参数张量。
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.scaling = config.scaling

        self.lora_A = nn.Parameter(torch.zeros(config.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, config.r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)  # B=0 保证初始时 ΔW=0，不破坏原模型
        self.dropout = nn.Dropout(config.lora_dropout)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            out = out + delta * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """把 ΔW 合并进原权重（推理加速，之后不可再分离训练）。"""
        if self.merged:
            return
        self.base.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self.merged = False


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """按 ``target_modules`` 正则替换 Linear 为 LoRALinear，就地修改并返回。"""
    pattern = re.compile("|".join(f"({t})" for t in config.target_modules))
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and pattern.search(child_name):
                setattr(parent, child_name, LoRALinear(child, config))
                replaced += 1
    if replaced == 0:
        raise ValueError(f"未匹配到任何目标模块: {config.target_modules}")
    return model


def mark_only_lora_trainable(model: nn.Module) -> None:
    """冻结除 LoRA 参数外的所有参数（含原 Linear 权重）。"""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只提取 LoRA 参数（适配器体积通常 <1% 主干）。"""
    return {name: p.detach().cpu() for name, p in model.named_parameters() if "lora_" in name}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """把适配器权重载入已 apply_lora 的模型。"""
    own = {name: p for name, p in model.named_parameters() if "lora_" in name}
    missing = set(own) - set(state)
    if missing:
        raise KeyError(f"适配器缺少参数: {sorted(missing)[:5]}")
    for name, tensor in state.items():
        own[name].data.copy_(tensor.to(own[name].device))


def merge_lora(model: nn.Module) -> nn.Module:
    """合并全部 LoRA 增量到主干权重（用于导出完整模型）。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model
