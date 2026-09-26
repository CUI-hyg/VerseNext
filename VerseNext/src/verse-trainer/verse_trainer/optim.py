"""优化器与学习率调度。

- AdamW：CUDA 上自动启用 fused 模式（单 kernel 融合，速度提升明显）。
- Warmup-Cosine：线性 warmup 后余弦衰减至 min_lr，LLM 训练标准配方。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from verse_core import global_registry
from verse_core.config import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig):
    """优化器配置。"""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10000
    """学习率衰减至谷值所用的总步数（warmup 之后的衰减地平线）。"""
    min_lr_ratio: float = 0.1
    """最低学习率为峰值 lr 的比例。"""

    def validate(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr 必须为正数")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1/beta2 必须在 [0, 1)")


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """构建 AdamW；embedding/norm 不做 weight decay，CUDA 上启用 fused。"""
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or "emb" in name.lower():
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        fused=fused,
    )


@global_registry.register_scheduler("warmup_cosine")
class WarmupCosineScheduler:
    """线性 warmup + 余弦衰减。"""

    def __init__(self, optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, step: int) -> float:
        """按当前 step 更新 lr，返回更新后的峰值 lr。"""
        cfg = self.config
        if step < cfg.warmup_steps:
            scale = (step + 1) / max(cfg.warmup_steps, 1)
        else:
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            scale = cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * min(progress, 1.0))
            )
        lr = cfg.lr * scale
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def build_lr_scheduler(name: str, optimizer: torch.optim.Optimizer, config: OptimizerConfig):
    cls = global_registry.get("scheduler", name)
    return cls(optimizer, config)
