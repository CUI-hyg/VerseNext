"""Individual：一个待进化个体 = 模型架构配置 + 训练配置。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from verse_core.config import BaseConfig
from verse_nn.transformer import VerseTransformerConfig
from verse_trainer.trainer import TrainerConfig


@dataclass
class Individual:
    """进化个体。

    fitness 为评估得分（越小越好，取验证 loss 与速度的加权组合）；
    未评估时为 None。``lineage`` 记录祖先链，便于追溯进化路径。
    """

    model_config: VerseTransformerConfig
    trainer_overrides: dict = field(default_factory=dict)
    """对 TrainerConfig 的覆盖项（lr、batch_size、grad_accum_steps 等）。"""
    fitness: float | None = None
    metrics: dict = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    """祖先 id 链（不含自身）。"""

    @property
    def id(self) -> str:
        payload = json.dumps(
            {"model": self.model_config.to_dict(), "trainer": self.trainer_overrides},
            sort_keys=True, default=str,
        )
        return hashlib.md5(payload.encode()).hexdigest()[:10]

    def copy(self) -> "Individual":
        return Individual(
            model_config=self.model_config,
            trainer_overrides=dict(self.trainer_overrides),
            lineage=self.lineage + [self.id],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_config": self.model_config.to_dict(),
            "trainer_overrides": self.trainer_overrides,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        return cls(
            model_config=VerseTransformerConfig.from_dict(data["model_config"]),
            trainer_overrides=data.get("trainer_overrides", {}),
            fitness=data.get("fitness"),
            metrics=data.get("metrics", {}),
            lineage=data.get("lineage", []),
        )
