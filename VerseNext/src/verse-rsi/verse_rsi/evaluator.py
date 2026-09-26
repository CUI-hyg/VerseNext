"""评估器：训练一个个体并计算适应度。

适应度设计（越小越好）::

    fitness = eval_loss + speed_penalty

其中 speed_penalty = speed_weight * (ref_tps / tps)，使「更准」与「更快」
之间可调权衡——这正是训练性能优化与模型架构优化的结合点。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from verse_core import EventBus, get_logger
from verse_core.config import BaseConfig
from verse_rsi.individual import Individual
from verse_trainer.data import DataConfig
from verse_trainer.trainer import Trainer, TrainerConfig

logger = get_logger("rsi.evaluator")


@dataclass
class EvaluatorConfig(BaseConfig):
    """个体评估配置（复用 TrainerConfig 的其余字段作为基础训练设置）。"""

    base_trainer: TrainerConfig = None  # type: ignore[assignment]
    """基础训练配置；各代训练预算（steps）通常远小于正式训练。"""

    speed_weight: float = 0.05
    """速度项权重；0 表示只看 eval_loss。"""
    reference_tps: float | None = None
    """速度基准 tokens/s；None 表示用第一个个体实测值。"""
    device: str = "auto"

    def validate(self) -> None:
        if self.base_trainer is None:
            self.base_trainer = TrainerConfig()


class FitnessEvaluator:
    """在共享数据上训练并评估单个个体。"""

    def __init__(self, config: EvaluatorConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self.event_bus = event_bus or EventBus()
        self._reference_tps = config.reference_tps

    def evaluate(self, individual: Individual, train_tokens, eval_tokens) -> Individual:
        base = self.config.base_trainer
        # 个体覆盖项 -> TrainerConfig（data 字段单独处理嵌套覆盖）
        overrides = dict(individual.trainer_overrides)
        data_overrides = overrides.pop("data", None)
        cfg: TrainerConfig = base.merge(**overrides)
        if data_overrides:
            cfg.data = cfg.data.merge(**data_overrides)
        # 个体的架构配置覆盖基础模型配置
        cfg.model = individual.model_config

        trainer = Trainer(cfg, event_bus=self.event_bus)
        summary = trainer.train(train_tokens, eval_tokens)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        eval_loss = summary.get("best_eval_loss") or summary["final_loss"]
        tps = summary["tokens_seen"] / max(summary["wall_time_s"], 1e-8)

        if self._reference_tps is None:
            self._reference_tps = tps
            speed_penalty = 0.0
        else:
            speed_penalty = self.config.speed_weight * (self._reference_tps / max(tps, 1e-8) - 1.0)
            speed_penalty = max(speed_penalty, 0.0)

        individual.fitness = eval_loss + speed_penalty
        individual.metrics = {
            "eval_loss": eval_loss,
            "final_loss": summary["final_loss"],
            "tokens_per_second": tps,
            "speed_penalty": speed_penalty,
            "num_params": trainer.model.num_parameters(),
            "wall_time_s": summary["wall_time_s"],
        }
        logger.info(
            "个体 %s | fitness %.4f (eval_loss %.4f, speed_penalty %.4f, %.1fk params)",
            individual.id, individual.fitness, eval_loss, speed_penalty,
            individual.metrics["num_params"] / 1000,
        )
        self.event_bus.emit("rsi/individual_evaluated", id=individual.id, **individual.metrics)
        return individual
