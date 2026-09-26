"""递归自进化主循环。

每代流程（递归：每代的最优个体作为下一代的父代继续进化）::

    for generation in range(G):
        1. 评估当前种群中未评估的个体（短训练预算）
        2. 选择 topk 作为精英
        3. 对精英做变异生成子代，加入种群
        4. 记录事件与档案；下一代以当前最优为起点重复

进化配置分两个阶段：进化期用短训练预算（few steps）快速比较个体，
胜出者最终可用完整预算重训（见 ``refine_best``）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from verse_core import EventBus, get_logger
from verse_core.config import BaseConfig
from verse_nn.transformer import VerseTransformerConfig
from verse_rsi.evaluator import EvaluatorConfig, FitnessEvaluator
from verse_rsi.individual import Individual
from verse_rsi.mutator import MutatorConfig, build_mutator
from verse_rsi.population import Population
from verse_trainer.trainer import TrainerConfig

logger = get_logger("rsi.evolve")


@dataclass
class EvolveConfig(BaseConfig):
    """进化循环配置。"""

    generations: int = 5
    population_size: int = 4
    """每代产生的子代数量（精英数 = 取 topk 中前 elite_k）。"""
    elite_k: int = 1
    """每代选出的精英数量，作为下一代的父代。"""
    mutator: str = "combined"
    mutator_config: MutatorConfig = field(default_factory=MutatorConfig)
    evaluator_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    seed: int = 42
    archive_path: str = "rsi_archive/population.json"
    """种群档案保存路径（每代结束都会覆盖写）。"""
    refine_best_steps: int = 0
    """>0 时，进化结束后用该步数对最优个体完整重训，返回其训练摘要。"""

    def validate(self) -> None:
        if self.generations <= 0 or self.population_size <= 0:
            raise ValueError("generations / population_size 必须为正数")
        if self.evaluator_config.base_trainer is None:
            self.evaluator_config.base_trainer = TrainerConfig()


class Evolver:
    """递归自进化驱动器。"""

    def __init__(self, config: EvolveConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.evaluator = FitnessEvaluator(config.evaluator_config, self.event_bus)
        self.mutator = build_mutator(config.mutator)
        self.population = Population()
        self.rng = random.Random(config.seed)

    def run(self, train_tokens, eval_tokens, seed_individual: Individual | None = None) -> Individual:
        """执行进化，返回最优个体。"""
        cfg = self.config
        if seed_individual is None:
            seed_individual = Individual(model_config=VerseTransformerConfig())
        self.population.add(seed_individual)

        for generation in range(cfg.generations):
            # 1) 评估所有未评估个体
            unevaluated = [i for i in self.population.individuals.values() if i.fitness is None]
            for individual in unevaluated:
                self.evaluator.evaluate(individual, train_tokens, eval_tokens)

            best = self.population.best()
            assert best is not None
            logger.info(
                "第 %d/%d 代 | 最优 %s fitness %.4f | 种群 %d",
                generation + 1, cfg.generations, best.id, best.fitness, len(self.population),
            )

            # 2) 精英变异 -> 子代（递归：以本代最优为父代继续进化）
            elites = self.population.topk(cfg.elite_k)
            for elite in elites:
                for _ in range(cfg.population_size):
                    child = self.mutator.mutate(elite, cfg.mutator_config, self.rng)
                    if child.id not in self.population.individuals:
                        self.population.add(child)

            # 3) 持久化档案 + 事件
            self.population.save(cfg.archive_path)
            self.event_bus.emit(
                "rsi/generation_end",
                generation=generation + 1,
                best_id=best.id,
                best_fitness=best.fitness,
                population_size=len(self.population),
            )

        best = self.population.best()
        logger.info("进化完成，最优个体 %s (fitness %.4f)", best.id, best.fitness)
        return best

    def refine_best(self, best: Individual, train_tokens, eval_tokens) -> dict:
        """用完整训练预算重训最优个体，返回训练摘要。"""
        # merge 会通过 from_dict 将 dict 覆盖项还原为对应子配置
        cfg: TrainerConfig = TrainerConfig().merge(
            **{**best.trainer_overrides, "model": best.model_config}
        )
        trainer = Trainer(cfg, event_bus=self.event_bus)
        return trainer.train(train_tokens, eval_tokens)
