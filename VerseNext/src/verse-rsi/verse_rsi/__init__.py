"""VerseNext RSI（递归自进化）包。

RSI = Recursive Self-Improvement：框架对「模型 + 训练配置」整体作为个体
（Individual）进行 评估 -> 选择 -> 变异 -> 再评估 的递归循环，
每代基于上代最优个体继续进化，实现训练策略与模型架构的共同自优化。
"""

from verse_rsi.individual import Individual
from verse_rsi.evaluator import FitnessEvaluator, EvaluatorConfig
from verse_rsi.mutator import (
    HyperparamMutator,
    ArchMutator,
    CombinedMutator,
    MutatorConfig,
)
from verse_rsi.population import Population
from verse_rsi.evolve import Evolver, EvolveConfig

__all__ = [
    "Individual",
    "FitnessEvaluator",
    "EvaluatorConfig",
    "HyperparamMutator",
    "ArchMutator",
    "CombinedMutator",
    "MutatorConfig",
    "Population",
    "Evolver",
    "EvolveConfig",
]
__version__ = "0.1.0"
