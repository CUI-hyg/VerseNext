"""变异算子：RSI 的「自进化」核心。

两类算子（均可通过 ``mutator/<name>`` 注册扩展）：
- HyperparamMutator：训练超参变异（lr / batch_size / grad_accum）
- ArchMutator：架构变异（层数 / 头数 / 隐藏维度 / KV 头数）

CombinedMutator 将多算子组合，按概率各自触发。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from verse_core import get_logger
from verse_core.config import BaseConfig
from verse_core.registry import global_registry, register_mutator
from verse_rsi.individual import Individual

logger = get_logger("rsi.mutator")


@dataclass
class MutatorConfig(BaseConfig):
    """变异强度配置。"""

    lr_mutation_scale: tuple[float, float] = (0.5, 2.0)
    batch_size_choices: tuple[int, ...] = (4, 8, 16, 32)
    layer_delta_choices: tuple[int, ...] = (-2, -1, 1, 2)
    head_choices: tuple[int, ...] = (2, 4, 8, 16)
    d_model_choices: tuple[int, ...] = (256, 384, 512, 768, 1024)
    arch_mutation_prob: float = 0.5
    hyper_mutation_prob: float = 0.8

    def validate(self) -> None:
        if not 0.0 <= self.arch_mutation_prob <= 1.0:
            raise ValueError("arch_mutation_prob 需在 [0,1]")
        if not 0.0 <= self.hyper_mutation_prob <= 1.0:
            raise ValueError("hyper_mutation_prob 需在 [0,1]")


class BaseMutator:
    """变异算子基类：输入一个个体，返回变异后的新个体（原个体不变）。"""

    name = "base"

    def mutate(self, individual: Individual, config: MutatorConfig, rng: random.Random) -> Individual:
        raise NotImplementedError


@register_mutator("hyperparam")
class HyperparamMutator(BaseMutator):
    """训练超参变异：lr 乘性扰动 + batch_size 离散跳变。"""

    name = "hyperparam"

    def mutate(self, individual: Individual, config: MutatorConfig, rng: random.Random) -> Individual:
        child = individual.copy()
        overrides = dict(child.trainer_overrides)

        if rng.random() < config.hyper_mutation_prob:
            base_lr = overrides.get(
                "optim", {}
            ).get("lr", None) or _current_lr(individual)
            lr_scale = rng.uniform(*config.lr_mutation_scale)
            new_lr = _clamp(base_lr * lr_scale, 1e-5, 1e-2)
            overrides.setdefault("optim", {})
            overrides["optim"]["lr"] = new_lr
            logger.debug("lr 变异: %.2e -> %.2e", base_lr, new_lr)

        if rng.random() < 0.3:
            overrides.setdefault("data", {})
            overrides["data"]["batch_size"] = rng.choice(config.batch_size_choices)

        child.trainer_overrides = overrides
        return child


def _current_lr(individual: Individual) -> float:
    return individual.trainer_overrides.get("optim", {}).get("lr", 3e-4)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@register_mutator("arch")
class ArchMutator(BaseMutator):
    """架构变异：层数/头数/隐藏维度/KV 头数。

    变异时保持架构合法性：d_model 被头数整除、num_kv_heads 整除 num_heads。
    """

    name = "arch"

    def mutate(self, individual: Individual, config: MutatorConfig, rng: random.Random) -> Individual:
        if rng.random() >= config.arch_mutation_prob:
            return individual.copy()

        child = individual.copy()
        m = child.model_config
        action = rng.choice(["layers", "heads", "d_model", "kv_heads"])

        if action == "layers":
            m.num_layers = _clamp(m.num_layers + rng.choice(config.layer_delta_choices), 1, 24)
        elif action == "heads":
            heads = rng.choice(config.head_choices)
            if m.d_model % heads == 0:
                m.num_heads = heads
                if m.num_kv_heads is not None and m.num_heads % m.num_kv_heads != 0:
                    m.num_kv_heads = heads
        elif action == "d_model":
            d_model = rng.choice(config.d_model_choices)
            if d_model % m.num_heads == 0:
                m.d_model = d_model
        elif action == "kv_heads":
            if m.num_kv_heads is None:
                m.num_kv_heads = max(1, m.num_heads // 2)  # MHA -> GQA
            else:
                m.num_kv_heads = rng.choice([k for k in (1, 2, 4, 8) if m.num_heads % k == 0])

        m.validate()
        logger.debug("架构变异(%s): %s", action, child.id)
        return child


@register_mutator("combined")
class CombinedMutator(BaseMutator):
    """先做超参变异，再做架构变异（可叠加、顺序可扩展）。"""

    name = "combined"

    def __init__(self) -> None:
        self._hyper = HyperparamMutator()
        self._arch = ArchMutator()

    def mutate(self, individual: Individual, config: MutatorConfig, rng: random.Random) -> Individual:
        child = self._hyper.mutate(individual, config, rng)
        child = self._arch.mutate(child, config, rng)
        return child


def build_mutator(name: str) -> BaseMutator:
    return global_registry.create("mutator", name)
