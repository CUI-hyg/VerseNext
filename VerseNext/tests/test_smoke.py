"""端到端冒烟测试：训练 + RSI 进化短循环。"""

import random

import pytest
import torch

from verse_nn.transformer import VerseTransformer, VerseTransformerConfig
from verse_rsi.evolve import EvolveConfig, Evolver
from verse_rsi.evaluator import EvaluatorConfig
from verse_rsi.individual import Individual
from verse_trainer.data import DataConfig, TokenBatchIterator
from verse_trainer.optim import OptimizerConfig
from verse_trainer.trainer import Trainer, TrainerConfig


VOCAB = 128


def _tokens(n: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    motif = [rng.randrange(VOCAB) for _ in range(32)]
    out = []
    while len(out) < n:
        out.extend(motif if rng.random() < 0.8 else [rng.randrange(VOCAB) for _ in range(8)])
    return out[:n]


def _small_trainer_config(steps: int = 20) -> TrainerConfig:
    return TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4, max_seq_len=128
        ),
        data=DataConfig(batch_size=4, seq_len=64),
        optim=OptimizerConfig(lr=3e-4, warmup_steps=5, max_steps=steps),
        steps=steps,
        log_every=10,
        eval_every=10,
        eval_batches=3,
        ckpt_every=0,
    )


def test_transformer_forward_loss_decreases():
    config = _small_trainer_config(steps=40)
    trainer = Trainer(config)
    assert isinstance(trainer.model, VerseTransformer)
    tokens = _tokens()
    summary = trainer.train(tokens)
    # 训练后 loss 应明显低于随机初始化水平（~ln(128)≈4.85）
    assert summary["final_loss"] < 4.0


def test_checkpoint_roundtrip(tmp_path):
    config = _small_trainer_config()
    trainer = Trainer(config)
    path = trainer.save_checkpoint(str(tmp_path / "ckpt.pt"))

    trainer2 = Trainer(config)
    trainer2.load_checkpoint(path)
    assert trainer2.global_step == trainer.global_step


def test_batch_iterator_shapes():
    it = TokenBatchIterator(_tokens(2000), DataConfig(batch_size=4, seq_len=32))
    x, y = it.next_batch()
    assert x.shape == (4, 32) and y.shape == (4, 32)
    # targets 是 inputs 右移一位
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_config_yaml_roundtrip(tmp_path):
    config = _small_trainer_config()
    path = tmp_path / "cfg.yaml"
    config.save(path)
    loaded = TrainerConfig.load(path)
    assert loaded.model.d_model == 64
    assert loaded.steps == config.steps


def test_rsi_evolution_finds_individual(tmp_path):
    tokens = _tokens()
    eval_tokens = tokens[-800:]
    config = EvolveConfig(
        generations=2,
        population_size=2,
        elite_k=1,
        evaluator_config=EvaluatorConfig(base_trainer=_small_trainer_config(steps=15)),
        archive_path=str(tmp_path / "rsi_pop.json"),
    )

    evolver = Evolver(config)
    best = evolver.run(tokens, eval_tokens)

    assert best.fitness is not None
    assert len(evolver.population) >= 5  # 1 seed + 每代 2 个子代
    assert evolver.population.best().fitness == pytest.approx(best.fitness)


def test_mutators_produce_valid_architectures():
    from verse_rsi.mutator import CombinedMutator, MutatorConfig

    seed = Individual(model_config=VerseTransformerConfig(
        vocab_size=VOCAB, d_model=64, num_layers=2, num_heads=4
    ))
    mutator = CombinedMutator()
    rng = random.Random(0)
    for _ in range(30):
        child = mutator.mutate(seed, MutatorConfig(), rng)
        child.model_config.validate()  # 架构合法性
        assert child.id != seed.id or child.trainer_overrides == seed.trainer_overrides


def test_population_persistence(tmp_path):
    from verse_rsi.population import Population

    ind = Individual(
        model_config=VerseTransformerConfig(vocab_size=VOCAB, d_model=64, num_layers=2),
        fitness=1.23,
    )
    pop = Population()
    pop.add(ind)
    path = tmp_path / "pop.json"
    pop.save(path)
    loaded = Population.load(path)
    assert loaded.best().fitness == pytest.approx(1.23)
    assert loaded.best().id == ind.id
