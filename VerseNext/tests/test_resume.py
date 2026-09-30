"""断点续训测试：完整状态恢复、LR 连续性、逐位一致、数据指纹、save_last。"""

import random

import numpy as np
import torch

from verse_nn.transformer import VerseTransformerConfig
from verse_trainer.data import DataConfig, TokenBatchIterator, fingerprint_source
from verse_trainer.optim import OptimizerConfig
from verse_trainer.trainer import Trainer, TrainerConfig

VOCAB = 128


def _tokens(n: int = 3000, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(VOCAB) for _ in range(n)]


def _cfg(steps: int, **kwargs) -> TrainerConfig:
    base = dict(log_every=1, eval_every=0, ckpt_every=0, seed=11)
    base.update(kwargs)
    return TrainerConfig(
        model=VerseTransformerConfig(vocab_size=VOCAB, d_model=32, num_layers=2,
                                     num_heads=4, max_seq_len=32, tie_weights=True),
        data=DataConfig(batch_size=2, seq_len=16, seed=7),
        optim=OptimizerConfig(lr=1e-3, warmup_steps=2, max_steps=steps),
        steps=steps,
        **base,
    )


# ------------------------------------------------------------- fingerprint

def test_fingerprint_tokens_stable_and_sensitive():
    tokens = np.arange(1000, dtype=np.int32)
    fp1 = fingerprint_source(tokens)
    fp2 = fingerprint_source(tokens.copy())
    assert fp1 == fp2 and fp1["kind"] == "tokens" and fp1["n"] == 1000
    changed = tokens.copy()
    changed[0] = 999
    assert fingerprint_source(changed)["sha256"] != fp1["sha256"]


def test_fingerprint_from_iterator_matches_array():
    tokens = _tokens(500)
    arr = np.asarray(tokens, dtype=np.int32)
    it = TokenBatchIterator(arr, DataConfig(batch_size=2, seq_len=16))
    assert fingerprint_source(it)["sha256"] == fingerprint_source(arr)["sha256"]


def test_fingerprint_examples():
    ds = type("DS", (), {"examples": [([1, 2, 3], [0, 0, 0]), ([4, 5], [0, 0])]})()
    fp = fingerprint_source(ds)
    assert fp["kind"] == "examples" and fp["n"] == 2
    assert fingerprint_source(object()) is None


# --------------------------------------------------------------- resume

def test_resume_restores_full_training_state(tmp_path):
    tokens = _tokens()
    full = Trainer(_cfg(4))
    full.train(tokens, None)

    part = Trainer(_cfg(4))
    part.config.steps = 2
    part.train(tokens, None)
    ckpt = part.save_checkpoint(str(tmp_path / "step_2.pt"))

    name, p_part = next(iter(part.model.named_parameters()))
    exp_before = float(part.optimizer.state[p_part]["exp_avg"].norm())
    best_before = part._best_loss
    hist_before = list(part._train_hist)
    fp_before = part._data_fingerprint
    tokens_before = part._tokens_seen

    resumed = Trainer(_cfg(4))
    resumed.config.steps = 4
    resumed.load_checkpoint(ckpt)

    assert resumed.global_step == 2
    assert abs(resumed._best_loss - best_before) < 1e-12
    assert resumed._train_hist == hist_before
    assert resumed._data_fingerprint == fp_before
    assert resumed._tokens_seen == tokens_before
    name_r, p_r = next(iter(resumed.model.named_parameters()))
    assert name_r == name
    assert abs(float(resumed.optimizer.state[p_r]["exp_avg"].norm()) - exp_before) < 1e-9
    for (n1, a), (n2, b) in zip(part.model.state_dict().items(),
                                resumed.model.state_dict().items()):
        assert n1 == n2 and torch.equal(a, b)
    # 调度器无状态：同 step 的 lr 必须一致
    assert abs(full.scheduler.step(2) - resumed.scheduler.step(2)) < 1e-15


def _accelerator_active() -> bool:
    """当前是否有非 CPU 加速器在跑（NPU/CUDA）。"""
    if torch.cuda.is_available():
        return True
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def test_resume_is_bitwise_continuous(tmp_path):
    """续训后的每一步 loss 与不间断训练一致（不丢失、不折损）。

    CPU 上要求逐位相同。NPU/CUDA 上浮点归约顺序由硬件决定，**两次完全相同的
    独立训练 run 都无法逐位复现**（已实测：同一进程内同种子跑两遍，step-1 loss
    就有 ~1e-7 差异），因此这里放宽为紧容差比较；它仍能抓住「状态丢失/错位」
    这类真实回归（那会造成远大于 1e-4 的偏差）。
    """
    tokens = _tokens()
    full = Trainer(_cfg(6))
    full.train(tokens, None)

    part = Trainer(_cfg(6))
    part.config.steps = 3
    part.train(tokens, None)
    ckpt = part.save_checkpoint(str(tmp_path / "step_3.pt"))

    resumed = Trainer(_cfg(6))
    resumed.config.steps = 6
    resumed.load_checkpoint(ckpt)
    resumed.train(tokens, None)

    assert resumed.global_step == 6
    assert len(resumed._train_hist) == len(full._train_hist)
    if _accelerator_active():
        for (s1, l1), (s2, l2) in zip(resumed._train_hist, full._train_hist):
            assert s1 == s2 and abs(l1 - l2) < 1e-4, (s1, l1, s2, l2)
    else:
        assert resumed._train_hist == full._train_hist
    assert resumed._tokens_seen == full._tokens_seen


def test_load_checkpoint_weights_only(tmp_path):
    tokens = _tokens()
    t = Trainer(_cfg(3))
    t.train(tokens, None)
    ckpt = t.save_checkpoint(str(tmp_path / "step_3.pt"))

    fresh = Trainer(_cfg(3))
    fresh.load_checkpoint(ckpt, resume_training=False)
    assert fresh.global_step == 3
    # 未恢复监控状态
    assert fresh._best_loss == float("inf")
    assert fresh._tokens_seen == 0


# --------------------------------------------------------------- save_last

def test_save_last_checkpoint_and_cleanup(tmp_path):
    cfg = _cfg(4, ckpt_every=2, save_best_only=True, save_last=True,
               ckpt_dir=str(tmp_path))
    trainer = Trainer(cfg)
    trainer.train(_tokens(), None)
    last = tmp_path / f"{cfg.model_name}_last.pt"
    assert last.exists()
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 4
    assert payload.get("optimizer_state") is not None
    # finalize 后 *_last.pt 一并清理，只留 final
    final = trainer.finalize("pretrain")
    assert not last.exists() and final.exists()


def test_eval_every_zero_with_eval_set_does_not_crash():
    """eval_every=0 表示关闭周期评估，即使提供了评估集也不应触发取模。"""
    tokens = _tokens()
    trainer = Trainer(_cfg(2, eval_every=0))
    trainer.train(tokens, tokens)
    assert trainer.global_step == 2
    assert trainer._eval_hist == []


def test_cumulative_tokens_seen_across_resume(tmp_path):
    tokens = _tokens()
    a = Trainer(_cfg(2))
    a.train(tokens, None)
    ckpt = a.save_checkpoint(str(tmp_path / "step_2.pt"))
    b = Trainer(_cfg(4))
    b.load_checkpoint(ckpt)
    summary = b.train(tokens, None)
    per_step = 2 * 16                     # batch_size × seq_len
    assert b._tokens_seen == 4 * per_step
    assert summary["tokens_seen"] == 4 * per_step
    assert summary["tokens_seen_run"] == 2 * per_step
