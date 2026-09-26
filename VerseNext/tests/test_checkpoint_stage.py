"""checkpoint 生命周期 / 阶段分级 / 参数口径测试。"""

import random

import pytest
import torch

from verse_nn.transformer import VerseTransformer, VerseTransformerConfig, count_parameters
from verse_trainer.data import DataConfig
from verse_trainer.optim import OptimizerConfig
from verse_trainer.trainer import Trainer, TrainerConfig

VOCAB = 128


def _tokens(n: int = 2000) -> list[int]:
    rng = random.Random(0)
    return [rng.randrange(VOCAB) for _ in range(n)]


def _trainer(tmp_path, steps: int = 3, **kwargs) -> Trainer:
    cfg = TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=VOCAB, d_model=32, num_layers=2, num_heads=4,
            max_seq_len=64, tie_weights=True,
        ),
        data=DataConfig(batch_size=2, seq_len=16),
        optim=OptimizerConfig(lr=1e-3, warmup_steps=1, max_steps=steps),
        steps=steps, log_every=1, eval_every=0, ckpt_every=0,
        ckpt_dir=str(tmp_path),
        **kwargs,
    )
    return Trainer(cfg)


def test_count_parameters_dedupes_tied_weights():
    cfg = VerseTransformerConfig(vocab_size=64, d_model=32, num_layers=1,
                                 num_heads=4, tie_weights=True)
    model = VerseTransformer(cfg)
    unique, total = count_parameters(model)
    assert unique == model.num_parameters()
    # 绑定权重在 state_dict 中出现两次（lm_head + token_emb）
    assert total > unique
    assert total - unique == model.token_emb.weight.numel()


def test_finalize_stage_naming_and_cleanup(tmp_path):
    trainer = _trainer(tmp_path, steps=2)
    trainer.train(_tokens())
    # 造两个周期 checkpoint
    step1 = trainer.save_checkpoint(str(tmp_path / "step_1.pt"))
    step2 = trainer.save_checkpoint(str(tmp_path / "step_2.pt"))
    assert step1.exists() and step2.exists()

    final = trainer.finalize("pretrain")
    assert final.name == "verse_transformer_pretrain_final.pt"
    assert final.exists()
    # 周期 checkpoint 被清理，final 保留
    assert not step1.exists() and not step2.exists()
    assert final.exists()


def test_finalize_payload_metadata(tmp_path):
    trainer = _trainer(tmp_path, steps=1)
    trainer.train(_tokens())
    out = trainer.finalize("finetune")
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["stage"] == "finetune"
    assert payload["final"] is True
    assert payload["param_count"] == trainer.model.num_parameters()
    assert payload["tie_weights"] is True
    # 默认保留优化器状态（可续训）
    assert payload.get("optimizer_state") is not None


def test_finalize_without_optimizer_state(tmp_path):
    trainer = _trainer(tmp_path, steps=1, final_keep_optimizer=False)
    trainer.train(_tokens())
    out = trainer.finalize("pretrain")
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload.get("optimizer_state") is None
    # 无优化器状态也能加载（不报错）
    trainer.load_checkpoint(out)


def test_load_checkpoint_detects_arch_mismatch(tmp_path):
    trainer = _trainer(tmp_path, steps=1)
    trainer.train(_tokens())
    out = trainer.finalize("pretrain")

    other = _trainer(tmp_path / "other", steps=1)
    # 篡改元数据模拟架构不匹配（实际模型参数量与声明的不同）
    payload = torch.load(out, map_location="cpu", weights_only=False)
    payload["param_count"] = payload["param_count"] + 1
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(RuntimeError, match="参数量不匹配"):
        other.load_checkpoint(bad)


def test_cleanup_intermediate_keeps_specified(tmp_path):
    trainer = _trainer(tmp_path, steps=1)
    (tmp_path / "step_9.pt").write_bytes(b"x")
    (tmp_path / "step_10.pt").write_bytes(b"x")
    keep = tmp_path / "keep_me.pt"
    keep.write_bytes(b"x")
    removed = trainer.cleanup_intermediate(keep={keep})
    assert removed == 2
    assert keep.exists()
    assert not (tmp_path / "step_9.pt").exists()
