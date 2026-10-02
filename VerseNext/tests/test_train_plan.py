"""接续训练（训练阶段链）测试。

覆盖三件事：
1. **LR 连续**：阶段之间不重开 warmup，整条链共用一条 warmup-cosine 曲线。
2. **换数据集**：每阶段用自己的数据，模型/优化器常驻，步号区间首尾相接。
3. **续训**：跳过已完成的阶段、中途阶段接着跑，CPU 上与不间断训练逐位一致；
   链结构/数据变化必须被拦住。
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from verse_core import EventBus
from verse_nn.transformer import VerseTransformerConfig
from verse_trainer.data import DataConfig
from verse_trainer.optim import OptimizerConfig
from verse_trainer.plan import (
    ChainPlan,
    StageSpec,
    TrainStage,
    chain_fingerprint,
    check_chain_resume,
    load_chain_plan,
    stage_bounds,
)
from verse_trainer.trainer import Trainer, TrainerConfig

VOCAB = 128


@pytest.fixture(autouse=True)
def _isolated_ckpt_dir(tmp_path, monkeypatch):
    """把工作目录切到临时目录：阶段产物默认落在相对路径 ``checkpoints/``，
    不隔离就会把测试 checkpoint 写进仓库。"""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _tokens(n: int = 3000, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    motif = [rng.randrange(VOCAB) for _ in range(32)]
    out: list[int] = []
    while len(out) < n:
        out.extend(motif)
    return out[:n]


def _config(steps: int = 10, seq_len: int = 16, batch_size: int = 2,
            warmup: int = 4, lr: float = 1e-3, device: str = "cpu",
            ckpt_dir: str = "checkpoints") -> TrainerConfig:
    """小模型配置。

    ``device`` 默认强制 CPU：这些用例验证的是阶段链的**调度语义**（步号区间、
    LR 连续性、续训逐位一致），加速器上的归约本身有 ~1e-7 的非确定性，会掩盖
    真正的逻辑回归。
    """
    return TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=VOCAB, d_model=32, num_layers=1, num_heads=2, max_seq_len=64
        ),
        data=DataConfig(batch_size=batch_size, seq_len=seq_len),
        optim=OptimizerConfig(lr=lr, warmup_steps=warmup, max_steps=steps),
        steps=steps,
        log_every=1,
        eval_every=0,
        ckpt_every=0,
        device=device,
        ckpt_dir=ckpt_dir,
    )


def _stage(name: str, steps: int, tokens, **kwargs) -> TrainStage:
    return TrainStage(name=name, steps=steps, train_source=tokens, **kwargs)


# --------------------------------------------------------------- 基础工具


def test_stage_bounds_are_contiguous():
    stages = [_stage("a", 3, _tokens(500)), _stage("b", 5, _tokens(500, 1))]
    assert stage_bounds(stages) == [(0, 3), (3, 8)]
    assert stage_bounds(stages, base=100) == [(100, 103), (103, 108)]


def test_chain_fingerprint_changes_with_data():
    a = _tokens(500, seed=1)
    b = _tokens(500, seed=2)
    same = chain_fingerprint([_stage("s", 4, a)])
    assert same == chain_fingerprint([_stage("s", 4, a)])
    assert same["sha256"] != chain_fingerprint([_stage("s", 4, b)])["sha256"]
    # 阶段名/步数变化也要改变指纹
    assert same["sha256"] != chain_fingerprint([_stage("t", 4, a)])["sha256"]
    assert same["sha256"] != chain_fingerprint([_stage("s", 5, a)])["sha256"]


# ------------------------------------------------------------------- plan


def test_load_chain_plan_parses_stages(tmp_path):
    path = tmp_path / "plan.yaml"
    path.write_text(
        "horizon_steps: 20\n"
        "stages:\n"
        "  - name: general\n"
        "    data: a.txt\n"
        "    steps: 8\n"
        "    text_column: text\n"
        "    lr: 3.0e-4\n"
        "    seq_len: 64\n"
        "  - name: domain\n"
        "    steps: 12\n"
        "    lr: 1.0e-4\n",
        encoding="utf-8",
    )
    plan = load_chain_plan(path)
    assert isinstance(plan, ChainPlan)
    assert [s.name for s in plan.stages] == ["general", "domain"]
    assert plan.total_steps() == 20
    assert plan.horizon_steps == 20
    assert plan.stages[0].text_kwargs() == {"text_column": "text"}
    assert plan.stages[1].text_kwargs() is None


def test_load_chain_plan_rejects_unknown_key(tmp_path):
    path = tmp_path / "plan.yaml"
    path.write_text("stages:\n  - name: a\n    steps: 1\n    lr_typo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知配置项"):
        load_chain_plan(path)


def test_load_chain_plan_rejects_duplicate_and_bad_steps(tmp_path):
    dup = tmp_path / "dup.yaml"
    dup.write_text(
        "stages:\n  - name: a\n    steps: 1\n  - name: a\n    steps: 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="重复"):
        load_chain_plan(dup)

    bad = tmp_path / "bad.yaml"
    bad.write_text("stages:\n  - name: a\n    steps: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="steps"):
        load_chain_plan(bad)


def test_chain_plan_to_dict_roundtrips_through_yaml(tmp_path):
    plan = ChainPlan(stages=[StageSpec(name="a", steps=2), StageSpec(name="b", steps=3)])
    path = tmp_path / "out.yaml"
    plan.save(path)
    assert load_chain_plan(path).total_steps() == 5


# ------------------------------------------------------------- 阶段链训练


def test_chain_runs_all_stages_and_accumulates_tokens():
    stages = [
        _stage("a", 4, _tokens(1000, 1), data_config=DataConfig(batch_size=2, seq_len=16)),
        _stage("b", 4, _tokens(1000, 2), data_config=DataConfig(batch_size=2, seq_len=16)),
    ]
    trainer = Trainer(_config(steps=8))
    summary = trainer.train_stages(stages)

    assert trainer.global_step == 8
    assert [s["name"] for s in summary["stages"]] == ["a", "b"]
    assert [(s["start_step"], s["end_step"]) for s in summary["stages"]] == [(0, 4), (4, 8)]
    assert all(not s["skipped"] for s in summary["stages"])
    # 每步 2*16=32 token，共 8 步
    assert summary["tokens_seen"] == 8 * 32
    assert summary["stage"] == "chain"


def test_chain_lr_is_continuous_with_flat_run():
    """阶段链的逐步 LR 必须与等价单段平跑完全一致（不重开 warmup）。"""
    total = 12

    flat_bus = EventBus()
    flat_lrs: list[float] = []
    flat_bus.subscribe("train/step_end", lambda e: flat_lrs.append(e.payload["lr"]))
    Trainer(_config(steps=total, warmup=4), event_bus=flat_bus).train(_tokens(1000))

    chain_bus = EventBus()
    chain_lrs: list[float] = []
    chain_bus.subscribe("train/step_end", lambda e: chain_lrs.append(e.payload["lr"]))
    trainer = Trainer(_config(steps=total, warmup=4), event_bus=chain_bus)
    trainer.train_stages([
        _stage("a", 5, _tokens(1000, 1)),
        _stage("b", 7, _tokens(1000, 2)),
    ])

    assert len(flat_lrs) == len(chain_lrs) == total
    assert chain_lrs == pytest.approx(flat_lrs, rel=0, abs=0)


def test_chain_horizon_defaults_to_sum_of_steps():
    trainer = Trainer(_config(steps=99))
    trainer.train_stages([_stage("a", 3, _tokens(500)), _stage("b", 4, _tokens(500, 1))])
    assert trainer.config.optim.max_steps == 7


def test_stage_overrides_apply_and_do_not_leak_between_iterators(monkeypatch):
    """每阶段的 seq_len/batch_size/lr/grad_accum 生效，且互不追溯影响。"""
    made: list = []
    original = Trainer._as_batch_source

    def spy(self, source, data_config=None):
        result = original(self, source, data_config)
        made.append(result)
        return result

    monkeypatch.setattr(Trainer, "_as_batch_source", spy)

    stages = [
        _stage("a", 2, _tokens(2000, 1),
               data_config=DataConfig(batch_size=2, seq_len=16), lr=1e-3, grad_accum_steps=1),
        _stage("b", 2, _tokens(2000, 2),
               data_config=DataConfig(batch_size=4, seq_len=32), lr=5e-4, grad_accum_steps=2),
    ]
    trainer = Trainer(_config(steps=4))
    lrs: list[float] = []
    trainer.event_bus.subscribe("train/step_end", lambda e: lrs.append(e.payload["lr"]))
    trainer.train_stages(stages)

    assert len(made) == 2
    assert made[0].config is not made[1].config, "两阶段必须用各自的 DataConfig 实例"
    assert (made[0].config.seq_len, made[0].config.batch_size) == (16, 2)
    assert (made[1].config.seq_len, made[1].config.batch_size) == (32, 4)
    # 阶段 1 的迭代器不能被阶段 2 追溯性改写
    assert made[0].config.seq_len == 16
    # warmup=4：step0 -> 1e-3*0.25；step2/3 走阶段 2 的 5e-4
    assert lrs[0] == pytest.approx(1e-3 * 0.25)
    assert lrs[2] == pytest.approx(5e-4 * 0.75)
    assert lrs[3] == pytest.approx(5e-4)
    assert trainer.config.grad_accum_steps == 2


def test_chain_emits_stage_events():
    events: list[tuple[str, dict]] = []
    bus = EventBus()
    for name in ("train/start", "train/stage_start", "train/stage_end"):
        bus.subscribe(name, lambda e, _n=name: events.append((_n, e.payload)))

    trainer = Trainer(_config(steps=6), event_bus=bus)
    trainer.train_stages([_stage("a", 2, _tokens(500)), _stage("b", 4, _tokens(500, 1))])

    names = [n for n, _ in events]
    assert names.count("train/stage_start") == 2
    assert names.count("train/stage_end") == 2
    start = next(p for n, p in events if n == "train/start")
    assert start["total_steps"] == 6 and start["stage"] == "chain"

    first = next(p for n, p in events if n == "train/stage_start")
    assert first == {"index": 0, "total": 2, "name": "a",
                     "start_step": 0, "end_step": 2, "resumed": False}


# ------------------------------------------------------------------- 续训


def test_chain_resume_skips_done_stage_and_is_bitwise_continuous(tmp_path):
    """中途续训：阶段 0 跳过、阶段 1 接续，loss 历史与不间断训练逐位一致。"""
    data_a, data_b = _tokens(1000, 1), _tokens(1000, 2)
    ckpt_dir = str(tmp_path / "ckpts")

    def cfg() -> TrainerConfig:
        return _config(steps=10, warmup=3, ckpt_dir=ckpt_dir)

    def stages():
        return [
            _stage("a", 4, data_a, data_config=DataConfig(batch_size=2, seq_len=16)),
            _stage("b", 6, data_b, data_config=DataConfig(batch_size=2, seq_len=16)),
        ]

    full = Trainer(cfg())
    full.train_stages(stages())
    assert full.global_step == 10

    # 同一份计划跑到阶段 b 的第 7 步时中断
    partial = Trainer(cfg())
    partial.event_bus.subscribe(
        "train/step_end",
        lambda e: setattr(partial, "_stop_requested", True) if e.payload["step"] == 7 else None,
    )
    partial.train_stages(stages())
    assert partial.global_step == 7
    ckpt = partial.finalize("chain", path=tmp_path / "mid.pt")

    resumed = Trainer(cfg())
    resumed.load_checkpoint(ckpt)
    assert resumed.global_step == 7
    assert resumed._restored_chain_plan is not None, "链计划必须随 checkpoint 一起保存"

    base = check_chain_resume(resumed._restored_chain_plan, stages(), checkpoint_step=7)
    assert base == 0
    summary = resumed.train_stages(stages(), plan_base=base)

    assert resumed.global_step == 10
    # 阶段 a 已完成 -> 跳过；阶段 b 从中断处续跑
    assert [s["skipped"] for s in summary["stages"]] == [True, False]
    assert summary["stages"][1]["resumed"] is True
    # 逐位连续：与不间断跑完整条链的 loss 历史完全一致
    assert resumed._train_hist == full._train_hist


def test_check_chain_resume_rejects_plan_change():
    stages = [_stage("a", 3, _tokens(500)), _stage("b", 3, _tokens(500, 1))]
    saved = {"base": 0, "names": ["a", "b"], "steps": [3, 3],
             "fingerprint": chain_fingerprint(stages)}
    # 结构一致 -> 通过，返回 base
    assert check_chain_resume(saved, stages, checkpoint_step=4) == 0
    # 步数变化 -> 报错
    with pytest.raises(RuntimeError, match="步数变化"):
        check_chain_resume(saved, [_stage("a", 3, _tokens(500)),
                                   _stage("b", 9, _tokens(500, 1))], checkpoint_step=4)
    # 断点早于链起点 -> 报错
    with pytest.raises(RuntimeError, match="早于链起点"):
        check_chain_resume(saved, stages, checkpoint_step=-1)


def test_check_chain_resume_guards_data_change():
    stages = [_stage("a", 3, _tokens(500, 1))]
    saved = {"base": 0, "names": ["a"], "steps": [3],
             "fingerprint": chain_fingerprint(stages)}
    changed = [_stage("a", 3, _tokens(500, 99))]
    with pytest.raises(RuntimeError, match="指纹变化"):
        check_chain_resume(saved, changed, checkpoint_step=1)
    # 放行后返回 base
    assert check_chain_resume(saved, changed, checkpoint_step=1,
                              allow_data_change=True) == 0


def test_check_chain_resume_from_single_stage_checkpoint():
    """checkpoint 来自单段训练（无链计划）时，以断点为链起点。"""
    stages = [_stage("a", 3, _tokens(500))]
    assert check_chain_resume(None, stages, checkpoint_step=120) == 120


def test_chain_resume_skips_completed_stages():
    """已完整跑过的阶段在重跑时必须被跳过。"""
    stages = [_stage("a", 2, _tokens(500)), _stage("b", 2, _tokens(500, 1))]
    trainer = Trainer(_config(steps=4))
    trainer.global_step = 2          # 阶段 a 已完成
    summary = trainer.train_stages(stages)
    assert summary["stages"][0]["skipped"] is True
    assert summary["stages"][1]["skipped"] is False
    assert trainer.global_step == 4


def test_chain_stops_on_early_stop():
    """早停终止整条链，后续阶段不再运行。"""
    cfg = _config(steps=10)
    cfg.eval_every = 2
    cfg.eval_batches = 1
    cfg.early_stop_patience = 1
    stages = [_stage("a", 2, _tokens(500), eval_source=_tokens(500, 7)),
              _stage("b", 8, _tokens(500, 1))]
    trainer = Trainer(cfg)
    # 把历史最优压到不可能被超越，保证第 2 步的评估立刻触发早停
    trainer._best_loss = -1e9
    summary = trainer.train_stages(stages)

    assert summary["early_stopped"] is True
    assert [s["name"] for s in summary["stages"]] == ["a"], "阶段 b 不应开始"
    assert trainer.global_step == 2
