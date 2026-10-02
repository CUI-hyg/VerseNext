"""训练进度事件与 run.py 进度条（ETA）语义测试。

回归背景：``run.py`` 曾把 **绝对** ``global_step`` 直接喂给 rich 的
``progress.update(completed=...)``，而 ``total`` 取的是 yaml 的 ``steps``。
断点续训时 ``global_step`` 从 checkpoint 的步号起步，于是
``completed > total``：进度条卡在 100%、``remaining`` 为负、``Task.speed``
被首帧的巨大增量冲高，``TimeRemainingColumn`` 恒显示 ``0:00:00``。

修法：trainer 在循环前广播 ``train/start{start_step,total_steps}``，
UI 用「本次 run 的相对区间」换算。本文件同时锁住这两半。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from verse_core import EventBus
from verse_nn.transformer import VerseTransformerConfig
from verse_trainer.data import DataConfig
from verse_trainer.optim import OptimizerConfig
from verse_trainer.trainer import Trainer, TrainerConfig

VOCAB = 128

# CometSpark 是工作区里的下游项目，不在 verse 包的 PYTHONPATH 上
_COMETSPARK = Path(__file__).resolve().parents[2] / "CometSpark"


def _tokens(n: int = 2000, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    motif = [rng.randrange(VOCAB) for _ in range(32)]
    out: list[int] = []
    while len(out) < n:
        out.extend(motif)
    return out[:n]


def _config(steps: int = 4) -> TrainerConfig:
    return TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=VOCAB, d_model=32, num_layers=1, num_heads=2, max_seq_len=64
        ),
        data=DataConfig(batch_size=2, seq_len=16),
        optim=OptimizerConfig(lr=1e-3, warmup_steps=2, max_steps=steps),
        steps=steps,
        log_every=1,
        eval_every=0,
        ckpt_every=0,
    )


# ------------------------------------------------------- trainer 侧事件


def test_train_emits_start_event_from_zero():
    """全新训练：start_step=0，total_steps 等于 cfg.steps。"""
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe("train/start", lambda e: seen.append(e.payload))

    trainer = Trainer(_config(steps=3), event_bus=bus)
    trainer.train(_tokens())

    assert len(seen) == 1, "train/start 每次 run 只应广播一次"
    assert seen[0]["start_step"] == 0
    assert seen[0]["total_steps"] == 3
    assert seen[0]["stage"] == "pretrain"


def test_train_start_event_reports_resume_offset():
    """续训：start_step 必须是恢复后的 global_step，而不是 0。"""
    cfg = _config(steps=6)
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe("train/start", lambda e: seen.append(e.payload))

    trainer = Trainer(cfg, event_bus=bus)
    trainer.global_step = 4          # 模拟 load_checkpoint 之后的起点
    trainer.train(_tokens())

    assert seen[0]["start_step"] == 4
    assert seen[0]["total_steps"] == 6


# ------------------------------------------------------- run.py UI 侧换算


class _FakeProgress:
    """记录 add_task/update 的假 Progress（不依赖真实终端）。"""

    instances: list["_FakeProgress"] = []

    def __init__(self, *columns, **kwargs) -> None:
        self.tasks: list[dict] = []
        self.updates: list[dict] = []
        self.console = _FakeConsole()
        _FakeProgress.instances.append(self)

    def __enter__(self) -> "_FakeProgress":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def add_task(self, description, total=None, **fields):
        self.tasks.append(
            {"description": description, "total": total, "completed": 0, "fields": dict(fields)}
        )
        return len(self.tasks) - 1

    def update(self, task_id, **kwargs) -> None:
        self.updates.append(dict(kwargs))
        task = self.tasks[task_id]
        for key, value in kwargs.items():
            if key in ("total", "completed"):
                task[key] = value
            else:
                task["fields"][key] = value


class _FakeConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args, **kwargs) -> None:
        self.lines.append(" ".join(str(a) for a in args))


@pytest.fixture
def run_module(monkeypatch):
    """导入 CometSpark/run.py 并把它的 Progress 换成假实现。"""
    if not (_COMETSPARK / "run.py").exists():
        pytest.skip("工作区无 CometSpark/run.py")
    pytest.importorskip("rich")
    if str(_COMETSPARK) not in sys.path:
        sys.path.insert(0, str(_COMETSPARK))
    import run  # noqa: PLC0415

    _FakeProgress.instances.clear()
    monkeypatch.setattr(run, "Progress", _FakeProgress)
    return run


def _drive(run, *, start_step: int, total_steps: int, emit_start: bool = True):
    """驱动 _run_training_ui，返回（最终任务状态, 每步 update 的 completed 序列）。"""
    bus_steps = list(range(start_step + 1, total_steps + 1))

    def run_fn(bus: EventBus) -> dict:
        if emit_start:
            bus.emit("train/start", start_step=start_step,
                     total_steps=total_steps, stage="pretrain")
        for step in bus_steps:
            bus.emit("train/step_end", step=step, loss=1.0, lr=1e-4,
                     tokens_per_second=100.0)
        return {"final_loss": 1.0, "steps": total_steps}

    rc = run._run_training_ui("t", [], total_steps, run_fn)
    assert rc == 0
    progress = _FakeProgress.instances[-1]
    # 只有 step 事件携带 loss 字段；据此把 train/start 的初始化 update 排除掉
    completions = [u["completed"] for u in progress.updates if "loss" in u]
    return progress.tasks[0], completions


def test_fresh_run_progress_is_relative(run_module):
    task, completions = _drive(run_module, start_step=0, total_steps=10)
    assert task["total"] == 10
    assert task["completed"] == 10
    assert completions == list(range(1, 11))


def test_resumed_run_progress_is_relative(run_module):
    """核心回归：续训时 completed 不能超过 total。"""
    task, completions = _drive(run_module, start_step=1000, total_steps=1010)
    assert task["total"] == 10, "total 应为本次 run 的步数（end - start）"
    assert task["completed"] == 10
    # 绝对步号 1001..1010 必须换算成相对 1..10，且全程不超过 total
    assert completions == list(range(1, 11))
    assert all(c <= task["total"] for c in completions)


def test_progress_without_start_event_falls_back_to_estimate(run_module):
    """train/start 缺失时退回调用方预估的 total（旧行为），不越界。"""
    task, completions = _drive(run_module, start_step=0, total_steps=4, emit_start=False)
    assert task["total"] == 4
    assert completions == [1, 2, 3, 4]
