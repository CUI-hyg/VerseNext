"""CometSpark 阶段链（接续训练）管线级测试。

框架层的调度语义由 ``test_train_plan.py`` 覆盖；这里验证**端到端接线**：
``trainer.pipeline.train_plan`` 能否解析计划、逐阶段换数据集、落盘阶段产物与
固化配置，以及 ``--resume`` 时跳过已完成的阶段。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_COMETSPARK = Path(__file__).resolve().parents[2] / "CometSpark"

_CONFIG = """\
model_name: cometspark
model:
  vocab_size: 248077
  d_model: 32
  num_layers: 1
  num_heads: 2
  max_seq_len: 64
  dropout: 0.0
  tie_weights: true
  gradient_checkpointing: false
  attn_type: sdpa
data:
  batch_size: 2
  seq_len: 16
  seed: 42
  drop_last: true
optim:
  name: adamw
  lr: 3.0e-4
  warmup_steps: 2
  max_steps: 4
  min_lr_ratio: 0.1
steps: 4
log_every: 1
eval_every: 0
ckpt_dir: checkpoints
device: cpu
dtype: fp32
compile: false
grad_accum_steps: 1
seed: 42
"""

_PLAN = """\
stages:
  - name: alpha
    data: data/a.txt
    steps: 2
    seq_len: 16
    batch_size: 2
    lr: 3.0e-4
  - name: beta
    data: data/b.txt
    steps: 2
    seq_len: 16
    batch_size: 2
    lr: 1.0e-4
"""


@pytest.fixture
def cometspark(tmp_path):
    """准备一个隔离的 CometSpark 运行根目录（配置 + 两份语料 + 计划）。"""
    if not (_COMETSPARK / "trainer" / "pipeline.py").exists():
        pytest.skip("工作区无 CometSpark/trainer")
    if str(_COMETSPARK) not in sys.path:
        sys.path.insert(0, str(_COMETSPARK))
    import trainer.pipeline as pipeline  # noqa: PLC0415

    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config" / "cfg.yaml").write_text(_CONFIG, encoding="utf-8")
    (tmp_path / "data" / "a.txt").write_text(
        "机器学习是人工智能的一个分支。" * 60, encoding="utf-8")
    (tmp_path / "data" / "b.txt").write_text(
        "语言模型通过预测下一个词来学习。" * 60, encoding="utf-8")
    (tmp_path / "config" / "plan.yaml").write_text(_PLAN, encoding="utf-8")
    return pipeline, tmp_path


def test_train_plan_runs_all_stages_and_writes_artifacts(cometspark):
    pipeline, root = cometspark
    summary = pipeline.train_plan(
        root / "config" / "plan.yaml", root / "config" / "cfg.yaml", root,
    )

    assert summary["stage"] == "chain"
    assert summary["steps"] == 4
    assert [(s["name"], s["start_step"], s["end_step"]) for s in summary["stages"]] == [
        ("alpha", 0, 2), ("beta", 2, 4),
    ]
    assert all(not s["skipped"] for s in summary["stages"])
    # 每阶段消耗的 token 分别统计（2 步 × 2 batch × 16 seq = 64）
    assert [s["tokens"] for s in summary["stages"]] == [64, 64]
    assert summary["tokens_seen"] == 128

    ckpt_dir = root / "checkpoints"
    assert (ckpt_dir / "cometspark_chain00_alpha.pt").exists()
    assert (ckpt_dir / "cometspark_chain01_beta.pt").exists()
    chain_final = Path(summary["checkpoint"])
    assert chain_final.exists() and chain_final.name.endswith("_chain_final.pt")

    resolved = yaml.safe_load(Path(summary["resolved_config"]).read_text(encoding="utf-8"))
    assert [s["name"] for s in resolved["chain"]["stages"]] == ["alpha", "beta"]
    # 阶段 2 的 lr 覆盖被记录下来（而不是被后一阶段覆盖成最终 cfg.optim.lr）
    assert resolved["chain"]["stages"][1]["lr"] == pytest.approx(1.0e-4)
    assert resolved["chain"]["stages"][0]["lr"] == pytest.approx(3.0e-4)
    assert resolved["chain"]["horizon_steps"] == 4


def test_train_plan_resume_skips_completed_stages(cometspark):
    pipeline, root = cometspark
    args = (root / "config" / "plan.yaml", root / "config" / "cfg.yaml", root)
    first = pipeline.train_plan(*args)
    assert first["steps"] == 4

    resumed = pipeline.train_plan(*args, resume_from=first["checkpoint"])
    # 已全部跑完 -> 两个阶段都跳过，步号不前进
    assert [s["skipped"] for s in resumed["stages"]] == [True, True]
    assert resumed["steps"] == 4


def test_train_plan_rejects_changed_data_without_opt_in(cometspark):
    pipeline, root = cometspark
    args = (root / "config" / "plan.yaml", root / "config" / "cfg.yaml", root)
    first = pipeline.train_plan(*args)

    # 换掉阶段 2 的数据 -> 链指纹变化，默认必须报错
    (root / "data" / "b.txt").write_text("完全不同的语料内容。" * 60, encoding="utf-8")
    with pytest.raises(RuntimeError, match="指纹变化"):
        pipeline.train_plan(*args, resume_from=first["checkpoint"])
    # 显式放行后可以继续
    resumed = pipeline.train_plan(*args, resume_from=first["checkpoint"],
                                 allow_data_change=True)
    assert resumed["stage"] == "chain"
