"""接续训练：训练阶段链（plan）。

把一次训练拆成**有序的多个阶段**，每个阶段用自己的数据集与（可选的）超参覆盖，
整条链只构建一次模型/优化器并常驻，阶段之间不重载权重。

为什么需要它
------------
真实语料往往分来源、分质量、分领域。传统做法是「跑完 A 存盘 → 起新进程载入 →
换数据接着跑」，代价是：

- 每段都要重新 ``torch.load`` 权重与优化器动量（0.6B 模型数 GB 的 IO）；
- 优化器动量在段之间被序列化/反序列化，容易因 dtype/设备转换而折损；
- LR 调度每段重开 warmup，轨迹不连续。

阶段链把这一切收进**同一个进程、同一个 Trainer**：只换数据迭代器，
LR 走同一条 warmup-cosine 曲线（地平线 = 各阶段步数之和），模型始终常驻。

三个概念
--------
- :class:`StageSpec`：YAML 里的**声明**（数据路径、步数、超参覆盖）。
- :class:`ChainPlan`：一份完整的计划（有序的 ``StageSpec``）。
- :class:`TrainStage`：**运行期**已解析的单元（数据源已加载成 token 数组/数据集对象）。

用法::

    plan = load_chain_plan("config/chain_example.yaml")
    trainer.train_stages([resolve(s) for s in plan.stages])
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import yaml

from verse_core import get_logger
from verse_core.config import BaseConfig
from verse_trainer.data import DataConfig, fingerprint_source

logger = get_logger("plan")

__all__ = [
    "StageSpec",
    "ChainPlan",
    "TrainStage",
    "load_chain_plan",
    "stage_bounds",
    "chain_fingerprint",
    "check_chain_resume",
]


@dataclass
class StageSpec(BaseConfig):
    """单个训练阶段的声明（来自 plan yaml）。

    除 ``name``/``steps`` 外全部可选：留 ``None`` 表示沿用 yaml 主配置的值。
    """

    name: str
    steps: int
    data: str | None = None
    """训练数据路径；None 表示主配置的默认训练集。"""
    val: str | None = None
    """验证数据路径；None 表示主配置的默认验证集。"""

    # ---- 文本字段抽取（透传给 prepare_tokens） ----
    text_column: str | None = None
    text_columns: list[str] | None = None
    text_template: str | None = None
    delimiter: str | None = None
    token_dtype: str = "int32"
    mmap_tokens: bool = False

    # ---- 超参覆盖 ----
    lr: float | None = None
    seq_len: int | None = None
    batch_size: int | None = None
    grad_accum_steps: int | None = None
    eval_every: int | None = None
    ckpt_every: int | None = None
    loss_chunk_size: int | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("阶段 name 不能为空")
        if self.steps < 1:
            raise ValueError(f"阶段 [{self.name}] steps 必须 >= 1，当前 {self.steps}")
        for key in ("lr", "seq_len", "batch_size", "grad_accum_steps"):
            value = getattr(self, key)
            if value is not None and value <= 0:
                raise ValueError(f"阶段 [{self.name}] {key} 必须为正数，当前 {value}")

    def text_kwargs(self) -> dict[str, Any] | None:
        """组装 ``prepare_tokens`` 需要的文本抽取参数（无则 None）。"""
        kwargs: dict[str, Any] = {}
        for key in ("text_column", "text_columns", "text_template", "delimiter"):
            value = getattr(self, key)
            if value:
                kwargs[key] = value
        return kwargs or None


@dataclass
class ChainPlan(BaseConfig):
    """有序的训练阶段链。"""

    stages: list[StageSpec]
    horizon_steps: int | None = None
    """cosine 衰减地平线；None 表示用各阶段步数之和。"""

    def validate(self) -> None:
        if not self.stages:
            raise ValueError("plan 必须包含至少一个阶段")
        seen: set[str] = set()
        for stage in self.stages:
            if stage.name in seen:
                raise ValueError(f"阶段名重复: {stage.name}")
            seen.add(stage.name)
        if self.horizon_steps is not None and self.horizon_steps < 1:
            raise ValueError(f"horizon_steps 必须 >= 1，当前 {self.horizon_steps}")

    def total_steps(self) -> int:
        """各阶段步数之和（不含续训起点）。"""
        return sum(stage.steps for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        """覆写基类：把 ``list[StageSpec]`` 也递归转成 dict（否则无法 YAML 序列化）。"""
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "horizon_steps": self.horizon_steps,
        }


@dataclass
class TrainStage:
    """运行期已解析的训练阶段（数据源已就绪，可直接交给 Trainer）。"""

    name: str
    steps: int
    train_source: Any
    eval_source: Any | None = None
    data_config: DataConfig | None = None
    """本阶段的 ``seq_len``/``batch_size``/``seed``；None 表示沿用 Trainer 主配置。"""
    lr: float | None = None
    grad_accum_steps: int | None = None
    eval_every: int | None = None
    ckpt_every: int | None = None
    loss_chunk_size: int | None = None


def load_chain_plan(path: str | Path) -> ChainPlan:
    """从 YAML 加载训练计划。

    不能直接用 ``BaseConfig.from_dict``：``stages`` 是 ``list[dict]``，基类只会
    原样透传而不会逐项转成 :class:`StageSpec`。这里手写解析并对未知键给出明确报错
    （静默忽略拼错的键会让用户以为覆盖生效了，实际没有）。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"plan 文件应为 mapping，实际是 {type(raw).__name__}: {path}")

    unknown = set(raw) - {"stages", "horizon_steps"}
    if unknown:
        raise ValueError(f"plan 未知配置项: {sorted(unknown)}")

    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"plan 的 stages 必须是非空列表: {path}")

    known_fields = {f.name for f in fields(StageSpec)}
    stages: list[StageSpec] = []
    for index, item in enumerate(raw_stages):
        if not isinstance(item, dict):
            raise ValueError(f"stages[{index}] 应为 mapping，实际是 {type(item).__name__}")
        extra = set(item) - known_fields
        if extra:
            raise ValueError(f"stages[{index}] 未知配置项: {sorted(extra)}")
        stages.append(StageSpec(**item))

    return ChainPlan(stages=stages, horizon_steps=raw.get("horizon_steps"))


def stage_bounds(stages: Sequence[TrainStage], base: int = 0) -> list[tuple[int, int]]:
    """把阶段链展开成全局步号区间 ``[(start, end), ...]``。

    ``base`` 是续训起点（checkpoint 的 ``global_step``）；全新训练传 0。
    区间左闭右开，相邻阶段的 ``end``/``start`` 首尾相接，因此「当前 step 落在
    哪个阶段」是唯一确定的。
    """
    bounds: list[tuple[int, int]] = []
    cursor = base
    for stage in stages:
        bounds.append((cursor, cursor + stage.steps))
        cursor += stage.steps
    return bounds


def chain_fingerprint(stages: Sequence[TrainStage]) -> dict:
    """阶段链指纹：有序 ``(name, steps, 每阶段数据指纹)`` 的哈希。

    单阶段的数据指纹不足以描述一条链（换数据集本就是链的语义），
    所以续训一致性校验必须比对整条链的指纹。
    """
    per_stage = [fingerprint_source(stage.train_source) for stage in stages]
    digest = hashlib.sha256()
    for stage, fingerprint in zip(stages, per_stage):
        digest.update(stage.name.encode())
        digest.update(b"\x00")
        digest.update(str(stage.steps).encode())
        digest.update(b"\x00")
        digest.update(str((fingerprint or {}).get("sha256", "")).encode())
        digest.update(b"\x00")
    return {
        "kind": "chain",
        "n_stages": len(stages),
        "sha256": digest.hexdigest()[:16],
        "stages": per_stage,
    }


def check_chain_resume(
    saved_plan: dict | None,
    stages: Sequence[TrainStage],
    *,
    checkpoint_step: int,
    allow_data_change: bool = False,
) -> int:
    """校验续训用的阶段链与 checkpoint 中记录的链是否兼容，返回链的 ``plan_base``。

    阶段链的「换数据集」是设计语义，所以**不能**沿用单段训练的
    ``seq_len``/``batch_size``/数据指纹逐项比对（那样每次续训都会误报）。这里改为
    比对**整条链**的结构（名字 + 步数）与链指纹。

    Args:
        saved_plan: checkpoint 里记录的链计划；``None`` 表示该 checkpoint 来自
            单段训练（pretrain/finetune/sft），此时以 ``checkpoint_step`` 作为链起点。
        stages: 本次要跑的阶段链。
        checkpoint_step: checkpoint 的 ``global_step``。
        allow_data_change: 链指纹不一致时是否放行（用于有意换数据续训）。

    Returns:
        本次训练应使用的 ``plan_base``（交给 :meth:`Trainer.train_stages`）。

    Raises:
        RuntimeError: 链结构变化、断点早于链起点，或链指纹变化且未放行。
    """
    if not saved_plan:
        logger.warning(
            "checkpoint 不含阶段链计划（来自单段训练）：以 step %d 作为链起点，"
            "数据内容不做校验", checkpoint_step,
        )
        return int(checkpoint_step)

    base = int(saved_plan.get("base", 0))
    names = [stage.name for stage in stages]
    steps = [stage.steps for stage in stages]
    problems: list[str] = []
    if saved_plan.get("names") != names:
        problems.append(f"阶段名变化: {saved_plan.get('names')} -> {names}")
    if saved_plan.get("steps") != steps:
        problems.append(f"阶段步数变化: {saved_plan.get('steps')} -> {steps}")
    if checkpoint_step < base:
        problems.append(f"断点 step {checkpoint_step} 早于链起点 {base}")
    if problems:
        raise RuntimeError(
            "阶段链续训失败：计划与 checkpoint 不兼容（区间会错位）:\n"
            "  - " + "\n  - ".join(problems)
        )

    current = chain_fingerprint(stages)
    saved_fp = saved_plan.get("fingerprint") or {}
    if saved_fp.get("sha256") != current["sha256"]:
        detail = (
            f"链数据指纹变化: {saved_fp.get('sha256')}(n_stages={saved_fp.get('n_stages')})"
            f" -> {current['sha256']}(n_stages={current['n_stages']})"
        )
        if not allow_data_change:
            raise RuntimeError(
                "阶段链续训一致性校验失败（确需换数据请显式放行 --allow-data-change）:\n"
                f"  - {detail}"
            )
        logger.warning("阶段链续训放行数据变化: %s", detail)

    return base
