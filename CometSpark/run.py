#!/usr/bin/env python3
"""CometSpark-Exp 终端 CLI（rich 界面）。

用法::

    python run.py version                       # 查看模型版本
    python run.py configs                       # 列出发现的配置与 checkpoint
    python run.py train [--config ...] [--steps N] [--threads N] [--data ...]  # 预训练
    python run.py train --resume [CKPT] [--steps N] [--allow-data-change]     # 断点续训
    python run.py train --plan config/chain_example.yaml                      # 接续训练（阶段链）
    python run.py finetune --ckpt ... [--lr ...] [--lora-rank ...]             # 微调（文本流）
    python run.py sft --data chats.json [--lora-rank ...]                     # 指令微调（对话）
    python run.py generate --prompt "..." [--ckpt ...] [--chat]               # 续写/对话
    python run.py chat [--ckpt ...]                                           # 交互式多轮对话

config 与 checkpoint 均可自动发现：
- config: ``config/cometspark_exp_<version>.yaml`` 中取版本最高者（当前 0.3 = Dense 0.6B）
- checkpoint: 优先后训练产物（sft > finetune > pretrain，merged 优先于 final），
  否则取最大 step 的 ``step_N.pt``

接续训练（``--plan``）：把一个模型的训练拆成有序的多个阶段，**每阶段换数据集**
（可各自覆盖 lr / seq_len / batch_size / grad_accum 等）。模型、优化器动量与 LR
调度只构建一次并全程常驻，阶段之间不重载权重、不重开 warmup，共用一条
warmup-cosine 曲线。计划格式见 ``config/chain_example.yaml``。
产物：每阶段 ``checkpoints/{MODEL_NAME}_chain{i:02d}_{name}.pt``，
链尾 ``checkpoints/{MODEL_NAME}_chain_final.pt``，
固化配置 ``config/{MODEL_NAME}_chain_resolved.yaml``。
``--resume`` 时会跳过已完成的阶段、从中断的阶段接着跑。

断点续训（``--resume``）：恢复权重 + 优化器动量 + LR 调度进度 + RNG + 监控/早停
进度 + 数据指纹，做到续训与不间断训练**逐位一致**（不丢失、不折损）。
- ``--steps N`` 表示新增 N 步（默认沿用原配置剩余步数）；
- LR 延续原 warmup-cosine，不重开 warmup；
- 数据/形状与 checkpoint 不一致会**报错**，换数据需显式 ``--allow-data-change``。

数据格式（train/finetune）：
- 文本：jsonl/json/txt/md/csv/tsv/parquet（``--text-column``/``--text-columns``/
  ``--text-template``/``--delimiter`` 控制字段抽取）
- 预分词：npy/bin/tokens（``--token-dtype`` 指定 raw 二进制类型，``--mmap-tokens`` 省内存）
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from model import MODEL_NAME, __version__  # noqa: E402

console = Console()

DEFAULT_CONFIG = ROOT / "config" / "cometspark_exp_0.3.yaml"

# ------------------------------------------------------------ 自动发现


def _version_key(p: Path) -> tuple[int, ...]:
    m = re.search(r"cometspark_exp_([\d.]+)\.yaml$", p.name)
    if not m:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))


def find_configs() -> list[Path]:
    """所有可用的核心配置（版本号降序，排除运行时生成的派生配置）。"""
    seen: dict[Path, Path] = {}
    for p in ROOT.glob("config/cometspark_exp_*.yaml"):
        if p.name.endswith(("_resolved.yaml", "_override.yaml", "_train_override.yaml")):
            continue
        seen.setdefault(p, p)
    resolved = sorted(
        (p for p in ROOT.glob("config/CometSpark-Exp-*_resolved.yaml")),
        key=_version_key,
    )
    return sorted(seen.values(), key=_version_key, reverse=True) + resolved


def find_config(explicit: str | None = None) -> Path:
    """自动寻找配置：显式指定 > 版本最高的核心配置 > 默认路径。"""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            console.print(f"[red]配置不存在:[/] {p}")
            raise SystemExit(2)
        return p
    configs = find_configs()
    if configs:
        return configs[0]
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    console.print("[red]未找到任何配置文件 (config/cometspark_exp_*.yaml)[/]")
    raise SystemExit(2)


def find_checkpoints() -> list[Path]:
    """checkpoints 目录下的所有可推理模型文件，按优先级排列。

    优先级：后训练阶段优先（sft > finetune > pretrain），同阶段内 merged
    （LoRA 合并后的完整模型）优先于 final，最后才回退到周期 ``step_N.pt``。
    LoRA 适配器（``*_adapter.pt``）不含完整权重，不参与自动发现。
    """
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    files = [
        p for p in ckpt_dir.rglob("*.pt")
        if "adapter" not in p.name and ".token_cache" not in p.parts
    ]

    def _rank(p: Path) -> tuple[int, int, str]:
        name = p.name
        for i, stage in enumerate(("sft", "finetune", "pretrain")):
            if f"_{stage}_" in name:
                kind = 0 if name.endswith("_merged.pt") else 1
                return (i, kind, name)
        return (len(("sft", "finetune", "pretrain")), 0, name)

    finals = sorted((p for p in files if p.name.endswith(("_final.pt", "_merged.pt"))),
                    key=_rank)
    steps = sorted(
        (p for p in files if re.search(r"step_(\d+)", p.name)),
        key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)),
        reverse=True,
    )
    return finals + steps


def find_checkpoint(explicit: str | None = None) -> Path:
    """自动寻找 checkpoint：显式指定 > final > 最大 step。"""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            console.print(f"[red]checkpoint 不存在:[/] {p}")
            raise SystemExit(2)
        return p
    ckpts = find_checkpoints()
    if ckpts:
        return ckpts[0]
    console.print("[red]未找到任何 checkpoint，请先运行 train[/]")
    raise SystemExit(2)


# ---------------------------------------------------------------- 界面


def banner(title: str, rows: list[tuple[str, str]]) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column()
    for k, v in rows:
        grid.add_row(k, str(v))
    console.print(Panel(grid, title=f"[bold magenta]{title}[/]",
                        border_style="magenta", box=box.ROUNDED))


def kv_table(pairs: list[tuple[str, object]], title: str) -> Table:
    table = Table(title=title, box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column(style="cyan", justify="right")
    table.add_column(style="bold")
    for k, v in pairs:
        table.add_row(k, str(v))
    return table


# ---------------------------------------------------------------- 子命令


def cmd_version(_args: argparse.Namespace) -> int:
    banner(MODEL_NAME, [("version", __version__), ("root", str(ROOT))])
    return 0


def cmd_configs(_args: argparse.Namespace) -> int:
    configs = find_configs()
    ckpts = find_checkpoints()
    console.print(kv_table([(f"[{i}]", c.relative_to(ROOT)) for i, c in enumerate(configs)],
                           title="可用配置（[0] 为默认选择）"))
    console.print(kv_table([(f"[{i}]", c.relative_to(ROOT)) for i, c in enumerate(ckpts)],
                           title="可用 checkpoint（[0] 为默认选择）"))
    return 0


def _yaml_steps(config_path: Path) -> int:
    import yaml

    return int((yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("steps", 300) or 300)


def _fmt_loss(value) -> str:
    """loss 展示：None/NaN 显示为 ``-``（例如阶段链已全部跑完、本次无新增步数）。"""
    try:
        if value is None or math.isnan(float(value)):
            return "-"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _run_training_ui(title: str, rows: list[tuple[str, str]],
                     total_steps: int, run_fn) -> int:
    """训练类子命令共享的 rich 进度/事件界面：run_fn(event_bus) -> summary。

    ``total_steps`` 只是**预估值**（``--steps`` 或 yaml 的 ``steps``），用于
    ``train/start`` 事件缺失时兜底。真实进度以 trainer 广播的
    ``train/start``（``start_step``/``total_steps``）为准，并把绝对
    ``global_step`` 换算成本次 run 的相对步号 —— 否则断点续训时
    ``completed`` 会远超 ``total``，进度条卡在 100%、ETA 恒为 ``0:00:00``。
    """
    import logging

    from verse_core import EventBus

    # 关闭框架 INFO 日志，避免与 rich 进度条重复输出（eval/checkpoint 已由事件渲染）
    logging.getLogger("verse").setLevel(logging.WARNING)
    logging.getLogger("trainer").setLevel(logging.WARNING)

    banner(title, rows)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[loss]}[/]"),
        TextColumn("[cyan]{task.fields[lr]}[/]"),
        TextColumn("[yellow]{task.fields[speed]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        # 先按调用方的预估总步数建确定态进度条；trainer 广播 train/start 后
        # 立即用「本次 run 的步号区间」校正（rich 在 total 变化时会重置采样，
        # 此刻尚无采样，无副作用）。这样断点续训/阶段链也不会算错 ETA。
        task = progress.add_task("train", total=total_steps, loss="", lr="", speed="")
        # 本次 run 的起始步号（trainer.global_step），用于把绝对步号换算成相对进度
        state = {"offset": 0}

        def on_start(event) -> None:
            p = event.payload
            start = int(p.get("start_step", 0) or 0)
            # trainer 未给出总步数时退回调用方的预估值
            total = int(p.get("total_steps", 0) or 0) or total_steps
            state["offset"] = start
            progress.update(task, total=max(total - start, 1), completed=0)

        def on_step(event) -> None:
            p = event.payload
            progress.update(
                task,
                completed=max(p["step"] - state["offset"], 0),
                loss=f"loss {p['loss']:.4f}",
                lr=f"lr {p['lr']:.2e}",
                speed=f"{p['tokens_per_second']:.0f} tok/s",
            )

        def on_eval(event) -> None:
            progress.console.print(
                f"  :mag: [bold]eval[/] step {event.payload['step']}: "
                f"[bold green]{event.payload['eval_loss']:.4f}[/]"
            )

        def on_ckpt(event) -> None:
            progress.console.print(f"  :floppy_disk: checkpoint -> {event.payload['path']}")

        def on_early_stop(event) -> None:
            progress.console.print(
                f"  :stop_sign: [bold red]early stop[/] step {event.payload['step']}: "
                f"loss 连续 {event.payload['patience']} 次未低于 best "
                f"{event.payload['best_loss']:.4f}"
            )

        def on_stage_start(event) -> None:
            p = event.payload
            progress.update(task, description=f"train [{p['index'] + 1}/{p['total']}] {p['name']}")

        def on_stage_end(event) -> None:
            p = event.payload
            progress.console.print(
                f"  :checkered_flag: 阶段 [bold]{p['name']}[/] 完成: "
                f"[bold green]{p['final_loss']:.4f}[/] -> {p['checkpoint']}"
            )

        event_bus = EventBus()
        event_bus.subscribe("train/start", on_start)
        event_bus.subscribe("train/step_end", on_step)
        event_bus.subscribe("train/eval_end", on_eval)
        event_bus.subscribe("train/checkpoint", on_ckpt)
        event_bus.subscribe("train/early_stop", on_early_stop)
        event_bus.subscribe("train/stage_start", on_stage_start)
        event_bus.subscribe("train/stage_end", on_stage_end)

        try:
            summary = run_fn(event_bus)
        except KeyboardInterrupt:
            console.print("[yellow]:warning: 训练被中断，进度已通过周期 checkpoint 保存[/]")
            return 130
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            # 这几类是「用户改配置/换数据就能解决」的问题（续训一致性校验、
            # 数据缺失、配置非法），直接给结论比甩一段 traceback 有用。
            console.print(f"[bold red]:x: 训练失败[/] ({type(exc).__name__})")
            console.print(str(exc))
            return 2

        # train/step_end 只在 global_step % log_every == 0 时发出；短 run（步数少于
        # log_every）或末段不足一个间隔时进度条会停在旧值。这里用实际步号收尾，
        # 让进度条如实反映「跑到哪了」（早停时也照样停在真实位置，不假装 100%）。
        final_step = int(summary.get("steps") or 0)
        if final_step > state["offset"]:
            progress.update(task, completed=final_step - state["offset"])

    rows_out = [
        ("model", summary.get("model")),
        ("mode", summary.get("mode", "pretrain")),
        ("params", f"{summary.get('model_params_m', '?')}M"),
        ("final loss", _fmt_loss(summary.get("final_loss"))),
        ("best eval", _fmt_loss(summary.get("best_eval_loss"))),
        ("steps", summary.get("steps")),
        ("tokens", f"{summary.get('tokens_seen', 0):,}"),
        ("best loss", _fmt_loss(summary.get("best_loss"))),
        ("early stop", "是" if summary.get("early_stopped") else "否"),
        ("耗时", f"{summary.get('wall_time_s', 0):.1f}s"),
        ("checkpoint", _display_path(Path(summary["checkpoint"]))
         if summary.get("checkpoint") else "-"),
    ]
    if summary.get("loss_plot"):
        rows_out.append(("loss 曲线", _display_path(Path(summary["loss_plot"]))))
    if summary.get("adapter"):
        rows_out.append(("adapter", _display_path(Path(summary["adapter"]))))
    if summary.get("resolved_config"):
        rows_out.append(("resolved config",
                         _display_path(Path(summary["resolved_config"]))))
    if summary.get("plan"):
        rows_out.append(("plan", _display_path(Path(summary["plan"]))))
    console.print(kv_table(rows_out, title=title))

    stages = summary.get("stages")
    if stages:
        console.print(_stages_table(stages, title))
    return 0


def _stages_table(stages: list[dict], title: str) -> Table:
    """阶段链的逐阶段结果表（单段训练没有 stages，不会走到这里）。"""
    table = Table(title=f"{title} · 阶段", box=box.SIMPLE, pad_edge=False)
    for column, justify in (("阶段", "left"), ("步号区间", "right"), ("loss", "right"),
                            ("eval", "right"), ("tokens", "right"),
                            ("耗时", "right"), ("状态", "left")):
        table.add_column(column, justify=justify)
    for stage in stages:
        if stage.get("skipped"):
            status = "[dim]跳过（已完成）[/]"
        elif stage.get("resumed"):
            status = "[yellow]续跑[/]"
        else:
            status = "[green]完成[/]"
        loss = stage.get("final_loss")
        eval_loss = stage.get("best_eval_loss")
        table.add_row(
            stage["name"],
            f"{stage['start_step']} → {stage['end_step']}",
            _fmt_loss(loss),
            _fmt_loss(eval_loss),
            f"{stage.get('tokens', 0):,}",
            f"{stage.get('wall_time_s', 0):.1f}s",
            status,
        )
    return table


def _display_path(p: Path) -> Path | str:
    """相对 ROOT 展示路径；项目外路径（如 /tmp 测试配置）原样返回。"""
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def _common_train_rows(config_path: Path, args: argparse.Namespace,
                       steps: int | None = None) -> list[tuple[str, str]]:
    rows = [
        ("config", _display_path(config_path)),
        ("model", f"{MODEL_NAME} (v{__version__})"),
        ("steps", steps or args.steps or _yaml_steps(config_path)),
        ("device/threads", f"{args.device or 'auto'} / {args.threads or 'config'}"),
    ]
    data = getattr(args, "data", None)
    if data:
        rows.append(("data", data))
    resume = getattr(args, "resume", None)
    if resume is not None:
        rows.append(("resume", resume if resume not in ("", "auto") else "auto（同阶段 final）"))
        if getattr(args, "allow_data_change", False):
            rows.append(("data change", "已放行（--allow-data-change）"))
    return rows


def _text_kwargs(args: argparse.Namespace) -> dict | None:
    """从命令行参数构建文本字段抽取 kwargs（无则返回 None）。"""
    kwargs: dict = {}
    if getattr(args, "text_column", None):
        kwargs["text_column"] = args.text_column
    if getattr(args, "text_columns", None):
        kwargs["text_columns"] = [
            c.strip() for c in args.text_columns.split(",") if c.strip()
        ]
    if getattr(args, "text_template", None):
        kwargs["text_template"] = args.text_template
    if getattr(args, "delimiter", None):
        kwargs["delimiter"] = args.delimiter
    return kwargs or None


def _add_data_args(p: argparse.ArgumentParser) -> None:
    """训练/微调共享的数据格式参数（支持多文本格式 + 预分词文件）。"""
    p.add_argument("--data", default=None,
                   help="训练数据路径（默认 data/train.jsonl）；支持 "
                        "jsonl/json/txt/md/csv/tsv/parquet 与预分词 npy/bin/tokens")
    p.add_argument("--val", default=None, help="验证数据路径（默认 data/val.jsonl）")
    p.add_argument("--text-column", default=None,
                   help="文本字段名（jsonl/json/csv/tsv/parquet）")
    p.add_argument("--text-columns", default=None,
                   help="多字段拼接为文本，逗号分隔（如 query,response）")
    p.add_argument("--text-template", default=None,
                   help="多字段模板（如 '{query} => {response}'）")
    p.add_argument("--delimiter", default=None,
                   help="CSV/TSV 分隔符（默认按后缀：.tsv -> \\t，其余 ,）")
    p.add_argument("--token-dtype", default="int32",
                   help="预分词 raw 二进制元素类型（int32/uint16/uint32/int64）")
    p.add_argument("--mmap-tokens", action="store_true",
                   help=".npy 预分词文件用内存映射加载（大文件省内存）")


def _add_resume_args(p: argparse.ArgumentParser) -> None:
    """训练/微调/SFT 共享的断点续训参数。"""
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="断点续训：不带值=自动找同阶段 final；或显式给 checkpoint 路径")
    p.add_argument("--allow-data-change", action="store_true",
                   help="续训时允许数据内容/seq_len/batch_size 与 checkpoint 不一致（换数据继续训练）")
    p.add_argument("--horizon-steps", type=int, default=None,
                   help="续训时重设 cosine 衰减地平线（默认延续 checkpoint 原值，避免 LR 回弹）")


def _resolve_path(p: str | None) -> Path | None:
    """把用户给的相对路径解析到项目根目录下（绝对路径原样返回）。"""
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _stage_final(stage: str) -> Path | None:
    """查找某训练阶段的 final（或 LoRA merged）产物。"""
    ckpt_dir = ROOT / "checkpoints"
    for name in (f"{MODEL_NAME}_{stage}_final.pt", f"{MODEL_NAME}_{stage}_merged.pt"):
        hits = sorted(ckpt_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def _resolve_resume(value: str | None, stage: str) -> Path | None:
    """``--resume`` 解析：None=不续训；``auto``/空=同阶段 final（回退最新）；否则为路径。"""
    if value is None:
        return None
    if value in ("", "auto"):
        found = _stage_final(stage)
        if found is not None:
            return found
        console.print(f"[yellow]未找到 {stage} 阶段 final，回退到最新 checkpoint[/]")
        return find_checkpoint(None)
    return _resolve_path(value)


def _load_plan(plan_path: Path):
    """加载训练计划（供 UI 显示阶段数与步数；真实进度由 train/start 事件给出）。"""
    from verse_trainer.plan import load_chain_plan

    return load_chain_plan(plan_path)


def cmd_train(args: argparse.Namespace) -> int:
    config_path = find_config(args.config)
    data_path = _resolve_path(args.data)
    val_path = _resolve_path(args.val)
    resume_path = _resolve_resume(args.resume, "pretrain")

    if args.plan:
        from trainer import train_plan as do_train_plan

        plan_path = _resolve_path(args.plan)
        if not plan_path.exists():
            console.print(f"[red]训练计划不存在:[/] {plan_path}")
            return 2
        # 阶段链的续训基准是链尾产物（{MODEL_NAME}_chain_final.pt）
        plan_resume = _resolve_resume(args.resume, "chain")
        plan = _load_plan(plan_path)
        total = args.steps or plan.total_steps()
        rows = _common_train_rows(config_path, args, steps=total) + [
            ("plan", _display_path(plan_path)),
            ("阶段", " → ".join(s.name for s in plan.stages)),
        ]
        return _run_training_ui(
            f"{MODEL_NAME} · {'阶段链续训' if plan_resume else '接续训练（阶段链）'}",
            rows, total,
            lambda bus: do_train_plan(
                str(plan_path), str(config_path), ROOT, event_bus=bus,
                threads_override=args.threads, device_override=args.device,
                resume_from=str(plan_resume) if plan_resume else None,
                allow_data_change=args.allow_data_change,
                horizon_steps=args.horizon_steps),
        )

    from trainer import train as do_train

    return _run_training_ui(
        f"{MODEL_NAME} · {'断点续训' if resume_path else '预训练'}",
        _common_train_rows(config_path, args),
        args.steps or _yaml_steps(config_path),
        lambda bus: do_train(str(config_path), ROOT, event_bus=bus,
                             steps_override=args.steps, threads_override=args.threads,
                             device_override=args.device,
                             text_kwargs=_text_kwargs(args),
                             data_path=str(data_path) if data_path else None,
                             val_path=str(val_path) if val_path else None,
                             token_dtype=args.token_dtype,
                             mmap_tokens=args.mmap_tokens,
                             resume_from=str(resume_path) if resume_path else None,
                             allow_data_change=args.allow_data_change,
                             horizon_steps=args.horizon_steps),
    )


def cmd_finetune(args: argparse.Namespace) -> int:
    from trainer import finetune as do_finetune

    config_path = find_config(args.config)
    resume_path = _resolve_resume(args.resume, "finetune")
    ckpt_path = resume_path if resume_path is not None else find_checkpoint(args.ckpt)
    data_path = _resolve_path(args.data)
    val_path = _resolve_path(args.val)
    title = "断点续训" if resume_path else "微调"
    return _run_training_ui(
        f"{MODEL_NAME} · {title} ({'LoRA r=%d' % args.lora_rank if args.lora_rank else '全参'})",
        _common_train_rows(config_path, args)
        + [("base ckpt", _display_path(ckpt_path)),
           ("lr", args.lr or "config"),
           ("lora", f"r={args.lora_rank} alpha={args.lora_alpha}" if args.lora_rank else "off")],
        args.steps or _yaml_steps(config_path),
        lambda bus: do_finetune(str(config_path), ROOT, str(ckpt_path), event_bus=bus,
                                steps_override=args.steps, threads_override=args.threads,
                                lr_override=args.lr, device_override=args.device,
                                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                                text_kwargs=_text_kwargs(args),
                                data_path=str(data_path) if data_path else None,
                                val_path=str(val_path) if val_path else None,
                                token_dtype=args.token_dtype,
                                mmap_tokens=args.mmap_tokens,
                                resume_from=str(resume_path) if resume_path else None,
                                allow_data_change=args.allow_data_change,
                                horizon_steps=args.horizon_steps),
    )


def cmd_sft(args: argparse.Namespace) -> int:
    from trainer import sft as do_sft

    data_path = _resolve_path(args.data)
    if not data_path.exists():
        console.print(f"[red]SFT 数据不存在:[/] {data_path}")
        return 2
    config_path = find_config(args.config)
    resume_path = _resolve_resume(args.resume, "sft")
    ckpt_path = resume_path if resume_path is not None else find_checkpoint(args.ckpt)
    val_path = _resolve_path(args.val)
    title = "断点续训" if resume_path else "SFT 指令微调"
    return _run_training_ui(
        f"{MODEL_NAME} · {title} ({'LoRA r=%d' % args.lora_rank if args.lora_rank else '全参'})",
        _common_train_rows(config_path, args)
        + [("base ckpt", _display_path(ckpt_path)),
           ("data", _display_path(data_path)),
           ("lr", args.lr or "config"),
           ("lora", f"r={args.lora_rank} alpha={args.lora_alpha}" if args.lora_rank else "off")],
        args.steps or _yaml_steps(config_path),
        lambda bus: do_sft(str(config_path), ROOT, str(ckpt_path), str(data_path),
                           str(val_path) if val_path else None, event_bus=bus,
                           steps_override=args.steps, threads_override=args.threads,
                           lr_override=args.lr, device_override=args.device,
                           lora_rank=args.lora_rank,
                           lora_alpha=args.lora_alpha, batch_size=args.batch_size,
                           max_seq_len=args.max_seq_len,
                           train_on_last_turns=args.train_on_last_turns,
                           resume_from=str(resume_path) if resume_path else None,
                           allow_data_change=args.allow_data_change,
                           horizon_steps=args.horizon_steps),
    )


def cmd_generate(args: argparse.Namespace) -> int:
    from trainer import generate as do_generate

    ckpt_path = find_checkpoint(args.ckpt)
    banner(
        f"{MODEL_NAME} · 生成",
        [
            ("checkpoint", _display_path(ckpt_path)),
            ("device/dtype", f"{args.device or 'auto'} / {args.dtype or 'auto'}"),
            ("sampling", f"temperature={args.temperature} top_k={args.top_k} "
             f"top_p={args.top_p} rep_pen={args.repetition_penalty} "
             f"no_repeat_ngram={args.no_repeat_ngram}"
             + (" greedy" if args.greedy else "")),
            ("format", "ChatML 对话" if args.chat else "自由续写"),
        ],
    )
    text = Text(args.prompt, style="bold")
    with console.status("[magenta]生成中...[/]"):
        output = do_generate(
            args.prompt,
            str(ckpt_path),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
            greedy=args.greedy,
            chat=args.chat,
            device=args.device or "auto",
            dtype=args.dtype or "auto",
        )
    text_out = output[len(args.prompt):].strip() if output.startswith(args.prompt) else output.strip()
    if args.chat:
        # ChatML：裁掉模板前缀，只展示 assistant 回答部分
        marker = "assistant\n"
        text_out = output.rsplit(marker, 1)[-1].strip() if marker in output else text_out
    console.print(Panel(
        text_out or "[dim](模型立即输出了结束符 — 试试更长的 prompt 或调低 temperature)[/]",
        title="[bold]输出[/]", border_style="green", box=box.ROUNDED,
        subtitle=f"prompt: {args.prompt!r}", subtitle_align="right",
    ))
    return 0


# ---------------------------------------------------------------- 对话 REPL

_CHAT_HELP = """\
[bold]命令[/]
  /help            显示本帮助
  /reset           清空对话历史（保留 system 提示）
  /history         打印当前消息历史
  /exit  /quit     退出

直接输入内容即发起一轮对话；模型会带上之前的上下文。
Ctrl-C / Ctrl-D 也可退出。"""


def _chat_history_table(history: list[dict]) -> Table:
    """把消息历史渲染成表格。"""
    table = Table(title="对话历史", box=box.SIMPLE, pad_edge=False, show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("role", style="cyan")
    table.add_column("content")
    for i, msg in enumerate(history):
        role = msg.get("role", "?")
        style = "bold green" if role == "assistant" else "white"
        content = str(msg.get("content", "")).replace("\n", " ⏎ ")
        table.add_row(str(i), role, Text(content, style=style, overflow="fold"))
    return table


def _handle_chat_command(session, line: str) -> bool:
    """处理 ``/`` 开头的命令；返回 False 表示应退出 REPL。"""
    cmd, _, _rest = line.partition(" ")
    cmd = cmd.lower()
    if cmd in ("/exit", "/quit", "/q"):
        return False
    if cmd == "/reset":
        session.reset()
        console.print("[yellow]对话历史已清空[/]")
    elif cmd == "/history":
        if session.history:
            console.print(_chat_history_table(session.history))
        else:
            console.print("[dim](暂无历史)[/]")
    elif cmd == "/help":
        console.print(Panel(_CHAT_HELP, title="[bold]对话命令[/]",
                            border_style="cyan", box=box.ROUNDED))
    else:
        console.print(f"[red]未知命令 {cmd}[/]（/help 查看可用命令）")
    return True


def _chat_repl(session) -> int:
    """交互式多轮对话循环：读一行 -> 生成 -> 打印，直到 /exit 或 EOF。"""
    while True:
        try:
            line = console.input("[bold cyan]你[/] [dim]›[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见[/]")
            break
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            if not _handle_chat_command(session, text):
                break
            continue
        try:
            with console.status("[magenta]思考中...[/]"):
                reply = session.ask(text)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]生成失败[/] ({type(exc).__name__}): {exc}")
            continue
        console.print(Panel(
            reply or "[dim](空回复 — 试试换个问法或调低 temperature)[/]",
            title=f"[bold green]助手[/] [dim](第 {session.turns} 轮)[/]",
            border_style="green", box=box.ROUNDED,
        ))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """交互式多轮对话（模型只加载一次，常驻显存/内存）。"""
    from trainer import ChatSession

    ckpt_path = find_checkpoint(args.ckpt)
    banner(
        f"{MODEL_NAME} · 对话",
        [
            ("checkpoint", _display_path(ckpt_path)),
            ("device/dtype", f"{args.device or 'auto'} / {args.dtype or 'auto'}"),
            ("history", f"最多保留 {args.max_history_turns} 轮" if args.max_history_turns
             else "不限制"),
            ("sampling", f"temperature={args.temperature} top_k={args.top_k} "
             f"top_p={args.top_p} rep_pen={args.repetition_penalty} "
             f"no_repeat_ngram={args.no_repeat_ngram}"
             + (" greedy" if args.greedy else "")),
        ],
    )
    try:
        with console.status("[magenta]加载模型中...[/]"):
            session = ChatSession(
                str(ckpt_path),
                device=args.device or "auto",
                dtype=args.dtype or "auto",
                system=args.system,
                max_history_turns=args.max_history_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram=args.no_repeat_ngram,
                greedy=args.greedy,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # LoRA 适配器 / 架构不匹配 / checkpoint 缺失都属于「用户能自己解决」
        console.print(f"[bold red]:x: 加载模型失败[/] ({type(exc).__name__})")
        console.print(str(exc))
        return 2
    console.print("[dim]输入内容开始对话；/help 查看命令，/exit 退出[/]")
    return _chat_repl(session)


# ---------------------------------------------------------------- 入口


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cometspark",
        description=f"{MODEL_NAME} 训练与推理 CLI（config/checkpoint 支持自动发现）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run.py train                     # 自动使用最新版配置(0.3 Dense 0.6B)预训练\n"
            "  python run.py train --steps 50 --threads 8\n"
            "  python run.py train --data data/corpus.txt        # 纯文本语料\n"
            "  python run.py train --data qa.csv --text-columns question,answer\n"
            "  python run.py train --data tokens.npy --mmap-tokens   # 预分词 token 流\n"
            "  python run.py train --resume              # 断点续训（自动找 pretrain final）\n"
            "  python run.py train --resume ckpt.pt --steps 500   # 续训并新增 500 步\n"
            "  python run.py train --resume --data new.jsonl --allow-data-change  # 换数据续训\n"
            "  python run.py train --plan config/chain_example.yaml  # 接续训练（多阶段换数据集）\n"
            "  python run.py train --plan config/chain_example.yaml --resume  # 阶段链断点续训\n"
            "  python run.py finetune --lr 1e-4        # 从 checkpoint 全参微调\n"
            "  python run.py finetune --lora-rank 8 --steps 100   # LoRA 微调\n"
            "  python run.py sft --data data/sft_demo.json --lora-rank 8  # 指令微调\n"
            "  python run.py generate --prompt '机器学习' --chat   # 对话式生成\n"
            "  python run.py chat                       # 交互式多轮对话（REPL）\n"
            "  python run.py chat --device npu --dtype bf16   # 指定设备与精度\n"
            "  python run.py configs                   # 查看可用配置与 checkpoint\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="打印模型名与版本号").set_defaults(func=cmd_version)
    sub.add_parser("configs", help="列出自动发现的配置与 checkpoint").set_defaults(func=cmd_configs)

    p_train = sub.add_parser("train", help="预训练 CometSpark 模型")
    p_train.add_argument("--config", default=None,
                         help="yaml 配置路径（默认自动发现版本最高的配置）")
    p_train.add_argument("--plan", default=None,
                         help="接续训练计划 yaml：把训练拆成多个阶段，每阶段可换数据集。"
                              "见 config/chain_example.yaml")
    p_train.add_argument("--steps", type=int, default=None, help="覆盖训练步数")
    p_train.add_argument("--device", default=None, help="覆盖设备 (auto/cpu/cuda/npu；默认 auto 按优先级自动选)")
    p_train.add_argument("--threads", type=int, default=None, help="覆盖 CPU 线程数")
    _add_data_args(p_train)
    _add_resume_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # ---- 后训练：微调 / SFT 共享参数 ----
    def _add_posttrain_args(p, with_data: bool) -> None:
        p.add_argument("--ckpt", default=None,
                       help="基座 checkpoint（默认 final > 最大 step 自动发现）")
        p.add_argument("--config", default=None,
                       help="yaml 配置路径（默认自动发现；模型架构以 ckpt 固化配置为准）")
        p.add_argument("--steps", type=int, default=None, help="覆盖训练步数")
        p.add_argument("--lr", type=float, default=None,
                       help="覆盖学习率（微调建议低于预训练值）")
        p.add_argument("--device", default=None, help="覆盖设备 (auto/cpu/cuda/npu；默认 auto 按优先级自动选)")
        p.add_argument("--threads", type=int, default=None, help="覆盖 CPU 线程数")
        p.add_argument("--lora-rank", type=int, default=None,
                       help=">0 时启用 LoRA 参数高效微调（默认关闭 = 全参）")
        p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha（默认 16）")
        if with_data:
            p.add_argument("--data", required=True, help="对话 JSON 数据路径")
            p.add_argument("--val", default=None, help="验证对话 JSON（可选）")
            p.add_argument("--batch-size", type=int, default=4, help="SFT 批大小（默认 4）")
            p.add_argument("--max-seq-len", type=int, default=None,
                           help="SFT 最大序列长度（默认取模型 max_seq_len）")
            p.add_argument("--train-on-last-turns", type=int, default=1,
                           help="只对最后 N 轮回答计算 loss；0 = 全部 assistant 轮")
        else:
            _add_data_args(p)
        _add_resume_args(p)

    p_ft = sub.add_parser("finetune", help="从 checkpoint 在文本流上继续训练（全参/LoRA）")
    _add_posttrain_args(p_ft, with_data=False)
    p_ft.set_defaults(func=cmd_finetune)

    p_sft = sub.add_parser("sft", help="从 checkpoint 做指令微调（对话数据，answer-only loss）")
    _add_posttrain_args(p_sft, with_data=True)
    p_sft.set_defaults(func=cmd_sft)

    p_gen = sub.add_parser("generate", help="从 checkpoint 续写文本")
    p_gen.add_argument("--prompt", default="机器学习是", help="提示文本")
    p_gen.add_argument("--ckpt", default=None,
                       help="checkpoint 路径（默认 final > 最大 step 自动发现）")
    p_gen.add_argument("--max-new-tokens", type=int, default=24)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--top-p", type=float, default=0.9, help="核采样阈值（1.0 = 关闭）")
    p_gen.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="重复惩罚（1.0 = 关闭，>1 抑制复读）")
    p_gen.add_argument("--no-repeat-ngram", type=int, default=3,
                       help="禁止重复 n-gram（0 = 关闭）")
    p_gen.add_argument("--greedy", action="store_true", help="贪心解码")
    p_gen.add_argument("--chat", action="store_true",
                       help="用 ChatML 模板包裹 prompt（SFT 后的对话模型使用）")
    p_gen.add_argument("--device", default=None,
                       help="推理设备 (auto/cpu/cuda/npu；默认 auto 按优先级自动选)")
    p_gen.add_argument("--dtype", default=None,
                       help="推理混合精度 (auto/fp32/bf16/fp16；默认 auto 按设备能力选)")
    p_gen.set_defaults(func=cmd_generate)

    p_chat = sub.add_parser("chat", help="交互式多轮对话（REPL，模型常驻）")
    p_chat.add_argument("--ckpt", default=None,
                        help="checkpoint 路径（默认 final > 最大 step 自动发现）")
    p_chat.add_argument("--device", default=None,
                        help="推理设备 (auto/cpu/cuda/npu；默认 auto 按优先级自动选)")
    p_chat.add_argument("--dtype", default=None,
                        help="推理混合精度 (auto/fp32/bf16/fp16；默认 auto 按设备能力选)")
    p_chat.add_argument("--system", default=None, help="system 提示（可选）")
    p_chat.add_argument("--max-history-turns", type=int, default=12,
                        help="最多保留的历史对话轮数（0 = 不限制，默认 12）")
    p_chat.add_argument("--max-new-tokens", type=int, default=128)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top-k", type=int, default=40)
    p_chat.add_argument("--top-p", type=float, default=0.9, help="核采样阈值（1.0 = 关闭）")
    p_chat.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="重复惩罚（1.0 = 关闭，>1 抑制复读）")
    p_chat.add_argument("--no-repeat-ngram", type=int, default=3,
                        help="禁止重复 n-gram（0 = 关闭）")
    p_chat.add_argument("--greedy", action="store_true", help="贪心解码")
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
