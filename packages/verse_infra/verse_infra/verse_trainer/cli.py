"""VerseTrainer CLI 入口（Part4K1 Task 6.4 + 6.5 + Part4K2 Task 8.4）。

6 个子命令（注册为 console_scripts）：

- ``verse-train``：预训练。
  参数：``--config`` / ``--device cpu|cuda|npu`` / ``--single-sample``
  / ``--parallel-chunks N`` / ``--max-steps`` / ``--resume`` / ``--amp``
  / ``--prompt`` / ``--completion`` / ``--single-file``
- ``verse-finetune``：微调。
  参数：``--config`` / ``--method lora|full`` / ``--device`` / ``--data``
- ``verse-posttrain``：后训练。
  参数：``--config`` / ``--rl nexrl|sft|dpo`` / ``--device`` / ``--data``
- ``verse-eval``：评估 + 打分。
  参数：``--config`` / ``--checkpoint`` / ``--prompts-file`` / ``--references-file`` / ``--score``
- ``verse-tokenize``：tokenizer 训练 / 加载 / 转换。
  参数：``--train`` / ``--load`` / ``--convert`` / ``--from-hf Qwen/Qwen3.5-35B-A3B``
- ``verse-download``：数据集下载器（任意 URL + HuggingFace datasets）。
  参数：``--url`` / ``--hf`` / ``--split`` / ``--output`` / ``--to-npz``
  / ``--text-key`` / ``--workers`` / ``--no-resume``
- ``verse-convert``：模型格式转换（.pt ↔ .vn，Part4K2 Task 1.8）。
  参数：``--input`` / ``--output`` / ``--chat-template`` / ``--tokenizer`` / ``--arch``

主函数 :func:`main` 用 ``sys.argv[1]`` 分发子命令。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

# 路径自举：确保 verse_trainer / verse_torch / verse_nex / verse_tokenizer 可被 import
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGES_DIR = os.path.dirname(_HERE)
for _dep in ("verse_torch", "verse_nex", "verse_tokenizer"):
    _dep_path = os.path.join(_PACKAGES_DIR, _dep)
    if os.path.isdir(_dep_path) and _dep_path not in sys.path:
        sys.path.insert(0, _dep_path)


# ---------------------------------------------------------------------------
# verse-train：预训练
# ---------------------------------------------------------------------------


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-train",
        description="VerseTrainer 预训练入口（支持单样本 / 并行 chunks / 断点续训）",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（config.yml）")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", "npu"],
                        help="设备（cpu/cuda/npu）；默认从 config 读取或 cpu")
    parser.add_argument("--single-sample", action="store_true",
                        help="单样本模式（配合 --prompt / --completion / --single-file）")
    parser.add_argument("--prompt", default=None,
                        help="单样本模式：prompt 文本")
    parser.add_argument("--completion", default=None,
                        help="单样本模式：completion 文本")
    parser.add_argument("--single-file", default=None,
                        help="单样本模式：单文件路径（内容当作纯文本）")
    parser.add_argument("--parallel-chunks", type=int, default=None,
                        help="并行训练 chunk 数（1=标准 Trainer，>1=ParallelTrainer）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖 config 的 max_steps")
    parser.add_argument("--resume", action="store_true",
                        help="从 last checkpoint 断点续训")
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度训练（GPU 后端）")
    parser.add_argument("--loss-optimizer", action="store_true",
                        help="启用 LossOptimizer（plateau 重走 + NaN/Inf 跳过）")
    parser.add_argument("--partition-training", action="store_true",
                        help="启用智能分区训练（LayerWiseTrainer，按 layer 分组训练+卸载+合并）")
    parser.add_argument("--partition-size", type=int, default=2,
                        help="智能分区训练：每组 layer 数量（默认 2）")
    parser.add_argument("--offload-dir", default=None,
                        help="智能分区训练：硬盘卸载目录（默认 save_dir/partition_offload）")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式（仅打印最终结果）")
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志模式（打印 chunk 拆分 / 排序等详细信息）")
    parser.add_argument("--parallel-strategy", default=None,
                        choices=["sequential", "round_robin"],
                        help="并行训练数据分配策略（默认 sequential，round_robin 为不重叠数据子集）")
    parser.add_argument("--base-dir", default=None,
                        help="配置中相对路径的基准目录（默认 config 同级目录）")
    parser.add_argument("--no-eval", action="store_true",
                        help="跳过训练后自动评估打分（默认训练后自动评估）")
    parser.add_argument("--eval-prompts", default=None,
                        help="自定义评估 prompt JSON 文件路径"
                             "（格式：[{\"prompt\": \"...\", \"reference\": \"...\"}, ...]）")
    return parser


def train_main(argv: Optional[List[str]] = None) -> int:
    """verse-train 主入口。"""
    parser = _build_train_parser()
    args = parser.parse_args(argv)

    # 单样本模式必须提供 prompt / completion / single-file 之一
    if args.single_sample:
        if not (args.prompt or args.completion or args.single_file):
            parser.error("--single-sample 需配合 --prompt / --completion / --single-file 使用")

    # 解析 config 与 base_dir
    config_path, base_dir = _resolve_config_and_base(args)
    if config_path is None:
        parser.error("--config 必填（除非用 --single-sample + --prompt）")

    # 覆盖 config 的 parallel_chunks / max_steps / parallel_strategy（通过临时 config 文件）
    overrides = {}
    if args.parallel_chunks is not None:
        overrides["parallel_chunks"] = int(args.parallel_chunks)
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.parallel_strategy is not None:
        overrides["parallel_strategy"] = str(args.parallel_strategy)
    effective_config = _apply_config_overrides(config_path, overrides)

    # 构造 single_sample dict
    single_sample = None
    if args.single_sample:
        if args.single_file:
            single_sample = None  # 走 single_file 路径
        else:
            single_sample = {
                "prompt": args.prompt or "",
                "completion": args.completion or "",
            }

    # Part4K2.5 Task 4: 解析 --eval-prompts（JSON 文件）
    eval_config = None
    if args.eval_prompts:
        import json as _json
        with open(args.eval_prompts, "r", encoding="utf-8") as f:
            prompts_data = _json.load(f)
        eval_config = {"prompts": prompts_data}

    from .trainer import train
    result = train(
        config_path=effective_config,
        base_dir=base_dir,
        device=args.device,
        single_sample=single_sample,
        single_file=args.single_file if args.single_sample else None,
        max_steps_override=args.max_steps,
        resume=args.resume,
        amp=args.amp,
        enable_loss_optimizer=args.loss_optimizer,
        partition_training=args.partition_training,
        partition_size=args.partition_size,
        offload_dir=args.offload_dir,
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        # Part4K2.5 Task 4: 默认 eval_after=True，--no-eval 跳过
        eval_after=not getattr(args, "no_eval", False),
        eval_config=eval_config,
    )
    print(f"\n[train] 结果：{result['best_val_loss']:.4f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verse-continue：持续训练（Part4K2 Task 7.3）
# ---------------------------------------------------------------------------


def _build_continue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-continue",
        description="VerseTrainer 持续训练入口（从 checkpoint 加载模型继续追加训练）",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint 文件路径（best.pt / resume.pt / 任意 pickle）")
    parser.add_argument("--additional-steps", type=int, required=True,
                        help="追加训练步数")
    parser.add_argument("--config", required=True, help="配置文件路径（config.yml）")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", "npu"],
                        help="设备（cpu/cuda/npu）；默认从 config 读取或 cpu")
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度训练（GPU 后端）")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式（仅打印最终结果）")
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志模式")
    parser.add_argument("--base-dir", default=None,
                        help="配置中相对路径的基准目录（默认 config 同级目录）")
    return parser


def continue_main(argv: Optional[List[str]] = None) -> int:
    """verse-continue 主入口。

    用法::

        verse-continue --checkpoint checkpoints/best.pt \\
            --additional-steps 1000 --config config.yml
        verse-continue --checkpoint checkpoints/resume.pt \\
            --additional-steps 500 --config config.yml --device cuda

    与 ``verse-train --resume`` 的区别：
    - ``--resume`` 是中断后恢复（从中断点继续，目标是完成原计划的步数）
    - ``verse-continue`` 是训练完成后继续追加训练
      （新目标 = additional_steps，独立步数）
    """
    parser = _build_continue_parser()
    args = parser.parse_args(argv)

    config_path, base_dir = _resolve_config_and_base(args)

    from .trainer import continue_train
    result = continue_train(
        checkpoint=args.checkpoint,
        additional_steps=int(args.additional_steps),
        config_path=config_path,
        base_dir=base_dir,
        device=args.device,
        amp=args.amp,
        quiet=args.quiet,
        verbose=args.verbose,
    )
    print(f"\n[continue] 结果：{result['best_val_loss']:.4f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verse-finetune：微调
# ---------------------------------------------------------------------------


def _build_finetune_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-finetune",
        description="VerseTrainer 微调入口（LoRA / 全量）",
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--method", default="full", choices=["lora", "full"],
                        help="微调方法（lora / full，默认 full）")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", "npu"],
                        help="设备（默认 cpu）")
    parser.add_argument("--data", default=None,
                        help="微调数据路径（jsonl，覆盖 config 的 data.train_path）")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA 秩（默认 8）")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha（默认 16.0）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖 config 的 max_steps")
    parser.add_argument("--base-dir", default=None,
                        help="配置中相对路径的基准目录")
    return parser


def finetune_main(argv: Optional[List[str]] = None) -> int:
    """verse-finetune 主入口。"""
    parser = _build_finetune_parser()
    args = parser.parse_args(argv)

    config_path, base_dir = _resolve_config_and_base(args)
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.data is not None:
        overrides["train_path"] = args.data
    effective_config = _apply_config_overrides(config_path, overrides)

    # 复用 train 入口，通过 enable_loss_optimizer=False 走标准 VerseNexTrainer
    # LoRA / 全量的区分由 config 控制（method=lora 时在 train 内部用 LoRATrainer）
    # 简化：本版本 verse-finetune 直接调用 train，method=lora 时打印提示
    # （真正的 LoRA 包装由 SFTTrainer/LoRATrainer 在数据格式匹配时触发）
    if args.method == "lora":
        print("[finetune] method=lora，将由 LoRATrainer 包装模型", flush=True)
        # 把 lora 参数写入 config overrides
        effective_config = _apply_config_overrides(
            effective_config,
            {"method": "lora", "lora_r": args.lora_r, "lora_alpha": args.lora_alpha},
        )
    else:
        print("[finetune] method=full，全量微调", flush=True)

    from .trainer import train
    result = train(
        config_path=effective_config,
        base_dir=base_dir,
        device=args.device,
        max_steps_override=args.max_steps,
        enable_loss_optimizer=False,
    )
    print(f"\n[finetune] 结果：{result['best_val_loss']:.4f}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verse-posttrain：后训练
# ---------------------------------------------------------------------------


def _build_posttrain_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-posttrain",
        description="VerseTrainer 后训练入口（RL / SFT / DPO）",
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--rl", default="sft", choices=["nexrl", "sft", "dpo"],
                        help="后训练算法（nexrl / sft / dpo，默认 sft）")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", "npu"],
                        help="设备（默认 cpu）")
    parser.add_argument("--data", default=None,
                        help="后训练数据路径（jsonl）")
    parser.add_argument("--prompts", default=None,
                        help="RL 模式（--rl nexrl）的 prompts 文件（每行一条）")
    parser.add_argument("--n-epochs", type=int, default=2,
                        help="RL 训练 epoch 数（默认 2）")
    parser.add_argument("--n-rollouts", type=int, default=2,
                        help="每个 prompt 的 rollout 数（默认 2）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="SFT/DPO 模式的 max_steps（覆盖 config）")
    parser.add_argument("--base-dir", default=None,
                        help="配置中相对路径的基准目录")
    return parser


def posttrain_main(argv: Optional[List[str]] = None) -> int:
    """verse-posttrain 主入口。"""
    parser = _build_posttrain_parser()
    args = parser.parse_args(argv)

    config_path, base_dir = _resolve_config_and_base(args)

    if args.rl == "nexrl":
        # RL 后训练：用 RLTrainer 包装 NexTrainer
        return _run_rl_posttrain(args, config_path, base_dir)

    # SFT / DPO 后训练：复用 train 入口
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.data is not None:
        overrides["train_path"] = args.data
    overrides["method"] = args.rl  # sft / dpo
    effective_config = _apply_config_overrides(config_path, overrides)

    print(f"[posttrain] --rl {args.rl}，复用 train 入口", flush=True)
    from .trainer import train
    result = train(
        config_path=effective_config,
        base_dir=base_dir,
        device=args.device,
        max_steps_override=args.max_steps,
        enable_loss_optimizer=False,
    )
    print(f"\n[posttrain] 结果：{result['best_val_loss']:.4f}", flush=True)
    return 0


def _run_rl_posttrain(args, config_path: str, base_dir: str) -> int:
    """RL 后训练（--rl nexrl）：用 RLTrainer 包装 NexTrainer。"""
    # 1. 加载 config 构建模型 + tokenizer
    from .trainer import _load_full_config, _build_model, _load_tokenizer, _resolve_path
    full_cfg = _load_full_config(config_path)
    model_cfg = full_cfg.get("model", {})
    tok_cfg = full_cfg.get("tokenizer", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})
    save_dir = _resolve_path(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))

    tok = _load_tokenizer(tok_cfg, base_dir, save_dir)
    vocab_size = len(tok)
    model, _ = _build_model(model_cfg, vocab_size)

    # 设备迁移
    if args.device is not None and args.device != "cpu":
        if hasattr(model, "to"):
            try:
                model.to(args.device)
            except Exception as e:
                print(f"[posttrain] 警告：迁移模型到 {args.device} 失败：{e}",
                      flush=True)

    # 2. 加载 prompts
    prompts = []
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if line.strip() and not line.strip().startswith("#"):
                    prompts.append(line)
    elif args.data:
        # 从 jsonl 读 prompts（取 prompt 字段或首条 user content）
        import json
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "prompt" in obj:
                        prompts.append(str(obj["prompt"]))
                    elif "messages" in obj:
                        for m in obj["messages"]:
                            if m.get("role") == "user":
                                prompts.append(str(m.get("content", "")))
                                break
                except Exception:
                    continue
    if not prompts:
        # 兜底默认 prompts
        prompts = ["1+1=", "你好", "hello"]
    print(f"[posttrain] RL prompts: {len(prompts)} 条", flush=True)

    # 3. RLTrainer 训练
    from .rl_trainer import RLTrainer
    rl_cfg = {
        "ppo_epochs": 2,
        "max_new_tokens": 8,
        "use_value": True,
        "lr": 1e-4,
        "target_kl": 10.0,
        "kl_adaptive": True,
    }
    trainer = RLTrainer(
        model=model,
        tokenizer=tok,
        cfg=rl_cfg,
        save_dir=save_dir,
    )
    losses, kls, rewards = trainer.fit(
        prompts=prompts,
        n_epochs=args.n_epochs,
        n_rollouts_per_prompt=args.n_rollouts,
    )
    print(f"\n[posttrain] RL 完成：losses={len(losses)} kls={len(kls)} "
          f"rewards={len(rewards)}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verse-eval：评估 + 打分
# ---------------------------------------------------------------------------


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-eval",
        description="VerseTrainer 评估 + 打分入口",
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 文件路径（默认自动查找 best.pt / cometspark.pt）")
    parser.add_argument("--prompts-file", default=None,
                        help="prompts 文件路径（每行一条，忽略空行与 # 注释）")
    parser.add_argument("--references-file", default=None,
                        help="参考答案文件路径（每行一条，与 prompts 一一对应）")
    parser.add_argument("--score", action="store_true",
                        help="启用打分模式（需 --references-file）")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="每条 prompt 生成最大 token 数；默认不限制（None），"
                             "模型生成到 EOS 自然停止（安全上限 100K 防无限循环）。"
                             "显式指定时按值限制。")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度（默认 1.0）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="top-k 采样")
    parser.add_argument("--top-p", type=float, default=None,
                        help="nucleus sampling 阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--base-dir", default=None,
                        help="配置中相对路径的基准目录")
    return parser


def eval_main(argv: Optional[List[str]] = None) -> int:
    """verse-eval 主入口。"""
    parser = _build_eval_parser()
    args = parser.parse_args(argv)

    config_path, base_dir = _resolve_config_and_base(args)

    # 加载 prompts
    prompts = None
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if line.strip() and not line.strip().startswith("#"):
                    prompts.append(line)
        if not prompts:
            prompts = None

    from .evaluate import evaluate
    result = evaluate(
        config_path=config_path,
        base_dir=base_dir,
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        score=args.score,
        references_file=args.references_file,
        checkpoint=args.checkpoint,
    )
    print(f"\n[eval] 生成 {len(result['results'])} 条样本", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verse-tokenize：tokenizer 训练 / 加载 / 转换
# ---------------------------------------------------------------------------


def _build_tokenize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-tokenize",
        description="VerseTokenizer 训练 / 加载 / 转换入口",
    )
    parser.add_argument("--train", default=None,
                        help="训练模式：corpus 文件路径")
    parser.add_argument("--vocab-size", type=int, default=256,
                        help="训练目标 vocab_size（默认 256）")
    parser.add_argument("--kind", default="bpe", choices=["bpe", "byte", "wordpiece"],
                        help="tokenizer 类型（bpe / byte / wordpiece，默认 bpe）")
    parser.add_argument("--save", default=None,
                        help="保存路径（默认 ./tokenizer.json）")
    parser.add_argument("--load", default=None,
                        help="加载模式：tokenizer 文件路径")
    parser.add_argument("--convert", default=None,
                        help="转换模式：HF tokenizer.json 文件路径，转为本包格式")
    parser.add_argument("--from-hf", default=None,
                        help="从 HuggingFace 下载 tokenizer（如 Qwen/Qwen3.5-35B-A3B）")
    parser.add_argument("--text", default=None,
                        help="加载 / 转换后测试编码的文本")
    return parser


def tokenize_main(argv: Optional[List[str]] = None) -> int:
    """verse-tokenize 主入口。"""
    parser = _build_tokenize_parser()
    args = parser.parse_args(argv)

    from verse_infra.verse_tokenizer import (
        BPETokenizer, ByteTokenizer, WordPieceTokenizer, load_tokenizer,
    )

    # 1. 训练模式
    if args.train:
        save_path = args.save or "tokenizer.json"
        if args.kind == "byte":
            tok = ByteTokenizer()
            tok.save(save_path)
            print(f"[tokenize] ByteTokenizer 已保存到 {save_path}", flush=True)
        elif args.kind == "bpe":
            with open(args.train, "r", encoding="utf-8") as f:
                corpus = f.read()
            tok = BPETokenizer.train(corpus, vocab_size=int(args.vocab_size))
            tok.save(save_path)
            print(f"[tokenize] BPETokenizer 已训练并保存到 {save_path} "
                  f"(vocab_size={len(tok)})", flush=True)
        elif args.kind == "wordpiece":
            with open(args.train, "r", encoding="utf-8") as f:
                corpus = f.read()
            tok = WordPieceTokenizer.train(corpus, vocab_size=int(args.vocab_size))
            tok.save(save_path)
            print(f"[tokenize] WordPieceTokenizer 已训练并保存到 {save_path}",
                  flush=True)
        if args.text:
            ids = _safe_tok_encode(tok, args.text)
            print(f"[tokenize] 测试编码 {args.text!r} → {ids}", flush=True)
        return 0

    # 2. 加载模式
    if args.load:
        kind = args.kind if args.kind != "byte" else "byte"
        tok = load_tokenizer(kind=kind, path=args.load)
        print(f"[tokenize] 已加载 tokenizer {args.load} (vocab_size={len(tok)})",
              flush=True)
        if args.text:
            ids = _safe_tok_encode(tok, args.text)
            decoded = _safe_tok_decode(tok, ids)
            print(f"[tokenize] 测试编码 {args.text!r} → {ids}", flush=True)
            print(f"[tokenize] 测试解码 → {decoded!r}", flush=True)
        return 0

    # 3. 转换模式（HF tokenizer.json → 本包格式）
    if args.convert:
        tok = BPETokenizer.from_pretrained(args.convert)
        save_path = args.save or "tokenizer_converted.json"
        tok.save(save_path)
        print(f"[tokenize] 已转换 {args.convert} → {save_path} "
              f"(vocab_size={len(tok)})", flush=True)
        if args.text:
            ids = _safe_tok_encode(tok, args.text)
            print(f"[tokenize] 测试编码 {args.text!r} → {ids}", flush=True)
        return 0

    # 4. 从 HuggingFace 下载
    if args.from_hf:
        try:
            tok = BPETokenizer.from_pretrained(args.from_hf)
            save_path = args.save or "tokenizer_hf.json"
            tok.save(save_path)
            print(f"[tokenize] 已从 HF 下载 {args.from_hf} → {save_path} "
                  f"(vocab_size={len(tok)})", flush=True)
            if args.text:
                ids = _safe_tok_encode(tok, args.text)
                print(f"[tokenize] 测试编码 {args.text!r} → {ids}", flush=True)
            return 0
        except Exception as e:
            print(f"[tokenize] 从 HF 下载失败：{e}", file=sys.stderr, flush=True)
            return 1

    parser.error("必须指定 --train / --load / --convert / --from-hf 之一")


def _safe_tok_encode(tok, text):
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        try:
            return list(tok.encode(text))
        except Exception:
            return []


def _safe_tok_decode(tok, ids):
    try:
        return tok.decode(list(ids))
    except TypeError:
        try:
            return tok.decode(list(ids), strip_special=True)
        except Exception:
            return tok.decode(list(ids))


# ---------------------------------------------------------------------------
# verse-convert：模型格式转换（.pt ↔ .vn，Part4K2 Task 1.8）
# ---------------------------------------------------------------------------


def _build_convert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-convert",
        description="模型格式转换（.pt ↔ .vn，基于 safetensors 的性能优化格式）",
    )
    parser.add_argument("--input", required=True,
                        help="输入文件路径（.pt 或 .vn，自动检测方向）")
    parser.add_argument("--output", required=True,
                        help="输出文件路径（.vn 或 .pt）")
    parser.add_argument("--chat-template", default=None,
                        help="chat_template.jinja 文件路径（仅 .pt → .vn 时附加）")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer.json 文件路径（仅 .pt → .vn 时附加）")
    parser.add_argument("--arch", default=None,
                        help="覆盖架构名（仅 .pt → .vn 时生效，默认从 .pt payload 读取）")
    return parser


def convert_main(argv: Optional[List[str]] = None) -> int:
    """verse-convert 主入口：在 .pt 与 .vn 之间互转。

    用法::

        verse-convert --input model.pt --output model.vn
        verse-convert --input model.vn --output model.pt
        verse-convert --input model.pt --output model.vn \\
            --chat-template chat_template.jinja --tokenizer tokenizer.json
    """
    parser = _build_convert_parser()
    args = parser.parse_args(argv)

    from verse_torch.vn_format import pt_to_vn, vn_to_pt, convert_format

    src = args.input
    dst = args.output
    src_lower = src.lower()
    dst_lower = dst.lower()

    # 读取附加内容（仅 pt→vn 有意义）
    chat_template = None
    if args.chat_template:
        with open(args.chat_template, "r", encoding="utf-8") as f:
            chat_template = f.read()
    tokenizer = args.tokenizer  # 透传路径给 pt_to_vn

    try:
        if src_lower.endswith(".pt") and dst_lower.endswith(".vn"):
            pt_to_vn(
                src, dst,
                arch=args.arch,
                chat_template=chat_template,
                tokenizer=tokenizer,
            )
            print(f"[convert] .pt → .vn 完成：{src} → {dst}", flush=True)
        elif src_lower.endswith(".vn") and dst_lower.endswith(".pt"):
            vn_to_pt(src, dst)
            print(f"[convert] .vn → .pt 完成：{src} → {dst}", flush=True)
        else:
            # 交给 convert_format 自动检测并给出明确错误
            convert_format(src, dst)
            print(f"[convert] 完成：{src} → {dst}", flush=True)
    except Exception as e:
        print(f"[convert] 转换失败：{type(e).__name__}: {e}", file=sys.stderr,
              flush=True)
        return 1
    return 0


# ---------------------------------------------------------------------------
# verse-download：数据集下载器
# ---------------------------------------------------------------------------


def _build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verse-download",
        description="VerseTrainer 数据集下载器（任意 URL + HuggingFace datasets）",
    )
    parser.add_argument("--url", default=None,
                        help="下载 URL（http/https）")
    parser.add_argument("--hf", default=None,
                        help="HuggingFace dataset repo ID（如 wikitext）")
    parser.add_argument("--split", default="train",
                        help="HF dataset split（默认 train）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出路径（文件或目录）；--to-npz 时为 .npz 路径")
    parser.add_argument("--to-npz", action="store_true",
                        help="下载后自动转 .npz 缓存（与 CachedDataset 对齐）")
    parser.add_argument("--text-key", default="text",
                        help="文本字段名（默认 text，用于 JSON/CSV 解析）")
    parser.add_argument("--workers", type=int, default=4,
                        help="下载线程数（默认 4）")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用断点续传")
    return parser


def download_main(argv: Optional[List[str]] = None) -> int:
    """verse-download 主入口。"""
    parser = _build_download_parser()
    args = parser.parse_args(argv)

    if not args.url and not args.hf:
        parser.error("必须指定 --url 或 --hf 之一")

    # 通过 verse_infra 顶层导出获取 DatasetDownloader
    # （verse_infra.__init__ 的 __getattr__ 会自动加载 data/downloader.py）
    from verse_infra import DatasetDownloader
    downloader = DatasetDownloader(num_workers=args.workers)

    if args.url:
        if args.to_npz:
            npz_path = downloader.download_and_cache(
                args.url, output_path=args.output, text_key=args.text_key,
            )
            print(f"[download] 已下载并缓存：{npz_path}", flush=True)
        else:
            path = downloader.download_url(
                args.url, output_path=args.output,
                resume=not args.no_resume,
            )
            print(f"[download] 已下载：{path}", flush=True)
    else:  # args.hf
        if args.to_npz:
            npz_path = downloader.download_and_cache(
                args.hf, output_path=args.output, text_key=args.text_key,
            )
            print(f"[download] 已下载并缓存：{npz_path}", flush=True)
        else:
            path = downloader.download_hf(
                args.hf, split=args.split, output_dir=args.output,
            )
            print(f"[download] 已下载：{path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _resolve_config_and_base(args) -> tuple:
    """解析 config 路径与 base_dir。

    若 args.config 为 None，返回 (None, base_dir)。
    若 args.base_dir 为 None，默认用 config 同级目录。
    """
    if args.config is None:
        return None, os.getcwd()
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    if args.base_dir:
        base_dir = os.path.abspath(args.base_dir)
    else:
        base_dir = os.path.dirname(config_path) or os.getcwd()
    return config_path, base_dir


def _apply_config_overrides(config_path: str, overrides: dict) -> str:
    """把 overrides 写入临时 config 文件，返回临时文件路径。

    overrides 是 {section.key: value} 形式的扁平 dict，如：
        {"training.max_steps": 100, "model.arch": "versenex"}
    或 {section: {key: value}} 形式的嵌套 dict。
    """
    if not overrides:
        return config_path
    try:
        from .trainer import _load_full_config
        full_cfg = _load_full_config(config_path)
    except Exception:
        full_cfg = {}

    # 把扁平 key 展开
    for k, v in overrides.items():
        if "." in k:
            section, key = k.split(".", 1)
            full_cfg.setdefault(section, {})[key] = v
        elif isinstance(v, dict):
            full_cfg.setdefault(k, {}).update(v)
        else:
            # 顶层标量：写到 training 段兜底
            full_cfg.setdefault("training", {})[k] = v

    # 保存临时文件
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".yml", prefix="verse_trainer_override_")
    os.close(fd)
    try:
        import yaml
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(full_cfg, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        # 无 PyYAML：用 verse_trainer.trainer._parse_yaml_minimal 的逆操作
        from .trainer import _parse_scalar
        lines = []
        for section, sub in full_cfg.items():
            if isinstance(sub, dict):
                lines.append(f"{section}:")
                for k, v in sub.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{section}: {sub}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------
# 主分发函数 main
# ---------------------------------------------------------------------------

# 子命令 → 入口函数映射
_SUBCOMMANDS = {
    "verse-train": ("train", train_main),
    "verse-continue": ("continue", continue_main),
    "verse-finetune": ("finetune", finetune_main),
    "verse-posttrain": ("posttrain", posttrain_main),
    "verse-eval": ("eval", eval_main),
    "verse-tokenize": ("tokenize", tokenize_main),
    "verse-download": ("download", download_main),
    "verse-convert": ("convert", convert_main),
    # 短别名
    "train": ("train", train_main),
    "continue": ("continue", continue_main),
    "finetune": ("finetune", finetune_main),
    "posttrain": ("posttrain", posttrain_main),
    "eval": ("eval", eval_main),
    "tokenize": ("tokenize", tokenize_main),
    "download": ("download", download_main),
    "convert": ("convert", convert_main),
}


def main(argv: Optional[List[str]] = None) -> int:
    """主分发函数：用 ``sys.argv[1]`` 分发子命令。

    用法：
        python -m verse_trainer.cli verse-train --config ...
        python -m verse_trainer.cli verse-eval --config ... --score

    若 ``argv`` 为 None，从 ``sys.argv[1:]`` 读取。
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(
            "VerseTrainer CLI\n"
            "用法：verse-train | verse-continue | verse-finetune | verse-posttrain | "
            "verse-eval | verse-tokenize | verse-download | verse-convert [options]\n\n"
            "子命令：\n"
            "  verse-train       预训练\n"
            "  verse-continue    持续训练（从 checkpoint 加载继续追加训练）\n"
            "  verse-finetune    微调（lora / full）\n"
            "  verse-posttrain   后训练（nexrl / sft / dpo）\n"
            "  verse-eval        评估 + 打分\n"
            "  verse-tokenize    tokenizer 训练 / 加载 / 转换\n"
            "  verse-download    数据集下载器（URL / HuggingFace）\n"
            "  verse-convert     模型格式转换（.pt ↔ .vn）\n",
            file=sys.stderr, flush=True,
        )
        return 1

    cmd = argv[0]
    rest = argv[1:]
    if cmd in ("-h", "--help"):
        print(
            "VerseTrainer CLI\n"
            "用法：verse-train | verse-continue | verse-finetune | verse-posttrain | "
            "verse-eval | verse-tokenize | verse-download | verse-convert [options]",
            file=sys.stderr, flush=True,
        )
        return 0

    if cmd not in _SUBCOMMANDS:
        print(f"未知子命令：{cmd!r}", file=sys.stderr, flush=True)
        print(f"可用子命令：{', '.join(_SUBCOMMANDS.keys())}", file=sys.stderr,
              flush=True)
        return 1

    _, fn = _SUBCOMMANDS[cmd]
    try:
        return fn(rest)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except Exception as e:
        import traceback
        print(f"\n[{cmd}] 执行失败：{type(e).__name__}: {e}", file=sys.stderr,
              flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
