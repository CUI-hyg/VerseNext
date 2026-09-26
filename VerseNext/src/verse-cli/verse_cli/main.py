"""VerseNext CLI。

子命令：
- ``verse train``  训练模型
- ``verse evolve`` 启动 RSI 自进化循环
- ``verse bench``  基准测速
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from verse_core import get_logger

logger = get_logger("cli")


# --------------------------------------------------------------------- data

def _make_tokens(vocab_size: int, length: int, seed: int):
    """生成合成 token 序列（带简单重复结构，使 loss 可以下降）。

    真实场景请替换为 tokenizer 产出的 token 文件。
    """
    rng = random.Random(seed)
    motif = [rng.randrange(vocab_size) for _ in range(64)]
    tokens = []
    while len(tokens) < length:
        # 70% 概率重复 motif 片段，30% 随机 token，形成可学习模式
        if rng.random() < 0.7:
            k = rng.randrange(8, 64)
            start = rng.randrange(0, len(motif) - k)
            tokens.extend(motif[start:start + k])
        else:
            tokens.extend(rng.randrange(vocab_size) for _ in range(rng.randrange(1, 16)))
    return tokens[:length]


def _build_trainer_config(args) -> "TrainerConfig":
    from verse_nn.transformer import VerseTransformerConfig
    from verse_trainer.data import DataConfig
    from verse_trainer.optim import OptimizerConfig
    from verse_trainer.trainer import TrainerConfig

    return TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ),
        data=DataConfig(batch_size=args.batch_size, seq_len=args.seq_len),
        optim=OptimizerConfig(lr=args.lr, warmup_steps=min(50, args.steps // 5 or 1), max_steps=args.steps),
        steps=args.steps,
        log_every=max(args.steps // 10, 1),
        eval_every=max(args.steps // 2, 1),
        eval_batches=5,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=0,
        compile=args.compile,
        device=args.device,
        seed=args.seed,
    )


def cmd_train(args) -> int:
    from verse_trainer.trainer import Trainer

    config = _build_trainer_config(args)
    tokens = _make_tokens(config.model.vocab_size, args.tokens, args.seed)
    trainer = Trainer(config)
    logger.info("模型参数量: %.2fM | 设备: %s", trainer.model.num_parameters() / 1e6, trainer.device)
    summary = trainer.train(tokens, tokens[-config.data.seq_len * 40:])
    logger.info("训练完成: %s", summary)
    if args.save:
        path = trainer.save_checkpoint(args.save)
        logger.info("模型已保存: %s", path)
    return 0


def cmd_evolve(args) -> int:
    from verse_nn.transformer import VerseTransformerConfig
    from verse_rsi.evolve import EvolveConfig, Evolver
    from verse_rsi.evaluator import EvaluatorConfig
    from verse_rsi.individual import Individual
    from verse_trainer.data import DataConfig
    from verse_trainer.optim import OptimizerConfig
    from verse_trainer.trainer import TrainerConfig

    # 进化期用短训练预算快速筛选
    base_trainer = TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ),
        data=DataConfig(batch_size=args.batch_size, seq_len=args.seq_len),
        optim=OptimizerConfig(warmup_steps=10, max_steps=args.steps),
        steps=args.steps,
        log_every=max(args.steps, 1),
        eval_every=max(args.steps, 1),
        eval_batches=3,
        ckpt_every=0,
        device=args.device,
        seed=args.seed,
    )
    config = EvolveConfig(
        generations=args.generations,
        population_size=args.population,
        elite_k=1,
        evaluator_config=EvaluatorConfig(base_trainer=base_trainer, speed_weight=args.speed_weight),
        archive_path=args.archive,
        seed=args.seed,
    )
    tokens = _make_tokens(args.vocab_size, args.tokens, args.seed)
    evolver = Evolver(config)
    # 种子个体继承 CLI 指定的基础架构
    seed = Individual(model_config=base_trainer.model)
    best = evolver.run(tokens, tokens[-base_trainer.data.seq_len * 40:], seed_individual=seed)
    logger.info("最优个体: %s", best.to_dict())

    if args.refine_steps > 0:
        from dataclasses import replace

        cfg = config.evaluator_config
        cfg.base_trainer = replace(cfg.base_trainer, steps=args.refine_steps)
        summary = evolver.refine_best(best, tokens, tokens[-base_trainer.data.seq_len * 40:])
        logger.info("最优个体重训结果: %s", summary)
    return 0


def cmd_bench(args) -> int:
    import time

    import torch

    from verse_nn.transformer import VerseTransformer, VerseTransformerConfig

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VerseTransformer(VerseTransformerConfig(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads,
    )).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    def _step():
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    model.train()
    for _ in range(3):  # warmup
        _step()
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        _step()
    dt = time.perf_counter() - t0
    tps = n * args.batch_size * args.seq_len / dt
    logger.info(
        "bench | %s | %.2fM params | %.1f ms/step | %.0f tokens/s",
        device, model.num_parameters() / 1e6, dt / n * 1000, tps,
    )
    return 0


def cmd_train_tokenizer(args) -> int:
    from verse_tokenizer import BPETokenizer, ChatTemplate

    if args.input:
        texts = [Path(p).read_text() for p in args.input]
    else:
        # 无输入文件时用内置合成语料演示
        texts = [_synth_corpus(args.tokens)]
        logger.info("未指定 --input，使用合成语料演示（正式训练请传文本文件）")
    tok = BPETokenizer()
    tok.train(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tok.chat_template = ChatTemplate.from_file(args.template).template_src if args.template else None
    out = tok.save_pretrained(args.out)
    logger.info("分词器已保存到 %s（词表 %d）", out, tok.vocab_size)
    return 0


def _synth_corpus(n_tokens: int) -> str:
    import random

    rng = random.Random(0)
    words = ["hello", "world", "the", "quick", "fox", "help", "morning", "today",
             "weather", "nice", "more", "about", "assistant", "tell", "me"]
    lines = []
    total = 0
    while total < n_tokens:
        line = " ".join(rng.choice(words) for _ in range(rng.randrange(4, 12)))
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def cmd_sft(args) -> int:
    from verse_nn import LoRAConfig, VerseTransformerConfig
    from verse_tokenizer import BPETokenizer, ChatTemplate
    from verse_trainer import SFTConfig, SFTDataset, Trainer, TrainerConfig
    from verse_trainer.data import DataConfig
    from verse_trainer.optim import OptimizerConfig

    tok = BPETokenizer.from_pretrained(args.tokenizer)
    tpl = ChatTemplate.from_pretrained(args.tokenizer)

    sft_cfg = SFTConfig(
        max_seq_len=args.seq_len, train_on_last_turns=args.train_on_last_turns,
        batch_size=args.batch_size, seed=args.seed,
    )
    train_ds = SFTDataset.from_file(args.data, tok, tpl, sft_cfg)
    eval_ds = SFTDataset.from_file(args.eval_data, tok, tpl, sft_cfg) if args.eval_data else None

    config = TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=tok.vocab_size, d_model=args.d_model,
            num_layers=args.num_layers, num_heads=args.num_heads,
            max_seq_len=args.seq_len,
        ),
        data=DataConfig(batch_size=args.batch_size, seq_len=args.seq_len),
        optim=OptimizerConfig(lr=args.lr, warmup_steps=min(50, args.steps // 5 or 1), max_steps=args.steps),
        steps=args.steps,
        log_every=max(args.steps // 10, 1),
        eval_every=max(args.steps // 2, 1),
        eval_batches=5,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=0,
        device=args.device,
        seed=args.seed,
        num_threads=args.num_threads,
        stage="sft",
        lora=LoRAConfig(r=args.lora_r, lora_alpha=args.lora_r * 2) if args.lora_r > 0 else None,
    )
    trainer = Trainer(config)
    trainer.train(train_ds, eval_ds)
    if args.lora_r > 0:
        path = trainer.save_adapter(args.save or "adapter.pt")
    else:
        path = trainer.finalize("sft", path=args.save or "sft_model.pt")
    logger.info("SFT 完成，模型已保存: %s", path)
    return 0


def cmd_generate(args) -> int:
    from verse_trainer import GenerateConfig, Generator

    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, greedy=args.greedy, quantize=args.quantize,
        num_threads=args.num_threads,
    )
    gen = Generator.from_pretrained(args.model, args.tokenizer, config)
    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        print(gen.chat(messages))
    else:
        print(gen.complete(args.prompt))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verse", description="VerseNext 神经网络框架 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vocab-size", type=int, default=1000)
        p.add_argument("--d-model", type=int, default=256)
        p.add_argument("--num-layers", type=int, default=4)
        p.add_argument("--num-heads", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--seq-len", type=int, default=128)
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--compile", action="store_true", help="启用 torch.compile")

    p_train = sub.add_parser("train", help="训练一个模型")
    _common(p_train)
    p_train.add_argument("--steps", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--tokens", type=int, default=200000, help="合成数据 token 总量")
    p_train.add_argument("--ckpt-dir", default="checkpoints")
    p_train.add_argument("--save", default=None, help="训练结束保存 checkpoint 路径")
    p_train.set_defaults(func=cmd_train)

    p_evolve = sub.add_parser("evolve", help="启动 RSI 自进化循环")
    _common(p_evolve)
    p_evolve.add_argument("--steps", type=int, default=60, help="进化期每个个体的训练步数")
    p_evolve.add_argument("--generations", type=int, default=3)
    p_evolve.add_argument("--population", type=int, default=3, help="每代子代数量")
    p_evolve.add_argument("--tokens", type=int, default=60000)
    p_evolve.add_argument("--speed-weight", type=float, default=0.05)
    p_evolve.add_argument("--archive", default="rsi_archive/population.json")
    p_evolve.add_argument("--refine-steps", type=int, default=0, help="进化后对最优个体重训步数")
    p_evolve.set_defaults(func=cmd_evolve)

    p_bench = sub.add_parser("bench", help="训练吞吐基准测速")
    _common(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    p_tok = sub.add_parser("train-tokenizer", help="训练 byte-level BPE 分词器")
    p_tok.add_argument("--input", nargs="*", default=None, help="语料文本文件（可多个）；缺省用合成语料演示")
    p_tok.add_argument("--vocab-size", type=int, default=4096)
    p_tok.add_argument("--min-frequency", type=int, default=2)
    p_tok.add_argument("--out", default="tokenizer/")
    p_tok.add_argument("--template", default=None, help="自定义 chat_template.jinja；缺省不写入文件（加载时回退内置模板）")
    p_tok.set_defaults(func=cmd_train_tokenizer)

    p_sft = sub.add_parser("sft", help="对话 SFT 微调（支持 LoRA）")
    _common(p_sft)
    p_sft.add_argument("--data", required=True, help="对话 JSON 文件")
    p_sft.add_argument("--eval-data", default=None)
    p_sft.add_argument("--tokenizer", required=True, help="tokenizer 目录（save_pretrained 产物）")
    p_sft.add_argument("--steps", type=int, default=200)
    p_sft.add_argument("--lr", type=float, default=1e-4)
    p_sft.add_argument("--train-on-last-turns", type=int, default=1)
    p_sft.add_argument("--lora-r", type=int, default=8, help="LoRA rank；0 = 全参微调")
    p_sft.add_argument("--num-threads", type=int, default=0, help="CPU 线程数；0 = 自动")
    p_sft.add_argument("--ckpt-dir", default="checkpoints")
    p_sft.add_argument("--save", default=None, help="保存路径（LoRA 存适配器，否则存完整 checkpoint）")
    p_sft.set_defaults(func=cmd_sft)

    p_gen = sub.add_parser("generate", help="加载模型生成文本/对话")
    p_gen.add_argument("--model", required=True, help="checkpoint 文件（.pt）")
    p_gen.add_argument("--tokenizer", required=True, help="tokenizer 目录")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--chat", action="store_true", help="使用 chat 模板渲染对话")
    p_gen.add_argument("--system", default=None, help="chat 模式下的 system 提示")
    p_gen.add_argument("--max-new-tokens", type=int, default=128)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-k", type=int, default=50)
    p_gen.add_argument("--greedy", action="store_true")
    p_gen.add_argument("--quantize", action="store_true", help="CPU int8 动态量化")
    p_gen.add_argument("--num-threads", type=int, default=0)
    p_gen.set_defaults(func=cmd_generate)

    return parser


def app(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(app())
