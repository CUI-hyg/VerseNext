"""训练入口：读取 config.yml → 构建数据 / 模型 / 优化器 → 调 Trainer.fit。

输出：
    checkpoints/best.pt
    checkpoints/last.pt
    checkpoints/loss_history.json
    checkpoints/loss_curve.png（或 .txt）
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from verse_torch.optim import AdamW, LambdaLR, warmup_cosine_lr
from verse_torch.training import Trainer

from model.config import CometSparkConfig, load_full_config
from model.model import CometSparkLM
from model.tokenizer import load_tokenizer
from src.data_loader import TextDataset, collate_fn, BatchLoader
from src.utils import set_seed, ensure_dir, num_threads


# 默认 tokenizer 文件名（相对于 checkpoints 同级目录）
_DEFAULT_TOKENIZER_FILE = "tokenizer.json"


def _resolve_path(base_dir: str, path_str: str) -> str:
    """把配置中的相对路径解析为相对 base_dir 的绝对路径。"""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


def train(
    config_path: str,
    base_dir: str = ".",
    n_threads: int = 0,
) -> dict:
    """主训练函数。

    Args:
        config_path: 配置文件路径（config.yml）
        base_dir: 配置中相对路径的基准目录（默认当前目录）
        n_threads: NumPy BLAS 线程数；0 表示不限制
    Returns:
        dict 包含：wall_clock / initial_loss / final_loss / best_val_loss /
        checkpoint_dir / loss_history_path
    """
    start_time = time.time()

    # 1. 读取配置
    full_cfg = load_full_config(config_path)
    model_cfg = full_cfg.get("model", {})
    train_cfg = full_cfg.get("training", {})
    tok_cfg = full_cfg.get("tokenizer", {})
    data_cfg = full_cfg.get("data", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})

    # 2. 环境准备
    if n_threads > 0:
        num_threads(n_threads)
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # 3. 加载 tokenizer
    tok_kind = str(tok_cfg.get("kind", "byte"))
    # tokenizer 文件路径：优先 ckpt_dir 下，其次 base_dir
    save_dir = _resolve_path(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))
    ensure_dir(save_dir)
    tok_path = os.path.join(save_dir, _DEFAULT_TOKENIZER_FILE)
    if not os.path.exists(tok_path):
        # 兼容 base_dir 下其他位置
        alt_tok_path = _resolve_path(base_dir, "tokenizer.json")
        if os.path.exists(alt_tok_path):
            tok_path = alt_tok_path
        else:
            raise FileNotFoundError(
                f"tokenizer 文件不存在：{tok_path}。请先调用 build_tokenizer。"
            )
    print(f"[train] 加载 tokenizer ({tok_kind}) from {tok_path}", flush=True)
    tok = load_tokenizer(tok_path, kind=tok_kind)
    vocab_size = len(tok)
    print(f"[train] vocab_size = {vocab_size}", flush=True)

    # 4. 构建数据集
    train_path = _resolve_path(base_dir, str(data_cfg.get("train_path", "data/train.jsonl")))
    val_path = _resolve_path(base_dir, str(data_cfg.get("val_path", "data/val.jsonl")))
    seq_len = int(model_cfg.get("seq_len", 128))

    print(f"[train] 加载训练数据 {train_path}", flush=True)
    train_ds = TextDataset(tok, train_path, seq_len=seq_len)
    print(f"[train] 加载验证数据 {val_path}", flush=True)
    val_ds = TextDataset(tok, val_path, seq_len=seq_len)
    print(f"[train] train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)

    batch_size = int(train_cfg.get("batch_size", 16))
    train_loader = BatchLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        seed=seed,
    )
    val_loader = BatchLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 5. 实例化模型（用实际 vocab_size 覆盖配置）
    config_dict = dict(model_cfg)
    config_dict["vocab_size"] = vocab_size
    config = CometSparkConfig.from_dict(config_dict)
    print(f"[train] 实例化模型 arch={config.arch} n_layer={config.n_layer} "
          f"n_embd={config.n_embd} seq_len={config.seq_len}", flush=True)
    model = CometSparkLM(config)

    # 6. 优化器 + 学习率调度
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = int(train_cfg.get("max_steps", 200))
    warmup = int(train_cfg.get("warmup", 20))
    scheduler = LambdaLR(
        optimizer, warmup_cosine_lr(warmup_steps=warmup, total_steps=max_steps)
    )

    # 7. Trainer
    patience = int(train_cfg.get("patience", 5))
    eval_interval = int(train_cfg.get("eval_interval", 20))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    log_interval = int(train_cfg.get("log_interval", 10))

    trainer_cfg = {
        "max_steps": max_steps,
        "eval_interval": eval_interval,
        "patience": patience,
        "save_dir": save_dir,
        "grad_accum": grad_accum,
        "log_interval": log_interval,
        "loss_rate_window": min(50, max(10, max_steps // 4)),
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=trainer_cfg,
    )
    print(f"[train] 开始训练 max_steps={max_steps} batch_size={batch_size} "
          f"lr={lr} warmup={warmup}", flush=True)
    train_losses, val_losses = trainer.fit()

    wall_clock = time.time() - start_time
    initial_loss = float(train_losses[0]) if train_losses else float("nan")
    final_loss = float(train_losses[-1]) if train_losses else float("nan")
    best_val_loss = float(trainer.best_val_loss)

    # 8. 保存完整模型（含 config）到 checkpoints/cometspark.pt
    full_model_path = os.path.join(save_dir, "cometspark.pt")
    try:
        model.save(full_model_path)
        print(f"[train] 完整模型已保存到 {full_model_path}", flush=True)
    except Exception as e:
        print(f"[train] 警告：保存完整模型失败：{e}", flush=True)

    print(
        f"[train] 训练完成 wall_clock={wall_clock:.2f}s "
        f"initial_loss={initial_loss:.4f} final_loss={final_loss:.4f} "
        f"best_val_loss={best_val_loss:.4f}",
        flush=True,
    )

    return {
        "wall_clock": wall_clock,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint_dir": save_dir,
        "loss_history_path": os.path.join(save_dir, "loss_history.json"),
        "full_model_path": full_model_path,
        "vocab_size": vocab_size,
        "config": config.to_dict(),
    }


__all__ = ["train"]
