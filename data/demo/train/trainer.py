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
from verse_torch.training import Trainer, ParallelTrainer, CheckpointManager, _default_collate

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
    # no_decay 参数组：bias 与 norm 类参数不参与 weight decay（标准正则化做法，
    # 避免对偏置/归一化缩放做衰减，提升收敛与泛化）
    no_decay = bool(train_cfg.get("no_decay", True))
    if no_decay:
        decay_params, nodecay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        n_decay, n_nodecay = len(decay_params), len(nodecay_params)
        print(f"[train] param groups: decay={n_decay} no_decay={n_nodecay}",
              flush=True)
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = int(train_cfg.get("max_steps", 200))
    warmup = int(train_cfg.get("warmup", 20))
    scheduler = LambdaLR(
        optimizer, warmup_cosine_lr(warmup_steps=warmup, total_steps=max_steps)
    )

    # 7. Trainer
    # Part3K2 Task 1.5: 根据 parallel_chunks 选择 Trainer 或 ParallelTrainer
    # parallel_chunks=1（默认）→ 标准 Trainer；>1 → ParallelTrainer（chunk 拆分 + 串行重训）
    parallel_chunks = int(train_cfg.get("parallel_chunks", 1))
    patience = int(train_cfg.get("patience", 5))
    eval_interval = int(train_cfg.get("eval_interval", 20))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    log_interval = int(train_cfg.get("log_interval", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    enable_progress_bar = bool(train_cfg.get("enable_progress_bar", True))
    realtime_plot = bool(train_cfg.get("realtime_plot", True))
    eta_window = int(train_cfg.get("eta_window", 20))

    if parallel_chunks > 1:
        # ParallelTrainer 分支：内部自建 BatchLoader + optimizer，仅需 dataset + cfg
        # 注意：ParallelTrainer.fit() 返回 history dict（含 train_loss/val_loss/steps 列表）
        parallel_cfg = {
            "parallel_chunks": parallel_chunks,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "lr": lr,
            "warmup": warmup,
            "eval_interval": eval_interval,
            "grad_clip": grad_clip,
            "label_smoothing": label_smoothing,
            "seed": seed,
            "patience": patience,
            "save_dir": save_dir,
            "log_interval": log_interval,
            "loss_rate_window": min(50, max(10, max_steps // 4)),
            "enable_progress_bar": enable_progress_bar,
            "realtime_plot": realtime_plot,
            "eta_window": eta_window,
        }
        # optimizer_kwargs 仅接受 AdamW 标准参数（weight_decay 等）；
        # no_decay 参数组分离由 ParallelTrainer 内部简化处理（不分离）
        optimizer_kwargs = {"weight_decay": weight_decay}
        checkpoint_mgr = CheckpointManager(save_dir)
        parallel_trainer = ParallelTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            optimizer_cls=AdamW,
            optimizer_kwargs=optimizer_kwargs,
            cfg=parallel_cfg,
            collate_fn=collate_fn,
            checkpoint_mgr=checkpoint_mgr,
        )
        print(f"[train] 开始训练 (ParallelTrainer) chunks={parallel_chunks} "
              f"max_steps={max_steps} batch_size={batch_size} "
              f"lr={lr} warmup={warmup} grad_clip={grad_clip} "
              f"label_smoothing={label_smoothing}", flush=True)
        history = parallel_trainer.fit()
        # 统一返回结构：把 history 的 train_loss/val_loss 列表作为 train_losses/val_losses
        train_losses = list(history.get("train_loss", []))
        val_losses = list(history.get("val_loss", []))
        best_val_loss = float(parallel_trainer.best_val_loss)
        # ParallelTrainer 不直接走 Trainer._save_history，手动补存 loss_history.json
        _save_parallel_history(save_dir, train_losses, val_losses, max_steps, eval_interval,
                                best_val_loss)
    else:
        trainer_cfg = {
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "patience": patience,
            "save_dir": save_dir,
            "grad_accum": grad_accum,
            "log_interval": log_interval,
            "loss_rate_window": min(50, max(10, max_steps // 4)),
            "grad_clip": grad_clip,
            "label_smoothing": label_smoothing,
            "enable_progress_bar": enable_progress_bar,
            "realtime_plot": realtime_plot,
            "eta_window": eta_window,
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
              f"lr={lr} warmup={warmup} grad_clip={grad_clip} "
              f"label_smoothing={label_smoothing}", flush=True)
        train_losses, val_losses = trainer.fit()
        best_val_loss = float(trainer.best_val_loss)

    wall_clock = time.time() - start_time
    initial_loss = float(train_losses[0]) if train_losses else float("nan")
    final_loss = float(train_losses[-1]) if train_losses else float("nan")

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


def _save_parallel_history(save_dir: str, train_losses, val_losses,
                            max_steps: int, eval_interval: int,
                            best_val_loss: float) -> None:
    """ParallelTrainer 分支的 loss 历史持久化（对齐 Trainer._save_history）。

    输出：
    - ``loss_history.json``：JSON 格式的 loss 历史
    - ``train_losses.txt`` / ``val_losses.txt``：纯文本每行一个值
    - ``loss_curve.png``（或 .txt）：loss 曲线图（matplotlib 不可用时降级 ASCII）
    """
    import json
    from verse_torch.training import plot_loss_curve

    os.makedirs(save_dir, exist_ok=True)
    history = {
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "max_steps": max_steps,
        "eval_interval": eval_interval,
        "best_val_loss": float(best_val_loss),
    }
    with open(os.path.join(save_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "train_losses.txt"), "w", encoding="utf-8") as f:
        for v in train_losses:
            f.write(f"{float(v):.6f}\n")
    with open(os.path.join(save_dir, "val_losses.txt"), "w", encoding="utf-8") as f:
        for v in val_losses:
            f.write(f"{float(v):.6f}\n")

    # 画曲线图（matplotlib 不可用时 plot_loss_curve 内部已自处理 ImportError）
    actual_path = plot_loss_curve(
        train_losses, val_losses,
        os.path.join(save_dir, "loss_curve.png"),
        eval_interval=eval_interval,
    )
    print(f"[train] loss 曲线已保存到: {actual_path}", flush=True)


__all__ = ["train"]
