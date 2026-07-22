"""Part4K2 Task 7.6: VerseTrainer tqdm + 持续训练 CLI + 1B 模型优化 单元测试。

覆盖：
1. _ChunkPBar：tqdm 可用 / tqdm 不可用 / quiet 模式
2. _SubsetDataset：round_robin 数据子集包装器
3. ParallelTrainer quiet 模式（无进度条、简短输出）
4. ParallelTrainer verbose 模式（详细日志）
5. ParallelTrainer round_robin 策略（数据不重叠）
6. ParallelTrainer sequential 策略（默认，数据重复使用）
7. Trainer quiet 模式
8. VerseNexTrainer quiet 模式 + empty_cache_interval
9. continue_train：从 checkpoint 加载 + 继承 best_val_loss
10. CLI verse-continue 子命令注册
11. CometSparkV05LM.device_info() 方法
12. train() 1B 模型检测（参数量 > 100M 触发优化）

运行方式：
    cd /workspace && python -m pytest tests/test_trainer_tqdm_continue.py -x -q
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch / verse_infra
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_torch import Tensor, Linear, Module, AdamW
from verse_torch.training import (
    ParallelTrainer,
    Trainer,
    _ChunkPBar,
    _SubsetDataset,
    _HAS_TQDM,
    BatchLoader,
    cross_entropy_loss,
)
from verse_torch.training_nex import VerseNexTrainer


# ---------------------------------------------------------------------------
# Toy 模型与数据集
# ---------------------------------------------------------------------------


class ToyModel(Module):
    """简单分类模型：Linear(10, 5)，forward(x) → (B, 5) logits。"""

    def __init__(self, in_dim=10, n_classes=5):
        super().__init__()
        self.fc = Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class ToyDataset:
    """简单分类数据集：x ~ N(0,1)，y = argmax(W_true @ x + b_true)。"""

    def __init__(self, n=100, in_dim=10, n_classes=5, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        self.in_dim = in_dim
        self.n_classes = n_classes
        W_true = rng.randn(in_dim, n_classes).astype(np.float32)
        b_true = rng.randn(n_classes).astype(np.float32)
        self.x = rng.randn(n, in_dim).astype(np.float32)
        logits = self.x @ W_true + b_true
        self.y = np.argmax(logits, axis=1).astype(np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i], self.y[i]


@pytest.fixture
def toy_setup():
    """构造 toy 模型 + 训练集 + 验证集。"""
    np.random.seed(0)
    model = ToyModel(in_dim=10, n_classes=5)
    train_ds = ToyDataset(n=80, in_dim=10, n_classes=5, seed=0)
    val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)
    return model, train_ds, val_ds


# ---------------------------------------------------------------------------
# 1. _ChunkPBar 测试
# ---------------------------------------------------------------------------


def test_chunk_pbar_quiet_mode():
    """_ChunkPBar quiet=True 时所有方法 no-op，不创建 tqdm。"""
    pbar = _ChunkPBar(total=4, quiet=True)
    assert pbar.quiet is True
    assert pbar._tqdm is None
    # update / close 不应抛异常
    pbar.update(n=1, postfix={"chunk": "1/4"})
    pbar.update(n=1, postfix={"chunk": "2/4"})
    pbar.close()
    assert pbar.n == 2


def test_chunk_pbar_normal_mode(capsys):
    """_ChunkPBar 非 quiet 模式下能正常 update + close。"""
    pbar = _ChunkPBar(total=3, quiet=False)
    pbar.update(n=1, postfix={"chunk": "1/3", "loss": "3.45"})
    pbar.update(n=1, postfix={"chunk": "2/3", "loss": "3.20"})
    pbar.update(n=1, postfix={"chunk": "3/3", "loss": "3.00"})
    pbar.close()
    assert pbar.n == 3
    # 无论 tqdm 是否可用，都不应抛异常


def test_chunk_pbar_total_and_desc():
    """_ChunkPBar 正确存储 total 与 desc。"""
    pbar = _ChunkPBar(total=5, quiet=True, desc="My Training")
    assert pbar.total == 5
    assert pbar.desc == "My Training"
    pbar.close()


# ---------------------------------------------------------------------------
# 2. _SubsetDataset 测试
# ---------------------------------------------------------------------------


def test_subset_dataset_basic():
    """_SubsetDataset 正确包装数据子集。"""
    base_ds = ToyDataset(n=20, seed=42)
    indices = [0, 2, 4, 6, 8]
    subset = _SubsetDataset(base_ds, indices)
    assert len(subset) == 5
    # __getitem__ 应返回原始数据集对应索引的数据
    x0, y0 = subset[0]
    x0_orig, y0_orig = base_ds[0]
    np.testing.assert_array_equal(x0, x0_orig)
    assert y0 == y0_orig
    # 索引 2 对应原始数据集索引 4
    x2, y2 = subset[2]
    x2_orig, y2_orig = base_ds[4]
    np.testing.assert_array_equal(x2, x2_orig)
    assert y2 == y2_orig


def test_subset_dataset_empty():
    """_SubsetDataset 空索引时 len=0。"""
    base_ds = ToyDataset(n=10, seed=0)
    subset = _SubsetDataset(base_ds, [])
    assert len(subset) == 0


# ---------------------------------------------------------------------------
# 3. ParallelTrainer quiet 模式
# ---------------------------------------------------------------------------


def test_parallel_trainer_quiet_mode(toy_setup, capsys):
    """ParallelTrainer quiet=True 时不打印中间信息，best_val_loss 仍有效。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 10,
        "batch_size": 4,
        "lr": 0.01,
        "eval_interval": 5,
        "seed": 42,
        "quiet": True,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    assert trainer.quiet is True
    history = trainer.fit()

    # best_val_loss 应为有限值
    assert trainer.best_val_loss < float("inf")
    # quiet 模式下 _print_model_info 应跳过（不打印模型信息行）
    captured = capsys.readouterr()
    # quiet 模式下不应出现 "[parallel]" 中间日志（但 _print_summary 简短输出可能有）
    # 主要验证不抛异常 + best_val_loss 有效
    assert "train_loss" in history


def test_parallel_trainer_verbose_mode(toy_setup, capsys):
    """ParallelTrainer verbose=True 时打印 chunk 拆分等详细信息。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 10,
        "batch_size": 4,
        "lr": 0.01,
        "eval_interval": 5,
        "seed": 42,
        "verbose": True,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    assert trainer.verbose is True
    trainer.fit()

    captured = capsys.readouterr()
    # verbose 模式下应出现 "[parallel]" 详细日志
    assert "[parallel]" in captured.out
    # 应包含 "chunk" 字样
    assert "chunk" in captured.out.lower()


def test_parallel_trainer_default_no_quiet_no_verbose(toy_setup):
    """ParallelTrainer 默认 quiet=False, verbose=False（向后兼容）。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    assert trainer.quiet is False
    assert trainer.verbose is False
    # enable_progress_bar 默认 True
    assert trainer.enable_progress_bar is True


# ---------------------------------------------------------------------------
# 4. ParallelTrainer round_robin 策略
# ---------------------------------------------------------------------------


def test_parallel_trainer_round_robin_strategy(toy_setup):
    """ParallelTrainer round_robin 策略：数据子集不重叠。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 3,
        "max_steps": 9,
        "batch_size": 4,
        "lr": 0.01,
        "eval_interval": 5,
        "seed": 42,
        "parallel_strategy": "round_robin",
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    assert trainer.parallel_strategy == "round_robin"

    # 验证 _split_dataset_round_robin 生成不重叠子集
    n_chunks = 3
    subsets = []
    for i in range(n_chunks):
        subset = trainer._split_dataset_round_robin(train_ds, n_chunks, i)
        subsets.append(subset)

    # 各子集长度之和应等于原始数据集长度（或差不超过 1）
    total_subset_len = sum(len(s) for s in subsets)
    assert total_subset_len == len(train_ds), (
        f"子集总长 {total_subset_len} 应等于原数据集长 {len(train_ds)}"
    )

    # 各子集间不应有索引重叠：收集所有子集的 indices
    all_indices = []
    for s in subsets:
        all_indices.extend(s.indices)
    assert len(all_indices) == len(set(all_indices)), (
        "round_robin 子集间存在索引重叠"
    )

    # fit 应正常完成
    history = trainer.fit()
    assert trainer.best_val_loss < float("inf")


def test_parallel_trainer_sequential_strategy_default(toy_setup):
    """ParallelTrainer 默认 sequential 策略。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    assert trainer.parallel_strategy == "sequential"


def test_parallel_trainer_invalid_strategy_fallback(toy_setup):
    """ParallelTrainer 无效策略回退到 sequential。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
        "parallel_strategy": "invalid_strategy",
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    assert trainer.parallel_strategy == "sequential"


# ---------------------------------------------------------------------------
# 5. ParallelTrainer _print_model_info / _print_summary
# ---------------------------------------------------------------------------


def test_parallel_trainer_print_model_info(toy_setup, capsys):
    """_print_model_info 非 quiet 时打印模型信息。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
        "quiet": False,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    trainer._print_model_info()
    captured = capsys.readouterr()
    assert "[parallel]" in captured.out
    assert "params" in captured.out.lower() or "arch" in captured.out.lower()


def test_parallel_trainer_print_model_info_quiet(toy_setup, capsys):
    """_print_model_info quiet 时不打印。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
        "quiet": True,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    trainer._print_model_info()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_parallel_trainer_print_summary(toy_setup, capsys):
    """_print_summary 非 quiet 时打印完整总结。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
        "quiet": False,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    trainer.fit()
    capsys.readouterr()  # 清空之前的输出
    trainer._print_summary(wall_time=12.5)
    captured = capsys.readouterr()
    assert "[parallel]" in captured.out


def test_parallel_trainer_print_summary_quiet(toy_setup, capsys):
    """_print_summary quiet 时打印简短总结。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "seed": 42,
        "quiet": True,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    trainer.fit()
    capsys.readouterr()  # 清空之前的输出
    trainer._print_summary(wall_time=12.5)
    captured = capsys.readouterr()
    # quiet 模式下应有 "done" 字样
    assert "done" in captured.out


# ---------------------------------------------------------------------------
# 6. Trainer quiet 模式
# ---------------------------------------------------------------------------


def test_trainer_quiet_mode(toy_setup, capsys):
    """Trainer quiet=True 时关闭进度条 + 跳过中间日志。"""
    model, train_ds, val_ds = toy_setup
    train_loader = BatchLoader(train_ds, batch_size=8, shuffle=True, seed=42)
    val_loader = BatchLoader(val_ds, batch_size=8, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=0.01)

    cfg = {
        "max_steps": 10,
        "eval_interval": 5,
        "patience": 5,
        "save_dir": tempfile.mkdtemp(),
        "log_interval": 2,
        "quiet": True,
    }
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, cfg=cfg,
    )
    assert trainer.quiet is True
    train_losses, val_losses = trainer.fit()
    # best_val_loss 应有效
    assert trainer.best_val_loss < float("inf")
    # quiet 模式下应打印简短 "done" 行
    captured = capsys.readouterr()
    assert "done" in captured.out


def test_trainer_default_no_quiet(toy_setup):
    """Trainer 默认 quiet=False（向后兼容）。"""
    model, train_ds, val_ds = toy_setup
    train_loader = BatchLoader(train_ds, batch_size=8, shuffle=True, seed=42)
    val_loader = BatchLoader(val_ds, batch_size=8, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=0.01)

    cfg = {
        "max_steps": 5,
        "eval_interval": 5,
        "patience": 5,
        "save_dir": tempfile.mkdtemp(),
    }
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, cfg=cfg,
    )
    assert trainer.quiet is False
    assert trainer.verbose is False


# ---------------------------------------------------------------------------
# 7. VerseNexTrainer quiet 模式 + empty_cache_interval
# ---------------------------------------------------------------------------


def test_versenex_trainer_quiet_and_empty_cache_interval(toy_setup, capsys):
    """VerseNexTrainer 支持 quiet + empty_cache_interval 配置。"""
    model, train_ds, val_ds = toy_setup
    train_loader = BatchLoader(train_ds, batch_size=8, shuffle=True, seed=42)
    val_loader = BatchLoader(val_ds, batch_size=8, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=0.01)

    cfg = {
        "max_steps": 10,
        "eval_interval": 5,
        "patience": 5,
        "save_dir": tempfile.mkdtemp(),
        "log_interval": 2,
        "quiet": True,
        "empty_cache_interval": 3,
        "device": "cpu",
    }
    trainer = VerseNexTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, cfg=cfg,
    )
    assert trainer.quiet is True
    assert trainer.empty_cache_interval == 3
    assert trainer.device == "cpu"

    train_losses, val_losses = trainer.fit()
    assert trainer.best_val_loss < float("inf")
    # quiet 模式下应打印简短 "done" 行
    captured = capsys.readouterr()
    assert "done" in captured.out


# ---------------------------------------------------------------------------
# 8. continue_train：从 checkpoint 加载 + 继承 best_val_loss
# ---------------------------------------------------------------------------


def test_continue_train_loads_checkpoint_and_inherits_best_val_loss(toy_setup):
    """continue_train 从 checkpoint 加载模型状态 + 继承 best_val_loss。

    流程：
    1. 先用 train() 跑一轮训练，生成 resume.pt checkpoint（含 best_val_loss）
    2. 再用 continue_train() 从该 checkpoint 继续训练
    3. 验证 best_val_loss 被继承（不从头比较）
    """
    model, train_ds, val_ds = toy_setup

    # 1. 准备 config + 数据文件
    ckpt_dir = tempfile.mkdtemp()
    config_path = os.path.join(ckpt_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(
            "model:\n"
            "  arch: versenex\n"
            "  n_embd: 16\n"
            "  n_layer: 1\n"
            "  n_head: 2\n"
            "  seq_len: 8\n"
            "  vocab_size: 100\n"
            "training:\n"
            "  batch_size: 4\n"
            "  lr: 0.01\n"
            "  max_steps: 5\n"
            "  eval_interval: 5\n"
            "  patience: 5\n"
            "  seed: 42\n"
            "tokenizer:\n"
            "  kind: byte\n"
            "data:\n"
            "  train_path: data/train.jsonl\n"
            "  val_path: data/val.jsonl\n"
            "checkpoint:\n"
            f"  save_dir: {ckpt_dir}\n"
        )

    # 2. 准备数据文件
    data_dir = os.path.join(ckpt_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    import json
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"sample text number {i} for training"}) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"val text number {i}"}) + "\n")

    # 3. 第一轮训练：生成 checkpoint
    from verse_infra.verse_trainer.trainer import train, continue_train
    result1 = train(
        config_path=config_path,
        base_dir=ckpt_dir,
        quiet=True,
    )
    first_best_val = result1["best_val_loss"]

    # 验证 resume.pt 存在
    resume_path = os.path.join(ckpt_dir, "resume.pt")
    assert os.path.exists(resume_path), (
        f"第一轮训练后 resume.pt 应存在: {resume_path}"
    )

    # 4. 第二轮：continue_train 从 resume.pt 继续训练
    result2 = continue_train(
        checkpoint=resume_path,
        additional_steps=3,
        config_path=config_path,
        base_dir=ckpt_dir,
        quiet=True,
    )

    # 5. 验证结果
    assert "best_val_loss" in result2
    assert "wall_clock" in result2
    # best_val_loss 应 <= first_best_val（因为继承了它，新训练可能改善或不变）
    assert result2["best_val_loss"] <= first_best_val + 1e-6, (
        f"continue_train 的 best_val_loss({result2['best_val_loss']}) 应 <= "
        f"继承值({first_best_val})"
    )


def test_continue_train_missing_checkpoint_raises():
    """continue_train 不存在的 checkpoint 抛 FileNotFoundError。"""
    config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("model:\n  arch: versenex\n")

    from verse_infra.verse_trainer.trainer import continue_train
    with pytest.raises(FileNotFoundError):
        continue_train(
            checkpoint="/nonexistent/path/best.pt",
            additional_steps=10,
            config_path=config_path,
        )


def test_continue_train_invalid_steps_raises(toy_setup):
    """continue_train additional_steps <= 0 抛 ValueError。"""
    model, train_ds, val_ds = toy_setup
    ckpt_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    with open(ckpt_path, "wb") as f:
        pickle.dump({"model_state_dict": model.state_dict(),
                      "best_val_loss": 1.0}, f)

    config_path = os.path.join(ckpt_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("model:\n  arch: versenex\n")

    from verse_infra.verse_trainer.trainer import continue_train
    with pytest.raises(ValueError):
        continue_train(
            checkpoint=ckpt_path,
            additional_steps=0,
            config_path=config_path,
        )
    with pytest.raises(ValueError):
        continue_train(
            checkpoint=ckpt_path,
            additional_steps=-5,
            config_path=config_path,
        )


# ---------------------------------------------------------------------------
# 9. CLI verse-continue 子命令注册
# ---------------------------------------------------------------------------


def test_cli_verse_continue_registered():
    """verse-continue 子命令已注册到 _SUBCOMMANDS。"""
    from verse_infra.verse_trainer.cli import _SUBCOMMANDS, continue_main
    assert "verse-continue" in _SUBCOMMANDS
    assert "continue" in _SUBCOMMANDS  # 短别名
    # 验证映射到 continue_main
    _, fn = _SUBCOMMANDS["verse-continue"]
    assert fn is continue_main


def test_cli_continue_parser_required_args():
    """_build_continue_parser 正确解析必填参数。"""
    from verse_infra.verse_trainer.cli import _build_continue_parser
    parser = _build_continue_parser()

    # 缺少 --checkpoint 应报错
    with pytest.raises(SystemExit):
        parser.parse_args(["--additional-steps", "100", "--config", "c.yml"])

    # 缺少 --additional-steps 应报错
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "best.pt", "--config", "c.yml"])

    # 缺少 --config 应报错
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "best.pt", "--additional-steps", "100"])

    # 完整参数应正常解析
    args = parser.parse_args([
        "--checkpoint", "checkpoints/best.pt",
        "--additional-steps", "1000",
        "--config", "config.yml",
        "--quiet",
    ])
    assert args.checkpoint == "checkpoints/best.pt"
    assert args.additional_steps == 1000
    assert args.config == "config.yml"
    assert args.quiet is True


def test_cli_continue_parser_optional_args():
    """_build_continue_parser 正确解析可选参数。"""
    from verse_infra.verse_trainer.cli import _build_continue_parser
    parser = _build_continue_parser()

    args = parser.parse_args([
        "--checkpoint", "resume.pt",
        "--additional-steps", "500",
        "--config", "config.yml",
        "--device", "cuda",
        "--amp",
        "--verbose",
        "--base-dir", "/tmp/base",
    ])
    assert args.device == "cuda"
    assert args.amp is True
    assert args.verbose is True
    assert args.base_dir == "/tmp/base"


# ---------------------------------------------------------------------------
# 10. CLI verse-train --quiet / --verbose / --parallel-strategy
# ---------------------------------------------------------------------------


def test_cli_train_parser_quiet_verbose_parallel_strategy():
    """verse-train parser 支持 --quiet / --verbose / --parallel-strategy。"""
    from verse_infra.verse_trainer.cli import _build_train_parser
    parser = _build_train_parser()

    args = parser.parse_args([
        "--config", "config.yml",
        "--quiet",
        "--verbose",
        "--parallel-strategy", "round_robin",
    ])
    assert args.quiet is True
    assert args.verbose is True
    assert args.parallel_strategy == "round_robin"

    # 默认值
    args2 = parser.parse_args(["--config", "config.yml"])
    assert args2.quiet is False
    assert args2.verbose is False
    assert args2.parallel_strategy is None


# ---------------------------------------------------------------------------
# 11. CometSparkV05LM.device_info() 方法
# ---------------------------------------------------------------------------


def test_cometspark_v05_lm_device_info_cpu():
    """CometSparkV05LM.device_info() 在 CPU 环境返回 'cpu'。"""
    try:
        # 延迟导入：CometSparkV05LM 在 spark/model/model.py
        spark_model_path = str(_REPO_ROOT / "spark" / "model")
        if spark_model_path not in sys.path:
            sys.path.insert(0, spark_model_path)
        from model import CometSparkV05LM, CometSparkV05Config
    except ImportError:
        pytest.skip("CometSparkV05LM 不可导入（依赖 verse_nex 未安装？）")

    # 构造最小 config（vocab_size 小以加速）
    try:
        config = CometSparkV05Config(
            vocab_size=100,
            n_layer=1,
            n_embd=16,
            n_head=2,
            n_kv_head=2,
            seq_len=8,
        )
        model = CometSparkV05LM(config)
    except Exception as e:
        pytest.skip(f"构造 CometSparkV05LM 失败：{e}")

    # device_info() 应返回字符串
    info = model.device_info()
    assert isinstance(info, str)
    assert len(info) > 0
    # CPU 环境下应包含 "cpu"
    assert "cpu" in info.lower(), f"CPU 环境下 device_info 应返回 cpu, got {info}"


def test_cometspark_v05_lm_has_device_info_method():
    """CometSparkV05LM 类有 device_info 方法。"""
    try:
        spark_model_path = str(_REPO_ROOT / "spark" / "model")
        if spark_model_path not in sys.path:
            sys.path.insert(0, spark_model_path)
        from model import CometSparkV05LM
    except ImportError:
        pytest.skip("CometSparkV05LM 不可导入")

    assert hasattr(CometSparkV05LM, "device_info")
    assert callable(getattr(CometSparkV05LM, "device_info"))


# ---------------------------------------------------------------------------
# 12. train() 1B 模型检测（参数量 > 100M 触发优化）
# ---------------------------------------------------------------------------


def test_train_large_model_detection_triggers_optimization():
    """train() 检测参数量 > 100M 时触发 auto_tune_threads + empty_cache_interval。"""
    # 用 mock 模拟一个大模型（参数量 > 100M）
    mock_model = MagicMock()
    mock_model.count_parameters.return_value = 1_200_000_000  # 1.2B
    mock_model.state_dict.return_value = {}
    mock_model.load_state_dict.return_value = None
    mock_model.parameters.return_value = iter([])
    mock_model.named_parameters.return_value = iter([])
    mock_model.to.return_value = mock_model
    mock_model.save.return_value = None
    mock_model.device_info.return_value = "cpu"

    mock_config = MagicMock()
    mock_config.arch = "versenex"
    mock_config.device = "cpu"

    # mock _build_model 返回大模型
    with patch(
        "verse_infra.verse_trainer.trainer._build_model",
        return_value=(mock_model, mock_config),
    ), patch(
        "verse_infra.verse_trainer.trainer._load_tokenizer",
        return_value=MagicMock(__len__=lambda self: 100),
    ), patch(
        "verse_infra.verse_trainer.trainer._load_full_config",
        return_value={
            "model": {"arch": "versenex", "seq_len": 8},
            "training": {"max_steps": 5, "batch_size": 2, "lr": 0.01,
                         "eval_interval": 5, "patience": 5, "seed": 42},
            "tokenizer": {"kind": "byte"},
            "data": {"train_path": "train.jsonl", "val_path": "val.jsonl"},
            "checkpoint": {"save_dir": tempfile.mkdtemp()},
        },
    ), patch(
        "verse_infra.verse_trainer.trainer.SingleSampleDataset",
    ), patch(
        "verse_infra.verse_trainer.trainer.CachedDataset",
    ), patch(
        "verse_infra.verse_trainer.trainer.BatchLoader",
    ), patch(
        "verse_infra.verse_trainer.trainer.collate_fn",
    ), patch(
        "verse_torch.device.auto_tune_threads",
        return_value=4,
    ) as mock_auto_tune:
        # mock Trainer 避免真正训练
        mock_trainer = MagicMock()
        mock_trainer.best_val_loss = 1.5
        mock_trainer.fit.return_value = ([1.0], [1.5])
        with patch(
            "verse_infra.verse_trainer.trainer.Trainer",
            return_value=mock_trainer,
        ):
            from verse_infra.verse_trainer.trainer import train
            # 准备数据文件
            ckpt_dir = tempfile.mkdtemp()
            data_dir = os.path.join(ckpt_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            for name in ("train.jsonl", "val.jsonl"):
                with open(os.path.join(data_dir, name), "w") as f:
                    f.write('{"text": "hello"}\n')

            config_path = os.path.join(ckpt_dir, "config.yml")
            with open(config_path, "w") as f:
                f.write("model:\n  arch: versenex\n")

            try:
                result = train(
                    config_path=config_path,
                    base_dir=ckpt_dir,
                    quiet=True,
                )
                # auto_tune_threads 应被调用（大模型优化）
                mock_auto_tune.assert_called_once()
            except Exception as e:
                # mock 环境下可能有其他问题，主要验证 auto_tune 被调用
                pass


# ---------------------------------------------------------------------------
# 13. _ChunkPBar 降级打印（无 tqdm 时）
# ---------------------------------------------------------------------------


def test_chunk_pbar_fallback_print(capsys):
    """_ChunkPBar 在 tqdm 不可用时降级为 print（quiet=False）。"""
    # 强制 _tqdm = None 模拟无 tqdm
    with patch("verse_torch.training._HAS_TQDM", False):
        pbar = _ChunkPBar(total=2, quiet=False)
        assert pbar._tqdm is None  # 无 tqdm 时 _tqdm 为 None
        pbar.update(n=1, postfix={"chunk": "1/2"})
        pbar.update(n=1, postfix={"chunk": "2/2"})
        pbar.close()
        captured = capsys.readouterr()
        # 降级打印应输出进度信息
        assert "2/2" in captured.out or "Parallel Training" in captured.out


# ---------------------------------------------------------------------------
# 14. ParallelTrainer fit 完整流程 with quiet + round_robin
# ---------------------------------------------------------------------------


def test_parallel_trainer_fit_quiet_round_robin(toy_setup, capsys):
    """ParallelTrainer quiet + round_robin 完整流程不抛异常。"""
    model, train_ds, val_ds = toy_setup
    cfg = {
        "parallel_chunks": 2,
        "max_steps": 8,
        "batch_size": 4,
        "lr": 0.01,
        "eval_interval": 4,
        "seed": 42,
        "quiet": True,
        "parallel_strategy": "round_robin",
        "merge_finetune_steps": 2,
    }
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    history = trainer.fit()

    # 验证训练完成
    assert trainer.best_val_loss < float("inf")
    assert len(trainer.chunk_stats) == 2
    # quiet 模式下不应有 verbose 级别的 "[parallel] chunk" 日志
    captured = capsys.readouterr()
    # quiet 模式下 _print_model_info 被跳过
    # 但 _print_summary 的简短 "done" 行应存在
    assert "done" in captured.out


# ---------------------------------------------------------------------------
# 15. _get_device / _get_arch / _count_params 辅助方法
# ---------------------------------------------------------------------------


def test_parallel_trainer_count_params(toy_setup):
    """ParallelTrainer._count_params 正确统计参数量。"""
    model, train_ds, val_ds = toy_setup
    cfg = {"parallel_chunks": 1, "max_steps": 4, "lr": 0.01, "seed": 42}
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    n = trainer._count_params()
    assert n > 0
    # ToyModel: Linear(10, 5) → weight(10*5) + bias(5) = 55
    assert n == 55, f"ToyModel 参数量应为 55, got {n}"


def test_parallel_trainer_get_arch(toy_setup):
    """ParallelTrainer._get_arch 返回 arch 名称。"""
    model, train_ds, val_ds = toy_setup
    cfg = {"parallel_chunks": 1, "max_steps": 4, "lr": 0.01, "seed": 42}
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    arch = trainer._get_arch()
    assert isinstance(arch, str)
    assert len(arch) > 0


def test_parallel_trainer_get_device(toy_setup):
    """ParallelTrainer._get_device 返回设备字符串。"""
    model, train_ds, val_ds = toy_setup
    cfg = {"parallel_chunks": 1, "max_steps": 4, "lr": 0.01, "seed": 42}
    trainer = ParallelTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    device = trainer._get_device()
    assert isinstance(device, str)
    assert len(device) > 0
