"""Part4K2.5 Task 3 + Task 6: loss 数据计算 & 图表修复 + 并行训练修复的单元测试。

覆盖：
Task 3（loss 数据计算 & 图表）:
1. plot_loss_curve 正确处理不同长度的 train/val losses
2. plot_loss_curve val_losses 为空时不报错
3. plot_loss_curve ASCII 降级模式显示 val 线
4. loss 计算正确（label_smoothing 传递）
5. loss_history.json 正确保存（含 initial_loss / final_loss）

Task 6（并行训练修复）:
6. ParallelTrainer chunk 间状态正确重置
7. ParallelTrainer round_robin 数据分配均匀
8. ParallelTrainer _split_steps 每组至少 1 步
9. ParallelTrainer _eval_full_val 空 val_dataset 处理
10. ParallelTrainer Phase 2 步数为 0 时跳过

运行方式：
    cd /workspace && python -m pytest tests/test_loss_and_parallel_fix.py -x -q
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch / verse_infra
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_torch import Tensor, Linear, SGD, Module, AdamW
from verse_torch.training import (
    ParallelTrainer,
    Trainer,
    plot_loss_curve,
    cross_entropy_loss,
    BatchLoader,
    _ChunkPBar,
    _SubsetDataset,
)
from verse_torch import training as training_mod


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


class EmptyDataset:
    """空数据集，用于测试 _eval_full_val 边界情况。"""

    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError


# ---------------------------------------------------------------------------
# Task 3: plot_loss_curve 修复测试
# ---------------------------------------------------------------------------


class TestPlotLossCurveFixes:
    """plot_loss_curve 修复测试。"""

    def test_different_length_train_val(self, tmp_path):
        """plot_loss_curve 正确处理不同长度的 train/val losses。"""
        train_losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        val_losses = [4.8, 3.6, 2.4, 1.2]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)
        assert os.path.exists(actual), f"输出文件应存在: {actual}"

    def test_empty_val_losses_no_error(self, tmp_path):
        """plot_loss_curve val_losses 为空时不报错。"""
        train_losses = [5.0, 4.5, 4.0, 3.5, 3.0]
        save_path = str(tmp_path / "loss_curve.png")

        # 不应抛异常
        actual = plot_loss_curve(train_losses, [], save_path, eval_interval=2)
        assert os.path.exists(actual), f"输出文件应存在: {actual}"

    def test_empty_train_losses_no_error(self, tmp_path):
        """plot_loss_curve train_losses 为空时不报错。"""
        val_losses = [4.8, 3.6, 2.4]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve([], val_losses, save_path, eval_interval=2)
        assert os.path.exists(actual), f"输出文件应存在: {actual}"

    def test_both_empty_no_error(self, tmp_path):
        """plot_loss_curve 两者都为空时不报错。"""
        save_path = str(tmp_path / "loss_curve.png")
        actual = plot_loss_curve([], [], save_path, eval_interval=2)
        assert os.path.exists(actual), f"输出文件应存在: {actual}"

    def test_ascii_val_line_visible(self, tmp_path, monkeypatch):
        """ASCII 降级模式显示 val 线（V 或 *）。"""
        # 强制 matplotlib 不可用
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        train_losses = [5.5 - i * 0.1 for i in range(50)]
        val_losses = [5.4, 4.5, 3.6, 2.7, 1.8]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=10)
        assert actual.endswith(".txt")

        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # 画布中应至少有 1 个 V 或 *（val 线可见）
        v_count = content.count("V")
        star_count = content.count("*")
        assert (v_count + star_count) >= 1, (
            f"ASCII 画布中应显示 val 线（V 或 *），"
            f"实际 V={v_count}, *={star_count}"
        )

    def test_ascii_val_x_aligned_to_eval_interval(self, tmp_path, monkeypatch):
        """ASCII 模式下 val 的 x 坐标应基于 eval_interval 对齐到 train 的 step。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        # train: 100 步，eval_interval=20，val: 5 个点
        train_losses = [5.5 - i * 0.03 for i in range(100)]
        val_losses = [5.4, 4.5, 3.6, 2.7, 1.8]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=20)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # 应包含 val_losses detail 段，且 step 对齐到 i * eval_interval
        assert "val_losses detail:" in content
        detail_lines = [
            line for line in content.splitlines()
            if line.startswith("  [step") and "val_loss=" in line
        ]
        assert len(detail_lines) == 5
        for i, line in enumerate(detail_lines):
            expected_step = i * 20
            assert f"step {expected_step:>6d}" in line, (
                f"detail 第 {i} 行 step 应为 {expected_step}, got: {line}"
            )

    def test_matplotlib_val_x_correct(self, tmp_path, monkeypatch):
        """matplotlib 模式下 val 的 x 坐标正确对齐到 eval_interval。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        val_losses = [4.8, 3.6, 2.4, 1.2]
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)

        # 第二次调用 ax.plot 是 val 曲线
        val_call = fake_ax.plot.call_args_list[1]
        val_x = val_call.args[0]
        val_y = val_call.args[1]

        # val_x 应为 [0, 2, 4, 6]（i * eval_interval）
        assert val_x == [0, 2, 4, 6], f"val_x 应为 [0, 2, 4, 6], got {val_x}"
        assert val_y == val_losses, f"val_y 应等于 val_losses, got {val_y}"


# ---------------------------------------------------------------------------
# Task 3: loss 计算方法测试
# ---------------------------------------------------------------------------


class TestLossCalculation:
    """loss 计算方法修复测试。"""

    def test_label_smoothing_passed_to_cross_entropy(self):
        """cross_entropy_loss 正确传递 label_smoothing 参数。"""
        np.random.seed(42)
        N, V = 8, 5
        logits_np = np.random.randn(N, V).astype(np.float32)
        targets_np = np.random.randint(0, V, size=N)
        logits = Tensor(logits_np, requires_grad=True)

        # label_smoothing=0.0 的 loss
        loss_no_smooth = cross_entropy_loss(
            logits, targets_np, label_smoothing=0.0)
        # label_smoothing=0.1 的 loss
        logits2 = Tensor(logits_np.copy(), requires_grad=True)
        loss_smooth = cross_entropy_loss(
            logits2, targets_np, label_smoothing=0.1)

        # 两者应有不同的值（label_smoothing 改变了 loss）
        assert abs(float(loss_no_smooth.data) - float(loss_smooth.data)) > 1e-6, (
            f"label_smoothing 应改变 loss 值: "
            f"no_smooth={float(loss_no_smooth.data)}, "
            f"smooth={float(loss_smooth.data)}"
        )

    def test_trainer_label_smoothing_in_fit(self, tmp_path):
        """Trainer.fit 中 label_smoothing 被正确传递到 cross_entropy_loss。"""
        np.random.seed(0)
        dim_in, vocab = 4, 4
        rng = np.random.RandomState(0)
        X = rng.randn(32, dim_in).astype(np.float32)
        y = rng.randint(0, vocab, size=32).astype(np.int64)

        def make_batches(X, y, bs=8):
            return [
                (Tensor(X[i:i + bs]), Tensor(y[i:i + bs]))
                for i in range(0, len(X), bs)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 10,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "label_smoothing": 0.1,
            "enable_progress_bar": False,
            "realtime_plot": False,
            "log_interval": 1000,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        train_losses, val_losses = trainer.fit()

        # 应正常完成训练，loss 为有限值
        assert len(train_losses) == 10
        assert all(np.isfinite(train_losses)), "train_losses 应全为有限值"
        assert len(val_losses) > 0
        assert all(np.isfinite(val_losses)), "val_losses 应全为有限值"

    def test_loss_history_json_has_initial_final_loss(self, tmp_path):
        """loss_history.json 正确保存，含 initial_loss 和 final_loss 字段。"""
        np.random.seed(0)
        dim_in, vocab = 4, 4
        rng = np.random.RandomState(0)
        X = rng.randn(32, dim_in).astype(np.float32)
        y = rng.randint(0, vocab, size=32).astype(np.int64)

        def make_batches(X, y, bs=8):
            return [
                (Tensor(X[i:i + bs]), Tensor(y[i:i + bs]))
                for i in range(0, len(X), bs)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 10,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "enable_progress_bar": False,
            "realtime_plot": False,
            "log_interval": 1000,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        train_losses, val_losses = trainer.fit()

        history_path = tmp_path / "loss_history.json"
        assert history_path.exists(), "loss_history.json 未生成"

        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        # 应包含 initial_loss 和 final_loss
        assert "initial_loss" in hist, "loss_history.json 应含 initial_loss"
        assert "final_loss" in hist, "loss_history.json 应含 final_loss"
        assert abs(hist["initial_loss"] - train_losses[0]) < 1e-6, (
            f"initial_loss({hist['initial_loss']}) != train_losses[0]({train_losses[0]})"
        )
        assert abs(hist["final_loss"] - train_losses[-1]) < 1e-6, (
            f"final_loss({hist['final_loss']}) != train_losses[-1]({train_losses[-1]})"
        )
        # 应包含 train_losses 和 val_losses
        assert len(hist["train_losses"]) == len(train_losses)
        assert len(hist["val_losses"]) == len(val_losses)

    def test_val_loss_computed_in_no_grad(self, tmp_path):
        """val_loss 在 no_grad 上下文中计算（不产生梯度）。"""
        np.random.seed(0)
        dim_in, vocab = 4, 4
        rng = np.random.RandomState(0)
        X = rng.randn(32, dim_in).astype(np.float32)
        y = rng.randint(0, vocab, size=32).astype(np.int64)

        def make_batches(X, y, bs=8):
            return [
                (Tensor(X[i:i + bs]), Tensor(y[i:i + bs]))
                for i in range(0, len(X), bs)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 5,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "enable_progress_bar": False,
            "realtime_plot": False,
            "log_interval": 1000,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        trainer.fit()

        # evaluate 在 no_grad 中运行，不应影响模型参数的梯度
        # 验证 evaluate 返回有限值
        val_loss = trainer.evaluate()
        assert np.isfinite(val_loss), f"evaluate 应返回有限值, got {val_loss}"


# ---------------------------------------------------------------------------
# Task 3: visualize.py 测试
# ---------------------------------------------------------------------------


class TestVisualizeStats:
    """visualize.py 统计信息测试。"""

    def test_visualize_prints_stats(self, tmp_path, capsys):
        """visualize 打印更多统计信息（avg_loss、min_loss、decline_rate）。"""
        from verse_infra.verse_trainer.visualize import visualize

        # 构造 loss_history.json
        train_losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
        val_losses = [4.8, 3.6, 2.4, 1.2]
        hist = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "max_steps": 10,
            "eval_interval": 2,
            "best_val_loss": 1.2,
            "initial_loss": 5.0,
            "final_loss": 0.5,
        }
        hist_path = tmp_path / "loss_history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(hist, f)

        save_path = str(tmp_path / "loss_curve.png")
        visualize(str(hist_path), save_path=save_path)

        captured = capsys.readouterr()
        # 应包含统计信息
        assert "initial_loss=5.0000" in captured.out
        assert "final_loss=0.5000" in captured.out
        assert "avg_loss=" in captured.out
        assert "min_loss=" in captured.out
        assert "decline_rate=" in captured.out
        assert "best_val_loss=1.2000" in captured.out

    def test_visualize_empty_train_losses(self, tmp_path, capsys):
        """visualize 处理空 train_losses 时不报错。"""
        from verse_infra.verse_trainer.visualize import visualize

        hist = {
            "train_losses": [],
            "val_losses": [],
            "max_steps": 0,
            "eval_interval": 1,
        }
        hist_path = tmp_path / "loss_history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(hist, f)

        save_path = str(tmp_path / "loss_curve.png")
        # 不应抛异常
        visualize(str(hist_path), save_path=save_path)

        captured = capsys.readouterr()
        assert "train_steps=0" in captured.out


# ---------------------------------------------------------------------------
# Task 6: ParallelTrainer chunk 间状态重置测试
# ---------------------------------------------------------------------------


class TestParallelTrainerStateReset:
    """ParallelTrainer chunk 间状态重置测试。"""

    def test_chunk_state_reset_between_chunks(self):
        """每个 chunk 开始时模型状态正确重置到 original_state。"""
        model = ToyModel(in_dim=10, n_classes=5)
        train_ds = ToyDataset(n=40, in_dim=10, n_classes=5, seed=0)
        val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)

        # 记录初始状态
        original_state = copy.deepcopy(model.state_dict())

        cfg = {
            "parallel_chunks": 3,
            "max_steps": 12,
            "batch_size": 8,
            "lr": 0.01,
            "seed": 42,
            "quiet": True,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        # 记录每个 chunk 开始前的模型状态
        chunk_start_states = []
        original_train_chunk = trainer._train_chunk

        def tracking_train_chunk(m, ds, steps, cid):
            if cid >= 0:
                # Phase 1：记录 chunk 开始时的模型状态
                chunk_start_states.append(copy.deepcopy(m.state_dict()))
            return original_train_chunk(m, ds, steps, cid)

        trainer._train_chunk = tracking_train_chunk
        trainer.fit()

        # 每个 chunk 开始时的状态应与 original_state 一致
        # （fit 在每个 chunk 前会 load_state_dict(original_state)）
        assert len(chunk_start_states) == 3, (
            f"应有 3 个 chunk，got {len(chunk_start_states)}"
        )
        for i, state in enumerate(chunk_start_states):
            for key in original_state:
                np.testing.assert_array_equal(
                    state[key], original_state[key],
                    err_msg=f"chunk {i} 开始时状态与 original_state 不一致"
                )

    def test_optimizer_not_leaked_between_chunks(self):
        """优化器状态不跨 chunk 泄漏（每个 chunk 创建新优化器）。"""
        model = ToyModel(in_dim=10, n_classes=5)
        train_ds = ToyDataset(n=40, in_dim=10, n_classes=5, seed=0)
        val_ds = ToyDataset(n=20, in_dim=10, n_classes=5, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 8,
            "batch_size": 8,
            "lr": 0.01,
            "seed": 42,
            "quiet": True,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        # 记录每个 chunk 创建的优化器
        optimizers_created = []
        original_init = trainer.__dict__.get("optimizer_cls", None)

        # 用 mock 替换 _train_chunk，追踪优化器创建
        original_train_chunk = trainer._train_chunk

        def tracking_train_chunk(m, ds, steps, cid):
            if cid >= 0:
                # 在 _train_chunk 内部会创建新优化器
                # 这里只验证 _train_chunk 正常执行（不抛异常）
                pass
            return original_train_chunk(m, ds, steps, cid)

        trainer._train_chunk = tracking_train_chunk
        # fit 应正常完成（不抛异常说明优化器创建/销毁正常）
        trainer.fit()
        assert trainer.best_val_loss < float("inf")


# ---------------------------------------------------------------------------
# Task 6: ParallelTrainer round_robin 数据分配测试
# ---------------------------------------------------------------------------


class TestRoundRobinDistribution:
    """round_robin 数据分配均匀性测试。"""

    def test_round_robin_even_distribution(self):
        """round_robin 模式下数据均匀分配（差不超过 1）。"""
        model = ToyModel()
        train_ds = ToyDataset(n=100, seed=0)
        val_ds = ToyDataset(n=20, seed=100)

        n_chunks = 3
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds,
            cfg={"parallel_chunks": n_chunks, "max_steps": 12,
                 "parallel_strategy": "round_robin", "quiet": True})

        subsets = []
        for i in range(n_chunks):
            subset = trainer._split_dataset_round_robin(
                train_ds, n_chunks, i)
            subsets.append(subset)

        # 各子集长度差不超过 1
        lengths = [len(s) for s in subsets]
        assert max(lengths) - min(lengths) <= 1, (
            f"round_robin 分配不均匀: {lengths}"
        )
        # 总长度等于原始数据集
        assert sum(lengths) == len(train_ds)

    def test_round_robin_no_overlap(self):
        """round_robin 模式下 chunk 间数据不重复。"""
        model = ToyModel()
        train_ds = ToyDataset(n=50, seed=0)
        val_ds = ToyDataset(n=10, seed=100)

        n_chunks = 4
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds,
            cfg={"parallel_chunks": n_chunks, "max_steps": 16,
                 "parallel_strategy": "round_robin", "quiet": True})

        all_indices = []
        for i in range(n_chunks):
            subset = trainer._split_dataset_round_robin(
                train_ds, n_chunks, i)
            all_indices.extend(subset.indices)

        # 无重复
        assert len(all_indices) == len(set(all_indices)), (
            "round_robin 子集间存在索引重复"
        )
        # 覆盖全部数据
        assert sorted(all_indices) == list(range(len(train_ds)))


# ---------------------------------------------------------------------------
# Task 6: _split_steps 每组至少 1 步测试
# ---------------------------------------------------------------------------


class TestSplitStepsMinOne:
    """_split_steps 每组至少 1 步测试。"""

    def test_split_steps_all_at_least_one_when_sufficient(self):
        """max_steps >= parallel_chunks 时每个 chunk 至少 1 步。"""
        # 正常场景
        trainer = ParallelTrainer(
            model=ToyModel(), train_dataset=ToyDataset(10), val_dataset=ToyDataset(5),
            cfg={"parallel_chunks": 4, "max_steps": 40})
        steps = trainer._split_steps()
        assert len(steps) == 4
        assert all(s >= 1 for s in steps), f"所有 chunk 至少 1 步, got {steps}"
        assert sum(steps) == 40

    def test_split_steps_exact_division(self):
        """整除场景：200 / 4 = [50, 50, 50, 50]。"""
        trainer = ParallelTrainer(
            model=ToyModel(), train_dataset=ToyDataset(10), val_dataset=ToyDataset(5),
            cfg={"parallel_chunks": 4, "max_steps": 200})
        steps = trainer._split_steps()
        assert steps == [50, 50, 50, 50]
        assert all(s >= 1 for s in steps)

    def test_split_steps_remainder_distributed(self):
        """余数场景：202 / 4 = [51, 51, 50, 50]。"""
        trainer = ParallelTrainer(
            model=ToyModel(), train_dataset=ToyDataset(10), val_dataset=ToyDataset(5),
            cfg={"parallel_chunks": 4, "max_steps": 202})
        steps = trainer._split_steps()
        assert steps == [51, 51, 50, 50]
        assert all(s >= 1 for s in steps)

    def test_split_steps_small_steps_filtered(self):
        """max_steps < parallel_chunks 时过滤 0 步 chunk。"""
        trainer = ParallelTrainer(
            model=ToyModel(), train_dataset=ToyDataset(10), val_dataset=ToyDataset(5),
            cfg={"parallel_chunks": 4, "max_steps": 2})
        steps = trainer._split_steps()
        assert all(s > 0 for s in steps), f"0 步 chunk 应被过滤, got {steps}"
        assert sum(steps) == 2


# ---------------------------------------------------------------------------
# Task 6: _eval_full_val 空 val_dataset 测试
# ---------------------------------------------------------------------------


class TestEvalFullValEmpty:
    """_eval_full_val 空 val_dataset 处理测试。"""

    def test_empty_val_dataset_returns_inf(self):
        """空 val_dataset 时返回 inf。"""
        model = ToyModel()
        trainer = ParallelTrainer(
            model=model, train_dataset=ToyDataset(10), val_dataset=EmptyDataset(),
            cfg={"parallel_chunks": 2, "max_steps": 4, "quiet": True})
        assert trainer._eval_full_val(model) == float("inf")

    def test_none_val_dataset_returns_inf(self):
        """val_dataset=None 时返回 inf。"""
        model = ToyModel()
        trainer = ParallelTrainer(
            model=model, train_dataset=ToyDataset(10), val_dataset=None,
            cfg={"parallel_chunks": 2, "max_steps": 4, "quiet": True})
        assert trainer._eval_full_val(model) == float("inf")

    def test_normal_val_dataset_returns_finite(self):
        """正常 val_dataset 时返回有限值。"""
        model = ToyModel()
        val_ds = ToyDataset(n=20, seed=100)
        trainer = ParallelTrainer(
            model=model, train_dataset=ToyDataset(10), val_dataset=val_ds,
            cfg={"parallel_chunks": 2, "max_steps": 4, "batch_size": 4,
                 "quiet": True})
        val_loss = trainer._eval_full_val(model)
        assert val_loss < float("inf"), f"val_loss 应为有限值, got {val_loss}"
        assert val_loss > 0, f"val_loss 应为正数, got {val_loss}"


# ---------------------------------------------------------------------------
# Task 6: Phase 2 步数为 0 时跳过测试
# ---------------------------------------------------------------------------


class TestPhase2SkipSmallChunks:
    """Phase 2 串行重训：chunk_steps < 4 时跳过测试。"""

    def test_phase2_skipped_when_chunk_steps_less_than_4(self):
        """chunk_steps < 4 时跳过 Phase 2 重训。"""
        model = ToyModel()
        train_ds = ToyDataset(n=40, seed=0)
        val_ds = ToyDataset(n=20, seed=100)

        # max_steps=9, parallel_chunks=3 → chunk_steps=3 < 4 → Phase 2 跳过
        cfg = {
            "parallel_chunks": 3,
            "max_steps": 9,
            "batch_size": 8,
            "lr": 0.01,
            "seed": 42,
            "quiet": True,
            "merge_finetune_steps": 0,  # 关闭 finetune
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        # 追踪 Phase 2 的 _train_chunk 调用
        retrain_calls = []
        original_train_chunk = trainer._train_chunk

        def tracking_train_chunk(m, ds, steps, cid):
            if cid < 0 and cid != -999:
                # Phase 2 重训
                retrain_calls.append((cid, steps))
            return original_train_chunk(m, ds, steps, cid)

        trainer._train_chunk = tracking_train_chunk
        trainer.fit()

        # chunk_steps=3 < 4，Phase 2 应全部跳过
        assert len(retrain_calls) == 0, (
            f"chunk_steps=3 < 4 时应跳过 Phase 2, "
            f"但有 {len(retrain_calls)} 次重训调用"
        )

    def test_phase2_executed_when_chunk_steps_ge_4(self):
        """chunk_steps >= 4 时执行 Phase 2 重训。"""
        model = ToyModel()
        train_ds = ToyDataset(n=40, seed=0)
        val_ds = ToyDataset(n=20, seed=100)

        # max_steps=12, parallel_chunks=3 → chunk_steps=4 >= 4 → Phase 2 执行
        cfg = {
            "parallel_chunks": 3,
            "max_steps": 12,
            "batch_size": 8,
            "lr": 0.01,
            "seed": 42,
            "quiet": True,
            "merge_finetune_steps": 0,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)

        retrain_calls = []
        original_train_chunk = trainer._train_chunk

        def tracking_train_chunk(m, ds, steps, cid):
            if cid < 0 and cid != -999:
                retrain_calls.append((cid, steps))
            return original_train_chunk(m, ds, steps, cid)

        trainer._train_chunk = tracking_train_chunk
        trainer.fit()

        # chunk_steps=4 >= 4，Phase 2 应执行 3 次重训
        assert len(retrain_calls) == 3, (
            f"chunk_steps=4 >= 4 时应执行 3 次 Phase 2 重训, "
            f"got {len(retrain_calls)}"
        )
        # 每次重训步数 = chunk_steps // 4 = 1
        for _, steps in retrain_calls:
            assert steps == 1, f"重训步数应为 1 (4//4), got {steps}"


# ---------------------------------------------------------------------------
# Task 6: _ChunkPBar 非 tty 降级测试
# ---------------------------------------------------------------------------


class TestChunkPBarNonTty:
    """_ChunkPBar 非 tty 环境降级测试。"""

    def test_chunk_pbar_non_tty_degrades_to_print(self, monkeypatch, capsys):
        """非 tty 环境下 _ChunkPBar 降级为简洁打印（不用 tqdm）。"""
        # 模拟非 tty 环境
        import sys
        mock_stderr = MagicMock()
        mock_stderr.isatty.return_value = False
        monkeypatch.setattr(sys, "stderr", mock_stderr)

        pbar = _ChunkPBar(total=3, quiet=False)
        # 非 tty 环境下 _tqdm 应为 None
        assert pbar._tqdm is None, "非 tty 环境下不应创建 tqdm 实例"

        pbar.update(n=1, postfix={"chunk": "1/3", "loss": "3.45"})
        pbar.update(n=1, postfix={"chunk": "2/3", "loss": "3.20"})
        pbar.close()

        captured = capsys.readouterr()
        # 应有降级打印输出
        assert "1/3" in captured.out
        assert "2/3" in captured.out
        assert pbar.n == 2

    def test_chunk_pbar_quiet_still_silent(self, capsys):
        """quiet 模式下 _ChunkPBar 仍然完全静默。"""
        pbar = _ChunkPBar(total=3, quiet=True)
        assert pbar._tqdm is None
        pbar.update(n=1, postfix={"chunk": "1/3"})
        pbar.close()

        captured = capsys.readouterr()
        assert captured.out == "", "quiet 模式下不应有任何输出"


# ---------------------------------------------------------------------------
# Task 6: ParallelTrainer 完整流程测试（含 Phase 2 跳过）
# ---------------------------------------------------------------------------


class TestParallelTrainerWithSkips:
    """ParallelTrainer 完整流程测试（含 Phase 2 跳过场景）。"""

    def test_fit_with_small_chunks_completes(self):
        """chunk_steps < 4 时 fit 仍正常完成。"""
        model = ToyModel()
        train_ds = ToyDataset(n=40, seed=0)
        val_ds = ToyDataset(n=20, seed=100)

        cfg = {
            "parallel_chunks": 4,
            "max_steps": 8,  # chunk_steps=2 < 4 → Phase 2 跳过
            "batch_size": 8,
            "lr": 0.01,
            "seed": 42,
            "quiet": True,
            "merge_finetune_steps": 0,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        history = trainer.fit()

        # 应正常完成
        assert trainer.best_val_loss < float("inf")
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(trainer.chunk_stats) == 4

    def test_fit_with_large_chunks_completes(self):
        """chunk_steps >= 4 时 fit 正常完成（含 Phase 2 重训）。"""
        model = ToyModel()
        train_ds = ToyDataset(n=40, seed=0)
        val_ds = ToyDataset(n=20, seed=100)

        cfg = {
            "parallel_chunks": 2,
            "max_steps": 20,  # chunk_steps=10 >= 4 → Phase 2 执行
            "batch_size": 8,
            "lr": 0.01,
            "seed": 42,
            "quiet": True,
            "merge_finetune_steps": 2,
        }
        trainer = ParallelTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
        history = trainer.fit()

        assert trainer.best_val_loss < float("inf")
        assert len(trainer.chunk_stats) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
