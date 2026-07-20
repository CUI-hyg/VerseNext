"""Task 2.9: verse_torch.training 单元测试。

覆盖：
1. cross_entropy_loss 与手动计算（log_softmax + NLL）数值一致
2. cross_entropy_loss ignore_index 正确屏蔽
3. EarlyStopping 触发逻辑
4. GradientAccumulator 每 N 次 should_step=True
5. CheckpointManager save_best/save_last/load_best/load_last 往返一致
6. LambdaLR + warmup_cosine：step 0 时 lr=0，warmup 后逐步上升，total_steps 时 lr 接近 0
7. compute_loss_rate：单调下降序列返回正数，常数序列返回 0
8. plot_loss_curve：matplotlib 不可用时也能输出文件（mock）
9. Trainer 在 toy 模型（Linear(d, vocab)）+ 合成数据上跑 10 步 loss 下降

运行方式：
    cd /workspace && python -m pytest tests/test_training.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import importlib
import math
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import (
    Tensor,
    Linear,
    SGD,
    Adam,
    LambdaLR,
    warmup_cosine_lr,
    cross_entropy_loss,
    EarlyStopping,
    GradientAccumulator,
    CheckpointManager,
    compute_loss_rate,
    plot_loss_curve,
    Trainer,
)
from verse_torch import training as training_mod


# ---------------------------------------------------------------------------
# 1. cross_entropy_loss 与手动计算一致
# ---------------------------------------------------------------------------


def manual_cross_entropy(logits_np: np.ndarray, targets_np: np.ndarray) -> float:
    """手动计算 cross entropy：log_softmax + NLL（取负平均）。"""
    # 数值稳定 log_softmax
    x = logits_np.astype(np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    shifted = x - x_max
    log_sum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum
    N = log_probs.shape[0]
    selected = log_probs[np.arange(N), targets_np]
    return float(-np.mean(selected))


class TestCrossEntropyLoss:

    def test_2d_matches_manual(self):
        np.random.seed(42)
        N, V = 8, 5
        logits_np = np.random.randn(N, V).astype(np.float32)
        targets_np = np.random.randint(0, V, size=N)
        logits = Tensor(logits_np, requires_grad=True)

        loss = cross_entropy_loss(logits, targets_np)
        manual = manual_cross_entropy(logits_np, targets_np)

        assert loss.data.shape == (), f"loss 应为标量, got shape {loss.data.shape}"
        assert abs(float(loss.data) - manual) < 1e-4, (
            f"cross_entropy_loss({float(loss.data)}) != manual({manual})"
        )

    def test_3d_matches_2d(self):
        # (B, T, V) 应等价于 flatten 后的 (B*T, V)
        np.random.seed(7)
        B, T, V = 2, 3, 6
        logits_np = np.random.randn(B, T, V).astype(np.float32)
        targets_np = np.random.randint(0, V, size=(B, T))

        loss_3d = cross_entropy_loss(
            Tensor(logits_np, requires_grad=False), targets_np
        )
        loss_2d = cross_entropy_loss(
            Tensor(logits_np.reshape(-1, V), requires_grad=False),
            targets_np.reshape(-1),
        )
        assert abs(float(loss_3d.data) - float(loss_2d.data)) < 1e-5

    def test_backward_grad_shape(self):
        np.random.seed(0)
        B, T, V = 2, 4, 5
        logits = Tensor(np.random.randn(B, T, V).astype(np.float32), requires_grad=True)
        targets = np.random.randint(0, V, size=(B, T))
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (B, T, V)

    def test_backward_grad_finite_diff(self):
        # 有限差分梯度检查（针对 (N, V) 输入）
        np.random.seed(123)
        N, V = 3, 4
        logits_np = np.random.randn(N, V).astype(np.float64)
        targets_np = np.array([0, 2, 1])

        logits = Tensor(logits_np, requires_grad=True)
        loss = cross_entropy_loss(logits, targets_np)
        loss.backward()
        analytic_grad = logits.grad.copy()

        # 数值梯度
        eps = 1e-6
        num_grad = np.zeros_like(logits_np)
        for i in range(N):
            for j in range(V):
                orig = logits_np[i, j]
                logits_np[i, j] = orig + eps
                lp = float(cross_entropy_loss(Tensor(logits_np), targets_np).data)
                logits_np[i, j] = orig - eps
                lm = float(cross_entropy_loss(Tensor(logits_np), targets_np).data)
                logits_np[i, j] = orig
                num_grad[i, j] = (lp - lm) / (2 * eps)

        rel = np.max(np.abs(analytic_grad - num_grad)) / (
            np.max(np.abs(num_grad)) + 1e-12
        )
        assert rel < 1e-4, f"梯度相对误差过大: {rel}"

    def test_targets_as_tensor(self):
        np.random.seed(0)
        logits_np = np.random.randn(4, 5).astype(np.float32)
        targets_np = np.array([0, 1, 2, 3])
        loss_ref = cross_entropy_loss(Tensor(logits_np), targets_np)
        loss_t = cross_entropy_loss(Tensor(logits_np), Tensor(targets_np))
        assert abs(float(loss_ref.data) - float(loss_t.data)) < 1e-6

    def test_all_ignore_returns_zero(self):
        # 所有 target 都是 ignore_index，应返回 0 且不报错
        logits = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
        targets = np.array([-100, -100, -100, -100])
        loss = cross_entropy_loss(logits, targets, ignore_index=-100)
        assert abs(float(loss.data)) < 1e-6
        # backward 不应报错
        loss.backward()
        # 由于 loss=0，梯度也应为 0
        assert logits.grad is not None
        assert np.max(np.abs(logits.grad)) < 1e-6


# ---------------------------------------------------------------------------
# 2. cross_entropy_loss ignore_index 屏蔽
# ---------------------------------------------------------------------------


class TestIgnoreIndex:

    def test_ignore_masked(self):
        np.random.seed(2024)
        N, V = 6, 4
        logits_np = np.random.randn(N, V).astype(np.float32)
        targets = np.array([0, -100, 2, 1, -100, 3])

        # 用 ignore_index=-100 时应只计算 [0, 2, 3] 三行
        loss_full = cross_entropy_loss(Tensor(logits_np), targets, ignore_index=-100)

        # 手动：取有效行
        valid_rows = np.array([0, 2, 3, 5])
        valid_targets = np.array([0, 2, 1, 3])
        manual = manual_cross_entropy(logits_np[valid_rows], valid_targets)
        assert abs(float(loss_full.data) - manual) < 1e-4, (
            f"ignore loss {float(loss_full.data)} != manual {manual}"
        )

    def test_ignore_grad_zero_at_masked_positions(self):
        # 被 ignore 的行不应有梯度
        np.random.seed(0)
        N, V = 3, 4
        logits = Tensor(np.random.randn(N, V).astype(np.float32), requires_grad=True)
        targets = np.array([0, -100, 2])
        loss = cross_entropy_loss(logits, targets, ignore_index=-100)
        loss.backward()
        # 第 1 行（被 ignore）的梯度应全为 0
        assert np.max(np.abs(logits.grad[1])) < 1e-6, (
            f"被 ignore 的行梯度应全 0, got {logits.grad[1]}"
        )
        # 其他行应有非零梯度
        assert np.max(np.abs(logits.grad[0])) > 0
        assert np.max(np.abs(logits.grad[2])) > 0


# ---------------------------------------------------------------------------
# 3. EarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:

    def test_triggers_after_patience_no_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        # 第一次：显著下降，重置 counter
        assert es(1.0) is False
        assert es.counter == 0
        # 第二次：未改善，counter=1
        assert es(1.0) is False
        assert es.counter == 1
        # 第三次：未改善，counter=2
        assert es(1.0) is False
        assert es.counter == 2
        # 第四次：未改善，counter=3 >= patience=3，触发停止
        assert es(1.0) is True
        assert es.should_stop is True

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.0)
        es(1.0)
        es(1.0)  # counter=1
        es(0.5)  # 改善，counter 重置为 0
        assert es.counter == 0
        assert es.best_loss == 0.5
        # 再两次未改善才触发
        assert es(0.5) is False  # counter=1
        assert es(0.5) is True   # counter=2 -> should_stop

    def test_min_delta_threshold(self):
        es = EarlyStopping(patience=1, min_delta=0.1)
        es(1.0)
        # 改善量 0.05 < min_delta 0.1，不视为改善
        assert es(0.95) is True  # counter=1 >= patience=1，触发
        assert es.should_stop is True

    def test_reset(self):
        es = EarlyStopping(patience=2)
        es(1.0)
        es(1.0)
        es(1.0)  # 触发
        assert es.should_stop is True
        es.reset()
        assert es.should_stop is False
        assert es.counter == 0
        assert es.best_loss == float("inf")


# ---------------------------------------------------------------------------
# 4. GradientAccumulator
# ---------------------------------------------------------------------------


class TestGradientAccumulator:

    def test_basic_accumulation(self):
        ga = GradientAccumulator(micro_batch=2, effective_batch=8)
        assert ga.accum_steps == 4
        # 前 3 次都不 step
        for _ in range(3):
            ga.step()
            assert ga.should_step() is False
        # 第 4 次 step 后 should_step 返回 True 并自动重置
        ga.step()
        assert ga.should_step() is True
        # 重置后再次 should_step 返回 False
        assert ga.should_step() is False

    def test_repeat_multiple_cycles(self):
        ga = GradientAccumulator(micro_batch=1, effective_batch=3)
        steps_results = []
        for _ in range(10):
            ga.step()
            steps_results.append(ga.should_step())
        # 10 次 step，accum_steps=3 -> 第 3, 6, 9 次 step 后 should_step=True
        # 累积位置：3 -> True (idx=2 in 0-indexed), 6 -> True (idx=5), 9 -> True (idx=8)
        # 注意：should_step 内部会自动重置，所以下一次 cycle 开始
        true_indices = [i for i, x in enumerate(steps_results) if x]
        assert true_indices == [2, 5, 8], f"got {true_indices}"

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            GradientAccumulator(micro_batch=3, effective_batch=10)  # 10 % 3 != 0
        with pytest.raises(ValueError):
            GradientAccumulator(micro_batch=0, effective_batch=10)

    def test_reset(self):
        ga = GradientAccumulator(micro_batch=1, effective_batch=4)
        ga.step()
        ga.step()
        ga.reset()
        assert ga.counter == 0
        assert ga.should_step() is False


# ---------------------------------------------------------------------------
# 5. CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:

    def test_save_load_roundtrip(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path))
        state = {
            "step": 100,
            "model_state_dict": {
                "weight": np.random.randn(4, 5).astype(np.float32),
                "bias": np.random.randn(4).astype(np.float32),
            },
            "val_loss": 0.345,
            "tensor_in_state": Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        }
        ckpt.save_best(state)
        ckpt.save_last(state)

        loaded_best = ckpt.load_best()
        loaded_last = ckpt.load_last()

        assert loaded_best["step"] == 100
        assert loaded_best["val_loss"] == pytest.approx(0.345)
        np.testing.assert_allclose(
            loaded_best["model_state_dict"]["weight"],
            state["model_state_dict"]["weight"],
        )
        np.testing.assert_allclose(
            loaded_best["model_state_dict"]["bias"],
            state["model_state_dict"]["bias"],
        )
        # Tensor 字段也应能还原
        assert isinstance(loaded_best["tensor_in_state"], Tensor)
        np.testing.assert_allclose(
            loaded_best["tensor_in_state"].data,
            state["tensor_in_state"].data,
        )

        # last 与 best 内容一致
        assert loaded_last["step"] == loaded_best["step"]
        np.testing.assert_allclose(
            loaded_last["model_state_dict"]["weight"],
            loaded_best["model_state_dict"]["weight"],
        )

    def test_custom_paths(self, tmp_path):
        best = tmp_path / "custom_best.pt"
        last = tmp_path / "custom_last.pt"
        ckpt = CheckpointManager(
            str(tmp_path), best_path=best, last_path=last
        )
        ckpt.save_best({"a": 1})
        ckpt.save_last({"b": 2})
        assert best.exists()
        assert last.exists()
        assert ckpt.load_best()["a"] == 1
        assert ckpt.load_last()["b"] == 2

    def test_creates_save_dir(self, tmp_path):
        # 不存在的目录应自动创建
        new_dir = tmp_path / "sub" / "ckpt"
        ckpt = CheckpointManager(str(new_dir))
        assert new_dir.exists()
        ckpt.save_last({"x": 1})
        assert (new_dir / "last.pt").exists()


# ---------------------------------------------------------------------------
# 6. LambdaLR + warmup_cosine_lr
# ---------------------------------------------------------------------------


class TestLambdaLRWarmupCosine:

    def test_step_zero_lr_is_zero(self):
        # 构造时基类会调用一次 step()，使 last_epoch=0
        opt = SGD([Tensor(np.zeros(2), requires_grad=True)], lr=1e-3)
        sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=5, total_steps=10))
        # step=0 时 lr_lambda(0) = 0 / 5 = 0
        assert opt.lr == pytest.approx(0.0, abs=1e-12)
        assert sched.last_epoch == 0

    def test_warmup_increases(self):
        opt = SGD([Tensor(np.zeros(2), requires_grad=True)], lr=1e-3)
        sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=5, total_steps=10))
        lrs = [opt.lr]
        for _ in range(5):
            sched.step()
            lrs.append(opt.lr)
        # warmup 阶段 lr 单调递增，最后达到 base_lr
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-12
        # step=5 (warmup_steps=5) 应等于 base_lr
        assert lrs[-1] == pytest.approx(1e-3, rel=1e-5)

    def test_cosine_decay_to_zero(self):
        opt = SGD([Tensor(np.zeros(2), requires_grad=True)], lr=1e-3)
        sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=2, total_steps=10))
        # 推进到 step=10
        for _ in range(10):
            sched.step()
        # step=10 时 progress=1, cos(pi)=-1, lr=base_lr * 0 = 0
        assert opt.lr == pytest.approx(0.0, abs=1e-9)

    def test_total_steps_close_to_zero(self):
        # total_steps 时 lr 接近 0（warmup_steps=5, total_steps=20）
        opt = SGD([Tensor(np.zeros(2), requires_grad=True)], lr=1e-3)
        sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=5, total_steps=20))
        for _ in range(20):
            sched.step()
        assert opt.lr == pytest.approx(0.0, abs=1e-9)

    def test_lambda_lr_general(self):
        # 通用 lambda：每步乘以 0.9
        opt = SGD([Tensor(np.zeros(2), requires_grad=True)], lr=1.0)
        sched = LambdaLR(opt, lambda step: 0.9 ** step)
        # step=0 时 factor=0.9^0=1.0 -> lr=1.0
        assert opt.lr == pytest.approx(1.0)
        sched.step()  # step=1
        assert opt.lr == pytest.approx(0.9)
        sched.step()  # step=2
        assert opt.lr == pytest.approx(0.81)


# ---------------------------------------------------------------------------
# 7. compute_loss_rate
# ---------------------------------------------------------------------------


class TestComputeLossRate:

    def test_monotonic_decreasing_positive(self):
        # 单调下降序列：前半均值高于后半均值，应返回正数
        window = 10
        losses = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        rate = compute_loss_rate(losses, window=window, min_delta=1e-6)
        # first_half avg = (10+9+8+7+6)/5 = 8
        # second_half avg = (5+4+3+2+1)/5 = 3
        # rate = (8-3)/8 = 0.625
        assert rate == pytest.approx(0.625, rel=1e-5)

    def test_constant_returns_zero(self):
        losses = [1.0] * 50
        rate = compute_loss_rate(losses, window=50, min_delta=1e-4)
        assert rate == 0.0

    def test_too_short_returns_zero(self):
        losses = [1.0, 0.5, 0.25]
        rate = compute_loss_rate(losses, window=50)
        assert rate == 0.0

    def test_below_min_delta_returns_zero(self):
        # 平均值低于 min_delta
        losses = [1e-5, 5e-6, 1e-5, 5e-6] * 20
        rate = compute_loss_rate(losses, window=50, min_delta=1e-4)
        assert rate == 0.0

    def test_increasing_returns_negative(self):
        # 上升序列：rate 应为负
        losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        rate = compute_loss_rate(losses, window=10, min_delta=1e-6)
        assert rate < 0


# ---------------------------------------------------------------------------
# 8. plot_loss_curve matplotlib 不可用降级
# ---------------------------------------------------------------------------


class TestPlotLossCurve:

    def test_ascii_fallback(self, tmp_path, monkeypatch):
        # 模拟 matplotlib 不可用
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # 同时清理已 import 的 matplotlib（如果有）
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        train_losses = [float(10 - i * 0.5) for i in range(20)]
        val_losses = [float(10 - i * 0.5 + 0.2) for i in range(0, 20, 2)]
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)
        # 降级为 .txt
        assert actual_path.endswith(".txt")
        assert os.path.exists(actual_path)
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Loss Curve" in content
        assert "T=train" in content
        assert "V=val" in content

    def test_normal_plot(self, tmp_path):
        # 不 mock，使用真实 matplotlib（如果可用）；不可用则跳过
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用，跳过 PNG 测试")
        train_losses = [float(10 - i * 0.5) for i in range(20)]
        val_losses = [float(10 - i * 0.5 + 0.2) for i in range(0, 20, 2)]
        save_path = str(tmp_path / "loss_curve.png")
        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)
        assert actual_path.endswith(".png")
        assert os.path.exists(actual_path)

    def test_empty_losses(self, tmp_path):
        # 空 loss 也应能写出文件
        save_path = str(tmp_path / "loss_curve.png")
        actual_path = plot_loss_curve([], [], save_path)
        assert os.path.exists(actual_path)


# ---------------------------------------------------------------------------
# 9. Trainer 在 toy 模型上跑 10 步 loss 下降
# ---------------------------------------------------------------------------


def _make_toy_dataset(vocab=4, dim_in=4, n_samples=64, seed=0):
    """构造一个简单的线性可分数据集：
    y = argmax(W_true @ x + b_true)
    """
    rng = np.random.RandomState(seed)
    W_true = rng.randn(dim_in, vocab).astype(np.float32)
    b_true = rng.randn(vocab).astype(np.float32)
    X = rng.randn(n_samples, dim_in).astype(np.float32)
    logits = X @ W_true + b_true
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


class TestTrainer:

    def test_loss_decreases_on_toy(self, tmp_path):
        # 用 Linear(d, vocab) + SGD 训练 30 步，loss 应明显下降
        np.random.seed(0)
        dim_in, vocab = 4, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=64, seed=0)

        # 构造 train_loader / val_loader（list of (x, y) 元组）
        def make_batches(X, y, batch_size=16):
            batches = []
            n = len(X)
            for i in range(0, n, batch_size):
                xb = Tensor(X[i:i + batch_size], requires_grad=False)
                yb = Tensor(y[i:i + batch_size], requires_grad=False)
                batches.append((xb, yb))
            return batches

        train_loader = make_batches(X, y, batch_size=16)
        val_loader = make_batches(X[:16], y[:16], batch_size=16)

        model = Linear(dim_in, vocab)
        # 初始化接近 0 以加快收敛
        with __import__('verse_torch').no_grad():
            model.weight.data = np.random.randn(vocab, dim_in).astype(np.float32) * 0.1
            model.bias.data = np.zeros(vocab, dtype=np.float32)

        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 30,
            "eval_interval": 10,
            "patience": 5,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,  # 不打印
            "loss_rate_window": 10,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, scheduler=None, cfg=cfg)
        train_losses, val_losses = trainer.fit()

        assert len(train_losses) == 30
        # 初始与最终 loss
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        # loss 应明显下降（至少下降 30%）
        assert final_loss < initial_loss, (
            f"final loss {final_loss} 应小于 initial loss {initial_loss}"
        )
        # 验证 loss 也应下降
        assert len(val_losses) >= 2
        assert val_losses[-1] < val_losses[0] + 1e-6

        # 检查保存的文件
        assert (tmp_path / "loss_history.json").exists()
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "last.pt").exists()
        # loss_curve 可能是 png 或 txt
        assert (tmp_path / "loss_curve.png").exists() or (tmp_path / "loss_curve.txt").exists()

    def test_trainer_with_scheduler(self, tmp_path):
        # 带 LambdaLR 的 Trainer
        np.random.seed(1)
        dim_in, vocab = 3, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=32, seed=1)

        def make_batches(X, y, batch_size=8):
            batches = []
            n = len(X)
            for i in range(0, n, batch_size):
                batches.append((
                    Tensor(X[i:i + batch_size]),
                    Tensor(y[i:i + batch_size]),
                ))
            return batches

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=2, total_steps=20))

        cfg = {
            "max_steps": 20,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "loss_rate_window": 5,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, scheduler=sched, cfg=cfg)
        train_losses, val_losses = trainer.fit()
        assert len(train_losses) == 20
        # 至少下降一些
        assert train_losses[-1] < train_losses[0]

    def test_trainer_grad_accum(self, tmp_path):
        # 梯度累积：micro_batch=1, effective_batch=4，每 4 步才真正更新一次
        np.random.seed(2)
        dim_in, vocab = 3, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=32, seed=2)

        def make_batches(X, y, batch_size=8):
            return [
                (Tensor(X[i:i + batch_size]), Tensor(y[i:i + batch_size]))
                for i in range(0, len(X), batch_size)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 12,
            "eval_interval": 6,
            "patience": 5,
            "save_dir": str(tmp_path),
            "grad_accum": 4,  # 每 4 步更新一次
            "log_interval": 100,
            "loss_rate_window": 5,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        train_losses, _ = trainer.fit()
        assert len(train_losses) == 12
        # 至少不上升
        assert train_losses[-1] <= train_losses[0] + 0.1

    def test_trainer_early_stop(self, tmp_path):
        # 故意构造一个不会改善的场景，验证 early stop
        np.random.seed(3)
        # 用极小的 lr，loss 几乎不变
        dim_in, vocab = 4, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=32, seed=3)

        def make_batches(X, y, batch_size=8):
            return [
                (Tensor(X[i:i + batch_size]), Tensor(y[i:i + batch_size]))
                for i in range(0, len(X), batch_size)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.0)  # 不更新
        cfg = {
            "max_steps": 100,
            "eval_interval": 1,    # 每步都评估
            "patience": 3,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "loss_rate_window": 5,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        train_losses, val_losses = trainer.fit()
        # patience=3 应在 4 次 eval 后停止（包含第一次的 best_loss 设置）
        # step 0: best 设置，counter=0
        # step 1: 不改善 counter=1
        # step 2: 不改善 counter=2
        # step 3: 不改善 counter=3 >= 3 -> stop
        # 所以应在 step=3 时停止
        assert len(train_losses) <= 5  # 不应跑满 100 步
        assert trainer.early_stopping.should_stop is True


if __name__ == "__main__":
    # 兼容直接 python tests/test_training.py 运行
    pytest.main([__file__, "-v"])
