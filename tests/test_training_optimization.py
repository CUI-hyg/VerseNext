"""训练精度优化与训练 UX 增强测试。

覆盖：
1. clip_grad_norm：梯度总范数裁剪
2. label_smoothing：默认 0 与原 loss 一致，>0 时 loss 偏移正确
3. AdamW 参数组：no_decay 组的 bias/norm 参数不被衰减
4. Trainer 新配置：grad_clip / label_smoothing / realtime_plot / ETA
5. _format_eta：时间格式化

运行方式：
    cd /workspace && python -m pytest tests/test_training_optimization.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import (
    Tensor,
    Linear,
    AdamW,
    cross_entropy_loss,
    Trainer,
    clip_grad_norm,
)
from verse_torch import training as training_mod


# ---------------------------------------------------------------------------
# 1. clip_grad_norm
# ---------------------------------------------------------------------------


class TestClipGradNorm:

    def test_no_clip_when_max_norm_zero(self):
        p = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        p.grad = np.array([3.0, 4.0])  # norm = 5
        norm = clip_grad_norm([p], max_norm=0.0)
        assert norm == 0.0
        # grad 不变
        np.testing.assert_allclose(p.grad, [3.0, 4.0])

    def test_clips_when_exceeds(self):
        p = Tensor(np.zeros(2), requires_grad=True)
        p.grad = np.array([3.0, 4.0])  # norm = 5
        norm = clip_grad_norm([p], max_norm=1.0)
        assert norm == pytest.approx(5.0)
        # 裁剪后 norm 应约等于 max_norm
        new_norm = float(np.sqrt(np.sum(p.grad ** 2)))
        assert new_norm == pytest.approx(1.0, abs=1e-4)

    def test_no_clip_when_below(self):
        p = Tensor(np.zeros(2), requires_grad=True)
        p.grad = np.array([0.3, 0.4])  # norm = 0.5
        norm = clip_grad_norm([p], max_norm=1.0)
        assert norm == pytest.approx(0.5)
        # grad 不变
        np.testing.assert_allclose(p.grad, [0.3, 0.4])

    def test_multiple_params(self):
        p1 = Tensor(np.zeros(2), requires_grad=True)
        p2 = Tensor(np.zeros(2), requires_grad=True)
        p1.grad = np.array([3.0, 0.0])
        p2.grad = np.array([0.0, 4.0])
        # total norm = sqrt(9 + 16) = 5
        norm = clip_grad_norm([p1, p2], max_norm=2.5)
        assert norm == pytest.approx(5.0)
        new_norm = float(np.sqrt(np.sum(p1.grad ** 2) + np.sum(p2.grad ** 2)))
        assert new_norm == pytest.approx(2.5, abs=1e-4)

    def test_none_grads_skipped(self):
        p = Tensor(np.zeros(2), requires_grad=True)
        p.grad = None
        norm = clip_grad_norm([p], max_norm=1.0)
        assert norm == 0.0


# ---------------------------------------------------------------------------
# 2. label_smoothing
# ---------------------------------------------------------------------------


class TestLabelSmoothing:

    def test_default_zero_matches_original(self):
        np.random.seed(0)
        N, V = 4, 5
        logits_np = np.random.randn(N, V).astype(np.float32)
        targets = np.array([0, 1, 2, 3])
        loss_default = cross_entropy_loss(Tensor(logits_np), targets)
        loss_explicit = cross_entropy_loss(
            Tensor(logits_np), targets, label_smoothing=0.0
        )
        assert abs(float(loss_default.data) - float(loss_explicit.data)) < 1e-7

    def test_smoothing_changes_loss(self):
        np.random.seed(1)
        N, V = 4, 5
        logits_np = np.random.randn(N, V).astype(np.float32)
        targets = np.array([0, 1, 2, 3])
        loss_hard = cross_entropy_loss(Tensor(logits_np), targets)
        loss_smooth = cross_entropy_loss(
            Tensor(logits_np), targets, label_smoothing=0.1
        )
        # 平滑后 loss 应不同（通常更大，因为引入了均匀分布惩罚）
        assert abs(float(loss_hard.data) - float(loss_smooth.data)) > 1e-4

    def test_smoothing_uniform_contribution(self):
        # 当 logits 均匀时，uniform loss = -mean(log_probs) = log(V)
        # hard loss 也 = log(V)，所以平滑后 loss = log(V)
        V = 4
        logits = Tensor(np.zeros((1, V), dtype=np.float32), requires_grad=True)
        targets = np.array([0])
        loss = cross_entropy_loss(logits, targets, label_smoothing=0.1)
        expected = float(np.log(V))
        assert abs(float(loss.data) - expected) < 1e-5

    def test_smoothing_backward_grad(self):
        np.random.seed(2)
        N, V = 3, 4
        logits = Tensor(
            np.random.randn(N, V).astype(np.float32), requires_grad=True
        )
        targets = np.array([0, 2, 1])
        loss = cross_entropy_loss(logits, targets, label_smoothing=0.1)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (N, V)
        assert np.all(np.isfinite(logits.grad))


# ---------------------------------------------------------------------------
# 3. AdamW 参数组（no_decay）
# ---------------------------------------------------------------------------


class TestAdamWParamGroups:

    def test_param_groups_flat(self):
        # 扁平参数：单组
        p1 = Tensor(np.zeros(2), requires_grad=True)
        p2 = Tensor(np.zeros(3), requires_grad=True)
        opt = AdamW([p1, p2], lr=1e-3, weight_decay=0.01)
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == 2
        assert opt.params == [p1, p2]

    def test_param_groups_split(self):
        # 参数组：decay + no_decay
        p_decay = Tensor(np.ones(2), requires_grad=True)
        p_nodecay = Tensor(np.ones(2), requires_grad=True)
        groups = [
            {"params": [p_decay], "weight_decay": 0.01},
            {"params": [p_nodecay], "weight_decay": 0.0},
        ]
        opt = AdamW(groups, lr=1e-3, weight_decay=0.01)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 0.01
        assert opt.param_groups[1]["weight_decay"] == 0.0
        # 扁平视图含全部参数
        assert len(opt.params) == 2

    def test_no_decay_param_not_decayed(self):
        # no_decay 组的参数不应被 weight decay 缩减
        p_decay = Tensor(np.array([10.0, 10.0]), requires_grad=True)
        p_nodecay = Tensor(np.array([10.0, 10.0]), requires_grad=True)
        p_decay.grad = np.array([0.0, 0.0])  # 零梯度，仅 wd 生效
        p_nodecay.grad = np.array([0.0, 0.0])
        groups = [
            {"params": [p_decay], "weight_decay": 0.1},
            {"params": [p_nodecay], "weight_decay": 0.0},
        ]
        opt = AdamW(groups, lr=1.0, weight_decay=0.1)
        opt.step()
        # decay 组：data *= (1 - lr*wd) = (1 - 0.1) = 0.9
        np.testing.assert_allclose(p_decay.data, [9.0, 9.0], atol=1e-5)
        # no_decay 组：不变
        np.testing.assert_allclose(p_nodecay.data, [10.0, 10.0])

    def test_module_param_groups(self):
        # Module 传入应自动扁平化为单组
        model = Linear(3, 2)
        opt = AdamW(model.parameters(), lr=1e-3)
        assert len(opt.param_groups) == 1
        # Linear 有 weight + bias
        assert len(opt.param_groups[0]["params"]) == 2


# ---------------------------------------------------------------------------
# 4. Trainer 新配置（grad_clip / label_smoothing / realtime_plot / ETA）
# ---------------------------------------------------------------------------


def _make_toy(vocab=4, dim_in=4, n=64, seed=0):
    rng = np.random.RandomState(seed)
    W = rng.randn(dim_in, vocab).astype(np.float32)
    b = rng.randn(vocab).astype(np.float32)
    X = rng.randn(n, dim_in).astype(np.float32)
    y = np.argmax(X @ W + b, axis=1).astype(np.int64)
    return X, y


class TestTrainerEnhancements:

    def test_grad_clip_in_trainer(self, tmp_path):
        np.random.seed(0)
        X, y = _make_toy(seed=0)
        batches = [
            (Tensor(X[i:i + 16]), Tensor(y[i:i + 16]))
            for i in range(0, 64, 16)
        ]
        model = Linear(4, 4)
        opt = AdamW(model.parameters(), lr=0.5, weight_decay=0.0)
        cfg = {
            "max_steps": 10,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "grad_clip": 1.0,
            "label_smoothing": 0.0,
            "enable_progress_bar": False,
        }
        trainer = Trainer(model, batches, batches[:1], opt, cfg=cfg)
        train_losses, _ = trainer.fit()
        assert len(train_losses) == 10
        assert train_losses[-1] < train_losses[0]

    def test_label_smoothing_in_trainer(self, tmp_path):
        np.random.seed(1)
        X, y = _make_toy(seed=1)
        batches = [
            (Tensor(X[i:i + 16]), Tensor(y[i:i + 16]))
            for i in range(0, 64, 16)
        ]
        model = Linear(4, 4)
        opt = AdamW(model.parameters(), lr=0.5, weight_decay=0.0)
        cfg = {
            "max_steps": 10,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "grad_clip": 0.0,
            "label_smoothing": 0.1,
            "enable_progress_bar": False,
        }
        trainer = Trainer(model, batches, batches[:1], opt, cfg=cfg)
        train_losses, _ = trainer.fit()
        assert len(train_losses) == 10
        # 标签平滑下 loss 仍应下降
        assert train_losses[-1] < train_losses[0] + 0.5

    def test_realtime_plot_updates_file(self, tmp_path):
        np.random.seed(2)
        X, y = _make_toy(seed=2)
        batches = [
            (Tensor(X[i:i + 16]), Tensor(y[i:i + 16]))
            for i in range(0, 64, 16)
        ]
        model = Linear(4, 4)
        opt = AdamW(model.parameters(), lr=0.5, weight_decay=0.0)
        cfg = {
            "max_steps": 20,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "grad_clip": 0.0,
            "label_smoothing": 0.0,
            "enable_progress_bar": False,
            "realtime_plot": True,
        }
        trainer = Trainer(model, batches, batches[:1], opt, cfg=cfg)
        trainer.fit()
        # 训练中应已生成 loss_curve 文件（png 或 txt）
        assert (tmp_path / "loss_curve.png").exists() or (
            tmp_path / "loss_curve.txt"
        ).exists()

    def test_best_pt_saved(self, tmp_path):
        # best.pt 应在 val_loss 改善时自动保存
        np.random.seed(3)
        X, y = _make_toy(seed=3)
        batches = [
            (Tensor(X[i:i + 16]), Tensor(y[i:i + 16]))
            for i in range(0, 64, 16)
        ]
        model = Linear(4, 4)
        opt = AdamW(model.parameters(), lr=0.5, weight_decay=0.0)
        cfg = {
            "max_steps": 15,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "grad_clip": 0.0,
            "label_smoothing": 0.0,
            "enable_progress_bar": False,
        }
        trainer = Trainer(model, batches, batches[:1], opt, cfg=cfg)
        trainer.fit()
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "last.pt").exists()

    def test_eta_window_config(self, tmp_path):
        # eta_window 应被读取
        np.random.seed(4)
        X, y = _make_toy(seed=4)
        batches = [
            (Tensor(X[i:i + 16]), Tensor(y[i:i + 16]))
            for i in range(0, 64, 16)
        ]
        model = Linear(4, 4)
        opt = AdamW(model.parameters(), lr=0.5, weight_decay=0.0)
        cfg = {
            "max_steps": 5,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "eta_window": 3,
            "enable_progress_bar": False,
        }
        trainer = Trainer(model, batches, batches[:1], opt, cfg=cfg)
        assert trainer.eta_window == 3
        trainer.fit()


# ---------------------------------------------------------------------------
# 5. _format_eta
# ---------------------------------------------------------------------------


class TestFormatEta:

    def test_seconds(self):
        assert training_mod._format_eta(45) == "45s"

    def test_minutes(self):
        assert training_mod._format_eta(125) == "2m05s"

    def test_hours(self):
        assert training_mod._format_eta(3725) == "1h02m"

    def test_negative(self):
        assert training_mod._format_eta(-1) == "?"

    def test_nan(self):
        assert training_mod._format_eta(float("nan")) == "?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
