"""Task 3.9a: verse_torch.optim_extras 单元测试。

覆盖 Lion 与 Adafactor 两个优化器：
1. test_lion_basic: Lion 训练简单线性回归，loss 下降
2. test_lion_sign_update: 验证更新方向是 sign(...)
3. test_lion_weight_decay: weight_decay 不为 0 时参数衰减
4. test_lion_param_groups: 支持参数组（no_decay 等）
5. test_adafactor_basic: Adafactor 训练简单线性回归，loss 下降
6. test_adafactor_factored: 2D 参数使用 factored 二阶矩（row/col）
7. test_adafactor_1d_param: 1D 参数使用普通 AdaGrad 风格（v）
8. test_adafactor_clipping: 更新裁剪（trust ratio clipping）生效

运行方式：
    cd /workspace && python -m pytest tests/test_optim_extras.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor, Lion, Adafactor


# ---------------------------------------------------------------------------
# Lion 测试
# ---------------------------------------------------------------------------


class TestLion:
    """Lion 优化器（sign 更新，无二阶矩）。"""

    def test_lion_basic(self):
        """Lion 训练简单线性回归，loss 应明显下降。"""
        np.random.seed(42)
        # 构造线性回归数据：y = X @ W_true + b_true
        N, d = 32, 4
        X_np = np.random.randn(N, d).astype(np.float32)
        W_true = np.array([2.0, -1.0, 0.5, 1.0], dtype=np.float32)
        b_true = 1.0
        Y_np = (X_np @ W_true + b_true).reshape(-1, 1)

        X = Tensor(X_np, requires_grad=False)
        Y = Tensor(Y_np, requires_grad=False)
        # 模型参数：W (d, 1), b (1,)
        W = Tensor(np.random.randn(d, 1).astype(np.float32) * 0.1, requires_grad=True)
        b = Tensor(np.zeros((1,), dtype=np.float32), requires_grad=True)

        opt = Lion([W, b], lr=1e-2, weight_decay=0.0)

        def loss_fn():
            pred = X @ W + b               # (N, 1)
            diff = pred - Y                # (N, 1)
            return (diff * diff).mean()    # 标量 MSE

        initial_loss = float(loss_fn().data)

        for _ in range(100):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()

        final_loss = float(loss_fn().data)
        # loss 应至少降到初始的 50% 以下
        assert final_loss < initial_loss * 0.5, (
            f"Lion 训练未收敛：initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_lion_sign_update(self):
        """验证 Lion 更新方向是 sign(m·β1 + g·(1-β1))。

        第一次 step 时 m=0，update = g·(1-β1)，sign(update) = sign(g)。
        """
        np.random.seed(0)
        # 标量向量参数，已知梯度
        p = Tensor(np.array([1.0, -1.0, 2.0, -3.0], dtype=np.float32), requires_grad=True)
        opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

        # 手动设置梯度（不通过 backward）
        g = np.array([0.5, -0.3, 0.0, 1.2], dtype=np.float32)
        p.grad = g

        p_before = p.data.copy()
        opt.step()

        # 第一次 step：m=0 → update = 0*0.9 + g*0.1 = 0.1*g
        # sign(update) = sign(g)（0.1 > 0，不改变符号）
        # p_new = p - lr * sign(g)
        expected = p_before - 0.1 * np.sign(g)
        np.testing.assert_allclose(p.data, expected, atol=1e-6)

        # 验证动量状态已写入
        state = opt.state.get(id(p), {})
        assert "exp_avg" in state, "Lion 应维护 exp_avg 动量状态"

    def test_lion_weight_decay(self):
        """weight_decay 不为 0 时参数应衰减。

        构造 grad=0 场景：sign(0)=0，所以只有 weight_decay 起作用。
        p_new = p - lr*sign(0) - lr*wd*p = p - lr*wd*p = p*(1 - lr*wd)
        """
        p = Tensor(np.array([1.0, 2.0, -1.5], dtype=np.float32), requires_grad=True)
        opt = Lion([p], lr=0.1, weight_decay=0.5)

        # grad=0 → sign(update)=0，仅 weight_decay 生效
        p.grad = np.zeros_like(p.data)

        p_before = p.data.copy()
        opt.step()

        # p_new = p * (1 - lr * wd) = p * (1 - 0.1*0.5) = p * 0.95
        expected = p_before * (1.0 - 0.1 * 0.5)
        np.testing.assert_allclose(p.data, expected, atol=1e-6)

        # 参数范数应减小
        assert np.linalg.norm(p.data) < np.linalg.norm(p_before)

    def test_lion_param_groups(self):
        """支持参数组（如 no_decay 组 vs decay 组）。"""
        p1 = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        p2 = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)

        param_groups = [
            {"params": [p1], "weight_decay": 0.0},   # no_decay 组
            {"params": [p2], "weight_decay": 0.5},   # decay 组
        ]
        opt = Lion(param_groups, lr=0.1, weight_decay=0.1)

        # 两者梯度都为 0，只有 weight_decay 起作用
        p1.grad = np.zeros_like(p1.data)
        p2.grad = np.zeros_like(p2.data)

        p1_before = p1.data.copy()
        p2_before = p2.data.copy()
        opt.step()

        # p1 在 no_decay 组：weight_decay=0 → 不衰减
        np.testing.assert_allclose(p1.data, p1_before, atol=1e-6)

        # p2 在 decay 组：weight_decay=0.5 → 衰减
        expected_p2 = p2_before * (1.0 - 0.1 * 0.5)
        np.testing.assert_allclose(p2.data, expected_p2, atol=1e-6)

        # 验证两组都被正确注册
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 0.0
        assert opt.param_groups[1]["weight_decay"] == 0.5


# ---------------------------------------------------------------------------
# Adafactor 测试
# ---------------------------------------------------------------------------


class TestAdafactor:
    """Adafactor 优化器（factored 二阶矩）。"""

    def test_adafactor_basic(self):
        """Adafactor 训练简单线性回归，loss 应明显下降。"""
        np.random.seed(42)
        N, d = 32, 4
        X_np = np.random.randn(N, d).astype(np.float32)
        W_true = np.array([2.0, -1.0, 0.5, 1.0], dtype=np.float32)
        Y_np = (X_np @ W_true + 1.0).reshape(-1, 1)

        X = Tensor(X_np, requires_grad=False)
        Y = Tensor(Y_np, requires_grad=False)
        W = Tensor(np.random.randn(d, 1).astype(np.float32) * 0.1, requires_grad=True)
        b = Tensor(np.zeros((1,), dtype=np.float32), requires_grad=True)

        # eps2 调大到 1.0：默认 1e-3 的 trust-ratio clipping 对小模型从头训练过严，
        # 会导致 update 被裁剪到极小值而无法收敛；eps2=1.0 允许 update_rms 与
        # param_rms 同量级，相当于 Adam 的 trust ratio = 1。
        opt = Adafactor([W, b], lr=0.5, eps2=1.0)

        def loss_fn():
            pred = X @ W + b
            diff = pred - Y
            return (diff * diff).mean()

        initial_loss = float(loss_fn().data)

        for _ in range(300):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()

        final_loss = float(loss_fn().data)
        assert final_loss < initial_loss * 0.5, (
            f"Adafactor 训练未收敛：initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_adafactor_factored(self):
        """2D 参数使用 factored 二阶矩（row/col 统计）。"""
        np.random.seed(0)
        p = Tensor(np.random.randn(4, 6).astype(np.float32), requires_grad=True)
        opt = Adafactor([p], lr=0.01)

        p.grad = np.random.randn(4, 6).astype(np.float32)
        opt.step()

        state = opt.state.get(id(p), {})
        # 2D 参数应使用 row/col 而非 v
        assert "row" in state, "2D 参数应使用 row 统计"
        assert "col" in state, "2D 参数应使用 col 统计"
        assert "v" not in state, "2D 参数不应使用 v（应使用 factored）"
        # row 形状应为 (4,)，col 形状应为 (6,)
        assert state["row"].shape == (4,), f"row shape 错误: {state['row'].shape}"
        assert state["col"].shape == (6,), f"col shape 错误: {state['col'].shape}"

    def test_adafactor_1d_param(self):
        """1D 参数使用普通 AdaGrad 风格（v 而非 row/col）。"""
        p = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        opt = Adafactor([p], lr=0.01)

        p.grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        opt.step()

        state = opt.state.get(id(p), {})
        # 1D 参数应使用 v（普通 AdaGrad 风格）
        assert "v" in state, "1D 参数应使用 v（普通 AdaGrad 风格）"
        assert "row" not in state, "1D 参数不应使用 row"
        assert "col" not in state, "1D 参数不应使用 col"
        # v 形状应与参数一致
        assert state["v"].shape == (3,), f"v shape 错误: {state['v'].shape}"

    def test_adafactor_clipping(self):
        """更新裁剪（trust ratio clipping）生效。

        构造参数极小但梯度极大的场景：
        - update 被 1/(sqrt(v_hat)+eps) 归一化后 ≈ 1
        - param_rms 极小 → clip_threshold = eps2*param_rms 极小
        - update_rms(1) > clip_threshold → 触发裁剪
        - 最终参数变化被限制在极小范围内
        """
        # 参数极小
        p = Tensor(np.array([1e-3, 1e-3], dtype=np.float32), requires_grad=True)
        # lr 设为 1.0 让裁剪效果可见；beta1=0 让 exp_avg 直接等于 update
        opt = Adafactor([p], lr=1.0, beta1=0.0, beta2=0.999, eps2=1e-3)

        # 极大梯度
        p.grad = np.array([100.0, 100.0], dtype=np.float32)

        p_before = p.data.copy()
        opt.step()

        param_change = float(np.abs(p.data - p_before).max())
        # 无裁剪时 update≈1, p 变化 ≈ 1.0（会变负数）
        # 有裁剪时 update 被 clip 到 ~1e-6, p 变化 << 0.1
        assert param_change < 0.1, (
            f"裁剪未生效：参数变化 {param_change} 应远小于无裁剪时的 ~1.0"
        )
        # 进一步验证裁剪确实把 update 压到很小
        assert param_change > 0, "参数应有微小变化（裁剪不是完全置零）"
