"""Task 3.9b: verse_torch.scheduler_extras 单元测试。

覆盖 OneCycleLR、ReduceLROnPlateau、CosineRestartsLR 三个调度器：
1. test_onecycle_warmup: 前 25% 步 lr 升到 max_lr
2. test_onecycle_anneal: 后 75% 步 lr 余弦退火
3. test_onecycle_initial_lr: 初始 lr = max_lr / div_factor
4. test_onecycle_final_lr: 末尾 lr 接近 max_lr / (div_factor * final_div_factor)
5. test_reduce_lr_on_plateau_decrease: val_loss 不下降时 lr 衰减
6. test_reduce_lr_on_plateau_improve: val_loss 下降时 lr 不变
7. test_reduce_lr_on_plateau_min_lr: 不低于 min_lr
8. test_cosine_restarts_cycle: 第一个周期结束后重新升温

运行方式：
    cd /workspace && python -m pytest tests/test_scheduler_extras.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor, SGD, OneCycleLR, ReduceLROnPlateau, CosineRestartsLR


def _make_opt(lr: float = 0.1):
    """构造一个带单个可训练参数的 SGD 优化器（用于调度器测试）。"""
    p = Tensor.zeros(1, requires_grad=True)
    return SGD([p], lr=lr)


# ---------------------------------------------------------------------------
# OneCycleLR 测试
# ---------------------------------------------------------------------------


class TestOneCycleLR:
    """OneCycleLR（super-convergence）：warmup + cosine anneal。"""

    def test_onecycle_initial_lr(self):
        """初始 lr = max_lr / div_factor。"""
        max_lr = 0.1
        div_factor = 25.0
        opt = _make_opt(lr=max_lr / div_factor)  # base_lr 记录此值
        sched = OneCycleLR(
            opt, max_lr=max_lr, total_steps=100,
            pct_start=0.25, div_factor=div_factor, final_div_factor=1e4,
        )
        # 构造后基类已调用一次 step()（last_epoch: -1 -> 0）
        # get_lr(0): warmup progress=0 → lr = initial_lr = max_lr / div_factor
        expected_initial = max_lr / div_factor
        assert abs(opt.lr - expected_initial) < 1e-7, (
            f"初始 lr {opt.lr} 不等于 max_lr/div_factor {expected_initial}"
        )

    def test_onecycle_warmup(self):
        """前 25% 步 lr 单调递增到 max_lr。"""
        max_lr = 0.1
        opt = _make_opt(lr=max_lr / 25.0)
        sched = OneCycleLR(opt, max_lr=max_lr, total_steps=100, pct_start=0.25)

        lrs = [opt.lr]
        # warmup 阶段：step 25 次（last_epoch 0 -> 25）
        for _ in range(25):
            sched.step()
            lrs.append(opt.lr)

        # step 25 时 last_epoch=25，进入 anneal 起点，lr = max_lr
        assert abs(opt.lr - max_lr) < 1e-6, (
            f"step 25 时 lr {opt.lr} 未达到 max_lr {max_lr}"
        )
        # warmup 阶段 lr 应单调递增
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-9, (
                f"warmup 阶段 lr 在 step {i} 时未递增：{lrs[i]} < {lrs[i-1]}"
            )

    def test_onecycle_anneal(self):
        """后 75% 步 lr 余弦退火（单调递减）。"""
        max_lr = 0.1
        opt = _make_opt(lr=max_lr / 25.0)
        sched = OneCycleLR(opt, max_lr=max_lr, total_steps=100, pct_start=0.25)

        # 先走完 warmup（step 25 次，到达 anneal 起点）
        for _ in range(25):
            sched.step()
        peak_lr = opt.lr
        # peak_lr 应等于 max_lr
        assert abs(peak_lr - max_lr) < 1e-6

        # anneal 阶段：step 75 次（last_epoch 25 -> 100）
        lrs = [opt.lr]
        for _ in range(75):
            sched.step()
            lrs.append(opt.lr)

        # anneal 阶段 lr 应单调递减（允许 1e-9 浮点误差）
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-9, (
                f"anneal 阶段 lr 在 step {i} 时未递减：{lrs[i]} > {lrs[i-1]}"
            )
        # 末尾 lr 应远小于 peak
        assert opt.lr < peak_lr * 0.01, (
            f"末尾 lr {opt.lr} 应远小于 peak {peak_lr}"
        )

    def test_onecycle_final_lr(self):
        """末尾 lr 接近 max_lr / (div_factor * final_div_factor)。"""
        max_lr = 0.1
        div_factor = 25.0
        final_div_factor = 1e4
        opt = _make_opt(lr=max_lr / div_factor)
        sched = OneCycleLR(
            opt, max_lr=max_lr, total_steps=100,
            pct_start=0.25, div_factor=div_factor, final_div_factor=final_div_factor,
        )
        # 构造后 last_epoch=0，再 step 100 次到 last_epoch=100
        for _ in range(100):
            sched.step()
        # last_epoch=100 时 anneal progress=1.0 → lr = final_lr
        final_lr = max_lr / (div_factor * final_div_factor)
        assert abs(opt.lr - final_lr) < 1e-7, (
            f"末尾 lr {opt.lr} 不接近 final_lr {final_lr}"
        )


# ---------------------------------------------------------------------------
# ReduceLROnPlateau 测试
# ---------------------------------------------------------------------------


class TestReduceLROnPlateau:
    """按 val_loss 衰减 lr。"""

    def test_reduce_lr_on_plateau_decrease(self):
        """val_loss 不下降时 lr 衰减。"""
        opt = _make_opt(lr=0.1)
        sched = ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=2, threshold=1e-4
        )

        initial_lr = opt.lr
        # 第 1 次：best=inf → 任何有限值都视为改善
        sched.step(1.0)
        assert abs(opt.lr - initial_lr) < 1e-12, "首次改善不应衰减 lr"

        # 第 2 次：相同 val_loss → 不改善，num_bad=1
        sched.step(1.0)
        assert abs(opt.lr - initial_lr) < 1e-12, "num_bad=1 未达 patience 不应衰减"

        # 第 3 次：相同 val_loss → 不改善，num_bad=2 >= patience=2 → 衰减
        sched.step(1.0)
        expected_lr = initial_lr * 0.5
        assert abs(opt.lr - expected_lr) < 1e-9, (
            f"lr 应衰减到 {expected_lr}, got {opt.lr}"
        )

    def test_reduce_lr_on_plateau_improve(self):
        """val_loss 持续下降时 lr 不变。"""
        opt = _make_opt(lr=0.1)
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

        initial_lr = opt.lr
        # 传入持续递减的 val_loss
        sched.step(1.0)
        sched.step(0.5)
        sched.step(0.2)
        sched.step(0.1)
        sched.step(0.01)

        assert abs(opt.lr - initial_lr) < 1e-12, (
            f"val_loss 持续下降时 lr 不应衰减，仍应为 {initial_lr}, got {opt.lr}"
        )

    def test_reduce_lr_on_plateau_min_lr(self):
        """多次衰减后 lr 不低于 min_lr。"""
        opt = _make_opt(lr=0.1)
        min_lr = 1e-3
        # patience=1 让每次不改善都触发衰减
        sched = ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=1, min_lr=min_lr
        )

        # 第 1 次改善（best=1.0）
        sched.step(1.0)
        # 连续 20 次不改善，每次都触发衰减
        for _ in range(20):
            sched.step(1.0)

        # lr 应稳定在 min_lr，不低于
        assert opt.lr >= min_lr - 1e-12, (
            f"lr {opt.lr} 不应低于 min_lr {min_lr}"
        )
        assert abs(opt.lr - min_lr) < 1e-12, (
            f"lr 应稳定在 min_lr {min_lr}, got {opt.lr}"
        )


# ---------------------------------------------------------------------------
# CosineRestartsLR 测试
# ---------------------------------------------------------------------------


class TestCosineRestartsLR:
    """带 warm restarts 的余弦退火（SGDR）。"""

    def test_cosine_restarts_cycle(self):
        """第一个周期结束后重新升温到 base_lr。"""
        base_lr = 0.1
        opt = _make_opt(lr=base_lr)
        # T_0=10：每 10 步一个周期
        sched = CosineRestartsLR(opt, T_0=10, T_mult=1, eta_min=0.0)

        # 构造后基类已 step 一次（last_epoch: -1 -> 0, T_cur: 0 -> 1）
        # get_lr 用 T_cur=0 计算 → lr = base_lr（cos(0)=1）
        assert abs(opt.lr - base_lr) < 1e-6, (
            f"初始 lr {opt.lr} 应等于 base_lr {base_lr}"
        )

        # step 5 次（T_cur 1 -> 6），lr 应退火到中点附近
        for _ in range(5):
            sched.step()
        mid_lr = opt.lr
        # step 5 时 T_cur=5, progress=0.5, lr = 0.5*base_lr*(1+cos(pi/2)) = 0.5*base_lr
        expected_mid = 0.5 * base_lr * (1.0 + math.cos(math.pi * 0.5))
        assert abs(mid_lr - expected_mid) < 1e-6, (
            f"step 5 时 lr {mid_lr} 不等于期望值 {expected_mid}"
        )
        assert mid_lr < base_lr, f"退火后 lr {mid_lr} 应小于 base_lr {base_lr}"

        # 继续 step 5 次（共 10 次），T_cur 到 10 触发 restart
        for _ in range(5):
            sched.step()
        restart_lr = opt.lr
        # restart 后 T_cur 重置为 0, lr 重新升到 base_lr
        assert abs(restart_lr - base_lr) < 1e-6, (
            f"restart 后 lr {restart_lr} 应重新升到 base_lr {base_lr}"
        )
