"""扩展 LR 调度器：OneCycleLR + ReduceLROnPlateau + CosineRestartsLR。

设计要点：
- ``OneCycleLR`` / ``CosineRestartsLR`` 继承 ``optim.LRScheduler`` 基类
  （基类在 ``__init__`` 末尾自动调用一次 ``step()``，使 ``last_epoch`` 从 -1 变为 0）
- ``ReduceLROnPlateau`` 不继承基类：因为它需要外部传入 ``metric``（如 val_loss），
  与基类的无参 ``step()`` 接口不兼容；直接操作 ``optimizer.param_groups`` 中每个 group 的 lr
- 所有 ``get_lr()`` 返回标量（与基类 ``self.optimizer.lr = new_lr`` 的赋值语义一致）
- 依赖最小化：仅 numpy + 标准库
"""

from __future__ import annotations

import math

from .optim import LRScheduler


class OneCycleLR(LRScheduler):
    """OneCycleLR（super-convergence）。

    论文: https://arxiv.org/abs/1708.07120

    调度曲线：
    - 前 ``pct_start`` 比例的步：从 ``max_lr / div_factor`` 线性升到 ``max_lr``
    - 后 ``1 - pct_start`` 比例的步：从 ``max_lr`` 余弦退火到 ``max_lr / (div_factor * final_div_factor)``

    Args:
        optimizer: 优化器（需有 ``lr`` 属性，由基类 ``base_lr`` 记录）
        max_lr: 峰值学习率
        total_steps: 总步数
        pct_start: 升温阶段占比（默认 0.25）
        div_factor: 初始 lr = ``max_lr / div_factor``（默认 25）
        final_div_factor: 末尾 lr = ``max_lr / (div_factor * final_div_factor)``（默认 1e4）
    """

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.25,
                 div_factor=25.0, final_div_factor=1e4):
        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = self.last_epoch
        warmup_steps = self.total_steps * self.pct_start
        if step < warmup_steps:
            # 升温阶段：从 initial_lr 线性升到 max_lr
            # 注意 warmup_steps 可能为 0（pct_start=0），用 max(...) 防止除零
            progress = step / max(warmup_steps, 1.0)
            initial_lr = self.max_lr / self.div_factor
            return initial_lr + (self.max_lr - initial_lr) * progress
        else:
            # 退火阶段：从 max_lr 余弦退火到 final_lr
            anneal_steps = self.total_steps * (1.0 - self.pct_start)
            progress = (step - warmup_steps) / max(anneal_steps, 1.0)
            # 限制到 [0, 1]，避免 step 超过 total_steps 时出现反弹
            progress = max(0.0, min(1.0, progress))
            final_lr = self.max_lr / (self.div_factor * self.final_div_factor)
            return final_lr + (self.max_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ReduceLROnPlateau:
    """按 val_loss 衰减 lr（不继承 ``LRScheduler``，因为需要外部 metric 输入）。

    当监控指标（如 ``val_loss``）连续 ``patience`` 次 epoch 无显著改善时，
    将每个 param_group 的 lr 乘以 ``factor``（不低于 ``min_lr``）。

    Args:
        optimizer: 优化器
        mode: ``'min'`` 或 ``'max'``（默认 ``'min'``，监控 val_loss）
        factor: 衰减系数（默认 0.1）
        patience: 多少 epoch 无改善后衰减（默认 10）
        min_lr: 最小 lr 下界（默认 0）
        threshold: 改善阈值（默认 1e-4，相对改善量）

    用法:
        >>> scheduler = ReduceLROnPlateau(opt, mode='min', patience=5)
        >>> for epoch in range(n_epochs):
        ...     train_one_epoch()
        ...     val_loss = evaluate()
        ...     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 min_lr=0, threshold=1e-4):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode 必须为 'min' 或 'max'，got {mode!r}")
        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.threshold = float(threshold)
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.num_bad_epochs = 0

    def step(self, metric):
        """传入当前 epoch 的 metric（如 val_loss），按需衰减 lr。"""
        current = float(metric)
        if self.mode == 'min':
            # 改善条件：current < best * (1 - threshold)
            # 当 best=inf 时，inf * (1-threshold) = inf，所以任何有限值都视为改善
            improved = current < self.best * (1.0 - self.threshold)
        else:
            improved = current > self.best * (1.0 + self.threshold)
        if improved:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                for group in self.optimizer.param_groups:
                    new_lr = max(group['lr'] * self.factor, self.min_lr)
                    group['lr'] = new_lr
                # 同步 self.lr 以保持与 LRScheduler 基类的兼容性
                if hasattr(self.optimizer, 'lr'):
                    self.optimizer.lr = max(
                        self.optimizer.lr * self.factor, self.min_lr
                    )
                self.num_bad_epochs = 0


class CosineRestartsLR(LRScheduler):
    """带 warm restarts 的余弦退火（SGDR）。

    论文: https://arxiv.org/abs/1608.03983

    每个周期内 lr 从 ``base_lr`` 余弦退火到 ``eta_min``；周期结束后重新升温到 ``base_lr``。
    周期长度按 ``T_mult`` 倍增：第 k 个周期长度 = ``T_0 * T_mult^k``。

    Args:
        optimizer: 优化器
        T_0: 第一个周期长度（步数）
        T_mult: 每个周期长度乘以 T_mult（默认 1，即所有周期等长）
        eta_min: 最小 lr（默认 0）
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        # 当前周期长度与已用步数
        self.T_i = self.T_0
        self.T_cur = 0
        self.cycle = 0
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        # 检查是否需要进入下一周期
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            # T_mult=0 时会陷入 0 长度周期，做最小保护
            if self.T_i <= 0:
                self.T_i = 1
        progress = self.T_cur / max(self.T_i, 1)
        # 余弦退火：从 base_lr 降到 eta_min
        return self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self):
        # 重写 step 以维护 T_cur 计数器
        # 注意：基类 __init__ 末尾会调用一次 step()，此时 last_epoch=-1 -> step() 中
        # 先 last_epoch += 1 -> last_epoch=0，然后 get_lr() 用 T_cur=0 计算第一个 lr
        result = super().step()
        # 推进当前周期内的步数计数
        self.T_cur += 1
        return result


__all__ = ["OneCycleLR", "ReduceLROnPlateau", "CosineRestartsLR"]
