"""VerseTorch: 优化器与学习率调度器。

设计参考 PyTorch `torch.optim`：
- Optimizer 基类提供 zero_grad() 与 step() 接口
- SGD / Adam / AdamW 实现标准更新规则
- LRScheduler 基类与 StepLR / ExponentialLR / CosineAnnealingLR
"""

from __future__ import annotations

import math
import numpy as np

from .tensor import Tensor


# ---------------------------------------------------------------------------
# 优化器 (Task 1.10)
# ---------------------------------------------------------------------------


class Optimizer:
    """优化器基类。"""

    def __init__(self, params, defaults: dict):
        # params 可以是 generator / list / Module
        if hasattr(params, "parameters"):
            params = list(params.parameters())
        else:
            params = list(params)
        self.params = params
        self.defaults = defaults
        self.state = {}  # 每个参数的状态（如 momentum buffer）

    def zero_grad(self):
        """清空所有参数的梯度。"""
        for p in self.params:
            p.grad = None

    def step(self):
        """执行一步参数更新（子类实现）。"""
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降（带 momentum, weight_decay, dampening, nesterov）。

    更新规则：
        g_t = grad + weight_decay * p
        if dampening != 0:
            buf = dampening * buf + (1 - dampening) * g_t
        else:
            buf = momentum * buf + g_t
        if nesterov:
            g_t = g_t + momentum * buf
        else:
            g_t = buf
        p -= lr * g_t
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening.")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0:
                key = id(p)
                buf = self.state.get(key, None)
                if buf is None:
                    buf = g.copy()
                    self.state[key] = buf
                else:
                    buf = self.momentum * buf + (1.0 - self.dampening) * g
                    self.state[key] = buf
                if self.nesterov:
                    g = g + self.momentum * buf
                else:
                    g = buf
            # 更新参数
            p.data = p.data - self.lr * g


class Adam(Optimizer):
    """Adam 优化器。

    更新规则：
        m_t = beta1 * m_{t-1} + (1 - beta1) * g
        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # t 是全局步数
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0:
                # L2 正则化（耦合 weight decay）
                g = g + self.weight_decay * p.data
            key = id(p)
            state = self.state.get(key, None)
            if state is None:
                state = {
                    "m": np.zeros_like(p.data, dtype=np.float32),
                    "v": np.zeros_like(p.data, dtype=np.float32),
                }
                self.state[key] = state
            m = state["m"]
            v = state["v"]
            # 更新一阶矩与二阶矩
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            state["m"] = m
            state["v"] = v
            # bias correction
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            # 参数更新
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW 优化器（解耦 weight decay）。

    与 Adam 的区别：weight_decay 直接作用于参数 p，而不是耦合到梯度 g。

    更新规则：
        m_t = beta1 * m + (1 - beta1) * g
        v_t = beta2 * v + (1 - beta2) * g^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            key = id(p)
            state = self.state.get(key, None)
            if state is None:
                state = {
                    "m": np.zeros_like(p.data, dtype=np.float32),
                    "v": np.zeros_like(p.data, dtype=np.float32),
                }
                self.state[key] = state
            m = state["m"]
            v = state["v"]
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            state["m"] = m
            state["v"] = v
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            # 解耦 weight decay：先做 Adam 更新，再单独乘 (1 - lr * wd)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data - update
            if self.weight_decay != 0:
                p.data = p.data * (1.0 - self.lr * self.weight_decay)


# ---------------------------------------------------------------------------
# 学习率调度器 (Task 1.11)
# ---------------------------------------------------------------------------


class LRScheduler:
    """学习率调度器基类。"""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        # 记录初始学习率
        self.base_lr = optimizer.lr
        self._last_lr = optimizer.lr
        self.step()  # 初始化第一步

    def get_lr(self):
        """子类实现：根据当前 epoch 计算新的学习率。"""
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr
        self._last_lr = new_lr
        return new_lr


class StepLR(LRScheduler):
    """每 step_size 步将学习率乘以 gamma。

    例如: StepLR(opt, step_size=10, gamma=0.1) 每 10 步 lr *= 0.1
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return self.base_lr
        # 每 step_size 步衰减
        decay = self.gamma ** (self.last_epoch // self.step_size)
        return self.base_lr * decay


class ExponentialLR(LRScheduler):
    """每步将学习率乘以 gamma。"""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.99, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return self.base_lr
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LRScheduler):
    """余弦退火学习率。

    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * T / T_max))

    每 T_max 步完成一个余弦周期。
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0,
                 last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return self.base_lr
        if self.last_epoch == 0:
            return self.base_lr
        if self.last_epoch > self.T_max:
            # 超过 T_max，重置周期（PyTorch 行为是 SGDR 重启）
            # 这里简化为继续余弦到 eta_min
            return self.eta_min
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * self.last_epoch / self.T_max)
        )
