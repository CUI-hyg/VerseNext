"""VerseTorch: 优化器与学习率调度器。

设计参考 PyTorch `torch.optim`：
- Optimizer 基类提供 zero_grad() 与 step() 接口
- SGD / Adam / AdamW / NAdamW / RMSProp 实现标准更新规则
- LRScheduler 基类与 StepLR / ExponentialLR / CosineAnnealingLR

GPU 路径委托
============
当 ``Tensor.data`` 为 ``torch.Tensor``（GPU/NPU 路径）时，优化器更新
自动切换为 torch 原生算子（``torch.zeros_like`` / ``torch.sqrt`` 等），
以保证 dtype 与 device 一致；CPU 路径继续使用 NumPy 实现。
"""

from __future__ import annotations

import math
import numpy as np

from .tensor import Tensor, _is_torch_data
from .device import get_torch_module

# 模块级缓存 torch 模块（无 torch 时为 None）
_TORCH = get_torch_module()


# ---------------------------------------------------------------------------
# 优化器 (Task 1.10)
# ---------------------------------------------------------------------------


class Optimizer:
    """优化器基类。

    支持两种 params 传入方式（与 PyTorch 对齐）：
    1. 扁平参数：``Module`` / ``list[Tensor]`` / generator —— 全部用 defaults
    2. 参数组：``list[dict]``，每个 dict 含 ``"params"`` 与可覆盖的超参
       （如 ``{"params": [...], "weight_decay": 0.0}``）。
    """

    def __init__(self, params, defaults: dict):
        if hasattr(params, "parameters"):
            flat = list(params.parameters())
            self.param_groups = [{"params": flat, **dict(defaults)}]
        else:
            items = list(params)
            if items and isinstance(items[0], dict):
                # 参数组：每组合并 defaults 后用组内值覆盖
                self.param_groups = []
                for g in items:
                    group = dict(defaults)
                    group.update(g)
                    group["params"] = list(g["params"])
                    self.param_groups.append(group)
            else:
                self.param_groups = [{"params": items, **dict(defaults)}]
        # 扁平视图，保持向后兼容（zero_grad / 旧 step 逻辑仍可用）
        self.params = [p for g in self.param_groups for p in g["params"]]
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
        for group in self.param_groups:
            wd = group.get("weight_decay", self.weight_decay)
            for p in group["params"]:
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
                if wd != 0:
                    p.data = p.data * (1.0 - self.lr * wd)


class NAdamW(Optimizer):
    """NAdamW 优化器（Nesterov-accelerated AdamW）。

    在 AdamW 基础上引入 Nesterov 动量前瞻：用 ``beta1 * m_t + (1 - beta1) * g``
    作为"Nesterov 一阶矩"再做 bias correction，使更新方向先看一步再校正，
    收敛通常比 AdamW 更快、更稳。

    更新规则（解耦 weight decay）::

        m_t   = beta1 * m + (1 - beta1) * g
        v_t   = beta2 * v + (1 - beta2) * g^2
        m_hat = (beta1 * m_t + (1 - beta1) * g) / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p    -= lr * m_hat / (sqrt(v_hat) + eps)
        p    *= (1 - lr * weight_decay)   # 解耦 weight decay

    Args:
        params: 可迭代的参数 / Module / 参数组 dict 列表
        lr: 学习率（默认 1e-3）
        betas: (beta1, beta2) 一阶/二阶矩衰减系数（默认 (0.9, 0.999)）
        eps: 分母稳定常数（默认 1e-8）
        weight_decay: 解耦权重衰减系数（默认 0.01）

    GPU 路径委托
    ------------
    当 ``p.data`` 为 ``torch.Tensor`` 时，momentum buffer 与算子均用
    torch 原生实现（``torch.zeros_like`` / ``torch.sqrt``），保持 device
    与 dtype 一致；否则退回 NumPy 路径。
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for group in self.param_groups:
            wd = group.get("weight_decay", self.weight_decay)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                is_torch = _is_torch_data(p.data)
                # 选择后端工具函数
                if is_torch:
                    _sqrt = _TORCH.sqrt
                    _zeros_like = _TORCH.zeros_like
                else:
                    _sqrt = np.sqrt
                    _zeros_like = lambda x: np.zeros_like(x, dtype=np.float32)
                key = id(p)
                state = self.state.get(key, None)
                if state is None:
                    state = {
                        "m": _zeros_like(p.data),
                        "v": _zeros_like(p.data),
                    }
                    self.state[key] = state
                m = state["m"]
                v = state["v"]
                # 一阶/二阶矩更新
                m = self.beta1 * m + (1.0 - self.beta1) * g
                v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
                state["m"] = m
                state["v"] = v
                # Nesterov 前瞻一阶矩 + bias correction
                m_nesterov = self.beta1 * m + (1.0 - self.beta1) * g
                m_hat = m_nesterov / bc1
                v_hat = v / bc2
                # 参数更新
                update = self.lr * m_hat / (_sqrt(v_hat) + self.eps)
                p.data = p.data - update
                # 解耦 weight decay
                if wd != 0:
                    p.data = p.data * (1.0 - self.lr * wd)


class RMSProp(Optimizer):
    """RMSProp 优化器。

    用滑动二阶矩自适应学习率，对非平稳目标（如 RNN / RL）较稳定。

    更新规则::

        v_t   = alpha * v + (1 - alpha) * g^2
        p    -= lr * g / (sqrt(v_t) + eps)

    可选 momentum（默认 0）：在自适应步长之上再加标准动量::

        buf   = momentum * buf + g
        p    -= lr * buf / (sqrt(v_t) + eps)

    可选 centered（默认 False）：用方差而非二阶矩归一化（与 PyTorch 对齐）::

        avg_g = alpha * avg_g + (1 - alpha) * g
        v_t   = alpha * v + (1 - alpha) * g^2
        p    -= lr * g / (sqrt(v_t - avg_g^2) + eps)

    Args:
        params: 可迭代的参数 / Module / 参数组 dict 列表
        lr: 学习率（默认 1e-2）
        alpha: 二阶矩滑动平均系数（默认 0.99）
        eps: 分母稳定常数（默认 1e-8）
        weight_decay: L2 权重衰减（耦合到梯度，默认 0.0）
        momentum: 动量系数（默认 0.0）
        centered: 是否使用 centered RMSProp（默认 False）

    GPU 路径委托
    ------------
    当 ``p.data`` 为 ``torch.Tensor`` 时，buffer 与算子均用 torch 原生实现。
    """

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 momentum: float = 0.0, centered: bool = False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, centered=centered)
        super().__init__(params, defaults)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

    def step(self):
        for group in self.param_groups:
            wd = group.get("weight_decay", self.weight_decay)
            mom = group.get("momentum", self.momentum)
            centered = group.get("centered", self.centered)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p.data
                is_torch = _is_torch_data(p.data)
                if is_torch:
                    _sqrt = _TORCH.sqrt
                    _zeros_like = _TORCH.zeros_like
                else:
                    _sqrt = np.sqrt
                    _zeros_like = lambda x: np.zeros_like(x, dtype=np.float32)
                key = id(p)
                state = self.state.get(key, None)
                if state is None:
                    state = {"v": _zeros_like(p.data)}
                    if mom != 0:
                        state["buf"] = _zeros_like(p.data)
                    if centered:
                        state["avg"] = _zeros_like(p.data)
                    self.state[key] = state
                v = state["v"]
                v = self.alpha * v + (1.0 - self.alpha) * (g * g)
                state["v"] = v
                if centered:
                    avg = state["avg"]
                    avg = self.alpha * avg + (1.0 - self.alpha) * g
                    state["avg"] = avg
                    denom = _sqrt(v - avg * avg) + self.eps
                else:
                    denom = _sqrt(v) + self.eps
                if mom != 0:
                    buf = state["buf"]
                    buf = mom * buf + g
                    state["buf"] = buf
                    update = self.lr * buf / denom
                else:
                    update = self.lr * g / denom
                p.data = p.data - update


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


class LambdaLR(LRScheduler):
    """Lambda 学习率调度器。

    通过自定义函数 `lr_lambda(step)` 返回学习率的乘法因子：

        lr = base_lr * lr_lambda(last_epoch)

    常用于 warmup + cosine decay 等自定义调度。

    Args:
        optimizer: 已构造的 Optimizer（持有 base_lr）
        lr_lambda: 接受 step（int）返回 float 的函数
        last_epoch: 起始 epoch（-1 表示初始化前）

    示例:
        >>> opt = Adam(model.parameters(), lr=1e-3)
        >>> sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=50, total_steps=1000))
    """

    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        # 注意：必须在调用 super().__init__() 之前设置 lr_lambda，
        # 因为基类 __init__ 末尾会调用 self.step()，进而调用 get_lr()。
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return self.base_lr
        factor = float(self.lr_lambda(self.last_epoch))
        return self.base_lr * factor


def warmup_cosine_lr(warmup_steps: int, total_steps: int):
    """Warmup + Cosine Decay 的 lr_lambda 工厂函数。

    - step < warmup_steps: 线性 warmup，lr 从 0 升至 base_lr
    - warmup_steps <= step <= total_steps: 余弦衰减，从 base_lr 降至 0

    返回一个闭包，可传给 `LambdaLR`：

        >>> opt = Adam(model.parameters(), lr=1e-3)
        >>> sched = LambdaLR(opt, warmup_cosine_lr(warmup_steps=50, total_steps=1000))

    Args:
        warmup_steps: 预热步数（达到 base_lr 时的 step）
        total_steps: 总训练步数（衰减到 0 时的 step）

    Returns:
        lr_lambda: (step: int) -> float
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # 线性 warmup：step=0 -> 0，step=warmup_steps -> 1.0
            return step / max(1, warmup_steps)
        # 余弦衰减阶段
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # 限制 progress 到 [0, 1]，避免 step 超过 total_steps 时出现反弹
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda
