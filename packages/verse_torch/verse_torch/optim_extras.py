"""扩展优化器：Lion + Adafactor。

参考论文：
- Lion: https://arxiv.org/abs/2302.06675
- Adafactor: https://arxiv.org/abs/1804.04235

设计要点：
- 继承 ``optim.Optimizer`` 基类，复用 zero_grad / param_groups / state 机制
- 与 ``LRScheduler`` 兼容：在 ``__init__`` 中设置 ``self.lr``，调度器更新 ``optimizer.lr`` 后即生效
- 依赖最小化：仅 numpy + 标准库
"""

from __future__ import annotations

import numpy as np

from .optim import Optimizer


class Lion(Optimizer):
    """Lion 优化器（Lion-eats-AdamW）。

    论文: https://arxiv.org/abs/2302.06675

    特点：
    - 无二阶矩，节省约 50% 优化器状态内存
    - 用 ``sign(m·β1 + g·(1-β1))`` 作为更新方向
    - 比 AdamW 通常更省内存、效果相当或更好

    更新规则（解耦 weight decay，与 AdamW 一致）::

        update = m * β1 + g * (1 - β1)
        p = p - lr * sign(update)
        if weight_decay != 0:
            p = p - lr * weight_decay * p
        m = m * β2 + g * (1 - β2)

    Args:
        params: 模型参数或参数组列表
        lr: 学习率（默认 1e-4，比 AdamW 小 3-10x）
        betas: (β1, β2) 动量衰减系数（默认 (0.9, 0.99)）
        weight_decay: 权重衰减（默认 0.1）
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # 与 AdamW / SGD 一致：保留 self.lr 以兼容 LRScheduler 基类
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        # 全局步数（用于 bias correction / 日志）
        self.t = 0

    def step(self):
        self.t += 1
        for group in self.param_groups:
            # betas / weight_decay 支持参数组级覆盖（如 no_decay 组）
            beta1, beta2 = group.get("betas", (self.beta1, self.beta2))
            wd = group.get("weight_decay", self.weight_decay)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state.get(id(p), {})
                if "exp_avg" not in state:
                    state["exp_avg"] = np.zeros_like(p.data)
                m = state["exp_avg"]
                # Lion 更新：sign(m·β1 + g·(1-β1))
                update = m * beta1 + g * (1.0 - beta1)
                p.data = p.data - self.lr * np.sign(update)
                # 解耦权重衰减（decoupled，类似 AdamW）
                if wd != 0:
                    p.data = p.data - self.lr * wd * p.data
                # 更新动量（用 β2，区别于 Adam 的 β1）
                # 用 in-place 写回，保持 state["exp_avg"] 是同一个 ndarray 对象
                m[...] = m * beta2 + g * (1.0 - beta2)
                self.state[id(p)] = state


class Adafactor(Optimizer):
    """Adafactor 优化器（factored 二阶矩）。

    论文: https://arxiv.org/abs/1804.04235

    特点：
    - 用行/列统计近似二阶矩，节省内存（参数矩阵 W (m,n) 二阶矩从 O(mn) 降到 O(m+n)）
    - 适合大模型训练（T5/Flan-T5 等使用）
    - clipping + 衰减更新

    Args:
        params: 模型参数
        lr: 学习率（None 时使用 1e-3 作为默认值；PyTorch 原版用相对学习率，
            此处简化为绝对值以与 LRScheduler 兼容）
        beta1: 动量衰减（默认 0.9）
        beta2: RMS 衰减（默认 0.999）
        eps1: 防止除零（默认 1e-30）
        eps2: clipping 阈值（默认 1e-3）
        weight_decay: 权重衰减（默认 0）
    """

    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999,
                 eps1=1e-30, eps2=1e-3, weight_decay=0.0):
        # lr=None 时默认 1e-3（保持与 LRScheduler 兼容）
        effective_lr = lr if lr is not None else 1e-3
        defaults = dict(lr=effective_lr, beta1=beta1, beta2=beta2,
                        eps1=eps1, eps2=eps2, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = effective_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        # 全局步数
        self.t = 0

    def step(self):
        self.t += 1
        for group in self.param_groups:
            lr = self.lr  # 与 LRScheduler 兼容：从 self.lr 读取（忽略 per-group lr）
            beta1 = group.get("beta1", self.beta1)
            beta2 = group.get("beta2", self.beta2)
            eps1 = group.get("eps1", self.eps1)
            eps2 = group.get("eps2", self.eps2)
            wd = group.get("weight_decay", self.weight_decay)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state.get(id(p), {})
                # 1D 参数（bias/norm）：用普通 AdaGrad 风格的二阶矩
                if g.ndim == 1:
                    if "v" not in state:
                        state["v"] = np.zeros_like(g)
                    state["v"] = state["v"] * beta2 + (1.0 - beta2) * (g * g)
                    v_hat = state["v"] / (1.0 - beta2 ** self.t)
                    update = g / (np.sqrt(v_hat) + np.sqrt(eps1))
                else:
                    # 2D+ 参数：factored 二阶矩（row / col 统计）
                    # 高维 g 先 reshape 到 2D，计算完再 reshape 回来
                    original_shape = g.shape
                    if g.ndim > 2:
                        g_2d = g.reshape(g.shape[0], -1)
                    else:
                        g_2d = g
                    if "row" not in state:
                        state["row"] = np.zeros(g_2d.shape[0], dtype=g.dtype)
                        state["col"] = np.zeros(g_2d.shape[1], dtype=g.dtype)
                    row = state["row"]
                    col = state["col"]
                    row_new = row * beta2 + (1.0 - beta2) * np.sum(g_2d * g_2d, axis=1)
                    col_new = col * beta2 + (1.0 - beta2) * np.sum(g_2d * g_2d, axis=0)
                    state["row"] = row_new
                    state["col"] = col_new
                    # 估计 RMS：用 outer(row, col) / max(...) 近似二阶矩
                    # 注意：保持数值稳定，max 中混入 eps1 防止全 0 时除零
                    denom_max = max(float(np.max(row_new)), float(np.max(col_new)), eps1)
                    v_hat = np.outer(row_new, col_new) / denom_max
                    update_2d = g_2d / (np.sqrt(v_hat) + np.sqrt(eps1))
                    # reshape 回原始形状
                    update = update_2d.reshape(original_shape)
                # 更新裁剪（trust ratio clipping）
                update_rms = float(np.sqrt(np.mean(update * update)))
                param_rms = float(np.sqrt(np.mean(p.data * p.data)))
                clip_threshold = eps2 * param_rms + eps1
                if update_rms > clip_threshold:
                    update = update * (clip_threshold / (update_rms + eps1))
                # 动量
                if "exp_avg" not in state:
                    state["exp_avg"] = np.zeros_like(g)
                state["exp_avg"] = beta1 * state["exp_avg"] + (1.0 - beta1) * update
                # 保持 p.data 的 dtype 不变（避免 float64 污染）
                p.data = (p.data - lr * state["exp_avg"]).astype(p.data.dtype, copy=False)
                # 解耦权重衰减
                if wd != 0:
                    p.data = (p.data - lr * wd * p.data).astype(p.data.dtype, copy=False)
                self.state[id(p)] = state


__all__ = ["Lion", "Adafactor"]
