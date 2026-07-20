"""VerseTorch: 损失函数。

包含常用损失：
- cross_entropy(logits, targets): 多分类 softmax 交叉熵
- binary_cross_entropy(pred, target): 二分类交叉熵（接受概率）
- binary_cross_entropy_with_logits(logits, target): 二分类（接受 logits）
- mse_loss(pred, target): 均方误差
- nll_loss(log_probs, targets): 负对数似然

所有损失返回标量 Tensor，requires_grad 自动传播。
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """多分类交叉熵损失。

    等价于 PyTorch `F.cross_entropy(logits, targets)`，内部用 log_softmax + nll_loss。

    Args:
        logits: (N, C) 未归一化的预测
        targets: (N,) int 类别索引（可传入 list / ndarray / Tensor）

    Returns:
        标量 Tensor
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets).astype(np.int64)
    N = logits.shape[0]
    # log_softmax 沿 dim=-1
    log_probs = logits.log_softmax(dim=-1)
    # 选取正确类别的 log_prob
    # 用 advanced indexing: log_probs[arange(N), targets]
    arange = np.arange(N)
    # 通过 __getitem__ 实现可微
    selected = log_probs[arange, targets]
    # 取负平均
    loss = -selected.mean()
    return loss


def nll_loss(log_probs: Tensor, targets) -> Tensor:
    """负对数似然损失。

    Args:
        log_probs: (N, C) 已经是 log 概率
        targets: (N,) int 类别索引

    Returns:
        标量 Tensor
    """
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets).astype(np.int64)
    N = log_probs.shape[0]
    arange = np.arange(N)
    selected = log_probs[arange, targets]
    return -selected.mean()


def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """二分类交叉熵，输入为概率（已经过 sigmoid）。

    L = -mean(target * log(pred) + (1 - target) * log(1 - pred))

    数值稳定：clip pred 到 [eps, 1-eps]
    """
    if not isinstance(target, Tensor):
        target = Tensor(target, requires_grad=False)
    eps = 1e-12
    # 限制 pred 范围以避免 log(0)
    # 用 clip：不可微但保持梯度方向（仅在边界处 subgradient）
    p_data = np.clip(pred.data, eps, 1.0 - eps)
    # 用 Tensor 包装 clip 后的数据，但保持计算图（用一个变换替代）
    # 这里直接用 pred 计算，但加上 eps 避免数值问题
    # 我们用 max/min 替代 clip 以保持可微：
    #   pred_safe = pred clipped to [eps, 1-eps]
    # 但简单起见，我们直接用 clip 数据构建新 Tensor，让 grad 流过原始 pred
    # 实现思路：loss = -(target * log(pred_safe) + (1-target) * log(1 - pred_safe))
    # 但 grad 需要回到 pred。这里采用一种近似：
    #   使用 pred 直接计算 log，但用 np.clip 把数据范围限制，不影响 grad 计算
    #   grad 仍是 -target/pred + (1-target)/(1-pred)
    p_safe = np.clip(pred.data, eps, 1.0 - eps)
    out_data = -(target.data * np.log(p_safe) + (1.0 - target.data) * np.log(1.0 - p_safe))
    out_data = np.asarray(out_data.mean(), dtype=np.float32)

    requires_grad = pred.requires_grad or target.requires_grad

    def _backward():
        if pred.requires_grad:
            # dL/dpred = (-target/pred + (1-target)/(1-pred)) / N
            N = pred.data.size
            grad = (-target.data / p_safe + (1.0 - target.data) / (1.0 - p_safe)) / N
            pred._accumulate_grad(out.grad * grad)
        if target.requires_grad:
            N = target.data.size
            grad = (-np.log(p_safe) + np.log(1.0 - p_safe)) / N
            target._accumulate_grad(out.grad * grad)

    out = Tensor(out_data, requires_grad=requires_grad, _children=(pred, target), _op="bce")
    if out.requires_grad:
        out._backward = _backward
    return out


def binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor:
    """二分类交叉熵，输入为 logits（更数值稳定）。

    L = -mean(target * log(sigmoid(logits)) + (1 - target) * log(1 - sigmoid(logits)))
      = mean(max(logits, 0) - logits * target + log(1 + exp(-|logits|)))

    这是 PyTorch `F.binary_cross_entropy_with_logits` 的实现。
    """
    if not isinstance(target, Tensor):
        target = Tensor(target, requires_grad=False)
    x = logits.data
    t = target.data
    # 数值稳定公式
    out_data = np.maximum(x, 0.0) - x * t + np.log1p(np.exp(-np.abs(x)))
    out_data = np.asarray(out_data.mean(), dtype=np.float32)

    requires_grad = logits.requires_grad or target.requires_grad

    def _backward():
        if logits.requires_grad:
            # dL/dx = sigmoid(x) - target，再除以 N
            N = logits.data.size
            sx = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            grad = (sx - t) / N
            logits._accumulate_grad(out.grad * grad)
        if target.requires_grad:
            N = target.data.size
            grad = -x / N
            target._accumulate_grad(out.grad * grad)

    out = Tensor(out_data, requires_grad=requires_grad, _children=(logits, target), _op="bce_logits")
    if out.requires_grad:
        out._backward = _backward
    return out


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """均方误差损失: L = mean((pred - target)^2)

    Args:
        pred: 任意形状的预测 Tensor
        target: 同形状的真实值 Tensor（或 ndarray/list）
    """
    if not isinstance(target, Tensor):
        target = Tensor(target, requires_grad=False)
    diff = pred - target
    return (diff * diff).mean()


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 损失: L = mean(|pred - target|)"""
    if not isinstance(target, Tensor):
        target = Tensor(target, requires_grad=False)
    diff = pred - target
    # |x| 的反向是 sign(x)
    out_data = np.abs(diff.data).mean()
    out_data = np.asarray(out_data, dtype=np.float32)

    requires_grad = pred.requires_grad or target.requires_grad

    def _backward():
        if pred.requires_grad:
            N = pred.data.size
            grad = np.sign(diff.data) / N
            pred._accumulate_grad(out.grad * grad)
        if target.requires_grad:
            N = target.data.size
            grad = -np.sign(diff.data) / N
            target._accumulate_grad(out.grad * grad)

    out = Tensor(out_data, requires_grad=requires_grad, _children=(pred, target), _op="l1")
    if out.requires_grad:
        out._backward = _backward
    return out


def kl_div_loss(log_probs: Tensor, target_probs: Tensor) -> Tensor:
    """KL 散度损失: L = sum(target * (log(target) - log_probs))

    Args:
        log_probs: log 概率（输入）
        target_probs: 目标概率分布
    """
    if not isinstance(target_probs, Tensor):
        target_probs = Tensor(target_probs, requires_grad=False)
    # 在 target_probs=0 处避免 log(0)
    safe_t = np.where(target_probs.data > 0, target_probs.data, 1.0)
    log_t = np.log(safe_t)
    out_data = (target_probs.data * (log_t - log_probs.data)).sum(axis=-1).mean()
    out_data = np.asarray(out_data, dtype=np.float32)

    requires_grad = log_probs.requires_grad or target_probs.requires_grad

    def _backward():
        if log_probs.requires_grad:
            N = log_probs.shape[0] if log_probs.ndim >= 2 else 1
            grad = -target_probs.data / N
            log_probs._accumulate_grad(out.grad * grad)
        if target_probs.requires_grad:
            N = target_probs.shape[0] if target_probs.ndim >= 2 else 1
            grad = (log_t - log_probs.data + 1.0 - safe_t) / N
            target_probs._accumulate_grad(out.grad * grad)

    out = Tensor(out_data, requires_grad=requires_grad, _children=(log_probs, target_probs), _op="kl_div")
    if out.requires_grad:
        out._backward = _backward
    return out
