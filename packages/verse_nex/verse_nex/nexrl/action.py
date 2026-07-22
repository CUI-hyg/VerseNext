"""NexAction: RL 动作抽象与采样策略。

包含：
- NexAction: 动作数据类（token_id + logprob）
- ActionSampler: 多种采样策略（ε-greedy / softmax / nucleus / top-k）
- ExplorationSchedule: 探索率衰减 schedule（linear / cosine / exponential）
- repeat_penalty: 对已生成 token 的 logits 施加惩罚
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NexAction:
    """RL 动作数据类。

    Attributes:
        token_id: 选择的 token id
        logprob: 该 token 的对数概率
    """

    token_id: int = 0
    logprob: float = 0.0


def _softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """数值稳定的 softmax（numpy 实现）。

    Args:
        logits: (..., vocab) 的 logits
        temperature: 温度参数

    Returns:
        与 logits 同形状的概率数组
    """
    scaled = logits / max(temperature, 1e-8)
    # 减去最大值防止溢出
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 log_softmax（numpy 实现）。"""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


class ActionSampler:
    """动作采样器：支持多种采样策略。

    所有方法接受 logits（numpy ndarray）并返回采样的 token id。
    """

    @staticmethod
    def epsilon_greedy(
        logits: np.ndarray,
        epsilon: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """ε-greedy 采样。

        以 ε 概率随机采样，否则选 argmax。

        Args:
            logits: (vocab,) 或 (..., vocab) 的 logits，取最后一维
            epsilon: 探索率
            rng: 随机数生成器

        Returns:
            采样的 token id
        """
        if rng is None:
            rng = np.random.default_rng()
        # 取最后一维的 logits
        if logits.ndim > 1:
            logits_1d = logits.reshape(-1)[-logits.shape[-1]:]
        else:
            logits_1d = logits
        if rng.random() < epsilon:
            return int(rng.integers(0, len(logits_1d)))
        return int(np.argmax(logits_1d))

    @staticmethod
    def softmax(
        logits: np.ndarray,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """温度缩放 softmax 采样。

        Args:
            logits: (vocab,) 的 logits
            temperature: 采样温度（>1 更平坦，<1 更尖锐）
            rng: 随机数生成器

        Returns:
            采样的 token id
        """
        if rng is None:
            rng = np.random.default_rng()
        if logits.ndim > 1:
            logits_1d = logits.reshape(-1)[-logits.shape[-1]:]
        else:
            logits_1d = logits
        probs = _softmax_np(logits_1d, temperature)
        return int(rng.choice(len(probs), p=probs))

    @staticmethod
    def nucleus(
        logits: np.ndarray,
        top_p: float = 0.9,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Nucleus (top-p) 采样。

        选择累积概率达到 top_p 的最小 token 集合，从中采样。

        Args:
            logits: (vocab,) 的 logits
            top_p: 累积概率阈值
            temperature: 采样温度
            rng: 随机数生成器

        Returns:
            采样的 token id
        """
        if rng is None:
            rng = np.random.default_rng()
        if logits.ndim > 1:
            logits_1d = logits.reshape(-1)[-logits.shape[-1]:]
        else:
            logits_1d = logits
        # 计算概率
        probs = _softmax_np(logits_1d, temperature)
        # 按概率降序排列
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        # 累积概率
        cumsum = np.cumsum(sorted_probs)
        # 找到累积概率 >= top_p 的最小集合
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        cutoff = min(cutoff, len(sorted_probs))
        # 在 top-p 集合内采样
        nucleus_probs = sorted_probs[:cutoff]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()  # 归一化
        choice = rng.choice(cutoff, p=nucleus_probs)
        return int(sorted_idx[choice])

    @staticmethod
    def topk(
        logits: np.ndarray,
        k: int = 10,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Top-k 采样。

        从概率最高的 k 个 token 中按概率采样。

        Args:
            logits: (vocab,) 的 logits
            k: 保留的 token 数
            temperature: 采样温度
            rng: 随机数生成器

        Returns:
            采样的 token id
        """
        if rng is None:
            rng = np.random.default_rng()
        if logits.ndim > 1:
            logits_1d = logits.reshape(-1)[-logits.shape[-1]:]
        else:
            logits_1d = logits
        k = min(k, len(logits_1d))
        # 找 top-k 的索引
        top_idx = np.argpartition(-logits_1d, kth=k - 1)[:k]
        # 在 top-k 中按 softmax 概率采样
        top_logits = logits_1d[top_idx]
        top_probs = _softmax_np(top_logits, temperature)
        choice = rng.choice(k, p=top_probs)
        return int(top_idx[choice])

    @staticmethod
    def sample(
        logits: np.ndarray,
        strategy: str = "softmax",
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> tuple:
        """通用采样入口。

        Args:
            logits: (vocab,) 的 logits
            strategy: 采样策略名
                ("epsilon_greedy" / "softmax" / "nucleus" / "topk")
            rng: 随机数生成器
            **kwargs: 策略参数（epsilon / temperature / top_p / k）

        Returns:
            (token_id, logprob) 元组
        """
        if rng is None:
            rng = np.random.default_rng()
        if strategy == "epsilon_greedy":
            token_id = ActionSampler.epsilon_greedy(
                logits, epsilon=kwargs.get("epsilon", 0.1), rng=rng,
            )
        elif strategy == "softmax":
            token_id = ActionSampler.softmax(
                logits, temperature=kwargs.get("temperature", 1.0), rng=rng,
            )
        elif strategy == "nucleus":
            token_id = ActionSampler.nucleus(
                logits, top_p=kwargs.get("top_p", 0.9),
                temperature=kwargs.get("temperature", 1.0), rng=rng,
            )
        elif strategy == "topk":
            token_id = ActionSampler.topk(
                logits, k=kwargs.get("k", 10),
                temperature=kwargs.get("temperature", 1.0), rng=rng,
            )
        else:
            raise ValueError(f"未知采样策略: {strategy!r}")

        # 计算对数概率
        if logits.ndim > 1:
            logits_1d = logits.reshape(-1)[-logits.shape[-1]:]
        else:
            logits_1d = logits
        log_probs = _log_softmax_np(logits_1d)
        logprob = float(log_probs[token_id])
        return token_id, logprob


def repeat_penalty(
    logits: np.ndarray,
    generated_tokens,
    penalty: float = 1.2,
) -> np.ndarray:
    """对已生成 token 的 logits 施加重复惩罚。

    对已出现过的 token，logits 减去 penalty * |logit|，
    降低重复生成的概率。

    Args:
        logits: (vocab,) 的 logits
        generated_tokens: list[int]，已生成的 token id 列表
        penalty: 惩罚因子（>1 更强惩罚）

    Returns:
        惩罚后的 logits（副本）
    """
    out = logits.copy().astype(np.float64)
    seen = set(int(t) for t in generated_tokens)
    for tid in seen:
        if 0 <= tid < len(out):
            # 对已生成的 token 降低 logits
            out[tid] = out[tid] - penalty * abs(float(out[tid]))
    return out


class ExplorationSchedule:
    """探索率衰减 schedule。

    支持三种衰减模式：
    - linear: 线性衰减 ε_t = ε_start - (ε_start - ε_end) * t / total_steps
    - cosine: 余弦衰减 ε_t = ε_end + (ε_start - ε_end) * (1 + cos(π*t/total_steps)) / 2
    - exponential: 指数衰减 ε_t = ε_start * (ε_end/ε_start)^(t/total_steps)
    """

    VALID_MODES = ("linear", "cosine", "exponential")

    def __init__(
        self,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.02,
        total_steps: int = 1000,
        mode: str = "cosine",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode 必须为 {self.VALID_MODES}，got {mode!r}")
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.total_steps = max(1, int(total_steps))
        self.mode = mode

    def value(self, step: int) -> float:
        """返回当前步数的探索率。

        Args:
            step: 当前步数（0-indexed）

        Returns:
            当前探索率
        """
        step = max(0, min(step, self.total_steps))
        if self.mode == "linear":
            return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
                step / self.total_steps
            )
        elif self.mode == "cosine":
            return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
                1 + math.cos(math.pi * step / self.total_steps)
            ) / 2
        else:  # exponential
            if self.epsilon_start <= 0:
                return self.epsilon_end
            ratio = self.epsilon_end / self.epsilon_start
            return self.epsilon_start * (ratio ** (step / self.total_steps))
