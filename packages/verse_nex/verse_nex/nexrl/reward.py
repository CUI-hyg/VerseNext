"""NexReward: 多维奖励 + 奖励归一化 + 奖励塑形。

包含：
- NexReward: 多维奖励计算（correctness / fluency / safety / length_penalty）
- RewardNormalizer: running mean/std 归一化（Welford 算法）
- RewardShaper: potential-based reward shaping（γ 折扣）
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 安全关键词黑名单（简化版，可按需扩展）
# ---------------------------------------------------------------------------
_DEFAULT_SAFETY_BLACKLIST = (
    "暴力", "色情", "违法", "毒品", "枪支",
    "kill", "bomb", "hack", "illegal", "drug",
)


# ---------------------------------------------------------------------------
# NexReward: 多维奖励
# ---------------------------------------------------------------------------


class NexReward:
    """多维奖励计算器。

    计算四个维度的奖励：
    - correctness: 精确匹配 / 子串匹配 / BLEU（复用 verse_torch.scoring）
    - fluency: 困惑度反比（用策略模型 logprob 均值）
    - safety: 关键词黑名单检测（0/1）
    - length_penalty: -abs(len - target_len) / target_len

    最终 total = 加权和（各维度权重可配置）。

    Args:
        weights: dict，各维度权重，默认均衡
        correctness_mode: "exact" / "substring" / "bleu"
        target_len: 目标生成长度
        safety_blacklist: 安全关键词黑名单
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        correctness_mode: str = "exact",
        target_len: int = 32,
        safety_blacklist: Optional[Iterable[str]] = None,
    ):
        # 默认权重
        self.weights: Dict[str, float] = {
            "correctness": 0.4,
            "fluency": 0.2,
            "safety": 0.2,
            "length_penalty": 0.2,
        }
        if weights is not None:
            self.weights.update(weights)
        self.correctness_mode = correctness_mode
        self.target_len = int(target_len)
        self.safety_blacklist = tuple(safety_blacklist) if safety_blacklist is not None else _DEFAULT_SAFETY_BLACKLIST

    def correctness(self, generated: str, reference: str) -> float:
        """计算 correctness 奖励。

        根据 correctness_mode 选择：
        - exact: 精确匹配（1.0 / 0.0）
        - substring: 子串匹配率
        - bleu: BLEU 分数（复用 verse_torch.scoring）
        """
        gen = generated.strip()
        ref = reference.strip()
        if self.correctness_mode == "exact":
            return 1.0 if gen == ref else 0.0
        elif self.correctness_mode == "substring":
            if not ref:
                return 1.0 if not gen else 0.0
            return 1.0 if ref in gen else 0.0
        elif self.correctness_mode == "bleu":
            try:
                from verse_torch.scoring import bleu as _bleu
                return _bleu(gen, ref)
            except Exception:
                # 简化 BLEU fallback
                if not gen or not ref:
                    return 0.0
                common = set(gen) & set(ref)
                return len(common) / max(len(set(gen) | set(ref)), 1)
        else:
            return 1.0 if gen == ref else 0.0

    def fluency(self, logprobs: List[float]) -> float:
        """计算 fluency 奖励（困惑度反比）。

        fluency = 1 / (1 + mean_neg_logprob)
        logprobs 为负值，越接近 0 越好。

        Args:
            logprobs: 每步生成 token 的对数概率列表

        Returns:
            fluency 奖励（0-1）
        """
        if not logprobs:
            return 0.0
        # logprobs 是负值，取均值后取绝对值近似困惑度
        mean_neg_lp = -float(np_mean(logprobs))
        # fluency = 1 / (1 + perplexity_proxy)
        return 1.0 / (1.0 + mean_neg_lp)

    def safety(self, generated: str) -> float:
        """计算 safety 奖励（关键词黑名单检测）。

        Returns:
            1.0（安全）/ 0.0（不安全）
        """
        gen_lower = generated.lower()
        for kw in self.safety_blacklist:
            if kw.lower() in gen_lower:
                return 0.0
        return 1.0

    def length_penalty(self, generated: str, target_len: Optional[int] = None) -> float:
        """计算长度惩罚。

        penalty = -abs(len - target_len) / target_len

        Returns:
            负值惩罚（-1 到 0）
        """
        tl = target_len if target_len is not None else self.target_len
        tl = max(1, tl)
        actual_len = len(generated.strip())
        return -abs(actual_len - tl) / tl

    def compute(
        self,
        generated: str,
        reference: str = "",
        logprobs: Optional[List[float]] = None,
        generated_tokens: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """计算多维奖励。

        Args:
            generated: 生成的文本
            reference: 参考答案
            logprobs: 每步的对数概率列表（用于 fluency）
            generated_tokens: 生成的 token id 列表（用于估算长度）

        Returns:
            dict 含各维度分数 + "total" 加权总分
        """
        if logprobs is None:
            logprobs = []
        if generated_tokens is not None:
            # 用 token 数估算目标长度
            target_len = max(1, self.target_len)
            actual_len = len(generated_tokens)
            lp = -abs(actual_len - target_len) / target_len
        else:
            lp = self.length_penalty(generated)

        correctness_score = self.correctness(generated, reference)
        fluency_score = self.fluency(logprobs)
        safety_score = self.safety(generated)

        total = (
            self.weights["correctness"] * correctness_score
            + self.weights["fluency"] * fluency_score
            + self.weights["safety"] * safety_score
            + self.weights["length_penalty"] * lp
        )

        return {
            "correctness": float(correctness_score),
            "fluency": float(fluency_score),
            "safety": float(safety_score),
            "length_penalty": float(lp),
            "total": float(total),
        }


def np_mean(values):
    """简化 numpy mean（避免硬依赖 numpy 在顶层 import）。"""
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)


# ---------------------------------------------------------------------------
# RewardNormalizer: Welford running mean/std
# ---------------------------------------------------------------------------


class RewardNormalizer:
    """奖励归一化器（Welford 在线算法）。

    使用 Welford 算法维护 running mean 和 running std，
    避免 storing all rewards。

    用法：
        normalizer = RewardNormalizer()
        for reward in rewards:
            normalized = normalizer.normalize(reward)
            normalizer.update(reward)
    """

    def __init__(self, clip_range: float = 10.0):
        """初始化归一化器。

        Args:
            clip_range: 归一化后裁剪范围 ±clip_range
        """
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # M2 统计量（sum of squared deviations）
        self.clip_range = float(clip_range)

    def update(self, reward: float) -> None:
        """更新 running 统计量（Welford 算法）。

        Args:
            reward: 新的 reward 值
        """
        self.count += 1
        delta = float(reward) - self.mean
        self.mean += delta / self.count
        delta2 = float(reward) - self.mean
        self.m2 += delta * delta2

    def update_batch(self, rewards: List[float]) -> None:
        """批量更新。"""
        for r in rewards:
            self.update(r)

    @property
    def variance(self) -> float:
        """当前方差。"""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """当前标准差。"""
        return math.sqrt(max(self.variance, 1e-8))

    def normalize(self, reward: float) -> float:
        """归一化 reward。

        normalized = (reward - mean) / (std + eps)

        若 count < 2（无统计量），返回原始 reward。

        Args:
            reward: 原始 reward

        Returns:
            归一化后的 reward
        """
        if self.count < 2:
            return float(reward)
        normalized = (float(reward) - self.mean) / (self.std + 1e-8)
        # 裁剪防止极端值
        if self.clip_range > 0:
            normalized = max(-self.clip_range, min(self.clip_range, normalized))
        return float(normalized)

    def reset(self) -> None:
        """重置统计量。"""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


# ---------------------------------------------------------------------------
# RewardShaper: potential-based reward shaping
# ---------------------------------------------------------------------------


class RewardShaper:
    """Potential-based reward shaping。

    F(s', s) = γ * Φ(s') - Φ(s)

    其中 Φ 是势函数（potential function），用于评估状态的质量。
    塑形后的 reward = original_reward + F(s', s)。

    势函数选择：
    - "fluency": 用 fluency 奖励作为势函数
    - "length": 用长度接近度作为势函数
    - 自定义 callable

    Args:
        gamma: 折扣因子
        potential_fn: 势函数，接受 state_info dict 返回 float
            None 时用内置 fluency 势函数
    """

    def __init__(
        self,
        gamma: float = 0.99,
        potential_fn: Optional[Callable] = None,
    ):
        self.gamma = float(gamma)
        if potential_fn is not None:
            self.potential_fn = potential_fn
        else:
            self.potential_fn = self._default_potential

    def _default_potential(self, state_info: dict) -> float:
        """默认势函数：基于 fluency 的简化版本。

        Args:
            state_info: dict，含：
                - logprobs: list[float]
                - generated_len: int
                - target_len: int

        Returns:
            势函数值（0-1 范围）
        """
        logprobs = state_info.get("logprobs", [])
        if not logprobs:
            return 0.0
        mean_neg_lp = -sum(float(v) for v in logprobs) / max(len(logprobs), 1)
        return 1.0 / (1.0 + mean_neg_lp)

    def shape(
        self,
        reward: float,
        prev_state_info: dict,
        curr_state_info: dict,
    ) -> float:
        """对 reward 进行塑形。

        shaped_reward = reward + γ * Φ(s') - Φ(s)

        Args:
            reward: 原始 reward
            prev_state_info: 前一状态的信息 dict
            curr_state_info: 当前状态的信息 dict

        Returns:
            塑形后的 reward
        """
        phi_prev = self.potential_fn(prev_state_info)
        phi_curr = self.potential_fn(curr_state_info)
        shaped = float(reward) + self.gamma * phi_curr - phi_prev
        return float(shaped)
