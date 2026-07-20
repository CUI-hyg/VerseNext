"""Task 5.4.3: Sampler - Token 采样器。

设计目标
--------
把 LM 输出的 logits（``(vocab_size,)``）转换为下一个 token id，支持：

- ``temperature``: 温度缩放（``>1`` 平滑分布、``<1`` 锐化分布、``=0`` 等价 greedy）
- ``top_k``: 只保留 logits 最高的 k 个候选
- ``top_p`` (nucleus): 累计概率达到 p 的最小候选集
- ``softmax``: 转换为概率后采样

数值稳定
--------
- softmax 用 ``logits - max(logits)`` 减去最大值，避免 exp 溢出；
- top_p 在 sorted logits 上累计求和，避免漏选；
- 所有过滤都用 ``-inf`` 屏蔽（softmax 后概率为 0），不修改原 logits 比例。

随机性
------
- ``np.random.default_rng(seed)`` 可显式传入以复现；
- 默认 ``np.random`` 全局状态，便于多样性。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Sampler:
    """Token 采样器。

    Args:
        temperature: 温度，``>0``。``=1.0`` 不缩放；``<1.0`` 更确定；``>1.0`` 更随机。
            特殊值 ``0.0`` 等价于 greedy（返回 argmax）。
        top_k: 仅保留 logits 最高的 ``top_k`` 个候选；``0`` 表示不限制。
        top_p: 仅保留累计概率达到 ``top_p`` 的最小候选集；``1.0`` 表示不限制。
        seed: 随机种子（可选，用于复现）。

    用法
    ----
        sampler = Sampler(temperature=0.8, top_k=40, top_p=0.95)
        next_id = sampler.sample(logits)  # logits: np.ndarray of shape (vocab_size,)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ):
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def sample(self, logits) -> int:
        """从 logits 采样一个 token id。

        Args:
            logits: 形状 ``(vocab_size,)`` 的 ndarray（或 list）

        Returns:
            int: 采样得到的 token id

        步骤：
            1. temperature=0 -> 直接返回 argmax（greedy）
            2. temperature scaling: ``logits / temperature``
            3. top_k filtering: 把非 top_k 的 logits 置为 -inf
            4. top_p (nucleus) filtering: 在 softmax 后按累计概率截断
            5. softmax -> 概率分布 -> 多项分布采样
        """
        logits = np.asarray(logits, dtype=np.float32).flatten()
        if logits.size == 0:
            raise ValueError("logits is empty")

        # 特殊：temperature=0 等价于 greedy
        if self.temperature == 0.0:
            return int(np.argmax(logits))

        # 1. temperature scaling
        scaled = logits / self.temperature

        # 2. top_k filtering: 保留最高的 top_k 个，其余置 -inf
        if self.top_k > 0 and self.top_k < logits.size:
            # 找到 top_k 阈值（第 k 大的值）
            # np.partition 第 kth 从小到大数，所以 -top_k 是第 top_k 大
            kth_value = np.partition(scaled, -self.top_k)[-self.top_k]
            scaled = np.where(scaled >= kth_value, scaled, -np.inf)

        # 3. softmax（数值稳定）
        max_val = np.max(scaled)
        if not np.isfinite(max_val):
            # 全部 -inf，回退到 argmax 原始 logits
            return int(np.argmax(logits))
        shifted = scaled - max_val
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)

        # 4. top_p (nucleus) filtering
        if self.top_p < 1.0:
            # 按 probs 降序排列
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            # 找到累计概率 >= top_p 的最小位置
            # 保留 cumsum <= top_p 的，加上第一个超过 top_p 的（保证至少有一个 token）
            cutoff = np.searchsorted(cumsum, self.top_p)
            cutoff = min(cutoff, probs.size - 1)  # 防止越界
            # 把不在前 cutoff+1 的 token 概率清零
            keep_mask = np.zeros_like(probs, dtype=bool)
            keep_mask[sorted_idx[: cutoff + 1]] = True
            probs = np.where(keep_mask, probs, 0.0)
            # 重新归一化
            total = probs.sum()
            if total <= 0:
                # 兜底：均匀分布
                probs = np.ones_like(probs) / probs.size
            else:
                probs = probs / total

        # 5. 多项分布采样
        rng = self._rng if self._rng is not None else np.random
        token_id = int(rng.choice(probs.size, p=probs))
        return token_id

    def greedy(self, logits) -> int:
        """贪心采样：直接返回 argmax。"""
        return int(np.argmax(np.asarray(logits).flatten()))

    def __repr__(self) -> str:
        return (
            f"Sampler(temperature={self.temperature}, top_k={self.top_k}, "
            f"top_p={self.top_p}, seed={self.seed!r})"
        )


class GreedySampler(Sampler):
    """贪心采样器：始终返回 argmax（等价于 ``temperature=0``）。"""

    def __init__(self):
        super().__init__(temperature=1.0, top_k=0, top_p=1.0)

    def sample(self, logits) -> int:
        return self.greedy(logits)


__all__ = ["Sampler", "GreedySampler"]
