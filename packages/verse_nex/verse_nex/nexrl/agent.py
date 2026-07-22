"""NexAgent: RL 智能体（策略网络 + 参考网络 + KL 约束）。

NexAgent 封装 VerseNexLM 作为策略网络，并维护一个冻结的参考网络
（策略的 frozen 副本），用于 KL 散度约束。
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import numpy as np

from verse_torch import Tensor, no_grad

from .action import ActionSampler, NexAction
from .state import NexState


class NexAgent:
    """RL 智能体。

    封装 VerseNexLM 策略网络 + 参考网络（KL 约束）。

    Args:
        policy: VerseNexLM 策略网络（可训练）
        ref_model: 参考网络（None 则深拷贝 policy）
        sampler: ActionSampler 实例
        rng: 随机数生成器
    """

    def __init__(
        self,
        policy,
        ref_model=None,
        sampler: Optional[ActionSampler] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.policy = policy
        # 参考网络：冻结的 policy 副本
        if ref_model is None:
            self.ref_model = copy.deepcopy(policy)
        else:
            self.ref_model = ref_model
        # 冻结参考网络
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.sampler = sampler if sampler is not None else ActionSampler()
        self.rng = rng if rng is not None else np.random.default_rng()

    def forward_policy(self, input_ids, track_grad: bool = False) -> Tensor:
        """策略网络前向，返回 logits。

        Args:
            input_ids: (B, T) 整数索引
            track_grad: 是否追踪梯度

        Returns:
            logits: (B, T, vocab) Tensor
        """
        if not isinstance(input_ids, Tensor):
            input_ids = Tensor(np.asarray(input_ids, dtype=np.int64))
        elif input_ids.data.dtype != np.int64:
            input_ids = Tensor(input_ids.data.astype(np.int64))

        if track_grad:
            return self.policy(input_ids)
        else:
            with no_grad():
                return self.policy(input_ids)

    def forward_ref(self, input_ids) -> Tensor:
        """参考网络前向（no_grad），返回 logits。

        Args:
            input_ids: (B, T) 整数索引

        Returns:
            logits: (B, T, vocab) Tensor
        """
        if not isinstance(input_ids, Tensor):
            input_ids = Tensor(np.asarray(input_ids, dtype=np.int64))
        elif input_ids.data.dtype != np.int64:
            input_ids = Tensor(input_ids.data.astype(np.int64))

        with no_grad():
            logits = self.ref_model(input_ids)
        return logits

    def act(
        self,
        state: NexState,
        strategy: str = "softmax",
        temperature: float = 1.0,
        track_grad: bool = False,
        **kwargs,
    ) -> Tuple[NexAction, Tensor]:
        """根据当前状态采样动作。

        用策略网络对当前序列做一次前向，取最后一个位置的 logits，
        然后用 ActionSampler 采样。

        Args:
            state: 当前状态
            strategy: 采样策略
            temperature: 采样温度
            track_grad: 是否追踪梯度（用于 PPO 更新时需要）
            **kwargs: 采样策略额外参数

        Returns:
            (action, logits_tensor) 元组
        """
        # 构造输入序列
        all_tokens = state.all_tokens
        if not all_tokens:
            all_tokens = [0]
        input_ids = np.asarray([all_tokens], dtype=np.int64)  # (1, T)

        logits = self.forward_policy(input_ids, track_grad=track_grad)
        # 取最后一个位置的 logits: (1, vocab)
        if hasattr(logits, 'data'):
            logits_np = logits.data
        else:
            logits_np = np.asarray(logits)
        # (1, T, V) -> (V,)
        last_logits = logits_np[0, -1, :]

        # 采样
        kwargs["temperature"] = temperature
        token_id, logprob = self.sampler.sample(
            last_logits, strategy=strategy, rng=self.rng, **kwargs,
        )

        action = NexAction(token_id=token_id, logprob=logprob)
        return action, logits

    def compute_kl(self, new_logits: Tensor, ref_logits: Tensor) -> Tensor:
        """计算 KL 散度。

        KL(ref || new) = sum( ref_probs * (log(ref_probs) - log(new_probs)) )

        Args:
            new_logits: 策略网络 logits (..., V) Tensor
            ref_logits: 参考网络 logits (..., V) Tensor

        Returns:
            kl: 标量 Tensor
        """
        # log_softmax
        new_log_probs = new_logits.log_softmax(dim=-1)
        ref_log_probs = ref_logits.log_softmax(dim=-1)

        # ref_probs = exp(ref_log_probs)
        ref_probs = ref_log_probs.exp()

        # KL = sum(ref_probs * (ref_log_probs - new_log_probs))
        kl = (ref_probs * (ref_log_probs - new_log_probs)).sum()
        return kl

    def compute_kl_scalar(self, new_logits, ref_logits) -> float:
        """计算 KL 散度的标量值（用于监控）。

        Args:
            new_logits: 策略 logits (numpy 或 Tensor)
            ref_logits: 参考 logits (numpy 或 Tensor)

        Returns:
            KL 散度标量值
        """
        # 转 numpy
        if isinstance(new_logits, Tensor):
            new_np = new_logits.data
        else:
            new_np = np.asarray(new_logits)
        if isinstance(ref_logits, Tensor):
            ref_np = ref_logits.data
        else:
            ref_np = np.asarray(ref_logits)

        # 取最后一维
        if new_np.ndim > 2:
            new_np = new_np.reshape(-1, new_np.shape[-1])
        if ref_np.ndim > 2:
            ref_np = ref_np.reshape(-1, ref_np.shape[-1])

        # log_softmax (numpy)
        def _log_softmax_np(x):
            shifted = x - np.max(x, axis=-1, keepdims=True)
            return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

        new_logp = _log_softmax_np(new_np)
        ref_logp = _log_softmax_np(ref_np)
        ref_p = np.exp(ref_logp)

        kl = np.sum(ref_p * (ref_logp - new_logp), axis=-1)
        return float(np.mean(kl))

    def sync_ref(self) -> None:
        """同步参考网络为当前策略网络权重。

        用于训练过程中定期更新参考网络。
        """
        ref_sd = self.policy.state_dict()
        self.ref_model.load_state_dict(ref_sd, strict=False)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def get_hidden_states(self, input_ids) -> np.ndarray:
        """获取策略网络最后一层隐藏状态（用于 value function）。

        Args:
            input_ids: (B, T) 整数索引

        Returns:
            hidden: (B, T, D) ndarray
        """
        if not isinstance(input_ids, Tensor):
            input_ids = Tensor(np.asarray(input_ids, dtype=np.int64))
        elif input_ids.data.dtype != np.int64:
            input_ids = Tensor(input_ids.data.astype(np.int64))

        with no_grad():
            # 获取最后一层 block 的输出（norm 前）
            x = self.policy.tok_emb(input_ids)
            for block in self.policy.blocks:
                x, _ = block(x, position_offset=0, kv_cache=None)
            # x: (B, T, D)
            if hasattr(x, 'data'):
                return x.data
            return np.asarray(x)
