"""NexTrainer: PPO 风格 RL 训练器。

实现：
- PPO clipped surrogate objective（clip_ratio 默认 0.2）
- GAE（广义优势估计，λ 默认 0.95）
- 可选 value function（Linear 投影到 1），无则纯策略梯度 fallback
- KL 祖父项：策略与参考模型 KL 散度惩罚，超阈值自适应增加 KL 权重
- policy_gradient_fallback：无 value function 时用 return - baseline(滑动均值)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.vnn import Linear
from verse_torch.optim import AdamW
from verse_torch.training import clip_grad_norm

from .agent import NexAgent
from .collector import ParallelRolloutCollector, Rollout
from .reward import NexReward, RewardNormalizer, RewardShaper


def _forward_with_hidden(model, input_ids) -> Tuple[Tensor, Tensor]:
    """前向计算，返回 (logits, hidden_states)，两者都带梯度。

    复现 VerseNexLM.forward 的逻辑，但在 norm/head 之前返回 hidden states，
    使 value head 可以基于 hidden states 计算价值估计。

    Args:
        model: VerseNexLM 策略网络
        input_ids: (B, T) 整数索引

    Returns:
        logits: (B, T, vocab) Tensor
        hidden: (B, T, dim) Tensor（norm 前的最后一层隐藏状态）
    """
    if not isinstance(input_ids, Tensor):
        input_ids = Tensor(np.asarray(input_ids, dtype=np.int64))
    elif input_ids.data.dtype != np.int64:
        input_ids = Tensor(input_ids.data.astype(np.int64))

    x = model.tok_emb(input_ids)
    for block in model.blocks:
        x, _ = block(x, position_offset=0, kv_cache=None)
    hidden = x  # (B, T, D)
    x_norm = model.norm(x)
    logits = model.head(x_norm)
    return logits, hidden


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    """numpy log_softmax。"""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


class NexTrainer:
    """PPO 风格 RL 训练器。

    Args:
        agent: NexAgent 实例（策略 + 参考网络）
        optimizer: 优化器（None 则自动创建 AdamW）
        cfg: dict，配置项：
            - clip_ratio: PPO clip 比率（默认 0.2）
            - gamma: 折扣因子（默认 0.99）
            - gae_lambda: GAE lambda（默认 0.95）
            - ppo_epochs: 每次 rollout 的 PPO 更新轮数（默认 4）
            - lr: 学习率（默认 1e-4）
            - ent_coef: 熵奖励系数（默认 0.01）
            - vf_coef: value loss 系数（默认 0.5）
            - max_grad_norm: 梯度裁剪范数（默认 0.5）
            - use_value: 是否使用 value function（默认 True）
            - kl_weight: 初始 KL 权重（默认 0.0）
            - target_kl: KL 目标阈值（默认 0.02）
            - kl_adaptive: 是否自适应 KL（默认 True）
            - max_new_tokens: rollout 最大生成 token 数（默认 16）
            - strategy: 采样策略（默认 "softmax"）
            - temperature: 采样温度（默认 1.0）
            - reward_fn: 自定义 reward 函数
            - use_reward_normalizer: 是否启用 reward 归一化（默认 True）
            - use_reward_shaper: 是否启用 reward 塑形（默认 False）
    """

    def __init__(self, agent: NexAgent, optimizer=None, cfg=None):
        self.agent = agent
        self.cfg = cfg if cfg is not None else {}

        # PPO 超参
        self.clip_ratio = float(self.cfg.get("clip_ratio", 0.2))
        self.gamma = float(self.cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg.get("gae_lambda", 0.95))
        self.ppo_epochs = int(self.cfg.get("ppo_epochs", 4))
        self.lr = float(self.cfg.get("lr", 1e-4))
        self.ent_coef = float(self.cfg.get("ent_coef", 0.01))
        self.vf_coef = float(self.cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(self.cfg.get("max_grad_norm", 0.5))

        # Value function
        self.use_value = bool(self.cfg.get("use_value", True))
        if self.use_value:
            dim = self.agent.policy.dim
            self.value_head = Linear(dim, 1, bias=True)
            # 初始化 value head（小权重，避免初始 value 过大）
            self.value_head.weight.data = (
                self.value_head.weight.data * 0.1
            ).astype(np.float32)
        else:
            self.value_head = None

        # KL 配置
        self.kl_weight = float(self.cfg.get("kl_weight", 0.0))
        self.target_kl = float(self.cfg.get("target_kl", 0.02))
        self.kl_adaptive = bool(self.cfg.get("kl_adaptive", True))

        # 优化器
        if optimizer is None:
            params = list(self.agent.policy.parameters())
            if self.value_head is not None:
                params += list(self.value_head.parameters())
            optimizer = AdamW(params, lr=self.lr, weight_decay=0.0)
            print(f"[NexTrainer] 自动构建 AdamW (lr={self.lr})", flush=True)
        self.optimizer = optimizer

        # Reward
        self.reward_fn = self.cfg.get("reward_fn", None)
        self.use_reward_normalizer = bool(
            self.cfg.get("use_reward_normalizer", True)
        )
        self.use_reward_shaper = bool(self.cfg.get("use_reward_shaper", False))
        self.reward_normalizer = RewardNormalizer() if self.use_reward_normalizer else None
        self.reward_shaper = RewardShaper(gamma=self.gamma) if self.use_reward_shaper else None

        # Rollout collector
        self.max_new_tokens = int(self.cfg.get("max_new_tokens", 16))
        self.strategy = self.cfg.get("strategy", "softmax")
        self.temperature = float(self.cfg.get("temperature", 1.0))

        # 构建 value_fn（用于 rollout collection 时记录 values）
        value_fn = None
        if self.use_value:
            value_fn = self._collect_value_fn

        self.collector = ParallelRolloutCollector(
            agent=self.agent,
            max_new_tokens=self.max_new_tokens,
            strategy=self.strategy,
            temperature=self.temperature,
            value_fn=value_fn,
            rng=np.random.default_rng(),
        )

        # Policy gradient fallback 的 baseline
        self.baseline = 0.0
        self.baseline_momentum = 0.9

        # 训练历史
        self.train_losses: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.kl_history: List[float] = []
        self.reward_history: List[float] = []
        self.kl_weights: List[float] = []

    # ------------------------------------------------------------------
    # Value function for rollout collection
    # ------------------------------------------------------------------

    def _collect_value_fn(self, agent: NexAgent, input_ids) -> float:
        """Rollout 采集时计算 value 估计（no_grad）。"""
        if self.value_head is None:
            return 0.0
        with no_grad():
            if not isinstance(input_ids, Tensor):
                input_ids = Tensor(np.asarray(input_ids, dtype=np.int64))
            elif input_ids.data.dtype != np.int64:
                input_ids = Tensor(input_ids.data.astype(np.int64))

            x = agent.policy.tok_emb(input_ids)
            for block in agent.policy.blocks:
                x, _ = block(x, position_offset=0, kv_cache=None)
            values = self.value_head(x)  # (B, T, 1)
            # 取最后一个位置的 value
            v_data = values.data
            return float(v_data.reshape(-1)[-1])

    # ------------------------------------------------------------------
    # GAE: 广义优势估计
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 GAE 优势和回报。

        GAE 公式：
            δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            A_t = Σ_{k=0}^{T-1-t} (γλ)^k * δ_{t+k}
            R_t = A_t + V(s_t)

        Args:
            rewards: 每步的 reward 列表
            values: 每步的 value 估计列表

        Returns:
            advantages: (T,) ndarray
            returns: (T,) ndarray
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0  # 终止状态 value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ------------------------------------------------------------------
    # Policy gradient fallback (无 value function)
    # ------------------------------------------------------------------

    def _compute_returns_baseline(
        self,
        rewards: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 return 和 baseline-based advantage（无 value function）。

        return_t = Σ_{k=0}^{T-1-t} γ^k * r_{t+k}
        advantage_t = return_t - baseline

        baseline 用滑动均值更新。

        Args:
            rewards: 每步的 reward 列表

        Returns:
            advantages: (T,) ndarray
            returns: (T,) ndarray
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)

        # 反向计算 discounted return
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        # advantage = return - baseline
        advantages = returns - self.baseline

        # 更新 baseline（滑动均值）
        if T > 0:
            mean_return = float(np.mean(returns))
            self.baseline = (
                self.baseline_momentum * self.baseline
                + (1 - self.baseline_momentum) * mean_return
            )

        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, rollout: Rollout) -> Tuple[float, float, float, float]:
        """对单条 rollout 执行 PPO 更新。

        Args:
            rollout: 单条 rollout 轨迹

        Returns:
            (total_loss, policy_loss, value_loss, kl) 元组
        """
        gen_tokens = rollout.generated_tokens
        prompt_tokens = rollout.prompt_tokens
        old_logprobs = rollout.logprobs
        advantages = rollout.advantages  # 预计算的 GAE 优势
        returns = rollout.returns  # 预计算的 returns
        values_collected = rollout.values  # 采集时记录的 values

        gen_len = len(gen_tokens)
        if gen_len == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 构造输入序列
        all_tokens = list(prompt_tokens) + gen_tokens
        if not all_tokens:
            all_tokens = [0]
        input_ids = np.asarray([all_tokens], dtype=np.int64)  # (1, T)

        # 前向（带梯度）
        logits, hidden = _forward_with_hidden(self.agent.policy, input_ids)
        # logits: (1, T, V), hidden: (1, T, D)

        # 提取生成 token 对应位置的 logits
        prompt_len = len(prompt_tokens)
        start = max(0, prompt_len - 1)
        end = min(start + gen_len, logits.shape[1])

        # 如果 start + gen_len 超出 logits 长度，调整
        actual_gen_len = end - start
        if actual_gen_len == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 切片获取相关 logits: (1, actual_gen_len, V)
        relevant_logits = logits[:, start:end, :]

        # log_softmax
        log_probs = relevant_logits.log_softmax(dim=-1)  # (1, actual_gen_len, V)

        # One-hot gather: 获取每个生成 token 的 log_prob
        V = logits.shape[-1]
        one_hot = np.zeros((1, actual_gen_len, V), dtype=np.float32)
        for i in range(actual_gen_len):
            tid = int(gen_tokens[i])
            if 0 <= tid < V:
                one_hot[0, i, tid] = 1.0

        # (log_probs * one_hot).sum(dim=-1) -> (1, actual_gen_len)
        action_log_probs = (log_probs * Tensor(one_hot)).sum(dim=-1)
        # reshape -> (actual_gen_len,)
        new_log_probs = action_log_probs.reshape(-1)

        # old log_probs（常数，不参与梯度）
        old_lp = np.array(old_logprobs[:actual_gen_len], dtype=np.float32)
        old_lp_tensor = Tensor(old_lp, requires_grad=False)

        # ratio = exp(new - old)
        ratio = (new_log_probs - old_lp_tensor).exp()  # (actual_gen_len,)

        # advantages（常数）
        adv = np.array(advantages[:actual_gen_len], dtype=np.float32)
        # 标准化优势（提升稳定性）
        if len(adv) > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_tensor = Tensor(adv, requires_grad=False)

        # Clipped surrogate
        surr1 = ratio * adv_tensor
        surr2 = ratio.clamp(
            1.0 - self.clip_ratio, 1.0 + self.clip_ratio
        ) * adv_tensor
        policy_loss = -surr1.minimum(surr2).mean()

        # Value loss
        value_loss = Tensor(np.array(0.0, dtype=np.float32))
        if self.value_head is not None:
            # 获取相关位置的 hidden states: (1, actual_gen_len, D)
            relevant_hidden = hidden[:, start:end, :]
            pred_values = self.value_head(relevant_hidden)  # (1, actual_gen_len, 1)
            pred_values_flat = pred_values.reshape(-1)  # (actual_gen_len,)

            # target returns
            ret = np.array(returns[:actual_gen_len], dtype=np.float32)
            ret_tensor = Tensor(ret, requires_grad=False)

            # value_loss = mean((pred - target)^2)
            diff = pred_values_flat - ret_tensor
            value_loss = (diff * diff).mean()

        # Entropy bonus
        probs = log_probs.exp()  # (1, actual_gen_len, V)
        entropy = -(probs * log_probs).sum(dim=-1).mean()  # scalar

        # KL penalty（如果启用）
        kl_loss = Tensor(np.array(0.0, dtype=np.float32))
        kl_scalar = 0.0
        if self.kl_weight > 0:
            # 计算策略与参考模型的 KL
            with no_grad():
                ref_input = Tensor(input_ids)
                ref_logits = self.agent.forward_ref(ref_input)
            kl_tensor = self.agent.compute_kl(logits, ref_logits)
            kl_loss = kl_tensor
            kl_scalar = float(kl_tensor.data.item() if kl_tensor.data.ndim == 0
                              else kl_tensor.data.sum())

        # Total loss
        total_loss = (
            policy_loss
            + self.vf_coef * value_loss
            - self.ent_coef * entropy
            + self.kl_weight * kl_loss
        )

        # Backward
        total_loss.backward()

        # 梯度裁剪
        params = list(self.agent.policy.parameters())
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        if self.max_grad_norm > 0:
            clip_grad_norm(params, self.max_grad_norm)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # 返回各 loss 值
        return (
            float(total_loss.data.item() if total_loss.data.ndim == 0
                  else total_loss.data.sum()),
            float(policy_loss.data.item() if policy_loss.data.ndim == 0
                  else policy_loss.data.sum()),
            float(value_loss.data.item() if value_loss.data.ndim == 0
                  else value_loss.data.sum()),
            kl_scalar,
        )

    # ------------------------------------------------------------------
    # Policy gradient fallback update (无 value function)
    # ------------------------------------------------------------------

    def _policy_gradient_fallback(self, rollout: Rollout) -> Tuple[float, float, float, float]:
        """纯策略梯度更新（无 value function）。

        loss = -mean(log_prob * advantage)

        Args:
            rollout: 单条 rollout 轨迹

        Returns:
            (total_loss, policy_loss, 0.0, kl) 元组
        """
        gen_tokens = rollout.generated_tokens
        prompt_tokens = rollout.prompt_tokens
        old_logprobs = rollout.logprobs
        advantages = rollout.advantages  # return - baseline
        returns = rollout.returns

        gen_len = len(gen_tokens)
        if gen_len == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 构造输入序列
        all_tokens = list(prompt_tokens) + gen_tokens
        if not all_tokens:
            all_tokens = [0]
        input_ids = np.asarray([all_tokens], dtype=np.int64)  # (1, T)

        # 前向（带梯度）
        logits = self.agent.policy(Tensor(input_ids))

        prompt_len = len(prompt_tokens)
        start = max(0, prompt_len - 1)
        end = min(start + gen_len, logits.shape[1])
        actual_gen_len = end - start
        if actual_gen_len == 0:
            return 0.0, 0.0, 0.0, 0.0

        # log_softmax
        relevant_logits = logits[:, start:end, :]
        log_probs = relevant_logits.log_softmax(dim=-1)

        # One-hot gather
        V = logits.shape[-1]
        one_hot = np.zeros((1, actual_gen_len, V), dtype=np.float32)
        for i in range(actual_gen_len):
            tid = int(gen_tokens[i])
            if 0 <= tid < V:
                one_hot[0, i, tid] = 1.0

        new_log_probs = (
            (log_probs * Tensor(one_hot)).sum(dim=-1).reshape(-1)
        )

        # advantages
        adv = np.array(advantages[:actual_gen_len], dtype=np.float32)
        if len(adv) > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_tensor = Tensor(adv, requires_grad=False)

        # Policy gradient loss: -mean(log_prob * advantage)
        policy_loss = -(new_log_probs * adv_tensor).mean()

        # Entropy bonus
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # KL penalty
        kl_loss = Tensor(np.array(0.0, dtype=np.float32))
        kl_scalar = 0.0
        if self.kl_weight > 0:
            with no_grad():
                ref_logits = self.agent.forward_ref(Tensor(input_ids))
            kl_tensor = self.agent.compute_kl(logits, ref_logits)
            kl_loss = kl_tensor
            kl_scalar = float(kl_tensor.data.item() if kl_tensor.data.ndim == 0
                              else kl_tensor.data.sum())

        # Total
        total_loss = policy_loss - self.ent_coef * entropy + self.kl_weight * kl_loss

        # Backward
        total_loss.backward()

        params = list(self.agent.policy.parameters())
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        if self.max_grad_norm > 0:
            clip_grad_norm(params, self.max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return (
            float(total_loss.data.item() if total_loss.data.ndim == 0
                  else total_loss.data.sum()),
            float(policy_loss.data.item() if policy_loss.data.ndim == 0
                  else policy_loss.data.sum()),
            0.0,
            kl_scalar,
        )

    # ------------------------------------------------------------------
    # KL 监控 + 自适应惩罚
    # ------------------------------------------------------------------

    def _compute_kl_for_rollout(self, rollout: Rollout) -> float:
        """计算单条 rollout 的 KL 散度（用于监控）。"""
        gen_tokens = rollout.generated_tokens
        prompt_tokens = rollout.prompt_tokens
        gen_len = len(gen_tokens)
        if gen_len == 0:
            return 0.0

        all_tokens = list(prompt_tokens) + gen_tokens
        if not all_tokens:
            all_tokens = [0]
        input_ids = np.asarray([all_tokens], dtype=np.int64)

        with no_grad():
            policy_logits = self.agent.forward_policy(input_ids, track_grad=False)
            ref_logits = self.agent.forward_ref(input_ids)

        return self.agent.compute_kl_scalar(policy_logits, ref_logits)

    def _update_kl_weight(self, kl: float) -> None:
        """自适应更新 KL 权重。"""
        if not self.kl_adaptive:
            return

        if kl > self.target_kl * 2:
            # KL 超阈值，增加惩罚
            self.kl_weight = min(self.kl_weight * 2.0 + 0.01, 1.0)
        elif kl < self.target_kl / 2:
            # KL 较低，减小惩罚
            self.kl_weight = max(self.kl_weight * 0.5, 0.0)

    # ------------------------------------------------------------------
    # fit: 主训练循环
    # ------------------------------------------------------------------

    def fit(
        self,
        prompts: List[str],
        n_epochs: int = 10,
        n_rollouts_per_prompt: int = 2,
        encode_fn=None,
        decode_fn=None,
    ) -> Tuple[List[float], List[float], List[float]]:
        """主训练循环。

        Args:
            prompts: prompt 文本列表
            n_epochs: 训练 epoch 数
            n_rollouts_per_prompt: 每个 prompt 的 rollout 数
            encode_fn: 编码函数
            decode_fn: 解码函数

        Returns:
            (train_losses, kl_history, reward_history) 元组
        """
        if not prompts:
            print("[NexTrainer] prompts 为空，跳过训练", flush=True)
            return self.train_losses, self.kl_history, self.reward_history

        print(
            f"[NexTrainer] 开始训练: n_epochs={n_epochs}, "
            f"n_prompts={len(prompts)}, "
            f"n_rollouts={n_rollouts_per_prompt}, "
            f"use_value={self.use_value}",
            flush=True,
        )

        # 默认 reward 函数
        if self.reward_fn is None:
            nex_reward = NexReward(target_len=self.max_new_tokens)

            def default_reward_fn(generated_text, prompt_text, logprobs, generated_tokens):
                return nex_reward.compute(
                    generated=generated_text,
                    reference="",
                    logprobs=logprobs,
                    generated_tokens=generated_tokens,
                )

            reward_fn = default_reward_fn
        else:
            reward_fn = self.reward_fn

        for epoch in range(n_epochs):
            # 1. Collect rollouts
            rollouts = self.collector.collect(
                prompts=prompts,
                n_rollouts_per_prompt=n_rollouts_per_prompt,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                reward_fn=reward_fn,
            )

            # 2. 处理 rewards（归一化 + 塑形）
            for r in rollouts:
                # reward shaping（可选）
                if self.reward_shaper is not None:
                    prev_info = {"logprobs": r.logprobs[:-1] if len(r.logprobs) > 1 else []}
                    curr_info = {"logprobs": r.logprobs}
                    r.reward = self.reward_shaper.shape(
                        r.reward, prev_info, curr_info,
                    )

                # reward normalization（可选）
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update(r.reward)
                    r.reward = self.reward_normalizer.normalize(r.reward)

            # 3. Compute advantages
            for r in rollouts:
                if self.use_value and r.values:
                    # GAE
                    # 把终末 reward 分配到最后一步
                    step_rewards = [0.0] * len(r.generated_tokens)
                    step_rewards[-1] = r.reward
                    advs, rets = self._compute_gae(step_rewards, r.values)
                    r.advantages = advs
                    r.returns = rets
                else:
                    # Policy gradient fallback
                    step_rewards = [0.0] * len(r.generated_tokens)
                    step_rewards[-1] = r.reward
                    advs, rets = self._compute_returns_baseline(step_rewards)
                    r.advantages = advs
                    r.returns = rets

            # 4. PPO update
            epoch_losses = []
            epoch_kls = []
            for ppo_epoch in range(self.ppo_epochs):
                for rollout in rollouts:
                    if self.use_value and self.value_head is not None:
                        result = self._ppo_update(rollout)
                    else:
                        result = self._policy_gradient_fallback(rollout)

                    total_loss, policy_loss, value_loss, kl = result
                    epoch_losses.append(total_loss)
                    if kl > 0:
                        epoch_kls.append(kl)

            # 5. KL 监控 + 自适应惩罚
            if rollouts:
                avg_kl = np.mean([
                    self._compute_kl_for_rollout(r) for r in rollouts
                ])
            else:
                avg_kl = 0.0

            self._update_kl_weight(float(avg_kl))

            # 记录历史
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            avg_reward = float(np.mean([r.reward for r in rollouts])) if rollouts else 0.0

            self.train_losses.append(avg_loss)
            self.kl_history.append(float(avg_kl))
            self.reward_history.append(avg_reward)
            self.kl_weights.append(self.kl_weight)

            print(
                f"[NexTrainer] epoch {epoch+1}/{n_epochs} "
                f"loss={avg_loss:.4f} kl={avg_kl:.4f} "
                f"kl_weight={self.kl_weight:.4f} "
                f"avg_reward={avg_reward:.4f}",
                flush=True,
            )

        print(
            f"[NexTrainer] 训练完成: {n_epochs} epochs, "
            f"final_loss={self.train_losses[-1] if self.train_losses else 0:.4f}",
            flush=True,
        )
        return self.train_losses, self.kl_history, self.reward_history
