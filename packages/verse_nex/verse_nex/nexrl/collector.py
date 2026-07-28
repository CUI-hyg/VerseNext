"""ParallelRolloutCollector: 并行 rollout 采集器。

实现多 prompt / 多 rollout 并行采样，支持 batched 前向。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verse_torch import Tensor, no_grad

from .action import repeat_penalty
from .agent import NexAgent
from .state import NexState


@dataclass
class Rollout:
    """单条 rollout 轨迹。

    Attributes:
        prompt: 输入提示文本
        prompt_tokens: 输入提示 token id 列表
        generated_tokens: 生成的 token id 列表
        logprobs: 每步生成 token 的对数概率列表
        values: 每步的 value 估计（可选，None 表示无 value function）
        reward: 最终 reward
        reward_components: reward 各维度分解
        done_reason: 结束原因（"eos" / "max_len"）
        truncated: 是否被 max_len 截断（True 时 GAE 应用 V(s_T) bootstrap）
        advantages: GAE 优势（由 trainer 计算，默认空 ndarray）
        returns: 回报（由 trainer 计算，默认空 ndarray）
    """

    prompt: str = ""
    prompt_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)  # 默认空列表
    reward: float = 0.0
    reward_components: Dict[str, float] = field(default_factory=dict)
    done_reason: str = ""
    truncated: bool = False  # 是否被 max_len 截断（True）或 eos 终止（False）
    advantages: Any = field(default=None)  # numpy ndarray，由 trainer 填充
    returns: Any = field(default=None)  # numpy ndarray，由 trainer 填充

    def __post_init__(self):
        if self.reward_components is None:
            self.reward_components = {}
        if self.advantages is None:
            self.advantages = np.array([], dtype=np.float32)
        if self.returns is None:
            self.returns = np.array([], dtype=np.float32)

    @property
    def length(self) -> int:
        """生成长度。"""
        return len(self.generated_tokens)

    @property
    def all_tokens(self) -> List[int]:
        """完整 token 序列。"""
        return list(self.prompt_tokens) + list(self.generated_tokens)


class ParallelRolloutCollector:
    """并行 rollout 采集器。

    对每个 prompt 生成 n_rollouts 条 rollout 轨迹，
    支持 batched 前向（多个 prompt 的 token 序列 batch 化）。

    Args:
        agent: NexAgent 实例
        max_new_tokens: 每条 rollout 最大生成 token 数
        eos_id: eos token id（None 表示不提前停止）
        strategy: 采样策略
        temperature: 采样温度
        use_repeat_penalty: 是否启用重复惩罚
        repeat_penalty_factor: 重复惩罚因子
        value_fn: 可选的 value 估计函数，接受 (agent, input_ids) 返回 float
    """

    def __init__(
        self,
        agent: NexAgent,
        max_new_tokens: int = 16,
        eos_id: Optional[int] = None,
        strategy: str = "softmax",
        temperature: float = 1.0,
        use_repeat_penalty: bool = False,
        repeat_penalty_factor: float = 1.2,
        value_fn=None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.agent = agent
        self.max_new_tokens = int(max_new_tokens)
        self.eos_id = eos_id
        self.strategy = strategy
        self.temperature = float(temperature)
        self.use_repeat_penalty = bool(use_repeat_penalty)
        self.repeat_penalty_factor = float(repeat_penalty_factor)
        self.value_fn = value_fn
        self.rng = rng if rng is not None else np.random.default_rng()

    def collect(
        self,
        prompts: List[str],
        n_rollouts_per_prompt: int = 1,
        encode_fn=None,
        decode_fn=None,
        reward_fn=None,
    ) -> List[Rollout]:
        """采集 rollout 轨迹。

        对每个 prompt 生成 n_rollouts_per_prompt 条轨迹。

        Args:
            prompts: prompt 文本列表
            n_rollouts_per_prompt: 每个 prompt 的 rollout 数
            encode_fn: 编码函数 str -> list[int]（None 用 agent 的 encode）
            decode_fn: 解码函数 list[int] -> str（None 用 agent 的 decode）
            reward_fn: reward 函数 (generated_text, prompt) -> dict
                None 时用简单长度奖励

        Returns:
            List[Rollout]，长度 = len(prompts) * n_rollouts_per_prompt
        """
        if encode_fn is None:
            encode_fn = self._default_encode
        if decode_fn is None:
            decode_fn = self._default_decode

        all_rollouts: List[Rollout] = []

        for prompt in prompts:
            prompt_tokens = encode_fn(prompt)
            for rollout_idx in range(n_rollouts_per_prompt):
                rollout = self._single_rollout(
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    decode_fn=decode_fn,
                    reward_fn=reward_fn,
                )
                all_rollouts.append(rollout)

        return all_rollouts

    def _single_rollout(
        self,
        prompt: str,
        prompt_tokens: List[int],
        decode_fn,
        reward_fn,
    ) -> Rollout:
        """执行单条 rollout。"""
        generated_tokens: List[int] = []
        logprobs: List[float] = []
        values: List[float] = []

        done = False
        done_reason = ""

        with no_grad():
            for step in range(self.max_new_tokens):
                # 构造输入序列
                all_tokens = list(prompt_tokens) + generated_tokens
                if not all_tokens:
                    all_tokens = [0]
                input_ids = np.asarray([all_tokens], dtype=np.int64)

                # 策略前向
                logits = self.agent.forward_policy(input_ids, track_grad=False)
                logits_np = logits.data if hasattr(logits, "data") else np.asarray(logits)
                # 取最后一个位置: (vocab,)
                last_logits = logits_np[0, -1, :]

                # 重复惩罚
                if self.use_repeat_penalty and generated_tokens:
                    last_logits = repeat_penalty(
                        last_logits, generated_tokens,
                        penalty=self.repeat_penalty_factor,
                    )

                # 采样（Part5K1.7 1.13：统一使用 agent.sampler 实例）
                token_id, logprob = self.agent.sampler.sample(
                    last_logits,
                    strategy=self.strategy,
                    rng=self.rng,
                    temperature=self.temperature,
                )

                # value 估计（如果有 value_fn）
                if self.value_fn is not None:
                    try:
                        v = float(self.value_fn(self.agent, input_ids))
                        values.append(v)
                    except Exception:
                        values.append(0.0)

                generated_tokens.append(int(token_id))
                logprobs.append(float(logprob))

                # 检查终止
                if self.eos_id is not None and token_id == self.eos_id:
                    done = True
                    done_reason = "eos"
                    break

        # Part5K1.7（1.13）：公共逻辑（bootstrap value + decode + reward）抽取到 _build_rollout
        return self._build_rollout(
            prompt=prompt,
            prompt_tokens=list(prompt_tokens),
            generated_tokens=generated_tokens,
            logprobs=logprobs,
            values=values,
            done=done,
            done_reason=done_reason,
            decode_fn=decode_fn,
            reward_fn=reward_fn,
        )

    def collect_batched(
        self,
        prompts: List[str],
        n_rollouts_per_prompt: int = 1,
        encode_fn=None,
        decode_fn=None,
        reward_fn=None,
    ) -> List[Rollout]:
        """Batched 并行采集（多个 prompt 的 token 序列 padding 后一次前向）。

        与 collect 不同的是，此方法将多个 prompt 的序列 padding 到
        相同长度后做一次批量前向，减少前向次数。

        Args:
            同 collect

        Returns:
            List[Rollout]
        """
        if encode_fn is None:
            encode_fn = self._default_encode
        if decode_fn is None:
            decode_fn = self._default_decode

        # 构造所有 rollout 的初始状态
        rollout_states: List[dict] = []
        for prompt in prompts:
            prompt_tokens = encode_fn(prompt)
            for _ in range(n_rollouts_per_prompt):
                rollout_states.append({
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": [],
                    "logprobs": [],
                    "values": [],
                    "done": False,
                    "done_reason": "",
                })

        if not rollout_states:
            return []

        # 批量生成
        with no_grad():
            for step in range(self.max_new_tokens):
                # 构造 batch input（padding 到同一长度）
                batch_inputs = []
                batch_indices = []
                for i, rs in enumerate(rollout_states):
                    if rs["done"]:
                        continue
                    all_tokens = list(rs["prompt_tokens"]) + rs["generated_tokens"]
                    if not all_tokens:
                        all_tokens = [0]
                    batch_inputs.append(all_tokens)
                    batch_indices.append(i)

                if not batch_inputs:
                    break  # 所有 rollout 都已完成

                # Part5K1.7（1.11）：左 padding + attention mask
                # 左 padding 使真实序列末尾对齐到 max_len-1，last_logits 取 [:, -1, :] 即真实最后 token
                # （选项 C：不改 forward_policy 接口，padding 对末尾 logits 影响最小）
                max_len = max(len(seq) for seq in batch_inputs)
                pad_id = 0
                padded = np.full((len(batch_inputs), max_len), pad_id, dtype=np.int64)
                attention_mask = np.zeros((len(batch_inputs), max_len), dtype=np.int32)
                for j, seq in enumerate(batch_inputs):
                    pad_len_j = max_len - len(seq)
                    padded[j, pad_len_j:] = seq
                    attention_mask[j, pad_len_j:] = 1

                # 批量前向
                input_tensor = Tensor(padded)
                logits = self.agent.forward_policy(input_tensor, track_grad=False)
                logits_np = logits.data if hasattr(logits, "data") else np.asarray(logits)

                # 为每个活跃 rollout 采样
                for j, idx in enumerate(batch_indices):
                    rs = rollout_states[idx]
                    # Part5K1.7（1.11）：左 padding 后真实序列末尾对齐到 -1 位置
                    last_logits = logits_np[j, -1, :]

                    # 重复惩罚
                    if self.use_repeat_penalty and rs["generated_tokens"]:
                        last_logits = repeat_penalty(
                            last_logits, rs["generated_tokens"],
                            penalty=self.repeat_penalty_factor,
                        )

                    # Part5K1.7（1.13）：统一使用 agent.sampler 实例
                    token_id, logprob = self.agent.sampler.sample(
                        last_logits,
                        strategy=self.strategy,
                        rng=self.rng,
                        temperature=self.temperature,
                    )

                    # value 估计（使用未 padding 的真实序列）
                    if self.value_fn is not None:
                        try:
                            v = float(self.value_fn(
                                self.agent,
                                np.asarray([batch_inputs[j]], dtype=np.int64),
                            ))
                            rs["values"].append(v)
                        except Exception:
                            rs["values"].append(0.0)

                    rs["generated_tokens"].append(int(token_id))
                    rs["logprobs"].append(float(logprob))

                    if self.eos_id is not None and token_id == self.eos_id:
                        rs["done"] = True
                        rs["done_reason"] = "eos"

        # Part5K1.7（1.13）：公共逻辑抽取到 _build_rollout
        all_rollouts: List[Rollout] = []
        for rs in rollout_states:
            rollout = self._build_rollout(
                prompt=rs["prompt"],
                prompt_tokens=list(rs["prompt_tokens"]),
                generated_tokens=rs["generated_tokens"],
                logprobs=rs["logprobs"],
                values=rs["values"],
                done=rs["done"],
                done_reason=rs["done_reason"],
                decode_fn=decode_fn,
                reward_fn=reward_fn,
            )
            all_rollouts.append(rollout)

        return all_rollouts

    # ------------------------------------------------------------------
    # 公共逻辑：构造 Rollout（bootstrap value + decode + reward）
    # ------------------------------------------------------------------

    def _build_rollout(
        self,
        prompt: str,
        prompt_tokens: List[int],
        generated_tokens: List[int],
        logprobs: List[float],
        values: List[float],
        done: bool,
        done_reason: str,
        decode_fn,
        reward_fn,
    ) -> Rollout:
        """构造 Rollout 对象（collect 与 collect_batched 公共逻辑）。

        处理：
        1. truncated 判定（not done → max_len 截断）
        2. 截断时多采集一个 V(s_T) 用于 GAE bootstrap
        3. 解码生成文本
        4. 计算 reward

        Args:
            prompt: 原始 prompt 文本
            prompt_tokens: prompt token id 列表
            generated_tokens: 生成的 token id 列表
            logprobs: 每步 logprob 列表
            values: 每步 value 列表（截断时会 append bootstrap value）
            done: 是否 eos 终止
            done_reason: 终止原因（"eos" / "max_len"）
            decode_fn: 解码函数
            reward_fn: reward 函数

        Returns:
            Rollout 对象
        """
        truncated = not done  # max_len 截断为 True，eos 终止为 False
        final_done_reason = done_reason if done else "max_len"
        values = list(values)  # 避免修改调用方列表

        # 截断时多采集一个 V(s_T) 用于 GAE bootstrap
        if truncated and self.value_fn is not None:
            all_tokens_final = list(prompt_tokens) + generated_tokens
            if not all_tokens_final:
                all_tokens_final = [0]
            input_ids_final = np.asarray([all_tokens_final], dtype=np.int64)
            with no_grad():
                try:
                    v = float(self.value_fn(self.agent, input_ids_final))
                    values.append(v)
                except Exception:
                    values.append(0.0)

        # 解码生成文本
        generated_text = decode_fn(generated_tokens)

        # 计算 reward
        if reward_fn is not None:
            reward_result = reward_fn(
                generated_text, prompt, logprobs, generated_tokens,
            )
            if isinstance(reward_result, dict):
                reward = float(reward_result.get("total", 0.0))
                reward_components = reward_result
            else:
                reward = float(reward_result)
                reward_components = {"total": reward}
        else:
            # 默认 reward: 基于长度的简单奖励
            reward = min(1.0, len(generated_tokens) / max(1, self.max_new_tokens))
            reward_components = {"total": reward, "length": reward}

        return Rollout(
            prompt=prompt,
            prompt_tokens=list(prompt_tokens),
            generated_tokens=generated_tokens,
            logprobs=logprobs,
            values=values,
            reward=reward,
            reward_components=reward_components,
            done_reason=final_done_reason,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # 默认编码/解码函数
    # ------------------------------------------------------------------

    def _default_encode(self, text: str) -> List[int]:
        """默认编码：用 agent 的 tokenizer 或 ord fallback。"""
        tokenizer = getattr(self.agent, "_tokenizer", None)
        if tokenizer is not None:
            try:
                return list(tokenizer.encode(text))
            except Exception:
                pass
        return [ord(c) % 256 for c in text]

    def _default_decode(self, token_ids: List[int]) -> str:
        """默认解码：用 agent 的 tokenizer 或 chr fallback。"""
        tokenizer = getattr(self.agent, "_tokenizer", None)
        if tokenizer is not None:
            try:
                return tokenizer.decode(token_ids)
            except Exception:
                pass
        return "".join(chr(int(t) % 256) for t in token_ids)
