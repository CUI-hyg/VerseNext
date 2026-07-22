"""NexState: RL 状态抽象。

NexState 封装了 RL 交互中的状态信息，包括：
- prompt: 输入提示文本
- generated_tokens: 已生成的 token id 列表
- kv_cache: 可选的 KV cache（用于高效推理）
- logprobs: 每步生成 token 的对数概率列表

支持并行多 state（batch 化）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class NexState:
    """RL 状态数据类。

    Attributes:
        prompt: 输入提示文本
        prompt_tokens: 输入提示的 token id 列表
        generated_tokens: 已生成的 token id 列表
        kv_cache: 可选 KV cache（dict 或 None）
        logprobs: 每步生成 token 的对数概率列表
        step: 当前生成步数
        done: 是否已完成（到达 eos 或 max_len）
    """

    prompt: str = ""
    prompt_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    kv_cache: Optional[Any] = None
    logprobs: List[float] = field(default_factory=list)
    step: int = 0
    done: bool = False

    @property
    def all_tokens(self) -> List[int]:
        """返回完整 token 序列（prompt + generated）。"""
        return list(self.prompt_tokens) + list(self.generated_tokens)

    @property
    def full_sequence(self) -> List[int]:
        """完整序列的别名。"""
        return self.all_tokens

    def append_action(self, token_id: int, logprob: float) -> None:
        """追加一个生成的动作。"""
        self.generated_tokens.append(int(token_id))
        self.logprobs.append(float(logprob))
        self.step += 1

    def reset(self) -> None:
        """重置状态（保留 prompt，清空生成部分）。"""
        self.generated_tokens = []
        self.logprobs = []
        self.kv_cache = None
        self.step = 0
        self.done = False

    def clone(self) -> "NexState":
        """浅拷贝状态（kv_cache 共享引用）。"""
        return NexState(
            prompt=self.prompt,
            prompt_tokens=list(self.prompt_tokens),
            generated_tokens=list(self.generated_tokens),
            kv_cache=self.kv_cache,
            logprobs=list(self.logprobs),
            step=self.step,
            done=self.done,
        )


def batch_states(states: List[NexState]) -> dict:
    """把多个 NexState batch 化为 dict。

    Returns:
        dict 含：
        - all_tokens: list[list[int]]，每条完整 token 序列
        - prompts: list[str]
        - generated_lens: list[int]
    """
    return {
        "all_tokens": [s.all_tokens for s in states],
        "prompts": [s.prompt for s in states],
        "generated_lens": [len(s.generated_tokens) for s in states],
    }
