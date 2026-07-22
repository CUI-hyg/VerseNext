"""NexEnv: RL 环境抽象 + 具体实现。

包含：
- NexEnv: 任务环境抽象基类
- ChatEnv: 对话任务环境（prompt-completion 评估）
- MathEnv: 数学任务环境（精确匹配 reward）
- CodeEnv: 代码续写环境（语法检查 reward）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state import NexState


class NexEnv(ABC):
    """RL 任务环境抽象基类。

    子类需实现：
    - reset() -> observation
    - step(action) -> (observation, reward, done, info)
    """

    def __init__(self, tokenizer=None):
        """初始化环境。

        Args:
            tokenizer: 可选的 tokenizer，用于 encode/decode
        """
        self.tokenizer = tokenizer

    @abstractmethod
    def reset(self, prompt: str = "") -> NexState:
        """重置环境，返回初始状态。

        Args:
            prompt: 输入提示

        Returns:
            初始 NexState
        """
        ...

    @abstractmethod
    def step(self, state: NexState, action_token: int) -> Tuple[NexState, float, bool, dict]:
        """执行一步。

        Args:
            state: 当前状态
            action_token: 选择的 token id

        Returns:
            (next_state, reward, done, info)
        """
        ...

    def decode(self, token_ids: List[int]) -> str:
        """解码 token 列表为文本。"""
        if self.tokenizer is not None:
            try:
                return self.tokenizer.decode(token_ids)
            except Exception:
                pass
        # fallback: 简单 chr
        return "".join(chr(int(t) % 256) for t in token_ids)

    def encode(self, text: str) -> List[int]:
        """编码文本为 token 列表。"""
        if self.tokenizer is not None:
            try:
                return list(self.tokenizer.encode(text))
            except Exception:
                pass
        # fallback: 简单 ord
        return [ord(c) % 256 for c in text]


class ChatEnv(NexEnv):
    """对话任务环境。

    用 prompt-completion 评估生成质量。
    每步追加 token，达到 max_len 或 eos 时计算 reward。
    """

    def __init__(
        self,
        reference_completion: str = "",
        max_len: int = 32,
        eos_id: Optional[int] = None,
        tokenizer=None,
        reward_fn=None,
    ):
        super().__init__(tokenizer)
        self.reference_completion = reference_completion
        self.max_len = int(max_len)
        self.eos_id = eos_id
        self.reward_fn = reward_fn  # 可选的自定义 reward 函数

    def reset(self, prompt: str = "") -> NexState:
        """重置为初始状态。"""
        prompt_tokens = self.encode(prompt) if prompt else []
        return NexState(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=[],
            step=0,
            done=False,
        )

    def step(self, state: NexState, action_token: int) -> Tuple[NexState, float, bool, dict]:
        """执行一步。"""
        state.append_action(action_token, 0.0)  # logprob 由 agent 填充

        done = False
        info: Dict[str, Any] = {}

        if self.eos_id is not None and action_token == self.eos_id:
            done = True
            info["reason"] = "eos"
        elif state.step >= self.max_len:
            done = True
            info["reason"] = "max_len"

        # 计算奖励
        reward = 0.0
        if done:
            generated_text = self.decode(state.generated_tokens)
            info["generated_text"] = generated_text
            if self.reward_fn is not None:
                reward = float(self.reward_fn(generated_text, self.reference_completion))
            else:
                # 默认：子串匹配
                if self.reference_completion:
                    reward = 1.0 if self.reference_completion in generated_text else 0.0
                else:
                    # 无参考答案时，reward = 生成长度的归一化值
                    reward = min(1.0, state.step / max(1, self.max_len))

        return state, reward, done, info


class MathEnv(NexEnv):
    """数学任务环境。

    用精确匹配评估数学答案。
    典型场景：prompt = "2+3=", reference = "5"
    """

    def __init__(
        self,
        reference_answer: str = "",
        max_len: int = 16,
        eos_id: Optional[int] = None,
        tokenizer=None,
    ):
        super().__init__(tokenizer)
        self.reference_answer = reference_answer.strip()
        self.max_len = int(max_len)
        self.eos_id = eos_id

    def reset(self, prompt: str = "") -> NexState:
        prompt_tokens = self.encode(prompt) if prompt else []
        return NexState(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=[],
            step=0,
            done=False,
        )

    def step(self, state: NexState, action_token: int) -> Tuple[NexState, float, bool, dict]:
        state.append_action(action_token, 0.0)

        done = False
        info: Dict[str, Any] = {}

        if self.eos_id is not None and action_token == self.eos_id:
            done = True
            info["reason"] = "eos"
        elif state.step >= self.max_len:
            done = True
            info["reason"] = "max_len"

        reward = 0.0
        if done:
            generated_text = self.decode(state.generated_tokens).strip()
            info["generated_text"] = generated_text
            if self.reference_answer:
                # 精确匹配
                reward = 1.0 if generated_text == self.reference_answer else 0.0
                # 子串匹配（部分奖励）
                if reward == 0.0 and self.reference_answer in generated_text:
                    reward = 0.5
            else:
                reward = 0.0

        return state, reward, done, info


class CodeEnv(NexEnv):
    """代码续写环境。

    用简化语法检查作为 reward。
    """

    def __init__(
        self,
        max_len: int = 64,
        eos_id: Optional[int] = None,
        tokenizer=None,
    ):
        super().__init__(tokenizer)
        self.max_len = int(max_len)
        self.eos_id = eos_id

    def reset(self, prompt: str = "") -> NexState:
        prompt_tokens = self.encode(prompt) if prompt else []
        return NexState(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generated_tokens=[],
            step=0,
            done=False,
        )

    def step(self, state: NexState, action_token: int) -> Tuple[NexState, float, bool, dict]:
        state.append_action(action_token, 0.0)

        done = False
        info: Dict[str, Any] = {}

        if self.eos_id is not None and action_token == self.eos_id:
            done = True
            info["reason"] = "eos"
        elif state.step >= self.max_len:
            done = True
            info["reason"] = "max_len"

        reward = 0.0
        if done:
            code = self.decode(state.generated_tokens)
            info["generated_code"] = code
            reward = self._syntax_check(code)

        return state, reward, done, info

    def _syntax_check(self, code: str) -> float:
        """简化的语法检查。

        检查括号匹配、缩进等基本规则。

        Returns:
            0.0-1.0 的语法质量分数
        """
        if not code.strip():
            return 0.0
        score = 0.0
        checks = 0

        # 检查 1: 括号匹配
        checks += 1
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        balanced = True
        for ch in code:
            if ch in pairs:
                stack.append(ch)
            elif ch in pairs.values():
                if not stack:
                    balanced = False
                    break
                top = stack.pop()
                if pairs[top] != ch:
                    balanced = False
                    break
        if balanced and not stack:
            score += 1.0

        # 检查 2: 无明显语法错误（简化）
        checks += 1
        if "..." not in code and code.count("  ") < 10:
            score += 1.0

        # 检查 3: 长度合理
        checks += 1
        if 0 < len(code.strip()) <= self.max_len * 4:
            score += 1.0

        return score / max(checks, 1)
