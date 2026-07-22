"""Task 5.4.4: StreamingGenerator - 流式生成器。

设计目标
--------
逐步产生 token，支持：

1. **Prefill**：用 ``model.forward_recurrent`` 逐 token 处理 prompt，
   在 SSM/RWKV 的递归状态下「预热」模型；
2. **Decode**：每步把上一步生成的 token 喂给模型，得到下一个 logits，
   交由 ``Sampler`` 采样；
3. **流式 yield**：每生成一个 token 就 ``yield``，便于上层打印 / 推送给客户端。

为什么用 recurrent 而非 parallel
--------------------------------
- recurrent 模式每个 token 只用 O(1) 内存（仅维护固定大小的 SSM 状态），
  不需要保存整个 KV cache 或 T×T attention 矩阵；
- 对于 Mamba-2 / RWKV-7 这类线性复杂度架构，recurrent 是天然推理模式；
- prompt 较长时，recurrent prefill 比 parallel 更省内存（虽然略慢）。

接口约定
--------
``model`` 必须实现：

.. code-block:: python

    def forward_recurrent(self, input_ids, states=None):
        # input_ids: (B, 1) int
        # states: list of per-layer state, or None
        # returns: (logits (B, 1, vocab_size), new_states list)

``verse_nex.HybridLM`` 已经满足此接口，所以可以直接传给 ``StreamingGenerator``。

可选的 ``state_cache``
---------------------
若用户希望在外部持有 state（例如多请求之间复用），可传入 ``StateCache`` 实例。
生成器会在每步之后调用 ``state_cache.set_all(new_states)`` 同步状态。
若不传，则生成器内部用一个普通 list 维护状态。
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Union

import numpy as np

from verse_torch import Tensor, no_grad

from .sampler import Sampler, GreedySampler
from .cache import StateCache


class StreamingGenerator:
    """流式生成器，逐步产生 token。

    Args:
        model: 语言模型（需实现 ``forward_recurrent``）。
        tokenizer: 可选分词器（仅用于 ``generate_from_text`` 便捷方法）。
        sampler: 采样器，默认 ``GreedySampler()``。
        state_cache: 可选 ``StateCache``，用于在外部持有状态；若为 None 则内部维护。

    用法
    ----
        gen = StreamingGenerator(model, tokenizer, sampler=Sampler(temperature=0.8))
        for token_id in gen.generate(prompt_ids, max_new_tokens=100):
            print(tokenizer.decode([token_id]), end="", flush=True)
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        sampler: Optional[Sampler] = None,
        state_cache: Optional[StateCache] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler if sampler is not None else GreedySampler()
        self.state_cache = state_cache
        # 内部状态：list of per-layer state
        self._states = None
        # 最后一次 forward 的 logits（用于外部调试）
        self._last_logits: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 主入口：generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: Union[list[int], np.ndarray, Tensor],
        max_new_tokens: Optional[int] = None,
        yield_tokens: bool = True,
        eos_token_id: Optional[int] = None,
        reset_state: bool = True,
        max_safe_limit: int = 100_000,
    ) -> Iterator[int]:
        """从 prompt 流式生成 token。

        Part4K2 Task 3 升级：默认不限长度（``max_new_tokens=None``），生成到
        ``eos_token_id`` 自然停止；达到 ``max_safe_limit`` 安全上限时强制停止
        以防无限循环。

        Args:
            prompt_ids: prompt 的 token id 序列（list / ndarray / 1D Tensor）。
            max_new_tokens: 最大生成 token 数；``None`` 表示不限（生成到
                ``eos_token_id`` 自然停止，或达到 ``max_safe_limit`` 安全上限）。
                指定值时按值生成（兼容旧调用）。
            yield_tokens: 若为 True，每生成一个 token 就 yield；
                若为 False，把所有 token 收集到 list 返回（仍用 generator 语法）。
            eos_token_id: 可选的 EOS token id；若生成的 token 等于它，提前停止。
            reset_state: 是否在开始时重置状态（默认 True）。
                若为 False，复用上一次 generate 结束时的状态（多轮对话场景）。
            max_safe_limit: 安全上限（默认 100K），防止无限循环；仅当
                ``max_new_tokens is None`` 时生效。

        Yields:
            int: 每个新生成的 token id
        """
        # 无限生成模式下：max_safe_limit 充当上限；旧调用按 max_new_tokens 限制
        effective_limit = max_safe_limit if max_new_tokens is None else int(max_new_tokens)
        if effective_limit <= 0:
            return

        # 1. 把 prompt_ids 转为 ndarray (T,)
        prompt_arr = self._to_id_array(prompt_ids)
        T = prompt_arr.shape[0]

        # 2. 重置状态（若需要）
        if reset_state:
            self._states = None
            if self.state_cache is not None:
                self.state_cache.reset()

        with no_grad():
            # 3. Prefill：逐 token 处理 prompt，更新 states
            #    用 recurrent 模式：每步 (B=1, T=1) -> (logits, new_states)
            if T > 0:
                logits = None
                for t in range(T):
                    tok = int(prompt_arr[t])
                    logits, self._states = self._step_recurrent(tok, self._states)
                # 最后一个 prompt token 的 logits 用于第一次采样
                last_logits = logits
            else:
                # 空 prompt：用一个零 token 启动（仅用于无 prompt 的极端场景）
                last_logits = np.zeros(
                    self._vocab_size(), dtype=np.float32
                )

            # 同步到外部 state_cache
            self._sync_cache()

            # 4. 自回归生成
            next_token = self.sampler.sample(last_logits)
            for _ in range(effective_limit):
                if yield_tokens:
                    yield next_token
                # EOS 提前停止
                if eos_token_id is not None and next_token == eos_token_id:
                    return
                # 单步：把上一步生成的 token 喂给模型
                logits, self._states = self._step_recurrent(next_token, self._states)
                self._last_logits = logits
                self._sync_cache()
                next_token = self.sampler.sample(logits)

    # ------------------------------------------------------------------
    # 便捷：从文本生成
    # ------------------------------------------------------------------

    def generate_from_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        add_special_tokens: bool = False,
        eos_token_id: Optional[int] = None,
        max_safe_limit: int = 100_000,
    ) -> Iterator[str]:
        """从文本 prompt 流式生成（yield 解码后的字符串片段）。

        Part4K2 Task 3 升级：默认不限长度（``max_new_tokens=None``），生成到
        ``eos_token_id`` 自然停止；达到 ``max_safe_limit`` 安全上限时强制停止。

        Args:
            prompt: 文本 prompt
            max_new_tokens: 最大生成 token 数；``None`` 表示不限（生成到
                ``eos_token_id`` 自然停止，或达到 ``max_safe_limit`` 安全上限）。
            add_special_tokens: encode prompt 时是否加 special tokens（默认 False，
                避免在中间插入 EOS）
            eos_token_id: 可选 EOS id
            max_safe_limit: 安全上限（默认 100K），防止无限循环。

        Yields:
            str: 每个新生成 token 解码后的字符串片段
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is None; pass a tokenizer to StreamingGenerator")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        for token_id in self.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            max_safe_limit=max_safe_limit,
        ):
            yield self.tokenizer.decode([token_id])

    # ------------------------------------------------------------------
    # 单步：内部辅助
    # ------------------------------------------------------------------

    def _step(self, prev_token_id: int) -> int:
        """单步生成：input last token -> output logits -> sample -> next token。

        注意：此方法 **不会** 更新 ``self._states``；若需要持久化状态，
        请用 ``generate`` 或直接调用 ``_step_recurrent``。

        Args:
            prev_token_id: 上一个 token id

        Returns:
            int: 采样得到的下一个 token id
        """
        logits, _ = self._step_recurrent(prev_token_id, self._states)
        self._last_logits = logits
        return self.sampler.sample(logits)

    def _step_recurrent(self, token_id: int, states):
        """调用 model.forward_recurrent 单步推进。

        Args:
            token_id: 当前 token id
            states: 当前状态（list 或 None）

        Returns:
            logits: (vocab_size,) ndarray
            new_states: list of per-layer state
        """
        # 构造 (B=1, T=1) 的 input_ids
        input_ids = Tensor(np.array([[int(token_id)]], dtype=np.int64), requires_grad=False)
        # 调用 model.forward_recurrent
        out = self.model.forward_recurrent(input_ids, states)
        # HybridLM.forward_recurrent 返回 (logits, new_states)
        if isinstance(out, tuple) and len(out) == 2:
            logits_tensor, new_states = out
        else:
            # 兼容某些只返回 Tensor 的实现
            logits_tensor = out
            new_states = None
        # logits_tensor: (B=1, T=1, vocab_size)
        if hasattr(logits_tensor, "data"):
            logits = logits_tensor.data
        else:
            logits = np.asarray(logits_tensor)
        # 取 (T=1, vocab_size) 的最后一行
        if logits.ndim == 3:
            logits = logits[0, -1, :]
        elif logits.ndim == 2:
            logits = logits[-1, :]
        elif logits.ndim == 1:
            pass
        else:
            logits = logits.flatten()
        return logits.astype(np.float32, copy=False), new_states

    def _sync_cache(self) -> None:
        """把内部 states 同步到外部 state_cache（如果有）。"""
        if self.state_cache is not None and self._states is not None:
            try:
                self.state_cache.set_all(self._states)
            except Exception:
                # 长度不匹配等，忽略
                pass

    def _vocab_size(self) -> int:
        """从 model 推断 vocab_size（用于空 prompt 的兜底）。"""
        # HybridLM 有 vocab_size 属性
        if hasattr(self.model, "vocab_size"):
            return int(self.model.vocab_size)
        # 从 lm_head 推断
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            return int(self.model.lm_head.out_features)
        # 从 embed 推断
        if hasattr(self.model, "embed"):
            return int(self.model.embed.num_embeddings)
        # 兜底
        return 256

    @staticmethod
    def _to_id_array(prompt_ids) -> np.ndarray:
        """把 list / ndarray / 1D Tensor 统一转为 1D int64 ndarray。"""
        if isinstance(prompt_ids, Tensor):
            arr = prompt_ids.data
        elif isinstance(prompt_ids, np.ndarray):
            arr = prompt_ids
        else:
            arr = np.asarray(prompt_ids)
        # 展平（如果是 (B, T) 也取第一 batch）
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.int64, copy=False)


__all__ = ["StreamingGenerator"]
