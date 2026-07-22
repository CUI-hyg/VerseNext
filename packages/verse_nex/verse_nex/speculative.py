"""VerseNex: 分离式并行预测（Speculative Decoding 风格）.

Part4K1 Task 3.2 实现：基于 Medusa 风格的多头并行 draft 预测。

核心思想
--------
传统自回归生成每次只能产生一个 token（一次前向 → 一个 argmax），
GEMM/softmax 算力利用率低。Speculative decoding 通过引入 **draft head**
并行预测 k 个候选 token，再让主模型一次前向验证所有 k 个 token：

1. draft_head（k 个并行预测头，每个预测未来第 i+1 个位置的 token）
   给定当前隐状态 hidden 一次性产生 k 个候选 token。
2. 主模型把 [context + k 个候选 token] 拼起来做一次前向，得到每个位置
   的 next-token argmax。
3. **verify-then-commit**：接受最长正确前缀（draft token 与主模型 argmax
   匹配的位置及之前），拒绝处用主模型预测替代，并触发重新 draft。

收益
----
- 主模型前向次数从 k 次降到 1 次（每次接受 ~k 个 token）
- draft_head 是浅 Linear，开销极低
- 算法上等价于 greedy decoding（draft 只是 hint，最终以主模型 argmax 为准）

复用的项目内已有功能
--------------------
- ``verse_torch.Tensor`` / ``verse_torch.nn.Linear / Module / ModuleList``
- 主模型只需要是可调用对象（接受 token id Tensor，返回 logits Tensor），
  与 ``VerseNexLM.forward`` / ``CometSparkNexLM.forward`` 接口对齐。

简化设计
--------
- draft_head 是 k 个独立的 ``Linear(dim, vocab)`` 头（Medusa 风格）。
  每个 head i 预测位置 current + i + 1 的 token。
- 可选共享 embedding（``shared_embedding``）作为 head 的输入特征；
  未提供时直接使用主模型 forward 输出的隐状态。
- verify 用主模型一次前向比较 logits argmax。
- ``verify_then_commit`` 返回 (接受的 token 列表, 是否全部接受)。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import Linear, Module, ModuleList


# ---------------------------------------------------------------------------
# SpeculativeDecoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder(Module):
    """分离式并行预测解码器（Medusa 风格）。

    Args:
        dim: 模型隐藏维度
        vocab_size: 词表大小
        num_draft_heads: 并行预测头数 k（默认 4），每个头预测未来第 i+1 个 token
        shared_embedding: 可选共享 Embedding（用于把 token id 投影回特征空间
            再喂入 draft head；为 None 时 draft 直接接受主模型隐状态）

    Attributes:
        draft_heads: ``ModuleList`` of ``Linear(dim, vocab)``，长度 = k

    用法::

        decoder = SpeculativeDecoder(dim=384, vocab_size=32000, num_draft_heads=4)
        # 1. 主模型 forward 获取隐状态
        logits, hidden = main_model(idx, return_hidden=True)
        # 2. draft 并行生成 k 个候选
        draft_tokens, draft_logits = decoder.draft(hidden)
        # 3. 主模型一次前向验证
        accepted, main_pred = decoder.verify(draft_tokens, main_model, idx)
        # 4. 接受最长正确前缀
        accepted_tokens, all_ok = decoder.verify_then_commit(
            draft_tokens, main_model, idx
        )
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_draft_heads: int = 4,
        shared_embedding: Optional[Module] = None,
    ):
        super().__init__()
        if num_draft_heads < 1:
            raise ValueError(
                f"num_draft_heads 必须 >= 1，got {num_draft_heads}"
            )
        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.num_draft_heads = int(num_draft_heads)
        self.shared_embedding = shared_embedding
        # k 个并行预测头：head_i 预测位置 current + i + 1
        # 偏置置 False 与项目内其它 head 一致（如 TriSparseAttention 的 wq/wk/wv）
        self.draft_heads = ModuleList(
            [Linear(self.dim, self.vocab_size, bias=False)
             for _ in range(self.num_draft_heads)]
        )

    # ------------------------------------------------------------------
    # draft: 并行生成 k 个候选 token
    # ------------------------------------------------------------------

    def draft(
        self,
        hidden: Union[Tensor, np.ndarray],
        last_position: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Tensor]]:
        """并行生成 k 个候选 token（一次前向，无串行循环）。

        Args:
            hidden: 主模型输出的隐状态
                - 形状 (B, T, D) 时取最后一个位置 → (B, 1, D)
                - 形状 (B, D) 时直接使用
                - 形状 (B, 1, D) 等价于 (B, T, D) 取 last
            last_position: 保留参数（与某些 draft 实现对齐），当前未使用

        Returns:
            draft_tokens: (B, k) ndarray，每个 head 的 argmax 预测
            draft_logits: list of k 个 Tensor，每个形状 (B, vocab)
                （保留供训练 draft head 时计算 CE loss）
        """
        # 统一 hidden 为 (B, 1, D) Tensor，取最后一个位置
        if not isinstance(hidden, Tensor):
            hidden = Tensor(np.asarray(hidden, dtype=np.float32))
        if hidden.ndim == 2:
            # (B, D) -> (B, 1, D)
            hidden = hidden.reshape(hidden.shape[0], 1, hidden.shape[1])
        elif hidden.ndim == 3:
            # (B, T, D) -> 取最后一个位置
            hidden = hidden[:, -1:, :]
        elif hidden.ndim == 4:
            # (B, H, T, D) 等奇葩形状：先 flatten 到 (B, T, D) 取 last
            B = hidden.shape[0]
            D = hidden.shape[-1]
            hidden = hidden.reshape(B, -1, D)[:, -1:, :]
        else:
            raise ValueError(
                f"hidden 维度 {hidden.ndim} 不支持，期望 2/3/4 维"
            )

        B = hidden.shape[0]
        draft_logits: List[Tensor] = []
        tokens_list: List[np.ndarray] = []
        # 一次循环展开 k 个 head（k 通常很小，开销可忽略）
        # 每个 head 的 forward 内部走批量 matmul，已并行
        for head in self.draft_heads:
            logits = head(hidden)  # (B, 1, vocab)
            logits = logits[:, 0, :]  # (B, vocab)
            draft_logits.append(logits)
            # argmax 取候选 token（detach，不参与反向）
            tokens_list.append(np.argmax(logits.data, axis=-1))

        # (k, B) -> (B, k)
        draft_tokens = np.stack(tokens_list, axis=1).astype(np.int64)
        return draft_tokens, draft_logits

    # ------------------------------------------------------------------
    # verify: 主模型一次前向验证 k 个候选 token
    # ------------------------------------------------------------------

    def verify(
        self,
        draft_tokens: np.ndarray,
        main_model: Callable[[Tensor], Tensor],
        context_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """主模型一次前向验证 k 个候选 token。

        Args:
            draft_tokens: (B, k) 候选 token ids
            main_model: 主模型可调用对象，接受 (B, T) Tensor 返回 (B, T, vocab) Tensor
                （或返回 (logits, hidden) 元组，本方法自动取 logits）
            context_tokens: (B, T_ctx) 上下文 token ids（不含 draft）

        Returns:
            accepted_mask: (B, k) bool ndarray，每个候选是否被主模型 argmax 验证通过
            main_pred: (B, k) ndarray，主模型在每个 draft 位置预测的 argmax token
        """
        draft_tokens = np.asarray(draft_tokens)
        context_tokens = np.asarray(context_tokens)
        if draft_tokens.ndim != 2 or context_tokens.ndim != 2:
            raise ValueError(
                "draft_tokens / context_tokens 必须为 2D (B, k) / (B, T_ctx)"
            )
        if draft_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                f"batch 不一致：draft {draft_tokens.shape[0]} "
                f"vs context {context_tokens.shape[0]}"
            )
        B, k = draft_tokens.shape
        T_ctx = context_tokens.shape[1]

        # 拼接 [context, draft] -> (B, T_ctx + k)
        # 用 int64 保持 token id 类型一致
        full_tokens = np.concatenate([context_tokens, draft_tokens], axis=1)
        full_t = Tensor(full_tokens.astype(np.int64), requires_grad=False)

        # 主模型一次前向（无梯度，纯推理验证）
        with no_grad():
            out = main_model(full_t)
            # 兼容返回 (logits, hidden) 元组的模型
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            logits_np = np.asarray(logits.data)

        # 主模型在每个 draft 位置 i (i=0..k-1) 预测的 next token
        # 位置 i 对应原序列位置 T_ctx - 1 + i（在拼接序列中的索引）
        # logits 形状 (B, T_ctx + k, vocab)，取 [T_ctx-1 : T_ctx-1+k]
        start = T_ctx - 1
        end = start + k
        main_pred = np.argmax(logits_np[:, start:end, :], axis=-1)  # (B, k)

        # 接受条件：draft_tokens[i] == main_pred[i]
        accepted_mask = (draft_tokens == main_pred)
        return accepted_mask, main_pred

    # ------------------------------------------------------------------
    # verify_then_commit: 接受最长正确前缀 + 拒绝处重新 draft
    # ------------------------------------------------------------------

    def verify_then_commit(
        self,
        draft_tokens: np.ndarray,
        main_model: Callable[[Tensor], Tensor],
        context_tokens: np.ndarray,
        max_redraft_rounds: int = 1,
    ) -> Tuple[List[int], bool, np.ndarray]:
        """接受最长正确前缀；拒绝处用主模型预测替代并触发重新 draft。

        简化策略（与原始 spec decoding 一致）：
        1. 验证 draft_tokens 与主模型 argmax 的匹配情况。
        2. 接受从位置 0 开始的最长连续正确前缀。
        3. 第一个不匹配处用主模型预测替代（"blessed" token，保证正确性）。
        4. 若存在不匹配处且 max_redraft_rounds > 0，则把替代后的序列作为
           新 context，重新 draft（用于 SubTask 3.5 测试）。

        Args:
            draft_tokens: (B, k) 候选 token ids
            main_model: 主模型可调用对象
            context_tokens: (B, T_ctx) 上下文 token ids
            max_redraft_rounds: 最大重新 draft 轮数（默认 1）

        Returns:
            accepted_tokens: list of int（batch 0 接受的 token 列表，
                含 blessed 替代 token）
            all_accepted: bool，是否初始 draft 全部被接受
            final_context: (1, T_ctx + len(accepted_tokens)) ndarray，
                更新后的 context（用于下一轮生成）
        """
        draft_tokens = np.asarray(draft_tokens)
        context_tokens = np.asarray(context_tokens)
        B, k = draft_tokens.shape
        if B != 1:
            # 简化：只处理 batch 0（spec decoding 通常单序列）
            draft_tokens_b = draft_tokens[:1]
            context_tokens_b = context_tokens[:1]
        else:
            draft_tokens_b = draft_tokens
            context_tokens_b = context_tokens

        accepted_mask, main_pred = self.verify(
            draft_tokens_b, main_model, context_tokens_b
        )
        acc_mask = accepted_mask[0]   # (k,) bool
        draft_b0 = draft_tokens_b[0]  # (k,)
        main_b0 = main_pred[0]        # (k,)

        accepted_tokens: List[int] = []
        all_accepted = True
        # 接受最长正确前缀
        for i in range(k):
            if acc_mask[i]:
                accepted_tokens.append(int(draft_b0[i]))
            else:
                # 拒绝处：用主模型预测替代（blessed token）
                accepted_tokens.append(int(main_b0[i]))
                all_accepted = False
                break  # 后续 draft token 作废（spec decoding 协议）

        # 构造更新后的 context：原 context + 接受的 token
        new_ctx_tokens = np.concatenate(
            [context_tokens_b, np.asarray(accepted_tokens, dtype=np.int64)[None, :]],
            axis=1,
        )

        # 若有拒绝处且允许重新 draft，则用更新后的 context 再 draft 一次
        # （仅作 demo / 测试可观测性，不影响 accepted_tokens 已确定的结果）
        if not all_accepted and max_redraft_rounds > 0:
            # 主模型前向取隐状态用于重新 draft
            with no_grad():
                out = main_model(
                    Tensor(new_ctx_tokens.astype(np.int64), requires_grad=False)
                )
                if isinstance(out, tuple):
                    logits_or_hidden = out[1] if len(out) > 1 else out[0]
                else:
                    logits_or_hidden = out
            # 重新 draft（结果不覆盖 accepted_tokens，仅触发流程）
            _ = self.draft(logits_or_hidden)

        return accepted_tokens, all_accepted, new_ctx_tokens

    # ------------------------------------------------------------------
    # 训练辅助：draft head CE loss（可选，用于训练 draft head）
    # ------------------------------------------------------------------

    def draft_loss(
        self,
        draft_logits: List[Tensor],
        target_tokens: np.ndarray,
    ) -> Tensor:
        """计算 draft head 的交叉熵损失（用于训练 draft head）。

        Args:
            draft_logits: list of k 个 Tensor，每个形状 (B, vocab)
            target_tokens: (B, k) 真实的未来 k 个 token

        Returns:
            scalar Tensor loss
        """
        from verse_torch.losses import cross_entropy

        target_tokens = np.asarray(target_tokens)
        if target_tokens.shape[1] != len(draft_logits):
            raise ValueError(
                f"target_tokens 第二维 {target_tokens.shape[1]} "
                f"与 draft_logits 长度 {len(draft_logits)} 不一致"
            )
        total_loss = Tensor(np.zeros((), dtype=np.float32))
        for i, logits in enumerate(draft_logits):
            # logits: (B, vocab)，target: (B,)
            # cross_entropy 期望 (B, C, ...) logits 与 (B, ...) target
            # 这里给 logits 增加一维 seq=1，target 也增加一维
            ce = cross_entropy(
                logits.reshape(logits.shape[0], 1, self.vocab_size),
                target_tokens[:, i].reshape(target_tokens.shape[0], 1),
            )
            total_loss = total_loss + ce
        return total_loss * (1.0 / len(draft_logits))


__all__ = ["SpeculativeDecoder"]
