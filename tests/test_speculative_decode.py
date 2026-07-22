"""测试：分离式并行预测 / Speculative Decoding（Part4K1 Task 3.5）。

覆盖：
- k=4 候选预测（draft_head 并行生成）
- 接受最长正确前缀
- 拒绝处重新 draft
- draft_head 生成 + verify 流程
"""

from __future__ import annotations

import os
import sys

import numpy as np

# PYTHONPATH 适配
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import pytest

from verse_torch import Tensor, no_grad
from verse_nex import SpeculativeDecoder


# ---------------------------------------------------------------------------
# FakeMainModel: 用于测试的可控主模型
# ---------------------------------------------------------------------------


class FakeMainModel:
    """可控的假主模型：embedding + 简单 linear 投影到 vocab。

    通过预定义的 next_token_map 可以精确控制主模型在每步预测的 token，
    从而测试 verify_then_commit 的接受/拒绝逻辑。
    """

    def __init__(
        self,
        vocab_size: int = 16,
        dim: int = 8,
        next_token_logits: np.ndarray = None,
    ):
        np.random.seed(0)
        self.vocab_size = vocab_size
        self.dim = dim
        # embedding (V, D) 与 head (D, V)
        self.E = np.random.randn(vocab_size, dim).astype(np.float32)
        self.W = np.random.randn(dim, vocab_size).astype(np.float32)
        # 可覆盖的 next-token 预测（强制 argmax）
        # shape (T_max, vocab_size)：在每个位置覆盖 logits
        self.next_token_logits = next_token_logits

    def __call__(self, idx_t: Tensor):
        idx = np.asarray(idx_t.data).astype(np.int64)
        # embedding lookup
        hidden = self.E[idx]  # (B, T, D)
        if self.next_token_logits is not None:
            T = idx.shape[1]
            # 强制每个位置的 logits 为 next_token_logits
            # (B, T, V) = next_token_logits[None, :T, :] broadcast
            logits = np.broadcast_to(
                self.next_token_logits[None, :T, :],
                (idx.shape[0], T, self.vocab_size),
            ).astype(np.float32).copy()
        else:
            logits = hidden @ self.W  # (B, T, V)
        return Tensor(logits, requires_grad=False), Tensor(hidden, requires_grad=False)


# ---------------------------------------------------------------------------
# SubTask 3.5.1: k=4 候选预测
# ---------------------------------------------------------------------------


def test_draft_k4_candidate_shape():
    """draft_head 并行生成 k=4 个候选 token，shape 正确。"""
    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    # 输入隐状态 (B=1, T=1, D=8)
    hidden = Tensor(np.random.randn(1, 1, 8).astype(np.float32), requires_grad=False)
    draft_tokens, draft_logits = decoder.draft(hidden)
    assert draft_tokens.shape == (1, 4)
    assert len(draft_logits) == 4
    for lg in draft_logits:
        assert lg.data.shape == (1, 16)
    # token id 范围合法
    assert np.all(draft_tokens >= 0)
    assert np.all(draft_tokens < 16)


def test_draft_various_hidden_shapes():
    """draft 接受不同形状的 hidden。"""
    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    # (B, D)
    h1 = Tensor(np.random.randn(2, 8).astype(np.float32))
    t1, _ = decoder.draft(h1)
    assert t1.shape == (2, 4)
    # (B, T, D) → 取 last
    h2 = Tensor(np.random.randn(2, 5, 8).astype(np.float32))
    t2, _ = decoder.draft(h2)
    assert t2.shape == (2, 4)
    # (B, 1, D)
    h3 = Tensor(np.random.randn(2, 1, 8).astype(np.float32))
    t3, _ = decoder.draft(h3)
    assert t3.shape == (2, 4)
    # ndarray 也接受
    h4 = np.random.randn(1, 8).astype(np.float32)
    t4, _ = decoder.draft(h4)
    assert t4.shape == (1, 4)


def test_draft_k_other_than_4():
    """k 可以是其它值（如 2 / 8）。"""
    for k in [2, 8]:
        decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=k)
        h = Tensor(np.random.randn(1, 1, 8).astype(np.float32))
        t, lg = decoder.draft(h)
        assert t.shape == (1, k)
        assert len(lg) == k


def test_draft_head_invalid_k():
    """非法 k 应抛 ValueError。"""
    with pytest.raises(ValueError):
        SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=0)
    with pytest.raises(ValueError):
        SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=-1)


# ---------------------------------------------------------------------------
# SubTask 3.5.2: 接受最长正确前缀
# ---------------------------------------------------------------------------


def test_verify_accept_all():
    """draft 与主模型 argmax 全匹配 → 全部接受。"""
    np.random.seed(0)
    # 主模型预测 token = [10, 11, 12, 13]
    # 主模型在拼接序列 [context(3) + draft(4)] 中位置 [2, 3, 4, 5] 处
    # 分别预测 [10, 11, 12, 13]（即 next_logits 的 argmax 序列）
    target_tokens = [10, 11, 12, 13]
    next_logits = np.full((32, 16), -10.0, dtype=np.float32)
    T_ctx = 3
    for i, t in enumerate(target_tokens):
        # 位置 T_ctx - 1 + i 对应预测第 i 个 draft token
        next_logits[T_ctx - 1 + i, t] = 10.0
    main_model = FakeMainModel(
        vocab_size=16, dim=8, next_token_logits=next_logits,
    )

    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    # 让 draft 也预测 [10, 11, 12, 13]
    # 通过手工设置 draft_heads 的权重，使每个 head 的 argmax 是目标 token
    for i, head in enumerate(decoder.draft_heads):
        # 让 head 对 token (10+i) 输出最大 logit
        # 简化：直接把 weight 全置 0，再把 (10+i) 列的 bias 设大
        # 但 head 没有 bias，weight shape (vocab, dim)
        # 设置 weight[10+i, :] = 大正数
        weight_data = np.zeros_like(head.weight.data)
        target_t = 10 + i
        weight_data[target_t, :] = 10.0
        head.weight = Tensor(weight_data, requires_grad=True)

    context = np.array([[1, 2, 3]], dtype=np.int64)
    # 主模型前向获取 hidden（用于 draft）
    with no_grad():
        out = main_model(Tensor(context))
        hidden = out[1]
    draft_tokens, _ = decoder.draft(hidden)
    # 验证 draft tokens 确为 [10, 11, 12, 13]
    assert draft_tokens[0].tolist() == [10, 11, 12, 13], (
        f"draft tokens 不对：{draft_tokens[0]}"
    )

    accepted_tokens, all_ok, _ = decoder.verify_then_commit(
        draft_tokens, main_model, context,
    )
    assert all_ok is True, "应当全部接受"
    assert accepted_tokens == [10, 11, 12, 13]


def test_verify_accept_partial_prefix():
    """draft 前缀部分匹配 → 接受最长正确前缀。"""
    np.random.seed(1)
    # 主模型预测 token = [10, 11, 12, 13]
    target_tokens = [10, 11, 12, 13]
    T_ctx = 3
    next_logits = np.full((32, 16), -10.0, dtype=np.float32)
    for i, t in enumerate(target_tokens):
        next_logits[T_ctx - 1 + i, t] = 10.0
    main_model = FakeMainModel(
        vocab_size=16, dim=8, next_token_logits=next_logits,
    )

    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    # 让 draft 预测 [10, 5, 12, 13] —— 第 2 个位置错（应为 11）
    for i, head in enumerate(decoder.draft_heads):
        weight_data = np.zeros_like(head.weight.data)
        if i == 1:
            # 第 2 个 head 预测错（99 不在 vocab_size=16 内，用 5 代替）
            weight_data[5, :] = 10.0
        else:
            # 其它预测正确
            target_t = 10 + i
            weight_data[target_t, :] = 10.0
        head.weight = Tensor(weight_data, requires_grad=True)

    context = np.array([[1, 2, 3]], dtype=np.int64)
    with no_grad():
        out = main_model(Tensor(context))
        hidden = out[1]
    draft_tokens, _ = decoder.draft(hidden)
    assert draft_tokens[0].tolist() == [10, 5, 12, 13], (
        f"draft tokens 不对：{draft_tokens[0]}"
    )

    accepted_tokens, all_ok, _ = decoder.verify_then_commit(
        draft_tokens, main_model, context, max_redraft_rounds=0,
    )
    # 应该接受 [10]（第一个正确），第二个不匹配 → 用主模型预测 11 替代
    assert all_ok is False, "应当未全部接受"
    assert accepted_tokens == [10, 11], (
        f"接受的 tokens 应为 [10, 11]，got {accepted_tokens}"
    )


def test_verify_reject_first_position():
    """第一个位置就拒绝 → 只返回主模型预测。"""
    np.random.seed(2)
    # 主模型预测 = [5, 6, 7, 8]
    target_tokens = [5, 6, 7, 8]
    T_ctx = 3
    next_logits = np.full((32, 16), -10.0, dtype=np.float32)
    for i, t in enumerate(target_tokens):
        next_logits[T_ctx - 1 + i, t] = 10.0
    main_model = FakeMainModel(
        vocab_size=16, dim=8, next_token_logits=next_logits,
    )

    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    # 让 draft 全部预测错（预测 [0, 1, 2, 3]）
    for i, head in enumerate(decoder.draft_heads):
        weight_data = np.zeros_like(head.weight.data)
        weight_data[i, :] = 10.0  # 预测 token i
        head.weight = Tensor(weight_data, requires_grad=True)

    context = np.array([[1, 2, 3]], dtype=np.int64)
    with no_grad():
        out = main_model(Tensor(context))
        hidden = out[1]
    draft_tokens, _ = decoder.draft(hidden)
    assert draft_tokens[0].tolist() == [0, 1, 2, 3]

    accepted_tokens, all_ok, _ = decoder.verify_then_commit(
        draft_tokens, main_model, context, max_redraft_rounds=0,
    )
    # 第一个位置就拒绝 → 用主模型预测 5 替代，后续作废
    assert all_ok is False
    assert accepted_tokens == [5], (
        f"接受的 tokens 应为 [5]，got {accepted_tokens}"
    )


# ---------------------------------------------------------------------------
# SubTask 3.5.3: 拒绝处重新 draft
# ---------------------------------------------------------------------------


def test_verify_reject_triggers_redraft():
    """拒绝处触发重新 draft（max_redraft_rounds=1）。"""
    np.random.seed(3)
    target_tokens = [10, 11, 12, 13]
    T_ctx = 3
    next_logits = np.full((32, 16), -10.0, dtype=np.float32)
    for i, t in enumerate(target_tokens):
        next_logits[T_ctx - 1 + i, t] = 10.0
    main_model = FakeMainModel(
        vocab_size=16, dim=8, next_token_logits=next_logits,
    )

    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    # 让 draft 预测错（第 0 个位置就错）
    for i, head in enumerate(decoder.draft_heads):
        weight_data = np.zeros_like(head.weight.data)
        weight_data[i, :] = 10.0  # 预测 token i（全部错）
        head.weight = Tensor(weight_data, requires_grad=True)

    context = np.array([[1, 2, 3]], dtype=np.int64)
    with no_grad():
        out = main_model(Tensor(context))
        hidden = out[1]
    draft_tokens, _ = decoder.draft(hidden)
    # draft 预测全错：[0, 1, 2, 3]

    # verify_then_commit 应触发重新 draft（max_redraft_rounds=1）
    # 不应抛异常
    accepted_tokens, all_ok, new_ctx = decoder.verify_then_commit(
        draft_tokens, main_model, context, max_redraft_rounds=1,
    )
    assert all_ok is False
    # 接受的 tokens 至少包含主模型替代的第 0 个 token
    assert accepted_tokens[0] == 10
    # new_ctx 应包含原 context + 接受的 token
    assert new_ctx.shape[1] == context.shape[1] + len(accepted_tokens)


# ---------------------------------------------------------------------------
# SubTask 3.5.4: draft_head 生成 + verify 完整流程
# ---------------------------------------------------------------------------


def test_full_draft_verify_pipeline():
    """端到端：draft → verify → accepted_mask 与 draft_logits 形状正确。"""
    np.random.seed(10)
    V, D = 32, 16
    main_model = FakeMainModel(vocab_size=V, dim=D)
    decoder = SpeculativeDecoder(dim=D, vocab_size=V, num_draft_heads=4)

    context = np.array([[5, 6, 7, 8, 9]], dtype=np.int64)
    with no_grad():
        out = main_model(Tensor(context))
        hidden = out[1]
    draft_tokens, draft_logits = decoder.draft(hidden)

    # verify 一次
    accepted_mask, main_pred = decoder.verify(draft_tokens, main_model, context)
    assert accepted_mask.shape == (1, 4)
    assert main_pred.shape == (1, 4)
    assert accepted_mask.dtype == np.bool_ or accepted_mask.dtype == bool

    # verify_then_commit 完整流程
    accepted_tokens, all_ok, new_ctx = decoder.verify_then_commit(
        draft_tokens, main_model, context,
    )
    assert isinstance(accepted_tokens, list)
    assert len(accepted_tokens) >= 1  # 至少接受 1 个（主模型替代）
    assert len(accepted_tokens) <= 4  # 至多 k=4
    assert isinstance(all_ok, bool)
    assert new_ctx.shape[0] == 1
    assert new_ctx.shape[1] == context.shape[1] + len(accepted_tokens)


def test_draft_loss_for_training():
    """draft_loss 接口可计算（用于训练 draft head）。"""
    from verse_torch import Tensor
    np.random.seed(20)
    V, D = 16, 8
    decoder = SpeculativeDecoder(dim=D, vocab_size=V, num_draft_heads=4)

    hidden = Tensor(np.random.randn(2, 1, D).astype(np.float32), requires_grad=False)
    draft_tokens, draft_logits = decoder.draft(hidden)

    # 假设真实未来 k 个 token = 随机生成
    target = np.random.randint(0, V, size=(2, 4))
    loss = decoder.draft_loss(draft_logits, target)
    assert loss.data.shape == ()  # 标量
    assert float(loss.data) > 0  # 损失应为正
    # 反向可流
    loss.backward()
    for head in decoder.draft_heads:
        assert head.weight.grad is not None


# ---------------------------------------------------------------------------
# 边界与错误处理
# ---------------------------------------------------------------------------


def test_verify_batch_mismatch_raises():
    """batch 维度不一致应抛 ValueError。"""
    decoder = SpeculativeDecoder(dim=8, vocab_size=16, num_draft_heads=4)
    draft = np.array([[1, 2, 3, 4]], dtype=np.int64)  # B=1
    context = np.array([[1, 2], [3, 4]], dtype=np.int64)  # B=2
    with pytest.raises(ValueError):
        decoder.verify(draft, lambda x: Tensor(np.zeros((1, 4, 16))), context)


def test_verify_main_model_returns_tuple_or_tensor():
    """verify 兼容主模型返回 Tensor 或 (Tensor, hidden) 元组。"""
    V, D = 16, 8

    # 1. 返回 Tensor
    class M1:
        def __call__(self, idx_t):
            B, T = idx_t.data.shape
            return Tensor(np.random.randn(B, T, V).astype(np.float32))

    # 2. 返回 (Tensor, hidden)
    class M2:
        def __call__(self, idx_t):
            B, T = idx_t.data.shape
            return (
                Tensor(np.random.randn(B, T, V).astype(np.float32)),
                Tensor(np.random.randn(B, T, D).astype(np.float32)),
            )

    decoder = SpeculativeDecoder(dim=D, vocab_size=V, num_draft_heads=4)
    context = np.array([[1, 2, 3]], dtype=np.int64)
    draft = np.array([[5, 6, 7, 8]], dtype=np.int64)

    for m in [M1(), M2()]:
        acc, pred = decoder.verify(draft, m, context)
        assert acc.shape == (1, 4)
        assert pred.shape == (1, 4)
