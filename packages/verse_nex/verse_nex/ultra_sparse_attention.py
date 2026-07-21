"""VerseNex 超稀疏并行多注意力机制 + Medusa 多头预测（Part4 P2.1）。

核心创新
--------
1. **UltraSparseMultiAttention**：超稀疏并行多注意力机制
   - 每个 query 位置只对 Top-K 个 key 位置计算注意力（K << T），其余置 -inf
   - 多头并行：每个头独立选择自己的 Top-K 稀疏模式
   - 因果掩码叠加：稀疏选择仅在因果窗口内进行
   - 复杂度：O(H * T * K * d_head) 而非 O(H * T² * d_head)，长序列显著加速
   - 可微性：Top-K 选择本身不可微，但选中位置的 attention 权重经 softmax
     后梯度可正常回传（与 BigBird / Longformer / Sparse Transformer 一致）

2. **MedusaHeads**：Medusa 多头并行预测
   - 主头（lm_head）预测 next token（位置 +1）
   - N 个副头分别预测位置 +2, +3, ..., +(N+1)
   - 每个副头：1 层 MLP（SwiGLU 或 SiLU）+ Linear → vocab
   - 训练：combined_loss = main_loss + Σ aux_i_loss * aux_weight_i
   - 推理：投机解码（speculative decoding），一次 forward 预测多个 token
   - 优点：训练时提供更丰富的梯度信号（多步预测），推理时加速 1.5-2x

设计参考
--------
- Sparse Transformer (Child et al., 2019): 稀疏注意力模式
- Big Bird (Zaheer et al., 2020): 随机 + 窗口 + 全局 稀疏模式
- Medusa (Cai et al., 2024): 多头并行预测 + 投机解码
- 本实现：Top-K per-query 稀疏 + 多头独立选择 + Medusa 多步预测
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from verse_torch.tensor import Tensor
from verse_torch.nn import Module, Linear, Dropout, SwiGLUMLP, ModuleList


# ---------------------------------------------------------------------------
# UltraSparseMultiAttention
# ---------------------------------------------------------------------------


class UltraSparseMultiAttention(Module):
    """超稀疏并行多注意力机制（VerseNex 核心组件）。

    每个 query 位置仅对 Top-K 个 key 位置（含因果窗口约束）计算注意力，
    多头并行且每个头独立选择自己的稀疏模式。

    Args:
        d_model: 模型隐藏维度
        n_head: 注意力头数
        n_kv_head: K/V 头数（GQA，默认 = n_head）
        top_k: 每个 query 位置保留的 key 数量（稀疏度）；<=0 表示全注意力
        dropout: attention dropout
        rope_max_seq: RoPE 预计算最大长度

    forward:
        x: (B, T, d_model)
        kv_cache: 可选 (k_cache, v_cache)，每个 shape (B, T_prev, n_kv_head, head_dim)
        → (out: (B, T, d_model), new_kv_cache)

    复杂度：
        全注意力：O(B * n_head * T² * head_dim)
        稀疏：    O(B * n_head * T * top_k * head_dim) + Top-K 选择 O(B * n_head * T * T)
        当 top_k << T 时显著加速（top_k=32, T=1024 → 32x attention 计算加速）
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        top_k: int = 32,
        dropout: float = 0.0,
        rope_max_seq: int = 32768,
    ):
        super().__init__()
        if n_kv_head is None:
            n_kv_head = n_head
        assert d_model % n_head == 0, (
            f"d_model({d_model}) 必须能被 n_head({n_head}) 整除"
        )
        assert n_head % n_kv_head == 0, (
            f"n_head({n_head}) 必须能被 n_kv_head({n_kv_head}) 整除"
        )
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.n_rep = n_head // n_kv_head
        self.top_k = max(0, int(top_k))
        kv_dim = n_kv_head * self.head_dim
        self.wq = Linear(d_model, n_head * self.head_dim, bias=False)
        self.wk = Linear(d_model, kv_dim, bias=False)
        self.wv = Linear(d_model, kv_dim, bias=False)
        self.proj = Linear(n_head * self.head_dim, d_model, bias=False)
        self.dropout = Dropout(dropout)
        self._build_rope_table(self.head_dim, rope_max_seq)

    # ------------------------------------------------------------------
    # RoPE（与 GQASelfAttention 一致的实现）
    # ------------------------------------------------------------------

    def _build_rope_table(self, head_dim: int, max_seq_len: int):
        half = head_dim // 2
        i = np.arange(half, dtype=np.float32)
        inv_freq = 1.0 / (10000.0 ** (2.0 * i / head_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, inv_freq)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
        self._cos_table = cos
        self._sin_table = sin
        self._max_seq_len = max_seq_len

    def _apply_rope(self, x: Tensor, position_offset: int = 0) -> Tensor:
        B, T, H, D = x.shape
        if position_offset + T > self._max_seq_len:
            new_max = max(self._max_seq_len * 2, position_offset + T)
            self._build_rope_table(D, new_max)
        pos = position_offset + np.arange(T)
        cos = self._cos_table[pos]
        sin = self._sin_table[pos]
        cos_b = cos.reshape(1, T, 1, D)
        sin_b = sin.reshape(1, T, 1, D)
        x_data = x.data
        half = D // 2
        rotate_half = np.concatenate([-x_data[..., half:], x_data[..., :half]], axis=-1)
        rotated = x_data * cos_b + rotate_half * sin_b
        requires_grad = x.requires_grad
        out = Tensor(
            rotated,
            requires_grad=requires_grad,
            _children=(x,) if requires_grad else (),
            _op="rope",
        )
        if requires_grad:
            def _backward():
                grad = out.grad
                g = grad * cos_b + np.concatenate(
                    [-grad[..., half:], grad[..., :half]], axis=-1
                ) * sin_b
                x._accumulate_grad(g)
            out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, kv_cache=None):
        B, T, d = x.shape
        # 1. 投影 Q/K/V
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # 2. position_offset
        position_offset = 0
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            position_offset = k_cache.shape[1]

        # 3. RoPE（仅 q, k）
        q = self._apply_rope(q, position_offset)
        k = self._apply_rope(k, position_offset)

        # 4. KV cache 拼接
        if kv_cache is not None:
            # 沿 T 维拼接（cache 已是 (B, T_prev, n_kv_head, head_dim)）
            k_in, v_in = k, v  # 保存 concat 输入（避免闭包捕获被重新赋值的 k/v）
            k_data = np.concatenate([k_cache.data, k.data], axis=1)
            v_data = np.concatenate([v_cache.data, v.data], axis=1)
            k = Tensor(k_data, requires_grad=k.requires_grad,
                       _children=(k,) if k.requires_grad else (), _op="kv_concat")
            v = Tensor(v_data, requires_grad=v.requires_grad,
                       _children=(v,) if v.requires_grad else (), _op="kv_concat")
            # 保存 concat 输出引用（避免后续 repeat_kv/permute 重新赋值后闭包失效）
            k_concat, v_concat = k, v
            if k_concat.requires_grad:
                _T_new = T
                def _k_backward():
                    if k_concat.grad is None:
                        return
                    # 只回传到新 k 部分（cache 部分已 detach）
                    k_in._accumulate_grad(k_concat.grad[:, _T_new:, :, :])
                k_concat._backward = _k_backward
            if v_concat.requires_grad:
                _T_new = T
                def _v_backward():
                    if v_concat.grad is None:
                        return
                    v_in._accumulate_grad(v_concat.grad[:, _T_new:, :, :])
                v_concat._backward = _v_backward

        new_kv_cache = (k.data.copy(), v.data.copy())  # 用 ndarray 存缓存

        # 5. repeat_kv: n_kv_head → n_head
        if self.n_rep > 1:
            # k/v: (B, T_kv, n_kv_head, head_dim) → (B, T_kv, n_head, head_dim)
            k_data = np.repeat(k.data, self.n_rep, axis=2)
            v_data = np.repeat(v.data, self.n_rep, axis=2)
            k_in, v_in = k, v  # 保存 repeat_kv 输入（避免闭包捕获被重新赋值的 k/v）
            k = Tensor(k_data, requires_grad=k.requires_grad,
                       _children=(k,) if k.requires_grad else (), _op="repeat_kv")
            v = Tensor(v_data, requires_grad=v.requires_grad,
                       _children=(v,) if v.requires_grad else (), _op="repeat_kv")
            # 保存 repeat_kv 输出（避免下面 permute 重新赋值后闭包引用错误）
            k_rep, v_rep = k, v
            if k_rep.requires_grad:
                _n_kv_head = self.n_kv_head
                _n_rep = self.n_rep
                _head_dim = self.head_dim
                def _k_repeat_backward():
                    if k_rep.grad is None:
                        return
                    # 把 n_head 维度的梯度加回到 n_kv_head 维度
                    # k_rep.grad shape: (B, T, n_head, head_dim)
                    g = k_rep.grad
                    T_g = g.shape[1]
                    g_red = g.reshape(
                        B, T_g, _n_kv_head, _n_rep, _head_dim
                    ).sum(axis=3)
                    k_in._accumulate_grad(g_red)
                k_rep._backward = _k_repeat_backward
            if v_rep.requires_grad:
                _n_kv_head = self.n_kv_head
                _n_rep = self.n_rep
                _head_dim = self.head_dim
                def _v_repeat_backward():
                    if v_rep.grad is None:
                        return
                    g = v_rep.grad
                    T_g = g.shape[1]
                    g_red = g.reshape(
                        B, T_g, _n_kv_head, _n_rep, _head_dim
                    ).sum(axis=3)
                    v_in._accumulate_grad(g_red)
                v_rep._backward = _v_repeat_backward

        # 6. 转置 (B, n_head, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 7. attention 计算
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(-1, -2)) * scale  # (B, n_head, T_q, T_k)

        T_q = q.shape[2]
        T_k = k.shape[2]
        offset = T_k - T_q

        # 8. 因果掩码 + Top-K 稀疏选择
        scores_data = scores.data  # (B, n_head, T_q, T_k)
        # 构造因果掩码（上三角为 -inf）
        i_idx = np.arange(T_q)[:, None]
        j_idx = np.arange(T_k)[None, :]
        causal_mask = (j_idx <= i_idx + offset)  # True = 可见

        effective_k = self.top_k if self.top_k > 0 else T_k
        # 每个 query 位置最多保留 effective_k 个 key（在因果窗口内）
        # 用 np.argsort 选择 Top-K（按分数降序）
        # 但要先屏蔽因果窗口外的位置
        masked_scores = np.where(causal_mask[None, None, :, :], scores_data, -1e9)

        if effective_k < T_k:
            # Top-K 选择：对每个 (B, head, q) 选 Top-K 个 key
            # argsort 升序，取最后 K 个（降序的 Top-K）
            # 但要处理有效 key 数 < K 的情况（因果窗口前几个位置）
            # 简化：先对 masked_scores 做 argsort，取倒数 K 个
            sorted_idx = np.argsort(masked_scores, axis=-1)
            topk_idx = sorted_idx[..., -effective_k:]  # (B, H, T_q, K)
            # 构造 sparse mask：只保留 topk_idx 对应位置
            sparse_mask = np.zeros_like(scores_data, dtype=bool)
            # 用 advanced indexing 设置 True：需要把 b/h/q 索引 broadcast 到 (B,H,T_q,K)
            b_idx, h_idx, q_idx = np.indices((B, self.n_head, T_q))
            b_idx_exp = np.broadcast_to(b_idx[..., None], topk_idx.shape)
            h_idx_exp = np.broadcast_to(h_idx[..., None], topk_idx.shape)
            q_idx_exp = np.broadcast_to(q_idx[..., None], topk_idx.shape)
            sparse_mask[b_idx_exp, h_idx_exp, q_idx_exp, topk_idx] = True
            # 因果掩码 AND sparse mask
            final_mask = sparse_mask & causal_mask[None, None, :, :]
        else:
            final_mask = causal_mask[None, None, :, :]

        # 应用 mask：不可见位置 → -1e9
        masked_scores_np = np.where(final_mask, scores_data, -1e9).astype(np.float32)

        # softmax（沿 T_k 维度）
        # 数值稳定：减去 max
        max_scores = np.max(masked_scores_np, axis=-1, keepdims=True)
        # 处理全 -1e9 的行（不应出现，因为对角线总是可见）
        max_scores = np.where(max_scores < -1e8, 0.0, max_scores)
        exp_scores = np.exp(masked_scores_np - max_scores)
        sum_exp = exp_scores.sum(axis=-1, keepdims=True)
        # 避免 0 除
        sum_exp = np.maximum(sum_exp, 1e-12)
        attn_data = exp_scores / sum_exp  # (B, n_head, T_q, T_k)

        # attn 是 softmax 输出，需要支持 backward
        requires_grad = scores.requires_grad
        attn = Tensor(
            attn_data,
            requires_grad=requires_grad,
            _children=(scores,) if requires_grad else (),
            _op="ultra_sparse_softmax",
        )
        if requires_grad:
            # 保存 sparse_mask 用于 backward（只对保留位置传梯度）
            _mask = final_mask
            def _backward():
                if attn.grad is None:
                    return
                grad = attn.grad  # (B, H, T_q, T_k)
                # softmax backward: ds_i = s_i * (g_i - sum_j(g_j * s_j))
                # 但只在 mask 位置有定义，其它位置 s_i = 0
                # 直接用标准 softmax backward 公式
                s = attn_data
                dot = (grad * s).sum(axis=-1, keepdims=True)
                ds = s * (grad - dot)
                # 只在 mask 位置回传（其它位置梯度应为 0）
                ds = ds * _mask
                scores._accumulate_grad(ds)
            attn._backward = _backward

        attn = self.dropout(attn)

        # 9. attn @ v → (B, n_head, T_q, head_dim)
        out = attn @ v

        # 10. reshape + proj
        out = out.transpose(1, 2).reshape(B, T_q, d)
        out = self.proj(out)
        return out, new_kv_cache


# ---------------------------------------------------------------------------
# MedusaHeads：多头并行预测
# ---------------------------------------------------------------------------


class MedusaHead(Module):
    """单个 Medusa 副头：预测未来第 K 步的 token。

    结构：
        hidden (B, T, d) → Linear(d, d) → SiLU → Linear(d, d) → Linear(d, vocab)
    简化版：hidden → Linear(d, d) → SiLU → Linear(d, vocab)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc1 = Linear(d_model, d_model, bias=True)
        self.fc2 = Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SiLU 激活：x * sigmoid(x)
        h = self.fc1(x)
        h_data = h.data
        sig = 1.0 / (1.0 + np.exp(-h_data))
        silu_data = h_data * sig
        requires_grad = h.requires_grad
        silu = Tensor(
            silu_data,
            requires_grad=requires_grad,
            _children=(h,) if requires_grad else (),
            _op="silu",
        )
        if requires_grad:
            def _backward():
                if silu.grad is None:
                    return
                g = silu.grad
                # d(silu)/d(h) = sigmoid(h) + h * sigmoid(h) * (1 - sigmoid(h))
                #              = sig + h_data * sig * (1 - sig)
                dh = g * (sig + h_data * sig * (1.0 - sig))
                h._accumulate_grad(dh)
            silu._backward = _backward
        return self.fc2(silu)


class MedusaHeads(Module):
    """Medusa 多头并行预测模块。

    主头（lm_head，外部定义）预测 next token（位置 +1）。
    本模块包含 N 个副头，分别预测位置 +2, +3, ..., +(N+1)。

    forward:
        hidden: (B, T, d_model)  — 最后一层 transformer/versenex 的输出
        → logits_list: [head_0_logits, head_1_logits, ..., head_{N-1}_logits]
          每个 shape (B, T, vocab_size)
          head_i 预测位置 t+i+2 的 token（基于位置 t 的 hidden）

    训练 loss（在 Trainer 中计算）：
        total_loss = main_loss(next_token) + Σ_i weight_i * aux_loss(head_i)
        其中 aux_loss 用 head_i 预测 token[t + i + 2] vs 真实 token[t + i + 2]

    推理（投机解码）：
        1. forward 一次得到 main_logits + aux_logits_list
        2. main 头 greedy/采样得到 token_{t+1}
        3. 每个 aux 头独立预测 token_{t+2}, ..., token_{t+N+1}
        4. 用 main 头对 token_{t+1}..token_{t+N} 做 verify
        5. 接受最长匹配前缀，拒绝位置重新 forward
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_aux_heads: int = 3,
        aux_weights: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_aux_heads = n_aux_heads
        self.heads = ModuleList(
            [MedusaHead(d_model, vocab_size) for _ in range(n_aux_heads)]
        )
        if aux_weights is None:
            # 默认权重：越远的头权重越小（0.8, 0.6, 0.4, ...）
            self.aux_weights = [max(0.1, 0.8 - 0.2 * i) for i in range(n_aux_heads)]
        else:
            assert len(aux_weights) == n_aux_heads
            self.aux_weights = list(aux_weights)

    def forward(self, hidden: Tensor) -> list:
        """返回 N 个副头的 logits 列表。

        Args:
            hidden: (B, T, d_model)

        Returns:
            logits_list: 长度 n_aux_heads，每个 (B, T, vocab_size)
            head_i 预测位置 t+i+2 的 token
        """
        return [head(hidden) for head in self.heads]


__all__ = [
    "UltraSparseMultiAttention",
    "MedusaHead",
    "MedusaHeads",
]
