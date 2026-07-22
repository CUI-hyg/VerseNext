"""VerseNex: 三路并行稀疏注意力 (Tri-Sparse Attention).

SWA (Sliding Window) + Global Token + ALiBi 三路并行计算并加权融合。

设计要点：
- 路径 A (SWA)：chunk-wise 滑动窗口，每个 query chunk 只 attend 最近 window_size 个 key，
  避免构造 T² 全张量，内存 O(T * window_size)。
  **Part4K1 Task 3 升级**：多 query chunk 并行计算（批量矩阵化），
  消除原串行 for 循环。每个 chunk 的 attention 计算批量堆叠为
  (n_chunks, B, H, W, K_max) 形式，一次 matmul/softmax 完成。
  GPU 路径委托 torch.bmm / batched matmul；CPU 路径用 numpy 批量 matmul。
  并行结果与原串行结果在 float32 下吻合到 1e-3（_swa_forward_serial 保留作对照）。
- 路径 B (Global)：num_global_tokens 个可学习全局 sink token（Embedding），
  每个 query 只 attend 这些 token，内存 O(T * num_global_tokens)。
- 路径 C (ALiBi)：标准 causal attention + ALiBi 位置偏置。
  T <= 1024 时直接构造 (B, H, T, T)；T > 1024 时降级（gate C 强制为 0）以避免 T² 内存。
- 三路输出按可学习 gate (3,) 加权求和：gate = sigmoid(logits)，logits 初始化为 [0,0,0]。
- 支持 GQA（n_kv_head < n_head），用 repeat_kv 复制 KV head。
- 支持位置编码：use_rope=True 时应用 RoPE（与 GQASelfAttention 一致）。
- forward：整序列并行计算（可微，用于训练）。
- forward_recurrent：单步递推（推理），维护滑动窗口 KV cache 与 global KV。

复用的项目内已有功能：
- verse_torch.nn.Linear / Dropout / Embedding / Module / normal_
- verse_torch.nn._concat（可微 Tensor 拼接，KV cache 用，与 GQASelfAttention 一致）
- verse_torch.nn.repeat_kv（GQA head 复制）
- verse_nex.sparse_attention._pad_last_dim（带梯度的轴向 padding）
- ALiBi slopes 公式与 verse_torch.nn.ALiBi 一致（m_h = 1/2^(h/n_head)，h=1..n_head）
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import (
    Linear,
    Module,
    Dropout,
    Embedding,
    _concat,
    normal_,
    repeat_kv,
)
from .sparse_attention import _pad_last_dim


# ---------------------------------------------------------------------------
# 推理用 numpy 工具（不构建计算图）
# ---------------------------------------------------------------------------


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 numpy softmax（forward_recurrent 中使用）。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 numpy sigmoid（tanh 近似，避免 overflow）。"""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# ---------------------------------------------------------------------------
# TriSparseAttention
# ---------------------------------------------------------------------------


class TriSparseAttention(Module):
    """三路并行稀疏注意力：Sliding Window + Global Token + ALiBi bias。

    三路并行计算并加权融合：
    - 路径 A (SWA)：每个 query chunk 只 attend 最近 window_size 个 key（chunk-wise 实现）
    - 路径 B (Global)：所有 query 都 attend num_global_tokens 个可学习全局 sink token
    - 路径 C (ALiBi)：标准 causal attention + ALiBi 位置偏置

    三路输出按可学习权重 gate (3,) 加权求和。

    Args:
        dim: 模型维度
        n_head: query head 数量
        n_kv_head: key/value head 数量（默认 = n_head；小于 n_head 时为 GQA）
        window_size: 滑动窗口大小（每个 query 最多 attend 最近 window_size 个 key）
        num_global_tokens: 全局 sink token 数量
        use_alibi: 是否启用 ALiBi 路径（路径 C）
        use_rope: 是否对 Q/K 应用 RoPE（v 与 global token 不应用）
        max_seq_len: RoPE 与 ALiBi 预计算的最大序列长度
        dropout: attention 权重 dropout 概率
        rope_theta: RoPE 的 base 频率
    """

    # 路径 C 的序列长度上限：超过则降级为 SWA + Global（gate C 强制为 0）
    _ALIBI_MAX_T = 1024

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int = None,
        window_size: int = 512,
        num_global_tokens: int = 64,
        use_alibi: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if n_kv_head is None:
            n_kv_head = n_head
        assert dim % n_head == 0, f"dim({dim}) 必须能被 n_head({n_head}) 整除"
        assert n_head % n_kv_head == 0, (
            f"n_head({n_head}) 必须能被 n_kv_head({n_kv_head}) 整除"
        )
        assert window_size >= 1, f"window_size 必须为正整数，got {window_size}"
        assert num_global_tokens >= 1, (
            f"num_global_tokens 必须为正整数，got {num_global_tokens}"
        )

        self.dim = dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.n_rep = n_head // n_kv_head
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.use_alibi = use_alibi
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        # QKV 投影与输出投影（bias=False，与 GQASelfAttention 一致）
        kv_dim = n_kv_head * self.head_dim
        self.wq = Linear(dim, n_head * self.head_dim, bias=False)
        self.wk = Linear(dim, kv_dim, bias=False)
        self.wv = Linear(dim, kv_dim, bias=False)
        self.proj = Linear(n_head * self.head_dim, dim, bias=False)
        self.dropout = Dropout(dropout)

        # 可学习全局 sink token（路径 B）：复用 Embedding 作为参数容器
        self.global_tokens = Embedding(num_global_tokens, dim)
        # Embedding 默认 std=1.0 偏大，重新初始化为 0.02 以匹配 Transformer 习惯
        normal_(self.global_tokens.weight, std=0.02)

        # 三路融合 gate（logits 初始化为 [0,0,0]）
        # 注：sigmoid(0)=0.5（而非 0.33）；logits=[0,0,0] 对应等权初始化意图。
        # gate 取值会随训练自适应，这里严格按契约 "logits=[0,0,0]" 初始化。
        self.gate_logits = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)

        # ALiBi slopes（与 verse_torch.nn.ALiBi 一致：m_h = 1/2^(h/n_head)，h=1..n_head）
        slopes = 1.0 / (
            2.0 ** (np.arange(1, n_head + 1, dtype=np.float32) / n_head)
        )
        self.alibi_slopes = slopes  # (n_head,)

        # RoPE 预计算表（仅 use_rope=True 时构建）
        if use_rope:
            self._build_rope_table(self.head_dim, max_seq_len)
        else:
            self._cos_table = None
            self._sin_table = None
            self._rope_max_seq_len = 0

    # ------------------------------------------------------------------
    # RoPE（与 GQASelfAttention 一致，自带实现以支持 position_offset）
    # ------------------------------------------------------------------

    def _build_rope_table(self, head_dim: int, max_seq_len: int) -> None:
        """预计算 RoPE 的 cos/sin 表。"""
        half = head_dim // 2
        i = np.arange(half, dtype=np.float32)
        inv_freq = 1.0 / (self.rope_theta ** (2.0 * i / head_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, inv_freq)  # (T, half)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (T, head_dim)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
        self._cos_table = cos
        self._sin_table = sin
        self._rope_max_seq_len = max_seq_len

    def _apply_rope(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """对 x 应用 RoPE。

        Args:
            x: (B, T, H, D) Tensor
            position_offset: 位置偏移（KV cache 场景下新 token 的起始位置）
        Returns:
            同形状 Tensor
        """
        B, T, H, D = x.shape
        if position_offset + T > self._rope_max_seq_len:
            new_max = max(self._rope_max_seq_len * 2, position_offset + T)
            self._build_rope_table(D, new_max)
        pos = position_offset + np.arange(T)
        cos = self._cos_table[pos]  # (T, D)
        sin = self._sin_table[pos]
        cos_b = cos.reshape(1, T, 1, D)
        sin_b = sin.reshape(1, T, 1, D)
        x_data = x.data
        half = D // 2
        # rotate_half(x) = concat(-x[half:], x[:half])
        rotate_half = np.concatenate(
            [-x_data[..., half:], x_data[..., :half]], axis=-1
        )
        rotated = x_data * cos_b + rotate_half * sin_b

        out = x._result(rotated, (x,), "rope")
        if out.requires_grad:
            def _backward():
                grad = out.grad
                g = grad * cos_b + np.concatenate(
                    [-grad[..., half:], grad[..., :half]], axis=-1
                ) * sin_b
                x._accumulate_grad(g)
            out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # 全局 token KV 计算（路径 B 共享 wk/wv 投影，保证 K/V 与序列在同一空间）
    # ------------------------------------------------------------------

    def _compute_global_kv(self):
        """计算全局 token 的 K, V（可微，用于并行 forward）。

        global_tokens.weight (n_g, dim) 经 wk/wv 投影到 K/V 空间，
        再 repeat_kv 复制到 n_head 个 head。

        Returns:
            k_g, v_g: 各为 (1, n_head, num_global_tokens, head_dim) Tensor
        """
        n_g = self.num_global_tokens
        g = self.global_tokens.weight  # (n_g, dim) Tensor
        k_g = self.wk(g)  # (n_g, n_kv_head * head_dim)
        v_g = self.wv(g)  # (n_g, n_kv_head * head_dim)
        k_g = k_g.reshape(1, n_g, self.n_kv_head, self.head_dim)
        v_g = v_g.reshape(1, n_g, self.n_kv_head, self.head_dim)
        k_g = repeat_kv(k_g, self.n_rep)  # (1, n_g, n_head, head_dim)
        v_g = repeat_kv(v_g, self.n_rep)
        k_g = k_g.permute(0, 2, 1, 3)  # (1, n_head, n_g, head_dim)
        v_g = v_g.permute(0, 2, 1, 3)
        return k_g, v_g

    def _compute_global_kv_numpy(self):
        """计算全局 token 的 K, V（numpy，用于 recurrent）。

        Returns:
            k_g, v_g: 各为 (n_head, num_global_tokens, head_dim) ndarray
        """
        k_g, v_g = self._compute_global_kv()
        # 去掉 batch 维：(1, H, n_g, d) -> (H, n_g, d)
        return k_g.data[0], v_g.data[0]

    # ------------------------------------------------------------------
    # 路径 A：滑动窗口注意力（chunk-wise，避免 T² 全张量）
    # ------------------------------------------------------------------

    def _swa_forward(self, q, k, v, position_offset):
        """chunk-wise 滑动窗口注意力（多 query chunk 并行，批量矩阵化）。

        Args:
            q: (B, H, T_q, d) Tensor
            k, v: (B, H, T_k, d) Tensor
            position_offset: query 在全局序列中的起始位置
        Returns:
            (B, H, T_q, d) Tensor

        Part4K1 Task 3 升级：消除串行 for 循环，把 n_chunks 个 query chunk 批量
        堆叠为 (n_chunks, B, H, W, d) 形式，一次 batched matmul/softmax 完成。
        GPU 路径委托 torch.matmul（autograd 自动构建）；CPU 路径用 numpy 批量
        matmul（self.data @ other.data 支持前导广播）。

        并行 vs 串行数值一致：每个 chunk 的可见 key 数量 K_ci ≤ 2W-1，统一 pad 到
        K_max=2W-1。padding 位置 mask 设为 -1e9（float32 下 exp(-1e9)=0），
        softmax 归一化与串行版本数学等价，float32 下吻合到 1e-3。
        """
        B, H, T_q, d = q.shape
        T_k = k.shape[2]
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        # 计算 chunk 数（用于决定走并行还是串行 fallback）
        n_chunks = (T_q + W - 1) // W

        # 退化路径：单 chunk 直接走串行（无 batched 收益，避免额外开销）
        # 注意：必须在 pad 之前判断，让 serial 自己处理 padding（serial 内部
        # 也会基于 T_q 决定是否 pad；预先 pad 会导致 serial 误判 T_q）
        if n_chunks == 1:
            return self._swa_forward_serial(q, k, v, position_offset)

        # 将 T_q pad 到 window_size 的整数倍（保持梯度路径）
        T_q_padded = n_chunks * W
        pad_len = T_q_padded - T_q
        if pad_len > 0:
            q = _pad_last_dim(q, pad_len, axis=2)

        # 每个 chunk 最多可见的 key 数量上限 = 2W - 1
        # （chunk 的 query 全局位置范围 [gq_lo, gq_hi-1]，可见 key 全局位置
        #   [gq_lo - W + 1, gq_hi - 1]，跨度 (gq_hi - 1) - (gq_lo - W + 1) + 1 = 2W - 1）
        K_max = 2 * W - 1

        # 右侧 pad K, V 以容纳任意 chunk 的 [k_lo : k_lo + K_max] 切片
        # （最坏情况 k_lo 接近 T_k - 1，slice 需延伸到 T_k + K_max - 2，
        #   pad K_max 个 0 即可覆盖）
        k_pad = _pad_last_dim(k, K_max, axis=2)  # (B, H, T_k + K_max, d)
        v_pad = _pad_last_dim(v, K_max, axis=2)

        # 预计算每个 chunk 的 q 切片 / k 切片 / v 切片 / mask（保持可微）
        q_chunks = []   # 每个为 (1, B, H, W, d) Tensor
        k_chunks = []   # 每个为 (1, B, H, K_max, d) Tensor
        v_chunks = []
        masks = []      # 每个为 (W, K_max) ndarray
        for ci in range(n_chunks):
            q_lo = ci * W
            q_hi = q_lo + W
            q_chunk = q[:, :, q_lo:q_hi, :]           # (B, H, W, d) Tensor
            q_chunks.append(q_chunk.reshape(1, B, H, W, d))

            # query 的全局位置范围：[gq_lo, gq_hi)
            gq_lo = position_offset + q_lo
            gq_hi = position_offset + q_hi  # exclusive

            # 可见 key 的索引范围（在 T_k 维度）
            k_lo = max(0, gq_lo - W + 1)
            k_hi = min(T_k, gq_hi)
            if k_lo >= k_hi:
                # 极端边界：至少取最后一个 key
                k_lo = max(0, k_hi - 1)
            K_ci = k_hi - k_lo  # 当前 chunk 真实可见 key 数

            # 统一长度 K_max 的切片：[k_lo : k_lo + K_max]
            # 注意 k_pad 已右 pad K_max，保证 k_lo + K_max ≤ k_pad.shape[2]
            k_chunk = k_pad[:, :, k_lo:k_lo + K_max, :]  # (B, H, K_max, d)
            v_chunk = v_pad[:, :, k_lo:k_lo + K_max, :]
            k_chunks.append(k_chunk.reshape(1, B, H, K_max, d))
            v_chunks.append(v_chunk.reshape(1, B, H, K_max, d))

            # causal + sliding window mask: (W, K_max)
            # 对 [0, K_ci) 的真实 key 应用 causal+window；对 [K_ci, K_max) 的
            # padding 位置统一置 -1e9（在 softmax 中权重为 0，与串行版本等价）
            q_gpos = np.arange(W) + gq_lo           # (W,)
            k_gpos = np.arange(K_max) + k_lo        # (K_max,)（超出 k_hi 的部分被下面覆盖）
            causal = k_gpos[None, :] <= q_gpos[:, None]      # (W, K_max)
            in_window = (q_gpos[:, None] - k_gpos[None, :]) < W
            mask_2d = np.where(causal & in_window, 0.0, -1e9).astype(np.float32)
            mask_2d[:, K_ci:] = -1e9  # padding 位置强制 -1e9
            masks.append(mask_2d)

        # 沿 chunk 轴 batched（_concat 可微，自动分流 CPU/GPU）
        q_batched = _concat(q_chunks, dim=0)   # (n_chunks, B, H, W, d)
        k_batched = _concat(k_chunks, dim=0)   # (n_chunks, B, H, K_max, d)
        v_batched = _concat(v_chunks, dim=0)

        mask_arr = np.stack(masks, axis=0)     # (n_chunks, W, K_max)
        mask_t = Tensor(
            mask_arr.reshape(n_chunks, 1, 1, W, K_max),
            requires_grad=False,
        )

        # scores: (n_chunks, B, H, W, K_max) = q @ k^T
        # numpy / torch 的 @ 都支持前导广播的 batched matmul
        scores = (q_batched @ k_batched.transpose(-1, -2)) * scale
        scores = scores + mask_t

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)

        # out: (n_chunks, B, H, W, d) = attn @ v
        out_batched = attn @ v_batched

        # reshape 回 (B, H, T_q_padded, d)
        # (n_chunks, B, H, W, d) -> permute (B, H, n_chunks, W, d) -> reshape (B, H, T_q_padded, d)
        out = out_batched.permute(1, 2, 0, 3, 4).reshape(B, H, T_q_padded, d)

        # 裁剪到原 T_q（丢弃 pad 部分的输出）
        if pad_len > 0:
            out = out[:, :, :T_q, :]
        return out

    def _swa_forward_serial(self, q, k, v, position_offset):
        """chunk-wise 滑动窗口注意力（串行版本，保留作并行实现的数值对照）。

        Args:
            q: (B, H, T_q, d) Tensor
            k, v: (B, H, T_k, d) Tensor
            position_offset: query 在全局序列中的起始位置
        Returns:
            (B, H, T_q, d) Tensor

        将 query 划分为大小 window_size 的 chunk，**串行 for 循环**处理每个
        query chunk。每个 chunk 只 gather 最近 window_size 个 key（约 2*window_size
        个候选 key），计算局部 attention。总内存 O(T_q * window_size)，
        远低于 O(T_q * T_k)。

        此方法为 SubTask 3.1 升级前的原始实现，保留作 _swa_forward 并行版本
        的数值一致性对照（测试 test_parallel_vs_serial_numerical_consistency）。
        """
        B, H, T_q, d = q.shape
        T_k = k.shape[2]
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        # 将 T_q pad 到 window_size 的整数倍（保持梯度路径）
        n_chunks = (T_q + W - 1) // W
        T_q_padded = n_chunks * W
        pad_len = T_q_padded - T_q
        if pad_len > 0:
            q = _pad_last_dim(q, pad_len, axis=2)

        out_chunks = []
        for ci in range(n_chunks):
            q_lo = ci * W
            q_hi = q_lo + W
            q_chunk = q[:, :, q_lo:q_hi, :]  # (B, H, W, d)

            # query 的全局位置范围：[gq_lo, gq_hi)
            gq_lo = position_offset + q_lo
            gq_hi = position_offset + q_hi  # exclusive

            # 可见 key 的索引范围（在 T_k 维度）
            # query 全局位置 p 可见 key 全局位置 [p - W + 1, p]
            # chunk 的 key 范围取并集：[gq_lo - W + 1, gq_hi - 1]
            k_lo = max(0, gq_lo - W + 1)
            k_hi = min(T_k, gq_hi)
            if k_lo >= k_hi:
                # 极端边界：至少取最后一个 key
                k_lo = max(0, k_hi - 1)

            k_chunk = k[:, :, k_lo:k_hi, :]  # (B, H, K_len, d)
            v_chunk = v[:, :, k_lo:k_hi, :]
            K_len = k_hi - k_lo

            # attention scores: (B, H, W, K_len)
            scores = (q_chunk @ k_chunk.transpose(-1, -2)) * scale

            # causal + sliding window mask
            q_gpos = np.arange(W) + gq_lo  # query 全局位置 (W,)
            k_gpos = np.arange(K_len) + k_lo  # key 全局位置 (K_len,)
            causal = k_gpos[None, :] <= q_gpos[:, None]  # (W, K_len)
            in_window = (q_gpos[:, None] - k_gpos[None, :]) < W  # (W, K_len)
            mask_2d = np.where(causal & in_window, 0.0, -1e9).astype(np.float32)
            mask = Tensor(mask_2d.reshape(1, 1, W, K_len), requires_grad=False)
            scores = scores + mask

            attn = scores.softmax(dim=-1)
            attn = self.dropout(attn)
            out_chunk = attn @ v_chunk  # (B, H, W, d)
            out_chunks.append(out_chunk)

        # 沿 T 轴拼接各 chunk 输出（可微）
        out = _concat(out_chunks, dim=2)  # (B, H, T_q_padded, d)
        # 裁剪到原 T_q（丢弃 pad 部分的输出）
        if pad_len > 0:
            out = out[:, :, :T_q, :]
        return out

    # ------------------------------------------------------------------
    # 路径 B：全局 token 注意力
    # ------------------------------------------------------------------

    def _global_forward(self, q):
        """全局 sink token 注意力。

        Args:
            q: (B, H, T_q, d) Tensor
        Returns:
            (B, H, T_q, d) Tensor

        每个 query 都 attend num_global_tokens 个全局 token（无 causal mask）。
        内存 O(T_q * num_global_tokens)。
        """
        B, H, T_q, d = q.shape
        scale = 1.0 / (d ** 0.5)

        k_g, v_g = self._compute_global_kv()  # 各 (1, H, n_g, d) Tensor
        # scores: (B, H, T_q, n_g) = q @ k_g^T（batch 维广播）
        scores = (q @ k_g.transpose(-1, -2)) * scale
        # 无 causal mask：全局 token 始终可见
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        # out: (B, H, T_q, n_g) @ (1, H, n_g, d) -> (B, H, T_q, d)
        out = attn @ v_g
        return out

    # ------------------------------------------------------------------
    # 路径 C：ALiBi 全注意力
    # ------------------------------------------------------------------

    def _alibi_forward(self, q, k, v, position_offset, T_q, T_k):
        """标准 causal attention + ALiBi 位置偏置。

        Args:
            q: (B, H, T_q, d) Tensor
            k, v: (B, H, T_k, d) Tensor
            position_offset: query 全局起始位置
            T_q, T_k: query / key 序列长度
        Returns:
            (B, H, T_q, d) Tensor

        直接构造 (B, H, T_q, T_k) scores 矩阵，仅用于 T_k <= _ALIBI_MAX_T 的场景。
        ALiBi bias: bias[h, i, j] = -slopes[h] * ((position_offset + i) - j)
        当 j <= position_offset + i（causal）；否则 -1e9。
        """
        H = q.shape[1]
        d = q.shape[3]
        scale = 1.0 / (d ** 0.5)
        scores = (q @ k.transpose(-1, -2)) * scale  # (B, H, T_q, T_k)

        # 构造 causal + ALiBi bias
        # query 全局位置: position_offset + i (i = 0..T_q-1)
        # key   全局位置: j (j = 0..T_k-1)
        i_idx = (np.arange(T_q) + position_offset)[:, None]  # (T_q, 1)
        j_idx = np.arange(T_k)[None, :]  # (1, T_k)
        dist = i_idx - j_idx  # (T_q, T_k)，causal 时为非负
        causal = (j_idx <= i_idx).astype(np.float32)  # (T_q, T_k)

        # ALiBi bias: (H, T_q, T_k)
        # causal 处: -slopes[h] * dist；非 causal 处: -1e9
        bias = np.where(
            causal[None, :, :] > 0,
            -self.alibi_slopes[:, None, None] * dist[None, :, :],
            np.float32(-1e9),
        ).astype(np.float32)

        bias_t = Tensor(bias.reshape(1, H, T_q, T_k), requires_grad=False)
        scores = scores + bias_t
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, H, T_q, d)
        return out

    # ------------------------------------------------------------------
    # 并行 forward（训练 / 整序列推理）
    # ------------------------------------------------------------------

    def forward(self, x, position_offset: int = 0, kv_cache: dict = None):
        """整序列并行计算（可微，用于训练）。

        Args:
            x: (B, T, D) Tensor
            position_offset: query 在全局序列中的起始位置
                （仅当 kv_cache 为 None 时生效；提供 kv_cache 时自动从其长度推导）
            kv_cache: 可选 KV cache，dict with keys 'k', 'v'
                （Tensor (B, T_prev, n_kv_head, head_dim)）
        Returns:
            out: (B, T, D) Tensor
            new_kv_cache: dict with keys 'k', 'v'（当前序列完整 K/V，已 detach）
        """
        B, T, D = x.shape
        H, d = self.n_head, self.head_dim
        n_kv = self.n_kv_head

        # 1. 投影 Q, K, V
        q = self.wq(x).reshape(B, T, H, d)
        k = self.wk(x).reshape(B, T, n_kv, d)
        v = self.wv(x).reshape(B, T, n_kv, d)

        # 2. KV cache 决定 position_offset（与 GQASelfAttention 一致：cache 长度即偏移）
        if kv_cache is not None:
            k_prev = kv_cache["k"]
            v_prev = kv_cache["v"]
            position_offset = k_prev.shape[1]

        # 3. 应用 RoPE（仅 q, k；v 与 global token 不应用）
        if self.use_rope:
            q = self._apply_rope(q, position_offset)
            k = self._apply_rope(k, position_offset)

        # 4. KV cache 拼接前缀（可微 concat）
        if kv_cache is not None:
            k = _concat([k_prev, k], dim=1)
            v = _concat([v_prev, v], dim=1)

        # detach 后存入新 cache，避免梯度跨越 step 传播（与 GQASelfAttention 一致）
        new_kv_cache = {"k": k.detach(), "v": v.detach()}

        # 5. GQA: repeat KV head 匹配 q head 数量
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # 6. 转置为 (B, H, T, d)
        q = q.permute(0, 2, 1, 3)  # (B, H, T_q, d)
        k_rep = k_rep.permute(0, 2, 1, 3)  # (B, H, T_k, d)
        v_rep = v_rep.permute(0, 2, 1, 3)

        T_q = T
        T_k = k_rep.shape[2]

        # 7. 三路融合 gate（sigmoid）
        gate = self.gate_logits.sigmoid()  # (3,)
        g_a, g_b, g_c = gate[0], gate[1], gate[2]

        # 8. 路径 A: SWA（chunk-wise）
        swa_out = self._swa_forward(q, k_rep, v_rep, position_offset)

        # 9. 路径 B: Global token
        global_out = self._global_forward(q)

        # 10. 路径 C: ALiBi 全注意力（仅当 use_alibi 且 T_k <= 阈值时启用）
        use_path_c = self.use_alibi and T_k <= self._ALIBI_MAX_T
        if use_path_c:
            alibi_out = self._alibi_forward(
                q, k_rep, v_rep, position_offset, T_q, T_k
            )
            out = g_a * swa_out + g_b * global_out + g_c * alibi_out
        else:
            # 降级：gate C 强制为 0（路径 C 不参与输出）
            out = g_a * swa_out + g_b * global_out

        # 11. reshape 回 (B, T, D) 并投影
        out = out.transpose(1, 2).reshape(B, T_q, D)
        out = self.proj(out)
        return out, new_kv_cache

    # ------------------------------------------------------------------
    # 递推 forward_recurrent（单步推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x_single, state):
        """单步递推推理。

        Args:
            x_single: (B, 1, D) Tensor
            state: dict 或 None，包含：
                - 'k_cache': (B, n_cached, n_kv_head, head_dim) ndarray
                - 'v_cache': 同上
                - 'global_k': (n_head, num_global_tokens, head_dim) ndarray
                - 'global_v': 同上
                - 'position': int，当前 token 的全局位置
        Returns:
            out: (B, 1, D) Tensor
            new_state: dict（同 state 结构）
        """
        B, T, D = x_single.shape
        assert T == 1, f"forward_recurrent requires T=1, got T={T}"
        H, d = self.n_head, self.head_dim
        n_kv = self.n_kv_head
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        # 初始化或加载状态
        if state is None:
            k_cache = None
            v_cache = None
            position = 0
            # 首次调用时计算 global K/V（后续复用，不随 step 变化）
            global_k, global_v = self._compute_global_kv_numpy()
        else:
            k_cache = state["k_cache"]
            v_cache = state["v_cache"]
            position = state["position"]
            global_k = state["global_k"]
            global_v = state["global_v"]

        with no_grad():
            # 1. 投影 Q, K, V（单 token）
            q = self.wq(x_single).reshape(B, H, d)
            k = self.wk(x_single).reshape(B, n_kv, d)
            v = self.wv(x_single).reshape(B, n_kv, d)

            # 2. 应用 RoPE（_apply_rope 期望 (B, T, H, d)）
            if self.use_rope:
                q_4d = q.reshape(B, 1, H, d)
                k_4d = k.reshape(B, 1, n_kv, d)
                q_4d = self._apply_rope(q_4d, position)
                k_4d = self._apply_rope(k_4d, position)
                q = q_4d.reshape(B, H, d)
                k = k_4d.reshape(B, n_kv, d)

            q_data = q.data

            # 3. 更新 KV cache（滑动窗口）
            k_new = k.data[:, None, :, :]  # (B, 1, n_kv, d)
            v_new = v.data[:, None, :, :]
            if k_cache is None:
                k_cache_arr = k_new
                v_cache_arr = v_new
            else:
                k_cache_arr = np.concatenate([k_cache, k_new], axis=1)
                v_cache_arr = np.concatenate([v_cache, v_new], axis=1)
            # 保留最近 window_size 个 token
            if k_cache_arr.shape[1] > W:
                k_cache_arr = k_cache_arr[:, -W:, :, :]
                v_cache_arr = v_cache_arr[:, -W:, :, :]
            n_cached = k_cache_arr.shape[1]

            # 4. GQA: repeat KV head（与 repeat_kv 行为一致：相邻复制）
            if self.n_rep > 1:
                k_rep = np.repeat(k_cache_arr, self.n_rep, axis=2)  # (B, n_cached, H, d)
                v_rep = np.repeat(v_cache_arr, self.n_rep, axis=2)
            else:
                k_rep = k_cache_arr
                v_rep = v_cache_arr
            # 转置为 (B, H, n_cached, d)
            k_rep_t = np.transpose(k_rep, (0, 2, 1, 3))
            v_rep_t = np.transpose(v_rep, (0, 2, 1, 3))

            # ===== 路径 A: SWA =====
            # cache 已裁剪到 window_size，所有 key 都在窗口内且为 past（causal）
            # scores: (B, H, n_cached) = q . k^T
            swa_scores = np.einsum("bhd,bhmd->bhm", q_data, k_rep_t) * scale
            swa_attn = _np_softmax(swa_scores, axis=-1)
            swa_out = np.einsum("bhm,bhmd->bhd", swa_attn, v_rep_t)  # (B, H, d)

            # ===== 路径 B: Global =====
            # global_k, global_v: (H, n_g, d)（跨 batch 共享）
            # scores: (B, H, n_g) = q . global_k^T
            global_scores = np.einsum("bhd,hmd->bhm", q_data, global_k) * scale
            global_attn = _np_softmax(global_scores, axis=-1)
            global_out = np.einsum("bhm,hmd->bhd", global_attn, global_v)  # (B, H, d)

            # ===== 路径 C: ALiBi =====
            T_total = position + 1
            use_path_c = self.use_alibi and T_total <= self._ALIBI_MAX_T
            if use_path_c:
                # key 全局位置: [position - n_cached + 1, position]
                key_positions = np.arange(n_cached) + (position - n_cached + 1)
                dist = position - key_positions  # (n_cached,) >= 0
                # ALiBi bias: (H, n_cached)
                alibi_bias = -self.alibi_slopes[:, None] * dist[None, :]
                alibi_scores = swa_scores + alibi_bias[None, :, :]  # (B, H, n_cached)
                alibi_attn = _np_softmax(alibi_scores, axis=-1)
                alibi_out = np.einsum(
                    "bhm,bhmd->bhd", alibi_attn, v_rep_t
                )  # (B, H, d)

            # ===== Gate 融合 =====
            gate = _np_sigmoid(self.gate_logits.data)  # (3,)
            if use_path_c:
                out = (
                    gate[0] * swa_out
                    + gate[1] * global_out
                    + gate[2] * alibi_out
                )
            else:
                # 降级：gate C 强制为 0
                out = gate[0] * swa_out + gate[1] * global_out

            # 5. reshape 并投影
            out = out.reshape(B, 1, D)
            out_tensor = Tensor(out, requires_grad=False)
            out_tensor = self.proj(out_tensor)

            new_state = {
                "k_cache": k_cache_arr,
                "v_cache": v_cache_arr,
                "global_k": global_k,
                "global_v": global_v,
                "position": position + 1,
            }

        return out_tensor, new_state


__all__ = ["TriSparseAttention"]
