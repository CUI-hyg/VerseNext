"""CometFuture VerseNext Delta Attention（VDA）与配套稀疏注意力（Part6 新增）。

设计目标
--------
- **VDA（VerseNext Delta Attention）**：三路混合注意力，融合
  - **DSA（DeepSeek 风格稀疏注意力）**：head 级 lightning gate 决定
    dense head（全因果注意力）与 sparse head（窗口 + 全局 sink）的分配；
  - **KDA（Kimi 风格 Anchor 注意力）**：每隔 ``anchor_interval`` 个 token
    选取一个 anchor（全局可见），普通 token 只 attend 窗口 + anchor，
    兼顾全局信息与线性复杂度（ATGT / MoBA 思想）；
  - **GMLA（VerseNext Gated MLA）**：DeepSeek-V2 风格低秩 KV 压缩
    （latent 空间缓存，KV cache 减半）+ head 级 gate 加权。
- 三条路径按可学习 gate ``(3,)``（sigmoid）加权融合，与
  :class:`TriSparseAttention` 的三路融合模式保持一致。
- **独立层类型**：DSA / KDA / GMLA 亦可作为独立注意力层使用，
  供 ``build_verse_delta_pattern`` 的层分配规则调用。

层分配规则（``build_verse_delta_pattern``）
------------------------------------------
- ``n_layer < 6``：全部使用 VDA
- ``6 <= n_layer < 10``：前 4 层 VDA，其余 GMLA
- ``n_layer >= 10``：前 4 层 VDA，后 2 层 DSA，其余 GMLA

统一接口（与 TriSparseAttention 对齐）
--------------------------------------
- ``forward(x, position_offset=0, kv_cache=None)`` → ``(out, new_kv_cache)``
- ``forward_recurrent(x_single, state)`` → ``(out, new_state)``
- 支持 GQA（``n_kv_head < n_head``）、RoPE（``use_rope=True``）、
  KV cache 拼接（``_concat`` 可微）、数值稳定 softmax。

复用的项目内已有功能：
- ``verse_torch.vnn``：``Linear / Module / Dropout / Embedding / normal_ /
  _concat / repeat_kv``
- ``verse_nex.sparse_attention._pad_last_dim``（带梯度的轴向 padding）
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.vnn import (
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
# 通用 numpy 工具（不构建计算图）
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
# 通用 RoPE 工具（与 TriSparseAttention 一致：rotate_half + cos/sin 表）
# ---------------------------------------------------------------------------


def _build_rope_table(head_dim: int, max_seq_len: int, rope_theta: float):
    """预计算 RoPE 的 cos/sin 表。返回 ``(cos, sin)``，各为 ``(T, head_dim)``。"""
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (rope_theta ** (2.0 * i / head_dim))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, inv_freq)  # (T, half)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos, sin


def _apply_rope(
    x: Tensor,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    rope_max_seq_len: int,
    position_offset: int = 0,
    rope_theta: float = 10000.0,
) -> Tensor:
    """对 ``(B, T, H, D)`` Tensor 应用 RoPE（可微）。

    Args:
        x: ``(B, T, H, D)`` Tensor
        cos_table / sin_table: ``(T, D)`` ndarray
        rope_max_seq_len: 当前预计算的最大长度
        position_offset: 位置偏移（KV cache 场景）
        rope_theta: 重建表时使用的基础频率
    Returns:
        同形状 Tensor
    """
    B, T, H, D = x.shape
    if position_offset + T > rope_max_seq_len:
        new_max = max(rope_max_seq_len * 2, position_offset + T)
        cos_table, sin_table = _build_rope_table(D, new_max, rope_theta)
        rope_max_seq_len = new_max
    pos = position_offset + np.arange(T)
    cos = cos_table[pos]  # (T, D)
    sin = sin_table[pos]
    cos_b = cos.reshape(1, T, 1, D)
    sin_b = sin.reshape(1, T, 1, D)
    x_data = x.data
    half = D // 2
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


# ---------------------------------------------------------------------------
# 公共基类：投影 / RoPE / KV cache / GQA / 输出投影
# ---------------------------------------------------------------------------


class _DeltaAttentionBase(Module):
    """VDA 家族注意力的公共基类。

    提供 QKV 投影、RoPE、KV cache 拼接、GQA head 复制与输出投影等公共逻辑。
    子类实现 ``_attend``（并行）与 ``_attend_recurrent``（单步递推）即可。
    """

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        window_size: int = 512,
        num_global_tokens: int = 64,
        anchor_interval: int = 128,
        latent_dim: Optional[int] = None,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        include_proj: bool = True,
    ):
        super().__init__()
        if n_kv_head is None:
            n_kv_head = n_head
        assert dim % n_head == 0, f"dim({dim}) 必须能被 n_head({n_head}) 整除"
        assert n_head % n_kv_head == 0, (
            f"n_head({n_head}) 必须能被 n_kv_head({n_kv_head}) 整除"
        )

        self.dim = dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.n_rep = n_head // n_kv_head
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.anchor_interval = anchor_interval
        self.latent_dim = latent_dim
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        # QKV 投影与输出投影（bias=False，与 GQASelfAttention 一致）
        self.wq = Linear(dim, n_head * self.head_dim, bias=False)
        self.wk = Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.wv = Linear(dim, n_kv_head * self.head_dim, bias=False)
        if include_proj:
            self.proj = Linear(n_head * self.head_dim, dim, bias=False)
        else:
            # 组合场景（VDA 内部子路径）：不投影，输出直接是 head 空间
            self.proj = None
        self.dropout = Dropout(dropout)

        # RoPE 预计算表（仅 use_rope=True 时构建）
        if use_rope:
            cos, sin = _build_rope_table(self.head_dim, max_seq_len, rope_theta)
            self._cos_table = cos
            self._sin_table = sin
            self._rope_max_seq_len = max_seq_len
        else:
            self._cos_table = None
            self._sin_table = None
            self._rope_max_seq_len = 0

    # ------------------------------------------------------------------
    # 投影与公共前向辅助
    # ------------------------------------------------------------------

    def _project(self, x: Tensor):
        """投影 Q/K/V。返回各为 ``(B, T, H, d)`` / ``(B, T, n_kv, d)``。"""
        B, T, D = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_head, self.head_dim)
        return q, k, v

    def _apply_rope(self, q: Tensor, k: Tensor, position_offset: int = 0):
        """对 q（可含 k）应用 RoPE。返回 ``(q, k)``。"""
        if self.use_rope:
            q = _apply_rope(
                q, self._cos_table, self._sin_table,
                self._rope_max_seq_len, position_offset, self.rope_theta,
            )
            if k is not None:
                k = _apply_rope(
                    k, self._cos_table, self._sin_table,
                    self._rope_max_seq_len, position_offset, self.rope_theta,
                )
        return q, k

    # ------------------------------------------------------------------
    # 并行 forward（训练 / 整序列推理）
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, position_offset: int = 0, kv_cache: dict = None):
        """整序列并行计算（可微，用于训练）。

        Args:
            x: ``(B, T, D)`` Tensor
            position_offset: query 在全局序列中的起始位置
            kv_cache: 可选 KV cache，dict with keys ``'k', 'v'``
        Returns:
            out: ``(B, T, D)`` Tensor
            new_kv_cache: dict with keys ``'k', 'v'``（已 detach）
        """
        B, T, D = x.shape
        q, k, v = self._project(x)

        # KV cache 决定 position_offset（cache 长度即偏移）
        if kv_cache is not None:
            k_prev = kv_cache["k"]
            v_prev = kv_cache["v"]
            position_offset = k_prev.shape[1]

        # 应用 RoPE（仅 q, k；v 不应用）
        if self.use_rope:
            q, k = self._apply_rope(q, k, position_offset)

        # KV cache 拼接前缀（可微 concat）
        if kv_cache is not None:
            k = _concat([k_prev, k], dim=1)
            v = _concat([v_prev, v], dim=1)

        # detach 后存入新 cache，避免梯度跨越 step 传播
        new_kv_cache = {"k": k.detach(), "v": v.detach()}

        # GQA: repeat KV head 匹配 q head 数量
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # 转置为 (B, H, T, d)
        q = q.permute(0, 2, 1, 3)
        k_rep = k_rep.permute(0, 2, 1, 3)
        v_rep = v_rep.permute(0, 2, 1, 3)

        attn_out = self._attend(q, k_rep, v_rep, position_offset)
        out = self._out_proj(attn_out, B, T)
        return out, new_kv_cache

    def _out_proj(self, attn_out: Tensor, B: int, T: int) -> Tensor:
        """把 head 空间输出 ``(B, H, T, d)`` reshape 回 ``(B, T, D)`` 并投影。"""
        out = attn_out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        if self.proj is not None:
            out = self.proj(out)
        return out

    # ------------------------------------------------------------------
    # 递推 forward_recurrent（单步推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x_single: Tensor, state: Optional[dict]):
        """单步递推推理（通用实现，子类提供 ``_attend_recurrent``）。

        Args:
            x_single: ``(B, 1, D)`` Tensor
            state: dict 或 None，包含 ``'k_cache' / 'v_cache' / 'position'``
        Returns:
            out: ``(B, 1, D)`` Tensor
            new_state: dict（同 state 结构）
        """
        B, T, D = x_single.shape
        assert T == 1, f"forward_recurrent requires T=1, got T={T}"
        H = self.n_head
        n_kv = self.n_kv_head

        # 初始化或加载状态
        if state is None:
            k_cache = None
            v_cache = None
            position = 0
        else:
            k_cache = state["k_cache"]
            v_cache = state["v_cache"]
            position = state["position"]

        with no_grad():
            # 1. 投影 Q, K, V（单 token）
            q = self.wq(x_single).reshape(B, H, self.head_dim)
            k = self.wk(x_single).reshape(B, n_kv, self.head_dim)
            v = self.wv(x_single).reshape(B, n_kv, self.head_dim)

            # 2. 应用 RoPE（_apply_rope 期望 (B, T, H, d)）
            if self.use_rope:
                q_4d, k_4d = self._apply_rope(
                    q.reshape(B, 1, H, self.head_dim),
                    k.reshape(B, 1, n_kv, self.head_dim),
                    position,
                )
                q = q_4d.reshape(B, H, self.head_dim)
                k = k_4d.reshape(B, n_kv, self.head_dim)

            q_data = q.data

            # 3. 更新 KV cache（全历史，dense 路径需要）
            k_new = k.data[:, None, :, :]  # (B, 1, n_kv, d)
            v_new = v.data[:, None, :, :]
            if k_cache is None:
                k_cache_arr = k_new
                v_cache_arr = v_new
            else:
                k_cache_arr = np.concatenate([k_cache, k_new], axis=1)
                v_cache_arr = np.concatenate([v_cache, v_new], axis=1)
            n_cached = k_cache_arr.shape[1]

            # 4. GQA: repeat KV head
            if self.n_rep > 1:
                k_rep = np.repeat(k_cache_arr, self.n_rep, axis=2)
                v_rep = np.repeat(v_cache_arr, self.n_rep, axis=2)
            else:
                k_rep = k_cache_arr
                v_rep = v_cache_arr
            k_rep_t = np.transpose(k_rep, (0, 2, 1, 3))
            v_rep_t = np.transpose(v_rep, (0, 2, 1, 3))

            # 5. 子类实现的单步 attention
            attn_out = self._attend_recurrent(
                q_data, k_rep_t, v_rep_t, position, state
            )  # (B, H, d)

            # 6. reshape 并投影
            out = attn_out.reshape(B, 1, D)
            out_tensor = Tensor(out, requires_grad=False)
            if self.proj is not None:
                out_tensor = self.proj(out_tensor)

            new_state = {
                "k_cache": k_cache_arr,
                "v_cache": v_cache_arr,
                "position": position + 1,
            }

        return out_tensor, new_state

    # 子类实现
    def _attend(self, q, k_rep, v_rep, position_offset):
        raise NotImplementedError

    def _attend_recurrent(self, q_data, k_rep_t, v_rep_t, position, state):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DSA：DeepSeek 风格稀疏注意力（head 级 lightning gate）
# ---------------------------------------------------------------------------


class DSAAttention(_DeltaAttentionBase):
    """DeepSeek 风格稀疏注意力（DSA）。

    每个 head 通过可学习的 lightning gate（sigmoid）分配两种计算路径：
    - **dense head**（gate → 1）：完整因果注意力，可访问全部历史 key；
    - **sparse head**（gate → 0）：局部滑动窗口注意力 + 全局 sink token
      （内存 O(T * window_size)，稀疏加速）。

    head 级 gate 为软门控（训练中自适应，收敛后可近似二值化裁剪）：
    ``out = dense_out * gate + sparse_out * (1 - gate)``。

    序列过长（``T_k > _MAX_T``）时 dense 路径自动降级为窗口 + 全局
    （避免构造 T² 全张量），与 :class:`TriSparseAttention` 的降级策略一致。
    """

    # dense 全注意力序列长度上限：超过则 dense 也走窗口 + 全局
    _MAX_T = 1024

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        window_size: int = 512,
        num_global_tokens: int = 64,
        anchor_interval: int = 128,
        latent_dim: Optional[int] = None,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        include_proj: bool = True,
    ):
        super().__init__(
            dim=dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            anchor_interval=anchor_interval,
            latent_dim=latent_dim,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
            include_proj=include_proj,
        )

        # head 级 lightning gate（logits 初始化为 0 → sigmoid=0.5 等权）
        self.head_gate_logits = Tensor(
            np.zeros(n_head, dtype=np.float32), requires_grad=True
        )

        # 全局 sink token（sparse head 的全局视图）
        self.global_tokens = Embedding(num_global_tokens, dim)
        normal_(self.global_tokens.weight, std=0.02)

    # ------------------------------------------------------------------
    # 全局 sink KV（与 TriSparseAttention 的 global token 路径一致）
    # ------------------------------------------------------------------

    def _compute_global_kv(self):
        """计算全局 token 的 K, V（可微）。返回各为 ``(1, H, n_g, d)``。"""
        n_g = self.num_global_tokens
        g = self.global_tokens.weight  # (n_g, dim)
        k_g = self.wk(g).reshape(1, n_g, self.n_kv_head, self.head_dim)
        v_g = self.wv(g).reshape(1, n_g, self.n_kv_head, self.head_dim)
        k_g = repeat_kv(k_g, self.n_rep).permute(0, 2, 1, 3)  # (1, H, n_g, d)
        v_g = repeat_kv(v_g, self.n_rep).permute(0, 2, 1, 3)
        return k_g, v_g

    def _compute_global_kv_numpy(self):
        """numpy 版全局 K/V（recurrent 用）：``(H, n_g, d)``。"""
        k_g, v_g = self._compute_global_kv()
        return k_g.data[0], v_g.data[0]

    # ------------------------------------------------------------------
    # dense 路径：完整因果注意力
    # ------------------------------------------------------------------

    def _dense_forward(self, q, k_rep, v_rep):
        """完整因果注意力。``q/k_rep/v_rep`` 各为 ``(B, H, T, d)``。"""
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]
        scale = 1.0 / (d ** 0.5)
        scores = (q @ k_rep.transpose(-1, -2)) * scale  # (B, H, T_q, T_k)

        i_idx = np.arange(T_q)[:, None]
        j_idx = np.arange(T_k)[None, :]
        causal = (j_idx <= i_idx).astype(np.float32)
        mask = np.where(causal > 0, 0.0, -1e9).astype(np.float32)
        scores = scores + Tensor(mask.reshape(1, 1, T_q, T_k), requires_grad=False)

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v_rep  # (B, H, T_q, d)
        return out

    # ------------------------------------------------------------------
    # sparse 路径：滑动窗口 + 全局 sink
    # ------------------------------------------------------------------

    def _window_forward_serial(self, q, k_rep, v_rep, position_offset):
        """chunk-wise 滑动窗口注意力（串行，内存 O(T * window_size)）。"""
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

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

            gq_lo = position_offset + q_lo
            gq_hi = position_offset + q_hi
            k_lo = max(0, gq_lo - W + 1)
            k_hi = min(T_k, gq_hi)
            if k_lo >= k_hi:
                k_lo = max(0, k_hi - 1)

            k_chunk = k_rep[:, :, k_lo:k_hi, :]
            v_chunk = v_rep[:, :, k_lo:k_hi, :]
            K_len = k_hi - k_lo

            scores = (q_chunk @ k_chunk.transpose(-1, -2)) * scale
            q_gpos = np.arange(W) + gq_lo
            k_gpos = np.arange(K_len) + k_lo
            causal = k_gpos[None, :] <= q_gpos[:, None]
            in_window = (q_gpos[:, None] - k_gpos[None, :]) < W
            mask_2d = np.where(causal & in_window, 0.0, -1e9).astype(np.float32)
            scores = scores + Tensor(
                mask_2d.reshape(1, 1, W, K_len), requires_grad=False
            )
            attn = scores.softmax(dim=-1)
            attn = self.dropout(attn)
            out_chunks.append(attn @ v_chunk)

        out = _concat(out_chunks, dim=2)  # (B, H, T_q_padded, d)
        if pad_len > 0:
            out = out[:, :, :T_q, :]
        return out

    def _global_forward(self, q):
        """全局 sink 注意力（无 causal mask）。``q: (B, H, T_q, d)``。"""
        B, H, T_q, d = q.shape
        scale = 1.0 / (d ** 0.5)
        k_g, v_g = self._compute_global_kv()  # (1, H, n_g, d)
        scores = (q @ k_g.transpose(-1, -2)) * scale
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v_g  # (B, H, T_q, d)
        return out

    # ------------------------------------------------------------------
    # 并行 _attend
    # ------------------------------------------------------------------

    def _attend(self, q, k_rep, v_rep, position_offset):
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]

        gate = self.head_gate_logits.sigmoid()  # (H,)

        # dense 路径：T 短时全因果；过长时降级为窗口 + 全局
        if T_k <= self._MAX_T:
            dense_out = self._dense_forward(q, k_rep, v_rep)
        else:
            dense_out = self._window_forward_serial(q, k_rep, v_rep, position_offset)
            dense_out = dense_out + self._global_forward(q)

        # sparse 路径：窗口 + 全局（两部分输出相加，提供局部 + 全局视图）
        sparse_out = self._window_forward_serial(q, k_rep, v_rep, position_offset)
        sparse_out = sparse_out + self._global_forward(q)

        gate_b = gate.reshape(1, H, 1, 1)
        out = dense_out * gate_b + sparse_out * (1.0 - gate_b)
        return out

    # ------------------------------------------------------------------
    # 单步递推 _attend_recurrent
    # ------------------------------------------------------------------

    def _attend_recurrent(self, q_data, k_rep_t, v_rep_t, position, state):
        """单步：dense 头 attend 全部缓存，sparse 头 attend 窗口 + 全局。"""
        B, H, d = q_data.shape
        n_cached = k_rep_t.shape[2]
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        # 全局 K/V（首次调用时计算并存入 state）
        if state is not None and "global_k" in state:
            global_k = state["global_k"]
            global_v = state["global_v"]
        else:
            global_k, global_v = self._compute_global_kv_numpy()

        # dense：全部缓存（causal，所有 past 可见）
        dense_scores = np.einsum("bhd,bhmd->bhm", q_data, k_rep_t) * scale
        dense_attn = _np_softmax(dense_scores, axis=-1)
        dense_out = np.einsum("bhm,bhmd->bhd", dense_attn, v_rep_t)

        # sparse：最近 window_size 个 key + 全局 sink
        w = min(n_cached, W)
        k_win = k_rep_t[:, :, -w:, :]
        v_win = v_rep_t[:, :, -w:, :]
        win_scores = np.einsum("bhd,bhmd->bhm", q_data, k_win) * scale
        win_attn = _np_softmax(win_scores, axis=-1)
        win_out = np.einsum("bhm,bhmd->bhd", win_attn, v_win)

        global_scores = np.einsum("bhd,hmd->bhm", q_data, global_k) * scale
        global_attn = _np_softmax(global_scores, axis=-1)
        global_out = np.einsum("bhm,hmd->bhd", global_attn, global_v)

        sparse_out = win_out + global_out

        gate = _np_sigmoid(self.head_gate_logits.data)  # (H,)
        gate_b = gate.reshape(1, H, 1)
        out = dense_out * gate_b + sparse_out * (1.0 - gate_b)
        return out


# ---------------------------------------------------------------------------
# KDA：Kimi 风格 Anchor 注意力（局部窗口 + 全局 anchor）
# ---------------------------------------------------------------------------


class KDAAttention(_DeltaAttentionBase):
    """Kimi 风格 Anchor 注意力（KDA）。

    每隔 ``anchor_interval`` 个 token 选取一个 **anchor**（全局可见），
    普通 query 只 attend 最近 ``window_size`` 个 key + 全部 anchor key：
    - anchor token 提供全局信息，代价 O(T * T / anchor_interval)；
    - 窗口 token 提供局部细节，代价 O(T * window_size)。

    与 Kimi 的 ATGT（Attention with Global Tokens）/ MoBA 思想一致，
    用稀疏 mask 把两者合并到同一次 softmax 中（数学上仍是合法分布）。

    序列过长（``T_k > _MAX_T``）时降级为 chunk-wise 实现，
    避免构造 (T_q, T_k) 全张量。
    """

    # 全矩阵（T_q, T_k）路径的序列长度上限
    _MAX_T = 1024

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        window_size: int = 512,
        num_global_tokens: int = 64,
        anchor_interval: int = 128,
        latent_dim: Optional[int] = None,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        include_proj: bool = True,
    ):
        super().__init__(
            dim=dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            anchor_interval=anchor_interval,
            latent_dim=latent_dim,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
            include_proj=include_proj,
        )
        assert anchor_interval >= 1, f"anchor_interval 必须 >= 1，got {anchor_interval}"
        self.anchor_interval = anchor_interval

    # ------------------------------------------------------------------
    # 快速路径：全矩阵 + anchor|window mask
    # ------------------------------------------------------------------

    def _anchor_window_forward(self, q, k_rep, v_rep, position_offset):
        """全矩阵注意力 + anchor|window mask（T_k <= _MAX_T 时使用）。"""
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]
        scale = 1.0 / (d ** 0.5)

        scores = (q @ k_rep.transpose(-1, -2)) * scale  # (B, H, T_q, T_k)

        i_idx = np.arange(T_q) + position_offset  # query 全局位置
        j_idx = np.arange(T_k)  # key 全局位置
        is_anchor = (j_idx % self.anchor_interval) == 0  # (T_k,)
        in_window = (i_idx[:, None] - j_idx[None, :]) < self.window_size
        causal = j_idx[None, :] <= i_idx[:, None]
        mask_2d = causal & (in_window | is_anchor[None, :])
        bias = np.where(mask_2d, 0.0, -1e9).astype(np.float32)
        scores = scores + Tensor(bias.reshape(1, 1, T_q, T_k), requires_grad=False)

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v_rep
        return out

    # ------------------------------------------------------------------
    # 长序列降级：chunk-wise（窗口 + anchor 合并到同一次 softmax）
    # ------------------------------------------------------------------

    def _anchor_window_serial(self, q, k_rep, v_rep, position_offset):
        """chunk-wise 实现：每 chunk 的 key = 窗口切片 + 全部 anchor。"""
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]
        W = self.window_size
        I = self.anchor_interval
        scale = 1.0 / (d ** 0.5)

        n_chunks = (T_q + W - 1) // W
        T_q_padded = n_chunks * W
        pad_len = T_q_padded - T_q
        if pad_len > 0:
            q = _pad_last_dim(q, pad_len, axis=2)

        # 全部 anchor 的 key（step slice，保持可微）
        k_anchor = k_rep[:, :, ::I, :]  # (B, H, n_a, d)
        v_anchor = v_rep[:, :, ::I, :]
        n_a = k_anchor.shape[2]
        anchor_pos = np.arange(0, T_k, I)  # anchor 全局位置

        out_chunks = []
        for ci in range(n_chunks):
            q_lo = ci * W
            q_hi = q_lo + W
            q_chunk = q[:, :, q_lo:q_hi, :]  # (B, H, W, d)

            gq_lo = position_offset + q_lo
            gq_hi = position_offset + q_hi
            k_lo = max(0, gq_lo - W + 1)
            k_hi = min(T_k, gq_hi)
            if k_lo >= k_hi:
                k_lo = max(0, k_hi - 1)

            k_win = k_rep[:, :, k_lo:k_hi, :]
            v_win = v_rep[:, :, k_lo:k_hi, :]
            K_win = k_hi - k_lo

            # 合并窗口 key 与 anchor key（沿 T 轴 concat，可微）
            k_all = _concat([k_win, k_anchor], dim=2)
            v_all = _concat([v_win, v_anchor], dim=2)
            K_all = K_win + n_a

            scores = (q_chunk @ k_all.transpose(-1, -2)) * scale

            q_gpos = np.arange(W) + gq_lo  # (W,)
            win_gpos = np.arange(K_win) + k_lo
            all_gpos = np.concatenate([win_gpos, anchor_pos])  # (K_all,)
            causal = all_gpos[None, :] <= q_gpos[:, None]
            in_window = (q_gpos[:, None] - all_gpos[None, :]) < W
            is_anchor = (all_gpos % I) == 0
            mask_2d = np.where(
                causal & (in_window | is_anchor[None, :]), 0.0, -1e9
            ).astype(np.float32)
            scores = scores + Tensor(
                mask_2d.reshape(1, 1, W, K_all), requires_grad=False
            )

            attn = scores.softmax(dim=-1)
            attn = self.dropout(attn)
            out_chunks.append(attn @ v_all)

        out = _concat(out_chunks, dim=2)
        if pad_len > 0:
            out = out[:, :, :T_q, :]
        return out

    # ------------------------------------------------------------------
    # 并行 _attend / 单步递推 _attend_recurrent
    # ------------------------------------------------------------------

    def _attend(self, q, k_rep, v_rep, position_offset):
        T_k = k_rep.shape[2]
        if T_k <= self._MAX_T:
            return self._anchor_window_forward(q, k_rep, v_rep, position_offset)
        return self._anchor_window_serial(q, k_rep, v_rep, position_offset)

    def _attend_recurrent(self, q_data, k_rep_t, v_rep_t, position, state):
        """单步：mask = causal & (anchor | 窗口)。"""
        B, H, d = q_data.shape
        n_cached = k_rep_t.shape[2]
        I = self.anchor_interval
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        scores = np.einsum("bhd,bhmd->bhm", q_data, k_rep_t) * scale

        # key 全局位置: [position - n_cached + 1, position]
        key_positions = np.arange(n_cached) + (position - n_cached + 1)
        is_anchor = (key_positions % I) == 0
        in_window = (position - key_positions) < W
        mask = np.where(
            (is_anchor | in_window).astype(np.float32)[None, None, :],
            0.0,
            -1e9,
        ).astype(np.float32)
        scores = scores + mask

        attn = _np_softmax(scores, axis=-1)
        out = np.einsum("bhm,bhmd->bhd", attn, v_rep_t)
        return out


# ---------------------------------------------------------------------------
# GMLA：VerseNext Gated MLA（低秩 KV 压缩 + head 级 gate）
# ---------------------------------------------------------------------------


class GMLAAttention(_DeltaAttentionBase):
    """VerseNext Gated MLA（Multi-head Latent Attention + gate）。

    基于 DeepSeek-V2 的 MLA 思想：
    - 把 K/V 压缩到低秩 latent 空间（``w_kv_down``），再经 ``w_k_up`` /
      ``w_v_up`` 重建，KV cache 只存 latent（约一半内存）；
    - Q 独立投影（``wq``），RoPE 应用于重建后的 K（与 MLA 一致）；
    - **head 级 gate**（sigmoid）对注意力输出加权，训练中自适应调整
      每个 head 的贡献（VerseNext 风格的可学习门控）。

    KV cache 结构：``{"c": (B, T, kv_dim)}``（latent，不存 K/V）。
    """

    # 全因果注意力序列长度上限：超过则降级为窗口注意力
    _MAX_T = 1024

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        latent_dim: Optional[int] = None,
        window_size: int = 512,
        num_global_tokens: int = 64,
        anchor_interval: int = 128,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        include_proj: bool = True,
    ):
        super().__init__(
            dim=dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            latent_dim=latent_dim,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            anchor_interval=anchor_interval,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
            include_proj=include_proj,
        )
        kv_dim = self.n_kv_head * self.head_dim
        # latent 维度默认等于 kv_dim（压缩 concat(K, V) 到单 latent）
        self.latent_dim = latent_dim if latent_dim is not None else kv_dim

        # MLA 专用投影（替换基类的 wk/wv；置 None 触发 __setattr__ 清理注册）
        self.w_kv_down = Linear(dim, self.latent_dim, bias=False)
        self.w_k_up = Linear(self.latent_dim, kv_dim, bias=False)
        self.w_v_up = Linear(self.latent_dim, kv_dim, bias=False)
        self.wk = None
        self.wv = None

        # head 级 gate（logits 初始化为 0 → sigmoid=0.5 等权）
        self.head_gate_logits = Tensor(
            np.zeros(n_head, dtype=np.float32), requires_grad=True
        )

    # ------------------------------------------------------------------
    # 并行 forward（latent KV 路径）
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, position_offset: int = 0, kv_cache: dict = None):
        """整序列并行计算（MLA：先压缩 K/V 到 latent，再重建 attend）。

        Args:
            x: ``(B, T, D)`` Tensor
            position_offset: query 在全局序列中的起始位置
            kv_cache: 可选 dict with key ``'c'``（latent ``(B, T_prev, kv_dim)``）
        Returns:
            out: ``(B, T, D)`` Tensor
            new_kv_cache: dict with key ``'c'``（已 detach）
        """
        B, T, D = x.shape
        H, d = self.n_head, self.head_dim
        n_kv = self.n_kv_head
        kv_dim = self.latent_dim

        # 1. Q 投影 + K/V 压缩
        q = self.wq(x).reshape(B, T, H, d)
        c = self.w_kv_down(x)  # (B, T, kv_dim)
        k = self.w_k_up(c).reshape(B, T, n_kv, d)
        v = self.w_v_up(c).reshape(B, T, n_kv, d)

        # 2. KV cache 决定 position_offset
        if kv_cache is not None:
            c_prev = kv_cache["c"]
            position_offset = c_prev.shape[1]

        # 3. 应用 RoPE（q 与重建后的 k）
        if self.use_rope:
            q, k = self._apply_rope(q, k, position_offset)

        # 4. latent cache 拼接（可微 concat）
        if kv_cache is not None:
            c = _concat([c_prev, c], dim=1)

        new_kv_cache = {"c": c.detach()}

        # 5. GQA repeat + 转置
        k_rep = repeat_kv(k, self.n_rep).permute(0, 2, 1, 3)
        v_rep = repeat_kv(v, self.n_rep).permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)

        # 6. 注意力：短序列全因果，长序列窗口降级
        T_k = k_rep.shape[2]
        if T_k <= self._MAX_T:
            attn_out = self._dense_causal(q, k_rep, v_rep)
        else:
            attn_out = self._window_serial(q, k_rep, v_rep, position_offset)

        # 7. head 级 gate 加权
        gate_b = self.head_gate_logits.sigmoid().reshape(1, H, 1, 1)
        attn_out = attn_out * gate_b

        out = self._out_proj(attn_out, B, T)
        return out, new_kv_cache

    def _dense_causal(self, q, k_rep, v_rep):
        """完整因果注意力。各参数 ``(B, H, T, d)``。"""
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]
        scale = 1.0 / (d ** 0.5)
        scores = (q @ k_rep.transpose(-1, -2)) * scale
        i_idx = np.arange(T_q)[:, None]
        j_idx = np.arange(T_k)[None, :]
        mask = np.where((j_idx <= i_idx).astype(np.float32) > 0, 0.0, -1e9)
        scores = scores + Tensor(
            mask.reshape(1, 1, T_q, T_k).astype(np.float32), requires_grad=False
        )
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        return attn @ v_rep

    def _window_serial(self, q, k_rep, v_rep, position_offset):
        """chunk-wise 窗口注意力（长序列降级路径）。"""
        B, H, T_q, d = q.shape
        T_k = k_rep.shape[2]
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        n_chunks = (T_q + W - 1) // W
        T_q_padded = n_chunks * W
        pad_len = T_q_padded - T_q
        if pad_len > 0:
            q = _pad_last_dim(q, pad_len, axis=2)

        out_chunks = []
        for ci in range(n_chunks):
            q_lo = ci * W
            q_hi = q_lo + W
            q_chunk = q[:, :, q_lo:q_hi, :]
            gq_lo = position_offset + q_lo
            gq_hi = position_offset + q_hi
            k_lo = max(0, gq_lo - W + 1)
            k_hi = min(T_k, gq_hi)
            if k_lo >= k_hi:
                k_lo = max(0, k_hi - 1)
            k_chunk = k_rep[:, :, k_lo:k_hi, :]
            v_chunk = v_rep[:, :, k_lo:k_hi, :]
            K_len = k_hi - k_lo

            scores = (q_chunk @ k_chunk.transpose(-1, -2)) * scale
            q_gpos = np.arange(W) + gq_lo
            k_gpos = np.arange(K_len) + k_lo
            causal = k_gpos[None, :] <= q_gpos[:, None]
            in_window = (q_gpos[:, None] - k_gpos[None, :]) < W
            mask_2d = np.where(causal & in_window, 0.0, -1e9).astype(np.float32)
            scores = scores + Tensor(
                mask_2d.reshape(1, 1, W, K_len), requires_grad=False
            )
            attn = scores.softmax(dim=-1)
            attn = self.dropout(attn)
            out_chunks.append(attn @ v_chunk)

        out = _concat(out_chunks, dim=2)
        if pad_len > 0:
            out = out[:, :, :T_q, :]
        return out

    # ------------------------------------------------------------------
    # 单步递推 forward_recurrent（只缓存 latent，重建 K/V）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x_single: Tensor, state: Optional[dict]):
        """单步递推推理（MLA：缓存 latent ``c``，逐步重建 K/V）。"""
        B, T, D = x_single.shape
        assert T == 1, f"forward_recurrent requires T=1, got T={T}"
        H, d = self.n_head, self.head_dim
        n_kv = self.n_kv_head
        W = self.window_size
        scale = 1.0 / (d ** 0.5)

        if state is None:
            c_cache = None
            position = 0
        else:
            c_cache = state["c_cache"]
            position = state["position"]

        with no_grad():
            # 1. Q 投影 + K/V 压缩（单 token）
            q = self.wq(x_single).reshape(B, H, d)
            c = self.w_kv_down(x_single).reshape(B, self.latent_dim)

            # 2. 更新 latent cache
            c_new = c.data[:, None, :]
            if c_cache is None:
                c_cache_arr = c_new
            else:
                c_cache_arr = np.concatenate([c_cache, c_new], axis=1)
            n_cached = c_cache_arr.shape[1]

            # 3. 从 latent 重建全部 K/V（numpy）
            c_t = Tensor(c_cache_arr, requires_grad=False)
            k_all = self.w_k_up(c_t).data.reshape(B, n_cached, n_kv, d)
            v_all = self.w_v_up(c_t).data.reshape(B, n_cached, n_kv, d)

            # 4. RoPE：q 与重建后的 k（全部历史）
            if self.use_rope:
                q_4d, _ = self._apply_rope(
                    q.reshape(B, 1, H, d),
                    None,
                    position,
                )
                q = q_4d.reshape(B, H, d)
                k_all_t = Tensor(k_all, requires_grad=False)
                k_rope = self._apply_rope(
                    k_all_t, None, 0,
                )[0].data.reshape(B, n_cached, n_kv, d)
                k_all = k_rope

            q_data = q.data

            # 5. GQA repeat
            if self.n_rep > 1:
                k_rep = np.repeat(k_all, self.n_rep, axis=2)
                v_rep = np.repeat(v_all, self.n_rep, axis=2)
            else:
                k_rep = k_all
                v_rep = v_all
            k_rep_t = np.transpose(k_rep, (0, 2, 1, 3))
            v_rep_t = np.transpose(v_rep, (0, 2, 1, 3))

            # 6. 注意力：短序列全因果；长序列窗口
            if n_cached <= self._MAX_T:
                scores = np.einsum("bhd,bhmd->bhm", q_data, k_rep_t) * scale
                attn = _np_softmax(scores, axis=-1)
                out = np.einsum("bhm,bhmd->bhd", attn, v_rep_t)
            else:
                w = min(n_cached, W)
                k_win = k_rep_t[:, :, -w:, :]
                v_win = v_rep_t[:, :, -w:, :]
                scores = np.einsum("bhd,bhmd->bhm", q_data, k_win) * scale
                attn = _np_softmax(scores, axis=-1)
                out = np.einsum("bhm,bhmd->bhd", attn, v_win)

            # 7. head 级 gate
            gate_b = _np_sigmoid(self.head_gate_logits.data).reshape(1, H, 1)
            out = out * gate_b

            out = out.reshape(B, 1, D)
            out_tensor = Tensor(out, requires_grad=False)
            if self.proj is not None:
                out_tensor = self.proj(out_tensor)

            new_state = {
                "c_cache": c_cache_arr,
                "position": position + 1,
            }

        return out_tensor, new_state


# ---------------------------------------------------------------------------
# VDA：三路混合（DSA + KDA + GMLA）gated 融合
# ---------------------------------------------------------------------------


class VDAAttention(Module):
    """CometFuture VerseNext Delta Attention（VDA）。

    三路并行计算并加权融合（与 :class:`TriSparseAttention` 的 gate 模式一致）：
    - 路径 A（DSA）：DeepSeek 风格 head-gated 稀疏注意力
    - 路径 B（KDA）：Kimi 风格 anchor 注意力（窗口 + 全局 anchor）
    - 路径 C（GMLA）：VerseNext Gated MLA（低秩 KV + head gate）

    三路输出按可学习 gate ``(3,)``（sigmoid）加权求和后经 ``proj`` 投影。
    KV cache 为嵌套结构：``{"dsa": ..., "kda": ..., "gmla": ...}``。
    """

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        window_size: int = 512,
        num_global_tokens: int = 64,
        anchor_interval: int = 128,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        # 三条路径（各自独立投影，输出不投影，由本层统一 proj）
        self.dsa = DSAAttention(
            dim=dim, n_head=n_head, n_kv_head=n_kv_head,
            window_size=window_size, num_global_tokens=num_global_tokens,
            use_rope=use_rope, max_seq_len=max_seq_len, dropout=dropout,
            rope_theta=rope_theta, include_proj=False,
        )
        self.kda = KDAAttention(
            dim=dim, n_head=n_head, n_kv_head=n_kv_head,
            window_size=window_size, anchor_interval=anchor_interval,
            use_rope=use_rope, max_seq_len=max_seq_len, dropout=dropout,
            rope_theta=rope_theta, include_proj=False,
        )
        self.gmla = GMLAAttention(
            dim=dim, n_head=n_head, n_kv_head=n_kv_head,
            latent_dim=latent_dim, window_size=window_size,
            use_rope=use_rope, max_seq_len=max_seq_len, dropout=dropout,
            rope_theta=rope_theta, include_proj=False,
        )
        # 输出投影（融合后）
        self.proj = Linear(dim, dim, bias=False)
        # 三路融合 gate（logits 初始化为 [0,0,0] → sigmoid=0.5 等权）
        self.gate_logits = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor, position_offset: int = 0, kv_cache: dict = None):
        """整序列并行计算（可微，用于训练）。

        Args:
            x: ``(B, T, D)`` Tensor
            position_offset: query 在全局序列中的起始位置
            kv_cache: 可选嵌套 dict（``'dsa' / 'kda' / 'gmla'`` 各为子路径 cache）
        Returns:
            out: ``(B, T, D)`` Tensor
            new_kv_cache: 嵌套 dict（已 detach）
        """
        sub_kv = kv_cache if kv_cache is not None else {}

        dsa_out, dsa_cache = self.dsa.forward(
            x, position_offset=position_offset, kv_cache=sub_kv.get("dsa"),
        )
        kda_out, kda_cache = self.kda.forward(
            x, position_offset=position_offset, kv_cache=sub_kv.get("kda"),
        )
        gmla_out, gmla_cache = self.gmla.forward(
            x, position_offset=position_offset, kv_cache=sub_kv.get("gmla"),
        )

        gate = self.gate_logits.sigmoid()  # (3,)
        out = gate[0] * dsa_out + gate[1] * kda_out + gate[2] * gmla_out
        out = self.proj(out)

        new_kv_cache = {
            "dsa": dsa_cache,
            "kda": kda_cache,
            "gmla": gmla_cache,
        }
        return out, new_kv_cache

    def forward_recurrent(self, x_single: Tensor, state: Optional[dict]):
        """单步递推推理。

        Args:
            x_single: ``(B, 1, D)`` Tensor
            state: 嵌套 dict 或 None（``'dsa' / 'kda' / 'gmla'``）
        Returns:
            out: ``(B, 1, D)`` Tensor
            new_state: 嵌套 dict
        """
        sub_state = state if state is not None else {}

        dsa_out, dsa_state = self.dsa.forward_recurrent(
            x_single, sub_state.get("dsa"),
        )
        kda_out, kda_state = self.kda.forward_recurrent(
            x_single, sub_state.get("kda"),
        )
        gmla_out, gmla_state = self.gmla.forward_recurrent(
            x_single, sub_state.get("gmla"),
        )

        gate = _np_sigmoid(self.gate_logits.data)  # (3,)
        out = gate[0] * dsa_out.data + gate[1] * kda_out.data + gate[2] * gmla_out.data
        out_tensor = Tensor(out, requires_grad=False)
        out_tensor = self.proj(out_tensor)

        new_state = {
            "dsa": dsa_state,
            "kda": kda_state,
            "gmla": gmla_state,
        }
        return out_tensor, new_state


# ---------------------------------------------------------------------------
# 层分配规则：build_verse_delta_pattern
# ---------------------------------------------------------------------------


def build_verse_delta_pattern(n_layer: int) -> list:
    """生成 VDA 架构的注意力层分配规则（Part6）。

    - ``n_layer < 6``：全部使用 VDA（混合注意力）
    - ``6 <= n_layer < 10``：前 4 层 VDA，其余 GMLA
    - ``n_layer >= 10``：前 4 层 VDA，后 2 层 DSA，其余 GMLA

    返回的 ``list[str]`` 每元素为 ``"vda" / "dsa" / "gmla"`` 之一，
    作为 :class:`CometSparkNexLM` 的 ``attn_pattern`` 使用。

    Examples:
        >>> build_verse_delta_pattern(2)
        ['vda', 'vda']
        >>> build_verse_delta_pattern(8)
        ['vda', 'vda', 'vda', 'vda', 'gmla', 'gmla', 'gmla', 'gmla']
        >>> build_verse_delta_pattern(10)
        ['vda', 'vda', 'vda', 'vda', 'gmla', 'gmla', 'gmla', 'gmla', 'dsa', 'dsa']
        >>> build_verse_delta_pattern(32)
        ['vda', 'vda', 'vda', 'vda', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla',
         'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla',
         'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'gmla',
         'gmla', 'gmla', 'gmla', 'gmla', 'gmla', 'dsa', 'dsa']
    """
    if n_layer < 1:
        raise ValueError(f"n_layer 必须 >= 1，got {n_layer}")
    if n_layer < 6:
        return ["vda"] * n_layer
    if n_layer < 10:
        return ["vda"] * 4 + ["gmla"] * (n_layer - 4)
    return ["vda"] * 4 + ["gmla"] * (n_layer - 6) + ["dsa"] * 2


# 注意力类型 → 实现类映射（供 VerseNexBlock 按 attn_kind 构造）
DELTA_ATTN_CLASSES = {
    "dsa": DSAAttention,
    "kda": KDAAttention,
    "gmla": GMLAAttention,
    "vda": VDAAttention,
}


__all__ = [
    "DSAAttention",
    "KDAAttention",
    "GMLAAttention",
    "VDAAttention",
    "build_verse_delta_pattern",
    "DELTA_ATTN_CLASSES",
]

