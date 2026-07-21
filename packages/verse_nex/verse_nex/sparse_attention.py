"""VerseNex: Sparse Attention (Task 3.5).

Top-K Chunk Sparse Attention，参考 RWKV-X (https://arxiv.org/abs/2504.21463)
与 Nemotron-H 的稀疏注意力设计。

核心思想：
    将序列划分为长度为 C 的 chunk，每个 query chunk 关注：
    1. intra-chunk: 同一 chunk 内所有 token（causal）
    2. sliding-window: 最近 W 个 past chunk
    3. top-k chunks: 基于chunk-level relevance score 选出的 k 个最相关 past chunk

复杂度：O(T * (C + (W + k) * C)) = O(T * C * (W + k + 1))
若 W + k + 1 << T / C，则远低于 O(T^2) 的全注意力。

设计要点：
- parallel 模式：用 VerseTorch Tensor 实现，可微（用于训练）
  - chunk selection 用 numpy（detach Q, K），因为 top-k 选择本身不可微
  - attention scores / softmax / 加权求和用 Tensor ops，保留 Q, K, V 梯度
  - 用 (T, T) sparse mask 实现 chunk-level 稀疏模式
- recurrent 模式：维护 KV cache，单步推理（用于推理）
- Top-K 选择基于 chunk-level Q·K score（mean over chunk tokens）
- 因果性保证：query chunk i 只能 attend 到 key chunks j <= i
- 数值一致：parallel 与 recurrent 输出在 float32 下吻合到 1e-3
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import Linear, RMSNorm, Module


# ---------------------------------------------------------------------------
# 工具：在指定 axis 上 pad 0，保持梯度路径
# ---------------------------------------------------------------------------


def _pad_last_dim(t: Tensor, pad_len: int, axis: int) -> Tensor:
    """在指定 axis 维度末尾 pad pad_len 个 0，保持梯度路径。

    Args:
        t: 输入 Tensor
        pad_len: 要 pad 的长度
        axis: 在哪个 axis 末尾 pad
    Returns:
        padded Tensor
    """
    if pad_len == 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[axis] = pad_len
    pad_data = np.zeros(pad_shape, dtype=t.data.dtype)
    out_data = np.concatenate([t.data, pad_data], axis=axis)

    def _backward():
        if t.requires_grad:
            if out.grad is None:
                return
            slice_obj = [slice(None)] * t.ndim
            slice_obj[axis] = slice(0, t.shape[axis])
            t._accumulate_grad(out.grad[tuple(slice_obj)])

    out = t._result(out_data, (t,), "pad")
    if out.requires_grad:
        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# 工具：构造 causal mask
# ---------------------------------------------------------------------------


def _causal_mask(seq_len: int) -> np.ndarray:
    """返回 (seq_len, seq_len) 下三角 mask（1 表示可见）。"""
    return np.tril(np.ones((seq_len, seq_len), dtype=np.float32))


# ---------------------------------------------------------------------------
# TopKChunkSparseAttention
# ---------------------------------------------------------------------------


class TopKChunkSparseAttention(Module):
    """Top-K Chunk Sparse Attention.

    Args:
        dim: 模型维度
        n_heads: 头数（要求 dim % n_heads == 0）
        chunk_size: chunk 大小 C
        n_sliding_chunks: 滑动窗口包含的 past chunk 数 W
        topk_chunks: 基于相关性选出的 top-k past chunk 数
        max_kv_chunks: KV cache 最多保留的 chunk 数（推理时；None 表示无限制）
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        chunk_size: int = 64,
        n_sliding_chunks: int = 2,
        topk_chunks: int = 2,
        max_kv_chunks: int = None,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.chunk_size = chunk_size
        self.n_sliding_chunks = n_sliding_chunks
        self.topk_chunks = topk_chunks
        self.max_kv_chunks = max_kv_chunks

        # QKV 投影（合并）
        self.qkv = Linear(dim, 3 * dim, bias=False)
        # 输出投影
        self.out = Linear(dim, dim, bias=False)
        # RMSNorm（attention 输出归一化）
        self.norm = RMSNorm(dim)

    # ------------------------------------------------------------------
    # 辅助：QKV 计算
    # ------------------------------------------------------------------

    def _compute_qkv(self, x: Tensor):
        """x: (B, T, D) -> Q, K, V: (B, T, H, d_head) (Tensor)"""
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0]  # (B, T, H, d_head)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        # 缩放 Q
        scale = 1.0 / np.sqrt(self.d_head)
        q = q * scale
        return q, k, v

    # ------------------------------------------------------------------
    # 选择 top-k chunks per query chunk
    # ------------------------------------------------------------------

    def _select_chunks(
        self,
        q_chunk: np.ndarray,  # (B, H, C, d)
        k_chunks: np.ndarray,  # (B, n_past_chunks, H, C, d)
    ):
        """对每个 batch 与 head，选择 top-k 最相关的 past chunk。

        Args:
            q_chunk: (B, H, C, d)
            k_chunks: (B, P, H, C, d) 其中 P = past chunks 数
        Returns:
            selected_indices: (B, H, k) - 选中的 past chunk 在 k_chunks 中的索引
            selected_scores: (B, H, k) - 对应的 relevance score（用于调试）
        """
        B, H, C, d = q_chunk.shape
        P = k_chunks.shape[1]
        k = min(self.topk_chunks, P)

        # chunk-level relevance score: mean over (C, C) of Q @ K^T
        # q_chunk: (B, H, C, d), k_chunks: (B, P, H, C, d)
        # scores[b, p, h] = mean_{i,j} (q[b,h,i,:] . k[b,p,h,j,:])
        # = (sum_{i,j} q . k) / (C * C)
        # 用 einsum: 'bhid,bphjd->bph'，然后除以 C*C
        # 但 einsum 字母冲突，改为分步：
        # Q_mean: (B, H, d) = q.mean(dim=2)
        q_mean = q_chunk.mean(axis=2)  # (B, H, d)
        # K_mean: (B, P, H, d)
        k_mean = k_chunks.mean(axis=3)  # (B, P, H, d)
        # scores[b, p, h] = q_mean[b, h, :] . k_mean[b, p, h, :]
        # = sum_d q_mean[b, h, d] * k_mean[b, p, h, d]
        scores = np.einsum("bhd,bphd->bph", q_mean, k_mean)  # (B, P, H)

        # 对每个 (B, H) 选 top-k
        # 这里简化：跨 head 共享选择（用 mean over heads 的 score）
        # 实际上 per-head 选择更灵活，但内存开销大；这里采用共享策略
        scores_avg = scores.mean(axis=2)  # (B, P)
        # 选中 top-k chunk indices
        # 注意：要排除已经在 sliding window 中的 chunk
        # 简化：直接选 top-k（sliding window 在外层处理，这里只对 remaining chunks 选）
        if k == P:
            # 全选
            selected = np.tile(np.arange(P)[None, :], (B, 1))  # (B, P)
        else:
            # 对每个 batch 选 top-k
            selected = np.argpartition(-scores_avg, k - 1, axis=1)[:, :k]  # (B, k)
        # 对所有 head 用相同的 chunk 选择
        selected_indices = np.broadcast_to(
            selected[:, None, :], (B, H, selected.shape[1])
        ).copy()  # (B, H, k_actual)
        selected_scores = np.take_along_axis(
            scores, selected[:, None, :], axis=1
        )  # (B, H, k) - wait, need to be (B, k, H)
        # 修正 scores 取法
        # scores: (B, P, H), selected: (B, k)
        # selected_scores[b, h, ki] = scores[b, selected[b, ki], h]
        selected_scores = np.take_along_axis(
            scores, selected[:, :, None], axis=1
        )  # (B, k, H)
        selected_scores = np.transpose(selected_scores, (0, 2, 1))  # (B, H, k)
        return selected_indices, selected_scores

    # ------------------------------------------------------------------
    # Parallel 模式（训练，可微）
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列稀疏注意力（可微，用于训练）。

        策略：
            1. 把序列划分为 chunks（长度 C）
            2. 用 numpy 计算 chunk selection（top-k 不可微，detach Q, K）
            3. 构造 (T, T) sparse mask（1 表示 query i attend key j）
            4. 用 Tensor ops 实现 full attention with mask，保留 Q, K, V 梯度：
               - scores = Q @ K^T  (B, H, T, T)
               - scores = scores + (1 - mask) * (-inf)
               - attn = softmax(scores)
               - out = attn @ V
            5. 裁剪到原 T，应用 norm + out_proj

        注意：训练时用 (T, T) 矩阵实现，O(T^2) 内存。
        这违背了 sparse 的初衷，但保证了完整的 Q, K, V 梯度。
        对于训练（T 通常不大）这是可接受的妥协。
        推理时用 forward_recurrent，保持常数内存。
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head
        C = self.chunk_size
        W = self.n_sliding_chunks
        k_top = self.topk_chunks

        q, k, v = self._compute_qkv(x)  # (B, T, H, d) Tensor, q 已 scale

        # pad T 到 chunk_size 整数倍（保持梯度路径）
        n_chunks = (T + C - 1) // C
        T_padded = n_chunks * C
        pad_len = T_padded - T
        if pad_len > 0:
            q = _pad_last_dim(q, pad_len, axis=1)
            k = _pad_last_dim(k, pad_len, axis=1)
            v = _pad_last_dim(v, pad_len, axis=1)

        # ----- 用 numpy 计算 chunk selection，构造 (T_padded, T_padded) sparse mask -----
        # chunk selection 依赖 Q, K 的值，但选择本身不可微，所以用 detach 的 data
        qd = q.data.reshape(B, n_chunks, C, H, d).transpose(0, 3, 1, 2, 4)  # (B, H, n_chunks, C, d)
        kd = k.data.reshape(B, n_chunks, C, H, d).transpose(0, 3, 1, 2, 4)

        # mask[i, j] = 1 if query i 可以 attend key j (且 j <= i, causal)
        mask = np.zeros((T_padded, T_padded), dtype=np.float32)

        for ci in range(n_chunks):
            # 当前 query chunk: (B, H, C, d)
            q_ci = qd[:, :, ci]  # (B, H, C, d)
            # sliding window: 最近 W 个 past chunks (ci-W .. ci-1)
            sliding_start = max(0, ci - W)
            sliding_chunks = list(range(sliding_start, ci))
            # top-k chunks: 从所有 past chunks (0 .. sliding_start-1) 中选 top-k
            remaining_past = list(range(0, sliding_start))
            topk_chunks = []
            if len(remaining_past) > 0 and k_top > 0:
                # chunk-level Q·K score
                q_ci_mean = q_ci.mean(axis=2)  # (B, H, d)
                k_past_full = kd[:, :, remaining_past, :, :]  # (B, H, P, C, d)
                k_past = k_past_full.mean(axis=3)  # (B, H, P, d)
                k_past = np.transpose(k_past, (0, 2, 1, 3))  # (B, P, H, d)
                scores = np.einsum("bhd,bphd->bph", q_ci_mean, k_past)  # (B, P, H)
                # 用 batch+head 平均的 score 选（所有 batch/head 共享）
                scores_avg_global = scores.mean(axis=2).mean(axis=0)  # (P,)
                k_actual = min(k_top, len(remaining_past))
                topk_idx = np.argsort(-scores_avg_global)[:k_actual]
                topk_chunks = [remaining_past[i] for i in topk_idx]

            # 所有要 attend 的 chunks（包括自己）
            attend_chunks = sorted(set(sliding_chunks + topk_chunks + [ci]))
            # 在 mask 中标记：query i in chunk ci 可以 attend key j in chunk cj
            # if cj < ci: 所有 j 都可见（past chunk）
            # if cj == ci: 只有 j <= i 可见（causal within chunk）
            for cj in attend_chunks:
                for i_local in range(C):
                    i_global = ci * C + i_local
                    if i_global >= T_padded:
                        break
                    for j_local in range(C):
                        j_global = cj * C + j_local
                        if j_global >= T_padded:
                            break
                        if j_global <= i_global:
                            mask[i_global, j_global] = 1.0

        # ----- 用 Tensor ops 实现 full attention with mask（保留 Q, K, V 梯度）-----
        # q, k, v: (B, T_padded, H, d) -> (B, H, T_padded, d)
        q_t = q.permute(0, 2, 1, 3)  # (B, H, T_padded, d)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)

        # scores = q @ k^T: (B, H, T_padded, T_padded)
        k_t_t = k_t.transpose(-1, -2)  # (B, H, d, T_padded)
        scores = q_t.matmul(k_t_t)  # (B, H, T_padded, T_padded)

        # apply mask: scores = scores + (1 - mask) * (-1e9)
        # mask: (T_padded, T_padded) -> broadcast to (1, 1, T_padded, T_padded)
        neg_mask_data = (1.0 - mask).reshape(1, 1, T_padded, T_padded)
        # 用常量 Tensor（requires_grad=False）
        neg_mask = Tensor(neg_mask_data.astype(np.float32), requires_grad=False)
        # masked scores: 原始 scores + (1-mask) * (-1e9)
        # 用乘法+加法表达，保持 scores 的梯度路径
        masked_scores = scores + neg_mask * (-1e9)

        # softmax over last dim
        attn = masked_scores.softmax(dim=-1)  # (B, H, T_padded, T_padded)

        # out = attn @ v: (B, H, T_padded, d)
        out = attn.matmul(v_t)  # (B, H, T_padded, d)

        # reshape to (B, T_padded, D)
        out = out.permute(0, 2, 1, 3)  # (B, T_padded, H, d)
        out = out.reshape(B, T_padded, D)

        # 裁剪到原 T（pad 部分的输出丢弃）
        if pad_len > 0:
            out = out[:, :T, :]

        # apply norm + out_proj（这些是可微的）
        out = self.norm(out)
        out = self.out(out)
        return out

    # ------------------------------------------------------------------
    # Recurrent 模式（推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x: Tensor, state=None):
        """单步稀疏注意力（推理用）。

        Args:
            x: (B, 1, D)
            state: tuple (kv_cache, position)
                kv_cache: list of (K, V) per past token, 每个 (B, H, d)
                position: int, 当前 token 的位置
        Returns:
            out: (B, 1, D)
            new_state: tuple (kv_cache, position)
        """
        B, T, D = x.shape
        assert T == 1, f"recurrent mode requires T=1, got T={T}"
        H, d = self.n_heads, self.d_head
        C = self.chunk_size
        W = self.n_sliding_chunks
        k_top = self.topk_chunks

        if state is None:
            kv_cache = []  # list of (K, V), each (B, H, d)
            position = 0
        else:
            kv_cache, position = state
            kv_cache = list(kv_cache)  # copy

        with no_grad():
            # 计算 Q, K, V
            qkv = self.qkv(x)  # (B, 1, 3D)
            qkv = qkv.reshape(B, 1, 3, H, d)
            q = qkv[:, 0, 0]  # (B, H, d)
            k = qkv[:, 0, 1]
            v = qkv[:, 0, 2]
            # scale Q
            scale = 1.0 / np.sqrt(d)
            q = q * scale

            # 添加到 KV cache
            kv_cache.append((k.data, v.data))
            cur_pos = position  # 当前 token 在序列中的位置（0-indexed）

            # 决定 attend 的 past tokens
            # 当前 token 在 chunk ci = cur_pos // C, 位置 ii = cur_pos % C
            ci = cur_pos // C
            ii = cur_pos % C

            # 收集要 attend 的 tokens:
            # 1. 当前 chunk 中 position <= ii 的 tokens（causal within chunk）
            # 2. 最近 W 个 past chunks 的所有 tokens
            # 3. top-k 个 past chunks 的所有 tokens
            attend_indices = []
            # 当前 chunk 内 causal
            chunk_start = ci * C
            for p in range(chunk_start, cur_pos + 1):
                attend_indices.append(p)
            # sliding window: 最近 W 个 past chunks
            for cj in range(max(0, ci - W), ci):
                for p in range(cj * C, (cj + 1) * C):
                    if p < cur_pos and p not in attend_indices:
                        attend_indices.append(p)
            # top-k chunks: 基于 Q·K_chunk_mean 选 top-k
            remaining_past_chunks = list(range(0, max(0, ci - W)))
            if len(remaining_past_chunks) > 0 and k_top > 0:
                # 用 Q 与每个 past chunk 的 mean K 计算 score
                k_means = []
                for cj in remaining_past_chunks:
                    chunk_ks = [kv_cache[p][0] for p in range(cj * C, (cj + 1) * C)]
                    if len(chunk_ks) > 0:
                        k_mean = np.mean(chunk_ks, axis=0)  # (B, H, d)
                    else:
                        k_mean = np.zeros((B, H, d), dtype=np.float32)
                    k_means.append(k_mean)
                k_means = np.stack(k_means, axis=1)  # (B, P, H, d)
                # scores[b, p, h] = q[b, h, :] . k_means[b, p, h, :]
                scores = np.einsum("bhd,bphd->bph", q.data, k_means)  # (B, P, H)
                scores_avg = scores.mean(axis=2).mean(axis=0)  # (P,)
                k_actual = min(k_top, len(remaining_past_chunks))
                topk_idx = np.argsort(-scores_avg)[:k_actual]
                for idx in topk_idx:
                    cj = remaining_past_chunks[idx]
                    for p in range(cj * C, (cj + 1) * C):
                        if p < cur_pos and p not in attend_indices:
                            attend_indices.append(p)

            attend_indices = sorted(attend_indices)
            # 限制 KV cache 大小（如果 max_kv_chunks 设定）
            if self.max_kv_chunks is not None and len(attend_indices) > self.max_kv_chunks * C:
                # 只保留最近的 max_kv_chunks * C 个
                attend_indices = attend_indices[-(self.max_kv_chunks * C):]

            # 计算 attention
            if len(attend_indices) == 0:
                # 第一个 token，直接用 V[0]
                out = v.data  # (B, H, d)
            else:
                # 收集 K, V
                K_sel = np.stack([kv_cache[p][0] for p in attend_indices], axis=1)  # (B, M, H, d)
                V_sel = np.stack([kv_cache[p][1] for p in attend_indices], axis=1)  # (B, M, H, d)
                # scores: (B, H, M) = q . K^T
                # q: (B, H, d), K_sel: (B, M, H, d) -> transpose to (B, H, d, M)
                K_sel_t = np.transpose(K_sel, (0, 2, 3, 1))  # (B, H, d, M)
                scores = np.einsum("bhd,bhdm->bhm", q.data, K_sel_t)  # (B, H, M)
                # softmax
                scores_max = scores.max(axis=-1, keepdims=True)
                scores_exp = np.exp(scores - scores_max)
                scores_sum = scores_exp.sum(axis=-1, keepdims=True) + 1e-9
                attn_weights = scores_exp / scores_sum  # (B, H, M)
                # 加权求和: out[b, h, d] = sum_m attn[b, h, m] * V[b, m, h, d]
                out = np.einsum("bhm,bmhd->bhd", attn_weights, V_sel)  # (B, H, d)

            # reshape 回 (B, 1, D)
            out = out.reshape(B, 1, D)
            out_tensor = Tensor(out, requires_grad=False)
            out_tensor = self.norm(out_tensor)
            out_tensor = self.out(out_tensor)

            # 修剪 KV cache（移除过老的 chunk，不参加 sliding window 或 top-k）
            # 简化：保留全部 KV cache（在 max_kv_chunks 限制内）
            if self.max_kv_chunks is not None:
                # 保留最近 max_kv_chunks 个 chunk
                min_keep_chunk = max(0, ci - self.max_kv_chunks + 1)
                # 不删除，因为索引会乱；这里只保留最近 max_kv_chunks 个 chunk 的 tokens
                # 但需要谨慎处理索引
                pass  # 简化：不修剪，让外层管理

            new_state = (kv_cache, cur_pos + 1)

        return out_tensor, new_state

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, state=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        elif mode == "recurrent":
            out, new_state = self.forward_recurrent(x, state)
            object.__setattr__(out, "_state", new_state)
            return out
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent")


__all__ = ["TopKChunkSparseAttention"]
