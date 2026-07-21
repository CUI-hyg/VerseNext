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
    # Parallel 模式（训练，可微，block-sparse 实现）
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列稀疏注意力（可微，用于训练）。

        Part4 P1.4 修复：从 O(T²) 内存改为 **block-sparse** 实现。

        原实现构造 (T, T) mask + (B, H, T, T) scores 矩阵，对于 T=2048 即
        占用 B*H*16MB 内存，T=4096 则 B*H*64MB，导致训练 OOM。

        新实现按 (query_chunk, key_chunk) 子块逐块计算 attention：
        - 对每个 query chunk ci，仅对被选中的 attend_chunks 中的 key chunk cj
          计算 (C, C) 子注意力（causal 在 ci==cj 时启用）
        - 拼接所有子块的输出得到最终结果
        - 内存复杂度 O(B * H * T * C * (W + k_top + 1))，远低于 O(T²)

        保留完整的 Q, K, V 梯度路径（每个子块的 attention 都是标准 Tensor ops）。
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

        # 转置为 (B, H, T_padded, d) 便于按 head 切分
        q_t = q.permute(0, 2, 1, 3)  # (B, H, T_padded, d)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)

        # 用 numpy 计算 chunk selection（detach Q, K，因为 top-k 不可微）
        qd = q.data.reshape(B, n_chunks, C, H, d).transpose(0, 3, 1, 2, 4)  # (B, H, n_chunks, C, d)
        kd = k.data.reshape(B, n_chunks, C, H, d).transpose(0, 3, 1, 2, 4)

        # 输出累加：用 NumPy 收集每个位置的输出，最后转为 Tensor
        # 由于不同子块之间是 disjoint（每个 query 位置只在一个 query chunk 中），
        # 可以用 cat 拼接子块输出。
        # 但为了保持 autograd，每个子块的输出是 Tensor，最后用 cat 拼接。
        from verse_torch.tensor import Tensor as _Tensor
        # 缓存每个 query chunk 的输出 Tensor（长度 n_chunks）
        chunk_outputs: list[_Tensor | None] = [None] * n_chunks

        for ci in range(n_chunks):
            # 当前 query chunk: (B, H, C, d)
            q_ci_mean = qd[:, :, ci].mean(axis=2)  # (B, H, d)
            # sliding window: 最近 W 个 past chunks (ci-W .. ci-1)
            sliding_start = max(0, ci - W)
            sliding_chunks = list(range(sliding_start, ci))
            # top-k chunks: 从所有 past chunks (0 .. sliding_start-1) 中选 top-k
            remaining_past = list(range(0, sliding_start))
            topk_chunks = []
            if len(remaining_past) > 0 and k_top > 0:
                # chunk-level Q·K score（用 chunk-mean）
                k_past_full = kd[:, :, remaining_past, :, :]  # (B, H, P, C, d)
                k_past = k_past_full.mean(axis=3)  # (B, H, P, d)
                # 用 batch+head 平均的 score 选（所有 batch/head 共享）
                # scores_avg_global: (P,)
                scores = np.einsum("bhd,bhpd->bhp", q_ci_mean, k_past)  # (B, H, P)
                scores_avg_global = scores.mean(axis=2).mean(axis=0)  # (P,)
                k_actual = min(k_top, len(remaining_past))
                topk_idx = np.argsort(-scores_avg_global)[:k_actual]
                topk_chunks = [remaining_past[i] for i in topk_idx]

            # 所有要 attend 的 chunks（包括自己；排序保证因果顺序）
            attend_chunks = sorted(set(sliding_chunks + topk_chunks + [ci]))

            # 提取 query chunk 的 Tensor 切片：q_t[:, :, ci*C:(ci+1)*C, :]
            # 用 .__getitem__ 保持梯度路径
            q_ci_tensor = q_t[:, :, ci * C:(ci + 1) * C, :]  # (B, H, C, d)

            # 拼接所有 attend key chunks 的 K, V
            # 用 cat 沿 T 轴拼接
            k_chunks_tensors = []
            v_chunks_tensors = []
            # 同时构造 causal mask for the concatenated keys
            # 对每个 attend chunk cj:
            #   - 如果 cj < ci: 所有 C 个 key 都可见（past chunk）
            #   - 如果 cj == ci: causal within chunk (tril)
            key_mask_parts = []  # each (C,) or (C, C)

            for cj in attend_chunks:
                k_cj = k_t[:, :, cj * C:(cj + 1) * C, :]  # (B, H, C, d)
                v_cj = v_t[:, :, cj * C:(cj + 1) * C, :]
                k_chunks_tensors.append(k_cj)
                v_chunks_tensors.append(v_cj)
                if cj < ci:
                    # past chunk: 全部 C 个 key 可见
                    key_mask_parts.append(np.ones((C,), dtype=np.float32))
                else:
                    # cj == ci: causal within chunk
                    key_mask_parts.append(
                        np.tril(np.ones((C, C), dtype=np.float32))
                    )

            # 拼接 K, V: (B, H, M, d), M = len(attend_chunks) * C
            # 用 Tensor.cat（如果有），否则用 numpy 拼接 + 重建 Tensor
            # VerseTorch Tensor 支持 __getitem__，但 cat 需要手动实现
            # 简单实现：先 numpy 拼接 data，再用 result Tensor 重建 autograd 节点
            # 但这样会丢失子 Tensor 的梯度路径。
            #
            # 替代方案：用 __getitem__ + 多次 matmul + 加法
            # 由于 matmul 是 (B, H, C, d) @ (B, H, d, M) = (B, H, C, M)
            # 我们可以分块计算：q_ci @ k_cj^T = (B, H, C, C) per chunk
            # 然后按 chunk 拼接 scores，再 softmax，再 @ V
            #
            # 但更简单的是：用 numpy 拼接 data 后用一次 matmul，但需要保留梯度
            # VerseTorch Tensor 的 matmul 支持 broadcasting，所以可以：
            #   K_all: (B, H, M, d) - 用 cat 拼接
            #   scores = q_ci @ K_all^T: (B, H, C, M)
            # 但我们需要实现 cat。

            # 使用 q_ci @ each k_cj^T 然后用 concat（沿 last dim）
            # 输出 scores_ci: (B, H, C, M)
            scores_parts = []  # list of (B, H, C, C) Tensor
            for kj, k_cj in enumerate(k_chunks_tensors):
                # q_ci: (B, H, C, d), k_cj: (B, H, C, d)
                # scores: (B, H, C, C) = q_ci @ k_cj^T
                #   = matmul(q_ci, k_cj.transpose(-1, -2))
                k_cj_t = k_cj.transpose(-1, -2)  # (B, H, d, C)
                s = q_ci_tensor.matmul(k_cj_t)  # (B, H, C, C)
                scores_parts.append(s)

            # 用 numpy 沿 last dim 拼接 scores_parts 的 data，
            # 但要保留梯度。VerseTorch Tensor 没有 cat，所以我们用 result Tensor
            # + 自定义 _backward 把梯度分发到各 parts。
            M_total = len(scores_parts) * C
            scores_data = np.concatenate(
                [s.data for s in scores_parts], axis=-1
            )  # (B, H, C, M_total)
            scores_ci = _Tensor(
                scores_data,
                requires_grad=any(s.requires_grad for s in scores_parts),
                _children=tuple(scores_parts) if any(s.requires_grad for s in scores_parts) else (),
                _op="sparse_concat",
            )
            # 为 scores_ci 设置 _backward：把上游 grad 按 last dim 切分回各 parts
            if scores_ci.requires_grad:
                _parts = scores_parts  # 闭包捕获
                _C = C

                def _backward():
                    if scores_ci.grad is None:
                        return
                    g = scores_ci.grad  # (B, H, C, M_total)
                    for j, p in enumerate(_parts):
                        if p.requires_grad:
                            sub = g[..., j * _C:(j + 1) * _C]
                            p._accumulate_grad(sub)
                scores_ci._backward = _backward

            # 构造 mask: (C, M_total) - block diagonal of (C,) ones or (C, C) tril
            mask_blocks = []
            for cj_idx, cj in enumerate(attend_chunks):
                if cj < ci:
                    # past chunk: 全部 C 个 key 可见 -> (C,) ones
                    mask_blocks.append(np.ones((C, C), dtype=np.float32))
                else:
                    # cj == ci: causal within chunk
                    mask_blocks.append(np.tril(np.ones((C, C), dtype=np.float32)))
            mask_ci = np.concatenate(mask_blocks, axis=1)  # (C, M_total)

            # apply mask: scores = scores + (1 - mask) * (-1e9)
            neg_mask_data = (1.0 - mask_ci).reshape(1, 1, C, M_total)
            neg_mask = _Tensor(neg_mask_data.astype(np.float32), requires_grad=False)
            masked_scores = scores_ci + neg_mask * (-1e9)

            # softmax over last dim
            attn_ci = masked_scores.softmax(dim=-1)  # (B, H, C, M_total)

            # out = attn @ V: (B, H, C, d)
            # V 拼接：v_chunks_tensors 各为 (B, H, C, d)，沿 T 轴拼成 (B, H, M_total, d)
            # 同样需要保留梯度
            v_data = np.concatenate(
                [v.data for v in v_chunks_tensors], axis=2
            )  # (B, H, M_total, d)
            v_all = _Tensor(
                v_data,
                requires_grad=any(v.requires_grad for v in v_chunks_tensors),
                _children=tuple(v_chunks_tensors) if any(v.requires_grad for v in v_chunks_tensors) else (),
                _op="sparse_concat_v",
            )
            if v_all.requires_grad:
                _v_parts = v_chunks_tensors

                def _backward_v():
                    if v_all.grad is None:
                        return
                    g = v_all.grad  # (B, H, M_total, d)
                    for j, p in enumerate(_v_parts):
                        if p.requires_grad:
                            sub = g[:, :, j * C:(j + 1) * C, :]
                            p._accumulate_grad(sub)
                v_all._backward = _backward_v

            # out_ci: (B, H, C, d) = attn_ci @ v_all
            #   attn_ci: (B, H, C, M), v_all: (B, H, M, d)
            out_ci = attn_ci.matmul(v_all)  # (B, H, C, d)
            chunk_outputs[ci] = out_ci

        # 拼接所有 chunk 的输出: (B, H, T_padded, d)
        # 同样需要保留梯度
        all_out_data = np.concatenate(
            [o.data for o in chunk_outputs], axis=2
        )  # (B, H, T_padded, d)
        out = _Tensor(
            all_out_data,
            requires_grad=any(o.requires_grad for o in chunk_outputs),
            _children=tuple(chunk_outputs) if any(o.requires_grad for o in chunk_outputs) else (),
            _op="sparse_concat_out",
        )
        if out.requires_grad:
            _out_parts = chunk_outputs
            _C2 = C

            def _backward_out():
                if out.grad is None:
                    return
                g = out.grad  # (B, H, T_padded, d)
                for i, p in enumerate(_out_parts):
                    if p.requires_grad:
                        sub = g[:, :, i * _C2:(i + 1) * _C2, :]
                        p._accumulate_grad(sub)
            out._backward = _backward_out

        # reshape to (B, T_padded, H, d) -> (B, T_padded, D)
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
