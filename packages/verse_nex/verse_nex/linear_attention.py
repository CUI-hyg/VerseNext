"""VerseNex: Linear Attention (Task 3.2, RetNet 风格).

RetNet: Retention Network (https://arxiv.org/abs/2307.08621)

核心公式：
    retention(x) = (Q @ K^T * D) @ V
    其中 D 是衰减矩阵：D[i,j] = gamma^(i-j) if i>=j else 0

三种计算模式：
- parallel (训练):  整个序列并行，O(T^2 * d) 但高度可并行
- recurrent (推理): 单步递推 s_t = K_t^T @ V_t + gamma * s_{t-1}; out_t = Q_t @ s_t
                    复杂度 O(T * d^2)，常数内存
- chunkwise (长序列): 块内 parallel + 块间 recurrent，平衡速度与内存

设计要点：
- parallel 路径用 VerseTorch Tensor 实现，保持可微（用于训练）
- recurrent 路径用 NumPy 实现，避免计算图开销（仅推理）
- 数值一致：parallel 与 recurrent 输出在 float32 下吻合到 1e-3
- 状态 shape: (B, n_heads, d_head, d_head)
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, nn, no_grad
from verse_torch.vnn import Linear, RMSNorm, Module


# ---------------------------------------------------------------------------
# 工具：构造衰减矩阵
# ---------------------------------------------------------------------------


def _decay_matrix(seq_len: int, gamma: float, dtype=np.float32) -> np.ndarray:
    """构造 (T, T) 衰减矩阵：D[i,j] = gamma^(i-j) if i>=j else 0。

    等价于 lower-triangular mask * gamma^rel，其中 rel = i-j >= 0。
    """
    # 相对距离矩阵 rel[i,j] = i - j
    idx = np.arange(seq_len, dtype=np.float32)
    rel = idx[:, None] - idx[None, :]  # (T, T)
    # 仅保留下三角
    mask = (rel >= 0).astype(dtype)
    D = (gamma ** rel) * mask  # (T, T)
    return D.astype(dtype)


def _decay_matrix_log(seq_len: int, log_gamma: float, dtype=np.float32) -> np.ndarray:
    """数值稳定版本：用 log 域 cumsum 构造衰减矩阵。

    log D[i,j] = (i-j) * log_gamma  for i>=j
    用 cumsum: log_decay[i,j] = cumsum_log[i] - cumsum_log[j]
    其中 cumsum_log[t] = sum_{s<=t} log_gamma_s（这里 log_gamma_s = log_gamma 恒定）
    """
    log_g = np.full(seq_len, log_gamma, dtype=np.float64)
    cumsum = np.cumsum(log_g)  # (T,)
    # log_decay[i,j] = (cumsum[i] - cumsum[j]) + log_gamma[j]（因为 D[i,j]=gamma^(i-j+1) or gamma^(i-j)）
    # 我们采用 D[i,j] = gamma^(i-j) for i>=j，所以从 j+1 到 i 的乘积
    # 即 log_decay[i,j] = cumsum[i] - cumsum[j]
    # 但当 i==j 时为 1，即 log_decay = 0
    log_decay = cumsum[:, None] - cumsum[None, :]  # (T, T)
    # 当 i < j 时为 0
    rel = np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]
    log_decay = np.where(rel >= 0, log_decay, -np.inf)
    return np.exp(log_decay).astype(dtype)


# ---------------------------------------------------------------------------
# RetNet
# ---------------------------------------------------------------------------


class RetNet(Module):
    """RetNet: Retention Network.

    Args:
        dim: 模型维度
        n_heads: 头数
        decay: 衰减系数 gamma（标量；不同 head 可有不同 gamma，这里简化为共享）
    """

    def __init__(self, dim: int, n_heads: int = 8, decay: float = 0.99):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.decay = float(decay)

        # QKV 投影：合并为单个 Linear 节省参数与计算
        self.qkv = Linear(dim, 3 * dim, bias=False)
        # group norm（每 head 独立归一化），用 RMSNorm 近似
        self.gnorm = RMSNorm(dim)
        # 输出投影
        self.out = Linear(dim, dim, bias=False)

        # 不同 head 用不同 gamma（基于几何级数，类似 RetNet 论文）
        # 论文里 gamma_h = 1 - 2^(-5-h)，这里简化：用同一个 gamma，但保留接口
        # 为提升表达能力，使用 per-head gamma buffer
        gammas = np.array(
            [1.0 - 2.0 ** (-5.0 - h) for h in range(n_heads)],
            dtype=np.float32,
        )
        # 把默认 decay 作为上界，gammas 缩放到 [decay, 1)
        gammas = self.decay + (1.0 - self.decay) * (gammas - gammas.min()) / max(1e-6, (gammas.max() - gammas.min()))
        self._gammas = gammas  # (n_heads,)

    # ------------------------------------------------------------------
    # 辅助：QKV 计算
    # ------------------------------------------------------------------

    def _compute_qkv(self, x: Tensor):
        """x: (B, T, D) -> Q, K, V: (B, T, H, d_head) (Tensor)"""
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        # reshape 为 (B, T, 3, H, d_head)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_head)
        # 拆分
        q = qkv[:, :, 0]  # (B, T, H, d_head)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        # 缩放 Q（retention 用 1/sqrt(d_head)）
        scale = 1.0 / np.sqrt(self.d_head)
        q = q * scale
        return q, k, v

    # ------------------------------------------------------------------
    # Parallel 模式（训练）
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列并行计算（训练用，可微）。

        保留公式：
            Y = (Q @ K^T * D) @ V
            对每个 head 独立计算，最后 merge heads + out_proj。

        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head
        q, k, v = self._compute_qkv(x)  # (B, T, H, d)

        # 把 head 维移到 batch，便于矩阵批量计算
        # (B, T, H, d) -> (B*H, T, d)
        qh = q.permute(0, 2, 1, 3).reshape(B * H, T, d)
        kh = k.permute(0, 2, 1, 3).reshape(B * H, T, d)
        vh = v.permute(0, 2, 1, 3).reshape(B * H, T, d)

        # scores: (B*H, T, T)
        scores = qh @ kh.transpose(-1, -2)

        # 衰减矩阵 D: (T, T)，对每个 head 不同 gamma
        # 这里为简洁，构造 (H, T, T) 然后广播到 (B, H, T, T)
        # 为保持可微，把 D 作为常量乘到 scores
        # 由于 gamma 是 buffer 不是 parameter，D 不需要梯度
        # 构造 (B*H, T, T) 的 mask
        D_list = []
        for h in range(H):
            D_list.append(_decay_matrix(T, float(self._gammas[h])))
        D_block = np.stack(D_list, axis=0)  # (H, T, T)
        # 复制到 batch 维：(B, H, T, T) -> (B*H, T, T)
        D_full = np.broadcast_to(D_block[None], (B, H, T, T)).reshape(B * H, T, T)
        D_tensor = Tensor(D_full.astype(np.float32), requires_grad=False)

        # 应用衰减
        scores = scores * D_tensor
        # 输出: (B*H, T, d)
        out = scores @ vh
        # 还原回 (B, T, H, d)
        out = out.reshape(B, H, T, d).permute(0, 2, 1, 3)  # (B, T, H, d)
        # merge heads
        out = out.reshape(B, T, D)
        # group norm + out proj
        out = self.gnorm(out)
        out = self.out(out)
        return out

    # ------------------------------------------------------------------
    # Recurrent 模式（推理）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x: Tensor, state=None):
        """单步递推（推理用）。

        Args:
            x: (B, 1, D)
            state: (B, H, d_head, d_head) 或 None
        Returns:
            out: (B, 1, D)
            new_state: (B, H, d_head, d_head)
        """
        B, T, D = x.shape
        assert T == 1, f"recurrent mode requires T=1, got T={T}"
        H, d = self.n_heads, self.d_head

        if state is None:
            state = np.zeros((B, H, d, d), dtype=np.float32)
        elif isinstance(state, Tensor):
            state = state.data

        # 用 no_grad 路径直接计算（不构建计算图）
        with no_grad():
            q, k, v = self._compute_qkv(x)  # Tensor (B, 1, H, d)
            qd = q.data  # (B, 1, H, d)
            kd = k.data  # (B, 1, H, d)
            vd = v.data

            # squeeze T=1
            qd = qd[:, 0]  # (B, H, d)
            kd = kd[:, 0]
            vd = vd[:, 0]

            new_state = state.copy()
            outs = np.zeros((B, H, d), dtype=np.float32)
            for h in range(H):
                gamma = float(self._gammas[h])
                # s_t = gamma * s_{t-1} + k_t^T outer v_t
                # k_t: (B, d) -> (B, d, 1)
                k_col = kd[:, h, :, None]  # (B, d, 1)
                v_row = vd[:, h, None, :]  # (B, 1, d)
                outer = k_col @ v_row  # (B, d, d)
                new_state[:, h] = gamma * state[:, h] + outer
                # out_t = q_t @ s_t  -> (B, d)
                q_row = qd[:, h, None, :]  # (B, 1, d)
                outs[:, h] = (q_row @ new_state[:, h])[:, 0]

            # 合并 head: (B, H, d) -> (B, D)
            out = outs.reshape(B, D)[:, None, :]  # (B, 1, D)
            out_tensor = Tensor(out, requires_grad=False)
            # group norm + out proj（这两层也是 Module，需要走前向）
            out_tensor = self.gnorm(out_tensor)
            out_tensor = self.out(out_tensor)

        return out_tensor, new_state

    # ------------------------------------------------------------------
    # Chunkwise 模式（长序列平衡）
    # ------------------------------------------------------------------

    def forward_chunkwise(self, x: Tensor, chunk_size: int = 64) -> Tensor:
        """分块并行：块内 parallel，块间 recurrent 传递状态。

        适用于长序列训练（比 full parallel 省内存）。
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        # 先计算 QKV（一次过）
        with no_grad():
            q, k, v = self._compute_qkv(x)
        # 注意：上面用 no_grad 会切断梯度。为保持可微，重新走可微路径
        q, k, v = self._compute_qkv(x)  # (B, T, H, d) Tensor

        n_chunks = (T + chunk_size - 1) // chunk_size
        # pad 到 chunk_size 整数倍
        pad_len = n_chunks * chunk_size - T
        if pad_len > 0:
            zeros = Tensor(np.zeros((B, pad_len, H, d), dtype=np.float32), requires_grad=False)
            q = Tensor(np.concatenate([q.data, zeros.data], axis=1), requires_grad=q.requires_grad)
            k = Tensor(np.concatenate([k.data, zeros.data], axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.concatenate([v.data, zeros.data], axis=1), requires_grad=v.requires_grad)

        T_padded = n_chunks * chunk_size
        # 状态 (B, H, d, d) - 用 numpy 维护（chunk 间的递推是顺序的）
        state = np.zeros((B, H, d, d), dtype=np.float32)

        outs = []
        for c in range(n_chunks):
            start = c * chunk_size
            end = start + chunk_size
            # 块内 Q, K, V
            qc = q[:, start:end]  # (B, C, H, d)
            kc = k[:, start:end]
            vc = v[:, start:end]

            # 块内 scores: (B, H, C, C)
            # 转置 (B, C, H, d) -> (B, H, C, d)
            qc_h = qc.permute(0, 2, 1, 3)
            kc_h = kc.permute(0, 2, 1, 3)
            vc_h = vc.permute(0, 2, 1, 3)
            scores_c = qc_h @ kc_h.transpose(-1, -2)  # (B, H, C, C)

            # 块内衰减
            D_intra = np.stack(
                [_decay_matrix(chunk_size, float(self._gammas[h])) for h in range(H)],
                axis=0,
            )  # (H, C, C)
            D_intra_full = np.broadcast_to(D_intra[None], (B, H, chunk_size, chunk_size))
            scores_c = scores_c * Tensor(D_intra_full.astype(np.float32), requires_grad=False)

            # 块内输出
            intra_out = scores_c @ vc_h  # (B, H, C, d)

            # 跨块贡献：state @ K^T -> scores for previous state
            # 对每个 head: prev_contrib[b, h, i] = q_{b,h,i} @ state[b, h]
            # 然后 prev_contrib @ V_prev_cumulative
            # 简化：直接用 state 累积的 K@V 外积和
            # 跨块贡献 = q_c @ (state expanded) -> (B, H, C, d)
            # state shape: (B, H, d, d)
            # 我们需要 (B, H, C, d) = q (B, H, C, d) @ state (B, H, d, d)
            # 用 numpy 算（无梯度，因为 state 是 buffer）
            qd = qc.data  # (B, C, H, d)
            # 转置到 (B, H, C, d)
            qd_h = np.transpose(qd, (0, 2, 1, 3))
            cross_out = np.einsum("bhcd,bhde->bhce", qd_h, state)  # (B, H, C, d)
            # 块内衰减应用到跨块贡献：每个 query 位置 i 的跨块贡献需要乘以
            # gamma^(i+1)（因为 query 在块内位置 i，距离上一块结尾为 i+1 步）
            for h in range(H):
                gamma = float(self._gammas[h])
                decay_vec = gamma ** (np.arange(chunk_size) + 1)  # (C,)
                cross_out[:, h] = cross_out[:, h] * decay_vec[None, :, None]
            cross_out_tensor = Tensor(cross_out.astype(np.float32), requires_grad=False)

            out_h = intra_out + cross_out_tensor  # (B, H, C, d)

            # 更新 state：state = gamma^C * state + K_c^T @ V_c
            # K_c^T @ V_c 是 (B, H, d, d)，块内还要按位置衰减累加
            # state_new = sum over t: gamma^(C-1-t) * k_t outer v_t + gamma^C * state
            kd = kc.data  # (B, C, H, d)
            vd = vc.data
            kd_h = np.transpose(kd, (0, 2, 1, 3))  # (B, H, C, d)
            vd_h = np.transpose(vd, (0, 2, 1, 3))
            new_state = np.zeros_like(state)
            for h in range(H):
                gamma = float(self._gammas[h])
                # 衰减向量：对位置 t 的外积 k_t v_t^T，贡献到 state 的衰减为 gamma^(C-1-t)
                decay_vec = gamma ** (chunk_size - 1 - np.arange(chunk_size))  # (C,)
                # sum_t decay_vec[t] * k_t outer v_t
                # k_t: (B, C, d), v_t: (B, C, d)
                # 用 einsum: 'btd,bte,t->bde'
                outer_sum = np.einsum(
                    "btd,bte,t->bde",
                    kd_h[:, h], vd_h[:, h], decay_vec,
                )  # (B, d, d)
                new_state[:, h] = (gamma ** chunk_size) * state[:, h] + outer_sum
            state = new_state

            # 还原 (B, H, C, d) -> (B, C, H, d) -> (B, C, D)
            out_h = out_h.permute(0, 2, 1, 3)  # (B, C, H, d)
            outs.append(out_h.reshape(B, chunk_size, D))

        # 拼接并裁剪到原 T
        out_full = outs[0]
        for o in outs[1:]:
            out_full = Tensor(
                np.concatenate([out_full.data, o.data], axis=1),
                requires_grad=out_full.requires_grad or o.requires_grad,
            )
        # 取前 T 个位置
        out_full = out_full[:, :T]

        # group norm + out proj
        out_full = self.gnorm(out_full)
        out_full = self.out(out_full)
        return out_full

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, state=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        elif mode == "recurrent":
            out, new_state = self.forward_recurrent(x, state)
            # 把 state 挂到输出上，便于调用方读取（非标准做法，但简洁）
            object.__setattr__(out, "_state", new_state)
            return out
        elif mode == "chunkwise":
            return self.forward_chunkwise(x)
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent/chunkwise")


__all__ = ["RetNet"]
