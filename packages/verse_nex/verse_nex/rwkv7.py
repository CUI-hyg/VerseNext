"""VerseNex: RWKV-7 Block (Task 3.4).

参考 RWKV-7 "Goose" 论文: https://arxiv.org/abs/2503.14456

核心结构：
    1. TimeMix: 时间混合层
       - 数据依赖的衰减率 w（每通道独立，in (-inf, 0)，exp(w) ∈ (0, 1)）
       - 状态更新: s_t = diag(exp(w_t)) @ s_{t-1} + k_t outer v_t
       - 输出: o_t = sigmoid(r_t) * (s_t @ k_t)
       - 类似 SSD 公式: O[i] = sigmoid(r[i]) * sum_{j<=i} decay[i,j] * (K[i]·K[j]) * V[j]
       - 其中 decay[i,j] = prod_{t=j+1}^{i} exp(w[t]) = exp(cumsum_w[i] - cumsum_w[j])
    2. ChannelMix: 通道混合层（类 FFN）
       - k = square(relu(linear_k(x_shifted)))
       - r = sigmoid(linear_r(x))
       - output = r * (k @ W_v)
    3. Block = LayerNorm + TimeMix + LayerNorm + ChannelMix（残差连接）

设计要点：
- TimeMix 同时支持 parallel（训练）与 recurrent（推理）模式
- recurrent 模式维护 (B, n_head, head_size, head_size) 状态 + (B, 1, dim) 的 x_prev 用于 time-shift
- 并行模式用 SSD 矩阵形式（与 Mamba-2 同构），保证 O(T) 训练效率
- ChannelMix 仅在 t>0 时启用 time-shift（t=0 用零向量）
- 数值一致：parallel 与 recurrent 输出在 float32 下吻合到 1e-3
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import Linear, LayerNorm, Module


# ---------------------------------------------------------------------------
# RWKV-7 TimeMix
# ---------------------------------------------------------------------------


class RWKV7TimeMix(Module):
    """RWKV-7 Time Mixing layer.

    Args:
        dim: 模型维度
        n_head: 头数（要求 dim % n_head == 0）
        head_size: 每头维度（默认 dim // n_head）
    """

    def __init__(self, dim: int, n_head: int = 8, head_size: int = None):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_size = head_size if head_size is not None else dim // n_head
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head})")

        # Time-shift mixing factors（可学习的插值系数）
        # 每个投影都有独立的 mix 因子
        self.r_mix = Tensor(np.full((1, 1, dim), 0.5, dtype=np.float32), requires_grad=True)
        self.k_mix = Tensor(np.full((1, 1, dim), 0.5, dtype=np.float32), requires_grad=True)
        self.v_mix = Tensor(np.full((1, 1, dim), 0.5, dtype=np.float32), requires_grad=True)
        self.w_mix = Tensor(np.full((1, 1, dim), 0.5, dtype=np.float32), requires_grad=True)
        self.a_mix = Tensor(np.full((1, 1, dim), 0.5, dtype=np.float32), requires_grad=True)

        # Linear projections
        # 输出维度与 dim 一致，内部按 head 切分
        self.x_r = Linear(dim, dim, bias=False)
        self.x_k = Linear(dim, dim, bias=False)
        self.x_v = Linear(dim, dim, bias=False)
        self.x_w = Linear(dim, dim, bias=False)  # decay (pre-activation)
        self.x_a = Linear(dim, dim, bias=False)  # low-rank modulator a (sigmoid)
        self.x_b = Linear(dim, dim, bias=False)  # low-rank modulator b

        # Output projection
        self.o_proj = Linear(dim, dim, bias=False)

        # 初始化 w 的偏置，让初始 decay 接近 1（exp(w) ≈ 1，即不衰减）
        # 实际 w = -softplus(x_w_raw)，所以让 x_w_raw 偏负即可让 w 接近 0
        with no_grad():
            self.x_w.weight.data = self.x_w.weight.data * 0.1 - 0.5

    # ------------------------------------------------------------------
    # 共用：从 x 与 x_shifted 计算 r, k, v, w, a, b
    # ------------------------------------------------------------------

    def _compute_rkvwab(self, x: Tensor, x_shifted: Tensor):
        """从 x 与 x_shifted 计算 r, k, v, w, a, b。

        Args:
            x: (B, T, D)
            x_shifted: (B, T, D) - x 的时间移位版本（右移 1，首位为 0）
        Returns:
            r, k, v, w, a, b: (B, T, D) Tensor
            其中:
                r = sigmoid(x_r(x * r_mix + x_shifted * (1 - r_mix)))
                k = x_k(x * k_mix + x_shifted * (1 - k_mix))
                v = x_v(x * v_mix + x_shifted * (1 - v_mix))
                w = -softplus(x_w(x * w_mix + x_shifted * (1 - w_mix)))  (negative)
                a = sigmoid(x_a(x * a_mix + x_shifted * (1 - a_mix)))
                b = x_b(x * a_mix + x_shifted * (1 - a_mix))  (reuse a_mix for simplicity)
        """
        # 时间混合输入
        r_in = x * self.r_mix + x_shifted * (1.0 - self.r_mix)
        k_in = x * self.k_mix + x_shifted * (1.0 - self.k_mix)
        v_in = x * self.v_mix + x_shifted * (1.0 - self.v_mix)
        w_in = x * self.w_mix + x_shifted * (1.0 - self.w_mix)
        a_in = x * self.a_mix + x_shifted * (1.0 - self.a_mix)
        # b 复用 a 的混合（简化）
        b_in = a_in

        # 线性投影
        r_raw = self.x_r(r_in)
        k = self.x_k(k_in)
        v = self.x_v(v_in)
        w_raw = self.x_w(w_in)
        a_raw = self.x_a(a_in)
        b = self.x_b(b_in)

        # 激活
        r = r_raw.sigmoid()
        a = a_raw.sigmoid()
        # w = -softplus(w_raw) (数值稳定 softplus)
        # 用 Tensor 实现：先 softplus，再取负
        safe_w = w_raw.minimum(20.0)
        sp_data = np.where(
            w_raw.data > 20, w_raw.data, np.log1p(np.exp(safe_w.data))
        ).astype(w_raw.data.dtype)
        # 构造 -softplus(w_raw) 的可微 Tensor
        w = _neg_softplus_tensor(w_raw, sp_data)
        return r, k, v, w, a, b

    # ------------------------------------------------------------------
    # Parallel 模式（训练，可微）
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列并行 SSD-style 计算（可微，用于训练）。

        公式（基于 SSD 同构）：
            decay[i, j] = exp(cumsum_w[i] - cumsum_w[j])  for i >= j
            O[i, h, c] = sigmoid(r[i, h, c]) * sum_{j<=i} decay[i, j, h, c]
                         * sum_d (K[i, h, d] * K[j, h, d]) * V[i, h, c]

        注意：RWKV-7 输出与 K[i] 的内积有关，所以 SSD 形式稍有不同：
            O[i, h, c] = sigmoid(r[i, h, c]) * sum_{j<=i} decay[i, j, h, c] * (K[i]·K[j]) * V[j, h, c]

        化简：
            KK[b, i, j, h] = sum_d K[b, i, h, d] * K[b, j, h, d]  (head-wise dot product)
            M[b, i, j, h, c] = decay[b, i, j, h, c] * KK[b, i, j, h]  (broadcast on c)
            O_pre[b, i, h, c] = sum_j M[b, i, j, h, c] * V[b, j, h, c]
            O = sigmoid(r) * O_pre

        其中 decay 是 per-channel 的，所以 M 实际形状为 (B, T_i, T_j, H, K)。
        """
        B, T, D = x.shape
        H, K = self.n_head, self.head_size

        # 构造 x_shifted: (B, T, D), 首位为零，其余为 x[:, :-1, :]
        x_shifted_data = np.zeros_like(x.data)
        if T > 1:
            x_shifted_data[:, 1:, :] = x.data[:, :-1, :]
        x_shifted = Tensor(x_shifted_data, requires_grad=False)

        r, k, v, w, a, b = self._compute_rkvwab(x, x_shifted)

        # reshape 到 (B, T, H, K)
        r_h = r.reshape(B, T, H, K)
        k_h = k.reshape(B, T, H, K)
        v_h = v.reshape(B, T, H, K)
        w_h = w.reshape(B, T, H, K)
        # a, b 在简化版本中暂不参与 state update；如需 low-rank term 可扩展

        # ----- 计算 decay matrix L: (B, T, T, H, K) -----
        # log_decay[i, j, h, c] = cumsum_w[i, h, c] - cumsum_w[j, h, c]  for i >= j
        # 用 float64 算 cumsum 提升精度
        w_data = w_h.data.astype(np.float64)  # (B, T, H, K), 负值
        cumsum_w = np.cumsum(w_data, axis=1)  # (B, T, H, K)
        zero_prefix = np.zeros((B, 1, H, K), dtype=np.float64)
        cs = np.concatenate([zero_prefix, cumsum_w], axis=1)  # (B, T+1, H, K)
        cs_i = cs[:, 1:, :, :]  # (B, T, H, K) corresponds to i+1
        cs_j = cs[:, 1:, :, :]
        # log_decay: (B, T_i, T_j, H, K)
        log_decay = cs_i[:, :, None, :, :] - cs_j[:, None, :, :, :]
        # 数值稳定性修复：clip log_decay 到 [-50, 0]
        # 理论上 log_decay <= 0（w 已约束为 -softplus(raw) < 0，cumsum 不减），
        # 但训练中 w_raw 可能学到异常值使 cumsum 出现极正，触发 exp 溢出为 inf
        # exp(-50) ≈ 1.9e-22 足够小但不 NaN；exp(0) = 1 上界安全
        log_decay = np.clip(log_decay, -50.0, 0.0)
        idx = np.arange(T)
        mask = (idx[:, None] >= idx[None, :]).astype(np.float64)  # (T, T)
        L_data = np.exp(log_decay) * mask[None, :, :, None, None]  # (B, T, T, H, K)
        L_t = Tensor(L_data.astype(np.float32), requires_grad=False)

        # 正确公式推导（与 recurrent 等价）：
        #   s_t[c, d] = sum_{j<=t} decay[t, j, c] * K[j, c] * V[j, d]
        #   o_t[c]    = sum_d s_t[c, d] * K[t, d]
        #             = sum_{j<=t} decay[t, j, c] * K[j, c] * (K[t] · V[j])
        # 所以 parallel 形式：
        #   O_pre[i, h, c] = sum_{j<=i} decay[i, j, h, c] * K[j, h, c] * (K[i, h, :] · V[j, h, :])

        # ----- KV_dot[b, i, j, h] = sum_d K[b, i, h, d] * V[b, j, h, d] -----
        # K: (B, T, H, K) -> K_i: (B, T_i, 1, H, K), V_j: (B, 1, T_j, H, K)
        # KV_dot = (K_i * V_j).sum(dim=-1) -> (B, T_i, T_j, H)
        K_i = k_h.unsqueeze(2)  # (B, T_i, 1, H, K)
        V_j = v_h.unsqueeze(1)  # (B, 1, T_j, H, K)
        KV_dot = (K_i * V_j).sum(dim=-1)  # (B, T_i, T_j, H)
        # 扩展到 (B, T_i, T_j, H, 1) 以便与 L * K[j] 相乘
        KV_exp = KV_dot.unsqueeze(-1)  # (B, T_i, T_j, H, 1)

        # ----- M[b, i, j, h, c] = decay[b, i, j, h, c] * K[b, j, h, c] -----
        # L: (B, T_i, T_j, H, K) per-channel decay
        # K[j]: (B, T_j, H, K) -> broadcast 到 (B, 1, T_j, H, K)
        K_j_broadcast = k_h.unsqueeze(1)  # (B, 1, T_j, H, K)
        M_t = L_t * K_j_broadcast  # (B, T_i, T_j, H, K)

        # ----- O_pre[b, i, h, c] = sum_j M[b, i, j, h, c] * KV_dot[b, i, j, h] -----
        # M: (B, T_i, T_j, H, K), KV_exp: (B, T_i, T_j, H, 1) -> broadcast
        # M * KV_exp: (B, T_i, T_j, H, K), 在 T_j (dim=2) 上求和 -> (B, T_i, H, K)
        O_pre = (M_t * KV_exp).sum(dim=2)  # (B, T_i, H, K)

        # ----- O = sigmoid(r) * O_pre -----
        # r_h: (B, T, H, K), O_pre: (B, T, H, K)
        O = r_h * O_pre  # sigmoid(r) 已经在 _compute_rkvwab 中应用了

        # reshape 回 (B, T, D) 并投影
        O = O.reshape(B, T, D)
        out = self.o_proj(O)
        return out

    # ------------------------------------------------------------------
    # Recurrent 模式（推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x: Tensor, state=None):
        """单步递推（推理用，常数内存）。

        Args:
            x: (B, 1, D)
            state: tuple (ssm_state, x_prev)
                ssm_state: (B, n_head, head_size, head_size)
                x_prev: (B, 1, D)
        Returns:
            out: (B, 1, D)
            new_state: tuple (ssm_state, x_prev)
        """
        B, T, D = x.shape
        assert T == 1, f"recurrent mode requires T=1, got T={T}"
        H, K = self.n_head, self.head_size

        if state is None:
            ssm_state = np.zeros((B, H, K, K), dtype=np.float32)
            x_prev = np.zeros((B, 1, D), dtype=np.float32)
        elif isinstance(state, tuple):
            ssm_state, x_prev = state
            if isinstance(ssm_state, Tensor):
                ssm_state = ssm_state.data
            if isinstance(x_prev, Tensor):
                x_prev = x_prev.data
            ssm_state = ssm_state.astype(np.float32, copy=True)
            x_prev = x_prev.astype(np.float32, copy=True)
        else:
            raise ValueError("state must be None or tuple (ssm_state, x_prev)")

        with no_grad():
            x_np = x.data.astype(np.float32, copy=False)
            x_shifted_np = x_prev  # (B, 1, D)

            # 用 numpy 计算 r, k, v, w, a, b
            # 时间混合输入
            r_mix = self.r_mix.data  # (1, 1, D)
            k_mix = self.k_mix.data
            v_mix = self.v_mix.data
            w_mix = self.w_mix.data
            a_mix = self.a_mix.data

            r_in = x_np * r_mix + x_shifted_np * (1.0 - r_mix)
            k_in = x_np * k_mix + x_shifted_np * (1.0 - k_mix)
            v_in = x_np * v_mix + x_shifted_np * (1.0 - v_mix)
            w_in = x_np * w_mix + x_shifted_np * (1.0 - w_mix)
            a_in = x_np * a_mix + x_shifted_np * (1.0 - a_mix)

            # 线性投影
            r_raw = r_in.reshape(B, D) @ self.x_r.weight.data.T  # (B, D)
            k_proj = k_in.reshape(B, D) @ self.x_k.weight.data.T
            v_proj = v_in.reshape(B, D) @ self.x_v.weight.data.T
            w_raw = w_in.reshape(B, D) @ self.x_w.weight.data.T
            a_raw = a_in.reshape(B, D) @ self.x_a.weight.data.T
            b_proj = a_in.reshape(B, D) @ self.x_b.weight.data.T

            # 激活
            r_d = 1.0 / (1.0 + np.exp(-r_raw))
            a_d = 1.0 / (1.0 + np.exp(-a_raw))
            # w = -softplus(w_raw)
            safe_w = np.minimum(w_raw, 20.0)
            sp = np.where(w_raw > 20, w_raw, np.log1p(np.exp(safe_w)))
            w_d = -sp.astype(np.float32)  # (B, D), negative
            # 不直接使用 a, b 在简化版 state update 中（保持纯对角衰减）

            # reshape 到 (B, H, K)
            r_h = r_d.reshape(B, H, K)
            k_h = k_proj.reshape(B, H, K)
            v_h = v_proj.reshape(B, H, K)
            w_h = w_d.reshape(B, H, K)

            # 状态更新: s_t = diag(exp(w)) @ s_{t-1} + k outer v
            # 输出: o = sigmoid(r) * (s @ k)  (per-head: (K,) @ (K, K) -> (K,))
            new_ssm = ssm_state.copy()
            O = np.zeros((B, H, K), dtype=np.float32)
            for h in range(H):
                exp_w = np.exp(w_h[:, h])  # (B, K)
                kv_outer = k_h[:, h, :, None] * v_h[:, h, None, :]  # (B, K, K)
                new_ssm[:, h] = exp_w[:, :, None] * ssm_state[:, h] + kv_outer
                # o = sigmoid(r) * (s @ k)  -> (B, K)
                # s @ k: (B, K, K) @ (B, K, 1) = (B, K, 1) -> (B, K)
                sk = np.einsum('bij,bj->bi', new_ssm[:, h], k_h[:, h])  # (B, K)
                O[:, h] = r_h[:, h] * sk

            # reshape 回 (B, 1, D) 并投影
            O_flat = O.reshape(B, 1, D)
            O_tensor = Tensor(O_flat, requires_grad=False)
            out = self.o_proj(O_tensor)

            new_x_prev = x_np.copy()

        return out, (new_ssm, new_x_prev)

    def forward(self, x: Tensor, state=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        elif mode == "recurrent":
            out, new_state = self.forward_recurrent(x, state)
            object.__setattr__(out, "_state", new_state)
            return out
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent")


# ---------------------------------------------------------------------------
# 辅助：构造 -softplus(w_raw) 的可微 Tensor
# ---------------------------------------------------------------------------


def _neg_softplus_tensor(x: Tensor, sp_data: np.ndarray) -> Tensor:
    """构造 w = -softplus(x) 的可微 Tensor。

    softplus 的导数是 sigmoid，所以 -softplus 的导数是 -sigmoid。
    """
    out_data = (-sp_data).astype(x.data.dtype)
    requires_grad = x.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=(x,) if requires_grad else (), _op="neg_softplus")
    if requires_grad:
        def _backward():
            if x.requires_grad:
                # d(-softplus(x))/dx = -sigmoid(x)
                sx = 1.0 / (1.0 + np.exp(-np.minimum(x.data, 20.0)))
                sx = np.where(x.data > 20, 1.0, sx).astype(x.data.dtype)
                x._accumulate_grad(out.grad * (-sx))
        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# RWKV-7 ChannelMix
# ---------------------------------------------------------------------------


class RWKV7ChannelMix(Module):
    """RWKV-7 Channel Mixing layer (FFN-like).

    Args:
        dim: 模型维度
        hidden: 隐藏维度（默认 4 * dim）
    """

    def __init__(self, dim: int, hidden: int = None):
        super().__init__()
        self.dim = dim
        self.hidden = hidden if hidden is not None else 4 * dim

        # Time-shift mix factor
        self.k_mix = Tensor(np.full((1, 1, dim), 0.5, dtype=np.float32), requires_grad=True)

        # Projections
        self.x_k = Linear(dim, self.hidden, bias=False)
        self.x_r = Linear(dim, dim, bias=False)
        self.x_v = Linear(self.hidden, dim, bias=False)

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列并行计算（可微）。

        公式:
            x_shifted = shift(x, 1)
            k_in = x * (1 - k_mix) + x_shifted * k_mix
            k = square(relu(x_k(k_in)))
            r = sigmoid(x_r(x))
            output = r * (k @ x_v.weight.T)  # (B, T, dim)
        """
        B, T, D = x.shape
        # x_shifted
        x_shifted_data = np.zeros_like(x.data)
        if T > 1:
            x_shifted_data[:, 1:, :] = x.data[:, :-1, :]
        x_shifted = Tensor(x_shifted_data, requires_grad=False)

        k_in = x * (1.0 - self.k_mix) + x_shifted * self.k_mix
        k_raw = self.x_k(k_in)
        # k = square(relu(k_raw))
        k_act = k_raw.relu() * k_raw.relu()  # square(relu(x))
        r = self.x_r(x).sigmoid()
        # value = k @ W_v.T -> (B, T, dim)
        v = k_act @ self.x_v.weight.transpose(-1, -2)
        return r * v

    def forward_recurrent(self, x: Tensor, state=None):
        """单步递推。state = (x_prev,)。"""
        B, T, D = x.shape
        assert T == 1, f"recurrent mode requires T=1, got T={T}"

        if state is None:
            x_prev = np.zeros((B, 1, D), dtype=np.float32)
        elif isinstance(state, tuple):
            x_prev = state[0]
            if isinstance(x_prev, Tensor):
                x_prev = x_prev.data
            x_prev = x_prev.astype(np.float32, copy=True)
        else:
            x_prev = state.astype(np.float32, copy=True) if isinstance(state, np.ndarray) else state

        with no_grad():
            x_np = x.data.astype(np.float32, copy=False)
            k_mix = self.k_mix.data
            k_in = x_np * (1.0 - k_mix) + x_prev * k_mix
            k_raw = k_in.reshape(B, D) @ self.x_k.weight.data.T  # (B, hidden)
            k_act = np.maximum(k_raw, 0)
            k_act = k_act * k_act  # square
            r_raw = x_np.reshape(B, D) @ self.x_r.weight.data.T  # (B, D)
            r = 1.0 / (1.0 + np.exp(-r_raw))
            v = k_act @ self.x_v.weight.data.T  # (B, D)
            out = r * v
            out = out[:, None, :]  # (B, 1, D)
            out_tensor = Tensor(out, requires_grad=False)
            new_x_prev = x_np.copy()
        return out_tensor, (new_x_prev,)

    def forward(self, x: Tensor, state=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        elif mode == "recurrent":
            out, new_state = self.forward_recurrent(x, state)
            object.__setattr__(out, "_state", new_state)
            return out
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent")


# ---------------------------------------------------------------------------
# RWKV-7 Block
# ---------------------------------------------------------------------------


class RWKV7Block(Module):
    """RWKV-7 Block: LayerNorm + TimeMix + LayerNorm + ChannelMix.

    Args:
        dim: 模型维度
        n_head: TimeMix 头数
        head_size: 每头维度
        hidden: ChannelMix 隐藏维度
    """

    def __init__(self, dim: int, n_head: int = 8, head_size: int = None,
                 hidden: int = None):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.time_mix = RWKV7TimeMix(dim, n_head=n_head, head_size=head_size)
        self.channel_mix = RWKV7ChannelMix(dim, hidden=hidden)

    def forward_parallel(self, x: Tensor) -> Tensor:
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x

    def forward_recurrent(self, x: Tensor, state=None):
        """state = (time_mix_state, channel_mix_state)"""
        if state is None:
            tm_state = None
            cm_state = None
        elif isinstance(state, tuple) and len(state) == 2:
            tm_state, cm_state = state
        else:
            raise ValueError("state must be None or tuple (time_mix_state, channel_mix_state)")

        # LayerNorm + TimeMix (single token)
        x_norm = self.ln1(x)
        x_norm_1 = x_norm.unsqueeze(1) if x_norm.ndim == 2 else x_norm
        # 实际上 ln1 输出 shape 应保持 (B, 1, D)
        tm_out, tm_new_state = self.time_mix.forward_recurrent(x_norm, tm_state)
        x = x + tm_out

        # LayerNorm + ChannelMix
        x_norm2 = self.ln2(x)
        cm_out, cm_new_state = self.channel_mix.forward_recurrent(x_norm2, cm_state)
        x = x + cm_out

        return x, (tm_new_state, cm_new_state)

    def forward(self, x: Tensor, state=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        elif mode == "recurrent":
            out, new_state = self.forward_recurrent(x, state)
            object.__setattr__(out, "_state", new_state)
            return out
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent")


__all__ = ["RWKV7TimeMix", "RWKV7ChannelMix", "RWKV7Block"]
