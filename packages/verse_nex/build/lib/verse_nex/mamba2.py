"""VerseNex: Mamba-2 SSM Block (Task 3.3).

参考 Mamba-2 论文: https://arxiv.org/abs/2405.21060

核心结构（每个 block）：
    1. in_proj: x -> x_branch, z_branch（合并为单个 Linear，输出 2*expand*dim）
    2. conv1d（causal, depthwise）作用于 x_branch
    3. silu(x_conv)
    4. x_proj: 计算 Δ, B, C（B, C 共享 cross-head，Δ per-head）
    5. dt = softplus(dt_proj(Δ_branch))（per-head）
    6. A_log: 参数 (n_heads,)，A = -exp(A_log)（负值，保证衰减）
    7. SSM 离散化 + selective scan:
       - A_bar = exp(dt * A)  (in (0, 1])
       - B_bar = dt * B
       - parallel (SSD 矩阵形式):  Y = (L ⊙ (C @ B_bar^T)) @ X
         其中 L[i,j] = prod_{t=j+1}^{i} A_bar[t] = exp(cumsum_logA_bar[i] - cumsum_logA_bar[j])
       - recurrent: h_t = A_bar[t] * h_{t-1} + B_bar[t] outer x_t; y_t = C[t] @ h_t
    8. gate: y = silu(z_branch) * y
    9. out_proj

设计要点：
- parallel 路径用 VerseTorch Tensor 实现，保持可微（用于训练）
- recurrent 路径用 NumPy 实现，常数内存（仅推理）
- 数值一致：parallel 与 recurrent 输出在 float32 下吻合到 1e-3
- 状态 shape:
    - ssm_state: (B, n_heads, d_state, d_head)
    - conv_state: (B, d_conv - 1, d_inner) - 缓存最近 d_conv-1 个 x_branch token

实现说明：
- conv1d 用「左 pad + 滑动乘加」实现，可微
- SSD 矩阵形式：直接用 Tensor broadcast + sum 表达 einsum，
  避免引入自定义 op；L 衰减矩阵作为 requires_grad=False 的常量 Tensor
- recurrent 路径在 no_grad 下用纯 numpy 算
- recurrent 路径维护独立的 conv_state，确保 conv1d 能正确访问历史 token
  （否则只看当前 token 会与 parallel 模式产生偏差）
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import Linear, Module


# ---------------------------------------------------------------------------
# 工具：causal depthwise conv1d（可微，用于训练）
# ---------------------------------------------------------------------------


def _conv1d_causal(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Causal depthwise 1D convolution（并行模式，可微）。

    Args:
        x: (B, T, D)
        weight: (d_conv, D) - 每个通道独立的卷积核
        bias: (D,) - 每个通道的偏置
    Returns:
        (B, T, D)

    实现：
        out[b, t, d] = bias[d] + sum_{k=0}^{d_conv-1} weight[k, d] * x[b, t-k, d]
        （t-k < 0 时取 0）

    用 Tensor 的 primitive ops（切片相加）表达，自动可微。
    """
    B, T, D = x.shape
    K = weight.shape[0]

    # bias 扩展到 (B, T, D)，作为累加起点
    # 注意：用默认参数捕获 out_init 引用，避免后续循环中 out 被重赋值导致闭包读到错的 tensor
    out_init = Tensor(np.tile(bias.data.reshape(1, 1, D), (B, T, 1)),
                      requires_grad=bias.requires_grad,
                      _children=(bias,) if bias.requires_grad else (),
                      _op="conv_bias_expand")
    if bias.requires_grad:
        def _backward_b(out_ref=out_init):
            if out_ref.grad is None:
                return
            bias._accumulate_grad(out_ref.grad.sum(axis=(0, 1)))
        out_init._backward = _backward_b

    out = out_init
    for k in range(K):
        # shifted_x[b, t, d] = x[b, t-k, d] if t >= k else 0
        shifted_data = np.zeros((B, T, D), dtype=x.data.dtype)
        if k < T:
            shifted_data[:, k:, :] = x.data[:, :T - k, :]
        shifted = Tensor(
            shifted_data,
            requires_grad=x.requires_grad,
            _children=(x,) if x.requires_grad and k > 0 else (),
            _op=f"conv_shift{k}",
        )
        if x.requires_grad and k > 0:
            # 用默认参数捕获 k_val 和 shifted_ref 的值（否则闭包会捕获循环变量，
            # 所有 _backward 都引用最后一次循环的 shifted，导致 shifted.grad 为 None）
            def _backward(k_val=k, shifted_ref=shifted):
                if shifted_ref.grad is None:
                    return
                grad_x = np.zeros_like(x.data)
                if k_val < T:
                    grad_x[:, :T - k_val, :] = shifted_ref.grad[:, k_val:, :]
                x._accumulate_grad(grad_x)
            shifted._backward = _backward

        # weight[k]: (D,) -> 用 Tensor 索引保持可微
        w_k = weight[k]
        w_k_expanded = w_k.reshape(1, 1, D)
        contrib = shifted * w_k_expanded  # (B, T, D)
        out = out + contrib

    return out


def _conv1d_step(x_branch: np.ndarray, conv_state: np.ndarray,
                 weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """单步 causal depthwise conv1d（推理用，常数内存）。

    Args:
        x_branch: (B, 1, d_inner) 当前 token 的 x_branch
        conv_state: (B, d_conv - 1, d_inner) 历史 d_conv-1 个 token
            存储顺序为「最新在前」：[x[t-1], x[t-2], ..., x[t-d_conv+1]]
        weight: (d_conv, d_inner)
        bias: (d_inner,)
    Returns:
        out: (B, 1, d_inner)
        new_conv_state: (B, d_conv - 1, d_inner)

    实现：
        window = [x_branch, conv_state] = [x[t], x[t-1], ..., x[t-d_conv+1]]
        out[b, 0, d] = bias[d] + sum_k weight[k, d] * window[b, k, d]
            = bias + weight[0]*x[t] + weight[1]*x[t-1] + ... + weight[K-1]*x[t-K+1]
        new_conv_state = window[:, :K-1] = [x[t], x[t-1], ..., x[t-K+2]]
            （丢掉最老的 x[t-K+1]，把 x[t] 加到最前面）
    """
    B = x_branch.shape[0]
    K = weight.shape[0]
    # window: (B, K, d_inner) = concat([x_branch, conv_state], axis=1)
    # x_branch: (B, 1, d_inner) - 最新 token 在前
    # conv_state: (B, K-1, d_inner) - 次新到最老
    window = np.concatenate([x_branch, conv_state], axis=1)  # (B, K, d_inner)
    # out = bias + sum_k weight[k] * window[:, k]
    # weight[0] * x[t] + weight[1] * x[t-1] + ... + weight[K-1] * x[t-K+1]
    out = bias[None, None, :] + (window * weight[None, :, :]).sum(axis=1, keepdims=True)
    # new_conv_state = window[:, :K-1]  (B, K-1, d_inner)
    # 保留前 K-1 个：[x[t], x[t-1], ..., x[t-K+2]]，丢掉最老的 x[t-K+1]
    new_conv_state = window[:, :-1, :].copy()
    return out.astype(x_branch.dtype), new_conv_state


# ---------------------------------------------------------------------------
# 工具：softplus（可微）
# ---------------------------------------------------------------------------


def _softplus(x: Tensor) -> Tensor:
    """softplus(x) = log(1 + exp(x))，数值稳定 + 可微。"""
    x_data = x.data
    safe_x = np.minimum(x_data, 20.0)
    out_data = np.where(x_data > 20, x_data, np.log1p(np.exp(safe_x))).astype(x_data.dtype)

    requires_grad = x.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=(x,) if requires_grad else (), _op="softplus")
    if requires_grad:
        def _backward():
            if x.requires_grad:
                # d softplus / dx = sigmoid(x)
                sx = 1.0 / (1.0 + np.exp(-np.minimum(x_data, 20.0)))
                sx = np.where(x_data > 20, 1.0, sx).astype(x_data.dtype)
                x._accumulate_grad(out.grad * sx)
        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Mamba-2 Block
# ---------------------------------------------------------------------------


class Mamba2Block(Module):
    """Mamba-2 selective SSM block.

    Args:
        dim: 模型维度
        d_state: SSM 状态维度（论文默认 128）
        d_conv: conv1d 核大小（默认 4）
        expand: 内部扩展倍数（默认 2）
        n_heads: SSD 头数（默认 16；要求 d_inner % n_heads == 0）
    """

    def __init__(self, dim: int, d_state: int = 128, d_conv: int = 4,
                 expand: int = 2, n_heads: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_heads = n_heads
        self.d_inner = expand * dim
        # Mamba-2 SSD: head_dim 基于 d_inner，而非 dim
        if self.d_inner % n_heads != 0:
            raise ValueError(
                f"d_inner ({self.d_inner} = expand*dim) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.d_head = self.d_inner // n_heads

        # 1. in_proj: x -> x_branch (d_inner), z_branch (d_inner)
        self.in_proj = Linear(dim, 2 * self.d_inner, bias=False)

        # 2. conv1d (depthwise causal)
        conv_w = Tensor.empty(d_conv, self.d_inner, requires_grad=True)
        with no_grad():
            conv_w.data = np.zeros((d_conv, self.d_inner), dtype=np.float32)
            conv_w.data[0] = 1.0
        self.conv1d_weight = conv_w
        conv_b = Tensor.zeros(self.d_inner, requires_grad=True)
        self.conv1d_bias = conv_b

        # 3. x_proj: 从 x_branch 计算 Δ_branch, B, C
        self.dt_rank = max(8, dim // 16)
        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # 4. dt_proj: Δ_branch -> dt (n_heads,)
        self.dt_proj = Linear(self.dt_rank, n_heads, bias=True)

        # 5. A_log: (n_heads,) 参数
        A_init = np.arange(1, n_heads + 1, dtype=np.float32)
        self.A_log = Tensor(np.log(A_init), requires_grad=True)

        # 6. D: skip connection (n_heads,) 参数
        self.D = Tensor.ones(n_heads, requires_grad=True)

        # 7. out_proj
        self.out_proj = Linear(self.d_inner, dim, bias=False)

    # ------------------------------------------------------------------
    # 共用：x_branch 处理（in_proj -> conv -> silu -> x_proj/dt）
    # ------------------------------------------------------------------

    def _prepare_parallel(self, x: Tensor):
        """从 x 计算 SSM 输入 X 与 gating z（并行模式）。

        Returns:
            X: (B, T, n_heads, d_head) Tensor
            z: (B, T, d_inner) Tensor (gating 分支)
            dt: (B, T, n_heads) Tensor (softplus 后)
            B_mat: (B, T, d_state) Tensor
            C_mat: (B, T, d_state) Tensor
            A: (n_heads,) ndarray（负值）
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head
        d_inner = self.d_inner

        xz = self.in_proj(x)
        xz = xz.reshape(B, T, 2, d_inner)
        x_branch = xz[:, :, 0]
        z_branch = xz[:, :, 1]

        x_conv = _conv1d_causal(x_branch, self.conv1d_weight, self.conv1d_bias)
        x_act = x_conv.silu()

        x_proj_out = self.x_proj(x_act)
        dt_branch = x_proj_out[:, :, :self.dt_rank]
        BC = x_proj_out[:, :, self.dt_rank:]
        B_mat = BC[:, :, :self.d_state]
        C_mat = BC[:, :, self.d_state:]

        dt_raw = self.dt_proj(dt_branch)
        dt = _softplus(dt_raw)

        A = -np.exp(self.A_log.data)

        X = x_act.reshape(B, T, H, d)

        return X, z_branch, dt, B_mat, C_mat, A

    def _prepare_recurrent(self, x: np.ndarray, conv_state: np.ndarray):
        """从 x 计算 SSM 输入与 gating（递归模式，纯 numpy）。

        Args:
            x: (B, 1, D) ndarray
            conv_state: (B, d_conv - 1, d_inner) ndarray
        Returns:
            Xd: (B, H, d) ndarray
            zd: (B, d_inner) ndarray
            dt_d: (B, H) ndarray
            Bd: (B, N) ndarray
            Cd: (B, N) ndarray
            A: (H,) ndarray
            new_conv_state: (B, d_conv - 1, d_inner) ndarray
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head
        N = self.d_state
        d_inner = self.d_inner

        # in_proj (作为 numpy matmul)
        # x: (B, 1, D) -> (B, D) -> (B, 2*d_inner)
        x_flat = x.reshape(B, D)
        xz = x_flat @ self.in_proj.weight.data.T  # (B, 2*d_inner)
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.data
        xz = xz.reshape(B, 1, 2, d_inner)
        x_branch = xz[:, 0, 0]   # (B, d_inner)
        z_branch = xz[:, 0, 1]   # (B, d_inner)

        # conv1d step
        x_branch_3d = x_branch[:, None, :]  # (B, 1, d_inner)
        x_conv, new_conv_state = _conv1d_step(
            x_branch_3d, conv_state,
            self.conv1d_weight.data, self.conv1d_bias.data,
        )  # (B, 1, d_inner)
        x_conv = x_conv[:, 0, :]  # (B, d_inner)

        # silu
        sig = 1.0 / (1.0 + np.exp(-np.minimum(x_conv, 20.0)))
        x_act = x_conv * sig  # (B, d_inner)

        # x_proj
        x_proj_out = x_act @ self.x_proj.weight.data.T  # (B, dt_rank + 2N)
        if self.x_proj.bias is not None:
            x_proj_out = x_proj_out + self.x_proj.bias.data
        dt_branch = x_proj_out[:, :self.dt_rank]
        BC = x_proj_out[:, self.dt_rank:]
        Bd = BC[:, :N]
        Cd = BC[:, N:]

        # dt_proj + softplus
        dt_raw = dt_branch @ self.dt_proj.weight.data.T  # (B, H)
        if self.dt_proj.bias is not None:
            dt_raw = dt_raw + self.dt_proj.bias.data
        safe = np.minimum(dt_raw, 20.0)
        dt_d = np.where(dt_raw > 20, dt_raw, np.log1p(np.exp(safe))).astype(x_conv.dtype)

        A = -np.exp(self.A_log.data)

        # X: (B, H, d)
        Xd = x_act.reshape(B, H, d)

        return Xd, z_branch, dt_d, Bd, Cd, A, new_conv_state

    # ------------------------------------------------------------------
    # Parallel SSD（训练，可微）
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列并行 SSD 计算（可微，用于训练）。

        SSD 公式:
            Y[h, i, c] = sum_{j<=i} decay[i, j] * (C[i] . B_bar[j]) * X[h, j, c]
                        + D[h] * X[h, i, c]
            decay[i, j] = prod_{t=j+1}^{i} A_bar[t]
                        = exp(cumsum_logA_bar[i] - cumsum_logA_bar[j])   for i >= j
            A_bar[t] = exp(dt[t] * A)       # (B, T, H) ∈ (0, 1]
            B_bar[t] = dt[t] * B[t]         # (B, T, H, N) (dt per-head, B shared)
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        X, z_branch, dt, B_mat, C_mat, A = self._prepare_parallel(x)

        # ----- 计算 log_A_bar = dt * A -----
        # 用 float64 算 cumsum 提升数值精度
        A_b = A.reshape(1, 1, H).astype(np.float64)
        log_A_bar_data = dt.data.astype(np.float64) * A_b  # (B, T, H)

        cumsum_log = np.cumsum(log_A_bar_data, axis=1)  # (B, T, H)
        zero_prefix = np.zeros((B, 1, H), dtype=np.float64)
        cs = np.concatenate([zero_prefix, cumsum_log], axis=1)  # (B, T+1, H)
        # decay[i, j] = exp(cs[i+1] - cs[j+1]) for i >= j else 0
        # cs_i[k] = cs[k+1] (k from 0 to T-1) -> shape (B, T, H)
        cs_i = cs[:, 1:, :]
        cs_j = cs[:, 1:, :]
        log_decay = cs_i[:, :, None, :] - cs_j[:, None, :, :]  # (B, T_i, T_j, H)
        idx = np.arange(T)
        mask = (idx[:, None] >= idx[None, :]).astype(np.float64)
        L_data = np.exp(log_decay) * mask[None, :, :, None]  # (B, T, T, H) float64
        # L 是 requires_grad=False 的常量；保持 float64 以提升 SSD 求和的数值精度
        # (NumPy 在 float64 * float32 时会自动提升到 float64，梯度仍按参数 dtype 累积)
        L_t = Tensor(L_data, requires_grad=False)

        # ----- B_bar[b, t, h, n] = dt[b, t, h] * B[b, t, n] -----
        dt_t = dt.unsqueeze(-1)              # (B, T, H, 1)
        B_t = B_mat.unsqueeze(2)             # (B, T, 1, N)
        B_bar_t = dt_t * B_t                 # (B, T, H, N)

        # CB[b, i, j, h] = sum_n C[b, i, n] * B_bar[b, j, h, n]
        # C_mat: (B, T, N) -> (B, T_i, 1, 1, N) via unsqueeze(2).unsqueeze(3)
        C_t = C_mat.unsqueeze(2).unsqueeze(3)  # (B, T_i, 1, 1, N)
        BB_t = B_bar_t.unsqueeze(1)            # (B, 1, T_j, H, N)
        CB_t = (C_t * BB_t).sum(-1)            # (B, T_i, T_j, H)

        # M = L ⊙ CB （L 为 float64 常量，CB 为 float32；NumPy 提升到 float64）
        M_t = L_t * CB_t  # (B, T_i, T_j, H) float64

        # Y[b, i, h, c] = sum_j M[b, i, j, h] * X[b, j, h, c]
        M_exp = M_t.unsqueeze(-1)  # (B, T_i, T_j, H, 1)
        X_exp = X.unsqueeze(1)     # (B, 1, T_j, H, d)
        Y_t = (M_exp * X_exp).sum(dim=2)  # (B, T_i, H, d) float64

        # skip: D * X
        D_t = self.D.reshape(1, 1, H, 1)
        Y_t = Y_t + D_t * X

        # reshape -> (B, T, d_inner)；转回 float32 与 recurrent 路径对齐
        Y_t = Y_t.reshape(B, T, H * d).cast(np.float32)

        # gating: y = silu(z) * Y
        z_act = z_branch.silu()
        Y_final = z_act * Y_t

        # out_proj
        out = self.out_proj(Y_final)
        return out

    # ------------------------------------------------------------------
    # Recurrent（推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x: Tensor, state=None):
        """单步递推（推理用，常数内存）。

        Args:
            x: (B, 1, D)
            state: 可选，三种形式：
                - None: 初始化为零状态
                - tuple (ssm_state, conv_state):
                    ssm_state: (B, n_heads, d_state, d_head)
                    conv_state: (B, d_conv - 1, d_inner)
                - ndarray: 视为 ssm_state，conv_state 初始化为零
        Returns:
            out: (B, 1, D)
            new_state: tuple (ssm_state, conv_state)
        """
        B, T, D = x.shape
        assert T == 1, f"recurrent mode requires T=1, got T={T}"
        H, d = self.n_heads, self.d_head
        N = self.d_state

        # 解析 state
        if state is None:
            ssm_state = np.zeros((B, H, N, d), dtype=np.float32)
            conv_state = np.zeros((B, self.d_conv - 1, self.d_inner), dtype=np.float32)
        elif isinstance(state, tuple):
            ssm_state, conv_state = state
            if isinstance(ssm_state, Tensor):
                ssm_state = ssm_state.data
            if isinstance(conv_state, Tensor):
                conv_state = conv_state.data
            ssm_state = ssm_state.astype(np.float32, copy=True)
            conv_state = conv_state.astype(np.float32, copy=True)
        else:
            # 兼容旧 API：单 ndarray 视为 ssm_state
            if isinstance(state, Tensor):
                state = state.data
            ssm_state = state.astype(np.float32, copy=True)
            conv_state = np.zeros((B, self.d_conv - 1, self.d_inner), dtype=np.float32)

        with no_grad():
            x_np = x.data.astype(np.float32, copy=False)
            Xd, zd, dt_d, Bd, Cd, A, new_conv_state = self._prepare_recurrent(
                x_np, conv_state,
            )

            # 用 float64 累积 SSM 状态以匹配 parallel 路径的数值精度
            # (parallel 路径在 float64 下用 cumsum 计算 decay 矩阵)
            ssm_state_f64 = ssm_state.astype(np.float64)
            new_ssm_f64 = ssm_state_f64.copy()
            Y = np.zeros((B, H, d), dtype=np.float64)
            for h in range(H):
                A_h = float(A[h])
                dt_h = dt_d[:, h].astype(np.float64)  # (B,)
                A_bar = np.exp(dt_h * A_h)             # (B,) float64
                B_bar = dt_h[:, None] * Bd.astype(np.float64)  # (B, N) float64
                x_h = Xd[:, h, :].astype(np.float64)   # (B, d) float64
                outer = B_bar[:, :, None] * x_h[:, None, :]  # (B, N, d) float64
                new_ssm_f64[:, h] = A_bar[:, None, None] * ssm_state_f64[:, h] + outer
                y_h = np.einsum('bn,bnd->bd', Cd.astype(np.float64), new_ssm_f64[:, h])
                y_h = y_h + float(self.D.data[h]) * x_h
                Y[:, h, :] = y_h

            # 将 float64 状态转回 float32 存储（节省内存，与 parallel 输出 dtype 对齐）
            new_ssm = new_ssm_f64.astype(np.float32)

            # gating: y = silu(z) * Y（用与 parallel 一致的 numerically stable sigmoid）
            zd_f64 = zd.astype(np.float64)
            sig = np.where(zd_f64 >= 0,
                           1.0 / (1.0 + np.exp(-zd_f64)),
                           np.exp(zd_f64) / (1.0 + np.exp(zd_f64)))
            silu_z = zd_f64 * sig
            Y_flat = Y.reshape(B, H * d)
            Y_final = (silu_z * Y_flat).astype(np.float32)

            # out_proj
            Y_tensor = Tensor(Y_final, requires_grad=False)
            out = self.out_proj(Y_tensor)
            out = out.unsqueeze(1)

        return out, (new_ssm, new_conv_state)

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


__all__ = ["Mamba2Block"]
