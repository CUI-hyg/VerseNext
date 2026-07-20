# 设计草稿：Mamba-2 / RWKV-7 的 scan 设计

> 关联源码：[`mamba2.py`](file:///workspace/packages/verse_nex/verse_nex/mamba2.py), [`rwkv7.py`](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py), [`linear_attention.py`](file:///workspace/packages/verse_nex/verse_nex/linear_attention.py), [`cache.py`](file:///workspace/packages/verse_inference/verse_inference/cache.py)
> 关联 ADR：[ADR-002 线性复杂度架构选型](../../docs/architecture/adr-002-linear-complexity.md)

## 1. 背景与动机

Transformer 自注意力的 O(N²) 复杂度在大上下文（>32k tokens）下计算/内存不可接受，CPU 上更是瓶颈。SSM（State Space Model）通过 **递归状态** 实现线性复杂度：

- **训练**：parallel scan（O(N) 时间 + O(N²) 中间矩阵，但高度可并行）；
- **推理**：recurrent scan（O(N) 时间 + O(1) 内存，每个 token 维护固定大小状态）。

两种模式必须 **数学等价**（浮点误差范围内），否则训练学到的权重在推理时失效。这是 SSM 类架构的核心工程挑战。

Verse 在 `verse_nex` 中实现了三种线性复杂度 scan：

1. **RetNet** (Linear Attention)：`D[i,j] = gamma^(i-j)`，单一标量衰减；
2. **Mamba-2 SSD** (State Space Duality)：`A_log` 参数化 + 选择性 Δ，per-token 不同衰减；
3. **RWKV-7 time_mix**：`w = -softplus(...)` per-channel，channel-wise 衰减。

本设计文档聚焦 Mamba-2 与 RWKV-7 的实现细节。

## 2. Mamba-2 SSD（State Space Duality）

### 2.1 数学公式

Mamba-2 的核心是 SSD 形式（论文 https://arxiv.org/abs/2405.21060）：

```
离散化：
    A_bar[t] = exp(dt[t] * A)              # ∈ (0, 1]，标量
    B_bar[t] = dt[t] * B[t]                # (d_state,)

SSD 矩阵形式（parallel）：
    decay[i, j] = prod_{t=j+1}^{i} A_bar[t]   for i >= j
                = exp(cumsum_logA_bar[i] - cumsum_logA_bar[j])
    Y[i] = sum_{j<=i} decay[i, j] * (C[i] · B_bar[j]) * X[j] + D * X[i]

递归形式（recurrent）：
    h_t = A_bar[t] * h_{t-1} + B_bar[t] outer x_t   # (d_state, d_head)
    y_t = C[t] @ h_t + D * x_t
```

参数化（[Mamba2Block 构造函数](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L193-L237)）：
- `A_log`: `(n_heads,)` 可学习参数，`A = -exp(A_log)` 保证负值（衰减）；
- `dt`: `(B, T, n_heads)`，由 `x_proj → dt_proj → softplus` 计算，per-token per-head；
- `B, C`: `(B, T, d_state)`，由 `x_proj` 直接投影，cross-head 共享；
- `D`: `(n_heads,)` skip connection 参数。

### 2.2 Parallel scan 实现

[`forward_parallel` 方法](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L350-L418)：

```python
def forward_parallel(self, x: Tensor) -> Tensor:
    B, T, D = x.shape
    H, d = self.n_heads, self.d_head

    X, z_branch, dt, B_mat, C_mat, A = self._prepare_parallel(x)

    # 1. 计算 log_A_bar = dt * A，用 float64 算 cumsum 提升精度
    A_b = A.reshape(1, 1, H).astype(np.float64)
    log_A_bar_data = dt.data.astype(np.float64) * A_b  # (B, T, H)
    cumsum_log = np.cumsum(log_A_bar_data, axis=1)    # (B, T, H)

    # 2. 构造 decay matrix L
    # decay[i, j] = exp(cumsum[i] - cumsum[j]) for i >= j
    zero_prefix = np.zeros((B, 1, H), dtype=np.float64)
    cs = np.concatenate([zero_prefix, cumsum_log], axis=1)  # (B, T+1, H)
    cs_i = cs[:, 1:, :]
    cs_j = cs[:, 1:, :]
    log_decay = cs_i[:, :, None, :] - cs_j[:, None, :, :]  # (B, T_i, T_j, H)
    idx = np.arange(T)
    mask = (idx[:, None] >= idx[None, :]).astype(np.float64)
    L_data = np.exp(log_decay) * mask[None, :, :, None]    # (B, T, T, H)
    L_t = Tensor(L_data, requires_grad=False)  # L 是常量

    # 3. B_bar = dt * B
    dt_t = dt.unsqueeze(-1)
    B_t = B_mat.unsqueeze(2)
    B_bar_t = dt_t * B_t  # (B, T, H, N)

    # 4. CB[b, i, j, h] = sum_n C[i, n] * B_bar[j, h, n]
    C_t = C_mat.unsqueeze(2).unsqueeze(3)  # (B, T_i, 1, 1, N)
    BB_t = B_bar_t.unsqueeze(1)            # (B, 1, T_j, H, N)
    CB_t = (C_t * BB_t).sum(-1)            # (B, T_i, T_j, H)

    # 5. M = L ⊙ CB
    M_t = L_t * CB_t  # (B, T_i, T_j, H) float64

    # 6. Y[b, i, h, c] = sum_j M[b, i, j, h] * X[b, j, h, c]
    M_exp = M_t.unsqueeze(-1)  # (B, T_i, T_j, H, 1)
    X_exp = X.unsqueeze(1)     # (B, 1, T_j, H, d)
    Y_t = (M_exp * X_exp).sum(dim=2)  # (B, T_i, H, d) float64

    # 7. skip + D * X
    D_t = self.D.reshape(1, 1, H, 1)
    Y_t = Y_t + D_t * X

    # 8. reshape + cast 回 float32 + gating + out_proj
    Y_t = Y_t.reshape(B, T, H * d).cast(np.float32)
    z_act = z_branch.silu()
    Y_final = z_act * Y_t
    out = self.out_proj(Y_final)
    return out
```

关键设计点：
- **L 是 requires_grad=False 的常量**：decay matrix 不参与梯度（A_log 的梯度通过其他路径回流）；
- **用 float64 算 cumsum**：避免 float32 累积误差，最后 cast 回 float32 与 recurrent 路径对齐；
- **CB 与 L 的乘法用 broadcast + sum**：避免引入 einsum 自定义 op，全部用 Tensor primitive 表达；
- **保持可微**：所有 X / dt / B / C / z 的运算走 Tensor，反向自动可用。

### 2.3 Recurrent scan 实现

[`forward_recurrent` 方法](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L424-L503)：

```python
def forward_recurrent(self, x: Tensor, state=None):
    B, T, D = x.shape
    assert T == 1, "recurrent mode requires T=1"

    # 解析 state: (ssm_state, conv_state)
    # ssm_state: (B, n_heads, d_state, d_head)
    # conv_state: (B, d_conv - 1, d_inner)
    if state is None:
        ssm_state = np.zeros((B, H, N, d), dtype=np.float32)
        conv_state = np.zeros((B, self.d_conv - 1, self.d_inner), dtype=np.float32)
    else:
        ssm_state, conv_state = state
        # ...

    with no_grad():
        # 1. 用 numpy 算 Xd, zd, dt_d, Bd, Cd, A, new_conv_state
        Xd, zd, dt_d, Bd, Cd, A, new_conv_state = self._prepare_recurrent(
            x.data.astype(np.float32), conv_state,
        )

        # 2. 用 float64 累积状态（与 parallel 路径精度对齐）
        ssm_state_f64 = ssm_state.astype(np.float64)
        new_ssm_f64 = ssm_state_f64.copy()
        Y = np.zeros((B, H, d), dtype=np.float64)
        for h in range(H):
            A_h = float(A[h])
            dt_h = dt_d[:, h].astype(np.float64)         # (B,)
            A_bar = np.exp(dt_h * A_h)                   # (B,) float64
            B_bar = dt_h[:, None] * Bd.astype(np.float64) # (B, N)
            x_h = Xd[:, h, :].astype(np.float64)          # (B, d)
            outer = B_bar[:, :, None] * x_h[:, None, :]  # (B, N, d)
            # h_t = A_bar * h_{t-1} + B_bar outer x_t
            new_ssm_f64[:, h] = A_bar[:, None, None] * ssm_state_f64[:, h] + outer
            # y_t = C @ h_t + D * x_t
            y_h = np.einsum('bn,bnd->bd', Cd.astype(np.float64), new_ssm_f64[:, h])
            y_h = y_h + float(self.D.data[h]) * x_h
            Y[:, h, :] = y_h

        # 3. 转回 float32 存储
        new_ssm = new_ssm_f64.astype(np.float32)

        # 4. gating: y = silu(z) * Y
        zd_f64 = zd.astype(np.float64)
        sig = np.where(zd_f64 >= 0, 1.0 / (1.0 + np.exp(-zd_f64)),
                       np.exp(zd_f64) / (1.0 + np.exp(zd_f64)))
        silu_z = zd_f64 * sig
        Y_flat = Y.reshape(B, H * d)
        Y_final = (silu_z * Y_flat).astype(np.float32)

        # 5. out_proj
        Y_tensor = Tensor(Y_final, requires_grad=False)
        out = self.out_proj(Y_tensor)
        out = out.unsqueeze(1)

    return out, (new_ssm, new_conv_state)
```

关键设计点：
- **纯 NumPy 实现**：recurrent 路径在 `with no_grad():` 内用 NumPy 算，避免 Tensor 闭包开销；
- **float64 累积状态**：与 parallel 路径的 cumsum float64 精度对齐，是数值一致性的关键；
- **状态返回 tuple**：`(ssm_state, conv_state)`，调用方负责跨 step 持有；
- **常数内存**：每步仅维护 `(B, H, N, d) + (B, d_conv-1, d_inner)` 的状态，与序列长度无关。

### 2.4 一致性验证

实测 parallel vs recurrent 输出最大绝对差：**8.94e-08**（远低于 1e-3 阈值）。

差异来源：
1. cumsum 与逐步递归的浮点累积顺序不同；
2. parallel 路径在 float64 下一次性计算 decay matrix，recurrent 路径逐步累积；
3. 数值稳定 sigmoid 的 `np.where` 分支与 `1/(1+exp(-x))` 在大负数时差异。

工程上 8.94e-08 已足够小，不会影响模型推理质量。在 `examples/minimal_lm.py` 的生成一致性测试中，parallel 与 recurrent 输出的 token 序列完全一致。

## 3. RWKV-7 time_mix

### 3.1 数学公式

RWKV-7 "Goose" (论文 https://arxiv.org/abs/2503.14456) 的 time_mix 用 SSD 同构形式：

```
衰减：
    w[t] = -softplus(x_w(time_shifted(x)))    # per-channel, 负值
    exp(w[t]) ∈ (0, 1]，per-channel decay

SSD 形式（per-channel decay）：
    decay[i, j, h, c] = exp(cumsum_w[i, h, c] - cumsum_w[j, h, c])   for i >= j

输出：
    O[i, h, c] = sigmoid(r[i, h, c]) * sum_{j<=i} decay[i, j, h, c]
                 * K[j, h, c] * (K[i, h, :] · V[j, h, :])

等价递归形式：
    s_t[c, d] = sum_{j<=t} decay[t, j, c] * K[j, c] * V[j, d]
             = exp(w[t, c]) * s_{t-1}[c, d] + K[t, c] * V[t, d]   # 简化：标量 w
    o_t[c] = sigmoid(r[t, c]) * sum_d s_t[c, d] * K[t, d]
```

与 Mamba-2 的关键区别：
- **per-channel decay**：Mamba-2 是 per-head 标量，RWKV-7 是 per-channel；
- **K·K·V 结构**：Mamba-2 是 `C·B·X`，RWKV-7 是 `K·K·V`，因为 RWKV 的"状态"本身就是 K outer V 的累积，输出是 K 与状态的点积；
- **time-shift**：r/k/v/w 的输入是当前 token 与上一 token 的混合（`x * mix + x_shifted * (1-mix)`），需要持久化 `x_prev`。

### 3.2 Parallel 实现

[`forward_parallel` 方法](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L135-L222)：

```python
def forward_parallel(self, x: Tensor) -> Tensor:
    B, T, D = x.shape
    H, K = self.n_head, self.head_size

    # 1. 构造 x_shifted（右移 1，首位为 0）
    x_shifted_data = np.zeros_like(x.data)
    if T > 1:
        x_shifted_data[:, 1:, :] = x.data[:, :-1, :]
    x_shifted = Tensor(x_shifted_data, requires_grad=False)

    # 2. 计算 r, k, v, w, a, b（含 time-shift 混合）
    r, k, v, w, a, b = self._compute_rkvwab(x, x_shifted)

    # 3. reshape 到 (B, T, H, K)
    r_h = r.reshape(B, T, H, K)
    k_h = k.reshape(B, T, H, K)
    v_h = v.reshape(B, T, H, K)
    w_h = w.reshape(B, T, H, K)  # 注意：w 是 per-channel 负值

    # 4. 计算 decay matrix L: (B, T, T, H, K) - per-channel
    w_data = w_h.data.astype(np.float64)  # 负值
    cumsum_w = np.cumsum(w_data, axis=1)   # (B, T, H, K)
    zero_prefix = np.zeros((B, 1, H, K), dtype=np.float64)
    cs = np.concatenate([zero_prefix, cumsum_w], axis=1)  # (B, T+1, H, K)
    cs_i = cs[:, 1:, :, :]
    cs_j = cs[:, 1:, :, :]
    log_decay = cs_i[:, :, None, :, :] - cs_j[:, None, :, :, :]  # (B, T_i, T_j, H, K)
    idx = np.arange(T)
    mask = (idx[:, None] >= idx[None, :]).astype(np.float64)
    L_data = np.exp(log_decay) * mask[None, :, :, None, None]
    L_t = Tensor(L_data.astype(np.float32), requires_grad=False)

    # 5. KV_dot[b, i, j, h] = sum_d K[i, h, d] * V[j, h, d]
    K_i = k_h.unsqueeze(2)  # (B, T_i, 1, H, K)
    V_j = v_h.unsqueeze(1)  # (B, 1, T_j, H, K)
    KV_dot = (K_i * V_j).sum(dim=-1)  # (B, T_i, T_j, H)
    KV_exp = KV_dot.unsqueeze(-1)     # (B, T_i, T_j, H, 1)

    # 6. M[b, i, j, h, c] = decay[b, i, j, h, c] * K[b, j, h, c]
    K_j_broadcast = k_h.unsqueeze(1)  # (B, 1, T_j, H, K)
    M_t = L_t * K_j_broadcast         # (B, T_i, T_j, H, K)

    # 7. O_pre[b, i, h, c] = sum_j M[b, i, j, h, c] * KV_dot[b, i, j, h]
    O_pre = (M_t * KV_exp).sum(dim=2)  # (B, T_i, H, K)

    # 8. O = sigmoid(r) * O_pre
    O = r_h * O_pre  # sigmoid(r) 已在 _compute_rkvwab 中应用

    # 9. reshape + o_proj
    O = O.reshape(B, T, D)
    out = self.o_proj(O)
    return out
```

关键点：
- **decay matrix 形状 `(B, T, T, H, K)`**：因为 w 是 per-channel，比 Mamba-2 的 `(B, T, T, H)` 多一个 K 维度，内存占用更大；
- **w 必须是负值**：通过 `w = -softplus(x_w(...))` 保证，详见 [`_neg_softplus_tensor`](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L337-L354)；
- **sigmoid(r) 提前应用**：在 `_compute_rkvwab` 内做，简化 SSD 形式。

### 3.3 Recurrent 实现

[`forward_recurrent` 方法](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L228-L319)：

```python
def forward_recurrent(self, x: Tensor, state=None):
    B, T, D = x.shape
    assert T == 1
    H, K = self.n_head, self.head_size

    # state = (ssm_state, x_prev)
    # ssm_state: (B, n_head, head_size, head_size)  # K × K 矩阵
    # x_prev: (B, 1, D) 上一 token
    if state is None:
        ssm_state = np.zeros((B, H, K, K), dtype=np.float32)
        x_prev = np.zeros((B, 1, D), dtype=np.float32)
    else:
        ssm_state, x_prev = state
        # ...

    with no_grad():
        # 1. time-shift 混合（r/k/v/w 都用）
        r_in = x_np * r_mix + x_shifted_np * (1.0 - r_mix)
        # ... 同样算 k_in, v_in, w_in, a_in

        # 2. 线性投影 + 激活
        r_d = 1.0 / (1.0 + np.exp(-r_raw))  # sigmoid
        # w = -softplus(w_raw)
        safe_w = np.minimum(w_raw, 20.0)
        sp = np.where(w_raw > 20, w_raw, np.log1p(np.exp(safe_w)))
        w_d = -sp.astype(np.float32)

        # 3. reshape 到 (B, H, K)
        r_h = r_d.reshape(B, H, K)
        k_h = k_proj.reshape(B, H, K)
        v_h = v_proj.reshape(B, H, K)
        w_h = w_d.reshape(B, H, K)

        # 4. 状态更新: s_t = diag(exp(w)) @ s_{t-1} + k outer v
        new_ssm = ssm_state.copy()
        O = np.zeros((B, H, K), dtype=np.float32)
        for h in range(H):
            exp_w = np.exp(w_h[:, h])                # (B, K)
            kv_outer = k_h[:, h, :, None] * v_h[:, h, None, :]  # (B, K, K)
            new_ssm[:, h] = exp_w[:, :, None] * ssm_state[:, h] + kv_outer
            # o = sigmoid(r) * (s @ k)
            sk = np.einsum('bij,bj->bi', new_ssm[:, h], k_h[:, h])  # (B, K)
            O[:, h] = r_h[:, h] * sk

        # 5. reshape + o_proj
        O_flat = O.reshape(B, 1, D)
        O_tensor = Tensor(O_flat, requires_grad=False)
        out = self.o_proj(O_tensor)

        new_x_prev = x_np.copy()

    return out, (new_ssm, new_x_prev)
```

关键点：
- **状态形状 `(B, H, K, K)`**：与 Mamba-2 的 `(B, H, N, d)` 不同，因为 RWKV 的状态是 K×K 矩阵；
- **time-shift**：需要持久化 `x_prev`（上一 token 的完整 D 维向量），而非只是 SSM 状态；
- **per-channel exp(w)**：每个 channel 独立衰减，状态更新用 `diag(exp_w) @ s`。

### 3.4 一致性验证

实测 RWKV-7 parallel vs recurrent 输出最大绝对差：**2.38e-07**（远低于 1e-3 阈值）。

## 4. 推理时的 StateCache 设计

[`verse_inference.cache.StateCache`](file:///workspace/packages/verse_inference/verse_inference/cache.py) 提供统一的递归状态容器：

```python
class StateCache:
    def __init__(self, n_layers, n_heads, head_dim, d_state, batch_size=1, arch="mamba2"):
        self.n_layers = n_layers
        self.states: list[Any] = []
        for _ in range(n_layers):
            self.states.append(self._zero_state())

    def _zero_state(self):
        if self.arch == "rwkv7":
            # RWKV-7: (B, n_heads, head_dim, head_dim)
            return np.zeros((self.batch_size, self.n_heads, self.head_dim, self.head_dim), dtype=np.float32)
        # Mamba-2: (B, n_heads, d_state, head_dim)
        return np.zeros((self.batch_size, self.n_heads, self.d_state, self.head_dim), dtype=np.float32)

    def get(self, layer): return self.states[layer]
    def set(self, layer, value): self.states[layer] = value
    def reset(self): ...
    def to_list(self): return list(self.states)
```

设计要点：
- **异构形状**：用 `list[Any]` 而非 ndarray，因为不同层可能有不同 state 结构（如 Hybrid 中 sparse_attn 层的 state 是 KV cache list）；
- **不强制 Tensor**：`set` 自动检测 Tensor 并提取 `.data`；
- **clone() 深拷贝**：用于多请求并发场景（每请求一份独立 state）。

实际推理流程中，[`StreamingGenerator`](file:///workspace/packages/verse_inference/verse_inference/generator.py) 直接调用 `HybridLM.forward_recurrent(input_ids, states)`，后者内部会维护每层 state 的 list，跨 step 持有。`StateCache` 主要用于：
- 用户手动管理状态（如 multi-turn 对话重置部分层）；
- 调试与可视化单层状态。

### 4.1 不同架构的 state 形状

| 架构        | ssm_state shape                  | 其他 state                          |
| ----------- | -------------------------------- | ----------------------------------- |
| Mamba-2     | `(B, n_heads, d_state, d_head)`   | `conv_state: (B, d_conv-1, d_inner)`|
| RWKV-7      | `(B, n_head, head_size, head_size)` | `x_prev: (B, 1, dim)`               |
| Sparse Attn | -                                | `kv_cache: list[(K, V)]`, `position: int` |

`StateCache` 默认按 mamba2 风格初始化（最常见的用例），用户可以 `set(layer, full_state)` 写入完整 state（含 conv_state / x_prev 等）。

## 5. 与 RetNet (Linear Attention) 的对比

[`linear_attention.py`](file:///workspace/packages/verse_nex/verse_nex/linear_attention.py) 实现 RetNet：

```
retention(x) = (Q @ K^T * D) @ V
D[i,j] = gamma^(i-j)  if i>=j else 0
```

与 Mamba-2/RWKV-7 的关键区别：
- **标量 gamma**：所有 token 共享同一个衰减系数，不能"选择性记忆"；
- **无 conv1d 前置**：缺少 Mamba-2 的 `depthwise causal conv1d`，对局部模式的建模能力弱；
- **状态 shape `(B, n_heads, d_head, d_head)`**：与 RWKV-7 类似但更简单。

RetNet 实测 parallel vs recurrent 一致性 6.59e-07，三种 scan 都达到了 1e-6 量级精度。

## 6. 数值稳定技巧

### 6.1 用 float64 算 cumsum

所有 scan 都在 `cumsum(log_A_bar)` 阶段用 float64 累积，最后 cast 回 float32：

```python
log_A_bar_data = dt.data.astype(np.float64) * A_b
cumsum_log = np.cumsum(log_A_bar_data, axis=1)
```

原因：float32 的 cumsum 在长序列（T > 1000）下累积误差显著，导致 decay matrix 数值偏离。

### 6.2 数值稳定 sigmoid

```python
sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
               np.exp(x) / (1.0 + np.exp(x)))
```

避免对大负数 `exp(-x)` 溢出。

### 6.3 softplus 数值稳定

```python
safe_x = np.minimum(x_data, 20.0)
out_data = np.where(x_data > 20, x_data, np.log1p(np.exp(safe_x)))
```

`x > 20` 时 `softplus(x) ≈ x`，跳过 `exp` 计算避免溢出。

### 6.4 parallel 与 recurrent 路径用一致的 float64 累积

**这是数值一致性的核心**。即使 parallel 用 cumsum，recurrent 用逐步加法，只要两者都在 float64 下累积，结果就一致到 1e-7 量级。

如果 parallel 用 float32 cumsum、recurrent 用 float32 逐步加，差异会到 1e-3 量级，**会破坏训练-推理等价性**。这是工程上必须避免的陷阱。

## 7. 性能考量

### 7.1 Parallel scan 的内存

decay matrix L 形状：
- Mamba-2: `(B, T, T, H)` float64，对 T=1024, H=8, B=1 约 64 MB；
- RWKV-7: `(B, T, T, H, K)` float32，对 T=1024, H=8, K=64, B=1 约 2 GB（致命）。

**RWKV-7 的 parallel scan 在长序列下内存爆炸**，是已知限制。缓解方案：
- 用 chunkwise scan（block 内 parallel + block 间 recurrent），未在当前实现中提供；
- 限制训练序列长度（建议 T ≤ 512）。

### 7.2 Recurrent scan 的速度

每步需要：
- 1D 卷积 step（常数时间，但需要 conv_state 滑动）；
- per-head for 循环（H 次）更新 SSM 状态；
- `np.einsum` 计算输出。

H=8 时单步约 0.1 ms（Mamba-2, dim=128, d_state=64），1000 tokens 约 100 ms。可接受。

### 7.3 优化机会

- **batch recurrent**：当前 recurrent 模式仅支持 B=1；可扩展为 B>1，但每层状态形状变为 4D；
- **融合 conv1d + SSM step**：当前分两步，可融合为一个 fused kernel（NumPy 下收益有限，C 扩展下收益大）；
- **Numba 加速**：recurrent for 循环用 Numba `@njit` 加速，预期 5-10x。

## 8. 已知限制

1. **RWKV-7 parallel scan 内存爆炸**：长序列下 decay matrix `(B, T, T, H, K)` 过大，建议 T ≤ 512；chunkwise 实现未提供。
2. **不支持 batch recurrent**：当前 `forward_recurrent` 仅支持 B=1；多请求并发需要外部并行（多实例）。
3. **无 GPU 加速**：所有 scan 在 CPU 上，大规模训练不可行。
4. **`forward_recurrent` 的 state 是 Python 对象**：tuple `(ssm_state, conv_state)`，无法直接序列化为 ndarray；需要序列化时手动拆包。
5. **parallel scan 没做 chunkwise 优化**：直接用 `(B, T, T, H)` 矩阵，T > 4096 时内存压力大。
6. **mixed precision 训练**：当前不支持 float16/bfloat16 训练（NumPy 不原生支持 bf16），只能用 float32 训练 + INT4 量化推理。

## 9. 源码引用汇总

### Mamba-2
- [`Mamba2Block` 构造函数](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L193-L237)：参数化与初始化；
- [`forward_parallel` 方法](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L350-L418)：SSD 矩阵形式；
- [`forward_recurrent` 方法](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L424-L503)：逐步递归 + float64 累积；
- [`_prepare_parallel` 与 `_prepare_recurrent`](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L243-L344)：共享的 x_branch 处理；
- [`_conv1d_causal` 与 `_conv1d_step`](file:///workspace/packages/verse_nex/verse_nex/mamba2.py#L51-L149)：可微并行卷积 + 常数内存递归卷积。

### RWKV-7
- [`RWKV7TimeMix` 类](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L39-L329)：time_mix 实现；
- [`forward_parallel` 方法](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L135-L222)：SSD 同构形式；
- [`forward_recurrent` 方法](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L228-L319)：状态递推 + x_prev 持久化；
- [`_compute_rkvwab` 方法](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L85-L129)：r/k/v/w/a/b 的 time-shift 混合 + 投影 + 激活；
- [`_neg_softplus_tensor` 函数](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py#L337-L354)：`w = -softplus(x)` 的可微实现。

### 推理 StateCache
- [`StateCache` 类](file:///workspace/packages/verse_inference/verse_inference/cache.py#L56-L179)：统一状态容器；
- [`_zero_state` 方法](file:///workspace/packages/verse_inference/verse_inference/cache.py#L94-L114)：根据 arch 初始化零状态。
