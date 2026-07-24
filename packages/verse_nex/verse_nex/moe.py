"""VerseNex: MoD (Mixture of Dense Parts) 多稠密分区架构。

参考 MoD 论文的双层路由设计：
- 第一层（part_router）：soft routing，将 token 加权分配到 ``num_dense_parts`` 个稠密分区。
  当 ``top_k == num_dense_parts`` 时，所有分区都被选中，权重为 softmax 概率。
- 第二层（expert_router）：每个 ``DensePart`` 内部 top-k 硬路由到 ``num_experts`` 个 Expert。

总输出::

    out = Σ_{p=1}^{num_dense_parts} part_weights[p] × DensePart_p(x)

每个 Router 计算 Switch Transformer 风格的 load balancing aux loss::

    aux = num_routes × Σ_i (f_i × P_i)

其中：
- ``f_i`` 是被路由到 route i 的 token 比例（top-k 命中，不可微，视为常量）
- ``P_i`` 是 router 给 route i 的平均概率（可微，梯度通过 softmax 回传）

设计要点：
- Expert 复用 ``verse_torch.nn.SwiGLUMLP`` 的 SwiGLU MLP 结构（同 Linear / Dropout / silu）
- dispatch/combine 用 numpy 索引实现，手写 ``_backward`` 闭包保持梯度可微
- 不实现 capacity 限制、expert parallelism、token dropping
- 重点是正确性和可微性

典型用法::

    from verse_nex.moe import MoDLayer
    mod = MoDLayer(dim=512, num_dense_parts=5, num_experts_per_part=8, top_k=3)
    out, aux = mod(x)   # x: (B, T, 512) -> out: (B, T, 512), aux: scalar
    loss = task_loss + aux
    loss.backward()
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.vnn import (
    Module,
    ModuleList,
    Linear,
    LayerNorm,
    Dropout,
    SwiGLUMLP,
    normal_,
)


__all__ = [
    "Router",
    "Expert",
    "DensePart",
    "MoDLayer",
]


# ---------------------------------------------------------------------------
# 可微索引工具函数
# ---------------------------------------------------------------------------


def _gather_topk(x: Tensor, indices: np.ndarray) -> Tensor:
    """沿最后一轴做可微 gather（等价于 ``np.take_along_axis``）。

    Args:
        x: ``(B, T, D)`` 的 Tensor
        indices: ``(B, T, K)`` 的 int ndarray，表示要 gather 的索引

    Returns:
        ``(B, T, K)`` 的 Tensor，满足 ``out[b, t, k] = x[b, t, indices[b, t, k]]``

    反向传播：用 ``np.add.at`` 把上游梯度 scatter 回 ``x`` 的对应位置（支持重复索引）。
    """
    B, T, D = x.shape
    K = indices.shape[-1]
    # 构造与 indices 同形状的 batch 维索引（高级索引需要所有索引数组形状一致）
    b_idx = np.broadcast_to(
        np.arange(B, dtype=np.int64)[:, None, None], (B, T, K)
    ).copy()
    t_idx = np.broadcast_to(
        np.arange(T, dtype=np.int64)[None, :, None], (B, T, K)
    ).copy()
    idx_tuple = (b_idx, t_idx, indices.astype(np.int64))

    out_data = x.data[idx_tuple]  # (B, T, K)

    def _backward():
        if x.requires_grad:
            grad = np.zeros_like(x.data)
            np.add.at(grad, idx_tuple, out.grad)
            x._accumulate_grad(grad)

    out = x._result(out_data, (x,), "gather_topk")
    if out.requires_grad:
        out._backward = _backward
    return out


def _scatter_to_mask(values: Tensor, mask: np.ndarray, out_shape: tuple) -> Tensor:
    """把 ``values`` scatter 到 ``mask`` 对应位置，输出全尺寸 Tensor。

    Args:
        values: ``(N, D)`` 的 Tensor
        mask: ``(B, T)`` 的 bool ndarray，``True`` 表示该位置有值
        out_shape: 输出形状 ``(B, T, D)``

    Returns:
        ``(B, T, D)`` 的 Tensor，``out[mask] = values``，其余位置为 0

    反向传播：``dvalues = dout[mask]``（gather）。
    """
    out_data = np.zeros(out_shape, dtype=values.data.dtype)
    out_data[mask] = values.data

    def _backward():
        if values.requires_grad:
            values._accumulate_grad(out.grad[mask])

    out = values._result(out_data, (values,), "scatter_to_mask")
    if out.requires_grad:
        out._backward = _backward
    return out


def _dispatch_and_combine(
    x: Tensor,
    dispatched_indices: Tensor,
    dispatched_weights: Tensor,
    experts: ModuleList,
    num_experts: int,
) -> Tensor:
    """Token dispatch 到各 Expert 并加权 combine。

    对每个 expert ``e`` 执行：
        1. 找出所有被路由到 ``e`` 的 token（在 top-k 的任一槽位命中 ``e``）
        2. gather 这些 token 的输入 ``x_e``
        3. ``y_e = expert_e(x_e)``
        4. 用对应的 router weight 加权 ``y_e``
        5. scatter 回原位置，累加到输出

    最终::

        out[b, t] = Σ_k dispatched_weights[b, t, k] × expert_{idx[b,t,k]}(x[b, t])

    全程可微：gather 通过 ``Tensor.__getitem__`` 的 boolean mask 索引，
    scatter 通过 ``_scatter_to_mask``，加权与累加通过 ``__mul__`` / ``__add__``。

    Args:
        x: ``(B, T, D)`` 输入
        dispatched_indices: ``(B, T, top_k)`` int Tensor（不可微）
        dispatched_weights: ``(B, T, top_k)`` float Tensor（可微）
        experts: ``ModuleList`` of Expert
        num_experts: expert 数量

    Returns:
        ``(B, T, D)`` 的 combine 后输出（可微）
    """
    B, T, D = x.shape
    idx_np = dispatched_indices.data  # (B, T, top_k) int
    w = dispatched_weights  # (B, T, top_k) Tensor

    # 初始化输出为零张量（不参与梯度，后续 __add__ 会正确传播 requires_grad）
    out = Tensor(np.zeros((B, T, D), dtype=np.float32), requires_grad=False)

    for e in range(num_experts):
        # slot_mask: (B, T, top_k) bool，表示哪些槽位路由到 expert e
        slot_mask = (idx_np == e)
        # token_mask: (B, T) bool，表示哪些 token 至少有一个槽位路由到 e
        token_mask = slot_mask.any(axis=-1)

        if not token_mask.any():
            # 没有 token 被路由到该 expert，跳过（该 expert 不贡献输出与梯度）
            continue

        # --- gather 输入 ---
        # x[token_mask] 通过 __getitem__ 的 boolean mask 索引，可微
        x_e = x[token_mask]  # (N_e, D)

        # --- expert 前向 ---
        y_e = experts[e](x_e)  # (N_e, D)

        # --- 提取 expert e 对应的权重 ---
        # w_e_full[b, t] = Σ_k w[b, t, k] × slot_mask[b, t, k]
        # 由于每个 token 在 top-k 中最多命中 expert e 一次，等价于取出对应槽位的权重
        slot_mask_t = Tensor(slot_mask.astype(np.float32), requires_grad=False)
        w_e_full = (w * slot_mask_t).sum(dim=-1)  # (B, T)，可微
        w_e = w_e_full[token_mask]  # (N_e,)，可微

        # --- 加权 ---
        y_e_weighted = y_e * w_e.unsqueeze(-1)  # (N_e, D) × (N_e, 1) -> (N_e, D)

        # --- scatter 回原位置 ---
        scattered = _scatter_to_mask(y_e_weighted, token_mask, (B, T, D))
        out = out + scattered

    return out


def _expand_part_weights(
    dispatched_indices: Tensor,
    dispatched_weights: Tensor,
    num_parts: int,
) -> list:
    """将 Router 输出展开为每个 part 的 ``(B, T, 1)`` 权重张量。

    当 ``top_k == num_parts``（soft routing）时，每个 part 恰好出现一次，
    权重即为该 part 的 softmax 概率。

    Args:
        dispatched_indices: ``(B, T, top_k)`` int Tensor
        dispatched_weights: ``(B, T, top_k)`` float Tensor（可微）
        num_parts: part 数量

    Returns:
        list of ``num_parts`` 个 ``(B, T, 1)`` Tensor
    """
    idx_np = dispatched_indices.data
    w = dispatched_weights

    part_weights = []
    for p in range(num_parts):
        # mask: (B, T, top_k) float，1 表示该槽位指向 part p
        mask = (idx_np == p).astype(np.float32)
        mask_t = Tensor(mask, requires_grad=False)
        # w_p[b, t] = Σ_k w[b, t, k] × mask[b, t, k]  -> (B, T, 1)
        w_p = (w * mask_t).sum(dim=-1, keepdim=True)
        part_weights.append(w_p)

    return part_weights


# ---------------------------------------------------------------------------
# Router: Top-k token 路由器
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 load balancing aux loss + router z-loss。

    对输入的每个 token 计算 router logits（通过一个线性层 ``gate``），
    选出 top-k 个 route，并用 softmax 重归一化得到 ``dispatched_weights``。

    同时计算两种辅助损失：

    1. **load balancing loss**（Switch Transformer 风格）::

        f_i = (被路由到 route i 的 token 数) / (总 token 数)   # 不可微
        P_i = mean_{b,t}(softmax_prob[b, t, i])                # 可微
        load_balance = num_routes × Σ_i (f_i × P_i) × aux_loss_weight

    2. **router z-loss**（ST-MoE 风格，防 router logits 过大）::

        z_loss = z_loss_weight × (1/N) × Σ_{b,t,i} (logits_{b,t,i})^2

    总 aux loss = load_balance + z_loss。

    ``f_i`` 视为常量（detached），梯度仅通过 ``P_i`` 回传到 router logits。
    z-loss 的梯度直接通过 ``logits^2`` 回传到 gate 权重。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（expert 数或 part 数）
        top_k: 每个 token 选出的 route 数（``1 <= top_k <= num_routes``）
        aux_loss_weight: load balancing loss 的权重系数
        z_loss_weight: router z-loss 的权重系数（0 表示不计算 z-loss）
        jitter: 训练时加到输入的均匀噪声幅度（0 表示不加）
        mod_version: MoD 路由版本（Part5K1.1）。
            - ``"1.1"``：原版路由（gate 直接作用于输入）。
            - ``"1.2"``（默认）：路由输入 LayerNorm + 熵正则 + 更稳的 gate 初始化，
              提升路由稳定性与能力，缓解训练初期 router collapse。
        entropy_weight: 路由熵正则权重（仅 V1.2，默认 1e-3）。鼓励 router 概率分布
            更均匀，与 load_balance loss 互补（前者作用于 soft prob，后者作用于
            hard 派发比例）。
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        jitter: float = 0.0,
        mod_version: str = "1.2",
        entropy_weight: float = 1e-3,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k 必须 >= 1，got {top_k}")
        if top_k > num_routes:
            raise ValueError(f"top_k({top_k}) 不能大于 num_routes({num_routes})")

        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.jitter = jitter
        self.mod_version = str(mod_version)
        self.entropy_weight = float(entropy_weight)

        # 路由线性层（无 bias，与 Switch Transformer 一致）
        self.gate = Linear(dim, num_routes, bias=False)
        # V1.2：更小的 init std，配合 router_norm 让初始 logits 更平稳，
        # 减少训练初期某 expert 被过度偏好的倾向
        init_std = 0.01 if self.mod_version >= "1.2" else 0.02
        normal_(self.gate.weight, std=init_std)

        # V1.2：路由输入 LayerNorm（GShard 风格），稳定 gate 输入分布。
        # 默认初始化（gamma=1, beta=0）近似 no-op，加载旧 V1.1 权重时无突变。
        if self.mod_version >= "1.2":
            self.router_norm = LayerNorm(dim)
        else:
            self.router_norm = None

        # 最近一次 forward 的 aux loss 分项（供外部读取）
        self._last_load_balance_loss = None
        self._last_z_loss = None
        self._last_entropy_loss = None

    def forward(self, x: Tensor):
        """前向计算。

        Args:
            x: ``(B, T, D)`` 的输入 Tensor

        Returns:
            dispatched_indices: ``(B, T, top_k)`` int Tensor（不可微）
            dispatched_weights: ``(B, T, top_k)`` float Tensor（可微，softmax 重归一化）
            aux_loss: 标量 Tensor（可微，load_balance + z_loss）
        """
        B, T, D = x.shape

        # 训练时加 jitter 噪声（有助于负载均衡，参考 Switch Transformer）
        if self.training and self.jitter > 0:
            noise = Tensor(
                ((np.random.rand(*x.shape).astype(np.float32) * 2.0 - 1.0) * self.jitter),
                requires_grad=False,
            )
            router_input = x + noise
        else:
            router_input = x

        # V1.2：路由输入 LayerNorm，稳定 gate 输入分布（GShard 风格）
        if self.router_norm is not None:
            router_input = self.router_norm(router_input)

        # Router logits: (B, T, num_routes)
        logits = self.gate(router_input)

        # 全 softmax 概率（用于 aux loss 的 P_i）
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # Top-k 选择（不可微，用 numpy）
        logits_np = logits.data
        # argsort 降序取前 top_k 个索引
        topk_idx = np.argsort(-logits_np, axis=-1)[..., :self.top_k]  # (B, T, top_k)
        topk_idx = topk_idx.astype(np.int64)

        # Gather top-k logits（可微，通过 _gather_topk 手写反向）
        topk_logits = _gather_topk(logits, topk_idx)  # (B, T, top_k)

        # 重归一化权重：对 top-k logits 做 softmax
        dispatched_weights = topk_logits.softmax(dim=-1)  # (B, T, top_k)

        # --- load balancing loss（Switch Transformer 风格）---
        # topk_onehot: (B, T, num_routes)，1 表示该 route 在 top-k 中
        topk_onehot = np.zeros((B, T, self.num_routes), dtype=np.float32)
        np.put_along_axis(topk_onehot, topk_idx, 1.0, axis=-1)

        # f_i = 被路由到 route i 的 token 比例（top-k 命中率，不可微）
        f = topk_onehot.reshape(-1, self.num_routes).mean(axis=0)  # (num_routes,)
        f_tensor = Tensor(f, requires_grad=False)

        # P_i = router 给 route i 的平均概率（可微，梯度回传到 logits）
        P = probs.mean(dim=(0, 1))  # (num_routes,)

        # load_balance = num_routes × Σ_i (f_i × P_i) × aux_loss_weight
        load_balance_loss = (f_tensor * P).sum() * float(self.num_routes)
        load_balance_loss = load_balance_loss * self.aux_loss_weight

        # --- router z-loss（ST-MoE 风格，防 logits 过大）---
        # z_loss = z_loss_weight × (1/N) × Σ_{b,t,i} (logits_{b,t,i})^2
        # 其中 N = B × T（token 总数），梯度通过 logits^2 回传到 gate 权重
        if self.z_loss_weight > 0:
            logits_sq = logits * logits  # (B, T, num_routes)，可微
            z_loss = logits_sq.sum() * (self.z_loss_weight / float(B * T))
        else:
            z_loss = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)

        # --- V1.2：路由熵正则（鼓励 soft prob 均匀，防 collapse）---
        # H = -Σ_i P_i log(P_i + eps)；最大化 H → 最小化 -H
        # 与 load_balance 互补：load_balance 作用于 hard 派发比例 f_i，
        # 熵正则直接作用于 soft prob P_i，在 top-k collapse 早期即可生效
        if self.mod_version >= "1.2" and self.entropy_weight > 0:
            probs_np = probs.data.astype(np.float64)
            P_mean = probs_np.mean(axis=(0, 1))  # (num_routes,)
            eps = 1e-8
            entropy = -float(np.sum(P_mean * np.log(P_mean + eps)))
            max_entropy = float(np.log(self.num_routes))  # 均匀分布的熵
            # 归一化到 [0,1]：1 - H/H_max ∈ [0,1]，0=均匀，1=完全 collapse
            entropy_gap = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
            entropy_loss = Tensor(
                np.asarray(self.entropy_weight * entropy_gap, dtype=np.float32),
                requires_grad=False,
            )
        else:
            entropy_loss = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)

        # 总 aux loss = load_balance + z_loss + entropy
        aux_loss = load_balance_loss + z_loss + entropy_loss

        # 缓存分项供外部读取
        # 用 object.__setattr__ 绕过 nn.Module.__setattr__ 的自动注册，
        # 避免这些临时 forward 缓存进入 _parameters / state_dict 导致
        # ParallelTrainer 合并 chunk 时 "Unexpected keys" 报错
        object.__setattr__(self, "_last_load_balance_loss", load_balance_loss)
        object.__setattr__(self, "_last_z_loss", z_loss)
        object.__setattr__(self, "_last_entropy_loss", entropy_loss)

        # 索引张量（不可微）
        dispatched_indices = Tensor(topk_idx, requires_grad=False)

        return dispatched_indices, dispatched_weights, aux_loss


# ---------------------------------------------------------------------------
# Expert: 单个 SwiGLU MLP Expert
# ---------------------------------------------------------------------------


class Expert(Module):
    """单个 SwiGLU MLP Expert（复用 ``verse_torch.nn.SwiGLUMLP`` 结构）。

    前向计算::

        h = w_down( silu(w_gate(x)) × w_up(x) )
        out = dropout(h)

    与 ``SwiGLUMLP`` 的区别：``hidden`` 维度由外部指定（不使用 ``hidden_multiple`` 公式），
    以支持 MoD 中 ``expert_hidden`` 的灵活配置。

    Args:
        dim: 输入/输出维度
        hidden: 隐藏层维度
        dropout: dropout 概率
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        # 三个线性层 + dropout，结构与 SwiGLUMLP 完全一致
        self.w_gate = Linear(dim, hidden, bias=False)
        self.w_up = Linear(dim, hidden, bias=False)
        self.w_down = Linear(hidden, dim, bias=False)
        self.dropout = Dropout(dropout)
        # 参数初始化（与 VerseNexLM._init_weights 风格一致）
        normal_(self.w_gate.weight, std=0.02)
        normal_(self.w_up.weight, std=0.02)
        normal_(self.w_down.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """前向计算: ``(..., D) → (..., D)``"""
        gate = self.w_gate(x).silu()
        up = self.w_up(x)
        h = gate * up
        h = self.w_down(h)
        h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# DensePart: 单个稠密分区
# ---------------------------------------------------------------------------


class DensePart(Module):
    """单个稠密分区，包含 ``num_experts`` 个 Expert + 内部 Router。

    接收外层的 ``part_weights`` 对输出进行加权（实现双层路由的融合）。
    内部 Router 将 token 路由到 top-k 个 Expert。

    前向计算::

        idx, w, aux = self.router(x)            # expert 路由
        combined = dispatch_and_combine(x, idx, w, experts)
        out = combined × part_weights           # 外层加权
        return out, aux

    Args:
        dim: 输入维度
        num_experts: expert 数量
        top_k: 每个 token 选出的 expert 数
        expert_hidden: expert 隐藏层维度（None 则自动计算为 ``int(dim×8/3/64)×64``）
        dropout: dropout 概率
        aux_loss_weight: load balancing loss 权重
        z_loss_weight: router z-loss 权重
        mod_version: MoD 路由版本（Part5K1.1，默认 ``"1.2"``，透传给内部 Router）
        entropy_weight: 路由熵正则权重（透传给内部 Router）
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 3,
        expert_hidden: int = None,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        mod_version: str = "1.2",
        entropy_weight: float = 1e-3,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k({top_k}) 不能大于 num_experts({num_experts})")

        if expert_hidden is None:
            # 默认 hidden = floor(dim × 8/3 / 64) × 64，与 SwiGLUMLP 的 4× 比例一致
            expert_hidden = max(int(dim * 8 / 3 / 64) * 64, 64)

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden = expert_hidden

        # Expert 列表（每个是独立实例，参数不共享）
        self.experts = ModuleList(
            [Expert(dim, expert_hidden, dropout) for _ in range(num_experts)]
        )

        # 内部 expert router（透传 V1.2 参数）
        self.router = Router(
            dim,
            num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
            mod_version=mod_version,
            entropy_weight=entropy_weight,
        )

    def forward(self, x: Tensor, part_weights: Tensor):
        """前向计算。

        Args:
            x: ``(B, T, D)`` 输入
            part_weights: ``(B, T, 1)`` 外层 router 给该分区的权重

        Returns:
            out: ``(B, T, D)`` 加权后的分区输出
            aux_loss: 标量 Tensor（expert router 的负载均衡 loss）
        """
        # Expert 路由
        dispatched_indices, dispatched_weights, aux_loss = self.router(x)

        # Dispatch + combine（可微）
        combined = _dispatch_and_combine(
            x,
            dispatched_indices,
            dispatched_weights,
            self.experts,
            self.num_experts,
        )

        # 乘以外层 part weights（广播: (B,T,D) × (B,T,1) -> (B,T,D)）
        out = combined * part_weights

        return out, aux_loss


# ---------------------------------------------------------------------------
# MoDLayer: MoD 顶层（双层路由）
# ---------------------------------------------------------------------------


class MoDLayer(Module):
    """MoD (Mixture of Dense Parts) 顶层：双层路由。

    **第一层 (part_router)**: soft routing，将 token 加权分配到
    ``num_dense_parts`` 个 ``DensePart``。当 ``top_k == num_dense_parts`` 时，
    所有 part 都被选中，权重为 softmax 概率（等价于 soft routing）。

    **第二层 (每个 DensePart 内 expert_router)**: top-k Expert 硬路由。

    总输出::

        out = Σ_{p=1}^{num_dense_parts} part_weights[p] × DensePart_p(x)

    总 aux loss（含 load_balance + z_loss）::

        total_aux = Σ_all_routers (load_balance_loss + z_loss)

    使用 ``aux_loss()`` 获取最近一次 forward 的总辅助损失标量，
    ``get_aux_loss_dict()`` 获取分项 breakdown。

    Args:
        dim: 输入维度
        num_dense_parts: 稠密分区数量（默认 5）
        num_experts_per_part: 每个分区内的 expert 数量（默认 8）
        top_k: 每个 token 在 expert 层选出的 expert 数（默认 3）
        expert_hidden: expert 隐藏层维度（None 则自动计算）
        dropout: dropout 概率
        aux_loss_weight: load balancing loss 权重
        z_loss_weight: router z-loss 权重（防 logits 过大）
        dense_part_names: 分区名称列表（可选，用于调试/可视化）
        mod_version: MoD 路由版本（Part5K1.1，默认 ``"1.2"``，透传给所有 Router）
        entropy_weight: 路由熵正则权重（透传给所有 Router）

    Note:
        - 5 DensePart × 8 Expert × top-3 双层门控结构（默认）
        - 不实现 capacity 限制、expert parallelism、token dropping
        - 全程可微，支持 ``loss.backward()`` 回传梯度到输入与所有参数
    """

    def __init__(
        self,
        dim: int,
        num_dense_parts: int = 5,
        num_experts_per_part: int = 8,
        top_k: int = 3,
        expert_hidden: int = None,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        dense_part_names: list = None,
        mod_version: str = "1.2",
        entropy_weight: float = 1e-3,
    ):
        super().__init__()
        if num_dense_parts < 1:
            raise ValueError(f"num_dense_parts 必须 >= 1，got {num_dense_parts}")
        if num_experts_per_part < 1:
            raise ValueError(
                f"num_experts_per_part 必须 >= 1，got {num_experts_per_part}"
            )
        if top_k > num_experts_per_part:
            raise ValueError(
                f"top_k({top_k}) 不能大于 num_experts_per_part({num_experts_per_part})"
            )

        if expert_hidden is None:
            expert_hidden = max(int(dim * 8 / 3 / 64) * 64, 64)

        self.dim = dim
        self.num_dense_parts = num_dense_parts
        self.num_experts_per_part = num_experts_per_part
        self.top_k = top_k
        self.expert_hidden = expert_hidden
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        self.mod_version = str(mod_version)

        # 分区名称（可选）
        if dense_part_names is not None:
            if len(dense_part_names) != num_dense_parts:
                raise ValueError(
                    f"dense_part_names 长度({len(dense_part_names)}) "
                    f"必须等于 num_dense_parts({num_dense_parts})"
                )
            self.dense_part_names = list(dense_part_names)
        else:
            self.dense_part_names = [f"part_{i}" for i in range(num_dense_parts)]

        # 第一层：part router（soft routing，top_k = num_dense_parts 选所有 part）
        self.part_router = Router(
            dim,
            num_dense_parts,
            top_k=num_dense_parts,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
            mod_version=mod_version,
            entropy_weight=entropy_weight,
        )

        # 第二层：DensePart 列表
        self.parts = ModuleList(
            [
                DensePart(
                    dim,
                    num_experts=num_experts_per_part,
                    top_k=top_k,
                    expert_hidden=expert_hidden,
                    dropout=dropout,
                    aux_loss_weight=aux_loss_weight,
                    z_loss_weight=z_loss_weight,
                    mod_version=mod_version,
                    entropy_weight=entropy_weight,
                )
                for _ in range(num_dense_parts)
            ]
        )

        # 最近一次 forward 的 aux loss breakdown（供外部读取）
        self._last_aux_dict: Optional[dict] = None

    def forward(self, x: Tensor):
        """前向计算。

        Args:
            x: ``(B, T, D)`` 输入

        Returns:
            out: ``(B, T, D)`` MoD 输出
            total_aux_loss: 标量 Tensor（所有 Router 的 aux loss 之和，
                            含 load_balance + z_loss）
        """
        B, T, D = x.shape

        # --- 第一层路由：part routing（soft） ---
        part_indices, part_weights_dispatched, part_aux = self.part_router(x)
        # part_indices: (B, T, num_dense_parts) — 所有 part，按 logit 降序
        # part_weights_dispatched: (B, T, num_dense_parts) — softmax 权重

        # 展开为每个 part 的 (B, T, 1) 权重（可微）
        part_weights_list = _expand_part_weights(
            part_indices, part_weights_dispatched, self.num_dense_parts
        )

        # --- 第二层路由 + 各分区前向 ---
        total_aux = part_aux
        # 分别累加 load_balance 与 z_loss（用于 breakdown）
        total_load_balance = self.part_router._last_load_balance_loss
        total_z_loss = self.part_router._last_z_loss
        # 初始化输出为零张量（不参与梯度，后续 __add__ 会正确传播）
        out = Tensor(np.zeros((B, T, D), dtype=np.float32), requires_grad=False)

        for p in range(self.num_dense_parts):
            part_out, expert_aux = self.parts[p](x, part_weights_list[p])
            out = out + part_out
            total_aux = total_aux + expert_aux
            # 累加分项（每个 DensePart 的 router 缓存了 _last_load_balance_loss / _last_z_loss）
            if self.parts[p].router._last_load_balance_loss is not None:
                total_load_balance = (
                    total_load_balance
                    + self.parts[p].router._last_load_balance_loss
                )
            if self.parts[p].router._last_z_loss is not None:
                total_z_loss = total_z_loss + self.parts[p].router._last_z_loss

        # 缓存 aux loss breakdown 供外部读取
        self._last_aux_dict = {
            "load_balance": total_load_balance,
            "z_loss": total_z_loss,
            "total": total_aux,
        }

        return out, total_aux

    # ------------------------------------------------------------------
    # aux loss 访问接口
    # ------------------------------------------------------------------

    def aux_loss(self):
        """返回最近一次 forward 的总辅助损失标量 Tensor。

        含所有 Router（part_router + 各 DensePart 内 expert_router）的
        load_balance_loss + z_loss 之和。

        Returns:
            标量 Tensor（可微）；若尚未 forward 则返回 None。
        """
        if self._last_aux_dict is None:
            return None
        return self._last_aux_dict["total"]

    def get_aux_loss_dict(self) -> Optional[dict]:
        """返回最近一次 forward 的辅助损失分项 breakdown。

        Returns:
            dict with keys:
                - ``"load_balance"``: 所有 Router 的 load_balance_loss 之和（标量 Tensor）
                - ``"z_loss"``: 所有 Router 的 z_loss 之和（标量 Tensor）
                - ``"total"``: 总辅助损失（标量 Tensor，= load_balance + z_loss）
            若尚未 forward 则返回 None。
        """
        return self._last_aux_dict
