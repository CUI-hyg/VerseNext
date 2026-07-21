"""VerseNex MoD（多稠密分区架构技术）— Part4 P2.2。

灵感来源
--------
人脑的不同区域负责不同能力：前额叶负责数学推理与认知规划、颞叶负责语言理解、
海马体负责记忆、枕叶负责视觉等。每个区域内又有专门的神经元集群（"专家"），
根据当前任务动态激活相关集群。

MoD（Mixture of Dense Parts）架构
----------------------------------
1. **DensePart（稠密分区）**：相当于大脑的一个功能分区
   - 每个 DensePart 内部包含多个 Expert
   - 每个 DensePart 有自己的 Inner Router（专家路由器）
   - DensePart 之间相互独立，各自擅长不同能力域

2. **Expert（专家）**：DensePart 内部的子网络
   - 每个 Expert 是一个 SwiGLU MLP
   - Inner Router 动态选择 Top-K 个 Expert 激活
   - 实现"效率与质量相互照顾"：稀疏激活省算力，多专家保质量

3. **MoDRouter（外层路由器）**：把每个 token 路由到 Top-K 个 DensePart
   - 学习每个 DensePart 的能力倾向
   - 不同 token 激活不同 DensePart 组合

4. **Load Balancing**：辅助 loss 鼓励均匀路由（避免某些 Part/Expert 被冷落）
   - Part-level balance loss
   - Expert-level balance loss（在每个 DensePart 内部）
   - z-loss 防止 router logits 过大

数据流
------
    x (B, T, d)
        ↓
    MoDRouter → part_logits (B, T, n_parts)
        ↓ softmax + Top-K
    part_weights (B, T, top_k_parts), part_idx (B, T, top_k_parts)
        ↓
    For each selected Part p:
        DensePart_p.InnerRouter → expert_logits (B, T, n_experts)
            ↓ softmax + Top-K
        expert_weights (B, T, top_k_experts), expert_idx (B, T, top_k_experts)
            ↓
        For each selected Expert e:
            expert_out = Expert_e(x)  # SwiGLU MLP
        part_out = Σ expert_weights * expert_out
        ↓
    output = Σ part_weights * part_out

复杂度
------
- 全激活：O(n_parts × n_experts × d_ff × d) per token
- Top-K Parts × Top-K Experts：O(top_k_parts × top_k_experts × d_ff × d) per token
- 当 top_k_parts=2, top_k_experts=2, n_parts=4, n_experts=4 时，激活 4/16 = 25%
- 参数量保持 16x（全专家），但计算量仅 4x → 高参数效率
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from verse_torch.tensor import Tensor
from verse_torch.nn import Module, Linear, Dropout, ModuleList


# ---------------------------------------------------------------------------
# Expert：单个专家（SwiGLU MLP，自定义 d_ff）
# ---------------------------------------------------------------------------


class Expert(Module):
    """单个 Expert：SwiGLU MLP，直接指定中间维度 d_ff。

    结构：x (B, T, d) → SiLU(W_gate(x)) * W_up(x) → W_down → (B, T, d)
    其中 W_gate/W_up: d → d_ff，W_down: d_ff → d
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_gate = Linear(d_model, d_ff, bias=False)
        self.w_up = Linear(d_model, d_ff, bias=False)
        self.w_down = Linear(d_ff, d_model, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x).silu()  # SiLU 激活
        up = self.w_up(x)
        h = gate * up
        h = self.w_down(h)
        h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# DensePart：稠密分区（含多个 Expert + Inner Router）
# ---------------------------------------------------------------------------


class DensePart(Module):
    """稠密分区：人脑功能分区的抽象。

    每个 DensePart 包含：
        - n_experts 个 Expert（SwiGLU MLP）
        - Inner Router：把 token 路由到 Top-K Expert

    Args:
        d_model: 模型维度
        d_ff: Expert MLP 中间维度
        n_experts: 分区内专家数量
        top_k_experts: 每个 token 激活的专家数
        dropout: dropout 概率

    forward:
        x: (B, T, d_model)
        → (out: (B, T, d_model), aux_loss: Tensor scalar)

    aux_loss: Expert-level load balancing loss（鼓励专家均匀使用）
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        top_k_experts: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k_experts = min(top_k_experts, n_experts)
        # Inner Router：d_model → n_experts
        self.inner_router = Linear(d_model, n_experts, bias=False)
        # n_experts 个 Expert
        self.experts = ModuleList(
            [Expert(d_model, d_ff, dropout=dropout) for _ in range(n_experts)]
        )

    def forward(self, x: Tensor):
        B, T, d = x.shape
        K = self.top_k_experts
        E = self.n_experts

        # 1. Inner Router logits
        router_logits = self.inner_router(x)  # (B, T, E)
        logits_data = router_logits.data

        # 2. softmax 得到路由权重
        max_logits = np.max(logits_data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_data - max_logits)
        sum_exp = exp_logits.sum(axis=-1, keepdims=True)
        sum_exp = np.maximum(sum_exp, 1e-12)
        probs = exp_logits / sum_exp  # (B, T, E)

        # 3. Top-K 选择
        if K < E:
            sorted_idx = np.argsort(probs, axis=-1)
            topk_idx = sorted_idx[..., -K:]  # (B, T, K)
        else:
            topk_idx = np.broadcast_to(
                np.arange(E), (B, T, E)
            ).copy()

        # 4. 收集 Top-K 路由权重
        b_idx, t_idx = np.indices((B, T))
        b_idx_exp = np.broadcast_to(b_idx[..., None], topk_idx.shape)
        t_idx_exp = np.broadcast_to(t_idx[..., None], topk_idx.shape)
        topk_weights = probs[b_idx_exp, t_idx_exp, topk_idx]  # (B, T, K)

        # 5. 归一化 Top-K 权重（使其和为 1）
        weight_sum = topk_weights.sum(axis=-1, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-12)
        topk_weights_norm = topk_weights / weight_sum  # (B, T, K)

        # 6. 计算所有 Expert 的输出，然后按 Top-K 选择加权
        #    为了简化实现，先计算所有 Expert 输出（小规模时性能可接受）
        #    大规模时可改为只计算被选中的 Expert
        all_expert_outs = []
        for e_idx in range(E):
            expert_out = self.experts[e_idx](x)  # (B, T, d)
            all_expert_outs.append(expert_out)
        # stack: (B, T, E, d) — 但这会占用大量内存，改为按需选择
        # 改用循环计算 Top-K Expert 的输出并加权
        # 注意：上面的循环已经计算了所有 Expert，现在按 Top-K 选择

        # 7. 按 Top-K 选择 Expert 输出并加权
        #    topk_idx: (B, T, K)，对每个 (b, t) 选 K 个 Expert
        out_data = np.zeros((B, T, d), dtype=np.float32)
        # 保存梯度所需信息
        requires_grad = (
            x.requires_grad
            or any(p.requires_grad for p in self.parameters())
        )

        # 为 backward 准备：保存每个被选中 Expert 的输出 Tensor 和权重
        selected_outs = []  # list of (expert_idx_tensor, weight_array)
        for k in range(K):
            # 对每个 k 位置，收集 (B, T) 个 token 各自选中的 Expert
            # topk_idx[:, :, k] → (B, T) Expert 索引
            k_expert_idx = topk_idx[:, :, k]  # (B, T)
            k_weight = topk_weights_norm[:, :, k]  # (B, T)

            # 对每个 (b, t)，取 all_expert_outs[k_expert_idx[b,t]][b, t]
            # 用 advanced indexing
            b_grid, t_grid = np.meshgrid(
                np.arange(B), np.arange(T), indexing="ij"
            )
            # 选择每个 token 对应的 Expert 输出
            selected_out_list = []
            for b in range(B):
                for t in range(T):
                    e = k_expert_idx[b, t]
                    selected_out_list.append(all_expert_outs[e].data[b, t])
            selected_out = np.stack(selected_out_list, axis=0).reshape(B, T, d)
            out_data += k_weight[..., None] * selected_out

            # 记录梯度信息
            selected_outs.append((k_expert_idx, k_weight, selected_out))

        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            _children=tuple(all_expert_outs) + (x,) if requires_grad else (),
            _op="dense_part",
        )

        if requires_grad:
            # 保存 backward 所需的上下文
            _all_expert_outs = all_expert_outs
            _selected_outs = selected_outs
            _K = K
            _d = d
            _B, _T = B, T

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad  # (B, T, d)
                # 对每个 k 位置，把 grad * weight 传回对应的 Expert
                for k in range(_K):
                    k_expert_idx, k_weight, _ = _selected_outs[k]
                    # grad_k: (B, T, d) = grad * k_weight[..., None]
                    grad_k = grad * k_weight[..., None]
                    # 把 grad_k 传回每个被选中的 Expert
                    for b in range(_B):
                        for t in range(_T):
                            e = k_expert_idx[b, t]
                            # 给 all_expert_outs[e] 的 (b, t) 位置累加梯度
                            expert_grad = np.zeros((_B, _T, _d), dtype=np.float32)
                            expert_grad[b, t] = grad_k[b, t]
                            _all_expert_outs[e]._accumulate_grad(expert_grad)
                # x 的梯度由各 Expert 内部 backward 传递

            out._backward = _backward

        # 8. Load balancing aux loss（Expert 级别）
        #    鼓励每个 Expert 被均匀路由
        #    balance_loss = n_experts * sum(mean_prob_e * fraction_tokens_e)
        #    其中 mean_prob_e = probs[:, :, e].mean()（平均路由概率）
        #         fraction_tokens_e = (topk_idx == e).sum() / total_tokens
        aux_loss_data = self._compute_balance_loss(probs, topk_idx, B, T, E, K)
        aux_loss = Tensor(
            aux_loss_data,
            requires_grad=router_logits.requires_grad,
            _children=(router_logits,) if router_logits.requires_grad else (),
            _op="expert_balance_loss",
        )
        if router_logits.requires_grad:
            _probs = probs
            _topk_idx = topk_idx
            _E, _K = E, K

            def _aux_backward():
                if aux_loss.grad is None:
                    return
                g_scalar = float(aux_loss.grad)
                # balance_loss = E * sum_e(mean_prob_e * frac_e)
                # d(balance_loss)/d(router_logits) 涉及 softmax Jacobian
                # 简化：用 probs 的梯度近似
                # mean_prob_e = probs[:, :, e].mean()
                # frac_e = (topk_idx == e).sum() / (B*T*K) （近似）
                # d mean_prob_e / d logits = (probs_e * (1 - probs_e)) / (B*T) 等
                # 简化实现：直接用 probs * (1 - probs) 形式
                total_tokens = B * T
                frac = np.zeros(_E, dtype=np.float32)
                for e in range(_E):
                    frac[e] = np.sum(_topk_idx == e) / max(total_tokens * _K, 1)
                # d loss / d probs_e = E * frac_e / (B*T)
                dprobs = np.zeros_like(_probs)
                for e in range(_E):
                    dprobs[:, :, e] = _E * frac[e] / max(total_tokens, 1)
                # softmax backward: dloss/dlogits = dprobs * probs - probs * sum(dprobs * probs)
                dot = (dprobs * _probs).sum(axis=-1, keepdims=True)
                dlogits = _probs * (dprobs - dot)
                router_logits._accumulate_grad(dlogits * g_scalar)

            aux_loss._backward = _aux_backward

        return out, aux_loss

    def _compute_balance_loss(
        self, probs: np.ndarray, topk_idx: np.ndarray,
        B: int, T: int, E: int, K: int,
    ) -> np.ndarray:
        """Expert-level load balancing loss（Switch Transformer 风格）。

        loss = E * sum_e(mean_prob_e * fraction_e)
        其中:
            mean_prob_e = mean(probs[:, :, e])  # 平均路由概率
            fraction_e = (topk_idx == e).sum() / (B * T * K)  # 被选中频率

        当路由完全均匀时，loss ≈ 1.0（最小值）。
        """
        total_tokens = B * T
        total_selected = max(total_tokens * K, 1)
        loss = 0.0
        for e in range(E):
            mean_prob_e = float(probs[:, :, e].mean())
            fraction_e = float(np.sum(topk_idx == e)) / total_selected
            loss += E * mean_prob_e * fraction_e
        return np.array(loss, dtype=np.float32)


# ---------------------------------------------------------------------------
# MoDBlock：完整的多稠密分区块
# ---------------------------------------------------------------------------


class MoDBlock(Module):
    """MoD（多稠密分区）完整块。

    结构：
        x → MoDRouter → Top-K Parts → 每个 Part 内部 Top-K Experts → 加权求和

    Args:
        d_model: 模型维度
        d_ff: Expert MLP 中间维度
        n_parts: DensePart 数量（大脑分区数）
        n_experts_per_part: 每个 DensePart 内的 Expert 数
        top_k_parts: 每个 token 激活的 DensePart 数
        top_k_experts: 每个 DensePart 内激活的 Expert 数
        dropout: dropout 概率
        aux_loss_weight: load balancing loss 权重

    forward:
        x: (B, T, d_model)
        → (out: (B, T, d_model), total_aux_loss: Tensor scalar)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_parts: int = 4,
        n_experts_per_part: int = 4,
        top_k_parts: int = 2,
        top_k_experts: int = 2,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_parts = n_parts
        self.n_experts_per_part = n_experts_per_part
        self.top_k_parts = min(top_k_parts, n_parts)
        self.top_k_experts = top_k_experts
        self.aux_loss_weight = aux_loss_weight

        # 外层 Router：d_model → n_parts
        self.router = Linear(d_model, n_parts, bias=False)
        # n_parts 个 DensePart
        self.parts = ModuleList(
            [
                DensePart(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_experts=n_experts_per_part,
                    top_k_experts=top_k_experts,
                    dropout=dropout,
                )
                for _ in range(n_parts)
            ]
        )

    def forward(self, x: Tensor):
        B, T, d = x.shape
        P = self.n_parts
        Kp = self.top_k_parts

        # 1. 外层 Router logits
        router_logits = self.router(x)  # (B, T, P)
        logits_data = router_logits.data

        # 2. softmax
        max_logits = np.max(logits_data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_data - max_logits)
        sum_exp = exp_logits.sum(axis=-1, keepdims=True)
        sum_exp = np.maximum(sum_exp, 1e-12)
        probs = exp_logits / sum_exp  # (B, T, P)

        # 3. Top-K Parts 选择
        if Kp < P:
            sorted_idx = np.argsort(probs, axis=-1)
            topk_part_idx = sorted_idx[..., -Kp:]  # (B, T, Kp)
        else:
            topk_part_idx = np.broadcast_to(
                np.arange(P), (B, T, P)
            ).copy()

        b_idx, t_idx = np.indices((B, T))
        b_idx_exp = np.broadcast_to(b_idx[..., None], topk_part_idx.shape)
        t_idx_exp = np.broadcast_to(t_idx[..., None], topk_part_idx.shape)
        topk_part_weights = probs[b_idx_exp, t_idx_exp, topk_part_idx]  # (B, T, Kp)

        # 归一化
        weight_sum = topk_part_weights.sum(axis=-1, keepdims=True)
        weight_sum = np.maximum(weight_sum, 1e-12)
        topk_part_weights_norm = topk_part_weights / weight_sum

        # 4. 对每个被选中的 DensePart，计算其输出
        #    为了高效，先计算所有 Part 的输出，再按 Top-K 选择
        #    （小规模时可接受，大规模时可改为只计算被选中的 Part）
        all_part_outs = []
        all_part_aux_losses = []
        for p_idx in range(P):
            part_out, part_aux = self.parts[p_idx](x)
            all_part_outs.append(part_out)
            all_part_aux_losses.append(part_aux)

        # 5. 按 Top-K 选择 Part 输出并加权
        out_data = np.zeros((B, T, d), dtype=np.float32)
        requires_grad = (
            x.requires_grad
            or any(p.requires_grad for p in self.parameters())
        )

        selected_part_info = []  # (part_idx_array, weight_array, part_out_array)
        for k in range(Kp):
            k_part_idx = topk_part_idx[:, :, k]  # (B, T)
            k_weight = topk_part_weights_norm[:, :, k]  # (B, T)

            # 收集每个 token 对应的 Part 输出
            selected_part_out = np.zeros((B, T, d), dtype=np.float32)
            for b in range(B):
                for t in range(T):
                    p = k_part_idx[b, t]
                    selected_part_out[b, t] = all_part_outs[p].data[b, t]
            out_data += k_weight[..., None] * selected_part_out
            selected_part_info.append((k_part_idx, k_weight, selected_part_out))

        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            _children=tuple(all_part_outs) + (x,) if requires_grad else (),
            _op="mod_block",
        )

        if requires_grad:
            _all_part_outs = all_part_outs
            _selected_part_info = selected_part_info
            _Kp = Kp
            _B, _T, _d = B, T, d

            def _backward():
                if out.grad is None:
                    return
                grad = out.grad  # (B, T, d)
                for k in range(_Kp):
                    k_part_idx, k_weight, _ = _selected_part_info[k]
                    grad_k = grad * k_weight[..., None]
                    # 把 grad_k 传回每个被选中的 Part
                    for b in range(_B):
                        for t in range(_T):
                            p = k_part_idx[b, t]
                            part_grad = np.zeros((_B, _T, _d), dtype=np.float32)
                            part_grad[b, t] = grad_k[b, t]
                            _all_part_outs[p]._accumulate_grad(part_grad)

            out._backward = _backward

        # 6. 汇总 aux loss（Part 级 + 所有 Expert 级）
        # Part 级 balance loss
        part_aux_data = self._compute_part_balance_loss(
            probs, topk_part_idx, B, T, P, Kp
        )
        part_aux = Tensor(
            part_aux_data,
            requires_grad=router_logits.requires_grad,
            _children=(router_logits,) if router_logits.requires_grad else (),
            _op="part_balance_loss",
        )
        if router_logits.requires_grad:
            _probs = probs
            _topk_part_idx = topk_part_idx
            _P, _Kp = P, Kp

            def _part_aux_backward():
                if part_aux.grad is None:
                    return
                g_scalar = float(part_aux.grad)
                total_tokens = B * T
                frac = np.zeros(_P, dtype=np.float32)
                for p in range(_P):
                    frac[p] = np.sum(_topk_part_idx == p) / max(total_tokens * _Kp, 1)
                dprobs = np.zeros_like(_probs)
                for p in range(_P):
                    dprobs[:, :, p] = _P * frac[p] / max(total_tokens, 1)
                dot = (dprobs * _probs).sum(axis=-1, keepdims=True)
                dlogits = _probs * (dprobs - dot)
                router_logits._accumulate_grad(dlogits * g_scalar)

            part_aux._backward = _part_aux_backward

        # 总 aux loss = part_aux + sum(expert_aux)
        total_aux = part_aux
        for ea in all_part_aux_losses:
            total_aux = total_aux + ea

        return out, total_aux

    def _compute_part_balance_loss(
        self, probs: np.ndarray, topk_idx: np.ndarray,
        B: int, T: int, P: int, K: int,
    ) -> np.ndarray:
        """Part-level load balancing loss。"""
        total_tokens = B * T
        total_selected = max(total_tokens * K, 1)
        loss = 0.0
        for p in range(P):
            mean_prob_p = float(probs[:, :, p].mean())
            fraction_p = float(np.sum(topk_idx == p)) / total_selected
            loss += P * mean_prob_p * fraction_p
        return np.array(loss, dtype=np.float32)


__all__ = ["Expert", "DensePart", "MoDBlock"]
