"""VerseAWM: JEPA 基础组件 (Task 4.1, 4.3, 4.4).

JEPA (Joint-Embedding Predictive Architecture) 由 LeCun 提出，核心思想：
- 在潜在空间 (latent space) 中预测，而非在像素空间重建
- 使用非对称设计：context encoder + target encoder (EMA) + predictor
- 防止表征坍塌三件套：
  1. target_encoder 输出 stop-gradient (detach)：target 不接收梯度
  2. EMA 更新 target encoder（不通过反向传播更新）
  3. cosine loss（防止 trivial solution，例如输出常数）

参考论文：
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeCun "A Path Towards Autonomous Machine Intelligence" (2022)
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, nn, no_grad
from verse_torch.nn import Module, Linear, LayerNorm, Dropout, ModuleList


# ---------------------------------------------------------------------------
# 工具：Multi-Head Attention（自注意力 / 交叉注意力）
# ---------------------------------------------------------------------------


class MultiHeadAttention(Module):
    """Multi-Head Attention (scaled dot-product).

    支持两种模式：
    - self-attention: 只传 x，q/k/v 都来自 x
    - cross-attention: 传 x 与 context，q 来自 x，k/v 来自 context

    Args:
        embed_dim: 模型维度 D
        n_heads: 头数 H，要求 D % H == 0
        dropout: attention weights dropout 概率（仅在 training=True 时生效）
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) 必须能被 n_heads ({n_heads}) 整除"
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        # 分离 Q/K/V 投影，便于 cross-attention
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out = Linear(embed_dim, embed_dim)
        self.dropout_p = float(dropout)

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        if context is None:
            context = x
        B, N, _ = x.shape
        M = context.shape[1]
        q = self.q_proj(x)            # (B, N, D)
        k = self.k_proj(context)      # (B, M, D)
        v = self.v_proj(context)      # (B, M, D)
        # reshape to (B, H, N, head_dim)
        q = q.reshape(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # attention scores: (B, H, N, M)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        attn = scores.softmax(dim=-1)
        # 训练时对 attention 权重做 dropout
        if self.training and self.dropout_p > 0.0:
            mask = (np.random.rand(*attn.shape) >= self.dropout_p).astype(np.float32)
            attn = attn * Tensor(mask, requires_grad=False) / (1.0 - self.dropout_p)
        out = attn @ v                # (B, H, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.embed_dim)
        return self.out(out)


class MLP(Module):
    """两层 MLP with GELU 激活。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, out_dim)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x).gelu()
        x = self.drop(x)
        return self.fc2(x)


class TransformerBlock(Module):
    """Pre-LN Transformer Block.

    Args:
        embed_dim: 模型维度
        n_heads: 注意力头数
        mlp_ratio: MLP 隐藏层相对于 embed_dim 的倍数
        dropout: dropout 概率
        cross_attn: 是否包含 cross-attention 子层（用于 predictor）
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, cross_attn: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        if cross_attn:
            self.norm_ctx = LayerNorm(embed_dim)
            self.norm_q = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)

    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        """forward.

        - 若 context is None：纯 self-attention block
        - 若 context is not None 且 self.cross_attn：cross-attention block
          （x 是 query，context 是 key/value）
        """
        if context is None:
            # self-attention (Pre-LN)
            x = x + self.attn(self.norm1(x))
        else:
            q = self.norm_q(x) if hasattr(self, "norm_q") else self.norm1(x)
            ctx = self.norm_ctx(context)
            x = x + self.attn(q, ctx)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Task 4.1: Context / Target / Predictor
# ---------------------------------------------------------------------------


class ContextEncoder(Module):
    """JEPA Context Encoder（ViT-style）。

    输入：context patches 序列 (B, N_ctx, D)，已经 patch embed + 加位置编码
    输出：context 表征 s_x (B, N_ctx, D)

    内部：若干层 self-attention TransformerBlock + 最终 LayerNorm
    """

    def __init__(self, embed_dim: int, depth: int = 6, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
             for _ in range(depth)]
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class TargetEncoder(Module):
    """JEPA Target Encoder，结构与 ContextEncoder 完全相同，但参数独立。

    作用：作为 EMA target，输出 detach 用于计算 loss。
    本类不接收梯度（参数由 EMA 更新函数维护），但 forward 内部不强制 no_grad，
    调用者应在调用时使用 `with no_grad():` 或对输出 detach。
    """

    def __init__(self, embed_dim: int, depth: int = 6, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
             for _ in range(depth)]
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Predictor(Module):
    """JEPA Predictor：小型 Transformer + cross-attention 到 context。

    输入：
      - target_queries: (B, N_tgt, D) 仅包含位置编码（无内容），作为 query
      - context: (B, N_ctx, D) 来自 context encoder 的 s_x
    输出：
      - s_y_hat: (B, N_tgt, D) 在 target 位置处的预测表征

    实现：
      1. 将 target_queries 与 context 拼接做 self-attention（让 queries 互相互动 + 看到 context）
      2. 再做 cross-attention：query=target_queries, context=s_x
      3. 输出前 N_tgt 个 token 作为预测

    简化版：将 target_queries 与 s_x 拼接 → self-attention blocks → 取前 N_tgt 个 token
    """

    def __init__(self, embed_dim: int, depth: int = 4, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
             for _ in range(depth)]
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, s_x: Tensor, target_queries: Tensor,
                z: Tensor = None) -> Tensor:
        """predictor forward.

        Args:
            s_x: (B, N_ctx, D) context 表征
            target_queries: (B, N_tgt, D) target 位置编码（无内容）
            z: (B, N_z, D) optional latent（暂未使用，保留接口）
        Returns:
            s_y_hat: (B, N_tgt, D) 预测的 target 表征
        """
        # 拼接：[target_queries, context]
        # 这样 target queries 可以 attend 到 context（自注意力覆盖整个序列）
        # 通过切片取回前 N_tgt 个 token
        N_tgt = target_queries.shape[1]
        if z is not None:
            x = _concat([target_queries, z, s_x], dim=1)
        else:
            x = _concat([target_queries, s_x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # 取前 N_tgt 个 token 作为预测
        # 用切片保持可微
        return x[:, :N_tgt]


def _concat(tensors, dim=0):
    """沿指定轴拼接 Tensor 列表（保持可微）。

    VerseTorch 没有提供 torch.cat 等价物，这里手写一个基于 numpy concatenate 的版本。
    """
    arrays = [t.data for t in tensors]
    out_data = np.concatenate(arrays, axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(
        out_data, requires_grad=requires_grad,
        _children=tuple(tensors) if requires_grad else (),
        _op="concat",
    )
    if requires_grad:
        # 记录每个输入在输出中沿 dim 的切片范围
        sizes = [t.shape[dim] for t in tensors]
        offsets = []
        s = 0
        for sz in sizes:
            offsets.append((s, s + sz))
            s += sz

        def _backward():
            grads = np.split(out.grad, [off for off, _ in offsets[1:]], axis=dim)
            for t, g in zip(tensors, grads):
                if t.requires_grad:
                    # 处理 split 返回的 ndarray 形状
                    g = np.asarray(g)
                    # 形状可能与 t 不同（如 dim 不匹配），用 reshape
                    if g.shape != t.shape:
                        g = g.reshape(t.shape)
                    t._accumulate_grad(g)

        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Task 4.1: JEPABase
# ---------------------------------------------------------------------------


class JEPABase(Module):
    """JEPA 基类：组合 context_encoder + target_encoder + predictor。

    forward 接口约定（子类实现）：
        forward(*inputs) -> (loss: Tensor scalar, metrics: dict)

    本类提供通用方法：
        - compute_loss(pred, target): 调用 jepa_loss
        - update_target(decay): EMA 更新 target_encoder

    防止坍塌关键点：
        - target_encoder 输出必须 detach
        - target_encoder 参数不参与 optimizer
        - EMA decay 从 0.99 逐步升到 0.9999
    """

    def __init__(self, context_encoder: ContextEncoder,
                 target_encoder: TargetEncoder,
                 predictor: Predictor):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor

    def forward(self, *args, **kwargs):
        raise NotImplementedError("子类需实现 forward")

    def update_target(self, decay: float = 0.996) -> None:
        """EMA 更新 target_encoder 参数：θ_t = decay * θ_t + (1-decay) * θ_c"""
        update_target_encoder(self.context_encoder, self.target_encoder, decay)


# ---------------------------------------------------------------------------
# Task 4.3: EMA Target Encoder
# ---------------------------------------------------------------------------


def update_target_encoder(context_encoder: Module, target_encoder: Module,
                          decay: float = 0.996) -> None:
    """EMA 更新 target_encoder 参数。

    原理：
        θ_target <- decay * θ_target + (1 - decay) * θ_context

    - target_encoder 不接收梯度，所以参数更新由本函数显式完成
    - decay 越大，target 变化越慢；训练初期 decay 较小（如 0.99）让 target 快速适应，
      后期 decay 较大（如 0.9999）让 target 稳定，防止坍塌
    - 必须在 no_grad 下操作，避免影响计算图

    Args:
        context_encoder: 提供"在线"参数的 encoder
        target_encoder: 接收 EMA 更新的 encoder
        decay: EMA 衰减系数 ∈ [0, 1)
    """
    with no_grad():
        ctx_params = list(context_encoder.parameters())
        tgt_params = list(target_encoder.parameters())
        if len(ctx_params) != len(tgt_params):
            raise ValueError(
                f"参数数量不匹配: context={len(ctx_params)}, target={len(tgt_params)}"
            )
        for cp, tp in zip(ctx_params, tgt_params):
            tp.data = (decay * tp.data + (1.0 - decay) * cp.data).astype(tp.data.dtype)


def ema_decay_schedule(step: int, total_steps: int,
                       start_decay: float = 0.99,
                       end_decay: float = 0.9999) -> float:
    """EMA decay 调度：从 start_decay 线性升到 end_decay。

    训练初期 decay 较小，让 target_encoder 快速跟上 context_encoder；
    后期 decay 接近 1，让 target 稳定，防止坍塌。
    """
    if total_steps <= 0:
        return end_decay
    t = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return start_decay + (end_decay - start_decay) * t


# ---------------------------------------------------------------------------
# Task 4.4: JEPA Loss（防坍塌）
# ---------------------------------------------------------------------------


def jepa_loss(pred: Tensor, target: Tensor, loss_type: str = "cosine",
              vicreg_lambda: float = 0.0) -> Tensor:
    """JEPA 损失函数（防坍塌版本）。

    Args:
        pred: (B, ..., D) 预测的表征（来自 predictor，可微）
        target: (B, ..., D) 真实 target 表征（来自 target_encoder，必须 detach）
        loss_type: 损失类型
            - "cosine": 1 - mean(cos_sim(pred, target))
              余弦损失对 magnitude 不敏感，鼓励方向对齐；
              即使 target 是常数，pred 也只能学到方向（被 stop-grad 阻止梯度），
              配合 EMA target + stop-grad 三件套防止坍塌
            - "l2": 0.5 * mean((pred - target)^2)
              标准回归损失，对 magnitude 敏感；可能更易坍塌，需配合 EMA
            - "vicreg": variance + covariance 正则化（可选，加上 cosine 主体）
        vicreg_lambda: vicreg 正则项权重（仅 loss_type="vicreg" 时使用）

    Returns:
        标量 Tensor
    """
    # 确保 target 是 detach 的（防止梯度回流到 target_encoder）
    # 注意：调用者应在 target_encoder forward 时用 no_grad；这里再做一次保险
    target = target.detach()

    if pred.shape != target.shape:
        raise ValueError(
            f"pred 与 target 形状不匹配: {pred.shape} vs {target.shape}"
        )

    if loss_type == "cosine":
        # 余弦相似度：cos_sim = (pred · target) / (||pred|| * ||target||)
        # 在最后维度 D 上求和，其他维度取均值
        eps = 1e-12
        # 计算点积与范数
        dot = (pred * target).sum(dim=-1)  # (B, ...)
        pred_norm = ((pred * pred).sum(dim=-1) + eps).sqrt()
        tgt_norm = ((target * target).sum(dim=-1) + eps).sqrt()
        cos_sim = dot / (pred_norm * tgt_norm)  # (B, ...)
        loss = (1.0 - cos_sim).mean()
        return loss

    if loss_type == "l2":
        diff = pred - target
        return (diff * diff).mean() * 0.5

    if loss_type == "vicreg":
        # VICReg: variance + covariance 正则化（防止坍塌 / 退化）
        # 主损失用 cosine，再加 variance 与 covariance 正则
        eps = 1e-12
        dot = (pred * target).sum(dim=-1)
        pred_norm = ((pred * pred).sum(dim=-1) + eps).sqrt()
        tgt_norm = ((target * target).sum(dim=-1) + eps).sqrt()
        cos_sim = dot / (pred_norm * tgt_norm)
        main_loss = (1.0 - cos_sim).mean()

        # Variance 正则：让 pred 沿每个维度有合理方差（>= 1）
        # 沿 batch 维计算方差
        # pred shape (B, ..., D)；把 batch 与中间维度 flatten
        B = pred.shape[0]
        flat = pred.reshape(B, -1)  # (B, D_total)
        # var 沿 dim=0
        var = ((flat - flat.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
        # 软阈值：max(0, 1 - sqrt(var))
        var_reg = ((1.0 - var.sqrt()).maximum(Tensor(0.0, requires_grad=False))).mean()

        # Covariance 正则：让不同维度去相关
        # cov = (flat - mean)^T @ (flat - mean) / B
        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.transpose(-1, -2) @ centered) / B
        # 取非对角线元素的 L2 范数的平方 / D
        D_total = flat.shape[1]
        # 对角线 mask
        eye = np.eye(D_total, dtype=np.float32)
        off_diag = cov * Tensor(1.0 - eye, requires_grad=False)
        cov_reg = (off_diag * off_diag).mean()

        return main_loss + vicreg_lambda * (var_reg + cov_reg)

    raise ValueError(f"Unknown loss_type: {loss_type}")


__all__ = [
    "MultiHeadAttention",
    "MLP",
    "TransformerBlock",
    "ContextEncoder",
    "TargetEncoder",
    "Predictor",
    "JEPABase",
    "update_target_encoder",
    "ema_decay_schedule",
    "jepa_loss",
]
