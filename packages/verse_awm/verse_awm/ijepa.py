"""VerseAWM: I-JEPA 图像版 (Task 4.2).

I-JEPA (Image JEPA) 流程：
1. Patchify 图像为 (B, N, patch_dim) → 线性映射到 (B, N, D)
2. 加位置编码（learnable）
3. 生成 mask：
   - context mask: 1 个大块（覆盖 ~50% patches）—— 给 context encoder
   - target masks: 多个小块（每个 ~15-20% patches）—— 用于预测
4. context_encoder 处理 context patches → s_x (B, N_ctx, D)
5. target_encoder 处理所有 patches (no_grad) → s_y grid (B, N, D)
6. predictor 接收 (s_x, target_queries) → s_y_hat 在 target 位置 (B, N_tgt, D)
7. loss = Σ_block cosine(s_y_hat_block, s_y_block.detach())

关键点：
- target_encoder 输出 detach（stop-gradient）
- EMA 更新 target_encoder
- cosine loss + EMA + stop-grad 防止坍塌

参考: https://arxiv.org/abs/2301.08243
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, nn, no_grad
from verse_torch.nn import Module, Linear, LayerNorm

from .jepa import (
    ContextEncoder,
    TargetEncoder,
    Predictor,
    JEPABase,
    jepa_loss,
)


# ---------------------------------------------------------------------------
# Task 4.2: Patch Embedding
# ---------------------------------------------------------------------------


class PatchEmbed(Module):
    """Image → patch embeddings.

    输入: (B, C, H, W) 图像
    输出: (B, N, D) 其中 N = (H/patch_size) * (W/patch_size)

    实现：用 reshape + Linear（等价于卷积 stride=patch_size）。
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) 必须能被 patch_size ({patch_size}) 整除"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_patches_per_side = img_size // patch_size
        self.num_patches = self.n_patches_per_side ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        # 线性映射 patch_dim → embed_dim
        self.proj = Linear(self.patch_dim, embed_dim)

    def forward(self, images: Tensor) -> Tensor:
        """images: (B, C, H, W) → (B, N, D)"""
        B, C, H, W = images.shape
        p = self.patch_size
        # reshape: (B, C, H//p, p, W//p, p)
        # 调整轴顺序把 patch 内的维度合并
        # 目标: (B, H//p, W//p, C, p, p) -> (B, N, C*p*p)
        x = images.reshape(B, C, H // p, p, W // p, p)
        # permute 到 (B, H//p, W//p, C, p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # flatten patch 维度
        x = x.reshape(B, self.num_patches, self.patch_dim)
        # 线性投影
        return self.proj(x)


# ---------------------------------------------------------------------------
# Task 4.2: Mask 生成（context block + target blocks）
# ---------------------------------------------------------------------------


def random_masking(B: int, n_patches_per_side: int,
                   context_ratio: float = 0.5, target_ratio: float = 0.2,
                   n_targets: int = 4, rng: np.random.Generator = None):
    """生成 I-JEPA 风格的随机块 mask。

    在 patches grid (S, S) 上：
    - context: 1 个矩形大块（覆盖约 context_ratio * total）
    - targets: n_targets 个矩形小块（每个约 target_ratio * total）

    Args:
        B: batch size
        n_patches_per_side: grid 边长 S（即 patches 是 SxS 排列）
        context_ratio: context block 占总 patches 的目标比例
        target_ratio: 每个 target block 占总 patches 的目标比例
        n_targets: target block 数量
        rng: numpy Generator（可选，便于复现）

    Returns:
        context_mask: (B, N) bool, True 表示该 patch 属于 context（输入 context encoder）
        target_masks: list of (B, N) bool, 每个 target block 一个 mask
    """
    if rng is None:
        rng = np.random.default_rng()

    S = n_patches_per_side
    N = S * S
    context_mask = np.zeros((B, N), dtype=bool)
    target_masks = [np.zeros((B, N), dtype=bool) for _ in range(n_targets)]

    # 计算 block 边长（正方形，使面积近似等于 ratio * S^2）
    def _block_side(ratio: float) -> int:
        side = max(1, int(round(S * np.sqrt(ratio))))
        return min(side, S)

    c_side = _block_side(context_ratio)
    t_side = _block_side(target_ratio)

    for b in range(B):
        # Context block：随机位置
        cy = int(rng.integers(0, S - c_side + 1))
        cx = int(rng.integers(0, S - c_side + 1))
        for i in range(c_side):
            for j in range(c_side):
                context_mask[b, (cy + i) * S + (cx + j)] = True
        # Target blocks：每个独立随机位置
        # 允许与 context 重叠（不严格禁止，I-JEPA 原文允许）
        for t in range(n_targets):
            ty = int(rng.integers(0, S - t_side + 1))
            tx = int(rng.integers(0, S - t_side + 1))
            for i in range(t_side):
                for j in range(t_side):
                    target_masks[t][b, (ty + i) * S + (tx + j)] = True
    return context_mask, target_masks


# ---------------------------------------------------------------------------
# Task 4.2: IJEPA 主类
# ---------------------------------------------------------------------------


class IJEPA(JEPABase):
    """I-JEPA: Image JEPA.

    Args:
        img_size: 图像边长（正方形）
        patch_size: patch 边长
        in_channels: 通道数
        embed_dim: 模型维度 D
        depth: encoder 层数
        n_heads: 注意力头数
        predictor_depth: predictor 层数
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 192,
                 depth: int = 6, n_heads: int = 4,
                 predictor_depth: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        # 1. 先创建三件套与 patch_embed（Python 对象，不挂到 self 上）
        patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches_per_side = img_size // patch_size
        num_patches = n_patches_per_side ** 2
        pos_embed = Tensor.empty(1, num_patches, embed_dim, requires_grad=True)
        with no_grad():
            pos_embed.data = (np.random.randn(
                1, num_patches, embed_dim,
            ).astype(np.float32)) * 0.02

        ctx_enc = ContextEncoder(embed_dim, depth, n_heads, mlp_ratio, dropout)
        tgt_enc = TargetEncoder(embed_dim, depth, n_heads, mlp_ratio, dropout)
        # 用 context_encoder 的参数初始化 target_encoder（EMA 起点相同）
        self._copy_params(ctx_enc, tgt_enc)
        predictor = Predictor(embed_dim, predictor_depth, n_heads, mlp_ratio, dropout)

        # 2. 调用 JEPABase.__init__ 注册 context/target/predictor
        super().__init__(ctx_enc, tgt_enc, predictor)

        # 3. 现在可以挂其他子模块/参数（Module.__setattr__ 已经可用）
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_patches_per_side = n_patches_per_side
        self.num_patches = num_patches

    @staticmethod
    def _copy_params(src: Module, dst: Module) -> None:
        """把 src 参数复制到 dst（仅用于初始化 target_encoder = context_encoder）."""
        with no_grad():
            src_params = list(src.parameters())
            dst_params = list(dst.parameters())
            assert len(src_params) == len(dst_params), \
                f"参数数量不匹配: {len(src_params)} vs {len(dst_params)}"
            for sp, dp in zip(src_params, dst_params):
                dp.data = sp.data.copy()

    def random_masking(self, B: int, context_ratio: float = 0.5,
                       target_ratio: float = 0.2, n_targets: int = 4,
                       rng: np.random.Generator = None):
        """便捷封装：基于当前模型的 grid 大小生成 mask."""
        return random_masking(
            B, self.n_patches_per_side,
            context_ratio, target_ratio, n_targets, rng,
        )

    def forward(self, images: Tensor,
                context_ratio: float = 0.5,
                target_ratio: float = 0.2,
                n_targets: int = 4,
                loss_type: str = "cosine",
                rng: np.random.Generator = None):
        """前向 + 损失计算。

        Args:
            images: (B, C, H, W)
            context_ratio / target_ratio / n_targets: mask 策略参数
            loss_type: "cosine" / "l2" / "vicreg"
            rng: numpy 随机数生成器（可选）

        Returns:
            loss: 标量 Tensor
            metrics: dict，包含各 target block 的 loss 与全局 loss
        """
        B = images.shape[0]
        # 1. patchify + pos emb
        patches = self.patch_embed(images)  # (B, N, D)
        N = patches.shape[1]
        x = patches + self.pos_embed  # broadcast (1, N, D)

        # 2. mask
        context_mask, target_masks = self.random_masking(
            B, context_ratio, target_ratio, n_targets, rng,
        )
        # 至少要有 1 个 context 与 1 个 target patch
        # 若某些样本 context 为空（极端情况），fallback 到第 0 个 patch
        if not context_mask.any():
            context_mask[:, 0] = True

        # 3. context encoder 处理 context patches
        # 用布尔索引取 patch（按样本分别 gather）
        # 简化处理：每个样本的 context patch 数量可能不同，这里统一取 max
        # 实际 I-JEPA 每个 batch 内 context 大小相同（因为是 block mask）
        # 为简化实现，假设 batch 内 context 大小一致
        ctx_counts = context_mask.sum(axis=1)
        n_ctx = int(ctx_counts[0])
        # 构造 context patches tensor (B, n_ctx, D)
        # 由于不同样本可能 ctx 不同，这里用每个样本独立 gather 然后 pad 到 n_ctx
        # 简化：假设所有样本 n_ctx 相同
        ctx_patches_list = []
        for b in range(B):
            idx = np.where(context_mask[b])[0]
            if len(idx) < n_ctx:
                # pad（重复最后一个）
                pad = np.tile(idx[-1:] if len(idx) > 0 else [0], n_ctx - len(idx))
                idx = np.concatenate([idx, pad])
            ctx_patches_list.append(x[b:b + 1, idx])
        # 拼接成 (B, n_ctx, D)
        ctx_input = _batch_gather(x, context_mask, n_ctx)
        s_x = self.context_encoder(ctx_input)  # (B, n_ctx, D)

        # 4. target encoder 处理所有 patches（no_grad + detach）
        with no_grad():
            s_y_grid = self.target_encoder(x)  # (B, N, D)
            s_y_grid = s_y_grid.detach()

        # 5. predictor：对每个 target block 做预测
        # target queries 是 target 位置上的 pos_embed（不含 patch content）
        total_loss = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)
        per_block_losses = []
        n_actual_blocks = 0
        for t_idx, tmask in enumerate(target_masks):
            # 该 block 的 target patch 数（假设 batch 内一致）
            tgt_counts = tmask.sum(axis=1)
            n_tgt = int(tgt_counts[0])
            if n_tgt == 0:
                continue
            # 构造 target queries: 用 pos_embed 在 target 位置
            tgt_queries = _batch_gather(
                self.pos_embed.expand((B, N, self.embed_dim)),
                tmask, n_tgt,
            )
            # target 真实表征
            tgt_repr = _batch_gather(s_y_grid, tmask, n_tgt)  # (B, n_tgt, D)
            # predictor 输出
            s_y_hat = self.predictor(s_x, tgt_queries)  # (B, n_tgt, D)
            # 损失
            blk_loss = jepa_loss(s_y_hat, tgt_repr, loss_type=loss_type)
            per_block_losses.append(float(blk_loss.data))
            total_loss = total_loss + blk_loss
            n_actual_blocks += 1

        if n_actual_blocks == 0:
            # fallback：用所有 patches 作为单一 target
            tgt_queries = self.pos_embed.expand((B, N, self.embed_dim))
            tgt_repr = s_y_grid
            s_y_hat = self.predictor(s_x, tgt_queries)
            total_loss = jepa_loss(s_y_hat, tgt_repr, loss_type=loss_type)
            per_block_losses.append(float(total_loss.data))
            n_actual_blocks = 1

        total_loss = total_loss * (1.0 / n_actual_blocks)

        metrics = {
            "loss": float(total_loss.data),
            "per_block_losses": per_block_losses,
            "n_ctx": int(n_ctx),
            "n_targets": n_actual_blocks,
        }
        return total_loss, metrics

    def extract_features(self, images: Tensor) -> Tensor:
        """提取图像表征（用于线性探针 / 下游任务）.

        返回 (B, D) 的全局表征：对 context_encoder 输出的所有 patch 取平均。
        评估模式下使用：无 mask，所有 patches 都过 context_encoder。
        """
        was_training = self.training
        self.eval()
        try:
            patches = self.patch_embed(images)
            x = patches + self.pos_embed
            s_x = self.context_encoder(x)  # (B, N, D)
            # 平均池化得到全局表征
            feat = s_x.mean(dim=1)  # (B, D)
        finally:
            if was_training:
                self.train()
        return feat


def _batch_gather(x: Tensor, mask: np.ndarray, n_target: int) -> Tensor:
    """根据 bool mask 从 (B, N, D) 中 gather 出 (B, n_target, D).

    假设 batch 内每个样本 mask True 的数量一致（=n_target）。
    若不足，用第 0 个 patch pad；若超出，截断。
    """
    B, N, D = x.shape
    indices = np.zeros((B, n_target), dtype=np.int64)
    for b in range(B):
        idx = np.where(mask[b])[0]
        if len(idx) >= n_target:
            indices[b] = idx[:n_target]
        else:
            # 不足时用现有 idx 循环填充
            if len(idx) == 0:
                indices[b] = 0
            else:
                rep = int(np.ceil(n_target / len(idx)))
                padded = np.tile(idx, rep)[:n_target]
                indices[b] = padded

    # 用 advanced indexing 实现 gather
    # x shape (B, N, D)，indices (B, n_target)
    # 目标: out[b, t, d] = x[b, indices[b, t], d]
    arr = x.data
    out_data = np.take_along_axis(
        arr, indices[:, :, None], axis=1,
    )  # (B, n_target, D)

    requires_grad = x.requires_grad
    out = Tensor(
        out_data, requires_grad=requires_grad,
        _children=(x,) if requires_grad else (),
        _op="gather",
    )
    if requires_grad:
        def _backward():
            grad = np.zeros_like(arr, dtype=out.grad.dtype)
            np.add.at(grad, (np.arange(B)[:, None], indices), out.grad)
            x._accumulate_grad(grad)

        out._backward = _backward
    return out


__all__ = ["PatchEmbed", "IJEPA", "random_masking"]
