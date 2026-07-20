"""VerseAWM: V-JEPA 视频版 (Task 4.5).

V-JEPA (Video JEPA) 流程类似 I-JEPA，但输入是视频 (B, C, T, H, W)。
关键差异：
- Patchify 扩展为时空 tubelet：tubelet_t (时间) × patch_size (空间) × patch_size (空间)
- Mask 策略：
  - Temporal tube masking: 整个时间管（同一空间位置跨时间）—— 适合预测物体持续运动
  - Spatial block masking: 空间块（连续 patches 跨时间）—— 适合预测物体在空间上移动
- 位置编码：分离时间维与空间维，或合并为 3D 位置

参考: https://arxiv.org/abs/2506.09985 (V-JEPA 2)
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
# Task 4.5: SpatioTemporal Patch Embedding
# ---------------------------------------------------------------------------


class SpatioTemporalPatchEmbed(Module):
    """Video → spatiotemporal patch embeddings.

    输入: (B, C, T, H, W)
    输出: (B, N, D)，N = T' * H' * W'
        T' = T // tubelet_t
        H' = H // patch_size
        W' = W // patch_size

    tubelet_t: 时间维度的 patch 大小（每 tubelet_t 帧合并为一个 token）
    """

    def __init__(self, video_size=(16, 32, 32), tubelet_t: int = 2,
                 patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        T, H, W = video_size
        if T % tubelet_t != 0:
            raise ValueError(f"T ({T}) 必须能被 tubelet_t ({tubelet_t}) 整除")
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"H, W ({H}, {W}) 必须能被 patch_size ({patch_size}) 整除"
            )
        self.video_size = video_size
        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.T_patch = T // tubelet_t
        self.H_patch = H // patch_size
        self.W_patch = W // patch_size
        self.num_patches = self.T_patch * self.H_patch * self.W_patch
        self.patch_dim = in_channels * tubelet_t * patch_size * patch_size
        self.proj = Linear(self.patch_dim, embed_dim)

    def forward(self, videos: Tensor) -> Tensor:
        """videos: (B, C, T, H, W) → (B, N, D)"""
        B, C, T, H, W = videos.shape
        tt = self.tubelet_t
        p = self.patch_size
        # reshape: (B, C, T//tt, tt, H//p, p, W//p, p)
        x = videos.reshape(B, C, T // tt, tt, H // p, p, W // p, p)
        # permute 到 (B, T', H', W', C, tt, p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        # flatten 到 (B, N, patch_dim)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Task 4.5: Video Mask 生成
# ---------------------------------------------------------------------------


def video_random_masking(B: int, T_patch: int, H_patch: int, W_patch: int,
                         context_ratio: float = 0.5, target_ratio: float = 0.2,
                         n_targets: int = 4, mode: str = "tube",
                         rng: np.random.Generator = None):
    """视频 mask 生成。

    Args:
        B: batch size
        T_patch, H_patch, W_patch: 时空 patch grid 维度
        context_ratio / target_ratio / n_targets: mask 策略参数
        mode: "tube" 或 "block"
            - "tube": 时间管 mask—— 选一个空间区域，跨所有时间
            - "block": 空间块 mask—— 选一个时空矩形
        rng: numpy 随机数生成器

    Returns:
        context_mask: (B, N) bool
        target_masks: list of (B, N) bool
    """
    if rng is None:
        rng = np.random.default_rng()

    T, Hp, Wp = T_patch, H_patch, W_patch
    N = T * Hp * Wp
    context_mask = np.zeros((B, N), dtype=bool)
    target_masks = [np.zeros((B, N), dtype=bool) for _ in range(n_targets)]

    def _to_idx(t, h, w):
        return t * (Hp * Wp) + h * Wp + w

    def _block_dims(ratio: float):
        """根据 ratio 返回 (t_side, h_side, w_side) 使体积近似等于 ratio * N."""
        # 简化：t_side 用 cube_root(ratio)*T，h/w_side 用 sqrt(ratio)*H/W
        # 但更简单：让空间维度按 sqrt，时间维度按 ratio 调整
        h_side = max(1, int(round(Hp * np.sqrt(ratio))))
        w_side = max(1, int(round(Wp * np.sqrt(ratio))))
        h_side = min(h_side, Hp)
        w_side = min(w_side, Wp)
        return h_side, w_side

    c_h, c_w = _block_dims(context_ratio)
    t_h, t_w = _block_dims(target_ratio)

    for b in range(B):
        if mode == "tube":
            # tube mask: 选空间区域，跨所有时间
            cy = int(rng.integers(0, Hp - c_h + 1))
            cx = int(rng.integers(0, Wp - c_w + 1))
            for t in range(T):
                for i in range(c_h):
                    for j in range(c_w):
                        context_mask[b, _to_idx(t, cy + i, cx + j)] = True
            # target tubes
            for tidx in range(n_targets):
                ty = int(rng.integers(0, Hp - t_h + 1))
                tx = int(rng.integers(0, Wp - t_w + 1))
                for t in range(T):
                    for i in range(t_h):
                        for j in range(t_w):
                            target_masks[tidx][b, _to_idx(t, ty + i, tx + j)] = True
        else:
            # block mask: 时空矩形
            t_c = max(1, T // 2)
            ct0 = int(rng.integers(0, T - t_c + 1))
            cy = int(rng.integers(0, Hp - c_h + 1))
            cx = int(rng.integers(0, Wp - c_w + 1))
            for t in range(t_c):
                for i in range(c_h):
                    for j in range(c_w):
                        context_mask[b, _to_idx(ct0 + t, cy + i, cx + j)] = True
            for tidx in range(n_targets):
                t_t = max(1, T // 4)
                tt0 = int(rng.integers(0, T - t_t + 1))
                ty = int(rng.integers(0, Hp - t_h + 1))
                tx = int(rng.integers(0, Wp - t_w + 1))
                for t in range(t_t):
                    for i in range(t_h):
                        for j in range(t_w):
                            target_masks[tidx][b, _to_idx(tt0 + t, ty + i, tx + j)] = True
    return context_mask, target_masks


# ---------------------------------------------------------------------------
# Task 4.5: VJEPA 主类
# ---------------------------------------------------------------------------


class VJEPA(JEPABase):
    """V-JEPA: Video JEPA.

    Args:
        video_size: (T, H, W) 视频尺寸
        tubelet_t: 时间 patch 大小
        patch_size: 空间 patch 大小
        in_channels: 通道数
        embed_dim: 模型维度
        depth: encoder 层数
        n_heads: 注意力头数
        predictor_depth: predictor 层数
    """

    def __init__(self, video_size=(16, 32, 32), tubelet_t: int = 2,
                 patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192,
                 depth: int = 6, n_heads: int = 4, predictor_depth: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        # 1. 创建子组件
        patch_embed = SpatioTemporalPatchEmbed(
            video_size, tubelet_t, patch_size, in_channels, embed_dim,
        )
        T_patch = patch_embed.T_patch
        H_patch = patch_embed.H_patch
        W_patch = patch_embed.W_patch
        num_patches = patch_embed.num_patches
        pos_embed = Tensor.empty(1, num_patches, embed_dim, requires_grad=True)
        with no_grad():
            pos_embed.data = (np.random.randn(
                1, num_patches, embed_dim,
            ).astype(np.float32)) * 0.02

        ctx_enc = ContextEncoder(embed_dim, depth, n_heads, mlp_ratio, dropout)
        tgt_enc = TargetEncoder(embed_dim, depth, n_heads, mlp_ratio, dropout)
        self._copy_params(ctx_enc, tgt_enc)
        predictor = Predictor(embed_dim, predictor_depth, n_heads, mlp_ratio, dropout)

        # 2. 调用 JEPABase.__init__ 注册三件套
        super().__init__(ctx_enc, tgt_enc, predictor)

        # 3. 挂其他子模块/属性
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.video_size = video_size
        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.T_patch = T_patch
        self.H_patch = H_patch
        self.W_patch = W_patch
        self.num_patches = num_patches

    @staticmethod
    def _copy_params(src: Module, dst: Module) -> None:
        with no_grad():
            src_params = list(src.parameters())
            dst_params = list(dst.parameters())
            assert len(src_params) == len(dst_params)
            for sp, dp in zip(src_params, dst_params):
                dp.data = sp.data.copy()

    def random_masking(self, B: int, context_ratio: float = 0.5,
                       target_ratio: float = 0.2, n_targets: int = 4,
                       mode: str = "tube", rng: np.random.Generator = None):
        return video_random_masking(
            B, self.T_patch, self.H_patch, self.W_patch,
            context_ratio, target_ratio, n_targets, mode, rng,
        )

    def forward(self, videos: Tensor,
                context_ratio: float = 0.5,
                target_ratio: float = 0.2,
                n_targets: int = 4,
                mode: str = "tube",
                loss_type: str = "cosine",
                rng: np.random.Generator = None):
        """前向 + 损失计算。

        Args:
            videos: (B, C, T, H, W)
            mode: "tube" 时间管 / "block" 时空块
        Returns:
            loss, metrics
        """
        B = videos.shape[0]
        patches = self.patch_embed(videos)
        N = patches.shape[1]
        x = patches + self.pos_embed

        context_mask, target_masks = self.random_masking(
            B, context_ratio, target_ratio, n_targets, mode, rng,
        )
        if not context_mask.any():
            context_mask[:, 0] = True

        ctx_counts = context_mask.sum(axis=1)
        n_ctx = int(ctx_counts[0])

        # gather context patches
        from .ijepa import _batch_gather
        ctx_input = _batch_gather(x, context_mask, n_ctx)
        s_x = self.context_encoder(ctx_input)

        with no_grad():
            s_y_grid = self.target_encoder(x).detach()

        total_loss = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)
        per_block_losses = []
        n_actual_blocks = 0
        for tmask in target_masks:
            tgt_counts = tmask.sum(axis=1)
            n_tgt = int(tgt_counts[0])
            if n_tgt == 0:
                continue
            tgt_queries = _batch_gather(
                self.pos_embed.expand((B, N, self.embed_dim)),
                tmask, n_tgt,
            )
            tgt_repr = _batch_gather(s_y_grid, tmask, n_tgt)
            s_y_hat = self.predictor(s_x, tgt_queries)
            blk_loss = jepa_loss(s_y_hat, tgt_repr, loss_type=loss_type)
            per_block_losses.append(float(blk_loss.data))
            total_loss = total_loss + blk_loss
            n_actual_blocks += 1

        if n_actual_blocks == 0:
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


__all__ = ["VJEPA", "SpatioTemporalPatchEmbed", "video_random_masking"]
