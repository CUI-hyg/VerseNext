"""VerseAWM: H-JEPA 层次化 JEPA (Task 4.7).

H-JEPA (Hierarchical JEPA) 由 LeCun 提出，关键思想：
- 在多个时间尺度上做潜在预测
- 高层抽象动作 (abstract action) 控制长期预测
- 低层具体动作 (concrete action) 控制短期预测

简化实现（2 层 hierarchy）：
- Level 1（短期）: predictor_short(s_x_t, a_short) -> s_y_{t+1}
  - 直接预测下一时刻的 latent
- Level 2（中期）: predictor_long(s_x_t, abstract_action) -> s_y_{t+K}
  - 跳跃 K 步预测（K 由 horizon_ratios 决定）

每层共享一个 encoder + 各自的 predictor，target_encoder 仍用 EMA。

参考: LeCun "A Path Towards Autonomous Machine Intelligence" (2022)
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
    update_target_encoder,
    jepa_loss,
)


class _MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers.append(Linear(d, hidden_dim))
            d = hidden_dim
        layers.append(Linear(d, out_dim))
        self.layers = nn.ModuleList(layers)
        self.n = len(layers)

    def forward(self, x):
        for i in range(self.n - 1):
            x = self.layers[i](x).gelu()
        return self.layers[-1](x)


class HJEPA(Module):
    """Hierarchical JEPA（简化版，2 层时间尺度）.

    结构：
    - 共享 context_encoder / target_encoder（EMA）
    - 两个 predictor：
      - predictor_short：预测 t+1
      - predictor_long：预测 t+K (K = horizon_ratios[1] / horizon_ratios[0])
    - 两个 abstract action encoder（可选；本实现简化为 learnable token）

    Args:
        obs_dim: 观测维度（已 flatten）
        embed_dim: latent 表征维度
        n_levels: 层次数（本实现固定为 2）
        horizon_ratios: 每层的时间尺度比例（如 (1, 8) 表示 short=1步, long=8步）
    """

    def __init__(self, obs_dim: int, embed_dim: int = 256,
                 n_levels: int = 2, horizon_ratios=(1, 8),
                 encoder_depth: int = 4, n_heads: int = 4,
                 predictor_depth: int = 3, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        assert n_levels == 2, "本实现仅支持 2 层 hierarchy"
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.horizon_ratios = tuple(horizon_ratios)
        # K：长期预测的跳跃步数
        self.K = horizon_ratios[1] // horizon_ratios[0]

        # 观测编码：obs_dim -> embed_dim（单 token 表征）
        self.obs_encoder = Linear(obs_dim, embed_dim)

        # 共享 encoder（处理 latent sequence）
        self.context_encoder = ContextEncoder(
            embed_dim, encoder_depth, n_heads, mlp_ratio, dropout,
        )
        self.target_encoder = TargetEncoder(
            embed_dim, encoder_depth, n_heads, mlp_ratio, dropout,
        )
        # 初始化 target = context
        self._copy_params(self.context_encoder, self.target_encoder)

        # 每层独立的 predictor
        self.predictor_short = Predictor(
            embed_dim, predictor_depth, n_heads, mlp_ratio, dropout,
        )
        self.predictor_long = Predictor(
            embed_dim, predictor_depth, n_heads, mlp_ratio, dropout,
        )

        # abstract action token（learnable，代表"高层动作"）
        # short 与 long 各一个
        self.action_token_short = Tensor.empty(1, 1, embed_dim, requires_grad=True)
        self.action_token_long = Tensor.empty(1, 1, embed_dim, requires_grad=True)
        with no_grad():
            self.action_token_short.data = np.random.randn(1, 1, embed_dim).astype(np.float32) * 0.02
            self.action_token_long.data = np.random.randn(1, 1, embed_dim).astype(np.float32) * 0.02

    @staticmethod
    def _copy_params(src: Module, dst: Module) -> None:
        with no_grad():
            src_params = list(src.parameters())
            dst_params = list(dst.parameters())
            assert len(src_params) == len(dst_params)
            for sp, dp in zip(src_params, dst_params):
                dp.data = sp.data.copy()

    def update_target(self, decay: float = 0.996) -> None:
        update_target_encoder(self.context_encoder, self.target_encoder, decay)

    def encode(self, x: Tensor) -> Tensor:
        """观测序列编码为 latent sequence.

        Args:
            x: (B, T, obs_dim) 或 (B, obs_dim)
        Returns:
            s: (B, T, embed_dim) 或 (B, embed_dim)
        """
        if x.ndim == 2:
            # (B, obs_dim) -> (B, 1, embed_dim) -> encoder -> (B, 1, embed_dim) -> (B, embed_dim)
            t = self.obs_encoder(x).unsqueeze(1)  # (B, 1, D)
            s = self.context_encoder(t).squeeze(1)
            return s
        # (B, T, obs_dim)
        t = self.obs_encoder(x)  # (B, T, D)
        s = self.context_encoder(t)
        return s

    def encode_target(self, x: Tensor) -> Tensor:
        """target_encoder 编码（no_grad + detach）."""
        with no_grad():
            if x.ndim == 2:
                t = self.obs_encoder(x).unsqueeze(1)
                s = self.target_encoder(t).squeeze(1)
            else:
                t = self.obs_encoder(x)
                s = self.target_encoder(t)
            return s.detach()

    def forward(self, x_short: Tensor, x_long: Tensor = None,
                loss_type: str = "cosine"):
        """前向 + 双层损失.

        Args:
            x_short: (B, T_short, obs_dim) 短期观测序列
                     假设 T_short >= 2，用 x[:, :-1] 预测 x[:, 1:]
            x_long: (B, T_long, obs_dim) 长期观测序列
                     用 x[:, 0] 预测 x[:, K]（K = self.K）
                     若 None，则从 x_short 提取
            loss_type: 损失类型

        Returns:
            loss: 标量 Tensor
            metrics: dict
        """
        B, T_short, _ = x_short.shape

        # ----- Level 1：短期预测（t -> t+1）-----
        # 编码全部时刻
        s_short = self.encode(x_short)  # (B, T, D)
        s_short_tgt = self.encode_target(x_short)  # (B, T, D)

        # 用第 0..T-2 时刻预测第 1..T-1 时刻
        # target query: 第 t+1 时刻的 obs 编码（含信息但作为 query 位置）
        # 简化：把第 t 时刻的 latent 作为 context，预测第 t+1 时刻
        # 为了用 Predictor（接受序列），把 s_x 视为单 token，target query 也是单 token
        s_x_short = s_short[:, :-1]  # (B, T-1, D) 作为 context
        s_y_short_tgt = s_short_tgt[:, 1:]  # (B, T-1, D) 真实 target

        # short action token 作为额外 query（broadcast 到 T-1 个位置）
        n_pred = T_short - 1
        act_short = self.action_token_short.expand((B, n_pred, self.embed_dim))
        # predictor 接受 (s_x, target_queries)
        # 这里 target_queries = action token（不含 obs content）
        s_y_short_pred = self.predictor_short(s_x_short, act_short)
        # 损失
        loss_short = jepa_loss(s_y_short_pred, s_y_short_tgt, loss_type=loss_type)

        # ----- Level 2：长期预测（t -> t+K）-----
        if x_long is None:
            # 从 x_short 提取长期对
            # 需要 T_short >= K + 1
            if T_short < self.K + 1:
                # 退化为 short
                loss_long = loss_short
                s_y_long_pred = s_y_short_pred
                s_y_long_tgt = s_y_short_tgt
            else:
                x_long_in = x_short[:, :T_short - self.K]  # (B, T-K, obs_dim)
                x_long_tgt = x_short[:, self.K:]  # (B, T-K, obs_dim)
                s_x_long = self.encode(x_long_in)  # (B, T-K, D)
                s_y_long_tgt = self.encode_target(x_long_tgt)  # (B, T-K, D)
                n_long = x_long_in.shape[1]
                act_long = self.action_token_long.expand((B, n_long, self.embed_dim))
                s_y_long_pred = self.predictor_long(s_x_long, act_long)
                loss_long = jepa_loss(s_y_long_pred, s_y_long_tgt, loss_type=loss_type)
        else:
            s_x_long = self.encode(x_long[:, :1])  # (B, 1, D) 仅第 0 帧
            # target：第 K 帧
            x_long_tgt = x_long[:, self.K:self.K + 1] if x_long.shape[1] > self.K else x_long[:, -1:]
            s_y_long_tgt = self.encode_target(x_long_tgt)
            act_long = self.action_token_long.expand((B, 1, self.embed_dim))
            s_y_long_pred = self.predictor_long(s_x_long, act_long)
            loss_long = jepa_loss(s_y_long_pred, s_y_long_tgt, loss_type=loss_type)

        # 总损失：两层加权
        loss = loss_short + loss_long

        metrics = {
            "loss": float(loss.data),
            "loss_short": float(loss_short.data),
            "loss_long": float(loss_long.data),
            "K": self.K,
        }
        return loss, metrics


__all__ = ["HJEPA"]
