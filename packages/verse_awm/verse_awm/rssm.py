"""VerseAWM: RSSM 循环状态空间模型 (Task 4.6).

RSSM (Recurrent State-Space Model) 由 Dreamer V3 提出，是世界模型的核心组件。

状态分解：
- recurrent state h_t (deterministic)：由 GRU 递归更新
- posterior z_t (stochastic, categorical)：训练时从观测推断 q(z|x_t, h_t)
- prior z_hat_t (stochastic, categorical)：推理时从 h_t 直接预测 q(z|h_t)

Dynamics：
    h_t = GRU(cat(z_{t-1}, a_{t-1}), h_{t-1})
    z_t ~ q(z | x_t, h_t)         # posterior (training)
    z_hat_t ~ q(z | h_t)           # prior (inference)
    x_hat_t = decoder(cat(h_t, flatten(z_t)))

Loss:
    - reconstruction loss: decoder 重构观测
    - KL loss: KL(q_posterior || q_prior) — 平衡 posterior 与 prior
    - 可选 reward loss

Categorical latent (Dreamer V3 风格)：
- 32 classes × 32 dims = 1024 维 one-hot / softmax 样本
- Gumbel-softmax straight-through 用于可微采样
- KL 用 categorical KL divergence

参考: Dreamer V3 https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor, nn, no_grad
from verse_torch.nn import Module, Linear, LayerNorm


# ---------------------------------------------------------------------------
# GRU Cell (verse_torch 没有，这里手写)
# ---------------------------------------------------------------------------


class GRUCell(Module):
    """GRU 单元（简化版）。

    更新公式：
        z_t = sigmoid(W_z x_t + U_z h_{t-1})
        r_t = sigmoid(W_r x_t + U_r h_{t-1})
        h_hat_t = tanh(W_h x_t + U_h (r_t * h_{t-1}))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_hat_t

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏状态维度
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 合并 z, r, h_hat 三个门，用一个 Linear 处理 input 与 hidden
        self.x2h = Linear(input_dim, 3 * hidden_dim)
        self.h2h = Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """x: (B, input_dim), h_prev: (B, hidden_dim) -> h_t (B, hidden_dim)"""
        H = self.hidden_dim
        x_proj = self.x2h(x)        # (B, 3H)
        h_proj = self.h2h(h_prev)   # (B, 3H)
        # 切片
        x_z = x_proj[..., :H]
        x_r = x_proj[..., H:2 * H]
        x_h = x_proj[..., 2 * H:]
        h_z = h_proj[..., :H]
        h_r = h_proj[..., H:2 * H]
        h_h = h_proj[..., 2 * H:]
        z = (x_z + h_z).sigmoid()
        r = (x_r + h_r).sigmoid()
        h_hat = (x_h + r * h_h).tanh()
        h_t = (1.0 - z) * h_prev + z * h_hat
        return h_t


# ---------------------------------------------------------------------------
# Categorical latent 采样（Gumbel-softmax straight-through）
# ---------------------------------------------------------------------------


def gumbel_softmax(logits: Tensor, tau: float = 1.0, hard: bool = True,
                   rng: np.random.Generator = None) -> Tensor:
    """Gumbel-Softmax 采样（可微的 categorical 采样）.

    原理：
    - 标准 Gumbel-Max: argmax(log_softmax(p) + gumbels) 不可微
    - Gumbel-Softmax: softmax((logits + gumbels) / tau) 可微，但输出是 soft one-hot
    - Straight-through: 前向用 hard one-hot，反向用 soft 梯度
      实现：y_hard + (y_soft - y_soft.detach())
      前向 = y_hard + 0 = y_hard
      反向 = 0 + dy_soft - 0 = dy_soft（梯度从 y_soft 流过）

    Args:
        logits: (..., n_classes) 未归一化的 logits
        tau: 温度，越低越接近 argmax
        hard: 是否使用 straight-through one-hot
    Returns:
        采样结果，hard=True 时为 one-hot，hard=False 时为 soft 概率
    """
    if rng is None:
        rng = np.random.default_rng()
    # 采样 Gumbel 噪声: g = -log(-log(u)), u ~ Uniform(0, 1)
    # 数值稳定：clip u 到 [eps, 1-eps] 避免 log(0) 或 log(1)=0
    eps = 1e-6
    u = rng.uniform(eps, 1.0 - eps, size=logits.shape).astype(np.float32)
    gumbels_np = -np.log(-np.log(u))
    gumbels = Tensor(gumbels_np, requires_grad=False)
    # 加噪声 + softmax
    y = (logits + gumbels) / tau
    y_soft = y.softmax(dim=-1)
    if not hard:
        return y_soft
    # straight-through one-hot
    idx = np.argmax(y_soft.data, axis=-1)
    n_classes = logits.shape[-1]
    y_hard_np = np.zeros_like(y_soft.data)
    # 构建 one-hot
    idx_flat = idx.reshape(-1)
    y_hard_flat = y_hard_np.reshape(-1, n_classes)
    y_hard_flat[np.arange(len(idx_flat)), idx_flat] = 1.0
    y_hard_np = y_hard_flat.reshape(y_soft.shape)
    y_hard = Tensor(y_hard_np, requires_grad=False)
    # y_hard - y_soft.detach() + y_soft：
    #   forward: y_hard - y_soft_data + y_soft_data = y_hard
    #   backward: 0 - 0 + dy_soft = dy_soft
    return y_hard + y_soft - y_soft.detach()


def categorical_kl(logits_q: Tensor, logits_p: Tensor) -> Tensor:
    """两个 categorical 分布的 KL 散度 KL(q || p).

    KL(q||p) = sum_i q_i * (log q_i - log p_i)
            = sum_i q_i * (log_softmax(logits_q)_i - log_softmax(logits_p)_i)

    在 Dreamer V3 中使用 "balanced KL"：先分别计算 KL(q||p) 与 KL(p||q)，
    再取最大值或加权平均。这里实现标准 KL(q||p)。

    Args:
        logits_q: (..., C) posterior logits
        logits_p: (..., C) prior logits
    Returns:
        (...,) 每个 batch 的 KL（最后保留 C 维求和后）
    """
    log_q = logits_q.log_softmax(dim=-1)
    log_p = logits_p.log_softmax(dim=-1)
    # q = exp(log_q)
    q = log_q.exp()
    # KL = sum(q * (log_q - log_p))
    kl = (q * (log_q - log_p)).sum(dim=-1)
    return kl


# ---------------------------------------------------------------------------
# RSSM 主类
# ---------------------------------------------------------------------------


class RSSM(Module):
    """Recurrent State-Space Model (Dreamer V3 style).

    状态分解：
    - h_t (deterministic, GRU): (B, deter_dim)
    - z_t (stochastic, categorical): (B, stoch_dim, stoch_classes) one-hot/softmax

    Args:
        obs_dim: 观测维度（ flattened ）
        action_dim: 动作维度（无控制信号时为 0）
        deter_dim: 确定性状态维度 h_t
        stoch_dim: 随机状态分组数
        stoch_classes: 每组的类别数
        hidden_dim: MLP 隐藏维度
        gru_layers: GRU 层数（这里简化为 1）
    """

    def __init__(self, obs_dim: int, action_dim: int = 0,
                 deter_dim: int = 512, stoch_dim: int = 32,
                 stoch_classes: int = 32, hidden_dim: int = 512,
                 gru_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        # 随机状态总维度 (flatten 后)
        self.stoch_total = stoch_dim * stoch_classes

        # GRU cell：输入 = cat(z_{t-1}, a_{t-1})，输出 = h_t
        gru_input_dim = self.stoch_total + action_dim
        self.gru = GRUCell(gru_input_dim, deter_dim)

        # posterior encoder: q(z | x_t, h_t) -> logits over (stoch_dim, stoch_classes)
        # 输入维度 = obs_dim + deter_dim
        self.posterior_net = _MLP(
            obs_dim + deter_dim, hidden_dim, self.stoch_total, n_layers=2,
        )

        # prior encoder: q(z | h_t) -> logits over (stoch_dim, stoch_classes)
        self.prior_net = _MLP(
            deter_dim, hidden_dim, self.stoch_total, n_layers=2,
        )

        # decoder: cat(h_t, flatten(z_t)) -> obs_hat
        self.decoder = _MLP(
            deter_dim + self.stoch_total, hidden_dim, obs_dim, n_layers=2,
        )

        # 可选 reward head（保留接口，本实现未启用训练 loss）
        # self.reward_head = _MLP(deter_dim + self.stoch_total, hidden_dim, 1, n_layers=2)

        # 默认 KL 平衡系数（Dreamer V3: 0.8）
        self.kl_balance = 0.8
        # 自由比特（free bits）—— KL 不会降到 0 以下太多
        # 设置较小值，让 KL 在初期也能贡献梯度
        self.kl_free = 0.2

    # ----- 单步前向 -----

    def step_recurrent(self, z_prev: Tensor, h_prev: Tensor,
                       a_prev: Tensor = None) -> Tensor:
        """GRU 递归步：h_t = GRU(cat(z_{t-1}, a_{t-1}), h_{t-1}).

        Args:
            z_prev: (B, stoch_dim, stoch_classes) 上一时刻的随机状态
            h_prev: (B, deter_dim) 上一时刻的确定性状态
            a_prev: (B, action_dim) 上一时刻的动作；若 action_dim=0 可为 None
        Returns:
            h_t: (B, deter_dim)
        """
        B = z_prev.shape[0]
        # flatten z_prev
        z_flat = z_prev.reshape(B, -1)  # (B, stoch_total)
        if a_prev is not None and self.action_dim > 0:
            gru_in = _concat([z_flat, a_prev], dim=-1)
        else:
            gru_in = z_flat
        h_t = self.gru(gru_in, h_prev)
        return h_t

    def posterior(self, obs: Tensor, h: Tensor):
        """后验：q(z | obs, h) -> logits, sample.

        Args:
            obs: (B, obs_dim)
            h: (B, deter_dim)
        Returns:
            z_sample: (B, stoch_dim, stoch_classes) one-hot (straight-through)
            posterior_logits: (B, stoch_dim, stoch_classes)
            posterior_probs: (B, stoch_dim, stoch_classes)
        """
        B = obs.shape[0]
        # MLP 输出 logits
        inp = _concat([obs, h], dim=-1)
        logits_flat = self.posterior_net(inp)  # (B, stoch_total)
        # reshape 到 (B, stoch_dim, stoch_classes)
        logits = logits_flat.reshape(B, self.stoch_dim, self.stoch_classes)
        # Gumbel-softmax straight-through 采样
        z_sample = gumbel_softmax(logits, tau=1.0, hard=True)
        # 概率（用于 KL 计算）
        probs = logits.softmax(dim=-1)
        return z_sample, logits, probs

    def prior(self, h: Tensor):
        """先验：q(z | h) -> logits, sample.

        Args:
            h: (B, deter_dim)
        Returns:
            z_sample, prior_logits, prior_probs
        """
        B = h.shape[0]
        logits_flat = self.prior_net(h)  # (B, stoch_total)
        logits = logits_flat.reshape(B, self.stoch_dim, self.stoch_classes)
        z_sample = gumbel_softmax(logits, tau=1.0, hard=True)
        probs = logits.softmax(dim=-1)
        return z_sample, logits, probs

    def reconstruct(self, h: Tensor, z: Tensor) -> Tensor:
        """解码器：cat(h, flatten(z)) -> obs_hat.

        Args:
            h: (B, deter_dim)
            z: (B, stoch_dim, stoch_classes)
        Returns:
            obs_hat: (B, obs_dim)
        """
        B = h.shape[0]
        z_flat = z.reshape(B, -1)
        inp = _concat([h, z_flat], dim=-1)
        return self.decoder(inp)

    # ----- 序列前向 -----

    def forward(self, observations: Tensor, actions: Tensor = None,
                initial_h: Tensor = None, initial_z: Tensor = None):
        """序列前向 + 损失计算。

        Args:
            observations: (B, T, obs_dim) 观测序列
            actions: (B, T, action_dim) 或 None（无控制信号）
            initial_h: (B, deter_dim) 初始 h_0；默认全 0
            initial_z: (B, stoch_dim, stoch_classes) 初始 z_0；默认全 0 one-hot

        Returns:
            dict 包含：
                - reconstructions: (B, T, obs_dim)
                - posterior_samples: list of (B, stoch_dim, stoch_classes)
                - prior_samples: list
                - posterior_logits, prior_logits
                - recurrent_states: list of (B, deter_dim)
                - kl_loss: 标量 Tensor
                - recon_loss: 标量 Tensor
                - loss: 总损失（recon + kl）
        """
        B, T, _ = observations.shape
        device_dtype = observations.dtype

        # 初始状态
        if initial_h is None:
            h = Tensor(np.zeros((B, self.deter_dim), dtype=np.float32),
                       requires_grad=False)
        else:
            h = initial_h
        if initial_z is None:
            # one-hot at class 0 for each group
            z_np = np.zeros((B, self.stoch_dim, self.stoch_classes), dtype=np.float32)
            z_np[:, :, 0] = 1.0
            z = Tensor(z_np, requires_grad=False)
        else:
            z = initial_z

        posterior_samples = []
        prior_samples = []
        posterior_logits_list = []
        prior_logits_list = []
        recurrent_states = []
        reconstructions = []

        total_kl = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)
        total_recon = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)

        for t in range(T):
            # 取当前 obs 与 action
            obs_t = observations[:, t]  # (B, obs_dim)
            a_prev = None
            if actions is not None and self.action_dim > 0:
                # 在 t=0 时用零动作
                if t == 0:
                    a_prev = Tensor(
                        np.zeros((B, self.action_dim), dtype=np.float32),
                        requires_grad=False,
                    )
                else:
                    a_prev = actions[:, t - 1]
            # 1. GRU 递归
            h = self.step_recurrent(z, h, a_prev)
            recurrent_states.append(h)

            # 2. posterior（从真实 obs 推断 z）
            z_post, post_logits, post_probs = self.posterior(obs_t, h)
            # 3. prior（从 h 预测 z_hat）
            z_prior, prior_logits, prior_probs = self.prior(h)

            # 4. 用 posterior 样本做下一步递归与解码（训练时）
            z = z_post

            # 5. 解码重构
            obs_hat = self.reconstruct(h, z)
            reconstructions.append(obs_hat)

            # 6. 计算损失
            # recon loss: MSE
            diff = obs_hat - obs_t
            recon_loss_t = (diff * diff).mean()
            total_recon = total_recon + recon_loss_t

            # KL loss: KL(post || prior)
            kl_t = categorical_kl(post_logits, prior_logits)
            kl_t_mean = kl_t.mean()
            # Dreamer V3 "balanced KL"：用 0.5 * (KL(post||prior) + KL(prior||post)) 或者
            # 加 free bits 防止 KL 过小（posterior 坍塌到 prior）
            # 这里简化：直接用 KL(post||prior)，并加 free bits
            kl_t_clipped = (kl_t_mean - self.kl_free).maximum(
                Tensor(0.0, requires_grad=False)
            ) + self.kl_free
            total_kl = total_kl + kl_t_clipped

            posterior_samples.append(z_post)
            prior_samples.append(z_prior)
            posterior_logits_list.append(post_logits)
            prior_logits_list.append(prior_logits)

        # 平均
        recon_loss = total_recon * (1.0 / T)
        kl_loss = total_kl * (1.0 / T)
        # 总损失
        loss = recon_loss + kl_loss

        # 把 list 堆叠成 (B, T, ...) Tensor
        reconstructions_t = _stack(reconstructions, dim=1)

        return {
            "reconstructions": reconstructions_t,
            "posterior_samples": posterior_samples,
            "prior_samples": prior_samples,
            "posterior_logits": posterior_logits_list,
            "prior_logits": prior_logits_list,
            "recurrent_states": recurrent_states,
            "kl_loss": kl_loss,
            "recon_loss": recon_loss,
            "loss": loss,
        }

    # ----- 推理：用 prior roll-out 未来轨迹 -----

    def rollout(self, observations: Tensor, n_predict: int,
                actions: Tensor = None):
        """给定上下文观测，用 prior 预测未来 n_predict 步。

        1. 在上下文上跑 posterior 模式（用真实观测推断 z）
        2. 从最后一个时间步开始，用 prior 模式 roll-out 未来

        Args:
            observations: (B, T_ctx, obs_dim) 上下文
            n_predict: 预测步数
            actions: (B, T_ctx + n_predict, action_dim) 或 None
        Returns:
            predictions: (B, n_predict, obs_dim) 预测的观测
            final_state: (h, z) 用于继续 rollout
        """
        B, T_ctx, _ = observations.shape
        # 先跑上下文，得到最终的 h, z
        out = self.forward(observations, actions[:, :T_ctx] if actions is not None else None)
        h = out["recurrent_states"][-1]
        z = out["posterior_samples"][-1]

        preds = []
        for t in range(n_predict):
            a_prev = None
            if actions is not None and self.action_dim > 0:
                a_prev = actions[:, T_ctx + t - 1] if t > 0 else \
                    actions[:, T_ctx - 1]
            # 递归
            h = self.step_recurrent(z, h, a_prev)
            # prior
            z, _, _ = self.prior(h)
            # decode
            obs_hat = self.reconstruct(h, z)
            preds.append(obs_hat)
        predictions = _stack(preds, dim=1)
        return predictions, (h, z)


# ---------------------------------------------------------------------------
# VideoRSSM：适配视频预测
# ---------------------------------------------------------------------------


class VideoRSSM(RSSM):
    """RSSM 适配视频预测（Moving MNIST 风格）.

    - obs_dim = H * W * C (帧展平)
    - action_dim = 0 (无控制信号，纯自预测)
    - 提供帧打包/解包辅助方法
    """

    def __init__(self, frame_size=(64, 64), in_channels: int = 1,
                 deter_dim: int = 256, stoch_dim: int = 32,
                 stoch_classes: int = 32, hidden_dim: int = 256,
                 gru_layers: int = 1):
        H, W = frame_size
        obs_dim = H * W * in_channels
        # 先调用 RSSM.__init__ 完成 Module 初始化与子模块注册
        super().__init__(
            obs_dim=obs_dim, action_dim=0,
            deter_dim=deter_dim, stoch_dim=stoch_dim,
            stoch_classes=stoch_classes, hidden_dim=hidden_dim,
            gru_layers=gru_layers,
        )
        # 之后才能设置非参数属性
        self.frame_size = frame_size
        self.in_channels = in_channels

    def frames_to_obs(self, frames: Tensor) -> Tensor:
        """(B, T, C, H, W) -> (B, T, obs_dim)"""
        B, T, C, H, W = frames.shape
        return frames.reshape(B, T, C * H * W)

    def obs_to_frames(self, obs: Tensor) -> Tensor:
        """(B, T, obs_dim) -> (B, T, C, H, W)"""
        B, T, _ = obs.shape
        C = self.in_channels
        H, W = self.frame_size
        return obs.reshape(B, T, C, H, W)

    def forward_frames(self, frames: Tensor):
        """frames: (B, T, C, H, W) -> dict (含原始 obs 与重构 frames)"""
        obs = self.frames_to_obs(frames)
        out = self.forward(obs, actions=None)
        # 重构 frames
        recon_obs = out["reconstructions"]  # (B, T, obs_dim)
        out["recon_frames"] = self.obs_to_frames(recon_obs)
        return out

    def rollout_frames(self, frames_ctx: Tensor, n_predict: int):
        """给定上下文 frames，预测未来 n_predict 帧.

        Args:
            frames_ctx: (B, T_ctx, C, H, W)
            n_predict: 预测帧数
        Returns:
            pred_frames: (B, n_predict, C, H, W)
        """
        obs_ctx = self.frames_to_obs(frames_ctx)
        preds, _ = self.rollout(obs_ctx, n_predict, actions=None)
        return self.obs_to_frames(preds)


# ---------------------------------------------------------------------------
# 辅助：MLP 与 Tensor 操作
# ---------------------------------------------------------------------------


class _MLP(Module):
    """简单 MLP with GELU."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2):
        super().__init__()
        layers = []
        d_in = in_dim
        for i in range(n_layers - 1):
            layers.append(Linear(d_in, hidden_dim))
            d_in = hidden_dim
        layers.append(Linear(d_in, out_dim))
        self.layers = nn.ModuleList(layers)
        self.n_layers = len(layers)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.n_layers - 1):
            x = self.layers[i](x).gelu()
        return self.layers[-1](x)


def _concat(tensors, dim=0):
    """沿指定轴拼接 Tensor 列表（保持可微）。"""
    arrays = [t.data for t in tensors]
    out_data = np.concatenate(arrays, axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(
        out_data, requires_grad=requires_grad,
        _children=tuple(tensors) if requires_grad else (),
        _op="concat",
    )
    if requires_grad:
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
                    g = np.asarray(g)
                    if g.shape != t.shape:
                        g = g.reshape(t.shape)
                    t._accumulate_grad(g)

        out._backward = _backward
    return out


def _stack(tensors, dim=0):
    """沿新轴堆叠 Tensor 列表（保持可微）。

    等价于 torch.stack。实现：先 unsqueeze 再 concat。
    """
    # 每个 tensor 在 dim 处插入一维
    new_dim = dim if dim >= 0 else dim + 1
    unsqueezed = [t.unsqueeze(new_dim) for t in tensors]
    return _concat(unsqueezed, dim=new_dim)


__all__ = [
    "RSSM",
    "VideoRSSM",
    "GRUCell",
    "gumbel_softmax",
    "categorical_kl",
]
