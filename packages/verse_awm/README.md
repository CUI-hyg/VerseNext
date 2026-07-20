# VerseAWM

> 中文定位：**自主世界模型包**（Autonomous World Model），在 VerseTorch 之上提供非生成式 JEPA 系列（I-JEPA / V-JEPA / H-JEPA）与生成式 RSSM（Dreamer 风格）实现，统一使用 EMA target encoder + energy-based loss 防止表征坍塌。

[返回主 README](../../README.md)

## 特性

- **非生成式 JEPA 系列**：在潜在空间预测，不在像素空间重建
  - `IJEPA`：图像 JEPA（参考 [I-JEPA, arXiv:2301.08243](https://arxiv.org/abs/2301.08243)）
  - `VJEPA`：视频 JEPA（参考 [V-JEPA 2, arXiv:2506.09985](https://arxiv.org/abs/2506.09985)）
  - `HJEPA`：层次化 JEPA，多时间尺度 + 抽象动作（参考 LeCun 2022 路线图）
- **生成式 RSSM**：Dreamer V3 风格递归状态空间模型
  - `RSSM` / `VideoRSSM`：deterministic (GRU) + stochastic (categorical) 双状态
  - Gumbel-Softmax straight-through 可微采样
  - KL balance + free bits 防 posterior 坍塌
- **通用 JEPA 基础组件**（`jepa.py`）：
  - `ContextEncoder` / `TargetEncoder`：ViT 风格 Transformer 编码器
  - `Predictor`：基于 cross-attention 的小型预测器
  - `update_target_encoder`：EMA 更新
  - `ema_decay_schedule`：从 0.99 线性升到 0.9999 的 decay 调度
  - `jepa_loss`：cosine / l2 / vicreg 三种损失
- **防坍塌三件套**：stop-gradient + EMA target + cosine loss
- **零重型依赖**：仅依赖 `verse_torch`（NumPy + Python 标准库）

## 安装

```bash
pip install -e packages/verse_torch
pip install -e packages/verse_awm
```

`verse_awm` 运行时**不依赖** PyTorch / TensorFlow / JAX，全部计算经由 `verse_torch.Tensor`。

## 模块导出

`verse_awm/__init__.py` 暴露以下符号：

```python
from verse_awm import (
    # JEPA 基础（jepa.py）
    JEPABase, ContextEncoder, TargetEncoder, Predictor,
    MultiHeadAttention, MLP, TransformerBlock,
    update_target_encoder, ema_decay_schedule, jepa_loss,
    # I-JEPA（ijepa.py）
    IJEPA, PatchEmbed, random_masking,
    # V-JEPA（vjepa.py）
    VJEPA, SpatioTemporalPatchEmbed, video_random_masking,
    # RSSM（rssm.py）
    RSSM, VideoRSSM, GRUCell, gumbel_softmax, categorical_kl,
    # H-JEPA（hjepa.py）
    HJEPA,
)
```

## 模块详解

### I-JEPA（图像 JEPA）`ijepa.py`

非生成式图像自监督预训练。流程：

1. `PatchEmbed` 将 `(B, C, H, W)` 切成 patch 并线性映射到 `(B, N, D)`，加可学习位置编码
2. `random_masking` 在 patch grid 上随机生成 1 个 context 大块（覆盖 ~50%）和 `n_targets` 个 target 小块（每个 ~20%）
3. `ContextEncoder` 处理 context patches → `s_x`
4. `TargetEncoder` 在 `no_grad` 下处理全部 patches → `s_y_grid`，输出 detach
5. `Predictor(s_x, target_queries)` → `s_y_hat`
6. `jepa_loss(s_y_hat, s_y.detach(), loss_type="cosine")` 对每个 target block 求损失后取平均

`IJEPA.extract_features(images)` 提供线性探针接口：返回 `(B, D)` 全局表征（patch 平均池化）。

构造参数：

```python
IJEPA(
    img_size=32, patch_size=4, in_channels=3,
    embed_dim=192, depth=6, n_heads=4,
    predictor_depth=4, mlp_ratio=4.0, dropout=0.0,
)
```

### V-JEPA（视频 JEPA）`vjepa.py`

视频版 JEPA，输入 `(B, C, T, H, W)`。关键差异：

- `SpatioTemporalPatchEmbed`：扩展为时空 tubelet（`tubelet_t × patch_size × patch_size`）
- `video_random_masking` 提供两种 mask 模式：
  - `"tube"`：时间管——同一空间区域跨所有时间步（适合预测物体持续运动）
  - `"block"`：时空矩形——连续 patches 跨时间子段（适合预测物体空间位移）
- 位置编码沿用 I-JEPA 的 learnable pos_emb（按 patch 数量自动构造）

```python
VJEPA(
    video_size=(16, 32, 32), tubelet_t=2, patch_size=4, in_channels=3,
    embed_dim=192, depth=6, n_heads=4, predictor_depth=4,
    mlp_ratio=4.0, dropout=0.0,
)
```

### H-JEPA（层次化 JEPA）`hjepa.py`

LeCun 层次化 JEPA 简化版（2 层时间尺度）：

- **Level 1（短期）**：`predictor_short(s_x_t, action_short)` 预测 `t+1` 的 latent
- **Level 2（长期）**：`predictor_long(s_x_t, action_long)` 跳跃 `K = horizon_ratios[1] // horizon_ratios[0]` 步预测

共享 `context_encoder` 与 `target_encoder`（EMA），抽象动作用 learnable token 表征。

```python
HJEPA(
    obs_dim=64, embed_dim=256, n_levels=2, horizon_ratios=(1, 8),
    encoder_depth=4, n_heads=4, predictor_depth=3,
    mlp_ratio=4.0, dropout=0.0,
)
# forward: x_short (B, T, obs_dim) → loss, metrics
```

### RSSM（递归状态空间模型）`rssm.py`

Dreamer V3 风格世界模型：

- 状态分解：
  - `h_t`（deterministic，GRU 递归更新）
  - `z_t`（stochastic，categorical，32 组 × 32 类）
- 训练：`posterior(obs, h) → z`（从真实观测推断），`prior(h) → z_hat`（从 h 预测）
- 推理：用 `prior` roll-out 未来轨迹（无需观测）
- 损失：`recon_loss + kl_loss`（KL 含 free bits 防坍塌）
- `VideoRSSM` 是 RSSM 的视频适配：`obs_dim = H * W * C`，提供 `frames_to_obs` / `obs_to_frames` / `forward_frames` / `rollout_frames`

```python
RSSM(
    obs_dim=4096, action_dim=0,
    deter_dim=512, stoch_dim=32, stoch_classes=32,
    hidden_dim=512, gru_layers=1,
)
# forward: observations (B, T, obs_dim) → dict（含 reconstructions / kl_loss / loss）
# rollout: observations (B, T_ctx, obs_dim), n_predict → predictions (B, n_predict, obs_dim)
```

底层工具：`GRUCell`（自实现 GRU）、`gumbel_softmax`（straight-through 采样）、`categorical_kl`（KL 散度）。

### JEPA 基础组件 `jepa.py`

- `ContextEncoder` / `TargetEncoder`：均为 ViT 风格（N 层 TransformerBlock + LayerNorm），结构相同但参数独立
- `Predictor`：将 `[target_queries, s_x]` 拼接做 self-attention，取前 `N_tgt` 个 token 作为预测
- `MultiHeadAttention`：支持 self / cross 两种模式，dropout 仅在 training=True 时生效
- `TransformerBlock`：Pre-LN 设计，可选 cross-attention 子层
- `JEPABase`：组合三件套的基类，子类实现 `forward → (loss, metrics)`，并提供 `update_target(decay)` EMA 接口

#### EMA 与防坍塌

```python
from verse_awm import update_target_encoder, ema_decay_schedule

# 训练循环中：
for step in range(n_steps):
    loss, metrics = model(batch)
    loss.backward(); opt.step()
    decay = ema_decay_schedule(step, n_steps, start_decay=0.99, end_decay=0.9999)
    update_target_encoder(model.context_encoder, model.target_encoder, decay)
```

`ema_decay_schedule` 从 `start_decay`（默认 0.99）线性升到 `end_decay`（默认 0.9999）：训练初期 target 快速适应，后期保持稳定防止坍塌。

#### 损失函数 `jepa_loss`

支持三种 loss_type：

| 类型 | 公式 | 适用场景 |
|---|---|---|
| `"cosine"`（默认） | `1 - mean(cos_sim(pred, target))` | 推荐主线，对 magnitude 不敏感，配合 EMA 三件套 |
| `"l2"` | `0.5 * mean((pred - target)²)` | magnitude 敏感，更易坍塌 |
| `"vicreg"` | cosine + variance + covariance 正则 | 显式防退化 |

`target` 在函数内部会再做一次 `detach()`，但调用者仍应在 `target_encoder` forward 时使用 `with no_grad():` 以避免构建多余计算图。

## 快速开始

完整示例见 [`examples/jepa_demo.py`](../../examples/jepa_demo.py)（8×8 色块网格上的 I-JEPA 训练）：

```python
import numpy as np
from verse_torch import Tensor, optim
from verse_awm import IJEPA, update_target_encoder, ema_decay_schedule

# 1. 构建极小模型（图像 8x8，patch 4x4）
model = IJEPA(
    img_size=8, patch_size=4, in_channels=3,
    embed_dim=32, depth=2, n_heads=2, predictor_depth=2,
)
# target_encoder 不接收梯度，仅 context_encoder / predictor / pos_embed / patch_embed 参与优化
params = (
    list(model.context_encoder.parameters())
    + list(model.predictor.parameters())
    + [model.pos_embed]
    + list(model.patch_embed.parameters())
)
opt = optim.Adam(params, lr=1e-3)

# 2. 训练循环
images = Tensor(images_np.astype(np.float32), requires_grad=False)
for step in range(n_steps):
    opt.zero_grad()
    loss, metrics = model(images, n_targets=4, rng=rng)
    loss.backward()
    opt.step()
    # EMA 更新 target_encoder
    decay = ema_decay_schedule(step, n_steps)
    update_target_encoder(model.context_encoder, model.target_encoder, decay)

# 3. 提取表征用于下游任务
feat = model.extract_features(images)  # (B, D)
```

实测：50 步训练后 cosine loss 显著下降，详见 [`verse_data/experiments/jepa_demo/loss_log.json`](../../verse_data/experiments/jepa_demo/loss_log.json)。

## 测试

| 测试文件 | 覆盖内容 |
|---|---|
| [`tests/test_ijepa_cifar10.py`](../../tests/test_ijepa_cifar10.py) | I-JEPA 在 CIFAR-10 子集上的自监督预训练 + 表征质量验证 |
| [`tests/test_rssm_moving_mnist.py`](../../tests/test_rssm_moving_mnist.py) | VideoRSSM 在 Moving MNIST 上的视频预测 + rollout 评估 |

运行：

```bash
PYTHONPATH=packages/verse_torch:packages/verse_awm pytest tests/test_ijepa_cifar10.py tests/test_rssm_moving_mnist.py
```

## 设计原理

- **为什么用 JEPA 而非 MAE？** JEPA 在 latent 空间预测，避免像素重建带来的高频细节负担，更利于学到高层语义；cosine loss + EMA + stop-grad 三件套是防止 representation collapse 的标准组合。
- **为什么同时提供 RSSM？** JEPA 是非生成式路线（LeCun），RSSM 是生成式路线（Hafner Dreamer V3）。两者各有适用场景，本包统一实现以便对比与混合。
- **为什么自实现 GRU？** VerseTorch 暂未提供 RNN 算子，RSSM 需要 deterministic 递归路径，故在 `rssm.py` 内部用 `Linear` 组合 3 个门（z / r / h_hat）手写 GRU。

## 相关文档

- [ADR-003 世界模型路线](../../docs/architecture/adr-003-world-model-route.md) —— JEPA + RSSM 双主线决策
- [JEPA EMA 设计](../../verse_data/designs/jepa_ema_design.md) —— EMA decay 调度与防坍塌论证
- [JEPA demo 说明](../../examples/README_jepa_demo.md) —— 8×8 色块网格 demo 的运行方式
- [VerseTorch README](../verse_torch/README.md) —— Tensor / nn / autograd 基础
- [主 README](../../README.md)
