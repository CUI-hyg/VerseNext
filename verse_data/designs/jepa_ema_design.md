# 设计草稿：JEPA EMA + 防坍塌设计

> 关联源码：[`jepa.py`](file:///workspace/packages/verse_awm/verse_awm/jepa.py), [`ijepa.py`](file:///workspace/packages/verse_awm/verse_awm/ijepa.py), [`vjepa.py`](file:///workspace/packages/verse_awm/verse_awm/vjepa.py), [`hjepa.py`](file:///workspace/packages/verse_awm/verse_awm/hjepa.py), [`rssm.py`](file:///workspace/packages/verse_awm/verse_awm/rssm.py)
> 关联 ADR：[ADR-003 世界模型路线选型](../../docs/architecture/adr-003-world-model-route.md)

## 1. 背景与动机

JEPA（Joint-Embedding Predictive Architecture）由 Yann LeCun 提出（参考论文 [A Path Towards Autonomous Machine Intelligence, 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)），核心思想：

> 在 **latent 空间** 中预测，而非在像素/原始观测空间重建。

与生成式自监督（MAE、SimMask）和对比学习（SimCLR、MoCo）相比，JEPA 的优势：

1. **避免像素级重建的浪费**：预测下一帧的 latent 比预测像素更高效，专注于语义信息；
2. **避免对比学习的负样本工程**：不需要 large batch / memory bank / 难例挖掘；
3. **支持多模态扩展**：图像（I-JEPA）/ 视频（V-JEPA）/ 跨模态（A-JEPA）共用同一框架。

但 JEPA 面临一个致命问题：**表征坍塌（representation collapse）**。

### 1.1 什么是表征坍塌？

如果 predictor 直接学到一个常数函数 `predictor(s_x) = c`，且 target encoder 也输出常数 `c`，那么 cosine loss = 0，模型"完美"收敛，但学不到任何有用信息。

更广义的坍塌形式：
- **常数坍塌**：所有输入映射到同一个 latent；
- **维度坍塌**：latent 的部分维度恒为常数，有效维度 < 设计维度；
- **子空间坍塌**：latent 全部落在一个低维子空间内。

### 1.2 防坍塌三件套

Verse 在 `verse_awm` 中实现了三件套防坍塌：

1. **Stop-gradient**：target encoder 的输出在计算 loss 前 `detach`，梯度不回流；
2. **EMA target encoder**：target encoder 参数不通过 optimizer 更新，而是 context encoder 的 EMA；
3. **Cosine loss**：归一化后预测，对 magnitude 不敏感，避免 trivial 解。

这三件套必须 **同时存在**，缺一不可。下面分别详述。

## 2. Stop-gradient（stop-grad）

### 2.1 原理

如果 target encoder 的梯度回流，predictor 会通过反传让 target encoder 把所有输出拉到同一个点，瞬间坍塌。所以必须切断梯度。

[`jepa.py` 第 398-400 行](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L398-L400)：

```python
def jepa_loss(pred: Tensor, target: Tensor, loss_type="cosine", ...):
    # 确保 target 是 detach 的（防止梯度回流到 target_encoder）
    # 注意：调用者应在 target_encoder forward 时用 no_grad；这里再做一次保险
    target = target.detach()
```

并在 I-JEPA 的 forward 中也用 `no_grad` 包裹 target encoder 调用（[`ijepa.py` 第 274-276 行](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L274-L276)）：

```python
# 4. target encoder 处理所有 patches（no_grad + detach）
with no_grad():
    s_y_grid = self.target_encoder(x)  # (B, N, D)
    s_y_grid = s_y_grid.detach()
```

### 2.2 双重保险

为什么 `no_grad()` 与 `detach()` 都要？

- `no_grad()` 在上下文管理器层切断：target encoder 内部的所有算子都不构建计算图，节省内存与时间；
- `detach()` 在 Tensor 层切断：即使 target encoder 已经在 no_grad 下，输出的 Tensor 仍可能因为外部代码错误被加入计算图，detach 是最后一道防线。

实际工程中，单独用任何一个都不够安全。Verse 采用双重保险策略。

### 2.3 失败模式

如果 stop-grad 缺失：

```python
# 错误示例
s_y_grid = self.target_encoder(x)  # 没有 no_grad，没有 detach
loss = jepa_loss(s_y_hat, s_y_grid)  # jepa_loss 内部会 detach，但梯度已经回流到 target_encoder
loss.backward()
```

此时 target encoder 的参数会被 optimizer 更新（如果在 optimizer 中），predictor 会迅速学会"输出常数 = target encoder 输出"，loss 在几步内降到 0，但表征坍塌。

## 3. EMA target encoder

### 3.1 原理

target encoder 的参数 **不能** 通过 optimizer 更新（否则会被梯度下降拉到坍塌解）。但 target encoder 又必须跟随 context encoder 的演化（否则预测目标过时，predictor 学不到有用信息）。

EMA（Exponential Moving Average）是折中方案：

```
θ_target ← decay * θ_target + (1 - decay) * θ_context
```

- `decay` 接近 1：target 慢慢跟随 context，提供稳定目标；
- `decay` 接近 0：target 快速跟随 context，但目标不稳定，容易坍塌。

### 3.2 实现

[`jepa.py` 第 331-356 行](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L331-L356)：

```python
def update_target_encoder(context_encoder: Module, target_encoder: Module,
                          decay: float = 0.996) -> None:
    """EMA 更新 target_encoder 参数。
    
    原理：
        θ_target <- decay * θ_target + (1 - decay) * θ_context
    
    - target_encoder 不接收梯度，所以参数更新由本函数显式完成
    - decay 越大，target 变化越慢；训练初期 decay 较小（如 0.99）让 target 快速适应，
      后期 decay 较大（如 0.9999）让 target 稳定，防止坍塌
    - 必须在 no_grad 下操作，避免影响计算图
    """
    with no_grad():
        ctx_params = list(context_encoder.parameters())
        tgt_params = list(target_encoder.parameters())
        if len(ctx_params) != len(tgt_params):
            raise ValueError(f"参数数量不匹配: context={len(ctx_params)}, target={len(tgt_params)}")
        for cp, tp in zip(ctx_params, tgt_params):
            tp.data = (decay * tp.data + (1.0 - decay) * cp.data).astype(tp.data.dtype)
```

关键点：
- `with no_grad():` 包裹，避免影响计算图；
- 直接操作 `tp.data`（绕过 autograd）；
- dtype 强制对齐 `tp.data.dtype`，避免 float64 累积。

### 3.3 EMA decay 调度

[`ema_decay_schedule` 函数](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L359-L370)：

```python
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
```

调度曲线：
- `step=0`: `decay = 0.99`（target 每 step 更新 1%，快速适应）；
- `step=total/2`: `decay = 0.9949`（中等稳定）；
- `step=total-1`: `decay = 0.9999`（target 每 step 更新 0.01%，几乎冻结）。

这种 ramp 是 I-JEPA 论文（https://arxiv.org/abs/2301.08243）和 BYOL 都推荐的最佳实践。

### 3.4 在 I-JEPA 中的使用

[`jepa_demo.py` 第 113-114 行](file:///workspace/examples/jepa_demo.py#L113-L114)：

```python
decay = ema_decay_schedule(step, n_steps)
update_target_encoder(model.context_encoder, model.target_encoder, decay)
```

每个训练 step 后调用一次 EMA 更新。注意 `target_encoder.parameters()` **不在** optimizer 的 params 列表中：

```python
# jepa_demo.py 第 89-94 行
params = (
    list(model.context_encoder.parameters())
    + list(model.predictor.parameters())
    + [model.pos_embed]
    + list(model.patch_embed.parameters())
)
opt = optim.Adam(params, lr=lr)
# 注意：target_encoder.parameters() 不在 optimizer 中！
```

这是 JEPA 实现中最容易踩的坑：如果不小心把 target_encoder 加进了 optimizer，所有防坍塌机制都失效。

### 3.5 为什么不用 ModeSeeking / Sinkhorn 等其他防坍塌方法？

JEPA 防坍塌方法谱系：

| 方法                  | 机制                              | 优点                  | 缺点                    |
| --------------------- | --------------------------------- | --------------------- | ----------------------- |
| **stop-grad + EMA**   | target 不接收梯度，EMA 慢跟随     | 实现简单，效果好       | 对 EMA 调度敏感         |
| VICReg                | variance + covariance 正则         | 不需要 EMA            | 需要正则权重调参        |
| Sinkhorn-Knopp        | batch 内 representation 重新分配  | 严格防常数坍塌        | 计算开销大，对小 batch 不友好 |
| InfoNCE               | 对比学习，需要负样本              | 表达能力强            | 需要大 batch / memory bank |

Verse 选择 **stop-grad + EMA + cosine** 的组合，因为：
- 实现最简单，对教学友好；
- 在小 batch（16）下能稳定工作（demo 验证）；
- 不需要调参（cosine loss 无超参）。

VICReg 在 `jepa_loss(loss_type="vicreg")` 中也实现了，作为备选；但默认仍用 cosine。

## 4. Cosine loss

### 4.1 原理

`jepa_loss` 的 cosine 选项（[`jepa.py` 第 407-417 行](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L407-L417)）：

```python
if loss_type == "cosine":
    eps = 1e-12
    dot = (pred * target).sum(dim=-1)
    pred_norm = ((pred * pred).sum(dim=-1) + eps).sqrt()
    tgt_norm = ((target * target).sum(dim=-1) + eps).sqrt()
    cos_sim = dot / (pred_norm * tgt_norm)
    loss = (1.0 - cos_sim).mean()
    return loss
```

计算：
- `cos_sim = (pred · target) / (||pred|| * ||target||)`，范围 `[-1, 1]`；
- `loss = 1 - cos_sim`，范围 `[0, 2]`，越小越好。

### 4.2 为什么 cosine 防坍塌？

考虑 L2 loss `0.5 * mean((pred - target)^2)`：
- **trivial 解**：`pred = target = 0`，loss = 0；
- 但这个解无意义（表征全为零）。

考虑 cosine loss：
- 如果 `pred = 0`，`cos_sim = 0 / 0 = NaN`（实际用 eps 保护后接近 0），`loss = 1 - 0 = 1`，**不收敛**；
- 如果 `pred = c * target`（方向对齐，magnitude 不同），`cos_sim = 1`，`loss = 0`，**收敛**；
- 如果 `pred = c`（常数），但 target 不是 c，`cos_sim = (c · target) / (|c| * |target|)`，**只在 target 也是 c 时为 1**，否则不收敛。

所以 cosine loss 鼓励的是 **方向对齐**，而非 magnitude 匹配。结合 stop-grad（target 不动）和 EMA（target 慢动），predictor 必须学到 target 的方向，无法坍塌到常数。

### 4.3 与 L2 的对比

L2 loss 的失败模式：
```
predictor(s_x) = 0  for all s_x
target_encoder(x) = 0  (因为 predictor 的梯度反传让它变成 0)
loss = 0
```

Cosine loss 的成功模式：
```
predictor(s_x) ≠ 0 (因为 cos_sim = 0 时 loss = 1)
predictor 必须学到 target 的方向
```

### 4.4 VICReg 作为备选

[`jepa_loss(loss_type="vicreg")` 实现](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L423-L454) 在 cosine 主损失基础上加 variance + covariance 正则：

```python
# Variance 正则：让 pred 沿每个维度有合理方差（>= 1）
var = ((flat - flat.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
var_reg = ((1.0 - var.sqrt()).maximum(0.0)).mean()

# Covariance 正则：让不同维度去相关
centered = flat - flat.mean(dim=0, keepdim=True)
cov = (centered.transpose(-1, -2) @ centered) / B
eye = np.eye(D_total, dtype=np.float32)
off_diag = cov * Tensor(1.0 - eye, requires_grad=False)
cov_reg = (off_diag * off_diag).mean()

return main_loss + vicreg_lambda * (var_reg + cov_reg)
```

VICReg 不依赖 stop-grad（理论上），但实践上仍建议配合 EMA。Verse 默认用 cosine，VICReg 作为可选项。

## 5. I-JEPA / V-JEPA / H-JEPA 三种变体

### 5.1 I-JEPA（图像版）

[`ijepa.py` 第 150-321 行](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L150-L321)：

```
images (B, 3, H, W)
   │
   ▼
PatchEmbed → patches (B, N, D)
   │
   ▼
+ pos_embed
   │
   ├──→ ContextEncoder（仅处理 context mask 内的 patches）
   │       ↓
   │     s_x (B, N_ctx, D)
   │       │
   │       ▼
   │     Predictor (target_queries 是位置编码 only, cross-attention to s_x)
   │       │
   │       ▼
   │     s_y_hat (B, N_tgt, D)
   │
   └──→ TargetEncoder (EMA, 处理所有 patches, no_grad + detach)
           ↓
         s_y_grid (B, N, D)
           │
           ▼
         gather target 位置的 s_y
           │
           ▼
       cosine_loss(s_y_hat, s_y.detach())
```

特点：
- **mask 策略**：1 个 context block（覆盖 ~50% patches）+ 4 个 target blocks（每个 ~20% patches）；
- **target_queries 仅含位置信息**：predictor 学到"位置 → 表征"映射，不能"作弊"复制 patch content；
- **预测多个 target**：每个 target block 独立计算 loss，平均后回传。

### 5.2 V-JEPA（视频版）

[`vjepa.py`](file:///workspace/packages/verse_awm/verse_awm/vjepa.py) 扩展到时序：

```
video (B, C, T, H, W)
   │
   ▼
Spatiotemporal PatchEmbed → patches (B, N_t * N_s, D)
   │
   ▼
+ pos_embed（含时间维）
   │
   ▼
... 同 I-JEPA 但 mask 是 3D（time + space）
```

特点：
- **mask 策略**：context 是前几帧的全部 patches，target 是后几帧的随机 patches（temporal split）；
- **预测未来**：模型学到"过去 → 未来"的 latent 转移，是世界模型能力的早期形式。

### 5.3 H-JEPA（层次化版）

[`hjepa.py` 第 54-60 行](file:///workspace/packages/verse_awm/verse_awm/hjepa.py#L54-L60)：

```
共享 context_encoder + target_encoder (EMA)
   │
   ├──→ predictor_short(s_x_t, action_short) → s_y_{t+1}      # 短期预测（1 步）
   │
   └──→ predictor_long(s_x_t, abstract_action) → s_y_{t+K}    # 长期预测（K 步跳跃）
```

特点：
- **多时间尺度**：短期 predictor 学具体细节，长期 predictor 学抽象动作；
- **abstract action**：高层动作（如"向左走"）控制长期预测，与低层动作（如"每个肌肉怎么动"）解耦；
- **LeCun 路线**：H-JEPA 是 LeCun 提出的"通向自主机器智能"路径的核心组件。

### 5.4 三者对比

| 变体  | 输入      | mask 维度 | 预测目标       | 用途                  |
| ----- | --------- | --------- | -------------- | --------------------- |
| I-JEPA| 图像      | 2D (H×W)  | 同帧不同 patches | 图像表征学习          |
| V-JEPA| 视频      | 3D (T×H×W)| 未来帧 patches   | 视频表征 + 时序预测   |
| H-JEPA| 序列（任意）| 时间      | t+1 与 t+K       | 多时间尺度抽象        |

## 6. RSSM 的 categorical latent + Gumbel-softmax 对比

RSSM（Dreamer V3 路线）也面临"如何采样 latent 让训练可微"的问题，但解法完全不同。

### 6.1 RSSM 的 categorical latent

[`rssm.py` 第 88-130 行附近](file:///workspace/packages/verse_awm/verse_awm/rssm.py#L88-L130)：

```python
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
    """
```

设计要点：
- **32 classes × 32 dims = 1024 维** latent（Dreamer V3 配置）；
- 每组 32 类用 Gumbel-Softmax 采样得到 one-hot（前向）/ soft（反向）；
- 共 32 组 → 总 latent 维度 1024。

### 6.2 与 JEPA 的对比

| 维度            | JEPA                            | RSSM                            |
| --------------- | ------------------------------- | ------------------------------- |
| latent 类型     | 连续（cosine distance）          | 离散（categorical, one-hot）    |
| 采样方式        | 不采样（直接用 encoder 输出）    | Gumbel-Softmax straight-through |
| 防坍塌机制       | stop-grad + EMA + cosine         | KL(posterior ∥ prior) 限制       |
| 训练目标         | 预测 target encoder 输出         | 重建观测 + 预测下一 latent      |
| 是否生成式       | 否（latent 空间预测）             | 是（decoder 重建观测）          |
| 推理时是否采样   | 否（直接前向）                   | 是（prior 采样）                |
| 计算开销        | 低（仅 forward + cosine）         | 高（GRU + posterior + prior + decoder）|

### 6.3 何时选 JEPA，何时选 RSSM？

- **JEPA 适合**：表征学习，下游任务（分类、检测）的预训练；
- **RSSM 适合**：世界模型，需要"想象"未来观测做规划（Dreamer 系列强化学习）；
- **H-JEPA 适合**：长期规划，多时间尺度抽象（LeCun 路线）。

Verse 在 ADR-003 中决定 **JEPA + RSSM 双轨实现**，H-JEPA 作为长期路线，详见 [ADR-003](../../docs/architecture/adr-003-world-model-route.md)。

## 7. 工程实现注意事项

### 7.1 target_encoder 与 context_encoder 的初始化

[`ijepa.py` 第 198-207 行](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L198-L207)：

```python
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
```

初始化时 target = context（参数完全相同），随后 target 通过 EMA 慢慢演化。这避免了初始时 target 与 context 输出差异过大导致 loss 偏高。

### 7.2 Predictor 的输入设计

predictor 的输入是 `(target_queries, s_x)`：
- `target_queries`：仅位置编码（不含 patch content）；
- `s_x`：context encoder 输出。

[`Predictor.forward` 方法](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L225-L249)：

```python
def forward(self, s_x: Tensor, target_queries: Tensor, z: Tensor = None) -> Tensor:
    N_tgt = target_queries.shape[1]
    if z is not None:
        x = _concat([target_queries, z, s_x], dim=1)
    else:
        x = _concat([target_queries, s_x], dim=1)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    # 取前 N_tgt 个 token 作为预测
    return x[:, :N_tgt]
```

设计意图：
- **target_queries 不含 patch content**：避免 predictor "作弊"复制输入；
- **拼接后 self-attention**：target queries 可以 attend 到 context（整个拼接序列做 self-attn 等价于 cross-attn）；
- **切片取前 N_tgt**：predictor 输出 target 位置的预测。

### 7.3 EMA decay 的边界情况

- `decay = 0`：target = context（无 EMA，等价于直接复制，立刻坍塌）；
- `decay = 1`：target 永远不变（无法适应 context 的演化，loss 不下降）；
- `decay = 0.99`：target 每 step 更新 1%，初期快速跟上；
- `decay = 0.9999`：target 每 step 更新 0.01%，后期稳定。

调度从 0.99 → 0.9999 是经验上的最佳实践，避免初期 target 过慢跟不上 context，又避免后期 target 过快导致坍塌。

## 8. 已知限制

1. **VICReg 未默认启用**：`vicreg_lambda=0.0` 时 VICReg 正则不生效，仅作为可选项；实际训练中默认走 cosine。
2. **predictor 的 cross-attention 用拼接 + self-attention 模拟**：等价但计算开销略大（O((N_tgt + N_ctx)^2) vs O(N_tgt * N_ctx)）。
3. **target_encoder 不接收梯度，但仍占内存**：参数量是 context encoder 的 2 倍（两个 encoder 同时存在）；可考虑用 weight sharing 减半，但会破坏 EMA 语义。
4. **Gumbel-Softmax straight-through 的梯度偏差**：前向是 hard one-hot，反向是 soft，梯度估计有偏差；Dreamer V3 通过 KL loss 校正，但仍有训练不稳定风险。
5. **不支持多 GPU 训练**：所有 JEPA 变体在单 CPU 上训练；大规模训练需要后续 GPU 支持。
6. **H-JEPA 的 abstract action 是占位符**：当前实现中 `abstract_action` 是随机生成的，没有真正的"高层动作"语义；完整实现需要分层强化学习或自监督动作发现。
7. **未实现 stop-grad 的形式化验证**：理论上 stop-grad 必须在 target encoder 的所有反向路径上都生效；当前实现依赖 `no_grad()` 上下文管理器，如果用户在 forward 中混用 `enable_grad()` 可能破坏不变量。

## 9. 源码引用汇总

### JEPA 基础组件
- [`ContextEncoder` / `TargetEncoder` / `Predictor` 类](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L146-L249)：三件套定义；
- [`JEPABase` 基类](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L294-L323)：组合三件套，提供通用方法；
- [`update_target_encoder` 函数](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L331-L356)：EMA 更新逻辑；
- [`ema_decay_schedule` 函数](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L359-L370)：decay 调度；
- [`jepa_loss` 函数](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L378-L456)：cosine / L2 / VICReg 三种损失。

### I-JEPA 图像版
- [`PatchEmbed` 类](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L43-L82)：图像 → patch embedding；
- [`random_masking` 函数](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L90-L142)：context + target block 生成；
- [`IJEPA` 类](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L150-L340)：完整 I-JEPA 实现，含 forward / extract_features；
- [`_batch_gather` 函数](file:///workspace/packages/verse_awm/verse_awm/ijepa.py#L343-L385)：可微的 batched gather（用 `np.add.at` 实现反向 scatter）。

### H-JEPA 层次化
- [`HJEPA` 类](file:///workspace/packages/verse_awm/verse_awm/hjepa.py#L54-L100)：双时间尺度 predictor。

### RSSM（与 JEPA 对比）
- [`GRUCell` 类](file:///workspace/packages/verse_awm/verse_awm/rssm.py#L42-L80)：verse_torch 没有原生 GRU，这里手写；
- [`gumbel_softmax` 函数](file:///workspace/packages/verse_awm/verse_awm/rssm.py#L88-L130)：straight-through Gumbel-Softmax。
