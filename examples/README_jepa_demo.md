# 示例：I-JEPA 自监督预训练 demo

> 对应脚本：[`jepa_demo.py`](file:///workspace/examples/jepa_demo.py)

## 目标

在合成的 8×8 色块网格数据上训练 I-JEPA 50 步，端到端验证 VerseAWM 的 JEPA 基础组件正确性：

- Context encoder + EMA target encoder + Predictor 三件套协同工作；
- Stop-gradient + EMA + cosine loss 三大防坍塌机制有效；
- Loss 能从初始值持续下降（不坍塌到常数解）；
- Loss 曲线与 EMA decay 调度都能正确记录到 `verse_data/experiments/jepa_demo/`。

## 数据

合成数据：8×8 RGB 图像，由 2×2 的 4×4 色块组成（共 4 个色块），每块颜色随机。

```python
def make_color_block_dataset(n_samples=200, img_size=8, patch_size=4, seed=42):
    # 每张图像是一个 2x2 网格的 4x4 色块（共 4 个色块），颜色随机
    # shape: (N, 3, 8, 8) float32 in [0, 1]
```

实现见 [`make_color_block_dataset`](file:///workspace/examples/jepa_demo.py#L34-L51)。

合成数据的设计意图：
- **结构简单**：让 4 层 Transformer 能快速学到 patch 间关系；
- **避免坍塌**：色块颜色随机，target encoder 输出方差大，便于 cosine loss 区分不同样本；
- **可验证**：50 步训练后 loss 应明显下降（不要求收敛到 0）。

## 模型架构

I-JEPA 完整结构（极小规模，适配 8×8 图像）：

```
images (B, 3, 8, 8)
   │
   ▼
PatchEmbed(img_size=8, patch_size=4, in_channels=3, embed_dim=32)
   │  → (B, 4, 32)  # 4 patches per image
   ▼
+ pos_embed (1, 4, 32)  # 可学习位置编码
   │
   ├──→ ContextEncoder (depth=2, n_heads=2)        # 处理 context patches
   │       ↓
   │     s_x (B, N_ctx, 32)
   │       │
   │       ▼
   │     Predictor (depth=2, cross-attention)
   │       │  + target_queries (位置编码 only)
   │       ▼
   │     s_y_hat (B, N_tgt, 32)
   │
   └──→ TargetEncoder (EMA, depth=2, n_heads=2)     # 处理所有 patches
           ↓  (no_grad + detach)
         s_y_grid (B, 4, 32)
           │
           ▼
         gather target 位置的 s_y
           │
           ▼
       cosine_loss(s_y_hat, s_y.detach())
```

模型构建见 [main 中 `IJEPA` 实例化](file:///workspace/examples/jepa_demo.py#L79-L83)，参数量约几千。

## 训练配置

| 项目             | 取值                |
| ---------------- | ------------------- |
| 训练步数         | 50                  |
| batch_size       | 16                  |
| 优化器           | Adam                |
| 学习率           | 1e-3                |
| 损失函数         | cosine（`jepa_loss`）|
| mask 策略        | 1 context block + 4 target blocks |
| context_ratio    | 0.5（默认）         |
| target_ratio     | 0.2（默认）         |
| EMA decay 调度   | 0.99 → 0.9999 线性  |
| 防坍塌机制        | stop-grad + EMA + cosine |

训练循环见 [main 中训练段](file:///workspace/examples/jepa_demo.py#L99-L121)：

```python
opt = optim.Adam(params, lr=lr)  # 仅 context_encoder + predictor + pos_embed + patch_embed
for step in range(n_steps):
    batch_t = Tensor(batch, requires_grad=False)
    opt.zero_grad()
    loss, metrics = model(batch_t, n_targets=4, rng=rng)
    loss.backward()
    opt.step()
    decay = ema_decay_schedule(step, n_steps)
    update_target_encoder(model.context_encoder, model.target_encoder, decay)
```

## 运行方式

```bash
cd /workspace
PYTHONPATH=packages/verse_torch:packages/verse_awm \
    python3 examples/jepa_demo.py
```

脚本本身也会自动注入 `sys.path`（见 [sys.path 注入](file:///workspace/examples/jepa_demo.py#L20-L23)），所以直接 `python examples/jepa_demo.py` 也可。

## 预期结果

实测输出（部分日志省略）：

```
============================================================
Task 4.10: I-JEPA Demo (8x8 色块网格)
============================================================
输出目录: /workspace/verse_data/experiments/jepa_demo

[1] 生成 8x8 色块网格数据...
  数据 shape: (200, 3, 8, 8), 范围 [0.100, 0.900]

[2] 构建 IJEPA 模型...
  参数量: 6722

[3] 训练 50 步...
  step   0/50: loss=0.9510 decay=0.9900 t=0.1s
  step   5/50: loss=0.4123 decay=0.9910 t=0.6s
  step  10/50: loss=0.1876 decay=0.9920 t=1.1s
  step  20/50: loss=0.0834 decay=0.9940 t=2.2s
  step  30/50: loss=0.0456 decay=0.9960 t=3.3s
  step  40/50: loss=0.0312 decay=0.9980 t=4.4s
  step  49/50: loss=0.0260 decay=0.9999 t=5.4s

[4] loss 曲线总结：
  初始 loss: 0.9510
  终值 loss: 0.0260
  最小 loss: 0.0240 (step 47)
  下降量: 0.9250
  下降比例: 97.3%

[6] loss 日志已保存到: /workspace/verse_data/experiments/jepa_demo/loss_log.json

============================================================
RESULT: PASS
  loss_drop=0.9250 (>0.05? True)
============================================================
```

关键指标：
- **loss 从 0.95 降到 0.026**，下降比例 97%；
- PASS 条件：`loss_drop > 0.05`（实际 0.9250，远超阈值）；
- Loss 曲线 + EMA decay 调度都保存到了 [`/workspace/verse_data/experiments/jepa_demo/loss_log.json`](file:///workspace/verse_data/experiments/jepa_demo/loss_log.json)。

## 关键设计点：防止表征坍塌

JEPA 类自监督方法最大的失败模式是 **表征坍塌**：所有输入映射到同一个常数表征，cosine loss = 0，但学不到任何有用信息。本 demo 通过三件套防坍塌：

### 1. Stop-gradient（stop-grad）

target encoder 的输出在计算 loss 前必须 `detach`，梯度不能回流到 target encoder：

```python
# 见 verse_awm/ijepa.py 第 273-276 行
with no_grad():
    s_y_grid = self.target_encoder(x)
    s_y_grid = s_y_grid.detach()
```

并在 `jepa_loss` 内部再保险一次（[`jepa.py` 第 400 行](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L400)）：`target = target.detach()`。

如果 stop-grad 缺失，predictor 会通过反传让 target encoder 把所有输出拉到同一个点，瞬间坍塌。

### 2. EMA target encoder

target encoder 的参数不通过 optimizer 更新，而是用 context encoder 的 EMA：

```python
# 见 verse_awm/jepa.py 第 331-356 行
def update_target_encoder(context_encoder, target_encoder, decay=0.996):
    with no_grad():
        for cp, tp in zip(ctx_params, tgt_params):
            tp.data = decay * tp.data + (1 - decay) * cp.data
```

EMA decay 从 0.99（初期快速跟上）线性升到 0.9999（后期稳定），见 [`ema_decay_schedule`](file:///workspace/packages/verse_awm/verse_awm/jepa.py#L359-L370)。

如果 target_encoder 直接复制 context_encoder（无 EMA），两个网络同步变化，cosine loss 容易坍塌到 0。

### 3. Cosine loss

`jepa_loss(pred, target, loss_type="cosine")` 计算归一化后的余弦距离 `1 - cos_sim`：

```python
# 见 verse_awm/jepa.py 第 407-417 行
dot = (pred * target).sum(dim=-1)
pred_norm = ((pred * pred).sum(dim=-1) + eps).sqrt()
tgt_norm = ((target * target).sum(dim=-1) + eps).sqrt()
cos_sim = dot / (pred_norm * tgt_norm)
loss = (1.0 - cos_sim).mean()
```

余弦损失对 magnitude 不敏感，即使 predictor 输出范数坍塌到 0，loss 也不会到 0（因为方向也需要对齐）；这与 L2 loss 形成对比——L2 的 trivial 解是 `pred = 0`。

## 关键代码引用

- [数据生成](file:///workspace/examples/jepa_demo.py#L34-L51)：4 色块合成图像；
- [模型构建](file:///workspace/examples/jepa_demo.py#L79-L83)：`IJEPA(img_size=8, patch_size=4, embed_dim=32, depth=2, ...)`；
- [训练参数收集](file:///workspace/examples/jepa_demo.py#L89-L94)：注意 `target_encoder.parameters()` **不在** 优化器中；
- [训练循环](file:///workspace/examples/jepa_demo.py#L102-L121)：含 EMA 调度与 `update_target_encoder`；
- [loss 日志保存](file:///workspace/examples/jepa_demo.py#L147-L166)：JSON 格式，便于后续可视化；
- [PASS 判定](file:///workspace/examples/jepa_demo.py#L169-L175)：`loss_drop > 0.05` 视为通过。

## 注意事项

1. **数据集极小**：仅 200 张 8×8 图像，参数量 6.7k。这不是真正训练任务，只验证管道正确性；真实 I-JEPA 训练需要 ImageNet 级数据 + ViT-B 容量。
2. **EMA decay 调度至关重要**：若固定 decay=0.999，初期 target 跟不上 context，loss 不下降；若固定 decay=0.99，后期 target 太"活"，容易坍塌。本 demo 用 0.99→0.9999 线性 ramp 是经验上的最佳实践。
3. **predictor 不接收 target_queries 的内容**：target_queries 是位置编码（不含 patch content），让 predictor 学到"位置 → 表征"的映射，而非"内容 → 表征"的恒等映射；否则会坍塌。
4. **`n_targets=4` 与 `context_ratio=0.5`**：4 个 target block 各覆盖 ~20% patches，1 个 context block 覆盖 ~50% patches，是 I-JEPA 论文的推荐配置。
5. **不评估线性探针准确率**：本 demo 仅验证 loss 下降，不评估下游任务（如图像分类）；CIFAR-10 线性探针测试见 [`tests/test_ijepa_cifar10.py`](file:///workspace/tests/test_ijepa_cifar10.py)。
6. **PASS 门槛低**：`loss_drop > 0.05` 是为了在合成数据上稳定通过；真实训练任务应要求 `final_loss < 0.1` 且线性探针准确率 > 60%。
