# 模型压缩管线设计（trillion → billion 路线图）

- **状态**：Design Draft（PoC 待 Stage 5 验证）
- **日期**：2026-07-20
- **作者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关 ADR**：[ADR-001: CPU 优先设计决策](file:///workspace/docs/architecture/adr-001-cpu-first.md)、[ADR-004: CPU 并行计算方案选型](file:///workspace/docs/architecture/adr-004-cpu-parallel.md)
- **相关规范**：[`/workspace/.trae/specs/evolve2-cometspark/spec.md`](../../../.trae/specs/evolve2-cometspark/spec.md)
- **相关实现**：`packages/verse_torch/verse_torch/compress.py`（Stage 5 待实现）、[`tests/test_compression_poc.py`](file:///workspace/tests/test_compression_poc.py)

## 1. 上下文

### 1.1 终极目标

Verse 框架的端侧部署愿景是：**将万亿级别（trillion，~1T 参数）大模型压缩到十亿级别（billion，~1B 参数）**，实现 **1000× 压缩**，让端侧 CPU / 嵌入式设备也能跑高能力 LLM。这与 [ADR-001](file:///workspace/docs/architecture/adr-001-cpu-first.md) 的 CPU 优先路线相辅相成：

- CPU 优先解决了"算子能在 CPU 上跑"的问题；
- 线性复杂度架构（[ADR-002](file:///workspace/docs/architecture/adr-002-linear-complexity.md)）解决了"长上下文不爆显存"的问题；
- **压缩管线**解决"模型本身能塞进端侧内存"的问题——1T 参数 FP16 需要 2 TB 内存，端侧不可行；压到 1B 后 FP16 仅需 2 GB，INT4 后仅需 500 MB，可在 8 GB 内存的 Intel N100 / 树莓派 5 上运行。

### 1.2 本轮 PoC 目标

本轮 PoC（Stage 5）的目标是：**在小模型（1M 参数）上验证 10× 压缩可行且 loss 差异 ≤ 5%**。

- 选择 1M 参数而非更大模型，是为了在 sandbox 3 核 CPU 上 5 分钟内完成端到端验证；
- 10× 压缩是 1000× 路线图的第一步，验证组合技术（prune + INT4 + LoRA）的可行性；
- loss 差异 ≤ 5% 是经验阈值，超过该阈值说明压缩破坏了模型表达能力。

### 1.3 与现有模块的关系

- **依赖** `verse_torch.nn.TransformerLM`：作为 PoC 的压缩对象（1M 参数小模型）；
- **依赖** `verse_torch.training.Trainer`：压缩后用 LoRA 微调恢复精度；
- **被依赖** 于 `data/demo/`（CometSpark-v0.1）：压缩管线是端侧部署的关键路径；
- **被依赖** 于 `verse_inference`：压缩后的模型需通过 `ModelLoader` 加载并推理。

## 2. 压缩技术选型

### 2.1 候选技术总览

| 技术 | 压缩比 | 训练成本 | 推理加速 | 适用阶段 |
|---|---|---|---|---|
| **OSP（Outlier-Safe Pre-Training）** | 1.5–2×（间接，通过让权重更平滑利于量化） | 高（需重训） | 间接（与量化协同） | 阶段 2+ |
| **BitNet b1.58** | 8–16×（相比 FP16） | 极高（from scratch） | 4–6× | 阶段 3+ |
| **QLoRA** | 4×（INT4 量化 + LoRA 微调） | 低（4-bit 量化后微调） | 2–3× | PoC 起步 |
| **知识蒸馏** | 灵活（teacher → student 任意压缩比） | 中（需 teacher 模型） | 取决于 student | 阶段 2+ |
| **结构化剪枝** | 2–4×（按 head/channel） | 低（剪枝后微调） | 1.5–2× | PoC 起步 |
| **INT4 量化** | 4×（相比 FP16） | 极低（PTQ） | 2–3× | PoC 起步 |

### 2.2 OSP（Outlier-Safe Pre-Training）

**核心思想**：训练时显式分离 outlier 通道（少数幅值极大的权重），让主体权重分布更平滑，利于后续量化。代表工作：LLM.int8()、SmoothQuant、Outlier-Safe Layer。

- **优点**：
  - 量化误差显著降低（INT4 量化误差可降 50%+）；
  - 不改变模型结构，仅改变训练过程；
  - 与下游量化 / 剪枝 / 蒸馏兼容。
- **缺点**：
  - 需要从头训练（pre-training 阶段技术），不能直接用于已训练模型；
  - 训练复杂度增加（需维护 outlier mask）；
  - PoC 阶段用剪枝代替，无法直接验证。
- **PoC 替代**：用 `OutlierSafePruner` 在已训练模型上做后置剪枝，模拟 OSP 的"分离 outlier"效果。

### 2.3 BitNet b1.58

**核心思想**：将权重量化到 1.58-bit 三值 `{−1, 0, +1}`，相比 INT4 进一步压缩 2.5×（FP16 → INT4 是 4×，FP16 → 1.58-bit 是 10×）。代表工作：BitNet b1.58（Microsoft, 2024）。

- **优点**：
  - 极致压缩比（10× vs FP16）；
  - 推理时 matmul 退化为加减法，CPU 友好（参考 BitNet.cpp 在 CPU 上 6.17× 提速）；
  - 与 [ADR-001](file:///workspace/docs/architecture/adr-001-cpu-first.md) 的 CPU 优先目标高度契合。
- **缺点**：
  - 必须 from-scratch 训练（不能直接量化已训练模型）；
  - 训练稳定性需要特殊技巧（STE 直通估计、梯度裁剪）；
  - 生态较新，开源实现少。
- **PoC 替代**：用 INT4 量化代替，验证"低位宽 + 微调恢复"的基本路径。

### 2.4 QLoRA

**核心思想**：4-bit 量化 + LoRA 微调，量化后仍可训练。代表工作：QLoRA（Dettmers et al., 2023）。

- **优点**：
  - 4× 压缩 + 微调恢复精度，性价比高；
  - LoRA 仅训练少量参数（< 1%），微调成本低；
  - 与 CPU 优先兼容（INT4 matmul 在 CPU 上有优化）。
- **缺点**：
  - 仅 4× 压缩，离 1000× 目标还有 250× 距离，需配合其他技术；
  - 4-bit 量化误差可能让 loss 差异超过 5%（需调 sparsity 或改用 INT8）。
- **PoC 用途**：作为 PoC 阶段的核心技术之一（prune + INT4 + LoRA wrap）。

### 2.5 知识蒸馏

**核心思想**：teacher（大模型）→ student（小模型），用 soft labels（teacher 的 logits 温度软化后）传递暗知识。代表工作：Hinton distillation、TinyBERT、MiniLM。

- **优点**：
  - 灵活的压缩比（teacher / student 任意大小）；
  - 可与其他技术叠加（量化后蒸馏、剪枝后蒸馏）；
  - 保留 teacher 的语义信息。
- **缺点**：
  - 需要 teacher 模型，PoC 阶段无大模型可用；
  - 蒸馏训练时间较长；
  - teacher / student 容量差距过大时效果下降。
- **PoC 替代**：用 self-distill（同模型作 teacher），验证蒸馏管线代码正确性。

### 2.6 结构化剪枝

**核心思想**：按 head / channel 剪枝，移除不重要的结构单元，硬件友好（不像非结构化剪枝需要稀疏矩阵支持）。代表工作：LLM-Pruner、Sheared LLaMA、Wanda。

- **优点**：
  - 硬件友好（剪枝后是稠密小矩阵，无需稀疏支持）；
  - 推理加速明显（1.5–2×）；
  - 与量化 / 蒸馏兼容。
- **缺点**：
  - 压缩比受限（2–4×，远低于量化的 4–10×）；
  - 剪枝后需微调恢复精度；
  - 重要性评估（按 |weight|_mean 还是按 gradient）需要调参。
- **PoC 用途**：作为 PoC 阶段的核心技术之一（sparsity=0.3，剪掉 bottom 30% 的 head/channel）。

## 3. trillion → billion 路线图

### 3.1 总体路线

```
1T ──[阶段4: 全栈]──> 100B ──[阶段3: BitNet+QLoRA]──> 1B ──[阶段2: OSP+INT4+蒸馏]──> 10M ──[PoC: prune+INT4+LoRA]──> 100K
                                                                                          ↑
                                                                                  1M 起步（PoC 验证）
```

每个阶段 10× 压缩，共 4 个阶段，总目标 1000×。

### 3.2 阶段详情

#### PoC 阶段（本轮，Stage 5）

- **目标**：1M → 100K，10× 压缩，loss ≤ 5%
- **技术栈**：
  1. `OutlierSafePruner(model, sparsity=0.3)`：剪掉 bottom 30% 的 head / channel，约 1.4× 压缩
  2. `quantize_only(model, bits=4)`：INT4 量化，约 4× 压缩（FP16 → INT4）
  3. `lora_wrap(model, r=8, alpha=16)`：LoRA 包裹所有 Linear，准备微调
  4. `Trainer.fit()`：用同一份数据微调 1–2 epoch 恢复精度
- **验证**：`tests/test_compression_poc.py`
  - 压缩比 ≥ 10×
  - loss 差异 ≤ 5%
  - 生成 `docs/benchmarks/compression_poc.md` 对照表

#### 阶段 2：100M → 10M

- **目标**：100M → 10M，10× 压缩，loss ≤ 10%
- **技术栈**：
  1. **OSP 重训**：用 outlier-safe 训练策略从头训练 100M 模型，让权重分布平滑
  2. **INT4 量化**：OSP 后量化误差显著降低，可安全 INT4
  3. **知识蒸馏微调**：用未量化的 100M 模型作 teacher，蒸馏微调量化后的 10M 模型
- **风险**：OSP 重训需要 GPU 资源（约 100 GPU·小时），CPU 上不可行

#### 阶段 3：10B → 1B

- **目标**：10B → 1B，10× 压缩，loss ≤ 15%
- **技术栈**：
  1. **BitNet b1.58 重训**：用 1.58-bit 三值权重 from-scratch 训练 10B 模型
  2. **QLoRA 微调**：量化后用 LoRA 微调对齐下游任务
  3. **知识蒸馏**：用 FP16 的 10B 模型作 teacher，蒸馏 1.58-bit 的 10B student
- **风险**：BitNet b1.58 训练稳定性需要 STE + 梯度裁剪；GPU 资源需求大

#### 阶段 4：1T → 100B

- **目标**：1T → 100B，10× 压缩，loss ≤ 20%
- **技术栈**：**全栈组合**
  1. **OSP + BitNet b1.58 联合训练**：OSP 让权重平滑，BitNet 量化到 1.58-bit
  2. **结构化剪枝**：剪掉 50% 不重要的 head（基于 Wanda 重要性评估）
  3. **QLoRA 微调**：对齐下游任务
  4. **知识蒸馏**：用 FP16 的 1T teacher 蒸馏
- **风险**：全栈组合调参复杂；1T teacher 蒸馏成本极高

### 3.3 数值验证表格

| 阶段 | 原参数量 | 压缩后 | 压缩比 | 主要技术 | 目标 loss 差异 |
|------|---------|--------|--------|---------|--------------|
| PoC | 1M | 100K | 10× | prune + INT4 + LoRA | ≤ 5% |
| 阶段 2 | 100M | 10M | 10× | OSP + INT4 + 蒸馏 | ≤ 10% |
| 阶段 3 | 10B | 1B | 10× | BitNet + QLoRA | ≤ 15% |
| 阶段 4 | 1T | 100B | 10× | 全栈（OSP + BitNet + QLoRA + 蒸馏 + 剪枝） | ≤ 20% |
| **总计** | **1T** | **1B** | **1000×** | - | - |

### 3.4 存储与显存对比

| 阶段 | 原始存储（FP16） | 压缩后存储 | 端侧可行性 |
|---|---|---|---|
| PoC | 2 MB | 200 KB | 任意设备 |
| 阶段 2 | 200 MB | 20 MB | 树莓派 / N100 |
| 阶段 3 | 20 GB | 2 GB | 消费级 CPU |
| 阶段 4 | 2 TB | 200 GB | 服务器 CPU（需进一步量化到 1.58-bit 才能到 ~30 GB） |
| **总目标** | **2 TB** | **2 GB** | **消费级 CPU（INT4 量化后）** |

## 4. 关键决策

### 4.1 优先用组合技术而非单一技术

- **理由**：单一技术的压缩比有上限（INT4 仅 4×，剪枝仅 2–4×），无法独立达到 10× / 阶段目标；
- **组合策略**：剪枝（结构化）+ 量化（位宽）+ 蒸馏（恢复精度）三件套，乘法关系而非加法；
- **PoC 验证**：`compress_pipeline` 依次 prune → quantize → lora_wrap，验证组合路径。

### 4.2 OSP 是预训练阶段技术，PoC 阶段用剪枝代替

- **理由**：OSP 需要重训，PoC 阶段无 GPU 资源；`OutlierSafePruner` 在已训练模型上模拟"分离 outlier"效果；
- **代价**：PoC 的压缩误差可能略高于真正 OSP；但 5% loss 阈值仍可达；
- **未来路径**：阶段 2 引入 GPU 后，从 OSP 重训开始。

### 4.3 BitNet b1.58 需要 from-scratch 训练，PoC 阶段用 INT4 量化

- **理由**：BitNet 必须从零训练，PoC 阶段无资源；INT4 量化是后置量化（PTQ），可直接用于已训练模型；
- **代价**：BitNet 的 10× 压缩优势无法在 PoC 体现；PoC 用 INT4 的 4× + 剪枝 1.4× + LoRA 微调恢复凑够 10×；
- **未来路径**：阶段 3 引入 BitNet，从 10B 模型 from-scratch 训练。

### 4.4 蒸馏需要 teacher 模型，PoC 阶段用 self-distill

- **理由**：PoC 阶段无大模型作 teacher；self-distill（同模型压缩前后作 teacher / student）可验证蒸馏管线代码；
- **代价**：self-distill 的精度恢复效果弱于真正的 teacher-student 蒸馏；
- **未来路径**：阶段 2 起用未压缩模型作 teacher 蒸馏压缩后的 student。

### 4.5 压缩管线 API 设计

```python
from verse_torch.compress import (
    compress_pipeline, quantize_only, prune_only, lora_only, distill_only,
    OutlierSafePruner, LoRALinear, KnowledgeDistiller,
)

# 端到端压缩
report = compress_pipeline(
    model,
    target_ratio=0.1,  # 压缩到原参数量的 10%（10× 压缩）
    eval_fn=lambda m: evaluate_loss(m, val_loader),
)
# report = {
#     "original_params": 1_000_000,
#     "compressed_params": 100_000,
#     "ratio": 0.1,
#     "original_loss": 5.23,
#     "compressed_loss": 5.41,
#     "loss_diff_pct": 3.4%,  # ≤ 5% 阈值
#     "steps": ["prune", "quantize", "lora_wrap", "finetune"],
#     "prune_report": {...},
#     "quantize_report": {...},
# }

# 单技术函数（用于 ablation）
pruned = prune_only(model, sparsity=0.3)
quantized = quantize_only(model, bits=4)
lora_wrapped = lora_only(model, r=8, alpha=16)
distilled = distill_only(student, teacher=original_model, T=2.0, alpha=0.5)
```

## 5. PoC 实施计划

### 5.1 实现路径（Stage 5 任务）

| Task | 内容 | 依赖 |
|---|---|---|
| 5.1 | `docs/papers/compression_references.md`：收集 ≥ 10 篇参考 | 无 |
| 5.2 | `OutlierSafePruner(model, sparsity=0.3)`：按 head/channel 重要性剪枝 | `verse_torch.nn.TransformerLM` |
| 5.3 | `LoRALinear(d_in, d_out, r=8, alpha=16)`：base frozen + A/B trainable | `verse_torch.nn.Linear` |
| 5.4 | `KnowledgeDistiller(teacher, student, T=2.0, alpha=0.5)`：KL + hard label 联合 | `cross_entropy_loss` |
| 5.5 | `compress_pipeline(model, target_ratio=0.1, eval_fn)`：端到端流程 | 5.2 / 5.3 / 5.4 |
| 5.6 | `quantize_only` / `prune_only` / `lora_only` / `distill_only` 单技术函数 | 5.2 / 5.3 / 5.4 |
| 5.7 | `tests/test_compression_poc.py`：1M 模型上验证 10× + loss ≤ 5% | 5.5 |

### 5.2 PoC 验收标准

- [ ] 压缩比 ≥ 10×（`report["ratio"] <= 0.1`）
- [ ] loss 差异 ≤ 5%（`report["loss_diff_pct"] <= 5.0`）
- [ ] 生成 `docs/benchmarks/compression_poc.md` 对照表
- [ ] 单元测试 `tests/test_compression_poc.py` 全部 PASS
- [ ] 在 sandbox 3 核 CPU 上 5 分钟内完成端到端验证

### 5.3 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| INT4 量化误差让 loss 差异超过 5% | 调整 sparsity（从 0.3 降到 0.2）；或改用 INT8 量化（4× → 2× 压缩，但精度更高） |
| `OutlierSafePruner` 剪枝破坏模型结构 | 剪枝后必须微调（LoRA wrap + 1–2 epoch）；剪枝报告输出每层保留比例便于诊断 |
| multiprocessing IPC 开销让并行训练变慢 | `ParallelLinear` 设置 `batch_threshold=16`，小 batch 自动降级为单进程（参考 [ADR-004](file:///workspace/docs/architecture/adr-004-cpu-parallel.md)） |
| LoRA 微调不收敛 | 调整 `r`（rank）和 `alpha`（scaling）；增加微调 epoch；或改用 full fine-tune |
| 1M 模型太小，压缩效果不显著 | PoC 主要验证管线代码正确性；阶段 2 在 100M 模型上验证真实压缩效果 |
| 蒸馏 teacher / student 容量差距过小（self-distill） | self-distill 主要验证代码路径；阶段 2 起用真实 teacher |

## 6. 与其他模块的集成

### 6.1 与 `verse_torch.nn` 的集成

- `OutlierSafePruner` 接受任何 `nn.Module`，遍历 `ModuleList` 找到所有 `Linear` / `MultiHeadAttention`，按 head / channel 重要性剪枝；
- `LoRALinear` 继承 `nn.Linear`，API 兼容，可直接替换；
- `KnowledgeDistiller` 接受两个 `nn.Module`（teacher / student），共享 `cross_entropy_loss`。

### 6.2 与 `verse_torch.training` 的集成

- 压缩后的 LoRA 微调通过 `Trainer.fit()` 完成；
- `Trainer` 的 `optimizer` 仅更新 LoRA 的 A/B 矩阵（base frozen）；
- `CheckpointManager` 保存压缩后的模型 + LoRA 权重。

### 6.3 与 `verse_inference` 的集成

- 压缩后的模型通过 `ModelLoader.load(path, arch="cometspark")` 加载；
- `StreamingGenerator` 兼容 INT4 量化模型（matmul 退化为 INT4 × FP16）；
- 端侧部署时压缩模型可直接在 4 核 CPU 上推理。

### 6.4 与 `data/demo/`（CometSpark-v0.1）的集成

- CometSpark 训练完成后可选调用 `compress_pipeline` 压缩；
- 压缩报告写入 `data/demo/checkpoints/compression_report.json`；
- 压缩模型可用于端侧推理演示。

## 7. 后续工作

### 7.1 阶段 2 起的扩展

- **OSP 训练策略**：在 `verse_torch.training` 中新增 `OSPOptimizer` / `OSPScheduler`；
- **真实蒸馏**：`KnowledgeDistiller` 支持外部 teacher 模型（不再限于 self-distill）；
- **GPU 后端**：OSP 重训需要 GPU，依赖后续 spec 引入 GPU 后端。

### 7.2 阶段 3 起的扩展

- **BitNet b1.58 实现**：新增 `BitNetLinear`（权重三值化 + STE 反向）；
- **BitNet 训练循环**：`Trainer` 支持 BitNet 模式的梯度裁剪与 STE；
- **1.58-bit matmul kernel**：CPU 上用 SIMD 加速三值 matmul（参考 BitNet.cpp）。

### 7.3 阶段 4 起的扩展

- **全栈组合**：OSP + BitNet + QLoRA + 蒸馏 + 剪枝 的联合优化；
- **Wanda 重要性评估**：替代 |weight|_mean，用 |weight| × |activation| 评估重要性；
- **自动调参**：根据 target_ratio 自动选择 sparsity / bits / r 的组合。

## 8. 参考

### 8.1 论文

- **BitNet b1.58**: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"（Ma et al., 2024）— [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **QLoRA**: "QLoRA: Efficient Finetuning of Quantized LLMs"（Dettmers et al., 2023）— [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **OSP / Outlier-Safe**: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"（Dettmers et al., 2022）— [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- **SmoothQuant**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"（Xiao et al., 2022）— [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- **知识蒸馏**: "Distilling the Knowledge in a Neural Network"（Hinton et al., 2015）— [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **LLM-Pruner**: "LLM-Pruner: On the Structural Pruning of Large Language Models"（Ma et al., 2023）— [arXiv:2305.11627](https://arxiv.org/abs/2305.11627)
- **Wanda**: "A Simple and Effective Pruning Approach for Large Language Models"（Sun et al., 2023）— [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
- **Sheared LLaMA**: "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"（Xia et al., 2023）— [arXiv:2310.06694](https://arxiv.org/abs/2310.06694)
- **BitNet**: "BitNet: Scaling 1-bit Transformers for Large Language Models"（Wang et al., 2023）— [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- **BitNet.cpp**: "BitNet.cpp: Efficient Edge Inference for 1-bit LLMs"（Microsoft, 2024）— [GitHub](https://github.com/microsoft/BitNet)

### 8.2 工程参考

- [BitNet.cpp](https://github.com/microsoft/BitNet)：1.58-bit LLM CPU 推理，6.17× 提速
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)：INT4 量化工具链
- [PEFT](https://github.com/huggingface/peft)：LoRA / QLoRA 实现参考
- [Verse `compress.py`](file:///workspace/packages/verse_torch/verse_torch/compress.py)（Stage 5 待实现）
- [`tests/test_compression_poc.py`](file:///workspace/tests/test_compression_poc.py)（PoC 验证结果）

### 8.3 相关 ADR

- [ADR-001: CPU 优先设计决策](file:///workspace/docs/architecture/adr-001-cpu-first.md) — 压缩是 CPU 优先的关键支撑
- [ADR-002: 线性复杂度架构选型](file:///workspace/docs/architecture/adr-002-linear-complexity.md) — 压缩 + 线性复杂度共同保证端侧可行
- [ADR-004: CPU 并行计算方案选型](file:///workspace/docs/architecture/adr-004-cpu-parallel.md) — 压缩后微调用 ParallelLinear 加速
