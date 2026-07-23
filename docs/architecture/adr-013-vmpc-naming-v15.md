# ADR-013: VMPC 命名 + V1.5 设计

- **状态**：Accepted
- **日期**：2026-07-23
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：Part5K1 升级任务集
- **前置 ADR**：[ADR-012 压缩技术 V1.3](adr-012-compression-v13.md)（V1.5 在 V1.3 基础上演进）
- **相关 ADR**：[ADR-014 双模型并行](adr-014-dual-model-small-mate.md)（small / mate 预设由 `vmpc_compress` 提供）、[ADR-015 VMT 完整三档策略](adr-015-vmt-full-strategy.md)（freeze 档复用 VMPC INT4 量化）

## 上下文

Part5K1 之前，`verse_torch.compress` 已提供压缩管线 V1.3（[ADR-012](adr-012-compression-v13.md)），组合剪枝 + 量化 + 知识蒸馏 + LoRA 实现"大模型→小模型能力转移"。但存在以下问题：

1. **缺乏统一品牌**：压缩 / 量化 / 蒸馏 / 剪枝四项技术散落在 `compress.py` 各个类与函数中，对外没有统一命名入口，用户难以建立"VerseNext 的压缩技术栈"心智模型。
2. **V1.3 蒸馏保序性不足**：三重损失（软标签 KL + 硬标签 CE + 中间层特征 MSE）保证 student 学到 teacher 的分布与特征，但**不直接约束 student 与 teacher 的 top-k 排序一致性**，推理侧 student 可能出现"分布近似但 top-1 漂移"的情况。
3. **推理 logits 未校准**：量化后的 student 在推理时 logits 尺度可能偏移（INT4 反量化误差累积），缺少 temperature-aware 的校准机制。
4. **反量化精度损失集中在 outlier 通道**：均匀 INT4 量化对所有通道一视同仁，但权重中存在少量 outlier 通道（幅度显著大于其他），这些通道被强制压到 int4 后反量化误差最大，是压缩后质量下降的主因。
5. **缺乏过拟合防护 + 压缩感知**：压缩训练阶段 student 容易过拟合 teacher 的分布（尤其小模型），且缺乏"压缩感知"的正则化（即训练时主动收紧稀疏度，让模型更适配后续剪枝）。

同时需要保持向后兼容：V1.3 的 `compress_pipeline(version="1.3")` 接口与 stats 结构不能破坏。

## 决策

**正式命名模型参数压缩技术为 VMPC（Versenext Model Parameters Compression），升级到 V1.5；通过 `vmpc.py` 门面统一入口，新增 contrastive_distill / logit_calibration / outlier-aware 反量化 + VMPCRegularizer + vmpc_compress 便捷预设。**

具体含义：

1. **命名：VMPC 统一品牌**
   - `verse_torch.vmpc` 模块作为压缩 / 量化 / 蒸馏 / 剪枝的统一门面，re-export `compress.py` 中的核心对象（`compress_pipeline` / `OutlierSafePruner` / `LoRALinear` / `KnowledgeDistiller` / `QLinear` / `compression_report` 等）。
   - 调用者通过 `verse_torch.vmpc` 与通过 `verse_torch.compress` 拿到的是同一对象（`verse_torch.vmpc.compress_pipeline is verse_torch.compress.compress_pipeline` 必须为 `True`）。

2. **V1.5 在 V1.3 基础上新增三项算法升级**：
   - **contrastive_distill（对比蒸馏）**：基于 margin ranking loss，保证 student 与 teacher 的 top-k token 排序一致。`KnowledgeDistiller` 新增 `distill_contrastive` / `contrastive_loss_weight` / `contrastive_margin` / `contrastive_top_k` 参数；`compute_loss` 在三重损失基础上叠加 `contrastive_loss_weight * contrastive_loss(...)`。
   - **logit_calibration（推理 logits 校准）**：temperature-aware sharpening。`_compress_pipeline_v15` 计算校准因子并存到模型 `_vmpc_logit_calib_factor` 元数据，推理时（`CometSparkNexLM.generate`）按校准因子对 logits 做锐化，弥补量化后的尺度漂移。
   - **outlier-aware 反量化**：outlier channel 保留 fp16，其余 int4。`QLinear` 新增 `outlier_aware` / `outlier_ratio` / `outlier_threshold` 参数；反量化时对 outlier 通道叠加 fp32 residual（识别 |w| > threshold 的通道，单独保留高精度副本）。

3. **VMPCRegularizer：防过拟合 + 压缩感知稀疏收紧**
   - 组成：参数幅度 L2 正则 + 压缩感知 dropout（随机置零部分权重模拟压缩损失）+ early-exit 自适应稀疏收紧。
   - `step(val_loss)` 维护 `val_loss_history`，连续 `patience` 步 val_loss 不下降（val_loss 平台期）→ `target_sparsity *= sparsity_decay`，自动收紧 sparsity。
   - `target_sparsity` 降到 `SPARSITY_FLOOR`（0.05）以下时返回 `should_stop=True`，触发早停。
   - 轻量、无状态依赖，通过 `attach(trainer)` monkey-patch `_compute_loss` 接入训练循环。

4. **vmpc_compress(model, profile="small"|"mate") 一键预设**
   - `profile="small"`：ternary 量化 + 高稀疏（sparsity=0.5），适配 0.06zB 小模型（[ADR-014](adr-014-dual-model-small-mate.md)）。
   - `profile="mate"`：int4 量化 + 中稀疏（sparsity=0.3）+ LoRA + 蒸馏，适配 0.2zB 旗舰模型（[ADR-014](adr-014-dual-model-small-mate.md)）。
   - 内部调用 `compress_pipeline(version="1.5")`，极端情况回退到 `version="1.3"` 保证可用性。

5. **版本分派**：
   - `compress_pipeline` 内部按 `version` 字段分派：`version >= 1.5` 走 `_compress_pipeline_v15`（V1.3 + contrastive_distill + logit_calibration），`version >= 1.3` 走 `_compress_pipeline_v13` 路径。
   - V1.5 的 stats 与 V1.3 完全兼容（同名同义键），追加 V1.5 专属字段：`"vmpc_version": "1.5"` / `"logit_calib_factor"` / `"contrastive_distill"`。
   - 模型元数据 `_vmpc_version` / `_vmpc_compressed` / `_vmpc_logit_calib_factor` / `_vmpc_contrastive_distill` 供下游推理识别。

## 后果

### 优点

- **统一品牌**：VMPC 作为 VerseNext 压缩技术栈的统一入口，降低用户认知成本；`vmpc.py` 门面 + re-export 保证向后兼容。
- **推理保序**：contrastive_distill 直接约束 top-k 排序一致性，避免"分布近似但 top-1 漂移"。
- **推理校准**：logit_calibration 弥补量化后 logits 尺度漂移，提升生成质量。
- **outlier 精度**：outlier-aware 反量化把误差集中在非 outlier 通道，关键通道保留 fp16，质量损失最小化。
- **过拟合防护**：VMPCRegularizer 在压缩训练阶段自动收紧 sparsity，让模型更适配后续剪枝，val_loss 平台期触发早停。
- **一键预设**：`vmpc_compress(profile=...)` 降低双模型场景的配置门槛。
- **向后兼容**：V1.3 通过 `version="1.3"` 仍可访问，stats 结构同构。

### 缺点

- **V1.5 计算开销**：contrastive_distill 需额外计算 top-k margin ranking loss；outlier-aware 反量化需维护 outlier 通道的 fp16 副本。但这是离线一次性成本。
- **校准因子依赖推理侧配合**：`logit_calibration` 校准因子存到模型元数据，需推理侧（`CometSparkNexLM.generate`）主动读取并应用，否则不生效。
- **outlier 识别阈值需调优**：`outlier_threshold` 默认值可能不适合所有模型分布，极端情况下需用户手动调参。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| contrastive_distill 的 top-k margin 失衡 | 默认 `contrastive_loss_weight=0.5` / `contrastive_margin=0.5` / `contrastive_top_k=10`，用户可调 |
| logit_calibration 校准因子过拟合 teacher | 校准因子仅在压缩时一次性计算，推理时固定应用，不随训练更新 |
| outlier 通道识别不准 | `outlier_ratio` 控制保留比例（默认 0.01），用户可按模型分布调整 |
| V1.5 与 V1.3 行为差异导致回归 | `version` 参数显式选择；V1.5 stats 与 V1.3 同构 + 追加字段；测试覆盖两路径 |

## 替代方案（已否决）

### 方案 A：仅升级 V1.3 不命名 VMPC

**描述**：保留 `compress.py` 现有命名，仅在 V1.3 基础上追加 contrastive_distill / logit_calibration / outlier-aware，不引入 `vmpc.py` 门面。

**否决理由**：缺乏统一品牌，压缩 / 量化 / 蒸馏 / 剪枝四项技术散落，用户难以建立心智模型；`vmpc_compress` 一键预设无处安放；双模型场景（[ADR-014](adr-014-dual-model-small-mate.md)）的 small / mate 预设需要统一入口。

### 方案 B：强制 V1.5 为默认路径（破坏性升级）

**描述**：`compress_pipeline` 默认走 V1.5，移除 V1.3 路径。

**否决理由**：破坏现有用户代码（V1.3 stats 结构 / 行为）；违反向后兼容承诺；V1.5 的 contrastive_distill 训练成本可能不适用于所有场景。

### 方案 C：outlier-aware 用混合精度训练替代

**描述**：不维护 outlier 通道 fp16 副本，训练时整体用混合精度。

**否决理由**：混合精度是训练期技术，不解决推理期反量化误差；outlier 通道的精度损失主要发生在量化→反量化环节，需在反量化路径专门处理。

## 备注

- 本 ADR 是 Part5K1 "VMPC 命名 + V1.5 算法升级"的核心决策。
- VMPC 品牌借鉴自 VerseNext 命名体系（V = Versenext，MPC = Model Parameters Compression）。
- `vmpc.py` 门面的 re-export 模式借鉴自 `verse_infra` 总包聚合（[ADR-006](adr-006-verse-infra-aggregation.md)）。
- V1.5 的 contrastive_distill 灵感来源于 CRD（Contrastive Representation Distillation）的 margin ranking 思路，但简化为 top-k token 排序约束。
- 相关测试：`tests/test_vmpc.py` 覆盖门面 re-export / VMPCRegularizer / vmpc_compress / V1.5 算法路径。
- 相关文档：[Verse 训练指南 - VMPC V1.5 压缩指南](../training_guide.md)、[Verse 性能调优 - VMPC V1.5 调优](../performance_tuning.md)
