# ADR-015: VMT 完整三档策略

- **状态**：Accepted
- **日期**：2026-07-23
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：Part5K1 升级任务集
- **前置 ADR**：[ADR-011 智能分区训练](adr-011-layerwise-training.md)（LayerWiseTrainer 是 VMT 的 unload 档基础）、[ADR-013 VMPC 命名 + V1.5 设计](adr-013-vmpc-naming-v15.md)（freeze 档复用 VMPC INT4 量化）
- **相关 ADR**：[ADR-004 CPU 并行](adr-004-cpu-parallel.md)（VMT 与 chunk 并行互补）、[ADR-014 双模型并行](adr-014-dual-model-small-mate.md)（双模型训练可配合 VMT）

## 上下文

Part4K2 的 `LayerWiseTrainer`（[ADR-011](adr-011-layerwise-training.md)）实现了"按 layer 分组训练 + `.vn` 分片卸载"，解决了大模型在有限内存 CPU / 单卡 GPU 上的训练问题。但其策略单一——所有分区统一走 **unload 档**（训完一组卸载到硬盘），存在以下局限：

1. **冻结场景缺失**：对于底层（embedding 附近）已经收敛稳定的层，最理想的是直接**冻结**（`requires_grad=False`），既不卸载也不参与梯度计算，节省内存与计算。但 LayerWiseTrainer 只能卸载，无法表达"冻结但留在内存"语义。
2. **高频训练优化缺失**：对于顶层（lm_head 附近）需要高频训练的层，逐块前向 + 卸载的开销大，理想的是**层融合 + 梯度累积**专项优化。但 LayerWiseTrainer 的 forward 是逐块串行，没有融合路径。
3. **策略无法差异化**：所有分区统一 unload，无法按层位置分配不同策略（如"底层 freeze、中层 optimize、顶层 unload"），无法发挥不同层的训练特性。
4. **VMT 品牌缺失**：`LayerWiseTrainer` 命名偏向实现细节（layer-wise），缺少 VerseNext 品牌化的"内存感知训练"统一命名。

同时需要保持向后兼容：`LayerWiseTrainer` 的现有接口与行为不能破坏。

## 决策

**实现 VMT（Versenext Memory-aware Training）完整三档策略：unload（卸载到硬盘 .vn 分片，已有）/ freeze（INT4 量化 + requires_grad=False，压缩冻结）/ optimize（层融合 + 梯度累积，高频训练专项优化）；VMTTrainer 继承 LayerWiseTrainer，支持 vmt_strategy 解析（"auto" 或显式语法）；LayerWiseTrainer 保留为简化版（仅 unload）。**

具体含义：

1. **VMT 三档策略**：
   - **unload 档**（已有，LayerWiseTrainer 基础能力）：训完一组卸载到硬盘 `.vn` 分片（复用 [ADR-009](adr-009-vn-format.md)），内存超阈值时自动卸载已训练的非当前组。适用于内存极度受限场景。
   - **freeze 档**（新增）：INT4 量化 + `requires_grad=False`（压缩冻结）。训练期间该组 block 参数被 INT4 量化（in-place 量化→反量化，模拟压缩），并冻结梯度；训练结束后从 fp32 备份精确恢复（反量化误差 = 0）。适用于底层已收敛稳定的层，冻结后既不卸载也不参与梯度，节省内存与计算。
   - **optimize 档**（新增，Task 7 层融合 + Task 8 VMT 集成）：层融合 + 梯度累积（高频训练专项优化）。前向走 `_fused_forward_blocks`（Task 7 已实现，数值与逐块前向严格一致），大 batch 时按 `micro_batch_size` 分微批累积梯度。适用于顶层需要高频训练的层。

2. **VMTTrainer 继承 LayerWiseTrainer**：
   - `VMTTrainer` 继承 `LayerWiseTrainer`，复用其 unload 档能力（分区 / 卸载 / 加载 / 合并 / fine-tune）。
   - 新增 freeze 档：`_freeze_partition`（INT4 量化 + 冻结 + fp32 备份）/ `_unfreeze_partition`（从备份恢复 fp32）。
   - 新增 optimize 档：`_optimize_partition`（层融合前向 + 微批梯度累积）。
   - 对外接口与 `LayerWiseTrainer` 一致（`fit` / `evaluate`），训练前后模型对象保持统一实体（同一 id）。

3. **vmt_strategy 解析**：
   - **`"auto"`**（默认）：按层位置自动分配——前 1/3 freeze（底层稳定）、中间 1/3 optimize（中层高频）、后 1/3 unload（顶层卸载）。
   - **显式语法**：`"layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload"`，支持 `layers[start:end]` 切片语法 + `=tier` 档名。
   - 解析校验：档名合法（freeze / optimize / unload）、层区间合法（不越界）、层区间不重叠且连续覆盖全部层。

4. **LayerWiseTrainer 保留为简化版**：
   - `LayerWiseTrainer` 保留原有行为（仅 unload 档），作为 VMT 的简化版入口。
   - 用户无需三档策略时仍可用 `LayerWiseTrainer`，接口与行为完全不变（向后兼容）。
   - `VMTTrainer` 是 `LayerWiseTrainer` 的超集，需要差异化策略时升级到 `VMTTrainer`。

5. **CLI / 编程入口**：
   - 编程：`VMTTrainer(model, config, vmt_strategy="auto", partition_size=2)`。
   - CLI：通过 `--vmt-strategy` 参数指定（与 `--partition-training` 配合）。

## 后果

### 优点

- **差异化策略**：按层位置分配 freeze / optimize / unload，发挥不同层的训练特性（底层冻结省内存、中层融合加速、顶层卸载保精度）。
- **freeze 档省内存**：底层冻结后既不卸载也不参与梯度，比 unload 档少一次 I/O，比全量训练省梯度内存。
- **optimize 档加速**：层融合前向减少 Python 循环开销，梯度累积支持大 batch 等效训练。
- **VMT 品牌化**：VMT（Versenext Memory-aware Training）作为内存感知训练的统一命名，与 VMPC（压缩）品牌呼应。
- **向后兼容**：`LayerWiseTrainer` 保留为简化版，现有代码零修改；`VMTTrainer` 是超集，按需升级。

### 缺点

- **策略解析复杂度**：显式语法（`layers[0:8]=freeze, ...`）需解析与校验，比单一 unload 策略复杂。
- **freeze 档量化误差**：训练期间 freeze 档走 INT4 量化前向，存在量化误差（但训练后从 fp32 备份恢复，最终模型无误差）。
- **optimize 档实现耦合**：依赖 Task 7 的 `_fused_forward_blocks`，若层融合实现有 bug 会影响 optimize 档。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 显式语法解析错误 | 严格校验档名 / 层区间 / 覆盖完整性，解析失败抛 `ValueError` 明确提示 |
| freeze 档量化误差影响训练 | 训练期间量化仅用于前向（模拟压缩），梯度不回传；训练后从 fp32 备份精确恢复 |
| optimize 档层融合数值不一致 | Task 7 已验证 `_fused_forward_blocks` 与逐块前向数值严格一致（测试覆盖） |
| auto 策略不适配所有模型 | 用户可改用显式语法按需分配；auto 是合理默认（前 1/3 freeze / 中 1/3 optimize / 后 1/3 unload） |
| VMTTrainer 与 LayerWiseTrainer 行为差异 | VMTTrainer 默认 `vmt_strategy="auto"`，与 LayerWiseTrainer 的全 unload 不同；用户需显式选择 |

## 替代方案（已否决）

### 方案 A：仅扩展 LayerWiseTrainer

**描述**：不引入 VMTTrainer，直接在 `LayerWiseTrainer` 中新增 freeze / optimize 档 + `vmt_strategy` 参数。

**否决理由**：三档策略需要更复杂的策略解析（`layers[start:end]=tier` 语法），与 `LayerWiseTrainer` 的"仅 unload 简化版"定位冲突；破坏 `LayerWiseTrainer` 的向后兼容（新增参数改变默认行为）；`LayerWiseTrainer` 保留为简化版更清晰。

### 方案 B：三档独立三个 Trainer 类

**描述**：`UnloadTrainer` / `FreezeTrainer` / `OptimizeTrainer` 三个独立类。

**否决理由**：三档需要按层位置混合分配（一个模型可能同时用三档），独立类无法表达混合策略；类数量膨胀；合并 / fine-tune 逻辑重复。

### 方案 C：freeze 档用 fp16 替代 INT4

**描述**：freeze 档用 fp16 量化（而非 INT4）减少内存。

**否决理由**：fp16 是 16-bit，内存节省不如 INT4（4-bit）显著；fp16 不需要量化→反量化，但也失去了"模拟压缩"的语义（freeze 档的目的是模拟压缩后的冻结状态，INT4 更贴近实际部署形态）。

## 备注

- 本 ADR 是 Part5K1 "VMT 完整三档策略"的核心决策。
- VMT（Versenext Memory-aware Training）与 VMPC（[ADR-013](adr-013-vmpc-naming-v15.md)）品牌呼应，分别覆盖"训练期内存感知"与"压缩期参数压缩"。
- freeze 档的 INT4 量化复用 VMPC 的量化能力（[ADR-013](adr-013-vmpc-naming-v15.md)），保证训练期模拟与部署期压缩一致。
- optimize 档的层融合复用 Task 7 的 `_fused_forward_blocks`（64+ 层训练加速），梯度累积复用 `GradientAccumulator`。
- `VMTTrainer` 的合并阶段（`_vmt_merge`）需同时处理：加载 unload 档卸载分片 + 解冻 freeze 档分区 + 恢复 requires_grad 原始状态。
- 相关测试：`tests/test_vmt_trainer.py` 覆盖三档策略 / 策略解析 / freeze 量化恢复 / optimize 层融合 / 合并 / fine-tune。
- 相关文档：[Verse 训练指南 - VMT 三档策略指南](../training_guide.md)、[Verse 性能调优 - VMT 性能调优](../performance_tuning.md)
