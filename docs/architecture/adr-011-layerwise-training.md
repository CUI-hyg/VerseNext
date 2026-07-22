# ADR-011: 智能分区训练

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k2-arch-model-upgrade/spec.md`](../../../.trae/specs/part4k2-arch-model-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)、[ADR-005 GPU/NPU 后端](adr-005-gpu-npu-backend.md)
- **相关 ADR**：[ADR-009 .vn 文件格式](adr-009-vn-format.md)（分区卸载复用 `.vn` 分片）、[ADR-004 CPU 并行](adr-004-cpu-parallel.md)（分区训练与 chunk 并行互补）

## 上下文

Part4K2 的目标之一是让 CometSpark V0.5-1B（约 1.12B 参数）这类大模型能在**有限内存的 CPU / 单卡 GPU** 上完成训练。现有的 `ParallelTrainer` / `ParallelTrainerSafe` 虽然支持 `--parallel-chunks N` 分块训练与 OOM 兜底，但其并行粒度是**时间维度**（把 `max_steps` 拆成 N 个 chunk 顺序执行），**模型参数始终全量驻留内存**。对于 1B 参数模型：

1. **内存墙**：1B 参数 fp32 ≈ 4GB 权重 + 优化器状态（AdamW 两阶矩 ≈ 8GB）+ 激活值，峰值内存轻松超过 12GB，消费级 CPU / 入门级 GPU 无法承载。
2. **梯度计算浪费**：传统全量训练每个 step 对所有层计算梯度，但深层网络的底层（embedding 附近）与顶层（lm_head 附近）梯度分布差异大，全量同步更新并非最优。
3. **现有分块不解决参数驻留**：`--parallel-chunks` 只拆 step 不拆参数，OOM 时仅缩小 batch，无法把已训练的层卸载到硬盘。
4. **ZeRO / FSDP 等方案过重**：工业界的 ZeRO-3 / FSDP 需要分布式集群 + 通信库，与 Verse 的"单机 CPU 优先"定位冲突。

同时需要满足：
- **统一实体**：训练过程中模型对象不变（外部代码看到的始终是完整模型），不能要求用户手动拼装分片。
- **无损往返**：卸载 / 加载参数数值完全一致（复用 ADR-009 的 `.vn` 格式）。
- **与现有 Trainer 接口对齐**：`fit(train_loader, val_loader, max_steps)` 返回 `(train_losses, val_losses)`。

## 决策

**实现 `LayerWiseTrainer`：将模型按 layer 分组训练，训完一组卸载到硬盘 `.vn` 分片，保持统一实体；内存超阈值时自动卸载已训练的非当前组；全部组训练完成后合并所有分片为完整模型，可选整体 fine-tune。**

具体含义：

1. **按 layer 分组**：
   - 模型需有 `.blocks` 属性（如 `VerseNexLM` / `CometSparkNexLM` 的 `ModuleList`）。
   - 按 `partition_size`（默认 2）分组，每组包含 `partition_size` 个 block。
   - `embedding` / `lm_head` / `norm` 等非 block 参数**始终在内存中且可训练**（不参与分区卸载），保证 loss 能持续下降。

2. **逐组训练 + 冻结**：
   - 训练当前组时，其他组 blocks 的 `requires_grad=False`（不参与梯度计算），当前组 + 非 block 参数 `requires_grad=True`。
   - 每组训练 `max_steps // n_partitions` 步（余数均摊到前几个分区）。
   - 用 `AdamW` 优化器仅收集 `requires_grad=True` 的参数。

3. **`.vn` 分片卸载**（复用 ADR-009）：
   - 训完一组用 `VNFileWriter` 写到 `offload_dir/partition_{idx}.vn`（`arch="layerwise"`，config 含 `partition` / `block_indices` / `partition_size`）。
   - 卸载后默认**不置零**（保持 forward 正确性）；内存压力场景可 `zero=True` 把该组参数置零释放内存（置零后该组需重新加载才能用）。
   - 加载用 `VNFileReader.read_weights()` + `model.load_state_dict(sd, strict=False)`（仅加载该组 block 参数）。

4. **内存监控 + 自动卸载**：
   - 调用 `get_memory_info("cpu")`（Part4K2 Task 5 实现）读取当前内存使用。
   - 超过 `memory_threshold_mb`（默认 512MB）时，把已训练的非当前组参数卸载到硬盘（若尚未卸载）。
   - 每组训练过程中按 `max_steps // 4` 间隔检查内存。

5. **合并 + 可选 fine-tune**：
   - 全部组训练完成后，`_merge_partitions` 从硬盘加载所有已卸载的分区参数到模型，恢复完整状态。
   - 恢复所有参数的 `requires_grad` 原始状态（训练前快照）。
   - 可选 `finetune_steps`（config 字段，默认 0）：合并后整体 fine-tune N 步（全部参数可训练，`lr * 0.5`），弥合分组训练的层间边界。

6. **统一实体**：训练过程中 `self.model` 对象不变，只是内部参数在内存 / 硬盘之间备份；对外接口与普通 `Trainer` 一致（`fit` / `evaluate`）。

7. **CLI 集成**：`verse-train --partition-training --partition-size N --max-steps 1000`（`--partition-training` 开关在 `verse_trainer.cli` 中路由到 `LayerWiseTrainer`）。

## 后果

### 优点

- **低内存训练大模型**：1B 模型可按 `partition_size=2` 拆成 10 组（n_layer=20），单组训练时仅当前组 + embedding/lm_head 在内存，峰值内存显著下降。
- **统一实体**：外部代码无需感知分区，模型对象始终完整，`fit` 接口与 `Trainer` 一致。
- **无损往返**：卸载 / 加载用 `.vn` 分片（safetensors / npz），数值完全一致。
- **内存自适应**：`memory_threshold_mb` 触发自动卸载，无需用户手动调参；`get_memory_info` 跨平台（CPU / GPU）。
- **可选 fine-tune**：`finetune_steps` 弥合分组训练的层间边界，提升最终模型质量。
- **与 ADR-001 一致**：仅依赖 NumPy + 标准库 + VerseTorch 现有组件，无分布式 / 通信库依赖。

### 缺点

- **训练速度折损**：逐组训练相当于层间串行，总步数相同时各组训练步数减少；但通过 fine-tune 弥合，且内存节省带来的"能跑起来"价值大于速度折损。
- **层间依赖**：当前组训练时其他组冻结，底层层无法及时响应顶层变化；`finetune_steps` 缓解但不能完全消除。
- **`.blocks` 属性要求**：模型需有 `.blocks`（`ModuleList`），非 block 化模型（如纯 SSM）不适用。
- **硬盘 I/O**：卸载 / 加载分片有 I/O 开销，但 `.vn` 的 safetensors mmap 路径读取较快。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 分组训练精度低于全量训练 | `finetune_steps` 整体微调弥合；`partition_size` 可调（越小越接近全量） |
| 内存阈值设置不当（频繁卸载 / 不触发） | 默认 512MB 适用于多数 CPU；用户可通过 `memory_threshold_mb` 调整 |
| 卸载目录磁盘空间不足 | `offload_dir` 可指定到大盘；`cleanup()` 清理（仅自动创建目录时） |
| 模型无 `.blocks` 属性 | 构造时抛 `ValueError` 明确提示（支持 VerseNexLM / CometSparkNexLM） |

## 替代方案（已否决）

### 方案 A：ZeRO-3 / FSDP 分布式分片

**描述**：引入分布式训练框架，把参数 / 梯度 / 优化器状态分片到多卡。

**否决理由**：需要多卡集群 + 通信库（NCCL / gloo）；与 Verse 的"单机 CPU 优先"定位冲突；部署门槛高。

### 方案 B：梯度检查点（activation checkpoint）Only

**描述**：仅用激活检查点（`use_checkpoint=True`）节省激活内存，不卸载参数。

**否决理由**：激活检查点只省激活内存，不省参数 / 优化器状态内存；1B 模型的 12GB 参数+优化器内存仍超限。激活检查点已作为 `VerseNexBlock` 的补充选项（Task 5.2）。

### 方案 C：CPU offload 全量参数到硬盘（非分区）

**描述**：把整个 `state_dict` 卸载到硬盘，每个 step 全量加载 / 卸载。

**否决理由**：每个 step 全量 I/O 开销巨大；不解决"当前训练层"的内存驻留问题；无分组训练的层间聚焦优势。

### 方案 D：逐层冻结（freeze-and-train）不卸载

**描述**：逐层冻结训练，但不卸载已训练层到硬盘。

**否决理由**：参数仍全量驻留内存，不解决内存墙；LayerWiseTrainer 的卸载能力是核心价值。

## 备注

- 本 ADR 是 Part4K2 "大模型低内存训练"的核心决策。
- `LayerWiseTrainer` 与 `ParallelTrainerSafe`（Part4K1）互补：前者拆参数（空间维度），后者拆 step（时间维度），可组合使用。
- `partition_size` 推荐值：CPU 8GB 内存 → `partition_size=2`；16GB → `partition_size=4`；GPU 24GB → `partition_size=8` 或不用分区训练。
- 相关测试：`tests/test_layerwise_trainer.py` 覆盖分区 / 卸载 / 加载 / 合并 / fine-tune / 内存监控。
- 相关文档：[Verse 训练指南 - 智能分区训练指南](../training_guide.md)、[Verse 性能调优 - 智能分区训练性能调优](../performance_tuning.md)
