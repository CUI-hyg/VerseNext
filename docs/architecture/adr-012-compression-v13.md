# ADR-012: 压缩技术 V1.3

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k2-arch-model-upgrade/spec.md`](../../../.trae/specs/part4k2-arch-model-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)（量化是一等公民）、[ADR-008 超稀疏并行注意力](adr-008-parallel-sparse-attention.md)
- **相关 ADR**：[ADR-011 智能分区训练](adr-011-layerwise-training.md)（压缩后小模型可配合分区训练）

## 上下文

Part4K2 之前，`verse_torch.compress` 已提供压缩管线 v1 / v2（`OutlierSafePruner` + `QuantizedLinear` + `LoRALinear` + `KnowledgeDistiller` + `compress_pipeline`），但存在以下局限：

1. **蒸馏损失单薄**：`KnowledgeDistiller` V1.0 仅用 Hinton KL 软标签蒸馏（`alpha * T^2 * KL(teacher/T || student/T) + (1-alpha) * CE`），缺乏中间层特征匹配，大模型→小模型的能力转移效果有限（小模型容易"记住分布"但"丢失能力"）。
2. **温度固定**：蒸馏温度 `T` 全程固定，前期软标签不够平滑（信息量不足）、后期不够尖锐（收敛慢），无法兼顾"前期学分布 / 后期收敛"两阶段需求。
3. **管线流程次优**：v2 流程为 `prune → quantize → lora`（蒸馏作为可选项且在 lora 之后），蒸馏在 LoRA 包装后进行会导致 student 模型梯度路径被 LoRA 增量层打断，蒸馏信号难以传到 base 层。
4. **缺乏吞吐率优化**：量化后 `QLinear` 未走 fused matmul 路径，INT4 权重的访存优势未充分转化为吞吐率提升。
5. **teacher 配置繁琐**：v2 要求 teacher 放在 `config["distill"]["teacher"]` 嵌套字段，用户需多层级配置；缺乏顶层便捷入口。
6. **缺乏压缩报告**：v2 stats 仅返回参数量 / 压缩比，缺乏吞吐率提升估算与标准化报告。

同时需要保持向后兼容：v1 / v2 的 `compress_pipeline` 接口与 stats 结构不能破坏；`KnowledgeDistiller` 的 `T` 参数别名需保留。

## 决策

**推出压缩管线 V1.3：`KnowledgeDistiller` 升级三重损失 + 自适应温度退火；`compress_pipeline` V1.3 流程重排为 `prune → quantize → distill → lora` + 吞吐率优化（fused matmul + batch 量化）；新增 `compression_report` 标准化报告与 teacher 便捷入口。**

具体含义：

1. **`KnowledgeDistiller` V1.3 三重损失**：
   ```
   Loss = alpha * T^2 * KL(teacher/T || student/T)              # 软标签蒸馏
        + (1 - alpha) * CE(student, labels)                     # 硬标签蒸馏
        + feature_loss_weight * MSE(student_feat, teacher_feat) # 中间层特征匹配
   ```
   - `alpha` 默认 0.7（软标签权重），`feature_loss_weight` 默认 0.3。
   - 中间层特征通过 `feature_extractor` 回调提取（teacher 特征 `detach()` 不回传梯度）；student / teacher 特征维度不一致时截断到较小者（`_align_features`）。
   - 无硬标签时软标签全权（`total = alpha * soft_loss`）；无特征时跳过特征项。

2. **自适应温度退火（temperature annealing）**：
   - 训练过程中温度从 `T_init`（用户设定，默认 4.0）线性退火到 `T_min = max(1.0, T_init * 0.25)`。
   - 前期高温：软标签平滑，信息量大，student 学 teacher 的分布。
   - 后期低温：分布尖锐，收敛到硬标签附近。
   - `distill(epochs, lr, anneal_temperature=True)` 默认启用；`epochs > 1` 时按 `frac = epoch / (epochs - 1)` 线性插值。

3. **`compress_pipeline` V1.3 流程重排**：
   - 新流程：`prune → quantize → distill → lora`（v2 为 `prune → quantize → lora`，蒸馏可选且在 lora 后）。
   - **重排理由**：蒸馏在量化之后（先量化减存储）、LoRA 包装之前（蒸馏信号直达 base 层，不被 LoRA 增量层打断），最后 LoRA 包装为微调准备（QLoRA 风格）。
   - 通过 `version="1.3"` 参数选择 V1.3 路径（`_compress_pipeline_v13`）；`version` 默认行为向后兼容（v1 / v2 路径保留）。

4. **吞吐率优化**：
   - 量化后 `QLinear` 内部走 fused matmul 路径（`matmul_int4`），INT4 权重的访存优势转化为吞吐率提升。
   - batch 量化：量化时对 batch 维度统一处理，减少 Python 循环开销。
   - `stats` 中 `steps[].fused_matmul = True` 标记是否启用 fused 路径。

5. **teacher 便捷入口**：
   - `config` 顶层可直接放 `teacher_model` / `teacher` / `train_loader`（`_resolve_teacher` 解析）。
   - `distill` 子配置优先级更高（`d_cfg.get("teacher", teacher)`）。
   - 无 `train_loader` 时仅冻结 teacher、就绪 student（不端到端蒸馏）。

6. **`compression_report` 标准化报告**：
   - `compression_report(model, compressed_model)` 返回 dict：`original_params` / `compressed_params` / `compression_ratio` / `sparsity` / `bits_per_param` / `estimated_throughput_improvement` / `version`。
   - 吞吐率估算：INT4 ≈ 4×、INT8 ≈ 2×、ternary ≈ 8×（相对 fp32 的权重访存）。
   - V1.3 的 `stats` 内嵌 `compression_report` 字段（与 v2 stats 同构 + V1.3 专属字段）。

7. **VerseNex 集成**：
   - `CometSparkNexLM.compress_v13(config)`：实例方法，调用 `compress_pipeline(self, config, version="1.3")`。
   - `CometSparkNexLM.distill_from(teacher, train_data)`：实例方法，构造 `KnowledgeDistiller(teacher, self)` 端到端蒸馏。

8. **向后兼容**：
   - `KnowledgeDistiller` 的 `T` 参数保留为 `temperature` 的旧别名（仅当显式传入非 None 时覆盖 `temperature`）。
   - `compress_pipeline(version=None)` 默认走 v1 / v2 路径；`version="1.3"` 走 V1.3。
   - v2 的 `stats` 结构在 V1.3 中保留（同名同义键），仅追加 V1.3 专属字段。

## 后果

### 优点

- **能力转移更强**：三重损失（软标签 + 硬标签 + 特征匹配）比单一 KL 蒸馏更能保留 teacher 的能力，小模型在压缩比相同的情况下质量更高。
- **温度自适应**：退火调度兼顾"前期学分布 / 后期收敛"两阶段，无需手动调温度。
- **流程更优**：蒸馏在 LoRA 之前，信号直达 base 层；LoRA 包装为后续微调准备（QLoRA 风格）。
- **吞吐率提升**：fused matmul 把 INT4 访存优势转化为实际加速（估算 4×）。
- **配置便捷**：teacher 顶层字段 + `distill` 子配置双层入口，降低使用门槛。
- **报告标准化**：`compression_report` 提供参数量 / 压缩比 / 吞吐率估算，便于选型决策。
- **向后兼容**：v1 / v2 路径与 `T` 别名保留，现有代码零修改。

### 缺点

- **蒸馏训练耗时**：三重损失 + 温度退火比单一 KL 计算量大（特征匹配需前向 teacher + student 中间层）；但这是离线一次性成本。
- **feature_extractor 回调约定**：用户需自行实现特征提取回调（`distill_layers` 目前仅作元数据记录），有一定集成成本。
- **V1.3 路径需显式选择**：`version="1.3"` 需用户显式指定，默认仍走 v2（避免破坏现有行为）。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 三重损失权重失衡（alpha / feature_loss_weight 不当） | 默认值（0.7 / 0.3）经实验调优；用户可按需调整 |
| 温度退火到 T_min 仍不够尖锐 | `T_min = max(1.0, T_init * 0.25)` 保底；用户可调 `T_init` |
| 特征维度不匹配 | `_align_features` 截断到较小者；用户可通过 `feature_extractor` 自定义对齐 |
| V1.3 与 v2 行为差异导致回归 | `version` 参数显式选择；V1.3 stats 与 v2 同构 + 追加字段；测试覆盖两路径 |

## 替代方案（已否决）

### 方案 A：仅升级 `KnowledgeDistiller`，不重排管线

**描述**：保留 v2 流程（`prune → quantize → lora`），仅升级蒸馏损失为三重。

**否决理由**：蒸馏在 LoRA 之后进行，信号被 LoRA 增量层打断，三重损失的效果被削弱；无法实现 QLoRA 风格的"蒸馏 + LoRA 微调准备"组合。

### 方案 B：强制 V1.3 为默认路径（破坏性升级）

**描述**：`compress_pipeline` 默认走 V1.3，移除 v1 / v2 路径。

**否决理由**：破坏现有用户代码（v2 stats 结构 / 行为）；违反向后兼容承诺；V1.3 的蒸馏训练成本可能不适用于所有场景。

### 方案 C：引入外部蒸馏库（如 `textbrewer` / `doudou`）

**描述**：不自研三重损失，依赖外部蒸馏库。

**否决理由**：外部库多为 PyTorch 生态，与 Verse 的 NumPy 后端不兼容；引入重型依赖违反 ADR-001；现有 `KnowledgeDistiller` 已有基础，增量升级成本低。

## 备注

- 本 ADR 是 Part4K2 "大模型→小模型能力转移"的核心决策。
- V1.3 的"以小博大"路线：大模型（teacher）→ 剪枝 + 量化 → 蒸馏 → LoRA 微调 → 小模型（student），对应 `verse_data/designs/compression_pipeline_design.md` 的 trillion → billion 路线图。
- `KnowledgeDistiller` 的 `T` 别名模式借鉴自 ADR-010 的 `temperature` 参数设计。
- 相关测试：`tests/test_compress_v13.py` 覆盖三重损失 / 温度退火 / 管线重排 / 压缩报告 / VerseNex 集成。
- 相关文档：[Verse 性能调优 - 压缩技术 V1.3 调优](../performance_tuning.md)
