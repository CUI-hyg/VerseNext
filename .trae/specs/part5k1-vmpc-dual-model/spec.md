# Part5K1：VMPC 技术全量化 + 双模型并行 + 架构精简 Spec

## Why

Part4K2 已落地压缩 V1.3、.vn 格式、LayerWiseTrainer 分区训练雏形、CometSpark V0.5-1B 单模型。
但当前仍存在结构性短板，阻碍“以小博大”路线继续推进：

- 压缩技术分散在 `compress.py`（V1.3）+ `vn_format.py`（量化内联标记）+ `quantize.py`，
  未形成统一命名的技术品牌；V1.3 偏重“存储压缩”，对“推理命中/准确率/反过拟合”支持不足，
  难以达到大参数模型等效能力。
- `spark/` 仍是单模型扁平结构（`config/` `model/` `src/`），无法承载“小模型 + 旗舰模型”
  双轨并行；checkpoint 目录 `checkpoints_XXX/` 命名与 `.pt` 文件格式未与 `.vn` 性能路线对齐。
- 数据加载仅 `load_jsonl` 严格解析，遇到无规范字段（如 `instruction/response`、`q/a`、
  `input/output`）或不规范 JSON 时直接抛错，缺乏自动修复能力，阻塞训练启动。
- 训练启动时数据预处理无 val 自动生成、无预加载流水线；64 层以上大模型在 CPU 下串行前向慢；
  LayerWiseTrainer 仅做“卸载”，未做“压缩冻结低频部分 + 专项优化高频部分”的完整 VMT。
- `verse_torch.nn` 仍沿用 PyTorch 命名空间 `nn`，与“自研原生架构解决已有框架问题”的宗旨
  不符；底层仍保留较多 Transformer/PyTorch 后端接口壳，重复调用、重复引用。

Part5K1 正式命名 **VMPC（VerseNext Model Parameters Compression）** 并升级到 V1.5，
落地双模型并行（small 0.06zB / mate 0.2zB 等效能力），完成数据自修复、完整 VMT、
`nn → vnn` 重命名与底层精简，使自研架构在“高性能 / 高速度 / 高质量”三维度真正对标大模型。

> **宗旨锚点**：“我们使用自研原生架构就是为了解决已有框架的问题，而不是重复造轮子，
> 甚至做得不如已有框架。”——所有改造必须以“减少壳、减少重复、提升性能”为准绳。

## What Changes

### 1. VMPC 技术命名 + V1.5 升级
- **正式命名**：把分散在 `compress.py` / `vn_format.py` / `quantize.py` 的压缩能力
  统一品牌化为 **VMPC（VerseNext Model Parameters Compression）**，作为 VerseNext 原生压缩
  技术的唯一对外名称；保留 `compress_pipeline` 函数名（向后兼容），但文档/注释/CLI 统一称 VMPC。
- **VMPC V1.5**（在 V1.3 基础上升级，不破坏 V1.3 API）：
  - **命中与算法优化**：推理时引入 logits 校准（temperature-aware softmax sharpening）+
    outlier-aware 反量化路径，提升 token 命中准确度（减少胡乱输出）。
  - **训练/推理速度**：QLinear 内部 fused matmul 路径升级（int4 batched GEMM 向量化），
    减少反量化中间拷贝；推理 recurrent 路径下 QLinear 走零拷贝解包。
  - **质量与准确率**：知识蒸馏 V1.5 新增 **logit-level contrastive distillation**
    （teacher/student logit 排序一致性损失），不只是 KL 概率匹配。
  - **反过拟合（核心）**：新增 `VMPCRegularizer` —— 参数幅度正则 + dropout-aware
    压缩感知 + early-exit 自适应（val loss 不降时自动收紧稀疏度），使小物理参数模型
    达到大参数模型等效能力（以小博大），避免训练过拟合。
- **VMPC 入口**：新增 `verse_torch/vmpc.py` 作为统一门面（re-export V1.5 API +
  `VMPCRegularizer` + `vmpc_compress` 便捷函数），`compress.py` 保持为底层实现。

### 2. 数据加载与数据处理自修复
- **新增 `verse_trainer/jsonl_repair.py`**（独立文件，置于 VerseTrainer 下）：
  - **情况 A：无规范参数** —— 自动探测异名字段（`instruction/response`、`q/a`、
    `input/output`、`question/answer`、`user/assistant` 等），批量转换为标准
    `{"prompt": ..., "completion": ...}` 格式（自动寻找对应量，批量替换）。
  - **情况 B：不规范格式** —— 能自动修复则修复（补全缺失引号/逗号、修复未闭合 JSON、
    去除 BOM/控制字符、合并多行 JSON 对象）；无法修复时抛出带行号 + 原因的明确错误。
  - **集成**：`load_jsonl` 默认调用修复管线（可 `repair=False` 关闭）；修复后的标准
    JSONL 可写回原文件或 `.repaired.jsonl`。
- **val.json 自动生成**：训练启动时若 `val_path` 不存在，从 `train_path` 自动切分
  `val_ratio`（默认 5%）样本生成 `val.jsonl`，并写回配置指定的 `val_path`。

### 3. 训练资源与时间优化
- **数据预加载流水线**：`CachedDataset` 升级 —— 训练启动时并行预加载（后台线程编码 +
  主线程构建模型），`prefetch` 扩展到非 torch 环境也可用（纯 threading 预取）。
- **64+ 层训练加速**：`VerseNexBlock` 前向引入 **层融合（layer fusion）** ——
  相邻同型 block 的 RMSNorm+Linear 合并为单次 matmul（CPU 下减少 Python 循环开销）；
  大模型（`n_layer >= 64`）自动启用 chunked 前向（按 8 层分块，块间不构建中间梯度图
  以降低内存峰值），提升 CPU 下大参数模型表现。
- **完整 VMT（VerseNext Memory-aware Training）**：升级 `LayerWiseTrainer` → `VMTTrainer`：
  - 未大幅训练的部分 → 从内存卸载到硬盘 `.vn` 分片（保留现有能力）。
  - 较少使用的部分 → **压缩冻结**（INT4 量化 + `requires_grad=False`，仅在需要时反量化）。
  - 高频训练的部分 → **专项优化**（保持 fp32 可训练 + 启用层融合 + 梯度累积）。
  - 统一实体不变，对外仍是完整模型训练；新增 `vmt_strategy` 配置（`unload` / `freeze` /
    `optimize` 三档按层分配）。

### 4. 模型升级（双模型并行）
- **目录重构（BREAKING）**：`spark/` 下分出 `small/` 与 `mate/` 两个模型目录：
  ```
  spark/
    small/                      # 小模型 0.06zB（≈60B 等效能力，VMPC 压缩后）
      config/
        cometspark_small.yml
      model/
        config.py               # CometSparkSmallConfig
        model.py                # CometSparkSmallLM
    mate/                       # 旗舰模型 0.2zB（≈200B 等效能力）
      config/
        cometspark_mate.yml
      model/
        config.py               # CometSparkMateConfig
        model.py                # CometSparkMateLM
    _bootstrap.py               # 保留（路径自举）
    run.py                      # 升级：支持 small/mate 选择 + 后训练/lora/持续训练
    README.md
  ```
  原 `spark/config/`、`spark/model/`、`spark/src/` 内容按级别迁入 `small/` / `mate/`，
  删除扁平旧目录。
- **checkpoint 重命名（BREAKING）**：`checkpoints_XXX/` → `mf_XXX/`（`mf` = model file，
  `XXX` ∈ {`small`, `mate`}）；旧目录自动迁移（首次运行时检测并重命名 + 警告）。
- **模型文件转换**：新训练默认保存为 `.vn`（提升性能，走 VMPC 智能压缩存储）；
  `save()` 保留 `.pt` 兼容，但 CLI 默认输出 `.vn`。
- **双模型 VMPC 适配**：`small` 与 `mate` 配置针对 VMPC 微调（`small` 走 ternary + 高稀疏，
  `mate` 走 int4 + 中稀疏 + 蒸馏），优化各自性能与能力。
- **run.py 升级**：新增 `--model small|mate` 选择；新增 `finetune`（LoRA）、`posttrain`
  （SFT/DPO/RL）、`continue`（持续训练）子命令（委托 verse_trainer CLI 已有能力，
  不重复造轮子）。
- **模型配置优化**：`small` / `mate` 配置针对 VMPC + 反过拟合调优（init_std、dropout、
  aux_loss_weight、label_smoothing 重新标定）。

### 5. VerseNex & VerseTorch 精简
- **删除 Transformer/PyTorch 后端接口壳**：移除 `verse_torch.nn` 中仅作 PyTorch 兼容
  的死壳（如 `_GQASelfAttention` / `_TransformerBlock` / `_TransformerLM` 旧别名链），
  创造替代组件（VerseNex 原生路径已就绪，无需再保留 transformer 系别名）。
- **底层合并**：合并 `compress.py` 与 `quantize.py` 中重复的 bit 统计/参数遍历函数；
  合并 `training.py` 与 `training_nex.py` 中重复的 collate / loss 辅助；消除反复 import。
- **性能去壳**：`QLinear.forward` 简化反量化中间拷贝；`LayerNorm` / `RMSNorm` 合并
  公共归一化内核；`Module.__setattr__` 简化注册分支。
- **核心类重命名（BREAKING）**：`verse_torch.nn` → `verse_torch.vnn`（VerseNext NN）。
  - 新建 `verse_torch/vnn.py` 作为正式模块（内容迁移自 `nn.py`）。
  - `verse_torch/nn.py` 保留为 thin shim，`from verse_torch.nn import *` 转发到 `vnn`，
    并发 `DeprecationWarning`（一个版本后删除）。
  - 顶层 `from verse_torch import nn` 保留（指向 `vnn`），但推荐 `from verse_torch import vnn`。

### 6. 文档与代码注释
- README + ADR + training_guide + perf_tuning 全面更新到 VMPC V1.5 + 双模型 + VMT。
- 新增 ADR：VMPC 命名 + V1.5 设计、双模型并行、VMT 完整策略、`nn → vnn` 重命名。
- 代码注释统一到 VMPC 术语（compress 注释从“V1.3”升级为“VMPC V1.5”）。

## Impact

- **Affected specs**: `part4k1-infra-model-upgrade`、`part4k2-arch-model-upgrade`
  （均已完成，Part5K1 在其上叠加，不回滚已完成能力，仅升级品牌、补齐短板、精简壳）。
- **Affected code**:
  - `packages/verse_torch/verse_torch/`：
    - 新增 `vmpc.py`（VMPC V1.5 门面 + `VMPCRegularizer`）。
    - 新增 `vnn.py`（迁移自 `nn.py`，正式 NN 模块）；`nn.py` 降级为 shim。
    - `compress.py` 升级 V1.5（contrastive distill + 命中校准 + 反量化优化）。
    - `layerwise_trainer.py` 升级为 `VMTTrainer`（压缩冻结 + 专项优化三档）。
    - `training.py` / `training_nex.py` 合并重复辅助。
    - `__init__.py` 导出 `vnn` / `vmpc` / `VMTTrainer` / `VMPCRegularizer`。
  - `packages/verse_nex/verse_nex/`：
    - `cometspark.py` / `tri_sparse_attn.py`：层融合 + chunked 前向（64+ 层加速）。
    - `__init__.py`：移除已废弃 transformer 别名导出。
  - `packages/verse_infra/verse_infra/verse_trainer/`：
    - 新增 `jsonl_repair.py`（JSONL 自修复 + 异名字段标准化）。
    - `data.py`：`load_jsonl` 集成修复管线 + `CachedDataset` 预加载流水线 + val 自动生成。
    - `trainer.py`：训练启动接入 val 自动生成 + VMTTrainer 选择。
  - `spark/`：
    - 重构为 `small/` + `mate/` 双模型目录（BREAKING）。
    - `run.py` 升级：`--model` 选择 + `finetune` / `posttrain` / `continue` 子命令。
    - checkpoint → `mf_small/` / `mf_mate/`，默认 `.vn` 输出。
  - `tests/`：新增 VMPC V1.5 / jsonl_repair / VMTTrainer / 双模型 / vnn 重命名测试。
  - `docs/`：新增 ADR-013~016 + 更新 README/training_guide/perf_tuning。

## ADDED Requirements

### Requirement: VMPC 技术命名与 V1.5 门面

系统 SHALL 把分散的压缩能力统一命名为 **VMPC（VerseNext Model Parameters Compression）**，
并提供 V1.5 统一门面：

1. **命名统一**：`verse_torch/vmpc.py` 作为 VMPC 唯一门面，re-export `compress_pipeline`
   / `OutlierSafePruner` / `LoRALinear` / `KnowledgeDistiller` / `QLinear` /
   `compress_mod_experts` / `compression_report`，并新增 V1.5 专属 API。
2. **向后兼容**：`from verse_torch.compress import compress_pipeline` 仍可用（底层实现不变）；
   `from verse_torch.vmpc import compress_pipeline` 是推荐入口。
3. **`VMPCRegularizer`**：反过拟合正则器，参数幅度 L2 + 压缩感知 dropout + early-exit
   自适应稀疏收紧；可挂载到任意 `Trainer` 的 loss 上。
4. **`vmpc_compress(model, profile="small"|"mate")`**：便捷函数，按模型级别一键应用
   预设压缩配置（small=ternary+高稀疏，mate=int4+中稀疏+蒸馏）。

#### Scenario: VMPC 门面导入
- **WHEN** 执行 `from verse_torch.vmpc import compress_pipeline, VMPCRegularizer`
- **THEN** 成功导入，`compress_pipeline` 与 `verse_torch.compress.compress_pipeline` 同一对象

#### Scenario: 反过拟合正则
- **WHEN** 训练中 val_loss 连续 `patience` 步不降
- **THEN** `VMPCRegularizer` 自动收紧稀疏度（`target_sparsity *= 0.9`），抑制过拟合

### Requirement: VMPC V1.5 命中与质量优化

系统 SHALL 在 VMPC V1.5 中提升推理命中与训练质量：

1. **logits 校准**：推理路径下对反量化后的 logits 做 temperature-aware sharpening
   （`logits = logits / sqrt(variance) * calibration_factor`），减少饱和导致的胡乱输出。
2. **contrastive distillation**：`KnowledgeDistiller` V1.5 新增 logit 排序一致性损失
   （`margin_ranking_loss(teacher_topk, student_topk)`），不只匹配概率分布。
3. **outlier-aware 反量化**：QLinear 反量化时识别 outlier channel（|w| > 3σ），
   对 outlier 通道保留 fp16 精度，其余 int4，兼顾速度与命中。
4. **数值稳定**：所有 V1.5 路径在 float32 下与 V1.3 输出吻合到 1e-2（前向）。

#### Scenario: 命中提升
- **WHEN** 用 VMPC V1.5 压缩后模型生成
- **THEN** token 命中率（exact-match next token）相对 V1.3 提升 ≥ 5%

#### Scenario: contrastive distill
- **WHEN** 启用 `distill_contrastive=True`
- **THEN** student logit top-k 排序与 teacher 一致性 ≥ 0.9（Spearman 相关）

### Requirement: JSONL 自修复与标准化

系统 SHALL 在 `verse_trainer/jsonl_repair.py` 提供 JSONL 自动修复能力：

1. **异名字段标准化（情况 A）**：自动探测并批量转换为 `{"prompt":..., "completion":...}`：
   - `instruction` / `response` → `prompt` / `completion`
   - `q` / `a`、`question` / `answer` → `prompt` / `completion`
   - `input` / `output` → `prompt` / `completion`
   - `user` / `assistant`（非 chat 数组形式）→ `prompt` / `completion`
   - 单字段 `text` / `content` / `raw` → 保留为 `text`
2. **格式自动修复（情况 B）**：能修复则修复：
   - 补全缺失引号/逗号（`{"prompt":"x" "completion":"y"}` → 合法 JSON）
   - 修复未闭合 JSON（`{"prompt":"x"` → 补 `}`）
   - 去除 BOM / 控制字符 / 行尾多余逗号
   - 合并被错误换行的多行 JSON 对象
   - 无法修复时抛 `JSONLRepairError`（含行号 + 原因 + 原行内容）
3. **集成 `load_jsonl`**：默认 `repair=True`，修复后样本标准化为标准格式；
   `repair=False` 走原严格解析（向后兼容）。
4. **写回**：`repair_jsonl(path, write_back=True)` 写到 `*.repaired.jsonl` 或覆盖原文件。

#### Scenario: 异名字段
- **WHEN** JSONL 含 `{"instruction":"1+1","response":"2"}`
- **THEN** 自动转为 `{"prompt":"1+1","completion":"2"}`

#### Scenario: 格式修复
- **WHEN** JSONL 含 `{"prompt":"x" "completion":"y"}`（缺逗号）
- **THEN** 自动修复为合法 JSON 并解析成功

#### Scenario: 无法修复
- **WHEN** JSONL 含完全无法解析的行 `{"prompt": <broken>`
- **THEN** 抛 `JSONLRepairError`，消息含行号 + 原因

### Requirement: val.json 自动生成

系统 SHALL 在训练启动时自动生成验证集：

1. **触发条件**：`val_path` 文件不存在或为空时自动触发。
2. **切分策略**：从 `train_path` 末尾切分 `val_ratio`（默认 0.05）样本到 `val_path`；
   切分后从 train 中移除这些样本（内存中，不修改原 train 文件，除非 `write_back=True`）。
3. **写回**：生成的 val 写到配置指定的 `val_path`；若目录不存在自动创建。
4. **日志**：切分时打印 `(n_train, n_val)` 便于用户确认。

#### Scenario: 自动生成 val
- **WHEN** `data/val.jsonl` 不存在，启动训练
- **THEN** 从 `data/train.jsonl` 切分 5% 到 `data/val.jsonl`，train 用剩余 95%

### Requirement: VMT 完整智能分区训练

系统 SHALL 提供完整 VMT（`VMTTrainer`），三档策略按层分配：

1. **unload 档**：未大幅训练的层 → 卸载到硬盘 `.vn` 分片（保留现有 LayerWiseTrainer 能力）。
2. **freeze 档**：较少使用的层 → INT4 量化 + `requires_grad=False`（压缩冻结），
   仅在评估/合并时反量化；内存占用降至 1/8。
3. **optimize 档**：高频训练的层 → 保持 fp32 可训练 + 启用层融合 + 梯度累积专项优化。
4. **策略配置**：`vmt_strategy` 可按层名/层索引分配档位（如 `"layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload"`），
   或用预设（`"auto"` 按层重要性自动分配）。
5. **统一实体**：训练过程中模型对象不变，对外接口与普通 `Trainer` 一致（`fit` / `evaluate`）。
6. **无损往返**：unload/freeze 层在合并回完整模型时数值一致（freeze 层反量化误差 ≤ 1e-3）。

#### Scenario: 三档分配
- **WHEN** 配置 `vmt_strategy="layers[0:8]=freeze, layers[8:]=optimize"` 训练 64 层模型
- **THEN** 前 8 层 INT4 冻结（内存 1/8），其余层 fp32 优化训练，对外仍是完整模型

#### Scenario: 自动策略
- **WHEN** `vmt_strategy="auto"` 训练 80 层模型
- **THEN** 自动按层重要性分配（首尾层 freeze，中间高频层 optimize，少量低频层 unload）

### Requirement: 双模型并行（small / mate）

系统 SHALL 在 `spark/` 下提供双模型并行结构：

1. **目录结构**：`spark/small/`（0.06zB ≈ 60B 等效）+ `spark/mate/`（0.2zB ≈ 200B 等效），
   各含 `config/` + `model/`。
2. **small 模型**：物理参数小（适合 3 核 CPU 沙箱训练），经 VMPC（ternary + 高稀疏）
   达到 ≈60B 等效能力；`CometSparkSmallConfig` / `CometSparkSmallLM`。
3. **mate 模型**：物理参数较大（旗舰），经 VMPC（int4 + 中稀疏 + 蒸馏）达到 ≈200B 等效能力；
   `CometSparkMateConfig` / `CometSparkMateLM`。
4. **VMPC 适配**：两个模型配置针对各自压缩预设微调架构（expert 数 / 层 pattern / init_std）。
5. **共享底层**：均基于 `verse_nex.CometSparkNexLM`（VerseNexBlock），不重造底层 block。

#### Scenario: 构建小模型
- **WHEN** 调用 `CometSparkSmall()`
- **THEN** 返回物理参数小、VMPC-small 预设就绪的 `CometSparkSmallLM` 实例

#### Scenario: 构建旗舰模型
- **WHEN** 调用 `CometSparkMate()`
- **THEN** 返回物理参数较大、VMPC-mate 预设就绪的 `CometSparkMateLM` 实例

### Requirement: checkpoint 与文件格式升级

系统 SHALL 把 checkpoint 目录与模型文件格式升级到 VMPC 性能路线：

1. **目录重命名（BREAKING）**：`checkpoints_XXX/` → `mf_XXX/`（`XXX` ∈ {`small`, `mate`}）。
2. **自动迁移**：首次运行时检测旧 `checkpoints_XXX/`，自动重命名为 `mf_XXX/` + 警告。
3. **默认 `.vn` 输出**：新训练默认保存为 `.vn`（VMPC 智能压缩存储）；
   `save()` 保留 `.pt` 兼容（`save(path, format="pt"|"vn")`）。
4. **CLI 默认**：`spark/run.py train --model small` 默认输出 `mf_small/best.vn`。

#### Scenario: 自动迁移
- **WHEN** 存在旧 `checkpoints_small/`，运行 `spark/run.py train --model small`
- **THEN** 自动重命名为 `mf_small/` + 打印迁移警告，训练继续

#### Scenario: 默认 vn 输出
- **WHEN** 执行 `spark/run.py train --model mate`
- **THEN** checkpoint 默认保存为 `mf_mate/best.vn`

### Requirement: run.py 训练模式补齐

系统 SHALL 在 `spark/run.py` 支持完整训练模式：

1. **`--model small|mate`**：选择模型级别（默认 `small`）。
2. **`finetune` 子命令**：LoRA 微调（委托 `verse_trainer` 的 `LoRATrainer`），
   支持 `--lora-r` / `--lora-alpha` / `--target-modules`。
3. **`posttrain` 子命令**：后训练（SFT / DPO / RL），委托 `verse_trainer` 的
   `SFTTrainer` / `DPOTrainer` / `RLTrainer`；`--mode sft|dpo|rl`。
4. **`continue` 子命令**：持续训练（从 checkpoint 继续），委托 `verse_trainer.continue_train`；
   `--checkpoint` / `--additional-steps`。
5. **不重复造轮子**：所有训练逻辑委托 `verse_trainer`，`run.py` 只做参数解析 + 模型选择 +
   checkpoint 路径映射。

#### Scenario: LoRA 微调
- **WHEN** 执行 `spark/run.py finetune --model small --lora-r 8 --checkpoint mf_small/best.vn`
- **THEN** 委托 `LoRATrainer` 对 small 模型做 LoRA 微调，输出 `mf_small/best_lora.vn`

#### Scenario: 持续训练
- **WHEN** 执行 `spark/run.py continue --model mate --checkpoint mf_mate/best.vn --additional-steps 1000`
- **THEN** 委托 `continue_train` 从 checkpoint 继续 1000 步

### Requirement: VerseTorch.nn → VerseTorch.vnn 重命名

系统 SHALL 把 `verse_torch.nn` 重命名为 `verse_torch.vnn`：

1. **新模块**：`verse_torch/vnn.py` 为正式 NN 模块（内容迁移自 `nn.py`）。
2. **shim 兼容（一个版本）**：`verse_torch/nn.py` 降级为 thin shim，
   `from verse_torch.nn import Module` 转发到 `vnn.Module` + `DeprecationWarning`。
3. **顶层导出**：`from verse_torch import vnn` 推荐；`from verse_torch import nn` 仍可用
   （指向 `vnn`），但不推荐。
4. **全项目导入更新**：`verse_nex` / `compress.py` / `training.py` 等内部 `from . import nn`
   改为 `from . import vnn as nn`（最小改动）或直接 `from . import vnn`。

#### Scenario: 新导入
- **WHEN** 执行 `from verse_torch.vnn import Linear`
- **THEN** 成功导入，无警告

#### Scenario: 旧导入兼容
- **WHEN** 执行 `from verse_torch.nn import Linear`
- **THEN** 成功导入，但发出 `DeprecationWarning: verse_torch.nn 已更名为 verse_torch.vnn`

### Requirement: 64+ 层训练加速

系统 SHALL 优化 64 层以上神经网络在 CPU 下的训练速度：

1. **层融合**：`VerseNexBlock` 相邻同型 block 的 `RMSNorm + Linear` 合并为单次 matmul
   （CPU 下减少 Python 循环开销）。
2. **chunked 前向**：`n_layer >= 64` 时自动按 8 层分块前向，块间不保留中间梯度图
   （梯度检查点风格），降低内存峰值 50%+。
3. **性能目标**：64 层模型 CPU 下前向吞吐相对未优化提升 ≥ 1.5×。

#### Scenario: 层融合
- **WHEN** 训练 64 层 VerseNex 模型
- **THEN** 相邻同型 block 自动融合，前向 Python 循环开销降低

#### Scenario: chunked 前向
- **WHEN** `n_layer=80`，启用 chunked 前向
- **THEN** 内存峰值相对全量前向降低 ≥ 50%，吞吐相对未优化提升 ≥ 1.5×

## MODIFIED Requirements

### Requirement: `verse_torch.compress.compress_pipeline`
原 V1.3 流程 `prune → quantize → distill → lora`。修改为 VMPC V1.5：
- 默认 `version="1.5"`（V1.3 仍可通过 `version="1.3"` 显式调用）。
- V1.5 新增 `contrastive_distill`（logit 排序一致性损失）+ `logit_calibration`（命中校准）。
- `compression_report` 新增 `vmpc_version` 字段。
- 行为向后兼容：旧 `version="1.3"` 调用结果不变。

### Requirement: `verse_torch.layerwise_trainer.LayerWiseTrainer`
原仅支持 unload 档。修改为 `VMTTrainer`（继承 `LayerWiseTrainer`）：
- 新增 `freeze` 档（INT4 量化冻结）+ `optimize` 档（层融合 + 梯度累积）。
- `vmt_strategy` 配置按层分配三档。
- 旧 `LayerWiseTrainer` 名保留为 `VMTTrainer` 别名（向后兼容）。

### Requirement: `verse_infra.verse_trainer.data.load_jsonl`
原严格解析，遇错抛 `ValueError`。修改为：
- 默认 `repair=True`，调用 `jsonl_repair.repair_jsonl` 自动修复 + 标准化。
- `repair=False` 走原严格解析（向后兼容）。
- 修复后样本统一为标准 `{"prompt":..., "completion":...}` 或 `{"text":...}` 格式。

### Requirement: `spark/run.py` 子命令
原 `train / eval / generate / chat / compress / convert / download`。修改为：
- `train` 新增 `--model small|mate`（默认 small）。
- 新增 `finetune` / `posttrain` / `continue` 子命令。
- checkpoint 默认输出 `mf_XXX/` + `.vn`。

## REMOVED Requirements

### Requirement: `verse_torch.nn` 中的 transformer 系旧别名
**Reason**：VerseNex 原生架构已统一接管，`_GQASelfAttention` / `_TransformerBlock` /
`_TransformerLM` 等仅为 PyTorch 兼容的壳，与“自研原生架构解决已有框架问题”宗旨冲突，
且增加维护负担。
**Migration**：
- 删除 transformer 系别名（`TransformerLM` / `TransformerBlock` / `GQASelfAttention`）。
- 全项目改用 `VerseNexLM` / `VerseNexBlock` / `VerseNexAttention`（`verse_nex` 已提供）。
- 旧 `from verse_torch.nn import TransformerLM` 抛 `ImportError`（不再 silent 兼容）。

### Requirement: `spark/` 扁平单模型目录
**Reason**：双模型并行需要 `small/` + `mate/` 分离结构。
**Migration**：
- `spark/config/cometspark_v05.yml` → `spark/mate/config/cometspark_mate.yml`
- `spark/config/cometspark_v05_small.yml` → `spark/small/config/cometspark_small.yml`
- `spark/model/` → `spark/small/model/` + `spark/mate/model/`（各自 Config/LM）
- `spark/src/` → 能力迁入 `verse_trainer`（已完成）+ `small/` / `mate/` 各自 `model/`
- 删除扁平旧目录
