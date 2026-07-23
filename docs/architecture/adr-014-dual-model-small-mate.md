# ADR-014: 双模型并行（small / mate）

- **状态**：Accepted
- **日期**：2026-07-23
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：Part5K1 升级任务集
- **前置 ADR**：[ADR-013 VMPC 命名 + V1.5 设计](adr-013-vmpc-naming-v15.md)（small / mate 预设由 `vmpc_compress` 提供）、[ADR-009 .vn 文件格式](adr-009-vn-format.md)（默认输出 `.vn`）
- **相关 ADR**：[ADR-015 VMT 完整三档策略](adr-015-vmt-full-strategy.md)（双模型训练可配合 VMT 分区）

## 上下文

Part5K1 之前，CometSpark 训练仓库（`spark/`）只有单一模型规格（V0.5-1B 主力 + small 调试配置），通过同一套 `config.yml` 切换规模。Part5K1 进入"双模型并行"阶段后，需要同时维护两个定位差异明显的模型：

1. **small 模型**：0.06zB 极小模型，面向端侧 / 嵌入式 / 树莓派部署，追求极致压缩比与低内存推理。需要 ternary 量化（1.58-bit）+ 高稀疏剪枝，对蒸馏质量要求相对宽松（小模型容量有限）。
2. **mate 模型**：0.2zB 旗舰模型，面向消费级 CPU / 单卡 GPU 推理，追求能力上限。需要 int4 量化 + 中等稀疏 + LoRA 微调适配 + 蒸馏保能力，对推理质量要求高。

两者在 VMPC 压缩预设（[ADR-013](adr-013-vmpc-naming-v15.md)）、checkpoint 命名、训练配置上差异显著，若仍用单一 `config.yml` + 运行时参数切换，会导致：

- **配置耦合**：small 与 mate 的 VMPC 预设、checkpoint 目录、默认参数混在同一配置文件，难以维护。
- **VMPC 预设无处安放**：`vmpc_compress(profile="small"|"mate")` 需要类级别的预设适配，单模型 + 不同 config 无法表达"profile"语义。
- **checkpoint 命名冲突**：两者训练产物若都写 `checkpoints/`，会相互覆盖。
- **能力档位语义不清**：1zB ≈ 1000B 等效能力的档位表达需要在目录结构上显式体现。

## 决策

**正式开始双模型并行：`spark/small/` 承载 0.06zB 小模型（VMPC-small 预设），`spark/mate/` 承载 0.2zB 旗舰模型（VMPC-mate 预设）；双模型均基于 `verse_nex.CometSparkNexLM`，VMPC 适配通过 config 传入；checkpoint 目录改用 `mf_small/` / `mf_mate/`，模型文件默认 `.vn` 格式。**

具体含义：

1. **small 模型（0.06zB）**：
   - 目录：`spark/small/`
   - VMPC 预设：`vmpc_compress(profile="small")` = ternary 量化（2bit/值）+ 高稀疏剪枝（sparsity=0.5）。
   - 定位：端侧 / 嵌入式 / 树莓派，极致压缩比，低内存推理。
   - 1zB ≈ 1000B 等效能力，0.06zB ≈ 60B 等效能力档位。

2. **mate 模型（0.2zB）**：
   - 目录：`spark/mate/`
   - VMPC 预设：`vmpc_compress(profile="mate")` = int4 量化 + 中稀疏（sparsity=0.3）+ LoRA（rank=8, alpha=16）+ 蒸馏（需用户提供 teacher）。
   - 定位：消费级 CPU / 单卡 GPU 推理，能力上限。
   - 0.2zB ≈ 200B 等效能力档位。

3. **统一基座**：
   - 双模型均基于 `verse_nex.CometSparkNexLM`（VerseNexBlock = TriSparse + MoD），不重造底层 block。
   - VMPC 适配通过 config 传入（`config["vmpc_profile"]` / `config["vmpc_compress"]`），不在类级别硬编码 profile。

4. **checkpoint 目录命名**：
   - small 模型：`mf_small/`（替代旧 `checkpoints_small/`）。
   - mate 模型：`mf_mate/`（替代旧 `checkpoints_mate/`）。
   - `mf_` 前缀统一，避免与旧 `checkpoints_XXX/` 混淆。

5. **模型文件默认 `.vn`**：
   - checkpoint 默认输出 `.vn` 格式（[ADR-009](adr-009-vn-format.md)），safetensors mmap 零拷贝 + pickle-free 安全。
   - `.pt` 格式仍可通过 `verse-convert` 互转，但不再是默认输出。

6. **`spark/run.py` 支持 `--model small|mate`**：
   - `python spark/run.py train --model small`：训练 small 模型，checkpoint 写 `mf_small/`。
   - `python spark/run.py train --model mate`：训练 mate 模型，checkpoint 写 `mf_mate/`。
   - 不指定 `--model` 时保留原有默认行为（向后兼容）。

## 后果

### 优点

- **职责清晰**：small / mate 目录分离，配置 / 数据 / checkpoint 互不干扰。
- **VMPC 预设落地**：`vmpc_compress(profile=...)` 的类级别适配有目录归宿，`--model` 参数直接映射到 profile。
- **能力档位显式**：0.06zB / 0.2zB 在目录名与文档中显式体现，便于用户按需选择。
- **checkpoint 隔离**：`mf_small/` / `mf_mate/` 避免相互覆盖，`mf_` 前缀与旧 `checkpoints_` 区分。
- **`.vn` 默认**：性能优化格式成为默认，mmap 零拷贝 + 自描述元数据。
- **向后兼容**：`--model` 不指定时保留原行为，旧 checkpoint 目录仍可读取。

### 缺点

- **目录膨胀**：`spark/` 下新增 `small/` / `mate/` 两个子目录，结构变复杂。
- **双模型维护成本**：两个模型的 config / 数据 / 测试需分别维护，发布流程更重。
- **`--model` 参数学习成本**：用户需理解 small / mate 的定位差异才能选择。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| small / mate 配置漂移导致能力档位混乱 | 目录隔离 + 文档明确 0.06zB / 0.2zB 定位 |
| `mf_` 前缀与旧 `checkpoints_` 共存导致用户困惑 | 旧目录仍可读（向后兼容）；文档引导用户迁移到 `mf_` |
| mate 模型蒸馏缺 teacher 时退化 | `vmpc_compress(profile="mate")` 蒸馏默认不启用，用户需显式传 `teacher_model` |
| `.vn` 默认输出在无 safetensors 环境降级 | 自动降级 npz（`allow_pickle=False`），功能不变仅失去 mmap 零拷贝 |

## 替代方案（已否决）

### 方案 A：单模型 + 不同 config

**描述**：保留单一 `spark/` 目录，通过 `cometspark_small.yml` / `cometspark_mate.yml` 两套 config 切换规模，不引入 `--model` 参数。

**否决理由**：VMPC 预设（`vmpc_compress(profile=...)`）需要类级别适配，单模型 + 不同 config 无法表达"profile"语义；checkpoint 目录仍会冲突；能力档位（0.06zB / 0.2zB）在目录结构上无法显式体现。

### 方案 B：完全独立的两个仓库

**描述**：`spark-small/` 与 `spark-mate/` 完全独立两个仓库。

**否决理由**：重复维护 `_bootstrap.py` / `run.py` / `src/` 等基础设施；双模型共享 `verse_nex.CometSparkNexLM` 基座，独立仓库会导致基座代码漂移；违背"一个训练仓库多模型并行"的初衷。

### 方案 C：用 `--size small|mate` 替代 `--model`

**描述**：参数名用 `--size` 而非 `--model`。

**否决理由**：`--size` 语义偏向"规模"，而 small / mate 的差异不仅是规模，还包括 VMPC 预设 / 能力档位 / 定位；`--model` 更能表达"模型规格"的完整语义。

## 备注

- 本 ADR 是 Part5K1 "双模型并行"的核心决策。
- 1zB ≈ 1000B 等效能力是 Part5K1 引入的能力档位单位（zB = zeta-Billion equivalent），用于横向对比不同压缩比下的等效能力。
- `mf_` 前缀含义：model-format（与 `.vn` 默认格式呼应），同时与旧 `checkpoints_` 区分。
- `spark/run.py` 的 `--model` 参数在 Task 11 补齐 `finetune` / `posttrain` / `continue` 子命令时统一支持。
- 相关测试：`tests/test_spark_dual_model.py` 覆盖 small / mate 目录结构 + `--model` 参数 + checkpoint 命名。
- 相关文档：[Verse 训练指南 - 双模型训练指南](../training_guide.md)、[主 README - 双模型并行](../../README.md)
