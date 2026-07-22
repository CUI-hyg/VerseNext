# Part4K1：基础设施全面升级 + 模型能力升级 + 优化 Spec

## Why

Part3K2 完成了 CometSpark + VerseNext 的能力补齐与并行训练基础，但当前基础设施仍存在结构性债务：
- 训练代码与 demo 耦合在 `data/demo/`，无法作为正式训练栈复用；
- `config.yml` 仍保留已坏的 `hybrid` 模式（seq_len≥64 时 NaN）；
- `VerseNex` 仍以 `TransformerLM` 等旧命名暴露，未完成“脱离 Transformer”的品牌落地；
- 超稀疏注意力只验证了“稀疏”未实现“并行”；
- 缺少 RL 算法（NexRL）、缺少 GPU/NPU 加速、缺少单样本/微调/后训练的 CLI 一等支持；
- `CometSpark` 仍是 demo，训练存在 Loss 难压、输出胡乱的问题。

Part4K1 将训练代码剥离为独立 `VerseTrainer` 包，把 4 个基础设施包聚合为 `VerseInfra`，全面落地 VerseNex 新技术（MoD / 超稀疏并行注意力 / NexRL），新增 GPU/NPU 后端抽象，并把 CometSpark 升级为正式 V0.5-1B 模型迁移到 `spark/` 专用目录。

## What Changes

### 1. 基础设施升级
- **新增 `VerseTrainer` 包**（从 `data/demo/` 训练代码完全剥离，CLI + 后端）：强化并行训练并修复“莫名终止退出”；解决数据集加载耗时问题；新增单样本 / 微调 / 后训练的 CLI 一等支持（`verse-train` / `verse-finetune` / `verse-posttrain` / `verse-eval` / `verse-tokenize`）；参考 `GPT_teacher-3.37M-cn` 彻底解决 Loss 无法优化问题（不停压低 Loss，遇 plateau 自动重走）；清除无效代码与旧实现。
- **新增 `VerseInfra` 总包（BREAKING）**：把 `verse_compat/` + `verse_inference/` + `verse_tokenizer/` + 新增 `VerseTrainer` 全部移动到 `VerseInfra/` 下，作为单可安装包 + 子模块结构；全项目导入路径更新为 `from verse_infra.verse_xxx import ...`。
- **彻底删除 `config.yml` 中的 `hybrid` 模式（BREAKING）**：`arch` 仅保留 `versenex`（原 `transformer` 路径由 VerseNexLM 统一接管）。
- **升级 `VerseNex`**：把 `TransformerLM` / `TransformerBlock` / `GQASelfAttention` 等“Transformer 系”类在 VerseNex 命名空间下统一更名为 `VerseNexLM` / `VerseNexBlock` / `VerseNexAttention`（保留兼容别名一个版本）；完全落地 **MoD**、**超稀疏并行注意力**、**NexRL** 三项新技术。
- **新增 `NexRL`**：优化 RL 算法。并行 RL + 优化 RL 五要素（Agent + 环境 + 状态 + 动作 + 奖励），重点优化奖励（reward shaping / 多维奖励）与动作（动作空间采样 / 探索-利用平衡）。
- **优化超稀疏并行注意力**：在已验证的稀疏注意力基础上实现并行机制 —— 多 token 同时处理 + 分离式并行预测未来多 token（speculative-decoding 风格的 verify-then-commit）。
- **优化 `VerseTokenizer`**：借鉴成熟 tokenizer 框架（sentencepiece / tokenizers / tiktoken）补齐能力，并接入 NexRL 提升 token 命中准确度。
- **升级 `VerseTorch`**：新增缺失组件，并按新能力优化。
- **支持 GPU/NPU 训练（重点）**：新增 `DeviceBackend` 设备抽象层，PyTorch 可用时自动委托到 `torch`（CUDA / MPS / NPU via `torch_npu`），不可用时回退 NumPy CPU；最大化复用社区支持；针对训练与推理分别优化。

### 2. CometSpark → V0.5-1B 重要更新
- **chore（BREAKING）**：删除 `scripts/`，合并 `src/` + `train/` → `src/`，彻底删除旧实现。
- **迁移（BREAKING）**：把 CometSpark 从 `data/demo/` 移动到 `spark/` 专用目录；删除旧 `data/demo/`。
- **全面接入新框架**：VerseNex + VerseTrainer + VerseTokenizer 等，更新所有代码。
- **完全重写 CometSpark 模型**：VerseNex 已包含创新架构，新模型聚焦架构优化与能力提升（在 VerseNex block 之上构建，而非重造）。
- **使用 `Qwen3.5-35B-A3B` 的 Tokenizer**（HuggingFace `Qwen/Qwen3.5-35B-A3B`，vocab 248320）。
- **针对 VerseNex 优化**：提升训练时 CPU 利用率；解决模型能力不足 / 胡乱输出问题；支持后训练、增强训练；保证 1B 模型下较好质量。

### 3. 文档与代码注释
- 全面完善文档与代码注释到最新状态（README / 架构文档 / 模块 docstring / CLI 帮助）。

## Impact

- **Affected specs**: `build-verse-framework`、`evolve2-cometspark`、`part3-arch-model-optimization`、`part3k2-major-upgrade`（均已完成，Part4K1 在其上叠加，不回滚已完成能力，仅重组结构与升级能力）。
- **Affected code**:
  - `packages/verse_torch/`：新增 `device.py`（DeviceBackend）、`backend_torch.py`（PyTorch 委托后端）、补齐缺失 nn/optim/loss 组件；`Tensor` 支持设备迁移。
  - `packages/verse_nex/`：`cometspark.py` / `__init__.py` 重命名 Transformer 系 → VerseNex 系（保留别名）；`tri_sparse_attn.py` 升级为并行；`moe.py` 完善 MoD；新增 `nexrl/`（NexRL 算法包）。
  - `packages/verse_tokenizer/`：借鉴 tokenizers / sentencepiece 补齐能力；新增 NexRL 集成接口。
  - `packages/verse_compat/`、`packages/verse_inference/`：迁入 VerseInfra 后更新内部导入。
  - **新增** `packages/verse_infra/`（总包，含 `verse_compat/` `verse_inference/` `verse_tokenizer/` `verse_trainer/` 子模块）。
  - **新增** `packages/verse_infra/verse_trainer/`：从 `data/demo/` 剥离的训练栈 + CLI。
  - **新增** `spark/`：CometSpark V0.5-1B 正式模型目录（config / model / src）。
  - **删除** `data/demo/`（训练能力迁入 VerseTrainer，模型能力迁入 spark/）。
  - `pyproject.toml`：新增 `verse_infra`、`verse_trainer` 包声明；`tests/`、`examples/`、`docs/` 全量更新导入路径与文档。
  - `tests/`：新增 device backend / nexrl / parallel attn / versetrainer / cometspark-v05 / qwen-tokenizer 等测试。

## ADDED Requirements

### Requirement: VerseTrainer 独立训练包

系统 SHALL 提供独立可安装包 `verse_trainer`，从 `data/demo/` 训练代码完全剥离，提供 CLI + 后端 API：

1. **CLI 入口**（基于 argparse，`console_scripts` 注册）：
   - `verse-train`：预训练（含 `--config`、`--device cpu|cuda|npu`、`--single-sample`、`--parallel-chunks N`）。
   - `verse-finetune`：微调（LoRA / 全量）。
   - `verse-posttrain`：后训练（SFT / DPO / RL via NexRL）。
   - `verse-eval`：评估 + 打分（BLEU/ROUGE/exact-match）。
   - `verse-tokenize`：训练 / 加载 / 转换 tokenizer。
2. **强化并行训练**：`ParallelTrainer` 升级，修复“莫名终止退出”（子进程异常捕获 + 信号处理 + OOM 兜底 + 断点续训 checkpoint）；新增 `_safe_chunk_run()` 包裹每个 chunk 执行。
3. **数据集加载加速**：实现 `CachedDataset`（首次扫描后缓存为 `.npz`，后续启动毫秒级加载）+ 流式 lazy load；解决加载耗时问题。
4. **单样本支持**：`--single-sample` 接受单条 prompt/completion 或单文件，支持单样本训练 / 推理调试。
5. **微调 / 后训练**：复用 Part3K2 的 `SFTTrainer` / `DPOTrainer` / `LoRATrainer`，CLI 直接调用；新增 `RLTrainer`（基于 NexRL）。
6. **Loss 优化（参考 GPT_teacher-3.37M-cn）**：
   - 梯度裁剪 + LR warmup + cosine + ReduceLROnPlateau 组合策略；
   - loss plateau 检测：连续 N 步未下降则触发“重走”（回退 best checkpoint + 扰动 LR + 重置优化器动量）；
   - 数值稳定：NaN/Inf 检测 + 跳过该 batch；
   - 目标：在保证质量前提下不停压低 Loss。
7. **清除无效代码**：删除 `data/demo/` 中的旧实现、重复代码、注释掉的死代码。

#### Scenario: CLI 预训练
- **WHEN** 执行 `verse-train --config spark/config/cometspark_v05.yml --device cuda --parallel-chunks 4`
- **THEN** 加载配置 → 构建 CachedDataset → VerseNexLM → ParallelTrainer（4 chunk）→ 训练输出 checkpoint + loss 曲线

#### Scenario: 单样本调试
- **WHEN** 执行 `verse-train --single-sample --prompt "1+1=" --completion "2" --max-steps 50`
- **THEN** 构造单样本 batch，训练 50 步，输出 loss 下降曲线用于快速调试

#### Scenario: Loss plateau 重走
- **WHEN** 训练中连续 `patience` 步 val_loss 未下降
- **THEN** 触发 `_rollback_and_perturb()`：回退 best_state_dict + LR × 0.3 + 重置 Adam 动量，继续训练

### Requirement: VerseInfra 总包

系统 SHALL 提供总包 `verse_infra`，作为模型训练 / 运行的基础设施入口：

1. **单包 + 子模块结构**：`verse_infra` 是一个可安装包（独立 `pyproject.toml`），`verse_compat` / `verse_inference` / `verse_tokenizer` / `verse_trainer` 作为其子模块：
   ```
   packages/verse_infra/
     pyproject.toml
     verse_infra/
       __init__.py            # 重导出公共 API
       verse_compat/
       verse_inference/
       verse_tokenizer/
       verse_trainer/
   ```
2. **物理迁移**：把现有 `packages/verse_compat/`、`packages/verse_inference/`、`packages/verse_tokenizer/` 的源码移动到 `packages/verse_infra/verse_infra/` 下作为子模块；删除原 `packages/verse_xxx/` 顶层目录。
3. **导入路径更新（BREAKING）**：全项目所有 `from verse_tokenizer import ...` → `from verse_infra.verse_tokenizer import ...`（其余同理）；`verse_infra/__init__.py` 提供顶层便捷重导出 `from verse_infra import BPETokenizer`。
4. **verse_torch / verse_nex 保持独立**：不并入 VerseInfra（它们是引擎层与架构层，VerseInfra 是基础设施层）。
5. **向后兼容（一个版本）**：在原 `packages/verse_tokenizer/` 等位置保留一个 thin shim，`from verse_tokenizer import *` 转发到 `verse_infra.verse_tokenizer`，并发出 `DeprecationWarning`。

#### Scenario: 子模块导入
- **WHEN** 用户写 `from verse_infra.verse_tokenizer import BPETokenizer`
- **THEN** 成功导入，无 DeprecationWarning

#### Scenario: 便捷重导出
- **WHEN** 用户写 `from verse_infra import BPETokenizer, VerseTrainer`
- **THEN** 成功导入公共 API

#### Scenario: 旧路径兼容
- **WHEN** 用户写 `from verse_tokenizer import BPETokenizer`
- **THEN** 经 shim 转发成功，但发出 `DeprecationWarning: 请改用 verse_infra.verse_tokenizer`

### Requirement: VerseNex 品牌落地 + 类重命名

系统 SHALL 在 `verse_nex` 完成 Transformer 系类的 VerseNex 系重命名：

1. **重命名映射**（在 `verse_nex` 命名空间内）：
   - `TransformerLM` → `VerseNexLM`（顶层 LM）
   - `TransformerBlock` → `VerseNexBlock`（已存在，统一为唯一名）
   - `GQASelfAttention` → `VerseNexAttention`
   - 相关工厂 / 配置类同步更名。
2. **保留兼容别名（一个版本）**：旧名作为 `DeprecationWarning` 别名保留，下一大版本删除。
3. **`arch` 配置统一**：`config.yml` 的 `arch` 字段仅保留 `versenex`；旧 `transformer` / `hybrid` 值映射到 `versenex` 并发警告。
4. **完成 MoD 落地**：`MoDLayer`（5 DensePart × 8 Expert × top-3 双层门控）经测试验证可训练、可推理、aux loss 收敛。
5. **完成超稀疏并行注意力落地**：见独立 requirement。
6. **完成 NexRL 落地**：见独立 requirement。

#### Scenario: 类重命名
- **WHEN** 调用 `from verse_nex import VerseNexLM`
- **THEN** 成功导入，与旧 `TransformerLM` 行为一致

#### Scenario: 旧名兼容
- **WHEN** 调用 `from verse_torch.nn import TransformerLM`
- **THEN** 经别名导入成功，但发出 `DeprecationWarning: TransformerLM 已更名为 VerseNexLM`

### Requirement: NexRL 优化强化学习算法

系统 SHALL 提供新 RL 算法 `NexRL`，位于 `verse_nex/nexrl/`：

1. **RL 五要素**：
   - **Agent**：`NexAgent`，封装 VerseNexLM 策略网络 + 参考网络（KL 约束）。
   - **Environment**：`NexEnv`，任务环境（对话 / 数学 / 代码续写），提供 observation + reward。
   - **State**：`NexState`，包含 prompt + 已生成 token + KV cache（支持并行多 state）。
   - **Action**：`NexAction`，token 级动作 + 动作空间采样（top-k / nucleus + temperature schedule + 探索-利用 ε-greedy）。
   - **Reward**：`NexReward`，多维奖励组合（正确性 + 流畅度 + 安全性 + 长度惩罚），支持 reward shaping + reward normalization + GAE 优势估计。
2. **并行 RL**：多 prompt / 多 rollout 并行采样（batched），GPU/NPU 后端批量前向；`ParallelRolloutCollector` 收集 rollout。
3. **优化重点**：
   - **奖励优化**：多维加权 + reward normalization（running mean/std）+ reward shaping（potential-based）+ KL 惩罚防策略崩溃。
   - **动作优化**：动作空间采样策略可配置（ε-greedy / softmax / nucleus）+ 探索衰减 schedule + 重复动作惩罚。
4. **算法基础**：PPO 风格（clip ratio + value function + GAE）+ 参考模型 KL 祖父项；支持纯策略梯度 fallback（无 value）。
5. **CLI 集成**：`verse-posttrain --rl nexrl --config ...` 触发 NexRL 训练。

#### Scenario: 多维奖励
- **WHEN** NexReward 评估一个生成样本
- **THEN** 返回 `{"correctness": 0.8, "fluency": 0.9, "safety": 1.0, "length_penalty": -0.1}` 加权总分

#### Scenario: 并行 rollout
- **WHEN** 配置 `rollout_batch=8`
- **THEN** 8 个 prompt 同时前向采样，GPU 批量计算，吞吐 ≈ 8× 单条

#### Scenario: KL 防崩溃
- **WHEN** 策略与参考模型 KL 散度 > 阈值
- **THEN** 自动增加 KL 惩罚权重，限制策略漂移

### Requirement: 超稀疏并行注意力机制

系统 SHALL 在已有稀疏注意力（`TriSparseAttention` / `TopKChunkSparseAttention`）基础上实现并行机制：

1. **多 token 并行处理**：query 序列按 chunk 划分后，多个 query chunk 并行计算 attention（批量矩阵化 / GPU 并行），而非串行循环。
2. **分离式并行预测**：支持并行预测未来多个 token（speculative-decoding 风格）：
   - draft 模型 / draft head 并行生成 k 个候选 token；
   - 主模型一次前向验证 k 个 token（并行 attention）；
   - verify-then-commit：接受最长正确前缀，拒绝处重新 draft。
3. **KV cache 并行维护**：并行预测时 KV cache 批量更新。
4. **数值一致**：并行结果与串行结果在 float32 下吻合到 1e-3（前向）；预测接受/拒绝语义明确。
5. **性能目标**：长序列（seq_len ≥ 512）下，并行实现相对串行实现吞吐提升 ≥ 2×（GPU 后端）。

#### Scenario: 多 chunk 并行 attention
- **WHEN** seq_len=512, chunk_size=64, 8 个 query chunk
- **THEN** 8 个 chunk 并行计算 attention，GPU 利用率提升，吞吐相对串行 ≥ 2×

#### Scenario: 分离式预测
- **WHEN** 配置 `speculative_k=4`
- **THEN** draft 并行生成 4 个候选 token，主模型一次前向验证，接受最长正确前缀

### Requirement: VerseTokenizer 优化

系统 SHALL 优化 `verse_tokenizer`，借鉴成熟框架补齐能力：

1. **借鉴 tokenizers / sentencepiece / tiktoken**：
   - BPE 训练支持 `min_frequency`、`max_token_length`、并行 merge（多线程训练加速）。
   - 新增 `WordLevel` / `WordPiece` tokenizer 备选。
   - 编码 / 解码向量化（批量 encode/decode 加速）。
   - 支持 `add_bos` / `add_eos` 独立开关 + `truncation` / `padding` 策略对齐 HF `BatchEncoding`。
2. **NexRL 集成**：新增 `NexTokenizerWrapper`，在 token 边界注入 RL 信号（reward-weighted token preference），提升 token 命中准确度（高频高奖励子串优先成 token）。
3. **Qwen3.5-35B-A3B tokenizer 兼容**：支持从 HuggingFace `Qwen/Qwen3.5-35B-A3B` 加载 `tokenizer.json`（vocab 248320），作为 CometSpark V0.5-1B 的默认 tokenizer。
4. **byte-aligned decode**：保留并强化，杜绝 U+FFFD 乱码。

#### Scenario: 并行 BPE 训练
- **WHEN** 用大语料训练 BPE，`workers=4`
- **THEN** merge 阶段多线程并行，训练耗时 < 串行的 40%

#### Scenario: NexRL 集成
- **WHEN** 用 reward 标注的语料训练 tokenizer，启用 `nexrl_integration=True`
- **THEN** 高 reward 子串优先被合并为 token，token 命中率提升

#### Scenario: 加载 Qwen tokenizer
- **WHEN** 调用 `BPETokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")`
- **THEN** 从 HuggingFace 下载 tokenizer.json，成功加载 vocab 248320

### Requirement: VerseTorch 组件补齐 + GPU/NPU 后端

系统 SHALL 升级 `verse_torch`，新增缺失组件并支持 GPU/NPU：

1. **GPU/NPU 设备抽象（重点）**：
   - 新增 `DeviceBackend` 抽象基类 + `NumpyBackend`（默认 CPU）+ `TorchBackend`（PyTorch 委托）。
   - PyTorch 可用时（`import torch` 成功）自动启用 TorchBackend，支持 `cuda` / `mps` / `npu`（via `torch_npu`）设备。
   - 不可用时回退 NumpyBackend，行为与现状一致。
   - `Tensor.to(device)` / `Tensor.device` / `Tensor.cuda()` / `Tensor.npu()` API 对齐 PyTorch。
   - 模型 / 优化器 / 数据加载支持 `.to(device)` 迁移。
   - 最大化复用社区：CUDA kernel 走 PyTorch，NPU 走 `torch_npu`，不自研 kernel。
2. **缺失组件补齐**：
   - nn：`Conv1d`（卷积，备选架构）、`GroupNorm`、`LayerNorm` 优化版、`RotaryEmbedding`（RoPE 独立类，可复用）、`KVCache` 抽象（推理专用）、`StaticCache` / `DynamicCache`。
   - optim：`NAdamW`、`RMSProp`。
   - losses：`contrastive_loss`（RL/DPO 备选）、`perplexity`。
   - training：`DistributedTrainer` 占位接口（多卡数据并行 API 预留，单卡实现先行）。
3. **训练 / 推理优化**：GPU 后端开启混合精度（`autocast`）、梯度累积、KV cache 推理加速。
4. **向后兼容**：无 PyTorch 环境下所有现有测试不变通过。

#### Scenario: 自动后端选择
- **WHEN** 环境安装了 PyTorch + CUDA
- **THEN** `import verse_torch` 时 `Tensor` 默认设备为 CPU，`Tensor([1.0]).cuda()` 迁移到 GPU，matmul 在 GPU 执行

#### Scenario: 无 PyTorch 回退
- **WHEN** 环境无 PyTorch
- **THEN** `Tensor` 行为与现状一致（NumPy 后端），`Tensor.cuda()` 抛 `RuntimeError("未安装 PyTorch，无法使用 GPU")`

#### Scenario: 混合精度训练
- **WHEN** `--device cuda --amp`
- **THEN** forward/ backward 在 autocast fp16 下执行，loss 与 fp32 一致到 1e-2，显存占用降低

### Requirement: CometSpark V0.5-1B 正式模型

系统 SHALL 把 CometSpark 升级为正式模型 V0.5-1B，迁移到 `spark/` 专用目录：

1. **目录结构**：
   ```
   spark/
     config/
       cometspark_v05.yml       # 默认 1B 配置
       cometspark_v05_small.yml # 调试用小配置
     model/
       config.py                # CometSparkV05Config
       model.py                  # CometSparkV05LM（基于 VerseNex block）
     src/
       data_loader.py
       trainer.py                # 调用 VerseTrainer
       evaluate.py
       utils.py
     README.md
   ```
2. **chore**：删除 `data/demo/scripts/`；合并 `src/` + `train/` → `src/`；删除旧实现。
3. **迁移**：把 `data/demo/` 训练能力迁入 `VerseTrainer`，模型能力迁入 `spark/`，删除 `data/demo/`。
4. **完全重写模型**：`CometSparkV05LM` 基于 `VerseNexBlock`（TriSparse + MoD）构建，不重造底层；聚焦架构优化（层 pattern / 规模 / 初始化）与能力提升。
5. **Qwen3.5-35B-A3B tokenizer**：默认使用 `Qwen/Qwen3.5-35B-A3B` tokenizer（vocab 248320）。
6. **1B 参数预算**：通过 `n_layer` / `n_embd` / MoD expert 配置达到 ≈1B 参数；配置工厂 `CometSparkV05()` / `CometSparkV05Small()`。
7. **VerseNex 优化**：
   - 提升 CPU 利用率（BLAS 线程 + numba + 多线程数据加载）。
   - 解决胡乱输出：embedding scale + init scale + 输出 projection tie + temperature scaling。
   - 支持后训练 / 增强训练（接入 VerseTrainer CLI）。
   - 1B 模型保证质量：loss 收敛 + 生成连贯 + 打分达标。

#### Scenario: 模型构建
- **WHEN** 调用 `CometSparkV05()`
- **THEN** 返回 ≈1B 参数的 `CometSparkV05LM` 实例，基于 VerseNexBlock

#### Scenario: Qwen tokenizer
- **WHEN** 加载 spark/ 默认配置
- **THEN** tokenizer 从 `Qwen/Qwen3.5-35B-A3B` 加载，vocab=248320

#### Scenario: 训练 CLI
- **WHEN** 执行 `verse-train --config spark/config/cometspark_v05.yml --device cuda`
- **THEN** VerseTrainer 加载 spark/ 模型 + Qwen tokenizer + 训练数据，输出 checkpoint + loss 曲线

## MODIFIED Requirements

### Requirement: `config.yml` 的 `arch` 字段
原 `arch` 支持 `transformer` / `hybrid` / `versenex`。修改为：
- 仅保留 `versenex`（唯一合法值）。
- `transformer` / `hybrid` 值映射到 `versenex` 并发 `DeprecationWarning`（一个版本后报错）。
- 删除 `hybrid` 相关代码路径（`HybridBlock` / `HybridLM` 标记 deprecated，保留只读兼容）。

### Requirement: `verse_torch.Tensor`
原 `Tensor` 仅 NumPy 后端。修改为：
- 新增 `device` 属性（默认 `cpu`）。
- 新增 `.to(device)` / `.cuda()` / `.npu()` / `.cpu()` 方法。
- 当 `device != cpu` 时，底层由 `TorchBackend` 驱动（PyTorch 委托）；CPU 时仍由 NumPy 驱动。
- autograd 路径在 GPU 下走 PyTorch autograd（委托），CPU 下保持自研 autograd。

### Requirement: `verse_nex.CometSparkNexLM`
原 `CometSparkNexLM` 是 VerseNex 原生顶层架构。修改为：
- 作为 `CometSparkV05LM` 的基类被复用，不重复实现底层 block。
- 工厂 `CometSparkV02` 保留为调试用小模型。

## REMOVED Requirements

### Requirement: `data/demo/` 目录
**Reason**：训练能力剥离为 `VerseTrainer` 包，模型能力迁移到 `spark/`，demo 目录已完成历史使命。
**Migration**：
- `data/demo/train/` → `packages/verse_infra/verse_trainer/`
- `data/demo/model/` → `spark/model/`（重写为 V0.5）
- `data/demo/src/` + `data/demo/scripts/` → `spark/src/` + `VerseTrainer` CLI
- `data/demo/data/` → `datasets/`（已有目录）
- `data/demo/config/` → `spark/config/`
- 删除 `data/demo/` 整个目录

### Requirement: `config.yml` 的 `hybrid` 模式
**Reason**：seq_len≥64 时 `np.exp(log_decay)` 数值溢出（NaN），且 VerseNex 已统一接管所有架构。
**Migration**：
- `arch: hybrid` → `arch: versenex`（自动映射 + 警告）
- `HybridBlock` / `HybridLM` 标记 deprecated，保留只读兼容，不在新配置中使用

### Requirement: `packages/verse_compat/`、`packages/verse_inference/`、`packages/verse_tokenizer/` 顶层包
**Reason**：聚合为 `verse_infra` 总包的子模块，统一基础设施入口。
**Migration**：
- 源码移动到 `packages/verse_infra/verse_infra/` 下
- 原位置保留 thin shim（一个版本）发出 DeprecationWarning
- 全项目导入路径更新

### Requirement: `data/demo/scripts/` 与重复实现
**Reason**：脚本能力已由 VerseTrainer CLI 覆盖，重复实现需清除。
**Migration**：删除 `scripts/`，功能由 `verse-tokenize` / `verse-train` 等 CLI 替代
