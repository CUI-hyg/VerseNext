# Part3K2：CometSpark + VerseNext 重大升级 Spec

## Why

Part3 完成了架构与模型基础优化（tokenizer 抽象、PyYAML、NaN 修复、CLI/递归修复、自定义 prompt、val_loss 可视化、medium/large 配置、numba 加速）。Part3K2 在此基础上进行七项重大升级：训练数据格式现代化（chat / prompt-completion）、tokenizer 全面升级（自带 BPE + 正则预分词 + 中文优化）、对齐 Transformer/PyTorch 补齐缺失能力、CometSpark 架构向大参数模型看齐并深度集成压缩技术、基础框架与训练体系支持并行训练、训练框架支持推理 + 自由温度 + 打分、以及全项目 check-loop 审计清零 BUG。

## What Changes

- **训练数据格式现代化（BREAKING）**：`train.jsonl` / `val.jsonl` 完全移除 `{"text": "..."}` 格式，改为支持两种 JSON 类型——`[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`（chat 数组）与 `{"prompt":"...","completion":"..."}`（prompt-completion 对象）。两种格式可在同一文件混用。
- **Tokenizer 全面升级**：升级 `verse_tokenizer`，新增 GPT-2/GPT-4 风格正则预分词（中文整字、数字、英文单词、标点独立成块），BPE 训练支持 `vocab_size` 自适应、特殊 token 注册、`add_special_tokens` 开关、`apply_chat_template`/`apply_prompt_template` 助手；统一 NFKC + byte-aligned decode；新增 `SentencePieceUnigramTokenizer`（unigram 语言模型）。
- **对齐 Transformer / PyTorch 补齐能力**：补齐缺失优化器（Lion、Adafactor）、LR 调度器（OneCycleLR、ReduceLROnPlateau、CosineRestarts）、激活函数（GeGLU、Mish、SiLU 别名）、注意力（SlidingWindowAttention、ALiBi 位置偏置）、归一化（DeepNorm）、损失（focal loss、label smoothing 默认参数对齐 HF）、`torch.utils.data` 风格 Dataset/DataLoader（已有 BatchLoader 升级）。同时进一步减少依赖（去掉 `requests` 软依赖、合并 numba 软依赖路径）。
- **CometSpark 架构升级 + 压缩技术深度集成**：模型升级为 Llama/Qwen 风格（pre-norm RMSNorm + GQA + RoPE + SwiGLU + tie_weights，已具备），新增 `n_kv_head` GQA 自适应、`rope_theta` 可配置、`max_position_embeddings`、`attention_dropout`、`hidden_dropout`、`embedding_dropout` 分别配置；架构支持 `CometSparkConfig.from_pretrained` / `save_pretrained` 标准接口；压缩技术（OutlierSafePruner、LoRALinear、KnowledgeDistiller、INT4、Ternary）深度集成到 `CometSparkLM`，提供 `compress(config)` 一键管线 API；新增 `CometSparkSmall` / `CometSparkMedium` / `CometSparkLarge` 三档预设工厂；开始试验压缩 + 训练组合。
- **并行训练支持（基础框架 + 训练体系）**：`verse_torch.training` 新增 `ParallelTrainer`，支持把训练步数拆成 N 个 chunk，并行训练后合并；合并策略：对比每个 chunk 的 train_loss + val_loss，效果差的 chunk 放前面、好的放后面重训，并按 val_loss 对整体进行 fine-tune（优化 loss 较差的部分）；**修复 val_loss 更新漏洞**——并行 chunk 训练时 val_loss 需在每个 chunk 完成后基于完整 val 数据集更新（旧实现只更新了 chunk 局部 val，存在 BUG）；目标：当前阶段做到尽可能等同于一体训练的水平，算法优化后续迭代。
- **训练框架升级**：`Trainer` 新增 `inference(prompts, temperature, top_k, top_p, max_tokens)` 方法（自由温度 + top-k/top-p 采样）；`evaluate.py` 测试模块升级为 `ScoringEvaluator`，支持 BLEU/ROUGE-L/exact-match/prefix-accuracy 多指标打分（参考 GPT_teacher-3.37M-cn）；新增 `tests/test_scoring.py`。
- **全项目 check-loop 审计**：检查所有 packages（verse_torch/verse_nex/verse_tokenizer/verse_inference/verse_awm/verse_compat）+ data/demo；修复严重错误、漏洞、可优化部分；合并重复/低效代码；清零 BUG/漏洞/错误。

## Impact

- **Affected specs**: `build-verse-framework`（Part1）、`evolve2-cometspark`（Part2）、`part3-arch-model-optimization`（Part3）—— 均已完成，Part3K2 在其上叠加，不回滚。
- **Affected code**:
  - `packages/verse_tokenizer/verse_tokenizer/`：新增 `unigram.py`、`preprocess.py`（正则预分词）、`chat_template.py`；升级 `bpe.py`、`byte.py`、`char.py`
  - `packages/verse_torch/verse_torch/`：新增 `optim_extras.py`（Lion/Adafactor）、`scheduler_extras.py`（OneCycle/ReduceLROnPlateau/CosineRestarts）、`activations.py`（GeGLU/Mish/SiLU）；升级 `nn.py`（SlidingWindowAttention/ALiBi/DeepNorm）、`training.py`（ParallelTrainer/inference）、`compress.py`（深度集成 API）
  - `packages/verse_nex/verse_nex/`：保持现有 Mamba2/RWKV7/RetNet，新增 `cometspark.py` 升级版
  - `data/demo/src/data_loader.py`：重写支持 chat / prompt-completion 双格式
  - `data/demo/data/train.jsonl`、`val.jsonl`：完全重写为新格式
  - `data/demo/model/model.py`、`config.py`：升级 CometSpark 架构 + 压缩集成
  - `data/demo/train/trainer.py`、`evaluate.py`：升级并行训练 + 推理 + 打分
  - `data/demo/run.py`：新增 `--parallel-chunks N`、`--temperature`、`--top-p`、`--score` 参数
  - `tests/`：新增 `test_chat_data_loader.py`、`test_tokenizer_upgrade.py`、`test_parallel_trainer.py`、`test_scoring.py`、`test_compression_integration.py`、`test_optim_extras.py`、`test_scheduler_extras.py`

## ADDED Requirements

### Requirement: 多格式训练数据加载器

系统 SHALL 支持 `train.jsonl` / `val.jsonl` 中每行一个 JSON 样本，支持两种格式：
1. **Chat 数组**：`[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`，按对话顺序拼接 user/assistant 内容，使用 `apply_chat_template` 包装（含 `<|user|>` / `<|assistant|>` 等控制 token）。
2. **Prompt-Completion 对象**：`{"prompt":"...","completion":"..."}`，拼接为 `prompt + <|assistant|> + completion`，并标注 completion 部分参与 loss 计算（prompt 部分 loss 屏蔽，`ignore_index=-100`）。

系统 SHALL 在加载时自动检测每行格式（数组 → chat；对象含 prompt+completion → prompt-completion；其它 → 抛出 `ValueError`）。

#### Scenario: 加载 chat 数组样本
- **WHEN** `train.jsonl` 一行为 `[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]`
- **THEN** 加载器调用 `tokenizer.apply_chat_template` 生成 `<|user|>你好<|assistant|>你好！<|eos|>` 序列，loss 仅在 assistant 内容上计算

#### Scenario: 加载 prompt-completion 样本
- **WHEN** `train.jsonl` 一行为 `{"prompt":"1+1=","completion":"2"}`
- **THEN** 加载器生成 `1+1=<|assistant|>2<|eos|>`，loss 仅在 `2<|eos|>` 部分计算

#### Scenario: 拒绝旧 text 格式
- **WHEN** `train.jsonl` 一行为 `{"text":"..."}`
- **THEN** 加载器抛出 `ValueError("旧版 text 格式已废弃，请使用 chat 数组或 prompt-completion 格式")`

### Requirement: 升级版 Tokenizer

系统 SHALL 提供以下升级能力：
1. **GPT-2/GPT-4 风格正则预分词**：使用 regex 把文本切成中文字、英文单词、数字、标点、空白独立块，再送入 BPE。
2. **BPE 训练优化**：支持 `vocab_size` 自适应（达不到目标时回退到最大可达）、特殊 token 注册（bos/eos/pad/unk + chat 控制符 `<|user|>`/`<|assistant|>`/`<|system|>`）、`add_special_tokens` 编码开关。
3. **apply_chat_template**：输入 chat 数组，输出 `<|user|>{content}<|assistant|>{content}<|eos|>` 风格 token 序列。
4. **apply_prompt_template**：输入 prompt 字符串，输出 `<|user|>{prompt}<|assistant|>` 前缀（用于推理）。
5. **Unigram tokenizer**：新增 `SentencePieceUnigramTokenizer`，基于 unigram 语言模型（Viterbi 解码），作为 BPE 的备选。
6. **byte-aligned decode**：所有 tokenizer 统一实现 `_trim_to_utf8_boundary`，杜绝 U+FFFD 乱码。

#### Scenario: 中文预分词
- **WHEN** 用升级 BPE encode `"床前明月光"`
- **THEN** 预分词先切成 `["床","前","明","月","光"]`，每个汉字作为独立 piece 进入 BPE 训练词表，不再因字节合并产生乱码

#### Scenario: chat template
- **WHEN** 调用 `tok.apply_chat_template([{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}])`
- **THEN** 返回 token id 序列对应 `<|user|>你好<|assistant|>你好！<|eos|>`

### Requirement: 对齐 Transformer / PyTorch 补齐能力

系统 SHALL 在 `verse_torch` 中补齐以下能力（参考 HuggingFace Transformers + PyTorch）：
1. **优化器**：`Lion`（无动量项，sign 更新）、`Adafactor`（Factored 二阶矩，内存友好）。
2. **LR 调度器**：`OneCycleLR`（super-convergence）、`ReduceLROnPlateau`（按 val_loss 降 lr）、`CosineRestartsLR`（带 warm restarts）。
3. **激活函数**：`GeGLU`、`Mish`、`SiLU`（与 PyTorch `torch.nn.SiLU` 对齐，作为 `SwiGLU` 内部使用的别名）。
4. **注意力变体**：`SlidingWindowAttention`（滑动窗口，长上下文场景）、`ALiBi` 位置偏置（替代 RoPE 的备选）。
5. **归一化**：`DeepNorm`（pre-norm 变体，深网络稳定训练）。
6. **损失函数**：`focal_loss`（类别不均衡场景）、`label_smoothing` 默认参数对齐 HF（`0.0` 关闭，`0.1` 轻度）。
7. **数据接口**：升级 `BatchLoader` 对齐 `torch.utils.data.DataLoader`（支持 `num_workers=0` 接口、`pin_memory` 占位、`persistent_workers` 占位，但 CPU-only 实现保持单线程）。

#### Scenario: Lion 优化器
- **WHEN** 用 `Lion(params, lr=1e-4, weight_decay=0.1)` 训练模型
- **THEN** 参数更新方向为 `sign(m · β1 + g · (1-β1))`，无二阶矩，比 AdamW 节省 ~50% 优化器状态内存

#### Scenario: OneCycleLR
- **WHEN** 配置 `lr_scheduler=onecycle`，`max_lr=0.01`，`total_steps=200`
- **THEN** lr 在前 25% 步升到 max_lr，后 75% 步余弦退火到 `max_lr/div=25`

### Requirement: CometSpark 架构升级 + 压缩集成

系统 SHALL 把 `CometSparkLM` 升级为 Llama/Qwen 风格：
1. **架构升级**：pre-norm RMSNorm + GQA（已具备）+ RoPE（`rope_theta` 可配置）+ SwiGLU（已具备）+ tie_weights；新增 `attention_dropout` / `hidden_dropout` / `embedding_dropout` 分别配置；新增 `max_position_embeddings` 与 `seq_len` 分离。
2. **from_pretrained / save_pretrained**：标准 HF 接口，支持 `from_pretrained(checkpoint_dir)` / `save_pretrained(checkpoint_dir)` 加载/保存 config + state_dict + tokenizer。
3. **压缩深度集成**：`CometSparkLM.compress(compress_config)` 一键应用压缩管线（prune + quantize + lora + ternary + distill 任意组合），返回压缩后模型；`CometSparkLM.compression_stats()` 返回压缩前/后参数量、稀疏度、bit 数。
4. **预设工厂**：`CometSparkSmall()` / `CometSparkMedium()` / `CometSparkLarge()` 返回标准配置的模型实例。
5. **试验压缩训练**：新增 `examples/compress_train_demo.py`，演示 prune→quantize→finetune→evaluate 完整流程。

#### Scenario: 一键压缩
- **WHEN** 调用 `model.compress({"prune": {"sparsity": 0.5}, "quantize": {"bits": 4}})`
- **THEN** 模型权重中 50% 通道被裁剪，剩余权重 INT4 量化，`compression_stats()` 显示参数量减少 ~50%，bit 数从 32 降到 4

#### Scenario: 预设工厂
- **WHEN** 调用 `CometSparkSmall()`
- **THEN** 返回 n_layer=2 n_embd=64 的 `CometSparkLM` 实例，参数量约 131K

### Requirement: 并行训练支持

系统 SHALL 在 `verse_torch.training.ParallelTrainer` 中提供：
1. **步数拆分**：把 `max_steps` 拆成 N 个 chunk（如 200 步拆 4 chunk × 50 步）。
2. **并行训练**：每个 chunk 独立 `Trainer` 实例并行执行（CPU 多进程或串行，CPU-only 默认串行避免 GIL 竞争，但接口对齐并行）。
3. **合并策略**：对比每个 chunk 的 train_loss + val_loss，**效果差的放前面、好的放后面**，串行重训（这样差的部分先收敛、好的部分微调）。
4. **val_loss 修复**：每个 chunk 完成后，**基于完整 val 数据集**更新 val_loss（旧实现只用了 chunk 局部 val，存在 BUG）；val_loss 必须包含完整数据集统计而非 batch 平均。
5. **整体调整**：合并后按 val_loss 对整体 fine-tune 若干步（默认 `merge_finetune_steps = max_steps // 10`），优化 loss 较差的部分。
6. **目标水平**：当前阶段做到尽可能等同于一体训练的水平（不要求超越，避免过度算法化），后续迭代再做算法优化。

#### Scenario: 200 步拆 4 chunk
- **WHEN** 配置 `parallel_chunks=4`，`max_steps=200`
- **THEN** 每个 chunk 训练 50 步，按 train_loss+val_loss 排序后串行重训，最后整体 fine-tune 20 步

#### Scenario: val_loss 完整更新
- **WHEN** chunk 0 训练完成
- **THEN** 在 `ParallelTrainer._eval_full_val()` 中跑完整 val 数据集计算 val_loss（不是只跑一个 batch），更新 `best_val_loss` 与 `best_state_dict`

### Requirement: 训练框架推理 + 自由温度 + 测试打分

系统 SHALL 升级 `Trainer` 与 `evaluate.py`：
1. **Trainer.inference**：新增方法 `inference(prompts: list[str], temperature=1.0, top_k=None, top_p=None, max_tokens=30) -> list[str]`，支持自由温度、top-k、top-p（nucleus）采样。
2. **ScoringEvaluator**：`evaluate.py` 升级为支持多指标打分：
   - `exact_match`：精确匹配率
   - `prefix_accuracy`：前缀匹配率（适合续写任务）
   - `char_f1`：字符级 F1
   - `bleu`：BLEU-4（简化版，无 smoothing）
   - `rouge_l`：ROUGE-L（最长公共子序列）
3. **打分报告**：`ScoringEvaluator.evaluate(prompts, references) -> dict` 返回各指标分数；`ScoringEvaluator.report(score_dict) -> str` 返回可读报告。
4. **CLI 参数**：`run.py` 新增 `--score`（启用打分）、`--references-file`（参考答案文件，每行一个）、`--temperature`、`--top-p`。

#### Scenario: 推理 + 温度
- **WHEN** 调用 `trainer.inference(["你好","1+1="], temperature=0.8, top_k=10, max_tokens=50)`
- **THEN** 返回 2 个生成字符串，temperature=0.8 增加随机性，top_k=10 限制候选

#### Scenario: 打分
- **WHEN** `--score --references-file refs.txt` 启用打分
- **THEN** `ScoringEvaluator` 跑完所有 prompt → 生成 → 与参考答案比对，输出 `exact_match=0.4 bleu=0.32 rouge_l=0.55 ...` 报告

### Requirement: 全项目 check-loop 审计

系统 SHALL 执行全项目审计，覆盖：
1. **严重错误**：递归栈溢出、NaN/Inf 数值溢出、Unicode 乱码、import 循环、文件路径硬编码。
2. **漏洞**：资源泄漏（文件未关闭）、除零、整数溢出、未捕获异常、SQL/路径注入（如适用）。
3. **可优化部分**：重复代码（如多个 `cross_entropy` 实现）、低效算法（如 `for` 循环可向量化）、冗余 import、死代码。
4. **合并**：合并 `verse_torch.losses.cross_entropy` 与 `verse_torch.training.cross_entropy_loss` 的重复实现（保持两个 API 入口，内部共用）；合并重复的 NFKC 实现。
5. **清零 BUG**：修复所有发现的 BUG/漏洞/错误，提供测试覆盖。

#### Scenario: check-loop 输出
- **WHEN** 执行 check-loop
- **THEN** 输出 `audit_report.md`（项目根目录），列出发现的所有问题 + 修复状态 + 测试覆盖

## MODIFIED Requirements

### Requirement: `verse_tokenizer.BaseTokenizer` 接口
原 `BaseTokenizer` 仅定义 `encode` / `decode` / `save` / `load` / `__len__`。修改为新增：
- `apply_chat_template(messages: list[dict]) -> list[int]`
- `apply_prompt_template(prompt: str) -> list[int]`
- `special_tokens: dict[str, int]`（属性）
- `add_special_tokens: bool`（构造参数，默认 True）
- `preprocess(text: str) -> str`（NFKC + 大小写归一化等）

### Requirement: `data/demo/src/data_loader.TextDataset`
原 `TextDataset` 只读 `text` 字段。修改为：
- 自动检测每行格式（chat 数组 / prompt-completion）
- 调用 `tokenizer.apply_chat_template` 或 `apply_prompt_template` 生成 token 序列
- loss mask：prompt 部分用 `ignore_index=-100` 屏蔽，仅 completion 部分参与 loss
- 旧 `text` 格式：抛 `ValueError` 提示迁移

### Requirement: `data/demo/train/trainer.py` 训练循环
原 `trainer.py` 直接调用 `Trainer.fit()`。修改为：
- 根据 `training.parallel_chunks` 配置选择 `Trainer` 或 `ParallelTrainer`
- `ParallelTrainer` 内部按 chunk 训练 → 排序 → 合并重训 → 整体 fine-tune
- 每个 chunk 完成后基于完整 val 数据集更新 val_loss（修复 BUG）

### Requirement: `data/demo/train/evaluate.py`
原 `evaluate.py` 仅生成示例文本。修改为：
- 升级为 `ScoringEvaluator` 类，支持生成 + 打分
- `--score` 时加载 `references-file`，跑多指标打分
- 不打分时保持原行为（仅生成示例）

### Requirement: `CometSparkLM`
原 `CometSparkLM` 仅支持 forward/generate/count_parameters。修改为：
- 新增 `compress(compress_config) -> CometSparkLM`（返回压缩后模型）
- 新增 `compression_stats() -> dict`
- 新增 `from_pretrained(dir) / save_pretrained(dir)` 类方法
- 新增 `rope_theta / max_position_embeddings / attention_dropout / hidden_dropout / embedding_dropout` 配置

## REMOVED Requirements

### Requirement: 旧版 `{"text": "..."}` 训练数据格式
**Reason**：用户要求完全替换为新 JSON 格式（chat 数组 + prompt-completion），删除 text 格式
**Migration**：
- `data/demo/data/train.jsonl` / `val.jsonl` 重写为新格式（每行 chat 数组或 prompt-completion 对象）
- `TextDataset` 拒绝 `{"text": "..."}`，抛出明确错误提示用户迁移
- `data/README.md` 更新数据格式文档

### Requirement: `verse_torch` 中的最小 YAML 解析器（已在 Part3 移除）
**Reason**：Part3 已用 PyYAML 替换，Part3K2 不再保留 fallback 路径（进一步减少依赖复杂度）
**Migration**：删除 `_minimal_yaml_load` 函数（如有残留），统一走 `yaml.safe_load`，安装时要求 `pyyaml` 为必选依赖（已是）

### Requirement: 重复的 cross_entropy 实现
**Reason**：`verse_torch.losses.cross_entropy` 与 `verse_torch.training.cross_entropy_loss` 存在重复实现
**Migration**：`training.cross_entropy_loss` 内部调用 `losses.cross_entropy`，保持两个 API 入口（用户习惯），但实现共用
