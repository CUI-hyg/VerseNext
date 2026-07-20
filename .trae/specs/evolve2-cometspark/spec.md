# CometSpark-v0.1 与 Verse 第二次进化 Spec

## Why

Verse 框架 v0.1（已完成）建立了 VerseTorch + VerseNex + VerseAWM 三大包，但仅完成了核心引擎、架构库与世界模型的"骨架"。要走向"可用框架"，仍需：

1. **训练栈不完整**：现有 `examples/` 只是 smoke test，没有完整的多步训练、自动交叉熵、loss 可视化、早停、checkpoint 周期保存等"训练基础设施"。用户无法用一行命令训练自己的 LM。
2. **Tokenizer 与 nn 层不够丰富**：`verse_tokenizer` 仅有最小 BPE/Char，`verse_torch.nn` 缺少 `SwiGLU MLP`、`GQA SelfAttention`、`Embedding` tied head、`Sequential` 链式搭建多层网络所需的常用模块；与 PyTorch/torch 生态对接时 API 缺口大。
3. **CPU 性能未榨干**：纯 NumPy 默认单线程，`np.dot` 即使底层是 MKL/OpenBLAS 也无法在 Python 层面利用 batch 维度并行；`forward`/`backward` 中的元素级 op 完全串行。在 4-8 核 CPU 上训练吞吐量有 3-5× 提升空间。
4. **缺少"压缩模型参数"创新点**：行业趋势是 trillion-param → billion-param（Phi-3、Nanbeige4.1-3B、BitNet b1.58），但 Verse 没有端到端的"超压缩"流程。需要验证组合量化（INT4/1.58-bit）+ 结构化剪枝 + 知识蒸馏 + LoRA 低秩适配能否在 VerseTorch 上跑通，作为可行性 PoC。

同时，用户要求基于 VerseNex 创建一个名为 **CometSpark-v0.1** 的端到端 LM 训练仓库（参考 GPT_teacher 项目结构），让用户用 `python run.py` 一键完成"构建 tokenizer → 预训练 → 验证"全流程。

## What Changes

### Verse 框架第二次进化

- **ADDED** `verse_torch.nn`：补齐多层神经网络常用层
  - `SwiGLUMLP`（Llama 风格 gate × up × down，hidden 对齐 64）
  - `GQASelfAttention`（Grouped Query Attention，含 RoPE + KV cache + causal mask）
  - `TransformerBlock`（pre-norm RMSNorm + GQA + SwiGLU MLP + residual）
  - `TransformerLM`（token emb + N × TransformerBlock + tied LM head）
  - `ModuleList`/`Sequential` 已有，补充 `Parameter` 别名
- **ADDED** `verse_torch.training`：训练基础设施新模块
  - `Trainer` 类：封装 forward → cross_entropy → backward → step → log → checkpoint 周期
  - `cross_entropy_loss(logits, targets, ignore_index=-100)` 自动 reshape + softmax + NLL
  - `compute_loss_rate`：滑动窗口 loss 下降率（用于"自主计算损失率"）
  - `plot_loss_curve(train_losses, val_losses, save_path)`：matplotlib 可选，无则纯文本 ASCII 曲线
  - `EarlyStopping`（patience + min_delta）
  - `CheckpointManager`（best.pt / last.pt / quantized.pt 周期保存）
  - `GradientAccumulator`（micro_batch → effective batch）
  - `LRSchedulers`：补充 `LambdaLR`（warmup + cosine）
- **ADDED** `verse_torch.parallel`：初级 CPU 并行计算
  - `parallel_matmul(A, B, n_workers=None)`：将 batch 维度切片到 multiprocessing.Pool
  - `ParallelLinear`（继承 Linear，forward 时自动 batch-split）
  - `parallel_map(fn, iterable, n_workers=None)`：通用并行 map，autograd 安全（仅用于 forward 不依赖梯度路径的数据预处理或推理）
  - 默认 n_workers = `os.cpu_count() // 2`，避免过度订阅影响 BLAS 线程
- **ADDED** `verse_tokenizer`：扩展 BPE + 提供 HF tokenizer.json 加载的统一接口
  - `BPETokenizer.train(corpus, vocab_size)`：从语料训练 BPE
  - `BPETokenizer.add_special_tokens([bos, eos, pad, unk])`
  - `ByteTokenizer`：纯字节级 fallback（vocab_size=259，含 bos/eos/pad）
  - `load_tokenizer(kind="bpe"|"byte"|"hf", path=None)`：统一工厂
- **ADDED** `verse_torch.compress`：模型压缩 PoC 模块（第一轮可行性验证）
  - `OutlierSafePruner`：结构化剪枝（按 |weight|_mean per head/channel）
  - `LoRALinear`：低秩适配层（W_frozen + A·B^T，A 高斯，B 零初始化）
  - `KnowledgeDistiller`：teacher_logits + student_logits 的 KL loss
  - `compress_pipeline(model, target_ratio)`：组合量化（INT4）+ 剪枝（30%）+ LoRA（r=8）的端到端 PoC 流程
  - 在 1M → 100K 参数的小模型上验证组合压缩比与精度损失
- **MODIFIED** `verse_inference.generator`：支持 CometSpark 模型加载与流式生成
- **ADDED** ADR-004：CPU 并行计算策略（multiprocessing vs threading vs numexpr 选型）

### CometSpark-v0.1 训练仓库

新增目录 `/workspace/data/demo/`（注意：用户要求放 `data/` 下，但 `datasets/` 已存在；按用户原文 `data/` 解读，与现有 `datasets/` 并列），目录结构：

```
data/demo/
├── model/              # 模型核心代码
│   ├── __init__.py
│   ├── config.py       # CometSparkConfig（dataclass，含 vocab_size/n_layer/n_head/n_embd/seq_len/dropout 等）
│   ├── model.py        # CometSparkLM（基于 verse_nex.HybridLM 或 verse_torch.nn.TransformerLM）
│   └── tokenizer.py    # 调用 verse_tokenizer 构建 BPE tokenizer
├── data/               # 数据集
│   ├── README.md       # 数据格式说明
│   ├── train.jsonl     # 训练样本（每行 {"text": "..."}），先用小样本占位
│   └── val.jsonl       # 验证样本
├── train/              # 训练代码
│   ├── __init__.py
│   ├── trainer.py      # 调用 verse_torch.training.Trainer
│   ├── evaluate.py     # 验证 + 验收测试（生成示例文本）
│   └── visualize.py    # loss 曲线绘制
├── src/                # 辅助代码
│   ├── __init__.py
│   ├── utils.py        # set_seed / num_threads / ensure_dir
│   └── data_loader.py  # JSONL → Tensor dataset
├── config/
│   └── config.yml      # 默认配置（参考 GPT_teacher）
├── checkpoints/        # 训练输出（best.pt / last.pt / loss_curve.png / loss_history.json）
└── run.py              # 一键入口：构建 tokenizer → 训练 → 验证
```

- **ADDED** `CometSparkLM`：基于 `verse_nex.HybridLM`（Mamba-2 backbone 默认）或 `verse_torch.nn.TransformerLM`（可选），使用 `verse_torch.nn` 的 RMSNorm/SwiGLU/GQA
- **ADDED** `run.py`：参考 `GPT_teacher/run.py`，subprocess 链式调用：`build_tokenizer → train → evaluate`
- **BREAKING**：无（完全新增功能，不修改现有包的对外 API；`verse_torch.nn` 现有类保持向后兼容）

## Impact

- **Affected specs**：
  - `build-verse-framework`（v0.1 已完成）：本次为增量扩展，不修改其 spec
  - 后续 ADR-004 将补充 CPU 并行计算决策
- **Affected code**：
  - `packages/verse_torch/verse_torch/nn.py`（扩展层）
  - `packages/verse_torch/verse_torch/training.py`（新文件）
  - `packages/verse_torch/verse_torch/parallel.py`（新文件）
  - `packages/verse_torch/verse_torch/compress.py`（新文件）
  - `packages/verse_tokenizer/verse_tokenizer/bpe.py`（扩展 train/add_special_tokens）
  - `packages/verse_tokenizer/verse_tokenizer/__init__.py`（导出 load_tokenizer）
  - `data/demo/`（新目录，整个 CometSpark 训练仓库）
- **Affected docs**：
  - `docs/architecture/adr-004-cpu-parallel.md`（新增）
  - `verse_data/designs/compression_pipeline_design.md`（新增，压缩流程设计草稿）

## ADDED Requirements

### Requirement: 多步训练基础设施

系统 SHALL 提供 `verse_torch.training.Trainer` 类，封装 LM 训练完整生命周期。

#### Scenario: 多步训练 + 自动交叉熵 + 早停
- **WHEN** 用户调用 `Trainer(model, train_loader, val_loader, optimizer, scheduler, cfg).fit()`
- **THEN** 系统按 step 循环：forward → `cross_entropy_loss(logits, targets)` → backward → grad accumulation → optimizer.step → scheduler.step → log
- **AND** 每 `eval_interval` step 在 val_loader 上计算验证 loss
- **AND** 验证 loss 连续 `patience` 次未改善时触发 `EarlyStopping`
- **AND** 每次新最佳验证 loss 时保存 `best.pt`，每 `eval_interval` 保存 `last.pt`
- **AND** 训练结束自动调用 `plot_loss_curve` 保存 `loss_curve.png`

#### Scenario: 自主计算损失率
- **WHEN** 训练过程中维护长度为 `window`（默认 50）的滑动窗口
- **THEN** 系统每 step 计算 `loss_rate = (mean(prev_window) - mean(curr_window)) / mean(prev_window)`
- **AND** 当 `loss_rate < min_delta`（默认 1e-4）持续 `patience` 次时输出"训练收敛"提示

### Requirement: Tokenizer 系统完善

系统 SHALL 提供统一的 tokenizer 工厂与至少三种实现。

#### Scenario: 从语料训练 BPE
- **WHEN** 用户调用 `BPETokenizer.train(corpus_text, vocab_size=8000)`
- **THEN** 系统按字节级 BPE 算法迭代合并最频繁 pair 直到达目标词表大小
- **AND** 自动添加 `<bos>` / `<eos>` / `<pad>` / `<unk>` 特殊 token
- **AND** 提供 `save(path)` / `load(path)` 以 JSON 格式持久化

#### Scenario: 加载 HF tokenizer.json
- **WHEN** 用户调用 `load_tokenizer(kind="hf", path="tokenizer.json")`
- **THEN** 系统尝试用 `tokenizers` 库加载（如已安装）
- **AND** 失败时 fallback 到 `BPETokenizer.load(path)` 或 `ByteTokenizer`
- **AND** 返回的对象统一暴露 `encode(text) -> List[int]` / `decode(ids) -> str` / `vocab_size` / `bos_id` / `eos_id` / `pad_id`

### Requirement: 多层神经网络层补齐

系统 SHALL 在 `verse_torch.nn` 中提供与 Llama/Qwen 风格对齐的核心层。

#### Scenario: 构建 GQA SelfAttention
- **WHEN** 用户实例化 `GQASelfAttention(d=256, n_head=4, n_kv_head=2)`
- **THEN** 层包含 `wq` / `wk` / `wv` / `proj` 四个 Linear（bias=False）
- **AND** forward 支持 `kv_cache` 参数（前缀 (k, v) Tensor），返回 `(output, new_kv_cache)`
- **AND** 自动应用 RoPE（调用 `verse_nex.positional.RoPE`）
- **AND** 自动应用 causal mask（下三角）

#### Scenario: 构建 TransformerLM
- **WHEN** 用户实例化 `TransformerLM(vocab_size=1000, n_layer=4, n_head=4, n_embd=256, seq_len=128)`
- **THEN** 模型包含 `tok_emb` + 4 × `TransformerBlock` + `RMSNorm` + `head`（与 tok_emb 权重 tied）
- **AND** `forward(idx)` 返回 logits，shape `(B, T, vocab_size)`

### Requirement: 初级 CPU 并行计算

系统 SHALL 提供 `verse_torch.parallel` 模块，在多核 CPU 上加速训练/推理。

#### Scenario: 并行 batched matmul
- **WHEN** 用户调用 `parallel_matmul(A, B, n_workers=4)`，A.shape=(B, M, K), B.shape=(K, N)
- **THEN** 系统将 batch 维度 B 切成 4 份，分别提交到 `multiprocessing.Pool(4)`
- **AND** 子进程内调用 `np.matmul`，结果合并回 (B, M, N)
- **AND** 在 B ≥ 16 且 M*N ≥ 1024 时实测加速比 ≥ 2×（4 核 CPU）

#### Scenario: 自动并行 Linear
- **WHEN** 用户将 `Linear` 替换为 `ParallelLinear(n_workers=4)`
- **THEN** forward 时若 batch 维度 ≥ 阈值（默认 16），自动启用并行
- **AND** batch < 阈值时 fallback 到普通 `Linear.forward`，避免进程开销
- **AND** 反向传播梯度正确（与单线程数值一致到 1e-6）

### Requirement: 模型压缩 PoC（第一轮可行性验证）

系统 SHALL 提供 `verse_torch.compress` 模块，验证组合压缩流程。

#### Scenario: 端到端压缩 pipeline
- **WHEN** 用户在 1M 参数的小 LM 上调用 `compress_pipeline(model, target_ratio=0.1)`
- **THEN** 系统依次执行：
  1. 结构化剪枝 30%（按 head 重要性）
  2. INT4 量化（W4A16）
  3. LoRA 适配（r=8，frozen base + trainable A/B）
- **AND** 在原模型与压缩模型上跑同一测试集，输出压缩比、参数量、loss 差异
- **AND** 当压缩比 ≥ 10× 且 loss 差异 ≤ 5% 时输出"可行性验证 PASS"

#### Scenario: 单技术对照
- **WHEN** 用户分别调用 `quantize_only` / `prune_only` / `lora_only` / `distill_only`
- **THEN** 系统输出每种技术单独的压缩比与精度损失
- **AND** 在 `docs/benchmarks/compression_poc.md` 中生成对照表

### Requirement: CometSpark-v0.1 一键训练

系统 SHALL 在 `/workspace/data/demo/` 下提供完整 LM 训练仓库。

#### Scenario: 一键运行
- **WHEN** 用户在 `data/demo/` 下执行 `python run.py`
- **THEN** 系统依次执行：
  1. 检查 `data/tokenizer.json`，不存在则用 `verse_tokenizer.BPETokenizer.train` 从 `data/train.jsonl` 训练
  2. 调用 `train/trainer.py` 开始训练（参考 `config/config.yml`）
  3. 训练结束自动调用 `train/evaluate.py` 跑验收测试（生成示例文本）
  4. 输出 `checkpoints/best.pt` / `checkpoints/loss_curve.png` / `checkpoints/loss_history.json`
- **AND** 全程使用 VerseTorch/VerseNex，不依赖 PyTorch/torch
- **AND** 在 4 核 CPU 上 5 分钟内完成 1000 step 训练（small config）

#### Scenario: CometSpark 模型架构
- **WHEN** 用户查看 `model/model.py`
- **THEN** `CometSparkLM` 基于 `verse_nex.HybridLM`（默认 SSM:Attention = 3:1）或 `verse_torch.nn.TransformerLM`（config 可切换）
- **AND** 默认 config：vocab_size=auto, n_layer=4, n_embd=128, n_head=4, seq_len=128, dropout=0.1
- **AND** 模型参数量 ≤ 5M（small 版本，验证流程用）

## MODIFIED Requirements

### Requirement: verse_inference 兼容 CometSpark

`verse_inference.ModelLoader` 现有支持 mamba2/rwkv7/hybrid 三种 arch，新增 `cometspark` arch：
- 自动检测 `config.json` 中的 `architectures` 字段
- 加载 CometSparkConfig + CometSparkLM
- 流式生成接口不变

## REMOVED Requirements

无（本次为纯增量扩展，不删除任何现有功能）。
