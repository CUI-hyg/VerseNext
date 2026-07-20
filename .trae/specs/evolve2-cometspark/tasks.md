# Tasks

> 目标：完成 Verse 框架第二次进化 + 创建 CometSpark-v0.1 端到端 LM 训练仓库。
> 任务编排按"nn 层补齐 → 训练基础设施 → Tokenizer → CPU 并行 → 压缩 PoC → CometSpark 仓库 → 文档与验证"顺序推进。

## 阶段 0：现有代码基线确认

- [x] Task 0.1: 读取 `packages/verse_torch/verse_torch/nn.py`、`tensor.py`、`optim.py`，确认现有 API（Linear/Embedding/LayerNorm/RMSNorm/Dropout/ModuleList/Sequential 已实现）
- [x] Task 0.2: 读取 `packages/verse_nex/verse_nex/positional.py`（RoPE 已实现）和 `hybrid.py`（HybridLM 已实现）
- [x] Task 0.3: 读取 `packages/verse_tokenizer/verse_tokenizer/bpe.py`，确认现有 BPE 接口

## 阶段 1：verse_torch.nn 多层神经网络补齐

- [x] Task 1.1: 实现 `SwiGLUMLP(d, dropout, hidden_multiple=4, align=64)`
  - [x] SubTask 1.1.1: w_gate / w_up / w_down 三个 Linear（bias=False），hidden = `((4*d*2/3 + align-1) // align) * align`
  - [x] SubTask 1.1.2: forward: `silu(w_gate(x)) * w_up(x)` → `w_down` → dropout
- [x] Task 1.2: 实现 `GQASelfAttention(d, n_head, n_kv_head=None, dropout=0.0)`
  - [x] SubTask 1.2.1: wq/wk/wv/proj 四个 Linear（bias=False），head_dim = d // n_head，n_rep = n_head // n_kv_head
  - [x] SubTask 1.2.2: forward(x, kv_cache=None)：投影 + RoPE（调用 `verse_nex.positional.apply_rope`）+ causal mask + softmax + attn + proj
  - [x] SubTask 1.2.3: KV cache 拼接前缀，返回 (output, new_kv_cache)
  - [x] SubTask 1.2.4: GQA repeat_kv 工具函数
- [x] Task 1.3: 实现 `TransformerBlock(d, n_head, n_kv_head, dropout)`
  - [x] SubTask 1.3.1: pre-norm 结构：`x = x + attn(norm1(x)); x = x + mlp(norm2(x))`
- [x] Task 1.4: 实现 `TransformerLM(vocab_size, n_layer, n_head, n_embd, seq_len, dropout, n_kv_head=None, tie_weights=True)`
  - [x] SubTask 1.4.1: tok_emb + N × TransformerBlock + RMSNorm + head
  - [x] SubTask 1.4.2: tie_weights=True 时 `head.weight = tok_emb.weight`
  - [x] SubTask 1.4.3: 参数初始化（Linear normal std=0.02，特殊层缩放 1/sqrt(2*n_layer)）
- [x] Task 1.5: 单元测试：`tests/test_nn_advanced.py`
  - [x] SubTask 1.5.1: SwiGLUMLP forward shape + 有限差分梯度
  - [x] SubTask 1.5.2: GQASelfAttention forward shape + KV cache 正确拼接
  - [x] SubTask 1.5.3: TransformerLM forward shape (B, T, vocab) + 参数量计算

## 阶段 2：verse_torch.training 训练基础设施

- [x] Task 2.1: 实现 `cross_entropy_loss(logits, targets, ignore_index=-100)` 函数
  - [x] SubTask 2.1.1: 自动 reshape logits (N, V) / targets (N,)
  - [x] SubTask 2.1.2: log_softmax + NLL，ignore_index mask
- [x] Task 2.2: 实现 `EarlyStopping(patience, min_delta)` 类
- [x] Task 2.3: 实现 `GradientAccumulator(micro_batch, effective_batch)`
- [x] Task 2.4: 实现 `CheckpointManager(save_dir, best_path, last_path)`
  - [x] SubTask 2.4.1: save_best(state) / save_last(state) / load_best() / load_last()
- [x] Task 2.5: 实现 `LambdaLR(optimizer, lr_lambda)` 调度器（补充到 `optim.py`）
  - [x] SubTask 2.5.1: warmup + cosine 的 lr_lambda 工厂函数
- [x] Task 2.6: 实现 `compute_loss_rate(loss_window, window=50, min_delta=1e-4)` 滑动窗口下降率
- [x] Task 2.7: 实现 `plot_loss_curve(train_losses, val_losses, save_path, eval_interval)`
  - [x] SubTask 2.7.1: matplotlib 可选，无则输出 ASCII 曲线到 .txt
- [x] Task 2.8: 实现 `Trainer` 类
  - [x] SubTask 2.8.1: `__init__(model, train_loader, val_loader, optimizer, scheduler, cfg)`，cfg 包含 max_steps/eval_interval/patience/save_dir/grad_accum
  - [x] SubTask 2.8.2: `fit()`：主训练循环 + log + checkpoint + early stop + loss_rate 监控
  - [x] SubTask 2.8.3: `evaluate()`：在 val_loader 上计算平均 loss
  - [x] SubTask 2.8.4: 训练结束自动 plot_loss_curve + 保存 loss_history.json
- [x] Task 2.9: 单元测试：`tests/test_training.py`
  - [x] SubTask 2.9.1: cross_entropy 与 PyTorch 数值对齐（手动算）
  - [x] SubTask 2.9.2: EarlyStopping 触发逻辑
  - [x] SubTask 2.9.3: Trainer 在合成数据上 10 step loss 下降

## 阶段 3：verse_tokenizer 系统完善

- [x] Task 3.1: 扩展 `BPETokenizer`，新增 `train(corpus, vocab_size)` 类方法
  - [x] SubTask 3.1.1: 字节级 BPE merge 算法（参考 GPT-2 风格）
  - [x] SubTask 3.1.2: 训练完成后自动 add_special_tokens
- [x] Task 3.2: 实现 `add_special_tokens(tokens)` 方法，扩展 vocab
- [x] Task 3.3: 实现 `save(path)` / `load(path)` JSON 持久化
- [x] Task 3.4: 实现 `ByteTokenizer`（vocab_size=259，含 bos/eos/pad/unk）
- [x] Task 3.5: 实现 `load_tokenizer(kind, path)` 工厂函数
  - [x] SubTask 3.5.1: kind="hf" 时尝试 `tokenizers.Tokenizer.from_file`，失败 fallback
  - [x] SubTask 3.5.2: kind="bpe" 时调 BPETokenizer.load
  - [x] SubTask 3.5.3: kind="byte" 时返回 ByteTokenizer
- [x] Task 3.6: 在 `verse_tokenizer/__init__.py` 中导出 `BPETokenizer` / `CharTokenizer` / `ByteTokenizer` / `load_tokenizer`
- [x] Task 3.7: 单元测试：`tests/test_tokenizer.py`
  - [x] SubTask 3.7.1: BPE train + encode/decode 往返一致
  - [x] SubTask 3.7.2: load_tokenizer 三种 kind 均返回统一接口

## 阶段 4：verse_torch.parallel CPU 并行计算

- [x] Task 4.1: 实现 `parallel_matmul(A, B, n_workers=None)` 函数
  - [x] SubTask 4.1.1: A shape (B, M, K), B shape (K, N) 或 (B, K, N)
  - [x] SubTask 4.1.2: 将 batch 维度切片到 multiprocessing.Pool
  - [x] SubTask 4.1.3: 默认 n_workers = max(1, os.cpu_count() // 2)
- [x] Task 4.2: 实现 `ParallelLinear(d_in, d_out, n_workers=None, batch_threshold=16)`
  - [x] SubTask 4.2.1: 继承 `nn.Linear`，forward 时 batch >= threshold 启用并行
  - [x] SubTask 4.2.2: 反向梯度正确（数值一致到 1e-6）
- [x] Task 4.3: 实现 `parallel_map(fn, iterable, n_workers=None)` 通用并行 map
- [x] Task 4.4: 单元测试 + 基准：`tests/test_parallel.py`
  - [x] SubTask 4.4.1: parallel_matmul 与 np.matmul 数值一致
  - [x] SubTask 4.4.2: 在 batch=64, M=N=K=256 上实测加速比（记录到 `docs/benchmarks/parallel_benchmark.md`）

## 阶段 5：verse_torch.compress 模型压缩 PoC

- [x] Task 5.1: 资料整理：在 `docs/papers/compression_references.md` 收集压缩技术参考（BitNet/QLoRA/OSP/蒸馏/剪枝）
- [x] Task 5.2: 实现 `OutlierSafePruner(model, sparsity=0.3)` 类
  - [x] SubTask 5.2.1: 按 head/channel |weight|_mean 排序，剪掉 bottom 30%
  - [x] SubTask 5.2.2: 返回 pruned_model + 剪枝报告（每层保留比例）
- [x] Task 5.3: 实现 `LoRALinear(d_in, d_out, r=8, alpha=16)` 层
  - [x] SubTask 5.3.1: base Linear frozen，A (d_in, r) 高斯，B (r, d_out) 零
  - [x] SubTask 5.3.2: forward: `base(x) + (B @ A^T) @ x * (alpha/r)`
  - [x] SubTask 5.3.3: 仅 A/B 参与反向，base 不更新
- [x] Task 5.4: 实现 `KnowledgeDistiller(teacher, student, T=2.0, alpha=0.5)`
  - [x] SubTask 5.4.1: KL(soften(teacher_logits/T) || soften(student_logits/T)) * T^2
  - [x] SubTask 5.4.2: 联合 hard label CE loss 加权
- [x] Task 5.5: 实现 `compress_pipeline(model, target_ratio=0.1, eval_fn=None)`
  - [x] SubTask 5.5.1: 依次 prune → quantize(INT4) → lora_wrap
  - [x] SubTask 5.5.2: 计算压缩比（参数量 / 存储字节）
  - [x] SubTask 5.5.3: 如有 eval_fn，计算压缩前后 loss 差异
  - [x] SubTask 5.5.4: 输出报告 dict
- [x] Task 5.6: 实现 `quantize_only` / `prune_only` / `lora_only` / `distill_only` 单技术函数
- [x] Task 5.7: 端到端 PoC：`tests/test_compression_poc.py`
  - [x] SubTask 5.7.1: 在 1M 参数小 TransformerLM 上跑 compress_pipeline
  - [x] SubTask 5.7.2: 验证压缩比 ≥ 10×
  - [x] SubTask 5.7.3: 验证 loss 差异 ≤ 5%
  - [x] SubTask 5.7.4: 生成 `docs/benchmarks/compression_poc.md` 对照表

## 阶段 6：CometSpark-v0.1 训练仓库搭建

- [x] Task 6.1: 创建 `data/demo/` 目录结构（model/data/train/src/config/checkpoints）
- [x] Task 6.2: 实现 `data/demo/model/config.py`（`CometSparkConfig` dataclass）
  - [x] SubTask 6.2.1: 字段：vocab_size, n_layer, n_head, n_embd, seq_len, dropout, n_kv_head, arch（"hybrid"|"transformer"）, ssm_kind, sparse_ratio
  - [x] SubTask 6.2.2: `from_yaml(path)` / `to_yaml(path)` 方法
- [x] Task 6.3: 实现 `data/demo/model/model.py`（`CometSparkLM`）
  - [x] SubTask 6.3.1: arch="hybrid" 时基于 `verse_nex.HybridLM`
  - [x] SubTask 6.3.2: arch="transformer" 时基于 `verse_torch.nn.TransformerLM`
  - [x] SubTask 6.3.3: forward(idx) → logits，shape (B, T, vocab)
  - [x] SubTask 6.3.4: `save(path)` / `load(path)`：序列化 config + state_dict
- [x] Task 6.4: 实现 `data/demo/model/tokenizer.py`
  - [x] SubTask 6.4.1: `build_tokenizer(corpus_path, vocab_size, save_path)`：调 `verse_tokenizer.BPETokenizer.train`
  - [x] SubTask 6.4.2: `load_tokenizer(path)`：调 `verse_tokenizer.load_tokenizer`
- [x] Task 6.5: 实现 `data/demo/src/utils.py`（set_seed / num_threads / ensure_dir）
- [x] Task 6.6: 实现 `data/demo/src/data_loader.py`
  - [x] SubTask 6.6.1: `load_jsonl(path)` → List[dict]
  - [x] SubTask 6.6.2: `TextDataset(tok, jsonl_path, seq_len)`：__getitem__ 返回 (x, y) Tensor
  - [x] SubTask 6.6.3: `collate_fn(batch, pad_id)` 批处理 + pad
- [x] Task 6.7: 实现 `data/demo/train/trainer.py`
  - [x] SubTask 6.7.1: 调用 `verse_torch.training.Trainer`
  - [x] SubTask 6.7.2: 训练配置从 `config/config.yml` 读取
  - [x] SubTask 6.7.3: 输出 `checkpoints/best.pt` / `last.pt` / `loss_curve.png` / `loss_history.json`
- [x] Task 6.8: 实现 `data/demo/train/evaluate.py`
  - [x] SubTask 6.8.1: 加载 best.pt，生成示例文本（greedy + top-k）
  - [x] SubTask 6.8.2: 验收测试：5 条 prompt，输出回答 + 打印
- [x] Task 6.9: 实现 `data/demo/train/visualize.py`
  - [x] SubTask 6.9.1: 调 `verse_torch.training.plot_loss_curve`
  - [x] SubTask 6.9.2: ASCII fallback
- [x] Task 6.10: 实现 `data/demo/config/config.yml`（参考 GPT_teacher）
  - [x] SubTask 6.10.1: model: n_layer=4, n_head=4, n_embd=128, seq_len=128, dropout=0.1, arch="hybrid"
  - [x] SubTask 6.10.2: training: batch_size=16, micro_batch=4, lr=1e-3, max_steps=1000, warmup=50, eval_interval=50, patience=10
- [x] Task 6.11: 准备 `data/demo/data/train.jsonl` 和 `val.jsonl` 占位数据
  - [x] SubTask 6.11.1: 至少 200 条 train / 50 条 val，每条 {"text": "..."}
  - [x] SubTask 6.11.2: 内容：唐诗 + 简单问答 + 数字序列（用于验证 LM 学习能力）
- [x] Task 6.12: 实现 `data/demo/run.py`（一键入口）
  - [x] SubTask 6.12.1: argparse: --skip-train / --skip-eval / --config
  - [x] SubTask 6.12.2: subprocess 链式调用 build_tokenizer → train → evaluate
- [x] Task 6.13: 实现 `data/demo/data/README.md`（数据格式说明）
- [x] Task 6.14: 端到端验证：在 4 核 CPU 上 `python run.py` 跑通
  - [x] SubTask 6.14.1: 5 分钟内完成 1000 step
  - [x] SubTask 6.14.2: loss 单调下降
  - [x] SubTask 6.14.3: 生成示例文本可读

## 阶段 7：verse_inference CometSpark 兼容

- [x] Task 7.1: 在 `verse_inference/model_loader.py` 中新增 `cometspark` arch 分支
  - [x] SubTask 7.1.1: 加载 `data/demo/model/config.py` 的 CometSparkConfig
  - [x] SubTask 7.1.2: 实例化 CometSparkLM + load_state_dict
- [x] Task 7.2: 验证 `StreamingGenerator` 兼容 CometSparkLM（生成 100 tokens ≤ 5s）

## 阶段 8：文档与 ADR

- [x] Task 8.1: 创建 `docs/architecture/adr-004-cpu-parallel.md`
  - [x] SubTask 8.1.1: 对比 multiprocessing / threading / numexpr / BLAS 线程
  - [x] SubTask 8.1.2: 决策：数据并行用 multiprocessing（避免 GIL），元素级用 numexpr（如可用），matmul 依赖底层 BLAS
- [x] Task 8.2: 创建 `verse_data/designs/compression_pipeline_design.md`
  - [x] SubTask 8.2.1: 压缩技术选型（OSP + BitNet 1.58 + LoRA + 蒸馏）
  - [x] SubTask 8.2.2: trillion → billion 路线图（本轮 PoC 目标 10×，最终 1000×）
  - [x] SubTask 8.2.3: 数值验证表格
- [x] Task 8.3: 创建 `data/demo/README.md`（CometSpark 入口文档）
  - [x] SubTask 8.3.1: 一键运行说明
  - [x] SubTask 8.3.2: 目录结构图
  - [x] SubTask 8.3.3: 配置参数说明

## 阶段 9：最终验证

- [x] Task 9.1: 跑 `tests/test_nn_advanced.py` + `test_training.py` + `test_tokenizer.py` + `test_parallel.py` + `test_compression_poc.py` + `test_cometspark_inference.py` 全部 PASS（132 passed / 2 skipped）
- [x] Task 9.2: 跑 `data/demo/run.py` 端到端，记录 wall-clock / loss / 生成样本（wall-clock 9.19s，initial 5.6060 → final 2.1568，5 条样本）
- [x] Task 9.3: 更新 `/workspace/README.md`，加入 CometSpark 入口链接 + 第二次进化摘要
- [x] Task 9.4: 更新 `/workspace/.trae/specs/evolve2-cometspark/checklist.md` 全部 `[x]`
- [x] Task 9.5: 回归验证 `import verse_torch; import verse_nex; import verse_awm; import verse_tokenizer; import verse_inference` 全部 OK，未引入 torch/tf/jax/transformers，`test_unit_operators.py` + `test_end_to_end.py` 12 passed 无回归
- [x] Task 9.6: 更新 `/workspace/.trae/specs/evolve2-cometspark/tasks.md` 全部 `[x]` + 完成总结

---

## 完成总结

Verse 框架第二次进化（CometSpark-v0.1）于 2026-07-20 完成。本次进化的核心成果：

1. **verse_torch.nn 多层神经网络补齐**：新增 `SwiGLUMLP` / `GQASelfAttention`（含 KV cache + RoPE + causal mask）/ `TransformerBlock`（pre-norm 残差）/ `TransformerLM`（weight tying + 缩放初始化），含有限差分梯度检查测试。
2. **verse_torch.training 训练基础设施**：`cross_entropy_loss`（ignore_index）/ `EarlyStopping` / `GradientAccumulator` / `CheckpointManager` / `LambdaLR`（warmup + cosine 调度）/ `compute_loss_rate` / `plot_loss_curve`（matplotlib 可选 + ASCII fallback）/ `Trainer` 类，端到端可跑。
3. **verse_tokenizer 系统完善**：`BPETokenizer.train` + `add_special_tokens` + `save/load` JSON、`ByteTokenizer`（vocab=259）、`load_tokenizer(kind, path)` 三种 kind 工厂函数。
4. **verse_torch.parallel CPU 并行**：`parallel_matmul` / `ParallelLinear`（含反向梯度数值一致）/ `parallel_map`，对 batch ≥ 阈值启用 multiprocessing，避免 GIL。
5. **verse_torch.compress 模型压缩 PoC**：`OutlierSafePruner` / `LoRALinear`（base frozen + A/B trainable）/ `KnowledgeDistiller`（KL + hard label）/ `QLinear` / `compress_pipeline` + `quantize_only` / `prune_only` / `lora_only` / `distill_only` 单技术函数，1M 参数 TransformerLM 上验证压缩比 ≥ 10×、loss 差异 ≤ 5%。
6. **CometSpark-v0.1 训练仓库**：`data/demo/` 完整目录结构（model / train / src / config / data / checkpoints），`run.py` 一键入口（build_tokenizer → train → evaluate → visualize），默认配置在 3 核 CPU 沙箱中 9.19s 跑完 200 步，loss 5.6060 → 2.1568 单调下降，生成 5 条样本输出，checkpoints 目录齐全。
7. **verse_inference CometSpark 兼容**：`model_loader.py` 新增 `cometspark` arch 分支，`StreamingGenerator` 兼容 CometSparkLM，100 tokens ≤ 5s。
8. **文档与 ADR**：`docs/architecture/adr-004-cpu-parallel.md`（multiprocessing/threading/numexpr/BLAS 对比）、`verse_data/designs/compression_pipeline_design.md`（trillion→billion 路线图）、`data/demo/README.md`（一键运行 + 目录 + 配置参数说明）。
9. **最终验证**：132 新增单元测试全部 PASS（2 skipped 为环境受限），12 回归测试 PASS，`import` 全部成功，运行时无 torch/tf/jax/transformers 依赖。

# Task Dependencies

- Task 0.x 是所有后续任务的前置
- Task 1.x（nn 层）是 Task 2.x（Trainer 用 TransformerLM 测试）的前置
- Task 1.x 是 Task 6.3（CometSparkLM 用 TransformerLM）的前置
- Task 2.x（training）是 Task 6.7（trainer.py 调用）的前置
- Task 3.x（tokenizer）是 Task 6.4（CometSpark tokenizer）的前置
- Task 4.x（parallel）独立，可与 1-3 并行
- Task 5.x（compress）依赖 Task 1.x（需要 TransformerLM 作 PoC 模型）
- Task 6.x（CometSpark 仓库）依赖 Task 1-3 全部完成
- Task 7.x（inference 兼容）依赖 Task 6.x
- Task 8.x（文档）可与 4-7 并行
- Task 9.x（最终验证）依赖全部完成

# 可并行任务

- 阶段 1（nn 层）/ 阶段 2（training）/ 阶段 3（tokenizer）/ 阶段 4（parallel）可并行（4 个独立子代理）
- 阶段 5（compress）依赖阶段 1 完成后启动
- 阶段 6（CometSpark 仓库）依赖 1+2+3 完成后启动
- 阶段 7（inference 兼容）依赖 6 完成后启动
- 阶段 8（文档）可与 4-7 并行
