# Checklist

> 用于阶段验证。每完成一项检查后，将对应 `[ ]` 改为 `[x]`。

## 阶段 0：现有代码基线确认

- [x] 已读取 `verse_torch/nn.py` 确认 Linear/Embedding/LayerNorm/RMSNorm/Dropout/ModuleList/Sequential 可用
- [x] 已读取 `verse_nex/positional.py` 确认 RoPE 可用
- [x] 已读取 `verse_nex/hybrid.py` 确认 HybridLM 可用
- [x] 已读取 `verse_tokenizer/bpe.py` 确认现有 BPE 接口

## 阶段 1：verse_torch.nn 多层神经网络补齐

- [x] `SwiGLUMLP` 实现，hidden 对齐 64，forward shape 正确
- [x] `GQASelfAttention` 实现，含 wq/wk/wv/proj，支持 kv_cache，自动 RoPE + causal mask
- [x] `TransformerBlock` 实现，pre-norm 残差结构
- [x] `TransformerLM` 实现，含 tok_emb + N × Block + RMSNorm + head，支持 tie_weights
- [x] `Parameter` 别名导出
- [x] `tests/test_nn_advanced.py` 全部 PASS（含有限差分梯度检查）

## 阶段 2：verse_torch.training 训练基础设施

- [x] `cross_entropy_loss(logits, targets, ignore_index)` 实现，与手动计算一致
- [x] `EarlyStopping(patience, min_delta)` 实现，触发逻辑正确
- [x] `GradientAccumulator(micro_batch, effective_batch)` 实现
- [x] `CheckpointManager(save_dir)` 实现，save_best/save_last/load_best/load_last 可用
- [x] `LambdaLR(optimizer, lr_lambda)` 调度器补充到 `optim.py`
- [x] `compute_loss_rate(loss_window, window, min_delta)` 滑动窗口下降率实现
- [x] `plot_loss_curve(train_losses, val_losses, save_path)` 实现，matplotlib 可选 + ASCII fallback
- [x] `Trainer` 类实现，fit/evaluate 方法可用
- [x] `tests/test_training.py` 全部 PASS

## 阶段 3：verse_tokenizer 系统完善

- [x] `BPETokenizer.train(corpus, vocab_size)` 类方法实现
- [x] `BPETokenizer.add_special_tokens(tokens)` 实现
- [x] `BPETokenizer.save(path)` / `load(path)` JSON 持久化实现
- [x] `ByteTokenizer` 实现（vocab_size=259）
- [x] `load_tokenizer(kind, path)` 工厂函数实现，三种 kind 均返回统一接口
- [x] `verse_tokenizer/__init__.py` 导出全部公共 API
- [x] `tests/test_tokenizer.py` 全部 PASS

## 阶段 4：verse_torch.parallel CPU 并行计算

- [x] `parallel_matmul(A, B, n_workers)` 实现，数值与 np.matmul 一致
- [x] `ParallelLinear(d_in, d_out, n_workers, batch_threshold)` 实现，反向梯度正确
- [x] `parallel_map(fn, iterable, n_workers)` 通用并行 map 实现
- [x] `tests/test_parallel.py` 全部 PASS
- [x] `docs/benchmarks/parallel_benchmark.md` 生成，记录加速比

## 阶段 5：verse_torch.compress 模型压缩 PoC

- [x] `docs/papers/compression_references.md` 收集 ≥ 10 篇参考
- [x] `OutlierSafePruner(model, sparsity)` 实现，输出剪枝报告
- [x] `LoRALinear(d_in, d_out, r, alpha)` 实现，base frozen + A/B trainable
- [x] `KnowledgeDistiller(teacher, student, T, alpha)` 实现，KL + hard label 联合
- [x] `compress_pipeline(model, target_ratio, eval_fn)` 端到端流程实现
- [x] `quantize_only` / `prune_only` / `lora_only` / `distill_only` 单技术函数实现
- [x] `tests/test_compression_poc.py` 在 1M 参数小模型上验证：
  - [x] 压缩比 ≥ 10×
  - [x] loss 差异 ≤ 5%
- [x] `docs/benchmarks/compression_poc.md` 生成对照表

## 阶段 6：CometSpark-v0.1 训练仓库

- [x] `data/demo/` 目录结构完整（model/data/train/src/config/checkpoints）
- [x] `model/config.py`：CometSparkConfig dataclass + from_yaml/to_yaml
- [x] `model/model.py`：CometSparkLM 支持 hybrid + transformer 两种 arch
- [x] `model/tokenizer.py`：build_tokenizer + load_tokenizer
- [x] `src/utils.py`：set_seed / num_threads / ensure_dir
- [x] `src/data_loader.py`：load_jsonl + TextDataset + collate_fn
- [x] `train/trainer.py`：调用 verse_torch.training.Trainer
- [x] `train/evaluate.py`：加载 best.pt 生成示例文本 + 验收测试
- [x] `train/visualize.py`：plot_loss_curve + ASCII fallback
- [x] `config/config.yml`：默认配置（n_layer=4, n_embd=128, max_steps=1000）
- [x] `data/train.jsonl` ≥ 200 条，`data/val.jsonl` ≥ 50 条
- [x] `run.py`：一键入口，subprocess 链式调用
- [x] `data/README.md`：数据格式说明
- [x] 端到端验证：`python run.py` 在 4 核 CPU 上 5 分钟内完成 1000 step
- [x] loss 单调下降，生成示例文本可读

## 阶段 7：verse_inference CometSpark 兼容

- [x] `verse_inference/model_loader.py` 新增 `cometspark` arch 分支
- [x] `StreamingGenerator` 兼容 CometSparkLM，100 tokens ≤ 5s

## 阶段 8：文档与 ADR

- [x] `docs/architecture/adr-004-cpu-parallel.md` 完整（含 multiprocessing/threading/numexpr/BLAS 对比）
- [x] `verse_data/designs/compression_pipeline_design.md` 完整（含 trillion→billion 路线图）
- [x] `data/demo/README.md` 完整（一键运行说明 + 目录结构 + 配置参数）

## 阶段 9：最终验证

- [x] 所有新增单元测试 PASS（test_nn_advanced / test_training / test_tokenizer / test_parallel / test_compression_poc / test_cometspark_inference，共 132 passed / 2 skipped）
- [x] `python data/demo/run.py` 端到端跑通，记录 wall-clock / loss / 生成样本（wall-clock 9.19s，initial 5.6060 → final 2.1568，5 条生成样本）
- [x] `/workspace/README.md` 加入 CometSpark 入口链接 + 第二次进化摘要
- [x] `import verse_torch, verse_nex, verse_awm, verse_tokenizer, verse_inference` 全部成功（无回归）
- [x] 运行时仍不依赖 `torch` / `transformers` / `tensorflow` / `jax`（sys.modules 检查无输出）
- [x] checklist.md 全部 `[x]`
