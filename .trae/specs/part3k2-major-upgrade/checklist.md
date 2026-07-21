# Checklist — Part3K2：CometSpark + VerseNext 重大升级

每个 checkpoint 验证完毕后请勾选。失败的 checkpoint 需在 `tasks.md` 新增修复任务并重新验证。

## Task 1: 训练数据格式现代化

- [ ] `TextDataset` 支持自动检测 chat 数组与 prompt-completion 两种格式
- [ ] `TextDataset` 拒绝旧 `{"text":"..."}` 格式，抛出明确 `ValueError`
- [ ] chat 数组样本通过 `apply_chat_template` 包装为 `<|user|>...<|assistant|>...<|eos|>` 序列
- [ ] prompt-completion 样本拼接为 `prompt<|assistant|>completion<|eos|>`
- [ ] loss mask 正确屏蔽 prompt 部分（`ignore_index=-100`），仅 completion 参与 loss
- [ ] `train.jsonl` / `val.jsonl` 完全重写为新格式，无 `{"text":"..."}` 行
- [ ] `data/README.md` 文档已更新新格式说明
- [ ] `tests/test_chat_data_loader.py` 覆盖 chat/prompt-completion/拒绝 text/loss mask 四种场景
- [ ] `python data/demo/run.py` 端到端跑通新格式

## Task 2: Tokenizer 全面升级

- [ ] `verse_tokenizer/preprocess.py` 实现 GPT-2/GPT-4 风格正则预分词（中文整字独立成块）
- [ ] `BPETokenizer` 接入正则预分词，`vocab_size` 自适应
- [ ] `BPETokenizer` 支持注册特殊 token（bos/eos/pad/unk + `<|user|>`/`<|assistant|>`/`<|system|>`）
- [ ] `BPETokenizer` 支持 `add_special_tokens` 编码开关
- [ ] `chat_template.py` 实现 `apply_chat_template` 与 `apply_prompt_template`
- [ ] `BaseTokenizer` ABC 新增 `apply_chat_template` / `apply_prompt_template` / `special_tokens` / `add_special_tokens` 接口
- [ ] `unigram.py` 实现 `SentencePieceUnigramTokenizer`（EM 训练 + Viterbi 解码）
- [ ] `ByteTokenizer` / `CharTokenizer` 统一 NFKC + byte-aligned decode
- [ ] `verse_tokenizer/__init__.py` 导出新 API
- [ ] `tests/test_tokenizer_upgrade.py` 覆盖正则预分词/chat template/unigram/byte-aligned/中文乱码防护
- [ ] 中文样本 encode + decode 无 U+FFFD 乱码

## Task 3: 对齐 Transformer / PyTorch 补齐能力

- [x] `optim_extras.py` 实现 `Lion`（sign 更新，无二阶矩）
- [x] `optim_extras.py` 实现 `Adafactor`（factored 二阶矩）
- [x] `scheduler_extras.py` 实现 `OneCycleLR` / `ReduceLROnPlateau` / `CosineRestartsLR`
- [x] `activations.py` 实现 `GeGLU` / `Mish` / `SiLU` 别名
- [x] `nn.py` 新增 `SlidingWindowAttention`
- [x] `nn.py` 新增 `ALiBi` 位置偏置
- [x] `nn.py` 新增 `DeepNorm`
- [x] `losses.py` 新增 `focal_loss`
- [x] `losses.py` 的 `label_smoothing` 默认参数对齐 HF（`0.0` 关闭）
- [x] `BatchLoader` 对齐 `torch.utils.data.DataLoader` 接口（占位参数）
- [x] 减少依赖：`requests` 软依赖移除，`numba` 软依赖路径合并
- [x] `verse_torch/__init__.py` 导出新 API
- [x] `tests/test_optim_extras.py` 覆盖 Lion/Adafactor
- [x] `tests/test_scheduler_extras.py` 覆盖 OneCycle/ReduceLROnPlateau/CosineRestarts

## Task 4: CometSpark 架构升级 + 压缩深度集成

- [ ] `CometSparkConfig` 新增 `rope_theta` / `max_position_embeddings` / `attention_dropout` / `hidden_dropout` / `embedding_dropout`
- [ ] `CometSparkConfig` 支持 `from_pretrained(dir)` / `save_pretrained(dir)`
- [ ] `CometSparkLM` 应用新配置（RoPE theta、分离 dropout、max_position_embeddings 与 seq_len 分离）
- [ ] `CometSparkLM` 新增 `from_pretrained(dir)` / `save_pretrained(dir)` 类方法
- [ ] `CometSparkLM.compress(compress_config)` 一键应用压缩管线
- [ ] `CometSparkLM.compression_stats()` 返回压缩前/后参数量、稀疏度、bit 数
- [ ] `CometSparkSmall()` / `CometSparkMedium()` / `CometSparkLarge()` 工厂函数
- [ ] `compress_pipeline` 支持任意组合 prune+quantize+lora+ternary+distill
- [ ] `examples/compress_train_demo.py` 演示完整压缩训练流程
- [ ] `tests/test_compression_integration.py` 覆盖 compress/compression_stats/from_pretrained/save_pretrained/工厂函数

## Task 5: 并行训练支持

- [ ] `ParallelTrainer` 类实现，支持步数拆分（`parallel_chunks`）
- [ ] 合并策略：效果差放前面、好的放后面串行重训
- [ ] **val_loss 修复**：`_eval_full_val()` 基于完整 val 数据集更新 best_val_loss 与 best_state_dict
- [ ] 整体 fine-tune（`merge_finetune_steps = max_steps // 10`）
- [ ] `data/demo/train/trainer.py` 根据 `parallel_chunks` 选择 `Trainer` 或 `ParallelTrainer`
- [ ] `config*.yml` 新增 `parallel_chunks` 字段（默认 1）
- [ ] `tests/test_parallel_trainer.py` 覆盖 4 chunk 拆分/合并排序/val_loss 完整更新/整体 fine-tune
- [ ] 4 chunk 并行训练 val_loss 与一体训练差距 < 5%

## Task 6: 训练框架推理 + 自由温度 + 测试打分

- [ ] `Trainer.inference(prompts, temperature, top_k, top_p, max_tokens)` 方法实现
- [ ] `ScoringEvaluator` 类实现 `exact_match` 指标
- [ ] `ScoringEvaluator` 实现 `prefix_accuracy` 指标
- [ ] `ScoringEvaluator` 实现 `char_f1` 指标
- [ ] `ScoringEvaluator` 实现 `bleu` 指标（BLEU-4 简化版）
- [ ] `ScoringEvaluator` 实现 `rouge_l` 指标
- [ ] `ScoringEvaluator.evaluate(prompts, references) -> dict` 返回多指标分数
- [ ] `ScoringEvaluator.report(score_dict) -> str` 返回可读报告
- [ ] `run.py` 新增 `--score` / `--references-file` / `--top-p` 参数
- [ ] `tests/test_scoring.py` 覆盖五个指标 + 报告格式
- [ ] `data/demo/README.md` 补充推理 + 打分 CLI 用法

## Task 7: 全项目 check-loop 审计 + BUG 清零

- [ ] 严重错误扫描完成（递归/NaN/乱码/import 循环/路径硬编码）
- [ ] 漏洞扫描完成（资源泄漏/除零/整数溢出/未捕获异常）
- [ ] 可优化部分扫描完成（重复代码/低效算法/冗余 import/死代码）
- [ ] `losses.cross_entropy` 与 `training.cross_entropy_loss` 内部共用实现
- [ ] NFKC 实现合并到 `verse_tokenizer/preprocess.py`
- [ ] 所有发现的 BUG/漏洞/错误已修复，每项配套测试
- [ ] `audit_report.md` 在项目根目录生成（列出发现 + 修复状态 + 测试覆盖）
- [ ] `pytest tests/` 全量测试零失败

## 综合验收

- [ ] 所有 packages 测试通过（verse_torch / verse_nex / verse_tokenizer / verse_inference / verse_awm / verse_compat）
- [ ] `python data/demo/run.py` 默认配置端到端跑通（包含新数据格式 + 升级 tokenizer）
- [ ] `python data/demo/run.py --score --references-file refs.txt` 打分模式跑通
- [ ] `python data/demo/run.py --config config/config_medium.yml` medium 配置跑通
- [ ] `python data/demo/run.py` 配合 `parallel_chunks=4` 配置跑通并行训练
- [ ] `examples/compress_train_demo.py` 演示脚本跑通
- [ ] `audit_report.md` 报告完整，无未修复项
- [ ] `data/demo/README.md` 文档与实现一致
