# Tasks — Part3K2：CometSpark + VerseNext 重大升级

按依赖顺序排列，独立任务可并行。每个任务完成后请勾选对应 checkbox。

- [x] Task 1: 训练数据格式现代化（chat + prompt-completion，BREAKING）
  - [x] SubTask 1.1: 重写 `data/demo/src/data_loader.py` 的 `TextDataset`，支持 chat 数组与 prompt-completion 双格式自动检测，loss mask 屏蔽 prompt 部分（`ignore_index=-100`）
  - [x] SubTask 1.2: 完全重写 `data/demo/data/train.jsonl` 与 `val.jsonl` 为新格式（保留唐诗/问答/数字序列三类数据，转为 chat 数组 + prompt-completion 混合），删除旧 `{"text":"..."}` 行
  - [x] SubTask 1.3: 更新 `data/demo/data/README.md` 文档新格式说明
  - [x] SubTask 1.4: 新增 `tests/test_chat_data_loader.py`，覆盖 chat/prompt-completion/拒绝 text 格式/loss mask 四种场景
  - [x] SubTask 1.5: 更新 `data/demo/run.py` 验证端到端可跑通新格式

- [x] Task 2: Tokenizer 全面升级
  - [x] SubTask 2.1: 在 `verse_tokenizer` 新增 `preprocess.py`，实现 GPT-2/GPT-4 风格正则预分词（中文整字、英文单词、数字、标点、空白独立成块）+ NFKC 归一化
  - [x] SubTask 2.2: 升级 `bpe.py`：`BPETokenizer` 接入正则预分词，支持 `vocab_size` 自适应、特殊 token 注册（bos/eos/pad/unk + `<|user|>`/`<|assistant|>`/`<|system|>`）、`add_special_tokens` 编码开关
  - [x] SubTask 2.3: 新增 `chat_template.py`，实现 `apply_chat_template(messages)` 与 `apply_prompt_template(prompt)`，并集成到 `BaseTokenizer` ABC
  - [x] SubTask 2.4: 新增 `unigram.py`，实现 `SentencePieceUnigramTokenizer`（EM 训练 + Viterbi 解码）
  - [x] SubTask 2.5: 升级 `byte.py` / `char.py`：统一 NFKC + byte-aligned decode（保证与 BPE 行为一致），实现 `apply_chat_template`（byte/char 版本使用简化模板）
  - [x] SubTask 2.6: 更新 `verse_tokenizer/__init__.py` 导出新 API
  - [x] SubTask 2.7: 新增 `tests/test_tokenizer_upgrade.py`，覆盖正则预分词/chat template/unigram/byte-aligned decode/中文乱码防护

- [x] Task 3: 对齐 Transformer / PyTorch 补齐能力
  - [x] SubTask 3.1: 在 `verse_torch` 新增 `optim_extras.py`，实现 `Lion`（sign 更新）与 `Adafactor`（factored 二阶矩）
  - [x] SubTask 3.2: 新增 `scheduler_extras.py`，实现 `OneCycleLR`、`ReduceLROnPlateau`、`CosineRestartsLR`
  - [x] SubTask 3.3: 新增 `activations.py`，实现 `GeGLU`、`Mish`、`SiLU`（别名）
  - [x] SubTask 3.4: 升级 `nn.py`：新增 `SlidingWindowAttention`、`ALiBi` 位置偏置、`DeepNorm`
  - [x] SubTask 3.5: 升级 `losses.py`：新增 `focal_loss`，确认 `label_smoothing` 参数与 HF 对齐（默认 0.0）
  - [x] SubTask 3.6: 升级 `training.py` 的 `BatchLoader` 对齐 `torch.utils.data.DataLoader` 接口（`num_workers`/`pin_memory`/`persistent_workers` 占位参数）
  - [x] SubTask 3.7: 检查并减少依赖：确认 `requests` 软依赖可移除，`numba` 软依赖路径合并
  - [x] SubTask 3.8: 更新 `verse_torch/__init__.py` 导出新 API
  - [x] SubTask 3.9: 新增 `tests/test_optim_extras.py` 与 `tests/test_scheduler_extras.py`

- [x] Task 4: CometSpark 架构升级 + 压缩深度集成
  - [x] SubTask 4.1: 升级 `data/demo/model/config.py`：新增 `rope_theta` / `max_position_embeddings` / `attention_dropout` / `hidden_dropout` / `embedding_dropout` 字段，提供 `from_pretrained` / `save_pretrained` 类方法
  - [x] SubTask 4.2: 升级 `data/demo/model/model.py`：`CometSparkLM` 应用新配置（RoPE theta、分离的 dropout、max_position_embeddings 与 seq_len 分离），新增 `from_pretrained` / `save_pretrained` / `compress(compress_config)` / `compression_stats()` 方法
  - [x] SubTask 4.3: 新增 `CometSparkSmall()` / `CometSparkMedium()` / `CometSparkLarge()` 工厂函数
  - [x] SubTask 4.4: 在 `verse_torch/compress.py` 新增 `compress_pipeline` 一键管线 API（已部分存在，升级支持任意组合 prune+quantize+lora+ternary+distill），并暴露 `CometSparkLM.compress` 调用入口
  - [x] SubTask 4.5: 新增 `examples/compress_train_demo.py`，演示 prune→quantize→finetune→evaluate 完整流程
  - [x] SubTask 4.6: 新增 `tests/test_compression_integration.py`，覆盖 compress/compression_stats/from_pretrained/save_pretrained/工厂函数

- [x] Task 5: 并行训练支持（基础框架 + 训练体系）
  - [x] SubTask 5.1: 在 `verse_torch/training.py` 新增 `ParallelTrainer` 类，支持步数拆分（`parallel_chunks` 配置）+ 串行执行 chunk（CPU-only 避免 GIL 竞争）
  - [x] SubTask 5.2: 实现合并策略：对比每个 chunk 的 train_loss + val_loss，**效果差放前面、好的放后面**串行重训
  - [x] SubTask 5.3: **修复 val_loss 更新漏洞**：新增 `_eval_full_val()` 方法，每个 chunk 完成后基于完整 val 数据集更新 `best_val_loss` 与 `best_state_dict`（不再用 batch 局部 val）
  - [x] SubTask 5.4: 实现整体 fine-tune（`merge_finetune_steps = max_steps // 10`），优化 loss 较差的部分
  - [x] SubTask 5.5: 在 `data/demo/train/trainer.py` 接入 `ParallelTrainer`：根据 `training.parallel_chunks` 配置选择 `Trainer` 或 `ParallelTrainer`
  - [x] SubTask 5.6: 更新 `data/demo/config/config*.yml`，新增 `parallel_chunks`（默认 1，即不并行）字段
  - [x] SubTask 5.7: 新增 `tests/test_parallel_trainer.py`，覆盖 4 chunk 拆分/合并排序/val_loss 完整更新/整体 fine-tune 四种场景
  - [x] SubTask 5.8: 验证 4 chunk 并行训练效果尽可能等同于一体训练（val_loss 差距 < 5%）

- [x] Task 6: 训练框架推理 + 自由温度 + 测试打分
  - [x] SubTask 6.1: 在 `verse_torch/training.py` 的 `Trainer` 新增 `inference(prompts, temperature, top_k, top_p, max_tokens)` 方法
  - [x] SubTask 6.2: 升级 `data/demo/train/evaluate.py` 为 `ScoringEvaluator` 类，实现 `exact_match` / `prefix_accuracy` / `char_f1` / `bleu` / `rouge_l` 五个指标
  - [x] SubTask 6.3: 实现 `ScoringEvaluator.evaluate(prompts, references) -> dict` 与 `report(score_dict) -> str`
  - [x] SubTask 6.4: 更新 `data/demo/run.py`：新增 `--score` / `--references-file` / `--top-p` 参数，`--score` 启用打分模式
  - [x] SubTask 6.5: 新增 `tests/test_scoring.py`，覆盖五个指标 + 报告格式
  - [x] SubTask 6.6: 更新 `data/demo/README.md`：补充推理 + 打分 CLI 用法

- [x] Task 7: 全项目 check-loop 审计 + BUG 清零
  - [x] SubTask 7.1: 用 search sub-agent 扫描所有 packages 与 data/demo，列出严重错误（递归/NaN/乱码/import 循环/路径硬编码）
  - [x] SubTask 7.2: 列出漏洞（资源泄漏/除零/整数溢出/未捕获异常）
  - [x] SubTask 7.3: 列出可优化部分（重复代码、低效算法、冗余 import、死代码）
  - [x] SubTask 7.4: 合并重复实现：`losses.cross_entropy` 与 `training.cross_entropy_loss` 内部共用；NFKC 实现合并到 `verse_tokenizer/preprocess.py`
  - [x] SubTask 7.5: 修复所有发现的 BUG/漏洞/错误，每项修复配套测试
  - [x] SubTask 7.6: 在项目根目录生成 `audit_report.md`（check-loop 报告，列出发现 + 修复状态 + 测试覆盖）
  - [x] SubTask 7.7: 跑全量测试（`pytest tests/`）确保零失败

# Task Dependencies

- Task 1（数据格式）与 Task 2（tokenizer）相互依赖：`TextDataset` 调用 `apply_chat_template` / `apply_prompt_template`，需 Task 2.3 先完成 chat_template 接口；建议 Task 2.3 与 Task 1.1 同步推进。
- Task 3（对齐 PyTorch）独立，可与 Task 1/2/4/5/6 并行。
- Task 4（架构升级 + 压缩）依赖 Task 3.4（SlidingWindowAttention 等可选，但 CometSpark 不强制使用）。
- Task 5（并行训练）依赖 Task 1（新数据格式）与 Task 2（tokenizer），且依赖 Task 6.1（inference 方法）用于 chunk 间生成验证（可选）。
- Task 6（推理 + 打分）依赖 Task 1（新数据格式用于 references）。
- Task 7（check-loop）应在 Task 1-6 全部完成后执行（最终审计）。
