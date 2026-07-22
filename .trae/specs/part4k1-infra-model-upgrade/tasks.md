# Tasks — Part4K1：基础设施全面升级 + 模型能力升级 + 优化

按依赖顺序排列，独立任务可并行。每个任务完成后请勾选对应 checkbox。

## 阶段 A：基础设施升级

- [x] Task 1: VerseTorch GPU/NPU 设备抽象层 + 缺失组件补齐
  - [x] SubTask 1.1: 新增 `verse_torch/device.py`，实现 `DeviceBackend` 抽象基类 + `NumpyBackend`（默认 CPU）+ `TorchBackend`（PyTorch 委托，支持 cuda/mps/npu via torch_npu）
  - [x] SubTask 1.2: 升级 `verse_torch/tensor.py`：`Tensor` 新增 `device` 属性 + `.to(device)` / `.cuda()` / `.npu()` / `.cpu()` 方法；GPU 下委托 PyTorch autograd，CPU 下保持自研 autograd
  - [x] SubTask 1.3: 新增 `verse_torch/backend_torch.py`，实现 PyTorch 委托后端（matmul / linear / attention / softmax 等 op 委托；CUDA kernel 走 PyTorch，NPU 走 torch_npu，不自研 kernel）
  - [x] SubTask 1.4: 升级 `verse_torch/nn.py`：Module 新增 `.to(device)` / `.device` 属性；新增 `RotaryEmbedding`（RoPE 独立类）、`KVCache` 抽象、`StaticCache` / `DynamicCache`、`GroupNorm`、`Conv1d`、`LayerNorm` 优化版
  - [x] SubTask 1.5: 升级 `verse_torch/optim.py`：新增 `NAdamW`、`RMSProp`
  - [x] SubTask 1.6: 升级 `verse_torch/losses.py`：新增 `contrastive_loss`（RL/DPO 备选）、`perplexity`
  - [x] SubTask 1.7: 升级 `verse_torch/training.py`：新增 `DistributedTrainer` 占位接口（多卡数据并行 API 预留）；混合精度 `autocast` 支持（GPU 后端）
  - [x] SubTask 1.8: 更新 `verse_torch/__init__.py` 导出新 API；升级 `pyproject.toml`（`torch` 为可选依赖）
  - [x] SubTask 1.9: 新增 `tests/test_device_backend.py`，覆盖 NumpyBackend / TorchBackend / device 迁移 / 无 PyTorch 回退 / autocast

- [x] Task 2: VerseNex 品牌落地 + 类重命名 + MoD 完善验证
  - [x] SubTask 2.1: 在 `verse_nex` 内重命名 `TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention`，统一 `VerseNexBlock` 为唯一名；更新 `cometspark.py` / `__init__.py` / 相关导入
  - [x] SubTask 2.2: 在 `verse_torch.nn` 保留旧名 `TransformerLM` / `TransformerBlock` / `GQASelfAttention` 作为 `DeprecationWarning` 别名（一个版本）
  - [x] SubTask 2.3: 完善 `verse_nex/moe.py` 的 `MoDLayer`：5 DensePart × 8 Expert × top-3 双层门控，aux loss 计算；补齐 load_balance_loss + router z-loss
  - [x] SubTask 2.4: 新增 `tests/test_mod_complete.py`：MoD 前向 / 反向 / aux loss 收敛 / parallel-vs-recurrent 一致性
  - [x] SubTask 2.5: 统一 `config.yml` 的 `arch` 字段为 `versenex` 唯一值；`transformer` / `hybrid` 映射 + DeprecationWarning；标记 `HybridBlock` / `HybridLM` deprecated

- [x] Task 3: 超稀疏并行注意力机制优化
  - [x] SubTask 3.1: 升级 `verse_nex/tri_sparse_attn.py`：多 query chunk 并行计算 attention（批量矩阵化 / GPU 并行），消除串行循环
  - [x] SubTask 3.2: 新增 `verse_nex/speculative.py`：分离式并行预测 —— draft head 并行生成 k 个候选 token + 主模型一次前向验证 + verify-then-commit
  - [x] SubTask 3.3: 实现 KV cache 并行批量更新（`KVCache.batch_update`）
  - [x] SubTask 3.4: 新增 `tests/test_parallel_sparse_attn.py`：多 chunk 并行 vs 串行数值一致（float32 吻合 1e-3）+ GPU 吞吐 ≥ 2×
  - [x] SubTask 3.5: 新增 `tests/test_speculative_decode.py`：k=4 候选预测 + 接受最长正确前缀 + 拒绝处重 draft

- [x] Task 4: NexRL 优化强化学习算法
  - [x] SubTask 4.1: 新增 `verse_nex/nexrl/__init__.py`，定义 RL 五要素抽象：`NexAgent`（策略 + 参考网络 + KL 约束）/ `NexEnv`（observation + reward）/ `NexState`（prompt + tokens + KV cache）/ `NexAction`（token 动作 + 采样策略）/ `NexReward`（多维奖励）
  - [x] SubTask 4.2: 实现 `NexReward`：多维加权（correctness + fluency + safety + length_penalty）+ reward normalization（running mean/std）+ reward shaping（potential-based）
  - [x] SubTask 4.3: 实现 `NexAction`：动作空间采样（ε-greedy / softmax / nucleus）+ 探索衰减 schedule + 重复动作惩罚
  - [x] SubTask 4.4: 实现 `ParallelRolloutCollector`：多 prompt / 多 rollout 并行采样（batched），GPU 批量前向
  - [x] SubTask 4.5: 实现 `NexTrainer`：PPO 风格（clip ratio + GAE + KL 祖父项）+ 纯策略梯度 fallback；支持 `verse-posttrain --rl nexrl`
  - [x] SubTask 4.6: 新增 `tests/test_nexrl.py`：多维奖励 / 并行 rollout / KL 防崩溃 / 动作采样策略

- [ ] Task 5: VerseTokenizer 优化 + NexRL 集成
  - [x] SubTask 5.1: 升级 `verse_tokenizer/bpe.py`：BPE 训练支持 `min_frequency` / `max_token_length` / 并行 merge（多线程训练加速）
  - [x] SubTask 5.2: 新增 `verse_tokenizer/wordpiece.py`（WordPiece tokenizer）+ 升级 `unigram.py` 对齐 sentencepiece；编码/解码向量化（批量 encode/decode）
  - [x] SubTask 5.3: 对齐 HF `BatchEncoding`：`add_bos` / `add_eos` 独立开关 + `truncation` / `padding` 策略
  - [x] SubTask 5.4: 新增 `verse_tokenizer/nex_wrapper.py`：`NexTokenizerWrapper`，token 边界注入 RL 信号（reward-weighted token preference），高频高奖励子串优先成 token
  - [x] SubTask 5.5: 实现 `BPETokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")`，从 HuggingFace 下载 tokenizer.json（vocab 248320）加载
  - [x] SubTask 5.6: 新增 `tests/test_tokenizer_optimization.py`：并行 BPE 训练 / NexRL 集成 / Qwen tokenizer 加载 / wordpiece / BatchEncoding

- [x] Task 6: VerseTrainer 独立训练包（从 data/demo 剥离）
  - [x] SubTask 6.1: 创建 `packages/verse_infra/verse_trainer/`（暂在 verse_infra 占位，待 Task 7 完成物理迁移），新增 `pyproject.toml` + `__init__.py`
  - [x] SubTask 6.2: 把 `data/demo/train/` 训练代码迁入 `verse_trainer/`：`trainer.py`（ParallelTrainer 升级 + `_safe_chunk_run` + 断点续训）/ `evaluate.py`（ScoringEvaluator）/ `visualize.py`
  - [x] SubTask 6.3: 把 `data/demo/src/data_loader.py` 升级为 `CachedDataset`（首次扫描缓存 `.npz` + 流式 lazy load）；迁入 `verse_trainer/data.py`
  - [x] SubTask 6.4: 实现 CLI 入口 `verse_trainer/cli.py`：`verse-train` / `verse-finetune` / `verse-posttrain` / `verse-eval` / `verse-tokenize`（argparse + console_scripts）
  - [x] SubTask 6.5: 实现单样本支持：`--single-sample` 接受单条 prompt/completion 或单文件
  - [x] SubTask 6.6: 接入微调 / 后训练：复用 `SFTTrainer` / `DPOTrainer` / `LoRATrainer`；新增 `RLTrainer`（基于 NexRL）
  - [x] SubTask 6.7: 实现 Loss 优化策略（参考 GPT_teacher-3.37M-cn）：梯度裁剪 + LR warmup + cosine + ReduceLROnPlateau + loss plateau 重走（`_rollback_and_perturb`）+ NaN/Inf 跳过
  - [x] SubTask 6.8: 清除 `data/demo/` 旧实现 / 重复代码 / 死代码（在迁移过程中同步清理）
  - [x] SubTask 6.9: 新增 `tests/test_verse_trainer.py`：CLI 端到端 / 单样本 / CachedDataset 加速 / plateau 重走 / 断点续训

- [x] Task 7: VerseInfra 总包聚合（物理迁移 + 导入路径更新）
  - [x] SubTask 7.1: 创建 `packages/verse_infra/` 目录结构 + `pyproject.toml` + `verse_infra/__init__.py`（重导出公共 API）
  - [x] SubTask 7.2: 把 `packages/verse_compat/` 源码移动到 `packages/verse_infra/verse_infra/verse_compat/`；删除原顶层目录
  - [x] SubTask 7.3: 把 `packages/verse_inference/` 源码移动到 `packages/verse_infra/verse_infra/verse_inference/`；删除原顶层目录
  - [x] SubTask 7.4: 把 `packages/verse_tokenizer/` 源码移动到 `packages/verse_infra/verse_infra/verse_tokenizer/`；删除原顶层目录
  - [x] SubTask 7.5: 把 Task 6 的 `verse_trainer/` 放入 `packages/verse_infra/verse_infra/verse_trainer/`
  - [x] SubTask 7.6: 在原 `packages/verse_compat/`、`verse_inference/`、`verse_tokenizer/` 位置保留 thin shim（`from verse_infra.verse_xxx import *` + DeprecationWarning，一个版本）
  - [x] SubTask 7.7: 全项目导入路径更新：`tests/`、`examples/`、`data/`、`packages/verse_nex/`、`packages/verse_torch/`、`spark/`、`docs/` 中所有 `from verse_tokenizer/verse_compat/verse_inference import` → `from verse_infra.verse_xxx import`
  - [x] SubTask 7.8: 更新根 `pyproject.toml`：声明 `verse_infra`、`verse_trainer`（作为 verse_infra 子模块）包；删除旧包声明
  - [x] SubTask 7.9: 新增 `tests/test_verse_infra_imports.py`：子模块导入 / 便捷重导出 / 旧路径 shim DeprecationWarning

## 阶段 B：CometSpark V0.5-1B

- [x] Task 8: CometSpark V0.5-1B 模型迁移 + 完全重写
  - [x] SubTask 8.1: 创建 `spark/` 目录结构：`config/` `model/` `src/` `README.md`
  - [x] SubTask 8.2: chore：删除 `data/demo/scripts/`；合并 `data/demo/src/` + `data/demo/train/` → `spark/src/`（data_loader / trainer / evaluate / utils）；删除旧实现
  - [x] SubTask 8.3: 实现 `spark/model/config.py`：`CometSparkV05Config`（基于 VerseNex 配置 + 1B 参数预算 + Qwen tokenizer 字段 + `from_pretrained` / `save_pretrained`）
  - [x] SubTask 8.4: 完全重写 `spark/model/model.py`：`CometSparkV05LM` 基于 `VerseNexBlock`（TriSparse + MoD）构建，不重造底层；聚焦层 pattern / 规模 / 初始化；工厂 `CometSparkV05()` / `CometSparkV05Small()`
  - [x] SubTask 8.5: 全面接入新框架：`spark/src/trainer.py` 调用 `VerseTrainer`；tokenizer 用 Qwen3.5-35B-A3B
  - [x] SubTask 8.6: 生成 `spark/config/cometspark_v05.yml`（1B 默认）+ `cometspark_v05_small.yml`（调试）；删除 `config.yml` 的 hybrid 模式
  - [x] SubTask 8.7: VerseNex 优化：embedding scale + init scale + 输出 projection tie + temperature scaling（解决胡乱输出）；CPU 利用率优化（BLAS 线程 + numba + 多线程数据加载）
  - [x] SubTask 8.8: 支持后训练 / 增强训练：接入 VerseTrainer CLI（`verse-posttrain --config spark/config/cometspark_v05.yml`）
  - [x] SubTask 8.9: 删除 `data/demo/` 整个目录（训练能力已迁入 VerseTrainer，模型能力已迁入 spark/）
  - [x] SubTask 8.10: 新增 `tests/test_cometspark_v05.py`：模型构建（≈1B 参数）/ Qwen tokenizer 加载 / 训练 CLI 端到端 / 生成连贯性 / 打分达标

## 阶段 C：文档与代码注释

- [x] Task 9: 文档与代码注释全面更新
  - [x] SubTask 9.1: 更新根 `README.md`：新增 VerseInfra / VerseTrainer / NexRL / CometSpark V0.5-1B / GPU-NPU 支持 章节；删除 data/demo 相关说明
  - [x] SubTask 9.2: 更新 `packages/verse_infra/README.md`：总包结构 + 子模块说明 + 导入路径迁移指南
  - [x] SubTask 9.3: 更新 `packages/verse_nex/README.md`：VerseNexLM 重命名说明 + MoD / 超稀疏并行注意力 / NexRL 说明
  - [x] SubTask 9.4: 更新 `packages/verse_torch/README.md`：DeviceBackend / GPU-NPU 后端 / 新组件说明
  - [x] SubTask 9.5: 新增 `spark/README.md`：CometSpark V0.5-1B 模型说明 + 配置 + 训练/推理 CLI
  - [x] SubTask 9.6: 补齐关键模块 docstring：`device.py` / `backend_torch.py` / `nexrl/` / `speculative.py` / `verse_trainer/cli.py` / `spark/model/`
  - [x] SubTask 9.7: 更新 `docs/architecture/`：新增 ADR 记录（GPU-NPU 后端抽象、VerseInfra 聚合、NexRL 设计、超稀疏并行注意力）；更新现有 ADR
  - [x] SubTask 9.8: 更新 `docs/training_guide.md` + `docs/performance_tuning.md`：GPU/NPU 训练 + 并行训练 + NexRL 后训练 指南

## 阶段 D：综合验收

- [x] Task 10: 全项目 check-loop + 测试通过
  - [x] SubTask 10.1: 跑全量 `pytest tests/` 确保零失败（含新测试 + 旧测试）
  - [x] SubTask 10.2: 验证 `from verse_infra import ...` / `from verse_nex import VerseNexLM` / `from verse_torch import ...` 全部导入成功
  - [x] SubTask 10.3: 验证 `verse-train --config spark/config/cometspark_v05.yml --device cpu` 端到端跑通（无 GPU 环境回退 CPU）
  - [x] SubTask 10.4: 验证旧路径 shim 发出 DeprecationWarning 但仍可工作
  - [x] SubTask 10.5: 更新 `audit_report.md` 记录 Part4K1 变更与修复

# Task Dependencies

- Task 1（VerseTorch GPU/NPU）是基础，Task 2/3/4/6/8 均依赖其 `Tensor.device` 与 `Module.to(device)`。
- Task 2（VerseNex 重命名 + MoD）独立于 Task 3/4，可并行；但 Task 3/4 依赖 VerseNex 命名稳定。
- Task 3（超稀疏并行注意力）依赖 Task 1（GPU 后端）+ Task 2（VerseNex 命名）。
- Task 4（NexRL）依赖 Task 1（GPU 并行）+ Task 2（VerseNexLM 策略网络）；Task 5（VerseTokenizer）的 NexRL 集成依赖 Task 4。
- Task 5（VerseTokenizer）的 tokenizer 主体优化可独立先行，NexRL 集成部分依赖 Task 4。
- Task 6（VerseTrainer）依赖 Task 1-5（训练栈需调用 VerseTorch/VerseNex/VerseTokenizer/NexRL）。
- Task 7（VerseInfra 聚合）依赖 Task 6（VerseTrainer 完成后才能物理聚合）；可与其余并行做路径准备。
- Task 8（CometSpark V0.5-1B）依赖 Task 1-7（全面接入新框架）；最后执行。
- Task 9（文档）依赖 Task 1-8 完成；可与 Task 10 验收并行收尾。
- Task 10（综合验收）依赖 Task 1-9 全部完成。
