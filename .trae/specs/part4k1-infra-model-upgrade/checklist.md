# Checklist — Part4K1：基础设施全面升级 + 模型能力升级 + 优化

每个 checkpoint 验证完毕后请勾选。失败的 checkpoint 需在 `tasks.md` 新增修复任务并重新验证。

## Task 1: VerseTorch GPU/NPU 设备抽象层 + 缺失组件补齐

- [x] `device.py` 实现 `DeviceBackend` 抽象基类 + `NumpyBackend`（CPU 默认）+ `TorchBackend`（PyTorch 委托）
- [x] `TorchBackend` 支持 `cuda` / `mps` / `npu`（via `torch_npu`）设备
- [x] `Tensor` 新增 `device` 属性 + `.to(device)` / `.cuda()` / `.npu()` / `.cpu()` 方法
- [x] GPU 下 matmul / linear / attention / softmax 等算子委托 PyTorch 执行
- [x] GPU 下 autograd 走 PyTorch autograd，CPU 下保持自研 autograd
- [x] 无 PyTorch 环境下 `Tensor.cuda()` 抛 `RuntimeError("未安装 PyTorch，无法使用 GPU")`
- [x] 无 PyTorch 环境下所有现有测试不变通过（向后兼容）
- [x] `nn.py` 新增 `RotaryEmbedding` / `KVCache` / `StaticCache` / `DynamicCache` / `GroupNorm` / `Conv1d` / `LayerNorm` 优化版
- [x] `Module` 新增 `.to(device)` / `.device` 属性
- [x] `optim.py` 新增 `NAdamW` / `RMSProp`
- [x] `losses.py` 新增 `contrastive_loss` / `perplexity`
- [x] `training.py` 新增 `DistributedTrainer` 占位接口 + 混合精度 `autocast` 支持（GPU 后端）
- [x] `verse_torch/__init__.py` 导出新 API；`pyproject.toml` 把 `torch` 声明为可选依赖
- [x] `tests/test_device_backend.py` 覆盖 NumpyBackend / TorchBackend / device 迁移 / 无 PyTorch 回退 / autocast
- [ ] GPU 后端混合精度训练 loss 与 fp32 一致到 1e-2，显存占用降低（需 GPU 环境验证）

## Task 2: VerseNex 品牌落地 + 类重命名 + MoD 完善验证

- [x] `verse_nex` 内 `TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention` 重命名完成
- [x] `VerseNexBlock` 统一为唯一名（无重复定义）
- [x] `verse_torch.nn` 保留旧名作为 `DeprecationWarning` 别名（一个版本）
- [x] 旧别名导入时发出 `DeprecationWarning: TransformerLM 已更名为 VerseNexLM`
- [x] `MoDLayer` 完整实现：5 DensePart × 8 Expert × top-3 双层门控 + aux loss
- [x] `MoDLayer` 包含 `load_balance_loss` + `router z-loss`
- [x] `tests/test_mod_complete.py` 覆盖 MoD 前向 / 反向 / aux loss 收敛 / parallel-vs-recurrent 一致性
- [x] `config.yml` 的 `arch` 字段仅保留 `versenex` 唯一值
- [x] `arch: transformer` / `arch: hybrid` 自动映射到 `versenex` + DeprecationWarning
- [x] `HybridBlock` / `HybridLM` 标记 deprecated（保留只读兼容）

## Task 3: 超稀疏并行注意力机制优化

- [x] `tri_sparse_attn.py` 实现多 query chunk 并行计算 attention（批量矩阵化 / GPU 并行）
- [x] 消除串行循环，attention 计算并行化
- [x] 新增 `verse_nex/speculative.py`：draft head 并行生成 k 个候选 token
- [x] 实现主模型一次前向验证 + verify-then-commit（接受最长正确前缀）
- [x] 实现 KV cache 并行批量更新（`KVCache.batch_update`）
- [x] `tests/test_parallel_sparse_attn.py` 验证多 chunk 并行 vs 串行数值一致（float32 吻合 1e-3）
- [ ] 长序列（seq_len ≥ 512）下并行实现吞吐 ≥ 2×（GPU 后端）（需 GPU 环境验证）
- [x] `tests/test_speculative_decode.py` 覆盖 k=4 候选预测 + 接受最长正确前缀 + 拒绝处重 draft

## Task 4: NexRL 优化强化学习算法

- [x] `verse_nex/nexrl/__init__.py` 定义 RL 五要素抽象：`NexAgent` / `NexEnv` / `NexState` / `NexAction` / `NexReward`
- [x] `NexAgent` 封装 VerseNexLM 策略网络 + 参考网络 + KL 约束
- [x] `NexEnv` 提供 observation + reward
- [x] `NexState` 包含 prompt + 已生成 token + KV cache（支持并行多 state）
- [x] `NexAction` 支持 token 级动作 + 动作空间采样
- [x] `NexReward` 多维加权（correctness + fluency + safety + length_penalty）+ reward normalization（running mean/std）+ reward shaping
- [x] `NexAction` 实现 ε-greedy / softmax / nucleus 采样 + 探索衰减 schedule + 重复动作惩罚
- [x] `ParallelRolloutCollector` 实现多 prompt / 多 rollout 并行采样（batched）
- [x] `NexTrainer` 实现 PPO 风格（clip ratio + GAE + KL 祖父项）+ 纯策略梯度 fallback
- [x] KL 散度超阈值时自动增加 KL 惩罚权重
- [x] `verse-posttrain --rl nexrl` CLI 集成可用
- [x] `tests/test_nexrl.py` 覆盖多维奖励 / 并行 rollout / KL 防崩溃 / 动作采样策略

## Task 5: VerseTokenizer 优化 + NexRL 集成

- [x] `bpe.py` BPE 训练支持 `min_frequency` / `max_token_length` / 并行 merge（多线程）
- [ ] 并行 merge 训练耗时 < 串行的 40%
- [x] 新增 `wordpiece.py`（WordPiece tokenizer）
- [x] `unigram.py` 对齐 sentencepiece
- [x] 编码 / 解码向量化（批量 encode/decode 加速）
- [x] `add_bos` / `add_eos` 独立开关 + `truncation` / `padding` 策略对齐 HF `BatchEncoding`
- [x] 新增 `nex_wrapper.py` 的 `NexTokenizerWrapper`：token 边界注入 RL 信号
- [x] reward-weighted token preference：高频高奖励子串优先成 token
- [x] `BPETokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")` 从 HuggingFace 加载 tokenizer.json（vocab 248320）
- [x] byte-aligned decode 保留并强化，无 U+FFFD 乱码
- [x] `tests/test_tokenizer_optimization.py` 覆盖并行 BPE / NexRL 集成 / Qwen tokenizer / wordpiece / BatchEncoding

## Task 6: VerseTrainer 独立训练包

- [x] `packages/verse_infra/verse_trainer/` 包创建（pyproject.toml + __init__.py）
- [x] `data/demo/train/` 训练代码迁入 `verse_trainer/`（trainer / evaluate / visualize）
- [x] `ParallelTrainer` 升级：`_safe_chunk_run` 包裹每个 chunk + 子进程异常捕获 + 信号处理 + OOM 兜底 + 断点续训
- [x] 修复“莫名终止退出”：异常捕获 + 信号处理 + OOM 兜底
- [x] `CachedDataset` 实现：首次扫描缓存 `.npz` + 流式 lazy load，解决加载耗时
- [x] CLI 入口 `verse-train` / `verse-finetune` / `verse-posttrain` / `verse-eval` / `verse-tokenize` 可用
- [x] `verse-train --config ... --device cpu|cuda|npu --single-sample --parallel-chunks N` 参数支持
- [x] `--single-sample` 单条 prompt/completion / 单文件支持
- [x] 复用 `SFTTrainer` / `DPOTrainer` / `LoRATrainer`；新增 `RLTrainer`（NexRL）
- [x] Loss 优化策略：梯度裁剪 + LR warmup + cosine + ReduceLROnPlateau + loss plateau 重走
- [x] `_rollback_and_perturb`：回退 best_state_dict + LR × 0.3 + 重置 Adam 动量
- [x] NaN/Inf 检测 + 跳过该 batch
- [x] `data/demo/` 旧实现 / 重复代码 / 死代码已清除
- [x] `tests/test_verse_trainer.py` 覆盖 CLI 端到端 / 单样本 / CachedDataset 加速 / plateau 重走 / 断点续训

## Task 7: VerseInfra 总包聚合

- [x] `packages/verse_infra/` 目录结构 + `pyproject.toml` + `verse_infra/__init__.py`（重导出公共 API）
- [x] `verse_compat` 源码移动到 `verse_infra/verse_compat/`，原顶层目录删除
- [x] `verse_inference` 源码移动到 `verse_infra/verse_inference/`，原顶层目录删除
- [x] `verse_tokenizer` 源码移动到 `verse_infra/verse_tokenizer/`，原顶层目录删除
- [x] `verse_trainer` 放入 `verse_infra/verse_trainer/`
- [x] 单包 + 子模块结构：`from verse_infra.verse_tokenizer import BPETokenizer` 可用
- [x] 便捷重导出：`from verse_infra import BPETokenizer, VerseTrainer` 可用
- [x] 原 `packages/verse_xxx/` 位置保留 thin shim（`from verse_infra.verse_xxx import *` + DeprecationWarning）
- [x] 旧路径 `from verse_tokenizer import BPETokenizer` 经 shim 转发成功 + DeprecationWarning
- [x] 全项目导入路径更新（tests / examples / data / verse_nex / verse_torch / spark / docs）
- [x] 根 `pyproject.toml` 声明 `verse_infra`、`verse_trainer`；删除旧包声明
- [x] `tests/test_verse_infra_imports.py` 覆盖子模块导入 / 便捷重导出 / 旧路径 shim DeprecationWarning
- [x] `verse_torch` / `verse_nex` 保持独立未并入 VerseInfra

## Task 8: CometSpark V0.5-1B 模型迁移 + 完全重写

- [x] `spark/` 目录结构创建：`config/` `model/` `src/` `README.md`
- [x] `data/demo/scripts/` 删除
- [x] `data/demo/src/` + `data/demo/train/` 合并 → `spark/src/`（data_loader / trainer / evaluate / utils）
- [x] 旧实现彻底删除
- [x] `spark/model/config.py` 实现 `CometSparkV05Config`（VerseNex 配置 + 1B 预算 + Qwen tokenizer + from_pretrained / save_pretrained）
- [x] `spark/model/model.py` 完全重写 `CometSparkV05LM`：基于 `VerseNexBlock`（TriSparse + MoD），不重造底层
- [x] 工厂 `CometSparkV05()` 返回 ≈1B 参数实例
- [x] 工厂 `CometSparkV05Small()` 返回调试小配置
- [x] `spark/src/trainer.py` 调用 VerseTrainer
- [x] tokenizer 用 Qwen3.5-35B-A3B（vocab 248320）
- [x] `spark/config/cometspark_v05.yml`（1B 默认）+ `cometspark_v05_small.yml`（调试）生成
- [x] `config.yml` 的 hybrid 模式删除（arch 仅 versenex）
- [x] VerseNex 优化：embedding scale + init scale + 输出 projection tie + temperature scaling（解决胡乱输出）
- [x] CPU 利用率优化：BLAS 线程 + numba + 多线程数据加载
- [x] 后训练 / 增强训练接入：`verse-posttrain --config spark/config/cometspark_v05.yml`
- [x] `data/demo/` 整个目录已删除
- [x] `tests/test_cometspark_v05.py` 覆盖模型构建（≈1B）/ Qwen tokenizer / 训练 CLI / 生成连贯性 / 打分达标
- [x] 1B 模型 loss 收敛 + 生成连贯 + 打分达标

## Task 9: 文档与代码注释全面更新

- [x] 根 `README.md` 新增 VerseInfra / VerseTrainer / NexRL / CometSpark V0.5-1B / GPU-NPU 章节删除 data/demo 说明
- [x] `packages/verse_infra/README.md` 总包结构 + 子模块说明 + 导入路径迁移指南
- [x] `packages/verse_nex/README.md` VerseNexLM 重命名说明 + MoD / 超稀疏并行注意力 / NexRL 说明
- [x] `packages/verse_torch/README.md` DeviceBackend / GPU-NPU 后端 / 新组件说明
- [x] `spark/README.md` CometSpark V0.5-1B 模型说明 + 配置 + 训练/推理 CLI
- [x] 关键模块 docstring 补齐：`device.py` / `backend_torch.py` / `nexrl/` / `speculative.py` / `verse_trainer/cli.py` / `spark/model/`
- [x] `docs/architecture/` 新增 ADR：GPU-NPU 后端抽象 / VerseInfra 聚合 / NexRL 设计 / 超稀疏并行注意力
- [x] 现有 ADR 更新到最新状态
- [x] `docs/training_guide.md` + `docs/performance_tuning.md` 更新 GPU/NPU 训练 + 并行训练 + NexRL 后训练指南

## 综合验收

- [x] `pytest tests/` 全量测试零失败（含新测试 + 旧测试）
- [x] `from verse_infra import ...` / `from verse_nex import VerseNexLM` / `from verse_torch import ...` 全部导入成功
- [x] `verse-train --config spark/config/cometspark_v05.yml --device cpu` 端到端跑通（无 GPU 回退 CPU）
- [x] `verse-finetune` / `verse-posttrain` / `verse-eval` / `verse-tokenize` CLI 可用
- [x] 旧路径 shim 发出 DeprecationWarning 但仍可工作
- [ ] GPU 环境下 `--device cuda` 训练加速（相对 CPU ≥ 3×，若环境有 GPU）
- [x] `audit_report.md` 更新记录 Part4K1 变更与修复
- [x] 无 `data/demo/` 残留文件
- [x] 无 `hybrid` 模式残留代码路径（除 deprecated 只读兼容）
- [x] 文档与实现一致，无过时说明
