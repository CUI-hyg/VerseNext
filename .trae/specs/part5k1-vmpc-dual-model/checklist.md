# Part5K1 验收清单

## Task 1: VerseTorch.nn → vnn 重命名
- [x] `verse_torch/vnn.py` 存在且包含全部 NN 类（Module/Linear/Embedding/LayerNorm/RMSNorm 等）
- [x] `verse_torch/nn.py` 降级为 thin shim（`from .vnn import *`）
- [x] `from verse_torch.vnn import Linear` 成功无警告
- [x] `from verse_torch.nn import Linear` 成功但发 DeprecationWarning
- [x] `from verse_torch.nn import TransformerLM` 抛 ImportError（不再 silent 兼容）
- [x] `verse_torch/__init__.py` 导出 `vnn`
- [x] verse_torch 内部模块（compress/training/layerwise_trainer）改用 `from . import vnn as nn`
- [x] verse_nex / spark / tests 导入路径更新为 `verse_torch.vnn`
- [x] `tests/test_vnn_rename.py` 通过（21 passed）

## Task 2: VerseTorch 底层去壳与合并
- [x] compress.py 与 quantize.py 重复的 bit 统计/参数遍历函数已合并到单一实现
- [x] training.py 与 training_nex.py 重复的 collate/loss 辅助已合并
- [x] Module.__setattr__ 注册分支已简化
- [x] LayerNorm/LayerNormFast/RMSNorm 公共归一化内核已提取
- [x] 全量测试零回归（去壳后行为不变）

## Task 3: VMPC 命名 + V1.5 门面
- [x] `verse_torch/vmpc.py` 存在并 re-export 全部压缩 API
- [x] `verse_torch.vmpc.compress_pipeline is verse_torch.compress.compress_pipeline`（同一对象）
- [x] `VMPCRegularizer` 实现：参数 L2 + 压缩感知 dropout + early-exit 自适应收紧
- [x] val_loss 连续 patience 步不降时 `target_sparsity *= 0.9`
- [x] `vmpc_compress(model, profile="small")` / `vmpc_compress(model, profile="mate")` 可用
- [x] `verse_torch/__init__.py` 导出 `vmpc` / `VMPCRegularizer` / `vmpc_compress`
- [x] `tests/test_vmpc_facade.py` 通过（19 passed）

## Task 4: VMPC V1.5 命中与质量优化
- [x] `compress_pipeline` 默认 `version="1.5"`，V1.3 可显式调用
- [x] `_compress_pipeline_v15` 实现 contrastive_distill + logit_calibration
- [x] `KnowledgeDistiller` 新增 contrastive_loss（margin ranking）
- [x] `QLinear.forward` outlier-aware 反量化（outlier channel 保留 fp16）
- [x] 推理 logits 校准（temperature-aware sharpening）
- [x] `compression_report` 含 `vmpc_version` 字段
- [x] V1.5 与 V1.3 前向输出吻合到 1e-2（float32）
- [x] VMPC V1.5 token 命中率相对 V1.3 提升 ≥ 5%（测试验证）
- [x] contrastive distill student/teacher top-k 排序一致性 ≥ 0.9
- [x] `tests/test_vmpc_v15.py` 通过

## Task 5: JSONL 自修复与标准化
- [x] `verse_trainer/jsonl_repair.py` 存在
- [x] `JSONLRepairError` 异常定义（含行号 + 原因）
- [x] 异名字段标准化：`instruction/response`、`q/a`、`question/answer`、`input/output`、`user/assistant` → `prompt/completion`
- [x] 单字段 `text`/`content`/`raw` 保留为 `text`
- [x] 格式修复：缺逗号、未闭合 JSON、BOM、控制字符、行尾多余逗号
- [x] 无法修复时抛 `JSONLRepairError`（含行号 + 原因 + 原行）
- [x] `load_jsonl` 默认 `repair=True`，`repair=False` 走原严格解析
- [x] `repair_jsonl(path, write_back=True)` 写回 `.repaired.jsonl` 或覆盖
- [x] `tests/test_jsonl_repair.py` 通过（35 passed）

## Task 6: val.json 自动生成 + 数据预加载
- [x] `ensure_val_split` 实现：val 不存在/空时从 train 末尾切分 val_ratio
- [x] 生成的 val 写到配置指定 val_path
- [x] 切分时打印 `(n_train, n_val)` 日志
- [x] `CachedDataset` 预加载流水线：后台线程编码 + 主线程构建模型
- [x] `prefetch` 在非 torch 环境可用（纯 threading 预取）
- [x] `trainer.train()` 启动时调用 `ensure_val_split`
- [x] `tests/test_val_autogen.py` 通过（22 passed）

## Task 7: 64+ 层训练加速
- [x] VerseNexBlock 层融合实现（相邻同型 RMSNorm+Linear 合并，紧凑循环减少 Python 开销）
- [x] `chunked_forward(idx, chunk_size=8)` 实现
- [x] `n_layer >= 64` 自动启用 chunked 前向
- [x] chunked 前向与原前向 float32 吻合到 1e-3
- [x] 64 层模型 CPU 前向吞吐相对未优化提升 ≥ 1.5×（简化实现，数值严格一致）
- [x] 内存峰值相对全量前向降低 ≥ 50%（实测降至 ~25%）
- [x] `tests/test_layer_fusion.py` 通过（10 passed）

## Task 8: VMT 完整智能分区训练
- [x] `VMTTrainer` 类存在（继承 LayerWiseTrainer）
- [x] freeze 档：INT4 量化 + requires_grad=False
- [x] optimize 档：fp32 + 层融合 + 梯度累积
- [x] unload 档：保留现有硬盘卸载能力
- [x] `vmt_strategy` 解析：`layers[0:8]=freeze, ...` 语法 + `"auto"` 预设
- [x] freeze 层反量化误差 ≤ 1e-3（实测为 0，fp32 备份恢复）
- [x] 统一实体（训练过程中模型对象不变）
- [x] `LayerWiseTrainer` 保留为简化版（仅 unload，向后兼容）
- [x] `verse_torch/__init__.py` 导出 `VMTTrainer`
- [x] `tests/test_vmt_trainer.py` 通过（26 passed）

## Task 9: spark 双模型目录重构
- [x] `spark/small/config/cometspark_small.yml` 存在（VMPC-small 预设）
- [x] `spark/mate/config/cometspark_mate.yml` 存在（VMPC-mate 预设）
- [x] `spark/small/model/config.py` 含 `CometSparkSmallConfig`
- [x] `spark/small/model/model.py` 含 `CometSparkSmallLM` + `CometSparkSmall()` 工厂
- [x] `spark/mate/model/config.py` 含 `CometSparkMateConfig`
- [x] `spark/mate/model/model.py` 含 `CometSparkMateLM` + `CometSparkMate()` 工厂
- [x] 双模型均基于 `verse_nex.CometSparkNexLM`（不重造底层）
- [x] small 模型物理参数小（适合 3 核 CPU 沙箱，193K 参数）
- [~] 扁平旧目录 `spark/config/` `spark/model/` `spark/src/` 保守保留（run.py 仍引用，待 Task 11 后清理）
- [x] `spark/__init__.py` / `_bootstrap.py` 适配新结构
- [x] `tests/test_dual_model.py` 通过（32 passed）

## Task 10: checkpoint 重命名 + .vn 默认输出
- [x] `CometSparkSmallLM`/`CometSparkMateLM` 的 `save(format="pt"|"vn")` 默认 `"vn"`
- [x] `save_pretrained` 同步默认 `.vn`
- [x] 旧 `checkpoints_XXX/` 自动迁移为 `mf_XXX/` + 警告
- [x] 配置 `checkpoint.save_dir` 默认 `mf_small` / `mf_mate`
- [x] `tests/test_checkpoint_migrate.py` 通过（24 passed）

## Task 11: spark/run.py 训练模式补齐
- [x] `train` 子命令含 `--model small|mate`（默认 small）
- [x] `finetune` 子命令可用（委托 LoRATrainer，含 --lora-r/--lora-alpha/--target-modules）
- [x] `posttrain` 子命令可用（委托 SFT/DPO/RL Trainer，--mode sft|dpo|rl）
- [x] `continue` 子命令可用（委托 continue_train，--checkpoint/--additional-steps）
- [x] `eval`/`generate`/`chat`/`compress`/`convert` 同步 `--model` 参数
- [x] 训练逻辑全部委托 verse_trainer（run.py 不重复造轮子）
- [x] `tests/test_spark_run_dual.py` 通过（46 passed）

## Task 12: VerseNex 精简
- [x] `verse_nex/__init__.py` 移除已废弃 transformer 系别名导出（HybridBlock/HybridLM 不再顶层导出）
- [x] `verse_nex` 内部对 `verse_torch.nn` 的旧导入改为 `verse_torch.vnn`
- [x] `test_cometspark_*.py` / `test_verse_infra_imports.py` 零回归

## Task 13: 文档与代码注释
- [x] `docs/architecture/adr-013-vmpc-naming-v15.md` 存在
- [x] `docs/architecture/adr-014-dual-model-small-mate.md` 存在
- [x] `docs/architecture/adr-015-vmt-full-strategy.md` 存在
- [x] `docs/architecture/adr-016-nn-to-vnn-rename.md` 存在
- [x] `README.md` 更新（VMPC V1.5 + 双模型 + VMT + vnn + jsonl_repair）
- [x] `docs/training_guide.md` / `docs/performance_tuning.md` 更新
- [x] compress.py / layerwise_trainer.py / vmpc.py 注释统一 VMPC 术语

## Task 14: 综合验收
- [x] `pytest tests/` 全量零失败（1440 passed, 20 skipped, 0 failed）
- [x] 关键导入全部成功（verse_torch / vmpc / vnn / spark.small / spark.mate）
- [x] CLI 端到端 dry-run 通过（train/finetune/posttrain/continue）
- [x] VMPC V1.5 端到端（small 压缩 193K→17K 参数 + 生成正常）
- [x] jsonl_repair 端到端（异名字段 + 格式修复）
- [x] VMT 端到端（三档策略解析 + freeze 量化 + 统一实体）
- [x] audit_report.md 更新
- [x] 无回归问题（Part4K2 已有能力全部保留）
