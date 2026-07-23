# Tasks — Part5K1：VMPC 技术全量化 + 双模型并行 + 架构精简

按依赖顺序排列，独立任务可并行。每个任务完成后请勾选对应 checkbox。

## 阶段 A：VerseTorch 底层精简（vnn 重命名 + 去壳）

- [x] Task 1: VerseTorch.nn → VerseTorch.vnn 重命名（BREAKING）
  - [x] SubTask 1.1: 新建 `packages/verse_torch/verse_torch/vnn.py`，把 `nn.py` 全部内容迁移过去（Module / Linear / Embedding / LayerNorm / RMSNorm / Dropout / Sequential / ModuleList / SwiGLUMLP / 初始化函数 / Conv1d / GroupNorm / KVCache 等）
  - [x] SubTask 1.2: `nn.py` 降级为 thin shim：`from .vnn import *` + 对 `TransformerLM`/`TransformerBlock`/`GQASelfAttention` 旧名抛 `ImportError`（不再 silent 兼容，落实 REMOVED Requirement）
  - [x] SubTask 1.3: 更新 `verse_torch/__init__.py`：新增 `from . import vnn` 导出，`nn` 仍指向 `vnn`（`nn = vnn` 别名），但推荐 `vnn`
  - [x] SubTask 1.4: 更新 `verse_torch` 内部模块导入：`compress.py` / `training.py` / `training_nex.py` / `layerwise_trainer.py` 中 `from . import nn` 改为 `from . import vnn as nn`（最小改动，保证不破坏现有逻辑）
  - [x] SubTask 1.5: 更新 `verse_nex` / `spark` / `tests` 中所有 `from verse_torch.nn import ...` → `from verse_torch.vnn import ...`（或 `from verse_torch import vnn; vnn.XXX`）
  - [x] SubTask 1.6: 新增 `tests/test_vnn_rename.py`：新导入路径可用 + 旧路径发 DeprecationWarning + transformer 系旧名抛 ImportError

- [x] Task 2: VerseTorch 底层去壳与合并
  - [x] SubTask 2.1: 合并 `compress.py` 与 `quantize.py` 中重复的 bit 统计 / 参数遍历函数（`_iter_all_tensors` / `_iter_module_tensors` / `count_parameters` 等），统一到 `compress.py` 的单一实现，`quantize.py` 复用
  - [x] SubTask 2.2: 合并 `training.py` 与 `training_nex.py` 中重复的 collate / loss 辅助（`_as_tensor` / `_scalar` / `_cfg_get` 等），消除反复 import
  - [x] SubTask 2.3: 简化 `Module.__setattr__` 注册分支（减少 isinstance 判断链）
  - [x] SubTask 2.4: 合并 `LayerNorm` / `LayerNormFast` / `RMSNorm` 的公共归一化内核（提取 `_normalize_kernel`）
  - [x] SubTask 2.5: 运行全量测试，确认去壳后行为不变（零回归）

## 阶段 B：VMPC V1.5 技术

- [x] Task 3: VMPC 命名 + V1.5 门面
  - [x] SubTask 3.1: 新建 `packages/verse_torch/verse_torch/vmpc.py`：re-export `compress_pipeline` / `OutlierSafePruner` / `LoRALinear` / `KnowledgeDistiller` / `QLinear` / `compress_mod_experts` / `compression_report`（从 `compress` 导入同一对象，保证 `verse_torch.vmpc.compress_pipeline is verse_torch.compress.compress_pipeline`）
  - [x] SubTask 3.2: 实现 `VMPCRegularizer`：参数幅度 L2 正则 + 压缩感知 dropout + early-exit 自适应稀疏收紧（val_loss 连续 `patience` 步不降 → `target_sparsity *= 0.9`）；提供 `attach(trainer)` 挂载到 Trainer loss
  - [x] SubTask 3.3: 实现 `vmpc_compress(model, profile="small"|"mate")` 便捷函数：按预设一键压缩（small=ternary+高稀疏 sparsity=0.5，mate=int4+中稀疏 sparsity=0.3+蒸馏）
  - [x] SubTask 3.4: 更新 `verse_torch/__init__.py` 导出 `vmpc` / `VMPCRegularizer` / `vmpc_compress`
  - [x] SubTask 3.5: 新增 `tests/test_vmpc_facade.py`：门面导入同一性 + VMPCRegularizer 收紧 + vmpc_compress 预设

- [x] Task 4: VMPC V1.5 命中与质量优化（升级 compress.py）
  - [x] SubTask 4.1: `compress.py` 新增 `_compress_pipeline_v15(model, config, return_stats)`：在 V1.3 流程基础上增加 `contrastive_distill` + `logit_calibration` 步骤；`compress_pipeline` 默认 `version="1.5"`，V1.3 走 `_compress_pipeline_v13`
  - [x] SubTask 4.2: `KnowledgeDistiller` 新增 `contrastive_loss`（margin ranking loss，teacher/student logit top-k 排序一致性）；`compute_loss` 增加 `distill_contrastive` 参数
  - [x] SubTask 4.3: `QLinear.forward` 升级 outlier-aware 反量化：识别 outlier channel（|w| > 3σ）保留 fp16，其余 int4；减少反量化中间拷贝
  - [x] SubTask 4.4: 推理 logits 校准：`CometSparkNexLM.generate` 路径下对反量化 logits 做 temperature-aware sharpening（`logits / sqrt(var) * calib_factor`）
  - [x] SubTask 4.5: `compression_report` 新增 `vmpc_version` 字段
  - [x] SubTask 4.6: 数值稳定：V1.5 路径与 V1.3 前向输出吻合到 1e-2（float32）
  - [x] SubTask 4.7: 新增 `tests/test_vmpc_v15.py`：contrastive distill 排序一致性 + outlier 反量化 + 命中率提升（≥5%）+ 数值稳定

## 阶段 C：数据加载与训练资源优化

- [x] Task 5: JSONL 自修复与标准化（独立文件）
  - [x] SubTask 5.1: 新建 `packages/verse_infra/verse_infra/verse_trainer/jsonl_repair.py`：定义 `JSONLRepairError` 异常 + 异名字段映射表（`instruction/response`、`q/a`、`question/answer`、`input/output`、`user/assistant`）
  - [x] SubTask 5.2: 实现 `_standardize_fields(item)`：自动探测异名字段并批量转换为 `{"prompt":..., "completion":...}`；单字段 `text`/`content`/`raw` 保留为 `text`
  - [x] SubTask 5.3: 实现 `_repair_line(line)`：补全缺失引号/逗号、修复未闭合 JSON、去除 BOM/控制字符/行尾多余逗号；无法修复返回 None
  - [x] SubTask 5.4: 实现 `repair_jsonl(path, write_back=False, repair=True)`：逐行修复 + 标准化，返回 `List[dict]`；`write_back=True` 写到 `*.repaired.jsonl` 或覆盖
  - [x] SubTask 5.5: 修改 `data.py` 的 `load_jsonl`：默认 `repair=True` 调用 `repair_jsonl`；`repair=False` 走原严格解析（向后兼容）
  - [x] SubTask 5.6: 新增 `tests/test_jsonl_repair.py`：异名字段标准化 + 缺逗号修复 + 未闭合修复 + BOM 去除 + 无法修复抛错

- [x] Task 6: val.json 自动生成 + 数据预加载流水线
  - [x] SubTask 6.1: 在 `data.py` 实现 `ensure_val_split(train_path, val_path, val_ratio=0.05, write_back=True)`：val_path 不存在/空时从 train 末尾切分；写回 val_path + 日志 `(n_train, n_val)`
  - [x] SubTask 6.2: `CachedDataset` 升级预加载流水线：`preload=True` 时后台线程编码 + 主线程构建模型；`prefetch` 扩展到非 torch 环境（纯 threading 预取，移除 `self._torch is None` 的降级）
  - [x] SubTask 6.3: `trainer.py` 训练入口 `train()` 启动时调用 `ensure_val_split`（读取 config 的 data 段）
  - [x] SubTask 6.4: 新增 `tests/test_val_autogen.py`：val 不存在自动生成 + 比例正确 + write_back

- [x] Task 7: 64+ 层训练加速（VerseNex 层融合 + chunked 前向）
  - [x] SubTask 7.1: `verse_nex/cometspark.py` 的 `CometSparkNexLM.forward` 新增层融合：相邻同型 VerseNexBlock 的 `RMSNorm + Linear` 合并为单次 matmul（CPU 下减少 Python 循环）
  - [x] SubTask 7.2: 实现 `chunked_forward(idx, chunk_size=8)`：`n_layer >= 64` 时自动启用，按 8 层分块前向，块间不保留中间梯度图（梯度检查点风格）
  - [x] SubTask 7.3: `forward` 自动检测 `n_layer >= 64` 走 chunked 路径，否则走原路径
  - [x] SubTask 7.4: 数值一致：chunked 前向与原前向 float32 吻合到 1e-3
  - [x] SubTask 7.5: 新增 `tests/test_layer_fusion.py`：层融合数值一致 + chunked 前向内存峰值降低 + 吞吐 ≥ 1.5×

- [x] Task 8: VMT 完整智能分区训练（VMTTrainer）
  - [x] SubTask 8.1: `verse_torch/layerwise_trainer.py` 新增 `VMTTrainer`（继承 `LayerWiseTrainer`）：新增 `freeze` 档（INT4 量化 + requires_grad=False）+ `optimize` 档（层融合 + 梯度累积）
  - [x] SubTask 8.2: 实现 `vmt_strategy` 解析：支持 `"layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload"` 语法 + `"auto"` 预设（按层重要性自动分配）
  - [x] SubTask 8.3: freeze 档实现：调用 `quantize_only(dtype="int4")` + `requires_grad=False`；评估/合并时反量化（误差 ≤ 1e-3）
  - [x] SubTask 8.4: optimize 档实现：保持 fp32 + 启用层融合 + 梯度累积
  - [x] SubTask 8.5: 保留 `LayerWiseTrainer` 作为 `VMTTrainer` 别名（向后兼容）
  - [x] SubTask 8.6: `verse_torch/__init__.py` 导出 `VMTTrainer`
  - [x] SubTask 8.7: 新增 `tests/test_vmt_trainer.py`：三档分配 + freeze 反量化误差 + auto 策略 + 统一实体

## 阶段 D：双模型并行（spark 重构）

- [x] Task 9: spark 目录重构（small / mate 双模型）
  - [x] SubTask 9.1: 创建 `spark/small/config/cometspark_small.yml`（从 `cometspark_v05_small.yml` 迁移 + VMPC-small 预设：ternary + 高稀疏 + 0.06zB 目标）
  - [x] SubTask 9.2: 创建 `spark/mate/config/cometspark_mate.yml`（从 `cometspark_v05.yml` 迁移 + VMPC-mate 预设：int4 + 中稀疏 + 蒸馏 + 0.2zB 目标）
  - [x] SubTask 9.3: 创建 `spark/small/model/config.py`（`CometSparkSmallConfig`，从 `CometSparkV05Config` 派生 + VMPC-small 字段）+ `spark/small/model/model.py`（`CometSparkSmallLM` + `CometSparkSmall()` 工厂）
  - [x] SubTask 9.4: 创建 `spark/mate/model/config.py`（`CometSparkMateConfig`）+ `spark/mate/model/model.py`（`CometSparkMateLM` + `CometSparkMate()` 工厂）
  - [x] SubTask 9.5: 双模型均基于 `verse_nex.CometSparkNexLM`，VMPC 适配微调架构（expert 数 / 层 pattern / init_std）
  - [~] SubTask 9.6: **保守策略**——保留扁平旧目录 `spark/config/`、`spark/model/`、`spark/src/`，只新建 `small/` 和 `mate/`。原因：`spark/run.py` 仍引用旧路径（`spark/config/cometspark_v05.yml` + `from spark.model.model import CometSparkV05LM`），删除会立即破坏 run.py。**旧目录待 Task 11 清理**（Task 11 升级 run.py 后再删除旧 config/model/src）。现有测试 `test_cometspark_v05.py` / `test_vmpc_facade.py` / `test_vnn_rename.py` 全部零回归验证通过。
  - [x] SubTask 9.7: 更新 `spark/__init__.py` / `spark/_bootstrap.py` 适配新结构
  - [x] SubTask 9.8: 新增 `tests/test_dual_model.py`：构建 small/mate + VMPC 预设 + 参数量在预期区间（32 个测试全部通过）

- [x] Task 10: checkpoint 重命名 + .vn 默认输出
  - [x] SubTask 10.1: `CometSparkSmallLM` / `CometSparkMateLM` 的 `save` 增加 `format="pt"|"vn"` 参数，默认 `"vn"`；`save_pretrained` 同步
  - [x] SubTask 10.2: 实现 checkpoint 目录自动迁移：检测旧 `checkpoints_XXX/` → 重命名为 `mf_XXX/` + 警告（在 run.py / trainer 启动时）
  - [x] SubTask 10.3: 配置文件 `checkpoint.save_dir` 默认 `mf_small` / `mf_mate`
  - [x] SubTask 10.4: 新增 `tests/test_checkpoint_migrate.py`：自动迁移 + 默认 vn 输出

- [x] Task 11: spark/run.py 训练模式补齐
  - [x] SubTask 11.1: `train` 子命令新增 `--model small|mate`（默认 small），按级别选择 config + 模型 + checkpoint 目录
  - [x] SubTask 11.2: 新增 `finetune` 子命令：委托 `verse_trainer` 的 `LoRATrainer`，支持 `--lora-r` / `--lora-alpha` / `--target-modules` / `--checkpoint`
  - [x] SubTask 11.3: 新增 `posttrain` 子命令：委托 `SFTTrainer` / `DPOTrainer` / `RLTrainer`，`--mode sft|dpo|rl`
  - [x] SubTask 11.4: 新增 `continue` 子命令：委托 `verse_trainer.continue_train`，`--checkpoint` / `--additional-steps`
  - [x] SubTask 11.5: `eval` / `generate` / `chat` / `compress` / `convert` 子命令同步 `--model` 参数
  - [x] SubTask 11.6: 所有训练逻辑委托 `verse_trainer`，run.py 只做参数解析 + 模型选择 + checkpoint 路径映射（不重复造轮子）
  - [x] SubTask 11.7: 新增 `tests/test_spark_run_dual.py`：`--model small/mate` 选择 + finetune/posttrain/continue dry-run

## 阶段 E：VerseNex 精简与文档

- [x] Task 12: VerseNex 精简（删除废弃别名）
  - [x] SubTask 12.1: `verse_nex/__init__.py` 移除已废弃 transformer 系别名导出（`HybridBlock`/`HybridLM` 标记 deprecated 后保留只读，但不新增导出）
  - [x] SubTask 12.2: 清理 `verse_nex` 内部对 `verse_torch.nn` 的旧导入（改用 `verse_torch.vnn`）
  - [x] SubTask 12.3: 运行 `tests/test_cometspark_*.py` / `test_verse_infra_imports.py` 确认零回归

- [x] Task 13: 文档与代码注释升级
  - [x] SubTask 13.1: 新增 `docs/architecture/adr-013-vmpc-naming-v15.md`（VMPC 命名 + V1.5 设计）
  - [x] SubTask 13.2: 新增 `docs/architecture/adr-014-dual-model-small-mate.md`（双模型并行）
  - [x] SubTask 13.3: 新增 `docs/architecture/adr-015-vmt-full-strategy.md`（VMT 完整三档策略）
  - [x] SubTask 13.4: 新增 `docs/architecture/adr-016-nn-to-vnn-rename.md`（nn → vnn 重命名）
  - [x] SubTask 13.5: 更新 `README.md`：VMPC V1.5 + 双模型 + VMT + vnn + jsonl_repair
  - [x] SubTask 13.6: 更新 `docs/training_guide.md` / `docs/performance_tuning.md`：双模型训练 + VMT 策略 + 64+ 层加速
  - [x] SubTask 13.7: `compress.py` / `layerwise_trainer.py` / `vmpc.py` 代码注释统一到 VMPC 术语（"V1.3" → "VMPC V1.5"）

## 阶段 F：验收

- [x] Task 14: 全量测试 + 综合验收
  - [x] SubTask 14.1: `pytest tests/` 全量零失败（含新增测试）
  - [x] SubTask 14.2: 关键导入验证：`import verse_torch` / `from verse_torch.vmpc import ...` / `from verse_torch.vnn import ...` / `from spark.small.model import CometSparkSmallLM` / `from spark.mate.model import CometSparkMateLM`
  - [x] SubTask 14.3: CLI 端到端：`spark/run.py train --model small --dry-run` / `finetune` / `posttrain` / `continue` dry-run 通过
  - [x] SubTask 14.4: VMPC V1.5 端到端：small 模型压缩 + 生成 + 命中率验证
  - [x] SubTask 14.5: jsonl_repair 端到端：异名字段 + 格式修复
  - [x] SubTask 14.6: VMT 端到端：64+ 层模型三档训练
  - [x] SubTask 14.7: 更新 `audit_report.md`

# Task Dependencies

- Task 1（vnn 重命名）→ Task 2（去壳合并）依赖 Task 1（vnn 就位后才能改内部 import）
- Task 3（VMPC 门面）依赖 Task 1（vnn 重命名后 compress.py 内部 import 已更新）
- Task 4（VMPC V1.5）依赖 Task 3（门面就位后扩展 V1.5）
- Task 5（jsonl_repair）独立，可与 Task 1/2/3 并行
- Task 6（val 自动生成 + 预加载）依赖 Task 5（load_jsonl 已升级）
- Task 7（层融合 + chunked）独立，可与 Task 1~6 并行
- Task 8（VMTTrainer）依赖 Task 4（VMPC 量化能力）+ Task 7（层融合）
- Task 9（双模型目录）依赖 Task 1（vnn）+ Task 3（VMPC 门面）
- Task 10（checkpoint + .vn）依赖 Task 9（双模型就位）
- Task 11（run.py 升级）依赖 Task 9 + Task 10
- Task 12（VerseNex 精简）依赖 Task 1（vnn）
- Task 13（文档）依赖所有功能任务
- Task 14（验收）依赖所有

# 并行策略

- 第一批（独立）：Task 1 + Task 5 + Task 7 + Task 12（VerseNex 精简可与 Task 1 并行，仅清理导入）
- 第二批：Task 2 + Task 3 + Task 6（Task 6 依赖 Task 5，第一批完成后启动）
- 第三批：Task 4 + Task 8 + Task 9（Task 4 依赖 Task 3，Task 8 依赖 Task 4+7，Task 9 依赖 Task 1+3）
- 第四批：Task 10 + Task 11
- 第五批：Task 13
- 第六批：Task 14
