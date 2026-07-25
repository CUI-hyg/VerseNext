# Tasks — Part5K1.3：漏洞修复 + 原生 .vn + Gigatoken + NPU/ROCm 兼容

按依赖顺序排列，独立任务可并行。每个任务完成后请勾选对应 checkbox。

## 阶段 A：并行训练递归修复（fix）

- [x] Task 1: 定位 + 修复 ParallelTrainer Phase 3 merge_finetune 递归崩溃
  - [x] SubTask 1.1: 在 `packages/verse_torch/verse_torch/training.py` 的 `ParallelTrainer.fit` 中定位 chunk_id=-999 路径（`_train_chunk(self.model, self.train_dataset, self.merge_finetune_steps, -999)`），确认 `copy.deepcopy(self.best_state_dict)` 与 `load_state_dict` 调用链
  - [x] SubTask 1.2: 把 `ParallelTrainer.fit` 中所有 `copy.deepcopy(state_dict)` 改为 `pickle.loads(pickle.dumps(state_dict, protocol=4))`（迭代序列化，避免递归对象图遍历）
  - [x] SubTask 1.3: 把 `packages/verse_torch/verse_torch/vnn.py` 的 `Module.state_dict()` / `load_state_dict()` 的子模块遍历从递归 DFS 改为迭代 BFS（显式栈，行为等价：state_dict 键值完全一致）
  - [x] SubTask 1.4: `packages/verse_infra/verse_infra/verse_trainer/trainer.py` 的 `_safe_chunk_run` 新增 `RecursionError` 捕获分支：`gc.collect()` + 临时上调 `sys.setrecursionlimit`（+500）重试一次；仍失败时走"跳过该 chunk"路径（保留现有 graceful degrade）
  - [x] SubTask 1.5: `ParallelTrainerSafe._train_chunk_safe` 增加 chunk 间 `gc.collect()` + `is_shutdown_requested()` 检查，避免残留状态泄漏
  - [x] SubTask 1.6: `Trainer.fit` 主循环捕获 `RecursionError` 时优雅保存当前 `best_state` 后退出（exit code != 0），不丢 checkpoint
  - [x] SubTask 1.7: 新增 `tests/test_parallel_recursion_fix.py`：构造 60+ 层 VerseNex 模型 + chunk_id=-999 路径，断言不触发 RecursionError + val_loss 收敛；现有 `tests/test_parallel_trainer.py` / `test_loss_and_parallel_fix.py` 零回归

- [x] Task 2: CheckpointManager 原子写 + 稳定性增强
  - [x] SubTask 2.1: `packages/verse_torch/verse_torch/training.py` 的 `CheckpointManager.save_best` / `save_last` 改为原子写：先写 `best.pt.tmp` / `last.pt.tmp`（或 `best.vn.tmp`），再 `os.replace` 重命名；写入失败时清理 `.tmp`
  - [x] SubTask 2.2: `CheckpointManager.__init__` 增加 `format: str = "auto"` 参数 + `use_vmpc: bool = False` 参数（默认 False 保持向后兼容）；`format="auto"` 时根据 `use_vmpc` 自动选择
  - [x] SubTask 2.3: 扩展 `save_best` / `save_last` 签名支持 `training_state=None, optimizer_state=None, extra_state=None` 参数（默认 None，保持现有调用向后兼容）
  - [x] SubTask 2.4: 扩展 `load_best` / `load_last` 返回统一 dict（含 `model_state_dict` / `training_state` / `optimizer_state` / `extra_state`，缺失字段为 None）
  - [x] SubTask 2.5: 实现 `use_vmpc=True` 时 `format="pt"` 抛 `ValueError`（与 `CometSparkSmallLM._enforce_vn_format` 一致）
  - [x] SubTask 2.6: 新增 `tests/test_checkpoint_atomic.py`：原子写 + 中断不损坏 + use_vmpc 强制 .vn

## 阶段 B：.vn v2 原生格式 + 复杂 Python 对象

- [x] Task 3: VN_FORMAT_VERSION v1 → v2 升级（向后兼容）
  - [x] SubTask 3.1: `packages/verse_torch/verse_torch/vn_format.py`：`VN_FORMAT_VERSION = 2`；新增 `training_state.json` / `optimizer_state.pkl` / `extra_state.pkl` 常量
  - [x] SubTask 3.2: 抽取 `VNEntry` 抽象（统一管理 ZIP 内条目的写入/读取/校验，减少 `VNFileWriter` / `VNFileReader` 重复的 `writestr` / `open` 模板代码）
  - [x] SubTask 3.3: `VNFileWriter` 新增 `write_training_state(state: dict)`：写 `training_state.json`（JSON-able dict，含 step / epoch / best_val_loss / patience_count / rng_state_hex / config_snapshot_hash）
  - [x] SubTask 3.4: `VNFileWriter` 新增 `write_optimizer_state(state: dict)`：写 `optimizer_state.pkl`（pickle，承载任意 Python 对象：AdamW 的 exp_avg / exp_avg_sq / step、scheduler state 等）
  - [x] SubTask 3.5: `VNFileWriter` 新增 `write_extra_state(state: Any)`：写 `extra_state.pkl`（pickle，用户自定义任意对象）
  - [x] SubTask 3.6: `VNFileWriter.close` 更新 `meta.json`：新增 `vn_format_version: 2` + `has_training_state` / `has_optimizer_state` / `has_extra_state` 布尔字段
  - [x] SubTask 3.7: `VNFileReader` 自动识别 v1/v2（基于 `meta.json` 的 `vn_format_version` 字段）；v1 文件读取新方法返回 None
  - [x] SubTask 3.8: `VNFileReader` 新增 `read_training_state() -> Optional[dict]` / `read_optimizer_state() -> Optional[dict]` / `read_extra_state() -> Optional[Any]`
  - [x] SubTask 3.9: 新增 `tests/test_vn_v2_format.py`：v2 写入 + 读取（training_state / optimizer_state / extra_state）+ v1 向后兼容（read 新方法返回 None）+ optimizer m/v 矩阵数值一致（float32 吻合 1e-7）

- [x] Task 4: CheckpointManager 原生 .vn 支持（集成 v2）
  - [x] SubTask 4.1: `CheckpointManager.save_best` 在 `format="vn"` 时调用 `VNFileWriter` 写 `.vn`：含 model state_dict + training_state + optimizer_state + extra_state
  - [x] SubTask 4.2: `CheckpointManager.load_best` / `load_last` 在 `format="vn"` 时调用 `VNFileReader` 读取（含 v1 向后兼容：缺失字段为 None）
  - [x] SubTask 4.3: 默认路径 `best.vn` / `last.vn`（format="vn"）或 `best.pt` / `last.pt`（format="pt"）；`_resolve_path` 自动加上对应扩展名
  - [x] SubTask 4.4: `Trainer.__init__` 中的 `CheckpointManager` 实例化传入 `format="auto"` + `use_vmpc=cfg.get("use_vmpc", False)`，让训练默认按 use_vmpc flag 走 .vn/.pt
  - [x] SubTask 4.5: 新增 `tests/test_checkpoint_vn.py`：vn checkpoint 含 optimizer state + 训练中断后 load_best 恢复 + use_vmpc 强制 .vn

- [x] Task 5: VerseNex / VerseTorch 原生 .vn 适配
  - [x] SubTask 5.1: `packages/verse_nex/verse_nex/cometspark.py` 的 `CometSparkNexLM.save(path, format="vn")`：新增 `format` 参数（默认 `"vn"`），`format="vn"` 走 `save_vn` 路径，`format="pt"` 走原 pickle 路径
  - [x] SubTask 5.2: `CometSparkNexLM.from_pretrained(path)`：识别 `.vn` / `.pt` / 目录（含 `model.vn` 或 `model.pt`）三种输入
  - [x] SubTask 5.3: `CometSparkNexLM.save_vn` 新增 `training_state` / `optimizer_state` / `extra_state` 参数，调用 `VNFileWriter` v2 API
  - [x] SubTask 5.4: `spark/src/base_model.py` 的 `CometSparkV05LM.save_vn` 同步升级支持 training_state / optimizer_state / extra_state 参数（基类提供，子类继承）
  - [x] SubTask 5.5: `spark/small/model/model.py` + `spark/mate/model/model.py` 的 `save_vn` 覆盖调用基类 + 透传 training_state / optimizer_state / extra_state
  - [x] SubTask 5.6: 现有测试 `test_vn_format.py` / `test_cometspark_v05.py` / `test_dual_model.py` 零回归

- [x] Task 6: ResumeManager 断点续训
  - [x] SubTask 6.1: `packages/verse_torch/verse_torch/training.py` 新增 `ResumeState` namedtuple（`model_state_dict` / `optimizer_state` / `step` / `rng_state` / `best_val_loss` / `epoch` / `patience_count`）
  - [x] SubTask 6.2: 新增 `ResumeManager` 类：`save(path, model, optimizer, step, **kwargs)` / `load(path) -> ResumeState` / `apply(trainer, path)` 三方法
  - [x] SubTask 6.3: `ResumeManager.save` 调用 `CheckpointManager.save_best` 写 `.vn`（含 model + optimizer + step + rng + best_val_loss + epoch + patience）
  - [x] SubTask 6.4: `ResumeManager.load` 调用 `CheckpointManager.load_best` 读取，v1 `.vn` / `.pt` 缺失字段返回 None + 警告日志
  - [x] SubTask 6.5: `ResumeManager.apply(trainer, path)` 把 `ResumeState` 应用到 `Trainer` / `ParallelTrainerSafe` 实例（恢复 model / optimizer / step / rng / best_val_loss）
  - [x] SubTask 6.6: `ParallelTrainerSafe._save_resume_state` / `_load_resume_state` 改用 `ResumeManager` + `.vn`；旧 `.pkl` resume 文件保留为兼容回退（`.vn` 不存在时尝试 `.pkl`）
  - [x] SubTask 6.7: `spark/run.py continue` 子命令委托 `ResumeManager.apply` 完成续训状态恢复
  - [x] SubTask 6.8: 新增 `tests/test_resume_manager.py`：断点续训 + v1 向后兼容 + optimizer state 恢复 + rng_state 恢复

## 阶段 C：Gigatoken 集成（默认 Tokenizer）

- [x] Task 7: GigaTokenizerWrapper 适配器
  - [x] SubTask 7.1: 新增 `packages/verse_infra/verse_infra/verse_tokenizer/giga.py`：定义 `GigaTokenizerWrapper(BaseTokenizer)`
  - [x] SubTask 7.2: lazy import gigatoken：模块 import 不触发加载，仅构造时 import；不可用时抛 `ImportError` 含安装提示（`pip install gigatoken`）
  - [x] SubTask 7.3: 内部用 `gt.Tokenizer(hf_tokenizer).as_hf()` 兼容模式（drop-in replacement），保证与 `VerseTokenizer` 输出一致；可选 `native=True` 走 `gt.Tokenizer(model_id)` 原生 API
  - [x] SubTask 7.4: 实现 `encode` / `decode` / `encode_batch` / `decode_batch` / `save` / `load` / `__len__` / `apply_chat_template` 接口（对齐 `VerseTokenizer`）
  - [x] SubTask 7.5: 缓存 `bos_id` / `eos_id` / `pad_id` / `vocab_size`（构造时一次解析）；`vocab` 属性懒加载
  - [x] SubTask 7.6: `apply_chat_template` 委托底层 HF tokenizer 的 `apply_chat_template`（gigatoken 兼容模式下保留 HF 接口）
  - [x] SubTask 7.7: `packages/verse_infra/verse_infra/verse_tokenizer/__init__.py` 导出 `GigaTokenizerWrapper`
  - [x] SubTask 7.8: `bpe.py` 的 `load_tokenizer` 新增 `kind="giga"` 分支：优先 `GigaTokenizerWrapper`，gigatoken 不可用时**自动降级**到 `VerseTokenizer`（打印警告）
  - [x] SubTask 7.9: 新增 `tests/test_giga_tokenizer.py`：encode/decode 与 VerseTokenizer 一致 + 批量 encode 加速（≥10×）+ 自动降级 + apply_chat_template

- [x] Task 8: Gigatoken 设为默认
  - [x] SubTask 8.1: `spark/small/config/cometspark_small.yml` 的 `tokenizer.kind` 默认从 `"verse"` 改为 `"giga"`
  - [x] 8.2: `spark/mate/config/cometspark_mate.yml` 同步修改
  - [x] SubTask 8.3: `pyproject.toml`（`packages/verse_infra/pyproject.toml`）的 `verse-tokenizer` extras 新增 `[giga]`：`gigatoken >= 0.1.0`（不强制安装）
  - [x] SubTask 8.4: 现有 `tests/test_verse_tokenizer.py` / `test_tokenizer_nex_wrapper.py` / `test_tokenizer_upgrade.py` 零回归（默认 verse 路径仍可用）
  - [x] SubTask 8.5: 端到端验证：`python spark/run.py train --model small --dry-run` 默认走 gigatoken 路径（gigatoken 不可用时降级到 verse，不报错）

## 阶段 D：NPU CANN & AMD ROCm 兼容

- [x] Task 9: device.py ROCm / CANN 探测与设备识别
  - [x] SubTask 9.1: `packages/verse_torch/verse_torch/device.py` 的 `_parse_device` 新增 `"rocm"` / `"rocm:0"` 识别（等价于 `"cuda"`，HIP-on-ROCm 走 PyTorch `cuda` 路径）
  - [x] SubTask 9.2: 新增 `has_rocm() -> bool`：检测 `torch.version.hip is not None`
  - [x] SubTask 9.3: 新增 `get_rocm_version() -> Optional[str]`：返回 ROCm 版本字符串
  - [x] SubTask 9.4: 新增 `has_cann() -> bool`：检测 `torch_npu` 可用 + CANN 版本可读
  - [x] SubTask 9.5: 新增 `get_cann_version() -> Optional[str]`：返回 CANN 版本字符串
  - [x] SubTask 9.6: `get_backend("rocm:0")` 返回 `TorchBackend(device="cuda:0")`（内部映射，对外暴露 "rocm" 字符串仅用于诊断）
  - [x] SubTask 9.7: `empty_cache("rocm")` / `get_memory_info("rocm")` 走 `torch.cuda.empty_cache()` / `torch.cuda.mem_get_info()`（PyTorch ROCm build 已通过 HIP 暴露 cuda API）
  - [x] SubTask 9.8: `get_memory_info("npu")` 增加 `torch_npu.npu.mem_get_info` 兜底（部分 CANN 版本 API 名差异：`memory_allocated` / `mem_get_info`）

- [x] Task 10: backend_torch.py TorchBackend 兼容 + autocast 日志
  - [x] SubTask 10.1: `packages/verse_torch/verse_torch/backend_torch.py` 的 `_torch_device` 接受 `"rocm"` / `"rocm:0"`，内部映射到 `torch.device("cuda:0")`
  - [x] SubTask 10.2: `TorchBackend.__init__` 接受 `"rocm"` device 字符串，内部 `_device_str` 保留原值（用于诊断），`_torch_device` 映射到 cuda
  - [x] SubTask 10.3: `autocast` 上下文管理器：`device_type="rocm"` 等价 `device_type="cuda"`（PyTorch ROCm build 原生支持 fp16 autocast via HIP）
  - [x] SubTask 10.4: `autocast` 在 NPU 路径首次启用时打印 CANN 版本 + PyTorch 版本 + torch_npu 版本日志（便于运维诊断）
  - [x] SubTask 10.5: 现有 `tests/test_device_backend.py` 零回归（CPU 路径不变）

- [x] Task 11: spark/run.py CLI 兼容
  - [x] SubTask 11.1: `spark/run.py` 的 `--device` 参数支持 `"rocm"` / `"rocm:0"` / `"npu"` / `"cuda"` / `"cpu"`
  - [x] SubTask 11.2: 启动时打印设备信息（device type + 版本 + 显存，调用 `has_rocm` / `get_rocm_version` / `has_cann` / `get_cann_version`）
  - [x] SubTask 11.3: 新增 `tests/test_rocm_npu_compat.py`：ROCm/CANN 探测 API（无 ROCm/NPU 环境时跳过相关断言）+ device 字符串识别 + autocast 等价性

## 阶段 E：文档与代码注释

- [x] Task 12: ADR + 文档更新
  - [x] SubTask 12.1: 新增 `docs/architecture/adr-017-vn-v2-resume.md`：`.vn` v2 格式 + 复杂 Python 对象 + 断点续训设计
  - [x] SubTask 12.2: 新增 `docs/architecture/adr-018-gigatoken-integration.md`：Gigatoken 集成（默认 tokenizer + lazy import + 降级策略 + 不复刻原则）
  - [x] SubTask 12.3: 新增 `docs/architecture/adr-019-rocm-cann-compat.md`：NPU CANN & AMD ROCm 生态兼容（不自研 kernel）
  - [x] SubTask 12.4: 更新 `README.md`：默认 tokenizer 改为 gigatoken + .vn v2 断点续训 + NPU/ROCm 兼容
  - [x] SubTask 12.5: 更新 `docs/training_guide.md`：断点续训用法 + `--device rocm` / `--device npu` 示例
  - [x] SubTask 12.6: `vn_format.py` / `training.py` / `device.py` / `backend_torch.py` 代码注释统一到 v2 / ROCm / CANN 术语

## 阶段 F：验收

- [x] Task 13: 全量测试 + 综合验收
  - [x] SubTask 13.1: `pytest tests/` 全量零失败（含新增测试 test_parallel_recursion_fix / test_vn_v2_format / test_resume_manager / test_giga_tokenizer / test_rocm_npu_compat / test_checkpoint_atomic / test_checkpoint_vn）
  - [x] SubTask 13.2: 关键导入验证：`import verse_torch` / `from verse_torch.vn_format import VNFileWriter, VNFileReader, VN_FORMAT_VERSION`（==2）/ `from verse_torch.training import CheckpointManager, ResumeManager, ResumeState` / `from verse_infra.verse_tokenizer import GigaTokenizerWrapper`
  - [x] SubTask 13.3: 端到端：`spark/run.py train --model small --dry-run` 默认走 gigatoken + .vn v2 + CPU 路径
  - [x] SubTask 13.4: 端到端：构造 60+ 层 VerseNex 模型 + ParallelTrainerSafe 训练，断言不触发 RecursionError + val_loss 收敛
  - [x] SubTask 13.5: 端到端：训练中断后 `spark/run.py continue --model small --checkpoint best.vn` 恢复 step / optimizer state / rng / best_val_loss
  - [x] SubTask 13.6: 端到端：`spark/run.py train --model small --device rocm --dry-run`（无 ROCm 环境时仅验证 device 字符串识别 + 错误信息明确）
  - [x] SubTask 13.7: v1 向后兼容：用 Part5K1 写出的 v1 `.vn` 文件加载成功 + `read_optimizer_state` 返回 None
  - [x] SubTask 13.8: 更新 `audit_report.md`：Part5K1.3 修复 + 升级总结

# Task Dependencies

- Task 1（并行训练递归修复）依赖 Task 2（CheckpointManager 原子写）的 `use_vmpc` flag 与
  `format` 参数（避免 v1 .pt 与 v2 .vn 路径冲突），但 Task 1 主要是 `Module.state_dict`
  迭代化，可独立推进；Task 2 完成后再集成
- Task 3（VN_FORMAT_VERSION v2）独立，可与 Task 1 / Task 7 / Task 9 并行
- Task 4（CheckpointManager .vn 集成）依赖 Task 3（v2 API 就位）
- Task 5（VerseNex / VerseTorch 原生 .vn）依赖 Task 3（v2 API）+ Task 4（CheckpointManager .vn）
- Task 6（ResumeManager）依赖 Task 4（CheckpointManager .vn）+ Task 5（VerseNex 原生 .vn）
- Task 7（GigaTokenizerWrapper）独立，可与 Task 1~6 / Task 9~10 并行
- Task 8（Gigatoken 设为默认）依赖 Task 7（GigaTokenizerWrapper 就位）
- Task 9（device.py ROCm/CANN）独立，可与 Task 1~8 并行
- Task 10（backend_torch.py 兼容）依赖 Task 9（device.py 探测 API 就位）
- Task 11（spark/run.py CLI）依赖 Task 9 + Task 10
- Task 12（文档）依赖所有功能任务
- Task 13（验收）依赖所有

# 并行策略

- **第一批（独立）**：Task 1（并行训练递归修复 - vnn.Module 部分）+ Task 3（VN v2 格式）+
  Task 7（GigaTokenizerWrapper）+ Task 9（device.py ROCm/CANN 探测）
- **第二批**：Task 2（CheckpointManager 原子写 + format 参数）+ Task 4（CheckpointManager .vn 集成，
  依赖 Task 3）+ Task 10（backend_torch.py 兼容，依赖 Task 9）+ Task 8（Gigatoken 设为默认，
  依赖 Task 7）
- **第三批**：Task 5（VerseNex / VerseTorch 原生 .vn，依赖 Task 3 + Task 4）+
  Task 11（spark/run.py CLI，依赖 Task 9 + Task 10）
- **第四批**：Task 6（ResumeManager，依赖 Task 4 + Task 5）
- **第五批**：Task 12（文档）
- **第六批**：Task 13（验收）

# 不重复造轮子约束

- **不重新实现 gigatoken**：仅做 `BaseTokenizer` 接口适配 wrapper，直接 import gigatoken 库
- **不自研 ROCm kernel**：所有 GPU 计算走 PyTorch 原生（MIOpen / rocBLAS / FlashAttention-ROCm）
- **不自研 CANN kernel**：NPU 计算走 `torch_npu`（HCCL / hccl_kernel）
- **不重新实现 BPE/Unigram**：gigatoken 兼容模式下复用其 Rust 实现
- **不重复 .pt 已有能力**：`CheckpointManager` 保留 `format="pt"` 路径，仅新增 `"vn"` 选项
- **不重新实现 pickle 序列化**：`optimizer_state.pkl` / `extra_state.pkl` 直接用 `pickle.dump` / `pickle.load`
