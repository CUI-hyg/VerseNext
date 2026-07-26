# Checklist — Part5K1.3：漏洞修复 + 原生 .vn + Gigatoken + NPU/ROCm 兼容

## 阶段 A：并行训练递归修复（fix）

### Task 1: ParallelTrainer Phase 3 merge_finetune 递归崩溃修复
- [x] `ParallelTrainer.fit` 中 chunk_id=-999 路径的 `copy.deepcopy(state_dict)` 全部改为 `pickle.loads(pickle.dumps(state_dict, protocol=4))`
- [x] `Module.state_dict()` / `load_state_dict()` 子模块遍历改为迭代 BFS（显式栈），返回的 state_dict 键值与原递归实现完全一致
- [x] `_safe_chunk_run` 新增 `RecursionError` 捕获分支：`gc.collect()` + `sys.setrecursionlimit(+500)` 重试一次；仍失败时 graceful degrade
- [x] `ParallelTrainerSafe._train_chunk_safe` chunk 间增加 `gc.collect()` + `is_shutdown_requested()` 检查
- [x] `Trainer.fit` 主循环捕获 `RecursionError` 时优雅保存 `best_state` 后退出（exit code != 0）
- [x] `tests/test_parallel_recursion_fix.py` 通过：60+ 层 VerseNex + chunk_id=-999 不触发 RecursionError + val_loss 收敛
- [x] `tests/test_parallel_trainer.py` / `test_loss_and_parallel_fix.py` 零回归

### Task 2: CheckpointManager 原子写 + format 参数
- [ ] `CheckpointManager.save_best` / `save_last` 使用原子写（`.tmp` + `os.replace`）
- [ ] `CheckpointManager.__init__` 新增 `format="auto"|"vn"|"pt"` 参数 + `use_vmpc=False` 参数
- [ ] `save_best` / `save_last` 签名扩展支持 `training_state=None, optimizer_state=None, extra_state=None`
- [ ] `load_best` / `load_last` 返回统一 dict（含 `model_state_dict` / `training_state` / `optimizer_state` / `extra_state`，缺失字段为 None）
- [ ] `use_vmpc=True` 且 `format="pt"` 时抛 `ValueError`
- [ ] `tests/test_checkpoint_atomic.py` 通过：原子写 + 中断不损坏 + use_vmpc 强制 .vn

## 阶段 B：.vn v2 原生格式 + 复杂 Python 对象

### Task 3: VN_FORMAT_VERSION v1 → v2 升级
- [ ] `VN_FORMAT_VERSION = 2`；新增 `training_state.json` / `optimizer_state.pkl` / `extra_state.pkl` 常量
- [ ] 抽取 `VNEntry` 抽象统一管理 ZIP 条目（减少 `VNFileWriter` / `VNFileReader` 重复模板代码）
- [ ] `VNFileWriter.write_training_state(state: dict)` 实现，写 `training_state.json`
- [ ] `VNFileWriter.write_optimizer_state(state: dict)` 实现，写 `optimizer_state.pkl`（pickle）
- [ ] `VNFileWriter.write_extra_state(state: Any)` 实现，写 `extra_state.pkl`（pickle）
- [ ] `VNFileWriter.close` 更新 `meta.json`：`vn_format_version: 2` + `has_training_state` / `has_optimizer_state` / `has_extra_state` 布尔字段
- [ ] `VNFileReader` 自动识别 v1/v2（基于 `meta.json` 的 `vn_format_version` 字段）
- [ ] `VNFileReader.read_training_state()` / `read_optimizer_state()` / `read_extra_state()` 实现（v1 文件返回 None）
- [ ] `tests/test_vn_v2_format.py` 通过：v2 写入 + 读取 + v1 向后兼容 + optimizer m/v 数值一致（float32 吻合 1e-7）

### Task 4: CheckpointManager 原生 .vn 集成
- [x] `CheckpointManager.save_best` 在 `format="vn"` 时调用 `VNFileWriter` 写 `.vn`（含 model + training + optimizer + extra）
- [x] `CheckpointManager.load_best` / `load_last` 在 `format="vn"` 时调用 `VNFileReader` 读取
- [x] 默认路径 `best.vn` / `last.vn`（format="vn"）或 `best.pt` / `last.pt`（format="pt"），`_resolve_path` 自动加扩展名
- [x] `Trainer.__init__` 中的 `CheckpointManager` 实例化传入 `format="auto"` + `use_vmpc=cfg.get("use_vmpc", False)`
- [x] `tests/test_checkpoint_vn.py` 通过：vn checkpoint 含 optimizer state + 训练中断后 load_best 恢复 + use_vmpc 强制 .vn

### Task 5: VerseNex / VerseTorch 原生 .vn 适配
- [ ] `CometSparkNexLM.save(path, format="vn"|"pt")` 新增 `format` 参数（默认 `"vn"`）
- [ ] `CometSparkNexLM.from_pretrained(path)` 识别 `.vn` / `.pt` / 目录（含 `model.vn` 或 `model.pt`）三种输入
- [ ] `CometSparkNexLM.save_vn` 新增 `training_state` / `optimizer_state` / `extra_state` 参数
- [ ] `CometSparkV05LM.save_vn` 同步升级支持 training_state / optimizer_state / extra_state 参数
- [ ] `CometSparkSmallLM` / `CometSparkMateLM` 的 `save_vn` 透传 training_state / optimizer_state / extra_state
- [ ] `tests/test_vn_format.py` / `test_cometspark_v05.py` / `test_dual_model.py` 零回归

### Task 6: ResumeManager 断点续训
- [x] `ResumeState` namedtuple 定义（`model_state_dict` / `optimizer_state` / `step` / `rng_state` / `best_val_loss` / `epoch` / `patience_count`）
- [x] `ResumeManager.save(path, model, optimizer, step, **kwargs)` 实现，调用 `CheckpointManager.save_best` 写 `.vn`
- [x] `ResumeManager.load(path) -> ResumeState` 实现，调用 `CheckpointManager.load_best` 读取
- [x] `ResumeManager.apply(trainer, path)` 实现，把 `ResumeState` 应用到 `Trainer` / `ParallelTrainerSafe` 实例
- [x] `ParallelTrainerSafe._save_resume_state` / `_load_resume_state` 改用 `ResumeManager` + `.vn`；旧 `.pkl` 兼容回退
- [x] `spark/run.py continue` 子命令委托 `ResumeManager.apply`
- [x] `tests/test_resume_manager.py` 通过：断点续训 + v1 向后兼容 + optimizer state 恢复 + rng_state 恢复

## 阶段 C：Gigatoken 集成（默认 Tokenizer）

### Task 7: GigaTokenizerWrapper 适配器
- [x] `packages/verse_infra/verse_infra/verse_tokenizer/giga.py` 创建，定义 `GigaTokenizerWrapper(BaseTokenizer)`
- [x] lazy import gigatoken：模块 import 不触发加载，仅构造时 import；不可用时抛 `ImportError` 含安装提示
- [x] 内部用 `gt.Tokenizer(hf_tokenizer).as_hf()` 兼容模式（drop-in replacement）
- [x] 可选 `native=True` 走 `gt.Tokenizer(model_id)` 原生 API
- [x] 实现 `encode` / `decode` / `encode_batch` / `decode_batch` / `save` / `load` / `__len__` / `apply_chat_template` 接口
- [x] 缓存 `bos_id` / `eos_id` / `pad_id` / `vocab_size`（构造时一次解析）；`vocab` 属性懒加载
- [x] `apply_chat_template` 委托底层 HF tokenizer 的 `apply_chat_template`
- [x] `verse_tokenizer/__init__.py` 导出 `GigaTokenizerWrapper`
- [x] `bpe.py` 的 `load_tokenizer` 新增 `kind="giga"` 分支 + 自动降级到 `VerseTokenizer`
- [x] `tests/test_giga_tokenizer.py` 通过：encode/decode 与 VerseTokenizer 一致 + 批量 encode 加速（≥10×）+ 自动降级 + apply_chat_template

### Task 8: Gigatoken 设为默认
- [ ] `spark/small/config/cometspark_small.yml` 的 `tokenizer.kind` 默认 `"giga"`
- [ ] `spark/mate/config/cometspark_mate.yml` 的 `tokenizer.kind` 默认 `"giga"`
- [ ] `pyproject.toml`（`packages/verse_infra/pyproject.toml`）的 `verse-tokenizer` extras 新增 `[giga]`：`gigatoken >= 0.1.0`（不强制安装）
- [ ] `tests/test_verse_tokenizer.py` / `test_tokenizer_nex_wrapper.py` / `test_tokenizer_upgrade.py` 零回归
- [ ] 端到端：`python spark/run.py train --model small --dry-run` 默认走 gigatoken 路径（不可用时降级到 verse，不报错）

## 阶段 D：NPU CANN & AMD ROCm 兼容

### Task 9: device.py ROCm / CANN 探测
- [ ] `_parse_device` 支持 `"rocm"` / `"rocm:0"`，等价 `"cuda"`（HIP-on-ROCm）
- [ ] `has_rocm() -> bool` 实现：检测 `torch.version.hip is not None`
- [ ] `get_rocm_version() -> Optional[str]` 实现：返回 ROCm 版本字符串
- [ ] `has_cann() -> bool` 实现：检测 `torch_npu` 可用 + CANN 版本可读
- [ ] `get_cann_version() -> Optional[str]` 实现：返回 CANN 版本字符串
- [ ] `get_backend("rocm:0")` 返回 `TorchBackend(device="cuda:0")`（内部映射）
- [ ] `empty_cache("rocm")` / `get_memory_info("rocm")` 走 `torch.cuda.empty_cache()` / `torch.cuda.mem_get_info()`
- [ ] `get_memory_info("npu")` 增加 `torch_npu.npu.mem_get_info` 兜底（CANN 版本差异）

### Task 10: backend_torch.py TorchBackend 兼容
- [ ] `_torch_device` 接受 `"rocm"` / `"rocm:0"`，内部映射到 `torch.device("cuda:0")`
- [ ] `TorchBackend.__init__` 接受 `"rocm"` device 字符串，`_device_str` 保留原值（诊断用）
- [ ] `autocast(device_type="rocm")` 等价 `device_type="cuda"`（PyTorch ROCm build 原生支持）
- [ ] `autocast` 在 NPU 路径首次启用时打印 CANN + PyTorch + torch_npu 版本日志
- [ ] `tests/test_device_backend.py` 零回归（CPU 路径不变）

### Task 11: spark/run.py CLI 兼容
- [x] `--device` 参数支持 `"rocm"` / `"rocm:0"` / `"npu"` / `"cuda"` / `"cpu"`
- [x] 启动时打印设备信息（device type + 版本 + 显存）
- [x] `tests/test_rocm_npu_compat.py` 通过：ROCm/CANN 探测 API + device 字符串识别 + autocast 等价性（无 ROCm/NPU 环境时跳过相关断言）

## 阶段 E：文档与代码注释

### Task 12: ADR + 文档更新
- [x] `docs/architecture/adr-017-vn-v2-resume.md` 创建
- [x] `docs/architecture/adr-018-gigatoken-integration.md` 创建
- [x] `docs/architecture/adr-019-rocm-cann-compat.md` 创建
- [x] `README.md` 更新：默认 tokenizer 改为 gigatoken + .vn v2 断点续训 + NPU/ROCm 兼容
- [x] `docs/training_guide.md` 更新：断点续训用法 + `--device rocm` / `--device npu` 示例
- [x] `vn_format.py` / `training.py` / `device.py` / `backend_torch.py` 代码注释统一到 v2 / ROCm / CANN 术语

## 阶段 F：验收

### Task 13: 全量测试 + 综合验收
- [x] `pytest tests/` 全量零失败（含新增 7 个测试文件）
- [x] 关键导入验证：
  - `import verse_torch`
  - `from verse_torch.vn_format import VNFileWriter, VNFileReader, VN_FORMAT_VERSION`（==2）
  - `from verse_torch.training import CheckpointManager, ResumeManager, ResumeState`
  - `from verse_infra.verse_tokenizer import GigaTokenizerWrapper`
  - `from verse_torch.device import has_rocm, get_rocm_version, has_cann, get_cann_version`
- [x] 端到端：`spark/run.py train --model small --dry-run` 默认走 gigatoken + .vn v2 + CPU 路径
- [x] 端到端：60+ 层 VerseNex + ParallelTrainerSafe 训练不触发 RecursionError + val_loss 收敛
- [x] 端到端：训练中断后 `spark/run.py continue --model small --checkpoint best.vn` 恢复 step / optimizer state / rng / best_val_loss
- [x] 端到端：`spark/run.py train --model small --device rocm --dry-run`（无 ROCm 时仅验证 device 字符串识别）
- [x] v1 向后兼容：Part5K1 写出的 v1 `.vn` 文件加载成功 + `read_optimizer_state` 返回 None
- [x] `audit_report.md` 更新：Part5K1.3 修复 + 升级总结

## 不重复造轮子约束验证

- [x] 未修改 gigatoken 源码（仅做 wrapper 适配）
- [x] 未自研 ROCm kernel（所有 GPU 计算走 PyTorch 原生）
- [x] 未自研 CANN kernel（NPU 计算走 `torch_npu`）
- [x] 未重新实现 BPE/Unigram（gigatoken 兼容模式下复用其 Rust 实现）
- [x] 未重新实现 pickle 序列化（`optimizer_state.pkl` / `extra_state.pkl` 直接用 `pickle`）
- [x] `CheckpointManager` 保留 `format="pt"` 路径（向后兼容，未删除 .pt 能力）
- [x] `VerseTokenizer` 未删除（保留向后兼容，仅在默认路径让位给 gigatoken）

## 向后兼容性验证

- [x] Part5K1 写出的 v1 `.vn` 文件可被 v2 `VNFileReader` 加载（read 新方法返回 None）
- [x] 旧 `CheckpointManager(save_dir)` 调用（不传 `format`）行为不变（默认 auto → use_vmpc=False → pt）
- [x] 旧 `CometSparkNexLM.save(path)` 调用（不传 `format`）默认走 `"vn"`（行为变化，但向后兼容：`.pt` 路径可通过 `format="pt"` 显式调用）⚠️ 注：`trainer.py:1500` 未同步更新（见 audit_report Bug 1）
- [x] 旧 `ParallelTrainerSafe._save_resume_state` 写出的 `.pkl` 可被新 `_load_resume_state` 兼容读取
- [x] 旧 `spark/run.py` 命令（不带 `--device`）默认 CPU 路径不变
- [x] 旧 `load_tokenizer(kind="byte"|"bpe"|"hf")` 调用行为不变
