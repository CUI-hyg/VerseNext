# Part5K1.3：漏洞修复 + 原生 .vn + Gigatoken + NPU/ROCm 兼容 Spec

## Why

Part5K1 已落地 VMPC V1.5 / 双模型并行 / VMT / vnn 重命名 / jsonl_repair 等完整能力，
但在实际训练与部署中暴露出 4 个阻塞性短板，影响"自研原生架构稳定可用"的承诺：

1. **并行训练崩溃**：`ParallelTrainerSafe` 在 Phase 3 整体 fine-tune 阶段（chunk_id=-999）
   抛出 `maximum recursion depth exceeded in __instancecheck__`，导致 merge_finetune
   阶段被跳过、训练结果不稳定。根因在 `copy.deepcopy(self.best_state_dict)` 与
   `Module.__setattr__` 的递归 `isinstance` 链在大模型 / 深层 module 树下爆栈。
2. **`.vn` 格式仍是壳**：`CheckpointManager`（`verse_torch.training`）始终写
   `best.pt` / `last.pt` / `cometspark_emergency.pt`，即便 `use_vmpc=True`、
   `CometSparkSmallLM.save(format="vn")` 也只覆盖显式 `save()` 调用，训练过程中
   仍走 `.pt` 路径；且 `.vn` v1 仅存权重（state_dict），**不支持 optimizer 状态、
   训练步数、调度器状态、RNG 状态等任意 Python 对象**，无法支持断点续训。
3. **Tokenizer 速度瓶颈**：当前默认走 `VerseTokenizer`（lazy import transformers），
   HuggingFace `AutoTokenizer` 在大语料 encode 时吞吐量受限；社区已有
   `gigatoken`（Rust 实现，~1000× 快于 HF tokenizers，drop-in 兼容），但项目
   未集成，违背"优先复用已有优秀库，不自研"的宗旨。
4. **NPU/ROCm 生态兼容不完整**：`device.py` 已支持 `npu`（通过 `torch_npu`），
   但 `cuda` 路径未显式声明对 AMD ROCm（HIP-on-ROCm）的支持，ROCm 环境下
   `--device rocm` 无法识别；NPU autocast / 显存管理路径未对 CANN 版本做适配；
   缺少 ROCm/CANN 探测 API，运维无法判断环境是否可用。

Part5K1.3 不引入新业务能力，专注**修 bug + 把壳换成原生 + 接优秀生态 + 兼容硬件**，
让自研架构在"稳定可用 / 原生高性能 / 生态兼容"三维度真正落地。

> **宗旨锚点**："我们使用自研原生架构就是为了解决已有框架的问题，
> 而不是重复造轮子，甚至做得不如已有框架。"
> ——本版所有改造严格遵循"修壳、接库、兼容生态"，**不自研 tokenizer、不自研 kernel、
> 不重复造 .pt 已有能力**。

## What Changes

### 1. 并行训练稳定性修复（fix）
- **根因修复**：`ParallelTrainer.fit` 的 Phase 3 merge_finetune（chunk_id=-999）
  路径不再触发 `RecursionError`：
  - `copy.deepcopy(self.best_state_dict)` → 改用迭代式深拷贝
    （`pickle.loads(pickle.dumps(sd, protocol=4))`，避免递归遍历对象图）
  - `Module.state_dict()` / `load_state_dict()` 的子模块遍历从**递归 DFS** 改为
    **迭代 BFS**（显式栈），消除深层 module 树下的 `__instancecheck__` 递归链。
  - `_safe_chunk_run` 新增 `RecursionError` 捕获分支：捕获后**清理调用栈上下文**
    （`gc.collect()` + `sys.setrecursionlimit` 临时上调）并重试一次；仍失败才走
    "跳过该 chunk 继续后续流程"路径（保留现有 graceful degrade 行为）。
- **稳定性增强**（其他部分）：
  - `CheckpointManager.save_best` / `save_last` 增加**原子写**（先写 `.tmp` 再
    `os.replace`），避免训练中断时写出半截文件导致下次启动无法加载。
  - `ParallelTrainer.fit` 在每个 chunk 之间增加 `gc.collect()` + 检查
    `is_shutdown_requested()`，避免 chunk 残留状态泄漏到下一个 chunk。
  - `Trainer.fit` 主循环捕获 `RecursionError` 时**优雅保存当前 best_state** 后退出
    （exit code != 0），不再让进程崩在栈溢出处丢失 checkpoint。
  - 新增 `tests/test_parallel_recursion_fix.py`：构造 60+ 层 VerseNex 模型 +
    chunk_id=-999 路径，断言不触发 RecursionError + val_loss 收敛。

### 2. 原生 .vn 格式 + 复杂 Python 对象 + 断点续训（feat/fix）
- **`.vn` 格式升级到 v2**（向后兼容 v1）：
  - `VN_FORMAT_VERSION` 从 `1` 升到 `2`；`VNFileReader` 自动识别 v1/v2。
  - v2 新增 ZIP 内条目（v1 文件不含，读取时按缺失处理）：
    - `training_state.json` —— 训练元信息（step / epoch / best_val_loss /
      patience_count / rng_state_hex / config_snapshot_hash）
    - `optimizer_state.pkl` —— optimizer 状态（pickle，承载 AdamW m/v 矩阵、
      step count、lr scheduler state 等任意 Python 对象）
    - `extra_state.pkl` —— 其他任意 Python 对象（用户自定义 callback state、
      EMA state、grad scaler state 等），由 `extra_state` 参数传入
  - v2 `meta.json` 新增字段：`"vn_format_version": 2` / `"has_training_state": bool` /
    `"has_optimizer_state": bool` / `"has_extra_state": bool`。
- **`VNFileWriter` 扩展**：
  - 新增 `write_training_state(state: dict)`：写 `training_state.json`
    （JSON-able dict，含 step / epoch / best_val_loss / rng 等）
  - 新增 `write_optimizer_state(state: dict)`：写 `optimizer_state.pkl`
    （pickle，承载任意 Python 对象：AdamW 的 exp_avg / exp_avg_sq / step、
    scheduler 的 base_lru / last_epoch 等）
  - 新增 `write_extra_state(state: Any)`：写 `extra_state.pkl`（pickle，
    用户自定义任意对象）
- **`VNFileReader` 扩展**：
  - 新增 `read_training_state() -> Optional[dict]`
  - 新增 `read_optimizer_state() -> Optional[dict]`
  - 新增 `read_extra_state() -> Optional[Any]`
  - v1 文件读取这些方法返回 `None`（向后兼容）
- **`CheckpointManager` 原生 .vn 支持**：
  - `__init__` 新增 `format: str = "auto"` 参数：
    - `"auto"`：根据 `use_vmpc` flag 自动选择（use_vmpc=True → "vn"，否则 "pt"）
    - `"vn"`：`save_best` / `save_last` 写 `.vn` 文件
    - `"pt"`：保留现有 `.pt` 行为（向后兼容）
  - `save_best(state, training_state=None, optimizer_state=None, extra_state=None)`：
    扩展签名，支持传入训练状态 + optimizer 状态 + 额外状态。
  - `load_best() -> dict` / `load_last() -> dict`：返回包含 `model_state_dict` /
    `training_state` / `optimizer_state` / `extra_state` 的统一 dict（缺失字段为 None）。
  - **强制 .vn**：`use_vmpc=True` 时 `format="pt"` 被拒绝并抛 `ValueError`
    （与 `CometSparkSmallLM._enforce_vn_format` 一致）。
- **VerseNex / VerseTorch 原生 .vn 适配**：
  - `verse_nex.CometSparkNexLM.save(path, format="vn"|"pt")`：新增 `format` 参数
    （默认 `"vn"`），与 `CometSparkSmallLM` 接口对齐；`format="vn"` 走
    `save_vn` 路径，`format="pt"` 走原 pickle 路径。
  - `verse_nex.CometSparkNexLM.from_pretrained(path)`：识别 `.vn` / `.pt` /
    目录（含 `model.vn` 或 `model.pt`）三种输入。
  - `CometSparkSmallLM` / `CometSparkMateLM` 的 `save_vn` 升级支持
    `training_state` / `optimizer_state` / `extra_state` 参数。
- **断点续训（ResumeManager）**：
  - 新增 `verse_torch/training.py.ResumeManager`：统一管理断点续训状态。
    - `save(path, model, optimizer, step, **kwargs)`：调用 `CheckpointManager`
      写 `.vn` checkpoint（含 model + optimizer + step + rng_state + best_val_loss）
    - `load(path) -> ResumeState`：返回 `ResumeState` namedtuple
      （`model_state_dict` / `optimizer_state` / `step` / `rng_state` / `best_val_loss`）
    - `apply(trainer, path)`：把 `ResumeState` 应用到 `Trainer` / `ParallelTrainerSafe`
      实例（恢复 model / optimizer / step / rng / best_val_loss）
  - `ParallelTrainerSafe._save_resume_state` / `_load_resume_state`：
    改用 `ResumeManager` + `.vn` 格式（原 `.pkl` resume 文件保留为兼容回退）。
  - `spark/run.py continue` 子命令委托 `ResumeManager.apply` 完成续训状态恢复。
- **核心架构优化**：
  - `vn_format.py` 抽取 `VNEntry` 抽象（统一管理 ZIP 内条目的写入/读取/校验），
    减少 `VNFileWriter` / `VNFileReader` 重复的 `writestr` / `open` 模板代码。
  - `CheckpointManager` 与 `ResumeManager` 共享 `_serialize_state_dict` /
    `_deserialize_state_dict` 辅助函数，避免两处分别实现 pickle/npz 序列化。
- **新增 ADR-017**：`.vn` v2 格式 + 复杂 Python 对象 + 断点续训设计。

### 3. Gigatoken 集成（feat，作为默认 Tokenizer）
- **新增 `verse_infra/verse_tokenizer/giga.py`**：
  - `GigaTokenizerWrapper(BaseTokenizer)`：包装 `gigatoken.Tokenizer`，
    实现 `encode` / `decode` / `encode_batch` / `decode_batch` / `save` / `load` /
    `__len__` / `apply_chat_template` 等接口（对齐 `VerseTokenizer`）。
  - **lazy import gigatoken**：模块 import 不触发 `gigatoken` 加载，
    仅构造 `GigaTokenizerWrapper` 时 import；不可用时抛 `ImportError` 含安装提示。
  - 内部用 `gt.Tokenizer(hf_tokenizer).as_hf()` 兼容模式（drop-in replacement），
    保证与 `VerseTokenizer` 输出一致；可选 `native=True` 走 `gt.Tokenizer(model_id)`
    原生 API（更快，但需要单独训练/加载）。
  - 缓存 `bos_id` / `eos_id` / `pad_id` / `vocab_size`（构造时一次解析）。
  - `apply_chat_template` 委托底层 HF tokenizer 的 `apply_chat_template`
    （gigatoken 兼容模式下保留 HF 接口）。
- **设为默认 Tokenizer**：
  - `load_tokenizer(kind="giga", ...)` 新增 `kind="giga"` 分支：
    优先 `GigaTokenizerWrapper`，gigatoken 不可用时**自动降级**到
    `VerseTokenizer`（保持向后兼容，不强制安装）。
  - `spark/small/config/cometspark_small.yml` 与 `spark/mate/config/cometspark_mate.yml`：
    `tokenizer.kind` 默认从 `"verse"` 改为 `"giga"`，`tokenizer.repo` 保留。
  - `VerseTokenizer` 不删除（向后兼容），仅在默认路径上让位给 `GigaTokenizerWrapper`。
  - `verse_tokenizer/__init__.py` 导出 `GigaTokenizerWrapper`。
- **直接导入库，不复刻**：
  - **不修改 gigatoken 源码**，**不重新实现 BPE/Unigram**；
  - 仅做接口适配（`BaseTokenizer` 抽象的 Python wrapper）；
  - `pyproject.toml` 的 `verse-tokenizer` extras 新增 `[giga]` 可选依赖：
    `gigatoken >= 0.1.0`。
- **新增 ADR-018**：Gigatoken 集成（默认 tokenizer + lazy import + 降级策略）。

### 4. NPU CANN & AMD ROCm 兼容（feat，不自研 kernel）
- **`device.py` 设备字符串扩展**：
  - `_parse_device` 新增 `"rocm"` 识别（与 `"cuda"` 等价，HIP-on-ROCm 走 PyTorch
    `cuda` 路径）；`"rocm:0"` / `"rocm"` 均可识别。
  - 新增 `has_rocm() -> bool`：检测当前 PyTorch 是否为 ROCm build
    （`torch.version.hip is not None` 或 `hasattr(torch.cuda, 'is_available')`
    且 `torch.cuda.device_count() > 0` 且 ROCm 环境）。
  - 新增 `has_cann() -> bool`：检测 `torch_npu` 是否可用 + CANN 版本可读
    （`torch_npu.version` 或 `torch_npu.cann_version`，便于诊断）。
  - 新增 `get_cann_version() -> Optional[str]` / `get_rocm_version() -> Optional[str]`。
- **`backend_torch.py` 兼容性增强**：
  - `TorchBackend` 接受 `"rocm"` device 字符串，内部映射到 `torch.device("cuda")`
    （HIP-on-ROCm 复用 cuda 路径，**不自研 ROCm kernel**）。
  - `autocast` 上下文管理器：`device_type="rocm"` 等价 `"cuda"`（PyTorch ROCm
    build 下 `torch.autocast(device_type="cuda")` 原生支持 ROCm fp16）。
  - `autocast` 在 NPU 路径下增加 CANN 版本日志（首次启用时打印 CANN 版本 +
    PyTorch 版本 + torch_npu 版本，便于运维诊断）。
- **`device.py` 显存管理兼容**：
  - `empty_cache` / `get_memory_info` 在 `"rocm"` 设备下走 `torch.cuda.empty_cache()`
    （PyTorch ROCm build 已通过 HIP 暴露 cuda API）。
  - NPU 路径增加 `torch_npu.npu.mem_get_info` 兜底（部分 CANN 版本 API 名差异）。
- **`spark/run.py` CLI 兼容**：
  - `--device` 参数支持 `"rocm"` / `"rocm:0"` / `"npu"` / `"cuda"` / `"cpu"`。
  - 启动时打印设备信息（device type + 版本 + 显存），便于用户确认环境。
- **不重复造轮子约束**：
  - **不自研 ROCm kernel**：所有 GPU 计算走 PyTorch 原生（cuBLAS / MIOpen /
    FlashAttention-ROCm）。
  - **不自研 CANN kernel**：NPU 计算走 `torch_npu`（HCCL / hccl_kernel）。
  - 仅做 device 字符串识别 + autocast 适配 + 版本探测。
- **新增 ADR-019**：NPU CANN & AMD ROCm 生态兼容（不自研 kernel）。

### 5. 文档与注释
- 新增 ADR-017 / ADR-018 / ADR-019（见上）。
- 更新 `README.md`：默认 tokenizer 改为 gigatoken + .vn v2 断点续训 + NPU/ROCm 兼容。
- 更新 `docs/training_guide.md`：断点续训用法 + `--device rocm` / `--device npu` 示例。
- 代码注释：`vn_format.py` / `training.py` / `device.py` 注释统一到 v2 / ROCm / CANN 术语。

## Impact

- **Affected specs**：`part5k1-vmpc-dual-model`（已完成，本版在其上叠加修复 + 升级，
  不回滚已完成能力）。
- **Affected code**:
  - `packages/verse_torch/verse_torch/`:
    - `vn_format.py`：`VN_FORMAT_VERSION=2` + `VNFileWriter` / `VNFileReader`
      新增 `write_training_state` / `write_optimizer_state` / `write_extra_state`
      / 对应 read 方法 + 抽取 `VNEntry` 抽象。
    - `vnn.py`：`Module.state_dict` / `load_state_dict` 改为迭代 BFS 遍历。
    - `training.py`：`CheckpointManager` 支持 `format="vn"|"pt"|"auto"` +
      `save_best(state, training_state, optimizer_state, extra_state)` 扩展 +
      原子写 + `RecursionError` 优雅保存；新增 `ResumeManager` + `ResumeState`。
    - `device.py`：`_parse_device` 支持 `"rocm"` + `has_rocm` / `has_cann` /
      `get_cann_version` / `get_rocm_version`。
    - `backend_torch.py`：`TorchBackend` 兼容 `"rocm"` device + autocast 日志。
    - `parallel.py`：可选（若 parallel 路径也涉及 state_dict 递归，同步改迭代）。
  - `packages/verse_nex/verse_nex/`:
    - `cometspark.py`：`CometSparkNexLM.save(path, format="vn"|"pt")` 新增 format
      参数；`from_pretrained` 识别 `.vn` / `.pt` / 目录（含 `model.vn`/`model.pt`）。
  - `packages/verse_infra/verse_infra/verse_trainer/`:
    - `trainer.py`：`ParallelTrainerSafe._save_resume_state` /
      `_load_resume_state` 改用 `ResumeManager` + `.vn`。
  - `packages/verse_infra/verse_infra/verse_tokenizer/`:
    - 新增 `giga.py`：`GigaTokenizerWrapper`（lazy import gigatoken）。
    - `__init__.py` 导出 `GigaTokenizerWrapper`。
    - `bpe.py` 的 `load_tokenizer`：新增 `kind="giga"` 分支 + 自动降级。
  - `spark/`:
    - `spark/small/config/cometspark_small.yml` + `spark/mate/config/cometspark_mate.yml`：
      `tokenizer.kind` 默认 `"giga"`。
    - `spark/run.py`：`--device` 支持 `"rocm"`；`continue` 子命令委托
      `ResumeManager.apply`。
  - `tests/`：
    - 新增 `test_parallel_recursion_fix.py` / `test_vn_v2_format.py` /
      `test_resume_manager.py` / `test_giga_tokenizer.py` /
      `test_rocm_npu_compat.py`。
  - `docs/`：
    - 新增 `adr-017-vn-v2-resume.md` / `adr-018-gigatoken-integration.md` /
      `adr-019-rocm-cann-compat.md`。
    - 更新 `README.md` / `training_guide.md`。

## ADDED Requirements

### Requirement: 并行训练递归崩溃修复

系统 SHALL 在 `ParallelTrainer.fit` 的 Phase 3 merge_finetune（chunk_id=-999）
路径下不触发 `RecursionError: maximum recursion depth exceeded in __instancecheck__`：

1. **state_dict 深拷贝**：`copy.deepcopy(state_dict)` 改为
   `pickle.loads(pickle.dumps(state_dict, protocol=4))`，避免递归遍历对象图。
2. **Module 遍历迭代化**：`Module.state_dict()` / `load_state_dict()` 的子模块
   遍历从递归 DFS 改为迭代 BFS（显式栈），消除深层 module 树下的递归链。
3. **RecursionError 捕获**：`_safe_chunk_run` 新增 `RecursionError` 分支：
   - 捕获后 `gc.collect()` + 临时上调 `sys.setrecursionlimit`（+500）重试一次。
   - 仍失败时走"跳过该 chunk 继续后续流程"路径（保留现有 graceful degrade）。
4. **chunk 间清理**：每个 chunk 结束后 `gc.collect()` + 检查
   `is_shutdown_requested()`，避免残留状态泄漏到下一个 chunk。
5. **Trainer.fit 主循环**：捕获 `RecursionError` 时优雅保存当前 `best_state`
   后退出（exit code != 0），不丢 checkpoint。

#### Scenario: chunk -999 不爆栈
- **WHEN** 训练 60+ 层 VerseNex 模型，进入 Phase 3 merge_finetune（chunk_id=-999）
- **THEN** 不触发 `RecursionError`，merge_finetune 正常完成，val_loss 收敛

#### Scenario: RecursionError 优雅降级
- **WHEN** chunk 抛出 `RecursionError`（极端深栈场景）
- **THEN** 自动重试一次（提高 recursionlimit + gc.collect），仍失败时跳过该 chunk
  继续后续流程，不丢失已训练的 best_state

### Requirement: CheckpointManager 原子写

系统 SHALL 在 `CheckpointManager.save_best` / `save_last` 写入时使用原子写：

1. **原子写**：先写到 `best.pt.tmp` / `last.pt.tmp`（或 `best.vn.tmp`），
   再 `os.replace` 重命名为目标文件，避免训练中断时写出半截文件。
2. **跨平台**：`os.replace` 在 Linux / Windows 均为原子操作（POSIX rename
   semantics；Windows 上 `os.replace` 覆盖目标）。
3. **异常处理**：写入失败时清理 `.tmp` 文件，不影响已有 checkpoint。

#### Scenario: 训练中断不损坏 checkpoint
- **WHEN** 训练过程中进程被 SIGKILL / 断电
- **THEN** `best.pt` / `best.vn` 文件要么是上次完整的 checkpoint，要么不存在
  （不会有半截文件导致下次启动无法加载）

### Requirement: `.vn` v2 格式与复杂 Python 对象

系统 SHALL 把 `.vn` 格式升级到 v2，向后兼容 v1，并支持任意 Python 对象：

1. **版本兼容**：`VN_FORMAT_VERSION = 2`；`VNFileReader` 自动识别 v1/v2。
   - v1 文件读取时 `read_training_state` / `read_optimizer_state` /
     `read_extra_state` 返回 `None`。
   - v2 文件写入时 `meta.json` 包含 `vn_format_version: 2` +
     `has_training_state` / `has_optimizer_state` / `has_extra_state` 布尔字段。
2. **`training_state.json`**：JSON-able dict，含 `step` / `epoch` /
   `best_val_loss` / `patience_count` / `rng_state_hex`（numpy RandomState 的
   `get_state()` 序列化为 hex 字符串）/ `config_snapshot_hash`。
3. **`optimizer_state.pkl`**：pickle，承载任意 Python 对象：
   - AdamW 的 `exp_avg` / `exp_avg_sq` / `step` 矩阵
   - lr scheduler 的 `base_lru` / `last_epoch` / `_step_count`
   - 用户自定义 optimizer 的任意 state dict
4. **`extra_state.pkl`**：pickle，承载用户自定义任意对象（EMA state / grad scaler
   state / callback state 等）。
5. **`VNFileWriter` API**：
   - `write_training_state(state: dict) -> None`
   - `write_optimizer_state(state: dict) -> None`
   - `write_extra_state(state: Any) -> None`
6. **`VNFileReader` API**：
   - `read_training_state() -> Optional[dict]`
   - `read_optimizer_state() -> Optional[dict]`
   - `read_extra_state() -> Optional[Any]`

#### Scenario: v1 向后兼容
- **WHEN** 用 `VNFileReader` 读取 Part5K1 写出的 v1 `.vn` 文件
- **THEN** `read_meta()["vn_format_version"] == 1`，`read_training_state()` 返回 `None`，
  `read_weights()` 正常返回权重

#### Scenario: v2 写入与读取
- **WHEN** 用 `VNFileWriter` v2 写入含 optimizer_state（AdamW 的 m/v 矩阵）的 `.vn`
- **THEN** `VNFileReader.read_optimizer_state()` 返回的 dict 与写入的 m/v 矩阵数值一致
  （float32 吻合到 1e-7）

### Requirement: CheckpointManager 原生 .vn 支持

系统 SHALL 在 `CheckpointManager` 支持原生 `.vn` 输出（不再始终 `.pt`）：

1. **`format` 参数**：`CheckpointManager(save_dir, format="auto"|"vn"|"pt")`。
   - `"auto"`：根据 `use_vmpc` flag 自动选择（use_vmpc=True → "vn"，否则 "pt"）
   - `"vn"`：`save_best` / `save_last` 写 `.vn` 文件
   - `"pt"`：保留现有 `.pt` 行为（向后兼容）
2. **扩展签名**：
   - `save_best(state, training_state=None, optimizer_state=None, extra_state=None)`
   - `save_last(state, training_state=None, optimizer_state=None, extra_state=None)`
   - `load_best() -> dict` / `load_last() -> dict`：返回包含 `model_state_dict` /
     `training_state` / `optimizer_state` / `extra_state` 的统一 dict（缺失字段为 None）。
3. **强制 .vn**：`use_vmpc=True` 时 `format="pt"` 被拒绝并抛 `ValueError`。
4. **默认路径**：`best.vn` / `last.vn`（format="vn"）或 `best.pt` / `last.pt`
   （format="pt"）。

#### Scenario: use_vmpc 强制 .vn
- **WHEN** `CheckpointManager(save_dir, format="pt", use_vmpc=True)` 构造
- **THEN** 抛 `ValueError`，提示"use_vmpc=True 时强制 .vn 格式"

#### Scenario: vn checkpoint 含 optimizer state
- **WHEN** `ckpt.save_best(state, optimizer_state={"m": m_arr, "v": v_arr, "step": 100})`
- **THEN** `ckpt.load_best()["optimizer_state"]["step"] == 100`，m/v 数值一致

### Requirement: VerseNex / VerseTorch 原生 .vn 适配

系统 SHALL 让 `CometSparkNexLM`（VerseNex 核心 LM）原生支持 `.vn` 格式：

1. **`CometSparkNexLM.save(path, format="vn"|"pt")`**：
   - `format="vn"`（默认）：调用 `save_vn` 路径（safetensors/npz + meta）
   - `format="pt"`：保留原 pickle 路径（向后兼容）
2. **`CometSparkNexLM.from_pretrained(path)`**：识别三种输入：
   - `.vn` 单文件：通过 `VNFileReader` 加载
   - `.pt` 单文件：原 pickle 路径
   - 目录：优先 `model.vn`，回退 `model.pt`
3. **`CometSparkSmallLM` / `CometSparkMateLM`** 的 `save_vn` 升级支持
   `training_state` / `optimizer_state` / `extra_state` 参数。

#### Scenario: CometSparkNexLM 原生 .vn
- **WHEN** 调用 `model.save("path.vn")`（不指定 format）
- **THEN** 默认 `format="vn"`，生成 `.vn` 文件，可通过 `from_pretrained` 加载

#### Scenario: from_pretrained 三路径
- **WHEN** 用 `CometSparkNexLM.from_pretrained` 加载 `.vn` / `.pt` / 目录
- **THEN** 三种输入均能正确加载模型权重与 config，无 ImportError / FileNotFoundError

### Requirement: 断点续训（ResumeManager）

系统 SHALL 提供统一的断点续训状态管理（`ResumeManager`）：

1. **`ResumeManager.save(path, model, optimizer, step, **kwargs)`**：
   - 调用 `CheckpointManager.save_best` 写 `.vn` checkpoint
   - 含 `model_state_dict` / `optimizer_state` / `step` / `rng_state` /
     `best_val_loss` / `epoch` / `patience_count`
2. **`ResumeManager.load(path) -> ResumeState`**：
   - 返回 `ResumeState` namedtuple（`model_state_dict` / `optimizer_state` /
     `step` / `rng_state` / `best_val_loss`）
   - v1 `.vn` / `.pt` 文件缺失字段返回 None（向后兼容）
3. **`ResumeManager.apply(trainer, path)`**：
   - 把 `ResumeState` 应用到 `Trainer` / `ParallelTrainerSafe` 实例
   - 恢复 model / optimizer / step / rng / best_val_loss
4. **集成**：
   - `ParallelTrainerSafe._save_resume_state` / `_load_resume_state` 改用 `ResumeManager`。
   - `spark/run.py continue` 子命令委托 `ResumeManager.apply`。

#### Scenario: 断点续训
- **WHEN** 训练到 step=500 中断，重启后调用 `ResumeManager.apply(trainer, "best.vn")`
- **THEN** model 权重 / optimizer 的 m/v / step=500 / rng_state / best_val_loss 全部恢复，
  继续训练从 step=500 开始

#### Scenario: v1 向后兼容续训
- **WHEN** 用 `ResumeManager.load` 加载 Part5K1 的 v1 `.vn` 文件
- **THEN** `optimizer_state` / `step` 等字段为 None，仅 `model_state_dict` 可用
  （用户得到明确警告："该 checkpoint 不含 optimizer state，从头初始化 optimizer"）

### Requirement: Gigatoken 集成（默认 Tokenizer）

系统 SHALL 集成 `gigatoken` 作为默认 tokenizer，**直接导入库，不复刻**：

1. **`GigaTokenizerWrapper(BaseTokenizer)`**（`verse_tokenizer/giga.py`）：
   - lazy import gigatoken：模块 import 不触发加载，仅构造时 import。
   - 内部用 `gt.Tokenizer(hf_tokenizer).as_hf()` 兼容模式（drop-in replacement），
     保证与 `VerseTokenizer` 输出一致。
   - 可选 `native=True` 走 `gt.Tokenizer(model_id)` 原生 API（更快）。
   - 实现 `encode` / `decode` / `encode_batch` / `decode_batch` / `save` / `load` /
     `__len__` / `apply_chat_template` 接口。
   - 缓存 `bos_id` / `eos_id` / `pad_id` / `vocab_size`。
2. **`load_tokenizer(kind="giga", ...)`**：
   - 优先 `GigaTokenizerWrapper`，gigatoken 不可用时**自动降级**到 `VerseTokenizer`。
3. **默认配置**：`spark/small/config/cometspark_small.yml` 与
   `spark/mate/config/cometspark_mate.yml` 的 `tokenizer.kind` 默认 `"giga"`。
4. **不删 `VerseTokenizer`**：保留向后兼容，仅在默认路径让位。
5. **不修改 gigatoken 源码 / 不重新实现 BPE/Unigram**：仅做接口适配。
6. **可选依赖**：`pyproject.toml` 的 `verse-tokenizer` extras 新增 `[giga]`：
   `gigatoken >= 0.1.0`。

#### Scenario: 默认 gigatoken
- **WHEN** 调用 `load_tokenizer(kind="giga", model_id="Qwen/Qwen3-32B")`
- **THEN** 返回 `GigaTokenizerWrapper` 实例，`encode("hello")` 与 `VerseTokenizer`
  输出一致

#### Scenario: 自动降级
- **WHEN** gigatoken 未安装，调用 `load_tokenizer(kind="giga")`
- **THEN** 自动降级到 `VerseTokenizer`，打印警告"gigatoken 未安装，降级到
  VerseTokenizer"

#### Scenario: 批量 encode 加速
- **WHEN** 用 `GigaTokenizerWrapper.encode_batch` 批量编码 10000 条文本
- **THEN** 吞吐量相对 `VerseTokenizer.encode_batch` 提升 ≥ 10×（gigatoken Rust 实现）

### Requirement: AMD ROCm 兼容

系统 SHALL 兼容 AMD GPU ROCm 生态（**不自研 kernel**）：

1. **`device.py` 设备识别**：
   - `_parse_device` 支持 `"rocm"` / `"rocm:0"`，等价于 `"cuda"`（HIP-on-ROCm
     走 PyTorch `cuda` 路径）。
   - `has_rocm() -> bool`：检测 `torch.version.hip is not None`。
   - `get_rocm_version() -> Optional[str]`：返回 ROCm 版本字符串。
2. **`backend_torch.py` TorchBackend 兼容**：
   - 接受 `"rocm"` device 字符串，内部映射到 `torch.device("cuda")`。
   - `autocast(device_type="rocm")` 等价 `device_type="cuda"`（PyTorch ROCm
     build 原生支持 fp16 autocast via HIP）。
3. **显存管理兼容**：`empty_cache` / `get_memory_info` 在 `"rocm"` 设备下走
   `torch.cuda.empty_cache()` / `torch.cuda.mem_get_info()`。
4. **CLI 集成**：`spark/run.py --device rocm` / `--device rocm:0` 可识别。
5. **不自研 ROCm kernel**：所有 GPU 计算走 PyTorch 原生（MIOpen / rocBLAS /
   FlashAttention-ROCm）。

#### Scenario: ROCm 设备识别
- **WHEN** 在 ROCm 环境（`torch.version.hip is not None`）执行 `get_backend("rocm:0")`
- **THEN** 返回 `TorchBackend(device="cuda:0")`，`device_type == "cuda"`
  （内部映射，对外暴露 "rocm" 字符串仅用于诊断）

#### Scenario: ROCm autocast
- **WHEN** 在 ROCm 环境执行 `with autocast(device="rocm", dtype=torch.float16):`
- **THEN** 启用 PyTorch ROCm fp16 autocast（HIP kernel），无 RuntimeError

### Requirement: NPU CANN 兼容增强

系统 SHALL 增强 NPU CANN 生态兼容性（**不自研 kernel**）：

1. **CANN 版本探测**：
   - `has_cann() -> bool`：检测 `torch_npu` 可用 + CANN 版本可读。
   - `get_cann_version() -> Optional[str]`：返回 CANN 版本字符串。
2. **autocast 日志**：NPU 路径首次启用 autocast 时打印 CANN 版本 + PyTorch
   版本 + torch_npu 版本（便于运维诊断）。
3. **显存管理兜底**：`get_memory_info("npu")` 增加 `torch_npu.npu.mem_get_info`
   兜底（部分 CANN 版本 API 名差异：`memory_allocated` / `mem_get_info`）。
4. **不自研 CANN kernel**：NPU 计算走 `torch_npu`（HCCL / hccl_kernel）。

#### Scenario: CANN 版本诊断
- **WHEN** 在 NPU 环境执行 `from verse_torch.device import get_cann_version; get_cann_version()`
- **THEN** 返回 CANN 版本字符串（如 `"8.0.0"`），便于运维诊断

#### Scenario: NPU autocast 日志
- **WHEN** 首次在 NPU 环境执行 `with autocast(device="npu"):`
- **THEN** 打印日志：`[autocast] NPU CANN version=8.0.0, torch=2.4.0, torch_npu=2.4.0`

## MODIFIED Requirements

### Requirement: `verse_torch.vn_format.VNFileWriter` / `VNFileReader`
原 v1 仅支持权重（state_dict）+ chat_template + tokenizer + meta。
修改为 v2：
- 新增 `write_training_state` / `write_optimizer_state` / `write_extra_state`
  及对应 read 方法。
- `meta.json` 新增 `vn_format_version: 2` + `has_training_state` /
  `has_optimizer_state` / `has_extra_state` 字段。
- `VNFileReader` 自动识别 v1/v2，v1 文件 read 新方法返回 None。
- 抽取 `VNEntry` 抽象统一管理 ZIP 条目（减少重复模板代码）。

### Requirement: `verse_torch.training.CheckpointManager`
原始终写 `best.pt` / `last.pt`。
修改为：
- `__init__` 新增 `format="auto"|"vn"|"pt"` 参数 + `use_vmpc` flag。
- `save_best` / `save_last` 扩展签名支持 `training_state` / `optimizer_state` /
  `extra_state`。
- `load_best` / `load_last` 返回统一 dict（含 `model_state_dict` / `training_state` /
  `optimizer_state` / `extra_state`，缺失字段为 None）。
- `use_vmpc=True` 时 `format="pt"` 抛 `ValueError`。
- 写入改为原子写（`.tmp` + `os.replace`）。

### Requirement: `verse_torch.vnn.Module.state_dict` / `load_state_dict`
原递归 DFS 遍历子模块。
修改为：
- 子模块遍历改为**迭代 BFS**（显式栈），消除深层 module 树下的递归链。
- 行为等价：返回的 state_dict 字典键值与原递归实现完全一致（保证 backward compat）。

### Requirement: `verse_nex.cometspark.CometSparkNexLM.save` / `from_pretrained`
原 `save(path)` 始终写 `.pt` pickle。
修改为：
- `save(path, format="vn"|"pt")`：默认 `"vn"`，`format="vn"` 走 `save_vn`，
  `format="pt"` 走原 pickle 路径。
- `from_pretrained(path)`：识别 `.vn` / `.pt` / 目录（含 `model.vn` 或
  `model.pt`）三种输入。

### Requirement: `verse_infra.verse_trainer.trainer.ParallelTrainerSafe`
原 `_save_resume_state` / `_load_resume_state` 用 `.pkl` pickle。
修改为：
- 改用 `ResumeManager` + `.vn` 格式（含 model + optimizer + step + rng）。
- 旧 `.pkl` resume 文件保留为兼容回退（读取时若 `.vn` 不存在则尝试 `.pkl`）。

### Requirement: `verse_infra.verse_tokenizer.bpe.load_tokenizer`
原支持 `kind="hf"|"bpe"|"byte"`。
修改为：
- 新增 `kind="giga"`：返回 `GigaTokenizerWrapper`，gigatoken 不可用时自动降级到
  `VerseTokenizer`。

### Requirement: `verse_torch.device._parse_device` / `get_backend`
原支持 `"cpu"|"cuda"|"mps"|"npu"`。
修改为：
- 新增 `"rocm"` / `"rocm:0"` 识别，等价于 `"cuda"`。
- 新增 `has_rocm` / `has_cann` / `get_rocm_version` / `get_cann_version` API。

### Requirement: `spark/run.py` CLI
原 `--device` 支持 `cpu|cuda|npu|mps`。
修改为：
- 新增 `--device rocm` / `--device rocm:0`。
- 启动时打印设备信息（device type + 版本 + 显存）。
- `continue` 子命令委托 `ResumeManager.apply` 完成续训状态恢复。

## REMOVED Requirements

### Requirement: `CheckpointManager` 始终写 `.pt`
**Reason**：与"`.vn` 原生支持 + `use_vmpc=True` 强制 `.vn`"宗旨冲突。
**Migration**：
- `CheckpointManager` 新增 `format` 参数，默认 `"auto"`（向后兼容，等价于原行为）。
- `use_vmpc=True` 时强制 `"vn"`，与 `CometSparkSmallLM._enforce_vn_format` 一致。
- 旧代码 `CheckpointManager(save_dir)` 不传 `format` 时，根据 `use_vmpc` 自动
  选择（use_vmpc=False → "pt"，与原行为完全一致；use_vmpc=True → "vn"）。

### Requirement: `Module.state_dict` / `load_state_dict` 递归 DFS
**Reason**：在深层 module 树下触发 `RecursionError`，阻塞并行训练。
**Migration**：
- 改为迭代 BFS 遍历（显式栈），行为等价（state_dict 键值完全一致）。
- 现有测试 `test_vnn_rename.py` / `test_training.py` 等零修改通过（断言字典相等）。
