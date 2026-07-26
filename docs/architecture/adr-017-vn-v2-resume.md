# ADR-017: .vn v2 格式 + 复杂 Python 对象 + 断点续训

- **状态**：Accepted
- **日期**：2026-07-25
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part5k1.3-bugfix-stability-upgrade/spec.md`](../../../.trae/specs/part5k1.3-bugfix-stability-upgrade/spec.md)
- **前置 ADR**：[ADR-009 .vn 文件格式](adr-009-vn-format.md)（v1 设计）、[ADR-005 GPU/NPU 后端](adr-005-gpu-npu-backend.md)（checkpoint 跨设备恢复）
- **相关 ADR**：[ADR-015 VMT 完整三档策略](adr-015-vmt-full-strategy.md)（VMT optimize 档需要 checkpoint 续训）、[ADR-016 nn → vnn 重命名](adr-016-nn-to-vnn-rename.md)（vnn.Module.state_dict 迭代 BFS 配合本 ADR 的递归修复）

## 上下文

ADR-009 定义的 `.vn` v1 格式已稳定承载模型权重（`model.safetensors` / `model.npz`）+ `config.yml` + `chat_template.jinja` + `tokenizer.json` + `meta.json`，并通过 `VNFileWriter` / `VNFileReader` / `pt_to_vn` / `vn_to_pt` / `convert_format` 提供 mmap 零拷贝与无损互转能力。但 Part5K1.3 在落地"自研原生架构稳定可用"承诺时，暴露出 v1 的 3 个阻塞性短板：

1. **无法承载训练状态**：v1 仅存权重（state_dict），训练步数（step）/ 训练轮数（epoch）/ 最佳验证 loss（best_val_loss）/ EarlyStopping patience 计数等"训练元信息"无处可写，断点续训时只能从 step=0 重新开始，浪费已训练算力。
2. **无法承载优化器状态**：AdamW 的一阶动量 `exp_avg` / 二阶动量 `exp_avg_sq` / `step`、lr scheduler 的 `base_lru` / `last_epoch` / `_step_count`、用户自定义 optimizer 的任意 state dict 均无法持久化。断点续训若不恢复 optimizer state，模型权重虽对但动量归零、学习率调度错位，导致 loss 突然飙升、训练发散。
3. **无法承载 RNG 状态**：numpy `RandomState.get_state()` / Python `random.getstate()` / PyTorch `torch.get_rng_state()` 等 RNG 状态在 v1 中无存储位置，断点续训后 RNG 重置，dataloader shuffle 顺序、dropout mask 与中断前不可复现，"训练中断 → 续训结果与未中断不一致"成为必然。
4. **CheckpointManager 仍是 `.pt` 壳**：即便 `use_vmpc=True`、`CometSparkSmallLM.save(format="vn")`，训练过程中 `CheckpointManager.save_best` / `save_last` 仍始终写 `best.pt` / `last.pt`，`.vn` 仅在显式 `model.save()` 时被使用——`.vn` 在训练链路里是个"装饰性格式"，并未真正承载训练产物。
5. **断点续训缺统一管理**：`ParallelTrainerSafe._save_resume_state` / `_load_resume_state` 用裸 `.pkl` pickle 写一个 dict，无版本管理、无原子写、无 v1 兼容；`spark/run.py continue` 子命令也只是粗粒度 reload，无 step / optimizer / rng 精细恢复。

同时必须保持 ADR-009 的核心承诺：**v1 文件可被新 reader 完整加载**（向后兼容，Part5K1 写出的 `.vn` checkpoint 不失效）。

## 决策

**把 `.vn` 格式升级到 v2（`VN_FORMAT_VERSION = 2`），新增 `training_state.json` / `optimizer_state.pkl` / `extra_state.pkl` 三个可选 ZIP 条目承载任意 Python 对象；`VNFileReader` 自动识别 v1/v2；`CheckpointManager` 增加 `format="vn"|"pt"|"auto"` 参数原生写 `.vn`；新增 `ResumeManager` 统一管理断点续训状态。**

具体含义：

1. **v2 ZIP 容器结构（向后兼容 v1）**：

   ```
   model.vn (ZIP, v2)
   ├── model.safetensors         # 权重（safetensors 可用时，mmap 零拷贝）
   ├── model.npz                 # 权重（safetensors 不可用时降级，allow_pickle=False）
   ├── config.yml                # 模型配置
   ├── chat_template.jinja       # 聊天模板（可选）
   ├── tokenizer.json            # tokenizer（可选）
   ├── meta.json                 # 元数据（含 vn_format_version / has_* 字段）
   ├── training_state.json       # v2 新增：训练元信息（JSON-able dict）
   ├── optimizer_state.pkl       # v2 新增：optimizer 状态（pickle，任意 Python 对象）
   └── extra_state.pkl           # v2 新增：用户自定义任意对象（pickle）
   ```

   v1 文件不含后三个条目，读取时按缺失处理。

2. **`meta.json` v2 字段扩展**：

   ```json
   {
       "vn_format_version": 2,
       "arch": "versenex",
       "weight_format": "safetensors" | "npz",
       "compression_info": {...} | null,
       "created_at": "ISO8601 时间戳",
       "weight_count": 12,
       "has_training_state": true,     // v2 新增，默认 false
       "has_optimizer_state": true,    // v2 新增，默认 false
       "has_extra_state": false        // v2 新增，默认 false
   }
   ```

   `VNFileReader` 基于 `vn_format_version` 字段自动识别 v1/v2（v1 文件无此字段默认 1）；v1 文件读取 `has_*` 默认 False，`read_training_state` / `read_optimizer_state` / `read_extra_state` 返回 `None`。

3. **`training_state.json`（JSON-able dict）**：

   训练元信息，由 `VNFileWriter.write_training_state(state: dict)` 写入。承载：

   - `step` / `epoch`：训练步数与轮数
   - `best_val_loss` / `patience_count`：EarlyStopping 状态
   - `rng_state_hex`：numpy `RandomState.get_state()` 序列化为 hex 字符串（避免 JSON 无法直接承载 tuple-of-ndarray）
   - `config_snapshot_hash`：训练配置的 hash（用于校验续训配置一致性）
   - 任何其他 JSON-able 字段（用户自定义）

   用 JSON 而非 pickle 的原因：训练元信息天然 JSON-able，JSON 可读 + 跨语言 + 无 RCE 风险，便于运维直接 `unzip -p model.vn training_state.json` 检查训练进度。

4. **`optimizer_state.pkl`（pickle，任意 Python 对象）**：

   optimizer 状态，由 `VNFileWriter.write_optimizer_state(state: dict)` 写入。承载：

   - AdamW 的 `exp_avg` / `exp_avg_sq` / `step` 矩阵
   - lr scheduler 的 `base_lru` / `last_epoch` / `_step_count`
   - 用户自定义 optimizer 的任意 state dict
   - grad scaler state（混合精度训练的 `_scale` / `_growth_tracker`）

   用 pickle 而非 JSON 的原因：optimizer state 含 `ndarray` / `Tensor` / 嵌套 dict，JSON 序列化会丢失 dtype 与 shape 信息，且需要手写 ndarray ↔ list 转换；pickle 是 PyTorch `torch.save` 的同等等价物，与 `torch.optim.AdamW.state_dict()` 的输出格式天然兼容。

5. **`extra_state.pkl`（pickle，任意 Python 对象）**：

   用户自定义任意对象，由 `VNFileWriter.write_extra_state(state: Any)` 写入。承载：

   - EMA（Exponential Moving Average）state
   - grad scaler state（fp16 训练）
   - callback state（如 `LossHistory` / `LossOptimizer` 的内部状态）
   - `best_state_dict`（ParallelTrainer 的最佳模型备份，避免与 `model.safetensors` 冲突）

6. **`VNFileWriter` / `VNFileReader` API 扩展**：

   - `VNFileWriter.write_training_state(state: dict) -> None`
   - `VNFileWriter.write_optimizer_state(state: dict) -> None`
   - `VNFileWriter.write_extra_state(state: Any) -> None`
   - `VNFileReader.read_training_state() -> Optional[dict]`（v1 文件返回 None）
   - `VNFileReader.read_optimizer_state() -> Optional[dict]`（v1 文件返回 None）
   - `VNFileReader.read_extra_state() -> Optional[Any]`（v1 文件返回 None）

   `VNFileReader.close` 时更新 `meta.json` 的 `has_training_state` / `has_optimizer_state` / `has_extra_state` 布尔字段。

7. **`VNEntry` 抽象（核心架构优化）**：

   抽取 `VNEntry` 抽象统一管理 ZIP 内条目的写入/读取/校验，减少 `VNFileWriter` / `VNFileReader` 重复的 `writestr` / `open` 模板代码。每个 entry 封装：条目名、序列化方式（JSON / pickle / safetensors / npz）、写入/读取方法、缺失处理（v1 文件返回 None）。

8. **`CheckpointManager` 原生 `.vn` 支持**：

   - `__init__` 新增 `format: str = "auto"` 参数：
     - `"auto"`：根据 `use_vmpc` flag 自动选择（`use_vmpc=True → "vn"`，否则 `"pt"`）
     - `"vn"`：`save_best` / `save_last` 写 `.vn` 文件（含 model + training_state + optimizer_state + extra_state）
     - `"pt"`：保留现有 `.pt` 行为（向后兼容）
   - `save_best(state, training_state=None, optimizer_state=None, extra_state=None)`：扩展签名
   - `load_best() -> dict` / `load_last() -> dict`：返回统一 dict（含 `model_state_dict` / `training_state` / `optimizer_state` / `extra_state`，缺失字段为 None）
   - **强制 `.vn`**：`use_vmpc=True` 时 `format="pt"` 抛 `ValueError`（与 `CometSparkSmallLM._enforce_vn_format` 一致）
   - **原子写**：先写 `best.pt.tmp` / `best.vn.tmp`，再 `os.replace` 重命名（POSIX 原子语义；Windows `os.replace` 同样原子）；写入失败时清理 `.tmp`

9. **`ResumeManager` 统一断点续训**：

   新增 `verse_torch/training.py.ResumeManager`，统一管理断点续训状态：

   - `ResumeState` namedtuple：`model_state_dict` / `optimizer_state` / `step` / `rng_state` / `best_val_loss` / `epoch` / `patience_count`
   - `save(path, model, optimizer, step, **kwargs)`：调用 `CheckpointManager.save_best` 写 `.vn` checkpoint
   - `load(path) -> ResumeState`：调用 `CheckpointManager.load_best` 读取；v1 `.vn` / `.pt` 文件缺失字段返回 None + 警告日志
   - `apply(trainer, path)`：把 `ResumeState` 应用到 `Trainer` / `ParallelTrainerSafe` 实例（恢复 model / optimizer / step / rng / best_val_loss）

10. **集成点**：

    - `ParallelTrainerSafe._save_resume_state` / `_load_resume_state`：改用 `ResumeManager` + `.vn`；旧 `.pkl` resume 文件保留为兼容回退（`.vn` 不存在时尝试 `.pkl`）
    - `spark/run.py continue` 子命令：委托 `ResumeManager.apply` 完成续训状态恢复
    - `Trainer.__init__`：实例化 `CheckpointManager` 时传入 `format="auto"` + `use_vmpc=cfg.get("use_vmpc", False)`

## 后果

### 优点

- **真正的断点续训**：v2 checkpoint 完整保存 model + optimizer + step + rng + best_val_loss + epoch + patience，续训后训练曲线与未中断训练不可区分（rng 一致 + 动量延续 + 学习率调度对齐）。
- **v1 完全向后兼容**：Part5K1 写出的 v1 `.vn` 文件被 v2 `VNFileReader` 完整加载（`read_weights` 正常返回权重，`read_training_state` 等新方法返回 None + 警告日志），无需迁移。
- **`.vn` 在训练链路落地**：`CheckpointManager` 默认 `format="auto"` + `use_vmpc=True` → `.vn`，训练产物不再是 `.pt` 壳；与 ADR-009 的"`.vn` 为推荐交付格式"承诺对齐。
- **安全性分级**：训练元信息（`training_state.json`）走 JSON（pickle-free，可读可审计），仅 optimizer / extra state 走 pickle（与 `torch.save` 同等信任级别）；npz 权重路径仍 `allow_pickle=False`。
- **原子写防损坏**：训练中断（SIGKILL / 断电）时 `best.vn` 要么是上次完整 checkpoint，要么不存在，不会写出半截文件导致下次启动无法加载。
- **统一续训入口**：`ResumeManager` 把"序列化 / 反序列化 / 应用到 trainer"三步封装，`ParallelTrainerSafe` 与 `spark/run.py continue` 共用同一逻辑，避免两处分别实现导致行为分歧。
- **VNEntry 抽象降复杂度**：减少 `VNFileWriter` / `VNFileReader` 重复的 `writestr` / `open` 模板代码，新增条目只需声明 entry 名 + 序列化方式。

### 缺点

- **v2 文件含 pickle**：`optimizer_state.pkl` / `extra_state.pkl` 使用 pickle，加载不可信 `.vn` 文件存在 RCE 风险（与 `torch.load` 同等级别）。文档需明确"仅用于可信来源的断点续训"。
- **v2 文件体积略增**：相比 v1 多出 `training_state.json`（KB 级）+ `optimizer_state.pkl`（与 model 同量级，AdamW 的 m/v 矩阵 = 2× model 大小）+ `extra_state.pkl`（用户自定义）。`optimizer_state.pkl` 是大头，但断点续训必需。
- **v1 文件续训降级**：v1 `.vn` 续训时 optimizer state / step / rng 全部为 None，`ResumeManager.apply` 仅能恢复 model 权重，optimizer 从头初始化（打印警告日志）；用户需明确知晓。
- **API 表面扩展**：`CheckpointManager.__init__` 新增 `format` / `use_vmpc` 参数，`save_best` / `save_last` 签名扩展；旧调用（不传新参数）行为完全一致（默认 `format="auto"` + `use_vmpc=False` → `"pt"`），但新参数文档需要清晰说明。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| v2 pickle RCE 风险 | `optimizer_state.pkl` / `extra_state.pkl` 仅用于可信来源（用户自己训练的 checkpoint）；文档明确警告；npz 权重路径仍 `allow_pickle=False` 不可被绕过 |
| v1 文件续训时 optimizer 重置导致 loss 飙升 | `ResumeManager.load` 检测到 `optimizer_state is None` 时打印警告日志："该 checkpoint 不含 optimizer state，从头初始化 optimizer"；用户得到明确信号 |
| `use_vmpc=True` + `format="pt"` 误用 | `CheckpointManager.__init__` 显式抛 `ValueError`，错误信息明确"use_vmpc=True 时强制 .vn 格式" |
| 原子写在 Windows 上不原子 | `os.replace` 在 Windows 上覆盖目标文件为原子操作（Python 3.3+ 保证）；测试 `test_checkpoint_atomic.py` 覆盖 Linux + Windows 路径 |
| `ResumeManager.apply` 与不同 trainer 类型不兼容 | `apply` 检测 trainer 类型（`Trainer` / `ParallelTrainerSafe`），分别走对应恢复路径；不支持类型抛 `TypeError` 明确报错 |
| `VNEntry` 抽象引入间接层增加阅读成本 | `VNEntry` 仅是写入/读取模板的封装，无业务逻辑；注释明确每个 entry 的序列化方式与缺失处理 |

## 替代方案（已否决）

### 方案 A：仅扩展 v1（在 `meta.json` 里塞 optimizer state）

**描述**：不升级 `VN_FORMAT_VERSION`，把 optimizer state / training_state 序列化为 JSON 字符串塞到 `meta.json` 的扩展字段。

**否决理由**：
- `meta.json` 应保持轻量自描述（架构 / 权重格式 / 创建时间），塞入 KB-MB 级 optimizer state 破坏可读性
- JSON 无法承载 `ndarray` / `Tensor`，需要手写转换层
- 无版本号无法区分"含 optimizer state 的 v1"与"纯 v1"，向后兼容检测复杂
- 与 ADR-009 "v1 仅权重"的设计意图冲突

### 方案 B：不向后兼容（v2 直接破坏 v1）

**描述**：v2 不识别 v1 文件，强制用户用 `verse-convert` 迁移。

**否决理由**：
- Part5K1 已有大量 v1 `.vn` checkpoint 在用户手中（`mf_small/` / `mf_mate/` 训练产物），破坏加载会导致用户训练成果丢失
- 违反 ADR-009 "未来 v2 可向后兼容 v1" 的承诺
- 与"自研原生架构稳定可用"宗旨冲突

### 方案 C：用 safetensors 扩展承载 optimizer state

**描述**：把 optimizer 的 m/v 矩阵也存为 safetensors 文件（`optimizer_state.safetensors`），仅训练元信息用 JSON。

**否决理由**：
- safetensors 仅支持 flat dict[str, tensor]，optimizer state 是嵌套 dict（`{param_id: {"exp_avg": tensor, "exp_avg_sq": tensor, "step": int}}`），需要额外拍平 / 还原逻辑
- scheduler state / grad scaler state 等非 tensor 对象无法用 safetensors 承载
- 增加格式复杂度，与 PyTorch `torch.save` 的 pickle 路径不互操作
- 安全收益有限（npz 路径已 `allow_pickle=False`，pickle 仅用于明确信任的 optimizer / extra state）

### 方案 D：`CheckpointManager` 始终写 `.pt`，仅 `model.save()` 写 `.vn`

**描述**：保持 Part5K1 现状，训练过程始终 `.pt`，仅显式 `save(format="vn")` 时写 `.vn`。

**否决理由**：
- `.vn` 在训练链路里仍是"装饰性格式"，违背 ADR-009 "`.vn` 为推荐交付格式"的承诺
- `use_vmpc=True` 强制 `.vn` 时，训练过程写 `.pt` 与最终交付 `.vn` 不一致，需要二次转换
- 断点续训需要 `ResumeManager` 直接读训练产物，若产物是 `.pt` 则 v2 能力无处施展

### 方案 E：自研二进制 optimizer 序列化格式

**描述**：定义自研二进制格式（magic + header + tensor table）承载 optimizer state，避免 pickle RCE。

**否决理由**：
- 维护成本极高（每种 optimizer state 结构都要单独处理）
- 失去与 PyTorch `torch.optim.AdamW.state_dict()` 输出的天然兼容
- 违反 ADR-001 "不重新发明底层工具"原则
- 与 ADR-009 "不重新实现 pickle 序列化"约束冲突

## 备注

- 本 ADR 是 ADR-009 "`.vn` 格式演进路线"的具体落地，`VN_FORMAT_VERSION` 从 1 升到 2，保持 v1 向后兼容
- v2 的 `optimizer_state.pkl` / `extra_state.pkl` **仅用于可信来源**（用户自己训练的 checkpoint），与 PyTorch `torch.save` 同等信任级别；加载第三方 `.vn` 文件时应仅信任 `model.safetensors` / `model.npz`（pickle-free 路径）
- 相关测试：`tests/test_vn_v2_format.py` 覆盖 v2 写入 + 读取 + v1 向后兼容 + optimizer m/v 数值一致（float32 吻合 1e-7）；`tests/test_checkpoint_vn.py` 覆盖 vn checkpoint 含 optimizer state + 训练中断后 `load_best` 恢复 + `use_vmpc` 强制 `.vn`；`tests/test_checkpoint_atomic.py` 覆盖原子写 + 中断不损坏；`tests/test_resume_manager.py` 覆盖断点续训 + v1 向后兼容 + optimizer state 恢复 + rng_state 恢复
- 相关代码：
  - [`verse_torch/vn_format.py`](../../packages/verse_torch/verse_torch/vn_format.py) —— `VN_FORMAT_VERSION=2` + `VNFileWriter` / `VNFileReader` 新增 `write_training_state` / `write_optimizer_state` / `write_extra_state` / 对应 read 方法 + `VNEntry` 抽象
  - [`verse_torch/training.py`](../../packages/verse_torch/verse_torch/training.py) —— `CheckpointManager` 支持 `format="vn"|"pt"|"auto"` + 原子写 + `ResumeManager` + `ResumeState`
  - [`verse_nex/cometspark.py`](../../packages/verse_nex/verse_nex/cometspark.py) —— `CometSparkNexLM.save(path, format="vn"|"pt")` + `from_pretrained` 三路径
  - [`verse_infra/verse_trainer/trainer.py`](../../packages/verse_infra/verse_infra/verse_trainer/trainer.py) —— `ParallelTrainerSafe._save_resume_state` / `_load_resume_state` 改用 `ResumeManager`
  - [`spark/run.py`](../../spark/run.py) —— `continue` 子命令委托 `ResumeManager.apply`

## 演进路线

- **v3（远期）**：若未来需要承载分布式训练状态（DDP rank 0 的 optimizer state + 各 rank 的 RNG state），可新增 `distributed_state.pkl` 条目 + `meta.json` 的 `has_distributed_state` 字段，v3 reader 自动识别 v1/v2/v3
- **`VNEntry` 抽象扩展**：未来新增条目（如 `tokenizer_config.json` / `metrics.json`）只需声明 entry 名 + 序列化方式，无需修改 `VNFileWriter` / `VNFileReader` 主体逻辑
