# Changelog

本文件记录 VerseNext 仓库各阶段的可观测变更（用户视角）。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，日期遵循版本发布时间。

## [Unreleased] — feat/rust-part1（Part1 性能内核化 + Part6 训练优化）

### Added

- **VerseTorch 可选 Rust 数值内核（`verse_rs`）**，覆盖训练循环 CPU 热点，Python 侧自动降级（详见 [ADR-020](docs/architecture/adr-020-rust-kernels.md)）：
  - `batched_matmul`：批量矩阵乘（rayon 按 batch 分片，替代 multiprocessing 进程池），接入 `parallel.py`
  - 优化器 step 内核：`adam_step` / `nadamw_step` / `lion_step` / `rmsprop_step` / `sgd_step`，接入 `optim.py` / `optim_extras.py`
  - 训练工具链内核：`log_softmax_forward/backward`（接入 `Tensor.log_softmax` 与 `cross_entropy`）、`grad_norm` / `scale_grads`（接入 `clip_grad_norm`）
  - 内核源码与同名 Python 文件同目录存放（`verse_torch/parallel.rs` 等），`src/lib.rs` 经 `#[path]` 引用
- **`checkpoint.py` / `plotting.py` 模块**：`training.py` 按职责拆分（3677 → 2281 行），`from verse_torch.training import ...` 保持完全兼容
- **训练优化（Part6）**：VerseNext Delta Attention、tokenizer.json 前置校验、prefetch 后台线程预取等
- **GigaToken 集成（Part6）**：支持直接加载标准 HuggingFace `tokenizer.json`

### Changed

- `verse_torch.nn` 与 `verse_torch.vnn` 合并为 `verse_torch.vnnn`（BREAKING：导入路径更新为 `verse_torch.vnnn`）
- 顶层 shim 包（`verse_tokenizer` / `verse_inference` / `verse_compat` / `verse_trainer`）移除，统一从 `verse_infra` 导入（BREAKING）
- `checkpoint.py` 子进程转换 worker 入口指向 `verse_torch.checkpoint`

### Fixed

- 全量测试回归中发现的既有问题经基线（`git stash` 对比 HEAD）确认与本次变更无关的项未在本分支修复（如 verse_infra tokenizer 校验与 `tests/test_trainer_tqdm_continue.py` 的兼容、`test_scoring.py` 缺少 `sys.path` 注入等，留待后续修复分支）

### 已知说明

- `verse_rs.so` 当前提交为 x86_64 Linux 构建产物；其他平台需在本仓库根目录用 cargo/maturin 重新构建，未构建时自动降级为纯 NumPy 路径
- 沙箱内存受限环境（如 4GB）下，同一进程内连续多次调用 verse_infra `train()` 可能触发 OOM 保护进程退出（与内核化无关，属资源限制）

## 早期阶段（main 分支基线）

main 分支此前阶段包含：VerseNext 框架主体（VerseTorch / VerseNex / VerseAWM / VerseInfra / CometSpark）、Part4/Part5 各阶段能力（线性复杂度架构、VerseNex 原生架构、量化压缩、GPU/NPU 后端、双模型 small/mate、VN v2 格式与断点续训、GigaToken 集成等），详见各 [ADR](docs/architecture/) 与包内 README。
