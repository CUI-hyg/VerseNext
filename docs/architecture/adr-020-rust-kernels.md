# ADR-020: verse_torch 可选 Rust 数值内核（verse_rs）

- **状态**：Accepted
- **日期**：2026-08-16
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：Part1 性能内核化任务集（3a 基建 → 3b matmul → 3c 优化器/训练工具链 → Task2 训练模块重构）
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)（VerseTorch 是纯 Python/CPU 核心引擎）、[ADR-004 CPU 并行](adr-004-cpu-parallel.md)（multiprocessing batch 并行）
- **相关 ADR**：[ADR-016 nn→vnn 重命名](adr-016-nn-to-vnn-rename.md)（模块重构参考）、[ADR-017 resume 升级](adr-017-vn-v2-resume.md)（checkpoint 基础设施）

## 上下文

VerseTorch 坚持 CPU-first、纯 Python + NumPy、零重型运行时依赖（ADR-001）。在消费级 CPU / 嵌入式设备上，训练循环的数值热点成为吞吐瓶颈：

1. **批量矩阵乘（batched matmul）**：`parallel.py` 的 chunk 训练大量调用 `np.matmul`；`multiprocessing` 进程池有 fork / pickle / 进程创建开销，小 batch 下收益有限（ADR-004 已记录）。
2. **优化器 step**：`AdamW` / `NAdamW` / `Lion` / `RMSProp` / `SGD` 的逐参数更新是标量 Python 循环 + NumPy 临时数组，每步分配大量中间张量。
3. **训练工具链**：`log_softmax`（cross-entropy 前向）在 NumPy 下需要 5~6 次临时数组遍历；`clip_grad_norm` 的全局范数归约 + 缩放是逐参数 Python 循环。
4. **模块结构**：`training.py` 累积至 3677 行，checkpoint / 绘图逻辑与训练循环耦合，导入面过大。

## 决策

**引入可选 Rust 数值内核模块 `verse_rs`（maturin + pyo3 + rayon 构建 cdylib），覆盖训练循环 CPU 热点；Python 侧「优先走 Rust、缺失自动降级」；同时按热点归属拆分训练模块。**

具体含义：

1. **构建与产物**：
   - 内核源码按「与 Python 同名文件同目录」约定存放（`verse_torch/parallel.rs` / `optim.rs` / `training.rs`），`src/lib.rs` 通过 `#[path]` 引用，保证同名 .rs 与 .py 源码并排可读。
   - `Cargo.toml`：`pyo3 0.23`（`abi3-py38`，兼容旧 Python）、`numpy 0.23`、`rayon 1.10`（batch 维度并行分片）、`matrixmultiply 0.3`（单核 GEMM，无 BLAS 依赖）。
   - `[profile.release]` 开启 `opt-level=3` + `lto` + `codegen-units=1`。
   - 产物 `verse_rs.so` 直接提交到包内（源码树开箱即用）；未构建环境下 `from . import verse_rs` 失败，Python 侧自动降级。

2. **接入模式（自动降级）**：
   - 各 Python 模块模块级 `try: from . import verse_rs except ImportError: verse_rs = None`。
   - 运行时先检查内核可用性再检查输入条件：**float32 + C 连续内存**（Rust 侧 `to_f32_vec` 处理非连续，但首选连续切片零拷贝）才走 Rust；否则降级 NumPy 原路径。
   - 任何 Rust 异常都 try/except 兜底降级，保证数值结果与 NumPy 参考一致（测试对拍 `atol=1e-5`）。

3. **内核清单与接入点**：

   | 内核（verse_rs 导出） | 对应 Python 接入 | 说明 |
   |---|---|---|
   | `batched_matmul` | `parallel.py` 批量 matmul 路径 | 支持 (B,M,K)x(K,N) / (B,M,K)x(B,K,N) / (M,K)x(K,N)，rayon 按 batch 分片，替代 multiprocessing 进程池 |
   | `default_threads` | `parallel.py` `_default_n_workers()` | CPU 核数一半（至少 1），与 Python 语义一致 |
   | `adam_step` / `nadamw_step` / `lion_step` / `rmsprop_step` / `sgd_step` | `optim.py` / `optim_extras.py` 对应优化器 step | 单次遍历更新参数 + 动量 + 解耦 weight decay；float32 时走 Rust |
   | `log_softmax_forward` / `log_softmax_backward` | `tensor.py` `Tensor.log_softmax`、`losses.py` `cross_entropy` | 沿最后一维单遍遍历（数值稳定：max 减缩），替代 NumPy 多趟临时数组 |
   | `grad_norm` / `scale_grads` | `training.py` `clip_grad_norm` | 全局 L2 范数归约 + 裁剪缩放，替代逐参数 Python 循环 |

4. **模块拆分（Task2）**：
   - `training.py`（3677 行）拆出 `checkpoint.py`（CheckpointManager / ResumeManager / ResumeState）与 `plotting.py`（compute_loss_rate / plot_loss_curve / ASCII 降级），`training.py` 顶部 re-export 全部符号，`from verse_torch.training import ...` 完全兼容。
   - 缩小导入面、消除循环导入风险；持久化与绘图逻辑按职责独立成模块。

## 备选方案

1. **整体重写为 Rust**：破坏「纯 Python 可读可改」的工程约定，放弃。
2. **Cython / numba**：引入编译期依赖，且非 Python 标准生态；pyo3 与项目现有工具链更贴合。
3. **继续纯 NumPy + multiprocessing**：保留，作为无 Rust 工具链环境下的默认路径（自动降级已保证）。

## 后果

**正面**：
- 训练循环热点获得单核 GEMM + 多线程 batch 分片加速（无 BLAS 依赖，嵌入式友好）。
- 优化器 step 与 log_softmax 减少中间分配与 Python 循环。
- 自动降级保证：无 Rust 工具链 / 非 float32 / 非连续内存时行为与纯 NumPy 完全一致（测试对拍）。
- 模块拆分后训练基础设施可独立演进、测试与复用。

**负面 / 成本**：
- 引入 Rust 工具链（cargo / maturin）作为可选构建依赖；`verse_rs.so` 需随平台重新构建（当前提交 x86_64 Linux 产物）。
- 内核仅覆盖 float32 CPU 路径，float64 / GPU（TorchBackend 委托 PyTorch）不走 Rust。
- 多一个二进制产物进入版本库（约 0.8MB）。

**兼容性**：非破坏。Python API 与导入路径不变；`verse_rs` 缺失时行为与拆分/内核化之前完全一致（拆分前基线测试复现确认）。
