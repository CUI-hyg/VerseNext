# Verse 性能调优指南

> 本文档面向希望榨干硬件性能的用户，覆盖 numba JIT 加速、BLAS 配置、batch_size 选择、CPU 线程数、量化加速、并行计算、**GPU/NPU 加速（Part4K1 新增）**、**混合精度 autocast（Part4K1 新增）**、**CachedDataset 数据加载加速（Part4K1 新增）** 九个维度。Verse 默认纯 CPU / 纯 NumPy（无需重新编译框架），Part4K1 起可选启用 GPU/NPU 加速（通过 PyTorch 委托后端）。

---

## 1. numba 可选 JIT 加速

### 1.1 安装

VerseNex 的 selective scan（Mamba-2 / RWKV-7 / Hybrid）热点函数已用 `@njit` 装饰，但 numba 是**可选依赖**——不安装也能运行，只是 @njit 退化为 no-op，函数按普通 NumPy 执行。

```bash
# 方式一：通过 verse-nex 的 speed extra 安装（推荐）
pip install "verse-nex[speed]"

# 方式二：直接安装 numba
pip install "numba>=0.60"
```

### 1.2 加速范围

`@njit` 装饰的热点函数（位于 `packages/verse_nex/verse_nex/mamba2.py`）：

| 函数 | 位置 | 加速点 |
|---|---|---|
| `_softplus_np` | mamba2.py | A_log 参数化约束的标量 softplus，每次 forward 都调用 |
| `_conv1d_step` | mamba2.py | 推理时单步 causal depthwise conv1d |
| `_ssm_recurrent_step_kernel` | mamba2.py | **核心热点**：selective scan 按 head 维度的递推循环，numba 将 Python 循环编译为机器码 |

实测收益：在 `n_heads=16, d_state=128, d_head=64` 配置下，recurrent 模式生成 512 tokens 的吞吐量提升约 1.8× ~ 3.2×（视 CPU 与 numba 版本而定）。第一次调用有 JIT 编译开销（约 1~3 秒），`cache=True` 后续进程启动直接加载缓存。

### 1.3 验证 numba 是否生效

```python
from verse_nex.mamba2 import _HAS_NUMBA
print(f"numba enabled: {_HAS_NUMBA}")  # True 表示已启用
```

### 1.4 注意事项

- numba 的 `@njit` 要求函数内仅用 NumPy 操作，不能引用 `verse_torch.Tensor` 对象。Verse 已将热点循环提取为纯 NumPy kernel（`_ssm_recurrent_step_kernel`），调用方负责 `astype(np.float64)`。
- 无 numba 时功能与有 numba 时完全一致，仅速度差异。所有单元测试在两种环境下均通过。
- parallel 路径（训练用）不依赖 numba，已用 `np.cumsum` + broadcasting 向量化，BLAS 加速。

---

## 2. BLAS 配置建议

NumPy 的矩阵乘法底层走 BLAS。不同 BLAS 实现性能差异可达 2× ~ 5×。

### 2.1 推荐实现

| BLAS 实现 | 适用场景 | 安装方式 |
|---|---|---|
| **OpenBLAS** | 通用推荐，跨平台稳定 | `pip install numpy`（默认即 OpenBLAS） |
| **Intel MKL** | Intel CPU 上最优，AVX-512 加速 | `pip install mkl numpy` 或 conda 安装 |
| **BLIS** | AMD CPU 上有优化 | 编译安装 |

### 2.2 检查当前 BLAS

```python
import numpy as np
np.show_config()
```

输出中查找 `blas_mkl_info` / `openblas64_get_info` / `blis_info` 字段确认后端。

### 2.3 升级 BLAS

```bash
# 用 conda 切换到 MKL
conda install numpy mkl

# 或用 pip 安装 OpenBLAS 版本
pip install numpy --upgrade --force-reinstall
```

---

## 3. batch_size 选择建议

### 3.1 训练场景

| 模型规模 | 推荐 batch_size | 说明 |
|---|---|---|
| < 1M 参数 | 32 ~ 64 | 小模型 batch 大易过拟合，配合 dropout |
| 1M ~ 10M 参数 | 16 ~ 32 | 平衡吞吐与显存（CPU 下为内存） |
| > 10M 参数 | 4 ~ 16 | 配合梯度累积 `GradientAccumulator` 模拟大 batch |

### 3.2 推理场景

- **流式生成**：`batch_size=1` 即可，recurrent 模式常数内存。
- **批量推理**：`batch_size=8 ~ 32`，parallel 模式利用 BLAS 批量 GEMM。

### 3.3 内存估算

```
batch_mem ≈ batch_size × seq_len × n_embd × 4_bytes × n_layers × factor
```

其中 `factor` 取 8~16（激活值 + 中间张量 + autograd 图）。CPU 5GB 约束下建议 `batch_size × seq_len × n_embd ≤ 2_000_000`。

---

## 4. CPU 线程数配置

NumPy / BLAS 默认使用全部 CPU 核心，但**多线程在以下场景反而变慢**：

- batch_size=1 的流式推理（线程切换开销 > 计算）
- 已用 `parallel_matmul` 的 multiprocessing 并行（避免线程 + 进程双重并行）

### 4.1 环境变量

```bash
# 限制 OpenBLAS 线程数（推理时建议 1~4）
export OPENBLAS_NUM_THREADS=4

# 限制 MKL 线程数
export MKL_NUM_THREADS=4

# 限制 OpenMP 线程数（numba 也用此变量）
export OMP_NUM_THREADS=4

# Python 进程总线程数上限
export NUMEXPR_NUM_THREADS=4
```

### 4.2 推荐配置

| 场景 | 线程数 | 说明 |
|---|---|---|
| 训练（batch ≥ 16） | 全部核心 | BLAS 批量 GEMM 受益于多线程 |
| 流式推理（batch=1） | 1 ~ 4 | 避免线程切换开销 |
| 已用 `parallel_matmul` | 1 | multiprocessing 已并行，BLAS 单线程避免争抢 |

### 4.3 在 Python 中动态设置

```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 必须在 import numpy 之前设置
import numpy as np
```

---

## 5. 量化加速

`verse_torch.quantize` 提供三种量化方案，均能在 CPU 上显著降低内存与提升推理吞吐。

### 5.1 方案对比

| 方案 | 比特数 | 内存压缩 | 推理加速 | 适用场景 |
|---|---|---|---|---|
| **INT8** | 8-bit | 4× | 1.5× ~ 2× | 通用，精度损失极小 |
| **INT4 (W4A16)** | 4-bit | 8× | 2× ~ 3× | 端侧 LLM，权重大于激活 |
| **1.58-bit ternary** | ≈1.58-bit | 20× | 3× ~ 6× | BitNet 风格，需训练时量化 |

### 5.2 使用示例

```python
from verse_torch.quantize import quantize_int8, dequantize_int8, QuantizedLinear

# 方式一：量化权重张量
q, scale = quantize_int8(linear.weight.data)
w_dequant = dequantize_int8(q, scale)  # 反量化用于推理

# 方式二：直接替换 Linear 为 QuantizedLinear
qlin = QuantizedLinear.from_linear(linear, bits=8)
out = qlin(x)  # 内部走 INT8 GEMM 路径
```

### 5.3 训练时量化（ternary）

参考 BitNet 训练范式，在训练全程保持权重为三值 {-1, 0, +1}：

```python
from verse_torch.quantize import quantize_ternary, matmul_ternary

# 每个 optimizer step 后量化
q, scale = quantize_ternary(linear.weight.data)
linear.weight.data = dequantize_ternary(q, scale)

# 推理时用专用 matmul
out = matmul_ternary(x, q, scale)
```

详细基准见 [量化基准](benchmarks/quantize_benchmark.md)。

---

## 6. 并行计算

`verse_torch.parallel` 提供 multiprocessing 版本的批量矩阵乘法与线性层，绕过 GIL。

### 6.1 API

| 函数 / 类 | 作用 |
|---|---|
| `parallel_matmul(A, B, n_workers=None)` | 跨进程切分 batch 维度的矩阵乘法 |
| `ParallelLinear(in_features, out_features)` | 替换 `nn.Linear`，对大 batch 自动启用并行 |
| `parallel_map(fn, iterable, n_workers=None)` | 跨进程 map，用于数据预处理 |

### 6.2 使用示例

```python
from verse_torch.parallel import parallel_matmul, ParallelLinear

# 大 batch 矩阵乘法
A = np.random.randn(1024, 768).astype(np.float32)
B = np.random.randn(768, 768).astype(np.float32)
C = parallel_matmul(A, B, n_workers=4)  # 4 进程并行

# 替换模型中的 Linear
model.blocks[0].attn.proj = ParallelLinear(768, 768)
```

### 6.3 何时启用

| batch_size | 建议 |
|---|---|
| < 64 | 不启用，进程创建开销 > 收益 |
| 64 ~ 256 | 启用 2~4 进程 |
| > 256 | 启用全部 CPU 核心数 |

启用并行时务必将 `OPENBLAS_NUM_THREADS=1`，避免线程 + 进程双重并行争抢 CPU。

详细基准见 [CPU 并行基准](benchmarks/parallel_benchmark.md)。

---

## 7. GPU/NPU 加速（Part4K1）

Part4K1 引入 `DeviceBackend` 抽象层（[`verse_torch.device`](../packages/verse_torch/verse_torch/device.py)），CPU-first 不变，PyTorch 仅作为可选加速后端。GPU 路径委托 `torch` 原生算子（含 `F.scaled_dot_product_attention` fused kernel），**不自研 CUDA kernel**；NPU 路径通过 `torch_npu` 扩展支持。详见 [ADR-005](architecture/adr-005-gpu-npu-backend.md)。

### 7.1 启用 GPU/NPU

```bash
# 安装可选依赖
pip install torch                  # CUDA / MPS
pip install torch_npu              # 华为昇腾 NPU

# CLI 启用
verse-train --config spark/config/cometspark_v05.yml --device cuda    # GPU
verse-train --config spark/config/cometspark_v05.yml --device npu     # NPU
```

### 7.2 Python API

```python
from spark.model.model import CometSparkV05

model = CometSparkV05()
model = model.to("cuda")          # 递归迁移所有参数到 GPU
# 之后 forward / backward 自动走 TorchBackend（委托 torch.Tensor）

# 或在构建时指定
# model = CometSparkV05(device="cuda")
```

### 7.3 性能预期

| 场景 | CPU 基线 | GPU 加速比 | 备注 |
|---|---|---|---|
| 1B 模型 forward（batch=8, seq=512） | ~2.5 s | ~10× ~ 20× | 视 GPU 型号（A100 / V100 / 4090） |
| 1B 模型训练步（forward + backward + optimizer） | ~8 s | ~8× ~ 15× | backward 委托 `torch.Tensor.backward()` |
| 大规模 attention（seq ≥ 512） | 受限于 BLAS | ~20× ~ 50× | GPU 走 `F.scaled_dot_product_attention` fused kernel |
| 小模型（< 10M 参数） | ~10 ms | 0.5× ~ 1.5× | GPU 启动开销可能抵消收益，建议保持 CPU |

### 7.4 GPU 显存估算

```
gpu_mem ≈ (params × 4_bytes)              # 模型权重
        + (params × 4_bytes) × 2          # AdamW 一阶/二阶动量
        + (batch × seq × n_embd × n_layer × 4_bytes × factor)  # 激活值
```

其中 `factor` 取 4~8（autograd 中间张量 + 临时缓冲）。1B 模型 + AdamW 状态约需 12 GB 显存（fp32）；启用混合精度可降到约 7 GB。

### 7.5 多卡数据并行（API 预留）

`verse_torch.training.DistributedTrainer` 提供 DDP API 占位（当前单进程 fallback）。未来接入 `torch.distributed` 后将启用真实多卡训练：每张卡持有完整模型副本，按 `DistributedSampler` 切分 batch，反向后 all-reduce 梯度。

---

## 8. 混合精度 autocast（Part4K1）

`verse_torch.backend_torch.autocast` 提供 fp16 混合精度上下文管理器，GPU/NPU 下启用 `torch.autocast`，CPU 下为 no-op。

### 8.1 启用方式

```bash
# CLI 启用
verse-train --config spark/config/cometspark_v05.yml --device cuda --amp
```

```python
# Python API
from verse_torch.backend_torch import autocast

with autocast(device="cuda", enabled=True):
    logits = model(x)              # fp16 前向（部分算子自动保持 fp32）
    loss = cross_entropy_loss(logits, y)
loss.backward()                    # 反向走 torch autograd
optimizer.step()
```

### 8.2 加速与显存收益

| 指标 | fp32 基线 | fp16 autocast | 收益 |
|---|---|---|---|
| GPU 显存占用 | 100% | ~60% | 节省约 40% |
| forward 吞吐 | 1× | 1.5× ~ 2× | Tensor Core 加速 matmul |
| 训练步耗时 | 1× | 0.6× ~ 0.8× | 含 backward + optimizer |
| 数值精度 | 基线 | loss 差异 < 1e-2 | softmax / layernorm 内部保持 fp32 |

### 8.3 注意事项

- **CPU 下 no-op**：`autocast(device="cpu")` 不启用任何混合精度，不损失精度
- **无 PyTorch 时 no-op**：未安装 torch 时 `autocast` 同样 no-op，向后兼容
- **未内置 GradScaler**：当前需自行在 `optimizer.step()` 前处理梯度缩放（若反向出现 NaN）
- **建议配合 `--device cuda` 使用**：CPU 下启用 `--amp` 不会有任何加速效果

---

## 9. CachedDataset 数据加载加速（Part4K1）

`verse_infra.verse_trainer.data.CachedDataset` 解决大规模数据集（数百万条样本）的 tokenize 重复开销：首次扫描时把每条样本的 token ID 缓存到 `.npz` 文件，后续启动直接 mmap 加载。

### 9.1 工作原理

```
首次启动：
  CachedDataset(jsonl_path) → 逐行 tokenize → 写入 {jsonl_path}.cache.npz
后续启动：
  CachedDataset(jsonl_path) → 检测 .cache.npz 存在 → mmap 加载（跳过 tokenize）
```

### 9.2 使用方式

```python
from verse_infra.verse_trainer.data import CachedDataset

# 首次启动：扫描 + 缓存（耗时与数据集规模成正比）
dataset = CachedDataset(tokenizer, "data/train.jsonl", seq_len=512)

# 后续启动：直接加载缓存（毫秒级）
dataset = CachedDataset(tokenizer, "data/train.jsonl", seq_len=512)

# 自动失效：jsonl 文件 mtime 变化时重新扫描
```

### 9.3 加速效果

| 数据集规模 | 首次扫描耗时 | 后续加载耗时 | 加速比 |
|---|---|---|---|
| 1,000 条 | ~0.5 s | < 10 ms | ~50× |
| 100,000 条 | ~50 s | < 100 ms | ~500× |
| 1,000,000 条 | ~8 min | < 1 s | ~480× |

### 9.4 缓存文件管理

- 缓存文件路径：`{原始 jsonl 路径}.cache.npz`（与数据集同目录）
- 自动失效：检测 jsonl 文件 mtime 变化时自动重新扫描
- 手动清除：删除 `.cache.npz` 文件即可
- 磁盘占用：约为原始 jsonl 的 1.2× ~ 1.5×（token ID 比 UTF-8 文本略大）

### 9.5 配合多线程数据加载

CachedDataset 的 `__getitem__` 仅做 ndarray 切片（O(1)），可与 `BatchLoader` 的多 worker 加载配合，进一步隐藏数据加载延迟：

```python
from verse_infra.verse_trainer.data import CachedDataset
from verse_torch.training import BatchLoader

dataset = CachedDataset(tokenizer, "data/train.jsonl", seq_len=512)
loader = BatchLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## 10. 综合调优 Checklist

针对典型场景的推荐配置：

### 10.1 端侧流式推理（树莓派 / 嵌入式）

```bash
pip install "verse-nex[speed]"     # numba 加速 recurrent
export OPENBLAS_NUM_THREADS=1      # 单线程，避免切换开销
export OMP_NUM_THREADS=1
```

模型用 INT4 量化 + recurrent 模式生成。

### 10.2 CPU 训练（消费级 x86，16 核心）

```bash
pip install "verse-nex[speed]"
export OPENBLAS_NUM_THREADS=8      # 留一半核心给数据加载
export OMP_NUM_THREADS=8
```

batch_size=16~32，训练时用 parallel 模式（Mamba2Block.forward_parallel）。

### 10.3 服务器批量推理（Xeon / EPYC，64+ 核心）

```bash
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
```

batch_size=32~64，parallel 模式批量推理，可选 INT8 量化进一步提速。

---

## 11. 性能 profiling

定位热点：

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... 运行模型 ...
profiler.disable()

stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(20)  # 打印 top 20 热点
```

常见热点：
- `mamba2.py:_ssm_recurrent_step_kernel` → 装 numba
- `tensor.py:__matmul__` → 检查 BLAS 后端
- `nn.py:forward` → 考虑 `ParallelLinear` 或量化
