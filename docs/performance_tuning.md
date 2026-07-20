# Verse 性能调优指南

> 本文档面向希望榨干 CPU 性能的用户，覆盖 numba JIT 加速、BLAS 配置、batch_size 选择、CPU 线程数、量化加速、并行计算六个维度。Verse 全程纯 CPU / 纯 NumPy，所有调优都在用户态完成，无需重新编译框架。

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

## 7. 综合调优 Checklist

针对典型场景的推荐配置：

### 7.1 端侧流式推理（树莓派 / 嵌入式）

```bash
pip install "verse-nex[speed]"     # numba 加速 recurrent
export OPENBLAS_NUM_THREADS=1      # 单线程，避免切换开销
export OMP_NUM_THREADS=1
```

模型用 INT4 量化 + recurrent 模式生成。

### 7.2 CPU 训练（消费级 x86，16 核心）

```bash
pip install "verse-nex[speed]"
export OPENBLAS_NUM_THREADS=8      # 留一半核心给数据加载
export OMP_NUM_THREADS=8
```

batch_size=16~32，训练时用 parallel 模式（Mamba2Block.forward_parallel）。

### 7.3 服务器批量推理（Xeon / EPYC，64+ 核心）

```bash
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
```

batch_size=32~64，parallel 模式批量推理，可选 INT8 量化进一步提速。

---

## 8. 性能 profiling

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
