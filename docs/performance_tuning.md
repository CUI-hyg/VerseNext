# Verse 性能调优指南

> 本文档面向希望榨干硬件性能的用户，覆盖 numba JIT 加速、BLAS 配置、batch_size 选择、CPU 线程数、量化加速、并行计算、**GPU/NPU 加速（Part4K1 新增）**、**混合精度 autocast（Part4K1 新增）**、**CachedDataset 数据加载加速（Part4K1 新增）**、**压缩技术 V1.3 调优（Part4K2 新增）**、**智能分区训练性能调优（Part4K2 新增）**、**资源利用优化（Part4K2 新增）**、**1B 模型训练优化建议（Part4K2 新增）** 十三个维度。Verse 默认纯 CPU / 纯 NumPy（无需重新编译框架），Part4K1 起可选启用 GPU/NPU 加速（通过 PyTorch 委托后端）。

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

---

## 12. 压缩技术 V1.3 调优（Part4K2 新增）

Part4K2 推出压缩管线 V1.3（`prune → quantize → distill → lora`），通过三重损失知识蒸馏实现大模型→小模型能力转移。详见 [ADR-012](architecture/adr-012-compression-v13.md)。

### 12.1 何时使用 V1.3

- 大模型（teacher）已训练完成，希望蒸馏出小模型（student）用于端侧部署
- 需要在压缩比（参数量 / bit 数）与模型质量之间取得平衡
- 希望压缩后仍能通过 LoRA 微调适配下游任务

### 12.2 V1.3 配置示例

```python
from verse_torch.compress import compress_pipeline, compression_report

config = {
    "prune":    {"sparsity": 0.3},           # 剪枝 30%
    "quantize": {"bits": 4},                  # INT4 量化
    "distill":  {                             # 知识蒸馏
        "teacher": teacher_model,
        "train_loader": train_loader,
        "epochs": 3,
        "lr": 1e-3,
        "temperature": 4.0,                   # 初始温度（自动退火到 T_min）
        "alpha": 0.7,                         # 软标签权重
        "feature_loss_weight": 0.3,           # 中间层特征匹配权重
    },
    "lora":     {"rank": 8, "alpha": 16},     # LoRA 包装（为微调准备）
}
compressed, stats = compress_pipeline(model, config, version="1.3", return_stats=True)
print(compression_report(model, compressed))
```

### 12.3 调优建议

| 维度 | 建议 |
|---|---|
| `prune.sparsity` | 0.2 ~ 0.4；过高会破坏模型结构，过低压缩比不足 |
| `quantize.bits` | 端侧部署用 4（INT4），平衡用 8（INT8） |
| `distill.temperature` | 初始 4.0；V1.3 自动退火到 `max(1.0, T * 0.25)`，无需手动调 |
| `distill.alpha` | 0.7（软标签主导）；无硬标签时软标签全权 |
| `distill.feature_loss_weight` | 0.3；设为 0 禁用特征匹配（退化为 V1.0 双损失） |
| `distill.epochs` | 3 ~ 5；过多会过拟合 teacher 的分布 |
| `lora.rank` | 8 ~ 16；rank 越大微调能力越强但参数越多 |

### 12.4 吞吐率优化

V1.3 量化后 `QLinear` 内部走 fused matmul 路径（`matmul_int4`），INT4 权重的访存优势转化为实际吞吐率提升：

| 量化类型 | 估算吞吐率提升（相对 fp32） |
|---|---|
| INT4 | ≈ 4× |
| INT8 | ≈ 2× |
| ternary（1.58-bit） | ≈ 8× |

> **提示**：`stats["estimated_throughput_improvement"]` 字段返回估算值；实际提升取决于 CPU / GPU 与 BLAS 后端。

### 12.5 VerseNex 集成

```python
# CometSparkNexLM 实例方法
compressed = model.compress_v13(config)
# 或从 teacher 蒸馏
student = small_model.distill_from(teacher=large_model, train_data=train_loader)
```

---

## 13. 智能分区训练性能调优（Part4K2 新增）

`LayerWiseTrainer` 按 layer 分组训练 + `.vn` 分片卸载，适用于低内存训练大模型。详见 [ADR-011](architecture/adr-011-layerwise-training.md)。

### 13.1 性能特征

- **内存换时间**：分区训练通过卸载已训练组到硬盘，降低峰值内存，但增加 I/O 开销
- **逐组串行**：总步数相同时各组训练步数减少（`max_steps // n_partitions`）
- **fine-tune 弥合**：合并后整体微调 `finetune_steps` 步，弥合层间边界

### 13.2 partition_size 调优

| 场景 | `partition_size` | 理由 |
|---|---|---|
| CPU 8GB 内存 | 2 | 最小分区，最大化内存节省 |
| CPU 16GB 内存 | 4 | 平衡内存与训练速度 |
| GPU 24GB 显存 | 8 | 大分区减少 I/O，或不用分区训练 |
| 调试 / 验证 | 1 | 极端逐层，仅用于验证流程 |

> **提示**：`partition_size` 越小，越接近逐层训练（内存最省但速度最慢）；越大越接近全量训练（速度最快但内存最高）。

### 13.3 memory_threshold_mb 调优

```python
from verse_torch import LayerWiseTrainer

trainer = LayerWiseTrainer(
    model,
    config={"lr": 1e-3, "finetune_steps": 20},
    partition_size=2,
    memory_threshold_mb=512,   # CPU 默认 512MB
    # memory_threshold_mb=4096,  # GPU 可设 4096MB
)
```

- **阈值过低**：频繁触发卸载，I/O 开销大
- **阈值过高**：不触发卸载，内存压力未缓解
- **建议**：CPU 设为物理内存的 25%~50%；GPU 设为显存的 50%~75%

### 13.4 与其他加速技术组合

| 组合 | 效果 |
|---|---|
| 分区训练 + `--amp`（GPU） | 降低显存 + 混合精度加速 |
| 分区训练 + INT4 量化 | 训练时量化权重，进一步降内存 |
| 分区训练 + `--parallel-chunks` | 分区拆参数 + chunk 拆 step，双重降内存 |
| 分区训练 + 激活检查点 | `use_checkpoint=True`，参数 + 激活双重省内存 |

### 13.5 I/O 优化

- `.vn` 分片用 safetensors（mmap 零拷贝），读取较快
- `offload_dir` 建议指定到 SSD（HDD 的随机读写会拖慢卸载 / 加载）
- `cleanup()` 清理自动创建的临时目录（仅当 `offload_dir` 未指定时）

---

## 14. 资源利用优化（Part4K2 新增）

Part4K2 Task 5 完善了 `verse_torch.device` 的资源监控与线程管理 API。

### 14.1 内存监控

```python
from verse_torch.device import get_memory_info, memory_usage, empty_cache

# 查询 CPU 内存
info = get_memory_info("cpu")
# {"total": ..., "used": ..., "free": ..., "used_percent": ...}

# 查询 GPU 内存（需 PyTorch）
info = get_memory_info("cuda")
print(memory_usage("cuda"))   # 返回 "X.X GB / Y.Y GB"

# 释放缓存（GPU 时调用 torch.cuda.empty_cache）
empty_cache("cuda")
```

### 14.2 CPU 线程数调优

```python
from verse_torch.device import set_num_threads, get_num_threads, auto_tune_threads

# 手动设置
set_num_threads(8)
print(get_num_threads())   # 8

# 自动调优（根据 CPU 核心数）
auto_tune_threads()
```

> **提示**：`auto_tune_threads` 根据 `os.cpu_count()` 自动设置线程数，适用于不确定硬件配置的场景。手动设置时建议设为物理核心数（非超线程数）。

### 14.3 激活检查点（GPU 大模型训练）

`VerseNexBlock` 的 `use_checkpoint=True` 开关启用激活检查点：

```python
from verse_nex import CometSparkNexLM

model = CometSparkNexLM(
    vocab_size=248320, dim=1024, n_layer=20,
    layer_pattern=["mod"]*5 + ["trisparse"]*15,
    use_checkpoint=True,   # 启用激活检查点
)
```

- **GPU 场景**：前向不保存中间激活，反向时重新计算，节省显存
- **CPU 场景**：自动降级为直接前向（CPU 内存通常不是瓶颈）
- **代价**：反向传播多一次前向计算，训练速度约慢 30%

### 14.4 NPU autocast 支持

`backend_torch.py` 的 `autocast` 上下文管理器支持 NPU：

```python
from verse_torch.backend_torch import autocast

# NPU 混合精度
with autocast(device_type="npu", dtype=torch.float16):
    logits = model(x)
    loss = cross_entropy(logits, y)
```

---

## 15. 1B 模型训练优化建议（Part4K2 新增）

CometSpark V0.5-1B（约 1.12B 参数）的训练需要综合运用多种优化技术。

### 15.1 内存预算

| 组件 | fp32 内存 | INT4 内存 |
|---|---|---|
| 权重 | 4.5 GB | 0.56 GB |
| AdamW 优化器状态（两阶矩） | 9.0 GB | 1.12 GB |
| 激活值（batch=8, seq=512） | ~2 GB | ~2 GB |
| **峰值合计** | **~15.5 GB** | **~3.7 GB** |

### 15.2 CPU 训练方案（8GB ~ 16GB 内存）

```bash
# 方案 A：智能分区训练（推荐 8GB 内存）
verse-train --config spark/config/cometspark_v05.yml \
    --partition-training --partition-size 2 --max-steps 1000

# 方案 B：分区训练 + 并行 chunk + 小 batch
verse-train --config spark/config/cometspark_v05.yml \
    --partition-training --partition-size 4 \
    --parallel-chunks 2 --single-sample --max-steps 2000

# 方案 C：小配置调试（cometspark_v05_small.yml）
verse-train --config spark/config/cometspark_v05_small.yml \
    --device cpu --max-steps 10
```

### 15.3 GPU 训练方案（24GB 显存）

```bash
# 方案 A：全量训练 + 混合精度 + 并行
verse-train --config spark/config/cometspark_v05.yml \
    --device cuda --amp --parallel-chunks 4 --loss-optimizer --max-steps 10000

# 方案 B：分区训练 + 混合精度（显存紧张时）
verse-train --config spark/config/cometspark_v05.yml \
    --partition-training --partition-size 8 \
    --device cuda --amp --max-steps 5000

# 方案 C：激活检查点 + 混合精度（模型代码需 use_checkpoint=True）
verse-train --config spark/config/cometspark_v05.yml \
    --device cuda --amp --max-steps 10000
```

### 15.4 持续训练优化

训练完成后可用 `verse-continue` 追加训练，无需从头开始：

```bash
# 追加 1000 步（GPU + 混合精度）
python -m verse_infra.verse_trainer.cli verse-continue \
    --checkpoint checkpoints/best.pt --additional-steps 1000 \
    --config spark/config/cometspark_v05.yml --device cuda --amp
```

### 15.5 压缩部署

训练完成后用 V1.3 压缩管线部署到端侧：

```python
from verse_torch.compress import compress_pipeline

config = {
    "prune":    {"sparsity": 0.3},
    "quantize": {"bits": 4},
    "distill":  {"teacher": large_model, "train_loader": train_loader, "epochs": 3},
    "lora":     {"rank": 8, "alpha": 16},
}
small_model, stats = compress_pipeline(large_model, config, version="1.3", return_stats=True)
# 压缩后约 280M 参数（INT4），峰值内存 ~1.5 GB，可在消费级 CPU 上推理
```

### 15.6 Checklist

- [ ] 确认 `safetensors` 已安装（`.vn` 格式 mmap 零拷贝）
- [ ] 确认 `numba` 已安装（selective scan 加速）
- [ ] CPU 训练：`OMP_NUM_THREADS` 设为物理核心数
- [ ] GPU 训练：`--amp` 混合精度 + `--parallel-chunks` 分块
- [ ] 内存不足：`--partition-training --partition-size 2`
- [ ] 训练完成：`verse-continue` 追加训练
- [ ] 部署：`compress_pipeline(version="1.3")` 压缩 + `verse-convert` 转 `.vn`
