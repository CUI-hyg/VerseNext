# ADR-005: GPU/NPU 后端抽象（DeviceBackend）

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k1-infra-model-upgrade/spec.md`](../../../.trae/specs/part4k1-infra-model-upgrade/spec.md)
- **前置 ADR**：[ADR-001 CPU 优先](adr-001-cpu-first.md)
- **相关 ADR**：[ADR-008 超稀疏并行注意力](adr-008-parallel-sparse-attention.md)（GPU 后端为多 chunk 并行提供算力）

## 上下文

ADR-001 确立了"CPU 优先 + 零重型依赖"的路线，所有算子先用纯 NumPy 实现并优化。随着 Part4K1 推出 **CometSpark V0.5-1B**（≈1.12B 参数），CPU 在大规模预训练上的吞吐量已无法满足：

1. **1B 参数预训练**：CPU + NumPy 在 1B 模型 + 200 步训练下耗时数小时到数天，无法快速迭代
2. **NPU 生态需求**：国产昇腾 NPU 通过 `torch_npu` 扩展支持 PyTorch，但 VerseTorch 当时完全没有 NPU 路径
3. **API 一致性要求**：用户希望在 CPU / GPU / NPU 之间零代码修改切换，类似 PyTorch 的 `Tensor.to(device)`
4. **不自研 kernel 约束**：ADR-001 明确"不重新发明 numpy / scipy / numba / safetensors"——CUDA kernel 同样不应自研，应复用 PyTorch 原生实现

同时，必须保持 ADR-001 的核心承诺：**无 PyTorch 环境下 CPU 路径完全不变**（向后兼容）。

## 决策

**引入 `DeviceBackend` 抽象基类 + `NumpyBackend`（默认）+ `TorchBackend`（PyTorch 委托），通过 `Tensor.device` 与 `Module.to(device)` 暴露统一 API；CUDA kernel 走 PyTorch 原生实现，NPU 走 `torch_npu` 扩展。**

具体含义：

1. **`DeviceBackend` 抽象基类**（`verse_torch/device.py`）：
   - 定义 `matmul / linear / softmax / attention / etc.` 抽象方法
   - `NumpyBackend`：默认实现，所有算子用 NumPy（与自研 autograd 完全等价）
   - `TorchBackend`：委托 PyTorch 原生 op（CUDA kernel 走 PyTorch，NPU 走 `torch_npu`），**不自研 kernel**
   - `get_backend(device)` 工厂：根据 device 字符串返回对应 backend
   - `has_torch()` / `has_torch_npu()` 探测函数

2. **`Tensor` 设备 API**（对齐 PyTorch）：
   - `Tensor.device`：返回当前设备字符串（`"cpu"` / `"cuda"` / `"cuda:0"` / `"npu"` / `"npu:0"`）
   - `Tensor.to(device)` / `.cuda()` / `.npu()` / `.cpu()`：迁移数据并切换 backend
   - GPU 下 autograd 委托 PyTorch autograd（不再维护自研计算图）
   - CPU 下保持自研 autograd（向后兼容 ADR-001）

3. **`Module.to(device)`**：递归迁移所有参数到目标设备

4. **`autocast` 混合精度**（`verse_torch/backend_torch.py`）：
   - `with autocast(enabled=True, dtype="float16"):` 上下文管理器
   - GPU 下启用 PyTorch `torch.autocast`，fp16 GEMM 加速
   - CPU 下为 no-op（不破坏 CPU 路径）

5. **设备字符串规范**：`"cpu"` / `"cuda"` / `"cuda:0"` / `"npu"` / `"npu:0"` / `"mps"`（Apple Silicon）

6. **CLI 集成**：`verse-train --device cuda --amp` / `--device npu` 通过 `DeviceBackend` 切换

## 后果

### 优点

- **API 一致**：用户代码 `Tensor.to("cuda")` / `Module.to("npu")` 与 PyTorch 完全一致，迁移成本接近零
- **不自研 kernel**：CUDA kernel 全部走 PyTorch 原生（cuBLAS / cuDNN / FlashAttention），NPU 走 `torch_npu`，避免维护成本与正确性风险
- **CPU 路径不变**：无 PyTorch 环境下 `Tensor.cuda()` 抛 `RuntimeError("未安装 PyTorch，无法使用 GPU")`，所有现有测试零修改通过
- **混合精度**：`autocast` 在 GPU 下自动启用 fp16 GEMM，显存占用减半 + 吞吐量提升 1.5×~2×
- **NPU 原生支持**：昇腾设备通过 `torch_npu` 扩展直接可用，无需额外适配
- **autograd 委托**：GPU 下 autograd 走 PyTorch autograd，避免在 GPU 上维护自研计算图的反向开销

### 缺点

- **GPU 路径硬依赖 PyTorch**：`--device cuda` 必须先 `pip install torch>=2.2`，违反"零重型依赖"原则——但这是可选依赖，CPU 路径无需安装
- **两套 autograd 实现**：CPU 用自研 autograd（拓扑排序 + 闭包），GPU 用 PyTorch autograd——理论上存在数值差异（float32 下吻合到 1e-5，但极端情况可能分歧）
- **NPU 依赖 `torch_npu` 版本对齐**：`torch_npu` 必须与 `torch` 版本严格匹配，否则 import 失败
- **设备迁移开销**：`Tensor.to(device)` 在 CPU↔GPU 之间拷贝数据，频繁迁移会拖慢训练；用户需理解"数据与模型在同一设备"的约束

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| CPU 与 GPU 数值不一致导致测试分歧 | `tests/test_device_backend.py` 覆盖 NumpyBackend / TorchBackend 数值一致性（float32 吻合 1e-5）；GPU 测试标记 `@pytest.mark.gpu` 在无 GPU 环境跳过 |
| PyTorch 版本升级破坏委托 | `TorchBackend` 只用 PyTorch 稳定 API（`torch.matmul` / `torch.nn.functional.softmax` 等），避免用 `aten` 内部 op |
| 用户误在 CPU 环境调用 `.cuda()` | 抛 `RuntimeError("未安装 PyTorch，无法使用 GPU")`，错误信息明确 |
| `torch_npu` 不可用时 `--device npu` 静默失败 | `has_torch_npu()` 探测 + 明确报错 `RuntimeError("未安装 torch_npu，无法使用 NPU")` |

## 替代方案（已否决）

### 方案 A：自研 CUDA kernel（CuPy / PyCUDA）

**描述**：用 CuPy 或 PyCUDA 直接写 CUDA kernel 实现 matmul / attention。

**否决理由**：
- 维护成本极高（每种 op 都要写 kernel + 反向 kernel）
- 性能不及 PyTorch 原生（PyTorch 已集成 cuBLAS / cuDNN / FlashAttention）
- 违反 ADR-001 "不重新发明轮子"原则
- NPU 无法复用（CuPy 只支持 CUDA）

### 方案 B：强制依赖 PyTorch（删除 NumPy 后端）

**描述**：放弃 CPU 优先，直接 `import torch` 作为唯一后端。

**否决理由**：
- 违反 ADR-001 的核心承诺（端侧零依赖）
- 破坏所有现有 CPU-only 测试与示例
- 安装包体积膨胀（PyTorch > 800MB）
- 树莓派 / 嵌入式设备无法安装 PyTorch

### 方案 C：基于 tinygrad 多后端

**描述**：用 tinygrad 的 lazy 计算图 + 多后端（CUDA / Metal / NPU）。

**否决理由**：
- tinygrad API 与 PyTorch 差异大（lazy + shape tracking），迁移成本高
- tinygrad 仍处于快速迭代期，API 不稳定
- NPU 支持不成熟

### 方案 D：仅支持 CUDA，不支持 NPU

**描述**：只委托 PyTorch CUDA，NPU 留待后续。

**否决理由**：
- 国产 NPU 生态需求明确（昇腾是 Part4K1 目标硬件之一）
- `torch_npu` 已提供 PyTorch 风格 API，委托成本与 CUDA 几乎相同
- 排除 NPU 会让"可选 GPU/NPU 后端"的承诺不完整

## 备注

- 本 ADR 是 ADR-001 "GPU 后端路线图"的具体落地，不否定 ADR-001 的 CPU 优先原则
- `torch` 与 `torch_npu` 均为**可选依赖**，`pip install verse-torch` 不会拉取它们
- GPU 后端默认禁用，仅在 `Tensor.cuda()` / `--device cuda` / `autocast` 时触发
- 相关测试：`tests/test_device_backend.py` 覆盖 NumpyBackend / TorchBackend / device 迁移 / 无 PyTorch 回退 / autocast
- 相关代码：[`verse_torch/device.py`](../../packages/verse_torch/verse_torch/device.py) / [`verse_torch/backend_torch.py`](../../packages/verse_torch/verse_torch/backend_torch.py)
