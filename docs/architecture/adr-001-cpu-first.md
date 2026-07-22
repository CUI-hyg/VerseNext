# ADR-001: CPU 优先设计决策

- **状态**：Accepted
- **日期**：2026-07-20
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/build-verse-framework/spec.md`](../../../.trae/specs/build-verse-framework/spec.md)
- **后续 ADR**：ADR-002（线性复杂度架构选型）、ADR-003（世界模型路线选型）、[ADR-005（GPU/NPU 后端抽象，Part4K1）](adr-005-gpu-npu-backend.md)

## 上下文

当前深度学习与大模型生态被 **PyTorch + Transformer** 双重锁定：

1. **算力依赖严重**：Transformer 自注意力 O(N²) 复杂度使得长上下文与超大规模模型对 GPU/HBM 资源的需求呈平方级膨胀。据 Epoch AI 估计，到 2030 年训练前沿 AI 将需要近 2000 万颗 H100 级 GPU，这条路已接近天花板。
2. **端侧/边缘部署困难**：PyTorch 体积庞大、依赖复杂（含 CUDA、cuDNN、TorchVision 等），难以在消费级 CPU、嵌入式设备、树莓派等场景下"开箱即用"地运行超大规模模型。
3. **GPU 锁定风险**：一旦生态以 PyTorch + CUDA 为默认底座，研究者与开发者就被绑定在 NVIDIA 硬件与闭源驱动栈上，难以迁移到国产 CPU、RISC-V、ARM 等替代平台；同时 GPU 短缺与出口管制加剧了这一风险。
4. **门槛过高**：现有 LLM 框架默认要求至少一张消费级 GPU 才能跑通最小示例，把大量学生、研究者、嵌入式开发者挡在门外。

Verse 的核心目标之一是 **在普通 CPU 上推理/训练超大规模模型**，并服务于端侧高能力 LLM 与世界模型。这要求框架在设计与实现层面就明确"CPU 优先"的取舍。

## 决策

**所有算子先在 CPU 上正确实现并优化，使用 NumPy 作为默认张量后端；GPU 后端延后到后续 spec。**

具体含义：

1. **运行时零重型依赖**：VerseTorch 运行时只依赖 NumPy（≥ 1.26）与 Python 标准库；可选依赖包括 Numba（CPU 加速）、safetensors（权重加载）、FastAPI（HTTP server）。**不依赖** PyTorch / Transformers / TensorFlow / JAX 作为运行时（仅 `verse_compat` 可在用户已安装时调用其加载器）。
2. **CPU 优先实现顺序**：每个算子先在纯 NumPy 上实现正确版本（与 PyTorch 数值对齐到 1e-5 ~ 1e-6），再考虑 Numba/Cython 加速，最后考虑量化（INT4/INT8/1.58-bit ternary）。GPU kernel 不在阶段 0–1 的范围内。
3. **NumPy 后端选型理由**：
   - 用户已普遍安装，跨平台（x86 / ARM / RISC-V / WebAssembly via Pyodide）；
   - 向量化 API 与 PyTorch 张量语义高度一致，便于移植；
   - 可作为后续多后端（CPU-Numba / CPU-Cython / GPU）的"参考实现"与数值回归基线。
4. **量化优先**：CPU 上算力受限，因此 INT4 (W4A16) 与 1.58-bit ternary 量化是一等公民而非可选优化——`QuantizedLinear` 与 `Linear` API 兼容，默认走量化路径。参考 BitNet.cpp 在 CPU 上 6.17× 提速的工程证据。
5. **GPU 后端路线图**：本 ADR **不否定** GPU 的价值，仅将其延后。大规模预训练仍需 GPU 集群；后续 spec 会引入可选 GPU 后端（候选：CuPy / 自实现 CUDA kernel），但 API 必须保持与 CPU 路径一致，用户代码零修改切换。

## 后果

### 优点

- **零重型依赖**：`pip install verse-torch` 只拉取 NumPy，安装包体积小、启动快，可在 CI、Docker、嵌入式镜像中无负担使用。
- **跨平台**：NumPy 在 Linux / macOS / Windows / ARM / RISC-V / 树莓派 / 嵌入式 Linux 上均有预编译 wheel，Verse 可直接运行。
- **端侧部署**：可在树莓派 5、Intel N100、Apple Silicon（通过 NumPy/Metal 后续可加速）等设备上跑通示例；为端侧高能力 LLM 与世界模型提供原生底座。
- **可审计性**：纯 Python + NumPy 实现易于阅读、调试与教学；数值梯度检查可直接对照有限差分法。
- **架构自由度**：不被 PyTorch 的 `nn.Module` / `aten` 内部抽象锁死，可以为 SSM 递归状态、JEPA EMA target encoder 等设计专门的抽象。
- **生态友好**：通过 `verse_compat` 仍可读取 HuggingFace `state_dict` / `.bin` / `.safetensors`，迁移成本低。

### 缺点

- **训练吞吐量低于 GPU**：纯 CPU + NumPy 在大规模预训练（>1B 参数、>10B token）上吞吐量比 A100/H100 低 1–2 个数量级。阶段 0–1 不适合做大规模预训练，仅适合：
  - 算法验证（小规模 MNIST / CIFAR / 字符级 LM）；
  - 端侧推理（≤ 1B 参数量化模型）；
  - 教学与原型。
- **大规模预训练仍需 GPU 集群**：这是已知限制，将在后续 spec 中通过可选 GPU 后端解决。
- **部分算子需要手写**：例如 selective scan、parallel scan、chunkwise attention 等在 NumPy 下需要显式循环或 reshape 技巧，性能可能不及手写 CUDA kernel。
- **内存带宽瓶颈**：CPU 内存带宽（~50 GB/s DDR5）远低于 HBM（~3 TB/s），大模型推理的内存-bound 特性会暴露。缓解策略：INT4/ternary 量化降低带宽需求。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| CPU 性能不达标导致示例无法跑通 | 引入 Numba/Cython 可选加速；INT4 量化默认开启；基准测试门槛设为"4 核 CPU、16GB RAM、5 分钟内 100 tokens"而非 GPU 指标 |
| API 与 PyTorch 偏离导致生态不友好 | `verse_compat.torch_api` 提供 `torch.nn.Linear` 等别名；`Tensor` API 严格对照 PyTorch 子集 |
| 后续 GPU 后端引入导致 API 破坏 | 后端抽象从一开始就预留（`Tensor.backend`），GPU 后端只实现同一接口 |

## 替代方案（已否决）

### 方案 A：直接基于 PyTorch

**描述**：不重写 Tensor 与 autograd，直接 `import torch` 作为底座，在其上实现 Mamba-2 / RWKV-7 / JEPA 等模块。

**否决理由**：
- **运行时依赖过重**：PyTorch 单包 > 800 MB（含 CUDA），与"端侧零依赖"目标冲突；
- **GPU 锁定**：PyTorch 的 CPU 路径是二等公民，许多优化（如 fused kernel）只在 CUDA 上生效；
- **抽象不匹配**：PyTorch 的 `nn.Module` 假设静态图友好的参数注册，与 SSM 递归状态、JEPA EMA target encoder 等需要"非参数状态"的设计存在张力，需要大量 workaround；
- **量化生态割裂**：PyTorch 的量化（torch.quantization）面向移动端 INT8，对 1.58-bit ternary 支持差，需要绕过框架自己实现。

### 方案 B：基于 tinygrad

**描述**：fork 或依赖 tinygrad 作为张量后端，复用其 lazy 计算图与多后端能力。

**否决理由**：
- **lazy 计算模型与 PyTorch API 不一致**：tinygrad 采用 lazy + shape tracking，与 PyTorch 的 eager + 动态图语义差异较大，移植现有 PyTorch 模型代码（含 `if`/`for` 控制流、in-place 操作）成本高；
- **API 稳定性**：tinygrad 仍处于快速迭代期，API 变动频繁，不适合作为长期底座；
- **调试体验**：lazy 模型在调试时难以直接查看中间张量值，与"教学/可审计"目标冲突；
- **CPU 路径非主线**：tinygrad 主线是 GPU/CLANG/LLVM 后端，NumPy 后端性能与优化优先级不高。

### 方案 C：基于 JAX

**描述**：使用 JAX 的函数式 + jit/vmap 作为底座。

**否决理由**：
- 函数式 API 与 PyTorch 风格差异极大，迁移成本最高；
- JAX 默认依赖 CPU/GPU/TPU 三后端，安装包同样庞大；
- 与"端侧零依赖"目标冲突。

### 方案 D：纯 C/C++ 实现（如 llama.cpp 路线）

**描述**：放弃 Python，用 C/C++ 实现整个框架（类似 llama.cpp / ggml）。

**否决理由**：
- 与"纯 Python / 易审计 / 易教学"目标冲突；
- 开发迭代速度慢，调试困难；
- 失去 NumPy 生态（自动微分、scipy、matplotlib 等无缝集成）。
- **保留作为后续性能选项**：未来 CPU 关键路径（GEMM、selective scan）可以提供 C 扩展作为可选加速，但顶层 API 保持 Python。

## 备注

- 本 ADR 仅约束 **阶段 0–1** 的实现路线。GPU 后端的引入将由后续 ADR 单独决策。
- 本 ADR 与 spec.md 中"BREAKING 设计取舍"章节一致：不依赖 PyTorch / Transformers / TensorFlow / JAX 作为运行时；不重新发明 numpy / scipy / numba / safetensors。
- 相关工程参考：[llama.cpp](https://github.com/ggml-org/llama.cpp)、[BitNet.cpp](https://github.com/microsoft/BitNet)、[tinygrad](https://docs.tinygrad.org/)、[micrograd](https://github.com/karpathy/micrograd)。

## 演进更新（Part4K1）

本 ADR 确立的 **CPU 优先原则保持不变**，但 Part4K1 已通过 [ADR-005: GPU/NPU 后端抽象](adr-005-gpu-npu-backend.md) 引入可选 GPU/NPU 委托后端（PyTorch 委托，不自研 CUDA kernel；NPU 走 `torch_npu`）。关键要点：

- **CPU 仍是默认路径与数值回归基线**：无 PyTorch 时所有测试不变通过（向后兼容）。
- **GPU/NPU 仅作为可选加速**：用户代码零修改切换（`model.to("cuda")`），无 PyTorch 时自动回退 CPU。
- **不自研 CUDA kernel**：GPU 路径完全委托 `torch` 原生算子（含 `F.scaled_dot_product_attention` fused kernel），与 ADR-001 的"不重新发明 numpy / scipy / numba"原则一脉相承。
- **本 ADR 的"阶段 0–1 约束"已解除**：大规模预训练可通过 `--device cuda` 走 GPU 路径；端侧推理仍走 CPU + 量化路径。
