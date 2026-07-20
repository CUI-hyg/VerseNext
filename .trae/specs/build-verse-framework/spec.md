# Verse 框架（VerseNex + VerseTorch + VerseAWM）Spec

## Why

当前深度学习与大模型生态被 **PyTorch + Transformer** 双重锁定：

1. **算力依赖严重**：Transformer 自注意力 O(N²) 复杂度，使得长上下文与超大规模模型对 GPU/HBM 资源的需求呈平方级膨胀；据 Epoch AI 估计，到 2030 年训练前沿 AI 将需要近 2000 万颗 H100 级 GPU，这条路已接近天花板。
2. **端侧/边缘部署困难**：PyTorch 体积庞大、依赖复杂，难以在消费级 CPU、嵌入式设备、树莓派等场景下"开箱即用"地运行超大规模模型。
3. **架构层面单一**：纯 Transformer 路线在长程依赖建模、推理时 KV Cache 线性膨胀、长度外推等方面存在结构性缺陷，无法直接服务于"世界模型"这一下一代范式（需要持续状态、潜在空间预测、层次化规划能力）。

我们需要一个 **完全基于 Python、可纯 CPU 运行、不强制依赖 PyTorch/Transformers** 的深度学习与 LM 框架，用以：
- 在普通 CPU 上推理/训练超大规模模型；
- 用 **线性复杂度架构（SSM/Mamba/RWKV/Linear Attention）** 替代或混合 Transformer，根治注意力/上下文瓶颈；
- 为下一代 **世界模型（JEPA、RSSM、H-JEPA）** 与端侧高能力 LLM 提供原生支撑；
- 尽可能兼容现有 HuggingFace / PyTorch 生态，降低迁移成本。

## What Changes

### 新增模块（packages/ 下）

- **`packages/verse_torch/`**：纯 Python + NumPy 实现的轻量张量与自动微分引擎（PyTorch 替代），CPU 优先、可扩展到多核/SIMD/量化。
  - `Tensor` 类、动态计算图、反向模式 autograd（基于 VJP）；
  - 核心算子（matmul、conv、reduce、broadcasting-aware backward）；
  - 优化器栈（SGD/Adam/AdamW）、学习率调度器；
  - **量化子系统**：INT4/INT8/1.58-bit（ternary）权重量化与反量化 kernel；
  - **CPU 加速后端**：基于 NumPy + 可选 Numba/Cython 的高效 GEMM；
  - **兼容层**：可从 PyTorch `state_dict` / `.bin` / `.safetensors` 加载权重，API 与 `torch.nn` 子集对齐。
- **`packages/verse_nex/`**：Transformer 替代架构库（线性复杂度优先）。
  - **SSM 内核**：Mamba/Mamba-2 选择性状态空间模型（selective scan）；
  - **RWKV 内核**：RWKV-7 时间混合 + 通道混合；
  - **Linear Attention**：RetNet、Lightning Attention、Performer 风格近似；
  - **Hybrid Block**：SSM + Sparse Attention 混合（参考 RWKV-X / Nemotron-H / Samba）；
  - **位置编码**：RoPE、ALiBi、NoPE 适配；
  - **训练友好性**：parallel scan、chunkwise 训练 kernel；
  - **CPU 优化**：递归推理 O(1) 状态、并行训练 O(N) 计算。
- **`packages/verse_awm/`**：世界模型专用包（Autonomous World Model）。
  - **JEPA 系列**：I-JEPA / V-JEPA 风格的潜在空间预测架构；
  - **RSSM**：循环状态空间模型（PLATO/Dreamer 风格）；
  - **H-JEPA**：层次化 JEPA（高层抽象规划 + 低层动作控制）；
  - **Energy-Based Loss**：非生成式表征空间损失；
  - **EMA Target Encoder**：防止表征坍塌；
  - **后训练接口**：可与 VerseNex 的 LM backbone 联合微调。
- **`packages/verse_tokenizer/`**：轻量分词器（BPE/Unigram/SentencePiece 子集），无 `tokenizers` / `sentencepiece` 重依赖时仍可运行。
- **`packages/verse_inference/`**：模型加载、KV/状态缓存、流式生成、OpenAI 兼容 HTTP server（可选 FastAPI）。
- **`packages/verse_compat/`**：HuggingFace `transformers` / `torch` 兼容适配层（仅在用户已安装时启用，作为"初期过渡"）。

### 目录与基础设施

- **`datasets/`**：清洗后的训练/评测数据集（子目录：`raw/`, `cleaned/`, `tokenizer/`, `README.md`）。
- **`docs/`**：外来文档与资料（论文、架构图、benchmark）。
  - `docs/papers/`：参考资料 PDF/Markdown 摘要；
  - `docs/architecture/`：架构决策记录（ADR）；
  - `docs/benchmarks/`：性能基准结果。
- **`verse_data/`**：内部自有材料（设计草稿、内部实验、迁移笔记）。
- **`tests/`**：单元测试 + 数值梯度检查 + 端到端用例。
- **`examples/`**：示例代码（MNIST、最小 LM、Mamba 1B CPU 推理 demo、JEPA demo）。
- **`pyproject.toml`**：workspace 多包布局（PEP 621 + uv/pip 可编辑安装）。

### 关键技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 张量后端 | NumPy + 可选 Numba | CPU 优先、零重型依赖、用户已可安装 |
| Autograd | 反向模式 VJP、动态图 | 与 PyTorch API 一致、易于审计 |
| 主推架构 | Mamba-2 + RWKV-7 + Hybrid | 已公开、工业验证（MiniMax-01、混元 T1、Nemotron-H） |
| 量化默认 | INT4 (W4A16) + 1.58-bit ternary | BitNet.cpp 在 CPU 上已验证 6.17× 提速 |
| 兼容策略 | 初期可读 PyTorch state_dict，运行时无 PyTorch 依赖 | 生态友好但运行时零依赖 |
| 世界模型主线 | JEPA（非生成式）+ RSSM（生成式） | 兼顾 LeCun 路线与 Dreamer 路线 |

### **BREAKING** 设计取舍

- **不依赖** PyTorch / Transformers / TensorFlow / JAX 作为运行时（仅 `verse_compat` 可在用户已安装时调用其加载器）。
- **初期实现** 不追求 GPU 支持，所有算子先在 CPU 上正确实现并优化；GPU 后端延后到后续 spec。
- **不重新发明** numpy / scipy / numba / safetensors（这些是必要的轻量基础依赖）。

## Impact

- **Affected specs**：本仓库当前无既有 spec，本 spec 为根 spec。
- **Affected code**：
  - 全新仓库结构（`/workspace` 当前仅有 LICENSE + README.md，可视为绿地项目）；
  - 三个核心 Python 包（`verse_torch`, `verse_nex`, `verse_awm`）；
  - 工具链：pyproject.toml、CI（pytest）、格式化（ruff/black）。
- **影响外部**：
  - 用户需安装 Python ≥ 3.10、NumPy ≥ 1.26；
  - 可选：Numba（CPU 加速）、FastAPI（HTTP server）、safetensors（权重加载）、requests（数据下载）。
- **参考资料（已收集）**：
  - 论文：[The End of Transformers? Sub-Quadratic Architectures](https://arxiv.org/html/2510.05364v1)、[RWKV-X](https://arxiv.org/html/2504.21463v2)、[LongMamba (ICLR 2025)](https://www.jankautz.com/publications/LongMamba_ICLR25.pdf)、[On the Length Generalization of Mamba (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/file/1bfc9f74afa91b9b8add5a97a97001a1-Paper-Conference.pdf)、[Characterizing SSM/Hybrid LM Performance](https://arxiv.org/html/2507.12442v4)、[Achilles' Heel of Mamba](https://arxiv.org/html/2509.17514v1)；
  - 工程参考：[llama.cpp](https://github.com/ggml-org/llama.cpp)、[BitNet.cpp](https://github.com/microsoft/BitNet)、[lm.c](https://github.com/oderoi/lm.c)、[tinygrad](https://docs.tinygrad.org/)、[PureML](https://joss.theoj.org/papers/10.21105/joss.09631.pdf)；
  - 世界模型：[V-JEPA 2](https://aiwiki.ai/wiki/v_jepa_2)、[JEPA Deep Dive](https://dgallitelli.github.io/blog/world-models-series/part2-jepa-deep-dive/)；
  - 微信公众号文章（共 20 篇）：见 `docs/papers/wechat_references.md`。

## ADDED Requirements

### Requirement: VerseTorch 张量与 Autograd 引擎

系统 SHALL 提供一个纯 Python + NumPy 实现的张量类型 `verse_torch.Tensor`，支持动态计算图与反向模式自动微分。

#### Scenario: 基本前向与反向
- **WHEN** 用户执行 `x = Tensor([1.0, 2.0, 3.0], requires_grad=True); y = (x * x).sum(); y.backward()`
- **THEN** `x.grad` 应等于 `[2.0, 4.0, 6.0]`，与 PyTorch 数值一致到 1e-6。

#### Scenario: broadcasting-aware 反向
- **WHEN** 形状 `(B, T, d)` 的张量与形状 `(d,)` 的偏置相加后求和并反向
- **THEN** 偏置梯度形状为 `(d,)`，且等于沿 broadcast 轴求和的结果。

#### Scenario: matmul 反向
- **WHEN** `C = A @ B` 其中 `A:(m,k)`, `B:(k,n)`
- **THEN** `A.grad = C.grad @ B.T`，`B.grad = A.T @ C.grad`，数值与 PyTorch 一致到 1e-5。

### Requirement: 优化器与训练栈

系统 SHALL 提供 SGD（含 momentum）、Adam、AdamW 优化器，以及 step/exponential/cosine 学习率调度器。

#### Scenario: Adam 训练收敛
- **WHEN** 在 MNIST 上用 VerseTorch + Adam 训练一个 2 层 MLP 5 个 epoch
- **THEN** 测试集准确率应 ≥ 95%。

### Requirement: CPU 量化与加速

系统 SHALL 提供 INT4/INT8/1.58-bit（ternary）权重量化能力，并在反量化-矩阵乘法 fused kernel 上获得相比 FP32 至少 2× 的吞吐提升。

#### Scenario: INT4 量化
- **WHEN** 将一个 FP32 Linear 层量化为 INT4 权重并执行推理
- **THEN** 输出与 FP32 输出的最大绝对差 ≤ 0.05 × 输出范数；推理 tokens/s 至少为 FP32 的 1.5×。

### Requirement: HuggingFace 权重加载兼容

系统 SHALL 能从 HuggingFace 标准格式（`.bin` / `.safetensors`）加载模型权重到 `verse_torch.Tensor`，无需用户安装 PyTorch（仅 `safetensors` 可选）。

#### Scenario: 加载 HF 模型
- **WHEN** 用户执行 `from verse_compat import load_hf_state_dict; sd = load_hf_state_dict("Qwen/Qwen2.5-0.5B")`
- **THEN** `sd` 应为 `dict[str, verse_torch.Tensor]`，键名与原 PyTorch state_dict 完全一致。

### Requirement: VerseNex 线性复杂度架构库

系统 SHALL 提供 Mamba-2、RWKV-7、Linear Attention（RetNet 风格）三种线性复杂度 Block 的纯 Python 实现，并支持训练与推理两种模式。

#### Scenario: Mamba-2 推理恒定内存
- **WHEN** 用 VerseNex 的 Mamba-2 Block 处理长度 1k 与 100k 的序列（推理模式）
- **THEN** 单步解码的峰值内存应基本一致（差 ≤ 10%），与上下文长度无关。

#### Scenario: RWKV-7 训练并行
- **WHEN** 用 VerseNex 的 RWKV-7 Block 在长度 4k 的序列上执行前向 + 反向
- **THEN** 训练内存应 O(N) 而非 O(N²)；梯度数值与参考实现一致到 1e-4。

### Requirement: Hybrid Block（SSM + Sparse Attention）

系统 SHALL 提供可配置层数比例的 Hybrid Block，结合 SSM 层（短程）与 Sparse Attention 层（长程），参考 RWKV-X / Nemotron-H 设计。

#### Scenario: 长上下文 passkey 检索
- **WHEN** 用 Hybrid Block（90% SSM + 10% Sparse Attention）构建 350M 模型，在 64k passkey 检索任务上评测
- **THEN** 准确率应 ≥ 90%。

### Requirement: VerseAWM 世界模型包

系统 SHALL 提供 JEPA（I-JEPA / V-JEPA 风格）与 RSSM（Dreamer 风格）两种世界模型实现。

#### Scenario: JEPA 表征预测训练
- **WHEN** 用 `verse_awm.JEPA` 在 CIFAR-10 上进行自监督预训练 50 epoch
- **THEN** 线性探针准确率应 ≥ 60%（参考 I-JEPA 在 ImageNet 上的设置，CIFAR-10 上门槛适当放宽）。

#### Scenario: RSSM 视频预测
- **WHEN** 用 `verse_awm.RSSM` 在 Moving MNIST 上训练 100 epoch
- **THEN** 模型应能生成 10 帧未来帧，MSE ≤ 0.02。

### Requirement: 仓库目录结构与多包布局

仓库根目录 SHALL 遵循以下结构：

```
/workspace/
├── packages/
│   ├── verse_torch/
│   ├── verse_nex/
│   ├── verse_awm/
│   ├── verse_tokenizer/
│   ├── verse_inference/
│   └── verse_compat/
├── datasets/
│   ├── raw/
│   ├── cleaned/
│   └── tokenizer/
├── docs/
│   ├── papers/
│   ├── architecture/
│   └── benchmarks/
├── verse_data/
│   ├── designs/
│   ├── experiments/
│   └── migration_notes/
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

#### Scenario: 多包可编辑安装
- **WHEN** 执行 `pip install -e packages/verse_torch packages/verse_nex packages/verse_awm`
- **THEN** `python -c "import verse_torch, verse_nex, verse_awm; print('ok')"` 应成功输出 `ok`。

### Requirement: 纯 CPU 端到端示例

系统 SHALL 提供至少三个可运行示例：
1. `examples/mnist_mlp.py`：VerseTorch + Adam 训练 MNIST MLP；
2. `examples/minimal_lm.py`：VerseNex Mamba-2 + VerseTorch 训练一个微型字符级 LM；
3. `examples/cpu_inference_demo.py`：加载量化后的小模型（≤ 1B 参数），在纯 CPU 上完成 100 tokens 生成。

#### Scenario: CPU 推理可运行
- **WHEN** 在 4 核 CPU、16GB RAM 的机器上运行 `examples/cpu_inference_demo.py`
- **THEN** 应在 5 分钟内完成模型加载 + 100 tokens 生成，峰值 RSS ≤ 8GB。

## MODIFIED Requirements

### Requirement: 仓库根（README + LICENSE）

原仓库根仅包含 LICENSE 与 README.md（无内容描述）。修改为：README.md 描述 Verse 框架整体目标、三包定位、安装方式、最小示例；LICENSE 保持不变。

## REMOVED Requirements

无（绿地项目，无既有需求被移除）。
