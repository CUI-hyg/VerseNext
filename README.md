# VerseNext

> **VerseNext** —— 纯 Python / 纯 CPU 的深度学习与大语言模型框架（VerseTorch + VerseNex + VerseAWM），不强制依赖 PyTorch / Transformers，可在消费级 CPU、嵌入式设备与树莓派上开箱即用。

VerseNext 的目标是用 **线性复杂度架构（SSM / Mamba / RWKV / Linear Attention）** 替代或混合 Transformer，根治自注意力 O(N²) 与 KV Cache 线性膨胀问题；并为下一代 **世界模型（JEPA、RSSM、H-JEPA）** 与端侧高能力 LLM 提供原生支撑，同时尽可能兼容 HuggingFace / PyTorch 生态以降低迁移成本。

## 三包定位

| 包 | PyPI 名 | 定位 | 关键能力 |
|---|---|---|---|
| **VerseTorch** | `verse-torch` | 纯 Python + NumPy 的张量与自动微分引擎（PyTorch 替代） | `Tensor` 类、动态计算图、反向模式 autograd、`nn.Module`、优化器栈（SGD/Adam/AdamW）、INT4/INT8/1.58-bit 量化、可读 PyTorch `state_dict` |
| **VerseNex** | `verse-nex` | Transformer 替代架构库（线性复杂度优先） | Mamba-2 selective scan、RWKV-7 time/channel mixing、RetNet 风格 Linear Attention、SSM + Sparse Attention Hybrid Block、RoPE/ALiBi/NoPE |
| **VerseAWM** | `verse-awm` | 世界模型专用包（Autonomous World Model） | I-JEPA / V-JEPA 潜在空间预测、RSSM（Dreamer 风格）、H-JEPA 层次化规划、EMA target encoder、energy-based loss |

辅助包：

| 包 | PyPI 名 | 定位 |
|---|---|---|
| `verse-tokenizer` | `verse-tokenizer` | 轻量 BPE/Unigram 分词器，无 `tokenizers` / `sentencepiece` 重依赖时仍可运行 |
| `verse-inference` | `verse-inference` | 模型加载、KV/状态缓存、流式生成、OpenAI 兼容 HTTP server（可选 FastAPI） |
| `verse-compat` | `verse-compat` | HuggingFace `transformers` / `torch` 兼容适配层（仅在用户已安装时启用） |

## 安装

需要 Python ≥ 3.10、NumPy ≥ 1.26。本仓库采用 uv/pip workspace 多包布局，每个 package 都是独立的可编辑安装单元。

```bash
# 1) 克隆仓库
git clone <repo-url> verse && cd verse

# 2a) 方式一：pip 可编辑安装（按需选择包）
pip install -e packages/verse_torch packages/verse_nex packages/verse_awm

# 2b) 方式二：uv workspace 一次性安装全部成员
uv sync

# 3) 可选运行时依赖（按需安装）
pip install "verse-nex[speed]"  # 安装 numba 加速 selective scan（推荐）
pip install "safetensors>=0.4"   # 加载 .safetensors 权重
pip install "fastapi>=0.110"     # OpenAI 兼容 HTTP server
```

> **numba 加速说明**：`verse-nex[speed]` 会安装 `numba>=0.60`，对 Mamba-2 / Hybrid 的 selective scan 递推循环做 JIT 编译，recurrent 模式生成吞吐量提升约 1.8× ~ 3.2×。numba 是可选依赖——不安装也能运行，只是 `@njit` 装饰器退化为 no-op。详见 [性能调优指南](docs/performance_tuning.md)。

## 最小示例

```python
from verse_torch import Tensor

x = Tensor([1.0, 2.0], requires_grad=True)
y = (x * x).sum()        # 1 + 4 = 5
y.backward()
print(y)                 # 5.0
print(x.grad)            # [2. 4.]  与 PyTorch 一致
```

实测：上述代码与 PyTorch 数值一致到 1e-6（已通过 377 项单元测试 + 有限差分梯度检查）。

更多示例见 [`examples/`](examples/)：

- [`mnist_mlp.py`](examples/mnist_mlp.py) —— MNIST MLP，5 epoch 准确率 97.66%
- [`minimal_lm.py`](examples/minimal_lm.py) —— 字符级 LM（Mamba-2 backbone），parallel vs recurrent 一致
- [`jepa_demo.py`](examples/jepa_demo.py) —— I-JEPA 自监督预训练
- [`cpu_inference_demo.py`](examples/cpu_inference_demo.py) —— 纯 CPU 流式生成（715 tokens/s，峰值 RSS 44.5MB）

## 关键技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 张量后端 | **NumPy** + 可选 Numba/Cython | CPU 优先、零重型依赖、用户已可安装、跨平台 |
| Autograd | 反向模式 VJP、动态计算图 | 与 PyTorch API 一致、易于审计与调试 |
| 主推架构 | **Mamba-2 + RWKV-7 + Hybrid**（SSM + Sparse Attention） | 已公开、工业验证（MiniMax-01、混元 T1、Nemotron-H、RWKV-X） |
| 量化默认 | **INT4 (W4A16) + 1.58-bit ternary** | BitNet.cpp 在 CPU 上已验证 6.17× 提速；端侧友好 |
| 兼容策略 | 初期可读 PyTorch `state_dict`，运行时无 PyTorch 依赖 | 生态友好但运行时零依赖 |
| 世界模型主线 | **JEPA（非生成式）+ RSSM（生成式）** | 兼顾 LeCun 路线与 Dreamer 路线 |
| GPU 后端 | 阶段 0–1 不支持，延后到后续 spec | 先在 CPU 上正确实现并优化；详见 [ADR-001](docs/architecture/adr-001-cpu-first.md) |

详细架构决策记录见 [`docs/architecture/`](docs/architecture/)。

## 详细文档

### 训练指南

[Verse 训练指南](docs/training_guide.md) —— 从零训练 LM 的完整流程：数据准备 → tokenizer → 模型 → 训练 → 评估 → 压缩 → 推理。

### 性能调优

[Verse 性能调优指南](docs/performance_tuning.md) —— numba JIT 加速、BLAS 配置、batch_size 选择、CPU 线程数、量化加速、并行计算六个维度的 CPU 调优手册。

### 各包文档

| 包 | 文档 | 定位 |
|---|---|---|
| VerseTorch | [README](packages/verse_torch/README.md) | 张量 / autograd / nn / optim / losses / training / quantize / parallel / compress |
| VerseNex | [README](packages/verse_nex/README.md) | Mamba-2 / RWKV-7 / RetNet / Sparse Attention / Hybrid / 位置编码 |
| VerseAWM | [README](packages/verse_awm/README.md) | I-JEPA / V-JEPA / H-JEPA / RSSM 世界模型 |
| VerseTokenizer | [README](packages/verse_tokenizer/README.md) | BPE / Byte / Char 分词器 |
| VerseInference | [README](packages/verse_inference/README.md) | 模型加载 / 状态缓存 / 流式生成 / HTTP server |
| VerseCompat | [README](packages/verse_compat/README.md) | HuggingFace / PyTorch 兼容层 |

### 设计文档

- [压缩管线设计（trillion → billion 路线图）](verse_data/designs/compression_pipeline_design.md)
- [Autograd 设计](verse_data/designs/autograd_design.md)
- [SSM scan 设计](verse_data/designs/ssm_scan_design.md)
- [JEPA EMA 设计](verse_data/designs/jepa_ema_design.md)
- [PyTorch 迁移笔记](verse_data/migration_notes/pytorch_to_versetorch.md)

### 基准测试

- [CPU 并行基准](docs/benchmarks/parallel_benchmark.md)
- [模型压缩 PoC](docs/benchmarks/compression_poc.md)
- [量化基准](docs/benchmarks/quantize_benchmark.md)
- [v0.1 整体基准](docs/benchmarks/benchmark-v0.1.md)

## CometSpark 端到端训练仓库

[`data/demo/`](data/demo/) —— **CometSpark-v0.1** 是基于 VerseNext 的端到端 LM 训练仓库，纯 Python / 纯 CPU 一键训练，运行时无 PyTorch / TensorFlow / JAX / transformers 依赖。

```bash
cd data/demo && python run.py
```

详见 [`data/demo/README.md`](data/demo/README.md)。

### 第二次进化摘要

CometSpark-v0.1 是 Verse 框架"第二次进化"的产物，覆盖以下能力：

- **多步训练 + cross_entropy + loss 曲线**：`verse_torch.training` 提供 `Trainer` / `EarlyStopping` / `GradientAccumulator` / `CheckpointManager` / `LambdaLR`（warmup + cosine）/ `compute_loss_rate` / `plot_loss_curve`，自动保存 `best.pt` / `last.pt` / `loss_history.json` / `loss_curve.txt`。
- **BPE / ByteTokenizer**：`verse_tokenizer` 完善 `BPETokenizer.train` / `save` / `load` / `add_special_tokens`，新增 `ByteTokenizer`（vocab=259）与 `load_tokenizer(kind, path)` 工厂。
- **TransformerLM**：`verse_torch.nn` 补齐 `SwiGLUMLP` / `GQASelfAttention`（含 KV cache）/ `TransformerBlock` / `TransformerLM`（含 weight tying），支持 GQA 与 RoPE。
- **CPU 并行**：`verse_torch.parallel` 提供 `parallel_matmul` / `ParallelLinear` / `parallel_map`，对 batch >= 阈值启用 multiprocessing。
- **模型压缩 PoC**：`verse_torch.compress` 提供 `OutlierSafePruner` / `LoRALinear` / `KnowledgeDistiller` / `QLinear` / `compress_pipeline`，在 1M 参数 TransformerLM 上验证压缩比 ≥ 10×、loss 差异 ≤ 5%。
- **推理兼容**：`verse_inference` 新增 `cometspark` arch 分支，`StreamingGenerator` 兼容 CometSparkLM（100 tokens ≤ 5s）。

### Part3K2 重大升级摘要

Part3K2 在第二次进化基础上完成 7 大升级，详见 [审计报告](audit_report.md)：

- **训练数据格式现代化（BREAKING）**：`TextDataset` 支持 **chat 数组** 与 **prompt-completion** 双格式自动检测，loss mask 自动屏蔽 prompt 部分（`ignore_index=-100`），仅 completion 参与损失；旧版 `{"text":"..."}` 格式已废弃。
- **Tokenizer 全面升级**：新增 `preprocess.py`（GPT-4 风格正则预分词，中文整字独立成块 + NFKC 归一化 + UTF-8 边界修复）、`chat_template.py`（`render_chat` / `render_prompt` / `split_prompt_completion`）、`unigram.py`（SentencePiece Unigram，EM 训练 + Viterbi 解码）；`BPETokenizer` 接入预分词、`vocab_size` 自适应、`add_special_tokens` 编码开关。
- **对齐 PyTorch 能力**：新增 `Lion` / `Adafactor` 优化器、`OneCycleLR` / `ReduceLROnPlateau` / `CosineRestartsLR` 调度器、`GeGLU` / `Mish` / `SiLU` 激活、`SlidingWindowAttention` / `ALiBi` / `DeepNorm` 层、`focal_loss`，`cross_entropy` 支持 `ignore_index` + `label_smoothing`。
- **CometSpark 架构升级 + 压缩深度集成**：`CometSparkConfig` 新增 `rope_theta` / 分离 dropout / `max_position_embeddings`；`CometSparkLM` 新增 `from_pretrained` / `save_pretrained` / `compress(compress_config)` / `compression_stats()` 方法与 `CometSparkSmall/Medium/Large` 工厂函数。
- **并行训练 `ParallelTrainer`**：步数拆分为 N 个 chunk，合并策略「差前好后」串行重训 + 整体 fine-tune；**修复 val_loss 漏洞**（`_eval_full_val` 基于完整 val 数据集而非单 batch 更新）。
- **推理 + 自由温度 + 打分**：`Trainer.inference(prompts, temperature, top_k, top_p, max_tokens)` 批量生成；`ScoringEvaluator` 实现 `exact_match` / `prefix_accuracy` / `char_f1` / `bleu` / `rouge_l` 五指标；`run.py` 新增 `--score` / `--references-file` / `--top-p` / `--parallel-chunks` 参数。
- **全项目 check-loop 审计**：修复 sigmoid/silu/BCE overflow、硬编码路径等 6 项问题，全量测试 377 passed。

## 仓库结构

```
/workspace/
├── packages/
│   ├── verse_torch/        # 张量与 autograd 引擎
│   ├── verse_nex/          # 线性复杂度架构库
│   ├── verse_awm/          # 世界模型包
│   ├── verse_tokenizer/    # 轻量分词器
│   ├── verse_inference/    # 推理引擎
│   └── verse_compat/       # HF/PyTorch 兼容层
├── data/                   # CometSpark-v0.1 端到端 LM 训练仓库（demo 入口）
│   └── demo/               # run.py / model / train / config / checkpoints
├── datasets/               # raw / cleaned / tokenizer
├── docs/                   # papers / architecture / benchmarks
├── verse_data/             # designs / experiments / migration_notes（内部材料）
├── tests/                  # 单元测试 + 数值梯度检查 + 端到端用例
├── examples/               # MNIST / 最小 LM / CPU 推理 demo / JEPA demo
├── pyproject.toml          # workspace 级配置（PEP 621 + uv workspace）
└── README.md
```

## 参考资料链接

更完整的清单（含微信公众号文章）见 [`docs/papers/`](docs/papers/)。

### 论文（arXiv / 会议）

- [The End of Transformers? Sub-Quadratic Architectures](https://arxiv.org/html/2510.05364v1) —— 线性/亚二次架构综述
- [RWKV-X: A Linear Complexity Hybrid Language Model](https://arxiv.org/html/2504.21463v2) —— SSM + Sparse Attention 混合
- [LongMamba (ICLR 2025)](https://www.jankautz.com/publications/LongMamba_ICLR25.pdf) —— Mamba 长上下文增强
- [On the Length Generalization of Mamba (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/file/1bfc9f74afa91b9b8add5a97a97001a1-Paper-Conference.pdf)
- [Characterizing SSM/Hybrid LM Performance with Long Context](https://arxiv.org/html/2507.12442v4)
- [Achilles' Heel of Mamba (NeurIPS 2025 Spotlight)](https://arxiv.org/html/2509.17514v1)
- Nemotron-H: Hybrid Mamba-Transformer Models（NVIDIA, 2025-03）

### 工程参考

- [llama.cpp](https://github.com/ggml-org/llama.cpp) —— CPU 端侧 LLM 推理事实标准
- [BitNet.cpp](https://github.com/microsoft/BitNet) —— 1.58-bit ternary CPU 推理
- [lm.c](https://github.com/oderoi/lm.c) —— 极简 LM 训练参考
- [tinygrad](https://docs.tinygrad.org/) —— 轻量 autograd 引擎参考
- [PureML (JOSS paper)](https://joss.theoj.org/papers/10.21105/joss.09631.pdf)
- [micrograd](https://github.com/karpathy/micrograd) —— 教学级 autograd

### 世界模型

- [V-JEPA 2](https://aiwiki.ai/wiki/v_jepa_2) —— Meta 视频世界模型
- [JEPA Deep Dive](https://dgallitelli.github.io/blog/world-models-series/part2-jepa-deep-dive/)
- [Beyond Next-Token Prediction: World Models and JEPA](https://ai.briqmind.com/research_en/14-world-models-jepa/)
- LeCun, "A Path Towards Autonomous Machine Intelligence"（2022）

## License

[MIT License](LICENSE) © 2026 CometFuture.

---
