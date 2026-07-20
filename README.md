# Verse

> **Verse** —— 纯 Python / 纯 CPU 的深度学习与大语言模型框架（VerseTorch + VerseNex + VerseAWM），不强制依赖 PyTorch / Transformers，可在消费级 CPU、嵌入式设备与树莓派上开箱即用。

Verse 的目标是用 **线性复杂度架构（SSM / Mamba / RWKV / Linear Attention）** 替代或混合 Transformer，根治自注意力 O(N²) 与 KV Cache 线性膨胀问题；并为下一代 **世界模型（JEPA、RSSM、H-JEPA）** 与端侧高能力 LLM 提供原生支撑，同时尽可能兼容 HuggingFace / PyTorch 生态以降低迁移成本。

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
pip install "numba>=0.60"        # CPU GEMM 加速
pip install "safetensors>=0.4"   # 加载 .safetensors 权重
pip install "fastapi>=0.110"     # OpenAI 兼容 HTTP server
```

## 最小示例

> 阶段 0 仅完成脚手架，下面的 `Tensor` API 将在阶段 1（VerseTorch 核心引擎）落地。当前为占位伪代码，用以约定 API 形态。

```python
from verse_torch import Tensor

x = Tensor([1.0, 2.0], requires_grad=True)
y = (x * x).sum()        # 1 + 4 = 5
y.backward()
print(y)                 # Tensor(5.0)
print(x.grad)            # Tensor([2.0, 4.0])  与 PyTorch 一致
```

预期：在阶段 1 完成后，上述代码与 PyTorch 数值一致到 1e-6。

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

[MIT License](LICENSE) © 2026 CUI-hyg

---

## 历史版本

> 以下是本仓库 README 的原始内容，保留以备追溯。

```markdown
# VerseNext
下一代LM框架，注重低算力、高能力
by CometFuture.
```
