# 论文与工程参考资料

> 本文件收录 Verse 框架调研过程中已收集的论文与工程参考资料。
> 微信公众号文章（共 20 篇）见同目录下 [`wechat_references.md`](wechat_references.md)。
>
> 最后更新：2026-07-20

## 论文（arXiv / 会议）

### 线性 / 亚二次架构综述

| # | 标题 | 出处 | arXiv / 链接 | 与 Verse 的关系 |
|---|---|---|---|---|
| 1 | The End of Transformers? Sub-Quadratic Architectures | arXiv:2510.05364 | [arxiv.org/html/2510.05364v1](https://arxiv.org/html/2510.05364v1) | 线性/亚二次架构全景综述，为 VerseNex 架构选型提供理论支撑 |

### Mamba / SSM 系列

| # | 标题 | 出处 | arXiv / 链接 | 与 Verse 的关系 |
|---|---|---|---|---|
| 2 | LongMamba: Enhancing Mamba's Long Context Capabilities | ICLR 2025 | [jankautz.com/publications/LongMamba_ICLR25.pdf](https://www.jankautz.com/publications/LongMamba_ICLR25.pdf) | Mamba-2 长上下文增强，参考其设计提升 VerseNex 在 64k+ 上下文的表现 |
| 3 | On the Length Generalization of Mamba | NeurIPS 2025 | [papers.nips.cc/.../1bfc9f74afa91b9b8add5a97a97001a1-Paper-Conference.pdf](https://papers.nips.cc/paper_files/paper/2025/file/1bfc9f74afa91b9b8add5a97a97001a1-Paper-Conference.pdf) | Mamba 长度外推分析，影响 VerseNex 位置编码与训练长度策略 |
| 4 | Characterizing State Space Model and Hybrid Language Model Performance with Long Context | arXiv:2507.12442 | [arxiv.org/html/2507.12442v4](https://arxiv.org/html/2507.12442v4) | SSM/Hybrid LM 长上下文性能刻画，为 Hybrid Block 比例选型提供依据 |
| 5 | Achilles' Heel of Mamba | NeurIPS 2025 Spotlight, arXiv:2509.17514 | [arxiv.org/html/2509.17514v1](https://arxiv.org/html/2509.17514v1) | 揭示 Mamba 的结构性弱点，指导 VerseNex 在弱点场景下补充 Sparse Attention |

### RWKV / Hybrid 系列

| # | 标题 | 出处 | arXiv / 链接 | 与 Verse 的关系 |
|---|---|---|---|---|
| 6 | RWKV-X: A Linear Complexity Hybrid Language Model | arXiv:2504.21463 | [arxiv.org/html/2504.21463v2](https://arxiv.org/html/2504.21463v2) | SSM + Sparse Attention 混合架构，直接对应 VerseNex 的 Hybrid Block 设计 |
| 7 | Nemotron-H: Hybrid Mamba-Transformer Models | NVIDIA, 2025-03 | （NVIDIA 技术报告，未公开 arXiv 编号） | 工业 Hybrid 模型参考，验证 90% SSM + 10% Attention 的工程可行性 |

## 工程参考

| # | 项目 | 链接 | 用途 |
|---|---|---|---|
| 1 | llama.cpp | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | CPU 端侧 LLM 推理事实标准；ggml 张量库设计可参考 |
| 2 | BitNet.cpp | [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet) | 1.58-bit ternary CPU 推理；VerseTorch 量化子系统直接对标 |
| 3 | lm.c | [github.com/oderoi/lm.c](https://github.com/oderoi/lm.c) | 极简 LM 训练参考；最小化依赖示范 |
| 4 | tinygrad | [docs.tinygrad.org](https://docs.tinygrad.org/) | 轻量 autograd 引擎参考；多后端抽象可借鉴（详见 [ADR-001](../architecture/adr-001-cpu-first.md) 中"方案 B"的否决理由） |
| 5 | PureML | [joss.theoj.org/papers/10.21105/joss.09631.pdf](https://joss.theoj.org/papers/10.21105/joss.09631.pdf) | 纯 Python 深度学习框架的 JOSS 论文；学术可引用性参考 |
| 6 | micrograd | [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd) | Karpathy 的教学级 autograd；VerseTorch `Tensor` 类的设计灵感来源之一 |

## 世界模型

| # | 资源 | 链接 | 用途 |
|---|---|---|---|
| 1 | V-JEPA 2 | [aiwiki.ai/wiki/v_jepa_2](https://aiwiki.ai/wiki/v_jepa_2) | Meta 视频世界模型最新版本；VerseAWM V-JEPA 实现的直接参考 |
| 2 | JEPA Deep Dive | [dgallitelli.github.io/blog/world-models-series/part2-jepa-deep-dive/](https://dgallitelli.github.io/blog/world-models-series/part2-jepa-deep-dive/) | JEPA 架构深度解读；I-JEPA / V-JEPA 实现细节参考 |
| 3 | Beyond Next-Token Prediction: World Models and JEPA | [ai.briqmind.com/research_en/14-world-models-jepa/](https://ai.briqmind.com/research_en/14-world-models-jepa/) | 世界模型 vs next-token prediction 路线对比；VerseAWM 设计哲学参考 |
| 4 | A Path Towards Autonomous Machine Intelligence | Yann LeCun, 2022 | （OpenReview / Meta AI 公开论文） | JEPA 路线的奠基性论文；VerseAWM 非生成式世界模型的理论基础 |
