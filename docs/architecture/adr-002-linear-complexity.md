# ADR-002: 线性复杂度架构选型

- **状态**：Accepted
- **日期**：2026-07-20
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **前置 ADR**：[ADR-001: CPU 优先设计决策](file:///workspace/docs/architecture/adr-001-cpu-first.md)
- **后续 ADR**：[ADR-003: 世界模型路线选型](file:///workspace/docs/architecture/adr-003-world-model-route.md)
- **相关规范**：[`/workspace/.trae/specs/build-verse-framework/spec.md`](../../../.trae/specs/build-verse-framework/spec.md)
- **相关设计草稿**：[`ssm_scan_design.md`](file:///workspace/verse_data/designs/ssm_scan_design.md)

## 上下文

Transformer 的自注意力机制复杂度为 **O(N²·d)**（N 为序列长度，d 为头维度），在长上下文场景下计算与内存开销急剧膨胀：

| 序列长度 N | 注意力矩阵规模 | FP32 显存（仅 attention scores） |
|---|---|---|
| 4 k | 16 M | 64 MB |
| 32 k | 1.024 G | 4 GB |
| 128 k | 16.4 G | 64 GB |
| 1 M | 1 T | 4 TB |

在端侧 CPU 场景下（参考 [ADR-001](file:///workspace/docs/architecture/adr-001-cpu-first.md) 的"4 核 CPU、16 GB RAM"基准），32k 以上序列的 O(N²) 注意力已经 **不可接受**：
- 内存：16 GB RAM 无法容纳 64k 的 attention scores；
- 算力：CPU 单核 ~10 GFLOPS，128k 序列的一次 attention 前向需要约 10^12 FLOPS，耗时数十秒；
- 流式生成：自回归解码时 KV cache 线性增长，长对话场景下内存占用不可控。

**结论**：Transformer 的 O(N²) 注意力与 Verse 的 CPU 优先、长上下文、流式推理目标存在根本冲突，必须引入 **线性复杂度或近线性复杂度** 架构。

## 候选方案

### 方案 1：Sparse Attention（稀疏注意力）

**描述**：保留 Transformer 注意力机制，但每个 token 只 attend 到部分位置（如 top-k chunk、滑动窗口、全局 token）。
- **复杂度**：O(N·k·d)，其中 k 为每个 token 关注的位置数（k≪N 时近线性）
- **代表工作**：Longformer、BigBird、Mistral Sliding Window Attention、RWKV-X 的 top-k chunk sparse
- **优点**：
  - 保留注意力机制的可解释性与长程检索能力；
  - 易于在 PyTorch / 自研框架上实现，工程成熟度高；
  - 可微，训练稳定性好。
- **缺点**：
  - 仍是 O(N·k)，当 k 较大时仍有开销；
  - 推理时需要维护 KV cache（虽然稀疏，但状态大小与 N 成线性关系）；
  - 不天然支持 O(1) 内存的流式生成。

### 方案 2：Linear Attention（线性注意力）

**描述**：通过核函数 φ(·) 将 softmax 注意力改写为 `φ(Q)·(φ(K)^T·V)`，使得复杂度从 O(N²·d) 降到 O(N·d²)。
- **代表工作**：Linear Transformer（Katharopoulos et al.）、RetNet（Microsoft）、RWKV-4/5/6 的 time_mix 雏形
- **优点**：
  - 训练时可并行（O(N·d²)），推理时可递归（O(1) 每步）；
  - 与 Transformer 范式接近，迁移成本低；
  - 数学理论清晰。
- **缺点**：
  - 表达能力弱于 softmax attention，下游任务表现有差距；
  - 长程建模能力受核函数选择影响；
  - 工程上"线性"仍依赖 d² 项，d 较大时优势不明显。

### 方案 3：SSM（State Space Model，Mamba-2）

**描述**：基于状态空间模型的线性递归 `h_t = A·h_{t-1} + B·x_t`，通过 selective scan（输入相关的 A、B、C、Δ）获得表达能力。Mamba-2 进一步引入 SSD（State Space Duality），揭示 SSM 与 attention 的等价关系。
- **复杂度**：训练 O(N·d·state)，推理 O(1) 每步
- **代表工作**：Mamba、Mamba-2、Jamba
- **优点**：
  - 真正的线性复杂度（训练）与 O(1) 内存（推理）；
  - 长程建模能力强（selective 机制可让信息跨数千 token 流动）；
  - 推理时只需固定大小的 state，无 KV cache 膨胀；
  - 数值稳定，训练友好。
- **缺点**：
  - selective scan 在 NumPy 上需要 reshape 技巧或显式循环；
  - 与 Transformer 生态差异较大（无 attention、无 KV cache），需要重新设计推理引擎；
  - 单独使用时长程"精确检索"能力不如 attention（passkey retrieval 上 SSM 略弱）。

### 方案 4：RWKV-7

**描述**：RWKV 系列的最新版本，结合 Linear Attention 与 RNN 的优点，采用 per-channel decay（`w = -softplus(...)`）与 time_mix/channel_mix 双结构。
- **复杂度**：训练 O(N·d)，推理 O(1) 每步
- **代表工作**：RWKV-4/5/6/7
- **优点**：
  - 纯线性复杂度，CPU 友好；
  - time_mix 与 channel_mix 解耦，工程上易于优化；
  - 状态可持久化（wkvState 跨会话保存）；
  - 训练稳定性与 Mamba-2 相当。
- **缺点**：
  - 表达能力相对受限（参数化比 Mamba-2 更受限）；
  - 生态较新，社区支持不如 Mamba-2 成熟；
  - 单独使用时长程检索能力同样弱于 attention。

### 方案 5：Hybrid（SSM + Sparse Attention 混合）

**描述**：大部分层用 SSM（短程建模、线性复杂度），少量层用 Sparse Attention（长程检索），通过层比例配置平衡效率与能力。参考 RWKV-X / Nemotron-H / Samba。
- **复杂度**：取决于 sparse_ratio，常见配置（10%~25% sparse）下总体接近 O(N·d·state)
- **优点**：
  - 兼顾 SSM 的效率与 attention 的检索能力；
  - 在 passkey retrieval、long-context QA 等需要精确检索的任务上明显优于纯 SSM；
  - 配置灵活，可按任务需求调整 SSM:Attention 比例；
  - 工程上无需重新设计训练 / 推理路径，与纯 SSM 共享代码。
- **缺点**：
  - 实现复杂度高于纯 SSM；
  - 仍需维护 sparse attention 的 KV cache（虽然稀疏）；
  - 超参数（sparse_ratio、sparse_placement）需要根据任务调优。

## 决策

**全部实现，由 `HybridLM` 统一封装，用户可按需配置 SSM:Attention 比例。**

具体含义：

1. **全部 5 种架构都实现**，作为 `verse_nex` 包的可选组件：
   - Sparse Attention：[`TopKChunkSparseAttention`](file:///workspace/packages/verse_nex/verse_nex/sparse_attention.py)（top-k chunk sparse，参考 RWKV-X）
   - Linear Attention：[`LinearAttention`](file:///workspace/packages/verse_nex/verse_nex/linear_attention.py)（RetNet 风格，parallel/recurrent/chunkwise 三模式）
   - Mamba-2：[`Mamba2Block`](file:///workspace/packages/verse_nex/verse_nex/mamba2.py)（selective scan，A_log 参数化）
   - RWKV-7：[`RWKV7Block`](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py)（time_mix + channel_mix + FFN）
   - Hybrid：[`HybridBlock`](file:///workspace/packages/verse_nex/verse_nex/hybrid.py) + [`HybridLM`](file:///workspace/packages/verse_nex/verse_nex/hybrid.py)（统一封装）

2. **统一接口 `HybridLM`**：所有架构通过同一个 LM 类对外暴露，用户只需通过 `ssm_kind` 与 `sparse_ratio` 两个超参数即可切换架构：
   ```python
   from verse_nex import HybridLM

   # 纯 Mamba-2
   model = HybridLM(vocab_size=256, dim=128, n_layers=4, ssm_kind="mamba2", sparse_ratio=0.0)

   # 纯 RWKV-7
   model = HybridLM(vocab_size=256, dim=128, n_layers=4, ssm_kind="rwkv7", sparse_ratio=0.0)

   # 混合（默认 10% sparse attention，均匀分布）
   model = HybridLM(vocab_size=256, dim=128, n_layers=4, ssm_kind="mamba2", sparse_ratio=0.1)
   ```

3. **默认配置**：`sparse_ratio=0.1`（90% SSM + 10% Sparse Attention），`sparse_placement="spread"`（均匀分布）。理由：
   - 10% sparse attention 足以恢复 passkey retrieval 等长程检索能力；
   - 均匀分布（spread）比集中在最后几层（last）更稳定，避免长程信息在最后才被检索；
   - 默认 `ssm_kind="mamba2"`（生态更成熟，数值验证更充分）。

4. **统一的双模式前向**：
   - `forward_parallel(x)`：训练模式，整序列并行计算；
   - `forward_recurrent(input_ids, states)`：推理模式，单步递推，O(1) 内存；
   - 数值一致性已验证：parallel vs recurrent 在所有架构上 ≤ 1e-6（详见 [ssm_scan_design.md](file:///workspace/verse_data/designs/ssm_scan_design.md)）。

5. **状态持久化**：所有架构的 recurrent 状态都通过 [`StateCache`](file:///workspace/packages/verse_inference/verse_inference/cache.py) 统一管理，支持跨会话保存 / 恢复，为流式生成器 [`StreamingGenerator`](file:///workspace/packages/verse_inference/verse_inference/generator.py) 提供 O(1) 内存的推理基础。

## 后果

### 优点

- **灵活性**：用户可在同一 API 下切换 5 种架构，无需修改模型代码、训练循环或推理引擎；
- **CPU 友好**：所有架构都是线性或近线性复杂度，可在 4 核 CPU、16 GB RAM 上跑通 32k+ 序列；
- **流式推理**：recurrent 模式下 O(1) 内存，适合端侧长对话 / 长文档场景；
- **生态对齐**：默认 Hybrid 配置与业界主流（Nemotron-H 8B、Jamba 12B、RWKV-X）方向一致；
- **可扩展**：未来新增架构（如 Mamba-3、RWKV-8、Lightning Attention）只需实现 `forward_parallel` / `forward_recurrent` 接口即可接入 HybridLM。

### 缺点

- **实现复杂度**：5 种架构 + Hybrid 封装共 6 个核心模块，代码量约为纯 Transformer 的 3 倍；
- **超参数增多**：`ssm_kind`、`sparse_ratio`、`sparse_placement`、`ssm_kwargs`、`sparse_kwargs` 共 5 个关键超参数，对新手不友好；
- **数值验证成本**：每种架构都需要 parallel vs recurrent 的一致性检查（已通过，详见下表）；
- **生态迁移成本**：从 PyTorch + Transformer 迁移到 HybridLM 需要重新理解 SSM 状态、sparse 配置（参考 [pytorch_to_versetorch.md](file:///workspace/verse_data/migration_notes/pytorch_to_versetorch.md)）。

### 数值一致性验证结果（来自 [checklist.md](file:///workspace/.trae/specs/build-verse-framework/checklist.md) 阶段 3）

| 架构 | parallel vs recurrent 一致性 | 阈值 | 结论 |
|---|---|---|---|
| Linear Attention (RetNet) | 6.59e-07 | 1e-6 | PASS |
| Mamba-2 SSD | 8.94e-08 | 1e-6 | PASS |
| RWKV-7 | 2.38e-07 | 1e-6 | PASS |
| Sparse Attention | 3.49e-07 | 1e-6 | PASS |
| HybridLM | 1.88e-07 | 1e-6 | PASS |

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 纯 SSM 在 passkey retrieval 上准确率不足 | Hybrid 配置（10% sparse attention）显著改善；阶段 3 已用 121K 参数小模型做结构验证，loss 从 54 降到 6，Q/K/V 梯度路径已修复 |
| 长序列内存膨胀（recurrent 模式） | `StateCache` 固定大小，1k vs 10k 序列实测 RSS 差 0 KB / 0.00% |
| 架构选择困难症 | 默认 Hybrid（`sparse_ratio=0.1`）覆盖 80% 场景；用户可通过 [`ModelLoader`](file:///workspace/packages/verse_inference/verse_inference/model_loader.py) 的 `arch="mamba2"/"rwkv7"/"hybrid"` 一键切换 |
| 训练吞吐量低于 GPU | 阶段 0–1 不做大规模预训练；CPU 上只做算法验证与端侧推理（参考 [ADR-001](file:///workspace/docs/architecture/adr-001-cpu-first.md)） |

## 替代方案（已否决）

### 方案 A：只实现 Transformer + Sliding Window

**描述**：不引入 SSM，只在 Transformer 基础上加滑动窗口注意力。

**否决理由**：
- **长程建模能力弱**：滑动窗口无法跨窗口检索，passkey retrieval 准确率显著下降；
- **KV cache 仍线性增长**：流式推理时内存不可控，与 Verse 的 O(1) 内存目标冲突；
- **CPU 性能不足**：即使滑动窗口，O(N·k·d) 在 k=512 时仍比 SSM 的 O(N·d·state) 慢 5–10 倍。

### 方案 B：只实现 Mamba-2

**描述**：放弃 RWKV-7、Sparse Attention、Hybrid，只保留 Mamba-2 作为唯一架构。

**否决理由**：
- **长程检索能力不足**：纯 Mamba-2 在 passkey retrieval、long-context QA 等任务上准确率明显低于 Hybrid；
- **架构选择受限**：不同任务可能需要不同架构（如 RWKV-7 在某些任务上优于 Mamba-2），单一架构无法覆盖；
- **生态风险**：Mamba-2 仍处于快速迭代期，单一架构押注风险高。

### 方案 C：基于 Flash Attention / Triton

**描述**：不引入新架构，优化 Transformer 注意力的实现（如 Flash Attention、Triton kernel）。

**否决理由**：
- **CPU 不支持**：Flash Attention 依赖 GPU shared memory，在 CPU 上无加速效果；
- **仍是 O(N²)**：即使常数项优化，长上下文下内存与算力仍不可接受；
- **依赖重型工具链**：Triton 需要独立编译器栈，与"零重型依赖"目标冲突（参考 [ADR-001](file:///workspace/docs/architecture/adr-001-cpu-first.md)）。

### 方案 D：基于 RetNet 单一架构

**描述**：只用 RetNet（Linear Attention 变体）作为唯一架构。

**否决理由**：
- **表达能力弱于 Mamba-2**：selective 机制让 Mamba-2 在长程建模上明显优于固定参数的 RetNet；
- **下游任务表现一般**：RetNet 在公开 benchmark 上表现不及 Mamba-2 / RWKV-7；
- **社区生态不成熟**：相比 Mamba-2 / RWKV-7，RetNet 的开源实现与预训练权重较少。

## 备注

- 本 ADR 与 [ADR-001](file:///workspace/docs/architecture/adr-001-cpu-first.md) 的 CPU 优先决策一致：所有架构都在 NumPy 上实现正确版本，GPU 后端延后。
- 本 ADR 的实现细节见 [`ssm_scan_design.md`](file:///workspace/verse_data/designs/ssm_scan_design.md)（SSM scan 设计草稿）与 [`hybrid.py`](file:///workspace/packages/verse_nex/verse_nex/hybrid.py) 源码。
- 后续 ADR-003 将决策世界模型（JEPA / RSSM / H-JEPA）路线，与本 ADR 的架构选型互补。
- 相关工程参考：[Mamba-2](https://arxiv.org/abs/2405.21060)、[RWKV-7](https://arxiv.org/abs/2503.14456)、[Nemotron-H](https://arxiv.org/abs/2505.00861)、[Jamba](https://arxiv.org/abs/2403.19887)、[RetNet](https://arxiv.org/abs/2307.08621)。

## 演进更新（Part4K1）

本 ADR 选型的 Mamba-2 / RWKV-7 / Hybrid 路线已标记为 **deprecated**（`HybridBlock` / `HybridLM` 保留只读兼容）。Part4K1 起 `config.yml` 的 `arch` 字段仅保留 `versenex` 唯一值（`transformer` / `hybrid` 自动映射 + `DeprecationWarning`）。

CometSpark V0.5-1B 采用 **VerseNex 原生架构**（`TriSparseAttention` + `MoDLayer`），不依赖 SSM：

- **TriSparseAttention**：SWA + Global sink + ALiBi 三路并行稀疏注意力（详见 [ADR-008: 超稀疏并行注意力](adr-008-parallel-sparse-attention.md)）。
- **MoDLayer**：5 DensePart × 8 Expert × top-3 双层门控 Mixture-of-Depths。

本 ADR 中 SSM 相关的 parallel/recurrent 双模式设计仍被 VerseNex 架构继承（`forward_parallel` / `forward_recurrent`），但底层 block 不再是 Mamba/RWKV。

## 演进更新（Part4K2）

本 ADR 的架构选型在 Part4K2 保持不变（VerseNex 原生架构仍为主线）。Part4K2 的以下能力为 VerseNex 架构提供了工程化支撑：

- **压缩技术 V1.3**（[ADR-012](adr-012-compression-v13.md)）：`CometSparkNexLM.compress_v13()` / `distill_from(teacher, train_data)` 实例方法，支持对 VerseNex 架构模型进行剪枝 + 量化 + 蒸馏 + LoRA 压缩，大模型（1B）→ 小模型（280M INT4）能力转移。
- **智能分区训练**（[ADR-011](adr-011-layerwise-training.md)）：`LayerWiseTrainer` 按 `model.blocks` 分组训练，VerseNexLM / CometSparkNexLM 的 `ModuleList` 结构天然适配分区训练。
- **.vn 文件格式**（[ADR-009](adr-009-vn-format.md)）：VerseNex 模型的 `config.yml` `arch` 字段统一为 `versenex`，`.vn` 容器的 `meta.json` 记录该架构名，工具链可据此路由加载逻辑。
