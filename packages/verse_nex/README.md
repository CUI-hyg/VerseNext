# VerseNex

> 中文定位：Transformer 替代架构库，主推线性复杂度（SSM/Mamba/RWKV/Linear Attention）+ Hybrid 混合架构 + VerseNex 原生架构（TriSparse + MoD），O(1) 推理状态 + O(N) 训练并行。

[返回主 README](../../README.md)

## 特性

- 线性复杂度：O(N) 训练 + O(1) 推理状态，适配长上下文。
- parallel / recurrent 双模式，float32 下输出吻合到 1e-3。
- 多种架构：Mamba-2 / RWKV-7 / RetNet / Sparse Attention / Hybrid（已 deprecated）/ VerseNex 原生架构。
- **Part4 新增：VerseNex 原生架构**：TriSparseAttention（三路并行稀疏注意力）+ MoD（多稠密分区）+ CometSparkNexLM（顶层 LM）。
- **Part4K1 新增**：
  - **品牌落地**：`TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention`、`VerseNexBlock` 统一为唯一名（旧名作为 `DeprecationWarning` 别名）
  - **MoD 完善**：`load_balance_loss` + `router z-loss` 稳定训练
  - **超稀疏并行注意力**：多 query chunk 并行 + `SpeculativeDecoder`（Medusa 风格）+ `ParallelKVCache`
  - **NexRL 强化学习**：五要素抽象（`NexAgent` / `NexEnv` / `NexState` / `NexAction` / `NexReward`）+ `NexTrainer`（PPO + GAE + KL 自适应）
- 位置编码：RoPE / ALiBi / NoPE，统一接口。
- 纯 Python + NumPy 友好，无重型深度学习框架硬依赖（GPU/NPU 通过 `verse_torch.device.DeviceBackend` 可选委托）。

## 安装

```bash
pip install -e packages/verse_nex
```

## 架构总览

### Mamba-2（selective state space）

- `Mamba2Block`：基于 SSD（state-space duality）算法的 selective SSM 块。
- 支持 parallel（整序列并行）/ recurrent（单步递推）双模式。
- 适合长上下文、低显存推理。

### RWKV-7（time / channel mixing）

- `RWKV7TimeMix`：时间混合层，类似 RNN 的 token-shift。
- `RWKV7ChannelMix`：通道混合层，门控 FFN。
- `RWKV7Block`：完整 RWKV-7 block（TimeMix + ChannelMix）。
- 并行训练 + 递归推理，状态缓存友好。

### RetNet（linear attention）

- `RetNet`：retention + chunkwise 实现，线性复杂度。
- 兼具 Transformer 训练并行与 RNN 推理高效。

### Sparse Attention

- `TopKChunkSparseAttention`：top-k chunk 稀疏注意力。
- 在 chunk 内全量 + 跨 chunk top-k，平衡全局感受野与计算成本。

### Hybrid Block / LM（Part4K1 deprecated）

> **Part4K1 标记 deprecated**：`HybridBlock` / `HybridLM` 保留只读兼容。新代码请用 `VerseNexLM`（TriSparse + MoD 已覆盖 Hybrid 的所有能力，且更高效）。`config.yml` 的 `arch: hybrid` 会自动映射为 `arch: versenex` + `DeprecationWarning`。

- `HybridBlock`：SSM（mamba2 / rwkv7）+ Sparse Attention 混合 block。
- `HybridLM`：完整 LM（Embedding → N × HybridBlock → LayerNorm → Head）。
- 支持 `forward_parallel` / `forward_recurrent` / `forward(mode=)` / `generate`。

### VerseNex 原生架构（Part4 新增）

Part4 在 Hybrid 基础上引入 **VerseNex 原生架构**——不依赖 SSM，纯注意力 + MoE 路线，专为 CPU 训练优化。

#### TriSparseAttention（三路并行稀疏注意力）

`TriSparseAttention` 是 VerseNex 的核心注意力机制，将注意力拆分为三路并行计算后用 sigmoid gate 融合：

- **SWA（Sliding Window Attention）**：chunk-wise 实现，不在内存中构造 T² 矩阵，复杂度 O(T·W)
- **Global Attention**：可学习的 sink token（默认 64 个），承载长程依赖，复杂度 O(T·N_global)
- **ALiBi（Attention with Linear Biases）**：基于位置的线性偏置，T ≤ 1024 直接构造，T > 1024 降级为 SWA-only
- **Gate 融合**：三路输出通过 sigmoid gate 加权融合：`out = σ(g_swa)·swa + σ(g_global)·global + σ(g_alibi)·alibi`
- **GQA 支持**：`n_kv_head < n_head` 时自动启用 Grouped Query Attention

适用场景：长上下文（T > 4096）、CPU 训练、低内存环境。

#### MoDLayer（Mixture of Dense Parts）

`MoDLayer` 灵感来源于人大脑的功能分区，将 FFN 拆分为多个 DensePart，每个 DensePart 下还有多个 Experts：

- **5 DensePart**：`general`（通用）/ `language`（语言）/ `math`（数理）/ `biochem`（生化）/ `code`（代码）
- **双层门控**：
  - `part_router`（soft routing）：所有 DensePart 都参与计算，权重通过 softmax 归一化
  - `expert_router`（hard routing，top-k）：每个 DensePart 内仅 top-k 个 Expert 被激活（默认 top-3）
- **Switch Transformer 风格 aux loss**（Part4K1 完善）：
  - `load_balance_loss`：负载均衡损失，避免 Expert 坍缩
  - `router z-loss`：router logits 平方和惩罚，稳定训练（防止 router 输出过大导致 softmax 饱和）
- **参数预算**：每个 Expert 是独立的 SwiGLU MLP（w_gate + w_up + w_down）

数学形式：

```
part_logits = part_router(x)                # (B, T, num_parts)
part_weights = softmax(part_logits / τ)     # soft routing
for each part p:
    expert_logits = expert_router[p](x)     # (B, T, num_experts)
    topk_idx, topk_w = topk(expert_logits, k=top_k)
    expert_out = sum(topk_w[i] * experts[p][topk_idx[i]](x) for i in range(k))
    out += part_weights[p] * expert_out
aux_loss = switch_transformer_aux_loss(part_logits, expert_logits)
```

#### CometSparkNexLM（顶层 LM）

`CometSparkNexLM` 是 VerseNex 原生架构的顶层语言模型，将 TriSparseAttention 与 MoDLayer 组合为完整 LM：

- **layer_pattern 驱动**：每层类型显式指定，例如 `["trisparse", "trisparse", "mod", "trisparse"]`
- **Pre-Norm + 残差**：`x = x + attn(norm1(x)); x = x + ffn(norm2(x))`
- **残差缩放**：`1/sqrt(2*n_layer)`，应用于 attn.proj 与 SwiGLU.w_down
- **三种前向模式**：
  - `forward(idx)` → logits：标准前向，推理用
  - `forward_with_aux(idx)` → `(logits, aux_loss)`：训练用，累加所有 MoD 层的 aux_loss
  - `forward_recurrent(input_ids, states)` → `(logits, new_states)`：流式生成用
- **generate**：支持 greedy + recurrent（temperature=1.0）与采样（temperature/top_k）两条路径
- **持久化**：`save(path)` / `load(path)` / `from_pretrained(path)` / `save_pretrained(dir_path)`

#### CometSpark-V0.2 工厂

`CometSparkV02()` 工厂函数一键构建 CometSpark-V0.2 模型：

```python
from verse_nex import CometSparkV02

model = CometSparkV02(vocab_size=151936)
# 32 层 VerseNex（8 MoD + 24 trisparse）
# d_model=384, n_head=8, n_kv_head=4
# 约 0.5B 参数
print(model.count_parameters())  # ≈ 537,591,264
```

层模式生成：`_build_v02_pattern(n_layer=32, mod_every=4)` → `["mod", "trisparse", "trisparse", "trisparse"] × 8`（共 8 MoD + 24 trisparse）。

### VerseNexLM 重命名（Part4K1）

Part4K1 完成品牌落地：`TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention`、`VerseNexBlock` 统一为唯一名。

- **新名（推荐）**：

```python
from verse_nex import VerseNexLM, VerseNexAttention, MoDLayer, TriSparseAttention
```

- **旧名（DeprecationWarning，保留一个版本）**：`verse_torch.nn` 仍保留 `TransformerLM` / `TransformerBlock` / `GQASelfAttention` 作为别名，导入时发出 `DeprecationWarning: TransformerLM 已更名为 VerseNexLM`。
- **`config.yml` `arch` 字段统一**：仅保留 `arch: versenex` 唯一值；旧值 `transformer` / `hybrid` / `verse_nex` 自动映射 + DeprecationWarning。
- **`HybridBlock` / `HybridLM` deprecated**：保留只读兼容，新代码请用 `VerseNexLM`（TriSparse + MoD 已覆盖 Hybrid 的所有能力）。

### 超稀疏并行注意力（Part4K1）

Part4K1 对 `TriSparseAttention` 与推理路径做了三层并行加速：

| 组件 | 文件 | 说明 |
|---|---|---|
| 多 query chunk 并行 | `tri_sparse_attn.py` | 把 query 序列按 chunk 切分后批量 matmul，消除串行循环；GPU 下走 PyTorch 原生 kernel |
| `SpeculativeDecoder` | `speculative.py` | Medusa 风格：draft head 并行生成 k 个候选 token + 主模型一次前向验证 + verify-then-commit（接受最长正确前缀） |
| `ParallelKVCache` | `kv_cache_parallel.py` | 批量更新 KV cache（`batch_update`），预分配 buffer，避免顺序拷贝 |

- **数值一致**：并行实现与串行实现在 float32 下吻合到 1e-3（`tests/test_parallel_sparse_attn.py` 验证）
- **GPU 加速**：seq_len ≥ 512 下并行实现吞吐量 ≥ 2×（GPU 后端）
- **Speculative Decoding**：接受率 ≥ 75% 时吞吐量提升约 2-3×

```python
from verse_nex.speculative import SpeculativeDecoder

decoder = SpeculativeDecoder(model, draft_head, k=4)
output = decoder.generate(prompt_ids, max_new_tokens=128)
```

详见 [ADR-008 超稀疏并行注意力](../../docs/architecture/adr-008-parallel-sparse-attention.md)。

### NexRL 强化学习（Part4K1）

`verse_nex.nexrl` 子包实现 VerseNex 强化学习算法，五要素抽象 + PPO 训练器：

| 要素 | 类 | 职责 |
|---|---|---|
| `NexAgent` | `nexrl/agent.py` | 策略网络（VerseNexLM）+ 参考网络（冻结，KL 约束） |
| `NexEnv` | `nexrl/env.py` | 任务环境：`ChatEnv` / `MathEnv` / `CodeEnv` |
| `NexState` | `nexrl/state.py` | RL 状态：prompt + tokens + KV cache + logprobs |
| `NexAction` | `nexrl/action.py` | 动作采样：ε-greedy / softmax / nucleus + 探索衰减 + 重复惩罚 |
| `NexReward` | `nexrl/reward.py` | 多维奖励：correctness + fluency + safety + length_penalty + 归一化 + shaping |

训练组件：
- `ParallelRolloutCollector`：多 prompt / 多 rollout 并行采样（batched，GPU 批量前向）
- `NexTrainer`：PPO clipped surrogate + GAE + KL 自适应 + value function（纯策略梯度 fallback）

```python
from verse_nex.nexrl import NexAgent, NexTrainer, NexReward, NexEnv

agent = NexAgent(policy_model=verse_nex_lm, ref_model=ref_lm)
trainer = NexTrainer(agent=agent, env=env, reward=reward, cfg={...})
trainer.fit()
```

CLI 集成：`verse-posttrain --rl nexrl`。详见 [ADR-007 NexRL 设计](../../docs/architecture/adr-007-nexrl-design.md)。

### 位置编码

- `RoPE`：旋转位置编码，预计算 cos/sin，支持 `(B, T, H, D)`。
- `ALiBi`：注意力线性偏置，无位置嵌入。
- `NoPE`：无位置编码（用于依赖内部状态的架构）。

## 快速开始

```python
from verse_nex import HybridLM

# 1. 构造一个混合 LM（Mamba-2 + Sparse Attention）
model = HybridLM(
    vocab_size=259,
    dim=128,
    n_layers=4,
    sparse_ratio=0.25,         # 25% 的 block 使用 sparse attention
    ssm_kind="mamba2",
    ssm_kwargs={"d_state": 64, "d_conv": 4, "expand": 2, "n_heads": 4},
    sparse_kwargs={"n_heads": 4, "chunk_size": 16, "topk_chunks": 1},
)

# 2. 并行模式（训练）
input_ids = [[10, 20, 30, 40, 50]]   # (B=1, T=5)
logits_parallel = model.forward_parallel(input_ids)
print("parallel logits shape:", logits_parallel.shape)   # (1, 5, vocab_size)

# 3. 递归模式（推理，单步 O(1) 内存）
states = None
logits_step_list = []
for tok in input_ids[0]:
    logits_step, states = model.forward_recurrent([[tok]], states)
    logits_step_list.append(logits_step)

# 4. 验证数值一致（float32 下吻合到 1e-3）
import numpy as np
p = logits_parallel.data[0]              # (T, vocab)
r = np.stack([x.data[0, 0] for x in logits_step_list])  # (T, vocab)
print("max abs diff:", np.abs(p - r).max())   # 期望 < 1e-3
```

## parallel vs recurrent 模式

| 模式 | 用途 | 计算复杂度 | 内存 | 可微 |
| --- | --- | --- | --- | --- |
| `parallel` | 训练 | O(N)（整序列并行） | O(N) | ✅ |
| `recurrent` | 推理 | O(1) per step | O(1) 状态 | ❌ |

- **parallel**：整序列并行计算，可微，适合训练。
- **recurrent**：单步递推，常数内存，适合部署推理。
- **数值一致**：float32 下两种模式输出吻合到 1e-3，无需重新训练即可切换。

## 测试

| 文件 | 覆盖范围 |
|---|---|
| [test_mamba2_memory.py](../../tests/test_mamba2_memory.py) | Mamba-2 长序列内存与一致性 |
| [test_passkey.py](../../tests/test_passkey.py) | passkey 检索能力评测 |
| [test_mod_complete.py](../../tests/test_mod_complete.py) | **Part4K1** MoD 前向 / 反向 / aux loss 收敛 / parallel-vs-recurrent 一致性 |
| [test_parallel_sparse_attn.py](../../tests/test_parallel_sparse_attn.py) | **Part4K1** 多 chunk 并行 vs 串行数值一致（float32 吻合 1e-3）+ GPU 吞吐 |
| [test_speculative_decode.py](../../tests/test_speculative_decode.py) | **Part4K1** k=4 候选预测 + 接受最长正确前缀 + 拒绝处重 draft |
| [test_nexrl.py](../../tests/test_nexrl.py) | **Part4K1** 多维奖励 / 并行 rollout / KL 防崩溃 / 动作采样策略 |

## 相关文档

- [ADR-002 线性复杂度](../../docs/architecture/adr-002-linear-complexity.md)
- [ADR-003 世界模型路线](../../docs/architecture/adr-003-world-model-route.md)
- [ADR-007 NexRL 设计](../../docs/architecture/adr-007-nexrl-design.md)（**Part4K1 新增**）
- [ADR-008 超稀疏并行注意力](../../docs/architecture/adr-008-parallel-sparse-attention.md)（**Part4K1 新增**）
- [CometSpark V0.5-1B](../../spark/README.md)（基于 VerseNexLM 的 1B 参数 LM）
- [Verse 训练指南](../../docs/training_guide.md)
- [主 README](../../README.md)
