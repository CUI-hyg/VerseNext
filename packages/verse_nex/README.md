# VerseNex

> 中文定位：Transformer 替代架构库，主推线性复杂度（SSM/Mamba/RWKV/Linear Attention）+ Hybrid 混合架构，O(1) 推理状态 + O(N) 训练并行。

[返回主 README](../../README.md)

## 特性

- 线性复杂度：O(N) 训练 + O(1) 推理状态，适配长上下文。
- parallel / recurrent 双模式，float32 下输出吻合到 1e-3。
- 多种架构：Mamba-2 / RWKV-7 / RetNet / Sparse Attention / Hybrid。
- 位置编码：RoPE / ALiBi / NoPE，统一接口。
- 纯 Python + NumPy 友好，无重型深度学习框架硬依赖。

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

### Hybrid Block / LM

- `HybridBlock`：SSM（mamba2 / rwkv7）+ Sparse Attention 混合 block。
- `HybridLM`：完整 LM（Embedding → N × HybridBlock → LayerNorm → Head）。
- 支持 `forward_parallel` / `forward_recurrent` / `forward(mode=)` / `generate`。

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

- `tests/test_mamba2_memory.py`：Mamba-2 长序列内存与一致性。
- `tests/test_passkey.py`：passkey 检索能力评测。

## 相关文档

- [ADR-002 线性复杂度](../../docs/architecture/adr-002-linear-complexity.md)
- [ADR-003 世界模型路线](../../docs/architecture/adr-003-world-model-route.md)
- [CometSparkLM 使用 HybridLM](../../data/demo/model/model.py)
