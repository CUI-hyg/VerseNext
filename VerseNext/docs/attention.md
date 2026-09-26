# 注意力机制与推理优化指南

VerseNext 支持三种可插拔注意力（注册中心 `attention/<name>`），并配套
一系列推理性能优化。

## 注意力类型

通过 `VerseTransformerConfig.attn_type` 选择：

| 类型 | 注册名 | 机制 | 适用场景 |
|---|---|---|---|
| `sdpa`（默认） | `attention/sdpa` | 稠密缩放点积因果注意力 + GQA/MQA | 通用，短中上下文 |
| `dsa` | `attention/dsa` | DeepSeek 稀疏注意力：indexer 打分 + top-k 选择 | 长上下文解码（GPU 收益显著） |
| `kda` | `attention/kda` | Kimi Delta Attention：门控 delta 规则线性注意力 | 超长上下文、显存受限 |

```python
from verse_nn import VerseTransformer, VerseTransformerConfig

cfg = VerseTransformerConfig(
    vocab_size=32000, d_model=512, num_layers=6, num_heads=8,
    attn_type="dsa",      # 或 "kda"
    dsa_top_k=64,         # DSA：每个 query 保留的 token 数
    dsa_index_head_dim=32 # DSA：indexer 头维度
)
model = VerseTransformer(cfg)
```

### DSA（DeepSeek Sparse Attention，简化参考实现）

1. 轻量 indexer 计算每个 query-key 对的相关性分数
2. 每个 query 仅保留 top-k 个因果合法位置（自身强制保留，不足时挤占最低分位）
3. 解码时 gather 被选中的 KV 直接计算注意力（免全长度掩码）

> 注意：CPU 小规模下 indexer 的逐步开销会超过稀疏化收益（实测 2k 上下文
> 约 0.6-0.7x SDPA）；其优势体现在 GPU 融合内核与超长上下文。选择对
> 浮点噪声敏感：缓存解码与全量重算在分数近并列处可能选出不同成员
> （均语义合法）。

### KDA（Kimi Delta Attention，chunkwise 并行实现）

每头递归状态更新（`S ∈ (d_k, d_v)`）：

```
S ← diag(g_t) · S                          # 细粒度门控衰减
S ← S(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ     # delta 规则写入/修正
o_t = Sᵀ q_t
```

- 状态大小与序列长度无关：1024 上下文下缓存显存约为 KV 缓存的 **1/32**
- **chunkwise 并行内核**（`kda_chunk_size`，默认 64）：
  - 关键性质：门控的行缩放与右侧状态转移 `(I − βkkᵀ)` **可交换**，据此对
    chunk 内状态做归一化 `S̃ = Λ⁻¹S`（`Λ = exp(cumsum(log g))`），归一化后
    递归变为无门控 delta 规则；
  - chunk 内展开为 **严格上三角线性系统** `(I + β⊙Û)Mᵀ = ...`（WY 表示），
    批量 `solve_triangular` 求解——chunk 内全并行，仅 chunk 间顺序传状态；
  - 实测 CPU 512-token 前向 **10.5x**（1421ms → 136ms，b=2/4 层）；
  - 单步解码保持递归路径（常数开销）；与递归基准逐位等价性经 3 种子
    验证（差异 < 1.4e-6，纯浮点累加顺序噪声）；
  - 数值边界：chunk 内累计门控衰减超过 ~1e-30 时 `1/Λ` 截断（此时旧状态
    的绝对贡献本身已指数级衰减，精度损失与语义一致）。

## 推理性能优化

| 优化 | 位置 | 说明 | 实测收益（CPU） |
|---|---|---|---|
| 静态预分配 KV Cache | `verse_nn/kv_cache.py` | 一次性分配、原地写入、零拷贝视图，替代逐步 `torch.cat`（O(t²) 拷贝） | 对动态缓存 1.3x，对全量重算 2.6x |
| KDA chunkwise 并行 | `attention.py` | WY 三角表示 + 批量三角求解，chunk 内全并行 | KDA 前向 10.5x（512 token） |
| 融合 QKV 投影 | `attention.py` | 单 GEMM 产出 q/k/v，减少 kernel 启动与内存往返 | 训练/推理通用 |
| 融合 SwiGLU | `blocks.py` | gate/up 合并为单 GEMM（2×hidden 后切分） | 训练/推理通用 |
| `logits_to_keep` | `transformer.py` | 解码/prefill 只对最后 1 个位置做 vocab 投影 | 长上下文 prefill 显著 |
| KV 缓存解码 | `generate()` | prefill 一次 + 每步 1 token | 长上下文数量级提升 |
| `torch.inference_mode` | `generate()` | 全程关闭 autograd 与版本计数 | 默认启用 |
| int8 动态量化 | `Generator(quantize=True)` | Linear 权重 int8（oneDNN/FBGEMM） | 大模型 1.5-2x + 体积 1/4 |
| 线程数 / flush denormal | `optimize_cpu()` | 物理核数线程 + 关闭亚正规慢路径 | 训练/推理通用 |

## 自定义注意力

```python
from verse_core import global_registry
import torch.nn as nn

@global_registry.register_attention("my_attn")
class MyAttention(nn.Module):
    index_head_dim = 0  # >0 时静态缓存会分配 indexer 缓冲
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.0, **kw):
        ...
    def forward(self, x, layer_cache=None, past_kv=None, use_cache=False):
        ...  # 返回 out 或 (out, present)

# 使用
cfg = VerseTransformerConfig(attn_type="my_attn")  # validate 白名单需同步扩展
```

统一接口约定：`forward(x, layer_cache=None, past_kv=None, use_cache=False)`，
静态缓存对象见 `verse_nn/kv_cache.py`（`StaticKVCache` / `RecurrentStateCache`）。
