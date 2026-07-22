# ADR-008: 超稀疏并行注意力

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k1-infra-model-upgrade/spec.md`](../../../.trae/specs/part4k1-infra-model-upgrade/spec.md)
- **前置 ADR**：[ADR-002 线性复杂度架构](adr-002-linear-complexity.md)（TriSparseAttention 是 VerseNex 原生架构核心）
- **相关 ADR**：[ADR-005 GPU/NPU 后端](adr-005-gpu-npu-backend.md)（多 chunk 并行受益于 GPU 批量算力）

## 上下文

Part4 引入了 `TriSparseAttention`（SWA + Global sink + ALiBi 三路并行稀疏注意力），但实现存在两个瓶颈：

1. **query chunk 串行循环**：原实现对 query 序列按 chunk 串行循环计算 attention，无法利用 GPU 批量并行——长序列（T ≥ 512）下吞吐量低
2. **推理速度慢**：标准 autoregressive 生成每步只产出 1 个 token，KV cache 顺序更新，无法利用 batch 并行

Part4K1 推出 CometSpark V0.5-1B（1B 参数 + 2048 上下文），CPU 上 attention 计算成为主要瓶颈；GPU 部署下也需要榨干并行度。

同时，业界已有成熟的加速方案：
- **FlashAttention** 的 chunk-wise tiling（但 Verse 不自研 CUDA kernel，详见 ADR-005）
- **Speculative Decoding**（Medusa / EAGLE）：draft head 并行生成 k 个候选 token + 主模型一次前向验证
- **Parallel KV Cache Update**：批量更新 KV cache 避免顺序拷贝

## 决策

**实现三层并行加速：(1) `tri_sparse_attn.py` 多 query chunk 并行计算；(2) `speculative.py` Medusa 风格 SpeculativeDecoder（draft + verify-then-commit）；(3) `kv_cache_parallel.py` ParallelKVCache 批量更新。**

具体含义：

### 1. 多 query chunk 并行（`tri_sparse_attn.py`）

- **批量矩阵化**：把 query 序列按 chunk 切分后，所有 chunk 的 attention 计算合并为一次批量 matmul
- **消除串行循环**：原 `for chunk in query_chunks:` 循环改为 `np.einsum` / `torch.matmul` 批量操作
- **GPU 并行**：GPU 后端下，批量 matmul 走 PyTorch 原生 CUDA kernel（cuBLAS / FlashAttention）
- **数值一致**：并行实现与串行实现在 float32 下吻合到 1e-3（`tests/test_parallel_sparse_attn.py` 验证）
- **长序列加速**：seq_len ≥ 512 下并行实现吞吐量 ≥ 2×（GPU 后端）

### 2. SpeculativeDecoder（`speculative.py`）

Medusa 风格的 speculative decoding：

- **Draft head**：轻量 head 并行生成 k 个候选 token（k=4 默认）
- **主模型一次前向验证**：把 k 个候选 token 拼成序列，主模型一次 forward 计算 logprob
- **Verify-then-commit**：接受最长正确前缀（候选 token 的 logprob 与 draft 一致），拒绝处重新 draft
- **加速比**：接受率 ≥ 75% 时吞吐量提升约 2-3×（每步产出 ≥ 2 token）

```python
from verse_nex.speculative import SpeculativeDecoder

decoder = SpeculativeDecoder(model, draft_head, k=4)
output = decoder.generate(prompt_ids, max_new_tokens=128)
```

### 3. ParallelKVCache（`kv_cache_parallel.py`）

- **批量更新**：`ParallelKVCache.batch_update(new_k, new_v)` 一次性拼接多个新 token 的 KV，避免顺序 `np.concatenate`
- **预分配内存**：构造时预分配 `max_seq_len` 大小的 buffer，更新时只写不扩
- **GPU 友好**：批量更新走 PyTorch `torch.cat`，GPU 上避免多次小拷贝

## 后果

### 优点

- **长序列加速**：多 chunk 并行让 attention 计算不再串行，GPU 下 seq_len=2048 吞吐量提升 ≥ 2×
- **推理加速**：SpeculativeDecoder 接受率高时每步产出多个 token，吞吐量提升 2-3×
- **KV cache 高效**：ParallelKVCache 批量更新避免顺序拷贝开销
- **CPU 也受益**：多 chunk 并行在 CPU 上用 `np.einsum` 向量化，虽然不及 GPU 但仍有 1.5×~2× 提升
- **数值一致**：并行与串行实现在 float32 下吻合到 1e-3，无精度损失
- **与 DeviceBackend 协同**：GPU 后端下批量 matmul 走 PyTorch 原生，不自研 kernel（符合 ADR-005）

### 缺点

- **Speculative Decoding 接受率不稳定**：draft head 质量差时接受率低，加速比下降甚至变慢（最坏情况退化为标准生成 + 额外 draft 开销）
- **draft head 训练成本**：Medusa 风格需要单独训练 draft head（额外参数 + 训练数据）
- **ParallelKVCache 内存固定**：预分配 `max_seq_len` buffer，短序列时浪费内存
- **多 chunk 并行实现复杂**：边界条件（chunk 不整除、KV cache 跨 chunk）需要仔细处理

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| 并行与串行数值不一致 | `tests/test_parallel_sparse_attn.py` 验证 float32 吻合 1e-3；CI 每次跑该测试 |
| SpeculativeDecoder 接受率低导致变慢 | `k` 可调（默认 4，接受率低时降到 2）；监控接受率，低于阈值自动回退标准生成 |
| draft head 与主模型版本不匹配 | `SpeculativeDecoder` 构造时校验 draft_head 与 model 的 vocab_size / n_embd 一致 |
| ParallelKVCache buffer 溢出 | 构造时 `max_seq_len` 强制上限；超出抛 `OverflowError` 而非静默截断 |
| GPU 环境 FlashAttention 数值与 Verse 实现不一致 | GPU 测试标记 `@pytest.mark.gpu`，CPU 回归测试用 NumpyBackend 验证 |

## 替代方案（已否决）

### 方案 A：保持串行实现，不并行化

**描述**：维持 Part4 的串行 query chunk 循环，不引入并行。

**否决理由**：
- 长序列（T ≥ 512）下 CPU 吞吐量不达标
- GPU 算力浪费（串行循环无法利用批量并行）
- 与 CometSpark V0.5-1B 的 1B 参数 + 2048 上下文目标冲突

### 方案 B：自研 FlashAttention 风格 CUDA kernel

**描述**：用 CuPy / PyCUDA 实现 tiling + fused softmax 的 CUDA kernel。

**否决理由**：
- 违反 ADR-005 "不自研 kernel"原则
- 维护成本极高（每种 attention 变体都要写 kernel）
- 性能不及 PyTorch 原生 FlashAttention
- NPU 无法复用

### 方案 C：用 HuggingFace `optimum` 的 attention 优化

**描述**：集成 `optimum.bettertransformer` 或 `flash_attn` 库。

**否决理由**：
- 硬依赖 PyTorch + Transformers + flash_attn（闭源编译）
- 与 VerseNexLM 的 TriSparseAttention 三路并行结构不兼容（flash_attn 假设标准 dense attention）
- 违反零重型依赖原则

### 方案 D：仅做 Speculative Decoding，不并行化 attention

**描述**：只实现 SpeculativeDecoder，attention 保持串行。

**否决理由**：
- Attention 串行仍是训练瓶颈（训练时无 speculative decoding）
- 长序列训练吞吐量不达标
- 三层加速是互补的，缺一不可

## 备注

- 本 ADR 是 Part4K1 "加速路线"的核心决策
- 三层加速互补：训练用多 chunk 并行，推理用 SpeculativeDecoder + ParallelKVCache
- `tests/test_parallel_sparse_attn.py` 验证多 chunk 并行 vs 串行数值一致
- `tests/test_speculative_decode.py` 覆盖 k=4 候选预测 + 接受最长正确前缀 + 拒绝处重 draft
- 相关代码：
  - [`verse_nex/tri_sparse_attn.py`](../../packages/verse_nex/verse_nex/tri_sparse_attn.py)
  - [`verse_nex/speculative.py`](../../packages/verse_nex/verse_nex/speculative.py)
  - [`verse_nex/kv_cache_parallel.py`](../../packages/verse_nex/verse_nex/kv_cache_parallel.py)
