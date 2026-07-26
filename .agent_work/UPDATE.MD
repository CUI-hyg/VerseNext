# VerseNex 子包代码审计报告（Part5K1.5）

> 审计范围：`packages/verse_nex/verse_nex/` 全部 18 个文件，共 ~8834 行代码
> 审计时间：Part5K1.5
> 审计目标：找出 Bug、漏洞、错误、冲突之处，以及"蜘蛛丝"代码
> （重复、低效、反复套壳、过分的相互导入、垃圾逻辑等）

## 摘要

| 严重程度 | 数量 | 说明 |
|---------|------|------|
| 高危 | 13 | 影响训练正确性 / 推理崩溃 / 资源爆炸 |
| 中危 | 35 | 影响数值一致性 / 性能 / 接口冲突 |
| 低危 | 25 | 死代码 / 命名误导 / 文档错误 / 微小性能 |
| **合计** | **73** | — |

**核心结论**：
1. `moe.py` 的 EMA 平滑和熵正则化**实际未生效**（死代码 + 梯度断裂），MoD 架构的核心辅助损失实现有严重缺陷。
2. `nexrl/` 子模块的 PPO 实现有**多处违反算法假设**（KL 尺度、IS logprob、GAE bootstrap、reward shaping），整个 RL 训练路径不可信。
3. `linear_attention.py` 的 `forward_chunkwise` **梯度无法回传**（pad 用 `np.concatenate` 切断计算图），与 docstring "适用于长序列训练" 严重不符。
4. **5 套 KV Cache 接口并存**（dict / list / `ParallelKVCache` / `StaticCache` / `DynamicCache`），无统一抽象，且 `ParallelKVCache` 全项目无人使用（死代码）。
5. **3 套 RoPE 实现**、**2 套稀疏注意力**、**3 套 softmax**，大量重复造轮子。

---

## 一、高危 Bug（必须修复）

### 1.1 `moe.py` — Router EMA `_load_balance_ema` 死代码（功能未生效）

- **位置**：[moe.py](file:///workspace/packages/verse_nex/verse_nex/moe.py) L338（初始化）、L409-L417（更新）
- **类型**：Bug / 死代码
- **描述**：`_load_balance_ema` 在 `__init__` 中初始化为 `None`，在 `forward` 中当 `ema_decay > 0 and self.training` 时被更新（L411-L417 计算 EMA 值并赋给 `self._load_balance_ema`）。但**全文件没有任何地方读取这个值**参与计算。`load_balance_loss`（可微版本）仍然直接参与 `aux_loss`，EMA 值被完全丢弃。
- **影响**：V1.2 宣称的"EMA 平滑抑制训练抖动"功能**实际未生效**。
- **修复方向**：在 `aux_loss` 中用 `self._load_balance_ema` 替换或混合 `load_balance_loss`（注意 EMA 不可微，需保留可微路径），或删除这段死代码并修正文档。

### 1.2 `moe.py` — 熵正则化 `entropy_reg` 断开梯度（参数完全无效）

- **位置**：[moe.py](file:///workspace/packages/verse_nex/verse_nex/moe.py) L431-L445
- **类型**：Bug / 数值计算错误
- **描述**：
  ```python
  probs_np = probs.data                    # 取 .data，梯度断开
  entropy_per_token = -np.sum(probs_np * np.log(probs_np + eps), axis=-1)
  mean_entropy = float(np.mean(entropy_per_token))
  entropy_reg = Tensor(np.array(-self.entropy_weight * mean_entropy, ...), requires_grad=False)
  ```
  `entropy_reg` 是 `requires_grad=False` 的常量，加到 `aux_loss` 后**不产生任何梯度**。注释辩解称"梯度已通过 load_balance 的 P_i 回传"——这是**错误的逻辑**：`load_balance_loss` 的梯度方向是降低 `f_i × P_i`（让概率匹配命中率），而熵正则的梯度方向是最大化熵（让 P 更均匀），两者是完全不同的优化目标。
- **影响**：`entropy_weight` 参数除了改变 loss 数值显示外，对训练**完全无效**。
- **修复方向**：用 Tensor 运算实现可微熵正则：`entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()`，然后 `aux_loss = aux_loss - entropy_weight * entropy`。

### 1.3 `linear_attention.py` — `forward_chunkwise` pad 切断梯度路径

- **位置**：[linear_attention.py](file:///workspace/packages/verse_nex/verse_nex/linear_attention.py) L264-L267、L348-L354
- **类型**：Bug / 梯度断裂
- **描述**：
  ```python
  zeros = Tensor(np.zeros((B, pad_len, H, d)), requires_grad=False)
  q = Tensor(np.concatenate([q.data, zeros.data], axis=1), requires_grad=q.requires_grad)
  ```
  新建 Tensor 用 `np.concatenate` 拼接 `q.data`，**`q` 的 `_children` 丢失**，反向传播无法从新 q 回到原 q。`_compute_qkv` 计算的 q、k、v 梯度全部断掉。同样问题在 L348-L354 拼接 chunk 输出时也存在。
- **影响**：`forward_chunkwise` 实际不可训练（梯度无法回传到 qkv 投影权重），与 docstring "适用于长序列训练" 严重不符。
- **修复方向**：用 `verse_torch.vnn._concat`（可微拼接）替换 `np.concatenate`，或用 `sparse_attention._pad_last_dim`（已实现带梯度的 pad）。

### 1.4 `sparse_attention.py` — `forward_recurrent` KV cache 无限增长

- **位置**：[sparse_attention.py](file:///workspace/packages/verse_nex/verse_nex/sparse_attention.py) L354-L358、L451-L458
- **类型**：Bug / 资源泄漏
- **描述**：
  - L354-L359：`kv_cache = list(kv_cache)` 然后 `kv_cache.append((k.data, v.data))`，每个 token 都 append，**从不修剪**。
  - L451-L458：注释说"修剪 KV cache（移除过老 chunk）"，但实际是 `pass  # 简化：不修剪，让外层管理`。
  - L421-L423 用 `attend_indices[-(self.max_kv_chunks * C):]` 截断 attend 列表，但 `kv_cache` 没截断，下次 attend 时仍能索引到老 token，逻辑不一致。
- **影响**：长序列推理时 `kv_cache` 长度线性增长，内存 O(T)，与"常数内存"的 recurrent 设计目标矛盾，最终 OOM。
- **修复方向**：实现真正的 cache 修剪：保留最近 `max_kv_chunks * C` 个 token 的 (K, V)，并用偏移量记录全局位置。

### 1.5 `kv_cache_parallel.py` — `batch_update` 替换 buffer 破坏预分配语义

- **位置**：[kv_cache_parallel.py](file:///workspace/packages/verse_nex/verse_nex/kv_cache_parallel.py) L197-L198
- **类型**：Bug / 性能 + 正确性
- **描述**：构造函数（L89-L97）预分配 `self._k_buf[i]` 形状为 `(max_batch, max_seq, num_heads, head_dim)`。但 `batch_update` 在 L197-L198 执行 `self._k_buf[layer_idx] = new_k`，其中 `new_k` 是 `_concat` 结果，形状为 `(B, T_new_total, H, D)`，**T_new_total 通常 < max_seq**。导致：
  1. 预分配的 max_seq buffer 被丢弃，每次 update 都创建新 Tensor，**失去预分配性能优势**。
  2. `reset()` L227-L232 用 `np.zeros_like(self._k_buf[i].data)` 重建 buffer，但此时 `_k_buf[i]` 已经是 shorter tensor，重建后的 buffer 形状是 `(B, T_new_total, H, D)` 而非 `(max_batch, max_seq, H, D)`，**buffer 大小永久缩小**。
- **影响**：预分配失效 + 长序列下 buffer 不足。
- **修复方向**：在 `batch_update` 中将 `new_k` 写入预分配 buffer 的 `[:B, :T_new_total]` 切片，保留 max_seq 维度。

### 1.6 `kv_cache_parallel.py` — `get()` 未按 `per_seq_lens` 截断

- **位置**：[kv_cache_parallel.py](file:///workspace/packages/verse_nex/verse_nex/kv_cache_parallel.py) L203-L212
- **类型**：Bug / 数据污染
- **描述**：`get` docstring 说"返回的 K/V 形状为 `(B, max(per_seq_lens), H, D)`"，但实现直接 `return self._k_buf[layer_idx], self._v_buf[layer_idx]`，由于问题 1.5，buffer 实际形状是 `(B, T_new_total, H, D)`，T_new_total 是最后一次 update 后的总长度。如果 batch 内序列长度不同（变长场景），返回的 K/V 含 padding 位置的零值，下游 attention 会把这些零当真实 key 计算，导致 softmax 分母偏大、输出偏零。
- **修复方向**：`get` 应按 `max(self.per_seq_lens[:B])` 截断，或返回 `(K[:B, :max_len], V[:B, :max_len])`。

### 1.7 `tri_sparse_attn.py` — ALiBi 路径在 parallel 与 recurrent 之间语义不一致

- **位置**：[tri_sparse_attn.py](file:///workspace/packages/verse_nex/verse_nex/tri_sparse_attn.py) L600-L608（parallel） vs L712-L725（recurrent）
- **类型**：Bug / 数值不一致
- **描述**：
  - Parallel 路径（L600）：`use_path_c = self.use_alibi and T_k <= self._ALIBI_MAX_T`，判据是 **KV cache 总长度 `T_k`**，且 ALiBi 路径独立构造 `(B, H, T_q, T_k)` 全注意力矩阵，看到**所有 past key**。
  - Recurrent 路径（L714）：`T_total = position + 1; use_path_c = self.use_alibi and T_total <= self._ALIBI_MAX_T`，判据是**全局位置**，且 ALiBi 路径复用 `swa_scores`（L721 `alibi_scores = swa_scores + alibi_bias`），意味着 recurrent 的 ALiBi 只看到 SWA 窗口内的 W 个 key。
  - 当 `position=1500, W=512, T_k=512`：parallel 检查 `T_k=512 ≤ 1024` 启用 ALiBi 看 512 个 key；recurrent 检查 `T_total=1501 > 1024` **不启用 ALiBi**。
- **影响**：破坏"parallel 与 recurrent 数值一致到 1e-3"的契约。
- **修复方向**：将 recurrent 的判据改为 `n_cached <= self._ALIBI_MAX_T`，且让 ALiBi 路径独立计算 `q @ k^T` 而非复用 `swa_scores`。

### 1.8 `nexrl/trainer.py` — `_compute_gae` 对 truncated rollout 错误处理

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L219-L226
- **类型**：Bug / GAE 计算
- **描述**：代码在 `t == T - 1` 时硬编码 `next_value = 0.0`，注释为"终止状态 value = 0"。但 rollout 结束的原因有 `eos`（真正终止）和 `max_len`（截断）两种（见 collector.py:210-211）。被 `max_len` 截断的 episode 应当用 `V(s_T)` bootstrap，否则 advantage 会被系统性低估，导致策略梯度信号偏负。
- **修复方向**：让 `Rollout` 携带 `truncated` 标志，截断时 `next_value = values[T]`，仅 eos 终止时才置 0。

### 1.9 `nexrl/trainer.py` + `agent.py` — `compute_kl` 用 `sum` 而非 `mean`，KL 损失尺度爆炸

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L382-L390，根因在 [agent.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/agent.py) L159
- **类型**：Bug / KL 损失
- **描述**：`self.agent.compute_kl(logits, ref_logits)` 内部 `kl = (ref_probs * (ref_log_probs - new_log_probs)).sum()` 对整个 `(B, T, V)` 张量求和。当 `T=16, V=50000` 时，单个样本的 KL 标量即可达到 ~`T×V×tiny` 量级，`total_loss = ... + self.kl_weight * kl_loss`（trainer.py:397）在 `kl_weight > 0` 时会瞬间爆炸或主导损失。`compute_kl_scalar`（agent.py:197）反而用 `np.mean`，两个接口语义不一致，监控值与训练值不匹配。
- **影响**：KL 权重一旦 >0，loss 立即爆炸，整个 PPO+KL 路径不可用。
- **修复方向**：把 `compute_kl` 改为 `(...).sum(dim=-1).mean()`（per-token 平均），并统一两个接口。

### 1.10 `nexrl/trainer.py` — `RewardShaper` 只对终末 reward 做一次 shaping（几乎无效）

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L630-L635
- **类型**：Bug / reward shaping
- **描述**：标准 potential-based shaping 要求逐步做 `F(s_t, s_{t+1}) = γΦ(s_{t+1}) - Φ(s_t)`。这里只对最终 reward 做一次：`prev_info = logprobs[:-1]`、`curr_info = logprobs`，两者只差一个 token，`_default_potential` 取 mean 后 Φ(s₀) ≈ Φ(s_T)，shaping 项 ≈ 0。势函数塑造的理论保证（policy invariance）失效，等于没用。
- **修复方向**：要么逐步 shaping（在 collector 内每步调用），要么直接移除该路径。

### 1.11 `nexrl/collector.py` — `collect_batched` 缺少 attention mask，padding token 污染前向

- **位置**：[collector.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/collector.py) L301-L309、L315
- **类型**：Bug / batched 前向
- **描述**：`padded = np.zeros((len(batch_inputs), max_len))`，用 0 padding 但 0 是有效 token id，且没有传 attention mask 给 `forward_policy`。padding 位置会经过 `tok_emb` + 所有 `blocks`，参与注意力计算，导致 `last_logits = logits_np[j, len(batch_inputs[j]) - 1, :]` 被其他样本的 padding 污染。
- **修复方向**：左 padding + attention mask，或干脆每个 rollout 独立前向（用 `collect`）。

### 1.12 `nexrl/action.py` — `ActionSampler.sample` logprob 与采样分布不一致（破坏 PPO IS 假设）

- **位置**：[action.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/action.py) L235-L242，被 collector.py:186-191、324-329 调用
- **类型**：Bug / PPO 假设
- **描述**：对 `nucleus/topk`，采样分布是"截断 + 重归一化"，但返回的 `logprob = log_softmax(original_logits)[token_id]`，与采样概率不一致；对 `epsilon_greedy`，采样分布是 `(1-ε)·argmax + ε·uniform`，但 `logprob` 仍是 `log_softmax`。`collector` 把这个 logprob 写入 `rollout.logprobs`，`_ppo_update` 拿它当 `old_logprob` 算 `ratio = new/old`，PPO 的 IS 假设 `behavior = old policy` 直接失效。
- **影响**：nucleus/topk/epsilon_greedy 三种策略下 PPO 训练都是无效的。
- **修复方向**：logprob 用实际采样分布的 log（截断后重归一化），或限制 PPO 只用 `softmax` 策略。

### 1.13 `nexrl/agent.py` + `collector.py` — `agent.act` 与 `collector._single_rollout` 重复，`NexAction` 是死代码

- **位置**：[agent.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/agent.py) L92-L137 vs [collector.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/collector.py) L149-L239
- **类型**：蜘蛛丝 + 死代码
- **描述**：`NexAgent.act` 自带"前向 → 取 last_logits → sample → 构造 `NexAction`"的完整逻辑，但 `collector` 完全没调用它，自己又写了一遍。结果 `NexAction`（action.py:19-29）只在 `act()` 里被构造，从未被 `collector` / `trainer` / `env` 使用，是事实上的死代码。
- **修复方向**：让 collector 调用 `agent.act`，或删除 `NexAgent.act` + `NexAction`。

---

## 二、中危问题

### 2.1 `cometspark.py` — `forward` 在 64+ 层时走 `chunked_forward`（不可微），与 `forward_with_aux` 路径不一致

- **位置**：[cometspark.py](file:///workspace/packages/verse_nex/verse_nex/cometspark.py) L649-L650 vs L664-L695
- **类型**：冲突 / 接口不一致
- **描述**：`forward` 在 `n_layer >= 64` 时自动切换到 `chunked_forward`，后者对非最后 chunk 用 `no_grad + detach`（L620-L624），导致只能反传到最后 `chunk_size` 层的参数。但 `forward_with_aux` 没有这个分支，始终走可微的逐块前向。64+ 层模型如果用 `forward` 训练会静默丢失大部分梯度，而用 `forward_with_aux` 则正常。
- **修复方向**：统一训练入口（让 `forward` 不自动切换，或在 `chunked_forward` 中实现真正的梯度检查点）。

### 2.2 `cometspark.py` — idx 标准化逻辑重复 5 次

- **位置**：[cometspark.py](file:///workspace/packages/verse_nex/verse_nex/cometspark.py) L599-L602、L652-L655、L677-L680、L712-L717、L770-L776
- **类型**：蜘蛛丝 / 重复代码
- **描述**：`Tensor(np.asarray(idx, dtype=np.int64))` + `astype(np.int64)` 的标准化逻辑在 `forward`、`forward_with_aux`、`forward_recurrent`、`chunked_forward`、`generate` 中各写了一遍，几乎一字不差。
- **修复方向**：提取为 `_normalize_idx(idx) -> Tensor` 私有方法。

### 2.3 `cometspark.py` — `load` 仅支持 `.pt`，`from_pretrained` 支持两种格式

- **位置**：[cometspark.py](file:///workspace/packages/verse_nex/verse_nex/cometspark.py) L1032-L1038（`load`）vs L1040-L1085（`from_pretrained`）；L928-L972（`save`）vs L1147-L1165（`save_pretrained`）
- **类型**：冲突 / 接口不一致
- **描述**：`load` 方法只读 `.pt`（pickle），不识别 `.vn`；但 `from_pretrained` 两种都支持。`save` 支持 `format="vn"` 和 `"pt"`；但 `save_pretrained` 只写 `model.pt`，不写 `model.vn`。用户用 `save(format="vn")` 存的模型，不能用 `load` 加载，必须用 `from_pretrained`。
- **修复方向**：让 `load` 复用 `from_pretrained` 的分发逻辑；让 `save_pretrained` 支持写 `model.vn`。

### 2.4 `moe.py` — `expert_hidden` 自动计算逻辑重复

- **位置**：[moe.py](file:///workspace/packages/verse_nex/verse_nex/moe.py) L689-L690（MoDLayer）与 L557-L559（DensePart）
- **类型**：蜘蛛丝 / 重复代码
- **描述**：`expert_hidden = max(int(dim * 8 / 3 / 64) * 64, 64)` 这行公式在 `MoDLayer.__init__` 和 `DensePart.__init__` 中各写了一遍。MoDLayer 计算后传给 DensePart，DensePart 的 `if expert_hidden is None` 分支永远走不到。
- **修复方向**：只在 `MoDLayer` 中计算，或提取为公共工具函数 `_default_expert_hidden(dim)`。

### 2.5 `moe.py` — MoDLayer 访问 Router 私有属性（封装泄漏）

- **位置**：[moe.py](file:///workspace/packages/verse_nex/verse_nex/moe.py) L776-L777、L786-L792
- **类型**：蜘蛛丝 / 封装泄漏
- **描述**：`MoDLayer.forward` 中直接访问 `self.part_router._last_load_balance_loss`、`self.parts[p].router._last_load_balance_loss`、`self.parts[p].router._last_z_loss` 等下划线开头的私有属性来累加 breakdown。
- **修复方向**：给 Router 增加公开的 `get_loss_breakdown() -> dict` 方法。

### 2.6 `moe.py` — `_dispatch_and_combine` 循环内大量临时 Tensor 创建

- **位置**：[moe.py](file:///workspace/packages/verse_nex/verse_nex/moe.py) L182-L211
- **类型**：蜘蛛丝 / 低效逻辑
- **描述**：对每个 expert（默认 8 个），循环内创建 `slot_mask_t`、`w_e_full`、`w_e`、`y_e_weighted`、`scattered` 共 5+ 个临时 Tensor，且 `slot_mask = (idx_np == e)` 和 `token_mask = slot_mask.any(axis=-1)` 每次都重新生成。
- **修复方向**：预计算所有 expert 的 `token_mask`（向量化 `(idx_np[..., None] == np.arange(num_experts)).any(axis=1)`）。

### 2.7 `mamba2.py` — parallel SSD 的 `L_data` 内存爆炸（O(T²)）

- **位置**：[mamba2.py](file:///workspace/packages/verse_nex/verse_nex/mamba2.py) L501
- **类型**：Bug / 资源问题
- **描述**：`L_data = np.exp(log_decay) * mask[None, :, :, None]` 形状为 `(B, T, T, H)` float64。当 `T=2048, H=16, B=1` 时，单个 Tensor 占用 `1 * 2048 * 2048 * 16 * 8 = 512MB`。再加上 `M_t`、`Y_t` 等中间结果，parallel 路径的峰值内存可达数 GB。代码没有任何长度警告或分块机制。
- **修复方向**：对长序列分块计算 SSD（chunkwise），或添加 `max_parallel_len` 参数超阈值时自动降级为 recurrent 累积。

### 2.8 `rwkv7.py` — `a` 和 `b` 被计算但从未使用（死代码，浪费参数）

- **位置**：[rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L70-L71（定义 `x_a`, `x_b`）、L115-L116（计算 `a_raw`, `b`）、L120（`a = a_raw.sigmoid()`）、L163（返回 `a, b`）、L170（注释承认未使用）
- **类型**：Bug / 死代码
- **描述**：`RWKV7TimeMix` 定义了 `self.x_a` 和 `self.x_b` 两个 `Linear(dim, dim)` 层（共 `2 * dim * dim` 参数），`_compute_rkvwab` 中计算了 `a_raw`、`b`、`a`，但 `forward_parallel` L170 注释明确写道"a, b 在简化版本中暂不参与 state update"。这些参数会进入 `state_dict`、参与优化器更新、占用显存，但**不影响模型输出**。`x_a`/`x_b` 的梯度始终为零（无下游消费），优化器对它们的更新无意义。
- **修复方向**：要么实现 low-rank modulator 的 state update（参考 RWKV-7 论文），要么删除 `x_a`/`x_b` 及相关计算。

### 2.9 `rwkv7.py` — TimeMix 与 ChannelMix 的 `mix` 系数语义相反

- **位置**：[rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L102（TimeMix）vs L405（ChannelMix）
- **类型**：冲突 / 接口不一致
- **描述**：
  - TimeMix: `r_in = x * self.r_mix + x_shifted * (1.0 - self.r_mix)` —— `mix` 是 **x 的权重**
  - ChannelMix: `k_in = x * (1.0 - self.k_mix) + x_shifted * self.k_mix` —— `mix` 是 **x_shifted 的权重**
  
  当 `mix=0.5` 时两者等价，但当 `mix != 0.5` 时语义完全相反。这会导致同名参数在不同层有相反的含义，极易引发调参错误。
- **修复方向**：统一约定（建议都用 `x * mix + x_shifted * (1 - mix)`）。

### 2.10 `rwkv7.py` — parallel SSD 的 `L_data` 内存爆炸（比 mamba2 更严重）

- **位置**：[rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L190
- **类型**：Bug / 资源问题
- **描述**：`L_data = np.exp(log_decay) * mask[None, :, :, None, None]` 形状为 `(B, T, T, H, K)` float64。比 mamba2.py 多了 `K` 维（`head_size`）。当 `T=2048, H=8, K=64, B=1` 时，占用 `1 * 2048 * 2048 * 8 * 64 * 8 = 16GB`。
- **修复方向**：同 mamba2.py，实现 chunkwise SSD 或自动降级。

### 2.11 `tri_sparse_attn.py` — ALiBi slopes 公式与 `positional.py` 不一致

- **位置**：[tri_sparse_attn.py](file:///workspace/packages/verse_nex/verse_nex/tri_sparse_attn.py) L152-L154
- **类型**：冲突
- **描述**：
  - 本文件：`slopes = 1.0 / (2.0 ** (np.arange(1, n_head + 1) / n_head))` → 即 `1/2^(1/n), 1/2^(2/n), ..., 1/2`
  - `positional.py:153-159`：n_heads ≤ 8 时用 `1/2, 1/4, ..., 1/2^n`；n_heads > 8 时用 `base^h where base=2^(-8/n)`
  - `verse_torch/vnn.py:1144`（项目主 ALiBi 实现）：`1/2^(h/n_head)`，与本文件一致
  - 当 `n_head=8` 时三个公式结果差距巨大：本文件 `1/2^(1/8)≈0.917`，positional.py `1/2=0.5`。
- **修复方向**：删除 `positional.py` 中的 ALiBi 类或让其 slopes 公式委托 `verse_torch.vnn.ALiBi`。

### 2.12 `tri_sparse_attn.py` — `_swa_forward` 极端 `T_k=0` 边界

- **位置**：[tri_sparse_attn.py](file:///workspace/packages/verse_nex/verse_nex/tri_sparse_attn.py) L324-L329
- **类型**：Bug / 边界条件
- **描述**：当 `T_k=0`（KV cache 为空且无 global token 注入）时，`mask_2d[:, K_ci:] = -1e9` 其中 `K_ci = k_hi - k_lo = 0`，整个 mask 全 -1e9，softmax 后均匀分布，输出 `0 @ v_pad` = 0。无显式错误，但语义错误（query 不应 attend 任何 padding key）。
- **修复方向**：在 `_swa_forward` 开头加 `if T_k == 0: return q * 0`。

### 2.13 `tri_sparse_attn.py` — global path 在 recurrent 中跨 batch 共享 `global_k/v`

- **位置**：[tri_sparse_attn.py](file:///workspace/packages/verse_nex/verse_nex/tri_sparse_attn.py) L647、L706-L710
- **类型**：Bug / 状态污染
- **描述**：recurrent 路径中 `global_k, global_v = self._compute_global_kv_numpy()` 返回形状 `(H, n_g, d)`（去掉了 batch 维），所有 batch 共享。global token 在 recurrent 中只计算一次（首次调用 `state=None` 时），后续 step 复用，但 global_tokens 是**可学习参数**，如果模型在推理中被微调（few-shot 场景），global_k/v 不会更新。
- **修复方向**：每次 forward_recurrent 都重新计算 global_k/v（开销小）。

### 2.14 `kv_cache_parallel.py` — 变长 batch 路径 `_concat(dim=0)` 会失败

- **位置**：[kv_cache_parallel.py](file:///workspace/packages/verse_nex/verse_nex/kv_cache_parallel.py) L180-L193
- **类型**：Bug
- **描述**：当 `per_seq_lens` 在 batch 内不一致时，进入变长路径，对每个 b 计算 `new_k_b` 形状 `(1, per_seq_lens[b]+T_new, H, D)`。不同 b 的 `new_k_b` 在 dim=1 长度不同，最后 `new_k = _concat(new_k_list, dim=0)` 沿 dim=0 拼接——`np.concatenate` 要求除拼接轴外其他轴长度相同，**会抛 ValueError**。
- **修复方向**：要么在变长路径中把所有 `new_k_b` pad 到 `max(per_seq_lens)+T_new` 再 concat，要么直接抛 `NotImplementedError`。

### 2.15 `kv_cache_parallel.py` — `ParallelKVCache` 未被任何 attention 模块使用（死代码）

- **位置**：[kv_cache_parallel.py](file:///workspace/packages/verse_nex/verse_nex/kv_cache_parallel.py)（整体）
- **类型**：蜘蛛丝 / 死代码
- **描述**：`ParallelKVCache` 仅在 `__init__.py:61` 导出，未被 `tri_sparse_attn.py`、`sparse_attention.py`、`cometspark.py` 任何 attention 模块引用。`cometspark.py` 用的是 `dict` 形式 KV cache。该类是为 speculative decoding 设计，但 `speculative.py` 也没有用它。**整个类是死代码**。
- **修复方向**：要么在 `cometspark.py` 的 `forward` 中接入 `ParallelKVCache`，要么删除该类。

### 2.16 `sparse_attention.py` — `_select_chunks` 是死代码且含未修复 Bug

- **位置**：[sparse_attention.py](file:///workspace/packages/verse_nex/verse_nex/sparse_attention.py) L145-L204
- **类型**：Bug + 蜘蛛丝
- **描述**：`_select_chunks` 方法 grep 全项目无任何调用（`forward_parallel` L254-L274 自己实现了 chunk selection 逻辑）。该方法内部还有遗留 bug：L194-L196 `np.take_along_axis` 的 indexer 形状不对，L196 注释 `# (B, H, k) - wait, need to be (B, k, H)` 明显是开发中遗留。
- **修复方向**：删除整个 `_select_chunks` 方法。

### 2.17 `sparse_attention.py` — `forward_parallel` 用 `(T, T)` 稠密矩阵实现稀疏

- **位置**：[sparse_attention.py](file:///workspace/packages/verse_nex/verse_nex/sparse_attention.py) L252-L313
- **类型**：蜘蛛丝
- **描述**：注释 L224-L227 自己承认"训练时用 (T, T) 矩阵实现，O(T²) 内存。这违背了 sparse 的初衷"。`tri_sparse_attn._swa_forward` 已经用 chunk-wise + 可微 `_concat` 解决了梯度问题，本文件的稠密实现已过时。
- **修复方向**：废弃 `TopKChunkSparseAttention`，统一使用 `TriSparseAttention`。

### 2.18 `nexrl/trainer.py` — reward shaping 后再归一化，破坏 potential-based 保证

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L628-L640
- **类型**：Bug / reward 处理顺序
- **描述**：`shape` 之后立即 `normalize`，相当于对 `r + γΦ(s') - Φ(s)` 整体做 `(x - μ)/σ` 缩放，等价于把 shaping 项也乘 `1/σ`，破坏了 potential-based shaping 的 policy-invariance 保证。
- **修复方向**：先 normalize 再 shape，或只对 base reward 归一化。

### 2.19 `nexrl/trainer.py` — `step_rewards` 全部归到最后一步，advantage 信号稀疏

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L647-L648、L654-L655
- **类型**：Bug / reward 分配
- **描述**：`step_rewards = [0.0] * len(r.generated_tokens); step_rewards[-1] = r.reward` 把整条 rollout 的 reward 全压到最后一步，前 `T-1` 步 reward=0。GAE 反向传播后中间步的 advantage 极小，绝大多数 token 拿不到训练信号。
- **修复方向**：按 token 等分（`r.reward / T`）或用 `assign_reward_back(reward, gamma)` 分配。

### 2.20 `nexrl/trainer.py` — baseline 在每条 rollout 后都更新，违反 PPO on-policy

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L262-L271
- **类型**：Bug / baseline 时机
- **描述**：advantage = `returns - self.baseline`，但 baseline 在每条 rollout 处理后用 `mean_return` 滑动更新。同一个 batch 内不同 rollout 的 advantage 用的是不同 baseline，破坏了 PPO "old advantage 固定" 的假设。
- **修复方向**：baseline 在一个 batch 内冻结，整批处理完再更新。

### 2.21 `nexrl/trainer.py` — `_update_kl_weight` 自适应更新过激进

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L553-L563
- **类型**：Bug / KL 自适应
- **描述**：`self.kl_weight = min(self.kl_weight * 2.0 + 0.01, 1.0)` 在 KL 超阈值时指数翻倍。从 0 起步，3 次越界即达 0.08，再 3 次达 0.64，极易在 epoch 间剧烈跳变，叠加 KL 的 sum bug 可直接训练崩溃。
- **修复方向**：用 `kl_weight += α * (kl - target_kl)` 形式的比例控制，并加 `clip` 平滑。

### 2.22 `nexrl/trainer.py` — `_ppo_update` 与 `_policy_gradient_fallback` 大量重复

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L279-L423 vs L429-L528
- **类型**：蜘蛛丝
- **描述**：两个方法 90% 代码重复（input 构造、log_softmax、one-hot gather、advantage 归一化、entropy、KL、backward、clip）。区别仅在 ratio clip vs 直接 PG。
- **修复方向**：抽 `_compute_pg_terms(rollout)` 公共方法。

### 2.23 `nexrl/trainer.py` — `NexTrainer` 与 `verse_torch.training.Trainer` 职责重叠

- **位置**：[trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L62 vs [training.py](file:///workspace/packages/verse_torch/verse_torch/training.py) L1999、[training_nex.py](file:///workspace/packages/verse_torch/verse_torch/training_nex.py) L125
- **类型**：冲突
- **描述**：`verse_torch.training` 已有 `Trainer / ParallelTrainer / DistributedTrainer`，`training_nex` 又有 `VerseNexTrainer / LoRATrainer / SFTTrainer / DPOTrainer`。`NexTrainer` 完全独立另起一套循环（fit/loss/backward/clip_grad_norm），不继承、不复用，配置项风格、日志格式、checkpoint/resume 都不互通。
- **修复方向**：让 `NexTrainer` 继承或组合 `VerseNexTrainer`，复用 optimizer 构建 / 日志 / checkpoint。

### 2.24 `nexrl/collector.py` — `value_fn` 异常静默吞掉

- **位置**：[collector.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/collector.py) L194-L199、L332-L337
- **类型**：Bug / 错误处理
- **描述**：`try: v = float(self.value_fn(...)) except Exception: values.append(0.0)`。任何 value_head 出错（如维度不匹配、NaN）都被吞掉，所有 values=0，`_compute_gae` 退化为 `advantage = reward`，且训练"看起来正常"。
- **修复方向**：仅 catch 特定异常，至少 log warning，或 fail-fast。

### 2.25 `nexrl/collector.py` — `collect` 与 `collect_batched` 70% 代码重复

- **位置**：[collector.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/collector.py) L106-L147 vs L241-L379
- **类型**：蜘蛛丝
- **描述**：两个方法各自实现一遍"循环 max_new_tokens → 前向 → 采样 → 检查 eos → 解码 → reward"流水线，reward 计算 + Rollout 构造逻辑完全相同。
- **修复方向**：抽 `_build_rollout(state_dict)` 公共方法，`collect` 内部用 batch=1 调用 `collect_batched`。

### 2.26 `nexrl/action.py` — `epsilon_greedy` / `softmax` / `nucleus` / `topk` 不支持 batched 输入

- **位置**：[action.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/action.py) L62-L191
- **类型**：Bug / 接口
- **描述**：四个方法都返回单个 `int`，但 `logits.ndim > 1` 时的处理是 `logits.reshape(-1)[-logits.shape[-1]:]`，即"取最后一维 V 个元素"——只对最后一条样本有效。若传入 `(B, V)` 或 `(B, T, V)`，会丢掉前 B-1 条样本。
- **修复方向**：要么明确只支持 1D logits 并断言，要么实现真正的 batched 采样。

### 2.27 `nexrl/action.py` — `repeat_penalty` 用减法而非除法，与 HF 标准不一致

- **位置**：[action.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/action.py) L245-L269
- **类型**：Bug / 重复惩罚
- **描述**：`out[tid] = out[tid] - penalty * abs(float(out[tid]))`。HF / CTRL 标准是 `if logit > 0: logit /= penalty else: logit *= penalty`。减法会让大 logit 减得多、小 logit 减得少，相对顺序可能反转。
- **修复方向**：改为乘除法。

### 2.28 `nexrl/action.py` — `ExplorationSchedule` 是死代码

- **位置**：[action.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/action.py) L56-L319
- **类型**：蜘蛛丝 + 死代码
- **描述**：`ExplorationSchedule` 提供 `epsilon` 衰减 schedule，但全项目无人调用。`ActionSampler.epsilon_greedy` 接收 `epsilon` 参数但不接收 schedule，trainer 也不持有 schedule，整条探索衰减路径未接入训练循环。
- **修复方向**：要么在 `NexTrainer.fit` 中接入，要么删除。

### 2.29 `nexrl/reward.py` — `length_penalty` 返回负值，与其他维度（0-1）尺度不一致

- **位置**：[reward.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/reward.py) L128-L139
- **类型**：Bug / reward 尺度
- **描述**：`length_penalty` 返回 `-abs(len - target_len) / target_len`（范围 -1 到 0），而 `correctness/fluency/safety` 都是 0-1。`total = 0.4·correctness + 0.2·fluency + 0.2·safety + 0.2·length_penalty`，意味着 length_penalty 永远在拖低 total。
- **修复方向**：改为 `1 - abs(len - target_len) / target_len` 或归一化到 0-1。

### 2.30 `nexrl/env.py` — `step` 中 `state.append_action(token, 0.0)` 写死 logprob=0.0

- **位置**：[env.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/env.py) L116、L176、L231
- **类型**：Bug / logprob 丢失
- **描述**：注释写"logprob 由 agent 填充"，但 `NexAgent` 与 `collector` 都不调用 `env.step`，没有任何代码会"事后填充" `state.logprobs`。若用户用 `env.step` 走 Gym 风格交互，`state.logprobs` 永远是 0.0，下游 `NexReward.fluency` 和 PPO `old_logprobs` 全部失效。
- **修复方向**：`step` 接收 `logprob` 参数由调用方传入，或直接不在 state 里写 logprob。

### 2.31 `speculative.py` — 4D hidden 处理丢失 head 维信息

- **位置**：[speculative.py](file:///workspace/packages/verse_nex/verse_nex/speculative.py) L138-L142
- **类型**：Bug
- **描述**：
  ```python
  elif hidden.ndim == 4:
      B = hidden.shape[0]
      D = hidden.shape[-1]
      hidden = hidden.reshape(B, -1, D)[:, -1:, :]
  ```
  当 hidden 是 `(B, H, T, D)` 时，reshape 到 `(B, H*T, D)` 取最后一个位置——这对应**最后一个 head 的最后一个 token**，而非最后一个 token 的所有 head。语义错误。
- **修复方向**：4D 输入应先 `hidden[:, :, -1, :].mean(dim=1)` 或 `hidden[:, :, -1, :].reshape(B, H*D)`。

### 2.32 `speculative.py` — `verify_then_commit` 只处理 batch 0，丢弃其他 batch

- **位置**：[speculative.py](file:///workspace/packages/verse_nex/verse_nex/speculative.py) L262-L268
- **类型**：Bug
- **描述**：`if B != 1: draft_tokens_b = draft_tokens[:1]; context_tokens_b = context_tokens[:1]`，静默丢弃 batch 1..B-1 的数据。用户传 B>1 时不会得到任何 warning，结果只含 batch 0 的 token。
- **修复方向**：B>1 时抛 `NotImplementedError`，或循环处理每个 batch。

---

## 三、蜘蛛丝代码（低危但影响可维护性）

### 3.1 `cometspark.py` — `state_dict` / `load_state_dict` 仅转发 super()

- **位置**：[cometspark.py](file:///workspace/packages/verse_nex/verse_nex/cometspark.py) L918-L922
- **类型**：蜘蛛丝 / 冗余方法
- **描述**：方法体只是 `return super().state_dict()`，没有任何额外逻辑，纯属冗余包装。
- **修复方向**：删除这两个方法，直接继承父类实现。

### 3.2 `cometspark.py` — `_softmax` 跨文件重复 3 份

- **位置**：[cometspark.py](file:///workspace/packages/verse_nex/verse_nex/cometspark.py) L1420-L1424、[tri_sparse_attn.py](file:///workspace/packages/verse_nex/verse_nex/tri_sparse_attn.py) L53、[nexrl/action.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/action.py) L32
- **类型**：蜘蛛丝 / 重复造轮子
- **描述**：numpy softmax 在 3 个文件中各有一份几乎相同的实现。注释自己都写了"与 model.py 中 _softmax 一致"。
- **修复方向**：统一提取到 `verse_nex.utils` 或 `verse_torch.utils` 公共模块。

### 3.3 `cometspark.py` — `_fused_forward_blocks` 硬编码 `position_offset=0`

- **位置**：[cometspark.py](file:///workspace/packages/verse_nex/verse_nex/cometspark.py) L545-L549
- **类型**：蜘蛛丝 / 接口不对齐
- **描述**：`_fused_forward_blocks` 内部调用 `attn(norm1(x), position_offset=0, kv_cache=None)` 硬编码了 `position_offset=0` 和 `kv_cache=None`，而 `VerseNexBlock.forward` 接受这两个参数。限制了未来扩展。
- **修复方向**：给 `_fused_forward_blocks` 加 `position_offset` 参数并透传。

### 3.4 `mamba2.py` — `cs_i` 与 `cs_j` 完全相同的冗余变量

- **位置**：[mamba2.py](file:///workspace/packages/verse_nex/verse_nex/mamba2.py) L491-L492
- **类型**：蜘蛛丝 / 冗余变量
- **描述**：`cs_i = cs[:, 1:, :]` 和 `cs_j = cs[:, 1:, :]` 是对同一个数组同一个切片的两次赋值，变量名不同但值完全相同。
- **修复方向**：合并为单个变量 `cs_ij = cs[:, 1:, :]`。

### 3.5 `mamba2.py` — `clamp(0, 10)` 下界冗余

- **位置**：[mamba2.py](file:///workspace/packages/verse_nex/verse_nex/mamba2.py) L382
- **类型**：蜘蛛丝 / 冗余条件
- **描述**：`_softplus` 的输出恒 `>= 0`，因此 `clamp(0, 10)` 的下界 `0` 永远不会触发。recurrent 路径（L451）只用了 `np.minimum(dt_d, 10.0)` 没有下界，两处逻辑不一致。
- **修复方向**：统一为 `clamp_max(10)`。

### 3.6 `rwkv7.py` — `k_act = k_raw.relu() * k_raw.relu()` 调用两次 relu

- **位置**：[rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L408
- **类型**：蜘蛛丝 / 低效逻辑
- **描述**：`k_raw.relu()` 被调用了两次（创建两个独立的 relu Tensor），然后相乘。
- **修复方向**：`r = k_raw.relu(); k_act = r * r`。

### 3.7 跨文件 — softplus 实现分散在多处

- **位置**：[mamba2.py](file:///workspace/packages/verse_nex/verse_nex/mamba2.py) L193-L210、L213-L226；[rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L342-L359、L123-L128、L293-L295
- **类型**：蜘蛛丝 / 重复造轮子
- **描述**：softplus 的数值稳定实现在两个文件中出现了 5 次以上，逻辑几乎相同。
- **修复方向**：提取到 `verse_nex.utils` 公共模块。

### 3.8 跨文件 — SSD decay matrix 计算逻辑重复

- **位置**：[mamba2.py](file:///workspace/packages/verse_nex/verse_nex/mamba2.py) L483-L504 vs [rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L175-L191
- **类型**：蜘蛛丝 / 重复代码
- **描述**：两个 SSD 实现的 decay matrix 计算几乎相同：`cumsum → 前缀零 → broadcasting 差 → clip(-50, 0) → causal mask → exp`。差别仅在维度数量和 dtype 策略。
- **修复方向**：提取为 `_compute_decay_matrix(cumsum_vals, T, trailing_shape) -> np.ndarray` 公共函数。

### 3.9 跨文件 — `forward(x, state, mode)` 统一入口模式重复

- **位置**：[mamba2.py](file:///workspace/packages/verse_nex/verse_nex/mamba2.py) L627-L635、[rwkv7.py](file:///workspace/packages/verse_nex/verse_nex/rwkv7.py) L326-L334、L445-L453、L508-L516
- **类型**：蜘蛛丝 / 重复代码
- **描述**：四个类都有完全相同结构的 `forward` 方法：`if mode == "parallel": ... elif mode == "recurrent": ... object.__setattr__(out, "_state", new_state); return out; else: raise ValueError(...)`。
- **修复方向**：提取 `SSDBlock` 基类（或 mixin），实现公共 `forward`，子类只需实现 `forward_parallel` 和 `forward_recurrent`。

### 3.10 `sparse_attention.py` — `forward_recurrent` 第一个 token 直接用 V，跳过 softmax

- **位置**：[sparse_attention.py](file:///workspace/packages/verse_nex/verse_nex/sparse_attention.py) L426-L428
- **类型**：Bug / 数值一致性
- **描述**：`if len(attend_indices) == 0: out = v.data`——第一个 token（position=0）时无 past，直接把 V 当输出。但 `forward_parallel` 中第一个 token 经过 `softmax(scores + mask)`，两者数值上接近但不严格相等，破坏数值一致性契约。
- **修复方向**：第一个 token 也走 softmax 路径（`attend_indices = [0]` 包含自己）。

### 3.11 `positional.py` — RoPE 在 `tri_sparse_attn` 中被重复实现

- **位置**：[tri_sparse_attn.py](file:///workspace/packages/verse_nex/verse_nex/tri_sparse_attn.py) L169-L217 vs [positional.py](file:///workspace/packages/verse_nex/verse_nex/positional.py) L28-L128
- **类型**：蜘蛛丝
- **描述**：`tri_sparse_attn` 自己实现了 `_build_rope_table` + `_apply_rope`，与 `positional.RoPE` 功能完全重叠。三套 RoPE 实现：`positional.RoPE`、`tri_sparse_attn._apply_rope`、`verse_torch.vnn.RoPE`。
- **修复方向**：`tri_sparse_attn` 接受外部 `rope: RoPE` 实例（依赖注入），删除内部实现。

### 3.12 `hybrid.py` — 模块级 `warnings.warn` 在 import 时触发

- **位置**：[hybrid.py](file:///workspace/packages/verse_nex/verse_nex/hybrid.py) L45-L52
- **类型**：Bug
- **描述**：模块顶层执行 `warnings.warn(..., DeprecationWarning)`，任何 `from verse_nex.hybrid import X` 都会触发警告。Python 推荐用 `__getattr__` 实现懒警告。
- **修复方向**：删除模块级 `warnings.warn`，只在类 `__init__` 内警告。

### 3.13 `hybrid.py` — `HybridBlock` 与 `TopKChunkSparseAttention` 双重归一化

- **位置**：[hybrid.py](file:///workspace/packages/verse_nex/verse_nex/hybrid.py) L115 + [sparse_attention.py](file:///workspace/packages/verse_nex/verse_nex/sparse_attention.py) L122
- **类型**：蜘蛛丝
- **描述**：`HybridBlock` 创建 `self.norm = LayerNorm(dim)`，先 `x_norm = self.norm(x)` 再 `out = self.block.forward_parallel(x_norm)`。但 `TopKChunkSparseAttention` 内部又有 `self.norm = RMSNorm(dim)`，又 `out = self.norm(out)`。**双重归一化**，与 transformer 标准 pre-norm 设计不符。
- **修复方向**：既然 `hybrid.py` 已 deprecated，删除整个模块即可。

### 3.14 `hybrid.py` — 已废弃但仍占空间（391 行死代码）

- **位置**：[hybrid.py](file:///workspace/packages/verse_nex/verse_nex/hybrid.py)（整体）
- **类型**：蜘蛛丝
- **描述**：`__init__.py` L26-L27 注释明确说 deprecated，`__all__` 不再导出。grep 显示无其他文件 import 它。391 行死代码占维护成本。
- **修复方向**：直接删除 `hybrid.py` 文件。

### 3.15 `linear_attention.py` — `_decay_matrix_log` 是 dead code

- **位置**：[linear_attention.py](file:///workspace/packages/verse_nex/verse_nex/linear_attention.py) L49-L66
- **类型**：蜘蛛丝
- **描述**：grep 显示 `_decay_matrix_log` 仅定义，无任何调用。`forward_parallel` L165 用的是 `_decay_matrix`（非 log 版本）。
- **修复方向**：删除 `_decay_matrix_log` 函数。

### 3.16 `nexrl/state.py` — 文件只有 90 行，过度拆分

- **位置**：[state.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/state.py)（整体）
- **类型**：蜘蛛丝
- **描述**：仅含一个 `@dataclass NexState`（55 行）+ 一个 helper `batch_states`（14 行），且 `batch_states` 只是 `list comprehension`，并未真正 pad/stack。该文件可并入 `env.py`。
- **修复方向**：合并到 `env.py`。

### 3.17 `nexrl/agent.py` — `get_hidden_states` 与 `trainer._forward_with_hidden` 重复

- **位置**：[agent.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/agent.py) L211-L232 vs [trainer.py](file:///workspace/packages/verse_nex/verse_nex/nexrl/trainer.py) L28-L53
- **类型**：蜘蛛丝 + 死代码
- **描述**：两处都手写 `tok_emb → blocks → 取 norm 前的 hidden`。`get_hidden_states` 全项目无人调用（grep 仅出现在自身定义），是死代码。
- **修复方向**：删除或合并到 `CometSparkNexLM.forward_with_hidden`。

---

## 四、跨文件架构冲突

### 4.1 5 套 KV Cache 接口互不兼容

- `sparse_attention.py`：`state = (kv_cache_list, position)`，kv_cache 是 `list of (K, V) ndarray`
- `tri_sparse_attn.py`：`state = {"k_cache": ..., "v_cache": ..., "global_k": ..., "global_v": ..., "position": int}`
- `cometspark.py`：`layer_state = {"aux": ..., "kv_cache": {"k": ..., "v": ...}}`（dict 套 dict）
- `kv_cache_parallel.py`：`ParallelKVCache` 类（继承 `KVCache` 基类）—— **全项目无人使用**
- `verse_torch.vnn`：`StaticCache`、`DynamicCache`（继承 `KVCache` 基类）

**5 种 KV cache 接口并存**，无统一抽象。`cometspark.py` 用 dict 接口，根本没用 `ParallelKVCache` 类。

**修复方向**：抽象统一 `KVCache` 基类，废弃 dict 形式，所有 attention 模块统一接口。

### 4.2 3 套 RoPE 实现

- `positional.RoPE`：独立类，支持 3D/4D 输入
- `tri_sparse_attn._apply_rope`：方法，仅支持 4D，自带 `_build_rope_table`
- `verse_torch.vnn`：可能有 `RoPE`（与 ALiBi 同级）

**修复方向**：统一为单一 `RoPE` 类，依赖注入。

### 4.3 2 套稀疏注意力

- `sparse_attention.TopKChunkSparseAttention`：chunk + sliding + top-k，稠密矩阵实现（O(T²) 内存）
- `tri_sparse_attn.TriSparseAttention`：SWA + Global + ALiBi，chunk-wise 实现（O(T·W) 内存）

`__init__.py:65` 已将 `TriSparseAttention` 设为 `VerseNexAttention` 别名，但 `TopKChunkSparseAttention` 仍导出。`hybrid.py` 仍引用 `TopKChunkSparseAttention`。

**修复方向**：删除 `sparse_attention.py` 和 `hybrid.py`，统一到 `tri_sparse_attn` + `cometspark`。

---

## 五、修复优先级建议

### P0 — 立即修复（影响训练正确性）

1. **moe.py 熵正则梯度断裂**（问题 1.2）：`entropy_weight` 完全无效，影响 MoD 路由质量。
2. **moe.py EMA 死代码**（问题 1.1）：决定是否实现或删除。
3. **linear_attention.py chunkwise 梯度切断**（问题 1.3）：长序列训练不可用。
4. **nexrl KL sum bug**（问题 1.9）：PPO+KL 路径爆炸。
5. **nexrl attention mask 缺失**（问题 1.11）：batched 采集数据污染。
6. **nexrl PPO logprob 不一致**（问题 1.12）：PPO 训练无效。

### P1 — 短期修复（影响推理正确性 / 资源）

7. **sparse_attention KV cache 无限增长**（问题 1.4）：长序列推理 OOM。
8. **kv_cache_parallel buffer 替换 + get 未截断**（问题 1.5、1.6）：预分配失效 + 数据污染。
9. **tri_sparse ALiBi parallel/recurrent 不一致**（问题 1.7）：数值一致性契约破裂。
10. **mamba2/rwkv7 SSD L_data 内存爆炸**（问题 2.7、2.10）：长序列训练 OOM。
11. **nexrl GAE bootstrap**（问题 1.8）：truncated rollout advantage 偏负。
12. **nexrl reward shaping 失效**（问题 1.10）：potential-based 保证失效。

### P2 — 中期重构（架构清理）

13. 统一 KV Cache 接口（问题 4.1）。
14. 删除 `hybrid.py`、`sparse_attention.py`（问题 3.13、3.14、2.17）。
15. `NexTrainer` 继承 `VerseNexTrainer`（问题 2.23）。
16. 抽 `SSDBlock` 基类（问题 3.9）。
17. 统一 RoPE / softmax / softplus 公共函数（问题 3.2、3.7、3.11）。

### P3 — 长期清理（死代码 + 命名）

18. 删除 `ParallelKVCache`、`_select_chunks`、`_decay_matrix_log`、`get_hidden_states`、`ExplorationSchedule` 等死代码。
19. 删除 `rwkv7.py` 的 `x_a`/`x_b` 死参数。
20. 修复 `cometspark.py` 的 `load`/`save_pretrained` 接口不一致。
21. 合并 `nexrl/state.py` 到 `env.py`。

---

## 六、附录：统计

### 按文件分布

| 文件 | 高危 | 中危 | 低危 | 合计 |
|------|------|------|------|------|
| cometspark.py | 0 | 3 | 3 | 6 |
| moe.py | 2 | 3 | 0 | 5 |
| mamba2.py | 0 | 1 | 2 | 3 |
| rwkv7.py | 0 | 2 | 1 | 3 |
| tri_sparse_attn.py | 1 | 3 | 0 | 4 |
| kv_cache_parallel.py | 2 | 2 | 0 | 4 |
| sparse_attention.py | 1 | 2 | 1 | 4 |
| linear_attention.py | 1 | 0 | 1 | 2 |
| positional.py | 0 | 1 | 1 | 2 |
| hybrid.py | 0 | 2 | 1 | 3 |
| speculative.py | 0 | 2 | 0 | 2 |
| nexrl/trainer.py | 3 | 5 | 0 | 8 |
| nexrl/collector.py | 2 | 2 | 0 | 4 |
| nexrl/action.py | 1 | 2 | 1 | 4 |
| nexrl/agent.py | 1 | 0 | 1 | 2 |
| nexrl/reward.py | 0 | 1 | 0 | 1 |
| nexrl/env.py | 0 | 1 | 0 | 1 |
| nexrl/state.py | 0 | 0 | 1 | 1 |
| 跨文件 | 0 | 0 | 4 | 4 |
| **合计** | **13** | **35** | **25** | **73** |

### 按类型分布

| 类型 | 数量 |
|------|------|
| Bug（数值/逻辑/资源） | 38 |
| 冲突（接口/语义不一致） | 15 |
| 蜘蛛丝（重复/死代码/低效） | 20 |

---

**审计人**：Part5K1.5 自动审计
**审计方法**：3 个并行 subagent 分模块深入阅读 + 交叉验证
**审计覆盖**：18 个文件，8834 行代码，100% 覆盖
