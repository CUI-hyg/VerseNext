# Part5K1.7 高危修复 Spec

## Why

`UPDATE.MD` 审计出 13 个高危 Bug，覆盖 MoD 路由、长序列梯度、KV Cache、RL 训练四条核心路径：
- MoD 的 EMA 平滑与熵正则**实际未生效**，路由质量受损；
- `linear_attention.forward_chunkwise` 用 `np.concatenate` 切断梯度，长序列训练不可用；
- KV Cache 在 recurrent/parallel 路径下无限增长 / buffer 缩水 / 数据污染；
- nexrl 的 PPO+KL 路径因 logprob 不一致、KL 求和错误、attention mask 缺失而**完全不可信**。

同时 `verse_infra/verse_trainer/trainer.py` 存在两个阻塞性 Bug：`_load_tokenizer` 返回 `None` 导致 `len(tok)` 抛 `TypeError`；兜底导入 `from model.tokenizer import load_tokenizer` 找不到 `model` 模块。

本 spec 落地 K1.7 的高危修复 + 两个训练阻塞 Bug，并同步清理 `UPDATE.MD` 中已修复的条目。

## What Changes

### 一、MoD 模块重命名 + 高危修复（1.1 / 1.2）
- **BREAKING（包内）**：`packages/verse_nex/verse_nex/moe.py` → `mod.py`；同步更新 `__init__.py`、所有 `from .moe import` / `from verse_nex.moe import` 引用、文档字符串中的 `moe.py`。
- 1.1 EMA：在 `aux_loss` 中合并 EMA 与可微路径——可微 `load_balance_loss` 仍参与 backward；EMA 仅作为**监控/日志**标量保留（避免不可微 EMA 污染梯度），并修正文档"EMA 参与总 aux loss"的错误描述。
- 1.2 熵正则：用 Tensor 运算实现可微熵正则 `entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()`，`aux_loss = aux_loss - entropy_weight * entropy`，删除 `requires_grad=False` 的常量分支。

### 二、长序列 / KV Cache 高危修复（1.3 – 1.10）
- 1.3 `linear_attention.forward_chunkwise`：用 `verse_torch.vnn._concat`（可微拼接）+ `verse_nex.sparse_attention._pad_last_dim`（带梯度 pad）替换 `np.concatenate`，恢复 qkv 投影梯度路径。
- 1.4 `sparse_attention.forward_recurrent`：实现真正的 KV cache 修剪，保留最近 `max_kv_chunks * C` 个 token，并用偏移量记录全局位置。
- 1.5 `kv_cache_parallel.batch_update`：将 `new_k`/`new_v` 写入预分配 buffer 的 `[:B, :T_new_total]` 切片，保留 `max_seq` 维度；`reset()` 重建 buffer 时用初始 `max_seq` 形状。
- 1.6 `kv_cache_parallel.get()`：按 `max(self.per_seq_lens[:B])` 截断，返回 `(K[:B, :max_len], V[:B, :max_len])`。
- 1.7 `tri_sparse_attn` ALiBi 路径：recurrent 判据改为 `n_cached <= self._ALIBI_MAX_T`，并让 ALiBi 路径独立计算 `q @ k^T` 而非复用 `swa_scores`，统一 parallel/recurrent 语义。
- 1.8 `nexrl/trainer._compute_gae`：`Rollout` 携带 `truncated` 标志，截断时 `next_value = values[T]`，仅 eos 终止时 `next_value = 0`。
- 1.9 `nexrl compute_kl`：改为 `(...).sum(dim=-1).mean()`（per-token 平均），统一 `compute_kl` 与 `compute_kl_scalar` 接口语义。
- 1.10 `nexrl RewardShaper`：**直接删除该路径**（用户指定），同步删除 trainer 中相关调用与文档。

### 三、nexrl 采样 / 采集高危修复（1.11 – 1.13）
- 1.11 `collector.collect_batched`：改用**左 padding + attention mask**，padding 位置（id=0 替换为 pad_id）通过 mask 屏蔽，避免污染前向。
- 1.12 `ActionSampler.sample` logprob：用**实际采样分布的 log**（截断后重归一化）——nucleus/topk 截断后重算 softmax 取 log；epsilon_greedy 用混合分布 `(1-ε)·argmax + ε·uniform` 的 log。
- 1.13 `agent.act` 与 `collector._single_rollout` 重复：**让 collector 调用 `agent.act`**，删除 `NexAction` 死代码（collector/trainer/env 不再使用），合并相似组件。

### 四、训练阻塞 Bug 修复
- Bug 1：`trainer.py:990` `len(tok)` 抛 `TypeError`——`_load_tokenizer` 兜底失败时返回 `None`。修复：所有分支显式返回 tokenizer 或抛 `RuntimeError`，调用处增加 `None` 校验。
- Bug 2：`trainer.py:776` `from model.tokenizer import load_tokenizer` 找不到 `model`——兜底路径 `demo_dir` 解析错误。修复：修正 `demo_dir` 解析（指向真正的 `data/demo/` 根，使 `model.tokenizer` 可导入），或直接委托 `verse_infra.verse_tokenizer`，删除失效的 `model.tokenizer` 兜底。

### 五、文档同步
- 修复完成后，**清除 `UPDATE.MD` 中已修复的 1.1–1.13 条目**（用户指定），并更新摘要表的数量统计。
- 同步更新 `moe.py` → `mod.py` 的所有文档引用。

## Impact

- **Affected specs**：`part5k1-vmpc-dual-model`（MoD 配置）、`part5k1.3-bugfix-stability-upgrade`（训练稳定性）。
- **Affected code**：
  - `packages/verse_nex/verse_nex/moe.py` → `mod.py`（重命名 + 1.1/1.2）
  - `packages/verse_nex/verse_nex/__init__.py`（导入路径）
  - `packages/verse_nex/verse_nex/linear_attention.py`（1.3）
  - `packages/verse_nex/verse_nex/sparse_attention.py`（1.4）
  - `packages/verse_nex/verse_nex/kv_cache_parallel.py`（1.5/1.6）
  - `packages/verse_nex/verse_nex/tri_sparse_attn.py`（1.7）
  - `packages/verse_nex/verse_nex/nexrl/{trainer,collector,action,agent,env,state}.py`（1.8–1.13）
  - `packages/verse_infra/verse_infra/verse_trainer/trainer.py`（Bug 1/2）
  - 所有 `from verse_nex.moe import` / `from .moe import` 的引用点
  - `.agent_work/UPDATE.MD`（清除已修复条目）
- **BREAKING**：`moe.py` 重命名为 `mod.py`，外部若直接 `from verse_nex.moe import` 需改为 `from verse_nex.mod import`（包内引用全部同步更新；`__init__.py` 仍导出 `Router/Expert/DensePart/MoDLayer`，公共 API 不变）。
- **RL 路径**：nexrl 修复后 PPO+KL 路径才可用，之前 nucleus/topk/epsilon_greedy 策略下 PPO 训练无效。

## ADDED Requirements

### Requirement: 可微熵正则化
MoD Router 的 `aux_loss` SHALL 通过 Tensor 运算实现可微熵正则化，`entropy_weight > 0` 时 SHALL 产生梯度回传到 `gate` 权重。

#### Scenario: 熵正则生效
- **WHEN** `entropy_weight > 0` 且模型处于训练模式
- **THEN** `aux_loss` 中包含 `-entropy_weight * entropy` 项，且该项 `requires_grad=True`
- **AND** backward 后 `gate.weight.grad` 非零

### Requirement: KV Cache 修剪
`sparse_attention.forward_recurrent` SHALL 在每步 append 后修剪 KV cache，保留最近 `max_kv_chunks * C` 个 token。

#### Scenario: 长序列推理不 OOM
- **WHEN** 推理序列长度 > `max_kv_chunks * C`
- **THEN** `kv_cache` 长度恒定在 `max_kv_chunks * C`
- **AND** 内存占用 O(C) 而非 O(T)

### Requirement: PPO 采样分布一致性
`ActionSampler.sample` 返回的 `logprob` SHALL 等于实际采样分布（截断后重归一化）的 log。

#### Scenario: nucleus 采样 logprob 正确
- **WHEN** strategy="nucleus", top_p=0.9
- **THEN** `logprob == log(softmax(truncated_logits)[token_id])`
- **AND** PPO 的 IS 假设 `behavior = old policy` 成立

### Requirement: 训练 tokenizer 加载健壮
`_load_tokenizer` SHALL 始终返回有效 tokenizer 或抛 `RuntimeError`，不得返回 `None`。

#### Scenario: 兜底路径失效
- **WHEN** `verse_infra.verse_tokenizer` 与 `model.tokenizer` 均不可用
- **THEN** 抛 `RuntimeError("无法加载 tokenizer: ...")` 而非返回 `None`
- **AND** 错误信息包含已尝试的路径与 kind

## MODIFIED Requirements

### Requirement: MoD Router aux_loss 计算
Router.forward 返回的 `aux_loss` SHALL = `load_balance_loss + z_loss - entropy_weight * entropy`，其中 `load_balance_loss` 和 `entropy` 均为可微 Tensor；EMA 仅作为监控标量（`_load_balance_ema`）记录，不参与 `aux_loss` 数值。

### Requirement: ParallelKVCache 预分配语义
`batch_update` SHALL 将新 K/V 写入预分配 buffer 的 `[:B, :T_new_total]` 切片，保留 `(max_batch, max_seq, H, D)` 形状；`get()` SHALL 按 `max(per_seq_lens[:B])` 截断返回。

### Requirement: nexrl collector 采集
`collect_batched` SHALL 使用左 padding + attention mask，padding token 不参与前向计算；collector SHALL 调用 `agent.act` 完成采样，不得重复实现前向+采样逻辑。

## REMOVED Requirements

### Requirement: RewardShaper 势函数塑造
**Reason**: 当前实现只对终末 reward 做一次 shaping，势函数 policy-invariance 保证失效；按用户指定直接删除该路径。
**Migration**: `NexTrainer` 中 `RewardShaper` 调用点删除；reward 归一化保留（base reward normalize）。

### Requirement: NexAction 数据类
**Reason**: `agent.act` 内部构造但 collector/trainer/env 从不消费，是事实上的死代码；collector 改为调用 `agent.act` 后直接使用 `(token_id, logprob)` 元组。
**Migration**: 删除 `nexrl/action.py` 中的 `NexAction` 类；`agent.act` 返回 `(token_id, logprob, logits)` 元组；`__init__.py` 移除 `NexAction` 导出。
