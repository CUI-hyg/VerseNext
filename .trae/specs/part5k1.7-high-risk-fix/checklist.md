# Checklist

## 阶段一：MoD 重命名 + 1.1/1.2 高危修复

- [x] `packages/verse_nex/verse_nex/moe.py` 已重命名为 `mod.py`
- [x] `packages/verse_nex/verse_nex/__init__.py` 中 `from .moe import` 已改为 `from .mod import`
- [x] 全项目无残留的 `from verse_nex.moe` / `from .moe import` / `verse_nex.moe` 引用（除 .trae/specs 和 .agent_work/UPDATE.MD）
- [x] `python -c "from verse_nex import Router, Expert, DensePart, MoDLayer"` 成功
- [x] `mod.py` Router.forward 中 EMA 仅作为监控标量，不参与 `aux_loss` 数值
- [x] `mod.py` 文档已修正"EMA 参与总 aux loss"的错误描述
- [x] `mod.py` Router.forward 熵正则用 Tensor 运算实现（`-(probs * (probs+eps).log()).sum(dim=-1).mean()`）
- [x] `entropy_weight > 0` 时 `aux_loss` 中熵正则项 `requires_grad=True`
- [x] `entropy_weight > 0` 时 backward 后 `gate.weight.grad` 非零
- [x] 已删除 `requires_grad=False` 的常量熵正则分支

## 阶段二：长序列 / KV Cache 高危修复（1.3 – 1.7）

- [x] `linear_attention.forward_chunkwise` 用 `verse_torch.vnn._concat` 替换 `np.concatenate`
- [x] `linear_attention.forward_chunkwise` pad 用 `sparse_attention._pad_last_dim` 替换零张量拼接
- [x] `forward_chunkwise` 反向传播可回传到 qkv 投影权重（梯度非零）
- [x] `sparse_attention.forward_recurrent` 实现 KV cache 修剪（保留最近 `max_kv_chunks * C` 个 token）
- [x] `forward_recurrent` 用偏移量 `cache_offset` 记录全局位置
- [x] `attend_indices` 与 `kv_cache` 长度一致
- [x] `kv_cache_parallel.batch_update` 将 new_k/new_v 写入预分配 buffer 切片 `[:B, :T_new_total]`
- [x] `batch_update` 保留 `(max_batch, max_seq, H, D)` 形状
- [x] `kv_cache_parallel.reset()` 用初始 `max_seq` 形状重建 buffer（缓存初始形状）
- [x] `kv_cache_parallel.get()` 按 `max(per_seq_lens[:B])` 截断，返回 `(K[:B, :max_len], V[:B, :max_len])`
- [x] `tri_sparse_attn` recurrent ALiBi 判据改为 `n_cached <= self._ALIBI_MAX_T`
- [x] `tri_sparse_attn` ALiBi 路径独立计算 `q @ k^T`，不复用 `swa_scores`
- [x] parallel 与 recurrent 在 `position=1500, W=512, T_k=512` 场景下行为一致

## 阶段三：nexrl GAE / KL / RewardShaper 修复（1.8 – 1.10）

- [x] `Rollout` dataclass 增加 `truncated: bool` 字段
- [x] `collector` 在 `max_len` 截断时设置 `truncated=True`，eos 时 `truncated=False`
- [x] `_compute_gae` 截断时 `next_value = values[T]`，eos 时 `next_value = 0`
- [x] `agent.compute_kl` 改为 `(...).sum(dim=-1).mean()`
- [x] `compute_kl` 与 `compute_kl_scalar` 语义统一
- [x] `kl_weight > 0` 时 loss 不爆炸
- [x] `nexrl/reward.py` 中 `RewardShaper` 类已删除（注：实际位于 reward.py 而非 action.py）
- [x] `nexrl/trainer.py` 中 `RewardShaper` 调用与 import 已删除
- [x] `nexrl/__init__.py` 与 `verse_nex/__init__.py` 中 `RewardShaper` 导出已删除

## 阶段四：nexrl 采样 / 采集修复（1.11 – 1.13）

- [x] `collector.collect_batched` 改用左 padding（pad_id 放在序列左侧）
- [x] `collect_batched` 构造 attention mask，padding 位置为 0
- [x] 采用选项 C：`last_logits = logits_np[j, -1, :]`（左 padding 后末尾对齐，不改 forward_policy 接口）
- [x] `last_logits` 定位到真实序列末尾（左 padding 后末尾位置固定）
- [x] `ActionSampler.sample` nucleus：截断后重算 softmax 取 log 作为 logprob
- [x] `ActionSampler.sample` topk：截断后重归一化取 log
- [x] `ActionSampler.sample` epsilon_greedy：用混合分布 `(1-ε)·argmax + ε·uniform` 的 log
- [x] `ActionSampler.sample` softmax 策略保持 `log_softmax` 不变
- [x] `agent.act` 返回 `(token_id, logprob, logits)` 元组（不再构造 `NexAction`）
- [x] `nexrl/action.py` 中 `NexAction` 类已删除
- [x] `nexrl/__init__.py` 与 `verse_nex/__init__.py` 中 `NexAction` 导出已删除
- [x] `collector._single_rollout` 改为调用 `agent.sampler.sample`，删除重复的前向+采样逻辑
- [x] `collect_batched` 同样调用 `agent.sampler.sample`
- [x] `collect` 与 `collect_batched` 公共逻辑已抽取（`_build_rollout`）

## 阶段五：训练阻塞 Bug 修复

- [x] `_load_tokenizer` 所有分支显式返回 tokenizer，不得返回 `None`
- [x] 兜底失败时抛 `RuntimeError("无法加载 tokenizer: ...")` 含路径与 kind
- [x] 调用处 `tok = _load_tokenizer(...)` 后有 `None` 校验
- [x] `from model.tokenizer import load_tokenizer` ModuleNotFoundError 已修复（删除失效兜底）
- [x] `demo_dir` 解析正确（确认 data/demo 不存在），已委托 verse_infra.verse_tokenizer

## 阶段六：文档同步与验证

- [x] `UPDATE.MD` 中 1.1–1.13 的 13 个高危条目已删除
- [x] `UPDATE.MD` 摘要表：高危 13 → 0，合计 73 → 60
- [x] `UPDATE.MD`"按文件分布"表各文件高危列已清零
- [x] `UPDATE.MD`"修复优先级建议"中已完成项已标记
- [x] `UPDATE.MD` 中危/低危条目保留（本次不修复）
- [x] 全项目文档/注释中 `moe.py` 已更新为 `mod.py`（补充 docs/part4_upgrade_report.md）
- [x] `__init__.py` 导出不变（`Router/Expert/DensePart/MoDLayer` 仍可导入）
- [x] `python -m pytest tests/test_cometspark_v05.py tests/test_dual_model.py tests/test_spark_run.py tests/test_checkpoint_vn.py tests/test_mod_complete.py tests/test_nexrl.py -x -q` 通过（225 passed, 1 skipped）
- [x] nexrl 相关测试通过（test_nexrl.py 修复了 NexAction/RewardShaper/采样器返回值适配）
- [x] MoD 熵正则梯度测试通过（`entropy_weight > 0` 时 `gate.weight.grad` 非零）
- [x] 额外测试套件通过（test_parallel_sparse_attn/test_cometspark_nex/test_recursion_fix/test_hybrid_stability/test_training_nex/test_p10_parallel_compress → 95 passed, 1 skipped）
