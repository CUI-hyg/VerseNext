# Tasks

## 阶段一：MoD 重命名 + 1.1/1.2 高危修复

- [x] Task 1: 将 `packages/verse_nex/verse_nex/moe.py` 重命名为 `mod.py`，同步更新所有引用
  - [x] SubTask 1.1: `git mv`（或 Read+Write+Delete）将 `moe.py` → `mod.py`
  - [x] SubTask 1.2: 更新 `packages/verse_nex/verse_nex/__init__.py`：`from .moe import` → `from .mod import`
  - [x] SubTask 1.3: grep 全项目 `from verse_nex.moe` / `from .moe import` / `verse_nex.moe` 引用，逐一更新
  - [x] SubTask 1.4: 更新 `mod.py` 内部文档字符串中的 `moe.py` → `mod.py`
  - [x] SubTask 1.5: 运行 `python -c "from verse_nex import Router, MoDLayer"` 验证导入

- [x] Task 2: 修复 1.1 EMA 死代码（`mod.py` Router.forward）
  - [x] SubTask 2.1: EMA 计算保留（`_load_balance_ema` 仍更新），但仅作为监控标量
  - [x] SubTask 2.2: 确认 `aux_loss` 中不使用 `_load_balance_ema`（可微 `load_balance_loss` 仍参与 backward）
  - [x] SubTask 2.3: 修正文档：删除"EMA 参与总 aux loss"的描述，改为"EMA 仅用于监控/日志"
  - [x] SubTask 2.4: 验证 `ema_decay > 0` 时 backward 正常，`gate.weight.grad` 非零

- [x] Task 3: 修复 1.2 熵正则梯度断裂（`mod.py` Router.forward）
  - [x] SubTask 3.1: 用 Tensor 运算实现可微熵正则：`entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()`
  - [x] SubTask 3.2: `aux_loss = load_balance_loss + z_loss - entropy_weight * entropy`
  - [x] SubTask 3.3: 删除 `requires_grad=False` 的常量分支
  - [x] SubTask 3.4: 更新文档：熵正则可微，梯度回传到 `gate` 权重
  - [x] SubTask 3.5: 验证 `entropy_weight > 0` 时 `gate.weight.grad` 非零

## 阶段二：长序列 / KV Cache 高危修复（1.3 – 1.7）

- [x] Task 4: 修复 1.3 `linear_attention.forward_chunkwise` 梯度断裂
  - [x] SubTask 4.1: 用 `verse_torch.vnn._concat` 替换 q/k/v 的 `np.concatenate` 拼接
  - [x] SubTask 4.2: 用 `verse_nex.sparse_attention._pad_last_dim` 替换 pad 零张量拼接
  - [x] SubTask 4.3: 验证 chunk 输出拼接处也用可微 `_concat`
  - [x] SubTask 4.4: 更新文档：`forward_chunkwise` 现支持梯度回传

- [x] Task 5: 修复 1.4 `sparse_attention.forward_recurrent` KV cache 无限增长
  - [x] SubTask 5.1: 实现 cache 修剪：保留最近 `max_kv_chunks * C` 个 token
  - [x] SubTask 5.2: 用偏移量 `cache_offset` 记录全局位置
  - [x] SubTask 5.3: 修正 `attend_indices` 与 `kv_cache` 长度一致
  - [x] SubTask 5.4: 删除 `pass  # 简化：不修剪` 注释，实现真正修剪
  - [x] SubTask 5.5: 更新文档

- [x] Task 6: 修复 1.5 `kv_cache_parallel.batch_update` buffer 替换
  - [x] SubTask 6.1: 将 `new_k`/`new_v` 写入预分配 buffer 的 `[:B, :T_new_total]` 切片
  - [x] SubTask 6.2: 保留 `(max_batch, max_seq, H, D)` 形状
  - [x] SubTask 6.3: 修正 `reset()`：用初始 `max_seq` 形状重建 buffer（缓存初始形状）
  - [x] SubTask 6.4: 更新文档

- [x] Task 7: 修复 1.6 `kv_cache_parallel.get()` 未按 `per_seq_lens` 截断
  - [x] SubTask 7.1: 按 `max(self.per_seq_lens[:B])` 截断
  - [x] SubTask 7.2: 返回 `(K[:B, :max_len], V[:B, :max_len])`
  - [x] SubTask 7.3: 更新文档

- [x] Task 8: 修复 1.7 `tri_sparse_attn` ALiBi parallel/recurrent 不一致
  - [x] SubTask 8.1: recurrent 判据改为 `n_cached <= self._ALIBI_MAX_T`
  - [x] SubTask 8.2: ALiBi 路径独立计算 `q @ k^T`，不再复用 `swa_scores`
  - [x] SubTask 8.3: 验证 parallel 与 recurrent 在 `position=1500, W=512, T_k=512` 场景下行为一致
  - [x] SubTask 8.4: 更新文档

## 阶段三：nexrl GAE / KL / RewardShaper 修复（1.8 – 1.10）

- [x] Task 9: 修复 1.8 `nexrl/trainer._compute_gae` truncated rollout
  - [x] SubTask 9.1: `Rollout` dataclass 增加 `truncated: bool` 字段
  - [x] SubTask 9.2: `collector` 在 `max_len` 截断时设置 `truncated=True`，eos 时 `truncated=False`
  - [x] SubTask 9.3: `_compute_gae`：截断时 `next_value = values[T]`，eos 时 `next_value = 0`
  - [x] SubTask 9.4: 更新文档

- [x] Task 10: 修复 1.9 `nexrl compute_kl` sum → mean
  - [x] SubTask 10.1: `agent.compute_kl` 改为 `(...).sum(dim=-1).mean()`
  - [x] SubTask 10.2: 统一 `compute_kl` 与 `compute_kl_scalar` 语义
  - [x] SubTask 10.3: 验证 `kl_weight > 0` 时 loss 不爆炸
  - [x] SubTask 10.4: 更新文档

- [x] Task 11: 删除 1.10 `nexrl RewardShaper` 路径
  - [x] SubTask 11.1: 删除 `nexrl/reward.py` 中的 `RewardShaper` 类（注：实际位于 reward.py 而非 action.py）
  - [x] SubTask 11.2: 删除 `nexrl/trainer.py` 中 `RewardShaper` 的调用与 import
  - [x] SubTask 11.3: 删除 `nexrl/__init__.py` 与 `verse_nex/__init__.py` 中 `RewardShaper` 导出
  - [x] SubTask 11.4: 更新文档（删除"势函数塑造"相关描述）

## 阶段四：nexrl 采样 / 采集修复（1.11 – 1.13）

- [x] Task 12: 修复 1.11 `collector.collect_batched` attention mask 缺失
  - [x] SubTask 12.1: 改用左 padding（pad_id 放在序列左侧）
  - [x] SubTask 12.2: 构造 attention mask `(B, max_len)`，padding 位置为 0
  - [x] SubTask 12.3: 采用选项 C：`last_logits = logits_np[j, -1, :]`（左 padding 后末尾对齐，不改 forward_policy 接口）
  - [x] SubTask 12.4: 取 `last_logits` 时定位到真实序列末尾（左 padding 后末尾位置固定）
  - [x] SubTask 12.5: 更新文档

- [x] Task 13: 修复 1.12 `ActionSampler.sample` logprob 不一致
  - [x] SubTask 13.1: nucleus：截断后重算 softmax 取 log 作为 logprob
  - [x] SubTask 13.2: topk：截断后重归一化取 log
  - [x] SubTask 13.3: epsilon_greedy：用混合分布 `(1-ε)·argmax + ε·uniform` 的 log
  - [x] SubTask 13.4: softmax 策略保持 `log_softmax` 不变
  - [x] SubTask 13.5: 重构 `sample` 方法，让子方法返回 `(token_id, sampling_logprob)` 而非仅 `token_id`
  - [x] SubTask 13.6: 更新文档

- [x] Task 14: 修复 1.13 `agent.act` 与 collector 重复 + 删除 `NexAction` 死代码
  - [x] SubTask 14.1: `agent.act` 返回 `(token_id, logprob, logits)` 元组（删除 `NexAction` 构造）
  - [x] SubTask 14.2: 删除 `nexrl/action.py` 中的 `NexAction` 类
  - [x] SubTask 14.3: 删除 `nexrl/__init__.py` 与 `verse_nex/__init__.py` 中 `NexAction` 导出
  - [x] SubTask 14.4: `collector._single_rollout` 改为调用 `agent.sampler.sample`，删除重复的前向+采样逻辑
  - [x] SubTask 14.5: `collect_batched` 同样调用 `agent.sampler.sample`
  - [x] SubTask 14.6: 合并 `collect` 与 `collect_batched` 的公共逻辑（抽 `_build_rollout`）
  - [x] SubTask 14.7: 更新文档

## 阶段五：训练阻塞 Bug 修复

- [x] Task 15: 修复 Bug 1 `trainer.py:990` `len(tok)` TypeError
  - [x] SubTask 15.1: `_load_tokenizer` 所有分支显式返回 tokenizer，不得返回 `None`
  - [x] SubTask 15.2: 兜底失败时抛 `RuntimeError("无法加载 tokenizer: ...")` 含路径与 kind
  - [x] SubTask 15.3: 调用处 `tok = _load_tokenizer(...)` 后增加 `if tok is None: raise RuntimeError`
  - [x] SubTask 15.4: 更新文档

- [x] Task 16: 修复 Bug 2 `trainer.py:776` `from model.tokenizer import load_tokenizer` ModuleNotFoundError
  - [x] SubTask 16.1: 排查 `demo_dir` 解析路径，确认 `data/demo/model/tokenizer.py` 不存在
  - [x] SubTask 16.2: 直接委托 `verse_infra.verse_tokenizer`（已覆盖所有 kind）
  - [x] SubTask 16.3: 删除失效的 `model.tokenizer` 兜底
  - [x] SubTask 16.4: 更新文档

## 阶段六：文档同步与验证

- [x] Task 17: 清除 `UPDATE.MD` 中已修复的 1.1–1.13 条目
  - [x] SubTask 17.1: 删除 `UPDATE.MD` 中 1.1–1.13 的 13 个高危条目
  - [x] SubTask 17.2: 更新摘要表：高危 13 → 0，合计 73 → 60
  - [x] SubTask 17.3: 更新"按文件分布"表：各文件高危列清零
  - [x] SubTask 17.4: 更新"修复优先级建议"：P0/P1 中已完成项标记或删除
  - [x] SubTask 17.5: 保留中危/低危条目（本次不修复）

- [x] Task 18: 同步 `moe.py` → `mod.py` 文档引用
  - [x] SubTask 18.1: grep 全项目文档/注释中的 `moe.py`，逐一更新为 `mod.py`（补充 docs/part4_upgrade_report.md）
  - [x] SubTask 18.2: 验证 `__init__.py` 导出不变

- [x] Task 19: 运行核心测试套件验证
  - [x] SubTask 19.1: `python -m pytest tests/test_cometspark_v05.py tests/test_dual_model.py tests/test_spark_run.py tests/test_checkpoint_vn.py tests/test_mod_complete.py tests/test_nexrl.py -x -q` → 225 passed, 1 skipped
  - [x] SubTask 19.2: nexrl 相关测试通过（test_nexrl.py 全部通过，修复了 NexAction/RewardShaper/采样器返回值的测试适配）
  - [x] SubTask 19.3: MoD 相关测试通过（test_mod_complete.py 全部通过，熵正则梯度验证通过）
  - [x] SubTask 19.4: 额外运行 test_parallel_sparse_attn/test_cometspark_nex/test_recursion_fix/test_hybrid_stability/test_training_nex/test_p10_parallel_compress → 95 passed, 1 skipped

# Task Dependencies

- Task 2, 3 依赖 Task 1（重命名后才能改 `mod.py`）
- Task 9 的 SubTask 9.2 依赖 collector（Task 12/14 修改 collector），但可先在 trainer 侧实现 `truncated` 字段消费，collector 侧补齐
- Task 11（删除 RewardShaper）独立，可与 Task 9/10 并行
- Task 12（attention mask）依赖 `forward_policy` 支持 mask（可能需要改 agent.py / cometspark.py）
- Task 13（logprob）独立，可与 Task 12 并行
- Task 14 依赖 Task 13（`agent.act` 返回值需与 logprob 修复一致）
- Task 15, 16 独立，可与阶段一至四并行
- Task 17 依赖所有修复完成
- Task 18 依赖 Task 1
- Task 19 依赖所有任务完成
