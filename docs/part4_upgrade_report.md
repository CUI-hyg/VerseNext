# Part4 升级报告：CometSpark-V0.2 与 VerseNex 原生架构

> 本报告记录 Part4 阶段的所有升级内容，包括 VerseNex 原生架构、CometSpark-V0.2（0.5B 参数）、训练体系、数据脚本、压缩技术、并行训练增强等。

## 概览

Part4 在 Part3K2 基础上完成 **11 大升级**，正式推出：

- **CometSpark-V0.2**：32 层 VerseNex 原生架构，约 0.5B 参数
- **VerseNex 原生架构**：TriSparseAttention + MoD 多稠密分区，不依赖 Transformer 或 SSM
- **训练体系**：4 个训练器（VerseNexTrainer / LoRATrainer / SFTTrainer / DPOTrainer）

测试覆盖：**462 passed + 1 skipped**（全项目回归测试），新增 60+ 测试用例。

---

## 1. VerseNex 原生架构

### 1.1 TriSparseAttention（三路并行稀疏注意力）

**文件**：[`packages/verse_nex/verse_nex/tri_sparse_attn.py`](../packages/verse_nex/verse_nex/tri_sparse_attn.py)（624 行）

将注意力拆分为三路并行计算后用 sigmoid gate 融合：

| 路径 | 机制 | 复杂度 | 适用场景 |
|---|---|---|---|
| **SWA** | Sliding Window Attention，chunk-wise 实现，不构造 T² 矩阵 | O(T·W) | 局部依赖 |
| **Global** | 可学习 sink token（默认 64 个） | O(T·N_global) | 长程依赖 |
| **ALiBi** | 基于位置的线性偏置，T ≤ 1024 直接构造，T > 1024 降级为 SWA-only | O(T²) 或降级 | 位置感知 |

**Gate 融合**：`out = σ(g_swa)·swa + σ(g_global)·global + σ(g_alibi)·alibi`

支持 GQA（`n_kv_head < n_head`）。

### 1.2 MoDLayer（Mixture of Dense Parts）

**文件**：[`packages/verse_nex/verse_nex/moe.py`](../packages/verse_nex/verse_nex/moe.py)（621 行）

灵感来源于人大脑的功能分区：

- **5 DensePart**：`general`（通用）/ `language`（语言）/ `math`（数理）/ `biochem`（生化）/ `code`（代码）
- **8 Experts per DensePart**，每个 Expert 是独立的 SwiGLU MLP
- **top-3 激活**：每个 token 仅激活 top-3 Expert（hard routing）
- **双层门控**：
  - `part_router`（soft routing）：所有 DensePart 都参与计算
  - `expert_router`（hard routing）：top-k 选择
- **Switch Transformer 风格 aux loss**：负载均衡损失，避免 Expert 坍缩

数值梯度检查误差：**5.11e-08**（远低于 1e-6 阈值）。

### 1.3 CometSparkNexLM（顶层 LM）

**文件**：[`packages/verse_nex/verse_nex/cometspark.py`](../packages/verse_nex/verse_nex/cometspark.py)（~920 行）

将 TriSparseAttention 与 MoDLayer 组合为完整 LM：

- **layer_pattern 驱动**：每层类型显式指定（`"trisparse"` / `"mod"`）
- **Pre-Norm + 残差**：`x = x + attn(norm1(x)); x = x + ffn(norm2(x))`
- **残差缩放**：`1/sqrt(2*n_layer)`
- **三种前向模式**：
  - `forward(idx)` → logits：标准前向
  - `forward_with_aux(idx)` → `(logits, aux_loss)`：训练用
  - `forward_recurrent(input_ids, states)` → `(logits, new_states)`：流式生成
- **generate**：greedy + recurrent（temperature=1.0）与采样（temperature/top_k）两条路径
- **持久化**：`save` / `load` / `from_pretrained` / `save_pretrained`

测试：21 个测试全部通过。

---

## 2. CometSpark-V0.2

### 2.1 参数预算

| 配置 | 值 |
|---|---|
| n_layer | 32 |
| d_model | 384 |
| n_head | 8 |
| n_kv_head | 4（GQA） |
| layer_pattern | 8 MoD + 24 trisparse（mod_every=4） |
| DensePart | 5（通用/语言/数理/生化/代码） |
| Experts per DensePart | 8 |
| top_k | 3 |
| window_size | 512 |
| num_global_tokens | 64 |
| **总参数量** | **≈ 537,591,264（0.538B）** |

### 2.2 CometSparkLM 三架构统一

`CometSparkLM` 现支持三种架构，对外接口完全一致：

```python
config = CometSparkConfig(arch="verse_nex", ...)
model = CometSparkLM(config)
# 与 arch="transformer" / arch="hybrid" 使用相同的接口：
# model.forward(idx) / model.generate(idx, ...) / model.save(path) / ...
```

### 2.3 工厂函数

- `CometSparkV02Small(vocab_size=256, seq_len=128)`：~0.5M 参数，沙箱验证用
- `CometSparkV02(vocab_size=151936)`：~0.5B 参数，正式训练用

### 2.4 配置文件

- [`config_verse_nex.yml`](../data/demo/config/config_verse_nex.yml)：0.5B 参数预训练配置
- [`config_verse_nex_small.yml`](../data/demo/config/config_verse_nex_small.yml)：沙箱验证用

---

## 3. VerseTokenizer

**文件**：[`packages/verse_tokenizer/verse_tokenizer/verse.py`](../packages/verse_tokenizer/verse_tokenizer/verse.py)（865 行）

原 QwenTokenizer 改名为 VerseTokenizer，针对 Qwen 系列做了 9 项优化：

1. 高效 BPE merge（优先级队列，O(N·V)）
2. 特殊 token 管理（`<|endoftext|>` / `<|im_start|>` / `<|im_end|>`）
3. UTF-8 边界修复（防 U+FFFD 乱码）
4. chat template（`apply_chat_template` / `apply_prompt_template`）
5. NFKC 归一化
6. GPT-4 风格预分词（中文整字独立成块）
7. vocab 自适应
8. `add_special_tokens` 编码开关
9. 持久化兼容（可加载 Qwen 系列 `tokenizer.json`）

测试：22 个测试全部通过。旧版 `qwen.py` 已删除。

---

## 4. 训练体系

**文件**：[`packages/verse_torch/verse_torch/training_nex.py`](../packages/verse_torch/verse_torch/training_nex.py)

### 4.1 VerseNexTrainer（aux_loss-aware）

```python
loss = cross_entropy(logits, y) + aux_loss_weight * aux
```

- 自动检测 `model.forward_with_aux` 或 `model.net.forward_with_aux`
- `aux_loss_weight` 默认从 `model.config.aux_loss_weight` 读取，可在 cfg 中显式覆盖
- 与 `Trainer` 完全兼容（相同的 cfg 字段、EarlyStopping、CheckpointManager、plot_loss_curve）
- 额外保存 `aux_losses.txt` 与 `aux_losses` 字段到 `loss_history.json`

### 4.2 LoRATrainer

- `__init__` 时自动调用 `lora_only(model, r, alpha)` 包装所有 `Linear` 为 `LoRALinear`
- 自动冻结 base 参数，仅训练 A/B 矩阵
- `optimizer=None` 时自动基于 LoRA 参数构建 AdamW
- `merge_lora()` 方法把 ΔW 合并回 base，替换回标准 `Linear`
- `merge_after=True` 时 fit 结束后自动 merge

### 4.3 SFTTrainer

- 支持 chat 数据格式（`{"messages": [{"role","content"}, ...]}`）
- 仅 assistant 回复 token 参与 loss（user/system token 被 `ignore_index=-100` 屏蔽）
- 渲染格式：`<|system|>...<|user|>...<|assistant|>...<|endoftext|>`
- 兼容 `forward_with_aux`

### 4.4 DPOTrainer

- 偏好对数据格式：`{"prompt":"...","chosen":"...","rejected":"..."}`
- DPO loss = `-mean(log σ(β·((π_chosen - π_rejected) - (ref_chosen - ref_rejected))))`
- reference model 自动冻结（`ref_model=None` 时深拷贝 policy）
- 自动计算 accuracy = mean(π_chosen > π_rejected)
- 保存 `dpo_history.json` + `dpo_curve.png`（含 loss 与 accuracy 双曲线）
- 数值稳定性：极端值（±100）下无 NaN/Inf

### 4.5 数据集

- `SFTDataset`：jsonl 加载 + chat template 渲染
- `DPODataset`：偏好对加载 + prompt/chosen/rejected 拼接

测试：19 个测试全部通过。

---

## 5. 并行训练增强

**文件**：[`packages/verse_torch/verse_torch/training.py`](../packages/verse_torch/verse_torch/training.py)

`ParallelTrainer` 自动检测 `forward_with_aux`：

- `_train_chunk` 内部使用 `VerseNexTrainer` 而非 `Trainer`，正确处理 MoD aux_loss
- `_eval_full_val` 调用 `forward_with_aux` 取 logits，避免 (logits, aux) tuple 破坏 loss_fn
- `chunk_cfg` 自动写入 `aux_loss_weight`，与 `VerseNexTrainer` 配置一致

测试：6 个测试全部通过。

---

## 6. MoD Expert 压缩

**文件**：[`packages/verse_torch/verse_torch/compress.py`](../packages/verse_torch/verse_torch/compress.py)

新增 `compress_mod_experts` 函数：

```python
def compress_mod_experts(model, keep_ratio=0.5, min_experts_per_part=1, return_stats=False):
    """MoD Expert 结构化剪枝。
    
    1. 遍历所有 MoDLayer
    2. 对每个 DensePart 内的 Experts 按参数 L2 范数排序
    3. 保留范数最高的 max(min_experts_per_part, int(num_experts * keep_ratio)) 个
    4. 同步修改 expert_router 权重矩阵行、num_routes、top_k、num_experts
    """
```

适用场景：MoD 模型部署前的体积压缩，特别适合 CPU 推理。

---

## 7. 数据下载与处理脚本

**目录**：[`data/demo/scripts/`](../data/demo/scripts/)

| 脚本 | 功能 |
|---|---|
| `download_datasets.py` | 7 个数据源（BelleGroup/Firefly/code-alpaca 等），3 种下载方式回退（hf_hub → datasets → urllib） |
| `process_datasets.py` | 统一 schema + chat template + 过滤去重 + 统计输出 |
| `build_tokenizer.py` | Qwen3-32B tokenizer 下载与验证 |
| `README.md` | 数据源列表、运行步骤、配置说明、常见问题 |

输出格式：`{"text":..., "source":..., "lang":..., "task_type":...}` 或 `{"messages":[...]}`。

---

## 8. run.py 与 trainer.py 集成

- `run.py --arch` choices 增加 `verse_nex`
- `trainer.py` 在 `arch=="verse_nex"` 分支调用 `VerseNexTrainer`

端到端验证：verse_nex_small 配置训练 loss 5.54 → 3.08 持续下降。

---

## 9. check-loop 审计

### 9.1 测试统计

- **全项目回归测试**：462 passed + 1 skipped
- **新增测试**：
  - `test_cometspark_nex.py`：21 个（CometSparkNexLM）
  - `test_cometspark_v02_integration.py`：13 个（CometSparkLM verse_nex 集成）
  - `test_training_nex.py`：19 个（训练体系）
  - `test_p10_parallel_compress.py`：6 个（并行训练 + 压缩）

### 9.2 验证项

- 16 个关键文件语法检查 OK
- 端到端 verse_nex_small 训练：loss 持续下降
- LoRA + merge_lora：41 个 Linear → 41 个 LoRALinear → 41 个 Linear（merge 后 forward 正常）
- DPO 数值稳定性：极端值（±100）下无 NaN/Inf
- 0.5B 模型构建：537,591,264 参数（0.538B，符合预算）
- 旧版 `qwen.py` 已删除，无残留 QwenTokenizer 引用

---

## 10. 文件清单

### 新建文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `packages/verse_nex/verse_nex/tri_sparse_attn.py` | 624 | TriSparseAttention |
| `packages/verse_nex/verse_nex/moe.py` | 621 | MoDLayer |
| `packages/verse_nex/verse_nex/cometspark.py` | ~920 | CometSparkNexLM |
| `packages/verse_torch/verse_torch/training_nex.py` | ~900 | 4 个训练器 + 2 个数据集 |
| `packages/verse_tokenizer/verse_tokenizer/verse.py` | 865 | VerseTokenizer |
| `data/demo/config/config_verse_nex.yml` | - | 0.5B 参数配置 |
| `data/demo/config/config_verse_nex_small.yml` | - | 沙箱验证配置 |
| `data/demo/scripts/download_datasets.py` | ~669 | 数据下载 |
| `data/demo/scripts/process_datasets.py` | ~780 | 数据处理 |
| `data/demo/scripts/build_tokenizer.py` | ~443 | tokenizer 构建 |
| `data/demo/scripts/README.md` | ~340 | 脚本文档 |
| `tests/test_cometspark_nex.py` | - | 21 个测试 |
| `tests/test_cometspark_v02_integration.py` | - | 13 个测试 |
| `tests/test_training_nex.py` | - | 19 个测试 |
| `tests/test_p10_parallel_compress.py` | - | 6 个测试 |
| `docs/part4_upgrade_report.md` | - | 本报告 |

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `packages/verse_nex/verse_nex/__init__.py` | 导出 VerseNexBlock / CometSparkNexLM / CometSparkV02，版本 0.3.0 |
| `packages/verse_torch/verse_torch/__init__.py` | 导出 training_nex 与 compress_mod_experts |
| `packages/verse_torch/verse_torch/training.py` | ParallelTrainer 支持 aux_loss |
| `packages/verse_torch/verse_torch/compress.py` | 新增 compress_mod_experts |
| `data/demo/model/config.py` | CometSparkConfig 新增 verse_nex 字段 |
| `data/demo/model/model.py` | CometSparkLM 支持 arch=verse_nex + V02 工厂 |
| `data/demo/train/trainer.py` | arch=verse_nex 分支调用 VerseNexTrainer |
| `data/demo/run.py` | --arch 增加 verse_nex |

### 删除文件

| 文件 | 原因 |
|---|---|
| `packages/verse_tokenizer/verse_tokenizer/qwen.py` | 被 verse.py 替代 |

---

## 11. 用户下一步

1. **下载数据**：`python data/demo/scripts/download_datasets.py`
2. **处理数据**：`python data/demo/scripts/process_datasets.py --format both`
3. **构建 tokenizer**：`python data/demo/scripts/build_tokenizer.py`
4. **训练**：`python data/demo/run.py --config config/config_verse_nex.yml`

训练部分由用户自行执行，本报告仅提供材料。
