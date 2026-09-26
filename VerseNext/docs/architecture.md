# 架构总览

## 包依赖关系

```
verse-cli ──> verse-rsi ───────> verse-trainer ──> verse-nn ──> verse-core
                 │                    │
                 └────> verse-tokenizer ─┘（依赖 verse-core + jinja2）
```

单向依赖，下层不感知上层。

## 各包职责

### verse-core（基础设施）

| 模块 | 职责 |
|---|---|
| `registry.py` | 全局组件注册中心。模型（`model/<name>`）、调度器（`scheduler/<name>`）、RSI 变异算子（`mutator/<name>`）均注册于此；装饰器 `register_model` 等简化注册。 |
| `config.py` | `BaseConfig`：dataclass 配置基类，提供 `to_dict/from_dict/merge/save/load`；嵌套配置自动还原。 |
| `events.py` | `EventBus` 同步事件总线；约定事件：`train/step_end`、`train/eval_end`、`rsi/generation_end` 等。 |
| `logging.py` | 统一日志（`verse.*` logger 命名空间）。 |

### verse-nn（模型架构）

- `attention.py`：可插拔注意力（注册中心 `attention/<name>` + `build_attention` 工厂）
  - `CausalSelfAttention`（sdpa）：融合 QKV 单 GEMM、GQA/MQA、SDPA 内核、静态/动态缓存双路径
  - `DSAttention`（dsa）：indexer 打分 + top-k 稀疏化，解码期 gather 选中 KV 免掩码计算
  - `KDAAttention`（kda）：门控 delta 规则线性注意力，固定大小递归状态（显存与序列长度无关）
- `kv_cache.py`：`StaticKVCache`（预分配、原地写、零拷贝视图）+ `RecurrentStateCache`（KDA）
- `rope.py`：`RotaryEmbedding`，预计算 cos/sin，支持位置偏移（缓存解码），超长序列动态扩表。
- `blocks.py`：`TransformerBlock`（pre-norm）+ `RMSNorm` + `SwiGLU`（gate/up 融合 GEMM，隐藏维对齐 256）。
- `transformer.py`：`VerseTransformer` + `VerseTransformerConfig`；权重绑定、梯度检查点、
  静态缓存生成（`logits_to_keep` 裁剪 vocab 投影、eos 提前停止）、`generate()`。
  模型经 `register_model("verse_transformer")` 注册，`build_model()` 是统一构建入口。
- `lora.py`：LoRA 适配器——`apply_lora`/`mark_only_lora_trainable`/`lora_state_dict`/`merge_lora`；B=0 初始化保证起点等价原模型。

### verse-tokenizer（分词器）

- `bpe.py`：byte-level BPE——训练（字节对合并）、编码/解码、`return_offsets` 字节偏移、特殊 token 与 `<|...|>` 自动注册、`save_pretrained`/`from_pretrained`（tokenizer.json + chat_template.jinja）。
- `chat_template.py`：jinja2 渲染 HF 约定对话模板；内置 ChatML 模板回退；`encode_chat` 生成 SFT 掩码 labels（哨兵串定位回答区间，与模板写法解耦）。
- `masking.py`：字节区间 -> labels（区间外 -100）。

### verse-trainer（训练性能）

- `trainer.py`：`Trainer`/`TrainerConfig`。AMP（bf16 优先，fp16 自动配 GradScaler）、梯度累积、梯度裁剪、TF32、`torch.compile` 开关、checkpoint 保存/恢复、`_stop_requested` 协作式停止；CPU profile（自动线程数 + flush denormal）；`TrainerConfig.lora` 启用 LoRA 微调；`train()` 支持任意 `next_batch()` 数据源；`save_adapter` 独立保存适配器。
- `optim.py`：fused AdamW（embedding/norm 免 weight decay 分组）+ `WarmupCosineScheduler`。
- `data.py`：`TokenBatchIterator` 随机窗口采样，targets 为右移一位。
- `sft.py`：`SFTDataset` 对话微调数据集（编码 + padding + 右移掩码 targets）。
- `inference.py`：`Generator` 推理引擎（chat/complete、int8 动态量化、线程优化、`torch.inference_mode`）。
- `callbacks.py`：`CheckpointCallback`、`EarlyStoppingCallback`（基于 EventBus）。

### verse-rsi（递归自进化）

核心抽象：**Individual = 模型架构配置 + 训练配置覆盖项**，适应度 = `eval_loss + speed_weight * (ref_tps/tps - 1)`（兼顾质量与速度）。

- `individual.py`：个体定义、lineage 祖先链、确定性 id（配置指纹）。
- `evaluator.py`：短训练预算评估适应度；首个个体的速度作为基准。
- `mutator.py`：`HyperparamMutator`（lr 乘性扰动、batch_size 跳变）、`ArchMutator`（层数/头数/d_model/KV 头数，保持架构合法性）、`CombinedMutator`。
- `population.py`：种群档案 + JSON 持久化。
- `evolve.py`：`Evolver.run()` 每代「评估 → 精英选择 → 变异 → 档案持久化」，下一代以最优为父代递归进化；`refine_best()` 对胜出者完整重训。

## 关键设计决策

1. **RSI 与训练解耦**：RSI 不侵入 Trainer，只通过 `TrainerConfig` 组合与 `EventBus` 事件交互；因此进化策略可整体替换。
2. **一切可注册**：新模型/新变异算子加一个装饰器即可被 CLI 与进化循环使用。
3. **配置即数据**：全部配置是 dataclass，可序列化——个体 id 即配置指纹，种群档案即实验记录。
