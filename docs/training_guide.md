# Verse 训练指南

> 本指南介绍如何用 Verse 框架从零训练一个语言模型，覆盖**数据准备 → tokenizer 构建 → 模型配置 → 训练 → 评估与生成 → 压缩 → 推理部署**全流程。所有步骤均在纯 Python / 纯 CPU 环境下完成，无需 PyTorch / TensorFlow / JAX。

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [Tokenizer 构建](#3-tokenizer-构建)
4. [模型配置与构建](#4-模型配置与构建)
5. [训练](#5-训练)
6. [评估与生成](#6-评估与生成)
7. [模型压缩](#7-模型压缩)
8. [推理部署](#8-推理部署)
9. [完整示例：CometSpark](#9-完整示例cometspark)

---

## 1. 环境准备

### 1.1 安装

需要 Python ≥ 3.10、NumPy ≥ 1.26。本仓库采用 uv/pip workspace 多包布局，按需安装：

```bash
# 克隆仓库
git clone <repo-url> verse && cd verse

# 方式一：pip 可编辑安装（推荐初学者）
pip install -e packages/verse_torch \
            -e packages/verse_nex \
            -e packages/verse_tokenizer \
            -e packages/verse_inference \
            -e packages/verse_compat \
            -e packages/verse_awm

# 方式二：uv workspace 一次性安装全部成员
uv sync

# 可选运行时依赖（按需）
pip install "numba>=0.60"        # CPU GEMM 加速
pip install "safetensors>=0.4"   # 加载 .safetensors 权重
pip install "fastapi>=0.110"     # OpenAI 兼容 HTTP server
```

### 1.2 验证安装

```python
from verse_torch import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()        # 1 + 4 + 9 = 14
y.backward()
print(y)                 # 14.0
print(x.grad)            # [2. 4. 6.]  与 PyTorch 一致
```

数值与 PyTorch 一致到 1e-6（已通过 109 项单元测试 + 有限差分梯度检查）。

---

## 2. 数据准备

### 2.1 JSONL 格式

Verse 训练流程使用 JSONL 文件，每行一个 JSON 对象，至少包含 `text` 字段：

```json
{"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"}
{"text": "问：你好 答：你好，很高兴见到你。"}
{"text": "1,2,3,4,5,6,7,8,9,10"}
{"text": "Hello, world!"}
```

- 文件编码：UTF-8
- 换行符：`\n`
- 其他字段会被忽略（如 `{"text": "...", "meta": {...}}` 仍可加载）

### 2.2 数据集划分

建议训练集 ≥ 200 条、验证集 ≥ 50 条、单条文本 ≥ 10 字符。文件命名约定：

```
data/
├── train.jsonl   # 训练集
└── val.jsonl     # 验证集
```

参考示例：[`data/demo/data/train.jsonl`](../data/demo/data/train.jsonl)（200 行，覆盖唐诗 / 问答 / 数字序列 / 英文短句 4 类），数据格式说明见 [`data/demo/data/README.md`](../data/demo/data/README.md)。

### 2.3 加载与切分

`verse_torch` 没有内置 `Dataset` 抽象，CometSpark demo 中提供了 `TextDataset` 与 `BatchLoader` 作为参考实现（见 [`data/demo/src/data_loader.py`](../data/demo/src/data_loader.py)）：

```python
from src.data_loader import TextDataset, BatchLoader, collate_fn

train_ds = TextDataset(tokenizer, "data/train.jsonl", seq_len=64)
val_ds = TextDataset(tokenizer, "data/val.jsonl", seq_len=64)

train_loader = BatchLoader(
    train_ds, batch_size=8, shuffle=True,
    collate_fn=collate_fn, drop_last=False, seed=42,
)
val_loader = BatchLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
```

每个 batch 返回 `(x, y)`，其中 `x.shape = (B, T)`、`y.shape = (B, T)` 是 `x` 向左移一位的目标序列（next-token prediction）。

---

## 3. Tokenizer 构建

Verse 提供 3 种 tokenizer，由 [`verse_tokenizer`](../packages/verse_tokenizer/README.md) 包统一管理。

### 3.1 ByteTokenizer（推荐入门）

固定词表 259（256 字节 + BOS/EOS/PAD/UNK），无需训练：

```python
from verse_tokenizer import ByteTokenizer

tok = ByteTokenizer()
print(len(tok))  # 259
ids = tok.encode("你好，世界")
print(ids)       # [230, 189, 189, 229, 165, 189, ...]
text = tok.decode(ids)
```

优点：UTF-8 任意文本都可编码，零依赖；缺点：1 个汉字 ≈ 3 token，序列偏长。

### 3.2 BPETokenizer.train（推荐进阶）

需要语料训练 merges：

```python
from verse_tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(corpus_iter, vocab_size=2000)   # corpus_iter 是字符串迭代器
tok.save("tokenizer.json")
# 之后可 tok.add_special_tokens(["<bos>", "<eos>", "<pad>", "<unk>"])
```

加载：

```python
from verse_tokenizer import BPETokenizer
tok = BPETokenizer.load("tokenizer.json")
```

### 3.3 HF tokenizer（需 transformers）

若已有 HuggingFace `tokenizer.json`，可直接复用：

```python
from verse_tokenizer import BPETokenizer
tok = BPETokenizer.load("path/to/hf_tokenizer.json")
```

### 3.4 工厂函数

```python
from verse_tokenizer import load_tokenizer
tok = load_tokenizer("tokenizer.json", kind="byte")  # kind: byte / bpe / hf
```

CometSpark 的 `model/tokenizer.py` 中的 `build_tokenizer` 完成构建并保存到 `checkpoints/tokenizer.json`：

```python
from model.tokenizer import build_tokenizer, load_tokenizer
tok = build_tokenizer(kind="byte", save_path="checkpoints/tokenizer.json")
# 后续：tok = load_tokenizer("checkpoints/tokenizer.json", kind="byte")
```

---

## 4. 模型配置与构建

### 4.1 两种架构

CometSpark 支持 `arch="transformer"` 与 `arch="hybrid"` 两种：

| 架构 | 主干 | 适用场景 |
|---|---|---|
| `transformer` | `GQASelfAttention` + `SwiGLUMLP`（GQA + RoPE） | PoC 与稳定性优先 |
| `hybrid` | Mamba-2 + Sparse Attention 混合块（线性复杂度） | 长上下文 / 显存敏感（待 verse_nex Mamba2 数值修复后推荐） |

### 4.2 CometSparkConfig 字段

完整字段定义见 [`data/demo/model/config.py`](../data/demo/model/config.py)，由 `config.yml` 解析得到：

```yaml
model:
  arch: transformer        # transformer / hybrid
  vocab_size: 256          # 由 tokenizer 自动覆盖
  n_layer: 2               # Transformer block 数
  n_head: 4                # 注意力头数
  n_embd: 64               # 模型维度
  seq_len: 64              # 上下文长度
  dropout: 0.1
  n_kv_head: 2             # GQA 的 KV 头数（n_head // n_kv_head = repeat 因子）
  ssm_kind: mamba2         # hybrid 下的 SSM 种类
  sparse_ratio: 0.5        # hybrid block 中 sparse attention 比例
  tie_weights: true        # 是否共享 embedding 与 head 权重
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `arch` | str | `transformer` | 架构选择 |
| `vocab_size` | int | `256` | 词表大小（实际由 tokenizer 自动覆盖） |
| `n_layer` | int | `2` | block 数量 |
| `n_head` | int | `4` | 注意力头数 |
| `n_embd` | int | `64` | 模型维度 |
| `seq_len` | int | `64` | 上下文长度 |
| `dropout` | float | `0.1` | dropout 概率 |
| `n_kv_head` | int | `2` | GQA KV 头数 |
| `ssm_kind` | str | `mamba2` | hybrid 下的 SSM 种类 |
| `sparse_ratio` | float | `0.5` | hybrid block 中 sparse attention 比例 |
| `tie_weights` | bool | `true` | embedding / head 权重共享 |

### 4.3 构建 CometSparkLM

```python
from model.config import CometSparkConfig
from model.model import CometSparkLM

config = CometSparkConfig(
    arch="transformer", vocab_size=259,
    n_layer=2, n_head=4, n_embd=64, seq_len=64,
    dropout=0.1, n_kv_head=2, tie_weights=True,
)
model = CometSparkLM(config)

# forward: (B, T) int → (B, T, vocab_size) logits
logits = model(input_ids)
```

`CometSparkLM` 还提供：

- `model.save(path)`：保存 `config + state_dict` 到 pickle 文件
- `CometSparkLM.from_pretrained(path)`：从 pickle 文件重建模型
- `model.generate(idx, max_new_tokens, temperature, top_k)`：便捷生成接口

### 4.4 纯 verse_torch 构建（不用 CometSpark）

若不依赖 CometSpark 封装，可直接用 `verse_torch.nn.TransformerLM`：

```python
from verse_torch import nn

model = nn.TransformerLM(
    vocab_size=259, n_layer=2, n_head=4, n_embd=64,
    seq_len=64, n_kv_head=2, dropout=0.1, tie_weights=True,
)
```

---

## 5. 训练

### 5.1 Trainer API

`verse_torch.training.Trainer` 是端到端训练循环的入口（详见 [`packages/verse_torch/verse_torch/training.py`](../packages/verse_torch/verse_torch/training.py)）：

```python
from verse_torch.training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,        # 可选
    cfg={
        "max_steps": 200,
        "eval_interval": 20,
        "patience": 5,
        "save_dir": "checkpoints",
        "grad_accum": 1,
        "log_interval": 20,
        "loss_rate_window": 50,
    },
)
train_losses, val_losses = trainer.fit()
```

`cfg` 字段说明：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_steps` | int | `100` | 最大训练步数 |
| `eval_interval` | int | `10` | 每隔多少步在 val_loader 上评估 + checkpoint |
| `patience` | int | `10` | EarlyStopping 容忍的未改善轮数 |
| `save_dir` | str | `./checkpoints` | 检查点保存目录 |
| `grad_accum` | int | `1` | 梯度累积步数（每 N 次反向执行一次 optimizer.step） |
| `log_interval` | int | `10` | 日志打印间隔 |
| `loss_rate_window` | int | `50` | loss 下降率滑动窗口大小 |

### 5.2 完整训练流程

参考 [`data/demo/train/trainer.py`](../data/demo/train/trainer.py)：

```python
from verse_torch.optim import AdamW, LambdaLR, warmup_cosine_lr
from verse_torch.training import Trainer

# 1. 优化器 + 学习率调度
lr = 3e-3
weight_decay = 0.01
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

max_steps = 200
warmup = 20
scheduler = LambdaLR(
    optimizer, warmup_cosine_lr(warmup_steps=warmup, total_steps=max_steps)
)
# warmup_cosine_lr: step < warmup 线性升；warmup..total 之间余弦衰减到 0

# 2. Trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    cfg={
        "max_steps": max_steps,
        "eval_interval": 20,
        "patience": 5,
        "save_dir": "checkpoints",
        "grad_accum": 1,
        "log_interval": 20,
    },
)

train_losses, val_losses = trainer.fit()
```

### 5.3 训练循环内部组件

`Trainer.fit()` 内部依次调用以下组件（均可在 `verse_torch.training` 中单独导入）：

- **损失函数**：`cross_entropy_loss(logits, targets, ignore_index=-100)`
  - 支持 `(B, T, V)` / `(N, V)` 形状与 `ignore_index` 屏蔽
  - 内部用 `log_softmax + NLL`，对 `ignore_index` 位置不计入 loss 与梯度
- **优化器**：`AdamW`（带 weight_decay）；学习率调度 `LambdaLR + warmup_cosine_lr`
- **梯度累积**：`GradientAccumulator(micro_batch, effective_batch)`
  - `accum_steps = effective_batch // micro_batch`
  - 每 `accum_steps` 次 `step()` 后 `should_step()` 返回 True 并自动重置
  - 在 Trainer 中以"每 N 次反向 step 一次"的步数累积语义使用
- **早停**：`EarlyStopping(patience, min_delta=0.0)`
  - 连续 `patience` 次 val_loss 未显著下降（> `min_delta`）时触发
- **检查点**：`CheckpointManager(save_dir)`
  - `save_best(state)`：保存到 `best.pt`（仅在 val_loss 创新低时调用）
  - `save_last(state)`：保存到 `last.pt`（每个 eval_interval 调用）
  - `load_best()` / `load_last()`：返回 state dict（含 `step / model_state_dict / val_loss / train_loss`）
  - 用 pickle 序列化，Tensor 内部转为 `{"__tensor__": True, "data": ndarray, ...}` 形式
- **Loss 下降率诊断**：`compute_loss_rate(loss_window, window=50, min_delta=1e-4)`
  - 返回 `(avg_first_half - avg_second_half) / avg_first_half`
- **Loss 曲线绘制**：`plot_loss_curve(train_losses, val_losses, save_path, eval_interval)`
  - 优先用 matplotlib 输出 PNG
  - matplotlib 不可用时降级为 ASCII 文本图（80×20 字符画布，T=train / V=val）

训练结束后，`save_dir` 目录下会自动生成：

```
checkpoints/
├── best.pt              # 最佳验证 loss 模型
├── last.pt              # 最后一次评估的模型
├── loss_history.json    # 逐 step 训练 / 验证 loss
└── loss_curve.png       # 或 loss_curve.txt（ASCII fallback）
```

---

## 6. 评估与生成

### 6.1 加载 best.pt 生成文本

参考 [`data/demo/train/evaluate.py`](../data/demo/train/evaluate.py)：

```python
import pickle
from model.config import CometSparkConfig
from model.model import CometSparkLM
from model.tokenizer import load_tokenizer
from verse_torch import no_grad

# 1. 加载 tokenizer
tok = load_tokenizer("checkpoints/tokenizer.json", kind="byte")

# 2. 加载模型（优先 cometspark.pt，其次用 config 重建 + best.pt）
import os
if os.path.exists("checkpoints/cometspark.pt"):
    model = CometSparkLM.from_pretrained("checkpoints/cometspark.pt")
else:
    config = CometSparkConfig(arch="transformer", vocab_size=len(tok), ...)
    model = CometSparkLM(config)
    with open("checkpoints/best.pt", "rb") as f:
        payload = pickle.load(f)
    model.load_state_dict(payload["model_state_dict"], strict=False)

# 3. 生成
with no_grad():
    model.eval()
    prompt = "床前明月光，"
    ids = list(tok.encode(prompt, add_special_tokens=False))
    import numpy as np
    idx = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    generated = model.generate(idx, max_new_tokens=32, temperature=1.0, top_k=None)
    gen_ids = generated.data.reshape(-1).tolist() if hasattr(generated, "data") else list(generated)
    print(tok.decode(gen_ids))
```

### 6.2 采样策略

`CometSparkLM.generate` / `StreamingGenerator` 支持以下策略：

| 策略 | 参数 | 说明 |
|---|---|---|
| Greedy | `temperature=0` 或 `top_k=None` | 每步取 argmax |
| Temperature | `temperature > 0` | logits / T 后采样，T 越大越随机 |
| Top-k | `top_k=k` | 仅在 logits 最大的 k 个 token 上采样 |

5 条预设 prompt（CometSpark demo 默认）：

```python
prompts = ["床前明月光，", "白日依山尽，", "你好，", "1+1=", "春风"]
```

---

## 7. 模型压缩

### 7.1 compress_pipeline：端到端压缩

`verse_torch.compress.compress_pipeline` 提供"prune → quantize → (可选) lora_wrap"流水线：

```python
from verse_torch.compress import compress_pipeline

report = compress_pipeline(
    model,
    target_ratio=0.1,     # 目标压缩比（1/10）
    sparsity=0.3,         # 剪枝稀疏度
    qtype="int4",          # 量化类型：int4 / int8 / ternary
    use_lora=False,       # 是否挂 LoRA 适配器（QLoRA 风格）
    eval_fn=lambda m: float(eval_loss(m)),  # 可选：用于报告 loss 差异
)
print(report)
# {
#   'original_params': 1000000,
#   'compressed_params': 125000.0,
#   'compressed_bits': 4000000,
#   'original_bits': 32000000,
#   'compression_ratio': 8.0,
#   'original_loss': 2.16,
#   'compressed_loss': 2.27,
#   'loss_diff_pct': 5.09,
#   'steps': [{'step': 'prune', ...}, {'step': 'quantize', ...}]
# }
```

PoC 验证基线（详见 [`docs/benchmarks/compression_poc.md`](benchmarks/compression_poc.md)）：

- 在 1M 参数 TransformerLM 上验证
- 默认配置（sparsity=0.3 + int4）压缩比 ≥ 8×
- 切到 `qtype="ternary"`（BitNet b1.58 风格，2 bit/value）可达 10× 以上
- loss 差异 ≤ 5%

### 7.2 单技术函数

每一步也可单独使用：

```python
from verse_torch.compress import (
    prune_only, quantize_only, lora_only, ternary_only, distill_only,
)

# 1. 仅剪枝（mask + 冻结策略，原结构不变）
model, report = prune_only(model, sparsity=0.3)

# 2. 仅量化（把所有 nn.Linear 替换为 QLinear）
quantize_only(model, dtype="int4")    # int4 / int8 / ternary

# 3. 仅 ternary 量化（BitNet b1.58 风格）
ternary_only(model)

# 4. 仅 LoRA 包装（frozen base + A@B 增量）
lora_only(model, r=8, alpha=16.0)

# 5. 仅蒸馏（teacher → student）
distill_only(teacher, student, train_loader, max_steps=100, T=2.0, alpha=0.5)
```

### 7.3 压缩类组件

| 类 | 作用 |
|---|---|
| `OutlierSafePruner` | 按 head/channel 维度 `|weight|_mean` 结构化剪枝；跳过 `tok_emb` / `head` 避免破坏词表语义 |
| `LoRALinear` | `frozen base + A @ B * (alpha / r)`；`merge()` 可合并为 `nn.Linear`（仅 base 是 Linear 时支持） |
| `QLinear` | 把 `QuantizedLinear` 包装为 `nn.Module`，便于嵌入模型树 |
| `KnowledgeDistiller` | `Loss = α·T²·KL(teacher/T ‖ student/T) + (1-α)·CE(student, hard)` |

### 7.4 QLoRA 风格：量化 + LoRA 微调

```python
# 1. 量化基座
quantize_only(model, dtype="int4")

# 2. 包装 LoRA 适配器（base 冻结，仅 A/B 可训练）
lora_only(model, r=8, alpha=16.0)

# 3. 后续训练只更新 A/B（节省显存 / 加速）
```

---

## 8. 推理部署

### 8.1 verse_inference 包

[`verse_inference`](../packages/verse_inference/README.md) 提供：

| 类 | 作用 |
|---|---|
| `ModelLoader` | 从 HF repo / 本地路径加载 LM；支持 `arch=mamba2 / rwkv7 / hybrid / cometspark` |
| `StateCache` | Mamba / RWKV 递归状态缓存（O(1) 内存推理） |
| `Sampler` / `GreedySampler` | temperature / top_k / top_p 采样器 |
| `StreamingGenerator` | 流式生成器：prefill → decode → yield 逐 token |

### 8.2 加载 cometspark 模型

```python
from verse_inference import ModelLoader, StreamingGenerator, Sampler

# ModelLoader 会自动从 pickle 加载 config + state_dict 重建 CometSparkLM
loader = ModelLoader(arch="cometspark")
model = loader.load("checkpoints/cometspark.pt")
```

`cometspark` arch 分支由 Stage 7 新增，详见 [`packages/verse_inference/verse_inference/model_loader.py`](../packages/verse_inference/verse_inference/model_loader.py)。

### 8.3 CPU 流式生成

```python
from verse_inference import StreamingGenerator, Sampler

sampler = Sampler(temperature=0.8, top_k=40)
gen = StreamingGenerator(model, tokenizer=tok, sampler=sampler)

prompt_ids = list(tok.encode("床前明月光，"))
for token_id in gen.generate(prompt_ids, max_new_tokens=100):
    print(tok.decode([token_id]), end="", flush=True)
```

`StreamingGenerator` 通过 `model.forward_recurrent` 逐 token 处理，每个 token 仅维护固定大小的 SSM 状态（O(1) 内存），适合 Mamba-2 / RWKV-7 这类线性复杂度架构。

### 8.4 性能基线

参考 [`examples/README_cpu_inference_demo.md`](../examples/README_cpu_inference_demo.md)：

- 4 核 CPU 上构建一个 0.6M 参数的 Mamba-2 LM
- 100 token 流式生成，吞吐量约 **715 tokens/s**
- 峰值 RSS 约 **44.5 MB**

约束满足：5 分钟内完成、RSS ≤ 8 GB、参数量 < 50M。

### 8.5 OpenAI 兼容 HTTP server（可选）

若安装了 `fastapi`，可启动 OpenAI 兼容的 `/v1/chat/completions` 接口：

```python
from verse_inference.server import serve
serve(model, tokenizer=tok, host="0.0.0.0", port=8000)
```

详见 [`packages/verse_inference/verse_inference/server.py`](../packages/verse_inference/verse_inference/server.py)。

---

## 9. 完整示例：CometSpark

CometSpark-v0.1 是基于 VerseNext 的端到端 LM 训练仓库，覆盖本指南全部能力。详见 [`data/demo/README.md`](../data/demo/README.md)。

### 9.1 一键运行

```bash
cd /workspace/data/demo
python run.py
```

可选参数：

```bash
python run.py --skip-build       # 跳过 tokenizer 构建
python run.py --skip-train       # 跳过训练，直接加载 best.pt 评估
python run.py --skip-eval        # 跳过评估
python run.py --config path/to/config.yml
```

`run.py` 串联四步：`build_tokenizer → train → evaluate → visualize`。

### 9.2 实测性能

在 3 核 CPU / 5GB 内存沙箱中（默认配置 `n_layer=2, n_embd=64, seq_len=64`）：

| 指标 | 实测 |
|---|---|
| wall-clock | ~9 秒 |
| 初始 train loss | 5.61 |
| 最终 train loss | 2.16 |
| 最佳 val loss | 2.28 |
| 生成样本数 | 5 条 |
| checkpoints | `best.pt` / `last.pt` / `cometspark.pt` / `loss_history.json` / `loss_curve.txt` |

### 9.3 目录结构

```
data/demo/
├── run.py                # 一键入口
├── config/config.yml     # 模型 / 训练 / tokenizer / data / checkpoint 配置
├── model/                # CometSparkConfig + CometSparkLM + tokenizer 工厂
├── src/                  # 数据加载 + utils
├── train/                # trainer.py + evaluate.py + visualize.py
├── data/                 # train.jsonl / val.jsonl / README.md
└── checkpoints/          # 训练产物（自动生成）
```

### 9.4 依赖

仅依赖 `verse_torch` / `verse_nex` / `verse_tokenizer` / `verse_inference`（运行时**不需要** PyTorch / TensorFlow / JAX / transformers）。

---

## 相关文档

- [VerseTorch README](../packages/verse_torch/README.md) —— Tensor / nn / autograd 基础
- [VerseNex README](../packages/verse_nex/README.md) —— Mamba-2 / RWKV-7 / Hybrid 架构
- [VerseAWM README](../packages/verse_awm/README.md) —— JEPA / RSSM 世界模型
- [VerseInference README](../packages/verse_inference/README.md) —— 模型加载与流式生成
- [CometSpark 仓库](../data/demo/README.md) —— 端到端 LM 训练示例
- [压缩管线设计](../verse_data/designs/compression_pipeline_design.md) —— 剪枝 / 量化 / LoRA / 蒸馏的完整设计论证
- [CPU 并行 ADR](architecture/adr-004-cpu-parallel.md) —— multiprocessing 并行决策
- [压缩 PoC 基准](benchmarks/compression_poc.md) —— 1M 参数模型压缩实测数据
- [主 README](../README.md)
