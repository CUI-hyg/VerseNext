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

### 2.1 JSONL 格式（Part3K2 双格式）

> **Part3K2 BREAKING 变更**：旧版 `{"text": "..."}` 格式已废弃，`TextDataset` 加载时会抛 `ValueError`。请改用下列两种格式之一（同一文件可混用）。

Verse 训练流程使用 JSONL 文件，每行一个 JSON 对象，支持 **两种格式**：

#### 1. chat 数组格式（多轮对话）

```json
[{"role":"user","content":"你好"},{"role":"assistant","content":"你好，很高兴见到你。"}]
[{"role":"user","content":"你叫什么名字"},{"role":"assistant","content":"我叫 CometSpark。"}]
```

- 渲染为 `<|user|>你好<|assistant|>你好，很高兴见到你。<|eos|>`
- **loss mask**：仅 assistant content + `<|eos|>` 参与 loss，user 部分屏蔽（target 设为 `-100`）

#### 2. prompt-completion 格式（续写 / 单轮）

```json
{"prompt":"床前明月光，","completion":"疑是地上霜。举头望明月，低头思故乡。"}
{"prompt":"2,4,6,","completion":"8,10,12,14,16"}
```

- 渲染为 `<|user|>床前明月光，<|assistant|>疑是地上霜。<|eos|>`
- **loss mask**：仅 completion + `<|eos|>` 参与 loss，prompt 部分屏蔽

- 文件编码：UTF-8
- 换行符：`\n`
- 两种格式可在同一文件混用，`TextDataset` 自动检测每行格式

### 2.2 数据集划分

建议训练集 ≥ 200 条、验证集 ≥ 50 条、单条文本 ≥ 10 字符。文件命名约定：

```
data/
├── train.jsonl   # 训练集
└── val.jsonl     # 验证集
```

参考示例：[`data/demo/data/train.jsonl`](../data/demo/data/train.jsonl)（127 行，chat 数组 + prompt-completion 混用，覆盖唐诗续写 / 问答对话 / 数字序列三类），数据格式说明见 [`data/demo/data/README.md`](../data/demo/data/README.md)。

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

**loss mask（Part3K2）**：`TextDataset` 自动屏蔽 prompt 部分——`y` 中对应 prompt/user 的位置被设为 `-100`（`ignore_index`），`cross_entropy_loss` 会跳过这些位置不计入 loss 与梯度，仅 completion/assistant 部分参与训练。这是指令微调的标准做法，避免模型学习"复述 prompt"。

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

### 3.5 Part3K2 新增：预处理、Chat 模板与 Unigram

Part3K2 对 `verse_tokenizer` 做了全面升级，新增预处理模块、chat 模板系统与 SentencePiece Unigram 分词器，详见 [`packages/verse_tokenizer/README.md`](../packages/verse_tokenizer/README.md)。

#### 预处理（preprocess）

```python
from verse_tokenizer import nfkc_normalize, pre_tokenize, trim_to_utf8_boundary

# NFKC 归一化（全角→半角等）
nfkc_normalize("１２３ＡＢＣ")   # "123ABC"

# GPT-4 风格正则预分词：中文整字、英文单词、数字、标点、空白分别成块
pre_tokenize("床前明月光hello123")
# ['床', '前', '明', '月', '光', 'hello', '123']

# UTF-8 边界修复：防止字节序列在多字节字符中间截断导致 U+FFFD 乱码
trim_to_utf8_boundary(b"\xe4\xbd\xa0\xe5")  # b'\xe4\xbd\xa0'（"你"）
```

#### Chat 模板（chat_template）

```python
from verse_tokenizer import render_chat, render_prompt, split_prompt_completion

# chat 数组 → 渲染字符串
render_chat([{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}])
# '<|user|>你好<|assistant|>你好！<|eos|>'

# prompt → 推理前缀
render_prompt("床前明月光，")   # '<|user|>床前明月光，<|assistant|>'

# 拆分 prompt / completion（用于 loss mask）
prompt_part, completion_part = split_prompt_completion(rendered_str)
```

所有 tokenizer（BPE / Byte / Char / Unigram）通过 `apply_chat_template(messages)` / `apply_prompt_template(prompt)` 直接生成 token 序列，无需手动拼接。

#### SentencePieceUnigramTokenizer

基于 unigram 语言模型 + Viterbi 解码（EM 训练）：

```python
from verse_tokenizer import SentencePieceUnigramTokenizer

tok = SentencePieceUnigramTokenizer(vocab_size=1000)
tok.train(corpus_iter, vocab_size=1000)   # 5 轮 EM
ids = tok.encode("床前明月光", add_special_tokens=True)
text = tok.decode(ids)
chat_ids = tok.apply_chat_template([{"role":"user","content":"你好"}])
```

#### BPETokenizer 升级要点

- 接入 GPT-4 正则预分词（中文整字独立成块）
- `vocab_size` 自适应：数据不足时回退到最大可达
- 默认注册 11 个特殊 token：`<bos>`/`<eos>`/`<pad>`/`<unk>` + `<|bos|>`/`<|eos|>`/`<|pad|>`/`<|unk|>`/`<|user|>`/`<|assistant|>`/`<|system|>`
- `encode(text, add_special_tokens=True/False)` 编码开关
- `apply_chat_template(messages)` / `apply_prompt_template(prompt)` 方法

---

## 4. 模型配置与构建

### 4.1 三种架构

CometSpark 支持 `arch="transformer"` / `arch="hybrid"` / `arch="verse_nex"` 三种：

| 架构 | 主干 | 适用场景 |
|---|---|---|
| `transformer` | `GQASelfAttention` + `SwiGLUMLP`（GQA + RoPE） | PoC 与稳定性优先 |
| `hybrid` | Mamba-2 + Sparse Attention 混合块（线性复杂度） | 长上下文 / 显存敏感（待 verse_nex Mamba2 数值修复后推荐） |
| `verse_nex`（Part4） | `TriSparseAttention` + `MoDLayer`（VerseNex 原生架构） | CometSpark-V0.2 默认，0.5B 参数级训练 |

Part4 新增的 `arch="verse_nex"` 是纯 VerseNex 原生架构，**不依赖 Transformer 或 SSM**：

- **TriSparseAttention**：SWA + Global sink + ALiBi 三路并行稀疏注意力，sigmoid gate 融合
- **MoDLayer**：5 DensePart（通用/语言/数理/生化/代码）× 8 Experts × top-3，双层门控
- **layer_pattern**：显式指定每层类型，例如 `["mod", "trisparse", "trisparse", "trisparse"]`
- **forward_with_aux**：返回 `(logits, aux_loss)`，专为训练设计（aux_loss 用于 MoD 负载均衡）

### 4.2 CometSparkConfig 字段

完整字段定义见 [`data/demo/model/config.py`](../data/demo/model/config.py)，由 `config.yml` 解析得到：

```yaml
model:
  arch: transformer        # transformer / hybrid / verse_nex（Part4）
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
  # Part4 verse_nex 专用字段（arch=verse_nex 时生效，其他 arch 忽略）
  layer_pattern: null      # null → 按 mod_every 自动生成；list → 显式指定每层类型
  mod_every: 4             # 每 mod_every 层中第 0 层为 mod，其余为 trisparse
  num_dense_parts: 5       # MoD DensePart 数量（通用/语言/数理/生化/代码）
  num_experts_per_part: 8  # 每个 DensePart 的 Expert 数量
  top_k: 3                 # 每个 token 激活的 Expert 数
  window_size: 512         # TriSparse SWA 窗口
  num_global_tokens: 64    # TriSparse Global sink token 数
  use_alibi: true          # 启用 ALiBi 路径
  use_rope: false          # 启用 RoPE（与 ALiBi 互斥）
  aux_loss_weight: 0.01    # MoD aux loss 权重
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
| `rope_theta` | float | `10000.0` | RoPE 基础频率（Part3K2，与 Llama/Mistral 一致） |
| `max_position_embeddings` | int | `2048` | RoPE 预计算缓存上限（Part3K2，与 seq_len 分离） |
| `attention_dropout` | float | `0.0` | attention softmax 后的 dropout（Part3K2，独立于 dropout） |
| `hidden_dropout` | float | `0.0` | MLP 中间层 dropout（Part3K2，独立于 dropout） |
| `embedding_dropout` | float | `0.0` | embedding 后 dropout（Part3K2，独立于 dropout） |
| `layer_pattern` | list | `null` | 每层类型（`"trisparse"` / `"mod"`），null 按 mod_every 自动生成（Part4） |
| `mod_every` | int | `4` | 每 mod_every 层中第 0 层为 mod（Part4） |
| `num_dense_parts` | int | `5` | MoD DensePart 数量（Part4） |
| `num_experts_per_part` | int | `8` | 每个 DensePart 的 Expert 数（Part4） |
| `top_k` | int | `3` | 每个 token 激活的 Expert 数（Part4） |
| `window_size` | int | `512` | TriSparse SWA 窗口（Part4） |
| `num_global_tokens` | int | `64` | TriSparse Global sink token 数（Part4） |
| `use_alibi` | bool | `true` | 启用 ALiBi 路径（Part4） |
| `use_rope` | bool | `false` | 启用 RoPE（与 ALiBi 互斥，Part4） |
| `aux_loss_weight` | float | `0.01` | MoD aux loss 权重（Part4） |

Part3K2 还为 `CometSparkConfig` 新增 `from_pretrained(dir)` / `save_pretrained(dir)` 类方法，支持目录式持久化（config.json + 可扩展）。

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
- `model.save_pretrained(dir)` / `CometSparkLM.from_pretrained(dir)`（Part3K2）：目录式持久化
- `model.generate(idx, max_new_tokens=32, temperature=1.0, top_k=None, eos_id=None)`：便捷生成接口（注意：当前不支持 `top_p`，需 top_p 请用 `verse_inference.StreamingGenerator`）
- `model.compress(compress_config) -> CometSparkLM`（Part3K2）：一键压缩，返回新模型实例，不修改原模型
- `model.compression_stats() -> dict`（Part3K2）：返回 `{original_params, compressed_params, sparsity, bits, compression_ratio}`

Part3K2 还提供三个工厂函数（[`data/demo/model/model.py`](../data/demo/model/model.py)）：

```python
from model.model import CometSparkSmall, CometSparkMedium, CometSparkLarge

small  = CometSparkSmall()    # ~131K 参数，PoC 验证
medium = CometSparkMedium()   # ~853K 参数，容量提升
large  = CometSparkLarge()    # ~3M 参数，需大内存或量化
```

Part4 新增 verse_nex 工厂函数：

```python
from model.model import CometSparkV02Small, CometSparkV02

# 沙箱验证用（~0.5M 参数）
small_v02 = CometSparkV02Small(vocab_size=256, seq_len=128)
# 4 层 VerseNex（1 MoD + 3 trisparse），d_model=64

# CometSpark-V0.2（~0.5B 参数，需大内存）
v02 = CometSparkV02(vocab_size=151936)
# 32 层 VerseNex（8 MoD + 24 trisparse），d_model=384
print(v02.count_parameters())  # ≈ 537,591,264
```

verse_nex 配置样例见 [`data/demo/config/config_verse_nex.yml`](../data/demo/config/config_verse_nex.yml)（0.5B 参数）与 [`config_verse_nex_small.yml`](../data/demo/config/config_verse_nex_small.yml)（沙箱验证）。

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

### 5.2 VerseNexTrainer（Part4 新增）

`verse_torch.training_nex.VerseNexTrainer` 是专为 VerseNex 原生架构设计的训练器，关键区别在于 **aux_loss-aware**：

```python
from verse_torch.training_nex import VerseNexTrainer

trainer = VerseNexTrainer(
    model=model,                  # CometSparkLM(arch="verse_nex") 或 CometSparkNexLM
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    cfg={
        "max_steps": 200,
        "eval_interval": 20,
        "patience": 5,
        "save_dir": "checkpoints_verse_nex",
        "grad_accum": 8,           # 梯度累积 8 步 → 有效 batch = batch_size * 8
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "aux_loss_weight": 0.01,   # MoD aux loss 权重（None 时从 model.config 读取）
        "enable_progress_bar": True,
        "realtime_plot": True,
    },
)
train_losses, val_losses = trainer.fit()
```

**工作原理**：

- 初始化时自动检测 `model.forward_with_aux` 或 `model.net.forward_with_aux`
- 启用 aux 路径时：`loss = cross_entropy(logits, y) + aux_loss_weight * aux`
  - `aux` 来自 MoD 层的 Switch Transformer 风格负载均衡损失
  - `aux_loss_weight` 默认从 `model.config.aux_loss_weight` 读取，也可在 cfg 中显式覆盖
- 未启用 aux 路径时（如 `arch="transformer"`）：退化为标准 `cross_entropy` 训练
- evaluate() 不计入 aux_loss（仅用于训练时的负载均衡）

`VerseNexTrainer` 与 `Trainer` 完全兼容（相同的 cfg 字段、相同的 EarlyStopping / CheckpointManager / plot_loss_curve），但额外保存 `aux_losses.txt` 与 `aux_losses` 字段到 `loss_history.json`。

### 5.3 LoRATrainer / SFTTrainer / DPOTrainer（Part4 新增）

Part4 还提供 3 个专用训练器，均继承自 `VerseNexTrainer`：

#### LoRATrainer（LoRA 微调）

```python
from verse_torch.training_nex import LoRATrainer

trainer = LoRATrainer(
    model=model,                  # 已加载预训练权重的模型
    train_loader=train_loader,
    val_loader=val_loader,
    cfg={"max_steps": 100, "save_dir": "checkpoints_lora"},
    lora_r=8,                     # LoRA 秩
    lora_alpha=16.0,              # LoRA 缩放因子
    merge_after=True,             # fit 结束后自动 merge LoRA 到 base
)
trainer.fit()
# merge_after=True 时，fit 后模型恢复为标准 Linear 结构，可直接推理 / 保存
```

- `__init__` 时自动调用 `lora_only(model, r, alpha)` 包装所有 `Linear` 为 `LoRALinear`
- 自动冻结 base 参数，仅训练 A/B 矩阵
- `optimizer=None` 时自动基于 LoRA 参数构建 AdamW
- `merge_lora()` 方法把 ΔW 合并回 base，替换回标准 `Linear`

#### SFTTrainer（监督微调）

```python
from verse_torch.training_nex import SFTTrainer, SFTDataset, sft_collate
from verse_torch.training import BatchLoader

# 数据格式：{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
dataset = SFTDataset(tokenizer, "data/sft.jsonl", seq_len=512, ignore_index=-100)
loader = BatchLoader(dataset, batch_size=4, shuffle=True, collate_fn=sft_collate)

trainer = SFTTrainer(
    model=model, train_loader=loader, val_loader=val_loader,
    optimizer=optimizer, cfg={"max_steps": 200}, ignore_index=-100,
)
trainer.fit()
```

- 仅 assistant 回复 token 参与 loss（user/system token 被 `ignore_index=-100` 屏蔽）
- 渲染格式：`<|system|>...<|user|>...<|assistant|>...<|endoftext|>`
- 兼容 `forward_with_aux`（若模型支持，aux_loss 仍会合并入总 loss）

#### DPOTrainer（Direct Preference Optimization）

```python
from verse_torch.training_nex import DPOTrainer, DPODataset, dpo_collate

# 数据格式：{"prompt":"...","chosen":"...","rejected":"..."}
dataset = DPODataset(tokenizer, "data/dpo.jsonl", seq_len=256)
loader = BatchLoader(dataset, batch_size=2, shuffle=True, collate_fn=dpo_collate)

trainer = DPOTrainer(
    model=model,                  # policy 模型（可训练）
    ref_model=None,               # None → 深拷贝 policy 作为 reference
    train_loader=loader,
    val_loader=val_loader,
    cfg={"max_steps": 100, "beta": 0.1, "save_dir": "checkpoints_dpo"},
)
train_losses, val_losses, val_accuracies = trainer.fit()
```

- DPO loss = `-mean(log σ(β·((π_chosen - π_rejected) - (ref_chosen - ref_rejected))))`
- reference model 自动冻结（`requires_grad=False`）
- 自动计算 accuracy = mean(π_chosen > π_rejected)
- 保存 `dpo_history.json` + `dpo_curve.png`（含 loss 与 accuracy 双曲线）

### 5.4 完整训练流程

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

### 5.5 训练循环内部组件

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

### 5.6 ParallelTrainer — 并行训练器（Part3K2，Part4 增强 aux_loss）

`verse_torch.training.ParallelTrainer` 把 `max_steps` 拆成 N 个 chunk，每个 chunk 独立训练，再按「差前好后」串行重训 + 整体 fine-tune。CPU 友好（串行执行，避免 GIL 竞争）。

**关键设计**：
- **chunk 拆分**：`max_steps` 拆为 `parallel_chunks` 份，每份独立 `Trainer` 实例
- **合并策略**：按 `train_loss + val_loss` 排序，效果差放前面、好的放后面串行重训
- **val_loss 漏洞修复**：`_eval_full_val()` 基于完整 val 数据集更新 `best_val_loss`（旧实现只用单 batch，不可比且方差大）
- **整体 fine-tune**：`merge_finetune_steps = max_steps // 10`，在最佳状态上微调

```python
from verse_torch import ParallelTrainer

cfg = {
    "parallel_chunks": 4,          # 拆 4 个 chunk
    "max_steps": 200,
    "batch_size": 8,
    "lr": 3e-3,
    "warmup": 20,
    "eval_interval": 20,
    "grad_clip": 1.0,
    "label_smoothing": 0.1,
    "seed": 42,
    # merge_finetune_steps 默认 max_steps // 10
}
trainer = ParallelTrainer(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    cfg=cfg,
)
trainer.fit()
print(f"best_val_loss={trainer.best_val_loss:.4f}")
```

CometSpark demo 通过 `training.parallel_chunks` 配置或 `--parallel-chunks` CLI 参数切换 `Trainer` / `ParallelTrainer`（`1` = 标准 Trainer，`>1` = ParallelTrainer）。

**Part4 增强**：`ParallelTrainer` 现在自动检测 `forward_with_aux`，启用 aux_loss 路径时：
- `_train_chunk` 内部使用 `VerseNexTrainer` 而非 `Trainer`，正确处理 MoD aux_loss
- `_eval_full_val` 调用 `forward_with_aux` 取 logits，避免 (logits, aux) tuple 破坏 loss_fn
- `chunk_cfg` 自动写入 `aux_loss_weight`，与 `VerseNexTrainer` 配置一致

### 5.7 高级优化器与调度器（Part3K2）

Part3K2 对齐 PyTorch / HuggingFace 补齐了高级优化器与调度器（[`verse_torch.optim_extras`](../packages/verse_torch/verse_torch/optim_extras.py) / [`scheduler_extras`](../packages/verse_torch/verse_torch/scheduler_extras.py)）：

| 优化器 | 签名 | 特点 |
|---|---|---|
| `Lion` | `Lion(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)` | sign-based 更新，无二阶矩，内存省 |
| `Adafactor` | `Adafactor(params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30, eps2=1e-3, weight_decay=0.0)` | factored 二阶矩，省显存 |

| 调度器 | 签名 | 特点 |
|---|---|---|
| `OneCycleLR` | `OneCycleLR(opt, max_lr, total_steps, pct_start=0.25, div_factor=25.0, final_div_factor=1e4)` | super-convergence |
| `ReduceLROnPlateau` | `ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, min_lr=0, threshold=1e-4)` | val 不降时降 lr |
| `CosineRestartsLR` | `CosineRestartsLR(opt, T_0, T_mult=1, eta_min=0)` | SGDR 带热重启 |

```python
from verse_torch import Lion, OneCycleLR

opt = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
sched = OneCycleLR(opt, max_lr=1e-4, total_steps=200)
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
| Top-p（nucleus） | `top_p=p` | 仅在累积概率 ≥ p 的最小 token 集上采样。**注意**：`CometSparkLM.generate` 当前不支持 `top_p`，需用 `verse_inference.StreamingGenerator` 或 `Trainer.inference`（Part3K2） |

5 条预设 prompt（CometSpark demo 默认）：

```python
prompts = ["床前明月光，", "白日依山尽，", "你好，", "1+1=", "春风"]
```

### 6.3 ScoringEvaluator — 生成质量打分（Part3K2）

`verse_torch.scoring.ScoringEvaluator` 对模型生成结果与参考答案计算 5 个指标：

| 指标 | 说明 |
|---|---|
| `exact_match` | 精确匹配率（完全相等 1.0） |
| `prefix_accuracy` | 前缀匹配率（适合续写任务） |
| `char_f1` | 字符级 F1 |
| `bleu` | BLEU-4 简化版 |
| `rouge_l` | ROUGE-L（基于最长公共子序列的 F1） |

```python
from verse_torch import ScoringEvaluator

evaluator = ScoringEvaluator()
scores = evaluator.evaluate(
    predictions=["疑是地上霜。举头望明月"],
    references=["疑是地上霜。举头望明月，低头思故乡。"],
)
print(evaluator.report(scores))
# ==================================================
# 评分报告
# ==================================================
# 样本数: 1
# --------------------------------------------------
#   exact_match        : 0.0000
#   prefix_accuracy    : 0.7333
#   char_f1            : 0.8462
#   bleu               : 0.0000
#   rouge_l            : 0.8462
# ==================================================
```

CometSpark demo 通过 `--score --references-file refs.txt` 启用打分模式（详见 [`data/demo/README.md`](../data/demo/README.md)）。

### 6.4 Trainer.inference — 批量推理生成（Part3K2）

`Trainer.inference(prompts, temperature, top_k, top_p, max_tokens)` 提供批量生成入口，支持字符串 prompt（配合 tokenizer）或 token ID 序列：

```python
from verse_torch.training import Trainer

# trainer 已训练完成
outputs = trainer.inference(
    prompts=["床前明月光，", "你好，"],
    temperature=0.8, top_k=10, max_tokens=30,
)
# outputs: list[str]（有 tokenizer 时）或 list[list[int]]（无 tokenizer 时）
```

`Trainer.inference` 内部优先委托 `model.generate`，若模型未实现 `generate` 则手动循环 forward + 采样，完整支持 `temperature` / `top_k` / `top_p`。

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

### 7.5 CometSparkLM.compress / compression_stats（Part3K2）

`CometSparkLM` 内置一键压缩入口，返回压缩后的新模型实例（不修改原模型）：

```python
from model.model import CometSparkSmall

model = CometSparkSmall()

# 一键压缩：50% 稀疏 + INT4 量化
compress_config = {
    "prune":    {"sparsity": 0.5, "method": "outlier_safe"},
    "quantize": {"bits": 4, "schema": "symmetric"},
    # 可选：任意组合 lora / ternary / distill
    # "lora":    {"rank": 8, "alpha": 16},
    # "ternary": {},
    # "distill": {"teacher": teacher_model, "epochs": 10, "lr": 1e-4},
}
compressed = model.compress(compress_config)

# 查看压缩统计
stats = compressed.compression_stats()
# {'original_params': 131776, 'compressed_params': 32700.0,
#  'sparsity': 0.5, 'bits': 4.0, 'compression_ratio': 4.03}
print(f"压缩比: {stats['compression_ratio']:.2f}x")
```

### 7.6 MoD Expert 结构化剪枝（Part4 新增）

`verse_torch.compress.compress_mod_experts` 针对 VerseNex 原生架构的 MoD 层进行结构化剪枝：

```python
from verse_torch.compress import compress_mod_experts
from model.model import CometSparkV02Small

model = CometSparkV02Small()

# 保留 50% 的 Experts（按参数 L2 范数排序，丢弃低利用率的）
stats = compress_mod_experts(
    model,
    keep_ratio=0.5,              # 保留比例
    min_experts_per_part=1,      # 每个 DensePart 至少保留 1 个 Expert
    return_stats=True,
)
print(stats)
# {
#   'original_experts': 4,       # 2 DensePart × 2 Experts = 4
#   'kept_experts': 2,           # 保留 2 个
#   'compression_ratio': 0.5,    # 压缩率 50%
# }
```

**工作原理**：

1. 遍历模型中所有 `MoDLayer` 实例
2. 对每个 DensePart 内的 Experts，按参数 L2 范数排序
3. 保留范数最高的 `max(min_experts_per_part, int(num_experts * keep_ratio))` 个
4. 删除被裁 Expert 后，同步修改 `expert_router` 权重矩阵的对应行
5. 修改 `top_k = min(self.top_k, remaining_experts)`

适用场景：MoD 模型部署前的体积压缩，特别适合 CPU 推理（减少 Expert 加载开销）。

### 7.7 压缩训练演示脚本

[`examples/compress_train_demo.py`](../examples/compress_train_demo.py) 演示完整流程：创建基准模型 → 压缩 → 统计 → forward 验证：

```bash
cd /workspace
python examples/compress_train_demo.py
# [1] 创建基准模型 CometSparkSmall... 参数量: 131776
# [2] 压缩：50% 通道稀疏 + INT4 量化...
# [3] 压缩统计: 压缩比 4.03x
# [4] Forward 验证（输入随机 token）... 输出 shape 正确
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

# Part3K2 新增
python run.py --parallel-chunks 4 # 启用 ParallelTrainer（4 chunk 拆分）
python run.py --top-p 0.9         # nucleus sampling（降级为不限制，见 6.2）
python run.py --score --references-file refs.txt  # 启用 ScoringEvaluator 打分
```

完整 CLI 参数说明见 [`data/demo/README.md`](../data/demo/README.md)。

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
