# Verse 训练指南

> 本指南介绍如何用 Verse 框架从零训练一个语言模型，覆盖**数据准备 → tokenizer 构建 → 模型配置 → 训练 → 评估与生成 → 压缩 → 推理部署**全流程。Verse 默认纯 Python / 纯 CPU 环境运行（无 PyTorch / TensorFlow / JAX 硬依赖），Part4K1 起可选启用 GPU/NPU 加速（通过 PyTorch 委托后端）。

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [Tokenizer 构建](#3-tokenizer-构建)
4. [模型配置与构建](#4-模型配置与构建)
5. [训练](#5-训练)
6. [评估与生成](#6-评估与生成)
7. [模型压缩](#7-模型压缩)
8. [推理部署](#8-推理部署)
9. [完整示例：CometSpark 双模型](#9-完整示例cometspark-v05-1b)
10. [VerseTrainer CLI 速查](#10-versetrainer-cli-速查)
11. [智能分区训练指南（Part4K2 新增）](#11-智能分区训练指南part4k2-新增)
12. [持续训练指南（Part4K2 新增）](#12-持续训练指南part4k2-新增)
13. [jinja2 聊天模板使用指南（Part4K2 新增）](#13-jinja2-聊天模板使用指南part4k2-新增)
14. [.vn 格式使用指南（Part4K2 新增）](#14-vn-格式使用指南part4k2-新增)
15. [数据集下载指南（Part4K2 新增）](#15-数据集下载指南part4k2-新增)
16. [spark/run.py 快速训练指南（Part4K2.5 新增）](#16-sparkrunpy-快速训练指南part4k25-新增)
17. [训练后自动评估指南（Part4K2.5 新增）](#17-训练后自动评估指南part4k25-新增)
18. [并行训练修复说明（Part4K2.5 新增）](#18-并行训练修复说明part4k25-新增)
19. [loss 图表修复说明（Part4K2.5 新增）](#19-loss-图表修复说明part4k25-新增)
20. [VMPC V2.0 压缩指南（Part5K1.1 大升级）](#20-vmpc-v20-压缩指南part5k11-大升级)
21. [双模型训练指南（Part5K1 / Part5K1.1）](#21-双模型训练指南part5k1--part5k11)
22. [VMT 三档策略指南（Part5K1 新增）](#22-vmt-三档策略指南part5k1-新增)
23. [64+ 层训练加速指南（Part5K1 新增）](#23-64-层训练加速指南part5k1-新增)
24. [spark/run.py 训练模式补齐（Part5K1 新增）](#24-sparkrunpy-训练模式补齐part5k1-新增)
25. [JSONL 自修复指南（Part5K1 新增）](#25-jsonl-自修复指南part5k1-新增)
26. [vnn 重命名迁移指南（Part5K1 新增）](#26-vnn-重命名迁移指南part5k1-新增)

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
            -e packages/verse_infra        # 聚合 verse_tokenizer / verse_compat / verse_inference / verse_trainer

# 方式二：uv workspace 一次性安装全部成员
uv sync

# 可选运行时依赖（按需）
pip install "numba>=0.60"        # CPU GEMM 加速
pip install "safetensors>=0.4"   # 加载 .safetensors 权重
pip install "fastapi>=0.110"     # OpenAI 兼容 HTTP server

# Part4K1 新增：GPU/NPU 加速（可选）
pip install "torch>=2.0"          # GPU 委托后端（cuda / mps）
pip install torch_npu             # 华为昇腾 NPU 后端（仅在 NPU 设备上需要）
```

安装 `verse_infra` 后会注册 7 个 VerseTrainer CLI 入口（详见第 10 节）：`verse-train` / `verse-finetune` / `verse-posttrain` / `verse-eval` / `verse-tokenize` / `verse-convert` / `verse-download`，外加通过统一分发入口调用的 `verse-continue`（`python -m verse_infra.verse_trainer.cli verse-continue`）。

> **Part4K1 导入路径变更**：`verse_tokenizer` / `verse_inference` / `verse_compat` 已聚合为 `verse_infra` 子模块。新代码请用 `from verse_infra.verse_tokenizer import BPETokenizer`；旧路径 `from verse_tokenizer import BPETokenizer` 仍可用（经 shim 转发 + DeprecationWarning，一个版本）。

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

参考示例：[`spark/data/train.jsonl`](../spark/data/train.jsonl)（chat 数组 + prompt-completion 混用，覆盖唐诗续写 / 问答对话 / 数字序列三类），数据格式说明见 [`spark/data/README.md`](../spark/data/README.md)。

### 2.3 加载与切分

`verse_torch` 没有内置 `Dataset` 抽象，VerseTrainer 提供 `CachedDataset` 与 `BatchLoader` 作为参考实现（见 [`packages/verse_infra/verse_infra/verse_trainer/data.py`](../packages/verse_infra/verse_infra/verse_trainer/data.py)）：

```python
from verse_infra.verse_trainer.data import CachedDataset
from verse_torch.training import BatchLoader, collate_fn

train_ds = CachedDataset(tokenizer, "data/train.jsonl", seq_len=64)
val_ds = CachedDataset(tokenizer, "data/val.jsonl", seq_len=64)

train_loader = BatchLoader(
    train_ds, batch_size=8, shuffle=True,
    collate_fn=collate_fn, drop_last=False, seed=42,
)
val_loader = BatchLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
```

每个 batch 返回 `(x, y)`，其中 `x.shape = (B, T)`、`y.shape = (B, T)` 是 `x` 向左移一位的目标序列（next-token prediction）。

**CachedDataset（Part4K1）**：首次扫描数据集时把每条样本的 token ID 缓存到 `.npz` 文件（与数据集同目录），后续启动直接 mmap 加载，跳过 tokenize 重复开销。适合 mate 旗舰（≈1.12B 参数）模型的大规模数据集场景（数百万条样本），加速效果随数据集规模增长。

**loss mask（Part3K2）**：`CachedDataset` 自动屏蔽 prompt 部分——`y` 中对应 prompt/user 的位置被设为 `-100`（`ignore_index`），`cross_entropy_loss` 会跳过这些位置不计入 loss 与梯度，仅 completion/assistant 部分参与训练。这是指令微调的标准做法，避免模型学习"复述 prompt"。

---

## 3. Tokenizer 构建

Verse 提供 3 种 tokenizer，由 [`verse_tokenizer`](../packages/verse_tokenizer/README.md) 包统一管理。

### 3.1 ByteTokenizer（推荐入门）

固定词表 259（256 字节 + BOS/EOS/PAD/UNK），无需训练：

```python
from verse_infra.verse_tokenizer import ByteTokenizer

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
from verse_infra.verse_tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(corpus_iter, vocab_size=2000)   # corpus_iter 是字符串迭代器
tok.save("tokenizer.json")
# 之后可 tok.add_special_tokens(["<bos>", "<eos>", "<pad>", "<unk>"])
```

加载：

```python
from verse_infra.verse_tokenizer import BPETokenizer
tok = BPETokenizer.load("tokenizer.json")
```

### 3.3 HF tokenizer（需 transformers）

若已有 HuggingFace `tokenizer.json`，可直接复用：

```python
from verse_infra.verse_tokenizer import BPETokenizer
tok = BPETokenizer.load("path/to/hf_tokenizer.json")
```

### 3.4 工厂函数

```python
from verse_infra.verse_tokenizer import load_tokenizer
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
from verse_infra.verse_tokenizer import nfkc_normalize, pre_tokenize, trim_to_utf8_boundary

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
from verse_infra.verse_tokenizer import render_chat, render_prompt, split_prompt_completion

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
from verse_infra.verse_tokenizer import SentencePieceUnigramTokenizer

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

### 4.1 架构选择（Part4K1：仅保留 versenex）

CometSpark 历史 `arch` 字段曾支持 `transformer` / `hybrid` / `verse_nex` 三种，**Part4K1 起仅保留 `versenex` 唯一值**：

| 架构 | 主干 | 状态 |
|---|---|---|
| `versenex`（推荐，Part4K1 唯一） | `TriSparseAttention` + `MoDLayer`（VerseNex 原生架构） | 默认，CometSpark 双模型 使用 |
| `transformer` | `GQASelfAttention` + `SwiGLUMLP`（GQA + RoPE） | **deprecated**：`config.yml` 写 `transformer` 会自动映射为 `versenex` + DeprecationWarning |
| `hybrid` | Mamba-2 + Sparse Attention 混合块（线性复杂度） | **deprecated**：`HybridBlock` / `HybridLM` 保留只读兼容；`config.yml` 写 `hybrid` 同样映射 |

`versenex` 架构是纯 VerseNex 原生架构，**不依赖 Transformer 或 SSM**：

- **TriSparseAttention**：SWA + Global sink + ALiBi 三路并行稀疏注意力，sigmoid gate 融合，支持多 query chunk 并行（Part4K1）
- **MoDLayer**：5 DensePart（通用/语言/数理/生化/代码）× 8 Experts × top-3，双层门控 + `load_balance_loss` + `router z-loss`（Part4K1 完善）
- **layer_pattern**：显式指定每层类型，例如 `["mod", "trisparse", "trisparse", "trisparse"]`
- **forward_with_aux**：返回 `(logits, aux_loss)`，专为训练设计（aux_loss 用于 MoD 负载均衡 + router 稳定性）
- **品牌落地（Part4K1）**：`TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention`、`VerseNexBlock` 统一为唯一名（旧名作为 `DeprecationWarning` 别名）

### 4.2 CometSparkConfig 字段

完整字段定义见 [`spark/src/base_config.py`](../spark/src/base_config.py)（CometSpark 双模型的 `CometSparkV05Config` 基类）与 [`packages/verse_nex/verse_nex/cometspark.py`](../packages/verse_nex/verse_nex/cometspark.py)（VerseNex 通用配置），由 `config.yml` 解析得到：

```yaml
model:
  arch: versenex           # Part4K1：仅 versenex 唯一值（transformer / hybrid 自动映射 + DeprecationWarning）
  vocab_size: 256          # 由 tokenizer 自动覆盖
  n_layer: 2               # Transformer block 数
  n_head: 4                # 注意力头数
  n_embd: 64               # 模型维度
  seq_len: 64              # 上下文长度
  dropout: 0.1
  n_kv_head: 2             # GQA 的 KV 头数（n_head // n_kv_head = repeat 因子）
  ssm_kind: mamba2         # hybrid 下的 SSM 种类（hybrid 已 deprecated）
  sparse_ratio: 0.5        # hybrid block 中 sparse attention 比例（hybrid 已 deprecated）
  tie_weights: true        # 是否共享 embedding 与 head 权重
  # Part4 verse_nex 专用字段（arch=versenex 时生效）
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

> **CometSpark 双模型 配置**：mate 旗舰（≈1.12B 参数）级训练用 `CometSparkV05Config`，默认 `n_embd=1024, n_layer=20, 5 MoD + 15 trisparse, 4 DensePart × 4 Expert × top-2 + tie_weights=True`，约 1.12B 参数。完整 `config.yml` 见 [`spark/mate/config/cometspark_mate.yml`](../spark/mate/config/cometspark_mate.yml)。

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

Part3K2 还提供三个工厂函数（[`spark/src/base_model.py`](../spark/src/base_model.py)）：

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

verse_nex 配置样例见 [`spark/small/config/cometspark_small.yml`](../spark/small/config/cometspark_small.yml)（沙箱验证）。

#### CometSpark 双模型（Part4K1，推荐）

CometSpark 双模型是 Part4K1 起的主力模型，基于 `VerseNexBlock`（TriSparse + MoD），mate 旗舰约 1.12B 参数：

```python
from spark.small.model import CometSparkSmall, CometSparkSmallLM
from spark.mate.model import CometSparkMate, CometSparkMateLM
from spark.src.base_config import CometSparkV05Config

# 1B 主力模型（≈ 1.12B 参数，需大内存或量化）
v05 = CometSparkMate()         # vocab 来自 Qwen3.5-35B-A3B tokenizer (248320)
print(v05.count_parameters()) # ≈ 1,115,000,000

# 沙箱验证小配置（调试用）
small_v05 = CometSparkSmall()

# 从 config 构建
config = CometSparkV05Config.from_pretrained("spark/mate/config/cometspark_mate.yml")
model = CometSparkV05LM(config)
```

CometSpark 双模型关键优化（解决胡乱输出）：
- **embedding scale**：`tok_emb(idx) * sqrt(n_embd)` 防止 embedding 量级过小
- **tie_weights**：`lm_head` 与 `tok_emb` 共享权重（减少参数 + 稳定训练）
- **temperature scaling**：生成时 `logits / temperature` 控制随机性
- **合理初始化**：normal + 残差缩放，由 `CometSparkNexLM._init_weights` 完成

### 4.4 纯 verse_torch 构建（不用 CometSpark）

若不依赖 CometSpark 封装，可直接用 `verse_torch.nn.TransformerLM`（Part4K1 起在 `verse_nex` 包内有更名后的 `VerseNexLM`，行为等价）：

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

参考 [`spark/src/trainer.py`](../spark/src/trainer.py)（CometSpark 双模型的训练入口，内部调用 `verse_infra.verse_trainer`）：

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

### 5.8 GPU/NPU 训练（Part4K1）

Part4K1 引入 `DeviceBackend` 抽象层后，Verse 支持 CPU / GPU / NPU 三种设备训练。CPU-first 不变，PyTorch 仅作为可选加速后端（详见 [ADR-005](architecture/adr-005-gpu-npu-backend.md)）。

#### 5.8.1 设备选择

| 设备 | 命令行参数 | 后端 | 依赖 |
|---|---|---|---|
| CPU（默认） | `--device cpu` | `NumpyBackend`（自研 autograd） | 无 |
| GPU（CUDA） | `--device cuda` | `TorchBackend`（委托 `torch`） | `pip install torch` |
| Apple Silicon | `--device mps` | `TorchBackend`（委托 `torch`） | `pip install torch`（macOS） |
| 华为 NPU | `--device npu` | `TorchBackend`（委托 `torch_npu`） | `pip install torch torch_npu` |

#### 5.8.2 Python API 用法

```python
from spark.mate.model import CometSparkMate

model = CometSparkMate()
model = model.to("cuda")          # 迁移到 GPU（递归迁移所有参数）
# 或 model = model.to("npu")      # NPU
# 或 model = model.cpu()           # 回到 CPU

# 之后所有 forward / backward 自动走 GPU 后端
optimizer = AdamW(model.parameters(), lr=3e-4)
logits = model(input_ids)         # GPU 上 forward
loss = cross_entropy_loss(logits, targets)
loss.backward()                   # 委托 torch.Tensor.backward()
optimizer.step()
```

#### 5.8.3 CLI 用法

```bash
# GPU 训练（需安装 torch）
verse-train --config spark/mate/config/cometspark_mate.yml --device cuda --amp

# NPU 训练（需安装 torch + torch_npu）
verse-train --config spark/mate/config/cometspark_mate.yml --device npu --amp

# CPU 训练（默认，零依赖）
verse-train --config spark/mate/config/cometspark_mate.yml --device cpu
```

#### 5.8.4 混合精度（autocast）

`--amp` 启用混合精度训练（仅 GPU/NPU 后端生效，CPU 为 no-op）：

```python
from verse_torch.backend_torch import autocast

with autocast(device="cuda", enabled=True):
    logits = model(x)              # fp16 前向
    loss = cross_entropy_loss(logits, y)
loss.backward()                    # 反向走 torch autograd
optimizer.step()
```

混合精度可将 GPU 显存占用降低约 40%，吞吐提升 1.5×~2×（视模型规模与 GPU 架构）。CPU 后端 `autocast` 为 no-op，不损失精度。

#### 5.8.5 无 PyTorch 回退

未安装 PyTorch 时：
- `--device cpu` 完全可用（默认）
- `--device cuda` / `--device npu` 抛 `RuntimeError("未安装 PyTorch，无法使用 device 'cuda'")`
- `Tensor.cuda()` / `Tensor.npu()` 抛同样错误
- `autocast(...)` 为 no-op
- 所有现有 CPU 测试不变通过（向后兼容）

### 5.9 并行训练 CLI（Part4K1）

`--parallel-chunks N` 启用 `ParallelTrainer`，把 `max_steps` 拆成 N 个 chunk 并行训练（CPU 串行实现，接口对齐并行），训练完后按 `train_loss + val_loss` 排序串行重训 + 整体 fine-tune。

```bash
# 4 chunk 并行训练（chunk 内独立 Trainer，最后合并 + finetune）
verse-train --config spark/mate/config/cometspark_mate.yml \
    --parallel-chunks 4 --max-steps 200

# 单样本 + 并行训练（沙箱调试）
verse-train --config spark/small/config/cometspark_small.yml \
    --single-sample --prompt "床前明月光，" --completion "疑是地上霜。" \
    --parallel-chunks 2 --max-steps 50
```

`--parallel-chunks 1`（默认）= 标准 `Trainer`；`>1` = `ParallelTrainer`。详见 5.6 节 ParallelTrainer 工作原理。

#### 5.9.1 Loss 优化策略（Part4K1）

`--loss-optimizer` 启用 `LossOptimizer`（参考 GPT_teacher-3.37M-cn 实践）：

- **梯度裁剪**：`grad_clip` 阈值（cfg 字段）
- **LR 组合调度**：warmup + cosine + `ReduceLROnPlateau`
- **loss plateau 重走**（`maybe_rollback`）：连续 `patience` 步 val_loss 未下降时，回退 best_state_dict + LR × 0.3 + 重置 Adam 动量（m/v 清零）+ 继续
- **NaN/Inf 跳过**（`check_loss_finite`）：loss 为 NaN/Inf 时跳过该 batch，不更新参数

```bash
verse-train --config spark/mate/config/cometspark_mate.yml \
    --parallel-chunks 4 --loss-optimizer --max-steps 1000
```

### 5.10 NexRL 后训练（Part4K1）

`verse-posttrain --rl nexrl` 启用基于 NexRL 的强化学习后训练（PPO + GAE + KL 自适应 + value function）。NexRL 五要素：`NexAgent`（策略 + 参考网络）/ `NexEnv` / `NexState` / `NexAction` / `NexReward`，详见 [ADR-007](architecture/adr-007-nexrl-design.md)。

```bash
# NexRL 后训练
verse-posttrain --config spark/mate/config/cometspark_mate.yml \
    --rl nexrl --data data/rl_prompts.jsonl --device cuda

# SFT 后训练（监督微调）
verse-posttrain --config spark/mate/config/cometspark_mate.yml \
    --rl sft --data data/sft.jsonl

# DPO 后训练（Direct Preference Optimization）
verse-posttrain --config spark/mate/config/cometspark_mate.yml \
    --rl dpo --data data/dpo.jsonl
```

#### 5.10.1 Python API

```python
from verse_nex.nexrl import NexAgent, NexTrainer, NexReward, ParallelRolloutCollector

agent = NexAgent(policy=model)               # 自动深拷贝参考网络
collector = ParallelRolloutCollector(agent=agent, env=env)
trainer = NexTrainer(
    agent=agent,
    collector=collector,
    cfg={
        "clip_ratio": 0.2,                   # PPO clip
        "gae_lambda": 0.95,                  # GAE lambda
        "kl_coef": 0.1,                      # KL 自适应初始权重
        "kl_target": 6.0,                    # KL 目标值
        "n_epochs": 10,                      # 训练轮数
    },
)
trainer.fit(prompts=["1+1=", "2+2=", "3+3="])
```

#### 5.10.2 NexReward 多维奖励

`NexReward` 支持四维加权奖励 + reward normalization（running mean/std）+ reward shaping（potential-based）：

| 维度 | 说明 |
|---|---|
| `correctness` | 答案正确性（精确匹配 / 数学验证） |
| `fluency` | 流畅度（n-gram 重复率 / 困惑度） |
| `safety` | 安全性（敏感词过滤） |
| `length_penalty` | 长度惩罚（过短 / 过长都扣分） |

#### 5.10.3 KL 自适应防崩溃

NexTrainer 监控策略网络与参考网络的 KL 散度，超阈值时自动增加 KL 惩罚权重，防止策略崩溃：

```
if kl > kl_target * 2:
    kl_coef *= 2.0     # 加大 KL 约束
elif kl < kl_target * 0.5:
    kl_coef *= 0.5     # 放松约束，鼓励探索
```

---

## 6. 评估与生成

### 6.1 加载 best.pt 生成文本

参考 [`spark/src/evaluate.py`](../spark/src/evaluate.py)：

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
if os.path.exists("mf_mate/best.vn"):
    model = CometSparkLM.from_pretrained("mf_mate/best.vn")
else:
    config = CometSparkConfig(arch="transformer", vocab_size=len(tok), ...)
    model = CometSparkLM(config)
    with open("mf_mate/best.vn", "rb") as f:
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

CometSpark demo 通过 `--score --references-file refs.txt` 启用打分模式（详见 [`spark/README.md`](../spark/README.md)）。

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

[`verse_infra.verse_inference`](../packages/verse_infra/README.md)（Part4K1 起聚合为 `verse_infra` 子模块）提供：

| 类 | 作用 |
|---|---|
| `ModelLoader` | 从 HF repo / 本地路径加载 LM；支持 `arch=mamba2 / rwkv7 / hybrid / cometspark` |
| `StateCache` | Mamba / RWKV 递归状态缓存（O(1) 内存推理） |
| `Sampler` / `GreedySampler` | temperature / top_k / top_p 采样器 |
| `StreamingGenerator` | 流式生成器：prefill → decode → yield 逐 token |

### 8.2 加载 cometspark 模型

```python
from verse_infra.verse_inference import ModelLoader, StreamingGenerator, Sampler

# ModelLoader 会自动从 pickle 加载 config + state_dict 重建 CometSparkLM
loader = ModelLoader(arch="cometspark")
model = loader.load("mf_mate/best.vn")
```

`cometspark` arch 分支由 Stage 7 新增，详见 [`packages/verse_inference/verse_inference/model_loader.py`](../packages/verse_inference/verse_inference/model_loader.py)。

### 8.3 CPU 流式生成

```python
from verse_infra.verse_inference import StreamingGenerator, Sampler

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
from verse_infra.verse_inference.server import serve
serve(model, tokenizer=tok, host="0.0.0.0", port=8000)
```

详见 [`packages/verse_inference/verse_inference/server.py`](../packages/verse_inference/verse_inference/server.py)。

---

## 9. 完整示例：CometSpark 双模型

CometSpark 双模型是 Part4K1 起的端到端 mate 旗舰（≈1.12B 参数）语言模型训练仓库，基于 VerseNex 原生架构（TriSparse + MoD），覆盖本指南全部能力。详见 [`spark/README.md`](../spark/README.md)。

### 9.1 一键运行（VerseTrainer CLI）

```bash
cd /workspace

# CPU 训练（默认，零依赖）
verse-train --config spark/mate/config/cometspark_mate.yml

# GPU 训练 + 混合精度（需安装 torch）
verse-train --config spark/mate/config/cometspark_mate.yml --device cuda --amp

# 并行训练 + Loss 优化
verse-train --config spark/mate/config/cometspark_mate.yml \
    --parallel-chunks 4 --loss-optimizer --max-steps 1000

# 断点续训
verse-train --config spark/mate/config/cometspark_mate.yml --resume
```

### 9.2 沙箱调试（小配置）

```bash
# 单样本 + 小配置（沙箱验证）
verse-train --config spark/small/config/cometspark_small.yml \
    --single-sample --prompt "床前明月光，" --completion "疑是地上霜。" \
    --max-steps 50
```

### 9.3 后训练 / 微调 / 评估

```bash
# NexRL 强化学习后训练
verse-posttrain --config spark/mate/config/cometspark_mate.yml \
    --rl nexrl --data data/rl_prompts.jsonl

# LoRA 微调
verse-finetune --config spark/mate/config/cometspark_mate.yml \
    --method lora --data data/sft.jsonl

# 评估 + 打分
verse-eval --config spark/mate/config/cometspark_mate.yml \
    --checkpoint mf_mate/best.vn \
    --prompts-file data/prompts.txt --score --references-file data/refs.txt

# tokenizer 训练 / 加载 / 转换
verse-tokenize --from-hf Qwen/Qwen3.5-35B-A3B    # 加载 Qwen tokenizer（vocab 248320）
```

### 9.4 实测性能（沙箱小配置）

在 3 核 CPU / 5GB 内存沙箱中（`cometspark_v05_small.yml` 调试配置）：

| 指标 | 实测 |
|---|---|
| wall-clock | ~30 秒 |
| 初始 train loss | ~10.5 |
| 最终 train loss | ~3.2 |
| 最佳 val loss | ~3.8 |
| checkpoints | `best.vn` / `last.vn` / `loss_history.json` / `loss_curve.txt` |

> mate 旗舰模型（`cometspark_mate.yml`）需要大内存或 GPU 环境，沙箱不可直接训练。

### 9.5 目录结构（Part5K1.1）

```
spark/
├── README.md                       # CometSpark 双模型 模型说明
├── small/                          # 0.06zB 小模型（VMPC-small 预设）
│   ├── config/
│   │   └── cometspark_small.yml    # small 配置（n_embd=64, n_layer=2, vocab=256）
│   └── model/
│       ├── config.py               # CometSparkSmallConfig
│       └── model.py                # CometSparkSmallLM + CometSparkSmall 工厂
├── mate/                           # 0.2zB 旗舰模型（VMPC-mate 预设）
│   ├── config/
│   │   └── cometspark_mate.yml     # mate 配置（n_embd=1024, n_layer=20, 5 MoD + 15 trisparse）
│   └── model/
│       ├── config.py               # CometSparkMateConfig
│       └── model.py                # CometSparkMateLM + CometSparkMate 工厂
├── src/                            # 共享基础组件
│   ├── base_config.py              # CometSparkV05Config 基类
│   ├── base_model.py               # CometSparkV05LM 基类
│   ├── trainer.py                  # 训练入口（调用 verse_infra.verse_trainer）
│   ├── evaluate.py                 # ScoringEvaluator 评估
│   └── utils.py                    # 辅助工具
├── data/
│   ├── train.jsonl                 # 训练集
│   ├── val.jsonl                   # 验证集
│   └── README.md                   # 数据格式说明
└── mf_small/ / mf_mate/            # 训练产物（自动生成，.vn 格式）
```

### 9.6 依赖

- `verse_torch` / `verse_nex` / `verse_infra`（含 `verse_trainer` / `verse_tokenizer` / `verse_inference`）
- CPU 训练：运行时**不需要** PyTorch / TensorFlow / JAX / transformers
- GPU/NPU 训练：可选安装 `torch`（+ `torch_npu` for NPU）
- tokenizer：使用 HuggingFace `Qwen/Qwen3.5-35B-A3B` tokenizer（vocab 248320），由 `verse_infra.verse_tokenizer.BPETokenizer.from_pretrained` 加载

---

## 10. VerseTrainer CLI 速查

VerseTrainer 提供 8 个 CLI 入口（7 个注册为 console_scripts + 1 个通过统一分发入口调用）：

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `verse-train` | 预训练 | `--config` / `--device cpu\|cuda\|npu` / `--parallel-chunks N` / `--max-steps` / `--resume` / `--amp` / `--loss-optimizer` / `--single-sample` / `--partition-training` / `--partition-size N` / `--prompt` / `--completion` / `--single-file` |
| `verse-continue` | **持续训练（Part4K2）** | `--checkpoint` / `--additional-steps` / `--config` / `--device` / `--amp`（通过 `python -m verse_infra.verse_trainer.cli verse-continue` 调用） |
| `verse-finetune` | 微调 | `--config` / `--method lora\|full` / `--device` / `--data` |
| `verse-posttrain` | 后训练 | `--config` / `--rl nexrl\|sft\|dpo` / `--device` / `--data` |
| `verse-eval` | 评估 + 打分 | `--config` / `--checkpoint` / `--prompts-file` / `--references-file` / `--score` / `--max-tokens`（默认 None=EOS 自然停止） |
| `verse-tokenize` | tokenizer 训练 / 加载 / 转换 | `--train` / `--load` / `--convert` / `--from-hf Qwen/Qwen3.5-35B-A3B` |
| `verse-download` | **数据集下载（Part4K2）** | `--url` / `--hf` / `--split` / `--to-npz` / `-o` |
| `verse-convert` | **模型格式转换（Part4K2）** | `--input` / `--output` / `--chat-template` / `--tokenizer` / `--arch` |

完整参数说明见 [`packages/verse_infra/verse_infra/verse_trainer/cli.py`](../packages/verse_infra/verse_infra/verse_trainer/cli.py)。

### 10.1 常用组合

```bash
# 1. 从零预训练 1B 模型（GPU + 混合精度 + 并行 + Loss 优化）
verse-train --config spark/mate/config/cometspark_mate.yml \
    --device cuda --amp --parallel-chunks 4 --loss-optimizer --max-steps 10000

# 2. LoRA 微调
verse-finetune --config spark/mate/config/cometspark_mate.yml \
    --method lora --data data/sft.jsonl --device cuda

# 3. NexRL 后训练
verse-posttrain --config spark/mate/config/cometspark_mate.yml \
    --rl nexrl --data data/rl_prompts.jsonl --device cuda

# 4. 评估 + 打分
verse-eval --config spark/mate/config/cometspark_mate.yml \
    --checkpoint mf_mate/best.vn \
    --prompts-file data/prompts.txt --score --references-file data/refs.txt

# 5. 加载 Qwen tokenizer
verse-tokenize --from-hf Qwen/Qwen3.5-35B-A3B

# 6. 智能分区训练（Part4K2，低内存跑大模型）
verse-train --config spark/mate/config/cometspark_mate.yml \
    --partition-training --partition-size 2 --max-steps 1000

# 7. 持续训练（Part4K2，从 checkpoint 继续追加训练）
python -m verse_infra.verse_trainer.cli verse-continue \
    --checkpoint mf_mate/best.vn --additional-steps 1000 \
    --config spark/mate/config/cometspark_mate.yml --device cuda --amp

# 8. 模型格式互转（Part4K2，.pt ↔ .vn）
verse-convert --input mf_mate/best.vn --output model.vn \
    --chat-template chat_template.jinja --tokenizer tokenizer.json --arch versenex

# 9. 数据集下载（Part4K2，任意 URL + HF + 自动转 .npz）
verse-download --url https://example.com/data.jsonl --to-npz -o data/cached.npz
verse-download --hf wikitext --split train
```

---

## 11. 智能分区训练指南（Part4K2 新增）

智能分区训练（`LayerWiseTrainer`）把模型按 layer 分组训练，训完一组卸载到硬盘 `.vn` 分片，保持统一实体（对外表现为完整模型训练）。适用于**有限内存的 CPU / 单卡 GPU 训练大模型**场景。详见 [ADR-011](architecture/adr-011-layerwise-training.md)。

### 11.1 适用场景

- mate 旗舰（≈1.12B 参数）模型在 8GB 内存 CPU 上训练（全量训练需 12GB+）
- 单卡 GPU 显存不足以容纳完整模型 + 优化器状态
- 希望逐层聚焦训练（底层 / 顶层差异化学习率需求）

### 11.2 基本用法

```python
from verse_torch import LayerWiseTrainer

# model 需有 .blocks 属性（如 VerseNexLM / CometSparkNexLM）
trainer = LayerWiseTrainer(
    model,
    config={
        "lr": 1e-3,
        "weight_decay": 0.01,
        "log_interval": 10,
        "eval_interval": 50,
        "finetune_steps": 20,   # 合并后整体微调 20 步
    },
    partition_size=2,            # 每组 2 个 block
    memory_threshold_mb=512,     # 内存超 512MB 触发卸载
)
train_losses, val_losses = trainer.fit(train_loader, val_loader, max_steps=1000)
```

### 11.3 工作原理

1. **分组**：按 `partition_size` 把 `model.blocks` 分组（embedding / lm_head / norm 始终在内存）
2. **逐组训练**：训练当前组时其他组冻结（`requires_grad=False`），每组训练 `max_steps // n_partitions` 步
3. **卸载**：训完一组用 `VNFileWriter` 写到 `offload_dir/partition_{idx}.vn`（safetensors / npz，无损）
4. **内存监控**：超过 `memory_threshold_mb` 时自动卸载已训练的非当前组
5. **合并**：全部组训练完成后，从硬盘加载所有分片恢复完整模型
6. **可选 fine-tune**：`finetune_steps > 0` 时合并后整体微调（全部参数可训练，`lr * 0.5`）

### 11.4 CLI 用法

```bash
# 智能分区训练（CPU，partition_size=2）
verse-train --config spark/mate/config/cometspark_mate.yml \
    --partition-training --partition-size 2 --max-steps 1000

# 配合 GPU + 混合精度
verse-train --config spark/mate/config/cometspark_mate.yml \
    --partition-training --partition-size 4 --device cuda --amp --max-steps 5000
```

### 11.5 参数调优建议

| 参数 | CPU 8GB | CPU 16GB | GPU 24GB |
|---|---|---|---|
| `partition_size` | 2 | 4 | 8（或不用分区） |
| `memory_threshold_mb` | 512 | 1024 | 4096 |
| `finetune_steps` | 20 | 30 | 50 |

> **提示**：`partition_size` 越小越接近全量训练（精度越高但速度越慢）；`finetune_steps` 用于弥合分组训练的层间边界，建议至少 20 步。

---

## 12. 持续训练指南（Part4K2 新增）

持续训练（`verse-continue`）在训练完成后从 checkpoint 继续追加训练，与 `--resume` 的"中断恢复"语义不同：`--resume` 是从中断点继续同一轮训练，`verse-continue` 是在已完成训练基础上追加新步数。

### 12.1 与 `--resume` 的区别

| 特性 | `--resume` | `verse-continue` |
|---|---|---|
| 语义 | 中断恢复（继续同一轮训练） | 追加训练（已完成基础上加步数） |
| `best_val_loss` | 恢复中断前的值 | 继承之前的值（不从头比较） |
| 适用场景 | 训练意外中断 | 训练完成后想继续提升 |
| 调用方式 | `verse-train --resume` | `python -m verse_infra.verse_trainer.cli verse-continue` |

### 12.2 CLI 用法

```bash
# 从 checkpoint 继续追加 1000 步训练
python -m verse_infra.verse_trainer.cli verse-continue \
    --checkpoint mf_mate/best.vn --additional-steps 1000 \
    --config spark/mate/config/cometspark_mate.yml --device cuda --amp

# CPU 持续训练
python -m verse_infra.verse_trainer.cli verse-continue \
    --checkpoint mf_small/best.vn --additional-steps 500 \
    --config spark/small/config/cometspark_small.yml --device cpu
```

### 12.3 编程接口

```python
from verse_infra.verse_trainer import continue_train

# continue_train 通过 train(continue_from=checkpoint) 实现
continue_train(
    config_path="spark/mate/config/cometspark_mate.yml",
    checkpoint="mf_mate/best.vn",
    additional_steps=1000,
    device="cuda",
    amp=True,
)
```

### 12.4 注意事项

- `--checkpoint` 必须指向有效的 `.pt` 文件（含 `state_dict`）
- 自动继承之前的 `best_val_loss`，新训练中只有更低 val_loss 才会更新 best
- 支持 `--device cuda --amp` GPU 加速
- 持续训练的 checkpoint 仍保存为 `.pt` 格式（可用 `verse-convert` 转 `.vn`）

---

## 13. jinja2 聊天模板使用指南（Part4K2 新增）

Part4K2 将 ChatML 渲染升级为 jinja2 模板优先 + f-string 降级，并新增 Qwen3 官方工具调用格式支持。详见 [ADR-010](architecture/adr-010-jinja2-chat-template.md)。

### 13.1 基础 ChatML 渲染

```python
from verse_infra.verse_tokenizer import render_chat_qwen

# 渲染多轮对话
text = render_chat_qwen([
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！很高兴见到你。"},
])
# 输出：<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！很高兴见到你。<|im_end|>\n

# 推理前缀（add_generation_prompt=True）
text = render_chat_qwen(
    [{"role": "user", "content": "你好"}],
    add_generation_prompt=True,
)
# 输出末尾追加 <|im_start|>assistant\n，等待模型生成
```

### 13.2 工具调用渲染

```python
from verse_infra.verse_tokenizer import render_chat_qwen_with_tools

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}}]

messages = [
    {"role": "user", "content": "北京天气"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
    {"role": "tool", "content": '{"temp": 25}'},
]

out = render_chat_qwen_with_tools(messages, tools=tools)
# assistant 消息渲染为：
# <|im_start|>assistant
# <tool_call>
# {"name": "get_weather", "arguments": {"city": "北京"}}
# </tool_call>
# <|im_end|>
```

### 13.3 解析模型生成的工具调用

```python
from verse_infra.verse_tokenizer import extract_tool_calls_qwen3

# 模型生成的文本
generated = '<tool_call>\n{"name": "search", "arguments": {"q": "天气"}}\n</tool_call>'
calls = extract_tool_calls_qwen3(generated)
# 返回：[{"name": "search", "arguments": {"q": "天气"}}]

# 无工具调用时返回空列表
calls = extract_tool_calls_qwen3("普通回复")  # []
```

### 13.4 jinja2 可选依赖

- **jinja2 可用时**：优先用 `Template(template_str).render(**kwargs)` 渲染
- **jinja2 不可用时**：降级为 f-string 拼接，输出**完全等价**
- 安装：`pip install "jinja2>=3.0"`

> **提示**：jinja2 路径的输出与 f-string 降级路径完全等价（`ensure_ascii=True` + `sort_keys=True` 对齐 `tojson` 行为），无需担心两条路径输出差异。

### 13.5 模板常量

```python
from verse_infra.verse_tokenizer.chat_template import (
    CHATML_TEMPLATE,                  # 基础 ChatML
    CHATML_TEMPLATE_WITH_TOOLS,       # 含 tools 声明
    CHATML_TEMPLATE_WITH_TOOL_CALLS,  # 含 tools 声明 + assistant 工具调用
)
# 这些模板字符串可内嵌到 tokenizer.json 的 chat_template 字段
```

---

## 14. .vn 格式使用指南（Part4K2 新增）

`.vn` 是基于 safetensors 的 ZIP 容器格式，取代传统 pickle `.pt`，提供 mmap 零拷贝 + pickle-free 安全性 + 自描述元数据。详见 [ADR-009](architecture/adr-009-vn-format.md)。

### 14.1 CLI 互转

```bash
# .pt → .vn（可附加 chat_template / tokenizer）
verse-convert --input mf_mate/best.vn --output model.vn \
    --chat-template chat_template.jinja --tokenizer tokenizer.json --arch versenex

# .vn → .pt（无损回转）
verse-convert --input model.vn --output model.pt
```

### 14.2 编程接口：写入

```python
from verse_torch import VNFileWriter

with VNFileWriter("model.vn", arch="versenex", config=cfg_dict) as w:
    w.write_weights(model.state_dict())
    w.write_chat_template(template_str)   # 可选
    w.write_tokenizer("tokenizer.json")   # 可选
```

### 14.3 编程接口：读取

```python
from verse_torch import VNFileReader

with VNFileReader("model.vn") as r:
    meta = r.read_meta()              # {"vn_format_version": 1, "arch": ..., ...}
    cfg = r.read_config()             # 模型配置 dict
    sd = r.read_weights(mmap=True)    # 权重（safetensors mmap 零拷贝）
    tmpl = r.read_chat_template()     # Optional[str]
    tok = r.read_tokenizer()          # Optional[dict]
```

### 14.4 互转函数

```python
from verse_torch import pt_to_vn, vn_to_pt, convert_format

# .pt → .vn
pt_to_vn("model.pt", "model.vn", arch="versenex", config=cfg,
          chat_template=tmpl_str, tokenizer="tokenizer.json")

# .vn → .pt
vn_to_pt("model.vn", "model.pt")

# 自动检测后缀互转
convert_format("model.pt", "model.vn")
convert_format("model.vn", "model.pt")
```

### 14.5 格式结构

```
model.vn (ZIP)
├── model.safetensors   # 权重（safetensors 可用时，mmap 零拷贝）
├── model.npz           # 权重（safetensors 不可用时降级，allow_pickle=False）
├── config.yml          # 模型配置（YAML，优先 PyYAML，否则 JSON 兼容子集）
├── chat_template.jinja # 聊天模板（可选）
├── tokenizer.json      # tokenizer（可选）
└── meta.json           # 元数据（vn_format_version / arch / weight_format / compression_info / created_at）
```

> **提示**：safetensors 不可用时自动降级 npz，纯标准库 + numpy 即可工作。安装 `pip install "safetensors>=0.4"` 启用 mmap 零拷贝。

---

## 15. 数据集下载指南（Part4K2 新增）

`DatasetDownloader` 支持任意 URL + HuggingFace datasets 下载，断点续传 + 多线程 + 自动转 `.npz` 缓存。

### 15.1 CLI 用法

```bash
# 任意 URL 下载（多线程 + 断点续传 + 自动转 .npz）
verse-download --url https://example.com/data.jsonl --to-npz -o data/cached.npz

# HuggingFace datasets 下载
verse-download --hf wikitext --split train
verse-download --hf wikitext --split train --subset wikitext-2-raw-v1
```

### 15.2 编程接口

```python
from verse_infra import DatasetDownloader

dl = DatasetDownloader(cache_dir="data/datasets", num_workers=4)

# 任意 URL 下载 + 自动转 .npz
npz_path = dl.download_and_cache(
    "https://example.com/data.jsonl",
    output_path="data/datasets/cached.npz",
)

# HuggingFace datasets 下载
dir_ = dl.download_hf("wikitext", subset="wikitext-2-raw-v1", split="train")
```

### 15.3 支持的格式转换

下载后自动转为 `.npz`（含 `ids` / `mask` / `seq_len`，与 `CachedDataset` 对齐）：

| 源格式 | 转换说明 |
|---|---|
| `.json` / `.jsonl` | 解析每行 JSON，提取 `text` 字段 |
| `.csv` | 用标准库 `csv` 解析，提取指定列 |
| `.txt` | 每行作为一个样本 |
| `.parquet` | 用 `pyarrow` 解析（需安装 `pyarrow`） |

### 15.4 多线程与断点续传

- **多线程**：文件 ≥ 10MB（`_MULTITHREAD_THRESHOLD`）自动分块多线程下载
- **断点续传**：基于已下载字节数 + `Range` header，从中断点继续
- **缓存**：`cache_dir` 下按 URL hash 命名，重复下载直接命中缓存

> **提示**：HuggingFace datasets 需要 `pip install "datasets>=2.18"` 可选依赖。缺失时 `download_hf` 会提示安装。

---

## 16. spark/run.py 快速训练指南（Part4K2.5 新增）

`spark/run.py` 是基于 VerseTrainer API 封装的命令行快捷入口，提供 7 个子命令（train/eval/generate/chat/compress/convert/download），**所有命令都有合理默认值，最小化用户配置**。与 VerseTrainer CLI（`verse-train` 等）功能等价，但前者零安装可用、子命令更精简、默认值更友好。

路径自举由 `spark/_bootstrap.py` 统一完成（基于 `__file__` 推断，幂等注入 `sys.path`），无需手动设置 `PYTHONPATH`。

### 16.1 快速训练（最小命令）

```bash
# 小配置快速调试（约 10 秒完成，零安装可用）
python spark/run.py train --small

# 1B 正式训练（默认训练后自动评估）
python spark/run.py train

# 覆盖步数 / 设备 / 混合精度
python spark/run.py train --max-steps 1000 --device cuda --amp

# 断点续训
python spark/run.py train --resume
```

### 16.2 全部子命令速查

| 子命令 | 用途 | 示例 |
|---|---|---|
| `train` | 训练模型（训练后默认自动评估） | `python spark/run.py train --small` |
| `eval` | 评估 + 打分 | `python spark/run.py eval --checkpoint mf_mate/best.vn --score` |
| `generate` | 生成文本 | `python spark/run.py generate --prompt "床前明月光，"` |
| `chat` | 交互式聊天 | `python spark/run.py chat --checkpoint mf_mate/best.vn` |
| `compress` | 压缩模型 | `python spark/run.py compress --checkpoint mf_mate/best.vn --method prune,quantize` |
| `convert` | 模型格式互转（`.pt ↔ .vn`） | `python spark/run.py convert --input mf_mate/best.vn --output model.vn` |
| `download` | 下载数据集 | `python spark/run.py download --hf wikitext --split train` |

### 16.3 通用参数

所有子命令均支持：

- `--dry-run`：只打印将要执行的操作而不真正执行，便于在正式运行前核对参数。
- `--config`：指定配置文件路径（不指定时 `train` 用 1B 默认配置，`--small` 用小配置）。
- `--quiet` / `--verbose`：静默模式（仅打印最终结果）/ 详细日志模式。

### 16.4 与 VerseTrainer CLI 的关系

| 维度 | `spark/run.py` | `verse-train` 等 |
|---|---|---|
| 安装 | 零安装，直接 `python spark/run.py` | 需 `pip install -e packages/verse_infra` 注册 console_scripts |
| 子命令数 | 7 个（精简） | 8 个（含 `verse-continue` / `verse-tokenize` / `verse-finetune` / `verse-posttrain`） |
| 默认值 | 友好（`--small` 一键调试、`--eval-after` 默认开） | 偏向显式配置 |
| 适用场景 | 快速试用 / 沙箱验证 / 端到端演示 | 生产训练 / CI / 脚本编排 |

> **提示**：两者底层都调用 `verse_infra.verse_trainer.train()` 等同一套 API，训练结果完全一致，可按场景自由切换。

---

## 17. 训练后自动评估指南（Part4K2.5 新增）

Part4K2.5 Task 4 为 `train()` 新增 `eval_after` 参数（默认 `True`），训练完成后自动调用 `_auto_evaluate` 对 best checkpoint 做 5 指标打分，形成"训练 → 评估"闭环。

### 17.1 工作原理

1. 训练正常完成（或 EarlyStopping 触发）后，加载 `best.pt` checkpoint。
2. 用配置中的默认测试 prompt（或 `eval_config["prompts"]` 自定义）逐条生成。
3. 调用 `ScoringEvaluator` 计算 5 个指标：

| 指标 | 说明 |
|---|---|
| `exact_match` | 精确匹配率（生成与参考完全相等） |
| `prefix_accuracy` | 前缀匹配率（适合续写任务） |
| `char_f1` | 字符级 F1 |
| `bleu` | BLEU-4 简化版 |
| `rouge_l` | ROUGE-L（基于最长公共子序列的 F1） |

4. 评估结果写入 `train()` 返回 dict 的 `eval_result` 字段。

### 17.2 CLI 用法

```bash
# 默认开启自动评估（spark/run.py）
python spark/run.py train --small            # 训练后自动评估

# 跳过自动评估（仅训练）
python spark/run.py train --small --no-eval

# VerseTrainer CLI 同样支持（通过 config 的 training.eval_after 字段控制）
verse-train --config spark/small/config/cometspark_small.yml
```

### 17.3 编程接口

```python
from verse_infra.verse_trainer import train

# 默认 eval_after=True
result = train(
    config_path="spark/small/config/cometspark_small.yml",
    base_dir=".",
)
print(result["best_val_loss"])
print(result.get("eval_result"))   # 自动评估结果（5 指标）

# 显式关闭自动评估
result = train(
    config_path="spark/small/config/cometspark_small.yml",
    eval_after=False,
)

# 自定义评估 prompt
result = train(
    config_path="spark/small/config/cometspark_small.yml",
    eval_config={
        "prompts": [
            {"prompt": "床前明月光，", "reference": "疑是地上霜。"},
            {"prompt": "1+1=", "reference": "2"},
        ],
    },
)
```

### 17.4 注意事项

- 评估失败**不影响训练结果**：`spark/run.py` 内部用 `try/except` 包裹自动评估，失败时仅打印警告。
- 自动评估需要 `best.pt` 已生成（即训练至少完成一次 checkpoint 保存）。
- 5 指标打分需要参考答案；无参考答案时部分指标（exact_match / bleu）会退化为 0，prefix_accuracy / char_f1 / rouge_l 仍可反映生成质量。

---

## 18. 并行训练修复说明（Part4K2.5 新增）

Part4K2.5 Task 6 修复了 `ParallelTrainer`（见 5.6 节）的 4 个健壮性问题。

### 18.1 chunk 间状态重置

**问题**：旧实现中各 chunk 复用同一优化器实例，导致一阶/二阶动量跨 chunk 泄漏，后续 chunk 的训练受前面 chunk 残留动量干扰。

**修复**：`_train_chunk` 内部为每个 chunk 创建全新的优化器实例（`optimizer = optimizer_cls(model.parameters(), lr=self.lr, ...)`），确保优化器状态不跨 chunk 泄漏。同时 chunk 开始时备份模型状态，确保 chunk 训练后状态可追踪。

### 18.2 round_robin 均匀数据分配

**问题**：默认 `sequential` 策略下每个 chunk 都用完整 `train_dataset`，数据无差异化；希望 chunk 间数据不重复时无现成机制。

**修复**：新增 `parallel_strategy` 配置字段（`cfg["parallel_strategy"]`），支持两种策略：

| 策略 | 行为 | 适用场景 |
|---|---|---|
| `sequential`（默认） | 每个 chunk 用完整 train_dataset | 数据量小、希望各 chunk 充分训练 |
| `round_robin` | 数据集按索引轮询分配，chunk 间数据不重复 | 数据量大、希望覆盖更多数据 / 模拟数据并行 |

```bash
# round_robin 策略（CLI）
verse-train --config spark/mate/config/cometspark_mate.yml \
    --parallel-chunks 4 --parallel-strategy round_robin --max-steps 200
```

### 18.3 Phase 2 重训步数为 0 时崩溃

**问题**：当 `chunk_steps` 较小（如 `< 4`）时，Phase 2 重训步数 `chunk_steps // 4` 为 0，导致 `Trainer.fit(max_steps=0)` 崩溃。

**修复**：`chunk_steps < 4` 时跳过 Phase 2 重训（直接进入下一 chunk），并在重训步数计算中保证 `>= 1`。同时 Phase 2 内部创建新优化器，不复用旧优化器状态。

### 18.4 非 tty 环境 tqdm 输出垃圾字符

**问题**：CI / 输出重定向 / 容器日志场景下，stderr 不是 tty，tqdm 仍尝试用 `\r` 刷新进度条，产生大量控制字符污染日志。

**修复**：`_ChunkProgressBar` 仅在「tqdm 可用 + 未静默 + stderr 是 tty」时才启用 tqdm 进度条；否则降级为简洁打印（每完成一个 chunk 打印一行）。`Trainer.fit` 的内层进度条同样在非 tty 时降级为 `log_interval` 打印。

```python
# 编程接口：可手动控制
trainer = ParallelTrainer(
    model, train_ds, val_ds,
    cfg={
        "parallel_chunks": 4,
        "enable_progress_bar": False,   # 强制关闭进度条（CI 推荐）
        "quiet": True,                  # 静默模式
    },
)
```

---

## 19. loss 图表修复说明（Part4K2.5 新增）

Part4K2.5 Task 3 修复了 `plot_loss_curve`（见 5.5 节）的两个可视化问题。

### 19.1 x 轴偏移修复

**问题**：旧实现把 `train_losses`（逐 step 记录）与 `val_losses`（按 `eval_interval` 记录）画在同一 x 轴上时，val 线的 x 坐标未按 `eval_interval` 步长对齐，导致 val 曲线相对 train 曲线偏移，误判训练趋势。

**修复**：`plot_loss_curve` 现在按 `eval_interval` 步长为 val 点取正确 x 坐标（`x_val = [i * eval_interval for i in range(len(val_losses))]`），与 train 曲线的 step 坐标对齐。

### 19.2 ASCII 降级显示 val 线

**问题**：matplotlib 不可用时降级为 ASCII 文本图（80×20 字符画布），旧实现只画 train 线（`T`），val 线丢失，无法对比训练 / 验证 loss。

**修复**：ASCII 降级模式现在同时绘制 train 线（`T`）与 val 线（`V`），两条线在同一字符画布上对齐显示，便于在无 matplotlib 环境（如纯 SSH / 容器）下也能直观对比 loss 趋势。

### 19.3 visualize 统计增强

`verse_torch.visualize`（`visualize.py`）的统计输出增强，训练结束后除了 loss 曲线外，额外打印：

- 最终 train / val loss 与最佳 val loss
- loss 下降率（`compute_loss_rate`）
- 训练总步数与 wall-clock 耗时
- checkpoint 保存路径

---

## 20. VMPC V2.0 压缩指南（Part5K1.1 大升级）

Part5K1.1 将 **VMPC（VerseNext Model Parameters Compression）** 全面升级到 V2.0，**全面抛弃 V1.5 技术栈**。VMPC V2.0 明确 **VMPC ≠ 单纯的量化 / 剪枝**，而是基于三大支柱：**VN 格式文件 + 传统技术 + VSC（VerseNext Space Compression）**。详见 [ADR-013](architecture/adr-013-vmpc-naming-v15.md)。

> **等效能力**：在 VMPC V2.0 支持下，**1zB ≈ 1010B 模型能力**（等效处理）。物理压缩占 40%（量化 + 剪枝），专项算法优化 + 训练补偿占 60%。

### 20.1 统一开关 use_vmpc

所有配置通过 `vmpc.use_vmpc` 单项管理（默认开启）：

- `use_vmpc=true`：强制 `*.vn` 格式 + VSC 引擎压缩 + 训练 / 推理自动接入。
- `use_vmpc=false`：走 legacy 模式（传统技术直通，允许 `.pt`）。

```yaml
vmpc:
  use_vmpc: true              # 默认开启
  # legacy 字段（use_vmpc=false 时生效）
  profile: small
  prune_sparsity: 0.5
  quantize_dtype: ternary
  # V2.0 专属参数（VSC 三维空间压缩）
  target_ratio: 0.06
  quantize_bits: 2
  storage_weight: 0.5
  compute_weight: 0.3
  time_weight: 0.2
  force_vn_format: true
  enable_compensation: true
```

### 20.2 VSC 三维空间压缩引擎

`verse_torch.vsc.VSCEngine` 是独立算法引擎，从**存储 / 算力 / 时间**三维空间角度优化压缩策略：

```python
from verse_torch.vsc import VSCEngine, VSCConfig

# VSC 配置：三维权重
config = VSCConfig(
    target_ratio=0.06,        # 压到原大小 6%
    quantize_bits=2,          # ternary 量化
    storage_weight=0.5,       # 存储维度权重（小模型存储优先）
    compute_weight=0.3,       # 算力维度权重
    time_weight=0.2,          # 时间维度权重
    enable_compensation=True, # 启用训练补偿
)
engine = VSCEngine(config)
```

### 20.3 vmpc_compress 一键压缩

`vmpc_compress(model, use_vmpc=True)` 返回 `(compressed_model, stats)` 元组：

```python
from verse_torch.vmpc import vmpc_compress

# VMPC V2.0 路径（默认）：VSC 引擎 + 强制 .vn
compressed, stats = vmpc_compress(
    model, profile="small", use_vmpc=True,
    compensate_fn=train_fn,   # 可选：训练补偿（恢复压缩损失的能力）
)

# Legacy 路径：传统技术直通（允许 .pt）
compressed, stats = vmpc_compress(model, use_vmpc=False)
```

### 20.4 模型自带 VMPC 接口

`CometSparkSmallLM` / `CometSparkMateLM` 提供 `vmpc_compress_model()` 方法，返回新的模型实例（**不修改原模型**）：

```python
from spark.small.model import CometSparkSmall

model = CometSparkSmall()
new_model = model.vmpc_compress_model(use_vmpc=True)
print(f"压缩比: {model.count_parameters() / new_model.count_parameters():.2f}x")

# 保存（use_vmpc=True 时强制 .vn 格式）
new_model.save("model_vmpc.vn", format="vn")
```

### 20.5 训练补偿

`VMPCV2.compensate()` 通过 `train_fn` 恢复压缩损失的能力（算法优化 60% 的核心环节）：

```python
from verse_torch.vmpc import VMPCV2

vmpc = VMPCV2(config)
# 训练补偿：跑几步恢复能力
result = vmpc.compensate(
    compressed_model,
    train_fn=my_train_fn,     # train_fn(model, data, steps) -> loss_history
    train_data=train_data,
    steps=100,
)
print(f"final_loss={result['final_loss']}, recovered={result['recovered']:.2%}")
```

### 20.6 VN 多空间缓存

`VNCacheManager` 实现内存 LRU + 硬盘 mmap 混合缓存，按优先级自动调度：

- **混合缓存**（内存充足 + 硬盘充足）：优先级高的放内存，优先级低的放硬盘。
- **硬盘优先**（内存不充足时）：主要放硬盘。

为 VMPC V2.0 提供高吞吐、高速度的底层支撑。

### 20.7 spark/run.py compress 命令

```bash
# VMPC V2.0 压缩（默认 + 训练补偿）
python spark/run.py compress --model small --compensate

# Legacy 压缩（传统技术直通）
python spark/run.py compress --model small --no-vmpc --method prune,quantize
```

---

## 21. 双模型训练指南（Part5K1 / Part5K1.1）

Part5K1 正式开始双模型并行，`spark/small/` 与 `spark/mate/` 分别承载 0.06zB 小模型与 0.2zB 旗舰模型。Part5K1.1 精简目录为 `small/` + `mate/` + `src/` 三目录（弃用旧 `config/` / `model/`）。详见 [ADR-014](architecture/adr-014-dual-model-small-mate.md)。

### 21.1 模型定位

| 模型 | 目录 | 能力档位 | VMPC 预设 | 定位 |
|---|---|---|---|---|
| small | `spark/small/` | 0.06zB（≈1010B 等效能力） | ternary + 高稀疏 0.5 | 端侧 / 嵌入式 / 树莓派 |
| mate | `spark/mate/` | 0.2zB（≈2020B 等效能力） | int4 + 中稀疏 0.3 + LoRA + 蒸馏 | 消费级 CPU / 单卡 GPU |

> Part5K1.1 等效能力：在 VMPC V2.0 支持下，1zB ≈ 1010B 模型能力（等效处理）。

### 21.2 CLI 用法（Part5K1.1 自动调用）

Part5K1.1 起 `--model small|mate` 自动调用对应配置与 checkpoint 目录，无需指定配置 / 权重目录：

```bash
# 训练 small 模型（自动用 spark/small/config/cometspark_small.yml，checkpoint 写 mf_small/）
python spark/run.py train --model small

# 训练 mate 模型（自动用 spark/mate/config/cometspark_mate.yml，checkpoint 写 mf_mate/）
python spark/run.py train --model mate

# 生成 / 聊天也自动查找对应目录的最新 checkpoint
python spark/run.py generate --model small --prompt "你好"
python spark/run.py chat --model mate
```

### 21.3 checkpoint 目录

双模型 checkpoint 目录改用 `mf_` 前缀（替代旧 `checkpoints_XXX/`）：

```
mf_small/         # small 模型 checkpoint（.vn 默认格式）
├── best.vn
├── last.vn
└── loss_history.json

mf_mate/          # mate 模型 checkpoint
├── best.vn
├── last.vn
└── loss_history.json
```

### 21.4 模型文件强制 `.vn`（Part5K1.1）

Part5K1.1 起 `use_vmpc=True`（默认）时强制 `*.vn` 格式（不可替换修改）；`.pt` 格式仅在 legacy 模式（`use_vmpc=False`）下允许。`.vn` 格式支持多空间缓存（内存 LRU + 硬盘 mmap 混合调度）。

---

## 22. VMT 三档策略指南（Part5K1 新增）

Part5K1 推出 **VMT（Versenext Memory-aware Training）** 完整三档智能分区训练。详见 [ADR-015](architecture/adr-015-vmt-full-strategy.md)。

### 22.1 三档策略

| 档名 | 行为 | 适用场景 |
|---|---|---|
| `unload` | 卸载到硬盘 `.vn` 分片 | 内存极度受限（LayerWiseTrainer 基础能力） |
| `freeze` | INT4 量化 + `requires_grad=False` | 底层已收敛稳定的层（压缩冻结） |
| `optimize` | 层融合 + 梯度累积 | 顶层高频训练（专项优化） |

### 22.2 vmt_strategy 配置语法

支持两种格式：

**`"auto"`（默认）**：按层位置自动分配——前 1/3 freeze、中间 1/3 optimize、后 1/3 unload。

**显式语法**：`"layers[start:end]=tier, layers[start:end]=tier, ..."`

```python
from verse_torch.layerwise_trainer import VMTTrainer

# auto 策略
trainer = VMTTrainer(
    model, config={"lr": 1e-3},
    vmt_strategy="auto",
    partition_size=2,
)
trainer.fit(train_loader, val_loader, max_steps=1000)

# 显式策略：前 8 层 freeze，中间 48 层 optimize，剩余 unload
trainer = VMTTrainer(
    model, config={"lr": 1e-3, "micro_batch_size": 4},
    vmt_strategy="layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload",
    partition_size=2,
)
trainer.fit(train_loader, val_loader, max_steps=1000)
```

### 22.3 与 LayerWiseTrainer 的关系

`VMTTrainer` 继承 `LayerWiseTrainer`，是后者的超集：

- `LayerWiseTrainer`：保留为简化版（仅 unload 档），向后兼容。
- `VMTTrainer`：支持三档差异化策略，需要按层位置分配时升级。

### 22.4 optimize 档配置

optimize 档走层融合前向 + 梯度累积，config 额外读取 `micro_batch_size`：

```python
config = {
    "lr": 1e-3,
    "micro_batch_size": 4,   # optimize 档微批大小（0=不累积）
}
```

---

## 23. 64+ 层训练加速指南（Part5K1 新增）

Part5K1 针对 64+ 层大模型训练推出层融合 + chunked_forward 加速（Task 7）。

### 23.1 层融合（_fused_forward_blocks）

层融合把多个 VerseNexBlock 的前向融合为单次调用，减少 Python 循环开销：

- **数值等价**：层融合前向与逐块前向数值严格一致（测试覆盖）。
- **性能提升**：减少 Python 层循环开销，64 层模型前向提速约 1.5× ~ 2×。
- **VMT optimize 档集成**：层融合路径由 VMTTrainer 的 optimize 档内部调用。

### 23.2 chunked_forward 自动启用

`n_layer >= 64` 时自动启用 chunked_forward，把长序列按 chunk 分块前向，降低峰值内存：

```python
# 自动启用：构建 n_layer >= 64 的模型时自动走 chunked_forward
# 无需手动配置，模型 forward 内部自动检测 n_layer 并切换路径

# 手动调用层融合前向（VMT optimize 档内部使用）
logits = model._fused_forward_blocks(x, block_indices=range(0, 64))
```

### 23.3 性能预期

| 模型规模 | 逐块前向 | 层融合前向 | 提速 |
|---|---|---|---|
| 32 层 | 基线 | ≈ 1.1× | 轻微（循环开销占比小） |
| 64 层 | 基线 | ≈ 1.5× | 显著 |
| 128 层 | 基线 | ≈ 2.0× | 显著 |

---

## 24. spark/run.py 训练模式补齐（Part5K1 新增）

Part5K1 Task 11 为 `spark/run.py` 补齐 `finetune` / `posttrain` / `continue` 子命令，并统一支持 `--model small|mate` 参数。

### 24.1 子命令速查

| 子命令 | 用途 | 关键参数 |
|---|---|---|
| `train` | 预训练 | `--model small\|mate` / `--small` / `--config` / `--max-steps` / `--device` / `--amp` |
| `finetune` | 微调 | `--model` / `--method lora\|full` / `--checkpoint` / `--data` |
| `posttrain` | 后训练 | `--model` / `--rl nexrl\|sft\|dpo` / `--data` |
| `continue` | 持续训练 | `--model` / `--checkpoint` / `--additional-steps` |
| `eval` | 评估 + 打分 | `--checkpoint` / `--score` / `--references-file` |
| `generate` | 生成文本 | `--prompt` / `--max-tokens` / `--temperature` |
| `chat` | 交互式聊天 | `--checkpoint` / `--max-tokens` |
| `compress` | 压缩模型 | `--checkpoint` / `--method` / `--sparsity` |
| `convert` | 模型格式互转 | `--input` / `--output` |
| `download` | 下载数据集 | `--url` / `--hf` / `--to-npz` |

### 24.2 用法示例

```bash
# 训练 small 模型
python spark/run.py train --model small

# 微调 mate 模型（LoRA）
python spark/run.py finetune --model mate --method lora --checkpoint mf_mate/best.vn --data data/sft.jsonl

# 后训练 small 模型（SFT）
python spark/run.py posttrain --model small --rl sft --data data/sft.jsonl

# 持续训练 mate 模型（从 checkpoint 追加 1000 步）
python spark/run.py continue --model mate --checkpoint mf_mate/best.vn --additional-steps 1000
```

---

## 25. JSONL 自修复指南（Part5K1 新增）

Part5K1 Task 5 推出 `verse_torch.jsonl_repair`，提供 JSONL 文件自动修复与标准化。

### 25.1 常见损坏模式

- **截断的 JSON 行**：文件末尾或中途中断，最后一行 JSON 不完整。
- **未闭合的引号 / 括号**：JSON 字符串或对象 / 数组未正确闭合。
- **多行 JSON 拼接**：多个 JSON 对象被错误地拼接到同一行。

### 25.2 用法

```python
from verse_torch.jsonl_repair import repair_jsonl

# 修复损坏的 JSONL 文件，输出标准 JSONL
repair_jsonl("data/train_corrupted.jsonl", "data/train.jsonl")

# 修复后可直接用于训练
from verse_infra.verse_trainer.data import CachedDataset
dataset = CachedDataset(tokenizer, "data/train.jsonl", seq_len=64)
```

### 25.3 val.json 自动生成

Part5K1 Task 6 在训练数据准备阶段自动从 train.jsonl 切分验证集：

```bash
# 训练时若 val.jsonl 不存在，自动从 train.jsonl 切分（默认 10%）
python spark/run.py train --model small
```

---

## 26. vnn 重命名迁移指南（Part5K1 新增）

Part5K1 将 `verse_torch.nn` 重命名为 `verse_torch.vnn`（BREAKING）。详见 [ADR-016](architecture/adr-016-nn-to-vnn-rename.md)。

### 26.1 导入路径迁移

```python
# 推荐（新）
from verse_torch.vnn import Module, Linear, Embedding, VerseNexLM

# 兼容（仍可工作，已废弃）
from verse_torch.nn import Module, Linear, Embedding
from verse_torch import nn   # nn 是 vnn 的别名
```

### 26.2 transformer 系旧名迁移（BREAKING）

```python
# 旧（Part5K1 起抛 ImportError）
# from verse_torch.nn import TransformerLM       # ❌ ImportError
# from verse_torch.nn import TransformerBlock    # ❌ ImportError
# from verse_torch.nn import GQASelfAttention    # ❌ ImportError

# 新（VerseNex 命名）
from verse_torch.vnn import VerseNexLM           # ✅
from verse_torch.vnn import VerseNexBlock        # ✅
from verse_torch.vnn import VerseNexAttention    # ✅
```

### 26.3 行为等价性

`vnn` 与 `nn`（shim）是同一模块对象，所有类与函数完全等价：

```python
from verse_torch import vnn, nn
assert vnn.Module is nn.Module       # True
assert vnn.Linear is nn.Linear       # True
assert vnn.VerseNexLM is nn.VerseNexLM  # True
```

---

## 相关文档

- [VerseTorch README](../packages/verse_torch/README.md) —— Tensor / nn / autograd / GPU-NPU 后端基础
- [VerseNex README](../packages/verse_nex/README.md) —— VerseNexLM / TriSparse / MoD / NexRL 架构
- [VerseInfra README](../packages/verse_infra/README.md) —— 总包结构 + 导入路径迁移指南（Part4K1）
- [CometSpark 双模型 README](../spark/README.md) —— mate 旗舰（≈1.12B 参数）模型说明 + 配置 + 训练 CLI
- [压缩管线设计](../verse_data/designs/compression_pipeline_design.md) —— 剪枝 / 量化 / LoRA / 蒸馏的完整设计论证
- [CPU 并行 ADR](architecture/adr-004-cpu-parallel.md) —— multiprocessing 并行决策
- [GPU/NPU 后端 ADR](architecture/adr-005-gpu-npu-backend.md) —— Part4K1 设备抽象层决策
- [VerseInfra 聚合 ADR](architecture/adr-006-verse-infra-aggregation.md) —— Part4K1 总包聚合决策
- [NexRL 设计 ADR](architecture/adr-007-nexrl-design.md) —— Part4K1 强化学习设计决策
- [超稀疏并行注意力 ADR](architecture/adr-008-parallel-sparse-attention.md) —— Part4K1 三层并行加速决策
- [.vn 文件格式 ADR](architecture/adr-009-vn-format.md) —— Part4K2 模型交付格式决策
- [jinja2 聊天模板 ADR](architecture/adr-010-jinja2-chat-template.md) —— Part4K2 ChatML 模板决策
- [智能分区训练 ADR](architecture/adr-011-layerwise-training.md) —— Part4K2 大模型低内存训练决策
- [压缩技术 V1.3 ADR](architecture/adr-012-compression-v13.md) —— Part4K2 三重损失蒸馏决策
- [压缩 PoC 基准](benchmarks/compression_poc.md) —— 1M 参数模型压缩实测数据
- [性能调优指南](performance_tuning.md) —— CPU BLAS / numba / GPU 加速 / 混合精度 / CachedDataset
- [主 README](../README.md)
