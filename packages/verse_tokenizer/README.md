# VerseTokenizer

> 中文定位：轻量分词器，支持 BPE / Byte / Char / VerseTokenizer（Qwen 风格优化）四种模式；无 `tokenizers` / `sentencepiece` 重型依赖时仍可运行，可加载 HuggingFace `tokenizer.json`。

[返回主 README](../../README.md)

## 特性

- 四种 tokenizer：`BPETokenizer` / `ByteTokenizer` / `CharTokenizer` / `VerseTokenizer`。
- BPE 训练：`BPETokenizer.train(corpus, vocab_size)` 字节级 merge，GPT-2 风格预切分。
- 持久化：`save(path)` / `load(path)` JSON 格式。
- HuggingFace 兼容：可从 `tokenizer.json` 加载（`from_file` / `from_hf`）。
- 特殊 token：`<bos>` / `<eos>` / `<pad>` / `<unk>` 自动管理。
- 零重型依赖：`tokenizers` 包可选（仅在 `kind="hf"` 时使用）。

### Part4 新增能力（VerseTokenizer）

- **VerseTokenizer**：原 QwenTokenizer 改名而来，针对 Qwen 系列模型 tokenizer 做了 9 项深度优化：
  1. **高效 BPE merge**：基于优先级队列的 merge 算法，O(N·V) 复杂度
  2. **特殊 token 管理**：`<|endoftext|>` / `<|im_start|>` / `<|im_end|>` 等自动注册
  3. **UTF-8 边界修复**：`trim_to_utf8_boundary`，防 `errors="replace"` 产生 U+FFFD 乱码
  4. **chat template**：`apply_chat_template` / `apply_prompt_template`，与 `<|im_start|>system\n...<|im_end|>` 格式对齐
  5. **NFKC 归一化**：`nfkc_normalize`，统一全角/半角、组合字符
  6. **GPT-4 风格预分词**：中文整字独立成块，数字与英文连续
  7. **vocab 自适应**：训练数据不足时自动回退到更小 vocab
  8. **add_special_tokens 开关**：`encode` 时可选是否添加 bos/eos
  9. **持久化兼容**：可加载 Qwen 系列 `tokenizer.json`，也可保存为自有格式
- **替代旧版 tokenizer.json**：CometSpark-V0.2 默认使用 VerseTokenizer，不再依赖旧版 `tokenizer.json`。

### Part3K2 新增能力

- **GPT-4 风格正则预分词**（中文整字独立成块）+ **NFKC 归一化**：`preprocess.pre_tokenize` / `nfkc_normalize`。
- **Chat template 系统**：`render_chat` / `render_prompt` / `split_prompt_completion`，所有 tokenizer 统一接入 `apply_chat_template` / `apply_prompt_template`。
- **SentencePiece Unigram 分词器**：`SentencePieceUnigramTokenizer`，EM 训练 + Viterbi 解码。
- **UTF-8 边界修复**：`trim_to_utf8_boundary`，防 `errors="replace"` 产生 U+FFFD 乱码。
- **`add_special_tokens` 编码开关**：BPE / Byte / Char / Unigram 统一支持，控制 `encode` 时是否加 bos/eos。
- **BPETokenizer 升级**：GPT-4 正则预分词、`vocab_size` 自适应（数据不足回退）、特殊 token 注册（旧风格 4 + 新风格 7）、`apply_chat_template` / `apply_prompt_template` 方法。
- **ByteTokenizer / CharTokenizer 升级**：统一 NFKC + byte-aligned decode（无 U+FFFD 乱码）、`apply_chat_template` / `apply_prompt_template` 方法。

## 安装

```bash
pip install -e packages/verse_tokenizer
```

## 快速开始

### ByteTokenizer（最简单，PoC 推荐）

```python
from verse_tokenizer import ByteTokenizer

tok = ByteTokenizer()                          # vocab_size=259
ids = tok.encode("你好，verse", add_bos=True, add_eos=True)
print("ids:", ids)                              # [256, ...字节..., 257]
text = tok.decode(ids)                          # strip_special=True 默认丢弃 special
print("roundtrip:", text)                       # "你好，verse"
```

### BPETokenizer 训练 + 使用

```python
from verse_tokenizer import BPETokenizer

corpus = ["verse is a lightweight tokenizer", "hello world"] * 50
tok = BPETokenizer.train(corpus, vocab_size=300)   # 字节级 BPE
print("vocab_size:", len(tok))                     # 300（含 special tokens）

ids = tok.encode("verse hello", add_special_tokens=True)
print("ids:", ids)
text = tok.decode(ids)
print("roundtrip:", text)

tok.save("/tmp/verse_bpe.json")                    # 持久化
tok2 = BPETokenizer.load("/tmp/verse_bpe.json")     # 重新加载
assert tok2.encode("verse hello") == ids           # 与原 ids 一致
```

### load_tokenizer 工厂

```python
from verse_tokenizer import load_tokenizer

# 三种 kind：'byte' / 'bpe' / 'hf'
tok_byte = load_tokenizer(kind="byte")                 # ByteTokenizer（最简）
tok_bpe = load_tokenizer(kind="bpe", path="/tmp/verse_bpe.json")  # 加载本地 BPE
tok_hf = load_tokenizer(kind="hf", path="tokenizer.json")         # HF 兼容（无 tokenizers 则降级 Byte）

# 统一接口：encode / decode
ids = tok_byte.encode("hello verse")
print(tok_byte.decode(ids))
```

## API 详解

### BPETokenizer

| API | 说明 |
| --- | --- |
| `BPETokenizer.train(corpus, vocab_size)` | 类方法，从语料训练字节级 BPE，自动追加 `<bos>/<eos>/<pad>/<unk>` |
| `encode(text, add_special_tokens=True)` | 编码为 `List[int]`，特殊 token 视为 atomic |
| `decode(ids)` | 还原为字符串，byte-level 反查 |
| `add_special_tokens(tokens)` | 把 tokens 加入 vocab 并标记为 special |
| `save(path)` / `load(path)` | JSON 持久化（与 `from_file` 兼容） |
| `from_file(path)` | 从 HuggingFace `tokenizer.json` 加载 |
| `from_hf(repo_id, revision)` | 从 HF repo 下载 `tokenizer.json` 后加载 |
| `apply_chat_template(messages)` | **Part3K2** 渲染 chat 数组并编码（不加 bos/eos，`render_chat` 已含 `<|eos|>`） |
| `apply_prompt_template(prompt)` | **Part3K2** 渲染 prompt 并编码（推理前缀） |

**Part3K2 升级要点**：
- **GPT-4 风格正则预分词**：`train` 与 `encode` 内部统一调用 `preprocess.pre_tokenize`，中文整字、英文单词、数字、标点、空白分别独立成块。
- **`vocab_size` 自适应**：训练时若数据不足以达到目标 `vocab_size`，自动回退到最大可达 vocab 大小（不会无限循环）。
- **特殊 token 注册**：`train` 后自动 `add_special_tokens(DEFAULT_SPECIAL_TOKENS)`（11 个）：
  - 旧风格 4 个：`<bos>` / `<eos>` / `<pad>` / `<unk>`
  - 新风格 7 个：`<|bos|>` / `<|eos|>` / `<|pad|>` / `<|unk|>` / `<|user|>` / `<|assistant|>` / `<|system|>`
- **`encode` 编码开关**：`encode(text, add_special_tokens=True/False)`，`add_special_tokens=None` 时用构造参数 `add_special_tokens`（默认 `True`）作为默认值；`True` 时在首尾加 `<bos>` / `<eos>`。
- **UTF-8 字节边界检查**：`train` 时只接受合并后字节序列为合法 UTF-8 的 merge（避免 decode 时产生 U+FFFD 乱码）。

```python
from verse_tokenizer import BPETokenizer

corpus = ["床前明月光，疑是地上霜", "hello world 123", "verse tokenizer"] * 30
tok = BPETokenizer.train(corpus, vocab_size=400)
print("vocab_size:", len(tok))         # 自适应：数据不足时可能 < 400

# add_special_tokens 编码开关
ids_with = tok.encode("hello", add_special_tokens=True)    # 含 <bos>/<eos>
ids_without = tok.encode("hello", add_special_tokens=False)  # 不含

# chat template
messages = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}]
chat_ids = tok.apply_chat_template(messages)   # 渲染并编码
prompt_ids = tok.apply_prompt_template("你好")  # 推理前缀
```

### preprocess — 预处理（Part3K2）

`preprocess` 子模块提供 NFKC 归一化、GPT-4 风格正则预分词、UTF-8 边界修复，被 BPE / Byte / Char / Unigram 分词器复用。

| 函数 | 说明 |
| --- | --- |
| `nfkc_normalize(text)` | NFKC 归一化：全角字母数字 → 半角、组合字符 → 规范形式、兼容字符分解。等价于 `unicodedata.normalize("NFKC", text)` |
| `pre_tokenize(text)` | GPT-4 风格正则预分词：先 NFKC 归一化，再按 中文整字 / 英文单词 / 数字 / 标点 / 空白 / other 切分，返回 piece 列表（拼接后等于归一化原文，不丢字符） |
| `trim_to_utf8_boundary(bytes_data)` | UTF-8 边界修复：从字节序列末尾向前修剪不完整的多字节 UTF-8 字符，防 `errors="replace"` 产生 U+FFFD 乱码 |
| `trim_byte_ids_to_utf8_boundary(byte_ids)` | 同上，`list[int]` 版本（兼容旧 API） |

预分词分组顺序：`han`（CJK 基本汉字 `\u4e00-\u9fff`）→ `word`（ASCII 字母连续）→ `num`（ASCII 数字连续）→ `punct`（标点/符号）→ `space`（连续空白）→ `other`（兜底，如非 ASCII 字母 é、下划线 _）。

```python
from verse_tokenizer import pre_tokenize, nfkc_normalize, trim_to_utf8_boundary

# 预分词：中文整字独立成块，英文单词/数字整体保留
assert pre_tokenize("床前明月光") == ["床", "前", "明", "月", "光"]
assert pre_tokenize("床前明月光hello123") == ["床", "前", "明", "月", "光", "hello", "123"]

# NFKC 归一化：全角→半角
assert nfkc_normalize("ＡＢＣ１２３") == "ABC123"

# UTF-8 边界修复：中文 "你" = 0xE4 0xBD 0xA0（3 字节）
assert trim_to_utf8_boundary(b"\xe4\xbd\xa0") == b"\xe4\xbd\xa0"   # 完整保留
assert trim_to_utf8_boundary(b"\xe4\xbd") == b""                    # 截断 → 丢弃整个字符
assert trim_to_utf8_boundary(b"A\xe4\xbd\xa0") == b"A\xe4\xbd\xa0"  # ASCII + 完整中文
```

### chat_template — 对话模板（Part3K2）

`chat_template` 子模块把 chat 数组 / prompt 字符串转为可编码的渲染字符串，约定特殊 token 字符串：`<|user|>` / `<|assistant|>` / `<|system|>` / `<|eos|>` / `<|bos|>` / `<|pad|>` / `<|unk|>`。

| 函数 | 说明 |
| --- | --- |
| `render_chat(messages)` | 渲染 chat 数组为 `<|user|>{content}<|assistant|>{content}<|eos|>` 拼接字符串；`role` 不在 user/assistant/system 时按字面值渲染 |
| `render_prompt(prompt)` | 渲染 prompt 为推理前缀 `<|user|>{prompt}<|assistant|>`，模型从此处开始生成 |
| `split_prompt_completion(rendered)` | 拆分渲染后的字符串为 `(prompt_part, completion_part)`：找最后一个 `<|assistant|>` 位置，其前（含 marker）为 prompt，其后为 completion。用于 loss mask（prompt 部分屏蔽 `ignore_index=-100`，completion 部分参与 loss） |

```python
from verse_tokenizer import render_chat, render_prompt, split_prompt_completion

# chat 渲染
messages = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}]
assert render_chat(messages) == "<|user|>你好<|assistant|>你好！<|eos|>"

# prompt 渲染（推理前缀）
assert render_prompt("你好") == "<|user|>你好<|assistant|>"

# 拆分 prompt / completion（用于 loss mask）
prompt_part, completion_part = split_prompt_completion(
    "<|user|>你好<|assistant|>你好！<|eos|>"
)
assert prompt_part == "<|user|>你好<|assistant|>"
assert completion_part == "你好！<|eos|>"
```

### SentencePieceUnigramTokenizer — Unigram 分词器（Part3K2）

`SentencePieceUnigramTokenizer` 基于 unigram 语言模型：每个 piece 有一个概率（log 形式），编码时用 Viterbi 算法找概率最大的分割。接口与 `BPETokenizer` 对齐。

| API | 说明 |
| --- | --- |
| `SentencePieceUnigramTokenizer(vocab_size=1000, special_tokens=None, add_special_tokens=True)` | 构造器。`special_tokens` 默认为 `SpecialTokens`（含 `<|bos|>` / `<|eos|>` / `<|pad|>` / `<|unk|>` / `<|user|>` / `<|assistant|>` / `<|system|>`），占用 id 0~N-1 |
| `train(corpus, vocab_size=None)` | EM 训练：预分词 → 初始频次（所有前缀）→ 5 轮 EM（E 步 Viterbi 找最优分割，M 步重新估计概率）→ 保留 top-K vocab_size 个 piece（特殊 token 必留）。返回 `self`（链式） |
| `encode(text, add_special_tokens=None)` | Viterbi 解码：找概率最大的 piece 分割，转为 id 序列。`add_special_tokens=None` 时用 `self.add_special_tokens`；先按特殊 token 切分，普通 chunk 再 pre_tokenize + Viterbi |
| `decode(ids)` | 解码 id 序列：特殊 token（bos/eos/pad/unk/user/assistant/system）不输出，其余 piece 直接拼接 |
| `apply_chat_template(messages)` | 渲染 chat 数组并编码（不加 bos/eos，`render_chat` 已含 `<|eos|>`） |
| `apply_prompt_template(prompt)` | 渲染 prompt 并编码（推理前缀，不加 bos/eos） |
| `save(path)` / `load(path)` | JSON 持久化 |
| `__len__()` | 返回 vocab 大小 |

```python
from verse_tokenizer import SentencePieceUnigramTokenizer

corpus = ["床前明月光，疑是地上霜", "举头望明月，低头思故乡",
          "hello world", "verse tokenizer is lightweight"] * 20

tok = SentencePieceUnigramTokenizer(vocab_size=500)
tok.train(corpus, vocab_size=500)
print("vocab_size:", len(tok))

# Viterbi 解码
ids = tok.encode("床前明月光", add_special_tokens=True)   # 首尾加 <|bos|> / <|eos|>
text = tok.decode(ids)                                     # 还原字符串

# chat template
messages = [{"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"}]
chat_ids = tok.apply_chat_template(messages)
prompt_ids = tok.apply_prompt_template("你好")
```

### ByteTokenizer

- `vocab_size = 259`（256 字节 + `<bos>`/`<eos>`/`<pad>`/`<unk>`）。
- `encode(text, add_bos=False, add_eos=False)` → `List[int]`：UTF-8 字节序列。
- `decode(ids, strip_special=True)` → `str`：还原字符串；`strip_special=False` 时保留 special token 文本。
- `save(path)` / `load(path)`：JSON 持久化。

**Part3K2 升级要点**：
- **统一 NFKC + byte-aligned decode**：`encode` 内部 NFKC 归一化，`decode` 用 `trim_to_utf8_boundary` 修剪不完整字节，无 U+FFFD 乱码。
- **`add_special_tokens` 编码开关**：`encode` 兼容新旧两套 API：
  - 旧：`encode(text, add_bos=True, add_eos=True)`
  - 新：`encode(text, add_special_tokens=True)`（同时控制 bos 和 eos，优先级高于 `add_bos` / `add_eos`）
  - `add_special_tokens=None` 时用构造参数 `add_special_tokens`（默认 `False`，保持旧 API 行为）
- **`apply_chat_template(messages)` / `apply_prompt_template(prompt)`**：渲染并编码（不加 bos/eos，因为 chat template 已含 `<|eos|>`）。

```python
from verse_tokenizer import ByteTokenizer

tok = ByteTokenizer()                     # add_special_tokens 默认 False（旧 API 行为）
ids = tok.encode("你好，verse", add_special_tokens=True)   # 新 API：同时加 bos+eos
text = tok.decode(ids)                    # 无 U+FFFD 乱码

# chat template
chat_ids = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
)
```

### CharTokenizer

- 字符级 fallback：每个 unicode 字符对应一个 id。
- `id 0..3` 保留给 `<pad>` / `<unk>` / `<bos>` / `<eos>`。
- 首次 `encode` 时按需动态扩充 vocab。
- 无 merges、无依赖；当 BPE / HF 加载失败时使用。

**Part3K2 升级要点**：
- **统一 NFKC 归一化**（继承 `BaseTokenizer._preprocess`）。
- **`add_special_tokens` 编码开关**：构造参数 `add_special_tokens`（默认 `True`）控制 `encode` 时是否加 `<eos>`；`encode(text, add_special_tokens=None)` 时用默认值。
- **`apply_chat_template(messages)` / `apply_prompt_template(prompt)`**：继承自 `BaseTokenizer`，渲染并编码（不加 bos/eos）。

### load_tokenizer(kind, path)

| `kind` | 行为 |
| --- | --- |
| `"byte"` | 返回 `ByteTokenizer`，`path` 可选（用于加载已保存的配置） |
| `"bpe"` | 调用 `BPETokenizer.load(path)`；无 `path` 返回空 `BPETokenizer` |
| `"hf"` | 优先用 `tokenizers` 包加载 HF `tokenizer.json`，失败降级到 `ByteTokenizer` |

## 测试

- `tests/test_tokenizer.py`：17 项测试，覆盖 BPE train/encode/decode、Byte 往返、Char fallback、save/load 等。
- `tests/test_tokenizer_upgrade.py`：**Part3K2** 14 项测试，覆盖：
  - `pre_tokenize` 中文/混合切分、`nfkc_normalize` 全角转半角、`trim_to_utf8_boundary` 边界修复
  - `render_chat` / `render_prompt` / `split_prompt_completion`
  - BPE 中文训练、`add_special_tokens` 开关、Byte `apply_chat_template`、Byte 无乱码
  - Unigram train/encode/decode、Unigram `apply_chat_template`、特殊 token 注册

运行：

```bash
python -m pytest tests/test_tokenizer.py tests/test_tokenizer_upgrade.py -v
```

## 相关文档

- [CometSpark 使用 ByteTokenizer](../../data/demo/model/tokenizer.py)
