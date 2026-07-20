# VerseTokenizer

> 中文定位：轻量分词器，支持 BPE / Byte / Char 三种模式；无 `tokenizers` / `sentencepiece` 重型依赖时仍可运行，可加载 HuggingFace `tokenizer.json`。

[返回主 README](../../README.md)

## 特性

- 三种 tokenizer：`BPETokenizer` / `ByteTokenizer` / `CharTokenizer`。
- BPE 训练：`BPETokenizer.train(corpus, vocab_size)` 字节级 merge，GPT-2 风格预切分。
- 持久化：`save(path)` / `load(path)` JSON 格式。
- HuggingFace 兼容：可从 `tokenizer.json` 加载（`from_file` / `from_hf`）。
- 特殊 token：`<bos>` / `<eos>` / `<pad>` / `<unk>` 自动管理。
- 零重型依赖：`tokenizers` 包可选（仅在 `kind="hf"` 时使用）。

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

### ByteTokenizer

- `vocab_size = 259`（256 字节 + `<bos>`/`<eos>`/`<pad>`/`<unk>`）。
- `encode(text, add_bos=False, add_eos=False)` → `List[int]`：UTF-8 字节序列。
- `decode(ids, strip_special=True)` → `str`：还原字符串；`strip_special=False` 时保留 special token 文本。
- `save(path)` / `load(path)`：JSON 持久化。

### CharTokenizer

- 字符级 fallback：每个 unicode 字符对应一个 id。
- `id 0..3` 保留给 `<pad>` / `<unk>` / `<bos>` / `<eos>`。
- 首次 `encode` 时按需动态扩充 vocab。
- 无 merges、无依赖；当 BPE / HF 加载失败时使用。

### load_tokenizer(kind, path)

| `kind` | 行为 |
| --- | --- |
| `"byte"` | 返回 `ByteTokenizer`，`path` 可选（用于加载已保存的配置） |
| `"bpe"` | 调用 `BPETokenizer.load(path)`；无 `path` 返回空 `BPETokenizer` |
| `"hf"` | 优先用 `tokenizers` 包加载 HF `tokenizer.json`，失败降级到 `ByteTokenizer` |

## 测试

- `tests/test_tokenizer.py`：17 项测试，覆盖 BPE train/encode/decode、Byte 往返、Char fallback、save/load 等。

## 相关文档

- [CometSpark 使用 ByteTokenizer](../../data/demo/model/tokenizer.py)
