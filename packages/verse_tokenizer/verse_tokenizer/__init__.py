"""VerseTokenizer: Lightweight BPE/Unigram tokenizer (no heavy deps).

提供：
- ``BaseTokenizer``: 抽象基类，定义 encode/decode/save/load/__len__ 接口契约
- ``BPETokenizer``: 最小 BPE 分词器，可加载 HuggingFace tokenizer.json，支持 train/save/load
- ``CharTokenizer``: 字符级 fallback 分词器（无依赖、无 merges）
- ``ByteTokenizer``: 字节级 tokenizer（vocab_size=259，含 bos/eos/pad/unk）
- ``load_tokenizer``: 工厂函数，根据 kind 加载不同 tokenizer
"""

from .bpe import (
    BaseTokenizer,
    BPETokenizer,
    CharTokenizer,
    ByteTokenizer,
    load_tokenizer,
)

__version__ = "0.1.0"

__all__ = [
    "BaseTokenizer",
    "BPETokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "load_tokenizer",
]
