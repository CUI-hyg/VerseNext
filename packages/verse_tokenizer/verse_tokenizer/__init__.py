"""VerseTokenizer: Lightweight BPE/Unigram tokenizer (no heavy deps).

提供：
- ``BPETokenizer``: 最小 BPE 分词器，可加载 HuggingFace tokenizer.json
- ``CharTokenizer``: 字符级 fallback 分词器（无依赖、无 merges）
"""

from .bpe import BPETokenizer, CharTokenizer

__version__ = "0.1.0"

__all__ = ["BPETokenizer", "CharTokenizer"]
