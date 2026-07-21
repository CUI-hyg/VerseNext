"""VerseTokenizer: Lightweight BPE/Unigram tokenizer (no heavy deps).

提供：
- ``BaseTokenizer``: 抽象基类，定义 encode/decode/save/load/__len__/apply_chat_template 接口契约
- ``BPETokenizer``: 最小 BPE 分词器，可加载 HuggingFace tokenizer.json，支持 train/save/load
- ``CharTokenizer``: 字符级 fallback 分词器（无依赖、无 merges）
- ``ByteTokenizer``: 字节级 tokenizer（vocab_size=259，含 bos/eos/pad/unk）
- ``SentencePieceUnigramTokenizer``: SentencePiece Unigram 分词器（EM 训练 + Viterbi 解码）
- ``load_tokenizer``: 工厂函数，根据 kind 加载不同 tokenizer

Task 2 新增导出：
- 预处理：``nfkc_normalize`` / ``pre_tokenize`` / ``trim_to_utf8_boundary``
- Chat 模板：``render_chat`` / ``render_prompt`` / ``split_prompt_completion``
- Unigram：``SentencePieceUnigramTokenizer`` / ``SpecialTokens``
"""

from .bpe import (
    BaseTokenizer,
    BPETokenizer,
    CharTokenizer,
    ByteTokenizer,
    load_tokenizer,
)
from .preprocess import (
    nfkc_normalize,
    pre_tokenize,
    trim_to_utf8_boundary,
    trim_byte_ids_to_utf8_boundary,
)
from .chat_template import (
    render_chat,
    render_prompt,
    split_prompt_completion,
)
from .unigram import (
    SentencePieceUnigramTokenizer,
    SpecialTokens,
)

__version__ = "0.2.0"

__all__ = [
    # 基础 tokenizer
    "BaseTokenizer",
    "BPETokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "SentencePieceUnigramTokenizer",
    "load_tokenizer",
    # 预处理（Task 2.1）
    "nfkc_normalize",
    "pre_tokenize",
    "trim_to_utf8_boundary",
    "trim_byte_ids_to_utf8_boundary",
    # Chat 模板（Task 2.3）
    "render_chat",
    "render_prompt",
    "split_prompt_completion",
    # Unigram 特殊 token（Task 2.4）
    "SpecialTokens",
]
