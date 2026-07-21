"""VerseTokenizer: Lightweight BPE/Unigram tokenizer (no heavy deps).

提供：
- ``BaseTokenizer``: 抽象基类，定义 encode/decode/save/load/__len__/apply_chat_template 接口契约
- ``BPETokenizer``: 最小 BPE 分词器，可加载 HuggingFace tokenizer.json，支持 train/save/load
- ``CharTokenizer``: 字符级 fallback 分词器（无依赖、无 merges）
- ``ByteTokenizer``: 字节级 tokenizer（vocab_size=259，含 bos/eos/pad/unk）
- ``SentencePieceUnigramTokenizer``: SentencePiece Unigram 分词器（EM 训练 + Viterbi 解码）
- ``QwenTokenizer``: Qwen3 系列 tokenizer 包装器（lazy-import transformers）
- ``load_tokenizer``: 工厂函数，根据 kind 加载不同 tokenizer

Task 2 新增导出：
- 预处理：``nfkc_normalize`` / ``pre_tokenize`` / ``trim_to_utf8_boundary``
- Chat 模板：``render_chat`` / ``render_prompt`` / ``split_prompt_completion``
- Unigram：``SentencePieceUnigramTokenizer`` / ``SpecialTokens``

QwenTokenizer 新增导出：
- ``QwenTokenizer``：包装 transformers.AutoTokenizer（lazy import）
- Qwen3 ChatML：``render_chat_qwen`` / ``render_prompt_qwen`` / ``split_prompt_completion_qwen``
- Qwen3 特殊 token 常量：``QWEN_IM_START`` / ``QWEN_IM_END`` / ``QWEN_ENDOFTEXT``
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
    # Qwen3 ChatML
    QWEN_IM_START,
    QWEN_IM_END,
    QWEN_ENDOFTEXT,
    render_chat_qwen,
    render_prompt_qwen,
    split_prompt_completion_qwen,
)
from .unigram import (
    SentencePieceUnigramTokenizer,
    SpecialTokens,
)

# 注意：QwenTokenizer 采用 lazy import transformers，模块导入本身不依赖
# transformers。这里 import qwen 模块不会触发 transformers 加载——只有
# 真正调用 QwenTokenizer() 构造函数时才会触发。
from .qwen import QwenTokenizer

__version__ = "0.2.0"

__all__ = [
    # 基础 tokenizer
    "BaseTokenizer",
    "BPETokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "SentencePieceUnigramTokenizer",
    "QwenTokenizer",
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
    # Qwen3 ChatML（QwenTokenizer）
    "QWEN_IM_START",
    "QWEN_IM_END",
    "QWEN_ENDOFTEXT",
    "render_chat_qwen",
    "render_prompt_qwen",
    "split_prompt_completion_qwen",
]
