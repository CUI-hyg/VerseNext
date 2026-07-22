"""verse_tokenizer 兼容 shim —— 已迁入 verse_infra.verse_tokenizer。

旧路径仍可用（带 DeprecationWarning），推荐改用::

    from verse_infra.verse_tokenizer import BPETokenizer
"""

import os as _os
import sys as _sys
import warnings

# 路径自举：确保 verse_infra 可被导入
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PACKAGES_DIR = _os.path.dirname(_os.path.dirname(_THIS_DIR))  # → packages/
_VERSE_INFRA_PATH = _os.path.join(_PACKAGES_DIR, "verse_infra")
if _os.path.isdir(_VERSE_INFRA_PATH) and _VERSE_INFRA_PATH not in _sys.path:
    _sys.path.insert(0, _VERSE_INFRA_PATH)

warnings.warn(
    "verse_tokenizer 已迁入 verse_infra.verse_tokenizer，"
    "请改用 from verse_infra.verse_tokenizer import ...",
    DeprecationWarning,
    stacklevel=2,
)

from verse_infra.verse_tokenizer import *  # noqa: F401,E402,F403
from verse_infra.verse_tokenizer import (  # noqa: F401,E402  显式重导出常用 API
    BaseTokenizer, BPETokenizer, CharTokenizer, ByteTokenizer,
    WordPieceTokenizer, SentencePieceUnigramTokenizer, VerseTokenizer,
    QwenTokenizer, load_tokenizer, NexTokenizerWrapper,
    nfkc_normalize, pre_tokenize, trim_to_utf8_boundary,
    trim_byte_ids_to_utf8_boundary, render_chat, render_prompt,
    split_prompt_completion, SpecialTokens, QWEN_IM_START, QWEN_IM_END,
    QWEN_ENDOFTEXT, render_chat_qwen, render_prompt_qwen,
    split_prompt_completion_qwen,
)
from verse_infra.verse_tokenizer import __version__  # noqa: F401,E402


def __getattr__(name):
    """延迟重导出子模块（如 verse_tokenizer.verse → verse_infra.verse_tokenizer.verse）。"""
    try:
        _mod = __import__(f"verse_infra.verse_tokenizer.{name}", fromlist=[name])
        globals()[name] = _mod
        return _mod
    except ImportError:
        raise AttributeError(f"module 'verse_tokenizer' has no attribute {name!r}")
