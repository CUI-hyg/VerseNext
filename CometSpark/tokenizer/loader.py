"""tokenizer.json 加载适配器。

CometSpark 的 ``tokenizer/tokenizer.json`` 为 HuggingFace tokenizers 风格
（BPE 词表 + merges + added_tokens 特殊 token）。VerseNext 框架自带的
``BPETokenizer`` 使用同源 GPT-2 byte-level 编码，因此只需做一次格式转换
即可复用其 encode/decode 能力。
"""

from __future__ import annotations

import json
from pathlib import Path

from verse_tokenizer.bpe import BPETokenizer

TOKENIZER_DIR = Path(__file__).resolve().parent
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"

ENDOFTEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def load_tokenizer(path: str | Path = TOKENIZER_JSON) -> BPETokenizer:
    """从 HF 格式 ``tokenizer.json`` 构建 :class:`BPETokenizer`。

    Args:
        path: tokenizer.json 路径（默认为本目录下的 tokenizer.json）。

    Returns:
        可直接用于 encode/decode 的 BPETokenizer 实例。
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"不支持的分词器类型: {model.get('type')}")

    vocab: dict[str, int] = dict(model["vocab"])
    # HF merges 为 "a b" 字符串（或 ["a", "b"] 列表），统一转为 (a, b) 元组
    merges: list[tuple[str, str]] = [
        tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
        for m in model["merges"]
    ]
    # added_tokens 为特殊 token（<|endoftext|>、<|im_start|> 等）
    special_tokens: dict[str, int] = {
        t["content"]: t["id"] for t in payload.get("added_tokens", [])
    }

    tok = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    tok._tokenizer_dir = str(path.parent)
    return tok


def load_endoftext_id(tok: BPETokenizer) -> int | None:
    """返回 <|endoftext|> 的 token id（文档分隔符 / 结束符）。"""
    return tok.special_tokens.get(ENDOFTEXT)
