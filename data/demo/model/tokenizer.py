"""CometSpark tokenizer 工厂：build_tokenizer / load_tokenizer。

封装 verse_tokenizer 包，统一 PoC 阶段的 tokenizer 构建与加载流程。
"""

from __future__ import annotations

import os

from verse_tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer as _vload


def build_tokenizer(
    corpus_path: str,
    vocab_size: int,
    save_path: str,
    kind: str = "bpe",
) -> str:
    """从语料构建 tokenizer 并保存到 save_path。

    Args:
        corpus_path: 训练语料文件路径（纯文本，UTF-8）
        vocab_size: 目标词表大小（仅 kind="bpe" 时生效）
        save_path: 保存路径
        kind: "bpe" 调 BPETokenizer.train，"byte" 直接构造 ByteTokenizer

    Returns:
        实际保存路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)

    if kind == "bpe":
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = f.read()
        tok = BPETokenizer.train(corpus, vocab_size=int(vocab_size))
        tok.save(save_path)
        return save_path

    if kind == "byte":
        # ByteTokenizer 固定 vocab_size=259，vocab_size 参数忽略
        tok = ByteTokenizer()
        tok.save(save_path)
        return save_path

    raise ValueError(f"Unknown tokenizer kind: {kind!r} (expected 'bpe' or 'byte')")


def load_tokenizer(path: str, kind: str = "bpe"):
    """加载已保存的 tokenizer。

    Args:
        path: tokenizer 文件路径
        kind: "bpe" 或 "byte"，与 build_tokenizer 对应

    Returns:
        tokenizer 对象，统一接口：
            - ``encode(text)`` → List[int]
            - ``decode(ids)`` → str
            - ``__len__()`` → vocab_size
    """
    return _vload(kind=kind, path=path)


__all__ = ["build_tokenizer", "load_tokenizer"]
