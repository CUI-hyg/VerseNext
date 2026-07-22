"""Tokenizer 预处理：NFKC 归一化 + GPT 风格正则预分词 + UTF-8 边界修复。

设计目标
--------
- 统一 NFKC 归一化入口（全角→半角、组合→规范、兼容字符分解）；
- GPT-4 风格正则预分词：中文整字、英文单词、数字、标点、空白分别独立成块；
- ``trim_to_utf8_boundary``：从字节序列末尾向前修剪不完整的多字节 UTF-8 字符，
  避免 ``errors="replace"`` 产生 U+FFFD 乱码。

被以下模块复用：
- :mod:`verse_tokenizer.bpe` 的 ``BPETokenizer`` / ``ByteTokenizer`` / ``CharTokenizer``
- :mod:`verse_tokenizer.unigram` 的 ``SentencePieceUnigramTokenizer``
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# NFKC 归一化
# ---------------------------------------------------------------------------


def nfkc_normalize(text: str) -> str:
    """NFKC 归一化。

    - 全角字母数字 → 半角；
    - 组合字符 → 规范形式；
    - 兼容字符分解。

    与 :func:`unicodedata.normalize` 的 ``"NFKC"`` 等价，封装为独立函数
    便于在项目中统一调用入口（Task 7.4 要求 NFKC 实现合并到此模块）。
    """
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# GPT-4 风格正则预分词
# ---------------------------------------------------------------------------
# 分组顺序很重要：alternation 按从左到右匹配，先匹配 han 再匹配 word/num/punct/space。
# - han：CJK 基本汉字（\u4e00-\u9fff）整字独立成块
# - word：ASCII 字母连续作为整体
# - num：ASCII 数字连续作为整体
# - punct：非空白、非 \w、非汉字的字符（标点 / 符号）独立成块
# - space：连续空白作为整体
# - other：兜底，匹配任何剩余字符（如非 ASCII 字母 é、下划线 _ 等），
#   确保不会漏字符导致信息丢失
_GPT4_PATTERN = (
    r"(?P<han>[\u4e00-\u9fff])"
    r"|(?P<word>[A-Za-z]+)"
    r"|(?P<num>[0-9]+)"
    r"|(?P<punct>[^\s\w\u4e00-\u9fff])"
    r"|(?P<space>\s+)"
    r"|(?P<other>.)"
)
_GPT4_RE = re.compile(_GPT4_PATTERN, re.UNICODE)


def pre_tokenize(text: str) -> list[str]:
    """GPT 风格预分词：返回 piece 列表。

    中文整字、英文单词、数字、标点、空白分别独立成块。
    先做 NFKC 归一化（统一入口），再用 GPT-4 风格正则切分。

    Args:
        text: 原始文本

    Returns:
        piece 字符串列表，拼接后等于 NFKC 归一化后的原文（不丢失字符）

    Examples:
        >>> pre_tokenize("床前明月光")
        ['床', '前', '明', '月', '光']
        >>> pre_tokenize("你好hello123世界")
        ['你', '好', 'hello', '123', '世', '界']
    """
    text = nfkc_normalize(text)
    pieces: list[str] = []
    for m in _GPT4_RE.finditer(text):
        pieces.append(m.group(0))
    return pieces


# ---------------------------------------------------------------------------
# UTF-8 边界修复
# ---------------------------------------------------------------------------


def trim_to_utf8_boundary(bytes_data: bytes) -> bytes:
    """修剪字节序列到 UTF-8 边界（防止 U+FFFD 乱码）。

    从末尾向前检查最后一个 UTF-8 字符是否完整：
    - 先跳过所有末尾的 continuation bytes（10xxxxxx）找到首字节；
    - 根据首字节判断该字符需要的总字节数（1/2/3/4）；
    - 若末尾字符字节不完整，则丢弃该字符的所有字节。

    UTF-8 编码规则：
        - 单字节字符：``0xxxxxxx``
        - 双字节字符首字节：``110xxxxx``，需后续 1 字节
        - 三字节字符首字节：``1110xxxx``，需后续 2 字节
        - 四字节字符首字节：``11110xxx``，需后续 3 字节
        - 后续字节形如 ``10xxxxxx``

    Args:
        bytes_data: 可能末尾被截断的字节序列

    Returns:
        对齐到完整 UTF-8 字符边界的字节序列（前缀，不产生 U+FFFD）

    Examples:
        >>> # 中文 "你" = 0xE4 0xBD 0xA0（3 字节）
        >>> trim_to_utf8_boundary(b"\xe4\xbd\xa0")
        b'\xe4\xbd\xa0'
        >>> # 截断最后一个字节 → 丢弃整个字符
        >>> trim_to_utf8_boundary(b"\xe4\xbd")
        b''
        >>> # 完整 ASCII + 完整中文 → 全部保留
        >>> trim_to_utf8_boundary(b"A\xe4\xbd\xa0")
        b'A\xe4\xbd\xa0'
    """
    if not bytes_data:
        return bytes_data

    n = len(bytes_data)
    last = bytes_data[-1]
    # ASCII（单字节字符）必然完整
    if last < 0x80:
        return bytes_data

    # 从末尾向前找到当前字符的首字节（跳过 continuation bytes 10xxxxxx）
    i = n - 1
    while i >= 0 and (bytes_data[i] & 0xC0) == 0x80:
        i -= 1
    if i < 0:
        # 全是 continuation bytes，无法解码，全部丢弃
        return b""

    first = bytes_data[i]
    if first < 0x80:
        expected = 1
    elif first < 0xE0:
        expected = 2
    elif first < 0xF0:
        expected = 3
    else:
        expected = 4

    actual = n - i
    if actual < expected:
        # 末尾字符字节不完整，丢弃该字符的所有字节
        return bytes_data[:i]
    return bytes_data


def trim_byte_ids_to_utf8_boundary(byte_ids: list[int]) -> list[int]:
    """``trim_to_utf8_boundary`` 的 list[int] 版本（兼容旧 API）。

    与 :func:`trim_to_utf8_boundary` 相同逻辑，但接受 / 返回 ``list[int]``
    以便与 :mod:`verse_tokenizer.bpe` 中已有的 ``_trim_to_utf8_boundary`` 互换。
    """
    if not byte_ids:
        return byte_ids
    n = len(byte_ids)
    last = byte_ids[-1]
    if last < 0x80:
        return byte_ids
    i = n - 1
    while i >= 0 and (byte_ids[i] & 0xC0) == 0x80:
        i -= 1
    if i < 0:
        return []
    first = byte_ids[i]
    if first < 0x80:
        expected = 1
    elif first < 0xE0:
        expected = 2
    elif first < 0xF0:
        expected = 3
    else:
        expected = 4
    actual = n - i
    if actual < expected:
        return byte_ids[:i]
    return byte_ids


__all__ = [
    "nfkc_normalize",
    "pre_tokenize",
    "trim_to_utf8_boundary",
    "trim_byte_ids_to_utf8_boundary",
]
