"""Task 5.5: 乱码修复测试。

验证：
1. ``ByteTokenizer.decode`` 对截断字节序列不产生 U+FFFD（乱码）
2. ``BPETokenizer.decode`` 对截断字节序列不产生 U+FFFD
3. ``_trim_to_utf8_boundary`` 工具函数正确处理各种 UTF-8 边界情况
4. 完整文本 decode 不丢失字符

参考 Gpt_teacher-3.37M-cn 处理方法：解码前丢弃末尾不完整的多字节序列，
避免 ``errors="replace"`` 把不完整字节替换为 U+FFFD。

运行方式：
    cd /workspace && python -m pytest tests/test_no_garbled.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_tokenizer"))

from verse_tokenizer import BPETokenizer, ByteTokenizer
from verse_tokenizer.bpe import _trim_to_utf8_boundary


# 中文 "你" = 0xE4 0xBD 0xA0（3 字节 UTF-8 字符）
YOU_BYTES = [0xE4, 0xBD, 0xA0]
# 中文 "好" = 0xE5 0xA5 0xBD（3 字节 UTF-8 字符）
HAO_BYTES = [0xE5, 0xA5, 0xBD]
# Emoji 🎉 = 0xF0 0x9F 0x8E 0x89（4 字节 UTF-8 字符）
EMOJI_BYTES = [0xF0, 0x9F, 0x8E, 0x89]


# ---------------------------------------------------------------------------
# 测试 1: _trim_to_utf8_boundary 工具函数
# ---------------------------------------------------------------------------


def test_trim_empty():
    """空字节序列返回空。"""
    assert _trim_to_utf8_boundary([]) == []


def test_trim_ascii_complete():
    """ASCII 字节不受影响。"""
    assert _trim_to_utf8_boundary([0x41, 0x42, 0x43]) == [0x41, 0x42, 0x43]


def test_trim_chinese_complete():
    """完整的中文字节不受影响。"""
    assert _trim_to_utf8_boundary(YOU_BYTES) == YOU_BYTES


def test_trim_chinese_truncated_last_byte():
    """中文 "你" 截断最后一个字节（0xE4 0xBD）应丢弃整个字符。"""
    truncated = YOU_BYTES[:-1]  # [0xE4, 0xBD]
    result = _trim_to_utf8_boundary(truncated)
    assert result == [], f"截断字节未丢弃：{truncated} → {result}"


def test_trim_chinese_truncated_first_byte_only():
    """中文 "你" 只剩首字节（0xE4）应丢弃。"""
    result = _trim_to_utf8_boundary([0xE4])
    assert result == []


def test_trim_emoji_complete():
    """完整的 emoji 字节不受影响。"""
    assert _trim_to_utf8_boundary(EMOJI_BYTES) == EMOJI_BYTES


def test_trim_emoji_truncated():
    """Emoji 截断（前 3 字节）应丢弃整个字符。"""
    truncated = EMOJI_BYTES[:-1]  # [0xF0, 0x9F, 0x8E]
    result = _trim_to_utf8_boundary(truncated)
    assert result == []


def test_trim_mixed_complete_and_truncated():
    """混合完整字符 + 末尾截断字符：保留完整部分，丢弃截断部分。"""
    # "A你" = 0x41 0xE4 0xBD 0xA0，末尾再加截断的 0xE5
    mixed = [0x41] + YOU_BYTES + [0xE5]
    result = _trim_to_utf8_boundary(mixed)
    # 末尾的 0xE5 是 3 字节字符的首字节，但缺少后续，应丢弃
    assert result == [0x41] + YOU_BYTES


def test_trim_all_continuation_bytes():
    """全是 continuation bytes（10xxxxxx）应全部丢弃。"""
    all_cont = [0x80, 0x81, 0x82]
    result = _trim_to_utf8_boundary(all_cont)
    assert result == []


def test_trim_two_byte_char_complete():
    """2 字节字符（如 ¢ = 0xC2 0xA2）完整保留。"""
    # ¢ = U+00A2 = 0xC2 0xA2
    assert _trim_to_utf8_boundary([0xC2, 0xA2]) == [0xC2, 0xA2]


def test_trim_two_byte_char_truncated():
    """2 字节字符截断（只剩 0xC2）应丢弃。"""
    assert _trim_to_utf8_boundary([0xC2]) == []


# ---------------------------------------------------------------------------
# 测试 2: ByteTokenizer.decode 不产生 U+FFFD
# ---------------------------------------------------------------------------


def test_byte_decode_truncated_chinese_no_replacement():
    """ByteTokenizer.decode 截断的中文不产生 U+FFFD。"""
    tok = ByteTokenizer()
    # "你" 截断为 [0xE4, 0xBD]
    truncated = YOU_BYTES[:-1]
    decoded = tok.decode(truncated)
    assert "\ufffd" not in decoded, (
        f"decode 产生 U+FFFD：{truncated} → {decoded!r}"
    )


def test_byte_decode_truncated_emoji_no_replacement():
    """ByteTokenizer.decode 截断的 emoji 不产生 U+FFFD。"""
    tok = ByteTokenizer()
    # 🎉 截断为 [0xF0, 0x9F, 0x8E]
    truncated = EMOJI_BYTES[:-1]
    decoded = tok.decode(truncated)
    assert "\ufffd" not in decoded, (
        f"decode 产生 U+FFFD：{truncated} → {decoded!r}"
    )


def test_byte_decode_complete_chinese_no_loss():
    """ByteTokenizer.decode 完整中文不丢失字符。"""
    tok = ByteTokenizer()
    text = "你好世界"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"完整文本 decode 丢失字符：{text!r} → {decoded!r}"
    assert "\ufffd" not in decoded


def test_byte_decode_mixed_complete_and_truncated():
    """ByteTokenizer.decode 混合完整 + 截断字符：保留完整部分。"""
    tok = ByteTokenizer()
    # "A你" + 截断的 0xE5
    text = "A你"
    ids = tok.encode(text) + [0xE5]  # 末尾加一个截断字节
    decoded = tok.decode(ids)
    assert "\ufffd" not in decoded
    # "A你" 应保留
    assert "A" in decoded
    assert "你" in decoded


def test_byte_decode_empty_ids():
    """ByteTokenizer.decode 空序列返回空字符串。"""
    tok = ByteTokenizer()
    assert tok.decode([]) == ""


def test_byte_decode_strip_special_false_truncated():
    """strip_special=False 时截断也不产生 U+FFFD。"""
    tok = ByteTokenizer()
    # bos + "你"截断 + eos
    ids = [tok.bos_id] + YOU_BYTES[:-1] + [tok.eos_id]
    decoded = tok.decode(ids, strip_special=False)
    assert "\ufffd" not in decoded


def test_byte_decode_only_special_tokens():
    """只有 special tokens 的序列 decode 不产生 U+FFFD。"""
    tok = ByteTokenizer()
    ids = [tok.bos_id, tok.eos_id, tok.pad_id]
    decoded = tok.decode(ids, strip_special=False)
    assert "\ufffd" not in decoded


def test_byte_decode_truncated_first_byte_only():
    """只截断到首字节（0xE4）也不产生 U+FFFD。"""
    tok = ByteTokenizer()
    decoded = tok.decode([0xE4])
    assert "\ufffd" not in decoded


# ---------------------------------------------------------------------------
# 测试 3: BPETokenizer.decode 不产生 U+FFFD
# ---------------------------------------------------------------------------


def test_bpe_decode_complete_text_no_replacement():
    """BPETokenizer.decode 完整文本不产生 U+FFFD。"""
    # 用空 vocab 的 BPETokenizer，encode 会 fallback 到字符级
    tok = BPETokenizer({}, [], byte_level=True)
    text = "你好世界"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids)
    assert "\ufffd" not in decoded, (
        f"BPE decode 完整文本产生 U+FFFD：{text!r} → {decoded!r}"
    )


def test_bpe_decode_truncated_chinese_no_replacement():
    """BPETokenizer.decode 截断的中文不产生 U+FFFD。

    构造方式：encode 完整中文得到 byte-level id 列表，然后截断末尾 id。
    """
    tok = BPETokenizer({}, [], byte_level=True)
    text = "你好"
    full_ids = tok.encode(text, add_special_tokens=False)
    # 截断最后一个 id（对应 "好" 的最后一个字节）
    truncated_ids = full_ids[:-1]
    decoded = tok.decode(truncated_ids)
    assert "\ufffd" not in decoded, (
        f"BPE decode 截断中文产生 U+FFFD：{truncated_ids} → {decoded!r}"
    )


def test_bpe_decode_truncated_emoji_no_replacement():
    """BPETokenizer.decode 截断的 emoji 不产生 U+FFFD。"""
    tok = BPETokenizer({}, [], byte_level=True)
    text = "🎉"
    full_ids = tok.encode(text, add_special_tokens=False)
    # 截断最后一个字节
    truncated_ids = full_ids[:-1]
    decoded = tok.decode(truncated_ids)
    assert "\ufffd" not in decoded


def test_bpe_decode_mixed_complete_and_truncated():
    """BPETokenizer.decode 混合完整 + 截断字符：保留完整部分。"""
    tok = BPETokenizer({}, [], byte_level=True)
    text = "A你"
    full_ids = tok.encode(text, add_special_tokens=False)
    # 末尾加一个截断字节（"好" 的首字节 0xE5）
    # 注意：BPE encode 后的 id 是 byte-level 字符的 id，不是字节本身
    # 所以我们需要用 byte-level 字符的 id
    from verse_tokenizer.bpe import _BYTE_ENCODER
    truncated_byte_char = _BYTE_ENCODER[0xE5]
    truncated_id = tok.vocab.get(truncated_byte_char)
    if truncated_id is not None:
        mixed_ids = full_ids + [truncated_id]
        decoded = tok.decode(mixed_ids)
        assert "\ufffd" not in decoded
        assert "A" in decoded
        assert "你" in decoded


# ---------------------------------------------------------------------------
# 测试 4: BPE train 后的 tokenizer decode 不产生 U+FFFD
# ---------------------------------------------------------------------------


def test_bpe_trained_decode_no_replacement():
    """训练后的 BPETokenizer decode 中文不产生 U+FFFD。"""
    corpus = [
        "Hello world! This is a test corpus for BPE training.",
        "你好世界！这是一个用于 BPE 训练的测试语料。",
        "Machine learning is fun. 机器学习很有趣。",
    ]
    tok = BPETokenizer.train(corpus, vocab_size=300)
    test_texts = [
        "Hello world",
        "你好世界",
        "Mixed 中英 text 123",
        "机器学习",
    ]
    for text in test_texts:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert "\ufffd" not in decoded, (
            f"训练后 BPE decode 产生 U+FFFD：{text!r} → {decoded!r}"
        )


def test_bpe_trained_decode_truncated_no_replacement():
    """训练后的 BPETokenizer decode 截断序列不产生 U+FFFD。"""
    corpus = [
        "你好世界！这是一个用于 BPE 训练的测试语料。",
        "Machine learning is fun. 机器学习很有趣。",
    ]
    tok = BPETokenizer.train(corpus, vocab_size=300)
    text = "你好世界"
    full_ids = tok.encode(text, add_special_tokens=False)
    # 截断最后一个 id
    truncated_ids = full_ids[:-1]
    decoded = tok.decode(truncated_ids)
    assert "\ufffd" not in decoded


# ---------------------------------------------------------------------------
# 测试 5: prompt + 生成边界不产生 U+FFFD
# ---------------------------------------------------------------------------


def test_prompt_generation_boundary_no_replacement():
    """prompt + 生成拼接边界不产生 U+FFFD（Task 5.4 分别 decode 策略）。

    模拟 evaluate.py 的场景：
    - prompt = "你好"（原始文本）
    - 生成部分 = "世界" + 末尾截断字节（max_new_tokens 限制导致末尾不完整）
    - 用 Task 5.4 的策略：prompt 用原始文本，生成部分单独 decode
    """
    tok = ByteTokenizer()
    prompt = "你好"
    prompt_ids = tok.encode(prompt)
    # 生成部分 = "世界" 的字节 + 末尾截断的 "你" 首字节（0xE4）
    generated_ids = list("世界".encode("utf-8")) + [0xE4]
    full_ids = prompt_ids + generated_ids

    # Task 5.4 策略：prompt 用原始文本，生成部分单独 decode
    n_prompt = len(prompt_ids)
    generated_only = full_ids[n_prompt:]
    decoded_generated = tok.decode(generated_only)
    full_text = prompt + decoded_generated

    # 生成部分的 decode 不应产生 U+FFFD（_trim_to_utf8_boundary 处理末尾截断）
    assert "\ufffd" not in decoded_generated, (
        f"生成部分 decode 产生 U+FFFD：{generated_only} → {decoded_generated!r}"
    )
    # prompt 部分应完整保留
    assert full_text.startswith("你好")
    # "世界" 应保留在生成部分
    assert "世界" in decoded_generated


def test_full_decode_truncated_tail_no_replacement():
    """完整序列末尾截断（max_new_tokens 限制）decode 不产生 U+FFFD。

    场景：prompt + 生成拼接后末尾被截断（生成部分的最后一个字符不完整）。
    _trim_to_utf8_boundary 会丢弃末尾不完整字符。
    """
    tok = ByteTokenizer()
    # "你好世界" + 末尾截断的 0xE5（"好" 的首字节，但作为末尾不完整字符）
    full_ids = list("你好世界".encode("utf-8")) + [0xE5]
    decoded = tok.decode(full_ids)
    assert "\ufffd" not in decoded, (
        f"末尾截断 decode 产生 U+FFFD：{full_ids} → {decoded!r}"
    )
    # "你好世界" 应完整保留
    assert "你好世界" in decoded


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
