"""Task 2.8: Tokenizer 全面升级测试。

覆盖 Task 2 的 8 个子任务：
1. ``pre_tokenize``：GPT-4 风格正则预分词（中文整字、英文单词、数字、标点、空白独立成块）
2. ``nfkc_normalize``：NFKC 归一化（全角→半角、组合→规范）
3. ``trim_to_utf8_boundary``：UTF-8 边界修复（丢弃末尾不完整多字节序列）
4. ``render_chat`` / ``render_prompt`` / ``split_prompt_completion``：Chat 模板渲染
5. ``BPETokenizer``：接入 GPT-4 预分词 + vocab_size 自适应 + 特殊 token 注册 + add_special_tokens 开关
6. ``ByteTokenizer``：apply_chat_template + 无 U+FFFD 乱码
7. ``SentencePieceUnigramTokenizer``：EM 训练 + Viterbi 解码 + apply_chat_template
8. 特殊 token 注册验证（旧风格 <bos> 等 + 新风格 <|user|> 等）

运行方式：
    cd /workspace && python -m pytest tests/test_tokenizer_upgrade.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_infra.verse_tokenizer import (
    BPETokenizer,
    ByteTokenizer,
    CharTokenizer,
    SentencePieceUnigramTokenizer,
    SpecialTokens,
    nfkc_normalize,
    pre_tokenize,
    render_chat,
    render_prompt,
    split_prompt_completion,
    trim_to_utf8_boundary,
)


# ---------------------------------------------------------------------------
# 测试 1: pre_tokenize 中文整字独立成块
# ---------------------------------------------------------------------------


def test_pre_tokenize_chinese():
    """GPT-4 风格预分词：中文整字独立成块。"""
    pieces = pre_tokenize("床前明月光")
    assert pieces == ["床", "前", "明", "月", "光"], (
        f"中文整字预分词失败：{pieces}"
    )


# ---------------------------------------------------------------------------
# 测试 2: pre_tokenize 中英数字混合
# ---------------------------------------------------------------------------


def test_pre_tokenize_mixed():
    """GPT-4 风格预分词：中英数字混合，各类型独立成块。"""
    pieces = pre_tokenize("床前明月光hello123世界")
    # 中文整字、英文单词、数字、中文整字
    assert pieces == ["床", "前", "明", "月", "光", "hello", "123", "世", "界"], (
        f"中英数字混合预分词失败：{pieces}"
    )


# ---------------------------------------------------------------------------
# 测试 3: NFKC 归一化（全角→半角）
# ---------------------------------------------------------------------------


def test_nfkc_normalize_fullwidth():
    """NFKC 归一化：全角字母数字 → 半角。"""
    # 全角字母 Ａ(FF21) → A(0041)，ｚ(FF5A) → z(007A)
    assert nfkc_normalize("Ａｚ") == "Az"
    # 全角数字 １(FF11) → 1(0031)，９(FF19) → 9(0039)
    assert nfkc_normalize("１９") == "19"
    # 组合字符 e + ◌́ → é
    assert nfkc_normalize("e\u0301") == "\u00e9"


# ---------------------------------------------------------------------------
# 测试 4: trim_to_utf8_boundary UTF-8 边界修复
# ---------------------------------------------------------------------------


def test_trim_to_utf8_boundary():
    """UTF-8 边界修复：丢弃末尾不完整的多字节序列。"""
    # 中文 "你" = 0xE4 0xBD 0xA0（3 字节）
    # 完整保留
    assert trim_to_utf8_boundary(b"\xe4\xbd\xa0") == b"\xe4\xbd\xa0"
    # 截断最后一个字节 → 丢弃整个字符
    assert trim_to_utf8_boundary(b"\xe4\xbd") == b""
    # ASCII + 完整中文 → 全部保留
    assert trim_to_utf8_boundary(b"A\xe4\xbd\xa0") == b"A\xe4\xbd\xa0"
    # 空字节序列
    assert trim_to_utf8_boundary(b"") == b""


# ---------------------------------------------------------------------------
# 测试 5: render_chat 渲染 chat 数组
# ---------------------------------------------------------------------------


def test_render_chat():
    """render_chat：把 chat 数组渲染为 <|user|>...<|assistant|>...<|eos|> 格式。"""
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    rendered = render_chat(messages)
    expected = "<|user|>你好<|assistant|>你好！<|eos|>"
    assert rendered == expected, f"render_chat 失败：{rendered!r}"


# ---------------------------------------------------------------------------
# 测试 6: render_prompt 渲染推理前缀
# ---------------------------------------------------------------------------


def test_render_prompt():
    """render_prompt：把 prompt 渲染为 <|user|>{prompt}<|assistant|> 前缀。"""
    rendered = render_prompt("你好")
    expected = "<|user|>你好<|assistant|>"
    assert rendered == expected, f"render_prompt 失败：{rendered!r}"


# ---------------------------------------------------------------------------
# 测试 7: split_prompt_completion 拆分 prompt/completion
# ---------------------------------------------------------------------------


def test_split_prompt_completion():
    """split_prompt_completion：以最后一个 <|assistant|> 为界拆分。"""
    rendered = "<|user|>你好<|assistant|>你好！<|eos|>"
    prompt_part, completion_part = split_prompt_completion(rendered)
    assert prompt_part == "<|user|>你好<|assistant|>", (
        f"prompt_part 错误：{prompt_part!r}"
    )
    assert completion_part == "你好！<|eos|>", (
        f"completion_part 错误：{completion_part!r}"
    )


# ---------------------------------------------------------------------------
# 测试 8: BPE 中文训练 + 编解码往返
# ---------------------------------------------------------------------------


def test_bpe_train_chinese():
    """BPE 训练中文语料：vocab_size 自适应，encode/decode 往返无 U+FFFD。"""
    corpus = [
        "床前明月光，疑是地上霜。",
        "举头望明月，低头思故乡。",
        "春眠不觉晓，处处闻啼鸟。",
        "夜来风雨声，花落知多少。",
    ]
    tok = BPETokenizer.train(corpus, vocab_size=300)
    # vocab_size 自适应：不超过目标 300
    assert len(tok.vocab) <= 300, f"vocab 超过 300：{len(tok.vocab)}"
    # 至少有 256 个基础字节字符
    assert len(tok.vocab) >= 256, f"vocab 小于 256：{len(tok.vocab)}"
    # 中文编解码往返不产生 U+FFFD
    for text in ["床前明月光", "春眠不觉晓", "你好世界"]:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert "\ufffd" not in decoded, (
            f"BPE 中文 decode 产生 U+FFFD：{text!r} → {decoded!r}"
        )


# ---------------------------------------------------------------------------
# 测试 9: BPE add_special_tokens 编码开关
# ---------------------------------------------------------------------------


def test_bpe_add_special_tokens_switch():
    """add_special_tokens 开关：True 时首尾加 <bos>/<eos>，False 时不加。"""
    corpus = ["Hello world", "你好世界", "Machine learning"]
    tok = BPETokenizer.train(corpus, vocab_size=200)
    bos_id = tok.vocab.get("<bos>")
    eos_id = tok.vocab.get("<eos>")
    assert bos_id is not None, "<bos> 未注册到 vocab"
    assert eos_id is not None, "<eos> 未注册到 vocab"

    text = "Hello"
    # add_special_tokens=True：首尾加 <bos>/<eos>
    ids_with = tok.encode(text, add_special_tokens=True)
    assert ids_with[0] == bos_id, f"首 id 不是 <bos>：{ids_with[0]} != {bos_id}"
    assert ids_with[-1] == eos_id, f"尾 id 不是 <eos>：{ids_with[-1]} != {eos_id}"

    # add_special_tokens=False：不加 <bos>/<eos>
    ids_without = tok.encode(text, add_special_tokens=False)
    assert ids_without[0] != bos_id, f"首 id 不应是 <bos>：{ids_without[0]}"
    assert ids_without[-1] != eos_id, f"尾 id 不应是 <eos>：{ids_without[-1]}"


# ---------------------------------------------------------------------------
# 测试 10: ByteTokenizer apply_chat_template
# ---------------------------------------------------------------------------


def test_byte_tokenizer_apply_chat_template():
    """ByteTokenizer.apply_chat_template：渲染 chat 数组并编码。"""
    tok = ByteTokenizer()
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    ids = tok.apply_chat_template(messages)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    # 解码后应包含原始内容（special token 字符串被 strip_special=True 丢弃，
    # 但 <|user|> 等字符串的字节会被保留）
    # 注意：preprocess 会做 NFKC 归一化，全角 ！ → 半角 !
    decoded = tok.decode(ids)
    assert "你好" in decoded
    # NFKC 把全角 ！ 转为半角 !，所以检查 "你好!" 而非 "你好！"
    assert "你好!" in decoded or "你好！" in decoded, (
        f"assistant 内容未保留：{decoded!r}"
    )
    # 应包含 chat 模板标记的字节
    assert "<|user|>" in decoded
    assert "<|assistant|>" in decoded


# ---------------------------------------------------------------------------
# 测试 11: ByteTokenizer 无 U+FFFD 乱码
# ---------------------------------------------------------------------------


def test_byte_tokenizer_no_garbled():
    """ByteTokenizer：截断字节序列 decode 不产生 U+FFFD。"""
    tok = ByteTokenizer()
    # 中文 "你" = 0xE4 0xBD 0xA0，截断最后一个字节
    truncated = [0xE4, 0xBD]
    decoded = tok.decode(truncated)
    assert "\ufffd" not in decoded, (
        f"截断字节 decode 产生 U+FFFD：{truncated} → {decoded!r}"
    )
    # 完整中文往返不丢失
    text = "你好世界"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"完整文本往返丢失：{text!r} → {decoded!r}"


# ---------------------------------------------------------------------------
# 测试 12: SentencePieceUnigramTokenizer 训练 + 编解码
# ---------------------------------------------------------------------------


def test_unigram_train_encode_decode():
    """SentencePieceUnigramTokenizer：EM 训练 + Viterbi 编码 + decode 往返。"""
    corpus = [
        "床前明月光，疑是地上霜。",
        "举头望明月，低头思故乡。",
        "Hello world machine learning",
        "你好世界机器学习",
    ]
    tok = SentencePieceUnigramTokenizer(vocab_size=200)
    tok.train(corpus, vocab_size=200)
    # 词表大小不超过目标
    assert len(tok) <= 200 + 10, f"unigram vocab 过大：{len(tok)}"  # 容差
    # 编解码往返不产生 U+FFFD
    for text in ["床前明月光", "Hello", "你好世界"]:
        ids = tok.encode(text, add_special_tokens=False)
        assert isinstance(ids, list)
        decoded = tok.decode(ids)
        assert "\ufffd" not in decoded, (
            f"unigram decode 产生 U+FFFD：{text!r} → {decoded!r}"
        )


# ---------------------------------------------------------------------------
# 测试 13: SentencePieceUnigramTokenizer apply_chat_template
# ---------------------------------------------------------------------------


def test_unigram_apply_chat_template():
    """SentencePieceUnigramTokenizer.apply_chat_template：渲染 chat 并编码。"""
    corpus = ["你好世界", "机器学习很有趣", "Hello world"]
    tok = SentencePieceUnigramTokenizer(vocab_size=100)
    tok.train(corpus, vocab_size=100)
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    ids = tok.apply_chat_template(messages)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    # 解码后应包含原始内容（special token 被丢弃，其余 piece 拼接）
    decoded = tok.decode(ids)
    assert "你好" in decoded


# ---------------------------------------------------------------------------
# 测试 14: 特殊 token 注册验证
# ---------------------------------------------------------------------------


def test_special_tokens_registered():
    """BPE train 后默认注册 11 个特殊 token（旧风格 4 + 新风格 7）。"""
    corpus = ["Hello world", "你好世界"]
    tok = BPETokenizer.train(corpus, vocab_size=200)
    # 旧风格 4 个（向后兼容已有测试）
    for st in ["<bos>", "<eos>", "<pad>", "<unk>"]:
        assert st in tok.vocab, f"旧风格 special token {st!r} 未注册"
        assert st in tok.special_tokens, f"旧风格 special token {st!r} 不在 special_tokens 字典"
    # 新风格 7 个（chat_template 用）
    for st in ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
               "<|user|>", "<|assistant|>", "<|system|>"]:
        assert st in tok.vocab, f"新风格 special token {st!r} 未注册"
        assert st in tok.special_tokens, f"新风格 special token {st!r} 不在 special_tokens 字典"
    # SpecialTokens 字典（unigram 模块导出）应包含 7 个新风格 token
    assert SpecialTokens["user"] == "<|user|>"
    assert SpecialTokens["assistant"] == "<|assistant|>"
    assert SpecialTokens["system"] == "<|system|>"
    assert SpecialTokens["eos"] == "<|eos|>"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
