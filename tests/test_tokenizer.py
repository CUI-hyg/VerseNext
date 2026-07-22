"""Task 3.7: verse_tokenizer 单元测试。

覆盖：
1. BPE train：中英文混合 corpus，vocab_size=300 → vocab 长度 ≤ 300
2. BPE encode/decode 往返一致
3. BPE add_special_tokens：vocab 增长 4，special_tokens 有 4 项
4. BPE save/load：重建 tokenizer encode 结果一致
5. ByteTokenizer encode/decode 往返一致
6. ByteTokenizer bos/eos：首尾 id 正确
7. load_tokenizer 三种 kind 均返回统一接口

运行方式：
    cd /workspace && python -m pytest tests/test_tokenizer.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_infra.verse_tokenizer import BPETokenizer, CharTokenizer, ByteTokenizer, load_tokenizer


# ---------------------------------------------------------------------------
# 测试用混合语料（中英文）
# ---------------------------------------------------------------------------

CORPUS = [
    "Hello world! This is a test corpus for BPE training.",
    "你好世界！这是一个用于 BPE 训练的测试语料。",
    "The quick brown fox jumps over the lazy dog.",
    "敏捷的棕色狐狸跳过了懒狗。",
    "Machine learning is fun. 机器学习很有趣。",
    "Natural language processing 自然语言处理 is important.",
    "Tokens are the basic units of text processing.",
    "分词器是自然语言处理的基础组件。",
]


# ---------------------------------------------------------------------------
# 测试 1: BPE train
# ---------------------------------------------------------------------------


def test_bpe_train_vocab_size():
    """BPE train 在中英文混合语料上 vocab_size=300，结果 vocab 长度 ≤ 300。"""
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    assert len(tok.vocab) <= 300, f"vocab size {len(tok.vocab)} > 300"
    # 至少有 256 个基础字节字符
    assert len(tok.vocab) >= 256, f"vocab size {len(tok.vocab)} < 256"
    # 应包含 4 个 special tokens（train 自动 add）
    for st in ["<bos>", "<eos>", "<pad>", "<unk>"]:
        assert st in tok.vocab, f"special token {st!r} not in vocab"
        assert st in tok.special_tokens, f"special token {st!r} not in special_tokens"
    # 应有非空 merges
    assert len(tok.merge_ranks) > 0, "no merges learned"


# ---------------------------------------------------------------------------
# 测试 2: BPE encode/decode 往返一致
# ---------------------------------------------------------------------------


def test_bpe_encode_decode_roundtrip():
    """BPE encode/decode 往返一致：原文本 → encode → decode ≈ 原文本。"""
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    test_texts = [
        "Hello world",
        "你好世界",
        "Mixed 中英 text 混合文本 123",
        "The quick brown fox",
        "敏捷的棕色狐狸",
        "numbers 12345 and symbols !@#$%",
    ]
    for text in test_texts:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert decoded == text, (
            f"roundtrip failed: {text!r}\n"
            f"  ids    = {ids}\n"
            f"  decoded= {decoded!r}"
        )


# ---------------------------------------------------------------------------
# 测试 3: BPE add_special_tokens
# ---------------------------------------------------------------------------


def test_bpe_add_special_tokens():
    """add_special_tokens 调用后 vocab 增长 4，special_tokens 列表有 4 项。"""
    # 用一个空 vocab 的 BPETokenizer
    tok = BPETokenizer({}, [], byte_level=True)
    initial_vocab_size = len(tok.vocab)
    initial_special_count = len(tok.special_tokens)

    new_specials = ["<bos>", "<eos>", "<pad>", "<unk>"]
    tok.add_special_tokens(new_specials)

    assert len(tok.vocab) == initial_vocab_size + 4, (
        f"vocab should grow by 4: {initial_vocab_size} → {len(tok.vocab)}"
    )
    assert len(tok.special_tokens) == initial_special_count + 4
    for st in new_specials:
        assert st in tok.vocab
        assert st in tok.special_tokens
        assert isinstance(tok.vocab[st], int)


def test_bpe_add_special_tokens_idempotent():
    """重复 add 相同 special tokens 应幂等（不重复添加）。"""
    tok = BPETokenizer({}, [], byte_level=True)
    tok.add_special_tokens(["<bos>", "<eos>"])
    size_after_first = len(tok.vocab)
    tok.add_special_tokens(["<bos>", "<eos>"])  # 重复
    assert len(tok.vocab) == size_after_first, "add_special_tokens not idempotent"


# ---------------------------------------------------------------------------
# 测试 4: BPE save/load
# ---------------------------------------------------------------------------


def test_bpe_save_load():
    """save 后 load 重建的 tokenizer encode 结果一致。"""
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    test_text = "Hello 世界! BPE round trip test."
    original_ids = tok.encode(test_text, add_special_tokens=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenizer.json")
        tok.save(path)
        assert os.path.exists(path), "save did not create file"

        tok2 = BPETokenizer.load(path)
        loaded_ids = tok2.encode(test_text, add_special_tokens=False)

    assert loaded_ids == original_ids, (
        f"save/load encode mismatch:\n  original={original_ids}\n  loaded ={loaded_ids}"
    )
    # decode 也应一致
    assert tok2.decode(loaded_ids) == tok.decode(original_ids)


def test_bpe_save_load_preserves_special_tokens():
    """save/load 应保留 special_tokens。"""
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenizer.json")
        tok.save(path)
        tok2 = BPETokenizer.load(path)

    assert set(tok.special_tokens) == set(tok2.special_tokens)
    for st in tok.special_tokens:
        assert tok.vocab[st] == tok2.vocab[st], (
            f"special token {st!r} id mismatch: {tok.vocab[st]} vs {tok2.vocab[st]}"
        )


# ---------------------------------------------------------------------------
# 测试 5: ByteTokenizer encode/decode 往返一致
# ---------------------------------------------------------------------------


def test_byte_tokenizer_roundtrip():
    """ByteTokenizer encode/decode 往返一致：任意 UTF-8 文本可还原。"""
    tok = ByteTokenizer()
    assert tok.vocab_size == 259
    test_texts = [
        "Hello world",
        "你好世界",
        "Mixed 中英 text 123!@#",
        "🎉 Emoji test 🚀",
        "",  # 空字符串
        "Multi\nline\ntext\twith\ttabs",
        "日本語のテスト",
        "Русский текст",
    ]
    for text in test_texts:
        ids = tok.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = tok.decode(ids)
        assert decoded == text, (
            f"roundtrip failed: {text!r}\n  ids    = {ids}\n  decoded= {decoded!r}"
        )


def test_byte_tokenizer_default_ids():
    """ByteTokenizer 默认 bos/eos/pad/unk id 正确。"""
    tok = ByteTokenizer()
    assert tok.bos_id == 256
    assert tok.eos_id == 257
    assert tok.pad_id == 258
    assert tok.unk_id == 255
    assert len(tok) == 259


# ---------------------------------------------------------------------------
# 测试 6: ByteTokenizer bos/eos
# ---------------------------------------------------------------------------


def test_byte_tokenizer_bos_eos():
    """ByteTokenizer encode(add_bos=True, add_eos=True) 首尾 id 正确。"""
    tok = ByteTokenizer()
    text = "test"
    ids = tok.encode(text, add_bos=True, add_eos=True)

    assert ids[0] == tok.bos_id, (
        f"first id should be bos_id ({tok.bos_id}), got {ids[0]}"
    )
    assert ids[-1] == tok.eos_id, (
        f"last id should be eos_id ({tok.eos_id}), got {ids[-1]}"
    )
    # 中间部分应为 text 的 UTF-8 字节
    assert ids[1:-1] == list(text.encode("utf-8"))


def test_byte_tokenizer_strip_special():
    """decode(strip_special=True) 丢弃 special；strip_special=False 保留字符串。"""
    tok = ByteTokenizer()
    ids = [tok.bos_id] + list("hi".encode("utf-8")) + [tok.eos_id]

    # strip_special=True：special token 被丢弃
    decoded_strip = tok.decode(ids, strip_special=True)
    assert decoded_strip == "hi"

    # strip_special=False：special token 还原为字符串
    decoded_keep = tok.decode(ids, strip_special=False)
    assert "<bos>" in decoded_keep
    assert "<eos>" in decoded_keep
    assert "hi" in decoded_keep


# ---------------------------------------------------------------------------
# 测试 7: load_tokenizer 三种 kind
# ---------------------------------------------------------------------------


def test_load_tokenizer_byte():
    """load_tokenizer('byte') 返回 ByteTokenizer，有 encode/decode 方法。"""
    tok = load_tokenizer("byte")
    assert hasattr(tok, "encode")
    assert hasattr(tok, "decode")
    text = "hello 世界"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)
    assert decoded == text


def test_load_tokenizer_bpe_no_path():
    """load_tokenizer('bpe') 无 path 返回空 BPETokenizer，有 encode/decode。"""
    tok = load_tokenizer("bpe")
    assert hasattr(tok, "encode")
    assert hasattr(tok, "decode")
    # 即使 vocab 很小，encode 也不应崩溃
    ids = tok.encode("a")
    assert isinstance(ids, list)


def test_load_tokenizer_bpe_with_path():
    """load_tokenizer('bpe', path) 从文件加载，encode 结果与原 tokenizer 一致。"""
    trained = BPETokenizer.train(CORPUS, vocab_size=300)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bpe.json")
        trained.save(path)
        loaded = load_tokenizer("bpe", path)

    text = "hello world 你好"
    ids1 = trained.encode(text, add_special_tokens=False)
    ids2 = loaded.encode(text, add_special_tokens=False)
    assert ids1 == ids2, f"bpe load mismatch: {ids1} vs {ids2}"


def test_load_tokenizer_hf_fallback():
    """load_tokenizer('hf') 无 tokenizers 包时 fallback 到 ByteTokenizer。"""
    tok = load_tokenizer("hf")
    assert hasattr(tok, "encode")
    assert hasattr(tok, "decode")
    # 应能正常 encode/decode（无论是否安装 tokenizers 包）
    text = "test 文本"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)


def test_load_tokenizer_byte_with_path():
    """load_tokenizer('byte', path) 从 JSON 加载 ByteTokenizer。"""
    original = ByteTokenizer()
    text = "ByteTokenizer save/load 测试"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "byte.json")
        original.save(path)
        loaded = load_tokenizer("byte", path)

    ids1 = original.encode(text, add_bos=True, add_eos=True)
    ids2 = loaded.encode(text, add_bos=True, add_eos=True)
    assert ids1 == ids2


def test_load_tokenizer_unknown_kind():
    """未知 kind 应抛出 ValueError。"""
    with pytest.raises(ValueError):
        load_tokenizer("unknown_kind")


# ---------------------------------------------------------------------------
# 附加测试：CharTokenizer 仍可用（无回归）
# ---------------------------------------------------------------------------


def test_char_tokenizer_no_regression():
    """CharTokenizer 仍可用，不因新增 API 而回归。"""
    tok = CharTokenizer()
    text = "hello 你好"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids)
    # CharTokenizer 把 EOS 替换为 \n，这里 add_special_tokens=False 不追加 EOS
    assert decoded == text


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
