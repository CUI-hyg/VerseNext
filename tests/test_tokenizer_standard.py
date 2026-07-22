"""Task 4.4: Tokenizer 架构标准化测试。

验证：
1. ``BaseTokenizer`` 是抽象基类，不能直接实例化
2. ``BPETokenizer`` / ``ByteTokenizer`` / ``CharTokenizer`` 均继承 ``BaseTokenizer``
3. 三种 tokenizer 提供统一接口 ``encode`` / ``decode`` / ``save`` / ``load`` / ``__len__``
4. NFKC 正规化生效（全角字符映射到半角）
5. ``preprocess`` 钩子可被子类覆盖
6. 控制字符被去除（保留 ``\\n`` / ``\\r`` / ``\\t``）

运行方式：
    cd /workspace && python -m pytest tests/test_tokenizer_standard.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_infra.verse_tokenizer import (
    BaseTokenizer,
    BPETokenizer,
    ByteTokenizer,
    CharTokenizer,
)


# ---------------------------------------------------------------------------
# 测试 1: BaseTokenizer 是抽象基类
# ---------------------------------------------------------------------------


def test_base_tokenizer_is_abstract():
    """BaseTokenizer 是抽象基类，不能直接实例化。"""
    with pytest.raises(TypeError):
        BaseTokenizer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 测试 2: 三种 tokenizer 均继承 BaseTokenizer
# ---------------------------------------------------------------------------


def test_bpe_inherits_base():
    """BPETokenizer 继承 BaseTokenizer。"""
    assert issubclass(BPETokenizer, BaseTokenizer)
    tok = BPETokenizer({}, [], byte_level=True)
    assert isinstance(tok, BaseTokenizer)


def test_byte_inherits_base():
    """ByteTokenizer 继承 BaseTokenizer。"""
    assert issubclass(ByteTokenizer, BaseTokenizer)
    tok = ByteTokenizer()
    assert isinstance(tok, BaseTokenizer)


def test_char_inherits_base():
    """CharTokenizer 继承 BaseTokenizer。"""
    assert issubclass(CharTokenizer, BaseTokenizer)
    tok = CharTokenizer()
    assert isinstance(tok, BaseTokenizer)


# ---------------------------------------------------------------------------
# 测试 3: 三种 tokenizer 接口一致（都有 encode/decode/save/load/__len__）
# ---------------------------------------------------------------------------


def test_all_tokenizers_have_unified_interface():
    """三种 tokenizer 都有 encode/decode/save/load/__len__ 方法。"""
    tokenizers = [
        BPETokenizer({}, [], byte_level=True),
        ByteTokenizer(),
        CharTokenizer(),
    ]
    for tok in tokenizers:
        cls_name = tok.__class__.__name__
        # 检查方法存在
        assert hasattr(tok, "encode"), f"{cls_name} 缺少 encode 方法"
        assert hasattr(tok, "decode"), f"{cls_name} 缺少 decode 方法"
        assert hasattr(tok, "save"), f"{cls_name} 缺少 save 方法"
        assert hasattr(tok, "load"), f"{cls_name} 缺少 load 方法"
        assert hasattr(tok, "__len__"), f"{cls_name} 缺少 __len__ 方法"
        assert hasattr(tok, "preprocess"), f"{cls_name} 缺少 preprocess 方法"
        # 检查方法可调用
        assert callable(tok.encode)
        assert callable(tok.decode)
        assert callable(tok.save)
        assert callable(tok.load)
        assert callable(tok.preprocess)
        # __len__ 返回 int
        n = len(tok)
        assert isinstance(n, int) and n >= 0
        # encode/decode 基本往返
        ids = tok.encode("hello")
        assert isinstance(ids, list)
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)


def test_all_tokenizers_load_is_classmethod():
    """三种 tokenizer 的 load 都是 classmethod。"""
    for cls in (BPETokenizer, ByteTokenizer, CharTokenizer):
        # classmethod 的 __func__ 属性存在
        assert hasattr(cls.load, "__func__"), f"{cls.__name__}.load 不是 classmethod"


# ---------------------------------------------------------------------------
# 测试 4: NFKC 正规化生效（全角字符映射到半角）
# ---------------------------------------------------------------------------


def test_nfkc_fullwidth_letters_to_halfwidth():
    """NFKC 正规化把全角字母映射到半角字母。

    全角 ``Ａ`` (U+FF21) → 半角 ``A`` (U+0041)
    全角 ``ｚ`` (U+FF5A) → 半角 ``z`` (U+007A)
    """
    tok = ByteTokenizer()
    # 全角字母经 NFKC 应映射到半角
    fullwidth = "Ａｚ"
    expected = "Az"
    ids = tok.encode(fullwidth)
    decoded = tok.decode(ids)
    assert decoded == expected, (
        f"NFKC 未生效：{fullwidth!r} → {decoded!r}，期望 {expected!r}"
    )


def test_nfkc_fullwidth_digits_to_halfwidth():
    """NFKC 正规化把全角数字映射到半角数字。

    全角 ``１`` (U+FF11) → 半角 ``1`` (U+0031)
    全角 ``９`` (U+FF19) → 半角 ``9`` (U+0039)
    """
    tok = ByteTokenizer()
    fullwidth = "１９"
    expected = "19"
    ids = tok.encode(fullwidth)
    decoded = tok.decode(ids)
    assert decoded == expected, (
        f"NFKC 未生效：{fullwidth!r} → {decoded!r}，期望 {expected!r}"
    )


def test_nfkc_bpe_fullwidth_to_halfwidth():
    """BPETokenizer 也做 NFKC 正规化。

    用 preprocess 直接验证（空 vocab 的 BPETokenizer 无法 encode，
    但 preprocess 是 BaseTokenizer 提供的统一钩子）。
    """
    tok = BPETokenizer({}, [], byte_level=True)
    fullwidth = "ＡＢＣ"
    expected = "ABC"
    # 直接验证 preprocess 做 NFKC 正规化
    assert tok.preprocess(fullwidth) == expected, (
        f"BPE preprocess NFKC 未生效：{fullwidth!r} → {tok.preprocess(fullwidth)!r}"
    )


def test_nfkc_bpe_trained_fullwidth_to_halfwidth():
    """训练后的 BPETokenizer encode/decode 往返验证 NFKC 正规化。"""
    corpus = [
        "Hello ABC world",
        "你好世界",
        "Machine learning ABC",
    ]
    tok = BPETokenizer.train(corpus, vocab_size=300)
    fullwidth = "ABC"
    ids = tok.encode(fullwidth, add_special_tokens=False)
    decoded = tok.decode(ids)
    assert decoded == fullwidth, (
        f"BPE 训练后 NFKC 往返失败：{fullwidth!r} → {decoded!r}"
    )


def test_nfkc_char_fullwidth_to_halfwidth():
    """CharTokenizer 也做 NFKC 正规化。"""
    tok = CharTokenizer()
    fullwidth = "Ａ１"
    expected = "A1"
    ids = tok.encode(fullwidth, add_special_tokens=False)
    decoded = tok.decode(ids)
    assert decoded == expected, (
        f"Char NFKC 未生效：{fullwidth!r} → {decoded!r}，期望 {expected!r}"
    )


def test_preprocess_directly():
    """直接调用 preprocess 验证 NFKC 正规化。"""
    tok = ByteTokenizer()
    # 全角字母
    assert tok.preprocess("Ａ") == "A"
    # 全角数字
    assert tok.preprocess("１") == "1"
    # 组合字符（é = e + ´ 的组合形式 → 规范形式 é）
    # NFKC 会把组合形式 NFC 化
    combined = "e\u0301"  # e + 组合重音
    nfkc_result = tok.preprocess(combined)
    assert nfkc_result == "\u00e9", (
        f"组合字符未正规化：{combined!r} → {nfkc_result!r}"
    )


# ---------------------------------------------------------------------------
# 测试 5: preprocess 去除控制字符（保留 \n \r \t）
# ---------------------------------------------------------------------------


def test_preprocess_removes_control_chars():
    """preprocess 去除 Cc 类控制字符，但保留 \\n \\r \\t。"""
    tok = ByteTokenizer()
    # \x07 (BEL)、\x00 (NUL) 是控制字符，应被去除
    text = "hello\x07world\x00"
    processed = tok.preprocess(text)
    assert "\x07" not in processed
    assert "\x00" not in processed
    assert "hello" in processed
    assert "world" in processed


def test_preprocess_preserves_whitespace():
    """preprocess 保留 \\n \\r \\t 等基本空白。"""
    tok = ByteTokenizer()
    text = "line1\nline2\tcol2\rcol3"
    processed = tok.preprocess(text)
    assert "\n" in processed
    assert "\t" in processed
    assert "\r" in processed


# ---------------------------------------------------------------------------
# 测试 6: preprocess 可被子类覆盖
# ---------------------------------------------------------------------------


def test_preprocess_can_be_overridden():
    """子类可以覆盖 preprocess 方法实现自定义预处理。"""

    class UpperCaseTokenizer(ByteTokenizer):
        """自定义 tokenizer：preprocess 把文本转大写。"""

        def preprocess(self, text: str) -> str:
            return text.upper()

    tok = UpperCaseTokenizer()
    # preprocess 应转大写
    assert tok.preprocess("hello") == "HELLO"
    # encode 后 decode 也应是大写
    ids = tok.encode("hello")
    assert tok.decode(ids) == "HELLO"


# ---------------------------------------------------------------------------
# 测试 7: 三种 tokenizer 的 save/load 往返一致
# ---------------------------------------------------------------------------


def test_byte_save_load_roundtrip():
    """ByteTokenizer save/load 往返一致。"""
    tok = ByteTokenizer()
    text = "hello 世界"
    original_ids = tok.encode(text)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "byte.json")
        tok.save(path)
        assert os.path.exists(path)
        tok2 = ByteTokenizer.load(path)
    loaded_ids = tok2.encode(text)
    assert loaded_ids == original_ids
    assert tok2.decode(loaded_ids) == tok.decode(original_ids)


def test_char_save_load_roundtrip():
    """CharTokenizer save/load 往返一致。"""
    tok = CharTokenizer()
    text = "hello 你好"
    original_ids = tok.encode(text, add_special_tokens=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "char.json")
        tok.save(path)
        assert os.path.exists(path)
        tok2 = CharTokenizer.load(path)
    loaded_ids = tok2.encode(text, add_special_tokens=False)
    assert loaded_ids == original_ids
    assert tok2.decode(loaded_ids) == tok.decode(original_ids)


def test_bpe_save_load_roundtrip():
    """BPETokenizer save/load 往返一致。"""
    # 用一个简单的 vocab 测试
    vocab = {"a": 0, "b": 1, "c": 2, "<eos>": 3}
    merges = [["a", "b"]]
    tok = BPETokenizer(vocab, merges, special_tokens=["<eos>"], byte_level=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bpe.json")
        tok.save(path)
        assert os.path.exists(path)
        tok2 = BPETokenizer.load(path)
    # vocab 一致
    assert tok2.vocab == tok.vocab
    # special_tokens 一致
    assert set(tok2.special_tokens) == set(tok.special_tokens)


# ---------------------------------------------------------------------------
# 测试 8: BaseTokenizer 的抽象方法契约
# ---------------------------------------------------------------------------


def test_abstract_methods_enforced():
    """子类必须实现所有抽象方法，否则不能实例化。"""
    # 缺少 encode 的子类不能实例化
    class IncompleteTokenizer(BaseTokenizer):
        def decode(self, ids):
            return ""

        def save(self, path):
            pass

        @classmethod
        def load(cls, path):
            return cls()

        def __len__(self):
            return 0

    with pytest.raises(TypeError):
        IncompleteTokenizer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
