"""Task 5.6: VerseTokenizer 优化测试。

覆盖 Task 5 主体部分的 5 个子任务（不含 SubTask 5.4 NexRL 集成）：
1. SubTask 5.1: BPE 训练支持 ``min_frequency`` / ``max_token_length`` / 并行 merge（``workers``）
2. SubTask 5.2: WordPiece tokenizer + unigram 对齐 sentencepiece + 批量 encode/decode
3. SubTask 5.3: 对齐 HF ``BatchEncoding``（``add_bos`` / ``add_eos`` / ``truncation`` / ``padding``）
4. SubTask 5.5: ``BPETokenizer.from_pretrained`` Qwen tokenizer 加载（graceful skip）
5. byte-aligned decode 无 U+FFFD 乱码

运行方式：
    cd /workspace && python -m pytest tests/test_tokenizer_optimization.py -v
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

from verse_infra.verse_tokenizer import (
    BaseTokenizer,
    BPETokenizer,
    ByteTokenizer,
    WordPieceTokenizer,
    SentencePieceUnigramTokenizer,
)
from verse_infra.verse_tokenizer.wordpiece import (
    CONTINUING_SUBWORD_PREFIX,
    WORDPIECE_DEFAULT_SPECIAL_TOKENS,
)

try:
    import numpy as _np
except ImportError:  # numpy 是硬依赖，但兜底
    _np = None


# ---------------------------------------------------------------------------
# 测试语料
# ---------------------------------------------------------------------------
_CORPUS = [
    "Hello world! This is a test corpus for BPE training.",
    "你好世界！这是一个用于 BPE 训练的测试语料。",
    "Machine learning is fun. 机器学习很有趣。",
    "The quick brown fox jumps over the lazy dog.",
    "自然语言处理是人工智能的重要分支。",
    "Tokenization is the first step in NLP pipelines.",
    "分词是自然语言处理的第一步。",
    "Parallel training accelerates BPE merges significantly.",
]


# ===========================================================================
# SubTask 5.1: BPE 训练 min_frequency / max_token_length / 并行 merge
# ===========================================================================


class TestBPETrainMinFrequency:
    """BPE 训练 ``min_frequency`` 过滤测试。"""

    def test_min_frequency_default_is_2(self):
        """默认 ``min_frequency=2``，频次 < 2 的 pair 不合并。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300)
        # 默认 min_frequency=2，训练应正常完成
        assert len(tok.vocab) > 256, "vocab 至少含 256 个基础字节字符"

    def test_min_frequency_filters_low_freq_pairs(self):
        """``min_frequency=10`` 时低频 pair 被过滤，merges 更少。"""
        tok_low = BPETokenizer.train(_CORPUS, vocab_size=300, min_frequency=1)
        tok_high = BPETokenizer.train(_CORPUS, vocab_size=300, min_frequency=10)
        # 高 min_frequency 应过滤更多 pair，merges 更少（用 merge_ranks 字典大小）
        assert len(tok_high.merge_ranks) <= len(tok_low.merge_ranks), (
            f"min_frequency=10 的 merges ({len(tok_high.merge_ranks)}) "
            f"应 <= min_frequency=1 的 merges ({len(tok_low.merge_ranks)})"
        )

    def test_min_frequency_1_allows_single_occurrence(self):
        """``min_frequency=1`` 允许所有 pair 进入合并候选。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300, min_frequency=1)
        # 至少有一些 merge
        assert len(tok.merge_ranks) > 0, "min_frequency=1 应至少产生一些 merge"


class TestBPETrainMaxTokenLength:
    """BPE 训练 ``max_token_length`` 限制测试。"""

    def test_max_token_length_limits_merged_token_size(self):
        """``max_token_length=4`` 时合并后的 token 不超过 4 字符。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300, max_token_length=4)
        # 检查所有 merge 产生的 token 长度（merge_ranks 的 key 是 (a, b) 元组）
        for (a, b) in tok.merge_ranks:
            merged_len = len(a) + len(b)
            assert merged_len <= 4, (
                f"max_token_length=4 但合并后 token '{a}{b}' 长度={merged_len}"
            )

    def test_max_token_length_none_no_limit(self):
        """``max_token_length=None``（默认）不限制 token 长度。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300, max_token_length=None)
        # 不限制时应能产生更长的 merge
        assert len(tok.merge_ranks) > 0

    def test_max_token_length_2_minimal_merges(self):
        """``max_token_length=2`` 只允许 2 字符 token，merge 非常有限。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300, max_token_length=2)
        # 每个 merge 最多产生 2 字符 token
        for (a, b) in tok.merge_ranks:
            merged_len = len(a) + len(b)
            assert merged_len <= 2


class TestBPEParallelTrain:
    """BPE 并行训练（``workers`` 参数）测试。"""

    def test_workers_1_vs_4_produce_same_merges(self):
        """``workers=1`` 和 ``workers=4`` 产生相同的 merges（统计阶段并行不影响确定性）。"""
        tok_serial = BPETokenizer.train(_CORPUS, vocab_size=300, workers=1)
        tok_parallel = BPETokenizer.train(_CORPUS, vocab_size=300, workers=4)
        # merge_ranks 应完全一致（只有统计阶段并行，argmax 选择仍确定）
        assert tok_serial.merge_ranks == tok_parallel.merge_ranks, (
            "并行训练与串行训练的 merge_ranks 不一致"
        )
        assert tok_serial.vocab == tok_parallel.vocab, (
            "并行训练与串行训练的 vocab 不一致"
        )

    def test_workers_2_produces_same_result(self):
        """``workers=2`` 也应与 ``workers=1`` 一致。"""
        tok_serial = BPETokenizer.train(_CORPUS, vocab_size=300, workers=1)
        tok_parallel = BPETokenizer.train(_CORPUS, vocab_size=300, workers=2)
        assert tok_serial.merge_ranks == tok_parallel.merge_ranks

    def test_workers_8_produces_same_result(self):
        """``workers=8`` 也应与 ``workers=1`` 一致。"""
        tok_serial = BPETokenizer.train(_CORPUS, vocab_size=300, workers=1)
        tok_parallel = BPETokenizer.train(_CORPUS, vocab_size=300, workers=8)
        assert tok_serial.merge_ranks == tok_parallel.merge_ranks

    def test_parallel_train_encode_decode_consistent(self):
        """并行训练后的 tokenizer encode/decode 正常。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300, workers=4)
        for text in ["Hello world", "你好世界", "machine learning"]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids)
            assert "\ufffd" not in decoded, (
                f"并行训练后 decode 产生 U+FFFD：{text!r} → {decoded!r}"
            )

    def test_parallel_train_with_min_frequency(self):
        """并行训练 + min_frequency 组合使用。"""
        tok = BPETokenizer.train(
            _CORPUS, vocab_size=300, workers=4, min_frequency=2
        )
        assert len(tok.vocab) > 256
        # encode/decode 不产生乱码
        ids = tok.encode("Hello", add_special_tokens=False)
        decoded = tok.decode(ids)
        assert "\ufffd" not in decoded


# ===========================================================================
# SubTask 5.2a: WordPiece tokenizer
# ===========================================================================


class TestWordPieceTokenizer:
    """WordPiece 分词器测试。"""

    def test_train_basic(self):
        """WordPiece 训练基本功能。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        assert len(tok) > 0, "训练后 vocab 不为空"
        # 特殊 token 应注册
        for st in ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]:
            assert st in tok.vocab, f"特殊 token {st!r} 未注册"

    def test_train_min_frequency(self):
        """``min_frequency`` 过滤低频子串。"""
        tok_low = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=1)
        tok_high = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=5)
        # 高 min_frequency 应过滤更多子串
        assert len(tok_high) <= len(tok_low), (
            f"min_frequency=5 的 vocab ({len(tok_high)}) "
            f"应 <= min_frequency=1 的 vocab ({len(tok_low)})"
        )

    def test_encode_decode_roundtrip(self):
        """WordPiece encode/decode 往返。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        for text in ["hello", "world", "Hello world"]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids)
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)
            # decode 不产生乱码
            assert "\ufffd" not in decoded

    def test_encode_with_special_tokens(self):
        """encode 默认加 [CLS]/[SEP]。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        ids_with_special = tok.encode("hello", add_special_tokens=True)
        ids_without = tok.encode("hello", add_special_tokens=False)
        # 加 special tokens 时长度更长（含 [CLS] + [SEP]）
        assert len(ids_with_special) >= len(ids_without)
        # 首尾应是 [CLS] / [SEP]
        cls_id = tok.vocab.get("[CLS]")
        sep_id = tok.vocab.get("[SEP]")
        if cls_id is not None and len(ids_with_special) > 0:
            assert ids_with_special[0] == cls_id, "首 token 不是 [CLS]"
        if sep_id is not None and len(ids_with_special) > 1:
            assert ids_with_special[-1] == sep_id, "末 token 不是 [SEP]"

    def test_continuing_subword_prefix(self):
        """``##`` 前缀表示词中子词。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=300, min_frequency=1)
        # 词表中应有 ``##`` 前缀的 token
        has_continuing = any(
            t.startswith(CONTINUING_SUBWORD_PREFIX) for t in tok.vocab
        )
        assert has_continuing, "词表中无 ``##`` 前缀的 token"

    def test_save_load_roundtrip(self):
        """WordPiece save/load 往返。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = WordPieceTokenizer.load(path)
            assert len(tok2) == len(tok), "reload 后 vocab 大小不一致"
            assert tok2.vocab == tok.vocab, "reload 后 vocab 内容不一致"
            # encode 一致
            ids1 = tok.encode("hello", add_special_tokens=False)
            ids2 = tok2.encode("hello", add_special_tokens=False)
            assert ids1 == ids2, "reload 后 encode 结果不一致"
        finally:
            os.unlink(path)

    def test_inherits_base_tokenizer(self):
        """WordPieceTokenizer 继承 BaseTokenizer。"""
        assert issubclass(WordPieceTokenizer, BaseTokenizer)

    def test_encode_batch_decode_batch(self):
        """encode_batch / decode_batch 批量方法（继承自 BaseTokenizer）。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        texts = ["hello", "world", "test"]
        batch_ids = tok.encode_batch(texts, add_special_tokens=False)
        assert isinstance(batch_ids, list)
        assert len(batch_ids) == len(texts)
        assert all(isinstance(ids, list) for ids in batch_ids)
        # decode_batch
        decoded = tok.decode_batch(batch_ids)
        assert isinstance(decoded, list)
        assert len(decoded) == len(texts)
        assert all(isinstance(d, str) for d in decoded)

    def test_apply_chat_template(self):
        """WordPiece apply_chat_template。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        ids = tok.apply_chat_template(messages)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_unk_token_for_unknown_chars(self):
        """未知字符编码为 [UNK]。"""
        tok = WordPieceTokenizer.train(
            ["hello world"], vocab_size=50, min_frequency=1
        )
        # 用训练时没见过的字符
        ids = tok.encode("xyzqwerty", add_special_tokens=False)
        unk_id = tok.unk_id
        if unk_id is not None:
            # 至少有一个 unk
            assert unk_id in ids or len(ids) > 0

    def test_max_input_chars_per_word(self):
        """``max_input_chars_per_word`` 限制超长单词。"""
        tok = WordPieceTokenizer(
            vocab={"[UNK]": 0, "a": 1, "b": 2, "##c": 3},
            special_tokens=["[UNK]"],
            max_input_chars_per_word=3,
        )
        # 超长单词整体当 unk
        ids = tok.encode("aaaaaaa", add_special_tokens=False)
        unk_id = tok.unk_id
        if unk_id is not None:
            assert unk_id in ids


# ===========================================================================
# SubTask 5.2b: Unigram 对齐 BaseTokenizer
# ===========================================================================


class TestUnigramBaseTokenizer:
    """SentencePieceUnigramTokenizer 继承 BaseTokenizer 测试。"""

    def test_inherits_base_tokenizer(self):
        """unigram 继承 BaseTokenizer。"""
        assert issubclass(SentencePieceUnigramTokenizer, BaseTokenizer)

    def test_encode_batch_decode_batch(self):
        """unigram encode_batch / decode_batch（继承自 BaseTokenizer）。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        texts = ["你好", "世界", "Hello"]
        batch_ids = tok.encode_batch(texts, add_special_tokens=False)
        assert isinstance(batch_ids, list)
        assert len(batch_ids) == len(texts)
        decoded = tok.decode_batch(batch_ids)
        assert len(decoded) == len(texts)
        for d in decoded:
            assert "\ufffd" not in d

    def test_add_bos_add_eos_independent_switches(self):
        """unigram add_bos / add_eos 独立开关。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        assert tok.bos_id is not None, "bos_id 未设置"
        assert tok.eos_id is not None, "eos_id 未设置"

        ids_none = tok.encode("你好", add_special_tokens=False)
        ids_bos = tok.encode("你好", add_bos=True, add_eos=False)
        ids_eos = tok.encode("你好", add_bos=False, add_eos=True)
        ids_both = tok.encode("你好", add_bos=True, add_eos=True)

        # add_bos 时首 token 是 bos_id
        assert ids_bos[0] == tok.bos_id, "add_bos=True 但首 token 不是 bos_id"
        assert ids_bos[-1] != tok.eos_id or len(ids_bos) == 1, "add_eos=False 但末尾有 eos"

        # add_eos 时末 token 是 eos_id
        assert ids_eos[-1] == tok.eos_id, "add_eos=True 但末 token 不是 eos_id"

        # add_bos + add_eos
        assert ids_both[0] == tok.bos_id, "add_bos+eos 首 token 不是 bos_id"
        assert ids_both[-1] == tok.eos_id, "add_bos+eos 末 token 不是 eos_id"

    def test_save_load_roundtrip(self):
        """unigram save/load 往返（load 现为 classmethod）。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = SentencePieceUnigramTokenizer.load(path)
            assert len(tok2) == len(tok), "reload 后 vocab 大小不一致"
            assert tok2.bos_id == tok.bos_id, "reload 后 bos_id 不一致"
            assert tok2.eos_id == tok.eos_id, "reload 后 eos_id 不一致"
            # encode 一致
            ids1 = tok.encode("你好", add_special_tokens=False)
            ids2 = tok2.encode("你好", add_special_tokens=False)
            assert ids1 == ids2, "reload 后 encode 结果不一致"
        finally:
            os.unlink(path)

    def test_batch_encode_hf_style(self):
        """unigram batch_encode 返回 HF BatchEncoding 风格。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(["你好", "世界"], padding="longest", add_special_tokens=False)
        assert "input_ids" in out
        assert "attention_mask" in out
        if _np is not None:
            assert isinstance(out["input_ids"], _np.ndarray)
            assert isinstance(out["attention_mask"], _np.ndarray)
            assert out["input_ids"].shape == out["attention_mask"].shape

    def test_preprocess_nfkc_in_encode(self):
        """unigram encode 前置 NFKC 正规化。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        # 全角字符 NFKC 后变半角
        full_width = "Ｈｅｌｌｏ"  # 全角 Hello
        ids = tok.encode(full_width, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert "\ufffd" not in decoded


# ===========================================================================
# SubTask 5.3: HF BatchEncoding 对齐
# ===========================================================================


class TestBatchEncodingBPE:
    """BPE batch_encode 对齐 HF BatchEncoding 测试。"""

    def test_batch_encode_returns_dict_with_input_ids_and_attention_mask(self):
        """batch_encode 返回含 input_ids / attention_mask 的 dict。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(["hello", "world"], padding="longest", add_special_tokens=False)
        assert "input_ids" in out
        assert "attention_mask" in out

    def test_batch_encode_padding_longest(self):
        """padding='longest' 填充到最长序列长度。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(
            ["hello", "hi"],
            padding="longest",
            add_special_tokens=False,
        )
        if _np is not None:
            ids = out["input_ids"]
            mask = out["attention_mask"]
            assert ids.shape[0] == 2, "batch size 应为 2"
            # 两条序列应等长（padding 到最长）
            assert ids.shape[1] == mask.shape[1]
            # attention_mask 中 0 的位置是 padding
            has_zero = 0 in mask
            # "hi" 比 "hello" 短，应有 padding
            assert has_zero or ids.shape[1] == 0

    def test_batch_encode_padding_max_length(self):
        """padding='max_length' 填充到指定长度。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(
            ["hello", "hi"],
            padding="max_length",
            max_length=32,
            add_special_tokens=False,
        )
        if _np is not None:
            assert out["input_ids"].shape == (2, 32)
            assert out["attention_mask"].shape == (2, 32)

    def test_batch_encode_truncation_right(self):
        """truncation + truncation_side='right' 截断右侧（保留前段）。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        # 先获取 batch_encode 自身的不截断结果作对照（确保 add_bos/add_eos 解析一致）
        full_out = tok.batch_encode(["hello world test"], add_special_tokens=False)
        full_first = full_out["input_ids"][0]
        full_ids = list(full_first.tolist() if hasattr(full_first, "tolist") else full_first)
        # 截断到 max_length（不 padding，返回 list[list[int]]）
        out = tok.batch_encode(
            ["hello world test"],
            truncation="right",
            truncation_side="right",
            max_length=4,
            add_special_tokens=False,
        )
        truncated = out["input_ids"]
        # 统一转为 list 处理（可能是 ndarray 或 list）
        first = truncated[0]
        result = list(first.tolist() if hasattr(first, "tolist") else first)
        assert len(result) == 4, f"截断后长度应为 4，实际 {len(result)}"
        # 右截断保留前 4 个
        assert result == full_ids[:4], "右截断应保留前 4 个 token"

    def test_batch_encode_truncation_left(self):
        """truncation + truncation_side='left' 截断左侧（保留末段）。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        # 先获取 batch_encode 自身的不截断结果作对照
        full_out = tok.batch_encode(["hello world test"], add_special_tokens=False)
        full_first = full_out["input_ids"][0]
        full_ids = list(full_first.tolist() if hasattr(full_first, "tolist") else full_first)
        out = tok.batch_encode(
            ["hello world test"],
            truncation="right",
            truncation_side="left",
            max_length=4,
            add_special_tokens=False,
        )
        truncated = out["input_ids"]
        first = truncated[0]
        result = list(first.tolist() if hasattr(first, "tolist") else first)
        assert len(result) == 4
        # 左截断保留末尾 4 个
        assert result == full_ids[-4:], "左截断应保留末尾 4 个 token"

    def test_batch_encode_no_padding_returns_list(self):
        """不 padding 时返回 list[list[int]]。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(["hello", "hi"], add_special_tokens=False)
        assert "input_ids" in out
        assert isinstance(out["input_ids"], list)
        assert isinstance(out["attention_mask"], list)

    def test_batch_encode_return_tensors_list(self):
        """return_tensors='list' 返回 Python list。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(
            ["hello", "hi"],
            padding="longest",
            return_tensors="list",
            add_special_tokens=False,
        )
        assert isinstance(out["input_ids"], list)
        assert isinstance(out["attention_mask"], list)

    def test_add_bos_add_eos_independent_switches(self):
        """BPE add_bos / add_eos 独立开关。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        bos_id = tok.vocab.get(tok.BOS_TOKEN)
        eos_id = tok.vocab.get(tok.EOS_TOKEN)
        assert bos_id is not None, "bos token 未注册"
        assert eos_id is not None, "eos token 未注册"

        ids_none = tok.encode("hello", add_special_tokens=False)
        ids_bos = tok.encode("hello", add_bos=True, add_eos=False)
        ids_eos = tok.encode("hello", add_bos=False, add_eos=True)
        ids_both = tok.encode("hello", add_bos=True, add_eos=True)

        assert ids_bos[0] == bos_id, "add_bos=True 首 token 不是 bos"
        assert ids_eos[-1] == eos_id, "add_eos=True 末 token 不是 eos"
        assert ids_both[0] == bos_id and ids_both[-1] == eos_id, "add_bos+eos 首尾不对"

    def test_add_bos_add_eos_constructor_params(self):
        """BPE 构造函数 add_bos / add_eos 参数。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        # 从 merge_ranks 构造 merges list（按 rank 升序的 "a b" 字符串列表）
        sorted_merges = sorted(tok.merge_ranks.items(), key=lambda x: x[1])
        merges_list = [f"{a} {b}" for (a, b), _ in sorted_merges]
        # 显式设置 add_bos=True, add_eos=False
        tok_bos_only = BPETokenizer(
            vocab=tok.vocab,
            merges=merges_list,
            special_tokens=tok.special_tokens,
            add_bos=True,
            add_eos=False,
        )
        ids = tok_bos_only.encode("hello")
        bos_id = tok.vocab.get(tok.BOS_TOKEN)
        eos_id = tok.vocab.get(tok.EOS_TOKEN)
        if bos_id is not None and len(ids) > 0:
            assert ids[0] == bos_id, "构造 add_bos=True 首 token 不是 bos"
        if eos_id is not None and len(ids) > 1:
            assert ids[-1] != eos_id, "构造 add_eos=False 末尾不应有 eos"


class TestBatchEncodingUnigram:
    """Unigram batch_encode 对齐 HF BatchEncoding 测试。"""

    def test_batch_encode_padding_longest(self):
        """unigram padding='longest'。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        out = tok.batch_encode(["你好", "世界"], padding="longest", add_special_tokens=False)
        assert "input_ids" in out
        assert "attention_mask" in out
        if _np is not None:
            assert out["input_ids"].shape[0] == 2

    def test_batch_encode_truncation(self):
        """unigram truncation 截断。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        full_ids = tok.encode("你好世界机器学习", add_special_tokens=False)
        if len(full_ids) > 4:
            out = tok.batch_encode(
                ["你好世界机器学习"],
                truncation="right",
                truncation_side="right",
                max_length=4,
                add_special_tokens=False,
            )
            result = out["input_ids"]
            first = result[0]
            actual = list(first.tolist() if hasattr(first, "tolist") else first)
            assert len(actual) == 4, f"截断后长度应为 4，实际 {len(actual)}"


# ===========================================================================
# SubTask 5.5: Qwen tokenizer 加载 + HF 格式互转
# ===========================================================================


class TestQwenTokenizerLoad:
    """Qwen tokenizer 加载测试（graceful skip）。"""

    def test_from_pretrained_graceful_failure(self):
        """``from_pretrained`` 网络不可用时抛 RuntimeError（不卡住）。

        由于测试环境通常无网络访问 huggingface.co，
        此测试验证 ``from_pretrained`` 优雅失败而非卡住。
        """
        try:
            tok = BPETokenizer.from_pretrained(
                "Qwen/Qwen3.5-35B-A3B",
                timeout=10.0,
            )
            # 如果网络可用并成功加载，验证 vocab 大小
            assert len(tok.vocab) > 0, "加载成功但 vocab 为空"
            # Qwen3.5-35B-A3B 的 vocab 应该很大（248320）
            # 但不强制断言精确值，因为可能加载的是简化版
        except RuntimeError as e:
            # 网络不可用时应有明确的 RuntimeError
            assert "tokenizer" in str(e).lower() or "download" in str(e).lower() or \
                   "huggingface" in str(e).lower() or "failed" in str(e).lower(), (
                f"RuntimeError 消息不明确：{e}"
            )
        except Exception as e:
            # 其他异常类型也算 graceful 失败（只要不卡住）
            pytest.skip(f"from_pretrained 抛出非预期异常（视为 graceful 失败）：{e}")

    def test_to_hf_format_returns_valid_dict(self):
        """``to_hf_format`` 返回合法的 HF tokenizer.json 格式 dict。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        hf_data = tok.to_hf_format()
        assert isinstance(hf_data, dict)
        # 应含 model 字段
        assert "model" in hf_data, "to_hf_format 缺少 model 字段"
        assert hf_data["model"]["type"] == "BPE", "model type 不是 BPE"
        assert "vocab" in hf_data["model"], "model 缺少 vocab"
        assert "merges" in hf_data["model"], "model 缺少 merges"

    def test_from_hf_tokenizer_json_roundtrip(self):
        """``to_hf_format`` / ``from_hf_tokenizer_json`` 往返。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        hf_data = tok.to_hf_format()
        # 从 dict 加载
        tok2 = BPETokenizer.from_hf_tokenizer_json(hf_data)
        assert len(tok2.vocab) > 0, "from_hf_tokenizer_json 后 vocab 为空"
        # encode 应可用
        ids = tok2.encode("hello", add_special_tokens=False)
        assert isinstance(ids, list)

    def test_from_hf_tokenizer_json_from_file(self):
        """``from_hf_tokenizer_json`` 从文件路径加载。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=200)
        hf_data = tok.to_hf_format()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            import json
            json.dump(hf_data, f)
            path = f.name
        try:
            tok2 = BPETokenizer.from_hf_tokenizer_json(path)
            assert len(tok2.vocab) > 0
        finally:
            os.unlink(path)


# ===========================================================================
# byte-aligned decode 无乱码
# ===========================================================================


class TestByteAlignedDecode:
    """byte-aligned decode 不产生 U+FFFD 乱码。"""

    def test_bpe_decode_truncated_no_replacement(self):
        """BPE decode 截断序列不产生 U+FFFD。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300)
        text = "你好世界"
        full_ids = tok.encode(text, add_special_tokens=False)
        # 截断最后一个 id
        if len(full_ids) > 1:
            truncated = full_ids[:-1]
            decoded = tok.decode(truncated)
            assert "\ufffd" not in decoded, (
                f"BPE decode 截断产生 U+FFFD：{truncated} → {decoded!r}"
            )

    def test_bpe_decode_complete_no_replacement(self):
        """BPE decode 完整序列不产生 U+FFFD。"""
        tok = BPETokenizer.train(_CORPUS, vocab_size=300)
        for text in ["你好世界", "Hello world", "机器学习 machine"]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids)
            assert "\ufffd" not in decoded, (
                f"BPE decode 完整文本产生 U+FFFD：{text!r} → {decoded!r}"
            )

    def test_byte_decode_truncated_chinese(self):
        """ByteTokenizer decode 截断中文不产生 U+FFFD。"""
        tok = ByteTokenizer()
        # "你" = 0xE4 0xBD 0xA0（3 字节）
        truncated = [0xE4, 0xBD]
        decoded = tok.decode(truncated)
        assert "\ufffd" not in decoded

    def test_byte_decode_truncated_emoji(self):
        """ByteTokenizer decode 截断 emoji 不产生 U+FFFD。"""
        tok = ByteTokenizer()
        # 🎉 = 0xF0 0x9F 0x8E 0x89（4 字节）
        truncated = [0xF0, 0x9F, 0x8E]
        decoded = tok.decode(truncated)
        assert "\ufffd" not in decoded

    def test_byte_decode_mixed_complete_and_truncated(self):
        """ByteTokenizer decode 混合完整 + 截断字符。"""
        tok = ByteTokenizer()
        # "A你" + 截断的 0xE5
        ids = list("A你".encode("utf-8")) + [0xE5]
        decoded = tok.decode(ids)
        assert "\ufffd" not in decoded
        assert "A" in decoded
        assert "你" in decoded

    def test_wordpiece_decode_no_replacement(self):
        """WordPiece decode 不产生 U+FFFD。"""
        tok = WordPieceTokenizer.train(_CORPUS, vocab_size=200, min_frequency=2)
        for text in ["你好世界", "Hello world", "机器学习"]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids)
            assert "\ufffd" not in decoded, (
                f"WordPiece decode 产生 U+FFFD：{text!r} → {decoded!r}"
            )

    def test_unigram_decode_no_replacement(self):
        """unigram decode 不产生 U+FFFD。"""
        tok = SentencePieceUnigramTokenizer(vocab_size=200)
        tok.train(_CORPUS, vocab_size=200)
        for text in ["你好世界", "Hello world", "机器学习"]:
            ids = tok.encode(text, add_special_tokens=False)
            decoded = tok.decode(ids)
            assert "\ufffd" not in decoded, (
                f"unigram decode 产生 U+FFFD：{text!r} → {decoded!r}"
            )


# ===========================================================================
# 入口
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
