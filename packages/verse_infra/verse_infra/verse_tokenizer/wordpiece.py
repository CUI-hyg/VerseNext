"""WordPiece tokenizer：BERT 风格 WordPiece 分词器。

设计目标
--------
实现一个自包含、零依赖的 WordPiece 分词器，与 BERT / DistilBERT 等
HuggingFace 模型使用的 WordPiece 算法对齐：

- 词中子词用 ``##`` 前缀表示（如 ``"playing"`` → ``["play", "##ing"]``）；
- 编码采用「最长前缀贪心匹配」：从当前位置开始，找词表中最长的子串，
  若找不到则前进一步并加上 ``##`` 前缀继续匹配；
- 训练采用「贪心最大子串」算法：每次从所有词的所有子串中选频率最高的加入词表，
  直到达成 vocab_size。

接口与 :class:`verse_tokenizer.bpe.BPETokenizer` 对齐：
``train`` / ``encode`` / ``decode`` / ``apply_chat_template`` /
``apply_prompt_template`` / ``save`` / ``load`` / ``__len__``。

特殊 token
----------
默认注册 ``[PAD]`` / ``[UNK]`` / ``[CLS]`` / ``[SEP]`` / ``[MASK]``（BERT 风格）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional

from .bpe import BaseTokenizer, DEFAULT_SPECIAL_TOKENS
from .preprocess import pre_tokenize as _gpt4_pre_tokenize, nfkc_normalize
from .chat_template import render_chat as _render_chat, render_prompt as _render_prompt


# ---------------------------------------------------------------------------
# BERT 风格默认特殊 token
# ---------------------------------------------------------------------------

# WordPiece 默认使用方括号风格特殊 token（BERT 风格），
# 同时保留对 ``<bos>`` / ``<eos>`` 的兼容（与 BPETokenizer 共用）
WORDPIECE_DEFAULT_SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "<bos>", "<eos>", "<pad>", "<unk>",
]

# 子词前缀（``##`` 表示「词中续接」，BERT 标准约定）
CONTINUING_SUBWORD_PREFIX = "##"


# ---------------------------------------------------------------------------
# WordPieceTokenizer
# ---------------------------------------------------------------------------


class WordPieceTokenizer(BaseTokenizer):
    """BERT 风格 WordPiece 分词器。

    Args:
        vocab: ``{token_str: id}`` 词表。token 可能含 ``##`` 前缀。
        special_tokens: 特殊 token（``list[str]`` 或 ``dict[str, int]``）
        unk_token: 未知 token 字符串（默认 ``"[UNK]"``）
        continuing_subword_prefix: 词中续接前缀（默认 ``"##"``）
        max_input_chars_per_word: 单词最大字符数；超过则整体当 ``unk``
            （BERT 默认 100，防止恶意超长单词卡死）
        add_special_tokens: encode 时是否默认加 ``[CLS]`` / ``[SEP]``

    Task 5.2 升级：
        - 新增 WordPiece 实现，与 :class:`BPETokenizer` 接口对齐；
        - 训练采用贪心最大子串算法（频率最高的子串优先加入词表）；
        - encode 采用最长前缀贪心匹配（与 BERT ``WordPieceTokenizer`` 一致）；
        - 继承 :class:`BaseTokenizer`，自动获得 ``encode_batch`` / ``decode_batch``。
    """

    def __init__(
        self,
        vocab: dict,
        special_tokens: Optional = None,
        unk_token: str = "[UNK]",
        continuing_subword_prefix: str = CONTINUING_SUBWORD_PREFIX,
        max_input_chars_per_word: int = 100,
        add_special_tokens: bool = True,
    ):
        self.vocab = dict(vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        # 特殊 token 字典
        self.special_tokens: dict[str, int] = {}
        if special_tokens is None:
            specials_list = []
        elif isinstance(special_tokens, dict):
            specials_list = list(special_tokens.keys())
        else:
            specials_list = list(special_tokens)
        for st in specials_list:
            if st not in self.vocab:
                self.vocab[st] = len(self.vocab)
                self.id_to_token[self.vocab[st]] = st
            self.special_tokens[st] = self.vocab[st]
        # 构造参数：encode 时是否默认加 [CLS]/[SEP]
        self.auto_add_special_tokens = add_special_tokens
        # 兼容属性（与 BPETokenizer 对齐）
        self.byte_level = False
        self.add_bos = bool(add_special_tokens)
        self.add_eos = bool(add_special_tokens)
        # 缓存常用 id
        self.unk_id: Optional[int] = self.vocab.get(unk_token)
        self.pad_id: Optional[int] = self.vocab.get("[PAD]")
        self.cls_id: Optional[int] = self.vocab.get("[CLS]")
        self.sep_id: Optional[int] = self.vocab.get("[SEP]")

    # ------------------------------------------------------------------
    # 训练（贪心最大子串）
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus,
        vocab_size: int,
        min_frequency: int = 2,
        limit_alphabet: Optional[int] = None,
        special_tokens: Optional[list] = None,
    ) -> "WordPieceTokenizer":
        """从语料训练 WordPiece 词表。

        算法（贪心最大子串，与 BERT 训练脚本对齐）：
            1. pre-tokenize 语料，得到词列表（含空格分离的 token）；
            2. 统计每个词的频次；
            3. 初始化字母表（所有出现过的字符）+ special tokens；
            4. 对每个词，统计其所有子串的频次（加权 = 词频次）；
            5. 重复合并：
                - 选择频率最高的子串加入词表；
                - 频率 = sum(包含该子串的词的频次)；
            6. 直到 vocab_size 达到或无子串可加。

        Args:
            corpus: ``str`` 或 ``list[str]``
            vocab_size: 目标词表大小（含特殊 token）
            min_frequency: 子串最小频次（默认 2，BERT 默认）
            limit_alphabet: 字母表大小上限（``None`` 不限）
            special_tokens: 自定义特殊 token；默认用
                :data:`WORDPIECE_DEFAULT_SPECIAL_TOKENS`

        Returns:
            训练好的 :class:`WordPieceTokenizer` 实例
        """
        # 1. 统一为单一字符串
        if isinstance(corpus, (list, tuple)):
            text = "\n".join(str(c) for c in corpus)
        else:
            text = str(corpus)

        # 2. pre-tokenize：用 GPT-4 风格切词（中文整字、英文单词、数字、标点）
        pieces = _gpt4_pre_tokenize(text)
        # 词频统计（保留前导空格用于 wordpiece，但 strip 后存储）
        word_freq: Counter = Counter()
        for p in pieces:
            if not p:
                continue
            # 去除前导空格（WordPiece 输入约定：单词间用空格分隔）
            w = p.strip()
            if w:
                word_freq[w] += 1

        # 3. 字母表：所有出现过的字符（用于初始化词表）
        alphabet = set()
        for w in word_freq:
            for ch in w:
                alphabet.add(ch)
        if limit_alphabet is not None and len(alphabet) > limit_alphabet:
            # 按频次排序取 top-N
            char_freq: Counter = Counter()
            for w, f in word_freq.items():
                for ch in w:
                    char_freq[ch] += f
            alphabet = set(
                ch for ch, _ in char_freq.most_common(limit_alphabet)
            )

        # 4. 特殊 token
        if special_tokens is None:
            special_tokens = list(WORDPIECE_DEFAULT_SPECIAL_TOKENS)

        # 5. 初始化词表：先放特殊 token，再放字母表
        vocab: dict[str, int] = {}
        for st in special_tokens:
            if st not in vocab:
                vocab[st] = len(vocab)
        # 字母表按字典序排列（保证可复现）
        for ch in sorted(alphabet):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        # 6. 统计子串频次（用「贪心最大子串」训练）
        # 对每个词 w，统计其所有长度 ≥ 2 的子串频次（加权 = 词频次）
        # 子串在词中的位置决定前缀：
        #   - 词首位置：无前缀
        #   - 词中位置：加 ``##`` 前缀
        # 这样保证 encode 时的贪心匹配能命中正确子串。
        def _all_substrings_with_prefix(w: str) -> list[str]:
            """生成词 w 的所有子串，按位置加 ``##`` 前缀。"""
            subs = []
            n = len(w)
            for start in range(n):
                for end in range(start + 1, n + 1):
                    sub = w[start:end]
                    if start == 0:
                        # 词首：无前缀
                        subs.append(sub)
                    else:
                        # 词中：加 ## 前缀
                        subs.append(CONTINUING_SUBWORD_PREFIX + sub)
            return subs

        substring_freq: Counter = Counter()
        for w, f in word_freq.items():
            if len(w) > 100:
                # 超长单词跳过（防止训练卡死）
                continue
            for sub in _all_substrings_with_prefix(w):
                substring_freq[sub] += f

        # 7. 贪心加入最高频子串
        # 已加入的子串集合（避免重复加）
        added: set = set(vocab.keys())
        # 按频次降序排序子串候选（保证可复现：同频按字典序）
        candidates = sorted(
            ((s, f) for s, f in substring_freq.items() if f >= min_frequency and s not in added),
            key=lambda x: (-x[1], x[0]),
        )
        idx = 0
        while len(vocab) < vocab_size and idx < len(candidates):
            sub, freq = candidates[idx]
            idx += 1
            if sub in vocab:
                continue
            if freq < min_frequency:
                continue
            vocab[sub] = len(vocab)
            added.add(sub)

        # 8. 创建实例
        instance = cls(vocab, special_tokens=special_tokens)
        return instance

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: Optional[bool] = None,
    ) -> list[int]:
        """WordPiece 编码：最长前缀贪心匹配。

        Args:
            text: 输入文本
            add_special_tokens:
                - ``True``：首尾加 ``[CLS]`` / ``[SEP]``（如果存在）
                - ``False``：不加
                - ``None``：用 ``self.auto_add_special_tokens`` 默认值
        """
        # 前置 NFKC 正规化 + 控制字符去除（BaseTokenizer.preprocess）
        text = self.preprocess(text)
        if add_special_tokens is None:
            add_special_tokens = self.auto_add_special_tokens

        ids: list[int] = []

        # 先按特殊 token 切分（特殊 token 直接作为单 token）
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pat = re.compile("(" + "|".join(re.escape(s) for s in sorted_specials) + ")")
            chunks = pat.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens and chunk in self.vocab:
                ids.append(self.vocab[chunk])
                continue
            # 普通 chunk：pre-tokenize 后逐词 WordPiece
            for piece in _gpt4_pre_tokenize(chunk):
                ids.extend(self._encode_word(piece.strip()))

        if add_special_tokens:
            # 首尾加 [CLS] / [SEP]（BERT 风格）
            cls_id = self.vocab.get("[CLS]")
            if cls_id is not None and (not ids or ids[0] != cls_id):
                ids.insert(0, cls_id)
            sep_id = self.vocab.get("[SEP]")
            if sep_id is not None and (not ids or ids[-1] != sep_id):
                ids.append(sep_id)

        return ids

    def _encode_word(self, word: str) -> list[int]:
        """对单个词做 WordPiece 切分（最长前缀贪心匹配）。

        算法（与 BERT WordPiece 一致）：
            1. 若 word 为空，返回空；
            2. 若 word 字符数超过 max_input_chars_per_word，整体当 unk；
            3. 从 start=0 开始：
                - 在词表中找最长的子串 ``word[start:end]``（end 从最大向最小收缩）；
                - 词首位置不加前缀；词中位置加 ``##`` 前缀；
                - 找到则加入 tokens，start = end；
                - 找不到则标记为 unk，跳出。
        """
        if not word:
            return []
        if len(word) > self.max_input_chars_per_word:
            # 超长单词整体当 unk
            return [self.unk_id] if self.unk_id is not None else []

        tokens: list[int] = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            cur_token = None
            cur_id = None
            while start < end:
                sub = word[start:end]
                # 词首位置不加前缀；词中位置加 ## 前缀
                candidate = sub if start == 0 else self.continuing_subword_prefix + sub
                if candidate in self.vocab:
                    cur_token = candidate
                    cur_id = self.vocab[candidate]
                    break
                end -= 1
            if cur_id is None:
                # 找不到任何子串，整体当 unk
                return [self.unk_id] if self.unk_id is not None else []
            tokens.append(cur_id)
            start = end
        return tokens

    def decode(self, ids: list[int]) -> str:
        """解码 id 序列为字符串。

        - 特殊 token（``[CLS]`` / ``[SEP]`` / ``[PAD]`` / ``[UNK]`` / ``[MASK]``
          及 ``<bos>`` / ``<eos>`` 等）不输出到文本；
        - 普通 token：
            - ``##`` 前缀 token 与前一个 token 直接拼接（无空格）；
            - 非 ``##`` 前缀 token 前加空格（除非是第一个 token）。
        """
        special_set = set(self.special_tokens)
        out: list[str] = []
        is_first = True
        for i in ids:
            tok = self.id_to_token.get(int(i))
            if tok is None:
                continue
            if tok in special_set:
                # 特殊 token 跳过（不输出到文本）
                # 但 [SEP] 之后视作新词开始
                if tok == "[SEP]":
                    is_first = True
                continue
            if tok.startswith(self.continuing_subword_prefix):
                # 词中续接：去掉前缀，直接拼接
                out.append(tok[len(self.continuing_subword_prefix):])
            else:
                # 词首：前加空格（除非是第一个 token）
                if not is_first:
                    out.append(" ")
                out.append(tok)
                is_first = False
        return "".join(out)

    # ------------------------------------------------------------------
    # chat template（继承自 BaseTokenizer，覆盖以加 [CLS]/[SEP]）
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """渲染 chat 数组并编码（不加 [CLS]/[SEP]，因为 render_chat 已含 <|eos|>）。"""
        rendered = _render_chat(messages)
        return self.encode(rendered, add_special_tokens=False)

    def apply_prompt_template(self, prompt: str) -> list[int]:
        """渲染 prompt 并编码（不加 [CLS]/[SEP]）。"""
        rendered = _render_prompt(prompt)
        return self.encode(rendered, add_special_tokens=False)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """序列化为 JSON 文件。"""
        data = {
            "type": "wordpiece",
            "vocab": self.vocab,
            "special_tokens": dict(self.special_tokens),
            "unk_token": self.unk_token,
            "continuing_subword_prefix": self.continuing_subword_prefix,
            "max_input_chars_per_word": self.max_input_chars_per_word,
            "add_special_tokens": self.auto_add_special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordPieceTokenizer":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "wordpiece":
            raise ValueError(
                f"Not a WordPieceTokenizer JSON file (type={data.get('type')!r})"
            )
        vocab_raw = data.get("vocab", {})
        vocab = {k: int(v) for k, v in vocab_raw.items()}
        special_tokens = data.get("special_tokens")
        return cls(
            vocab=vocab,
            special_tokens=special_tokens,
            unk_token=data.get("unk_token", "[UNK]"),
            continuing_subword_prefix=data.get(
                "continuing_subword_prefix", CONTINUING_SUBWORD_PREFIX
            ),
            max_input_chars_per_word=data.get("max_input_chars_per_word", 100),
            add_special_tokens=data.get("add_special_tokens", True),
        )

    def __len__(self) -> int:
        return len(self.vocab)


__all__ = [
    "WordPieceTokenizer",
    "WORDPIECE_DEFAULT_SPECIAL_TOKENS",
    "CONTINUING_SUBWORD_PREFIX",
]
