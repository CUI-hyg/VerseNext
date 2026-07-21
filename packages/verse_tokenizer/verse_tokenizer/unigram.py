"""Unigram tokenizer：基于 unigram 语言模型 + Viterbi 解码。

算法概述
--------
SentencePiece Unigram 模型给每个 piece（subword）分配一个概率，
编码时用 Viterbi 算法找到概率最大的分割。

训练（简化版 EM）：
1. 初始：用 pre_tokenize 切分语料，所有 subword（每个 piece 的所有前缀）按频率初始化概率；
2. E 步：用 Viterbi 找最优分割，累计每个 piece 的频次；
3. M 步：重新估计每个 piece 的概率（频次归一化）；
4. 迭代 5 轮；
5. 保留 top-K vocab_size 个 piece（按频次排序），特殊 token 必留。

接口与 :class:`verse_tokenizer.bpe.BPETokenizer` 对齐：
``train`` / ``encode`` / ``decode`` / ``apply_chat_template`` /
``apply_prompt_template`` / ``save`` / ``load`` / ``__len__``。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Optional

from .preprocess import pre_tokenize
from .chat_template import render_chat, render_prompt


# ---------------------------------------------------------------------------
# 特殊 token
# ---------------------------------------------------------------------------

SpecialTokens = {
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "pad": "<|pad|>",
    "unk": "<|unk|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
}


# ---------------------------------------------------------------------------
# SentencePieceUnigramTokenizer
# ---------------------------------------------------------------------------


class SentencePieceUnigramTokenizer:
    """SentencePiece Unigram 分词器。

    基于 unigram 语言模型：每个 piece 有一个概率（log 形式），
    编码时用 Viterbi 找到概率最大的分割。

    Args:
        vocab_size: 目标词表大小（含特殊 token）
        special_tokens: 特殊 token 字典，默认为 :data:`SpecialTokens`
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        special_tokens: Optional[dict] = None,
        add_special_tokens: bool = True,
    ):
        self.vocab_size = vocab_size
        self.special_tokens = dict(special_tokens) if special_tokens else SpecialTokens.copy()
        # piece -> id, id -> piece
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}
        # piece 概率（log）
        self.piece_log_prob: dict[str, float] = {}
        # 特殊 token 占用 id 0~N-1
        self._init_special_tokens()
        # 编码时是否自动加 bos/eos
        self.add_special_tokens = add_special_tokens
        # 缓存的 Viterbi 频率表（encode 时用），train 后失效
        self._encode_freq_cache: Optional[dict[str, float]] = None
        # 编译好的特殊 token 切分正则（按长度降序匹配）
        self._special_split_re: Optional[re.Pattern] = None
        self._build_special_split_re()

    # ------------------------------------------------------------------
    # 特殊 token 初始化
    # ------------------------------------------------------------------

    def _init_special_tokens(self) -> None:
        """把 special_tokens 字典中的字符串注册到 vocab（id 0~N-1）。"""
        for name, tok in self.special_tokens.items():
            if tok not in self.piece_to_id:
                idx = len(self.piece_to_id)
                self.piece_to_id[tok] = idx
                self.id_to_piece[idx] = tok
                # 特殊 token 不参与概率（设为 0.0，对应概率 1.0，但不影响普通 piece）
                self.piece_log_prob[tok] = 0.0

    def _build_special_split_re(self) -> None:
        """构造用于切分特殊 token 的正则（按长度降序，避免短 token 抢匹配）。"""
        toks = sorted(self.special_tokens.values(), key=len, reverse=True)
        if toks:
            self._special_split_re = re.compile(
                "(" + "|".join(re.escape(t) for t in toks) + ")"
            )
        else:
            self._special_split_re = None

    # ------------------------------------------------------------------
    # 训练（EM）
    # ------------------------------------------------------------------

    def train(self, corpus, vocab_size: Optional[int] = None) -> "SentencePieceUnigramTokenizer":
        """EM 训练 unigram 模型。

        Args:
            corpus: ``str`` 或 ``list[str]``（原始文本）
            vocab_size: 目标词表大小（含特殊 token）；若数据太少达不到会自动回退

        Returns:
            self（链式调用）
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        # 1. 预分词
        if isinstance(corpus, str):
            corpus = [corpus]
        all_pieces: list[str] = []
        for text in corpus:
            all_pieces.extend(pre_tokenize(text))

        # 2. 初始频次：每个 piece 的所有前缀
        piece_freq: Counter = Counter()
        for piece in all_pieces:
            for i in range(1, len(piece) + 1):
                piece_freq[piece[:i]] += 1
        # 特殊 token 也算进去（保证 Viterbi 能命中）
        for tok in self.special_tokens.values():
            if tok not in piece_freq:
                piece_freq[tok] = 1
        total = sum(piece_freq.values())
        if total == 0:
            # 空语料兜底
            piece_freq[" "] = 1
            total = 1

        # 3. EM 迭代（5 轮）
        for _ in range(5):
            new_freq: Counter = Counter()
            for piece in all_pieces:
                best_path = self._viterbi(piece, piece_freq, total)
                if best_path:
                    for p in best_path:
                        new_freq[p] += 1
            # 特殊 token 频次保留为 1（不参与 EM，但保证在 vocab 中）
            for tok in self.special_tokens.values():
                if tok not in new_freq:
                    new_freq[tok] = 1
            if sum(new_freq.values()) > 0:
                piece_freq = new_freq
                total = sum(new_freq.values())

        # 4. 保留 top-K vocab_size 个 piece（按频次降序）
        special_set = set(self.special_tokens.values())
        # 特殊 token 必留
        keep_special = [(t, piece_freq.get(t, 1)) for t in self.special_tokens.values()]
        # 普通 piece 按频次排序，预留 special 的位置
        normal_pieces = [
            (p, f) for p, f in piece_freq.items() if p not in special_set
        ]
        normal_pieces.sort(key=lambda x: -x[1])
        remaining_slots = max(0, self.vocab_size - len(keep_special))
        keep_normal = normal_pieces[:remaining_slots]
        keep = keep_special + keep_normal

        # 5. 构建 vocab
        self.piece_to_id = {}
        self.id_to_piece = {}
        self.piece_log_prob = {}
        self._init_special_tokens()
        # 重新计算 total（仅含保留的 piece）
        kept_total = sum(f for _, f in keep)
        if kept_total <= 0:
            kept_total = 1
        for p, f in keep:
            if p not in self.piece_to_id:
                idx = len(self.piece_to_id)
                self.piece_to_id[p] = idx
                self.id_to_piece[idx] = p
                self.piece_log_prob[p] = math.log(f / kept_total) if f > 0 else -math.inf

        # 失效缓存
        self._encode_freq_cache = None
        self._build_special_split_re()
        return self

    # ------------------------------------------------------------------
    # Viterbi 解码
    # ------------------------------------------------------------------

    def _viterbi(
        self,
        text: str,
        piece_freq: dict,
        total: int,
        max_piece_len: int = 16,
    ) -> Optional[list[str]]:
        """Viterbi 解码：找概率最大的 piece 分割。

        Args:
            text: 待分割的字符串（一个 pre-tokenize piece）
            piece_freq: piece -> 频次 的 dict
            total: 所有频次之和（用于计算概率）
            max_piece_len: 单个 piece 最大长度（限制搜索范围）

        Returns:
            最优分割的 piece 列表；若无法分割（所有字符都不在词表）返回 None
        """
        n = len(text)
        if n == 0:
            return []
        # dp[i] = (best_score, best_path) 到位置 i
        dp: list[tuple[float, Optional[list[str]]]] = [(-math.inf, None)] * (n + 1)
        dp[0] = (0.0, [])
        for i in range(1, n + 1):
            j_start = max(0, i - max_piece_len)
            for j in range(j_start, i):
                if dp[j][1] is None:
                    continue
                piece = text[j:i]
                f = piece_freq.get(piece)
                if f is None or f <= 0:
                    continue
                log_p = math.log(f / total) if total > 0 else 0.0
                score = dp[j][0] + log_p
                if score > dp[i][0]:
                    dp[i] = (score, dp[j][1] + [piece])
        return dp[n][1]

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def _piece_freq_for_encode(self) -> dict[str, float]:
        """返回 piece -> 概率 的 dict（用于 Viterbi），带缓存。"""
        if self._encode_freq_cache is None:
            self._encode_freq_cache = {
                p: math.exp(lp) for p, lp in self.piece_log_prob.items()
                if lp != -math.inf and lp != float("-inf")
            }
        return self._encode_freq_cache

    def _encode_piece(self, piece: str, freq: dict[str, float]) -> list[int]:
        """对一个 pre-tokenize piece 做 Viterbi 分割并转为 id 序列。"""
        if not piece:
            return []
        best_path = self._viterbi(piece, freq, 1)
        unk_id = self.piece_to_id.get(self.special_tokens["unk"], 0)
        if best_path is None:
            # 无法分割，逐字符 fallback（已知字符用其 id，未知用 unk）
            ids: list[int] = []
            for ch in piece:
                if ch in self.piece_to_id:
                    ids.append(self.piece_to_id[ch])
                else:
                    ids.append(unk_id)
            return ids
        ids = []
        for p in best_path:
            if p in self.piece_to_id:
                ids.append(self.piece_to_id[p])
            else:
                ids.append(unk_id)
        return ids

    def encode(
        self,
        text: str,
        add_special_tokens: Optional[bool] = None,
    ) -> list[int]:
        """Viterbi 解码：找概率最大的 piece 分割，转为 id 序列。

        Args:
            text: 输入文本
            add_special_tokens: 是否在首尾加 bos/eos；None 时用 self.add_special_tokens
        """
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens

        ids: list[int] = []
        if add_special_tokens:
            bos_id = self.piece_to_id.get(self.special_tokens["bos"])
            if bos_id is not None:
                ids.append(bos_id)

        freq = self._piece_freq_for_encode()
        # 先按特殊 token 切分，特殊 token 作为单 token，其余 pre_tokenize + Viterbi
        if self._special_split_re is not None:
            chunks = self._special_split_re.split(text)
        else:
            chunks = [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens.values():
                if chunk in self.piece_to_id:
                    ids.append(self.piece_to_id[chunk])
                continue
            # 普通 chunk：pre_tokenize 后逐段 Viterbi
            for piece in pre_tokenize(chunk):
                ids.extend(self._encode_piece(piece, freq))

        if add_special_tokens:
            eos_id = self.piece_to_id.get(self.special_tokens["eos"])
            if eos_id is not None:
                ids.append(eos_id)
        return ids

    def decode(self, ids) -> str:
        """解码 id 序列为字符串。

        - 特殊 token（bos/eos/pad/unk/user/assistant/system）不输出到文本；
        - 其余 piece 直接拼接。
        """
        special_set = set(self.special_tokens.values())
        pieces: list[str] = []
        for i in ids:
            i = int(i)
            p = self.id_to_piece.get(i)
            if p is None:
                continue
            if p in special_set:
                # 特殊 token 跳过（不输出到文本）
                continue
            pieces.append(p)
        return "".join(pieces)

    # ------------------------------------------------------------------
    # chat template
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """渲染 chat 数组并编码（不加首尾 bos/eos，因为 render_chat 已含 <|eos|>）。"""
        rendered = render_chat(messages)
        return self.encode(rendered, add_special_tokens=False)

    def apply_prompt_template(self, prompt: str) -> list[int]:
        """渲染 prompt 并编码（不加首尾 bos/eos，prompt 模板用于推理前缀）。"""
        rendered = render_prompt(prompt)
        return self.encode(rendered, add_special_tokens=False)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存到 JSON 文件。"""
        data = {
            "type": "unigram",
            "vocab_size": self.vocab_size,
            "piece_to_id": self.piece_to_id,
            "piece_log_prob": self.piece_log_prob,
            "special_tokens": self.special_tokens,
            "add_special_tokens": self.add_special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> "SentencePieceUnigramTokenizer":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.piece_to_id = data["piece_to_id"]
        self.piece_log_prob = data["piece_log_prob"]
        self.special_tokens = data["special_tokens"]
        self.add_special_tokens = data.get("add_special_tokens", True)
        self.id_to_piece = {int(v): k for k, v in self.piece_to_id.items()}
        # JSON 反序列化后 key 可能是 str，id 需要 int
        self.piece_to_id = {k: int(v) for k, v in self.piece_to_id.items()}
        self._encode_freq_cache = None
        self._build_special_split_re()
        return self

    def __len__(self) -> int:
        return len(self.piece_to_id)


__all__ = ["SentencePieceUnigramTokenizer", "SpecialTokens"]
