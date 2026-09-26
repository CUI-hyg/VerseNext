"""Byte-level BPE 分词器。

设计：
- 文本先按 UTF-8 转字节，基础词表为 256 个字节 token——任意文本可编码，无 OOV。
- 在字节序列上训练 BPE 合并规则（merges），高频字节对逐步合并。
- 特殊 token（bos/eos/pad/用户自定义）在编码入口处理，永不参与 BPE 切分。
- 持久化格式为自定义 ``tokenizer.json``（vocab + merges + 特殊 token），
  与 ``chat_template.jinja`` 一起构成 ``save_pretrained`` 目录（对齐 HF 习惯）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from verse_core import get_logger

logger = get_logger("tokenizer")

# 模块级正则：避免在 encode 热路径里重复编译
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\n]{1,64}\|>")
_WORD_RE = re.compile(rb"\s*\S+|\s+$")
# 词级 BPE 结果缓存上限（缓存以 UTF-8 字节串为键，命中率在真实语料上很高）
_BPE_CACHE_MAX = 100_000

# byte-level BPE 的可打印化表示（GPT-2 风格 bytes_to_unicode），保证 JSON/终端可见
def _bytes_to_unicode() -> dict[int, str]:
    """将 0-255 字节映射到可见 unicode 字符。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

# 特殊 token 别名（按优先级）：兼容框架命名与 HF 常见命名
BOS_ALIASES = ("<|bos|>", "<s>", "<|startoftext|>", "<|begin_of_text|>")
EOS_ALIASES = ("<|eos|>", "<|endoftext|>", "</s>", "<|end_of_text|>", "<|im_end|>")
PAD_ALIASES = ("<|pad|>", "<pad>", "<|padding|>", "[PAD]")


class BPETokenizer:
    """Byte-level BPE。

    用法::

        tok = BPETokenizer()
        tok.train(corpus_text, vocab_size=1000)
        ids = tok.encode("你好 world")
        text = tok.decode(ids)
        tok.save_pretrained("my_tokenizer/")  # tokenizer.json + chat_template.jinja
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        special_tokens: dict[str, int] | None = None,
        chat_template: str | None = None,
    ) -> None:
        self.special_tokens: dict[str, int] = special_tokens or {}
        self.chat_template = chat_template
        # 热路径缓存（见 encode/decode）：特殊 token 版本号驱动失效重建
        self._special_version = 0
        self._split_specials: tuple[str, ...] | None = None
        self._split_re: re.Pattern[str] | None = None
        self._special_id_cache: set[int] | None = None
        self._bpe_cache: dict[bytes, tuple[str, ...]] = {}
        self._token_bytes_cache: dict[str, bytes] = {}
        self._max_id = -1
        if vocab is not None and merges is not None:
            self._setup(vocab, merges)
        else:
            self._setup(self._base_vocab(), [])

    # ------------------------------------------------------------- structure

    def _base_vocab(self) -> dict[str, int]:
        """基础词表：256 个字节 token。特殊 token 在词表构建时统一追加。"""
        return {BYTE_ENCODER[b]: b for b in range(256)}

    def _setup(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        self.vocab = dict(vocab)
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = list(merges)
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        # 特殊 token 追加到词表尾部，避免与 BPE token 冲突
        for name, tok_id in self.special_tokens.items():
            self.vocab[name] = tok_id
            self.id_to_token[tok_id] = name
        self._max_id = max(self.vocab.values()) if self.vocab else -1
        self._invalidate_special_cache()

    def _invalidate_special_cache(self) -> None:
        """特殊 token 集合变化时失效相关缓存（正则/ID 集合）。"""
        self._special_version += 1
        self._split_specials = None
        self._split_re = None
        self._special_id_cache = None

    def _split_pattern(self) -> re.Pattern[str] | None:
        """特殊 token 切分正则（按长度降序，惰性构建 + 缓存）。"""
        if self._split_specials is None:
            self._split_specials = tuple(
                sorted(self.special_tokens, key=len, reverse=True)
            )
            if self._split_specials:
                self._split_re = re.compile(
                    "(" + "|".join(re.escape(s) for s in self._split_specials) + ")"
                )
        return self._split_re

    def _special_ids(self) -> set[int]:
        if self._special_id_cache is None:
            self._special_id_cache = set(self.special_tokens.values())
        return self._special_id_cache

    def _token_bytes(self, token: str) -> bytes:
        """token 字符串 -> 原始字节（带缓存，避免逐字符查表）。"""
        cached = self._token_bytes_cache.get(token)
        if cached is not None:
            return cached
        out = bytearray()
        for ch in token:
            byte = BYTE_DECODER.get(ch)
            if byte is not None:
                out.append(byte)
        result = bytes(out)
        self._token_bytes_cache[token] = result
        return result

    # ------------------------------------------------------------------ train

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def bos_token_id(self) -> int | None:
        return self._first_special(BOS_ALIASES)

    @property
    def eos_token_id(self) -> int | None:
        return self._first_special(EOS_ALIASES)

    @property
    def pad_token_id(self) -> int | None:
        return self._first_special(PAD_ALIASES)

    def _first_special(self, names: tuple[str, ...]) -> int | None:
        """按别名优先级返回第一个已注册的特殊 token id。

        兼容不同来源的分词器：框架自带 ``<|bos|>`` 与 HF 风格
        ``<|endoftext|>`` / ``<s>`` 等，避免下游（如 SFT padding、
        generate 停止符）因缺少固定命名而拿不到 id。
        """
        for name in names:
            tok_id = self.special_tokens.get(name)
            if tok_id is not None:
                return tok_id
        return None

    def train(
        self,
        corpus: str | list[str],
        vocab_size: int = 2048,
        special_tokens: list[str] | None = None,
        min_frequency: int = 2,
    ) -> None:
        """在语料上训练 BPE 合并规则。

        Args:
            corpus: 语料文本或文档列表。
            vocab_size: 目标词表大小（含 256 字节基础 + 特殊 token）。
            special_tokens: 特殊 token，前三个自动为 ``<|bos|> <|eos|> <|pad|>``。
            min_frequency: 合并的最小出现次数。
        """
        if vocab_size < 260:
            raise ValueError("vocab_size 至少 260（256 字节 + 特殊 token）")
        specials = list(dict.fromkeys(["<|bos|>", "<|eos|>", "<|pad|>"] + (special_tokens or [])))

        # 预分词：按文档切成字节 word 序列（带频率），特殊 token 位置直接挖掉
        word_freqs: Counter[str] = Counter()
        docs = [corpus] if isinstance(corpus, str) else corpus
        import re

        pattern = "(" + "|".join(re.escape(s) for s in specials) + ")" if specials else None
        for doc in docs:
            parts = re.split(pattern, doc) if pattern else [doc]
            for part in parts:
                if not part or part in specials:
                    continue
                # 空格归并到词首（GPT-2 风格简化：不引入完整正则预分词）
                for word in part.split(" "):
                    if word:
                        word_freqs[" ".join(BYTE_ENCODER[b] for b in word.encode("utf-8"))] += 1

        # 每个词是符号列表
        words: list[list[str]] = []
        freqs: list[int] = []
        for w, f in word_freqs.items():
            words.append(w.split(" "))
            freqs.append(f)

        merges: list[tuple[str, str]] = []
        num_merges = vocab_size - 256 - len(specials)
        for i in range(num_merges):
            pair_freqs: Counter[tuple[str, str]] = Counter()
            for word, f in zip(words, freqs):
                for j in range(len(word) - 1):
                    pair_freqs[(word[j], word[j + 1])] += f
            if not pair_freqs:
                break
            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < min_frequency:
                break
            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            # 就地合并
            for word in words:
                j = 0
                while j < len(word) - 1:
                    if (word[j], word[j + 1]) == best_pair:
                        word[j:j + 2] = [new_token]
                    else:
                        j += 1

        special_map = {}
        next_id = 256 + len(merges)
        for name in specials:
            special_map[name] = next_id
            next_id += 1
        # 重建连续词表：字节 token -> merges -> 特殊 token
        final_vocab: dict[str, int] = self._base_vocab()
        for idx, (a, b) in enumerate(merges):
            final_vocab[a + b] = 256 + idx
        final_vocab.update(special_map)

        self.special_tokens = special_map
        self._setup(final_vocab, merges)
        logger.info(
            "BPE 训练完成：%d merges，词表 %d（目标 %d）", len(merges), self.vocab_size, vocab_size
        )

    # ----------------------------------------------------------------- encode

    def _bpe(self, symbols: list[str]) -> list[str]:
        """对符号序列反复应用最优合并。"""
        if len(symbols) < 2:
            return symbols
        while True:
            best_rank, best_pair = None, None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (symbols[i], symbols[i + 1])
            if best_pair is None:
                return symbols
            merged = best_pair[0] + best_pair[1]
            out, j = [], 0
            while j < len(symbols):
                if j < len(symbols) - 1 and (symbols[j], symbols[j + 1]) == best_pair:
                    out.append(merged)
                    j += 2
                else:
                    out.append(symbols[j])
                    j += 1
            symbols = out

    def _bpe_cached(self, raw: bytes) -> tuple[str, ...]:
        """词级 BPE（带缓存）：真实语料中词高度重复，缓存命中率高。"""
        cached = self._bpe_cache.get(raw)
        if cached is not None:
            return cached
        symbols = [BYTE_ENCODER[b] for b in raw]
        result = tuple(self._bpe(symbols))
        if len(self._bpe_cache) >= _BPE_CACHE_MAX:
            self._bpe_cache.clear()
        self._bpe_cache[raw] = result
        return result

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets: bool = False,
        auto_register_special: bool = True,
    ) -> list[int] | tuple[list[int], list[tuple[int, int]]]:
        """文本 -> token ids。

        Args:
            text: 输入文本（可含特殊 token 字符串）。
            add_special_tokens: 是否在开头追加 bos。
            return_offsets: True 时同时返回每个 token 的 ``(byte_start, byte_end)``
                字节偏移（用于 SFT loss 掩码等场景）。
            auto_register_special: 自动把文本中形如 ``<|name|>`` 的未知标记
                注册为特殊 token（让自定义 chat 模板开箱即用）。
        """
        # 仅在文本确实含 "<|" 时才扫描，避免每个普通文本都跑正则
        if auto_register_special and "<|" in text:
            for marker in set(_SPECIAL_TOKEN_RE.findall(text)):
                if marker not in self.special_tokens:
                    self.add_special_token(marker)

        ids, offsets = self._encode_core(text)
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        if return_offsets:
            return ids, offsets
        return ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = False,
        return_offsets: bool = False,
    ) -> list:
        """批量编码（复用热路径缓存，避免逐条重复建正则/查表）。"""
        return [
            self.encode(
                text, add_special_tokens=add_special_tokens,
                return_offsets=return_offsets,
            )
            for text in texts
        ]

    def _encode_core(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """encode 主体：按特殊 token 切分后逐词 BPE（返回 ids 与字节偏移）。"""
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        pattern = self._split_pattern()
        parts = pattern.split(text) if pattern is not None else [text]
        byte_pos = 0
        for part in parts:
            if not part:
                continue
            special_id = self.special_tokens.get(part)
            if special_id is not None:
                part_len = len(part.encode("utf-8"))
                ids.append(special_id)
                offsets.append((byte_pos, byte_pos + part_len))
                byte_pos += part_len
                continue
            raw_part = part.encode("utf-8")
            # 词 = 可选前导空白 + 非空白串（保留空格信息，decode 可还原原文）
            for match in _WORD_RE.finditer(raw_part):
                word_start = byte_pos + match.start()
                consumed = 0
                for sym in self._bpe_cached(match.group()):
                    ids.append(self.vocab[sym])
                    # 每个 byte-unicode 字符恰好对应 1 个原始字节
                    sym_len = len(sym)
                    offsets.append((word_start + consumed, word_start + consumed + sym_len))
                    consumed += sym_len
            byte_pos += len(raw_part)
        return ids, offsets

    def add_special_token(self, token: str) -> int:
        """注册一个特殊 token，返回其 id。"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        self._max_id += 1
        new_id = self._max_id
        self.special_tokens[token] = new_id
        self.vocab[token] = new_id
        self.id_to_token[new_id] = token
        self._invalidate_special_cache()
        return new_id

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """token ids -> 文本（token 级字节缓存 + 批量 join，避免逐字符查表）。"""
        special_ids = self._special_ids() if skip_special_tokens else None
        parts: list[bytes] = []
        for token_id in ids:
            if special_ids is not None and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id)
            if token is not None:
                parts.append(self._token_bytes(token))
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence

    def save_pretrained(self, directory: str | Path) -> Path:
        """保存 ``tokenizer.json`` + ``chat_template.jinja``（HF 目录风格）。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "verse_bpe",
            "vocab": self.vocab,
            "merges": [" ".join(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if self.chat_template:
            (directory / "chat_template.jinja").write_text(self.chat_template)
        return directory

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "BPETokenizer":
        directory = Path(directory)
        payload = json.loads((directory / "tokenizer.json").read_text())
        if payload.get("model_type") != "verse_bpe":
            raise ValueError(f"不支持的分词器格式: {payload.get('model_type')}")
        chat_template = None
        tpl_path = directory / "chat_template.jinja"
        if tpl_path.exists():
            chat_template = tpl_path.read_text()
        tok = cls(
            vocab=payload["vocab"],
            merges=[tuple(m.split(" ")) for m in payload["merges"]],
            special_tokens=payload["special_tokens"],
            chat_template=chat_template,
        )
        tok._tokenizer_dir = str(directory)
        return tok
