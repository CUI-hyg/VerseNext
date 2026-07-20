"""Task 5.3: 最小 BPE 分词器，可加载 HuggingFace tokenizer.json。

设计目标
--------
实现一个自包含、零依赖的 BPE 分词器，能够加载 HuggingFace ``tokenizer.json``
格式（仅支持 ``BPE`` 模型，不支持 WordPiece / Unigram / SentencePiece），
并提供 ``encode`` / ``decode`` API。

如果 ``tokenizer.json`` 不存在或解析失败，提供 ``CharTokenizer`` 字符级 fallback
（无 merges、无依赖），保证最简流程可用。

BPE 算法步骤
------------
1. **Pre-tokenize**：把输入文本切分为「词」（whitespace + punctuation）。
   HuggingFace tokenizer.json 的 ``pre_tokenizer`` 字段常见有
   ``ByteLevel`` / ``Whitespace`` / ``BertPreTokenizer`` 等。
   本实现默认采用 GPT-2 风格的 ``ByteLevel`` 预切分：
   - 用正则把文本切成 word / space / punctuation 段；
   - 每段字符末尾加 ``Ġ``（GPT-2 空格标记）做后续 BPE；
   - 字节级编码：用 GPT-2 的 ``bytes_to_unicode`` 把所有 256 个字节映射到可打印 unicode。
2. **BPE merge**：对每段的字符序列，按 ``merges`` 顺序贪心合并相邻 token，
   直到不能再合并为止。
3. **Vocab lookup**：把合并后的 token 序列映射为整数 id。

decode 步骤
-----------
1. 把 id 反查为 token 字符串；
2. 拼接后用 ``bytes_to_unicode`` 的逆映射还原为字节；
3. ``bytes.decode('utf-8', errors='replace')`` 解码为字符串。

special tokens
--------------
``<|endoftext|>`` 等 special tokens 在 encode 时如果命中会被作为单 token；
在 decode 时直接还原为其字符串形式。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from typing import Optional


# ---------------------------------------------------------------------------
# GPT-2 bytes_to_unicode：把 256 字节映射到稳定的 unicode 字符
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 字节到 unicode 的映射。

    可打印 ASCII 范围（!-~、¡-¬、®-ÿ）直接保留为自身；
    其它字节映射到 256 + 偏移的 unicode 字符，避免控制字符与空白。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


# GPT-2 风格 pre-tokenize 正则
# 注意：Python 标准库 re 不支持 \p{L} Unicode 属性，必须用 \w 与 \S 近似
# 使用 try/except 降级以保证在所有 Python 版本下可用
_GPT2_SPLIT_RE = None
_GPT2_SPLIT_RE_FALLBACK = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)
try:
    # 优先尝试使用 \p{...}（需要 regex 库或未来 Python 版本）
    _GPT2_SPLIT_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )
except re.error:
    _GPT2_SPLIT_RE = None  # 标准库 re 不支持，使用 fallback


def _gpt2_pretokenize(text: str) -> list[str]:
    """GPT-2 风格 pre-tokenize：把文本切成 word / number / punctuation / space 段。

    每段前导空格保留在 token 内（如 " hello"），后续 byte-level 编码时
    空格变成 ``Ġ``。
    """
    pat = _GPT2_SPLIT_RE if _GPT2_SPLIT_RE is not None else _GPT2_SPLIT_RE_FALLBACK
    return pat.findall(text)


def _byte_encode(piece: str) -> list[str]:
    """把一段字符串 byte-level 编码为 unicode 字符列表。"""
    return [_BYTE_ENCODER[b] for b in piece.encode("utf-8")]


# ---------------------------------------------------------------------------
# BPE 算法核心
# ---------------------------------------------------------------------------


def _bpe(token_chars: list[str], merge_ranks: dict[tuple[str, str], int]) -> list[str]:
    """对一段字符序列执行 BPE 合并。

    Args:
        token_chars: 字符列表（byte-level 编码后的 unicode 字符）
        merge_ranks: {("a", "b"): rank} 合并优先级，rank 越小越优先合并

    Returns:
        合并后的 token 字符串列表

    算法：
        重复以下步骤直到无法合并：
        1. 在当前 word 中找出所有相邻 pair；
        2. 选择 rank 最小的 pair；
        3. 把该 pair 合并为单个 token；
        4. 若不存在任何可合并 pair，停止。
    """
    word = list(token_chars)
    if len(word) < 2:
        return word

    while True:
        # 找出 rank 最小的 pair
        best_pair = None
        best_rank = None
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            r = merge_ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_pair = pair
        if best_pair is None:
            break
        # 合并所有出现的 best_pair
        new_word = []
        i = 0
        a, b = best_pair
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
        if len(word) == 1:
            break
    return word


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """最小 BPE 分词器，可加载 HuggingFace tokenizer.json。

    Args:
        vocab: {token_str: id}
        merges: ["token_a token_b", ...] 按 rank 升序排列
        special_tokens: 可选的 special token 字符串列表（如 ["<|endoftext|>"]）
        byte_level: 是否使用 GPT-2 byte-level 编码（默认 True）
    """

    def __init__(
        self,
        vocab: dict,
        merges: list,
        special_tokens: Optional[list[str]] = None,
        byte_level: bool = True,
    ):
        self.vocab = dict(vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        # merges 是字符串列表 ["a b", "c d", ...]，按顺序赋 rank
        self.merge_ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            if isinstance(m, str):
                parts = m.split(" ")
                if len(parts) == 2:
                    self.merge_ranks[(parts[0], parts[1])] = i
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                self.merge_ranks[(str(m[0]), str(m[1]))] = i
        self.byte_level = byte_level
        # special tokens：保留映射，便于 encode 时识别
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        for st in self.special_tokens:
            if st not in self.vocab:
                # 给 special token 分配 id（vocab 末尾）
                self.vocab[st] = len(self.vocab)
                self.id_to_token[self.vocab[st]] = st

    # ------------------------------------------------------------------
    # 构造方法
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        """从 HuggingFace ``tokenizer.json`` 加载。

        解析 ``model.vocab`` 与 ``model.merges``；
        ``added_tokens`` 与 ``added_tokens_decoder`` 中的 special tokens 也会注册。
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = data.get("model", {})
        if model.get("type", "BPE") != "BPE":
            raise ValueError(
                f"Only BPE model is supported, got {model.get('type')!r}"
            )
        vocab_raw = model.get("vocab", {})
        # vocab 可能是 {token: id} 或 {token: [id, ...]}
        vocab = {}
        for k, v in vocab_raw.items():
            if isinstance(v, (list, tuple)):
                vocab[k] = int(v[0])
            else:
                vocab[k] = int(v)
        merges = model.get("merges", [])

        # 收集 special tokens
        special_tokens: list[str] = []
        # added_tokens 顶层字段
        for at in data.get("added_tokens", []):
            content = at.get("content")
            if content is not None:
                special_tokens.append(content)
                # 同步到 vocab
                if content not in vocab:
                    vocab[content] = int(at.get("id", len(vocab)))

        # 判断是否 byte-level
        pre_tok = data.get("pre_tokenizer", {})
        byte_level = False
        if isinstance(pre_tok, dict) and pre_tok.get("type") == "ByteLevel":
            byte_level = True
        # 如果 model 配置里有 byte_fallback 等也认为是 byte_level
        if model.get("byte_fallback"):
            byte_level = True

        return cls(vocab, merges, special_tokens=special_tokens, byte_level=byte_level)

    @classmethod
    def from_hf(cls, repo_id: str, revision: str = "main") -> "BPETokenizer":
        """从 HuggingFace repo 下载 ``tokenizer.json`` 并加载。

        若安装了 ``huggingface_hub``，优先用 ``hf_hub_download``；
        否则用 ``urllib`` + ``https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json``。
        """
        # 优先使用 huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                revision=revision,
            )
            return cls.from_file(path)
        except ImportError:
            pass

        # 降级：urllib
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/tokenizer.json"
        tmp_dir = tempfile.mkdtemp(prefix="verse_tok_hf_")
        local_path = os.path.join(tmp_dir, "tokenizer.json")
        try:
            urllib.request.urlretrieve(url, local_path)  # noqa: S310
            return cls.from_file(local_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download tokenizer.json from {url}: {e}. "
                f"Install `huggingface_hub` for better HF support."
            ) from e

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """把文本编码为 token id 列表。

        Args:
            text: 输入文本
            add_special_tokens: 是否在末尾追加 ``<|endoftext|>``（如果存在）
        """
        ids: list[int] = []

        # 先按 special tokens 切分（special token 直接作为单 token）
        if self.special_tokens:
            # 构造一个匹配所有 special tokens 的正则
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pat = re.compile("(" + "|".join(re.escape(s) for s in sorted_specials) + ")")
            chunks = pat.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.vocab and chunk in self.special_tokens:
                ids.append(self.vocab[chunk])
                continue
            ids.extend(self._encode_chunk(chunk))

        if add_special_tokens:
            # 追加 <|endoftext|>（如果存在）
            eot = "<|endoftext|>"
            if eot in self.vocab and (not ids or ids[-1] != self.vocab[eot]):
                ids.append(self.vocab[eot])

        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """对一段（不含 special token 的）文本做 BPE encode。"""
        # pre-tokenize
        pieces = _gpt2_pretokenize(text)
        ids: list[int] = []
        for piece in pieces:
            if self.byte_level:
                chars = _byte_encode(piece)
            else:
                chars = list(piece)
            # BPE merge
            tokens = _bpe(chars, self.merge_ranks)
            for tok in tokens:
                if tok in self.vocab:
                    ids.append(self.vocab[tok])
                else:
                    # 未知 token：fallback 到字符级 id（如果有），否则用 <unk> 或跳过
                    unk = self.vocab.get("<unk>")
                    if unk is not None:
                        ids.append(unk)
                    else:
                        # 用每个字符的字节 id（若 vocab 有），否则跳过
                        for ch in tok:
                            if ch in self.vocab:
                                ids.append(self.vocab[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码为字符串。

        - 优先还原 byte-level 编码；
        - special tokens 直接还原为字符串；
        - 未知 id 跳过。
        """
        # 先按 special token 分段
        pieces: list[str] = []
        special_set = set(self.special_tokens)
        # 累积普通 token 的 unicode 字符
        normal_buffer: list[str] = []

        def flush_normal():
            if not normal_buffer:
                return
            if self.byte_level:
                # 用 byte decoder 还原：注意 buffer 中的元素可能是多字符 token
                # （如 BPE 合并后的 "ll"、"Wo"），需要展开为单字符再查 _BYTE_DECODER
                try:
                    byte_arr = bytes(
                        _BYTE_DECODER[ch]
                        for tok in normal_buffer
                        for ch in tok
                        if ch in _BYTE_DECODER
                    )
                    pieces.append(byte_arr.decode("utf-8", errors="replace"))
                except Exception:
                    pieces.append("".join(normal_buffer))
            else:
                pieces.append("".join(normal_buffer))
            normal_buffer.clear()

        for i in ids:
            tok = self.id_to_token.get(int(i))
            if tok is None:
                continue
            if tok in special_set:
                flush_normal()
                pieces.append(tok)
            else:
                normal_buffer.append(tok)
        flush_normal()
        return "".join(pieces)

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# CharTokenizer：字符级 fallback（无依赖、无 merges）
# ---------------------------------------------------------------------------


class CharTokenizer:
    """字符级 fallback 分词器。

    当 ``BPETokenizer.from_file`` / ``from_hf`` 失败时使用。
    每个 unicode 字符对应一个 id；id 0 保留给 ``<pad>``，1 给 ``<unk>``，2 给 ``<bos>``，3 给 ``<eos>``。

    用法与 ``BPETokenizer`` 一致（``encode`` / ``decode`` / ``__len__``）。
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, vocab: Optional[dict] = None, special_tokens: Optional[list[str]] = None):
        if vocab is None:
            # 默认 special tokens 占据 0..3
            vocab = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.BOS_TOKEN: 2,
                self.EOS_TOKEN: 3,
            }
            # 4.. 给所有可能的字符动态分配（懒加载，第一次 encode 时扩充）
        self.vocab = dict(vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.special_tokens = list(special_tokens) if special_tokens else [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN,
        ]
        self._next_id = max(self.vocab.values()) + 1 if self.vocab else 0

    def _ensure_char(self, ch: str) -> int:
        if ch not in self.vocab:
            self.vocab[ch] = self._next_id
            self.id_to_token[self._next_id] = ch
            self._next_id += 1
        return self.vocab[ch]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self._ensure_char(ch) for ch in text]
        if add_special_tokens:
            eos = self.vocab.get(self.EOS_TOKEN)
            if eos is not None:
                ids.append(eos)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.id_to_token.get(int(i))
            if tok is None or tok in (
                self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
            ):
                # special token 跳过（不输出到文本）
                if tok == self.EOS_TOKEN:
                    out.append("\n")  # EOS 替换为换行
                continue
            out.append(tok)
        return "".join(out)

    def __len__(self) -> int:
        return len(self.vocab)


__all__ = ["BPETokenizer", "CharTokenizer"]
