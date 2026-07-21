"""Task 5.3 / Task 2: 最小 BPE 分词器，可加载 HuggingFace tokenizer.json。

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
   本实现默认采用 GPT-4 风格预分词（见 :mod:`verse_tokenizer.preprocess`）：
   - 中文整字、英文单词、数字、标点、空白分别独立成块；
   - 字节级编码：用 GPT-2 的 ``bytes_to_unicode`` 把所有 256 个字节映射到可打印 unicode。
2. **BPE merge**：对每段的字符序列，按 ``merges`` 顺序贪心合并相邻 token，
   直到不能再合并为止。
3. **Vocab lookup**：把合并后的 token 序列映射为整数 id。

decode 步骤
-----------
1. 把 id 反查为 token 字符串；
2. 拼接后用 ``bytes_to_unicode`` 的逆映射还原为字节；
3. ``bytes.decode('utf-8', errors='ignore')`` 解码为字符串（避免 U+FFFD 乱码）。

special tokens
--------------
``<|endoftext|>`` 等 special tokens 在 encode 时如果命中会被作为单 token；
在 decode 时直接还原为其字符串形式。
Task 2 升级后默认注册 bos/eos/pad/unk（旧风格 ``<bos>`` 等 + 新风格 ``<|bos|>`` 等）
以及 chat 角色标记 ``<|user|>`` / ``<|assistant|>`` / ``<|system|>``。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, List

# Task 2: 引入 preprocess 与 chat_template 模块（统一 NFKC + 预分词 + UTF-8 边界修复）
from .preprocess import (
    nfkc_normalize,
    pre_tokenize as _gpt4_pre_tokenize,
    trim_byte_ids_to_utf8_boundary,
)
from .chat_template import render_chat as _render_chat, render_prompt as _render_prompt


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
# Task 5.1: UTF-8 字节对齐工具——丢弃末尾不完整的多字节序列
# ---------------------------------------------------------------------------


def _trim_to_utf8_boundary(byte_ids: List[int]) -> List[int]:
    """从末尾向前检查，丢弃不完整的 UTF-8 多字节序列。

    Task 2.5: 实现已统一到 :func:`verse_tokenizer.preprocess.trim_byte_ids_to_utf8_boundary`，
    这里保留为薄包装以兼容旧 API（:mod:`tests.test_no_garbled` 直接 import 此函数）。

    参考 UTF-8 编码规则：
        - 首字节 ``0xxxxxxx``：1 字节字符
        - 首字节 ``110xxxxx``：2 字节字符，需后续 1 字节
        - 首字节 ``1110xxxx``：3 字节字符，需后续 2 字节
        - 首字节 ``11110xxx``：4 字节字符，需后续 3 字节
        - 后续字节形如 ``10xxxxxx``

    若末尾字符的字节数不足，则丢弃该字符的所有字节，保证剩余字节解码时
    不会产生 U+FFFD（乱码）。

    Args:
        byte_ids: 字节 id 列表（每个元素 0-255）

    Returns:
        对齐到完整 UTF-8 字符边界的字节 id 列表
    """
    return trim_byte_ids_to_utf8_boundary(byte_ids)


# ---------------------------------------------------------------------------
# Task 2: 默认特殊 token（train 后自动注册）
# ---------------------------------------------------------------------------
# 旧风格（向后兼容已有测试：test_bpe_train_vocab_size 检查 <bos>/<eos>/<pad>/<unk>）
_DEFAULT_LEGACY_SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>", "<unk>"]
# 新风格（chat_template 用，与 unigram.SpecialTokens 对齐）
_DEFAULT_NEW_SPECIAL_TOKENS = [
    "<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
    "<|user|>", "<|assistant|>", "<|system|>",
]
# 合并后的默认特殊 token 列表
DEFAULT_SPECIAL_TOKENS = _DEFAULT_LEGACY_SPECIAL_TOKENS + _DEFAULT_NEW_SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Task 4.1 / 4.2 / 4.3: BaseTokenizer 抽象基类 + NFKC preprocess 钩子
# ---------------------------------------------------------------------------


class BaseTokenizer(ABC):
    """分词器抽象基类，向 GPT-4 / Llama tokenizer 设计看齐。

    子类必须实现以下抽象方法：
        - ``encode(text) -> List[int]``：文本 → token id 列表
        - ``decode(ids) -> str``：token id 列表 → 文本
        - ``save(path) -> None``：序列化到文件
        - ``load(path) -> BaseTokenizer``（classmethod）：从文件加载
        - ``__len__() -> int``：词表大小

    预处理钩子：
        - ``preprocess(text) -> str``：默认做 NFKC 正规化 + 去除控制字符
          （保留 ``\\n`` / ``\\r`` / ``\\t`` 等基本空白），子类可覆盖。
          Task 2.6: NFKC 实现统一调用 :func:`verse_tokenizer.preprocess.nfkc_normalize`。

    Task 2.6 新增默认方法（子类可覆盖）：
        - ``apply_chat_template(messages) -> List[int]``：渲染 chat 数组后 encode
        - ``apply_prompt_template(prompt) -> List[int]``：渲染 prompt 后 encode

    Task 2.6 新增属性约定：
        - ``special_tokens``：``dict[str, int]``，特殊 token 字符串 → id
        - ``auto_add_special_tokens``：``bool``，encode 时是否默认加 bos/eos
          （构造参数 ``add_special_tokens`` 控制默认值）

    设计目标：
        - 统一三种 tokenizer（BPE / Byte / Char）的接口契约；
        - 在 encode 前置 NFKC 正规化，确保全角字符与组合形式统一；
        - 保持向后兼容（不破坏现有 API）。
    """

    # 类级默认：encode 时是否自动加 bos/eos（子类可在 __init__ 中覆盖）
    auto_add_special_tokens: bool = True

    def preprocess(self, text: str) -> str:
        """文本预处理钩子：默认做 NFKC 正规化 + 去除控制字符。

        - NFKC：全角字母数字 → 半角，组合字符 → 规范形式，兼容字符分解；
          Task 2.6: 统一调用 :func:`verse_tokenizer.preprocess.nfkc_normalize`。
        - 去除控制字符（Cc 类），但保留 ``\\n`` / ``\\r`` / ``\\t`` 等基本空白。

        子类可覆盖此方法以实现自定义预处理（如 GPT-4 / Llama 风格的
        whitespace 规范化、特定字符映射等）。
        """
        if not isinstance(text, str):
            text = str(text)
        # NFKC 正规化：全角→半角、组合→规范、兼容字符分解（统一入口）
        text = nfkc_normalize(text)
        # 去除控制字符（Cc 类），但保留 \n \r \t
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\r", "\t")
            or unicodedata.category(ch) != "Cc"
        )
        return text

    # ------------------------------------------------------------------
    # Task 2.6: chat template 默认实现（子类可覆盖）
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """渲染 chat 数组为字符串后 encode。

        Args:
            messages: ``[{"role": "user", "content": "..."}, ...]``

        Returns:
            token id 列表（不加首尾 bos/eos，因为 render_chat 已含 ``<|eos|>``）
        """
        rendered = _render_chat(messages)
        # 子类的 encode 通常接受 add_special_tokens 参数
        try:
            return self.encode(rendered, add_special_tokens=False)  # type: ignore[misc]
        except TypeError:
            # 子类 encode 不接受 add_special_tokens 参数（如 ByteTokenizer 旧签名）
            return self.encode(rendered)

    def apply_prompt_template(self, prompt: str) -> list[int]:
        """渲染 prompt 为推理前缀后 encode。

        Args:
            prompt: 用户输入的 prompt 文本

        Returns:
            token id 列表（不加首尾 bos/eos，prompt 模板用于推理前缀）
        """
        rendered = _render_prompt(prompt)
        try:
            return self.encode(rendered, add_special_tokens=False)  # type: ignore[misc]
        except TypeError:
            return self.encode(rendered)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """文本 → token id 列表（抽象方法，子类必须实现）。"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """token id 列表 → 文本（抽象方法，子类必须实现）。"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """序列化到文件（抽象方法，子类必须实现）。"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """从文件加载（抽象方法，子类必须实现）。"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """返回词表大小（抽象方法，子类必须实现）。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------


class BPETokenizer(BaseTokenizer):
    """最小 BPE 分词器，可加载 HuggingFace tokenizer.json。

    Args:
        vocab: {token_str: id}
        merges: ["token_a token_b", ...] 按 rank 升序排列
        special_tokens: 可选的 special token（``list[str]`` 或 ``dict[str, int]``）
        byte_level: 是否使用 GPT-2 byte-level 编码（默认 True）
        add_special_tokens: encode 时是否默认加 bos/eos（默认 True）

    Task 2.2 升级：
        - ``special_tokens`` 内部存储为 ``dict[str, int]``（token 字符串 → id），
          与 :class:`BaseTokenizer` 约定一致；
        - ``encode(text, add_special_tokens=True)`` 在首尾加 ``<bos>`` / ``<eos>``；
        - ``train`` 后自动注册 ``DEFAULT_SPECIAL_TOKENS``（11 个，含旧风格与新风格）。
    """

    def __init__(
        self,
        vocab: dict,
        merges: list,
        special_tokens: Optional = None,
        byte_level: bool = True,
        add_special_tokens: bool = True,
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
        # Task 2.2: special_tokens 统一存为 dict[str, int]（token 字符串 → id）
        self.special_tokens: dict[str, int] = {}
        if special_tokens is None:
            pass
        elif isinstance(special_tokens, dict):
            # dict[str, int] 输入：忽略显式 id，统一追加到 vocab 末尾
            # （保持与 list[str] 分支一致的行为，避免 id 冲突）
            for st in special_tokens:
                if st not in self.vocab:
                    self.vocab[st] = len(self.vocab)
                    self.id_to_token[self.vocab[st]] = st
                self.special_tokens[st] = self.vocab[st]
        else:
            # list[str] 输入
            for st in special_tokens:
                if st not in self.vocab:
                    self.vocab[st] = len(self.vocab)
                    self.id_to_token[self.vocab[st]] = st
                self.special_tokens[st] = self.vocab[st]
        # Task 2.2: add_special_tokens 构造参数
        # （属性名避开与 add_special_tokens 方法冲突，用 auto_add_special_tokens）
        self.auto_add_special_tokens = add_special_tokens

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

    def encode(self, text: str, add_special_tokens: Optional[bool] = None) -> list[int]:
        """把文本编码为 token id 列表。

        Args:
            text: 输入文本
            add_special_tokens:
                - ``True``：在首尾追加 ``<bos>`` / ``<eos>``（如果存在于 vocab）
                - ``False``：不追加任何 special token
                - ``None``：使用 ``self.auto_add_special_tokens`` 默认值

        Task 2.2 升级：
            - 默认通过 ``auto_add_special_tokens`` 属性控制（构造参数 ``add_special_tokens``）；
            - ``add_special_tokens=True`` 时在首尾加 ``<bos>`` / ``<eos>``（兼容旧 ``<|endoftext|>``）。
        """
        # Task 4.2: 前置 NFKC 正规化 + 控制字符去除（BaseTokenizer.preprocess）
        text = self.preprocess(text)
        # 解析 add_special_tokens 参数：None 时用 auto_add_special_tokens 默认值
        if add_special_tokens is None:
            add_special_tokens = self.auto_add_special_tokens

        ids: list[int] = []

        # 先按 special tokens 切分（special token 直接作为单 token）
        if self.special_tokens:
            # 构造一个匹配所有 special tokens 的正则（按长度降序避免短 token 抢匹配）
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
            ids.extend(self._encode_chunk(chunk))

        if add_special_tokens:
            # Task 2.2: 在首尾加 <bos> / <eos>（向后兼容旧 <|endoftext|>）
            bos_id = self.vocab.get("<bos>")
            if bos_id is not None and (not ids or ids[0] != bos_id):
                ids.insert(0, bos_id)
            eos_id = self.vocab.get("<eos>")
            if eos_id is not None and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)

        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """对一段（不含 special token 的）文本做 BPE encode。

        Task 2.2: 预分词统一调用 :func:`verse_tokenizer.preprocess.pre_tokenize`
        （GPT-4 风格：中文整字、英文单词、数字、标点、空白独立成块）。
        """
        # Task 2.2: GPT-4 风格预分词（中文整字、英文单词、数字、标点、空白独立成块）
        pieces = _gpt4_pre_tokenize(text)
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
                    byte_list = [
                        _BYTE_DECODER[ch]
                        for tok in normal_buffer
                        for ch in tok
                        if ch in _BYTE_DECODER
                    ]
                    # Task 5.1: 字节对齐检查，丢弃末尾不完整的多字节序列
                    byte_list = _trim_to_utf8_boundary(byte_list)
                    # Task 5.5: 用 errors="ignore" 丢弃中间非法字节，避免 U+FFFD
                    pieces.append(bytes(byte_list).decode("utf-8", errors="ignore"))
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

    # ------------------------------------------------------------------
    # Task 3.2 / Task 2.2: add_special_tokens
    # ------------------------------------------------------------------

    def add_special_tokens(self, tokens) -> None:
        """将 tokens 加入 vocab 并标记为 special token。

        - 每个 token 分配新的 id（如果还未在 vocab 中）；
        - 更新 ``special_tokens`` 字典（Task 2.2: 统一存为 ``dict[str, int]``）；
        - encode 时这些 token 视为不可拆分（atomic）。

        Args:
            tokens: ``list[str]`` 或 ``dict[str, int]``（dict 时忽略显式 id，
                    统一追加到 vocab 末尾，避免 id 冲突）
        """
        # 统一输入为 list[str]
        if isinstance(tokens, dict):
            tokens = list(tokens.keys())
        elif isinstance(tokens, str):
            tokens = [tokens]
        for tok in tokens:
            if tok not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[tok] = new_id
                self.id_to_token[new_id] = tok
            # Task 2.2: special_tokens 是 dict[str, int]，用 setdefault 避免覆盖
            if tok not in self.special_tokens:
                self.special_tokens[tok] = self.vocab[tok]

    # ------------------------------------------------------------------
    # Task 3.1: train（字节级 BPE，参考 GPT-2 风格）
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, corpus, vocab_size: int) -> "BPETokenizer":
        """从语料训练字节级 BPE。

        算法步骤：
            1. 将 corpus 编码为字节序列（UTF-8），每个字节映射到 GPT-2 byte-level unicode 字符；
            2. 初始化词汇表为 256 个基础字节字符；
            3. 统计相邻 token pair 频率；
            4. 选择频率最高的 pair 合并为新 token，加入词汇表；
            5. 重复直到 vocab_size 达到或无 pair 可合并（vocab_size 自适应：
               数据太少时回退到最大可达 vocab 大小）；
            6. 记录 merges 列表（按合并顺序）；
            7. 训练完成后自动 ``add_special_tokens(DEFAULT_SPECIAL_TOKENS)``
               （11 个：旧风格 4 + 新风格 7）。

        Args:
            corpus: 训练语料，``str`` 或 ``List[str]``
            vocab_size: 目标词汇表大小（含 256 基础字节 + special tokens）

        Returns:
            训练好的 :class:`BPETokenizer` 实例
        """
        # 1. 统一为单一字符串
        if isinstance(corpus, (list, tuple)):
            text = "\n".join(str(c) for c in corpus)
        else:
            text = str(corpus)

        # 2. Task 2.2: pre-tokenize 改用 GPT-4 风格（中文整字、英文单词、
        #    数字、标点、空白独立成块）+ byte-level 编码
        pieces = _gpt4_pre_tokenize(text)
        # 每个 piece 转为 byte-level 字符元组（用于统计 pair）
        word_list: list[tuple[str, ...]] = []
        for p in pieces:
            if not p:
                continue
            chars = tuple(_byte_encode(p))
            if chars:
                word_list.append(chars)

        # 3. 初始化 vocab：256 个基础字节字符
        byte_chars = sorted(set(_BYTE_ENCODER.values()))
        vocab: dict[str, int] = {ch: i for i, ch in enumerate(byte_chars)}

        # 4. merges 列表（按合并顺序）
        merges: list[tuple[str, str]] = []

        # Task 2.2: 训练目标 merges 数 = vocab_size - 256(基础字节) - len(DEFAULT_SPECIAL_TOKENS)
        # （11 个 special tokens：旧风格 4 + 新风格 7）
        target_merges = max(0, vocab_size - 256 - len(DEFAULT_SPECIAL_TOKENS))

        # 5. 重复合并直到达到目标或无 pair 可合并
        # 已跳过的 pair 集合：避免同一不合法 pair 反复尝试
        skipped_pairs: set = set()
        while len(merges) < target_merges:
            # 统计相邻 pair 频率
            pair_counts: Counter = Counter()
            for word in word_list:
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += 1

            # 移除已跳过的不合法 pair
            for sp in skipped_pairs:
                pair_counts.pop(sp, None)
            if not pair_counts:
                break

            # 选择频率最高的 pair；同频率时按 pair 字典序保证可复现
            best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            if pair_counts[best_pair] < 1:
                break

            # Task 5.2: 字节边界检查——只接受合并后字节序列为合法 UTF-8
            # 字符序列的 merge，保证每个 BPE token 对应的字节序列都能独立
            # decode 为完整字符（向 GPT-4 / Llama tokenizer 设计看齐）。
            # 对于不合法的 merge（如跨多字节字符中间断开），跳过该 pair，
            # 避免后续 decode 时产生乱码（U+FFFD）。
            combined_bytes = [
                _BYTE_DECODER[ch]
                for ch in (best_pair[0] + best_pair[1])
                if ch in _BYTE_DECODER
            ]
            try:
                bytes(combined_bytes).decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                # 合并后字节序列不在 UTF-8 字符边界，跳过该 pair
                skipped_pairs.add(best_pair)
                continue

            # 合并 best_pair 产生新 token
            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            # 更新 word_list：合并所有出现的 best_pair
            new_word_list: list[tuple[str, ...]] = []
            a, b = best_pair
            for word in word_list:
                new_word: list[str] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_list.append(tuple(new_word))
            word_list = new_word_list

        # 6. 转换 merges 为字符串列表 ["a b", ...]（与 from_file 兼容）
        merges_str = [f"{a} {b}" for a, b in merges]

        # 7. Task 2.2: 创建实例（special_tokens 暂为空，train 后自动 add）
        #    注册 DEFAULT_SPECIAL_TOKENS（11 个：旧风格 <bos>/<eos>/<pad>/<unk>
        #    + 新风格 <|bos|>/<|eos|>/<|pad|>/<|unk|>/<|user|>/<|assistant|>/<|system|>）
        instance = cls(vocab, merges_str, special_tokens=None, byte_level=True)
        instance.add_special_tokens(DEFAULT_SPECIAL_TOKENS)
        return instance

    # ------------------------------------------------------------------
    # Task 3.3: save / load JSON 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """将 tokenizer 序列化为 JSON 文件。

        保存内容：``vocab`` (dict)、``merges`` (list of [str, str])、
        ``special_tokens`` (dict token→id)、``pattern`` (str, 如有)、
        ``byte_level`` (bool)。
        """
        # 把 merge_ranks 转回按 rank 升序排列的 list of [a, b]
        sorted_merges = sorted(self.merge_ranks.items(), key=lambda x: x[1])
        merges_list = [[a, b] for (a, b), _ in sorted_merges]

        data = {
            "type": "bpe",
            "vocab": self.vocab,
            "merges": merges_list,
            "special_tokens": {
                tok: self.vocab[tok]
                for tok in self.special_tokens
                if tok in self.vocab
            },
            "pattern": None,
            "byte_level": self.byte_level,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """从 JSON 文件加载 tokenizer（与 :meth:`save` 配对）。

        兼容 ``from_file``（HF tokenizer.json）：若 JSON 含 ``model`` 字段且无
        ``type`` 字段，自动转用 :meth:`from_file` 解析。
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # HF tokenizer.json 兼容
        if "model" in data and "type" not in data:
            return cls.from_file(path)

        vocab_raw = data.get("vocab", {})
        # JSON dict 的 key 始终是字符串，value 转回 int
        vocab = {k: int(v) for k, v in vocab_raw.items()}

        merges = data.get("merges", [])

        special_tokens_field = data.get("special_tokens", {})
        if isinstance(special_tokens_field, dict):
            special_tokens = list(special_tokens_field.keys())
        else:
            special_tokens = list(special_tokens_field)

        byte_level = data.get("byte_level", True)
        return cls(vocab, merges, special_tokens=special_tokens, byte_level=byte_level)

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# CharTokenizer：字符级 fallback（无依赖、无 merges）
# ---------------------------------------------------------------------------


class CharTokenizer(BaseTokenizer):
    """字符级 fallback 分词器。

    当 ``BPETokenizer.from_file`` / ``from_hf`` 失败时使用。
    每个 unicode 字符对应一个 id；id 0 保留给 ``<pad>``，1 给 ``<unk>``，2 给 ``<bos>``，3 给 ``<eos>``。

    Task 2.5 升级：
        - ``special_tokens`` 内部存为 ``dict[str, int]``（与 :class:`BaseTokenizer` 约定一致）；
        - 构造参数 ``add_special_tokens`` 控制 encode 时是否默认加 ``<eos>``；
        - ``apply_chat_template`` / ``apply_prompt_template`` 继承自 :class:`BaseTokenizer`
          （通过 try/except TypeError 兼容本类的 encode 签名）。

    用法与 ``BPETokenizer`` 一致（``encode`` / ``decode`` / ``__len__``）。
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(
        self,
        vocab: Optional[dict] = None,
        special_tokens: Optional = None,
        add_special_tokens: bool = True,
    ):
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
        # Task 2.5: special_tokens 统一存为 dict[str, int]
        self.special_tokens: dict[str, int] = {}
        default_specials = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN,
        ]
        if special_tokens is None:
            specials_list = default_specials
        elif isinstance(special_tokens, dict):
            specials_list = list(special_tokens.keys())
        else:
            specials_list = list(special_tokens)
        for st in specials_list:
            if st not in self.vocab:
                self.vocab[st] = len(self.vocab)
                self.id_to_token[self.vocab[st]] = st
            self.special_tokens[st] = self.vocab[st]
        self._next_id = max(self.vocab.values()) + 1 if self.vocab else 0
        # Task 2.5: add_special_tokens 构造参数
        self.auto_add_special_tokens = add_special_tokens

    def _ensure_char(self, ch: str) -> int:
        if ch not in self.vocab:
            self.vocab[ch] = self._next_id
            self.id_to_token[self._next_id] = ch
            self._next_id += 1
        return self.vocab[ch]

    def encode(self, text: str, add_special_tokens: Optional[bool] = None) -> list[int]:
        """编码文本为字符 id 序列。

        Args:
            text: 输入文本
            add_special_tokens:
                - ``True``：末尾追加 ``<eos>`` id
                - ``False``：不追加
                - ``None``：使用 ``self.auto_add_special_tokens`` 默认值
        """
        # Task 4.2: 前置 NFKC 正规化（全角字符与组合形式统一）
        text = self.preprocess(text)
        if add_special_tokens is None:
            add_special_tokens = self.auto_add_special_tokens
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

    def save(self, path: str) -> None:
        """序列化为 JSON 文件（与 BaseTokenizer 接口对齐）。"""
        data = {
            "type": "char",
            "vocab": self.vocab,
            # Task 2.5: special_tokens 存为 dict[str, int]
            "special_tokens": dict(self.special_tokens),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "char":
            raise ValueError(
                f"Not a CharTokenizer JSON file (type={data.get('type')!r})"
            )
        vocab_raw = data.get("vocab", {})
        vocab = {k: int(v) for k, v in vocab_raw.items()}
        special_tokens = data.get("special_tokens")
        return cls(vocab=vocab, special_tokens=special_tokens)

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Task 3.4: ByteTokenizer（vocab_size=259，含 bos/eos/pad/unk）
# ---------------------------------------------------------------------------


class ByteTokenizer(BaseTokenizer):
    """字节级 tokenizer。

    - ``vocab_size = 259``（256 字节 + bos + eos + pad + unk）；
    - ``encode`` 返回 UTF-8 字节序列（每个字节 0-255）；
    - ``decode`` 把字节序列还原为字符串。

    Task 2.5 升级：
        - ``special_tokens`` 内部存为 ``dict[str, int]``（与 :class:`BaseTokenizer` 约定一致）；
        - ``encode`` 兼容新旧两套 API：
            - 旧：``encode(text, add_bos=True, add_eos=True)``
            - 新：``encode(text, add_special_tokens=True)``（同时控制 bos 和 eos）
        - ``apply_chat_template`` / ``apply_prompt_template`` 显式覆盖，
          调用 ``encode(rendered)``（不加 bos/eos，因为 chat template 已含 ``<|eos|>``）。

    与 :class:`BPETokenizer` 接口对齐：
        - ``encode(text)`` → ``List[int]``
        - ``decode(ids)`` → ``str``
    """

    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        bos_id: int = 256,
        eos_id: int = 257,
        pad_id: int = 258,
        unk_id: int = 255,
        add_special_tokens: bool = False,
    ):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        # vocab_size 包含 256 字节 + bos/eos/pad（unk 复用 255，与 GPT_teacher 一致）
        self.vocab_size = 259
        # vocab 字典：包含 4 个 special token（与 BPETokenizer 接口对齐）
        self.vocab = {
            self.BOS_TOKEN: bos_id,
            self.EOS_TOKEN: eos_id,
            self.PAD_TOKEN: pad_id,
            self.UNK_TOKEN: unk_id,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        # Task 2.5: special_tokens 统一存为 dict[str, int]
        self.special_tokens: dict[str, int] = {
            self.BOS_TOKEN: bos_id,
            self.EOS_TOKEN: eos_id,
            self.PAD_TOKEN: pad_id,
            self.UNK_TOKEN: unk_id,
        }
        self.byte_level = True
        # Task 2.5: add_special_tokens 构造参数（默认 False，保持旧 API 行为）
        # 注意：ByteTokenizer 默认不加 bos/eos，因为旧测试依赖此行为
        self.auto_add_special_tokens = add_special_tokens

    def encode(
        self,
        text: str,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
        add_special_tokens: Optional[bool] = None,
    ) -> list[int]:
        """编码文本为字节 id 序列。

        Args:
            text: 输入文本
            add_bos: 是否在开头加 ``bos_id``（旧 API）
            add_eos: 是否在末尾加 ``eos_id``（旧 API）
            add_special_tokens:
                - 新 API：``True`` 同时加 bos 和 eos，``False`` 都不加
                - ``None``：使用 ``add_bos`` / ``add_eos`` 旧 API（默认均不加）

        Task 2.5: 兼容新旧两套 API。优先使用 ``add_special_tokens``；
        若未提供则回退到 ``add_bos`` / ``add_eos``。
        """
        # Task 4.2: 前置 NFKC 正规化（全角字符与组合形式统一）
        text = self.preprocess(text)
        ids = list(text.encode("utf-8"))
        # Task 2.5: 解析 add_special_tokens 与 add_bos/add_eos 的优先级
        if add_special_tokens is not None:
            should_add_bos = bool(add_special_tokens)
            should_add_eos = bool(add_special_tokens)
        else:
            should_add_bos = bool(add_bos) if add_bos is not None else False
            should_add_eos = bool(add_eos) if add_eos is not None else False
        if should_add_bos:
            ids = [self.bos_id] + ids
        if should_add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """解码 id 序列为字符串。

        Args:
            ids: token id 列表
            strip_special: ``True`` 时丢弃 special token；``False`` 时还原为对应字符串

        Task 5.1: 解码前做字节对齐检查，丢弃末尾不完整的多字节 UTF-8 序列，
        避免 ``errors="replace"`` 把不完整字节替换为 U+FFFD（乱码）。
        """
        if strip_special:
            byte_ids = [int(i) for i in ids if int(i) < 256]
            # Task 5.1: 字节对齐检查，丢弃末尾不完整的多字节序列
            byte_ids = _trim_to_utf8_boundary(byte_ids)
            # Task 5.5: 用 errors="ignore" 丢弃中间非法字节（如模型生成
            # 的 continuation byte 0x80-0xBF 作为首字节），避免产生 U+FFFD
            return bytes(byte_ids).decode("utf-8", errors="ignore")

        # 保留 special token 字符串：在 special token 处切段
        out_parts: list[str] = []
        byte_buf: list[int] = []
        for i in ids:
            i = int(i)
            if i < 256:
                byte_buf.append(i)
            else:
                if byte_buf:
                    # Task 5.1: 每段字节序列都做对齐检查
                    byte_buf = _trim_to_utf8_boundary(byte_buf)
                    # Task 5.5: 中间非法字节用 ignore 丢弃
                    out_parts.append(bytes(byte_buf).decode("utf-8", errors="ignore"))
                    byte_buf = []
                tok = self.id_to_token.get(i)
                out_parts.append(tok if tok else "")
        if byte_buf:
            # Task 5.1: 末尾段也做对齐检查
            byte_buf = _trim_to_utf8_boundary(byte_buf)
            # Task 5.5: 中间非法字节用 ignore 丢弃
            out_parts.append(bytes(byte_buf).decode("utf-8", errors="ignore"))
        return "".join(out_parts)

    # ------------------------------------------------------------------
    # Task 2.5: apply_chat_template / apply_prompt_template
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """渲染 chat 数组并编码（不加 bos/eos，因为 render_chat 已含 ``<|eos|>``）。"""
        rendered = _render_chat(messages)
        # 显式调用 encode，不传 add_special_tokens（保持旧 API 行为，不加 bos/eos）
        return self.encode(rendered)

    def apply_prompt_template(self, prompt: str) -> list[int]:
        """渲染 prompt 并编码（不加 bos/eos，prompt 模板用于推理前缀）。"""
        rendered = _render_prompt(prompt)
        return self.encode(rendered)

    def save(self, path: str) -> None:
        """序列化为 JSON 文件。"""
        data = {
            "type": "byte",
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "vocab_size": self.vocab_size,
            # Task 2.5: special_tokens 存为 dict[str, int]
            "special_tokens": dict(self.special_tokens),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "byte":
            raise ValueError(
                f"Not a ByteTokenizer JSON file (type={data.get('type')!r})"
            )
        return cls(
            bos_id=int(data.get("bos_id", 256)),
            eos_id=int(data.get("eos_id", 257)),
            pad_id=int(data.get("pad_id", 258)),
            unk_id=int(data.get("unk_id", 255)),
        )

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Task 3.5: load_tokenizer 工厂函数
# ---------------------------------------------------------------------------


def load_tokenizer(kind: str = "byte", path: Optional[str] = None):
    """工厂函数：根据 ``kind`` 加载 tokenizer。

    Args:
        kind: tokenizer 类型
            - ``"hf"``：尝试用 ``tokenizers`` 包加载 HF ``tokenizer.json``，
              失败 fallback 到 :class:`ByteTokenizer`
            - ``"bpe"``：调用 :meth:`BPETokenizer.load`；无 path 返回空 BPETokenizer
            - ``"byte"``：返回 :class:`ByteTokenizer`（path 可选）
        path: 文件路径（可选）

    Returns:
        统一接口对象：``encode(text)`` → ``List[int]``，``decode(ids)`` → ``str``
    """
    if kind == "hf":
        if path and os.path.exists(path):
            try:
                from tokenizers import Tokenizer as HFTokenizer

                hf_tok = HFTokenizer.from_file(path)

                # 包装为统一接口
                class _HFTokenizerWrapper:
                    """HF Tokenizer 的统一接口包装。"""

                    def __init__(self, t):
                        self.t = t
                        self.vocab_size = t.get_vocab_size()

                    def encode(self, text, add_special_tokens=True):
                        return self.t.encode(
                            text, add_special_tokens=add_special_tokens
                        ).ids

                    def decode(self, ids):
                        return self.t.decode(ids)

                    def __len__(self):
                        return self.vocab_size

                return _HFTokenizerWrapper(hf_tok)
            except Exception:
                # tokenizers 包未安装或加载失败，降级到 ByteTokenizer
                pass
        return ByteTokenizer()

    if kind == "bpe":
        if path and os.path.exists(path):
            return BPETokenizer.load(path)
        # 无 path：返回空 BPETokenizer（vocab 仅含基础字节字符）
        return BPETokenizer({}, [], byte_level=True)

    if kind == "byte":
        if path and os.path.exists(path):
            try:
                return ByteTokenizer.load(path)
            except Exception:
                pass
        return ByteTokenizer()

    raise ValueError(
        f"Unknown tokenizer kind: {kind!r} (expected 'hf'/'bpe'/'byte')"
    )


__all__ = [
    "BaseTokenizer",
    "BPETokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "load_tokenizer",
]
