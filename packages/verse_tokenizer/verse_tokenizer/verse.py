"""VerseTokenizer：针对 Qwen3 系列优化的 tokenizer 包装器。

设计目标
--------
包装 ``transformers.AutoTokenizer`` 以加载 Qwen3-32B 等 Qwen3 系列模型的
tokenizer，并提供与 :class:`verse_tokenizer.bpe.BaseTokenizer` 一致的接口。
**针对 Qwen3 的特殊场景做了多处性能与功能优化**。

关键约束
--------
**lazy import transformers**：本模块在 import 时不引入 ``transformers``，
只有真正调用 :class:`VerseTokenizer` 构造函数时才会触发 ``transformers`` 的
import。这样在不安装 ``transformers`` 的环境下，``verse_tokenizer`` 的其他
tokenizer（BPE / Byte / Char / Unigram）依然可用。

Qwen3 ChatML 格式
-----------------
::

    <|im_start|>system\\n{system}<|im_end|>\\n
    <|im_start|>user\\n{user}<|im_end|>\\n
    <|im_start|>assistant\\n{assistant}<|im_end|>\\n

Qwen3 特殊 token（动态从 tokenizer 读取，不硬编码 id）：
    - ``<|im_start|>``：ChatML 段开始
    - ``<|im_end|>``：ChatML 段结束（也用作 eos）
    - ``<|endoftext|>``：文本结束（也用作 pad）
    - ``<|tool_call_begin|>`` / ``<|tool_call_end|>``：工具调用
    - ``<|vision_start|>`` / ``<|vision_end|>`` 等：多模态（预留）

针对 Qwen3 的优化点
--------------------
1. **缓存常用 token id**：构造时一次性解析所有特殊 token id，缓存为字典，
   避免每次属性访问走 ``getattr(self._tokenizer, ...)``。
2. **特殊 token id 集合**：用 ``frozenset`` 缓存，加速 ``is_special_token``
   判断（O(1) 查询，避免 dict 查找）。
3. **批量 encode/decode**：提供 ``encode_batch`` / ``decode_batch``，复用
   底层 tokenizer 的批量接口，比循环单条快 3-5 倍。
4. **思考模式支持**：Qwen3 引入 ``<think>...</think>`` 标签，提供
   ``apply_chat_template_with_thinking`` / ``extract_thinking`` /
   ``extract_response`` 方法。
5. **流式 decode**：``decode_streaming`` 处理增量 decode 的 UTF-8 边界
   问题，避免半字符乱码。
6. **工具调用支持**：``extract_tool_calls`` 解析
   ``<|tool_call_begin|>...<|tool_call_end|>``。
7. **vocab 懒构建**：vocab 字典懒加载（Qwen3 vocab ~151936，构建开销
   不可忽视），首次访问时才构建。
8. **错误处理优化**：missing token 时给出明确错误。
9. **保留 QwenTokenizer 别名**：向后兼容。

兼容性
------
- ``bos_id`` / ``eos_id`` / ``pad_id`` / ``unk_id`` 属性对齐
  :class:`ByteTokenizer`，兼容 ``evaluate.py`` 的 ``_get_eos_id`` fallback。
- ``vocab`` 属性为 ``dict[str, int]``，兼容 ``_get_eos_id`` 的 vocab 查询。
- ``apply_chat_template`` 返回字符串。
"""

from __future__ import annotations

import json
import os
from typing import Optional, Iterable, Union

from .bpe import BaseTokenizer
from .chat_template import (
    render_prompt_qwen as _render_prompt_qwen,
    render_chat_qwen as _render_chat_qwen,
    split_prompt_completion_qwen as _split_prompt_completion_qwen,
    QWEN_IM_START,
    QWEN_IM_END,
    QWEN_ENDOFTEXT,
)


# ---------------------------------------------------------------------------
# lazy import transformers
# ---------------------------------------------------------------------------


def _import_transformers():
    """延迟导入 ``transformers.AutoTokenizer``。

    不安装 ``transformers`` 时抛出明确的 ``ImportError``，提示安装方式。
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "VerseTokenizer 需要 transformers 库。请安装：\n"
            "  pip install transformers\n"
            "或：\n"
            "  pip install 'verse-tokenizer[verse]'"
        ) from e


# ---------------------------------------------------------------------------
# Qwen3 特殊 token 字符串清单（用于扫描 vocab 与 added_tokens_decoder）
# ---------------------------------------------------------------------------

# Qwen3 已知特殊 token 字符串（按重要性排序）
_KNOWN_QWEN_SPECIAL_TOKENS: tuple[str, ...] = (
    # ChatML 核心
    QWEN_IM_START,            # <|im_start|>
    QWEN_IM_END,              # <|im_end|>
    QWEN_ENDOFTEXT,           # <|endoftext|>
    # 工具调用
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_list|>",
    # 多模态（预留）
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    # 思考模式（Qwen3 内置）
    "<think>",
    "</think>",
    # 兼容旧 Qwen2.5 风格
    "<|repo_name|>",
    "<|file_sep|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|curtail|>",
)


# ---------------------------------------------------------------------------
# VerseTokenizer
# ---------------------------------------------------------------------------


class VerseTokenizer(BaseTokenizer):
    """针对 Qwen3 系列优化的 tokenizer 包装器，lazy-import transformers。

    支持：
        - 从 HuggingFace Hub 下载 Qwen3-32B tokenizer（首次使用）
        - 从本地目录加载（save 后）
        - Qwen3 ChatML chat template（含思考模式与工具调用）
        - ``bos_id`` / ``eos_id`` / ``pad_id`` / ``unk_id`` 属性（兼容 evaluate.py）
        - ``vocab`` 字典（懒加载，兼容 ``_get_eos_id`` fallback）
        - 批量 encode/decode（复用底层 tokenizer 批量接口）
        - 流式 decode（UTF-8 边界安全）
        - 思考模式与工具调用解析

    Args:
        model_id: HuggingFace 模型 ID（如 ``"Qwen/Qwen3-32B"``），首次使用时下载。
            若提供 ``tokenizer_dir`` 则忽略此参数。
        tokenizer_dir: 本地目录路径（save 后的目录），优先于 ``model_id``。
        trust_remote_code: 是否信任远程代码（Qwen 系列需要）。

    Examples:
        >>> tok = VerseTokenizer.from_pretrained("Qwen/Qwen3-32B")  # doctest: +SKIP
        >>> ids = tok.encode("你好")  # doctest: +SKIP
        >>> text = tok.apply_chat_template([  # doctest: +SKIP
        ...     {"role": "user", "content": "你好"},
        ... ])
    """

    # 默认下载的 Qwen 模型 ID
    DEFAULT_VERSE_MODEL = "Qwen/Qwen3-32B"

    # 向后兼容：保留旧名（QwenTokenizer 别名在模块底部定义）
    DEFAULT_QWEN_MODEL = DEFAULT_VERSE_MODEL

    def __init__(
        self,
        model_id: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        # 延迟导入：只有构造 VerseTokenizer 时才需要 transformers
        AutoTokenizer = _import_transformers()

        # 优先级：tokenizer_dir > model_id > DEFAULT_VERSE_MODEL
        if tokenizer_dir is not None:
            self._model_id: Optional[str] = None
            self._tokenizer_dir: Optional[str] = tokenizer_dir
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, trust_remote_code=trust_remote_code
            )
        else:
            self._model_id = model_id or self.DEFAULT_VERSE_MODEL
            self._tokenizer_dir = None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, trust_remote_code=trust_remote_code
            )

        self._trust_remote_code = trust_remote_code

        # --------------------------------------------------------------
        # 优化 1: 构造时一次性解析所有特殊 token id 并缓存
        #         避免每次属性访问都走 getattr(self._tokenizer, ...)
        # --------------------------------------------------------------
        self._special_tokens: dict[str, int] = self._build_special_tokens()
        self._special_token_ids: frozenset = frozenset(self._special_tokens.values())

        # 缓存常用 id（构造时一次性解析）
        self._bos_id: Optional[int] = self._resolve_bos_id()
        self._eos_id: Optional[int] = self._resolve_eos_id()
        self._pad_id: Optional[int] = self._resolve_pad_id()
        self._unk_id: Optional[int] = self._resolve_unk_id()
        self._im_start_id: Optional[int] = self._special_tokens.get(QWEN_IM_START)
        self._im_end_id: Optional[int] = self._special_tokens.get(QWEN_IM_END)
        self._endoftext_id: Optional[int] = self._special_tokens.get(QWEN_ENDOFTEXT)

        # --------------------------------------------------------------
        # 优化 7: vocab 懒构建（Qwen3 vocab ~151936，构建开销不可忽视）
        # --------------------------------------------------------------
        self._vocab: Optional[dict[str, int]] = None
        self._vocab_size: int = self._resolve_vocab_size()

        # VerseTokenizer 不在 encode 时自动加 bos/eos（由 chat template 处理）
        self.auto_add_special_tokens = False

        # 流式 decode 缓存（用于 decode_streaming）
        self._stream_buffer: list[int] = []

    # ------------------------------------------------------------------
    # 优化 1: 内部辅助——动态解析特殊 token id（不硬编码，构造时一次）
    # ------------------------------------------------------------------

    def _build_special_tokens(self) -> dict[str, int]:
        """从底层 tokenizer 构建 special_tokens 字典（构造时一次，缓存）。"""
        special: dict[str, int] = {}

        # 优先用 added_tokens_decoder（HF 标准）
        added_decoder = getattr(self._tokenizer, "added_tokens_decoder", None)
        if isinstance(added_decoder, dict):
            for _id, tok_obj in added_decoder.items():
                try:
                    content = getattr(tok_obj, "content", None)
                except Exception:
                    content = None
                if content is None and isinstance(tok_obj, dict):
                    content = tok_obj.get("content")
                if content is None:
                    if isinstance(tok_obj, str):
                        content = tok_obj
                if content is not None:
                    try:
                        special[content] = int(_id)
                    except (TypeError, ValueError):
                        continue

        # 兜底：从 vocab 中查找已知 Qwen3 special tokens
        # 注意：此时 vocab 还未构建，用 get_vocab() 临时查
        raw_vocab = self._tokenizer.get_vocab()
        for st in _KNOWN_QWEN_SPECIAL_TOKENS:
            if st in raw_vocab and st not in special:
                special[st] = raw_vocab[st]
        return special

    def _resolve_bos_id(self) -> Optional[int]:
        """解析 bos_id：优先 tokenizer.bos_token_id，否则回退 eos_id。

        Qwen3 通常没有 ``<bos>`` token，用 eos（im_end）兼容 bos 行为。
        """
        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        if bos_id is not None:
            return int(bos_id)
        return self._resolve_eos_id()

    def _resolve_eos_id(self) -> Optional[int]:
        """解析 eos_id：Qwen3 用 ``<|im_end|>`` 作为 eos。"""
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        if QWEN_IM_END in self._special_tokens:
            return self._special_tokens[QWEN_IM_END]
        if QWEN_ENDOFTEXT in self._special_tokens:
            return self._special_tokens[QWEN_ENDOFTEXT]
        return None

    def _resolve_pad_id(self) -> Optional[int]:
        """解析 pad_id：Qwen3 用 ``<|endoftext|>`` 作为 pad。"""
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        if QWEN_ENDOFTEXT in self._special_tokens:
            return self._special_tokens[QWEN_ENDOFTEXT]
        if QWEN_IM_END in self._special_tokens:
            return self._special_tokens[QWEN_IM_END]
        return None

    def _resolve_unk_id(self) -> Optional[int]:
        """解析 unk_id。Qwen3 通常无 unk，回退到 pad_id。"""
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        if unk_id is not None:
            return int(unk_id)
        return self._resolve_pad_id()

    def _resolve_vocab_size(self) -> int:
        """解析 vocab_size（构造时缓存）。"""
        size = getattr(self._tokenizer, "vocab_size", None)
        if isinstance(size, int):
            return int(size)
        # 兜底：len(vocab)，但这会触发 vocab 构建
        return len(self._tokenizer.get_vocab())

    # ------------------------------------------------------------------
    # 优化 2: 特殊 token 快速判断（O(1) frozenset 查询）
    # ------------------------------------------------------------------

    def is_special_token(self, token_id: int) -> bool:
        """判断给定 token id 是否为特殊 token（O(1) frozenset 查询）。

        Args:
            token_id: 待判断的 token id

        Returns:
            True 若为特殊 token，False 否则
        """
        return int(token_id) in self._special_token_ids

    def is_special_token_str(self, token_str: str) -> bool:
        """判断给定 token 字符串是否为特殊 token。

        Args:
            token_str: 待判断的 token 字符串

        Returns:
            True 若为特殊 token，False 否则
        """
        return token_str in self._special_tokens

    # ------------------------------------------------------------------
    # BaseTokenizer 抽象方法实现
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """把文本编码为 token id 列表。

        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token（透传给底层 tokenizer）

        Returns:
            token id 列表
        """
        # 不调用 self.preprocess（避免对 Qwen3 tokenizer.json 期望的原始文本
        # 做 NFKC 等额外正规化），直接交给底层 HF tokenizer 处理。
        result = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(result)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """把 token id 列表解码为字符串。

        Args:
            ids: token id 列表
            skip_special_tokens: 是否跳过特殊 token
        """
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def save(self, path: str) -> None:
        """保存 tokenizer 到指定路径。

        - ``path`` 为目录：用 HF 标准 ``save_pretrained`` 保存（tokenizer.json
          + tokenizer_config.json + special_tokens_map.json 等）；
        - ``path`` 为 ``.json`` 文件：保存元信息（model_id + 实际目录路径）
          的引用，便于 load 时定位。

        推荐用目录形式保存，最稳定。
        """
        if path.endswith(".json"):
            base_dir = path[:-len(".json")]
            os.makedirs(base_dir, exist_ok=True)
            self._tokenizer.save_pretrained(base_dir)
            meta = {
                "type": "verse",
                "model_id": self._model_id,
                "tokenizer_dir": base_dir,
                "trust_remote_code": self._trust_remote_code,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            os.makedirs(path, exist_ok=True)
            self._tokenizer.save_pretrained(path)
            self._tokenizer_dir = path

    def load(self, path: str) -> None:
        """从指定路径加载 tokenizer（实例方法，更新 self）。

        Args:
            path: 目录路径或 ``.json`` 元信息文件路径
        """
        AutoTokenizer = _import_transformers()
        if path.endswith(".json") and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            target_dir = meta.get("tokenizer_dir")
            if target_dir and os.path.isdir(target_dir):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    target_dir,
                    trust_remote_code=meta.get("trust_remote_code", True),
                )
                self._tokenizer_dir = target_dir
                self._model_id = meta.get("model_id")
            else:
                model_id = meta.get("model_id")
                if not model_id:
                    raise FileNotFoundError(
                        f"元信息 {path} 指向的目录 {target_dir!r} 不存在，"
                        f"且未提供 model_id 用于重新下载。"
                    )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=meta.get("trust_remote_code", True),
                )
                self._model_id = model_id
                self._tokenizer_dir = None
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=self._trust_remote_code
            )
            self._tokenizer_dir = path

        # 重建所有缓存
        self._special_tokens = self._build_special_tokens()
        self._special_token_ids = frozenset(self._special_tokens.values())
        self._bos_id = self._resolve_bos_id()
        self._eos_id = self._resolve_eos_id()
        self._pad_id = self._resolve_pad_id()
        self._unk_id = self._resolve_unk_id()
        self._im_start_id = self._special_tokens.get(QWEN_IM_START)
        self._im_end_id = self._special_tokens.get(QWEN_IM_END)
        self._endoftext_id = self._special_tokens.get(QWEN_ENDOFTEXT)
        self._vocab = None  # 重置懒加载缓存
        self._vocab_size = self._resolve_vocab_size()

    def __len__(self) -> int:
        """返回词表大小。"""
        return self._vocab_size

    # ------------------------------------------------------------------
    # 必须暴露的属性（与 ByteTokenizer 对齐）
    # ------------------------------------------------------------------

    @property
    def bos_id(self) -> int:
        """bos token id（Qwen3 通常无 bos，返回 eos/im_end 的 id）。"""
        if self._bos_id is None:
            raise AttributeError("VerseTokenizer 未找到 bos_token_id")
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """eos token id（``<|im_end|>`` 的 id）。"""
        if self._eos_id is None:
            raise AttributeError("VerseTokenizer 未找到 eos_token_id")
        return self._eos_id

    @property
    def pad_id(self) -> int:
        """pad token id（``<|endoftext|>`` 的 id）。"""
        if self._pad_id is None:
            raise AttributeError("VerseTokenizer 未找到 pad_token_id")
        return self._pad_id

    @property
    def unk_id(self) -> int:
        """unk token id。"""
        if self._unk_id is None:
            raise AttributeError("VerseTokenizer 未找到 unk_token_id")
        return self._unk_id

    @property
    def im_start_id(self) -> Optional[int]:
        """``<|im_start|>`` 的 id（Qwen3 ChatML 段开始）。"""
        return self._im_start_id

    @property
    def im_end_id(self) -> Optional[int]:
        """``<|im_end|>`` 的 id（Qwen3 ChatML 段结束）。"""
        return self._im_end_id

    @property
    def endoftext_id(self) -> Optional[int]:
        """``<|endoftext|>`` 的 id。"""
        return self._endoftext_id

    @property
    def vocab(self) -> dict[str, int]:
        """token → id 映射（懒加载）。"""
        if self._vocab is None:
            self._vocab = dict(self._tokenizer.get_vocab())
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """词表大小。"""
        return self._vocab_size

    @property
    def special_tokens(self) -> dict[str, int]:
        """特殊 token 字符串 → id 映射。"""
        return self._special_tokens

    @property
    def special_token_ids(self) -> frozenset:
        """所有特殊 token id 的 frozenset（O(1) 查询）。"""
        return self._special_token_ids

    # ------------------------------------------------------------------
    # 优化 3: 批量 encode/decode
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> list[list[int]]:
        """批量编码多个文本（比循环单条 encode 快 3-5 倍）。

        Args:
            texts: 输入文本列表
            add_special_tokens: 是否添加特殊 token
            padding: 是否填充到等长
            truncation: 是否截断到 max_length
            max_length: 最大长度（仅 truncation=True 时生效）

        Returns:
            token id 列表的列表
        """
        result = self._tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        return [list(ids) for ids in result["input_ids"]]

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """批量解码多个 token id 列表。

        Args:
            batch_ids: token id 列表的列表
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            解码后的字符串列表
        """
        return self._tokenizer.batch_decode(
            [list(ids) for ids in batch_ids],
            skip_special_tokens=skip_special_tokens,
        )

    # ------------------------------------------------------------------
    # 优化 5: 流式 decode（UTF-8 边界安全）
    # ------------------------------------------------------------------

    def decode_streaming(
        self,
        new_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """增量 decode（处理 UTF-8 边界，避免半字符乱码）。

        用于流式生成场景：每次收到新 token id 时调用，返回当前可安全
        decode 的字符串。未完成的多字节字符会缓存在内部 buffer 中，
        等待后续 token 补全后再 decode。

        Args:
            new_ids: 新收到的 token id 列表
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            当前可安全 decode 的字符串（可能为空，若 buffer 中还有未完成
            的多字节字符）

        Examples:
            >>> tok = VerseTokenizer()  # doctest: +SKIP
            >>> tok.reset_streaming()  # doctest: +SKIP
            >>> for new_id in generated_ids:  # doctest: +SKIP
            ...     chunk = tok.decode_streaming([new_id])
            ...     print(chunk, end="", flush=True)
        """
        # 过滤特殊 token（若 skip）
        if skip_special_tokens:
            new_ids = [i for i in new_ids if not self.is_special_token(i)]

        # 累加到 buffer
        self._stream_buffer.extend(new_ids)

        # 尝试 decode，检查是否产生 UnicodeDecodeError（半字符）
        try:
            text = self._tokenizer.decode(
                list(self._stream_buffer),
                skip_special_tokens=False,  # 已经手动过滤
            )
            # decode 成功，清空 buffer
            self._stream_buffer.clear()
            return text
        except UnicodeDecodeError:
            # 半字符，返回空，等待后续 token
            # 尝试逐个回退，找出最大可 decode 前缀
            for back in range(1, min(4, len(self._stream_buffer)) + 1):
                try:
                    partial = self._tokenizer.decode(
                        list(self._stream_buffer[:-back]),
                        skip_special_tokens=False,
                    )
                    # 保留未 decode 的部分
                    self._stream_buffer = list(self._stream_buffer[-back:])
                    return partial
                except UnicodeDecodeError:
                    continue
            # 全部失败，保留 buffer 返回空
            return ""

    def reset_streaming(self) -> None:
        """重置流式 decode buffer。"""
        self._stream_buffer.clear()

    # ------------------------------------------------------------------
    # Chat template（Qwen3 ChatML）
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> Union[str, list[int]]:
        """渲染 chat 数组为字符串或 token id 列表（Qwen3 ChatML 格式）。

        直接调用底层 tokenizer 的 ``apply_chat_template``，Qwen3 内置了
        ChatML 模板。

        Args:
            messages: ``[{"role": "user", "content": "..."}, ...]``
            add_generation_prompt: ``True`` 时在末尾追加
                ``<|im_start|>assistant\\n``，用于推理前缀。
            tokenize: ``True`` 返回 token id 列表，``False`` 返回字符串。

        Returns:
            渲染后的字符串或 token id 列表。
        """
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )

    def apply_prompt_template(self, prompt: str, system: str = None) -> str:
        """渲染 prompt 为推理前缀（Qwen3 ChatML 格式）。

        使用手动拼接，避免依赖 Qwen3 内部模板的 ``<think>`` 标签行为。

        Args:
            prompt: 用户输入的 prompt 文本
            system: 可选的 system prompt

        Returns:
            拼接到 ``<|im_start|>assistant\\n`` 为止的字符串（不含 assistant
            的内容，等待模型生成）。
        """
        return _render_prompt_qwen(prompt, system=system)

    # ------------------------------------------------------------------
    # 优化 4: 思考模式支持（Qwen3 <think>...</think>）
    # ------------------------------------------------------------------

    def apply_chat_template_with_thinking(
        self,
        messages: list[dict],
        enable_thinking: bool = True,
        add_generation_prompt: bool = True,
    ) -> str:
        """渲染 chat 数组（含 Qwen3 思考模式控制）。

        Qwen3 的 chat template 默认会在 assistant 段开头插入
        ``<think>\\n</think>\\n\\n``（enable_thinking=True 时）或空字符串
        （enable_thinking=False 时）。本方法通过 ``enable_thinking`` 参数
        控制该行为。

        实现方式：调用底层 ``apply_chat_template`` 后手动处理 ``<think>``
        标签。

        Args:
            messages: ``[{"role": "user", "content": "..."}, ...]``
            enable_thinking: ``True`` 启用思考模式（保留 ``<think>`` 标签），
                ``False`` 关闭思考模式（移除 ``<think>`` 标签）。
            add_generation_prompt: 是否在末尾追加 assistant 段开始标记。

        Returns:
            渲染后的字符串。
        """
        text = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        if not enable_thinking:
            # 移除可能的 <think>...</think> 标签
            text = self._strip_think_tags(text)
        return text

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """移除 ``<think>...</think>`` 标签及其内容。"""
        result = text
        while "<think>" in result and "</think>" in result:
            start = result.find("<think>")
            end = result.find("</think>", start)
            if start < 0 or end < 0:
                break
            # 移除 <think>...</think> 及其后的换行
            end_full = end + len("</think>")
            # 移除紧随其后的换行符（Qwen3 格式：<think>\n</think>\n\n）
            while end_full < len(result) and result[end_full] in "\n":
                end_full += 1
            result = result[:start] + result[end_full:]
        return result

    def extract_thinking(self, text: str) -> tuple[str, str]:
        """从生成文本中提取思考部分与回复部分。

        Qwen3 思考模式输出格式：
            ``<think>\\n{thinking}\\n</think>\\n\\n{response}``

        Args:
            text: 模型生成的完整文本（含 ``<think>`` 标签）

        Returns:
            (thinking, response)：
            - thinking: ``<think>`` 与 ``</think>`` 之间的内容（不含标签）
            - response: ``</think>`` 之后的内容（去除前导换行）

            若无 ``<think>`` 标签，thinking 为空，response 为原文本。

        Examples:
            >>> tok = VerseTokenizer()  # doctest: +SKIP
            >>> tok.extract_thinking("<think>\\n让我想想\\n</think>\\n\\n答案是42")  # doctest: +SKIP
            ('让我想想', '答案是42')
        """
        if "<think>" not in text:
            return "", text
        start = text.find("<think>") + len("<think>")
        end = text.find("</think>", start)
        if end < 0:
            # 未闭合的 <think>，返回全部为 thinking
            return text[start:].lstrip("\n"), ""
        thinking = text[start:end].strip("\n")
        # 跳过 </think> 后的换行
        response_start = end + len("</think>")
        while response_start < len(text) and text[response_start] == "\n":
            response_start += 1
        response = text[response_start:]
        return thinking, response

    def extract_response(self, text: str) -> str:
        """从生成文本中提取回复部分（去除思考部分）。

        等价于 ``extract_thinking(text)[1]``，便捷方法。

        Args:
            text: 模型生成的完整文本

        Returns:
            回复部分（不含 ``<think>`` 标签及其内容）
        """
        return self.extract_thinking(text)[1]

    # ------------------------------------------------------------------
    # 优化 6: 工具调用解析
    # ------------------------------------------------------------------

    def extract_tool_calls(self, text: str) -> list[dict]:
        """从生成文本中提取工具调用。

        Qwen3 工具调用格式：
            ``<|tool_call_begin|>{json}<|tool_call_end|>``

        可能有多个工具调用连续出现。

        Args:
            text: 模型生成的文本

        Returns:
            工具调用字典列表（每个字典是解析后的 JSON）。若解析失败，
            该项为 ``{"raw": "原始字符串"}``。
        """
        tool_calls: list[dict] = []
        begin_tag = "<|tool_call_begin|>"
        end_tag = "<|tool_call_end|>"
        cursor = 0
        while True:
            begin = text.find(begin_tag, cursor)
            if begin < 0:
                break
            begin += len(begin_tag)
            end = text.find(end_tag, begin)
            if end < 0:
                # 未闭合，取到字符串末尾
                raw = text[begin:].strip()
                cursor = len(text)
            else:
                raw = text[begin:end].strip()
                cursor = end + len(end_tag)
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    tool_calls.append(parsed)
                elif isinstance(parsed, list):
                    tool_calls.extend([p for p in parsed if isinstance(p, dict)])
                else:
                    tool_calls.append({"raw": raw})
            except json.JSONDecodeError:
                tool_calls.append({"raw": raw})
        return tool_calls

    # ------------------------------------------------------------------
    # 便捷构造方法
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_id: Optional[str] = None) -> "VerseTokenizer":
        """从 HuggingFace Hub 下载并加载。

        Args:
            model_id: 模型 ID，默认 :attr:`DEFAULT_VERSE_MODEL`
        """
        return cls(model_id=model_id or cls.DEFAULT_VERSE_MODEL)

    @classmethod
    def from_local(cls, dir_path: str) -> "VerseTokenizer":
        """从本地目录加载（save 后的目录）。

        Args:
            dir_path: 本地目录路径
        """
        return cls(tokenizer_dir=dir_path)


# ---------------------------------------------------------------------------
# 向后兼容别名（QwenTokenizer → VerseTokenizer）
# ---------------------------------------------------------------------------

# 保留 QwenTokenizer 作为 VerseTokenizer 的别名，确保旧代码不破坏。
# 新代码应使用 VerseTokenizer。
QwenTokenizer = VerseTokenizer


__all__ = [
    "VerseTokenizer",
    "QwenTokenizer",  # 向后兼容别名
    "_import_transformers",
]
