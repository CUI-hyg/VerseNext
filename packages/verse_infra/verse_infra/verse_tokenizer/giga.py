"""GigaTokenizerWrapper：包装 gigatoken 库为 BaseTokenizer 接口。

设计目标
--------
包装社区高性能 tokenizer 库 ``gigatoken``（Rust 实现，~1000× 快于 HF tokenizers，
drop-in 兼容）为 :class:`verse_tokenizer.bpe.BaseTokenizer` 接口。

**不重新实现 BPE/Unigram**：直接 ``import gigatoken`` 复用其 Rust 实现。

关键约束
--------
**lazy import gigatoken**：本模块在 import 时不引入 ``gigatoken``，
只有真正调用 :class:`GigaTokenizerWrapper` 构造函数时才会触发 ``gigatoken``
的 import。这样在不安装 ``gigatoken`` 的环境下，``verse_tokenizer`` 的其他
tokenizer（BPE / Byte / Char / Verse）依然可用。

两种工作模式
------------
1. **兼容模式**（``native=False``，默认）：
   - 输入为字符串（model_id）→ 先用 HuggingFace ``AutoTokenizer.from_pretrained``
     加载 HF tokenizer，再用 ``gt.Tokenizer(hf_tok).as_hf()`` 包装为 HF 兼容对象；
   - 输入为 HF tokenizer 实例 → 直接 ``gt.Tokenizer(hf_tok).as_hf()``。
   - 兼容模式下 wrapper 内部持有 HF tokenizer（用于 ``apply_chat_template`` 等接口）。
   - encode/decode 委托给 ``as_hf()`` 返回的兼容对象（内部走 Rust，仍有显著加速）。

2. **原生模式**（``native=True``）：
   - ``gt.Tokenizer(model_id_or_tokenizer)`` 直接构造（gigatoken 原生 API）。
   - 适用于纯 gigatoken 工作流（最快，但不保留 HF 接口）。

向后兼容
--------
- ``bos_id`` / ``eos_id`` / ``pad_id`` / ``unk_id`` 属性对齐 :class:`VerseTokenizer`；
- ``vocab`` 属性为 ``dict[str, int]``（懒加载）；
- ``apply_chat_template`` 委托底层 HF tokenizer（兼容模式下）。
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union, List

from .bpe import BaseTokenizer


# ---------------------------------------------------------------------------
# lazy import helpers
# ---------------------------------------------------------------------------


def _import_gigatoken():
    """延迟导入 ``gigatoken``。

    不安装 ``gigatoken`` 时抛出明确的 ``ImportError``，提示安装方式与降级路径。
    """
    try:
        import gigatoken as gt
        return gt
    except ImportError as e:
        raise ImportError(
            "GigaTokenizerWrapper 需要 gigatoken 库，请安装：pip install gigatoken\n"
            "或自动降级：load_tokenizer(kind='giga', fallback='verse')"
        ) from e


def _import_auto_tokenizer():
    """延迟导入 HuggingFace ``AutoTokenizer``（兼容模式需要）。"""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "GigaTokenizerWrapper 兼容模式需要 transformers 库，请安装：\n"
            "  pip install transformers\n"
            "或使用原生模式：GigaTokenizerWrapper(model_id, native=True)\n"
            "或自动降级：load_tokenizer(kind='giga', fallback='verse')"
        ) from e


# ---------------------------------------------------------------------------
# GigaTokenizerWrapper
# ---------------------------------------------------------------------------


class GigaTokenizerWrapper(BaseTokenizer):
    """包装 ``gigatoken`` 库为 :class:`BaseTokenizer` 接口。

    gigatoken 是社区 Rust 实现的高性能 tokenizer（~1000× 快于 HF tokenizers），
    在兼容模式下用 ``gt.Tokenizer(hf_tokenizer).as_hf()`` 返回 HF 兼容对象，
    输出与 HF tokenizer 完全一致（drop-in replacement）。

    支持：
        - 兼容模式（``native=False``，默认）：保留 HF 接口（apply_chat_template 等）；
        - 原生模式（``native=True``）：gigatoken 原生 API（最快）；
        - ``bos_id`` / ``eos_id`` / ``pad_id`` / ``unk_id`` 属性（兼容 VerseTokenizer）；
        - ``vocab`` 字典（懒加载）；
        - 批量 encode/decode（复用 gigatoken 批量接口）；
        - ``apply_chat_template``（委托底层 HF tokenizer）。

    Args:
        model_id_or_tokenizer: HuggingFace 模型 ID（如 ``"Qwen/Qwen3-32B"``）、
            本地目录路径、或 HF tokenizer 实例。``None`` 时使用
            :attr:`DEFAULT_GIGA_MODEL`。
        native: ``True`` 走 gigatoken 原生 API（``gt.Tokenizer(model_id)``）；
            ``False``（默认）走兼容模式（``gt.Tokenizer(hf_tok).as_hf()``）。
        trust_remote_code: 透传给 ``AutoTokenizer.from_pretrained``（兼容模式）。
        **kwargs: 透传给底层构造（保留扩展点）。

    Examples:
        >>> # 兼容模式（默认，drop-in 替换 VerseTokenizer）
        >>> tok = GigaTokenizerWrapper("Qwen/Qwen3-32B")  # doctest: +SKIP
        >>> ids = tok.encode("你好")  # doctest: +SKIP
        >>> # 原生模式（最快）
        >>> tok = GigaTokenizerWrapper("Qwen/Qwen3-32B", native=True)  # doctest: +SKIP
    """

    # 默认模型 ID（与 VerseTokenizer 一致，便于 drop-in 替换）
    DEFAULT_GIGA_MODEL = "Qwen/Qwen3-32B"

    def __init__(
        self,
        model_id_or_tokenizer=None,
        native: bool = False,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        # lazy import：构造时才触发 gigatoken 加载
        gt = _import_gigatoken()
        self._native = bool(native)
        self._trust_remote_code = trust_remote_code
        # 兼容模式下保留 HF tokenizer 引用（用于 apply_chat_template 等接口）
        self._hf_tokenizer = None
        self._model_id: Optional[str] = None
        self._tokenizer_dir: Optional[str] = None

        # 解析输入
        if model_id_or_tokenizer is None:
            model_id_or_tokenizer = self.DEFAULT_GIGA_MODEL

        if self._native:
            # 原生模式：gt.Tokenizer(model_id) 直接构造
            # gigatoken 原生 API 接受 HF model name / 本地路径
            self._tokenizer = gt.Tokenizer(model_id_or_tokenizer)
            if isinstance(model_id_or_tokenizer, str):
                self._model_id = model_id_or_tokenizer
        else:
            # 兼容模式：gt.Tokenizer(hf_tok).as_hf()
            if isinstance(model_id_or_tokenizer, str):
                # 字符串：先用 HF AutoTokenizer 加载
                AutoTokenizer = _import_auto_tokenizer()
                # 兼容本地目录路径
                if os.path.isdir(model_id_or_tokenizer):
                    hf_tok = AutoTokenizer.from_pretrained(
                        model_id_or_tokenizer,
                        trust_remote_code=trust_remote_code,
                    )
                    self._tokenizer_dir = model_id_or_tokenizer
                else:
                    hf_tok = AutoTokenizer.from_pretrained(
                        model_id_or_tokenizer,
                        trust_remote_code=trust_remote_code,
                    )
                    self._model_id = model_id_or_tokenizer
                self._hf_tokenizer = hf_tok
                self._tokenizer = gt.Tokenizer(hf_tok).as_hf()
            else:
                # 假定是 HF tokenizer 实例
                hf_tok = model_id_or_tokenizer
                self._hf_tokenizer = hf_tok
                self._tokenizer = gt.Tokenizer(hf_tok).as_hf()

        # --------------------------------------------------------------
        # 缓存 vocab 信息（构造时一次解析）
        # --------------------------------------------------------------
        self._bos_id: Optional[int] = self._resolve_bos_id()
        self._eos_id: Optional[int] = self._resolve_eos_id()
        self._pad_id: Optional[int] = self._resolve_pad_id()
        self._unk_id: Optional[int] = self._resolve_unk_id()
        self._vocab_size: int = self._resolve_vocab_size()
        # vocab 字典懒加载（Qwen3 vocab ~151936，构建开销不可忽视）
        self._vocab: Optional[dict[str, int]] = None

        # GigaTokenizerWrapper 不在 encode 时自动加 bos/eos（由 chat template 处理）
        self.auto_add_special_tokens = False

    # ------------------------------------------------------------------
    # 内部辅助：解析特殊 token id（构造时一次，缓存）
    # ------------------------------------------------------------------

    def _resolve_bos_id(self) -> Optional[int]:
        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        if bos_id is not None:
            return int(bos_id)
        # 兼容模式下从 HF tokenizer 兜底
        if self._hf_tokenizer is not None:
            hf_bos = getattr(self._hf_tokenizer, "bos_token_id", None)
            if hf_bos is not None:
                return int(hf_bos)
        return None

    def _resolve_eos_id(self) -> Optional[int]:
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        if self._hf_tokenizer is not None:
            hf_eos = getattr(self._hf_tokenizer, "eos_token_id", None)
            if hf_eos is not None:
                return int(hf_eos)
        return None

    def _resolve_pad_id(self) -> Optional[int]:
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        if self._hf_tokenizer is not None:
            hf_pad = getattr(self._hf_tokenizer, "pad_token_id", None)
            if hf_pad is not None:
                return int(hf_pad)
        return None

    def _resolve_unk_id(self) -> Optional[int]:
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        if unk_id is not None:
            return int(unk_id)
        if self._hf_tokenizer is not None:
            hf_unk = getattr(self._hf_tokenizer, "unk_token_id", None)
            if hf_unk is not None:
                return int(hf_unk)
        return None

    def _resolve_vocab_size(self) -> int:
        # 优先用底层 tokenizer 的 vocab_size 属性
        size = getattr(self._tokenizer, "vocab_size", None)
        if isinstance(size, int):
            return int(size)
        # 兜底：len(tokenizer)（HF 兼容对象支持 __len__）
        try:
            return len(self._tokenizer)
        except (TypeError, AttributeError):
            # 最后兜底：get_vocab() 字典大小（会触发构建，开销大）
            return len(self._tokenizer.get_vocab())

    # ------------------------------------------------------------------
    # BaseTokenizer 抽象方法实现
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """把文本编码为 token id 列表。

        委托给底层 gigatoken 包装的 tokenizer（兼容模式下走 Rust 实现，
        性能远高于纯 Python 的 HF tokenizer）。

        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token（透传给底层 tokenizer）

        Returns:
            token id 列表
        """
        result = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(result)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """把 token id 列表解码为字符串。

        Args:
            ids: token id 列表
            skip_special_tokens: 是否跳过特殊 token
        """
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def save(self, path: str) -> None:
        """保存 tokenizer 到指定路径。

        - ``path`` 为目录：用 HF 标准 ``save_pretrained`` 保存（兼容模式）；
        - ``path`` 为 ``.json`` 文件：保存元信息（model_id + 目录路径）的引用。

        原生模式下仅保存 model_id 引用（gigatoken 原生 tokenizer 的持久化
        由 gigatoken 库自身管理）。
        """
        if self._native:
            # 原生模式：仅保存元信息引用
            if path.endswith(".json"):
                meta = {
                    "type": "giga",
                    "native": True,
                    "model_id": self._model_id,
                }
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            else:
                os.makedirs(path, exist_ok=True)
                meta = {
                    "type": "giga",
                    "native": True,
                    "model_id": self._model_id,
                }
                with open(os.path.join(path, "giga_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            return

        # 兼容模式：用底层 HF 兼容对象的 save_pretrained
        if path.endswith(".json"):
            base_dir = path[:-len(".json")]
            os.makedirs(base_dir, exist_ok=True)
            save_tok = self._hf_tokenizer if self._hf_tokenizer is not None else self._tokenizer
            if hasattr(save_tok, "save_pretrained"):
                save_tok.save_pretrained(base_dir)
            meta = {
                "type": "giga",
                "native": False,
                "model_id": self._model_id,
                "tokenizer_dir": base_dir,
                "trust_remote_code": self._trust_remote_code,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            os.makedirs(path, exist_ok=True)
            save_tok = self._hf_tokenizer if self._hf_tokenizer is not None else self._tokenizer
            if hasattr(save_tok, "save_pretrained"):
                save_tok.save_pretrained(path)
            self._tokenizer_dir = path

    def load(self, path: str) -> None:
        """从指定路径加载 tokenizer（实例方法，更新 self）。

        Args:
            path: 目录路径或 ``.json`` 元信息文件路径
        """
        gt = _import_gigatoken()
        if path.endswith(".json") and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            native = bool(meta.get("native", False))
            self._native = native
            self._model_id = meta.get("model_id")
            self._trust_remote_code = meta.get("trust_remote_code", True)
            target_dir = meta.get("tokenizer_dir")
            if native:
                # 原生模式重新构造
                if self._model_id:
                    self._tokenizer = gt.Tokenizer(self._model_id)
            else:
                AutoTokenizer = _import_auto_tokenizer()
                if target_dir and os.path.isdir(target_dir):
                    hf_tok = AutoTokenizer.from_pretrained(
                        target_dir, trust_remote_code=self._trust_remote_code
                    )
                    self._tokenizer_dir = target_dir
                elif self._model_id:
                    hf_tok = AutoTokenizer.from_pretrained(
                        self._model_id, trust_remote_code=self._trust_remote_code
                    )
                else:
                    raise FileNotFoundError(
                        f"元信息 {path} 既无有效 tokenizer_dir 也无 model_id"
                    )
                self._hf_tokenizer = hf_tok
                self._tokenizer = gt.Tokenizer(hf_tok).as_hf()
        else:
            # 目录路径：当作 HF tokenizer 目录加载（兼容模式）
            AutoTokenizer = _import_auto_tokenizer()
            hf_tok = AutoTokenizer.from_pretrained(
                path, trust_remote_code=self._trust_remote_code
            )
            self._hf_tokenizer = hf_tok
            self._tokenizer = gt.Tokenizer(hf_tok).as_hf()
            self._tokenizer_dir = path
            self._native = False

        # 重建所有缓存
        self._bos_id = self._resolve_bos_id()
        self._eos_id = self._resolve_eos_id()
        self._pad_id = self._resolve_pad_id()
        self._unk_id = self._resolve_unk_id()
        self._vocab = None
        self._vocab_size = self._resolve_vocab_size()

    def __len__(self) -> int:
        """返回词表大小。"""
        return self._vocab_size

    # ------------------------------------------------------------------
    # 必须暴露的属性（与 VerseTokenizer 对齐）
    # ------------------------------------------------------------------

    @property
    def bos_id(self) -> int:
        """bos token id（可能为 None，调用方需处理）。"""
        if self._bos_id is None:
            raise AttributeError("GigaTokenizerWrapper 未找到 bos_token_id")
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """eos token id。"""
        if self._eos_id is None:
            raise AttributeError("GigaTokenizerWrapper 未找到 eos_token_id")
        return self._eos_id

    @property
    def pad_id(self) -> int:
        """pad token id。"""
        if self._pad_id is None:
            raise AttributeError("GigaTokenizerWrapper 未找到 pad_token_id")
        return self._pad_id

    @property
    def unk_id(self) -> int:
        """unk token id。"""
        if self._unk_id is None:
            raise AttributeError("GigaTokenizerWrapper 未找到 unk_token_id")
        return self._unk_id

    @property
    def vocab(self) -> dict[str, int]:
        """token → id 映射（懒加载）。"""
        if self._vocab is None:
            raw_vocab = self._tokenizer.get_vocab()
            self._vocab = dict(raw_vocab) if raw_vocab is not None else {}
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """词表大小。"""
        return self._vocab_size

    @property
    def native(self) -> bool:
        """是否为原生模式。"""
        return self._native

    @property
    def hf_tokenizer(self):
        """兼容模式下持有的底层 HF tokenizer（原生模式下为 None）。"""
        return self._hf_tokenizer

    # ------------------------------------------------------------------
    # 批量 encode/decode（复用底层批量接口）
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """批量编码多个文本（复用 gigatoken 批量接口，性能优势显著）。

        Args:
            texts: 输入文本列表
            add_special_tokens: 是否添加特殊 token
            padding: 是否填充到等长
            truncation: 是否截断到 max_length
            max_length: 最大长度（仅 truncation=True 时生效）

        Returns:
            token id 列表的列表
        """
        # 优先用底层 __call__（HF 兼容对象支持，可走 Rust 批量路径）
        if hasattr(self._tokenizer, "__call__"):
            result = self._tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            return [list(ids) for ids in result["input_ids"]]
        # 兜底：逐条 encode
        return [
            self.encode(t, add_special_tokens=add_special_tokens)
            for t in texts
        ]

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """批量解码多个 token id 列表。

        Args:
            batch_ids: token id 列表的列表
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            解码后的字符串列表
        """
        if hasattr(self._tokenizer, "batch_decode"):
            return self._tokenizer.batch_decode(
                [list(ids) for ids in batch_ids],
                skip_special_tokens=skip_special_tokens,
            )
        # 兜底：逐条 decode
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    # ------------------------------------------------------------------
    # apply_chat_template（委托底层 HF tokenizer）
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        **kwargs,
    ) -> Union[str, list[int]]:
        """渲染 chat 数组为字符串或 token id 列表。

        兼容模式下委托底层 HF tokenizer 的 ``apply_chat_template``
        （gigatoken 兼容模式下保留 HF 接口）。

        Args:
            messages: ``[{"role": "user", "content": "..."}, ...]``
            **kwargs: 透传给底层 ``apply_chat_template``
                （如 ``add_generation_prompt`` / ``tokenize``）

        Returns:
            渲染后的字符串或 token id 列表。

        Raises:
            RuntimeError: 原生模式下无 HF tokenizer 可委托时抛出。
        """
        if self._hf_tokenizer is not None and hasattr(
            self._hf_tokenizer, "apply_chat_template"
        ):
            return self._hf_tokenizer.apply_chat_template(messages, **kwargs)
        # 兜底：底层 gigatoken 兼容对象可能也支持
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        raise RuntimeError(
            "GigaTokenizerWrapper.apply_chat_template 需要兼容模式（native=False）"
            "或底层 tokenizer 支持 apply_chat_template 接口"
        )

    # ------------------------------------------------------------------
    # 便捷构造方法
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: Optional[str] = None,
        native: bool = False,
    ) -> "GigaTokenizerWrapper":
        """从 HuggingFace Hub 下载并加载（兼容模式）。

        Args:
            model_id: 模型 ID，默认 :attr:`DEFAULT_GIGA_MODEL`
            native: ``True`` 走 gigatoken 原生 API
        """
        return cls(model_id_or_tokenizer=model_id or cls.DEFAULT_GIGA_MODEL, native=native)

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer, native: bool = False) -> "GigaTokenizerWrapper":
        """从已有的 HuggingFace tokenizer 实例构造（兼容模式）。

        Args:
            hf_tokenizer: HF tokenizer 实例
            native: ``True`` 走 gigatoken 原生 API（此时 hf_tokenizer 应为 model_id 字符串）
        """
        return cls(model_id_or_tokenizer=hf_tokenizer, native=native)


__all__ = [
    "GigaTokenizerWrapper",
    "_import_gigatoken",
]
