"""Qwen3 系列 tokenizer 包装器（lazy-import transformers）。

设计目标
--------
包装 ``transformers.AutoTokenizer`` 以加载 Qwen3-32B 等 Qwen3 系列模型的
tokenizer，并提供与 :class:`verse_tokenizer.bpe.BaseTokenizer` 一致的接口。

关键约束
--------
**lazy import transformers**：本模块在 import 时不引入 ``transformers``，
只有真正调用 :class:`QwenTokenizer` 构造函数时才会触发 ``transformers`` 的
import。这样在不安装 ``transformers`` 的环境下，``verse_tokenizer`` 的其他
tokenizer（BPE / Byte / Char / Unigram）依然可用。

Qwen3 ChatML 格式
-----------------
::

    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n{user}<|im_end|>\n
    <|im_start|>assistant\n{assistant}<|im_end|>\n

特殊 token（动态从 tokenizer 读取，不硬编码 id）：
    - ``<|im_start|>``：ChatML 段开始
    - ``<|im_end|>``：ChatML 段结束（也用作 eos）
    - ``<|endoftext|>``：文本结束（也用作 pad）

兼容性
------
- ``bos_id`` / ``eos_id`` / ``pad_id`` / ``unk_id`` 属性对齐
  :class:`ByteTokenizer`，兼容 ``evaluate.py`` 的 ``_get_eos_id`` fallback。
- ``vocab`` 属性为 ``dict[str, int]``，兼容 ``_get_eos_id`` 的 vocab 查询。
- ``apply_chat_template`` 返回字符串（注意与 ``BaseTokenizer`` 默认返回
  ``list[int]`` 不同，因为 QwenTokenizer 主要用于推理前缀拼接）。
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .bpe import BaseTokenizer
from .chat_template import (
    render_prompt_qwen as _render_prompt_qwen,
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
            "QwenTokenizer 需要 transformers 库。请安装：\n"
            "  pip install transformers\n"
            "或：\n"
            "  pip install 'verse-tokenizer[qwen]'"
        ) from e


# ---------------------------------------------------------------------------
# QwenTokenizer
# ---------------------------------------------------------------------------


class QwenTokenizer(BaseTokenizer):
    """Qwen3 系列 tokenizer 包装器，lazy-import transformers。

    支持：
        - 从 HuggingFace Hub 下载 Qwen3-32B tokenizer（首次使用）
        - 从本地目录加载（save 后）
        - Qwen3 ChatML chat template
        - ``bos_id`` / ``eos_id`` / ``pad_id`` / ``unk_id`` 属性（兼容 evaluate.py）
        - ``vocab`` 字典（兼容 ``_get_eos_id`` fallback）

    Args:
        model_id: HuggingFace 模型 ID（如 ``"Qwen/Qwen3-32B"``），首次使用时下载。
            若提供 ``tokenizer_dir`` 则忽略此参数。
        tokenizer_dir: 本地目录路径（save 后的目录），优先于 ``model_id``。
        trust_remote_code: 是否信任远程代码（Qwen 系列需要）。

    Examples:
        >>> tok = QwenTokenizer.from_pretrained("Qwen/Qwen3-32B")  # doctest: +SKIP
        >>> ids = tok.encode("你好")  # doctest: +SKIP
        >>> text = tok.apply_chat_template([  # doctest: +SKIP
        ...     {"role": "user", "content": "你好"},
        ... ])
    """

    # 默认下载的 Qwen 模型 ID
    DEFAULT_QWEN_MODEL = "Qwen/Qwen3-32B"

    def __init__(
        self,
        model_id: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        # 延迟导入：只有构造 QwenTokenizer 时才需要 transformers
        AutoTokenizer = _import_transformers()

        # 优先级：tokenizer_dir > model_id > DEFAULT_QWEN_MODEL
        if tokenizer_dir is not None:
            self._model_id: Optional[str] = None
            self._tokenizer_dir: Optional[str] = tokenizer_dir
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, trust_remote_code=trust_remote_code
            )
        else:
            self._model_id = model_id or self.DEFAULT_QWEN_MODEL
            self._tokenizer_dir = None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, trust_remote_code=trust_remote_code
            )

        self._trust_remote_code = trust_remote_code

        # 构造 vocab 字典（Qwen3 vocab 较大，约 151936，但一次性构建）
        self._vocab: dict[str, int] = dict(self._tokenizer.get_vocab())

        # 构建 special_tokens 字典（从 tokenizer.added_tokens_decoder 等动态获取）
        self._special_tokens: dict[str, int] = self._build_special_tokens()

        # 缓存属性
        self._bos_id: Optional[int] = self._resolve_bos_id()
        self._eos_id: Optional[int] = self._resolve_eos_id()
        self._pad_id: Optional[int] = self._resolve_pad_id()
        self._unk_id: Optional[int] = self._resolve_unk_id()

        # QwenTokenizer 不在 encode 时自动加 bos/eos（由 chat template 处理）
        self.auto_add_special_tokens = False

    # ------------------------------------------------------------------
    # 内部辅助：动态解析特殊 token id（不硬编码）
    # ------------------------------------------------------------------

    def _build_special_tokens(self) -> dict[str, int]:
        """从底层 tokenizer 构建 special_tokens 字典。"""
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
                    # tok_obj 可能直接是字符串
                    if isinstance(tok_obj, str):
                        content = tok_obj
                if content is not None:
                    try:
                        special[content] = int(_id)
                    except (TypeError, ValueError):
                        continue
        # 兜底：从 vocab 中查找常见 Qwen3 special tokens
        for st in (
            QWEN_IM_START, QWEN_IM_END, QWEN_ENDOFTEXT,
            "<|tool_call_begin|>", "<|tool_call_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>",
        ):
            if st in self._vocab and st not in special:
                special[st] = self._vocab[st]
        return special

    def _resolve_bos_id(self) -> Optional[int]:
        """解析 bos_id：优先 tokenizer.bos_token_id，否则回退 eos_id。

        Qwen3 通常没有 ``<bos>`` token，用 eos 兼容 bos 行为。
        """
        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        if bos_id is not None:
            return int(bos_id)
        # 回退到 eos_id（Qwen3 用 <|im_end|> 兼容）
        return self._resolve_eos_id()

    def _resolve_eos_id(self) -> Optional[int]:
        """解析 eos_id：Qwen3 用 ``<|im_end|>`` 作为 eos。"""
        # 优先用 tokenizer.eos_token_id
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        # 兜底：从 vocab 查 <|im_end|>
        if QWEN_IM_END in self._vocab:
            return self._vocab[QWEN_IM_END]
        if QWEN_ENDOFTEXT in self._vocab:
            return self._vocab[QWEN_ENDOFTEXT]
        return None

    def _resolve_pad_id(self) -> Optional[int]:
        """解析 pad_id：Qwen3 用 ``<|endoftext|>`` 作为 pad。"""
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        # 兜底：从 vocab 查 <|endoftext|>
        if QWEN_ENDOFTEXT in self._vocab:
            return self._vocab[QWEN_ENDOFTEXT]
        if QWEN_IM_END in self._vocab:
            return self._vocab[QWEN_IM_END]
        return None

    def _resolve_unk_id(self) -> Optional[int]:
        """解析 unk_id。"""
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        if unk_id is not None:
            return int(unk_id)
        # Qwen3 通常无 unk，回退到 pad_id
        return self._resolve_pad_id()

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
            # 元信息引用：保存 model_id 与一个指向实际目录的相对路径
            # 实际 tokenizer 文件保存到 path 同名目录（去掉 .json 后缀）
            base_dir = path[:-len(".json")]
            os.makedirs(base_dir, exist_ok=True)
            self._tokenizer.save_pretrained(base_dir)
            meta = {
                "type": "qwen",
                "model_id": self._model_id,
                "tokenizer_dir": base_dir,
                "trust_remote_code": self._trust_remote_code,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            # 目录形式
            os.makedirs(path, exist_ok=True)
            self._tokenizer.save_pretrained(path)
            # 同步更新内部 tokenizer_dir
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
                # 元信息指向的目录不存在，尝试用 model_id 重新加载
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
            # 目录形式
            self._tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=self._trust_remote_code
            )
            self._tokenizer_dir = path

        # 重建缓存
        self._vocab = dict(self._tokenizer.get_vocab())
        self._special_tokens = self._build_special_tokens()
        self._bos_id = self._resolve_bos_id()
        self._eos_id = self._resolve_eos_id()
        self._pad_id = self._resolve_pad_id()
        self._unk_id = self._resolve_unk_id()

    def __len__(self) -> int:
        """返回词表大小。"""
        # 优先用底层 tokenizer 的 vocab_size
        size = getattr(self._tokenizer, "vocab_size", None)
        if isinstance(size, int):
            return int(size)
        return len(self._vocab)

    # ------------------------------------------------------------------
    # 必须暴露的属性（与 ByteTokenizer 对齐）
    # ------------------------------------------------------------------

    @property
    def bos_id(self) -> int:
        """bos token id（Qwen3 通常无 bos，返回 eos/im_end 的 id）。"""
        if self._bos_id is None:
            raise AttributeError("QwenTokenizer 未找到 bos_token_id")
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """eos token id（``<|im_end|>`` 的 id）。"""
        if self._eos_id is None:
            raise AttributeError("QwenTokenizer 未找到 eos_token_id")
        return self._eos_id

    @property
    def pad_id(self) -> int:
        """pad token id（``<|endoftext|>`` 的 id）。"""
        if self._pad_id is None:
            raise AttributeError("QwenTokenizer 未找到 pad_token_id")
        return self._pad_id

    @property
    def unk_id(self) -> int:
        """unk token id。"""
        if self._unk_id is None:
            raise AttributeError("QwenTokenizer 未找到 unk_token_id")
        return self._unk_id

    @property
    def vocab(self) -> dict[str, int]:
        """token → id 映射。"""
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """词表大小。"""
        return len(self)

    @property
    def special_tokens(self) -> dict[str, int]:
        """特殊 token 字符串 → id 映射。"""
        return self._special_tokens

    # ------------------------------------------------------------------
    # Chat template（Qwen3 ChatML）
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict]) -> str:
        """渲染 chat 数组为字符串（Qwen3 ChatML 格式）。

        直接调用底层 tokenizer 的 ``apply_chat_template``，Qwen3 内置了
        ChatML 模板。

        Args:
            messages: ``[{"role": "user", "content": "..."}, ...]``

        Returns:
            渲染后的字符串（不 tokenize）。
        """
        return self._tokenizer.apply_chat_template(messages, tokenize=False)

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
    # 便捷构造方法
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_id: Optional[str] = None) -> "QwenTokenizer":
        """从 HuggingFace Hub 下载并加载。

        Args:
            model_id: 模型 ID，默认 :attr:`DEFAULT_QWEN_MODEL`
        """
        return cls(model_id=model_id or cls.DEFAULT_QWEN_MODEL)

    @classmethod
    def from_local(cls, dir_path: str) -> "QwenTokenizer":
        """从本地目录加载（save 后的目录）。

        Args:
            dir_path: 本地目录路径
        """
        return cls(tokenizer_dir=dir_path)


__all__ = [
    "QwenTokenizer",
    "_import_transformers",
]
