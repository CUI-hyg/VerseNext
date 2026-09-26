"""chat_template.jinja 支持：jinja2 渲染对话模板（对齐 HF transformers 约定）。

模板上下文：
- ``messages``: ``[{"role": str, "content": str}, ...]``
- ``bos_token`` / ``eos_token`` / ``pad_token``: 特殊 token 字符串
- ``add_generation_prompt``: bool，为 true 时在末尾追加助手指令前缀

默认模板为 ChatML 风格，可通过自定义 ``chat_template.jinja`` 替换。
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateError

from verse_core import get_logger

logger = get_logger("chat_template")

# 与 transformers 对齐的最小渲染环境：不含随机/IO 能力
_ENV = Environment(undefined=StrictUndefined, keep_trailing_newline=True)


class ChatTemplate:
    """对话模板封装。

    用法::

        tpl = ChatTemplate.from_pretrained("my_tokenizer/")   # 读 chat_template.jinja
        tpl = ChatTemplate.default()                           # 内置 ChatML 模板
        text = tpl.render(messages, add_generation_prompt=True)
    """

    def __init__(self, template_src: str, source_name: str = "<default>") -> None:
        try:
            self._template = _ENV.from_string(template_src)
        except TemplateError as e:
            raise ValueError(f"chat_template.jinja 解析失败（{source_name}）: {e}") from e
        self.template_src = template_src
        self.source_name = source_name

    @classmethod
    def default(cls) -> "ChatTemplate":
        """内置 ChatML 风格模板（同包打包的 chat_templates/chatml.jinja）。"""
        path = Path(__file__).parent / "chat_templates" / "chatml.jinja"
        return cls(path.read_text(), source_name=str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> "ChatTemplate":
        path = Path(path)
        return cls(path.read_text(), source_name=str(path))

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, fallback_default: bool = True) -> "ChatTemplate":
        """从 tokenizer 目录加载；无模板文件时可回退内置模板。"""
        path = Path(directory) / "chat_template.jinja"
        if path.exists():
            return cls.from_file(path)
        if fallback_default:
            logger.info("目录 %s 无 chat_template.jinja，使用内置 ChatML 模板", directory)
            return cls.default()
        raise FileNotFoundError(path)

    def render(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = False,
        bos_token: str = "<|bos|>",
        eos_token: str = "<|eos|>",
        pad_token: str = "<|pad|>",
    ) -> str:
        """渲染对话为完整文本（含特殊 token 字符串，可直接交给 tokenizer.encode）。"""
        if not messages:
            raise ValueError("messages 不能为空")
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

    def encode_chat(
        self,
        messages: list[dict],
        tokenizer,
        *,
        add_generation_prompt: bool = False,
        train_on_last_turns: int = 1,
        bos_token: str = "<|bos|>",
        eos_token: str = "<|eos|>",
        pad_token: str = "<|pad|>",
    ) -> tuple[list[int], list[int]]:
        """渲染并编码对话，返回 ``(input_ids, labels)``。

        labels 只保留最后 ``train_on_last_turns`` 轮 assistant 回答（含其后的
        结束标记 ``<|im_end|>``）对应的 token，其余位置为 -100——即 SFT 中
        「只对回答部分计算 loss」的掩码策略。

        bos/eos/pad token 可自定义：tokenizer 词表缺少对应特殊 token 时应
        传空串，避免 BPE 兜底编码产生越界 id。

        定位方式：将目标 assistant 轮的 content 替换为唯一哨兵串重新渲染，
        哨兵在文本中的字节位置即回答区间起点（与具体模板解耦）。
        """
        import uuid

        from verse_tokenizer.masking import labels_from_spans

        def _render(msgs: list[dict]) -> str:
            return self.render(
                msgs, add_generation_prompt=add_generation_prompt,
                bos_token=bos_token, eos_token=eos_token, pad_token=pad_token,
            )

        text = _render(messages)
        ids, offsets = tokenizer.encode(text, return_offsets=True)
        text_bytes = text.encode("utf-8")

        spans: list[tuple[int, int]] = []
        assistant_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        # add_generation_prompt 时最后一条 assistant 属于"待生成"内容，不参与训练
        if add_generation_prompt and assistant_idx and assistant_idx[-1] == len(messages) - 1:
            assistant_idx = assistant_idx[:-1]
        targets = assistant_idx[-train_on_last_turns:] if train_on_last_turns > 0 else []
        for idx in targets:
            content = messages[idx]["content"]
            sentinel = f"@@VERSE_{uuid.uuid4().hex}@@"
            sent_messages = [dict(m) for m in messages]
            sent_messages[idx]["content"] = sentinel
            sent_text = _render(sent_messages)
            start = sent_text.encode("utf-8").find(sentinel.encode("utf-8"))
            if start < 0:
                logger.warning("哨兵定位失败，跳过第 %d 轮 assistant 掩码", idx)
                continue
            end = start + len(content.encode("utf-8"))
            spans.append((start, end))
            # 回答后的结束标记（如 <|im_end|>）也纳入训练区间
            tail = text_bytes[end:end + 16]
            for marker in sorted(tokenizer.special_tokens, key=len, reverse=True):
                if marker.startswith("<|im_end") or marker in ("<|end|>", "<|eos|>"):
                    if tail.startswith(marker.encode("utf-8")):
                        spans[-1] = (start, end + len(marker))
                        break

        labels = labels_from_spans(ids, offsets, spans)
        return ids, labels
