"""Chat template：把 chat 数组 / prompt 字符串转为可编码的渲染字符串。

约定特殊 token 字符串
----------------------
- ``<|user|>`` / ``<|assistant|>`` / ``<|system|>``：对话角色标记
- ``<|eos|>``：序列结束标记
- ``<|bos|>``：序列开始标记

渲染格式
--------
- chat 数组：``<|user|>{content}<|assistant|>{content}<|eos|>``
- prompt 推理前缀：``<|user|>{prompt}<|assistant|>``

loss mask
---------
:meth:`split_prompt_completion` 用于把渲染后的字符串拆分为 prompt 部分
（loss 屏蔽，``ignore_index=-100``）与 completion 部分（参与 loss）。
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 特殊 token 字符串常量（与 unigram.SpecialTokens 保持一致）
# ---------------------------------------------------------------------------

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SYSTEM_TOKEN = "<|system|>"
EOS_TOKEN = "<|eos|>"
BOS_TOKEN = "<|bos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"


def render_chat(messages: list[dict]) -> str:
    """渲染 chat 数组为字符串。

    Args:
        messages: ``[{"role": "user", "content": "..."}, ...]``

    Returns:
        ``"<|user|>{content}<|assistant|>{content}<|eos|>"`` 这样的拼接字符串；
        若 ``role`` 不在 user/assistant/system 中也按相同模式渲染（用其字面值）。

    Examples:
        >>> render_chat([{"role": "user", "content": "你好"},
        ...              {"role": "assistant", "content": "你好！"}])
        '<|user|>你好<|assistant|>你好！<|eos|>'
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>{content}")
    parts.append(EOS_TOKEN)
    return "".join(parts)


def render_prompt(prompt: str) -> str:
    """渲染 prompt 字符串为推理前缀。

    Args:
        prompt: 用户输入的 prompt 文本

    Returns:
        ``"<|user|>{prompt}<|assistant|>"``，模型从此处开始生成回复。

    Examples:
        >>> render_prompt("你好")
        '<|user|>你好<|assistant|>'
    """
    return f"{USER_TOKEN}{prompt}{ASSISTANT_TOKEN}"


def split_prompt_completion(rendered: str) -> tuple[str, str]:
    """拆分渲染后的字符串为 (prompt_part, completion_part)。

    用于 loss mask：prompt 部分屏蔽（``ignore_index=-100``），
    completion 部分参与 loss。

    规则：找最后一个 ``<|assistant|>`` 的位置，其后的内容为 completion。

    Args:
        rendered: 渲染后的字符串（来自 :func:`render_chat` 或 :func:`render_prompt`）

    Returns:
        (prompt_part, completion_part)：
        - prompt_part 包含到 ``<|assistant|>``（含）
        - completion_part 是其后的内容（可能为空）

    Examples:
        >>> split_prompt_completion("<|user|>你好<|assistant|>你好！<|eos|>")
        ('<|user|>你好<|assistant|>', '你好！<|eos|>')
        >>> split_prompt_completion("no marker here")
        ('no marker here', '')
    """
    marker = ASSISTANT_TOKEN
    idx = rendered.rfind(marker)
    if idx < 0:
        return rendered, ""
    end = idx + len(marker)
    return rendered[:end], rendered[end:]


__all__ = [
    "USER_TOKEN",
    "ASSISTANT_TOKEN",
    "SYSTEM_TOKEN",
    "EOS_TOKEN",
    "BOS_TOKEN",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "render_chat",
    "render_prompt",
    "split_prompt_completion",
]
