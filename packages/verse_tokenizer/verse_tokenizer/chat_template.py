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


# ---------------------------------------------------------------------------
# Qwen3 ChatML 格式
# ---------------------------------------------------------------------------
# Qwen3 系列模型使用 ChatML 格式，与上面的 ``<|user|>`` / ``<|assistant|>``
# 风格不同，采用 ``<|im_start|>{role}\n{content}<|im_end|>\n`` 结构。

QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"
QWEN_ENDOFTEXT = "<|endoftext|>"

# ChatML 角色标记前缀（``<|im_start|>{role}\n``）
_QWEN_ROLE_PREFIX = "<|im_start|>{role}\n"
_QWEN_ROLE_SUFFIX = "<|im_end|>\n"


def render_chat_qwen(
    messages: list[dict],
    add_generation_prompt: bool = False,
) -> str:
    """Qwen3 ChatML 格式渲染。

    Args:
        messages: ``[{"role": "user", "content": "..."}, ...]``，role 支持
            ``system`` / ``user`` / ``assistant`` / ``tool`` 等。
        add_generation_prompt: ``True`` 时在末尾追加
            ``<|im_start|>assistant\\n``，用于推理前缀。

    Returns:
        ChatML 格式字符串，例如::

            <|im_start|>system\n{system}<|im_end|>\n
            <|im_start|>user\n{user}<|im_end|>\n
            <|im_start|>assistant\n{assistant}<|im_end|>\n

    Examples:
        >>> render_chat_qwen([
        ...     {"role": "user", "content": "你好"},
        ...     {"role": "assistant", "content": "你好！"},
        ... ])
        '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！<|im_end|>\n'
        >>> render_chat_qwen(
        ...     [{"role": "user", "content": "你好"}],
        ...     add_generation_prompt=True,
        ... )
        '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{_QWEN_ROLE_PREFIX.format(role=role)}{content}{_QWEN_ROLE_SUFFIX}")
    if add_generation_prompt:
        parts.append(f"{_QWEN_ROLE_PREFIX.format(role='assistant')}")
    return "".join(parts)


def render_prompt_qwen(prompt: str, system: str = None) -> str:
    """Qwen3 prompt 模板（推理前缀）。

    Args:
        prompt: 用户输入的 prompt 文本
        system: 可选的 system prompt；为 ``None`` 时不输出 system 段

    Returns:
        拼接到 ``<|im_start|>assistant\\n`` 为止的字符串（不含 assistant 的
        内容，等待模型生成）。

    Examples:
        >>> render_prompt_qwen("你好")
        '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
        >>> render_prompt_qwen("你好", system="你是助手")
        '<|im_start|>system\n你是助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
    """
    parts: list[str] = []
    if system:
        parts.append(f"{_QWEN_ROLE_PREFIX.format(role='system')}{system}{_QWEN_ROLE_SUFFIX}")
    parts.append(f"{_QWEN_ROLE_PREFIX.format(role='user')}{prompt}{_QWEN_ROLE_SUFFIX}")
    parts.append(f"{_QWEN_ROLE_PREFIX.format(role='assistant')}")
    return "".join(parts)


def split_prompt_completion_qwen(text: str) -> tuple[str, str]:
    """按 Qwen3 ChatML 的 ``<|im_start|>assistant\\n`` 分割。

    用于 loss mask：prompt 部分屏蔽（``ignore_index=-100``），completion
    部分参与 loss。

    规则：找最后一个 ``<|im_start|>assistant\\n`` 的位置，其后的内容为
    completion。

    Args:
        text: 渲染后的字符串（来自 :func:`render_chat_qwen` 或
            :func:`render_prompt_qwen`）

    Returns:
        (prompt_part, completion_part)：
        - prompt_part 包含到 ``<|im_start|>assistant\\n``（含）
        - completion_part 是其后的内容（可能为空）

    Examples:
        >>> split_prompt_completion_qwen(
        ...     '<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！<|im_end|>\n'
        ... )
        ('<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n', '你好！<|im_end|>\\n')
        >>> split_prompt_completion_qwen("no marker here")
        ('no marker here', '')
    """
    marker = "<|im_start|>assistant\n"
    idx = text.rfind(marker)
    if idx < 0:
        return text, ""
    end = idx + len(marker)
    return text[:end], text[end:]


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
    # Qwen3 ChatML（Task: QwenTokenizer）
    "QWEN_IM_START",
    "QWEN_IM_END",
    "QWEN_ENDOFTEXT",
    "render_chat_qwen",
    "render_prompt_qwen",
    "split_prompt_completion_qwen",
]
