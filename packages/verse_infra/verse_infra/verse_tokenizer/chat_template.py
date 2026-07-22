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

Qwen3 ChatML（jinja2 可选）
---------------------------
Part4K2 Task 2 升级：Qwen3 ChatML 渲染优先使用 jinja2 模板引擎（可选依赖），
jinja2 不可用时降级为现有的 f-string 拼接方式。新增工具调用模板（Qwen3 官方
``<tool_call>`` 格式）与 ``tools`` 声明支持。

- ``CHATML_TEMPLATE``：基础 ChatML jinja2 模板
- ``CHATML_TEMPLATE_WITH_TOOLS``：含 ``tools`` 声明的 ChatML jinja2 模板
- ``CHATML_TEMPLATE_WITH_TOOL_CALLS``：含 ``tools`` 声明 + assistant 工具调用
  （``<tool_call>...</tool_call>`` Qwen3 官方格式）的 jinja2 模板
"""

from __future__ import annotations

import json as _json
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Part4K2 Task 2: jinja2 可选依赖
# ---------------------------------------------------------------------------
# jinja2 是可选依赖：可用时优先用 Template 渲染 ChatML；不可用时降级为
# 现有的 f-string 拼接方式（render_chat_qwen 等函数的手写实现）。
try:
    from jinja2 import Template as _Template
    _HAS_JINJA2 = True
except ImportError:  # pragma: no cover - 取决于环境是否安装 jinja2
    _Template = None  # type: ignore[assignment]
    _HAS_JINJA2 = False


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


# ---------------------------------------------------------------------------
# Part4K2 Task 2: ChatML jinja2 模板定义（Qwen 风格）
# ---------------------------------------------------------------------------
# 当 jinja2 可用时，``render_chat_qwen`` / ``render_chat_qwen_with_tools``
# 优先用这些模板渲染；jinja2 不可用时降级为下面的 f-string 实现。
# 模板字符串中的 ``\n`` 由 Python 解析为换行符，jinja2 渲染时原样输出。

# 基础 ChatML 模板：``<|im_start|>{role}\n{content}<|im_end|>\n`` 循环拼接，
# 末尾按需追加 ``<|im_start|>assistant\n``。
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# 含 tools 声明的 ChatML 模板：在 messages 前插入一段 system 段声明可用工具。
# tools 为空列表 / None 时不输出 system 段，与基础模板等价。
CHATML_TEMPLATE_WITH_TOOLS = (
    "{% if tools %}<|im_start|>system\n"
    "You have access to the following tools:\n"
    "{{ tools | tojson }}\n"
    "<|im_end|>\n"
    "{% endif %}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# Part4K2 Task 2: 工具调用模板（Qwen3 官方 ``<tool_call>`` 格式）
# ---------------------------------------------------------------------------
# Qwen3 工具调用官方格式：
#   - assistant 消息携带 ``tool_calls`` 字段时，渲染为：
#       <|im_start|>assistant
#       <tool_call>
#       {"name": "func_name", "arguments": {"arg1": "value1"}}
#       </tool_call>
#       <|im_end|>
#   - tool 角色消息（工具返回结果）按普通消息渲染：
#       <|im_start|>tool\n{content}<|im_end|>
#   - 仍兼容普通 assistant 文本回复（无 tool_calls 字段时按 content 渲染）。
CHATML_TEMPLATE_WITH_TOOL_CALLS = (
    "{% if tools %}<|im_start|>system\n"
    "You have access to the following tools:\n"
    "{{ tools | tojson }}\n"
    "<|im_end|>\n"
    "{% endif %}"
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message.get('tool_calls') %}"
    "{% for tool_call in message['tool_calls'] %}"
    "<tool_call>\n{{ tool_call | tojson }}\n</tool_call>\n"
    "{% endfor %}"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{{ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# 工具调用相关 token 字符串（Qwen3 官方格式）
TOOL_CALL_BEGIN = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

# 工具声明 system 段前缀（与 CHATML_TEMPLATE_WITH_TOOLS 中的字面值保持一致）
_TOOL_SYSTEM_PREFIX = "<|im_start|>system\nYou have access to the following tools:\n"
_TOOL_SYSTEM_SUFFIX = "\n<|im_end|>\n"


def _jinja2_render(template_str: str, **kwargs: Any) -> str:
    """用 jinja2 渲染模板字符串（内部辅助）。

    Args:
        template_str: jinja2 模板字符串
        **kwargs: 渲染上下文变量

    Returns:
        渲染后的字符串

    Raises:
        RuntimeError: jinja2 不可用时调用此函数（调用方应先检查 ``_HAS_JINJA2``）
    """
    if not _HAS_JINJA2 or _Template is None:  # pragma: no cover - 防御性检查
        raise RuntimeError(
            "jinja2 不可用，无法渲染 ChatML 模板。请安装：pip install jinja2"
        )
    return _Template(template_str).render(**kwargs)


def render_chat_qwen(
    messages: list[dict],
    add_generation_prompt: bool = False,
) -> str:
    """Qwen3 ChatML 格式渲染。

    Part4K2 Task 2 升级：jinja2 可用时优先用 :data:`CHATML_TEMPLATE` 渲染；
    jinja2 不可用时降级为 f-string 拼接（输出完全等价）。

    Args:
        messages: ``[{"role": "user", "content": "..."}, ...]``，role 支持
            ``system`` / ``user`` / ``assistant`` / ``tool`` 等。
        add_generation_prompt: ``True`` 时在末尾追加
            ``<|im_start|>assistant\\n``，用于推理前缀。

    Returns:
        ChatML 格式字符串，例如（``\\n`` 表示换行）::

            <|im_start|>system\\n{system}<|im_end|>\\n
            <|im_start|>user\\n{user}<|im_end|>\\n
            <|im_start|>assistant\\n{assistant}<|im_end|>\\n

    Examples:
        >>> render_chat_qwen([
        ...     {"role": "user", "content": "你好"},
        ...     {"role": "assistant", "content": "你好！"},
        ... ])
        '<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n你好！<|im_end|>\\n'
        >>> render_chat_qwen(
        ...     [{"role": "user", "content": "你好"}],
        ...     add_generation_prompt=True,
        ... )
        '<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n'
    """
    if _HAS_JINJA2:
        return _jinja2_render(
            CHATML_TEMPLATE,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )
    # 降级：f-string 拼接（与 jinja2 输出等价）
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
        '<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n'
        >>> render_prompt_qwen("你好", system="你是助手")
        '<|im_start|>system\\n你是助手<|im_end|>\\n<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n'
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
        ...     '<|im_start|>user\\n你好<|im_end|>\\n<|im_start|>assistant\\n你好！<|im_end|>\\n'
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


# ---------------------------------------------------------------------------
# Part4K2 Task 2: 工具调用渲染与解析（Qwen3 官方格式）
# ---------------------------------------------------------------------------


def _render_tools_system_segment_fstring(tools: list[dict]) -> str:
    """f-string 降级路径：渲染 tools 声明的 system 段。

    与 :data:`CHATML_TEMPLATE_WITH_TOOLS` 中 ``{% if tools %}...{% endif %}``
    段输出完全等价。使用 ``ensure_ascii=True`` + ``sort_keys=True``
    （与 jinja2 ``tojson`` 默认行为一致，确保两条路径输出完全等价）。
    """
    if not tools:
        return ""
    # 注意：jinja2 的 tojson 过滤器默认使用 ensure_ascii=True + sort_keys=True，
    # 这里保持一致以确保 jinja2 / f-string 两条路径输出完全等价。
    tools_json = _json.dumps(tools, ensure_ascii=True, sort_keys=True)
    return f"{_TOOL_SYSTEM_PREFIX}{tools_json}{_TOOL_SYSTEM_SUFFIX}"


def _render_message_segment_fstring(msg: dict) -> str:
    """f-string 降级路径：渲染单条消息的 ChatML 段。

    与 :data:`CHATML_TEMPLATE_WITH_TOOL_CALLS`` 中 message 循环体输出等价。
    支持 assistant 消息携带 ``tool_calls`` 字段时输出 ``<tool_call>`` 块。
    使用 ``ensure_ascii=True`` + ``sort_keys=True``
    （与 jinja2 ``tojson`` 默认行为一致）。
    """
    role = msg.get("role", "user")
    parts = [f"{_QWEN_ROLE_PREFIX.format(role=role)}"]
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            tc_json = _json.dumps(tc, ensure_ascii=True, sort_keys=True)
            parts.append(f"{TOOL_CALL_BEGIN}\n{tc_json}\n{TOOL_CALL_END}\n")
    else:
        parts.append(msg.get("content", "") or "")
    parts.append(_QWEN_ROLE_SUFFIX)
    return "".join(parts)


def render_chat_qwen_with_tools(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    add_generation_prompt: bool = False,
    render_tool_calls: bool = True,
) -> str:
    """Qwen3 ChatML + 工具调用渲染（Qwen3 官方格式）。

    Part4K2 Task 2 新增。当 ``render_tool_calls=True`` 且 assistant 消息携带
    ``tool_calls`` 字段时，渲染为 Qwen3 官方的 ``<tool_call>`` 格式::

        <|im_start|>assistant
        <tool_call>
        {"name": "func", "arguments": {...}}
        </tool_call>
        <|im_end|>

    当 ``render_tool_calls=False`` 时，使用 :data:`CHATML_TEMPLATE_WITH_TOOLS`
    （仅声明 tools，不渲染 assistant 的 tool_calls 块，所有消息按 content 渲染）。

    jinja2 可用时优先用模板渲染；不可用时降级为 f-string 拼接（输出等价）。

    Args:
        messages: ``[{"role": "...", "content": "...", "tool_calls": [...]}, ...]``
            - ``role``: ``system`` / ``user`` / ``assistant`` / ``tool``
            - ``content``: 文本内容（assistant 携带 ``tool_calls`` 时可为空）
            - ``tool_calls``: 可选，assistant 工具调用列表，每项形如
              ``{"name": "func", "arguments": {...}}``
        tools: 可用工具声明列表（OpenAI function calling 风格），
            如 ``[{"type": "function", "function": {"name": "...", ...}}]``；
            为空 / ``None`` 时不输出 tools 声明 system 段。
        add_generation_prompt: ``True`` 时末尾追加 ``<|im_start|>assistant\\n``。
        render_tool_calls: ``True`` 时使用 :data:`CHATML_TEMPLATE_WITH_TOOL_CALLS`
            渲染 assistant 的 ``tool_calls`` 字段；``False`` 时仅声明 tools，
            不渲染 tool_calls 块（用 :data:`CHATML_TEMPLATE_WITH_TOOLS`）。

    Returns:
        ChatML 渲染字符串，含 tools 声明 + messages + 工具调用块。

    Examples:
        >>> tools = [{"type": "function", "function": {
        ...     "name": "get_weather",
        ...     "parameters": {"type": "object", "properties": {}},
        ... }}]
        >>> msgs = [
        ...     {"role": "user", "content": "北京天气"},
        ...     {"role": "assistant", "content": "",
        ...      "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]},
        ...     {"role": "tool", "content": '{"temp": 25}'},
        ... ]
        >>> out = render_chat_qwen_with_tools(msgs, tools=tools)
        >>> "<tool_call>" in out and "</tool_call>" in out
        True
        >>> "get_weather" in out
        True
    """
    # 选择模板：是否渲染 assistant 的 tool_calls 块
    if _HAS_JINJA2:
        template_str = (
            CHATML_TEMPLATE_WITH_TOOL_CALLS
            if render_tool_calls
            else CHATML_TEMPLATE_WITH_TOOLS
        )
        return _jinja2_render(
            template_str,
            messages=messages,
            tools=tools or [],
            add_generation_prompt=add_generation_prompt,
        )

    # 降级：f-string 拼接
    parts: list[str] = [_render_tools_system_segment_fstring(tools or [])]
    for msg in messages:
        if render_tool_calls:
            parts.append(_render_message_segment_fstring(msg))
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            parts.append(
                f"{_QWEN_ROLE_PREFIX.format(role=role)}{content}{_QWEN_ROLE_SUFFIX}"
            )
    if add_generation_prompt:
        parts.append(f"{_QWEN_ROLE_PREFIX.format(role='assistant')}")
    return "".join(parts)


def extract_tool_calls_qwen3(text: str) -> list[dict]:
    """从生成文本中提取 Qwen3 官方格式的工具调用。

    Part4K2 Task 2 新增。解析 ``<tool_call>\\n{json}\\n</tool_call>`` 块
    （Qwen3 官方格式），与 :func:`render_chat_qwen_with_tools` 中 assistant
    消息携带 ``tool_calls`` 字段的输出格式互逆。

    可能有多个工具调用连续出现。每个 ``<tool_call>...</tool_call>`` 块解析为
    一个 dict（``{"name": "...", "arguments": {...}}``）。

    Args:
        text: 模型生成的文本

    Returns:
        工具调用字典列表。JSON 解析失败时该项为 ``{"raw": "原始字符串"}``。
        若文本中无 ``<tool_call>`` 块，返回空列表。

    Examples:
        >>> text = (
        ...     '<tool_call>\\n'
        ...     '{"name": "search", "arguments": {"q": "天气"}}\\n'
        ...     '</tool_call>'
        ... )
        >>> calls = extract_tool_calls_qwen3(text)
        >>> len(calls)
        1
        >>> calls[0]["name"]
        'search'
        >>> calls[0]["arguments"]["q"]
        '天气'
        >>> extract_tool_calls_qwen3("普通回复")
        []
    """
    tool_calls: list[dict] = []
    cursor = 0
    while True:
        begin = text.find(TOOL_CALL_BEGIN, cursor)
        if begin < 0:
            break
        begin += len(TOOL_CALL_BEGIN)
        end = text.find(TOOL_CALL_END, begin)
        if end < 0:
            # 未闭合，取到字符串末尾
            raw = text[begin:].strip()
            cursor = len(text)
        else:
            raw = text[begin:end].strip()
            cursor = end + len(TOOL_CALL_END)
        if not raw:
            continue
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                tool_calls.append(parsed)
            elif isinstance(parsed, list):
                tool_calls.extend([p for p in parsed if isinstance(p, dict)])
            else:
                tool_calls.append({"raw": raw})
        except _json.JSONDecodeError:
            tool_calls.append({"raw": raw})
    return tool_calls


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
    # Part4K2 Task 2: jinja2 引擎标志
    "_HAS_JINJA2",
    # Qwen3 ChatML（Task: QwenTokenizer）
    "QWEN_IM_START",
    "QWEN_IM_END",
    "QWEN_ENDOFTEXT",
    "render_chat_qwen",
    "render_prompt_qwen",
    "split_prompt_completion_qwen",
    # Part4K2 Task 2: ChatML jinja2 模板（Qwen 风格）+ 工具调用
    "CHATML_TEMPLATE",
    "CHATML_TEMPLATE_WITH_TOOLS",
    "CHATML_TEMPLATE_WITH_TOOL_CALLS",
    "TOOL_CALL_BEGIN",
    "TOOL_CALL_END",
    "render_chat_qwen_with_tools",
    "extract_tool_calls_qwen3",
]
