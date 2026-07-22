# ADR-010: jinja2 聊天模板

- **状态**：Accepted
- **日期**：2026-07-22
- **决策者**：Verse 框架作者（CometFuture / CUI-hyg）
- **相关规范**：[`/workspace/.trae/specs/part4k2-arch-model-upgrade/spec.md`](../../../.trae/specs/part4k2-arch-model-upgrade/spec.md)
- **前置 ADR**：[ADR-006 VerseInfra 总包聚合](adr-006-verse-infra-aggregation.md)（`verse_tokenizer` 为子模块）
- **相关 ADR**：[ADR-009 .vn 文件格式](adr-009-vn-format.md)（`chat_template.jinja` 内嵌于 .vn 容器）

## 上下文

Part4K2 之前，`verse_tokenizer.chat_template` 已提供 Qwen3 ChatML 渲染（`render_chat_qwen` / `render_prompt_qwen` / `split_prompt_completion_qwen`），但实现为手写 f-string 拼接。随着模型能力扩展到工具调用（function calling）场景，现有方案暴露以下问题：

1. **工具调用格式缺失**：Qwen3 官方工具调用采用 `<tool_call>{"name":...,"arguments":...}</tool_call>` 格式，并需要 `tools` 声明的 system 段；手写拼接需要为每种组合（有/无 tools、有/无 tool_calls）维护独立分支，易出错。
2. **模板不可移植**：f-string 拼接逻辑硬编码在 Python 函数内，无法被 `transformers` / `tokenizers` 等生态工具直接读取（HF 生态以 jinja2 模板字符串作为 `chat_template` 标准）。
3. **模板不可内嵌**：HuggingFace `tokenizer.json` 支持内嵌 `chat_template` 字段（jinja2 字符串），Verse 的 f-string 实现无法序列化为可移植模板。
4. **维护成本高**：每新增一种消息变体（如 tool 角色返回结果）都需修改 Python 函数，而非仅更新模板字符串。

同时必须保持向后兼容：
- `render_chat_qwen` / `render_prompt_qwen` 等现有函数签名不变。
- jinja2 不应成为强制依赖（符合 ADR-001 的"零重型依赖"原则），无 jinja2 时渲染输出必须与有 jinja2 时完全等价。

## 决策

**将 ChatML 渲染升级为 jinja2 模板优先 + f-string 降级；jinja2 作为可选依赖（不可用时降级为等价的 f-string 实现）；新增 Qwen3 官方工具调用格式支持与三个模板常量。**

具体含义：

1. **jinja2 可选依赖 + 优雅降级**：
   - `chat_template.py` 顶部 `try: from jinja2 import Template` 探测，`_HAS_JINJA2` 标志。
   - jinja2 可用时：`render_chat_qwen` / `render_chat_qwen_with_tools` 优先用 `Template(template_str).render(**kwargs)` 渲染。
   - jinja2 不可用时：降级为手写 f-string 拼接（`_render_tools_system_segment_fstring` / `_render_message_segment_fstring`），输出与 jinja2 路径**完全等价**（`ensure_ascii=True` + `sort_keys=True` 与 jinja2 `tojson` 默认行为一致）。
   - `pyproject.toml` 声明 `chatml = ["jinja2>=3.0"]` 可选依赖。

2. **三个模板常量**（jinja2 字符串，可序列化到 `tokenizer.json`）：
   - `CHATML_TEMPLATE`：基础 ChatML，`<|im_start|>{role}\n{content}<|im_end|>\n` 循环拼接 + `add_generation_prompt`。
   - `CHATML_TEMPLATE_WITH_TOOLS`：在 messages 前插入 `tools` 声明 system 段（`tools | tojson`），tools 为空时与基础模板等价。
   - `CHATML_TEMPLATE_WITH_TOOL_CALLS`：tools 声明 + assistant 消息携带 `tool_calls` 字段时渲染为 `<tool_call>\n{tool_call | tojson}\n</tool_call>\n` 块（Qwen3 官方格式），兼容普通 assistant 文本回复。

3. **工具调用渲染与解析（互逆）**：
   - `render_chat_qwen_with_tools(messages, tools=None, add_generation_prompt=False, render_tool_calls=True)`：渲染含工具调用的完整对话。
   - `extract_tool_calls_qwen3(text)`：从模型生成文本中提取 `<tool_call>...</tool_call>` 块，解析为 `[{"name": ..., "arguments": ...}, ...]` 列表（与渲染互逆；JSON 解析失败时该项为 `{"raw": "原始字符串"}`）。
   - 工具调用 token 常量：`TOOL_CALL_BEGIN = "<tool_call>"` / `TOOL_CALL_END = "</tool_call>"`。

4. **tokenizer.json 内嵌**：`chat_template.jinja` 模板字符串可内嵌到 `tokenizer.json` 的 `chat_template` 字段，与 HuggingFace 标准对齐；`.vn` 容器（ADR-009）也支持 `write_chat_template` 单独存储。

5. **向后兼容**：
   - `render_chat_qwen` / `render_prompt_qwen` / `split_prompt_completion_qwen` 签名不变，行为升级（jinja2 优先）。
   - 旧的 `render_chat` / `render_prompt`（`<|user|>` / `<|assistant|>` 风格）保留不变。
   - `KnowledgeDistiller` 的 `T` 参数别名模式被借鉴到 `temperature` 参数（`T` 为旧别名，优先级低于 `temperature`）。

## 后果

### 优点

- **生态对接**：jinja2 模板字符串是 HuggingFace `chat_template` 事实标准，可被 `transformers` / `tokenizers` 直接读取，也可内嵌 `tokenizer.json`。
- **工具调用支持**：Qwen3 官方 `<tool_call>` 格式完整支持（渲染 + 解析互逆），为 RL 训练（NexRL）的工具调用场景铺路。
- **零强制依赖**：jinja2 为可选依赖，无 jinja2 时降级 f-string 输出完全等价，符合 ADR-001 原则。
- **模板可移植**：三个模板常量是纯字符串，可序列化、可 diff、可被外部工具加载。
- **维护成本低**：新增消息变体只需更新模板字符串，无需改 Python 函数（jinja2 路径）；f-string 降级路径仅维护等价性。
- **向后兼容**：现有函数签名不变，旧代码零修改。

### 缺点

- **双路径维护**：jinja2 路径与 f-string 降级路径需保持输出等价，新增模板变体时两处都要更新（已用 `ensure_ascii=True` + `sort_keys=True` 对齐 `tojson` 行为）。
- **jinja2 模板可读性**：jinja2 语法对不熟悉的用户有学习成本（但模板字符串本身是行业标准）。
- **`extract_tool_calls_qwen3` 容错**：模型生成的 `<tool_call>` 块可能格式不规范（未闭合、JSON 解析失败），已用 `{"raw": ...}` 兜底。

### 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| jinja2 / f-string 两路径输出不一致 | `ensure_ascii=True` + `sort_keys=True` 严格对齐 `tojson`；测试覆盖两路径输出等价 |
| jinja2 模板注入（用户自定义模板） | jinja2 默认沙箱化（`Template` 不启用 `autoescape`，但模板来自受信配置；`extract_tool_calls_qwen3` 仅解析 JSON） |
| 模型生成非法 `<tool_call>` 块 | `extract_tool_calls_qwen3` 用 `try/except JSONDecodeError` 兜底为 `{"raw": ...}` |
| 用户未装 jinja2 导致功能缺失 | 降级路径功能完整（仅失去模板可移植性）；文档提示 `pip install jinja2` |

## 替代方案（已否决）

### 方案 A：jinja2 作为强制依赖

**描述**：强制要求安装 jinja2，移除 f-string 降级路径。

**否决理由**：违反 ADR-001 的"零重型依赖"原则；嵌入式 / CI 等场景可能无法安装 jinja2；降级路径维护成本可控。

### 方案 B：继续用 f-string，不引入 jinja2

**描述**：仅扩展 f-string 实现工具调用，不引入模板常量。

**否决理由**：模板不可移植（无法内嵌 `tokenizer.json`）；无法被 HF 生态工具读取；每新增变体都要改 Python 函数。

### 方案 C：使用 `transformers` 的 `apply_chat_template`

**描述**：直接依赖 `transformers` 库的聊天模板实现。

**否决理由**：`transformers` 是重型依赖（与 ADR-001 冲突）；VerseTokenizer 需在无 `transformers` 时工作（Qwen tokenizer 仅 lazy import `transformers` 构造时触发）。

## 备注

- 本 ADR 是 Part4K2 "聊天模板工程化"的核心决策。
- jinja2 降级路径的等价性由 `tests/test_chat_template.py` 覆盖（jinja2 / f-string 两路径输出 diff 为空）。
- 工具调用格式参考 Qwen3 官方文档的 `<tool_call>` 规范。
- 相关文档：[Verse 训练指南 - jinja2 聊天模板使用指南](../training_guide.md)
