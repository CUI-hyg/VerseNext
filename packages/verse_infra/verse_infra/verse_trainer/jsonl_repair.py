"""JSONL 自修复与字段标准化（Part5K1 Task 5）。

处理两类常见数据质量问题：
1. **字段名不统一**：训练数据标准格式为 ``{"prompt": ..., "completion": ...}``
   或 ``{"text": ...}``，但实际数据可能用 ``instruction``/``response``、
   ``q``/``a``、``input``/``output`` 等异名。本模块通过 :data:`FIELD_ALIASES`
   映射表统一为标准字段名。
2. **格式损坏**：缺逗号、未闭合括号、BOM 头、行尾多余逗号、单引号、
   无引号键名等。:func:`_repair_line` 保守地逐项尝试修复，修复后必须通过
   ``json.loads`` 校验才算成功，避免"修复"出错误数据。

设计原则
========
- **仅依赖 Python 标准库**（``json`` / ``re`` / ``os`` / ``typing``），
  不引入 verse_trainer 其他模块，避免循环依赖。
- **修复保守**：宁可抛 :class:`JSONLRepairError` 也不产出错误数据。
  每个修复步骤后都用 ``json.loads`` 校验。
- **向后兼容**：:func:`repair_jsonl` 的 ``repair=False`` 走严格解析，
  不做任何修复。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class JSONLRepairError(Exception):
    """JSONL 修复 / 标准化过程中出现的错误。

    触发场景：
    - :func:`_standardize_fields` 遇到无法识别的字段结构
    - :func:`repair_jsonl` 中存在无法修复的行（``repair=True``）
    - :func:`repair_jsonl` 中 ``repair=False`` 模式下任意行解析失败
    """


# ---------------------------------------------------------------------------
# 字段异名映射表（SubTask 5.1）
# ---------------------------------------------------------------------------

#: 标准字段名 → 可能的异名列表。
#:
#: - ``prompt`` ← instruction / question / q / input / user / query / ask
#: - ``completion`` ← response / answer / a / output / assistant / reply / response_text
#: - ``text`` ← content / raw / body / passage（单字段场景）
FIELD_ALIASES: dict = {
    "prompt": ("instruction", "question", "q", "input", "user", "query", "ask"),
    "completion": (
        "response",
        "answer",
        "a",
        "output",
        "assistant",
        "reply",
        "response_text",
    ),
    "text": ("content", "raw", "body", "passage"),
}


# ---------------------------------------------------------------------------
# 字段标准化（SubTask 5.2）
# ---------------------------------------------------------------------------


def _standardize_fields(item: dict) -> dict:
    """把异名字段统一为标准 ``prompt``/``completion`` 或 ``text`` 结构。

    识别优先级（按顺序）：
        1. 已有 ``prompt`` + ``completion`` → 直接返回（标准 prompt-completion）
        2. 已有 ``text`` → 返回 ``{"text": ...}``（标准 text 格式）
        3. 探测到 prompt 别名 + completion 别名 → 转为 ``{"prompt":..., "completion":...}``
        4. 只有一个异名字段（任意类别）→ 转为 ``{"text": ...}``（单字段场景）
        5. 无法识别任何字段 → 抛 :class:`JSONLRepairError`

    未被消费的元数据字段（如 ``meta`` / ``id`` 等）原样拷贝到结果。

    Args:
        item: 单行 JSONL 解析出的 dict。
    Returns:
        标准化后的 dict。
    Raises:
        JSONLRepairError: 字段结构无法识别时。
    """
    # 1. 已是标准 prompt-completion
    if "prompt" in item and "completion" in item:
        result = {"prompt": item["prompt"], "completion": item["completion"]}
        for k, v in item.items():
            if k not in ("prompt", "completion"):
                result[k] = v
        return result

    # 2. 已是标准 text
    if "text" in item:
        result = {"text": item["text"]}
        for k, v in item.items():
            if k != "text":
                result[k] = v
        return result

    # 2b. 单边标准字段：只有 prompt 或只有 completion → 退化为 text 纯文本训练
    #     （单样本场景常见：只有 prompt 表示纯文本生成，只有 completion 表示续写）
    if "prompt" in item and "completion" not in item:
        result = {"text": item["prompt"]}
        for k, v in item.items():
            if k != "prompt":
                result[k] = v
        return result
    if "completion" in item and "prompt" not in item:
        result = {"text": item["completion"]}
        for k, v in item.items():
            if k != "completion":
                result[k] = v
        return result

    # 3. 探测异名字段
    prompt_key = _first_present(item, FIELD_ALIASES["prompt"])
    completion_key = _first_present(item, FIELD_ALIASES["completion"])

    # 3a. prompt + completion 别名齐全 → prompt-completion
    if prompt_key is not None and completion_key is not None:
        result = {
            "prompt": item[prompt_key],
            "completion": item[completion_key],
        }
        consumed = {prompt_key, completion_key}
        for k, v in item.items():
            if k not in consumed:
                result[k] = v
        return result

    # 4. 单字段场景：找到唯一的异名字段（任意类别）→ text
    single_key = _first_present(
        item,
        (*FIELD_ALIASES["prompt"], *FIELD_ALIASES["completion"], *FIELD_ALIASES["text"]),
    )
    if single_key is not None:
        result = {"text": item[single_key]}
        for k, v in item.items():
            if k != single_key:
                result[k] = v
        return result

    # 5. 无法识别
    raise JSONLRepairError(f"无法识别的字段结构: {list(item.keys())}")


def _first_present(item: dict, candidates: Tuple[str, ...]) -> Optional[str]:
    """返回 candidates 中第一个存在于 item 的键名，找不到返回 None。"""
    for key in candidates:
        if key in item:
            return key
    return None


# ---------------------------------------------------------------------------
# 单行修复（SubTask 5.3）
# ---------------------------------------------------------------------------

# 控制字符：0x00-0x1F 中除 \t(0x09) 外全部移除
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")

# 行尾多余逗号：,{" → }  ,] → ]
_TRAILING_COMMA_OBJ_RE = re.compile(r",\s*}")
_TRAILING_COMMA_ARR_RE = re.compile(r",\s*]")

# 缺失逗号：相邻的两个引号之间只有空白 → 插入逗号（如 "q" "completion" → "q", "completion"）
_MISSING_COMMA_RE = re.compile(r'"\s+"')

# 无引号键名：{key: 或 ,key: → {"key": 或 ,"key":
_UNQUOTED_KEY_RE = re.compile(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')


def _strip_bom_and_ws(line: str) -> str:
    """去除 BOM（\\ufeff）和首尾空白。"""
    return line.replace("\ufeff", "").strip()


def _strip_control_chars(line: str) -> str:
    """去除控制字符（0x00-0x1F 中除 \\t 外）。"""
    return _CONTROL_CHAR_RE.sub("", line)


def _strip_trailing_commas(line: str) -> str:
    """去除行尾多余逗号：``,}`` → ``}``，``,]`` → ``]``。"""
    line = _TRAILING_COMMA_OBJ_RE.sub("}", line)
    line = _TRAILING_COMMA_ARR_RE.sub("]", line)
    return line


def _fix_missing_commas(line: str) -> str:
    """补全相邻键值对之间缺失的逗号。

    匹配 ``"`` + 空白 + ``"``（如 ``"q" "completion"``）→ 替换为 ``", "``。
    仅在 json.loads 失败后调用，合法 JSON 不会进入此函数。
    """
    return _MISSING_COMMA_RE.sub('", "', line)


def _fix_unclosed(line: str) -> str:
    """补全未闭合的 ``{`` 或 ``[``：在末尾补对应数量的 ``}`` / ``]``。

    简单按计数差补全：先补 ``]`` 再补 ``}``（适配 ``[{...`` 嵌套）。
    字符串内的括号字符可能干扰计数，但后续 json.loads 会校验合法性。
    """
    open_braces = line.count("{") - line.count("}")
    open_brackets = line.count("[") - line.count("]")
    result = line
    if open_brackets > 0:
        result = result + "]" * open_brackets
    if open_braces > 0:
        result = result + "}" * open_braces
    return result


def _fix_single_quotes(line: str) -> str:
    """把单引号替换为双引号。

    仅在 json.loads 失败时调用。若值中含撇号（如 ``it's``），
    替换后 json.loads 仍会失败 → 返回 None，不会产出错误数据。
    """
    return line.replace("'", '"')


def _fix_unquoted_keys(line: str) -> str:
    """为无引号的键名补双引号：``{prompt:`` → ``{"prompt":``。"""
    return _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', line)


# 修复函数序列：按从保守到激进排序
_REPAIR_FIXES = (
    _fix_missing_commas,
    _fix_unclosed,
    _fix_single_quotes,
    _fix_unquoted_keys,
)


def _repair_line(line: str) -> Optional[dict]:
    """保守修复单行 JSONL，返回解析出的 dict 或 None。

    修复流程（按顺序）：
        1. 去除 BOM（``\\ufeff``）和首尾空白
        2. 去除控制字符（除 ``\\t`` 外的 0x00-0x1F）
        3. 去除行尾多余逗号（``,}`` → ``}``，``,]`` → ``]``）
        4. 尝试 ``json.loads``，成功则返回
        5. 失败则逐项尝试修复（缺逗号 / 未闭合 / 单引号 / 无引号键名），
           每应用一个修复后立即 ``json.loads`` 校验，成功即返回
        6. 全部修复应用后再做一次 ``json.loads`` 校验（组合修复可能生效）
        7. 仍然失败返回 None（让上层决定抛错或跳过）

    所有修复都保守：不会破坏合法 JSON（合法 JSON 在步骤 4 即返回）。

    Args:
        line: JSONL 文件中的一行原始文本。
    Returns:
        解析成功的 dict（或 list 等其他 JSON 类型），失败返回 None。
    """
    # 1. BOM + 空白
    s = _strip_bom_and_ws(line)
    if not s:
        return None
    # 2. 控制字符
    s = _strip_control_chars(s)
    # 3. 行尾多余逗号
    s = _strip_trailing_commas(s)

    # 4. 首次尝试解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 5. 逐项修复，每个修复后立即校验
    current = s
    for fix in _REPAIR_FIXES:
        candidate = fix(current)
        if candidate == current:
            # 该修复未产生变化，跳过
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 单项修复不够，保留修复结果叠加后续修复
            current = candidate

    # 6. 全部修复组合后再校验一次
    try:
        return json.loads(current)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# repair_jsonl（SubTask 5.4）
# ---------------------------------------------------------------------------


def repair_jsonl(
    path: str,
    write_back: bool = False,
    repair: bool = True,
) -> List[dict]:
    """读取 JSONL 文件，修复损坏行 + 标准化字段名，返回结果列表。

    Args:
        path: JSONL 文件路径。
        write_back: 若为 True，把修复+标准化后的结果写入 ``path + ".repaired.jsonl"``
            （不覆盖原文件，安全起见）。
        repair: 若为 True（默认），对无法直接 ``json.loads`` 的行调用
            :func:`_repair_line` 保守修复；若为 False，遇到任何解析失败
            立即抛 :class:`JSONLRepairError`（严格模式）。
    Returns:
        标准化后的 dict 列表（每行一个）。
    Raises:
        JSONLRepairError: ``repair=True`` 时存在无法修复的行；或 ``repair=False``
            时任意行解析失败。
        FileNotFoundError: 文件不存在。
    """
    # 1. 读取文件（utf-8，errors='replace' 兜底）
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    results: List[dict] = []
    failed_lines: List[int] = []
    total = 0  # 非空行数
    success = 0  # 成功解析行数
    repaired = 0  # 修复成功行数

    # 2. 逐行处理
    for line_no, raw in enumerate(raw_lines, start=1):
        line = raw.strip()
        if not line:
            # 空行跳过
            continue
        total += 1

        # 先尝试严格解析：成功则无需修复
        obj: Any = None
        need_repair = False
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            need_repair = True

        if need_repair:
            if not repair:
                # 严格模式：直接抛错
                raise JSONLRepairError(
                    f"JSONL 解析失败：{path} 第 {line_no} 行 - {line[:80]!r}"
                )
            # repair=True：调用 _repair_line 保守修复
            obj = _repair_line(line)
            if obj is None:
                failed_lines.append(line_no)
                continue
            repaired += 1

        success += 1

        # 3. 对 dict 调用 _standardize_fields；非 dict（如 list）原样保留
        if isinstance(obj, dict):
            obj = _standardize_fields(obj)
        results.append(obj)

    # 5. 存在无法修复的行 → 抛错
    if failed_lines:
        preview = failed_lines[:5]
        raise JSONLRepairError(
            f"{len(failed_lines)} 行无法修复: 行号 {preview}..."
        )

    # 6. write_back：写到 path + ".repaired.jsonl"（不覆盖原文件）
    if write_back:
        out_path = path + ".repaired.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 7. 打印日志
    print(
        f"[jsonl_repair] 读取 {total} 行，成功 {success} 行，修复 {repaired} 行",
        flush=True,
    )

    # 8. 返回结果
    return results


__all__ = [
    "JSONLRepairError",
    "FIELD_ALIASES",
    "repair_jsonl",
]
