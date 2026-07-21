#!/usr/bin/env python3
"""数据处理脚本：把 raw/ 下所有 jsonl 处理为统一 schema 合并到 train.jsonl。

统一 schema（每行一个 JSON 对象）
---------------------------------
预训练格式（``--format text``，默认）::

    {"text": "...", "source": "belle_chat", "lang": "zh", "task_type": "sft"}

SFT 对话格式（``--format messages``）::

    {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
     "source": "belle_chat", "lang": "zh", "task_type": "sft"}

混合格式（``--format both``）：同时输出 text 与 messages 两份::

    {"text": "<|user|>...<|assistant|>...<|endoftext|>",
     "messages": [...],
     "source": "belle_chat", "lang": "zh", "task_type": "sft"}

处理逻辑
--------
- **通用文本类**（``wiki_zh`` / ``*_lm``）：直接提取 ``text`` 字段
- **SFT 对话类**（``belle_chat`` / ``firefly_zh`` / ``code_alpaca`` / ``math_qa_zh`` / ``alpaca_en``）：
  把 ``instruction`` / ``input`` / ``output`` 拼成单条 text：
  ``<|user|>{instruction}\\n{input}<|assistant|>{output}<|endoftext|>``
  或保留 messages 格式供 SFTTrainer 使用
- **问答类**（``cmrc2018``）：把 ``question`` / ``context`` / ``answers`` 改写为 SFT 单轮

过滤规则
--------
- text 长度 < 50 字符的丢弃
- text 长度 > 8192 字符的截断到 8192
- 重复文本去重（用 set 跟踪 hash）
- 包含 ``<|imb|>`` 等 chat template 特殊标记的丢弃

统计输出
--------
- 每个源贡献的样本数
- 总样本数
- 平均长度、最大长度、最小长度
- 字符分布（中英占比）

使用方式
--------
    cd /workspace/data/demo/scripts
    # 处理默认 ./raw 目录
    python process_datasets.py --output ../data/train.jsonl

    # 自定义 raw 目录 + 限制每源最多 1000 条
    python process_datasets.py --raw-dir ./raw --output ./train.jsonl --max-per-source 1000

    # 输出 messages 格式（供 SFTTrainer 用）
    python process_datasets.py --format messages --output ./train.jsonl

    # 输出混合格式（text + messages 都有）
    python process_datasets.py --format both --output ./train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Optional, Iterable


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# chat template 特殊 token 列表（出现这些就丢弃样本，避免破坏 chat template 渲染）
SPECIAL_TOKENS_TO_REJECT: tuple[str, ...] = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|eos|>",
    "<|bos|>",
    "<|pad|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|imb|>",            # 任务示例里的占位 token
)

# 默认过滤阈值
DEFAULT_MIN_LEN = 50
DEFAULT_MAX_LEN = 8192

# 源名称 → (lang, task_type) 默认映射（可被 raw/<src>/meta.json 覆盖）
DEFAULT_SOURCE_META: dict[str, tuple[str, str]] = {
    "wiki_zh":     ("zh", "lm"),
    "belle_chat":  ("zh", "sft"),
    "firefly_zh":  ("zh", "sft"),
    "code_alpaca": ("code", "code"),
    "math_qa_zh":  ("zh", "sft"),
    "cmrc2018":    ("zh", "qa"),
    "alpaca_en":   ("en", "sft"),
}


# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------


def _log(msg: str, level: str = "INFO") -> None:
    """统一日志格式。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 源元信息读取：raw/<src>/meta.json
# ---------------------------------------------------------------------------


def _load_source_meta(src_dir: str, src_name: str) -> tuple[str, str]:
    """读取 raw/<src>/meta.json，未找到则用 DEFAULT_SOURCE_META 兜底。

    Returns:
        (lang, task_type)
    """
    meta_path = os.path.join(src_dir, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            lang = str(meta.get("lang", "")).strip()
            task_type = str(meta.get("task_type", "")).strip()
            if lang and task_type:
                return lang, task_type
        except Exception as e:  # noqa: BLE001
            _log(f"[{src_name}] meta.json 解析失败：{e}，用默认值兜底", "WARN")

    # 用源名匹配默认
    return DEFAULT_SOURCE_META.get(src_name, ("unknown", "unknown"))


# ---------------------------------------------------------------------------
# 输入样本解析：把不同来源的原始样本解析为 (user_text, assistant_text, raw_text) 三元组
# ---------------------------------------------------------------------------


def _extract_text_field(item: dict, keys: tuple[str, ...]) -> str:
    """从 dict 中按优先级尝试多个键，返回首个命中且为字符串的值。"""
    for k in keys:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if v is not None and not isinstance(v, (list, dict)):
            return str(v)
    return ""


def _parse_sft_sample(item: dict) -> tuple[str, str]:
    """解析 SFT 样本（instruction/input/output 或 prompt/completion）。

    Returns:
        (user_text, assistant_text)
    """
    # Belle / Firefly / Alpaca / CodeAlpaca 风格：instruction + input + output
    instruction = _extract_text_field(item, ("instruction", "prompt", "question", "query", "input_text"))
    extra_input = _extract_text_field(item, ("input", "context", "background"))
    output = _extract_text_field(item, ("output", "completion", "response", "answer", "target"))

    # 把 instruction + extra_input 合并为 user_text
    if extra_input and extra_input != instruction:
        user_text = f"{instruction}\n{extra_input}"
    else:
        user_text = instruction

    return user_text.strip(), output.strip()


def _parse_chat_sample(item: dict) -> tuple[str, str]:
    """解析 chat 风格样本（messages 数组 / conversations 数组）。

    Returns:
        (user_text, assistant_text)
        - 多轮对话：把所有 user 拼接 / 所有 assistant 拼接（取最后一轮）
        - 单轮：直接取
    """
    messages = item.get("messages") or item.get("conversations") or []
    if not isinstance(messages, list):
        return "", ""

    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # 兼容 role / from 两种字段
        role = str(msg.get("role") or msg.get("from") or "").lower()
        content = msg.get("content") or msg.get("value") or ""
        if not isinstance(content, str):
            content = str(content)
        if "user" in role or "human" in role:
            user_parts.append(content.strip())
        elif "assistant" in role or "gpt" in role or "model" in role:
            assistant_parts.append(content.strip())

    user_text = "\n".join(p for p in user_parts if p)
    assistant_text = "\n".join(p for p in assistant_parts if p)
    return user_text, assistant_text


def _parse_qa_sample(item: dict) -> tuple[str, str]:
    """解析 QA 样本（cmrc2018 风格：question / context / answers）。"""
    question = _extract_text_field(item, ("question", "query"))
    context = _extract_text_field(item, ("context", "passage", "background"))

    # answers 可能是 list[str] 或 {"text": [...]}
    answers = item.get("answers") or item.get("answer")
    answer_text = ""
    if isinstance(answers, list) and answers:
        answer_text = answers[0] if isinstance(answers[0], str) else str(answers[0])
    elif isinstance(answers, dict):
        texts = answers.get("text") or []
        if texts:
            answer_text = texts[0] if isinstance(texts[0], str) else str(texts[0])
    elif isinstance(answers, str):
        answer_text = answers

    # 把 context 拼到 user_text 中（让模型基于上下文回答）
    if context:
        user_text = f"阅读以下材料并回答问题：\n{context}\n\n问题：{question}"
    else:
        user_text = question

    return user_text.strip(), answer_text.strip()


def _parse_lm_sample(item: dict) -> str:
    """解析纯文本 LM 样本：text / content / passage / body。"""
    text = _extract_text_field(item, ("text", "content", "passage", "body", "sentence"))
    if not text:
        # wikiann NER 格式：tokens + tags
        tokens = item.get("tokens") or item.get("words")
        if isinstance(tokens, list):
            text = " ".join(str(t) for t in tokens if t is not None)
    return text.strip()


# ---------------------------------------------------------------------------
# 渲染：把 (user, assistant) 渲染为 text 与 messages
# ---------------------------------------------------------------------------


def _render_text(user: str, assistant: str) -> str:
    """渲染为预训练 text：``<|user|>{user}<|assistant|>{assistant}<|endoftext|>``。

    assistant 为空时只渲染 user 部分（纯 LM 用）。
    """
    parts = [f"<|user|>{user}"]
    if assistant:
        parts.append(f"<|assistant|>{assistant}")
        parts.append("<|endoftext|>")
    return "".join(parts)


def _render_messages(user: str, assistant: str) -> list[dict]:
    """渲染为 SFT messages 数组。

    assistant 为空时只输出 user 单条（视为 LM 上下文）。
    """
    msgs = [{"role": "user", "content": user}]
    if assistant:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


# ---------------------------------------------------------------------------
# 过滤与去重
# ---------------------------------------------------------------------------


def _contains_special_tokens(text: str) -> bool:
    """检测文本是否包含 chat template 特殊 token（避免破坏 chat template 渲染）。"""
    for tok in SPECIAL_TOKENS_TO_REJECT:
        if tok in text:
            return True
    return False


def _hash_text(text: str) -> str:
    """对 text 计算 SHA1 哈希（用于去重）。"""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _truncate(text: str, max_len: int) -> str:
    """截断到 max_len 字符。"""
    if len(text) > max_len:
        return text[:max_len]
    return text


# ---------------------------------------------------------------------------
# 字符分布统计
# ---------------------------------------------------------------------------


def _char_stats(text: str) -> tuple[int, int, int, int]:
    """统计文本中中文 / 英文字母 / 数字 / 其他字符数。

    Returns:
        (zh_count, en_count, digit_count, other_count)
    """
    zh = en = dig = other = 0
    for ch in text:
        cp = ord(ch)
        # CJK 统一汉字 + 扩展 A + 扩展 B（常见范围）
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            zh += 1
        elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
            en += 1
        elif 0x0030 <= cp <= 0x0039:
            dig += 1
        else:
            other += 1
    return zh, en, dig, other


# ---------------------------------------------------------------------------
# 处理单个源
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str) -> Iterable[dict]:
    """迭代 jsonl 文件，每行一个 dict。容错跳过解析失败的行。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                elif isinstance(obj, list):
                    # chat 数组格式：包装为 {"messages": [...]}
                    yield {"messages": obj}
            except json.JSONDecodeError as e:
                _log(f"  跳过无效 JSON 第 {line_no} 行：{e}", "WARN")
                continue


def _find_jsonl_files(src_dir: str) -> list[str]:
    """列出 src_dir 下所有 .jsonl 文件（按文件名排序，data.jsonl 优先）。"""
    if not os.path.isdir(src_dir):
        return []
    files = []
    for f in os.listdir(src_dir):
        full = os.path.join(src_dir, f)
        if os.path.isfile(full) and f.lower().endswith(".jsonl"):
            files.append(full)
    # data.jsonl 排在第一位
    files.sort(key=lambda p: (0 if os.path.basename(p) == "data.jsonl" else 1, p))
    return files


def process_source(
    src_name: str,
    src_dir: str,
    output_format: str,
    min_len: int,
    max_len: int,
    max_per_source: int,
    seen_hashes: set[str],
) -> tuple[list[dict], dict]:
    """处理单个源目录下的所有 jsonl，返回 (records, stats)。

    Args:
        src_name: 源名称
        src_dir: ``raw/<src_name>/`` 目录
        output_format: "text" / "messages" / "both"
        min_len: 文本最小长度（小于则丢弃）
        max_len: 文本最大长度（大于则截断）
        max_per_source: 该源最多保留的样本数（0 = 不限）
        seen_hashes: 全局已见 hash 集合（跨源去重）

    Returns:
        (records, stats)
        - records: 处理后的样本列表
        - stats: 该源的统计信息 dict
    """
    lang, task_type = _load_source_meta(src_dir, src_name)
    jsonl_files = _find_jsonl_files(src_dir)

    if not jsonl_files:
        _log(f"[{src_name}] 未找到 jsonl 文件（{src_dir}）", "WARN")
        return [], {"source": src_name, "lang": lang, "task_type": task_type, "total": 0}

    _log(f"[{src_name}] lang={lang} task_type={task_type} 文件数={len(jsonl_files)}")

    records: list[dict] = []
    n_in = 0
    n_filtered_short = 0
    n_filtered_special = 0
    n_dedup = 0
    n_kept = 0
    total_chars = 0
    min_chars = None
    max_chars = 0
    total_zh = total_en = total_dig = total_other = 0

    for jf in jsonl_files:
        for item in _iter_jsonl(jf):
            n_in += 1
            if max_per_source > 0 and n_kept >= max_per_source:
                break

            # 根据任务类型解析样本
            user_text = ""
            assistant_text = ""
            raw_text = ""

            if task_type == "lm" or task_type == "ner":
                raw_text = _parse_lm_sample(item)
                user_text = raw_text
                assistant_text = ""
            elif task_type == "qa":
                user_text, assistant_text = _parse_qa_sample(item)
            elif task_type in ("sft", "code"):
                # 优先 chat / messages 格式
                if "messages" in item or "conversations" in item:
                    user_text, assistant_text = _parse_chat_sample(item)
                else:
                    user_text, assistant_text = _parse_sft_sample(item)
            else:
                # 兜底：尝试 chat → sft → lm 顺序
                if "messages" in item or "conversations" in item:
                    user_text, assistant_text = _parse_chat_sample(item)
                elif "instruction" in item or "prompt" in item:
                    user_text, assistant_text = _parse_sft_sample(item)
                elif "text" in item:
                    raw_text = _parse_lm_sample(item)
                    user_text = raw_text

            # 渲染为 text
            text = _render_text(user_text, assistant_text)

            # 过滤：原始输入包含 chat template 特殊 token（避免破坏 chat template 渲染）
            # 注意：只检查 user_text / assistant_text / raw_text（原始输入），
            #       不检查渲染后的 text（其本身由 <|user|> 等构成，否则所有样本都会被过滤）
            if (
                _contains_special_tokens(user_text)
                or _contains_special_tokens(assistant_text)
                or (raw_text and _contains_special_tokens(raw_text))
            ):
                n_filtered_special += 1
                continue

            # 过滤：长度过短
            if len(text) < min_len:
                n_filtered_short += 1
                continue

            # 截断：长度过长
            text = _truncate(text, max_len)

            # 去重：基于 text hash
            h = _hash_text(text)
            if h in seen_hashes:
                n_dedup += 1
                continue
            seen_hashes.add(h)

            # 构造输出记录
            record: dict = {
                "source": src_name,
                "lang": lang,
                "task_type": task_type,
            }
            if output_format in ("text", "both"):
                record["text"] = text
            if output_format in ("messages", "both"):
                record["messages"] = _render_messages(user_text, assistant_text)
            records.append(record)
            n_kept += 1

            # 统计
            clen = len(text)
            total_chars += clen
            if min_chars is None or clen < min_chars:
                min_chars = clen
            if clen > max_chars:
                max_chars = clen
            zh, en, dig, other = _char_stats(text)
            total_zh += zh
            total_en += en
            total_dig += dig
            total_other += other

        if max_per_source > 0 and n_kept >= max_per_source:
            _log(f"[{src_name}] 达到 max_per_source={max_per_source}，停止读取后续文件", "INFO")
            break

    stats = {
        "source": src_name,
        "lang": lang,
        "task_type": task_type,
        "input": n_in,
        "kept": n_kept,
        "filtered_short": n_filtered_short,
        "filtered_special": n_filtered_special,
        "deduplicated": n_dedup,
        "total_chars": total_chars,
        "avg_chars": round(total_chars / n_kept, 2) if n_kept else 0,
        "min_chars": min_chars if min_chars is not None else 0,
        "max_chars": max_chars,
        "zh_chars": total_zh,
        "en_chars": total_en,
        "digit_chars": total_dig,
        "other_chars": total_other,
    }
    _log(
        f"[{src_name}] 输入={n_in} 保留={n_kept} "
        f"过滤短={n_filtered_short} 过滤特殊={n_filtered_special} 去重={n_dedup} "
        f"平均长度={stats['avg_chars']}"
    )
    return records, stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def process_all(
    raw_dir: str,
    output_path: str,
    output_format: str = "text",
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    max_per_source: int = 0,
    only_sources: Optional[list[str]] = None,
) -> dict:
    """处理 raw_dir 下所有源目录，合并到 output_path。

    Args:
        raw_dir: raw 根目录，包含若干 ``raw/<src>/`` 子目录
        output_path: 输出 jsonl 路径
        output_format: "text" / "messages" / "both"
        min_len: 文本最小长度（小于则丢弃）
        max_len: 文本最大长度（大于则截断）
        max_per_source: 每个源最多保留的样本数（0 = 不限）
        only_sources: 仅处理指定的源（None = 全部）

    Returns:
        全局统计 dict
    """
    if not os.path.isdir(raw_dir):
        _log(f"raw 目录不存在：{raw_dir}", "ERROR")
        return {"error": f"raw_dir not found: {raw_dir}"}

    # 枚举源目录
    source_dirs: list[tuple[str, str]] = []
    for name in sorted(os.listdir(raw_dir)):
        sd = os.path.join(raw_dir, name)
        if not os.path.isdir(sd):
            continue
        if only_sources and name not in only_sources:
            continue
        source_dirs.append((name, sd))

    if not source_dirs:
        _log(f"raw 目录下未找到任何源子目录：{raw_dir}", "WARN")
        # 仍然创建空输出文件，便于后续流程不中断
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        return {"sources": [], "total": 0}

    _log(f"待处理源数：{len(source_dirs)} -> {[n for n, _ in source_dirs]}")
    _log(f"输出格式：{output_format} | min_len={min_len} | max_len={max_len} | max_per_source={max_per_source}")

    seen_hashes: set[str] = set()
    all_records: list[dict] = []
    all_stats: list[dict] = []

    for name, sd in source_dirs:
        try:
            records, stats = process_source(
                src_name=name,
                src_dir=sd,
                output_format=output_format,
                min_len=min_len,
                max_len=max_len,
                max_per_source=max_per_source,
                seen_hashes=seen_hashes,
            )
            all_records.extend(records)
            all_stats.append(stats)
        except Exception as e:  # noqa: BLE001
            _log(f"[{name}] 处理异常：{e}", "ERROR")

    # 写出
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _log(f"输出写入：{output_path}（{len(all_records)} 条）")

    # 全局统计
    global_stats = _compute_global_stats(all_stats, all_records)
    _print_stats(global_stats)
    return global_stats


def _compute_global_stats(all_stats: list[dict], all_records: list[dict]) -> dict:
    """汇总所有源的统计信息。"""
    total_kept = sum(s.get("kept", 0) for s in all_stats)
    total_input = sum(s.get("input", 0) for s in all_stats)
    total_chars = sum(s.get("total_chars", 0) for s in all_stats)
    total_zh = sum(s.get("zh_chars", 0) for s in all_stats)
    total_en = sum(s.get("en_chars", 0) for s in all_stats)
    total_dig = sum(s.get("digit_chars", 0) for s in all_stats)
    total_other = sum(s.get("other_chars", 0) for s in all_stats)
    total_filtered_short = sum(s.get("filtered_short", 0) for s in all_stats)
    total_filtered_special = sum(s.get("filtered_special", 0) for s in all_stats)
    total_dedup = sum(s.get("deduplicated", 0) for s in all_stats)

    min_chars = min(
        (s["min_chars"] for s in all_stats if s.get("kept", 0) > 0),
        default=0,
    )
    max_chars = max(
        (s["max_chars"] for s in all_stats if s.get("kept", 0) > 0),
        default=0,
    )

    return {
        "sources": all_stats,
        "total_input": total_input,
        "total_kept": total_kept,
        "total_chars": total_chars,
        "avg_chars": round(total_chars / total_kept, 2) if total_kept else 0,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "zh_chars": total_zh,
        "en_chars": total_en,
        "digit_chars": total_dig,
        "other_chars": total_other,
        "zh_ratio": round(total_zh / total_chars, 4) if total_chars else 0,
        "en_ratio": round(total_en / total_chars, 4) if total_chars else 0,
        "digit_ratio": round(total_dig / total_chars, 4) if total_chars else 0,
        "filtered_short": total_filtered_short,
        "filtered_special": total_filtered_special,
        "deduplicated": total_dedup,
    }


def _print_stats(stats: dict) -> None:
    """打印统计信息到 stdout。"""
    print("\n" + "=" * 70)
    print("数据处理统计")
    print("=" * 70)

    print(f"\n{'源名称':<15} {'语言':<6} {'任务':<6} {'输入':>10} {'保留':>10} {'短过滤':>8} {'特殊过滤':>8} {'去重':>8} {'平均长度':>10}")
    print("-" * 95)
    for s in stats.get("sources", []):
        print(
            f"{s['source']:<15} {s['lang']:<6} {s['task_type']:<6} "
            f"{s['input']:>10} {s['kept']:>10} {s['filtered_short']:>8} "
            f"{s['filtered_special']:>8} {s['deduplicated']:>8} {s['avg_chars']:>10}"
        )

    print("\n" + "-" * 70)
    print(f"总输入样本数：    {stats['total_input']}")
    print(f"总保留样本数：    {stats['total_kept']}")
    print(f"过滤过短：        {stats['filtered_short']}")
    print(f"过滤特殊 token：  {stats['filtered_special']}")
    print(f"去重丢弃：        {stats['deduplicated']}")
    print(f"总字符数：        {stats['total_chars']}")
    print(f"平均长度：        {stats['avg_chars']}")
    print(f"最小长度：        {stats['min_chars']}")
    print(f"最大长度：        {stats['max_chars']}")
    print()
    print(f"中文字符：        {stats['zh_chars']}（占比 {stats['zh_ratio']*100:.2f}%）")
    print(f"英文字符：        {stats['en_chars']}（占比 {stats['en_ratio']*100:.2f}%）")
    print(f"数字字符：        {stats['digit_chars']}（占比 {stats['digit_ratio']*100:.2f}%）")
    print(f"其他字符：        {stats['other_chars']}（占比 {(1 - stats['zh_ratio'] - stats['en_ratio'] - stats['digit_ratio'])*100:.2f}%）")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CometSpark-V0.2 数据处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"),
        help="raw 根目录（默认：./raw）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "train.jsonl",
        ),
        help="输出 train.jsonl 路径（默认：../data/train.jsonl）",
    )
    parser.add_argument(
        "--format",
        choices=["text", "messages", "both"],
        default="text",
        help="输出格式：text=预训练 text 字段；messages=SFT messages 数组；both=两者都输出",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=DEFAULT_MIN_LEN,
        help=f"最小文本长度（小于则丢弃），默认 {DEFAULT_MIN_LEN}",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"最大文本长度（大于则截断），默认 {DEFAULT_MAX_LEN}",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="每个源最多保留的样本数（0 = 不限）",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="仅处理指定的源（空格分隔，如 --only belle_chat firefly_zh）",
    )
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    output_path = os.path.abspath(args.output)

    _log(f"raw 目录：{raw_dir}")
    _log(f"输出路径：{output_path}")

    stats = process_all(
        raw_dir=raw_dir,
        output_path=output_path,
        output_format=args.format,
        min_len=args.min_len,
        max_len=args.max_len,
        max_per_source=args.max_per_source,
        only_sources=args.only,
    )

    if stats.get("error"):
        return 1
    if stats.get("total_kept", 0) == 0:
        _log("未产生任何输出样本（所有样本被过滤或 raw/ 为空），输出文件已创建但为空", "WARN")
        _log("提示：可降低 --min-len 或检查 raw/ 数据格式", "WARN")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
