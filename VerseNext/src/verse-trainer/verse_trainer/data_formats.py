"""训练数据格式支持：文本文件与预分词文件。

支持的文本格式（统一产出 ``list[str]``，再交给分词器编码）::

    .jsonl / .json     每行/整体为对象或字符串；字段自动识别（见 extract_text）
    .txt / .md         UTF-8 纯文本，按非空行切分为文档
    .csv / .tsv        表格；用 text_column/text_columns 指定列，或拼接全部列
    .parquet           表格（需 pyarrow 或 pandas，可选依赖）

预分词格式（直接产出 token id 数组，跳过分词器与 token 缓存）::

    .npy               numpy 数组（int32/uint16 等）
    .bin / .tokens     raw 二进制 token 流，dtype 由参数指定

字段识别优先级：``text_template`` > ``text_columns`` > ``text_column`` >
通用字段（text/content/...）> 问答对（query+response/...）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from verse_core import get_logger

logger = get_logger("data_formats")

TEXT_SUFFIXES = frozenset(
    {".jsonl", ".json", ".txt", ".md", ".markdown", ".csv", ".tsv", ".parquet"}
)
TOKEN_SUFFIXES = frozenset({".npy", ".bin", ".tokens"})

TEXT_FIELDS = ("text", "content", "document", "passage", "raw")
QUERY_FIELDS = ("query", "question", "prompt", "instruction", "input", "human", "user")
RESPONSE_FIELDS = ("response", "answer", "output", "completion", "assistant", "gpt", "label")

_TOKEN_DTYPES = {
    "int32": np.int32,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int64": np.int64,
}


def is_pretokenized(path: str | Path) -> bool:
    """判断路径是否为预分词 token 文件（按后缀）。"""
    return Path(path).suffix.lower() in TOKEN_SUFFIXES


def detect_format(path: str | Path) -> str:
    """返回格式名（后缀小写，去掉点）；未知后缀返回 ``unknown``。"""
    suffix = Path(path).suffix.lower()
    if suffix in TEXT_SUFFIXES or suffix in TOKEN_SUFFIXES:
        return suffix.lstrip(".")
    return "unknown"


def _first_str(obj: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = obj.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_text(
    obj: dict,
    *,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    text_template: str | None = None,
) -> str | None:
    """从一条记录中抽取训练文本；无法识别返回 ``None``。"""
    if text_template:
        try:
            return text_template.format(**obj)
        except (KeyError, IndexError, ValueError):
            return None
    if text_columns:
        values = [obj.get(col) for col in text_columns]
        if any(isinstance(v, str) and v.strip() for v in values):
            return "\n".join("" if v is None else str(v) for v in values)
        return None
    if text_column:
        value = obj.get(text_column)
        return value if isinstance(value, str) and value.strip() else None

    text = _first_str(obj, TEXT_FIELDS)
    if text is not None:
        return text
    query = _first_str(obj, QUERY_FIELDS)
    response = _first_str(obj, RESPONSE_FIELDS)
    if query is not None or response is not None:
        return "\n".join(part for part in (query or "", response or "") if part)
    return None


def _record_to_text(
    obj,
    *,
    text_column: str | None,
    text_columns: list[str] | None,
    text_template: str | None,
    allow_json_fallback: bool,
) -> str | None:
    """对象/标量 -> 文本；字典走字段识别，标量直接 str。"""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False) if allow_json_fallback else str(obj)
    text = extract_text(
        obj, text_column=text_column, text_columns=text_columns,
        text_template=text_template,
    )
    if text is None and allow_json_fallback:
        return json.dumps(obj, ensure_ascii=False)
    return text


def load_texts_auto(
    path: str | Path,
    *,
    text_column: str | None = None,
    text_columns: list[str] | None = None,
    delimiter: str | None = None,
    text_template: str | None = None,
) -> list[str]:
    """按文件后缀加载文本，返回 ``list[str]``。

    Args:
        path: 数据文件路径。
        text_column: 指定单列作为文本（csv/tsv/parquet/json）。
        text_columns: 指定多列拼接为文本。
        delimiter: csv 分隔符（默认按后缀：``.tsv`` -> ``\\t``，其余 ``,``）。
        text_template: ``str.format`` 模板（如 ``"{query}\\n{response}"``）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix in TOKEN_SUFFIXES:
        raise ValueError(f"{path} 是预分词文件，请用 load_token_stream() 加载")
    if suffix == ".jsonl":
        return _load_jsonl(path, text_column, text_columns, text_template)
    if suffix == ".json":
        return _load_json(path, text_column, text_columns, text_template)
    if suffix in (".txt", ".md", ".markdown"):
        return _load_plain_text(path)
    if suffix in (".csv", ".tsv"):
        delim = delimiter or ("\t" if suffix == ".tsv" else ",")
        return _load_delimited(path, delim, text_column, text_columns, text_template)
    if suffix == ".parquet":
        return _load_parquet(path, text_column, text_columns, text_template)
    raise ValueError(
        f"不支持的数据格式: {path.suffix}（支持 {sorted(TEXT_SUFFIXES | TOKEN_SUFFIXES)}）"
    )


def _load_jsonl(path, text_column, text_columns, text_template) -> list[str]:
    texts: list[str] = []
    fallbacks = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _record_to_text(
                obj, text_column=text_column, text_columns=text_columns,
                text_template=text_template, allow_json_fallback=False,
            )
            if text is None:
                fallbacks += 1
                text = json.dumps(obj, ensure_ascii=False)
            texts.append(text)
    if fallbacks:
        logger.warning(
            "%s: %d/%d 条未识别到文本字段，已回退为整行 JSON（建议指定 text_column）",
            path.name, fallbacks, len(texts),
        )
    return _ensure_non_empty(path, texts)


def _load_json(path, text_column, text_columns, text_template) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # 可能是 {"data": [...]} 或单条记录
        for key in ("data", "records", "items", "conversations"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} 的 JSON 结构不受支持（需为数组或含数组字段的对象）")
    texts = []
    for obj in data:
        text = _record_to_text(
            obj, text_column=text_column, text_columns=text_columns,
            text_template=text_template, allow_json_fallback=True,
        )
        if text is not None:
            texts.append(text)
    return _ensure_non_empty(path, texts)


def _load_plain_text(path) -> list[str]:
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _ensure_non_empty(path, texts)


def _load_delimited(path, delimiter, text_column, text_columns, text_template) -> list[str]:
    texts = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            text = extract_text(
                row, text_column=text_column, text_columns=text_columns,
                text_template=text_template,
            )
            if text is None:
                # 未指定列时拼接全部列
                text = "\n".join(v for v in row.values() if v)
            if text:
                texts.append(text)
    return _ensure_non_empty(path, texts)


def _load_parquet(path, text_column, text_columns, text_template) -> list[str]:
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    except ImportError:
        try:
            import pandas as pd

            rows = pd.read_parquet(path).to_dict(orient="records")
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise ImportError(
                "读取 parquet 需要 pyarrow 或 pandas：pip install pyarrow"
            ) from exc
    texts = []
    for row in rows:
        text = extract_text(
            row, text_column=text_column, text_columns=text_columns,
            text_template=text_template,
        )
        if text is None:
            text = "\n".join(str(v) for v in row.values() if v is not None)
        if text:
            texts.append(text)
    return _ensure_non_empty(path, texts)


def _ensure_non_empty(path, texts: list[str]) -> list[str]:
    if not texts:
        raise ValueError(f"{path} 中没有可用数据")
    return texts


def load_token_stream(
    path: str | Path,
    *,
    dtype: str = "int32",
    mmap: bool = False,
) -> np.ndarray:
    """加载预分词 token 流，返回一维整型 numpy 数组。

    Args:
        path: ``.npy``（自带 dtype）或 ``.bin``/``.tokens``（按 ``dtype`` 解析）。
        dtype: raw 二进制的元素类型（int32/uint16/uint32/int64）。
        mmap: ``.npy`` 是否用内存映射（大文件省内存，随机访问会触发磁盘读）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"token 文件不存在: {path}")
    if dtype not in _TOKEN_DTYPES:
        raise ValueError(f"未知 token dtype: {dtype}（可选 {sorted(_TOKEN_DTYPES)}）")
    np_dtype = _TOKEN_DTYPES[dtype]
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r" if mmap else None)
        array = np.asarray(array).reshape(-1)
        if array.dtype != np_dtype and array.dtype.kind in "iu":
            array = array.astype(np_dtype, copy=False)
    else:
        array = np.fromfile(path, dtype=np_dtype)
    if array.size == 0:
        raise ValueError(f"{path} 为空")
    return array
