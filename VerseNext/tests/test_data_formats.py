"""训练数据 / 文件格式支持测试（data_formats + SFT 多格式加载）。"""

import json

import numpy as np
import pytest

from verse_trainer.data_formats import (
    detect_format,
    extract_text,
    is_pretokenized,
    load_texts_auto,
    load_token_stream,
)


def test_detect_and_pretokenized(tmp_path):
    assert detect_format("a.jsonl") == "jsonl"
    assert detect_format("a.parquet") == "parquet"
    assert detect_format("a.unknown") == "unknown"
    assert is_pretokenized("a.npy") and is_pretokenized("a.bin")
    assert not is_pretokenized("a.jsonl")


def test_extract_text_priority():
    obj = {"query": "Q", "response": "R", "text": "T"}
    assert extract_text(obj) == "T"                       # 通用字段优先
    assert extract_text(obj, text_column="query") == "Q"
    assert extract_text(obj, text_columns=["query", "response"]) == "Q\nR"
    assert extract_text(obj, text_template="{query} => {response}") == "Q => R"
    assert extract_text({"x": 1}) is None


def test_load_txt_md(tmp_path):
    (tmp_path / "a.txt").write_text("第一行\n\n第二行\n", encoding="utf-8")
    assert load_texts_auto(tmp_path / "a.txt") == ["第一行", "第二行"]
    (tmp_path / "a.md").write_text("# 标题\n正文", encoding="utf-8")
    assert load_texts_auto(tmp_path / "a.md") == ["# 标题", "正文"]


def test_load_jsonl_query_response(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps({"query": f"问{i}", "response": f"答{i}"})
                           for i in range(3)), encoding="utf-8")
    texts = load_texts_auto(p)
    assert texts == ["问0\n答0", "问1\n答1", "问2\n答2"]


def test_load_json_array(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"text": "x"}, {"text": "y"}]), encoding="utf-8")
    assert load_texts_auto(p) == ["x", "y"]


def test_load_csv_columns_and_template(tmp_path):
    p = tmp_path / "qa.csv"
    p.write_text("question,answer\n问1,答1\n问2,答2\n", encoding="utf-8")
    assert load_texts_auto(p, text_columns=["question", "answer"]) == ["问1\n答1", "问2\n答2"]
    assert load_texts_auto(p, text_template="{question} => {answer}") == ["问1 => 答1", "问2 => 答2"]
    assert load_texts_auto(p, text_column="answer") == ["答1", "答2"]


def test_load_tsv(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("text\n甲\n乙\n", encoding="utf-8")
    assert load_texts_auto(p) == ["甲", "乙"]


def test_load_parquet_optional(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"text": ["a", "b", "c"]}), tmp_path / "a.parquet")
    assert load_texts_auto(tmp_path / "a.parquet") == ["a", "b", "c"]


def test_load_token_stream_npy_bin(tmp_path):
    np.save(tmp_path / "t.npy", np.arange(100, dtype=np.int32))
    arr = load_token_stream(tmp_path / "t.npy")
    assert arr.dtype == np.int32 and arr.tolist()[:3] == [0, 1, 2]

    (tmp_path / "t.bin").write_bytes(np.arange(50, dtype=np.uint16).tobytes())
    raw = load_token_stream(tmp_path / "t.bin", dtype="uint16")
    assert raw.dtype == np.uint16 and raw.size == 50


def test_load_token_stream_mmap(tmp_path):
    np.save(tmp_path / "t.npy", np.arange(1000, dtype=np.int32))
    arr = load_token_stream(tmp_path / "t.npy", mmap=True)
    assert arr.size == 1000 and int(arr[999]) == 999


def test_pretokenized_rejected_by_text_loader(tmp_path):
    np.save(tmp_path / "t.npy", np.arange(10, dtype=np.int32))
    with pytest.raises(ValueError):
        load_texts_auto(tmp_path / "t.npy")


def test_load_conversations_formats(tmp_path):
    from verse_trainer.sft import load_conversations

    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    p1 = tmp_path / "s.jsonl"
    p1.write_text(json.dumps({"messages": msgs}), encoding="utf-8")
    assert len(load_conversations(p1)) == 1

    p2 = tmp_path / "s.json"
    p2.write_text(json.dumps([{"conversations": msgs}]), encoding="utf-8")
    assert len(load_conversations(p2)) == 1

    p3 = tmp_path / "s.csv"
    p3.write_text("system,user,assistant\nS,U,A\n", encoding="utf-8")
    conv = load_conversations(p3)
    assert [m["role"] for m in conv[0]] == ["system", "user", "assistant"]
