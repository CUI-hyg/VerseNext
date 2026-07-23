"""Part5K1 Task 5：JSONL 自修复与字段标准化单元测试。

覆盖：
1. 异名字段标准化（instruction/response → prompt/completion）
2. 单字段标准化（content → text）
3. 缺逗号修复
4. 未闭合括号修复
5. BOM 去除
6. 行尾多余逗号修复
7. 严重损坏行抛 JSONLRepairError
8. repair_jsonl 端到端（临时文件 + 读取 + 标准化 + 返回）
9. load_jsonl(repair=True) 集成测试
10. load_jsonl(repair=False) 严格模式仍抛错
11. 元数据字段保留
12. write_back 写出 .repaired.jsonl

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_infra:packages/verse_torch:\
        packages/verse_nex \
        python -m pytest tests/test_jsonl_repair.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# PYTHONPATH 适配：让 tests/ 能 import verse_infra 子模块
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_infra", "verse_torch", "verse_nex"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verse_infra.verse_trainer.jsonl_repair import (  # noqa: E402
    JSONLRepairError,
    FIELD_ALIASES,
    _repair_line,
    _standardize_fields,
    repair_jsonl,
)
from verse_infra.verse_trainer.data import load_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 异名字段标准化
# ---------------------------------------------------------------------------


class TestStandardizeFields:
    """SubTask 5.2：_standardize_fields 字段标准化。"""

    def test_alias_prompt_completion(self):
        """用例 1：instruction/response → prompt/completion。"""
        item = {"instruction": "Q", "response": "A"}
        result = _standardize_fields(item)
        assert result == {"prompt": "Q", "completion": "A"}

    def test_alias_q_a(self):
        """q/a 别名同样映射为 prompt/completion。"""
        item = {"q": "question", "a": "answer"}
        result = _standardize_fields(item)
        assert result == {"prompt": "question", "completion": "answer"}

    def test_alias_input_output(self):
        """input/output 别名映射。"""
        item = {"input": "in", "output": "out"}
        result = _standardize_fields(item)
        assert result == {"prompt": "in", "completion": "out"}

    def test_single_field_content_to_text(self):
        """用例 2：单字段 content → text。"""
        item = {"content": "hello"}
        result = _standardize_fields(item)
        assert result == {"text": "hello"}

    def test_single_field_passage_to_text(self):
        """单字段 passage → text。"""
        item = {"passage": "a long passage"}
        result = _standardize_fields(item)
        assert result == {"text": "a long passage"}

    def test_standard_prompt_completion_passthrough(self):
        """已是标准 prompt/completion 直接返回。"""
        item = {"prompt": "P", "completion": "C"}
        result = _standardize_fields(item)
        assert result == {"prompt": "P", "completion": "C"}

    def test_standard_text_passthrough(self):
        """已是标准 text 直接返回。"""
        item = {"text": "raw text"}
        result = _standardize_fields(item)
        assert result == {"text": "raw text"}

    def test_metadata_preserved_prompt_completion(self):
        """元数据字段（meta/id）在 prompt-completion 场景下保留。"""
        item = {"instruction": "Q", "response": "A", "id": 42, "meta": {"src": "x"}}
        result = _standardize_fields(item)
        assert result["prompt"] == "Q"
        assert result["completion"] == "A"
        assert result["id"] == 42
        assert result["meta"] == {"src": "x"}

    def test_metadata_preserved_text(self):
        """元数据字段在 text 场景下保留。"""
        item = {"content": "hello", "id": 7}
        result = _standardize_fields(item)
        assert result["text"] == "hello"
        assert result["id"] == 7

    def test_unrecognizable_raises(self):
        """无法识别的字段结构抛 JSONLRepairError。"""
        item = {"foo": "bar", "baz": "qux"}
        with pytest.raises(JSONLRepairError, match="无法识别的字段结构"):
            _standardize_fields(item)

    def test_field_aliases_table(self):
        """FIELD_ALIASES 映射表内容校验。"""
        assert "instruction" in FIELD_ALIASES["prompt"]
        assert "response" in FIELD_ALIASES["completion"]
        assert "content" in FIELD_ALIASES["text"]
        # 关键异名全覆盖
        for alias in ("instruction", "question", "q", "input", "user", "query", "ask"):
            assert alias in FIELD_ALIASES["prompt"]
        for alias in ("response", "answer", "a", "output", "assistant", "reply", "response_text"):
            assert alias in FIELD_ALIASES["completion"]
        for alias in ("content", "raw", "body", "passage"):
            assert alias in FIELD_ALIASES["text"]


# ---------------------------------------------------------------------------
# 2-7. _repair_line 单行修复
# ---------------------------------------------------------------------------


class TestRepairLine:
    """SubTask 5.3：_repair_line 单行修复。"""

    def test_missing_comma_repair(self):
        """用例 3：缺逗号修复 {"prompt": "q" "completion": "a"} → 成功。"""
        line = '{"prompt": "q" "completion": "a"}'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_unclosed_brace_repair(self):
        """用例 4：未闭合 { 修复。"""
        line = '{"prompt": "q", "completion": "a"'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_bom_stripped(self):
        """用例 5：BOM 去除。"""
        line = '\ufeff{"prompt": "q", "completion": "a"}'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_trailing_comma_repair(self):
        """用例 6：行尾多余逗号修复。"""
        line = '{"prompt": "q", "completion": "a",}'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_trailing_comma_in_array(self):
        """数组行尾多余逗号修复。"""
        line = '{"messages": [1, 2, 3,]}'
        result = _repair_line(line)
        assert result == {"messages": [1, 2, 3]}

    def test_single_quotes_repair(self):
        """单引号修复为双引号。"""
        line = "{'prompt': 'q', 'completion': 'a'}"
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_unquoted_keys_repair(self):
        """无引号键名修复。"""
        line = '{prompt: "q", completion: "a"}'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_control_chars_stripped(self):
        """控制字符（除 \\t）去除。"""
        line = '{"prompt": "q", "completion": "a"}\x00\x01'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_empty_line_returns_none(self):
        """空行返回 None。"""
        assert _repair_line("") is None
        assert _repair_line("   ") is None
        assert _repair_line("\n") is None

    def test_valid_json_passthrough(self):
        """合法 JSON 直接解析，不走修复。"""
        line = '{"prompt": "q", "completion": "a"}'
        result = _repair_line(line)
        assert result == {"prompt": "q", "completion": "a"}

    def test_severely_broken_returns_none(self):
        """用例 7 前置：严重损坏返回 None（让上层抛错）。"""
        line = '{"prompt": "q", "completion":'
        result = _repair_line(line)
        assert result is None


# ---------------------------------------------------------------------------
# 8. repair_jsonl 端到端
# ---------------------------------------------------------------------------


class TestRepairJsonl:
    """SubTask 5.4：repair_jsonl 端到端。"""

    def test_end_to_end_standardize_and_repair(self, tmp_path):
        """用例 8：写临时文件 + 读取 + 标准化 + 返回正确 list。"""
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"instruction": "Q1", "response": "A1"}\n'
            "\n"  # 空行跳过
            '{"content": "hello"}\n'
            '{"prompt": "q" "completion": "a"}\n'  # 缺逗号
            '{"prompt": "q", "completion": "a"\n',  # 未闭合
            encoding="utf-8",
        )
        results = repair_jsonl(str(path))
        assert len(results) == 4
        assert results[0] == {"prompt": "Q1", "completion": "A1"}
        assert results[1] == {"text": "hello"}
        assert results[2] == {"prompt": "q", "completion": "a"}
        assert results[3] == {"prompt": "q", "completion": "a"}

    def test_severely_broken_raises(self, tmp_path):
        """用例 7：严重损坏行 → repair_jsonl 抛 JSONLRepairError。"""
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"prompt": "q", "completion":\n',  # 严重损坏
            encoding="utf-8",
        )
        with pytest.raises(JSONLRepairError, match="无法修复"):
            repair_jsonl(str(path))

    def test_strict_mode_raises(self, tmp_path):
        """repair=False 严格模式：损坏行直接抛错。"""
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"prompt": "q" "completion": "a"}\n',  # 缺逗号
            encoding="utf-8",
        )
        with pytest.raises(JSONLRepairError, match="解析失败"):
            repair_jsonl(str(path), repair=False)

    def test_strict_mode_valid_json(self, tmp_path):
        """repair=False 严格模式：合法 JSON 正常解析。"""
        path = tmp_path / "valid.jsonl"
        path.write_text(
            '{"prompt": "q", "completion": "a"}\n'
            '{"text": "hello"}\n',
            encoding="utf-8",
        )
        results = repair_jsonl(str(path), repair=False)
        assert results == [
            {"prompt": "q", "completion": "a"},
            {"text": "hello"},
        ]

    def test_write_back(self, tmp_path):
        """用例 12：write_back 写到 .repaired.jsonl（不覆盖原文件）。"""
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"instruction": "Q", "response": "A"}\n',
            encoding="utf-8",
        )
        results = repair_jsonl(str(path), write_back=True)
        repaired_path = Path(str(path) + ".repaired.jsonl")
        assert repaired_path.exists()
        # 原文件仍存在
        assert path.exists()
        # 修复后的文件内容正确
        lines = repaired_path.read_text(encoding="utf-8").strip().split("\n")
        assert json.loads(lines[0]) == {"prompt": "Q", "completion": "A"}
        assert results == [{"prompt": "Q", "completion": "A"}]

    def test_empty_file(self, tmp_path):
        """空文件返回空列表。"""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        results = repair_jsonl(str(path))
        assert results == []

    def test_bom_file(self, tmp_path):
        """BOM 头文件可修复。"""
        path = tmp_path / "bom.jsonl"
        path.write_text(
            '\ufeff{"prompt": "q", "completion": "a"}\n',
            encoding="utf-8",
        )
        results = repair_jsonl(str(path))
        assert results == [{"prompt": "q", "completion": "a"}]

    def test_mixed_alias_formats(self, tmp_path):
        """混合多种异名格式全部标准化。"""
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            '{"instruction": "Q", "response": "A"}\n'
            '{"question": "Q2", "answer": "A2"}\n'
            '{"input": "I", "output": "O"}\n'
            '{"content": "single"}\n'
            '{"text": "raw"}\n',
            encoding="utf-8",
        )
        results = repair_jsonl(str(path))
        assert results[0] == {"prompt": "Q", "completion": "A"}
        assert results[1] == {"prompt": "Q2", "completion": "A2"}
        assert results[2] == {"prompt": "I", "completion": "O"}
        assert results[3] == {"text": "single"}
        assert results[4] == {"text": "raw"}


# ---------------------------------------------------------------------------
# 9-10. load_jsonl 集成测试
# ---------------------------------------------------------------------------


class TestLoadJsonlIntegration:
    """SubTask 5.5：load_jsonl 集成测试。"""

    def test_load_jsonl_repair_true(self, tmp_path):
        """用例 9：load_jsonl(repair=True) 自动修复 + 标准化。"""
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"instruction": "Q", "response": "A"}\n'
            '{"prompt": "q" "completion": "a"}\n',  # 缺逗号
            encoding="utf-8",
        )
        results = load_jsonl(str(path), repair=True)
        assert results[0] == {"prompt": "Q", "completion": "A"}
        assert results[1] == {"prompt": "q", "completion": "a"}

    def test_load_jsonl_repair_default(self, tmp_path):
        """load_jsonl 默认 repair=True。"""
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"content": "hello"}\n',
            encoding="utf-8",
        )
        results = load_jsonl(str(path))  # 不传 repair，默认 True
        assert results == [{"text": "hello"}]

    def test_load_jsonl_repair_false_strict(self, tmp_path):
        """用例 10：load_jsonl(repair=False) 严格模式抛错。"""
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"prompt": "q" "completion": "a"}\n',  # 缺逗号
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="JSONL 解析失败"):
            load_jsonl(str(path), repair=False)

    def test_load_jsonl_repair_false_valid(self, tmp_path):
        """load_jsonl(repair=False) 合法 JSON 正常解析（不标准化字段）。"""
        path = tmp_path / "valid.jsonl"
        path.write_text(
            '{"prompt": "q", "completion": "a"}\n'
            '{"text": "hello"}\n',
            encoding="utf-8",
        )
        results = load_jsonl(str(path), repair=False)
        # 严格模式不做字段标准化，原样返回
        assert results == [
            {"prompt": "q", "completion": "a"},
            {"text": "hello"},
        ]

    def test_load_jsonl_repair_false_keeps_aliases(self, tmp_path):
        """load_jsonl(repair=False) 不做异名标准化，保留原字段名。"""
        path = tmp_path / "aliases.jsonl"
        path.write_text(
            '{"instruction": "Q", "response": "A"}\n',
            encoding="utf-8",
        )
        results = load_jsonl(str(path), repair=False)
        # 严格模式：原字段名保留
        assert results == [{"instruction": "Q", "response": "A"}]
