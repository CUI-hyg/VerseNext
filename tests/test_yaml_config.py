"""Task 6.4: PyYAML 配置解析单元测试。

覆盖：
1. list 正确解析为 Python list（flow + block 两种写法）
2. 多行字符串 ``|``（literal）与 ``>``（folded）正确解析
3. 引号转义（含 ``:`` 的值用引号包裹后正确解析为 str）
4. 数值类型 int / float / bool / None 正确解析
5. 向后兼容：现有 ``data/demo/config/config.yml`` 仍能正确加载
6. ``save_full_config`` + ``load_full_config`` 往返一致（含 list / 中文）
7. 无 PyYAML 时 fallback 路径仍可解析标量子集（不失败）

运行方式：
    cd /workspace && python -m pytest tests/test_yaml_config.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 让 tests/ 目录能 import data/demo/model/config.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_DIR = _REPO_ROOT / "data" / "demo"
sys.path.insert(0, str(_DEMO_DIR))

from model import config as cfg_module  # noqa: E402
from model.config import load_full_config, save_full_config, CometSparkConfig  # noqa: E402

_HAS_YAML = cfg_module._HAS_YAML


# 若环境无 PyYAML，list / 多行字符串 / 引号转义相关测试需要 skip
requires_yaml = pytest.mark.skipif(
    not _HAS_YAML,
    reason="环境未安装 PyYAML，list/多行字符串/引号转义等高级语法不被极简解析器支持",
)


# ---------------------------------------------------------------------------
# 测试用 YAML 文本
# ---------------------------------------------------------------------------

# 含 list（flow 风格 + block 风格）、多行字符串、引号转义、各类标量
YAML_FULL = """\
model:
  arch: transformer
  n_layer: 4
  n_embd: 128
  dropout: 0.1
  tie_weights: true
  n_kv_head: null
  prompts:
    - "床前明月光"
    - "你好，世界"
  aliases: ["a", "b", "c"]
  description: |
    这是一个
    多行
    描述。
  summary: >
    折叠
    多行
    字符串。
  path_with_colon: "http://example.com:8080/path"
  ratio: 0.25
  count: 42
training:
  lr: 0.003
  warmup: 20
  enabled: false
"""


# 现有 config.yml 路径
_CONFIG_YML = _DEMO_DIR / "config" / "config.yml"


# ---------------------------------------------------------------------------
# 测试 1: list 正确解析（PyYAML 模式）
# ---------------------------------------------------------------------------

@requires_yaml
def test_list_flow_style():
    """flow 风格 list ``aliases: ["a", "b", "c"]`` 应解析为 Python list。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write('model:\n  aliases: ["a", "b", "c"]\n')
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    assert cfg["model"]["aliases"] == ["a", "b", "c"]


@requires_yaml
def test_list_block_style():
    """block 风格 list（``- item``）应解析为 Python list。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write('model:\n  prompts:\n    - "床前明月光"\n    - "你好，世界"\n')
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    assert cfg["model"]["prompts"] == ["床前明月光", "你好，世界"]


# ---------------------------------------------------------------------------
# 测试 2: 多行字符串 ``|`` 和 ``>``
# ---------------------------------------------------------------------------

@requires_yaml
def test_multiline_literal_block():
    """``|`` literal block 保留换行。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write("model:\n  description: |\n    这是一个\n    多行\n    描述。\n")
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    assert cfg["model"]["description"] == "这是一个\n多行\n描述。\n"


@requires_yaml
def test_multiline_folded_block():
    """``>`` folded block 折叠换行为空格。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write("model:\n  summary: >\n    折叠\n    多行\n    字符串。\n")
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    assert cfg["model"]["summary"] == "折叠 多行 字符串。\n"


# ---------------------------------------------------------------------------
# 测试 3: 引号转义（含 ``:`` 的值）
# ---------------------------------------------------------------------------

@requires_yaml
def test_quoted_value_with_colon():
    """含冒号的值用引号包裹应解析为完整 str，不被截断。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write('model:\n  url: "http://example.com:8080/path"\n')
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    assert cfg["model"]["url"] == "http://example.com:8080/path"


@requires_yaml
def test_single_quoted_value():
    """单引号包裹的值也应正确解析。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write("model:\n  name: 'comet:spark'\n")
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    assert cfg["model"]["name"] == "comet:spark"


# ---------------------------------------------------------------------------
# 测试 4: 数值类型 int / float / bool / None
# ---------------------------------------------------------------------------

def test_scalar_types_always_supported():
    """int / float / bool / None 在 PyYAML 与 fallback 两种模式下均应正确解析。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write(
            "model:\n"
            "  n_layer: 4\n"
            "  lr: 0.001\n"
            "  tie_weights: true\n"
            "  enabled: false\n"
            "  n_kv_head: null\n"
        )
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    m = cfg["model"]
    assert m["n_layer"] == 4 and isinstance(m["n_layer"], int)
    assert m["lr"] == 0.001 and isinstance(m["lr"], float)
    assert m["tie_weights"] is True
    assert m["enabled"] is False
    assert m["n_kv_head"] is None


# ---------------------------------------------------------------------------
# 测试 5: 向后兼容 - 现有 config.yml 仍可加载
# ---------------------------------------------------------------------------

def test_existing_config_yml_loads():
    """现有 ``data/demo/config/config.yml`` 在两种模式下均应成功加载，且 model 段字段类型正确。"""
    cfg = load_full_config(str(_CONFIG_YML))
    assert "model" in cfg
    assert "training" in cfg
    assert "tokenizer" in cfg
    m = cfg["model"]
    # 关键字段类型断言
    assert m["arch"] == "transformer"
    assert m["n_layer"] == 2 and isinstance(m["n_layer"], int)
    assert m["n_embd"] == 64 and isinstance(m["n_embd"], int)
    assert m["dropout"] == 0.1 and isinstance(m["dropout"], float)
    assert m["tie_weights"] is True
    assert m["n_kv_head"] == 2


def test_cometspark_config_from_yaml():
    """``CometSparkConfig.from_yaml`` 在两种模式下应正常构造。"""
    c = CometSparkConfig.from_yaml(str(_CONFIG_YML))
    assert c.arch == "transformer"
    assert c.n_layer == 2
    assert c.n_embd == 64
    assert c.tie_weights is True
    assert c.n_kv_head == 2


# ---------------------------------------------------------------------------
# 测试 6: save_full_config + load_full_config 往返一致
# ---------------------------------------------------------------------------

@requires_yaml
def test_roundtrip_with_list_and_unicode():
    """含 list + 中文的 dict 经 save → load 应保持等价。"""
    original = {
        "model": {
            "arch": "transformer",
            "prompts": ["床前明月光", "你好，世界"],
            "aliases": ["a", "b", "c"],
            "n_layer": 4,
            "ratio": 0.25,
            "enabled": False,
            "note": None,
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        path = f.name
    try:
        save_full_config(original, path)
        loaded = load_full_config(path)
    finally:
        os.unlink(path)
    assert loaded == original


# ---------------------------------------------------------------------------
# 测试 7: 完整 YAML_FULL 各项一并验证（PyYAML 模式）
# ---------------------------------------------------------------------------

@requires_yaml
def test_full_yaml_roundtrip():
    """YAML_FULL 一次性验证 list / 多行 / 引号 / 标量。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write(YAML_FULL)
        path = f.name
    try:
        cfg = load_full_config(path)
    finally:
        os.unlink(path)
    m = cfg["model"]
    # list
    assert m["prompts"] == ["床前明月光", "你好，世界"]
    assert m["aliases"] == ["a", "b", "c"]
    # 多行字符串
    assert m["description"] == "这是一个\n多行\n描述。\n"
    assert m["summary"] == "折叠 多行 字符串。\n"
    # 引号转义
    assert m["path_with_colon"] == "http://example.com:8080/path"
    # 数值类型
    assert m["ratio"] == 0.25 and isinstance(m["ratio"], float)
    assert m["count"] == 42 and isinstance(m["count"], int)
    assert m["tie_weights"] is True
    assert m["n_kv_head"] is None
    # training 段
    assert cfg["training"]["enabled"] is False


# ---------------------------------------------------------------------------
# 测试 8: fallback 路径仍可工作（直接调用 _parse_yaml_fallback）
# ---------------------------------------------------------------------------

def test_fallback_parser_basic_scalars():
    """直接调用极简解析器，确认标量子集仍可解析（不依赖 PyYAML 是否安装）。"""
    text = (
        "model:\n"
        "  n_layer: 4\n"
        "  arch: hybrid\n"
        "  tie_weights: true\n"
        "training:\n"
        "  lr: 0.001\n"
        "  enabled: false\n"
    )
    result = cfg_module._parse_yaml_fallback(text)
    assert result["model"]["n_layer"] == 4
    assert result["model"]["arch"] == "hybrid"
    assert result["model"]["tie_weights"] is True
    assert result["training"]["lr"] == 0.001
    assert result["training"]["enabled"] is False
