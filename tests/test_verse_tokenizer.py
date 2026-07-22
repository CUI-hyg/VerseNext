"""VerseTokenizer 测试：lazy import + Qwen3 ChatML + mock transformers。

测试策略
--------
所有测试都用 mock 模拟 ``transformers.AutoTokenizer``，**不下载真实 Qwen3 模型**。
覆盖：
1. lazy import 行为（不安装 transformers 时 import verse 模块不报错）
2. 调用 ``VerseTokenizer()`` 时若 transformers 缺失抛出明确 ImportError
3. ``apply_chat_template`` 输出格式正确（mock tokenizer）
4. ``apply_prompt_template`` 输出正确（手动拼接版本）
5. ``bos_id`` / ``eos_id`` / ``pad_id`` 属性
6. ``split_prompt_completion_qwen`` 函数
7. save/load 流程（mock save_pretrained / from_pretrained）
8. 带 system prompt 的 chat template
9. encode/decode 透传
10. from_pretrained / from_local 便捷方法
11. **Qwen 优化点 1: 缓存特殊 token id（O(1) frozenset 查询）**
12. **Qwen 优化点 2: 批量 encode/decode**
13. **Qwen 优化点 3: 思考模式 extract_thinking / extract_response**
14. **Qwen 优化点 4: 工具调用 extract_tool_calls**
15. **Qwen 优化点 5: 流式 decode（UTF-8 边界安全）**
16. **Qwen 优化点 6: im_start_id / im_end_id / endoftext_id 缓存属性**
17. **向后兼容：QwenTokenizer 是 VerseTokenizer 的别名**

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_infra python -m pytest tests/test_verse_tokenizer.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_infra.verse_tokenizer import (
    VerseTokenizer,
    QwenTokenizer,  # 向后兼容别名
    QWEN_IM_START,
    QWEN_IM_END,
    QWEN_ENDOFTEXT,
    render_chat_qwen,
    render_prompt_qwen,
    split_prompt_completion_qwen,
)
from verse_infra.verse_tokenizer.verse import _import_transformers


# ---------------------------------------------------------------------------
# MockQwenTokenizer：模拟 transformers.AutoTokenizer 的接口
# ---------------------------------------------------------------------------


class MockAddedToken:
    """模拟 transformers.AddedToken（含 content 属性）。"""

    def __init__(self, content: str):
        self.content = content


class MockQwenTokenizer:
    """模拟 Qwen3 tokenizer 的接口（用于测试，不下载真实模型）。

    模拟的字段：
        - ``vocab_size``：词表大小
        - ``bos_token_id`` / ``eos_token_id`` / ``pad_token_id`` / ``unk_token_id``
        - ``get_vocab()``：返回 vocab 字典
        - ``added_tokens_decoder``：special token 字典
        - ``encode(text, add_special_tokens=...)``：返回 id 列表
        - ``decode(ids, skip_special_tokens=...)``：返回字符串
        - ``apply_chat_template(messages, tokenize=False)``：返回 ChatML 字符串
        - ``save_pretrained(dir)``：写入 mock 文件
        - ``from_pretrained(path, trust_remote_code=...)``：类方法（返回实例）
        - ``__call__(texts, ...)``：批量 encode
        - ``batch_decode(batch_ids, skip_special_tokens=...)``：批量 decode
    """

    # 模拟 Qwen3-32B 的特殊 token id（仅测试用，与真实模型可能不同）
    IM_START_ID = 151644
    IM_END_ID = 151645
    ENDOFTEXT_ID = 151643

    def __init__(
        self,
        vocab: dict[str, int] = None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        unk_token_id=None,
    ):
        # 默认 vocab：3 个 special token + 一些普通 token
        self._vocab = vocab or {
            QWEN_IM_START: self.IM_START_ID,
            QWEN_IM_END: self.IM_END_ID,
            QWEN_ENDOFTEXT: self.ENDOFTEXT_ID,
            "你好": 1000,
            "世界": 1001,
            "hello": 1002,
            "world": 1003,
        }
        self.vocab_size = max(self._vocab.values()) + 1
        self.bos_token_id = bos_token_id  # Qwen3 通常 None
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.IM_END_ID
        self.pad_token_id = pad_token_id if pad_token_id is not None else self.ENDOFTEXT_ID
        self.unk_token_id = unk_token_id  # Qwen3 通常 None
        # 模拟 added_tokens_decoder（key 是 str 形式的 id）
        self.added_tokens_decoder = {
            str(self.IM_START_ID): MockAddedToken(QWEN_IM_START),
            str(self.IM_END_ID): MockAddedToken(QWEN_IM_END),
            str(self.ENDOFTEXT_ID): MockAddedToken(QWEN_ENDOFTEXT),
        }
        # 用于记录 save_pretrained 调用
        self._saved_dirs: list[str] = []

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        specials = sorted(self._vocab.keys(), key=len, reverse=True)
        import re
        pat = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
        for chunk in pat.split(text):
            if not chunk:
                continue
            if chunk in self._vocab:
                ids.append(self._vocab[chunk])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        id_to_tok = {v: k for k, v in self._vocab.items()}
        out: list[str] = []
        for i in ids:
            tok = id_to_tok.get(int(i))
            if tok is None:
                continue
            if skip_special_tokens and tok in (
                QWEN_IM_START, QWEN_IM_END, QWEN_ENDOFTEXT,
            ):
                continue
            out.append(tok)
        return "".join(out)

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False, **kwargs):
        # 忽略 enable_thinking 等额外 kwargs（mock 不模拟思考模式）
        text = render_chat_qwen(messages, add_generation_prompt=kwargs.get("add_generation_prompt", False))
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False, max_length=None):
        """模拟批量 encode（返回 dict，含 input_ids）。"""
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        }

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        """模拟批量 decode。"""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"mock": True, "vocab": self._vocab}, f)
        with open(os.path.join(dir_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"mock": True}, f)
        self._saved_dirs.append(dir_path)

    @classmethod
    def from_pretrained(cls, path, trust_remote_code=True, **kwargs):
        tok_json = os.path.join(str(path), "tokenizer.json")
        if os.path.exists(tok_json):
            with open(tok_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(vocab=data.get("vocab", {}))
        return cls()


# ---------------------------------------------------------------------------
# 辅助：用 mock 替换 _import_transformers 返回 MockQwenTokenizer
# ---------------------------------------------------------------------------


def _make_verse_tokenizer(**kwargs) -> VerseTokenizer:
    """用 MockQwenTokenizer 构造一个 VerseTokenizer 实例（绕过真实下载）。

    通过 patch ``verse_infra.verse_tokenizer.verse._import_transformers`` 返回
    ``MockQwenTokenizer`` 类，然后调用 ``VerseTokenizer()``。
    """
    with patch("verse_infra.verse_tokenizer.verse._import_transformers", return_value=MockQwenTokenizer):
        return VerseTokenizer(**kwargs)


# ---------------------------------------------------------------------------
# 测试 1: lazy import - 不安装 transformers 时 import verse 模块不报错
# ---------------------------------------------------------------------------


def test_lazy_import():
    """不安装 transformers 时 import verse 模块本身不报错。

    只有用 ``VerseTokenizer()`` 构造才会触发 ``_import_transformers`` 调用。
    """
    import importlib
    import verse_infra.verse_tokenizer.verse as verse_module
    importlib.reload(verse_module)
    assert hasattr(verse_module, "VerseTokenizer")
    assert hasattr(verse_module, "_import_transformers")
    assert verse_module.VerseTokenizer.DEFAULT_VERSE_MODEL == "Qwen/Qwen3-32B"
    # 向后兼容
    assert verse_module.VerseTokenizer.DEFAULT_QWEN_MODEL == "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# 测试 2: transformers 缺失时调用 VerseTokenizer() 抛出明确 ImportError
# ---------------------------------------------------------------------------


def test_import_transformers_missing():
    """模拟 transformers 缺失，调用 ``VerseTokenizer()`` 抛出明确 ImportError。"""
    with pytest.raises(ImportError) as exc_info:
        _import_transformers()
    msg = str(exc_info.value)
    assert "transformers" in msg
    assert "pip install" in msg

    with patch(
        "verse_infra.verse_tokenizer.verse._import_transformers",
        side_effect=ImportError("VerseTokenizer 需要 transformers 库。"),
    ):
        with pytest.raises(ImportError):
            VerseTokenizer(model_id="Qwen/Qwen3-32B")


# ---------------------------------------------------------------------------
# 测试 3: apply_chat_template 输出格式正确（mock tokenizer）
# ---------------------------------------------------------------------------


def test_chat_template_format():
    """apply_chat_template 输出 Qwen3 ChatML 格式。"""
    tok = _make_verse_tokenizer()
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    rendered = tok.apply_chat_template(messages)
    assert QWEN_IM_START in rendered
    assert QWEN_IM_END in rendered
    assert "user" in rendered
    assert "assistant" in rendered
    assert rendered.index("user") < rendered.index("assistant")
    assert "你好" in rendered


# ---------------------------------------------------------------------------
# 测试 4: apply_prompt_template 输出正确（手动拼接版本）
# ---------------------------------------------------------------------------


def test_prompt_template_format():
    """apply_prompt_template 输出正确的 ChatML 推理前缀。"""
    tok = _make_verse_tokenizer()
    result = tok.apply_prompt_template("你好")
    assert result.startswith(f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n")
    assert result.endswith(f"{QWEN_IM_START}assistant\n")
    assert "你好！" not in result

    result_with_sys = tok.apply_prompt_template("你好", system="你是助手")
    assert result_with_sys.startswith(f"{QWEN_IM_START}system\n你是助手{QWEN_IM_END}\n")
    assert f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n" in result_with_sys
    assert result_with_sys.endswith(f"{QWEN_IM_START}assistant\n")


# ---------------------------------------------------------------------------
# 测试 5: bos_id / eos_id / pad_id / unk_id / vocab / special_tokens 属性
# ---------------------------------------------------------------------------


def test_special_tokens_property():
    """mock tokenizer 暴露 bos_id/eos_id/pad_id。"""
    tok = _make_verse_tokenizer()
    assert tok.eos_id == MockQwenTokenizer.IM_END_ID
    assert tok.pad_id == MockQwenTokenizer.ENDOFTEXT_ID
    assert tok.bos_id == tok.eos_id  # Qwen3 无 bos，回退到 eos
    assert tok.unk_id == tok.pad_id  # Qwen3 无 unk，回退到 pad
    assert isinstance(tok.vocab, dict)
    assert QWEN_IM_START in tok.vocab
    assert QWEN_IM_END in tok.vocab
    assert QWEN_ENDOFTEXT in tok.vocab
    assert isinstance(tok.special_tokens, dict)
    assert QWEN_IM_START in tok.special_tokens
    assert QWEN_IM_END in tok.special_tokens
    assert tok.vocab_size == MockQwenTokenizer().vocab_size
    assert len(tok) == tok.vocab_size


# ---------------------------------------------------------------------------
# 测试 6: split_prompt_completion_qwen 函数
# ---------------------------------------------------------------------------


def test_qwen_chat_template_split():
    """测试 split_prompt_completion_qwen 函数（Qwen3 ChatML 版本）。"""
    text = (
        f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n"
        f"{QWEN_IM_START}assistant\n你好！{QWEN_IM_END}\n"
    )
    prompt, completion = split_prompt_completion_qwen(text)
    assert prompt == f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"
    assert completion == f"你好！{QWEN_IM_END}\n"

    prompt_only = render_prompt_qwen("测试")
    p, c = split_prompt_completion_qwen(prompt_only)
    assert p == prompt_only
    assert c == ""

    p2, c2 = split_prompt_completion_qwen("no marker here")
    assert p2 == "no marker here"
    assert c2 == ""

    multi = (
        f"{QWEN_IM_START}user\n问题1{QWEN_IM_END}\n"
        f"{QWEN_IM_START}assistant\n回答1{QWEN_IM_END}\n"
        f"{QWEN_IM_START}user\n问题2{QWEN_IM_END}\n"
        f"{QWEN_IM_START}assistant\n回答2{QWEN_IM_END}\n"
    )
    p3, c3 = split_prompt_completion_qwen(multi)
    assert p3.endswith(f"{QWEN_IM_START}assistant\n")
    assert c3 == f"回答2{QWEN_IM_END}\n"


# ---------------------------------------------------------------------------
# 测试 7: save/load 流程（mock save_pretrained / from_pretrained）
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    """mock save/load 流程：save 到目录，load 从目录读回。"""
    tok = _make_verse_tokenizer()
    original_eos = tok.eos_id
    original_pad = tok.pad_id
    original_vocab_size = tok.vocab_size

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "verse_tok")
        with patch("verse_infra.verse_tokenizer.verse._import_transformers", return_value=MockQwenTokenizer):
            tok.save(save_dir)
        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))

        new_tok = _make_verse_tokenizer()
        with patch("verse_infra.verse_tokenizer.verse._import_transformers", return_value=MockQwenTokenizer):
            new_tok.load(save_dir)
        assert new_tok.eos_id == original_eos
        assert new_tok.pad_id == original_pad
        assert new_tok.vocab_size == original_vocab_size
        assert QWEN_IM_START in new_tok.vocab

        # 测试 .json 元信息形式保存
        meta_path = os.path.join(tmpdir, "verse_meta.json")
        with patch("verse_infra.verse_tokenizer.verse._import_transformers", return_value=MockQwenTokenizer):
            tok.save(meta_path)
        assert os.path.isfile(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["type"] == "verse"
        assert "tokenizer_dir" in meta
        assert os.path.isdir(meta["tokenizer_dir"])


# ---------------------------------------------------------------------------
# 测试 8: 带 system prompt 的 chat template
# ---------------------------------------------------------------------------


def test_apply_chat_template_with_system():
    """带 system prompt 的 chat template 渲染。"""
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    rendered = render_chat_qwen(messages)
    assert rendered.count(QWEN_IM_START) == 3
    assert rendered.count(QWEN_IM_END) == 3
    assert rendered.index("system") < rendered.index("user")
    assert rendered.index("user") < rendered.index("assistant")
    assert "你是助手" in rendered
    assert "你好" in rendered
    assert "你好！" in rendered

    tok = _make_verse_tokenizer()
    rendered2 = tok.apply_chat_template(messages)
    assert rendered2 == rendered


# ---------------------------------------------------------------------------
# 测试 9: encode/decode 透传（mock 行为）
# ---------------------------------------------------------------------------


def test_encode_decode_passthrough():
    """encode/decode 透传到底层 mock tokenizer。"""
    tok = _make_verse_tokenizer()
    ids = tok.encode("你好", add_special_tokens=False)
    assert isinstance(ids, list)
    assert ids == [MockQwenTokenizer().get_vocab()["你好"]]
    text = tok.decode(ids, skip_special_tokens=True)
    assert text == "你好"


# ---------------------------------------------------------------------------
# 测试 10: from_pretrained / from_local 便捷方法
# ---------------------------------------------------------------------------


def test_from_pretrained_from_local():
    """from_pretrained / from_local 便捷构造方法。"""
    with patch("verse_infra.verse_tokenizer.verse._import_transformers", return_value=MockQwenTokenizer):
        tok1 = VerseTokenizer.from_pretrained()
        assert tok1._model_id == "Qwen/Qwen3-32B"

        tok2 = VerseTokenizer.from_pretrained("Qwen/Qwen3-14B")
        assert tok2._model_id == "Qwen/Qwen3-14B"

        tok3 = VerseTokenizer.from_local("/some/local/dir")
        assert tok3._tokenizer_dir == "/some/local/dir"
        assert tok3._model_id is None


# ===========================================================================
# 以下为 Qwen 优化点的专项测试
# ===========================================================================


# ---------------------------------------------------------------------------
# 测试 11: Qwen 优化点 1 - 缓存特殊 token id（O(1) frozenset 查询）
# ---------------------------------------------------------------------------


def test_special_token_id_cache():
    """缓存特殊 token id 集合，O(1) frozenset 查询。"""
    tok = _make_verse_tokenizer()
    # frozenset 类型的 special_token_ids
    assert isinstance(tok.special_token_ids, frozenset)
    # is_special_token 应在 O(1) 内判断
    assert tok.is_special_token(MockQwenTokenizer.IM_START_ID) is True
    assert tok.is_special_token(MockQwenTokenizer.IM_END_ID) is True
    assert tok.is_special_token(MockQwenTokenizer.ENDOFTEXT_ID) is True
    assert tok.is_special_token(999999) is False  # 不存在的 id
    assert tok.is_special_token(0) is False

    # is_special_token_str
    assert tok.is_special_token_str(QWEN_IM_START) is True
    assert tok.is_special_token_str(QWEN_IM_END) is True
    assert tok.is_special_token_str("普通token") is False


# ---------------------------------------------------------------------------
# 测试 12: Qwen 优化点 6 - im_start_id / im_end_id / endoftext_id 缓存属性
# ---------------------------------------------------------------------------


def test_qwen_special_id_properties():
    """im_start_id / im_end_id / endoftext_id 缓存属性（构造时一次解析）。"""
    tok = _make_verse_tokenizer()
    assert tok.im_start_id == MockQwenTokenizer.IM_START_ID
    assert tok.im_end_id == MockQwenTokenizer.IM_END_ID
    assert tok.endoftext_id == MockQwenTokenizer.ENDOFTEXT_ID
    # 这些属性应直接返回缓存值，不每次都查 dict
    assert isinstance(tok.im_start_id, int)
    assert isinstance(tok.im_end_id, int)
    assert isinstance(tok.endoftext_id, int)


# ---------------------------------------------------------------------------
# 测试 13: Qwen 优化点 2 - 批量 encode/decode
# ---------------------------------------------------------------------------


def test_batch_encode_decode():
    """批量 encode/decode 比循环单条更高效。"""
    tok = _make_verse_tokenizer()
    texts = ["你好", "世界", "hello", "world"]
    # 批量 encode
    batch_ids = tok.encode_batch(texts, add_special_tokens=False)
    assert isinstance(batch_ids, list)
    assert len(batch_ids) == 4
    for ids in batch_ids:
        assert isinstance(ids, list)
    # 验证与单条 encode 一致
    for text, ids in zip(texts, batch_ids):
        assert ids == tok.encode(text, add_special_tokens=False)

    # 批量 decode
    decoded = tok.decode_batch(batch_ids, skip_special_tokens=True)
    assert isinstance(decoded, list)
    assert len(decoded) == 4
    for orig, dec in zip(texts, decoded):
        assert dec == orig


# ---------------------------------------------------------------------------
# 测试 14: Qwen 优化点 3 - 思考模式 extract_thinking / extract_response
# ---------------------------------------------------------------------------


def test_thinking_mode_extract():
    """思考模式 extract_thinking / extract_response。"""
    tok = _make_verse_tokenizer()

    # 标准 Qwen3 思考模式输出
    text = "<think>\n让我想想\n</think>\n\n答案是42"
    thinking, response = tok.extract_thinking(text)
    assert thinking == "让我想想"
    assert response == "答案是42"

    # extract_response 便捷方法
    assert tok.extract_response(text) == "答案是42"

    # 无 think 标签
    text_no_think = "直接回复"
    thinking2, response2 = tok.extract_thinking(text_no_think)
    assert thinking2 == ""
    assert response2 == "直接回复"
    assert tok.extract_response(text_no_think) == "直接回复"

    # 未闭合的 <think>
    text_unclosed = "<think>\n未完成思考"
    thinking3, response3 = tok.extract_thinking(text_unclosed)
    assert thinking3 == "未完成思考"
    assert response3 == ""

    # 多个 <think> 块（取第一个）
    text_multi = "<think>\n第一段\n</think>\n\n回复1<think>\n第二段\n</think>\n\n回复2"
    thinking4, response4 = tok.extract_thinking(text_multi)
    assert thinking4 == "第一段"
    # response 应该是 </think> 后的全部（包括后续的 <think>）
    assert "回复1" in response4
    assert "第二段" in response4


# ---------------------------------------------------------------------------
# 测试 15: Qwen 优化点 3 - _strip_think_tags 静态方法
# ---------------------------------------------------------------------------


def test_strip_think_tags():
    """_strip_think_tags 移除 <think>...</think> 标签。"""
    text = "前文<think>\n思考内容\n</think>\n\n后文"
    stripped = VerseTokenizer._strip_think_tags(text)
    assert "<think>" not in stripped
    assert "</think>" not in stripped
    assert "思考内容" not in stripped
    assert "前文" in stripped
    assert "后文" in stripped

    # 多个 think 块
    text_multi = "a<think>x</think>\nb<think>y</think>\nc"
    stripped_multi = VerseTokenizer._strip_think_tags(text_multi)
    assert "x" not in stripped_multi
    assert "y" not in stripped_multi
    assert "a" in stripped_multi
    assert "b" in stripped_multi
    assert "c" in stripped_multi

    # 无 think 标签
    assert VerseTokenizer._strip_think_tags("普通文本") == "普通文本"


# ---------------------------------------------------------------------------
# 测试 16: Qwen 优化点 4 - 工具调用 extract_tool_calls
# ---------------------------------------------------------------------------


def test_extract_tool_calls():
    """extract_tool_calls 解析 <|tool_call_begin|>...<|tool_call_end|>。"""
    tok = _make_verse_tokenizer()

    # 单个工具调用
    text_single = (
        '我来调用工具<|tool_call_begin|>'
        '{"name": "search", "arguments": {"q": "天气"}}'
        '<|tool_call_end|>'
    )
    calls = tok.extract_tool_calls(text_single)
    assert len(calls) == 1
    assert calls[0]["name"] == "search"
    assert calls[0]["arguments"]["q"] == "天气"

    # 多个工具调用
    text_multi = (
        '<|tool_call_begin|>{"name": "search", "arguments": {"q": "a"}}<|tool_call_end|>'
        '<|tool_call_begin|>{"name": "calc", "arguments": {"x": 1}}<|tool_call_end|>'
    )
    calls_multi = tok.extract_tool_calls(text_multi)
    assert len(calls_multi) == 2
    assert calls_multi[0]["name"] == "search"
    assert calls_multi[1]["name"] == "calc"

    # 无工具调用
    assert tok.extract_tool_calls("普通回复") == []

    # JSON 解析失败时返回 raw
    text_bad = '<|tool_call_begin|>not valid json<|tool_call_end|>'
    calls_bad = tok.extract_tool_calls(text_bad)
    assert len(calls_bad) == 1
    assert calls_bad[0] == {"raw": "not valid json"}

    # 未闭合的工具调用
    text_unclosed = '<|tool_call_begin|>{"name": "test"}'
    calls_unclosed = tok.extract_tool_calls(text_unclosed)
    assert len(calls_unclosed) == 1
    assert calls_unclosed[0]["name"] == "test"


# ---------------------------------------------------------------------------
# 测试 17: Qwen 优化点 5 - 流式 decode（UTF-8 边界安全）
# ---------------------------------------------------------------------------


def test_decode_streaming_basic():
    """decode_streaming 处理正常 token 流（无半字符）。"""
    tok = _make_verse_tokenizer()
    tok.reset_streaming()

    # 正常 token 流：你好世界
    ids = [MockQwenTokenizer().get_vocab()["你好"], MockQwenTokenizer().get_vocab()["世界"]]
    # 一次性传入
    text = tok.decode_streaming(ids.copy(), skip_special_tokens=False)
    assert "你好" in text
    assert "世界" in text

    # 分批传入
    tok.reset_streaming()
    text1 = tok.decode_streaming([ids[0]], skip_special_tokens=False)
    text2 = tok.decode_streaming([ids[1]], skip_special_tokens=False)
    assert "你好" in text1
    assert "世界" in text2


def test_decode_streaming_skip_special():
    """decode_streaming 跳过特殊 token。"""
    tok = _make_verse_tokenizer()
    tok.reset_streaming()

    # 包含特殊 token 的流
    ids = [
        MockQwenTokenizer.IM_START_ID,
        MockQwenTokenizer().get_vocab()["你好"],
        MockQwenTokenizer.IM_END_ID,
    ]
    # skip_special_tokens=True 应过滤特殊 token
    text = tok.decode_streaming(ids, skip_special_tokens=True)
    assert "你好" in text
    assert QWEN_IM_START not in text
    assert QWEN_IM_END not in text


def test_decode_streaming_reset():
    """reset_streaming 清空 buffer。"""
    tok = _make_verse_tokenizer()
    # 模拟 buffer 中有内容
    tok._stream_buffer = [100, 200, 300]
    tok.reset_streaming()
    assert tok._stream_buffer == []


# ---------------------------------------------------------------------------
# 测试 18: 向后兼容 - QwenTokenizer 是 VerseTokenizer 的别名
# ---------------------------------------------------------------------------


def test_qwen_tokenizer_backward_compat():
    """QwenTokenizer 是 VerseTokenizer 的别名，确保旧代码不破坏。"""
    assert QwenTokenizer is VerseTokenizer
    # 通过 QwenTokenizer 也能构造
    with patch("verse_infra.verse_tokenizer.verse._import_transformers", return_value=MockQwenTokenizer):
        tok = QwenTokenizer()
        assert isinstance(tok, VerseTokenizer)
        assert isinstance(tok, QwenTokenizer)
        assert tok.DEFAULT_VERSE_MODEL == "Qwen/Qwen3-32B"
        assert tok.DEFAULT_QWEN_MODEL == "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# 测试 19: vocab 懒加载
# ---------------------------------------------------------------------------


def test_vocab_lazy_loading():
    """vocab 字典懒加载，首次访问才构建。"""
    tok = _make_verse_tokenizer()
    # 构造后 _vocab 应为 None（懒加载）
    assert tok._vocab is None
    # 首次访问触发构建
    v = tok.vocab
    assert tok._vocab is not None
    assert isinstance(v, dict)
    assert QWEN_IM_START in v
    # 再次访问应返回缓存的同一个 dict
    v2 = tok.vocab
    assert v is v2


# ---------------------------------------------------------------------------
# 测试 20: apply_chat_template 支持 add_generation_prompt 与 tokenize 选项
# ---------------------------------------------------------------------------


def test_apply_chat_template_options():
    """apply_chat_template 支持 add_generation_prompt 与 tokenize 选项。"""
    tok = _make_verse_tokenizer()
    messages = [{"role": "user", "content": "你好"}]

    # 默认 add_generation_prompt=False
    rendered = tok.apply_chat_template(messages)
    assert not rendered.endswith(f"{QWEN_IM_START}assistant\n")

    # add_generation_prompt=True
    rendered_gen = tok.apply_chat_template(messages, add_generation_prompt=True)
    assert rendered_gen.endswith(f"{QWEN_IM_START}assistant\n")

    # tokenize=True 返回 id 列表
    ids = tok.apply_chat_template(messages, tokenize=True)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
