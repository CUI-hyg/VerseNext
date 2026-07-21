"""QwenTokenizer 测试：lazy import + Qwen3 ChatML + mock transformers。

测试策略
--------
所有测试都用 mock 模拟 ``transformers.AutoTokenizer``，**不下载真实 Qwen3 模型**。
覆盖：
1. lazy import 行为（不安装 transformers 时 import qwen 模块不报错）
2. 调用 ``QwenTokenizer()`` 时若 transformers 缺失抛出明确 ImportError
3. ``apply_chat_template`` 输出格式正确（mock tokenizer）
4. ``apply_prompt_template`` 输出正确（手动拼接版本）
5. ``bos_id`` / ``eos_id`` / ``pad_id`` 属性
6. ``split_prompt_completion_qwen`` 函数
7. save/load 流程（mock save_pretrained / from_pretrained）
8. 带 system prompt 的 chat template

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_tokenizer python -m pytest tests/test_qwen_tokenizer.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# 让 tests/ 目录能 import verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_tokenizer"))

from verse_tokenizer import (
    QwenTokenizer,
    QWEN_IM_START,
    QWEN_IM_END,
    QWEN_ENDOFTEXT,
    render_chat_qwen,
    render_prompt_qwen,
    split_prompt_completion_qwen,
)
from verse_tokenizer.qwen import _import_transformers


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
        # 简单 mock：把文本按 special token 切分，匹配 vocab 的就查表，否则跳过
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

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False):
        # 用 verse_tokenizer 的 render_chat_qwen 实现，避免依赖真实模板
        text = render_chat_qwen(messages, add_generation_prompt=False)
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def save_pretrained(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        # 写入 mock 的 tokenizer.json 占位文件
        with open(os.path.join(dir_path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"mock": True, "vocab": self._vocab}, f)
        with open(os.path.join(dir_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"mock": True}, f)
        self._saved_dirs.append(dir_path)

    @classmethod
    def from_pretrained(cls, path, trust_remote_code=True, **kwargs):
        # 从目录加载：读取 mock 的 tokenizer.json
        tok_json = os.path.join(str(path), "tokenizer.json")
        if os.path.exists(tok_json):
            with open(tok_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(vocab=data.get("vocab", {}))
        # 从 model_id 加载（mock）：返回默认实例
        return cls()


# ---------------------------------------------------------------------------
# 辅助：用 mock 替换 _import_transformers 返回 MockQwenTokenizer
# ---------------------------------------------------------------------------


def _make_qwen_tokenizer(**kwargs) -> QwenTokenizer:
    """用 MockQwenTokenizer 构造一个 QwenTokenizer 实例（绕过真实下载）。

    通过 patch ``verse_tokenizer.qwen._import_transformers`` 返回
    ``MockQwenTokenizer`` 类，然后调用 ``QwenTokenizer()``。
    """
    with patch("verse_tokenizer.qwen._import_transformers", return_value=MockQwenTokenizer):
        # 构造时 tokenizer_dir=None, model_id=None，会调用
        # AutoTokenizer.from_pretrained(DEFAULT_QWEN_MODEL, ...)
        # 由于 MockQwenTokenizer.from_pretrained 是 classmethod，返回 mock 实例
        return QwenTokenizer(**kwargs)


# ---------------------------------------------------------------------------
# 测试 1: lazy import - 不安装 transformers 时 import qwen 模块不报错
# ---------------------------------------------------------------------------


def test_lazy_import():
    """不安装 transformers 时 import qwen 模块本身不报错。

    只有用 ``QwenTokenizer()`` 构造才会触发 ``_import_transformers`` 调用。
    """
    # 重新 import 模块，确保 import 时不需要 transformers
    # 由于测试环境未安装 transformers，直接 import 成功就说明 lazy import 生效
    import importlib
    import verse_tokenizer.qwen as qwen_module
    importlib.reload(qwen_module)
    # 模块级别应有 QwenTokenizer 类
    assert hasattr(qwen_module, "QwenTokenizer")
    assert hasattr(qwen_module, "_import_transformers")
    # 不应抛异常
    assert qwen_module.QwenTokenizer.DEFAULT_QWEN_MODEL == "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# 测试 2: transformers 缺失时调用 QwenTokenizer() 抛出明确 ImportError
# ---------------------------------------------------------------------------


def test_import_transformers_missing():
    """模拟 transformers 缺失，调用 ``QwenTokenizer()`` 抛出明确 ImportError。"""
    # 直接调用 _import_transformers()（环境未安装 transformers）
    with pytest.raises(ImportError) as exc_info:
        _import_transformers()
    msg = str(exc_info.value)
    # 错误信息应包含安装提示
    assert "transformers" in msg
    assert "pip install" in msg

    # 同样，调用 QwenTokenizer() 也应抛 ImportError
    with patch(
        "verse_tokenizer.qwen._import_transformers",
        side_effect=ImportError("QwenTokenizer 需要 transformers 库。"),
    ):
        with pytest.raises(ImportError):
            QwenTokenizer(model_id="Qwen/Qwen3-32B")


# ---------------------------------------------------------------------------
# 测试 3: apply_chat_template 输出格式正确（mock tokenizer）
# ---------------------------------------------------------------------------


def test_chat_template_format():
    """apply_chat_template 输出 Qwen3 ChatML 格式。"""
    tok = _make_qwen_tokenizer()
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    rendered = tok.apply_chat_template(messages)
    # 应包含 ChatML 标记
    assert QWEN_IM_START in rendered
    assert QWEN_IM_END in rendered
    assert "user" in rendered
    assert "assistant" in rendered
    # 顺序：user 在 assistant 前
    assert rendered.index("user") < rendered.index("assistant")
    # 应包含原始 content
    assert "你好" in rendered


# ---------------------------------------------------------------------------
# 测试 4: apply_prompt_template 输出正确（手动拼接版本）
# ---------------------------------------------------------------------------


def test_prompt_template_format():
    """apply_prompt_template 输出正确的 ChatML 推理前缀。"""
    tok = _make_qwen_tokenizer()
    result = tok.apply_prompt_template("你好")
    # 应以 <|im_start|>user\n 开头，以 <|im_start|>assistant\n 结尾
    assert result.startswith(f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n")
    assert result.endswith(f"{QWEN_IM_START}assistant\n")
    # 不应包含 assistant 的内容（留空等待生成）
    assert "你好！" not in result

    # 带 system prompt
    result_with_sys = tok.apply_prompt_template("你好", system="你是助手")
    assert result_with_sys.startswith(f"{QWEN_IM_START}system\n你是助手{QWEN_IM_END}\n")
    assert f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n" in result_with_sys
    assert result_with_sys.endswith(f"{QWEN_IM_START}assistant\n")


# ---------------------------------------------------------------------------
# 测试 5: bos_id / eos_id / pad_id / unk_id / vocab / special_tokens 属性
# ---------------------------------------------------------------------------


def test_special_tokens_property():
    """mock tokenizer 暴露 bos_id/eos_id/pad_id。"""
    tok = _make_qwen_tokenizer()
    # Qwen3 mock：bos_token_id=None → bos_id 回退到 eos_id
    assert tok.eos_id == MockQwenTokenizer.IM_END_ID
    assert tok.pad_id == MockQwenTokenizer.ENDOFTEXT_ID
    # bos_id 应回退到 eos_id（Qwen3 无 bos）
    assert tok.bos_id == tok.eos_id
    # unk_id 应回退到 pad_id（Qwen3 无 unk）
    assert tok.unk_id == tok.pad_id
    # vocab 是 dict[str, int]
    assert isinstance(tok.vocab, dict)
    assert QWEN_IM_START in tok.vocab
    assert QWEN_IM_END in tok.vocab
    assert QWEN_ENDOFTEXT in tok.vocab
    # special_tokens 包含 ChatML 标记
    assert isinstance(tok.special_tokens, dict)
    assert QWEN_IM_START in tok.special_tokens
    assert QWEN_IM_END in tok.special_tokens
    # vocab_size 与底层 tokenizer 一致
    assert tok.vocab_size == MockQwenTokenizer().vocab_size
    assert len(tok) == tok.vocab_size


# ---------------------------------------------------------------------------
# 测试 6: split_prompt_completion_qwen 函数
# ---------------------------------------------------------------------------


def test_qwen_chat_template_split():
    """测试 split_prompt_completion_qwen 函数（Qwen3 ChatML 版本）。"""
    # 完整对话：user + assistant
    text = (
        f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n"
        f"{QWEN_IM_START}assistant\n你好！{QWEN_IM_END}\n"
    )
    prompt, completion = split_prompt_completion_qwen(text)
    assert prompt == f"{QWEN_IM_START}user\n你好{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"
    assert completion == f"你好！{QWEN_IM_END}\n"

    # 只有 prompt（推理前缀，无 assistant 内容）
    prompt_only = render_prompt_qwen("测试")
    p, c = split_prompt_completion_qwen(prompt_only)
    assert p == prompt_only
    assert c == ""

    # 无 marker 的情况
    p2, c2 = split_prompt_completion_qwen("no marker here")
    assert p2 == "no marker here"
    assert c2 == ""

    # 多轮对话：找最后一个 <|im_start|>assistant\n
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
    tok = _make_qwen_tokenizer()
    original_eos = tok.eos_id
    original_pad = tok.pad_id
    original_vocab_size = tok.vocab_size

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "qwen_tok")
        # save 应调用底层 save_pretrained
        with patch("verse_tokenizer.qwen._import_transformers", return_value=MockQwenTokenizer):
            tok.save(save_dir)
        # 目录应存在且含 mock 文件
        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))

        # load：从目录加载（patch from_pretrained 返回 mock）
        new_tok = _make_qwen_tokenizer()
        # 先 patch _import_transformers 以便 load 时也用 mock
        with patch("verse_tokenizer.qwen._import_transformers", return_value=MockQwenTokenizer):
            new_tok.load(save_dir)
        # 加载后属性应与原 tokenizer 一致
        assert new_tok.eos_id == original_eos
        assert new_tok.pad_id == original_pad
        assert new_tok.vocab_size == original_vocab_size
        assert QWEN_IM_START in new_tok.vocab

        # 测试 .json 元信息形式保存
        meta_path = os.path.join(tmpdir, "qwen_meta.json")
        with patch("verse_tokenizer.qwen._import_transformers", return_value=MockQwenTokenizer):
            tok.save(meta_path)
        assert os.path.isfile(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["type"] == "qwen"
        assert "tokenizer_dir" in meta
        # 实际目录应已保存
        assert os.path.isdir(meta["tokenizer_dir"])


# ---------------------------------------------------------------------------
# 测试 8: 带 system prompt 的 chat template
# ---------------------------------------------------------------------------


def test_apply_chat_template_with_system():
    """带 system prompt 的 chat template 渲染。"""
    # 直接用 render_chat_qwen 函数测试
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
    ]
    rendered = render_chat_qwen(messages)
    # 应包含三段 ChatML
    assert rendered.count(QWEN_IM_START) == 3
    assert rendered.count(QWEN_IM_END) == 3
    # 顺序：system → user → assistant
    assert rendered.index("system") < rendered.index("user")
    assert rendered.index("user") < rendered.index("assistant")
    # 应包含所有 content
    assert "你是助手" in rendered
    assert "你好" in rendered
    assert "你好！" in rendered

    # 用 QwenTokenizer.apply_chat_template 也应工作（mock 透传到 render_chat_qwen）
    tok = _make_qwen_tokenizer()
    rendered2 = tok.apply_chat_template(messages)
    assert rendered2 == rendered


# ---------------------------------------------------------------------------
# 额外测试 9: encode/decode 透传（mock 行为）
# ---------------------------------------------------------------------------


def test_encode_decode_passthrough():
    """encode/decode 透传到底层 mock tokenizer。"""
    tok = _make_qwen_tokenizer()
    # encode 应返回 id 列表
    ids = tok.encode("你好", add_special_tokens=False)
    assert isinstance(ids, list)
    assert ids == [MockQwenTokenizer().get_vocab()["你好"]]
    # decode 应返回字符串
    text = tok.decode(ids, skip_special_tokens=True)
    assert text == "你好"


# ---------------------------------------------------------------------------
# 额外测试 10: from_pretrained / from_local 便捷方法
# ---------------------------------------------------------------------------


def test_from_pretrained_from_local():
    """from_pretrained / from_local 便捷构造方法。"""
    with patch("verse_tokenizer.qwen._import_transformers", return_value=MockQwenTokenizer):
        tok1 = QwenTokenizer.from_pretrained()
        assert tok1._model_id == "Qwen/Qwen3-32B"

        tok2 = QwenTokenizer.from_pretrained("Qwen/Qwen3-14B")
        assert tok2._model_id == "Qwen/Qwen3-14B"

        tok3 = QwenTokenizer.from_local("/some/local/dir")
        assert tok3._tokenizer_dir == "/some/local/dir"
        assert tok3._model_id is None
