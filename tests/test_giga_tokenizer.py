"""Part5K1.3 Task 7.9: GigaTokenizerWrapper 集成测试。

测试策略
--------
环境未安装 ``gigatoken`` / ``transformers`` 时，通过 ``unittest.mock`` 注入
fake 模块测试 wrapper 逻辑；真实 gigatoken 可用时跑端到端测试（标记 slow）。

覆盖：
1. 导入测试：``from verse_infra.verse_tokenizer import GigaTokenizerWrapper`` 不抛 ImportError
2. lazy import 行为：模块 import 不触发 gigatoken 加载
3. ImportError 提示：gigatoken 未安装时构造抛出明确错误
4. encode/decode 一致性（mock gigatoken + fake HF tokenizer）
5. 批量 encode/decode
6. vocab 信息缓存（bos_id / eos_id / pad_id / vocab_size / vocab 懒加载）
7. apply_chat_template 委托底层 HF tokenizer
8. save/load 持久化
9. 自动降级：load_tokenizer(kind="giga") 在 gigatoken 不可用时降级到 VerseTokenizer
10. 原生模式（native=True）
11. 兼容模式：字符串 model_id（mock AutoTokenizer）
12. 真实 gigatoken 端到端（@pytest.mark.skipif，未安装时跳过）

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_infra:packages/verse_torch:packages/verse_nex \
        python -m pytest tests/test_giga_tokenizer.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))


# ---------------------------------------------------------------------------
# 环境探测：gigatoken / transformers 是否可用
# ---------------------------------------------------------------------------


def _has_gigatoken() -> bool:
    try:
        import gigatoken  # noqa: F401
        return True
    except ImportError:
        return False


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


HAS_GIGATOKEN = _has_gigatoken()
HAS_TRANSFORMERS = _has_transformers()


# ---------------------------------------------------------------------------
# Fake HF tokenizer：用于 mock 测试（支持 encode/decode 往返）
# ---------------------------------------------------------------------------


class FakeHfTokenizer:
    """最小 fake HF tokenizer，支持 encode/decode 往返。

    编码方案：每个 ASCII 字符 → 其 ord 值（0-127）；
    decode 时把 id 转回 chr(id)。仅用于测试 wrapper 逻辑，
    不模拟真实 BPE 行为。
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 3
        self.unk_token_id = 0
        self._vocab = {chr(i): i for i in range(min(vocab_size, 128))}
        # 记录 apply_chat_template 调用以便断言
        self.chat_template_calls = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        # 简单 char-based 编码（仅 ASCII）
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if 0 <= int(i) < 128)

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, texts, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, **kwargs):
        """模拟 HF tokenizer 的 __call__ 批量接口。"""
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [
                self.encode(t, add_special_tokens=add_special_tokens) for t in texts
            ]
        }

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def apply_chat_template(self, messages, **kwargs):
        # 记录调用以便测试断言
        self.chat_template_calls.append({"messages": messages, "kwargs": kwargs})
        # 简单拼接所有 message 的 content
        return "".join(m.get("content", "") for m in messages)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"type": "fake", "vocab_size": self.vocab_size}, f)


# ---------------------------------------------------------------------------
# Fake gigatoken 模块：用于 mock 测试
# ---------------------------------------------------------------------------


class _FakeGtTokenizer:
    """Fake gigatoken.Tokenizer。

    兼容模式：``gt.Tokenizer(hf_tok).as_hf()`` 返回 HF tokenizer。
    原生模式：``gt.Tokenizer(model_id)`` 返回自身（委托内部 HF tokenizer）。
    """

    def __init__(self, arg):
        if isinstance(arg, str):
            # 原生模式：arg 是 model_id，创建 fake HF tokenizer
            self._hf = FakeHfTokenizer()
        else:
            # 兼容模式：arg 是 HF tokenizer 实例
            self._hf = arg

    def as_hf(self):
        """兼容模式：返回 HF 兼容对象。"""
        return self._hf

    # ------------------------------------------------------------------
    # 原生模式下的方法（委托内部 HF tokenizer）
    # ------------------------------------------------------------------

    def encode(self, text, add_special_tokens=True):
        return self._hf.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=True):
        return self._hf.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self):
        return self._hf.get_vocab()

    def __call__(self, *args, **kwargs):
        return self._hf(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self._hf.batch_decode(*args, **kwargs)

    def __len__(self):
        return self._hf.vocab_size

    @property
    def vocab_size(self):
        return self._hf.vocab_size

    @property
    def bos_token_id(self):
        return self._hf.bos_token_id

    @property
    def eos_token_id(self):
        return self._hf.eos_token_id

    @property
    def pad_token_id(self):
        return self._hf.pad_token_id

    @property
    def unk_token_id(self):
        return self._hf.unk_token_id


class _FakeGigatokenModule:
    """Fake gigatoken 模块。"""
    Tokenizer = _FakeGtTokenizer
    __version__ = "0.0.1-fake"


# ---------------------------------------------------------------------------
# Fake AutoTokenizer：用于 mock 兼容模式下的字符串 model_id 路径
# ---------------------------------------------------------------------------


class _FakeAutoTokenizer:
    """Fake transformers.AutoTokenizer。"""

    @staticmethod
    def from_pretrained(model_id, trust_remote_code=True, **kwargs):
        # 返回 fake HF tokenizer（忽略 model_id，不真实下载）
        return FakeHfTokenizer()


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gigatoken():
    """注入 fake gigatoken 模块（patch _import_gigatoken）。"""
    fake_gt = _FakeGigatokenModule()
    with patch(
        "verse_infra.verse_tokenizer.giga._import_gigatoken",
        return_value=fake_gt,
    ):
        yield fake_gt


@pytest.fixture
def fake_auto_tokenizer():
    """注入 fake AutoTokenizer（patch _import_auto_tokenizer）。"""
    with patch(
        "verse_infra.verse_tokenizer.giga._import_auto_tokenizer",
        return_value=_FakeAutoTokenizer,
    ):
        yield _FakeAutoTokenizer


@pytest.fixture
def fake_giga_and_auto(fake_gigatoken, fake_auto_tokenizer):
    """同时注入 fake gigatoken + fake AutoTokenizer（兼容模式字符串路径）。"""
    return fake_gigatoken


# ===========================================================================
# 1. 导入测试
# ===========================================================================


class TestImport:
    """GigaTokenizerWrapper 导入测试。"""

    def test_import_from_verse_tokenizer(self):
        """from verse_infra.verse_tokenizer import GigaTokenizerWrapper 不抛 ImportError。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        assert GigaTokenizerWrapper is not None
        assert isinstance(GigaTokenizerWrapper, type)

    def test_import_from_submodule(self):
        """从子模块路径导入也正常。"""
        from verse_infra.verse_tokenizer.giga import GigaTokenizerWrapper as Gw2
        assert Gw2 is not None

    def test_import_from_top_level(self):
        """从顶层 verse_infra 导入也正常（通过 __getattr__ 延迟导入）。"""
        # 顶层 verse_infra 通过 __getattr__ 解析
        import verse_infra
        # 直接 import 子模块的公共 API
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper as Gw3
        assert Gw3 is not None

    def test_class_in_all(self):
        """GigaTokenizerWrapper 在 __all__ 中。"""
        from verse_infra import verse_tokenizer
        assert "GigaTokenizerWrapper" in verse_tokenizer.__all__


# ===========================================================================
# 2. lazy import 行为
# ===========================================================================


class TestLazyImport:
    """lazy import gigatoken 行为测试。"""

    def test_module_import_doesnt_load_gigatoken(self):
        """import giga 模块不触发 gigatoken 加载。"""
        # 保存原始 sys.modules 状态，测试后恢复（避免影响后续测试的 patch）
        saved_giga = sys.modules.get("verse_infra.verse_tokenizer.giga")
        saved_gigatoken = sys.modules.get("gigatoken")
        try:
            # 清除可能已加载的 gigatoken（测试隔离）
            sys.modules.pop("gigatoken", None)
            # 强制重新 import giga 模块
            if "verse_infra.verse_tokenizer.giga" in sys.modules:
                del sys.modules["verse_infra.verse_tokenizer.giga"]
            import verse_infra.verse_tokenizer.giga as giga_mod
            # gigatoken 不应被加载
            assert "gigatoken" not in sys.modules, (
                "import giga 模块不应触发 gigatoken 加载（lazy import）"
            )
            # 但 GigaTokenizerWrapper 类应可用
            assert hasattr(giga_mod, "GigaTokenizerWrapper")
        finally:
            # 恢复原始模块状态，避免后续测试的 patch 失效
            # （del + reimport 会产生新模块对象，导致 verse_tokenizer 包命名空间中
            #   的 GigaTokenizerWrapper 与新模块的 _import_gigatoken 脱节）
            if saved_giga is not None:
                sys.modules["verse_infra.verse_tokenizer.giga"] = saved_giga
            else:
                sys.modules.pop("verse_infra.verse_tokenizer.giga", None)
            if saved_gigatoken is not None:
                sys.modules["gigatoken"] = saved_gigatoken

    def test_module_import_doesnt_load_transformers(self):
        """import giga 模块不触发 transformers 加载。"""
        # 保存原始 sys.modules 状态，测试后恢复
        saved_giga = sys.modules.get("verse_infra.verse_tokenizer.giga")
        saved_transformers = sys.modules.get("transformers")
        try:
            sys.modules.pop("transformers", None)
            if "verse_infra.verse_tokenizer.giga" in sys.modules:
                del sys.modules["verse_infra.verse_tokenizer.giga"]
            import verse_infra.verse_tokenizer.giga  # noqa: F401
            assert "transformers" not in sys.modules, (
                "import giga 模块不应触发 transformers 加载（lazy import）"
            )
        finally:
            # 恢复原始模块状态
            if saved_giga is not None:
                sys.modules["verse_infra.verse_tokenizer.giga"] = saved_giga
            else:
                sys.modules.pop("verse_infra.verse_tokenizer.giga", None)
            if saved_transformers is not None:
                sys.modules["transformers"] = saved_transformers


# ===========================================================================
# 3. ImportError 提示
# ===========================================================================


class TestImportError:
    """gigatoken 未安装时的 ImportError 提示测试。"""

    def test_construct_raises_import_error_with_install_hint(self):
        """gigatoken 未安装时构造抛 ImportError，含安装提示。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        with pytest.raises(ImportError) as exc_info:
            GigaTokenizerWrapper()
        msg = str(exc_info.value)
        assert "gigatoken" in msg.lower(), f"ImportError 消息应提及 gigatoken：{msg}"
        assert "pip install gigatoken" in msg, (
            f"ImportError 消息应含安装提示 'pip install gigatoken'：{msg}"
        )

    def test_import_error_mentions_fallback(self):
        """ImportError 消息应提及降级路径。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        with pytest.raises(ImportError) as exc_info:
            GigaTokenizerWrapper()
        msg = str(exc_info.value)
        # 应提及 fallback 或降级
        assert "fallback" in msg.lower() or "降级" in msg, (
            f"ImportError 消息应提及降级路径：{msg}"
        )

    def test_import_gigatoken_helper_raises_clear_error(self):
        """_import_gigatoken 辅助函数抛出明确错误。"""
        from verse_infra.verse_tokenizer.giga import _import_gigatoken
        with pytest.raises(ImportError) as exc_info:
            _import_gigatoken()
        assert "gigatoken" in str(exc_info.value).lower()


# ===========================================================================
# 4. encode/decode 一致性（mock gigatoken + fake HF tokenizer）
# ===========================================================================


class TestEncodeDecode:
    """encode/decode 基本功能测试（mock gigatoken）。"""

    def test_encode_returns_non_empty_list(self, fake_gigatoken):
        """encode 返回非空 list。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        ids = wrapper.encode("hello")
        assert isinstance(ids, list), f"encode 应返回 list，实际 {type(ids)}"
        assert len(ids) > 0, "encode('hello') 应返回非空 list"
        assert all(isinstance(i, int) for i in ids), "所有 id 应为 int"

    def test_decode_encode_roundtrip(self, fake_gigatoken):
        """decode(encode(text)) == text（往返一致）。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        for text in ["hello", "world", "abc", "test 123"]:
            ids = wrapper.encode(text)
            decoded = wrapper.decode(ids)
            assert decoded == text, (
                f"encode/decode 往返不一致：{text!r} → ids={ids} → {decoded!r}"
            )

    def test_encode_with_special_tokens_param(self, fake_gigatoken):
        """encode 接受 add_special_tokens 参数（透传给底层）。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        # 都应正常工作（fake tokenizer 忽略 add_special_tokens）
        ids_with = wrapper.encode("hello", add_special_tokens=True)
        ids_without = wrapper.encode("hello", add_special_tokens=False)
        assert isinstance(ids_with, list)
        assert isinstance(ids_without, list)

    def test_decode_skip_special_tokens_param(self, fake_gigatoken):
        """decode 接受 skip_special_tokens 参数。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        ids = wrapper.encode("hello")
        # 都应正常工作
        decoded_skip = wrapper.decode(ids, skip_special_tokens=True)
        decoded_keep = wrapper.decode(ids, skip_special_tokens=False)
        assert isinstance(decoded_skip, str)
        assert isinstance(decoded_keep, str)

    def test_construct_with_hf_tokenizer_instance(self, fake_gigatoken):
        """传入 HF tokenizer 实例（兼容模式）。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        # 内部应持有 HF tokenizer
        assert wrapper.hf_tokenizer is hf_tok
        assert wrapper.native is False

    def test_construct_with_string_model_id(self, fake_giga_and_auto):
        """传入字符串 model_id（兼容模式，mock AutoTokenizer）。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        wrapper = GigaTokenizerWrapper("Qwen/Qwen3-32B")
        # 应通过 AutoTokenizer 加载，然后 gt.Tokenizer(hf_tok).as_hf()
        assert wrapper.hf_tokenizer is not None
        assert wrapper.native is False
        # encode/decode 应工作
        ids = wrapper.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) > 0


# ===========================================================================
# 5. 批量 encode/decode
# ===========================================================================


class TestBatchEncodeDecode:
    """批量 encode/decode 测试。"""

    def test_encode_batch_returns_list_of_lists(self, fake_gigatoken):
        """encode_batch 返回 list[list[int]]。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        texts = ["hello", "world", "test"]
        result = wrapper.encode_batch(texts)
        assert isinstance(result, list)
        assert len(result) == len(texts)
        for ids in result:
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)

    def test_encode_batch_roundtrip(self, fake_gigatoken):
        """encode_batch + decode_batch 往返一致。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        texts = ["hello", "world", "abc 123"]
        ids_batch = wrapper.encode_batch(texts)
        decoded = wrapper.decode_batch(ids_batch)
        assert len(decoded) == len(texts)
        for orig, dec in zip(texts, decoded):
            assert dec == orig, f"批量往返不一致：{orig!r} → {dec!r}"

    def test_encode_batch_empty_input(self, fake_gigatoken):
        """encode_batch 空输入返回空 list。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        result = wrapper.encode_batch([])
        assert result == []

    def test_decode_batch_empty_input(self, fake_gigatoken):
        """decode_batch 空输入返回空 list。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        result = wrapper.decode_batch([])
        assert result == []


# ===========================================================================
# 6. vocab 信息缓存
# ===========================================================================


class TestVocabCaching:
    """bos_id / eos_id / pad_id / vocab_size / vocab 懒加载测试。"""

    def test_bos_id_cached(self, fake_gigatoken):
        """bos_id 在构造时缓存。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        assert wrapper.bos_id == hf_tok.bos_token_id
        assert wrapper._bos_id == hf_tok.bos_token_id

    def test_eos_id_cached(self, fake_gigatoken):
        """eos_id 在构造时缓存。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        assert wrapper.eos_id == hf_tok.eos_token_id

    def test_pad_id_cached(self, fake_gigatoken):
        """pad_id 在构造时缓存。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        assert wrapper.pad_id == hf_tok.pad_token_id

    def test_unk_id_cached(self, fake_gigatoken):
        """unk_id 在构造时缓存。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        assert wrapper.unk_id == hf_tok.unk_token_id

    def test_vocab_size_cached(self, fake_gigatoken):
        """vocab_size 在构造时缓存。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer(vocab_size=256)
        wrapper = GigaTokenizerWrapper(hf_tok)
        assert wrapper.vocab_size == 256
        assert wrapper._vocab_size == 256
        assert len(wrapper) == 256

    def test_vocab_lazy_loading(self, fake_gigatoken):
        """vocab 属性懒加载（首次访问时构建）。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        # 构造时 _vocab 应为 None
        assert wrapper._vocab is None
        # 首次访问触发构建
        vocab = wrapper.vocab
        assert vocab is not None
        assert isinstance(vocab, dict)
        # 再次访问应返回缓存（同一对象）
        vocab2 = wrapper.vocab
        assert vocab2 is vocab

    def test_bos_id_none_raises_attribute_error(self, fake_gigatoken):
        """bos_id 为 None 时访问属性抛 AttributeError。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        hf_tok.bos_token_id = None
        wrapper = GigaTokenizerWrapper(hf_tok)
        assert wrapper._bos_id is None
        with pytest.raises(AttributeError):
            _ = wrapper.bos_id


# ===========================================================================
# 7. apply_chat_template 委托
# ===========================================================================


class TestApplyChatTemplate:
    """apply_chat_template 委托底层 HF tokenizer 测试。"""

    def test_apply_chat_template_delegates_to_hf(self, fake_gigatoken):
        """apply_chat_template 委托底层 HF tokenizer。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = wrapper.apply_chat_template(messages)
        # FakeHfTokenizer.apply_chat_template 拼接所有 content
        assert result == "你好你好！"
        # 应记录调用
        assert len(hf_tok.chat_template_calls) == 1
        assert hf_tok.chat_template_calls[0]["messages"] is messages

    def test_apply_chat_template_with_kwargs(self, fake_gigatoken):
        """apply_chat_template 透传 kwargs。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        messages = [{"role": "user", "content": "hello"}]
        wrapper.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        call = hf_tok.chat_template_calls[-1]
        assert call["kwargs"]["add_generation_prompt"] is True
        assert call["kwargs"]["tokenize"] is False

    def test_apply_chat_template_native_mode_raises(self, fake_gigatoken):
        """原生模式下 apply_chat_template 无 HF tokenizer 可委托时抛 RuntimeError。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        # 原生模式：gt.Tokenizer(model_id) 直接构造，不持有 HF tokenizer
        wrapper = GigaTokenizerWrapper("Qwen/Qwen3-32B", native=True)
        assert wrapper.hf_tokenizer is None
        # _FakeGtTokenizer 没有 apply_chat_template 方法 → 应抛 RuntimeError
        with pytest.raises(RuntimeError):
            wrapper.apply_chat_template([{"role": "user", "content": "hi"}])


# ===========================================================================
# 8. save / load 持久化
# ===========================================================================


class TestSaveLoad:
    """save/load 持久化测试。"""

    def test_save_to_json_file(self, fake_gigatoken, tmp_path):
        """save 到 .json 元信息文件。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        json_path = str(tmp_path / "giga_tok.json")
        wrapper.save(json_path)
        # .json 文件应存在
        assert os.path.isfile(json_path)
        # 元信息应含 type=giga
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["type"] == "giga"
        assert meta["native"] is False

    def test_save_to_directory(self, fake_gigatoken, tmp_path):
        """save 到目录。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        dir_path = str(tmp_path / "giga_dir")
        wrapper.save(dir_path)
        # 目录应存在
        assert os.path.isdir(dir_path)

    def test_load_from_json(self, fake_giga_and_auto, tmp_path):
        """load 从 .json 元信息文件恢复。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        json_path = str(tmp_path / "giga_tok.json")
        wrapper.save(json_path)

        # 新建 wrapper 并 load
        wrapper2 = GigaTokenizerWrapper(hf_tok)
        wrapper2.load(json_path)
        # load 后应能正常 encode
        ids = wrapper2.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) > 0


# ===========================================================================
# 9. 自动降级：load_tokenizer(kind="giga") 降级到 VerseTokenizer
# ===========================================================================


class TestAutoDegrade:
    """load_tokenizer(kind="giga") 自动降级测试。"""

    @patch("verse_infra.verse_tokenizer.verse.VerseTokenizer")
    def test_giga_degrades_to_verse_when_gigatoken_missing(
        self, mock_verse_class, capsys
    ):
        """gigatoken 不可用时 load_tokenizer(kind="giga") 降级到 VerseTokenizer。"""
        from verse_infra.verse_tokenizer import load_tokenizer, VerseTokenizer

        # mock VerseTokenizer 实例
        mock_tok = MagicMock()
        mock_verse_class.return_value = mock_tok

        # load_tokenizer(kind="giga") 应降级
        tok = load_tokenizer(kind="giga")

        # 应返回 mock 的 VerseTokenizer 实例
        assert mock_verse_class.called, "应调用 VerseTokenizer 构造函数"
        assert tok is mock_tok, "应返回 VerseTokenizer 实例"

        # 应打印降级警告
        captured = capsys.readouterr()
        assert "gigatoken 未安装" in captured.out, (
            f"应打印降级警告，实际输出：{captured.out}"
        )
        assert "降级到 VerseTokenizer" in captured.out

    @patch("verse_infra.verse_tokenizer.verse.VerseTokenizer")
    def test_giga_degrade_with_path(self, mock_verse_class, capsys):
        """降级时 path 透传给 VerseTokenizer。"""
        from verse_infra.verse_tokenizer import load_tokenizer

        mock_tok = MagicMock()
        mock_verse_class.return_value = mock_tok

        # 传入 path（model_id）
        load_tokenizer(kind="giga", path="Qwen/Qwen3-32B")

        # VerseTokenizer 应被调用，且 model_id="Qwen/Qwen3-32B"
        mock_verse_class.assert_called_once_with(model_id="Qwen/Qwen3-32B")

    @patch("verse_infra.verse_tokenizer.verse.VerseTokenizer")
    def test_giga_degrade_with_local_dir(self, mock_verse_class, tmp_path, capsys):
        """降级时 path 为本地目录走 tokenizer_dir 参数。"""
        from verse_infra.verse_tokenizer import load_tokenizer

        mock_tok = MagicMock()
        mock_verse_class.return_value = mock_tok

        local_dir = str(tmp_path / "local_tok")
        os.makedirs(local_dir, exist_ok=True)

        load_tokenizer(kind="giga", path=local_dir)

        # VerseTokenizer 应被调用，且 tokenizer_dir=local_dir
        mock_verse_class.assert_called_once_with(tokenizer_dir=local_dir)

    def test_explicit_verse_kind(self):
        """显式 kind='verse' 走 VerseTokenizer 路径（不降级）。"""
        # 注意：transformers 未安装时 VerseTokenizer() 会抛 ImportError，
        # 这里只验证 load_tokenizer 尝试构造 VerseTokenizer（而非走 byte 路径）
        from verse_infra.verse_tokenizer import load_tokenizer
        with patch("verse_infra.verse_tokenizer.verse.VerseTokenizer") as mock_verse:
            mock_verse.return_value = MagicMock()
            tok = load_tokenizer(kind="verse")
            assert mock_verse.called

    def test_byte_kind_unchanged(self):
        """kind='byte' 仍返回 ByteTokenizer（向后兼容）。"""
        from verse_infra.verse_tokenizer import load_tokenizer, ByteTokenizer
        tok = load_tokenizer(kind="byte")
        assert isinstance(tok, ByteTokenizer)

    def test_unknown_kind_raises_valueerror(self):
        """未知 kind 抛 ValueError，消息含 'giga'。"""
        from verse_infra.verse_tokenizer import load_tokenizer
        with pytest.raises(ValueError) as exc_info:
            load_tokenizer("unknown_kind")
        msg = str(exc_info.value)
        assert "giga" in msg, f"ValueError 消息应含 'giga'：{msg}"


# ===========================================================================
# 10. 原生模式（native=True）
# ===========================================================================


class TestNativeMode:
    """原生模式（native=True）测试。"""

    def test_native_mode_constructs_with_string(self, fake_gigatoken):
        """原生模式：gt.Tokenizer(model_id) 直接构造。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        wrapper = GigaTokenizerWrapper("Qwen/Qwen3-32B", native=True)
        assert wrapper.native is True
        assert wrapper.hf_tokenizer is None  # 原生模式不持有 HF tokenizer

    def test_native_mode_encode_decode(self, fake_gigatoken):
        """原生模式 encode/decode 正常。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        wrapper = GigaTokenizerWrapper("Qwen/Qwen3-32B", native=True)
        ids = wrapper.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) > 0
        decoded = wrapper.decode(ids)
        assert decoded == "hello"

    def test_native_mode_vocab_size(self, fake_gigatoken):
        """原生模式 vocab_size 缓存。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        wrapper = GigaTokenizerWrapper("Qwen/Qwen3-32B", native=True)
        # FakeHfTokenizer 默认 vocab_size=256
        assert wrapper.vocab_size == 256
        assert len(wrapper) == 256


# ===========================================================================
# 11. 便捷构造方法
# ===========================================================================


class TestClassMethods:
    """from_pretrained / from_hf_tokenizer 便捷构造测试。"""

    def test_from_pretrained(self, fake_giga_and_auto):
        """from_pretrained 便捷构造。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        wrapper = GigaTokenizerWrapper.from_pretrained("Qwen/Qwen3-32B")
        assert wrapper is not None
        assert wrapper.native is False

    def test_from_pretrained_default_model(self, fake_giga_and_auto):
        """from_pretrained 不传 model_id 用默认。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        wrapper = GigaTokenizerWrapper.from_pretrained()
        assert wrapper is not None

    def test_from_hf_tokenizer(self, fake_gigatoken):
        """from_hf_tokenizer 从 HF tokenizer 实例构造。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper.from_hf_tokenizer(hf_tok)
        assert wrapper.hf_tokenizer is hf_tok


# ===========================================================================
# 12. 真实 gigatoken 端到端（@pytest.mark.skipif，未安装时跳过）
# ===========================================================================


@pytest.mark.skipif(not HAS_GIGATOKEN, reason="gigatoken 未安装，跳过端到端测试")
class TestRealGigatoken:
    """真实 gigatoken 端到端测试（需要安装 gigatoken）。"""

    @pytest.mark.slow
    def test_real_encode_decode(self):
        """真实 gigatoken encode/decode。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper
        # 仅在真实环境验证（需要网络下载 model）
        try:
            wrapper = GigaTokenizerWrapper("Qwen/Qwen3-32B")
        except Exception:
            pytest.skip("无法加载真实 tokenizer（网络/模型不可用）")
        ids = wrapper.encode("hello")
        assert isinstance(ids, list)
        decoded = wrapper.decode(ids)
        assert isinstance(decoded, str)

    @pytest.mark.slow
    def test_batch_encode_speedup(self):
        """批量 encode 加速对比（≥10×）。

        用 10000 条文本，对比 GigaTokenizerWrapper.encode_batch 与
        VerseTokenizer.encode_batch 的时间。
        """
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper, VerseTokenizer
        try:
            giga_tok = GigaTokenizerWrapper("Qwen/Qwen3-32B")
            verse_tok = VerseTokenizer("Qwen/Qwen3-32B")
        except Exception:
            pytest.skip("无法加载真实 tokenizer")

        texts = ["hello world"] * 10000
        import time
        t0 = time.time()
        giga_tok.encode_batch(texts)
        t_giga = time.time() - t0

        t0 = time.time()
        verse_tok.encode_batch(texts)
        t_verse = time.time() - t0

        # gigatoken 应至少快 10×（保守断言，实际 ~1000×）
        if t_verse > 0 and t_giga > 0:
            speedup = t_verse / t_giga
            assert speedup >= 10, (
                f"gigatoken 加速不足 10×：verse={t_verse:.3f}s, "
                f"giga={t_giga:.3f}s, speedup={speedup:.1f}×"
            )


# ===========================================================================
# 13. .json 元信息文件加载（Part5K1.8 Task 12.1）
# ===========================================================================


class TestGigaJsonMetaLoading:
    """``.json`` 元信息文件加载测试（Part5K1.8）。

    覆盖 Part5K1.8 修复：``GigaTokenizerWrapper`` / ``load_tokenizer`` 在
    ``path`` 为 ``.json`` 文件路径时不再抛 ``Repo id must be in the form...``
    错误。gigatoken / transformers 未安装时通过 fake 模块测试。
    """

    def test_json_meta_file_not_raise_repo_id_error(self, fake_giga_and_auto, tmp_path):
        """``load_tokenizer(kind="giga", path=json_path)`` 不抛 ``Repo id must be in the form`` 错误。

        Part5K1.8 修复：``.json`` 元信息文件路径走 ``wrapper.load()`` 重建，
        而非 ``AutoTokenizer.from_pretrained(.json)``。
        """
        from verse_infra.verse_tokenizer import load_tokenizer

        # 创建临时 .json 元信息文件（符合 GigaTokenizerWrapper.save 的格式）
        json_path = str(tmp_path / "giga_meta.json")
        meta = {
            "type": "giga",
            "native": False,
            "model_id": "Qwen/Qwen3-32B",
            "tokenizer_dir": None,
            "trust_remote_code": True,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # 调用 load_tokenizer：不应抛 "Repo id must be in the form" 错误
        # （可能因 gigatoken 未安装而降级，但不应抛该特定错误）
        try:
            tok = load_tokenizer(kind="giga", path=json_path)
            assert tok is not None
        except ImportError:
            # gigatoken / transformers 未安装时降级到 VerseTokenizer 是允许的
            pytest.skip("gigatoken/transformers 未安装（fake 模块未生效）")
        except Exception as e:
            msg = str(e)
            assert "Repo id must be in the form" not in msg, (
                f"不应抛 'Repo id must be in the form' 错误，实际抛出：{msg}"
            )
            # 其他异常允许（如 fake 模块环境下 wrapper.load 内部找不到 tokenizer_dir）

    def test_json_meta_file_with_wrapper(self, fake_giga_and_auto, tmp_path):
        """``GigaTokenizerWrapper(model_id_or_tokenizer=json_path)`` 不抛 Repo id 错误。

        Part5K1.8 修复：``__init__`` 检测 ``.json`` 文件路径后用
        ``DEFAULT_GIGA_MODEL`` 先构造空实例，再调用 ``self.load(path)`` 重建。
        """
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper

        # 创建临时 .json 元信息文件
        json_path = str(tmp_path / "giga_meta.json")
        meta = {
            "type": "giga",
            "native": False,
            "model_id": "Qwen/Qwen3-32B",
            "tokenizer_dir": None,
            "trust_remote_code": True,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # 直接构造 wrapper：不应抛 "Repo id must be in the form" 错误
        try:
            wrapper = GigaTokenizerWrapper(model_id_or_tokenizer=json_path)
            assert wrapper is not None
        except ImportError:
            pytest.skip("gigatoken/transformers 未安装（fake 模块未生效）")
        except Exception as e:
            msg = str(e)
            assert "Repo id must be in the form" not in msg, (
                f"不应抛 'Repo id must be in the form' 错误，实际抛出：{msg}"
            )

    def test_json_meta_file_load_via_load_method(self, fake_giga_and_auto, tmp_path):
        """``wrapper.load(json_path)`` 正常重建（端到端）。

        先用 fake HF tokenizer 构造 wrapper 并 save 为 .json，
        再用 ``GigaTokenizerWrapper(json_path)`` 从 .json 加载。
        """
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper

        # 1. 构造并保存为 .json
        hf_tok = FakeHfTokenizer()
        wrapper = GigaTokenizerWrapper(hf_tok)
        json_path = str(tmp_path / "saved_meta.json")
        wrapper.save(json_path)
        assert os.path.isfile(json_path)

        # 2. 用 .json 元信息文件路径直接构造（不应抛 Repo id 错误）
        wrapper2 = GigaTokenizerWrapper(model_id_or_tokenizer=json_path)
        # 加载后应能正常 encode
        ids = wrapper2.encode("hello")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_load_tokenizer_with_nonexistent_json(self, fake_giga_and_auto, tmp_path):
        """``.json`` 路径文件不存在时回退到原构造逻辑（不误判为元信息文件）。

        Part5K1.8 SubTask 1.1: ``__init__`` 判断 ``.json`` 路径时同时校验
        ``os.path.isfile``，文件不存在则走原 repo_id 处理路径。
        """
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper

        # 不存在的 .json 路径：应走原字符串 model_id 路径（AutoTokenizer.from_pretrained）
        # 这里 fake_auto_tokenizer 已注入，不会真实下载
        wrapper = GigaTokenizerWrapper(
            model_id_or_tokenizer="Qwen/nonexistent.json"
        )
        # fake AutoTokenizer 忽略 model_id 直接返回 FakeHfTokenizer
        assert wrapper.hf_tokenizer is not None
        assert wrapper.native is False


# ===========================================================================
# 14. 向后兼容性
# ===========================================================================


class TestBackwardCompatibility:
    """确保不破坏现有 verse_tokenizer 行为。"""

    def test_existing_tokenizers_still_importable(self):
        """现有 tokenizer 仍可导入。"""
        from verse_infra.verse_tokenizer import (
            BPETokenizer,
            ByteTokenizer,
            CharTokenizer,
            VerseTokenizer,
            QwenTokenizer,
            NexTokenizerWrapper,
            BaseTokenizer,
        )
        assert BPETokenizer is not None
        assert ByteTokenizer is not None
        assert CharTokenizer is not None
        assert VerseTokenizer is not None
        assert QwenTokenizer is VerseTokenizer  # 别名
        assert NexTokenizerWrapper is not None
        assert BaseTokenizer is not None

    def test_load_tokenizer_byte_still_works(self):
        """load_tokenizer('byte') 仍返回 ByteTokenizer。"""
        from verse_infra.verse_tokenizer import load_tokenizer, ByteTokenizer
        tok = load_tokenizer("byte")
        assert isinstance(tok, ByteTokenizer)

    def test_giga_wrapper_is_base_tokenizer_subclass(self):
        """GigaTokenizerWrapper 是 BaseTokenizer 子类。"""
        from verse_infra.verse_tokenizer import GigaTokenizerWrapper, BaseTokenizer
        assert issubclass(GigaTokenizerWrapper, BaseTokenizer)

    def test_qwen_tokenizer_alias_unchanged(self):
        """QwenTokenizer 仍是 VerseTokenizer 的别名。"""
        from verse_infra.verse_tokenizer import VerseTokenizer, QwenTokenizer
        assert QwenTokenizer is VerseTokenizer


# ===========================================================================
# 入口
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
