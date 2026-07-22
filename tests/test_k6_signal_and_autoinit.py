"""Part4K2.6 Task 2 + Task 3：Ctrl+C 强制退出 + 资源自动初始化 单元测试。

覆盖：
Task 2（Ctrl+C 强制退出 + 紧急保存）:
1. 信号处理器注册和恢复（install_signal_handlers / _restore_signal_handlers）
2. set_emergency_save_fn / clear_emergency_save_fn
3. 紧急保存函数被调用（mock os._exit，验证 _signal_handler 调用 save 后强制退出）
4. 无紧急保存函数时信号处理器仍正常退出
5. train() 注册和清除紧急保存
6. Trainer.fit 注册和清除更精确的紧急保存

Task 3（资源自动初始化）:
7. _auto_build_tokenizer byte kind
8. _auto_build_tokenizer char kind（保存 tokenizer.json）
9. _auto_build_tokenizer 未知 kind 降级 byte
10. _auto_generate_test_data 生成文件存在且格式正确
11. _is_test_config 小配置返回 True
12. _is_test_config 大配置返回 False
13. _load_tokenizer 自动构建（tokenizer 不存在时）
14. train() 数据自动生成（小配置 + 数据不存在时）
15. train() 大配置数据不存在时报错

运行方式：
    cd /workspace && python -m pytest tests/test_k6_signal_and_autoinit.py -x -q
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch / verse_infra
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_infra"))

from verse_torch import Tensor, Linear, SGD
from verse_torch.training import Trainer

import verse_infra.verse_trainer.trainer as _trainer_mod
from verse_infra.verse_trainer.trainer import (
    train,
    install_signal_handlers,
    reset_shutdown_flag,
    is_shutdown_requested,
    set_emergency_save_fn,
    clear_emergency_save_fn,
    _restore_signal_handlers,
    _signal_handler,
    _auto_build_tokenizer,
    _load_tokenizer,
    _is_test_config,
    _auto_generate_test_data,
    _TEST_TEXTS,
)


# ---------------------------------------------------------------------------
# 通用辅助：构造最小 config.yml
# ---------------------------------------------------------------------------


def _make_tiny_config(tmpdir, max_steps=10, vocab_size=259):
    """构造最小 config.yml（vocab=259, n_layer=2, n_embd=32, seq_len=16）。

    预先把 ByteTokenizer 保存到 <tmpdir>/checkpoints/tokenizer.json。
    """
    from verse_infra.verse_tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    ckpt_dir = os.path.join(tmpdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tok.save(os.path.join(ckpt_dir, "tokenizer.json"))

    config = {
        "model": {
            "arch": "versenex",
            "vocab_size": vocab_size,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 32,
            "seq_len": 16,
            "dropout": 0.0,
            "n_kv_head": 2,
            "tie_weights": True,
            "window_size": 8,
            "num_global_tokens": 2,
            "use_alibi": True,
            "use_rope": False,
        },
        "training": {
            "batch_size": 2,
            "lr": 0.003,
            "weight_decay": 0.0,
            "no_decay": False,
            "grad_clip": 1.0,
            "label_smoothing": 0.0,
            "max_steps": max_steps,
            "warmup": 2,
            "eval_interval": max_steps,
            "patience": 5,
            "grad_accum": 1,
            "log_interval": max_steps,
            "seed": 42,
            "enable_progress_bar": False,
            "realtime_plot": False,
            "eta_window": 5,
            "parallel_chunks": 1,
        },
        "tokenizer": {"kind": "byte"},
        "data": {
            "train_path": "data/train.jsonl",
            "val_path": "data/val.jsonl",
        },
        "checkpoint": {"save_dir": "checkpoints"},
    }
    cfg_path = os.path.join(tmpdir, "config.yml")
    import yaml
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return cfg_path


def _make_large_config(tmpdir):
    """构造大模型 config.yml（vocab=50000, n_layer=12, n_embd=512）。

    不保存数据文件，用于验证大配置数据缺失时报错。
    """
    config = {
        "model": {
            "arch": "versenex",
            "vocab_size": 50000,
            "n_layer": 12,
            "n_head": 8,
            "n_embd": 512,
            "seq_len": 128,
            "dropout": 0.0,
            "n_kv_head": 4,
            "tie_weights": True,
            "window_size": 64,
            "num_global_tokens": 16,
            "use_alibi": True,
            "use_rope": False,
        },
        "training": {
            "batch_size": 4,
            "lr": 0.003,
            "weight_decay": 0.0,
            "no_decay": False,
            "grad_clip": 1.0,
            "label_smoothing": 0.0,
            "max_steps": 5,
            "warmup": 2,
            "eval_interval": 5,
            "patience": 5,
            "grad_accum": 1,
            "log_interval": 100,
            "seed": 42,
            "enable_progress_bar": False,
            "realtime_plot": False,
            "eta_window": 5,
            "parallel_chunks": 1,
        },
        "tokenizer": {"kind": "byte"},
        "data": {
            "train_path": "data/train.jsonl",
            "val_path": "data/val.jsonl",
        },
        "checkpoint": {"save_dir": "checkpoints"},
    }
    cfg_path = os.path.join(tmpdir, "config.yml")
    import yaml
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return cfg_path


@pytest.fixture
def reset_signal_state():
    """保存/恢复信号模块的全局状态，避免测试间互相干扰。"""
    saved_installed = _trainer_mod._signal_handlers_installed
    saved_originals = dict(_trainer_mod._original_signal_handlers)
    saved_emg = _trainer_mod._emergency_save_fn
    saved_int = signal.getsignal(signal.SIGINT)
    saved_term = signal.getsignal(signal.SIGTERM)
    yield
    _trainer_mod._signal_handlers_installed = saved_installed
    _trainer_mod._original_signal_handlers = saved_originals
    _trainer_mod._emergency_save_fn = saved_emg
    try:
        signal.signal(signal.SIGINT, saved_int)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, saved_term)
    except Exception:
        pass


# ===========================================================================
# Task 2: Ctrl+C 强制退出 + 紧急保存
# ===========================================================================


class TestSignalHandlers:
    """信号处理器 + 紧急保存测试。"""

    def test_set_and_clear_emergency_save_fn(self, reset_signal_state):
        """set_emergency_save_fn / clear_emergency_save_fn 基本功能。"""
        assert _trainer_mod._emergency_save_fn is None

        marker = {"called": False}

        def my_save():
            marker["called"] = True

        set_emergency_save_fn(my_save)
        assert _trainer_mod._emergency_save_fn is my_save

        clear_emergency_save_fn()
        assert _trainer_mod._emergency_save_fn is None
        # clear 后 marker 未被调用
        assert marker["called"] is False

    def test_signal_handler_calls_emergency_save_and_exit(self, reset_signal_state):
        """收到信号时调用紧急保存函数并 os._exit(0)（mock os._exit）。"""
        called = {"save": False}

        def my_save():
            called["save"] = True

        set_emergency_save_fn(my_save)
        # 确保恢复路径是 no-op（不触碰真实信号）
        _trainer_mod._signal_handlers_installed = False
        _trainer_mod._original_signal_handlers = {}

        with patch.object(_trainer_mod.os, "_exit") as mock_exit:
            _signal_handler(signal.SIGINT, None)

        assert called["save"] is True, "紧急保存函数应被调用"
        mock_exit.assert_called_once_with(0)

    def test_signal_handler_without_save_fn_still_exits(self, reset_signal_state):
        """无紧急保存函数时，信号处理器仍应 os._exit(0)。"""
        clear_emergency_save_fn()
        assert _trainer_mod._emergency_save_fn is None
        _trainer_mod._signal_handlers_installed = False
        _trainer_mod._original_signal_handlers = {}

        with patch.object(_trainer_mod.os, "_exit") as mock_exit:
            _signal_handler(signal.SIGTERM, None)

        mock_exit.assert_called_once_with(0)

    def test_restore_signal_handlers_noop_when_not_installed(self, reset_signal_state):
        """_restore_signal_handlers 在未安装时为 no-op。"""
        _trainer_mod._signal_handlers_installed = False
        _trainer_mod._original_signal_handlers = {}
        _restore_signal_handlers()
        assert _trainer_mod._signal_handlers_installed is False

    def test_restore_signal_handlers_resets_flag(self, reset_signal_state):
        """_restore_signal_handlers 恢复后 _signal_handlers_installed 置 False。"""
        # 模拟已安装状态（使用 SIGUSR1，安全可注册）
        _trainer_mod._original_signal_handlers = {signal.SIGUSR1: signal.SIG_DFL}
        _trainer_mod._signal_handlers_installed = True
        try:
            _restore_signal_handlers()
            assert _trainer_mod._signal_handlers_installed is False
        finally:
            _trainer_mod._original_signal_handlers = {}

    def test_install_and_restore_signal_handlers_roundtrip(self, reset_signal_state):
        """install_signal_handlers + _restore_signal_handlers 往返。"""
        _trainer_mod._signal_handlers_installed = False
        _trainer_mod._original_signal_handlers = {}
        install_signal_handlers()
        # 主线程应安装成功（CI 非主线程时为 False，此处宽松断言）
        if _trainer_mod._signal_handlers_installed:
            _restore_signal_handlers()
            assert _trainer_mod._signal_handlers_installed is False


class TestTrainEmergencySave:
    """train() 注册和清除紧急保存。"""

    def test_train_registers_and_clears_emergency_save(self, tmp_path,
                                                       reset_signal_state):
        """train() 运行期间注册紧急保存，返回前清除。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=4)

        set_calls = []
        real_set = _trainer_mod.set_emergency_save_fn

        def spy_set(fn):
            set_calls.append(fn)
            real_set(fn)

        assert _trainer_mod._emergency_save_fn is None
        with patch.object(_trainer_mod, "set_emergency_save_fn", spy_set):
            result = train(
                config_path=cfg_path,
                base_dir=str(tmp_path),
                single_sample={"prompt": "1+1=", "completion": "2"},
                max_steps_override=4,
                eval_after=False,
            )
        # train() 应注册过紧急保存函数
        assert len(set_calls) >= 1, "train() 应调用 set_emergency_save_fn"
        assert callable(set_calls[0])
        # train() 返回后应已清除
        assert _trainer_mod._emergency_save_fn is None
        assert "best_val_loss" in result


class TestTrainerFitEmergencySave:
    """Trainer.fit 注册和清除更精确的紧急保存（Task 2.3）。"""

    def test_trainer_fit_registers_and_clears_emergency_save(self, tmp_path,
                                                             reset_signal_state):
        """Trainer.fit 开头注册紧急保存，结尾清除。"""
        np.random.seed(0)
        dim_in, vocab = 4, 4
        rng = np.random.RandomState(0)
        X = rng.randn(32, dim_in).astype(np.float32)
        y = rng.randint(0, vocab, size=32).astype(np.int64)

        def make_batches(arr_x, arr_y, bs=8):
            return [
                (Tensor(arr_x[i:i + bs]), Tensor(arr_y[i:i + bs]))
                for i in range(0, len(arr_x), bs)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 6,
            "eval_interval": 3,
            "patience": 10,
            "save_dir": str(tmp_path),
            "label_smoothing": 0.0,
            "enable_progress_bar": False,
            "realtime_plot": False,
            "log_interval": 1000,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)

        set_calls = []
        clear_calls = []
        real_set = _trainer_mod.set_emergency_save_fn
        real_clear = _trainer_mod.clear_emergency_save_fn

        def spy_set(fn):
            set_calls.append(fn)
            real_set(fn)

        def spy_clear():
            clear_calls.append(True)
            real_clear()

        clear_emergency_save_fn()
        with patch.object(_trainer_mod, "set_emergency_save_fn", spy_set), \
                patch.object(_trainer_mod, "clear_emergency_save_fn", spy_clear):
            trainer.fit()

        assert len(set_calls) == 1, "Trainer.fit 应注册一次紧急保存"
        assert callable(set_calls[0])
        assert len(clear_calls) == 1, "Trainer.fit 结尾应清除紧急保存"
        assert _trainer_mod._emergency_save_fn is None

    def test_trainer_fit_emergency_save_captures_latest_step(self, tmp_path,
                                                             reset_signal_state):
        """紧急保存函数能读取最新 step（通过 dict 持有）。"""
        np.random.seed(0)
        dim_in, vocab = 4, 4
        rng = np.random.RandomState(0)
        X = rng.randn(32, dim_in).astype(np.float32)
        y = rng.randint(0, vocab, size=32).astype(np.int64)

        def make_batches(arr_x, arr_y, bs=8):
            return [
                (Tensor(arr_x[i:i + bs]), Tensor(arr_y[i:i + bs]))
                for i in range(0, len(arr_x), bs)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 8,
            "eval_interval": 4,
            "patience": 10,
            "save_dir": str(tmp_path),
            "enable_progress_bar": False,
            "realtime_plot": False,
            "log_interval": 1000,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)

        captured_fn = []

        def spy_set(fn):
            captured_fn.append(fn)

        with patch.object(_trainer_mod, "set_emergency_save_fn", spy_set):
            trainer.fit()

        assert len(captured_fn) == 1
        # 调用捕获的紧急保存函数，应不抛异常（保存最新 checkpoint）
        emg_fn = captured_fn[0]
        # 应能正常执行（save_last + model.save）
        emg_fn()
        # last.pt 应被紧急保存函数创建
        assert os.path.exists(os.path.join(str(tmp_path), "last.pt"))


# ===========================================================================
# Task 3: 资源自动初始化
# ===========================================================================


class TestAutoBuildTokenizer:
    """_auto_build_tokenizer 测试。"""

    def test_byte_kind(self, tmp_path):
        """byte kind 返回 ByteTokenizer。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = _auto_build_tokenizer("byte", str(tmp_path))
        assert isinstance(tok, ByteTokenizer)
        # byte tokenizer vocab=259
        assert len(tok) == 259

    def test_bytes_kind_alias(self, tmp_path):
        """bytes 是 byte 的别名。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = _auto_build_tokenizer("bytes", str(tmp_path))
        assert isinstance(tok, ByteTokenizer)

    def test_char_kind(self, tmp_path):
        """char kind 构建 CharTokenizer 并保存 tokenizer.json。"""
        from verse_infra.verse_tokenizer import CharTokenizer
        save_dir = str(tmp_path)
        tok = _auto_build_tokenizer("char", save_dir)
        assert isinstance(tok, CharTokenizer)
        # 应保存 tokenizer.json
        assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))

    def test_charlevel_kind_alias(self, tmp_path):
        """charlevel 是 char 的别名。"""
        from verse_infra.verse_tokenizer import CharTokenizer
        tok = _auto_build_tokenizer("charlevel", str(tmp_path))
        assert isinstance(tok, CharTokenizer)

    def test_unknown_kind_fallback_byte(self, tmp_path):
        """未知 kind 降级为 byte。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = _auto_build_tokenizer("unknown_kind_xyz", str(tmp_path))
        assert isinstance(tok, ByteTokenizer)

    def test_bpe_kind_fallback_byte(self, tmp_path):
        """bpe kind（无训练语料）降级为 byte。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = _auto_build_tokenizer("bpe", str(tmp_path))
        assert isinstance(tok, ByteTokenizer)


class TestAutoGenerateTestData:
    """_auto_generate_test_data 测试。"""

    def test_generates_files_with_correct_format(self, tmp_path):
        """生成的文件存在且格式正确（JSONL，每行 {"text": ...}）。"""
        train_path = str(tmp_path / "data" / "train.jsonl")
        val_path = str(tmp_path / "data" / "val.jsonl")
        assert not os.path.exists(train_path)

        _auto_generate_test_data(train_path, val_path)

        assert os.path.exists(train_path)
        assert os.path.exists(val_path)

        # 训练数据 100 条（20 * 5）
        with open(train_path, "r", encoding="utf-8") as f:
            train_lines = [json.loads(line) for line in f if line.strip()]
        assert len(train_lines) == 100
        assert "text" in train_lines[0]
        assert isinstance(train_lines[0]["text"], str)

        # 验证数据 5 条
        with open(val_path, "r", encoding="utf-8") as f:
            val_lines = [json.loads(line) for line in f if line.strip()]
        assert len(val_lines) == 5
        assert "text" in val_lines[0]

    def test_creates_parent_directories(self, tmp_path):
        """自动创建父目录。"""
        train_path = str(tmp_path / "a" / "b" / "train.jsonl")
        val_path = str(tmp_path / "a" / "b" / "val.jsonl")
        _auto_generate_test_data(train_path, val_path)
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)

    def test_test_texts_nonempty(self):
        """_TEST_TEXTS 模板非空且含中英文。"""
        assert len(_TEST_TEXTS) == 20
        assert any("你好" in t for t in _TEST_TEXTS)
        assert any("Hello" in t for t in _TEST_TEXTS)


class TestIsTestConfig:
    """_is_test_config 测试。"""

    def test_small_vocab_returns_true(self):
        """vocab_size <= 1000 返回 True。"""
        assert _is_test_config({"vocab_size": 256, "n_embd": 512, "n_layer": 12}, {}) is True

    def test_small_embd_returns_true(self):
        """n_embd <= 128 返回 True。"""
        assert _is_test_config({"vocab_size": 50000, "n_embd": 64, "n_layer": 12}, {}) is True

    def test_small_layer_returns_true(self):
        """n_layer <= 4 返回 True。"""
        assert _is_test_config({"vocab_size": 50000, "n_embd": 512, "n_layer": 2}, {}) is True

    def test_large_config_returns_false(self):
        """大配置（全部超过阈值）返回 False。"""
        cfg = {"vocab_size": 50000, "n_embd": 512, "n_layer": 12}
        assert _is_test_config(cfg, {}) is False

    def test_empty_config_returns_false(self):
        """空配置返回 False。"""
        assert _is_test_config({}, {}) is False

    def test_cometspark_small_yml_is_test(self):
        """cometspark_v05_small.yml 判定为测试配置。"""
        # 模拟 spark/config/cometspark_v05_small.yml 的 model 段
        small_cfg = {
            "vocab_size": 256,
            "n_layer": 2,
            "n_embd": 64,
        }
        assert _is_test_config(small_cfg, {}) is True


class TestLoadTokenizerAutoBuild:
    """_load_tokenizer 自动构建测试。"""

    def test_auto_build_when_file_missing_char(self, tmp_path):
        """tokenizer 文件不存在 + kind=char 时自动构建。"""
        save_dir = str(tmp_path / "ckpt")
        os.makedirs(save_dir)
        base_dir = str(tmp_path)  # base_dir 下也没有 tokenizer.json
        # 确保没有现成 tokenizer
        assert not os.path.exists(os.path.join(save_dir, "tokenizer.json"))
        assert not os.path.exists(os.path.join(base_dir, "tokenizer.json"))

        tok = _load_tokenizer({"kind": "char"}, base_dir, save_dir)
        from verse_infra.verse_tokenizer import CharTokenizer
        assert isinstance(tok, CharTokenizer)
        # 自动构建应保存 tokenizer.json
        assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))

    def test_auto_build_when_file_missing_unknown_kind(self, tmp_path):
        """tokenizer 文件不存在 + 未知 kind 时降级 byte。"""
        save_dir = str(tmp_path / "ckpt")
        os.makedirs(save_dir)
        base_dir = str(tmp_path)
        tok = _load_tokenizer({"kind": "bpe"}, base_dir, save_dir)
        from verse_infra.verse_tokenizer import ByteTokenizer
        assert isinstance(tok, ByteTokenizer)

    def test_byte_kind_no_file_no_build(self, tmp_path):
        """byte kind 无文件时直接构造（不触发自动构建逻辑）。"""
        save_dir = str(tmp_path / "ckpt")
        os.makedirs(save_dir)
        base_dir = str(tmp_path)
        tok = _load_tokenizer({"kind": "byte"}, base_dir, save_dir)
        from verse_infra.verse_tokenizer import ByteTokenizer
        assert isinstance(tok, ByteTokenizer)
        # byte 不需要保存文件
        assert not os.path.exists(os.path.join(save_dir, "tokenizer.json"))

    def test_load_existing_tokenizer_file(self, tmp_path):
        """tokenizer.json 已存在时直接加载。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        save_dir = str(tmp_path / "ckpt")
        os.makedirs(save_dir)
        # 预存一个 byte tokenizer
        ByteTokenizer().save(os.path.join(save_dir, "tokenizer.json"))
        base_dir = str(tmp_path)
        tok = _load_tokenizer({"kind": "byte"}, base_dir, save_dir)
        assert len(tok) == 259


class TestTrainAutoInit:
    """train() 资源自动初始化集成测试。"""

    def test_train_auto_generates_data_small_config(self, tmp_path, reset_signal_state):
        """小配置 + 数据文件不存在时自动生成测试数据并完成训练。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=4)
        train_path = tmp_path / "data" / "train.jsonl"
        val_path = tmp_path / "data" / "val.jsonl"
        assert not train_path.exists()

        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            max_steps_override=4,
            eval_after=False,
        )

        # 数据文件应被自动生成
        assert train_path.exists(), "训练数据应被自动生成"
        assert val_path.exists(), "验证数据应被自动生成"
        # 训练应成功完成
        assert "best_val_loss" in result
        assert len(result["train_losses"]) == 4
        # 生成的数据格式正确
        with open(train_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 100
        assert "text" in lines[0]

    def test_train_large_config_missing_data_raises(self, tmp_path, reset_signal_state):
        """大配置 + 数据文件不存在时报 FileNotFoundError。"""
        cfg_path = _make_large_config(str(tmp_path))
        train_path = tmp_path / "data" / "train.jsonl"
        assert not train_path.exists()

        with pytest.raises(FileNotFoundError, match="训练数据文件不存在"):
            train(
                config_path=cfg_path,
                base_dir=str(tmp_path),
                max_steps_override=4,
                eval_after=False,
            )
        # 大配置不应自动生成数据
        assert not train_path.exists()

    def test_train_clears_emergency_save_after_auto_data(self, tmp_path,
                                                        reset_signal_state):
        """自动生成数据路径下 train() 结束后紧急保存函数已清除。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        assert _trainer_mod._emergency_save_fn is None
        train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            max_steps_override=3,
            eval_after=False,
        )
        assert _trainer_mod._emergency_save_fn is None
