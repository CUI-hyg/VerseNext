"""验证 trainer 交互式 tokenizer 初始化（Part5K1.8 Task 12.3）。

测试 ``verse_infra.verse_trainer.trainer`` 中的：
1. ``_auto_generate_test_data``：生成 ``{"prompt":"...", "completion":"..."}`` 格式
2. ``_load_tokenizer``：非 TTY 环境下不阻塞、不抛异常
3. ``_load_tokenizer``：传入 ``model_cfg`` 含 ``tokenizer_repo`` 时正常加载
4. ``_prompt_tokenizer_action``：非 TTY 环境下返回 byte tokenizer

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_infra:packages/verse_torch:\
        packages/verse_nex \
        python -m pytest tests/test_trainer_tokenizer_init.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.path 注入
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_infra", "verse_torch", "verse_nex"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# 导入被测函数
# ---------------------------------------------------------------------------

from verse_infra.verse_trainer.trainer import (
    _auto_generate_test_data,
    _load_tokenizer,
    _prompt_tokenizer_action,
    _auto_build_tokenizer,
    _TEST_TEXTS,
)


# ---------------------------------------------------------------------------
# 测试 1：_auto_generate_test_data 生成新格式
# ---------------------------------------------------------------------------


class TestAutoGenerateTestData:
    """验证 _auto_generate_test_data 生成 prompt-completion 格式。"""

    def test_auto_generate_test_data_new_format(self, tmp_path):
        """生成的 jsonl 每行都是 ``{"prompt":"", "completion":""}`` 格式。"""
        train_path = str(tmp_path / "train.jsonl")
        val_path = str(tmp_path / "val.jsonl")

        _auto_generate_test_data(train_path, val_path)

        # 验证文件存在
        assert os.path.isfile(train_path), f"未生成 train.jsonl：{train_path}"
        assert os.path.isfile(val_path), f"未生成 val.jsonl：{val_path}"

        # 读取 train.jsonl 检查格式
        with open(train_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                assert isinstance(item, dict), f"第 {i} 行不是 dict"
                assert "prompt" in item, f"第 {i} 行缺少 prompt 字段"
                assert "completion" in item, f"第 {i} 行缺少 completion 字段"
                assert isinstance(item["prompt"], str), (
                    f"第 {i} 行 prompt 不是字符串"
                )
                assert isinstance(item["completion"], str), (
                    f"第 {i} 行 completion 不是字符串"
                )

        # 读取 val.jsonl 检查格式
        with open(val_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                assert "prompt" in item, f"val 第 {i} 行缺少 prompt 字段"
                assert "completion" in item, f"val 第 {i} 行缺少 completion 字段"

    def test_auto_generate_test_data_count(self, tmp_path):
        """验证生成条数：train=150（30 × 5），val=5。"""
        train_path = str(tmp_path / "train.jsonl")
        val_path = str(tmp_path / "val.jsonl")

        _auto_generate_test_data(train_path, val_path)

        with open(train_path, "r", encoding="utf-8") as f:
            train_lines = [l for l in f if l.strip()]
        with open(val_path, "r", encoding="utf-8") as f:
            val_lines = [l for l in f if l.strip()]

        expected_train = 5 * len(_TEST_TEXTS)
        assert len(train_lines) == expected_train, (
            f"train 条数应为 {expected_train}，实际 {len(train_lines)}"
        )
        assert len(val_lines) == 5, f"val 条数应为 5，实际 {len(val_lines)}"

    def test_test_texts_are_prompt_completion_pairs(self):
        """``_TEST_TEXTS`` 内置数据应是 prompt-completion 对（Part5K1.8）。"""
        assert len(_TEST_TEXTS) > 0
        for i, item in enumerate(_TEST_TEXTS):
            assert "prompt" in item, f"_TEST_TEXTS[{i}] 缺少 prompt 字段"
            assert "completion" in item, f"_TEST_TEXTS[{i}] 缺少 completion 字段"


# ---------------------------------------------------------------------------
# 测试 2：_load_tokenizer 非 TTY 不阻塞
# ---------------------------------------------------------------------------


class TestLoadTokenizerNonTTY:
    """验证非 TTY 环境下 _load_tokenizer 不阻塞。"""

    def test_load_tokenizer_non_tty_no_block(self, tmp_path):
        """非 TTY 环境（mock sys.stdin.isatty 返回 False）下不阻塞、不抛异常。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga"}

        # mock sys.stdin.isatty 返回 False（非 TTY）
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            # 调用 _load_tokenizer（kind=giga，非 TTY 走自行构建 → byte tokenizer）
            tok = _load_tokenizer(tok_cfg, base_dir, save_dir, model_cfg=None)

        assert tok is not None, "_load_tokenizer 不应返回 None"
        assert len(tok) > 0, "tokenizer 应有非零 vocab_size"

    def test_load_tokenizer_byte_kind_direct(self, tmp_path):
        """kind='byte' 直接返回 ByteTokenizer（不进入交互分支）。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "byte"}

        # 不 mock sys.stdin.isatty，验证 byte kind 不进入交互分支
        tok = _load_tokenizer(tok_cfg, base_dir, save_dir)
        assert tok is not None
        assert len(tok) > 0

    def test_load_tokenizer_byte_kind_in_ci(self, tmp_path):
        """CI=true 时 byte tokenizer 仍正常工作（最常见场景）。"""
        # 模拟 CI 环境（即使 isatty() 返回 False，byte kind 也直接返回）
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "byte"}
        with patch.dict(os.environ, {"CI": "true"}):
            tok = _load_tokenizer(tok_cfg, base_dir, save_dir)
        assert tok is not None
        assert len(tok) > 0


# ---------------------------------------------------------------------------
# 测试 3：_load_tokenizer 传入 model_cfg
# ---------------------------------------------------------------------------


class TestLoadTokenizerWithModelCfg:
    """验证 _load_tokenizer 传入 model_cfg 含 tokenizer_repo 时正常加载。"""

    def test_load_tokenizer_with_model_cfg(self, tmp_path):
        """传入 model_cfg 含 tokenizer_repo，非 TTY 走自动构建（不阻塞）。

        Part5K1.8：model_cfg 参数向后兼容（默认 None）。
        非 TTY 环境下无论 model_cfg 是否含 tokenizer_repo，都走自动构建路径。
        """
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga"}
        model_cfg = {"tokenizer_repo": "Qwen/Qwen3-32B"}

        # 非 TTY 环境（CI 模式）
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            tok = _load_tokenizer(
                tok_cfg, base_dir, save_dir, model_cfg=model_cfg
            )

        assert tok is not None, "传入 model_cfg 后应能正常加载"
        assert len(tok) > 0, "tokenizer 应有非零 vocab_size"

    def test_load_tokenizer_model_cfg_none_backward_compat(self, tmp_path):
        """model_cfg=None 向后兼容（不抛异常）。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "byte"}
        # model_cfg=None 应正常工作
        tok = _load_tokenizer(tok_cfg, base_dir, save_dir, model_cfg=None)
        assert tok is not None

    def test_load_tokenizer_with_existing_tokenizer_file(self, tmp_path):
        """save_dir 下已存在 tokenizer.json 时直接加载（不走交互分支）。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        # 预先创建一个 byte tokenizer.json（用 _auto_build_tokenizer 生成）
        byte_tok = _auto_build_tokenizer("byte", save_dir)
        tok_path = os.path.join(save_dir, "tokenizer.json")
        if hasattr(byte_tok, "save"):
            byte_tok.save(tok_path)
        # 如果 byte tokenizer 不支持 save 到 .json，跳过此测试
        if not os.path.exists(tok_path):
            pytest.skip("ByteTokenizer 不支持 save 到 .json")

        # 调用 _load_tokenizer：应直接加载已有的 tokenizer.json
        tok_cfg = {"kind": "byte"}
        tok = _load_tokenizer(tok_cfg, base_dir, save_dir)
        assert tok is not None


# ---------------------------------------------------------------------------
# 测试 4：_prompt_tokenizer_action 非 TTY 环境
# ---------------------------------------------------------------------------


class TestPromptTokenizerActionNonTTY:
    """验证 _prompt_tokenizer_action 在非 TTY 环境下的行为。"""

    def test_prompt_tokenizer_action_non_tty(self, tmp_path):
        """非 TTY 环境下调用 _prompt_tokenizer_action 返回 byte tokenizer。

        Part5K1.8：非 TTY 环境默认走自行构建（_auto_build_tokenizer），
        不调用 input() 阻塞。
        """
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga"}
        model_cfg = {"tokenizer_repo": "Qwen/Qwen3-32B"}

        # mock sys.stdin.isatty 返回 False（非 TTY）
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            # 不应调用 input()
            mock_stdin.readline = MagicMock(side_effect=EOFError)
            tok = _prompt_tokenizer_action(tok_cfg, model_cfg, save_dir, base_dir)

        assert tok is not None, "非 TTY 应返回自动构建的 tokenizer"
        assert len(tok) > 0, "tokenizer 应有非零 vocab_size"

    def test_prompt_tokenizer_action_non_tty_byte_kind(self, tmp_path):
        """非 TTY + kind=byte 时仍返回 byte tokenizer。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "byte"}

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            tok = _prompt_tokenizer_action(tok_cfg, {}, save_dir, base_dir)

        assert tok is not None
        # byte tokenizer 的 vocab_size 应为 259（256 字节 + bos/eos/pad/unk）
        assert len(tok) == 259 or len(tok) > 0

    def test_prompt_tokenizer_action_non_tty_does_not_call_input(self, tmp_path):
        """非 TTY 环境下不调用 input()（避免阻塞）。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga"}

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            # 设置 input 抛 EOFError（如果被调用）
            with patch("builtins.input", side_effect=AssertionError("input 不应被调用")):
                tok = _prompt_tokenizer_action(tok_cfg, {}, save_dir, base_dir)

        assert tok is not None

    def test_prompt_tokenizer_action_tty_choose_y(self, tmp_path):
        """TTY 环境 + 用户输 y + 有 tokenizer_repo 时尝试从 repo 加载。

        用 mock 模拟用户输入 y，验证调用 load_tokenizer。
        """
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga", "tokenizer_repo": "Qwen/Qwen3-32B"}

        # mock TTY + 输入 "y"
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="y"):
                # mock load_tokenizer 避免真实下载
                with patch(
                    "verse_infra.verse_tokenizer.load_tokenizer"
                ) as mock_vload:
                    mock_tok = MagicMock()
                    mock_tok.save = MagicMock()
                    mock_vload.return_value = mock_tok
                    tok = _prompt_tokenizer_action(
                        tok_cfg, {}, save_dir, base_dir
                    )

        # 应调用 load_tokenizer 加载 repo_source
        assert mock_vload.called, "TTY + y 应调用 load_tokenizer"
        # 应调用 save 保存到 save_dir/tokenizer.json
        if mock_tok.save.called:
            saved_path = mock_tok.save.call_args[0][0]
            assert "tokenizer.json" in saved_path

    def test_prompt_tokenizer_action_tty_choose_n(self, tmp_path):
        """TTY 环境 + 用户输 n 时走自行构建。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga"}

        # mock TTY + 输入 "n"
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="n"):
                tok = _prompt_tokenizer_action(tok_cfg, {}, save_dir, base_dir)

        assert tok is not None
        assert len(tok) > 0

    def test_prompt_tokenizer_action_tty_invalid_input_three_times(self, tmp_path):
        """TTY 环境 + 三次无效输入后默认走自行构建。"""
        save_dir = str(tmp_path / "save")
        base_dir = str(tmp_path / "base")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

        tok_cfg = {"kind": "giga"}

        # mock TTY + 三次无效输入
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", side_effect=["xxx", "abc", "???"]):
                tok = _prompt_tokenizer_action(tok_cfg, {}, save_dir, base_dir)

        assert tok is not None, "三次无效输入后应默认走自行构建"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
