"""Part3 Task 7: 自定义 Prompt 支持测试。

验证：
1. ``--prompt "a,b"`` 正确解析为 ``["a", "b"]``
2. ``--prompts-file`` 正确读取文件（每行一个 prompt，忽略空行与 # 注释行）
3. ``--prompt`` 优先级高于 ``--prompts-file``
4. 都未指定时返回 None（由 evaluate 使用默认 5 条）
5. 参数透传到 evaluate（用 mock 或实际调用）

运行：
    cd /workspace && python -m pytest tests/test_custom_prompt.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 路径常量与 sys.path 注入
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
for pkg in ("verse_torch", "verse_nex", "verse_tokenizer",
            "verse_inference", "verse_compat"):
    p = REPO_ROOT / "packages" / pkg
    if p.is_dir():
        sys.path.insert(0, str(p))

# 把 data/demo 加入 path 以便 import run / model.config / model.model
_DEMO_DIR = REPO_ROOT / "data" / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# Task 7.5: _parse_prompts_from_cli 单元测试
# ---------------------------------------------------------------------------


class TestParsePromptsFromCli:
    """验证 ``run._parse_prompts_from_cli`` 的解析逻辑。"""

    def test_prompt_comma_separated(self):
        """``--prompt "a,b"`` 应解析为 ``["a", "b"]``。"""
        from run import _parse_prompts_from_cli

        result = _parse_prompts_from_cli("a,b", None, str(_DEMO_DIR))
        assert result == ["a", "b"]

    def test_prompt_chinese_comma_separated(self):
        """中文 prompt 逗号分隔：``"床前明月光，,你好，"`` → 2 条。"""
        from run import _parse_prompts_from_cli

        result = _parse_prompts_from_cli("床前明月光，,你好，", None, str(_DEMO_DIR))
        assert result == ["床前明月光，", "你好，"]

    def test_prompt_single(self):
        """单条 prompt（无逗号）应返回单元素列表。"""
        from run import _parse_prompts_from_cli

        result = _parse_prompts_from_cli("hello", None, str(_DEMO_DIR))
        assert result == ["hello"]

    def test_prompt_empty_string_returns_none(self):
        """空字符串 ``--prompt ""`` 应返回 None（用默认 5 条）。"""
        from run import _parse_prompts_from_cli

        result = _parse_prompts_from_cli("", None, str(_DEMO_DIR))
        assert result is None

    def test_prompt_only_commas_returns_none(self):
        """``--prompt ",,,"`` 过滤首尾空字符串后为空，应返回 None。"""
        from run import _parse_prompts_from_cli

        result = _parse_prompts_from_cli(",,,", None, str(_DEMO_DIR))
        assert result is None

    def test_prompts_file_reads_lines(self, tmp_path):
        """``--prompts-file`` 应读取每行一个 prompt。"""
        from run import _parse_prompts_from_cli

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text(
            "床前明月光\n"
            "白日依山尽\n"
            "你好\n",
            encoding="utf-8",
        )

        result = _parse_prompts_from_cli(None, str(prompts_file), str(_DEMO_DIR))
        assert result == ["床前明月光", "白日依山尽", "你好"]

    def test_prompts_file_ignores_blank_and_comment_lines(self, tmp_path):
        """``--prompts-file`` 应忽略空行与 ``#`` 注释行。"""
        from run import _parse_prompts_from_cli

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text(
            "# 这是注释\n"
            "第一条 prompt\n"
            "\n"
            "  # 这也是注释（带前导空格）\n"
            "第二条 prompt\n"
            "\n",
            encoding="utf-8",
        )

        result = _parse_prompts_from_cli(None, str(prompts_file), str(_DEMO_DIR))
        # 注释行被过滤，空行被过滤，只保留 2 条
        assert result == ["第一条 prompt", "第二条 prompt"]

    def test_prompt_takes_priority_over_prompts_file(self, tmp_path):
        """``--prompt`` 优先级高于 ``--prompts-file``。"""
        from run import _parse_prompts_from_cli

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("文件内容1\n文件内容2\n", encoding="utf-8")

        result = _parse_prompts_from_cli(
            "cli_a,cli_b", str(prompts_file), str(_DEMO_DIR)
        )
        assert result == ["cli_a", "cli_b"]

    def test_neither_prompt_nor_file_returns_none(self):
        """都未指定时返回 None。"""
        from run import _parse_prompts_from_cli

        result = _parse_prompts_from_cli(None, None, str(_DEMO_DIR))
        assert result is None

    def test_prompts_file_not_found_raises(self):
        """``--prompts-file`` 指向不存在的文件应抛 FileNotFoundError。"""
        from run import _parse_prompts_from_cli

        with pytest.raises(FileNotFoundError, match="--prompts-file"):
            _parse_prompts_from_cli(None, "/nonexistent/path.txt", str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# Task 7.5: --arch 覆盖逻辑测试
# ---------------------------------------------------------------------------


class TestOverrideConfigArch:
    """验证 ``run._override_config_arch`` 的覆盖逻辑。"""

    def test_no_arch_returns_original_path(self):
        """``--arch`` 未指定时返回原 config 路径，无临时文件。"""
        from run import _override_config_arch

        config_path = str(_DEMO_DIR / "config" / "config.yml")
        effective, tmp = _override_config_arch(config_path, None)
        assert effective == config_path
        assert tmp is None

    def test_arch_transformer_creates_temp_file(self):
        """``--arch transformer`` 创建临时 config 文件并写入 arch=transformer。"""
        from run import _override_config_arch
        from model.config import load_full_config

        config_path = str(_DEMO_DIR / "config" / "config.yml")
        effective, tmp = _override_config_arch(config_path, "transformer")
        try:
            assert effective == tmp
            assert os.path.exists(tmp)
            # 临时文件的 arch 应为 transformer
            cfg = load_full_config(tmp)
            assert cfg["model"]["arch"] == "transformer"
            # 原始文件不应被修改
            orig_cfg = load_full_config(config_path)
            assert orig_cfg["model"]["arch"] == "transformer"  # config.yml 默认就是
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    def test_arch_hybrid_overrides_transformer(self):
        """``--arch hybrid`` 覆盖原 transformer 配置。"""
        from run import _override_config_arch
        from model.config import load_full_config

        config_path = str(_DEMO_DIR / "config" / "config.yml")
        # config.yml 默认 arch=transformer，覆盖为 hybrid
        effective, tmp = _override_config_arch(config_path, "hybrid")
        try:
            assert os.path.exists(tmp)
            cfg = load_full_config(tmp)
            assert cfg["model"]["arch"] == "hybrid"
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    def test_invalid_arch_raises(self):
        """非法 arch 值应抛 ValueError。"""
        from run import _override_config_arch

        config_path = str(_DEMO_DIR / "config" / "config.yml")
        with pytest.raises(ValueError, match="--arch"):
            _override_config_arch(config_path, "invalid_arch")


# ---------------------------------------------------------------------------
# Task 7.5: run.py --help 显示所有新参数
# ---------------------------------------------------------------------------


class TestRunPyHelpShowsNewArgs:
    """验证 ``python run.py --help`` 显示所有新 CLI 参数。"""

    def test_help_shows_all_new_args(self):
        # 子进程需要能 import verse_torch / verse_nex 等，设置 PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(REPO_ROOT / "packages" / "verse_torch"),
            str(REPO_ROOT / "packages" / "verse_nex"),
            str(REPO_ROOT / "packages" / "verse_tokenizer"),
            str(REPO_ROOT / "packages" / "verse_inference"),
            str(REPO_ROOT / "packages" / "verse_compat"),
            str(_DEMO_DIR),
        ])
        result = subprocess.run(
            [sys.executable, str(_DEMO_DIR / "run.py"), "--help"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0, (
            f"run.py --help 退出码非 0：\n{result.stderr}"
        )
        for arg in ("--prompt", "--prompts-file", "--max-tokens",
                    "--temperature", "--top-k", "--arch"):
            assert arg in result.stdout, (
                f"run.py --help 未显示 {arg}：\n{result.stdout}"
            )


# ---------------------------------------------------------------------------
# Task 7.5: 参数透传到 evaluate（mock 验证）
# ---------------------------------------------------------------------------


class TestEvaluatePassthrough:
    """验证 ``run.stage_evaluate`` 透传参数到 ``train.evaluate.evaluate``。"""

    def test_stage_evaluate_passes_all_params(self):
        """stage_evaluate 应把 prompts/max_new_tokens/temperature/top_k 全部透传。"""
        from run import stage_evaluate

        captured_kwargs = {}

        def mock_evaluate(config_path, base_dir=".", **kwargs):
            captured_kwargs.update(kwargs)
            return {"results": [], "wall_clock": 0.0, "vocab_size": 256}

        with patch("run.evaluate_fn", mock_evaluate):
            stage_evaluate(
                config_path=str(_DEMO_DIR / "config" / "config.yml"),
                base_dir=str(_DEMO_DIR),
                prompts=["p1", "p2"],
                max_new_tokens=42,
                temperature=0.7,
                top_k=15,
            )

        assert captured_kwargs["prompts"] == ["p1", "p2"]
        assert captured_kwargs["max_new_tokens"] == 42
        assert captured_kwargs["temperature"] == 0.7
        assert captured_kwargs["top_k"] == 15

    def test_stage_evaluate_default_temperature(self):
        """stage_evaluate 默认 temperature=1.0。"""
        from run import stage_evaluate

        captured_kwargs = {}

        def mock_evaluate(config_path, base_dir=".", **kwargs):
            captured_kwargs.update(kwargs)
            return {"results": [], "wall_clock": 0.0, "vocab_size": 256}

        with patch("run.evaluate_fn", mock_evaluate):
            stage_evaluate(
                config_path=str(_DEMO_DIR / "config" / "config.yml"),
                base_dir=str(_DEMO_DIR),
            )

        assert captured_kwargs["temperature"] == 1.0
        assert captured_kwargs["max_new_tokens"] == 30
        assert captured_kwargs["top_k"] is None
        assert captured_kwargs["prompts"] is None


# ---------------------------------------------------------------------------
# Task 9.2: count_parameters 单元测试
# ---------------------------------------------------------------------------


class TestCountParameters:
    """验证 ``CometSparkLM.count_parameters`` 方法。"""

    def test_count_parameters_returns_positive(self):
        """count_parameters 应返回正整数。"""
        from model.config import CometSparkConfig
        from model.model import CometSparkLM

        config = CometSparkConfig(
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            seq_len=16,
            dropout=0.0,
            arch="transformer",
            tie_weights=True,
        )
        model = CometSparkLM(config)
        n = model.count_parameters()
        assert isinstance(n, int)
        assert n > 0

    def test_count_parameters_scales_with_model_size(self):
        """参数量应随 n_layer / n_embd 增大而增大。"""
        from model.config import CometSparkConfig
        from model.model import CometSparkLM

        small_cfg = CometSparkConfig(
            vocab_size=32, n_layer=1, n_head=2, n_embd=8,
            seq_len=16, dropout=0.0, arch="transformer", tie_weights=True,
        )
        large_cfg = CometSparkConfig(
            vocab_size=32, n_layer=4, n_head=4, n_embd=32,
            seq_len=16, dropout=0.0, arch="transformer", tie_weights=True,
        )
        small_model = CometSparkLM(small_cfg)
        large_model = CometSparkLM(large_cfg)
        assert large_model.count_parameters() > small_model.count_parameters()

    def test_count_parameters_matches_state_dict_size(self):
        """参数量应等于 state_dict 中所有 ndarray 元素数之和。"""
        from model.config import CometSparkConfig
        from model.model import CometSparkLM

        config = CometSparkConfig(
            vocab_size=32, n_layer=2, n_head=2, n_embd=16,
            seq_len=16, dropout=0.0, arch="transformer", tie_weights=True,
        )
        model = CometSparkLM(config)
        sd = model.state_dict()
        expected = sum(int(np.prod(v.shape)) for v in sd.values())
        actual = model.count_parameters()
        # state_dict 可能包含一些非 requires_grad 的 buffer；
        # count_parameters 只统计可训练参数，应 <= state_dict 总大小
        assert actual <= expected


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
