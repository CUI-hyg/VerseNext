"""Part5K1 Task 11：spark/run.py 训练模式补齐测试。

覆盖 SubTask 11.7 的全部测试用例：
1. ``_select_model_level`` 函数：small / mate / 默认 / 非法 / --config 覆盖
2. train 子命令 ``--model small|mate`` dry-run + 默认 small
3. finetune 子命令 dry-run（small / mate / lora / full）
4. posttrain 子命令 dry-run（sft / dpo / rl）
5. continue 子命令 dry-run（small / mate）
6. eval / generate / chat / compress / convert 同步 ``--model`` 参数
7. 非法 ``--model`` 抛错（argparse choices 校验）
8. 委托原则：finetune / posttrain / continue 实际调用 verse_trainer 函数（mock 验证）
9. ``--config`` 覆盖 ``--model`` 默认配置路径

运行方式：
    cd /workspace && python -m pytest tests/test_spark_run_dual.py -x -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# sys.path 注入（与 test_dual_model.py 一致）
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# 导入 spark.run（触发路径自举）
from spark import run as spark_run  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助：调用 CLI 并捕获输出
# ---------------------------------------------------------------------------


def _run_cli(argv, capsys):
    """调用 spark.run.main(argv)，返回 (exit_code, stdout, stderr)。"""
    code = spark_run.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# SubTask 11.1 / 11.5: _select_model_level 函数验证
# ---------------------------------------------------------------------------


class TestSelectModelLevel:
    """``_select_model_level`` 辅助函数验证。"""

    def test_small_level_returns_small_config_and_factory(self):
        """--model small 返回 small 配置路径 + CometSparkSmall 工厂。"""
        from spark.small.model import CometSparkSmall

        args = SimpleNamespace(model="small", config=None)
        config_path, factory, level = spark_run._select_model_level(args)

        assert level == "small"
        assert factory is CometSparkSmall
        assert config_path.endswith("spark/small/config/cometspark_small.yml")
        assert os.path.exists(config_path)

    def test_mate_level_returns_mate_config_and_factory(self):
        """--model mate 返回 mate 配置路径 + CometSparkMate 工厂。"""
        from spark.mate.model import CometSparkMate

        args = SimpleNamespace(model="mate", config=None)
        config_path, factory, level = spark_run._select_model_level(args)

        assert level == "mate"
        assert factory is CometSparkMate
        assert config_path.endswith("spark/mate/config/cometspark_mate.yml")
        assert os.path.exists(config_path)

    def test_default_model_is_small(self):
        """无 model 字段时默认 small。"""
        args = SimpleNamespace(config=None)
        config_path, factory, level = spark_run._select_model_level(args)

        assert level == "small"
        assert config_path.endswith("spark/small/config/cometspark_small.yml")

    def test_invalid_model_raises_value_error(self):
        """非法 model level 抛 ValueError。"""
        args = SimpleNamespace(model="huge", config=None)
        with pytest.raises(ValueError, match="未知 model level"):
            spark_run._select_model_level(args)

    def test_config_override_uses_config_path(self, tmp_path):
        """--config 覆盖默认配置路径，但工厂仍按 --model 选择。"""
        from spark.small.model import CometSparkSmall

        # 创建一个临时配置文件
        custom_cfg = tmp_path / "custom.yml"
        custom_cfg.write_text("model:\n  arch: versenex\n")

        args = SimpleNamespace(model="small", config=str(custom_cfg))
        config_path, factory, level = spark_run._select_model_level(args)

        assert level == "small"
        assert factory is CometSparkSmall
        assert config_path == str(custom_cfg)

    def test_config_override_with_mate_factory(self, tmp_path):
        """--config 覆盖路径 + --model mate → 工厂仍是 CometSparkMate。"""
        from spark.mate.model import CometSparkMate

        custom_cfg = tmp_path / "custom_mate.yml"
        custom_cfg.write_text("model:\n  arch: versenex\n")

        args = SimpleNamespace(model="mate", config=str(custom_cfg))
        config_path, factory, level = spark_run._select_model_level(args)

        assert level == "mate"
        assert factory is CometSparkMate
        assert config_path == str(custom_cfg)

    def test_nonexistent_config_raises_filenotfounderror(self):
        """--config 指向不存在的文件抛 FileNotFoundError。"""
        args = SimpleNamespace(model="small", config="/nonexistent/path.yml")
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            spark_run._select_model_level(args)


# ---------------------------------------------------------------------------
# SubTask 11.1: train 子命令 --model dry-run
# ---------------------------------------------------------------------------


class TestTrainModelDryRun:
    """train 子命令 ``--model small|mate`` dry-run。"""

    def test_train_model_small_dry_run(self, capsys):
        """train --model small --dry-run 打印 small 配置。"""
        code, out, _ = _run_cli(
            ["train", "--model", "small", "--dry-run"], capsys
        )
        assert code == 0
        assert "cometspark_small.yml" in out
        assert "small" in out.lower()

    def test_train_model_mate_dry_run(self, capsys):
        """train --model mate --dry-run 打印 mate 配置。"""
        code, out, _ = _run_cli(
            ["train", "--model", "mate", "--dry-run"], capsys
        )
        assert code == 0
        assert "cometspark_mate.yml" in out
        assert "mate" in out.lower()

    def test_train_default_model_is_small(self, capsys):
        """train --dry-run（不带 --model）默认 small。"""
        code, out, _ = _run_cli(["train", "--dry-run"], capsys)
        assert code == 0
        assert "cometspark_small.yml" in out
        assert "small" in out.lower()

    def test_train_small_flag_backward_compat(self, capsys):
        """train --small --dry-run 向后兼容旧 --small 标志。"""
        code, out, _ = _run_cli(["train", "--small", "--dry-run"], capsys)
        assert code == 0
        # --small 走旧 _SMALL_CONFIG 路径
        assert "cometspark_v05_small.yml" in out or "small" in out.lower()


# ---------------------------------------------------------------------------
# SubTask 11.2: finetune 子命令 dry-run
# ---------------------------------------------------------------------------


class TestFinetuneDryRun:
    """finetune 子命令 dry-run（Part5K1 Task 11.2）。"""

    def test_finetune_small_lora_dry_run(self, capsys):
        """finetune --model small --method lora --dry-run。"""
        code, out, _ = _run_cli(
            ["finetune", "--model", "small", "--method", "lora", "--dry-run"],
            capsys,
        )
        assert code == 0
        assert "finetune" in out
        assert "small" in out
        assert "lora" in out
        assert "cometspark_small.yml" in out

    def test_finetune_mate_full_dry_run(self, capsys):
        """finetune --model mate --method full --dry-run。"""
        code, out, _ = _run_cli(
            ["finetune", "--model", "mate", "--method", "full", "--dry-run"],
            capsys,
        )
        assert code == 0
        assert "finetune" in out
        assert "mate" in out
        assert "full" in out
        assert "cometspark_mate.yml" in out

    def test_finetune_lora_params_in_dry_run(self, capsys):
        """finetune --lora-r 16 --lora-alpha 32 --dry-run 打印参数。"""
        code, out, _ = _run_cli(
            [
                "finetune", "--model", "small", "--method", "lora",
                "--lora-r", "16", "--lora-alpha", "32", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "lora_r = 16" in out
        assert "lora_alpha = 32.0" in out

    def test_finetune_default_model_is_small(self, capsys):
        """finetune --dry-run（不带 --model）默认 small。"""
        code, out, _ = _run_cli(["finetune", "--dry-run"], capsys)
        assert code == 0
        assert "small" in out
        assert "cometspark_small.yml" in out


# ---------------------------------------------------------------------------
# SubTask 11.3: posttrain 子命令 dry-run
# ---------------------------------------------------------------------------


class TestPosttrainDryRun:
    """posttrain 子命令 dry-run（Part5K1 Task 11.3）。"""

    def test_posttrain_small_sft_dry_run(self, capsys):
        """posttrain --model small --mode sft --dry-run。"""
        code, out, _ = _run_cli(
            ["posttrain", "--model", "small", "--mode", "sft", "--dry-run"],
            capsys,
        )
        assert code == 0
        assert "posttrain" in out
        assert "sft" in out
        assert "small" in out
        assert "cometspark_small.yml" in out

    def test_posttrain_mate_dpo_dry_run(self, capsys):
        """posttrain --model mate --mode dpo --dry-run。"""
        code, out, _ = _run_cli(
            ["posttrain", "--model", "mate", "--mode", "dpo", "--dry-run"],
            capsys,
        )
        assert code == 0
        assert "dpo" in out
        assert "mate" in out
        assert "cometspark_mate.yml" in out

    def test_posttrain_rl_dry_run(self, capsys):
        """posttrain --model small --mode rl --dry-run。"""
        code, out, _ = _run_cli(
            ["posttrain", "--model", "small", "--mode", "rl", "--dry-run"],
            capsys,
        )
        assert code == 0
        assert "rl" in out

    def test_posttrain_default_model_is_small(self, capsys):
        """posttrain --dry-run（不带 --model）默认 small。"""
        code, out, _ = _run_cli(["posttrain", "--dry-run"], capsys)
        assert code == 0
        assert "small" in out
        assert "cometspark_small.yml" in out


# ---------------------------------------------------------------------------
# SubTask 11.4: continue 子命令 dry-run
# ---------------------------------------------------------------------------


class TestContinueDryRun:
    """continue 子命令 dry-run（Part5K1 Task 11.4）。"""

    def test_continue_small_dry_run(self, capsys):
        """continue --model small --checkpoint ck.pt --dry-run。"""
        code, out, _ = _run_cli(
            [
                "continue", "--model", "small",
                "--checkpoint", "ck.pt", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "continue" in out
        assert "small" in out
        assert "ck.pt" in out
        assert "cometspark_small.yml" in out

    def test_continue_mate_dry_run(self, capsys):
        """continue --model mate --checkpoint ck.pt --dry-run。"""
        code, out, _ = _run_cli(
            [
                "continue", "--model", "mate",
                "--checkpoint", "ck.pt", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "mate" in out
        assert "ck.pt" in out
        assert "cometspark_mate.yml" in out

    def test_continue_additional_steps_in_dry_run(self, capsys):
        """continue --additional-steps 50 --dry-run 打印步数。"""
        code, out, _ = _run_cli(
            [
                "continue", "--model", "small",
                "--checkpoint", "ck.pt",
                "--additional-steps", "50", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "additional_steps = 50" in out

    def test_continue_requires_checkpoint(self, capsys):
        """continue 不带 --checkpoint 应报错（argparse required）。"""
        with pytest.raises(SystemExit):
            spark_run.main(["continue", "--model", "small", "--dry-run"])


# ---------------------------------------------------------------------------
# SubTask 11.5: eval / generate / chat / compress / convert 同步 --model
# ---------------------------------------------------------------------------


class TestOtherSubcommandsModelSync:
    """eval / generate / chat / compress / convert 同步 --model 参数。"""

    def test_eval_model_small_dry_run(self, capsys):
        """eval --model small --dry-run 打印 model_level=small。"""
        code, out, _ = _run_cli(
            ["eval", "--model", "small", "--dry-run"], capsys
        )
        assert code == 0
        assert "small" in out
        assert "cometspark_small.yml" in out

    def test_eval_model_mate_dry_run(self, capsys):
        """eval --model mate --dry-run 打印 model_level=mate。"""
        code, out, _ = _run_cli(
            ["eval", "--model", "mate", "--dry-run"], capsys
        )
        assert code == 0
        assert "mate" in out
        assert "cometspark_mate.yml" in out

    def test_generate_model_small_dry_run(self, capsys):
        """generate --model small --dry-run 打印 model_level=small。"""
        code, out, _ = _run_cli(
            ["generate", "--model", "small", "--dry-run"], capsys
        )
        assert code == 0
        assert "small" in out

    def test_generate_model_mate_dry_run(self, capsys):
        """generate --model mate --dry-run 打印 model_level=mate。"""
        code, out, _ = _run_cli(
            ["generate", "--model", "mate", "--prompt", "hi", "--dry-run"],
            capsys,
        )
        assert code == 0
        assert "mate" in out

    def test_chat_model_small_dry_run(self, capsys):
        """chat --model small --dry-run 打印 model_level=small。"""
        code, out, _ = _run_cli(
            ["chat", "--model", "small", "--dry-run"], capsys
        )
        assert code == 0
        assert "small" in out

    def test_compress_model_small_dry_run(self, capsys):
        """compress --model small --checkpoint ck.pt --dry-run。"""
        code, out, _ = _run_cli(
            [
                "compress", "--model", "small",
                "--checkpoint", "ck.pt", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "small" in out

    def test_compress_model_mate_dry_run(self, capsys):
        """compress --model mate --checkpoint ck.pt --dry-run。"""
        code, out, _ = _run_cli(
            [
                "compress", "--model", "mate",
                "--checkpoint", "ck.pt", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "mate" in out

    def test_convert_model_small_dry_run(self, capsys):
        """convert --model small --input a.pt --output b.vn --dry-run。"""
        code, out, _ = _run_cli(
            [
                "convert", "--model", "small",
                "--input", "a.pt", "--output", "b.vn", "--dry-run",
            ],
            capsys,
        )
        assert code == 0
        assert "small" in out


# ---------------------------------------------------------------------------
# SubTask 11.5（续）: 非法 --model 抛错
# ---------------------------------------------------------------------------


class TestInvalidModel:
    """非法 ``--model`` 值由 argparse choices 校验拒绝。"""

    @pytest.mark.parametrize(
        "subcmd",
        ["train", "finetune", "posttrain", "eval", "generate",
         "chat", "compress"],
    )
    def test_invalid_model_raises_systemexit(self, subcmd):
        """各子命令 --model invalid 应触发 argparse 报错（SystemExit）。"""
        argv = [subcmd, "--model", "invalid"]
        if subcmd in ("compress",):
            argv += ["--checkpoint", "ck.pt"]
        with pytest.raises(SystemExit):
            spark_run.main(argv)

    def test_continue_invalid_model_raises_systemexit(self):
        """continue --model invalid 也应报错。"""
        with pytest.raises(SystemExit):
            spark_run.main(
                ["continue", "--model", "invalid", "--checkpoint", "ck.pt"]
            )


# ---------------------------------------------------------------------------
# SubTask 11.6: 委托原则验证（mock verse_trainer 函数）
# ---------------------------------------------------------------------------


class TestDelegatePrinciple:
    """验证 finetune / posttrain / continue 委托 verse_trainer（不重复造轮子）。

    通过 mock ``verse_infra.verse_trainer.train`` / ``continue_train``，验证
    非 dry-run 时确实调用了 verse_trainer 的对应函数。
    """

    def test_finetune_delegates_to_verse_trainer_train(self, tmp_path):
        """finetune 非 dry-run 委托 verse_trainer.train()。"""
        from unittest.mock import patch, MagicMock

        # 构造 args（非 dry-run）
        args = SimpleNamespace(
            model="small", config=None,
            method="lora", lora_r=8, lora_alpha=16.0,
            target_modules=None, checkpoint=None,
            data=None, lr=None, max_steps=None,
            device=None, dry_run=False,
            quiet=True, verbose=False,
        )

        mock_result = {"best_val_loss": 1.234, "save_dir": "mf_small"}
        with patch(
            "verse_infra.verse_trainer.train",
            return_value=mock_result,
        ) as mock_train, patch(
            "verse_infra.verse_trainer.cli._apply_config_overrides",
            side_effect=lambda cfg, ov: cfg,
        ):
            code = spark_run.cmd_finetune(args)

        assert code == 0
        assert mock_train.called
        # 验证传入的是 config_path（非 None）
        _call_kwargs = mock_train.call_args.kwargs
        assert "config_path" in _call_kwargs
        assert _call_kwargs["config_path"] is not None

    def test_posttrain_sft_delegates_to_verse_trainer_train(self, tmp_path):
        """posttrain --mode sft 非 dry-run 委托 verse_trainer.train()。"""
        from unittest.mock import patch

        args = SimpleNamespace(
            model="small", config=None,
            mode="sft", checkpoint=None,
            data=None, lr=None, max_steps=None,
            device=None, dry_run=False,
            quiet=True, verbose=False,
        )

        mock_result = {"best_val_loss": 0.567, "save_dir": "mf_small"}
        with patch(
            "verse_infra.verse_trainer.train",
            return_value=mock_result,
        ) as mock_train, patch(
            "verse_infra.verse_trainer.cli._apply_config_overrides",
            side_effect=lambda cfg, ov: cfg,
        ):
            code = spark_run.cmd_posttrain(args)

        assert code == 0
        assert mock_train.called

    def test_continue_delegates_to_verse_trainer_continue_train(self, tmp_path):
        """continue 非 dry-run 委托 verse_trainer.continue_train()。"""
        from unittest.mock import patch

        # 需要一个存在的 checkpoint 文件（continue_train 内部会检查）
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")

        args = SimpleNamespace(
            model="small", config=None,
            checkpoint=str(ckpt),
            additional_steps=10,
            lr=None, device=None, amp=False,
            dry_run=False,
            quiet=True, verbose=False,
        )

        mock_result = {"best_val_loss": 0.111, "save_dir": "mf_small"}
        with patch(
            "verse_infra.verse_trainer.continue_train",
            return_value=mock_result,
        ) as mock_cont:
            code = spark_run.cmd_continue(args)

        assert code == 0
        assert mock_cont.called
        # 验证传入 checkpoint 和 additional_steps
        _call_kwargs = mock_cont.call_args.kwargs
        assert _call_kwargs["checkpoint"] == str(ckpt)
        assert _call_kwargs["additional_steps"] == 10

    def test_train_delegates_to_verse_trainer_train(self, tmp_path):
        """train 非 dry-run 委托 verse_trainer.train()。"""
        from unittest.mock import patch

        args = SimpleNamespace(
            model="small", config=None, small=False,
            max_steps=None, batch_size=None,
            device=None, resume=False, amp=False,
            eval_after=False, dry_run=False,
            quiet=True, verbose=False,
        )

        mock_result = {"best_val_loss": 0.5, "save_dir": "mf_small"}
        with patch(
            "verse_infra.verse_trainer.train",
            return_value=mock_result,
        ) as mock_train:
            code = spark_run.cmd_train(args)

        assert code == 0
        assert mock_train.called


# ---------------------------------------------------------------------------
# SubTask 11.7: verse_trainer 导出 continue_train（委托前提）
# ---------------------------------------------------------------------------


class TestVerseTrainerExports:
    """验证 verse_trainer 包导出了 train / continue_train / evaluate。"""

    def test_verse_trainer_exports_train(self):
        """verse_infra.verse_trainer 导出 train。"""
        import verse_infra.verse_trainer as _vt
        assert hasattr(_vt, "train")
        assert callable(_vt.train)

    def test_verse_trainer_exports_continue_train(self):
        """verse_infra.verse_trainer 导出 continue_train（Part5K1 Task 11.4 委托前提）。"""
        import verse_infra.verse_trainer as _vt
        assert hasattr(_vt, "continue_train")
        assert callable(_vt.continue_train)

    def test_verse_trainer_exports_evaluate(self):
        """verse_infra.verse_trainer 导出 evaluate。"""
        import verse_infra.verse_trainer as _vt
        assert hasattr(_vt, "evaluate")
        assert callable(_vt.evaluate)
