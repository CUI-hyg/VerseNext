"""Part4K2.5 Task 4：训练后自动评估打分测试。

覆盖：
1. train() 默认 eval_after=True → result 含 eval_result
2. train() eval_after=False → result 不含 eval_result
3. _auto_evaluate 基本功能（dict 结构正确）
4. _auto_evaluate 无 reference 时只记录质量（scores 全 0 / n_samples=0）
5. _auto_evaluate 有 reference 时计算打分（5 个指标存在）
6. --no-eval CLI 选项解析
7. --eval-prompts CLI 选项解析
8. 评估报告打印（format_eval_report）
9. 默认测试用例正确（_DEFAULT_EVAL_PROMPTS）
10. _resolve_eval_prompts 优先级（eval_config > full_config > 默认）
11. evaluate_from_train_result 便捷函数

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_torch:packages/verse_nex:\
        packages/verse_infra \
        python -m pytest tests/test_auto_eval.py -x -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# PYTHONPATH 适配
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
for _sub in ("verse_torch", "verse_nex", "verse_infra"):
    _p = os.path.join(_WORKSPACE, "packages", _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from verse_infra.verse_trainer.trainer import (
    train,
    _auto_evaluate,
    _resolve_eval_prompts,
    _DEFAULT_EVAL_PROMPTS,
)
from verse_infra.verse_trainer.evaluate import (
    format_eval_report,
    evaluate_from_train_result,
)


# ---------------------------------------------------------------------------
# 通用辅助：构造最小 config + tokenizer
# ---------------------------------------------------------------------------


def _make_tiny_config(tmpdir, max_steps=3):
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
            "vocab_size": 259,
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
            "warmup": 1,
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
    try:
        import yaml
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        lines = []
        for section, sub in config.items():
            lines.append(f"{section}:")
            for k, v in sub.items():
                lines.append(f"  {k}: {v}")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return cfg_path


# ===========================================================================
# 1. 默认测试用例正确
# ===========================================================================


class TestDefaultEvalPrompts:
    """_DEFAULT_EVAL_PROMPTS 默认测试用例。"""

    def test_has_4_prompts(self):
        """默认测试用例应有 4 条。"""
        assert len(_DEFAULT_EVAL_PROMPTS) == 4

    def test_structure_correct(self):
        """每条用例应有 prompt 和 reference 字段。"""
        for item in _DEFAULT_EVAL_PROMPTS:
            assert "prompt" in item
            assert "reference" in item
            assert isinstance(item["prompt"], str)

    def test_has_reference_and_no_reference(self):
        """应同时包含有 reference 和无 reference 的用例。"""
        has_ref = [p for p in _DEFAULT_EVAL_PROMPTS if p["reference"] is not None]
        no_ref = [p for p in _DEFAULT_EVAL_PROMPTS if p["reference"] is None]
        assert len(has_ref) >= 1, "应至少有 1 条带 reference 的用例"
        assert len(no_ref) >= 1, "应至少有 1 条无 reference 的用例"

    def test_contains_expected_prompts(self):
        """应包含任务要求的测试用例。"""
        prompts = [p["prompt"] for p in _DEFAULT_EVAL_PROMPTS]
        assert "你好" in prompts
        assert "Hello" in prompts
        assert "1+1=" in prompts
        assert "中国的首都是" in prompts
        # 1+1= 的 reference 应为 "2"
        for p in _DEFAULT_EVAL_PROMPTS:
            if p["prompt"] == "1+1=":
                assert p["reference"] == "2"
            if p["prompt"] == "中国的首都是":
                assert p["reference"] == "北京"


# ===========================================================================
# 2. _resolve_eval_prompts 优先级
# ===========================================================================


class TestResolveEvalPrompts:
    """_resolve_eval_prompts 优先级测试。"""

    def test_default_when_no_config(self):
        """无 eval_config 和 full_config 时用默认。"""
        result = _resolve_eval_prompts(None, None)
        assert len(result) == 4
        assert result[0]["prompt"] == "你好"

    def test_eval_config_overrides_default(self):
        """eval_config["prompts"] 优先于默认。"""
        eval_config = {
            "prompts": [
                {"prompt": "测试", "reference": "结果"},
            ]
        }
        result = _resolve_eval_prompts(eval_config, None)
        assert len(result) == 1
        assert result[0]["prompt"] == "测试"
        assert result[0]["reference"] == "结果"

    def test_full_config_eval_section(self):
        """full_config["eval"]["prompts"] 作为第二优先级。"""
        full_config = {
            "eval": {
                "prompts": [
                    {"prompt": "配置测试", "reference": None},
                    {"prompt": "第二条", "reference": "答案"},
                ]
            }
        }
        result = _resolve_eval_prompts(None, full_config)
        assert len(result) == 2
        assert result[0]["prompt"] == "配置测试"
        assert result[1]["reference"] == "答案"

    def test_eval_config_overrides_full_config(self):
        """eval_config 优先于 full_config。"""
        eval_config = {"prompts": [{"prompt": "显式", "reference": None}]}
        full_config = {"eval": {"prompts": [{"prompt": "配置", "reference": None}]}}
        result = _resolve_eval_prompts(eval_config, full_config)
        assert len(result) == 1
        assert result[0]["prompt"] == "显式"

    def test_string_prompts_format(self):
        """支持纯字符串格式的 prompts。"""
        eval_config = {"prompts": ["hello", "world"]}
        result = _resolve_eval_prompts(eval_config, None)
        assert len(result) == 2
        assert result[0]["prompt"] == "hello"
        assert result[0]["reference"] is None

    def test_empty_prompts_falls_back_to_default(self):
        """空 prompts 列表兜底为默认。"""
        eval_config = {"prompts": []}
        result = _resolve_eval_prompts(eval_config, None)
        assert len(result) == 4  # 默认 4 条


# ===========================================================================
# 3. train() 集成测试：默认 eval_after=True
# ===========================================================================


class TestTrainAutoEval:
    """train() 自动评估集成测试。"""

    def test_train_default_eval_after_true(self, tmp_path):
        """train() 默认 eval_after=True，result 应含 eval_result。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "1+1=", "completion": "2"},
            max_steps_override=3,
            quiet=True,
            eval_config={"max_new_tokens": 8},
        )
        # eval_result 应存在
        assert "eval_result" in result
        assert result["eval_result"] is not None
        # 验证 eval_result 结构
        eval_result = result["eval_result"]
        assert "prompts" in eval_result
        assert "generations" in eval_result
        assert "references" in eval_result
        assert "scores" in eval_result
        assert "avg_length" in eval_result
        assert "has_eos_ratio" in eval_result
        # 默认 4 条 prompt
        assert len(eval_result["prompts"]) == 4
        assert len(eval_result["generations"]) == 4

    def test_train_eval_after_false_skips_eval(self, tmp_path):
        """train() eval_after=False 时跳过评估。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "1+1=", "completion": "2"},
            max_steps_override=3,
            quiet=True,
            eval_after=False,
        )
        # eval_after=False 时 eval_result 应为 None 或不存在
        assert result.get("eval_result") is None
        # 训练结果仍然完整
        assert "best_val_loss" in result
        assert "full_model_path" in result

    def test_train_eval_result_dict_structure(self, tmp_path):
        """eval_result dict 结构正确（所有必需字段）。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_config={"max_new_tokens": 8},
        )
        eval_result = result["eval_result"]
        assert eval_result is not None

        # 验证所有必需字段
        required_keys = {
            "prompts", "generations", "references", "scores",
            "avg_length", "has_eos_ratio",
        }
        assert required_keys.issubset(eval_result.keys()), (
            f"缺少字段：{required_keys - set(eval_result.keys())}"
        )

        # scores 应包含 5 个指标
        scores = eval_result["scores"]
        for metric in ("exact_match", "prefix_accuracy", "char_f1",
                        "bleu", "rouge_l"):
            assert metric in scores, f"scores 缺少指标 {metric}"
            assert isinstance(scores[metric], (int, float))

        # avg_length 和 has_eos_ratio 应为数值
        assert isinstance(eval_result["avg_length"], (int, float))
        assert isinstance(eval_result["has_eos_ratio"], (int, float))
        assert 0.0 <= eval_result["has_eos_ratio"] <= 1.0

    def test_train_eval_with_custom_prompts(self, tmp_path):
        """eval_config 自定义 prompts 生效。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        custom_prompts = [
            {"prompt": "测试A", "reference": "结果A"},
            {"prompt": "测试B", "reference": None},
        ]
        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_config={
                "prompts": custom_prompts,
                "max_new_tokens": 8,
            },
        )
        eval_result = result["eval_result"]
        assert eval_result is not None
        assert len(eval_result["prompts"]) == 2
        assert eval_result["prompts"][0] == "测试A"
        assert eval_result["prompts"][1] == "测试B"
        # 1 条有 reference → n_samples=1
        assert eval_result["scores"]["n_samples"] == 1


# ===========================================================================
# 4. _auto_evaluate 单元测试
# ===========================================================================


class TestAutoEvaluateUnit:
    """_auto_evaluate 直接调用测试。"""

    def test_auto_evaluate_basic(self, tmp_path):
        """_auto_evaluate 基本功能：返回正确结构的 dict。"""
        # 先训练一个模型
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        train_result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_after=False,  # 训练时不评估
        )
        model_path = train_result["full_model_path"]
        assert os.path.exists(model_path), f"模型文件不存在：{model_path}"

        # 加载 full config
        from verse_infra.verse_trainer.trainer import _load_full_config
        full_cfg = _load_full_config(cfg_path)

        # 加载 tokenizer
        from verse_infra.verse_trainer.trainer import _load_tokenizer, _resolve_path
        save_dir = _resolve_path(str(tmp_path), "checkpoints")
        tok = _load_tokenizer(full_cfg.get("tokenizer", {}), str(tmp_path), save_dir)
        vocab_size = len(tok)

        # 直接调用 _auto_evaluate
        eval_result = _auto_evaluate(
            model_path=model_path,
            config=full_cfg,
            tokenizer=tok,
            vocab_size=vocab_size,
            eval_config={"max_new_tokens": 8},
            base_dir=str(tmp_path),
            quiet=True,
        )

        # 验证结构
        assert "prompts" in eval_result
        assert "generations" in eval_result
        assert "references" in eval_result
        assert "scores" in eval_result
        assert "avg_length" in eval_result
        assert "has_eos_ratio" in eval_result
        assert len(eval_result["prompts"]) == 4
        assert len(eval_result["generations"]) == 4

    def test_auto_evaluate_no_reference(self, tmp_path):
        """_auto_evaluate 无 reference 时只记录质量（scores n_samples=0）。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        train_result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_after=False,
        )
        model_path = train_result["full_model_path"]

        from verse_infra.verse_trainer.trainer import (
            _load_full_config, _load_tokenizer, _resolve_path,
        )
        full_cfg = _load_full_config(cfg_path)
        save_dir = _resolve_path(str(tmp_path), "checkpoints")
        tok = _load_tokenizer(full_cfg.get("tokenizer", {}), str(tmp_path), save_dir)

        # 所有 prompt 都无 reference
        eval_config = {
            "prompts": [
                {"prompt": "测试1", "reference": None},
                {"prompt": "测试2", "reference": None},
            ],
            "max_new_tokens": 8,
        }
        eval_result = _auto_evaluate(
            model_path=model_path,
            config=full_cfg,
            tokenizer=tok,
            vocab_size=len(tok),
            eval_config=eval_config,
            base_dir=str(tmp_path),
            quiet=True,
        )

        # 无 reference → 不打分
        assert eval_result["scores"]["n_samples"] == 0
        assert eval_result["scores"]["per_sample"] == []
        # 指标值应为 0.0
        for metric in ("exact_match", "prefix_accuracy", "char_f1",
                        "bleu", "rouge_l"):
            assert eval_result["scores"][metric] == 0.0
        # 但仍然记录生成质量
        assert eval_result["avg_length"] >= 0
        assert 0.0 <= eval_result["has_eos_ratio"] <= 1.0

    def test_auto_evaluate_with_reference(self, tmp_path):
        """_auto_evaluate 有 reference 时计算打分。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        train_result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_after=False,
        )
        model_path = train_result["full_model_path"]

        from verse_infra.verse_trainer.trainer import (
            _load_full_config, _load_tokenizer, _resolve_path,
        )
        full_cfg = _load_full_config(cfg_path)
        save_dir = _resolve_path(str(tmp_path), "checkpoints")
        tok = _load_tokenizer(full_cfg.get("tokenizer", {}), str(tmp_path), save_dir)

        # 有 reference 的 prompt
        eval_config = {
            "prompts": [
                {"prompt": "1+1=", "reference": "2"},
                {"prompt": "你好", "reference": "世界"},
            ],
            "max_new_tokens": 8,
        }
        eval_result = _auto_evaluate(
            model_path=model_path,
            config=full_cfg,
            tokenizer=tok,
            vocab_size=len(tok),
            eval_config=eval_config,
            base_dir=str(tmp_path),
            quiet=True,
        )

        # 有 reference → 打分
        assert eval_result["scores"]["n_samples"] == 2
        assert len(eval_result["scores"]["per_sample"]) == 2
        # 5 个指标都应存在且为数值
        for metric in ("exact_match", "prefix_accuracy", "char_f1",
                        "bleu", "rouge_l"):
            assert metric in eval_result["scores"]
            assert 0.0 <= eval_result["scores"][metric] <= 1.0

    def test_auto_evaluate_quiet_mode(self, tmp_path, capsys):
        """quiet 模式只打印打分汇总。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        train_result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_after=False,
        )
        model_path = train_result["full_model_path"]

        from verse_infra.verse_trainer.trainer import (
            _load_full_config, _load_tokenizer, _resolve_path,
        )
        full_cfg = _load_full_config(cfg_path)
        save_dir = _resolve_path(str(tmp_path), "checkpoints")
        tok = _load_tokenizer(full_cfg.get("tokenizer", {}), str(tmp_path), save_dir)

        _auto_evaluate(
            model_path=model_path,
            config=full_cfg,
            tokenizer=tok,
            vocab_size=len(tok),
            eval_config={"max_new_tokens": 8},
            base_dir=str(tmp_path),
            quiet=True,
        )
        captured = capsys.readouterr()
        # quiet 模式不应打印每条 prompt 的详细输出
        assert "[prompt]" not in captured.out
        # 但应打印汇总行
        assert "[auto-eval]" in captured.out


# ===========================================================================
# 5. CLI 选项测试
# ===========================================================================


class TestCLIOptions:
    """--no-eval 和 --eval-prompts CLI 选项解析。"""

    def test_no_eval_default_false(self):
        """--no-eval 默认为 False（即默认评估）。"""
        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args(["--config", "dummy.yml"])
        assert args.no_eval is False

    def test_no_eval_flag(self):
        """--no-eval 设置为 True。"""
        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args(["--config", "dummy.yml", "--no-eval"])
        assert args.no_eval is True

    def test_eval_prompts_default_none(self):
        """--eval-prompts 默认为 None。"""
        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args(["--config", "dummy.yml"])
        assert args.eval_prompts is None

    def test_eval_prompts_file_path(self, tmp_path):
        """--eval-prompts 指定 JSON 文件路径。"""
        prompts_file = tmp_path / "prompts.json"
        prompts_data = [
            {"prompt": "测试", "reference": "结果"},
        ]
        prompts_file.write_text(
            json.dumps(prompts_data, ensure_ascii=False),
            encoding="utf-8",
        )

        from verse_infra.verse_trainer.cli import _build_train_parser
        parser = _build_train_parser()
        args = parser.parse_args([
            "--config", "dummy.yml",
            "--eval-prompts", str(prompts_file),
        ])
        assert args.eval_prompts == str(prompts_file)


# ===========================================================================
# 6. 评估报告打印测试
# ===========================================================================


class TestFormatEvalReport:
    """format_eval_report 报告格式化。"""

    def test_report_with_scores(self):
        """有打分时报告包含指标。"""
        eval_result = {
            "prompts": ["1+1=", "你好"],
            "generations": ["1+1=2", "你好世界"],
            "references": ["2", "世界"],
            "scores": {
                "exact_match": 0.5,
                "prefix_accuracy": 0.75,
                "char_f1": 0.8,
                "bleu": 0.3,
                "rouge_l": 0.6,
                "n_samples": 2,
                "per_sample": [],
            },
            "avg_length": 5.0,
            "has_eos_ratio": 0.5,
        }
        report = format_eval_report(eval_result)
        assert isinstance(report, str)
        assert "训练后评估报告" in report
        assert "exact_match" in report
        assert "prefix_accuracy" in report
        assert "char_f1" in report
        assert "bleu" in report
        assert "rouge_l" in report
        assert "avg_length" in report
        assert "has_eos_ratio" in report
        assert "样本数" in report

    def test_report_no_scores(self):
        """无打分时报告标注无 reference。"""
        eval_result = {
            "prompts": ["测试"],
            "generations": ["测试输出"],
            "references": [None],
            "scores": {},
            "avg_length": 3.0,
            "has_eos_ratio": 0.0,
        }
        report = format_eval_report(eval_result)
        assert "无打分" in report or "无 reference" in report

    def test_report_invalid_input(self):
        """无效输入返回错误信息。"""
        report = format_eval_report("not a dict")
        assert "无效" in report


# ===========================================================================
# 7. evaluate_from_train_result 便捷函数
# ===========================================================================


class TestEvaluateFromTrainResult:
    """evaluate_from_train_result 便捷函数。"""

    def test_reuses_existing_eval_result(self, tmp_path, capsys):
        """train_result 已含 eval_result 时直接复用。"""
        # 先训练（带自动评估）
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        train_result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_config={"max_new_tokens": 8},
        )
        assert train_result["eval_result"] is not None

        # 调用 evaluate_from_train_result 应直接返回已有的 eval_result
        result = evaluate_from_train_result(
            train_result, quiet=False
        )
        assert result is train_result["eval_result"]
        # 应打印报告
        captured = capsys.readouterr()
        assert "训练后评估报告" in captured.out

    def test_quiet_mode_no_print(self, tmp_path, capsys):
        """quiet 模式不打印报告。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=3)
        train_result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "hi", "completion": "hello"},
            max_steps_override=3,
            quiet=True,
            eval_config={"max_new_tokens": 8},
        )
        evaluate_from_train_result(train_result, quiet=True)
        captured = capsys.readouterr()
        # quiet 模式不应打印格式化报告
        assert "训练后评估报告" not in captured.out

    def test_raises_when_no_model_path(self):
        """无 full_model_path 且无 eval_result 时报错。"""
        train_result = {"best_val_loss": 1.0}  # 无 full_model_path / eval_result
        with pytest.raises(ValueError, match="full_model_path"):
            evaluate_from_train_result(train_result, config_path="dummy.yml")
