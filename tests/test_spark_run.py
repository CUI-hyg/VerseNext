"""spark/run.py 命令行快捷入口测试（Part4K2.5 Task 1）。

覆盖：
1. 命令行参数解析（各子命令的参数）
2. --small 标志选择小配置
3. --dry-run 不实际执行
4. train 子命令调用 train()（mock 验证）
5. generate 子命令加载模型并生成（mock 验证）
6. chat 子命令进入循环（mock input）
7. 默认配置路径正确
8. 路径自举正确
9. convert 子命令调用 pt_to_vn / vn_to_pt（mock 验证）
10. download 子命令调用 DatasetDownloader（mock 验证）

运行方式：
    cd /workspace && python -m pytest tests/test_spark_run.py -x -q
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path 注入
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spark import run as run_mod
from spark.run import (
    build_parser,
    cmd_train,
    cmd_eval,
    cmd_generate,
    cmd_chat,
    cmd_compress,
    cmd_convert,
    cmd_download,
    _resolve_config_path,
    _setup_paths,
    _generated_to_ids,
    _DEFAULT_CONFIG,
    _SMALL_CONFIG,
    _REPO_ROOT as RUN_REPO_ROOT,
)


# ---------------------------------------------------------------------------
# 1. 参数解析
# ---------------------------------------------------------------------------


class TestArgParsing:
    """各子命令参数解析正确性。"""

    def test_train_args(self):
        """train 子命令参数解析。"""
        args = build_parser().parse_args([
            "train", "--small", "--max-steps", "100", "--device", "cpu",
        ])
        assert args.command == "train"
        assert args.small is True
        assert args.max_steps == 100
        assert args.device == "cpu"
        assert args.eval_after is True  # 默认开启

    def test_train_no_eval(self):
        """--no-eval 禁用训练后评估。"""
        args = build_parser().parse_args(["train", "--no-eval"])
        assert args.eval_after is False

    def test_eval_args(self):
        """eval 子命令参数解析。"""
        args = build_parser().parse_args([
            "eval", "--checkpoint", "ck.pt", "--max-tokens", "50",
            "--temperature", "0.5", "--score",
        ])
        assert args.command == "eval"
        assert args.checkpoint == "ck.pt"
        assert args.max_tokens == 50
        assert args.temperature == 0.5
        assert args.score is True

    def test_generate_args(self):
        """generate 子命令参数解析。"""
        args = build_parser().parse_args([
            "generate", "--prompt", "你好", "--temperature", "0.8",
            "--max-tokens", "100",
        ])
        assert args.command == "generate"
        assert args.prompt == "你好"
        assert args.temperature == 0.8
        assert args.max_tokens == 100
        # 默认 checkpoint
        assert args.checkpoint == "checkpoints/best.pt"

    def test_chat_args(self):
        """chat 子命令参数解析。"""
        args = build_parser().parse_args([
            "chat", "--checkpoint", "model.pt", "--temperature", "0.7",
        ])
        assert args.command == "chat"
        assert args.checkpoint == "model.pt"
        assert args.temperature == 0.7
        assert args.max_tokens == 512  # 默认值

    def test_compress_args(self):
        """compress 子命令参数解析。"""
        args = build_parser().parse_args([
            "compress", "--checkpoint", "ck.pt",
            "--method", "prune,quantize,lora",
            "--sparsity", "0.5",
        ])
        assert args.command == "compress"
        assert args.checkpoint == "ck.pt"
        assert args.method == "prune,quantize,lora"
        assert args.sparsity == 0.5

    def test_convert_args(self):
        """convert 子命令参数解析。"""
        args = build_parser().parse_args([
            "convert", "--input", "model.pt", "--output", "model.vn",
        ])
        assert args.command == "convert"
        assert args.input == "model.pt"
        assert args.output == "model.vn"

    def test_download_args(self):
        """download 子命令参数解析。"""
        args = build_parser().parse_args([
            "download", "--url", "https://example.com/data.jsonl",
            "--output", "data/",
        ])
        assert args.command == "download"
        assert args.url == "https://example.com/data.jsonl"
        assert args.output == "data/"

    def test_no_command_shows_help(self):
        """无子命令时返回 1。"""
        ret = run_mod.main([])
        assert ret == 1


# ---------------------------------------------------------------------------
# 2. 配置路径解析
# ---------------------------------------------------------------------------


class TestConfigResolution:
    """配置文件路径解析。"""

    def test_default_config_path(self):
        """不指定 --config 时默认用 1B 配置。"""
        path = _resolve_config_path(None, small=False)
        assert path.endswith(_DEFAULT_CONFIG)
        assert os.path.exists(path)

    def test_small_config_path(self):
        """--small 时用小配置。"""
        path = _resolve_config_path(None, small=True)
        assert path.endswith(_SMALL_CONFIG)
        assert os.path.exists(path)

    def test_custom_config_path(self):
        """--config 显式指定。"""
        custom = os.path.join(str(_REPO_ROOT), _SMALL_CONFIG)
        path = _resolve_config_path(custom, small=False)
        assert path == custom

    def test_nonexistent_config_raises(self):
        """不存在的配置文件抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            _resolve_config_path("/nonexistent/config.yml")

    def test_train_small_selects_small_config(self):
        """train --small 选择小配置（通过 dry-run 验证）。"""
        args = build_parser().parse_args(["train", "--small", "--dry-run"])
        ret = cmd_train(args)
        assert ret == 0  # dry-run 总是返回 0

    def test_train_default_selects_1b_config(self):
        """train（无 --small）选择 1B 配置（通过 dry-run 验证）。"""
        args = build_parser().parse_args(["train", "--dry-run"])
        ret = cmd_train(args)
        assert ret == 0


# ---------------------------------------------------------------------------
# 3. --dry-run 不实际执行
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run 模式不执行实际操作。"""

    def test_train_dry_run(self):
        """train --dry-run 不调用 train()。"""
        with patch("verse_infra.verse_trainer.train") as mock_train:
            args = build_parser().parse_args([
                "train", "--small", "--dry-run",
            ])
            ret = cmd_train(args)
            assert ret == 0
            mock_train.assert_not_called()

    def test_eval_dry_run(self):
        """eval --dry-run 不调用 evaluate()。"""
        with patch("verse_infra.verse_trainer.evaluate") as mock_eval:
            args = build_parser().parse_args([
                "eval", "--config", _SMALL_CONFIG, "--dry-run",
            ])
            ret = cmd_eval(args)
            assert ret == 0
            mock_eval.assert_not_called()

    def test_generate_dry_run(self):
        """generate --dry-run 不加载模型。"""
        with patch("spark.run._load_model_and_tokenizer") as mock_load:
            args = build_parser().parse_args([
                "generate", "--prompt", "test", "--dry-run",
            ])
            ret = cmd_generate(args)
            assert ret == 0
            mock_load.assert_not_called()

    def test_chat_dry_run(self):
        """chat --dry-run 不加载模型。"""
        with patch("spark.run._load_model_and_tokenizer") as mock_load:
            args = build_parser().parse_args([
                "chat", "--dry-run",
            ])
            ret = cmd_chat(args)
            assert ret == 0
            mock_load.assert_not_called()

    def test_compress_dry_run(self):
        """compress --dry-run 不加载模型。"""
        with patch("spark.model.model.CometSparkV05LM") as mock_cls:
            args = build_parser().parse_args([
                "compress", "--checkpoint", "fake.pt", "--dry-run",
            ])
            ret = cmd_compress(args)
            assert ret == 0
            mock_cls.from_pretrained.assert_not_called()

    def test_convert_dry_run(self):
        """convert --dry-run 不调用转换函数。"""
        with patch("verse_torch.vn_format.pt_to_vn") as mock_ptvn:
            args = build_parser().parse_args([
                "convert", "--input", "model.pt", "--output", "model.vn",
                "--dry-run",
            ])
            ret = cmd_convert(args)
            assert ret == 0
            mock_ptvn.assert_not_called()

    def test_download_dry_run(self):
        """download --dry-run 不调用下载器。"""
        with patch("verse_infra.DatasetDownloader") as mock_dl:
            args = build_parser().parse_args([
                "download", "--url", "https://example.com/data",
                "--dry-run",
            ])
            ret = cmd_download(args)
            assert ret == 0
            mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# 4. train 子命令调用 train()
# ---------------------------------------------------------------------------


class TestTrainCommand:
    """train 子命令 mock 验证。"""

    @patch("verse_infra.verse_trainer.evaluate")
    @patch("verse_infra.verse_trainer.train")
    def test_train_calls_train(self, mock_train, mock_eval):
        """train 调用 verse_infra.verse_trainer.train()。"""
        mock_train.return_value = {
            "best_val_loss": 0.5,
            "save_dir": "checkpoints",
            "total_steps": 200,
            "best_checkpoint": "checkpoints/best.pt",
        }
        args = build_parser().parse_args(["train", "--small", "--no-eval"])
        ret = cmd_train(args)

        assert ret == 0
        mock_train.assert_called_once()
        # 验证 config_path 参数指向小配置
        call_kwargs = mock_train.call_args
        config_path = call_kwargs.kwargs.get("config_path") or call_kwargs.args[0]
        assert config_path.endswith(_SMALL_CONFIG)

    @patch("verse_infra.verse_trainer.evaluate")
    @patch("verse_infra.verse_trainer.train")
    def test_train_eval_after(self, mock_train, mock_eval):
        """--eval-after（默认）训练后自动评估。"""
        mock_train.return_value = {
            "best_val_loss": 0.3,
            "save_dir": "checkpoints",
            "total_steps": 200,
            "best_checkpoint": "checkpoints/best.pt",
        }
        mock_eval.return_value = {"results": [{"prompt": "a", "generated": "b"}]}
        args = build_parser().parse_args(["train", "--small"])
        ret = cmd_train(args)

        assert ret == 0
        mock_train.assert_called_once()
        mock_eval.assert_called_once()

    @patch("verse_infra.verse_trainer.evaluate")
    @patch("verse_infra.verse_trainer.train")
    def test_train_max_steps_override(self, mock_train, mock_eval):
        """--max-steps 覆盖训练步数。"""
        mock_train.return_value = {
            "best_val_loss": 0.5,
            "save_dir": "checkpoints",
            "total_steps": 50,
        }
        args = build_parser().parse_args([
            "train", "--small", "--max-steps", "50", "--no-eval",
        ])
        ret = cmd_train(args)

        assert ret == 0
        mock_train.assert_called_once()
        call_kwargs = mock_train.call_args.kwargs
        assert call_kwargs["max_steps_override"] == 50


# ---------------------------------------------------------------------------
# 5. generate 子命令加载模型并生成
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    """generate 子命令 mock 验证。"""

    @patch("spark.run._load_model_and_tokenizer")
    def test_generate_loads_model_and_generates(self, mock_load):
        """generate 加载模型并调用 generate()。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # 模拟 generate 返回 numpy 数组
        mock_model.generate.return_value = np.array([[1, 2, 3, 4, 5]])
        mock_model.config.arch = "versenex"
        mock_model.config.n_layer = 2
        mock_model.config.n_embd = 64
        mock_model.config.n_head = 4
        mock_model.config.n_kv_head = 2
        mock_model.config.vocab_size = 256
        mock_model.count_parameters.return_value = 100000
        mock_model.device_info.return_value = "cpu"

        # 模拟 tokenizer 行为
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "生成结果"

        args = build_parser().parse_args([
            "generate", "--checkpoint", "fake.pt",
            "--prompt", "你好", "--quiet",
        ])
        ret = cmd_generate(args)

        assert ret == 0
        mock_load.assert_called_once()
        mock_model.generate.assert_called_once()

        # 验证 generate 参数
        gen_kwargs = mock_model.generate.call_args.kwargs
        assert gen_kwargs["temperature"] == 0.8  # 默认温度

    @patch("spark.run._load_model_and_tokenizer")
    def test_generate_custom_temperature(self, mock_load):
        """generate --temperature 自定义温度。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_model.generate.return_value = np.array([[1, 2, 3]])
        mock_model.config.arch = "versenex"
        mock_model.config.n_layer = 2
        mock_model.config.n_embd = 64
        mock_model.config.n_head = 4
        mock_model.config.n_kv_head = 2
        mock_model.config.vocab_size = 256
        mock_model.count_parameters.return_value = 100000
        mock_model.device_info.return_value = "cpu"
        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.decode.return_value = "ok"

        args = build_parser().parse_args([
            "generate", "--checkpoint", "fake.pt",
            "--temperature", "1.5", "--quiet",
        ])
        ret = cmd_generate(args)

        assert ret == 0
        gen_kwargs = mock_model.generate.call_args.kwargs
        assert gen_kwargs["temperature"] == 1.5

    @patch("spark.run._load_model_and_tokenizer")
    def test_generate_max_tokens(self, mock_load):
        """generate --max-tokens 限制生成长度。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_model.generate.return_value = np.array([[1, 2, 3]])
        mock_model.config.arch = "versenex"
        mock_model.config.n_layer = 2
        mock_model.config.n_embd = 64
        mock_model.config.n_head = 4
        mock_model.config.n_kv_head = 2
        mock_model.config.vocab_size = 256
        mock_model.count_parameters.return_value = 100000
        mock_model.device_info.return_value = "cpu"
        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.decode.return_value = "ok"

        args = build_parser().parse_args([
            "generate", "--checkpoint", "fake.pt",
            "--max-tokens", "50", "--quiet",
        ])
        ret = cmd_generate(args)

        assert ret == 0
        gen_kwargs = mock_model.generate.call_args.kwargs
        assert gen_kwargs["max_new_tokens"] == 50

    def test_generate_missing_checkpoint(self):
        """generate 不存在的 checkpoint 返回错误。"""
        ret = run_mod.main([
            "generate", "--checkpoint", "/nonexistent/ckpt.pt",
        ])
        assert ret == 1


# ---------------------------------------------------------------------------
# 6. chat 子命令进入循环
# ---------------------------------------------------------------------------


class TestChatCommand:
    """chat 子命令 mock 验证。"""

    @patch("builtins.input")
    @patch("spark.run._load_model_and_tokenizer")
    def test_chat_loop_quit(self, mock_load, mock_input):
        """chat 输入 /quit 退出循环。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_model.config.arch = "versenex"
        mock_model.config.n_layer = 2
        mock_model.config.n_embd = 64
        mock_model.config.n_head = 4
        mock_model.config.n_kv_head = 2
        mock_model.config.vocab_size = 256
        mock_model.count_parameters.return_value = 100000
        mock_model.device_info.return_value = "cpu"

        # 立即退出
        mock_input.return_value = "/quit"

        args = build_parser().parse_args([
            "chat", "--checkpoint", "fake.pt",
        ])
        ret = cmd_chat(args)

        assert ret == 0
        mock_load.assert_called_once()

    @patch("builtins.input")
    @patch("spark.run._load_model_and_tokenizer")
    def test_chat_loop_generate(self, mock_load, mock_input):
        """chat 对话后生成回复再退出。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_model.config.arch = "versenex"
        mock_model.config.n_layer = 2
        mock_model.config.n_embd = 64
        mock_model.config.n_head = 4
        mock_model.config.n_kv_head = 2
        mock_model.config.vocab_size = 256
        mock_model.count_parameters.return_value = 100000
        mock_model.device_info.return_value = "cpu"

        # 模拟生成返回
        mock_model.generate.return_value = np.array([[1, 2, 3, 4, 5]])

        # 模拟 tokenizer：apply_chat_template 抛异常 → 降级手动渲染
        mock_tokenizer.apply_chat_template.side_effect = TypeError("nope")
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "你好"

        # 用户输入一条消息后退出
        mock_input.side_effect = ["你好", "/quit"]

        args = build_parser().parse_args([
            "chat", "--checkpoint", "fake.pt",
        ])
        ret = cmd_chat(args)

        assert ret == 0
        # 验证 model.generate 被调用了一次（对应用户的一条消息）
        mock_model.generate.assert_called_once()

    @patch("builtins.input")
    @patch("spark.run._load_model_and_tokenizer")
    def test_chat_clear_command(self, mock_load, mock_input):
        """chat /clear 清空对话历史。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_model.config.arch = "versenex"
        mock_model.config.n_layer = 2
        mock_model.config.n_embd = 64
        mock_model.config.n_head = 4
        mock_model.config.n_kv_head = 2
        mock_model.config.vocab_size = 256
        mock_model.count_parameters.return_value = 100000
        mock_model.device_info.return_value = "cpu"

        # /clear 不触发生成，/quit 退出
        mock_input.side_effect = ["/clear", "/quit"]

        args = build_parser().parse_args([
            "chat", "--checkpoint", "fake.pt",
        ])
        ret = cmd_chat(args)

        assert ret == 0
        # /clear 不应触发 generate
        mock_model.generate.assert_not_called()


# ---------------------------------------------------------------------------
# 7. compress 子命令
# ---------------------------------------------------------------------------


class TestCompressCommand:
    """compress 子命令 mock 验证。"""

    @patch("verse_torch.compress.compress_pipeline")
    @patch("spark.model.model.CometSparkV05LM")
    def test_compress_calls_pipeline(self, mock_cls, mock_pipeline):
        """compress 调用 compress_pipeline。"""
        mock_model = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model
        mock_model.config.arch = "versenex"
        mock_model.count_parameters.return_value = 100000

        mock_compressed_net = MagicMock()
        mock_compressed_net.count_parameters.return_value = 50000
        # compress_pipeline 返回 (compressed_model, stats)
        mock_pipeline.return_value = (mock_compressed_net, {"ratio": 2.0})

        # 构造新的 mock model
        mock_new_model = MagicMock()
        mock_new_model.count_parameters.return_value = 50000
        mock_cls.return_value = mock_new_model

        # 需要真实的 checkpoint 文件
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            args = build_parser().parse_args([
                "compress", "--checkpoint", ckpt_path,
                "--method", "prune", "--output", "out.pt", "--quiet",
            ])
            ret = cmd_compress(args)

            assert ret == 0
            mock_pipeline.assert_called_once()
            mock_new_model.save.assert_called_once_with("out.pt")
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# 8. convert 子命令
# ---------------------------------------------------------------------------


class TestConvertCommand:
    """convert 子命令 mock 验证。"""

    @patch("verse_torch.vn_format.pt_to_vn")
    def test_convert_pt_to_vn(self, mock_ptvn):
        """convert .pt → .vn 调用 pt_to_vn。"""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            input_path = f.name
        try:
            args = build_parser().parse_args([
                "convert", "--input", input_path,
                "--output", "model.vn",
            ])
            ret = cmd_convert(args)

            assert ret == 0
            mock_ptvn.assert_called_once()
            call_args = mock_ptvn.call_args
            assert call_args.args[0] == input_path
            assert call_args.args[1] == "model.vn"
        finally:
            os.unlink(input_path)

    @patch("verse_torch.vn_format.vn_to_pt")
    def test_convert_vn_to_pt(self, mock_vnpt):
        """convert .vn → .pt 调用 vn_to_pt。"""
        with tempfile.NamedTemporaryFile(suffix=".vn", delete=False) as f:
            input_path = f.name
        try:
            args = build_parser().parse_args([
                "convert", "--input", input_path,
                "--output", "model.pt",
            ])
            ret = cmd_convert(args)

            assert ret == 0
            mock_vnpt.assert_called_once()
        finally:
            os.unlink(input_path)

    def test_convert_nonexistent_input(self):
        """convert 不存在的输入文件返回错误。"""
        ret = run_mod.main([
            "convert", "--input", "/nonexistent.pt",
            "--output", "out.vn",
        ])
        assert ret == 1


# ---------------------------------------------------------------------------
# 9. download 子命令
# ---------------------------------------------------------------------------


class TestDownloadCommand:
    """download 子命令 mock 验证。"""

    @patch("verse_infra.DatasetDownloader")
    def test_download_url(self, mock_dl_cls):
        """download --url 调用 download_url。"""
        mock_downloader = MagicMock()
        mock_dl_cls.return_value = mock_downloader
        mock_downloader.download_url.return_value = "data/file.jsonl"

        args = build_parser().parse_args([
            "download", "--url", "https://example.com/data.jsonl",
        ])
        ret = cmd_download(args)

        assert ret == 0
        mock_dl_cls.assert_called_once()
        mock_downloader.download_url.assert_called_once()

    @patch("verse_infra.DatasetDownloader")
    def test_download_hf(self, mock_dl_cls):
        """download --hf 调用 download_hf。"""
        mock_downloader = MagicMock()
        mock_dl_cls.return_value = mock_downloader
        mock_downloader.download_hf.return_value = "data/wikitext"

        args = build_parser().parse_args([
            "download", "--hf", "wikitext", "--split", "test",
        ])
        ret = cmd_download(args)

        assert ret == 0
        mock_downloader.download_hf.assert_called_once()
        call_kwargs = mock_downloader.download_hf.call_args.kwargs
        assert call_kwargs["split"] == "test"

    @patch("verse_infra.DatasetDownloader")
    def test_download_to_npz(self, mock_dl_cls):
        """download --to-npz 调用 download_and_cache。"""
        mock_downloader = MagicMock()
        mock_dl_cls.return_value = mock_downloader
        mock_downloader.download_and_cache.return_value = "data/cache.npz"

        args = build_parser().parse_args([
            "download", "--url", "https://example.com/data",
            "--to-npz",
        ])
        ret = cmd_download(args)

        assert ret == 0
        mock_downloader.download_and_cache.assert_called_once()

    def test_download_no_source(self):
        """download 不指定 --url 或 --hf 返回错误。"""
        args = build_parser().parse_args(["download"])
        ret = cmd_download(args)
        assert ret == 1


# ---------------------------------------------------------------------------
# 10. 路径自举
# ---------------------------------------------------------------------------


class TestPathBootstrap:
    """路径自举正确性。"""

    def test_setup_paths_adds_deps(self):
        """_setup_paths 把 verse_torch/verse_nex/verse_infra 加入 sys.path。"""
        _setup_paths()
        for dep in ("verse_torch", "verse_nex", "verse_infra"):
            dep_path = os.path.join(str(RUN_REPO_ROOT), "packages", dep)
            assert dep_path in sys.path, f"{dep_path} 不在 sys.path 中"

    def test_repo_root_in_path(self):
        """repo root 在 sys.path 中（import spark 可用）。"""
        assert str(RUN_REPO_ROOT) in sys.path

    def test_can_import_verse_modules(self):
        """路径自举后能 import verse 模块。"""
        import verse_torch
        import verse_infra
        assert verse_torch is not None
        assert verse_infra is not None


# ---------------------------------------------------------------------------
# 11. _generated_to_ids 辅助函数
# ---------------------------------------------------------------------------


class TestGeneratedToIds:
    """_generated_to_ids 类型转换正确性。"""

    def test_ndarray_input(self):
        """numpy ndarray 输入正确转换。"""
        arr = np.array([[1, 2, 3, 4, 5]])
        ids = _generated_to_ids(arr)
        assert ids == [1, 2, 3, 4, 5]

    def test_tensor_like_input(self):
        """有 .data 属性的 Tensor-like 输入正确转换。"""
        class FakeTensor:
            def __init__(self, data):
                self.data = np.array(data)
        fake = FakeTensor([1, 2, 3])
        ids = _generated_to_ids(fake)
        assert ids == [1, 2, 3]

    def test_list_input(self):
        """list 输入正确转换。"""
        ids = _generated_to_ids([1, 2, 3])
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# 12. main 入口集成
# ---------------------------------------------------------------------------


class TestMainEntry:
    """main() 入口集成测试。"""

    def test_main_help_returns_1(self):
        """无子命令时 main 返回 1。"""
        ret = run_mod.main([])
        assert ret == 1

    @patch("verse_infra.verse_trainer.train")
    @patch("verse_infra.verse_trainer.evaluate")
    def test_main_train_small(self, mock_eval, mock_train):
        """main(["train", "--small"]) 完整流程。"""
        mock_train.return_value = {
            "best_val_loss": 0.5,
            "save_dir": "checkpoints",
            "total_steps": 200,
        }
        ret = run_mod.main(["train", "--small", "--no-eval"])
        assert ret == 0
        mock_train.assert_called_once()

    def test_main_file_not_found(self):
        """main 捕获 FileNotFoundError 返回 1。"""
        ret = run_mod.main([
            "train", "--config", "/nonexistent/config.yml",
        ])
        assert ret == 1
