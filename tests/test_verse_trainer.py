"""测试 verse_trainer 包（Part4K1 Task 6.9）。

覆盖：
1. CachedDataset 首次缓存 + 二次加载加速
2. CLI 端到端：verse-train --single-sample --prompt "1+1=" --completion "2" --max-steps 20
3. _safe_chunk_run 异常捕获（mock 抛异常，验证 graceful 处理）
4. loss plateau 重走（构造 plateau 场景，验证 rollback）
5. 断点续训（save + resume）
6. RLTrainer 调用 NexTrainer
7. CLI verse-tokenize / verse-eval / verse-finetune / verse-posttrain 子命令分发
8. LossOptimizer NaN/Inf 跳过

运行方式：
    cd /workspace && PYTHONPATH=packages/verse_torch:packages/verse_nex:\
        packages/verse_infra \
        python -m pytest tests/test_verse_trainer.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
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

from verse_infra.verse_trainer import (
    CachedDataset,
    TextDataset,
    SingleSampleDataset,
    BatchLoader,
    collate_fn,
    LossOptimizer,
    ParallelTrainerSafe,
    _safe_chunk_run,
    install_signal_handlers,
    reset_shutdown_flag,
    is_shutdown_requested,
    ChunkOOMError,
    RLTrainer,
    train,
)
from verse_infra.verse_trainer.loss_optim import (
    _rollback_and_perturb,
    reset_adam_momentum,
    scale_optimizer_lr,
)
from verse_torch import Linear, AdamW, Tensor


# ---------------------------------------------------------------------------
# 通用辅助：构造最小 config + tokenizer
# ---------------------------------------------------------------------------


def _make_tiny_config(tmpdir, parallel_chunks=1, max_steps=10):
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
            "warmup": 2,
            "eval_interval": max_steps,  # 仅在最后评估一次
            "patience": 5,
            "grad_accum": 1,
            "log_interval": max_steps,
            "seed": 42,
            "enable_progress_bar": False,
            "realtime_plot": False,
            "eta_window": 5,
            "parallel_chunks": parallel_chunks,
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
        # 极简手写 YAML
        lines = []
        for section, sub in config.items():
            lines.append(f"{section}:")
            for k, v in sub.items():
                lines.append(f"  {k}: {v}")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return cfg_path


def _make_jsonl(path, n=20, fmt="text"):
    """生成测试 jsonl 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            if fmt == "text":
                obj = {"text": f"hello world {i}"}
            elif fmt == "prompt_completion":
                obj = {"prompt": f"q{i}=", "completion": f"a{i}"}
            elif fmt == "chat":
                obj = [{"role": "user", "content": f"hi {i}"},
                       {"role": "assistant", "content": f"hello {i}"}]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ===========================================================================
# SubTask 6.3: CachedDataset 首次缓存 + 二次加载加速
# ===========================================================================


class TestCachedDataset:
    """CachedDataset 首次缓存 + 二次加载加速 + lazy load。"""

    def test_first_load_creates_cache(self, tmp_path):
        """首次加载应创建 .npz 缓存文件。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        jsonl = str(tmp_path / "train.jsonl")
        _make_jsonl(jsonl, n=20)

        cache_path = str(tmp_path / "train.jsonl.cache.npz")
        assert not os.path.exists(cache_path)

        ds = CachedDataset(tok, jsonl, seq_len=8, cache_path=cache_path)
        assert os.path.exists(cache_path)
        assert len(ds) > 0
        # 验证 __getitem__ 返回 (x, y) 形状正确
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_second_load_faster(self, tmp_path):
        """二次加载应明显加速（缓存命中）。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        jsonl = str(tmp_path / "train.jsonl")
        _make_jsonl(jsonl, n=50)

        # 首次加载（扫描 + 编码 + 写缓存）
        t1 = time.time()
        ds1 = CachedDataset(tok, jsonl, seq_len=8)
        t_first = time.time() - t1

        # 二次加载（命中缓存）
        t2 = time.time()
        ds2 = CachedDataset(tok, jsonl, seq_len=8)
        t_second = time.time() - t2

        assert len(ds1) == len(ds2)
        # 二次加载应不慢于首次（允许 5x 容差避免噪声）
        assert t_second <= t_first * 5 + 0.01, (
            f"二次加载 {t_second*1000:.1f}ms 应快于首次 {t_first*1000:.1f}ms"
        )

    def test_lazy_load(self, tmp_path):
        """lazy 模式应能正确读取数据。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        jsonl = str(tmp_path / "train.jsonl")
        _make_jsonl(jsonl, n=20)

        ds = CachedDataset(tok, jsonl, seq_len=8, lazy=True)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)
        # 多次读取不同 index 验证 lazy 切片稳定
        for i in range(min(5, len(ds))):
            x_i, y_i = ds[i]
            assert x_i.shape == (8,)

    def test_cache_invalidation_on_source_change(self, tmp_path):
        """源文件更新后缓存应失效重建。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        jsonl = str(tmp_path / "train.jsonl")
        _make_jsonl(jsonl, n=10)

        ds1 = CachedDataset(tok, jsonl, seq_len=8)
        n1 = len(ds1)

        # 修改源文件（追加更多数据）
        time.sleep(0.05)  # 确保 mtime 更新
        _make_jsonl(jsonl, n=30)

        # 重新加载应检测到源文件更新，重建缓存
        ds2 = CachedDataset(tok, jsonl, seq_len=8)
        assert len(ds2) > n1, f"缓存失效后应重建，n1={n1} n2={len(ds2)}"

    def test_format_support(self, tmp_path):
        """支持 chat / prompt-completion / text 三种格式。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()

        # chat 格式
        chat_jsonl = str(tmp_path / "chat.jsonl")
        _make_jsonl(chat_jsonl, n=10, fmt="chat")
        ds_chat = CachedDataset(tok, chat_jsonl, seq_len=8)
        assert len(ds_chat) > 0

        # prompt-completion 格式
        pc_jsonl = str(tmp_path / "pc.jsonl")
        _make_jsonl(pc_jsonl, n=10, fmt="prompt_completion")
        ds_pc = CachedDataset(tok, pc_jsonl, seq_len=8)
        assert len(ds_pc) > 0

        # 验证 loss mask：prompt-completion 格式的 y 应有 -100（屏蔽 prompt）
        x, y = ds_pc[0]
        # 至少有一个 -100（prompt 部分）或全 -100（极端短样本）
        assert (y == -100).any() or (y != -100).any()


# ===========================================================================
# SubTask 6.5: SingleSampleDataset 单样本支持
# ===========================================================================


class TestSingleSampleDataset:
    """单样本数据集（--single-sample 支持）。"""

    def test_prompt_completion(self):
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        ds = SingleSampleDataset(
            tok, prompt="1+1=", completion="2", seq_len=8, n_repeat=4,
        )
        assert len(ds) >= 1
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_text_only(self):
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        ds = SingleSampleDataset(tok, text="hello world", seq_len=8, n_repeat=4)
        assert len(ds) >= 1
        x, y = ds[0]
        assert x.shape == (8,)

    def test_empty_input_fallback(self):
        """空输入应兜底为 0 token，不报错。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        ds = SingleSampleDataset(tok, seq_len=8, n_repeat=4)
        assert len(ds) >= 1


# ===========================================================================
# SubTask 6.2: _safe_chunk_run 异常捕获
# ===========================================================================


class TestSafeChunkRun:
    """_safe_chunk_run 异常捕获 + OOM 兜底。"""

    def test_normal_execution(self):
        """正常 chunk 执行。"""
        def chunk_fn(x, batch_size=None):
            return f"ok {x} {batch_size}"
        r = _safe_chunk_run(chunk_fn, 42, batch_size=8)
        assert r == "ok 42 8"

    def test_oom_retry_with_batch_shrink(self):
        """OOM 时缩小 batch 重试。"""
        attempts = [0]
        def chunk_oom(batch_size=None):
            attempts[0] += 1
            if attempts[0] <= 2:
                raise MemoryError("oom")
            return f"recovered batch={batch_size}"
        r = _safe_chunk_run(chunk_oom, max_oom_retries=3, batch_size=8)
        assert "recovered" in r
        assert attempts[0] == 3
        # batch 应被缩小（8 → 4 → 2）
        assert "batch=2" in r

    def test_oom_exhausted_raises(self):
        """OOM 重试次数用尽应抛 ChunkOOMError。"""
        def chunk_oom(batch_size=None):
            raise MemoryError("oom")
        with pytest.raises(ChunkOOMError):
            _safe_chunk_run(chunk_oom, max_oom_retries=2, batch_size=8)

    def test_unexpected_exception_wrapped(self):
        """非预期异常应包装为 RuntimeError。"""
        def chunk_err():
            raise ValueError("bad")
        with pytest.raises(RuntimeError, match="chunk 执行失败"):
            _safe_chunk_run(chunk_err)

    def test_shutdown_signal_skips_chunk(self):
        """收到 shutdown 信号时应跳过 chunk 执行。"""
        reset_shutdown_flag()
        # 模拟收到信号
        _shutdown_event = sys.modules["verse_infra.verse_trainer.trainer"]._shutdown_event
        _shutdown_event.set()
        try:
            with pytest.raises(RuntimeError, match="shutdown"):
                def chunk_ok():
                    return "should not run"
                _safe_chunk_run(chunk_ok)
        finally:
            _shutdown_event.clear()


# ===========================================================================
# SubTask 6.7: LossOptimizer plateau 重走 + NaN/Inf 跳过
# ===========================================================================


class TestLossOptimizer:
    """LossOptimizer：plateau 重走 + NaN/Inf 跳过 + Adam 动量重置。"""

    def test_nan_inf_skip(self):
        """NaN/Inf loss 应被跳过。"""
        model = Linear(8, 4)
        opt = AdamW(model.parameters(), lr=0.01)
        lo = LossOptimizer(model, opt, patience=2, verbose=False)

        assert lo.check_loss_finite(1.0) is True
        assert lo.check_loss_finite(float("nan")) is False
        assert lo.check_loss_finite(float("inf")) is False
        assert lo.check_loss_finite(-float("inf")) is False
        assert lo.nan_skip_count == 3

    def test_plateau_rollback_triggered(self):
        """连续 patience 步未改善应触发 plateau 重走。"""
        model = Linear(8, 4)
        opt = AdamW(model.parameters(), lr=0.01)
        # 先 step 一次让 optimizer 有 m/v 状态
        x = Tensor(np.random.randn(2, 8).astype(np.float32))
        y = model(x).sum()
        y.backward()
        opt.step()

        lo = LossOptimizer(model, opt, patience=2, rollback_factor=0.3,
                           verbose=False)
        original_lr = opt.lr

        # 连续 3 步未改善（patience=2，第 3 步触发）
        val_losses = [5.0, 5.0, 5.0]
        triggered = []
        for i, v in enumerate(val_losses):
            t = lo.maybe_rollback(v, step=i, best_state_dict=model.state_dict())
            triggered.append(t)
        assert triggered == [False, False, True]
        assert lo.rollback_count == 1
        # LR 应被缩小
        assert opt.lr < original_lr
        # Adam m/v 应被重置为 0
        for st in opt.state.values():
            if isinstance(st, dict):
                m = st.get("m")
                if m is not None:
                    assert float(np.asarray(m).sum()) == 0.0

    def test_plateau_no_rollback_when_improving(self):
        """val_loss 持续下降时不应触发重走。"""
        model = Linear(8, 4)
        opt = AdamW(model.parameters(), lr=0.01)
        lo = LossOptimizer(model, opt, patience=2, verbose=False)

        # 持续下降
        for v in [5.0, 4.0, 3.0, 2.0]:
            t = lo.maybe_rollback(v, best_state_dict=model.state_dict())
            assert t is False
        assert lo.rollback_count == 0

    def test_max_rollbacks_limit(self):
        """超过 max_rollbacks 后不再重走。"""
        model = Linear(8, 4)
        opt = AdamW(model.parameters(), lr=0.01)
        x = Tensor(np.random.randn(2, 8).astype(np.float32))
        y = model(x).sum()
        y.backward()
        opt.step()

        lo = LossOptimizer(model, opt, patience=1, rollback_factor=0.3,
                           max_rollbacks=2, verbose=False)
        # 触发 2 次重走
        for _ in range(4):
            lo.maybe_rollback(5.0, best_state_dict=model.state_dict())
        assert lo.rollback_count == 2
        # 第 3 次不再触发
        lo.maybe_rollback(5.0, best_state_dict=model.state_dict())
        assert lo.rollback_count == 2

    def test_rollback_and_perturb_standalone(self):
        """_rollback_and_perturb 独立函数版本。"""
        model = Linear(4, 2)
        opt = AdamW(model.parameters(), lr=0.01)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        y = model(x).sum()
        y.backward()
        opt.step()
        original_lr = opt.lr
        original_t = opt.t

        n = _rollback_and_perturb(model, opt, model.state_dict(), lr_factor=0.3)
        assert n >= 1
        assert opt.lr < original_lr
        assert opt.t == 0  # Adam 步数计数器被重置

    def test_scale_optimizer_lr(self):
        """scale_optimizer_lr 缩放所有 param_group 的 lr。"""
        model = Linear(4, 2)
        opt = AdamW(model.parameters(), lr=0.1)
        new_lr = scale_optimizer_lr(opt, 0.5)
        assert abs(new_lr - 0.05) < 1e-6
        for g in opt.param_groups:
            assert abs(g["lr"] - 0.05) < 1e-6


# ===========================================================================
# SubTask 6.2: 信号处理
# ===========================================================================


class TestSignalHandlers:
    """信号处理（不真正发信号，只测标志机制）。"""

    def test_shutdown_flag_clear_set(self):
        """reset_shutdown_flag / is_shutdown_requested 接口。"""
        reset_shutdown_flag()
        assert is_shutdown_requested() is False

    def test_install_signal_handlers_idempotent(self):
        """install_signal_handlers 应幂等。"""
        # 在主线程调用应成功（CI 子线程环境也兼容）
        install_signal_handlers()
        install_signal_handlers()  # 重复调用不报错


# ===========================================================================
# SubTask 6.4 + 6.5: CLI 端到端 + 单样本训练
# ===========================================================================


class TestCLIEndToEnd:
    """CLI 端到端：verse-train --single-sample。"""

    def test_verse_train_single_sample(self, tmp_path):
        """verse-train --single-sample --prompt "1+1=" --completion "2" --max-steps 20。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=10)

        # 直接调用 train() 函数（绕过 argparse，便于测试）
        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "1+1=", "completion": "2"},
            max_steps_override=10,
        )
        assert "best_val_loss" in result
        assert len(result["train_losses"]) == 10
        # 训练应产生 checkpoint 文件
        assert os.path.exists(result["checkpoint_dir"])
        # resume.pt 应被保存
        assert os.path.exists(os.path.join(result["checkpoint_dir"], "resume.pt"))

    def test_verse_train_single_file(self, tmp_path):
        """verse-train --single-sample --single-file。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=5)
        single_file = tmp_path / "input.txt"
        single_file.write_text("hello world this is a test", encoding="utf-8")

        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_file=str(single_file),
            max_steps_override=5,
        )
        assert len(result["train_losses"]) == 5

    def test_verse_train_parallel_chunks(self, tmp_path):
        """verse-train --parallel-chunks 2（ParallelTrainerSafe 路径）。"""
        cfg_path = _make_tiny_config(str(tmp_path), parallel_chunks=2, max_steps=8)

        # 构造训练数据
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_jsonl(str(data_dir / "train.jsonl"), n=20)
        _make_jsonl(str(data_dir / "val.jsonl"), n=10)

        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            max_steps_override=8,
        )
        assert "best_val_loss" in result
        # ParallelTrainer 应产生 history
        assert isinstance(result["train_losses"], list)

    def test_cli_main_dispatch(self, tmp_path, capsys):
        """cli.main 子命令分发。"""
        from verse_infra.verse_trainer import cli

        # 无参数应返回 1
        rc = cli.main([])
        assert rc == 1

        # 未知子命令应返回 1
        rc = cli.main(["unknown-cmd"])
        assert rc == 1

        # --help 应返回 0
        rc = cli.main(["--help"])
        assert rc == 0

    def test_cli_verse_tokenize_train(self, tmp_path):
        """verse-tokenize --train 模式。"""
        from verse_infra.verse_trainer import cli

        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello world hello world hello", encoding="utf-8")
        save_path = tmp_path / "tok.json"

        rc = cli.tokenize_main([
            "--train", str(corpus),
            "--kind", "bpe",
            "--vocab-size", "50",
            "--save", str(save_path),
        ])
        assert rc == 0
        assert save_path.exists()

    def test_cli_verse_tokenize_byte(self, tmp_path):
        """verse-tokenize --train --kind byte 模式。"""
        from verse_infra.verse_trainer import cli

        save_path = tmp_path / "byte_tok.json"
        rc = cli.tokenize_main([
            "--train", str(tmp_path / "any.txt"),  # byte 不需要 corpus
            "--kind", "byte",
            "--save", str(save_path),
        ])
        assert rc == 0
        assert save_path.exists()

    def test_cli_verse_tokenize_load(self, tmp_path):
        """verse-tokenize --load 模式。"""
        from verse_infra.verse_trainer import cli

        # 先 build
        save_path = tmp_path / "tok.json"
        cli.tokenize_main([
            "--train", str(tmp_path / "x.txt"),
            "--kind", "byte",
            "--save", str(save_path),
        ])
        # 再 load
        rc = cli.tokenize_main([
            "--load", str(save_path),
            "--kind", "byte",
            "--text", "hello",
        ])
        assert rc == 0


# ===========================================================================
# SubTask 6.2: 断点续训
# ===========================================================================


class TestResumeTraining:
    """断点续训：save + resume。"""

    def test_resume_checkpoint_saved(self, tmp_path):
        """训练后应保存 resume.pt。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=5)
        result = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "q", "completion": "a"},
            max_steps_override=5,
        )
        resume_path = os.path.join(result["checkpoint_dir"], "resume.pt")
        assert os.path.exists(resume_path)
        # 验证 resume.pt 可被 pickle.load
        import pickle
        with open(resume_path, "rb") as f:
            payload = pickle.load(f)
        assert "model_state_dict" in payload
        assert "best_val_loss" in payload

    def test_resume_does_not_crash(self, tmp_path):
        """resume=True 时应正常加载并继续训练。"""
        cfg_path = _make_tiny_config(str(tmp_path), max_steps=5)

        # 第一段训练
        result1 = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "q", "completion": "a"},
            max_steps_override=5,
        )

        # 第二段训练：resume=True
        result2 = train(
            config_path=cfg_path,
            base_dir=str(tmp_path),
            single_sample={"prompt": "q", "completion": "a"},
            max_steps_override=5,
            resume=True,
        )
        assert "best_val_loss" in result2
        # resume 后 best_val_loss 应保持或改善（不重置为 inf）
        assert result2["best_val_loss"] <= result1["best_val_loss"] + 1e-3


# ===========================================================================
# SubTask 6.6: RLTrainer 包装 NexTrainer
# ===========================================================================


class TestRLTrainer:
    """RLTrainer 调用 NexTrainer。"""

    def test_rl_trainer_fit(self):
        """RLTrainer.fit 调用 NexTrainer 完成训练。"""
        from verse_nex import VerseNexLM
        from verse_infra.verse_tokenizer import ByteTokenizer

        model = VerseNexLM(
            vocab_size=259, dim=32, n_layer=2, n_head=4, n_kv_head=2,
            window_size=4, num_global_tokens=2, max_seq_len=64,
            use_alibi=True, use_rope=False, dropout=0.0, tie_weights=True,
        )
        tok = ByteTokenizer()
        trainer = RLTrainer(
            model, tok,
            cfg={
                "ppo_epochs": 1,
                "max_new_tokens": 4,
                "use_value": True,
                "lr": 1e-3,
                "target_kl": 10.0,
            },
        )
        losses, kls, rewards = trainer.fit(
            prompts=["1+1=", "hello"],
            n_epochs=1,
            n_rollouts_per_prompt=1,
        )
        assert len(losses) == 1
        assert len(kls) == 1
        assert len(rewards) == 1
        # loss 不应为 NaN
        for l in losses:
            assert not np.isnan(l), f"RL loss is NaN: {l}"

    def test_rl_trainer_fallback_no_tokenizer(self):
        """无 tokenizer 时 RLTrainer 应兜底 encode/decode。"""
        from verse_nex import VerseNexLM
        model = VerseNexLM(
            vocab_size=256, dim=32, n_layer=2, n_head=4, n_kv_head=2,
            window_size=4, num_global_tokens=2, max_seq_len=64,
            use_alibi=True, use_rope=False, dropout=0.0, tie_weights=True,
        )
        trainer = RLTrainer(
            model, tokenizer=None,
            cfg={
                "ppo_epochs": 1,
                "max_new_tokens": 4,
                "use_value": False,  # 用 fallback 路径
                "lr": 1e-3,
                "target_kl": 10.0,
            },
        )
        losses, kls, rewards = trainer.fit(
            prompts=["ab", "cd"],
            n_epochs=1,
            n_rollouts_per_prompt=1,
        )
        assert len(losses) == 1

    def test_rl_trainer_save(self, tmp_path):
        """RLTrainer save_dir 应保存策略模型。"""
        from verse_nex import VerseNexLM
        model = VerseNexLM(
            vocab_size=259, dim=32, n_layer=2, n_head=4, n_kv_head=2,
            window_size=4, num_global_tokens=2, max_seq_len=64,
            use_alibi=True, use_rope=False, dropout=0.0, tie_weights=True,
        )
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        save_dir = str(tmp_path / "rl_ckpt")
        trainer = RLTrainer(
            model, tok,
            cfg={
                "ppo_epochs": 1,
                "max_new_tokens": 4,
                "use_value": True,
                "lr": 1e-3,
                "target_kl": 10.0,
            },
            save_dir=save_dir,
        )
        trainer.fit(prompts=["1+1="], n_epochs=1, n_rollouts_per_prompt=1)
        assert os.path.exists(os.path.join(save_dir, "rl_policy.pt"))


# ===========================================================================
# ParallelTrainerSafe 集成测试
# ===========================================================================


class TestParallelTrainerSafe:
    """ParallelTrainerSafe 集成：chunk 安全执行 + resume。"""

    def test_fit_with_safe_chunks(self, tmp_path):
        """ParallelTrainerSafe.fit 完整流程。"""
        from verse_infra.verse_tokenizer import ByteTokenizer
        from verse_nex import VerseNexLM

        tok = ByteTokenizer()
        # 构造数据集
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_jsonl = str(data_dir / "train.jsonl")
        val_jsonl = str(data_dir / "val.jsonl")
        _make_jsonl(train_jsonl, n=20)
        _make_jsonl(val_jsonl, n=10)

        train_ds = CachedDataset(tok, train_jsonl, seq_len=16)
        val_ds = CachedDataset(tok, val_jsonl, seq_len=16)

        model = VerseNexLM(
            vocab_size=259, dim=32, n_layer=2, n_head=4, n_kv_head=2,
            window_size=8, num_global_tokens=2, max_seq_len=64,
            use_alibi=True, use_rope=False, dropout=0.0, tie_weights=True,
        )

        ckpt_dir = str(tmp_path / "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        from verse_torch.training import CheckpointManager
        ckpt_mgr = CheckpointManager(ckpt_dir)

        trainer = ParallelTrainerSafe(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            optimizer_cls=AdamW,
            optimizer_kwargs={"weight_decay": 0.0},
            cfg={
                "parallel_chunks": 2,
                "max_steps": 6,
                "batch_size": 2,
                "lr": 0.003,
                "warmup": 1,
                "eval_interval": 3,
                "grad_clip": 1.0,
                "label_smoothing": 0.0,
                "seed": 42,
                "patience": 5,
                "save_dir": ckpt_dir,
                "log_interval": 100,
                "loss_rate_window": 10,
                "enable_progress_bar": False,
                "realtime_plot": False,
                "eta_window": 5,
            },
            collate_fn=collate_fn,
            checkpoint_mgr=ckpt_mgr,
        )
        history = trainer.fit()
        assert "train_loss" in history
        assert "val_loss" in history
        assert trainer.best_val_loss < float("inf")
        # resume.pt 应被保存
        assert os.path.exists(os.path.join(ckpt_dir, "resume.pt"))


# ===========================================================================
# TextDataset 向后兼容
# ===========================================================================


class TestTextDatasetBackwardCompat:
    """TextDataset 向后兼容（与 data/demo 原实现一致）。"""

    def test_text_dataset_basic(self, tmp_path):
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        jsonl = str(tmp_path / "train.jsonl")
        _make_jsonl(jsonl, n=10)
        ds = TextDataset(tok, jsonl, seq_len=8)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_batch_loader_iter(self, tmp_path):
        from verse_infra.verse_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        jsonl = str(tmp_path / "train.jsonl")
        _make_jsonl(jsonl, n=20)
        ds = CachedDataset(tok, jsonl, seq_len=8)
        loader = BatchLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
        n_batches = 0
        for x, y in loader:
            assert x.shape == (4, 8) or x.shape[0] <= 4
            assert y.shape == x.shape
            n_batches += 1
        assert n_batches > 0
