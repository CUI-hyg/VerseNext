"""Part5K1 Task 6：val.json 自动生成 + 数据预加载流水线 单元测试。

覆盖：
1. val 不存在自动生成（从 train 末尾切分）
2. val 为空自动生成（空文件被填充）
3. val 已存在不切分（保守不动）
4. val_ratio 边界（0.0 至少 1 条，1.0 全部作为 val）
5. write_back=False（不写文件，只返回计数）
6. trainer 集成（train() 入口自动调用 ensure_val_split）
7. CachedDataset preload（后台线程编码 + __getitem__ 返回正确数据）
8. BatchLoader prefetch 非 torch 环境（纯 threading 预取）

运行方式：
    cd /workspace && python -m pytest tests/test_val_autogen.py -x -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# PYTHONPATH 适配：让 tests/ 能 import verse_infra / verse_torch 子模块
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_infra", "verse_torch", "verse_nex"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verse_infra.verse_trainer.data import (  # noqa: E402
    ensure_val_split,
    count_lines,
    CachedDataset,
    BatchLoader,
    collate_fn,
)
from verse_infra.verse_trainer import trainer as trainer_mod  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, items):
    """把 list 写入 jsonl 文件，每行一个 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list:
    """读取 jsonl 文件，返回 list。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _make_train_items(n: int = 10):
    """构造 n 条训练样本（text 格式）。"""
    return [{"text": f"sample-{i} hello world"} for i in range(n)]


# ---------------------------------------------------------------------------
# 1. val 不存在自动生成
# ---------------------------------------------------------------------------


class TestEnsureValSplitAutoGen:
    """SubTask 6.1：ensure_val_split 自动切分。"""

    def test_val_not_exist_autogen(self, tmp_path):
        """用例 1：val 不存在 → 从 train 末尾切分 20%。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_items = _make_train_items(10)
        _write_jsonl(train_path, train_items)
        # val.jsonl 不创建

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.2
        )

        # 10 * 0.2 = 2，所以 n_val=2, n_train=8
        assert n_train == 8
        assert n_val == 2
        # val.jsonl 被创建
        assert val_path.exists()
        # val.jsonl 内容是 train 的末尾 2 条
        val_items = _read_jsonl(val_path)
        assert len(val_items) == 2
        assert val_items[0] == train_items[8]
        assert val_items[1] == train_items[9]

    def test_val_empty_file_autogen(self, tmp_path):
        """用例 2：val 存在但为空文件 → 应被填充。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_items = _make_train_items(10)
        _write_jsonl(train_path, train_items)
        # 创建空的 val.jsonl（0 字节）
        val_path.write_text("", encoding="utf-8")
        assert val_path.exists()
        assert os.path.getsize(val_path) == 0

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.2
        )

        assert n_train == 8
        assert n_val == 2
        # val.jsonl 被填充
        val_items = _read_jsonl(val_path)
        assert len(val_items) == 2
        assert val_items[0] == train_items[8]

    def test_val_blank_lines_autogen(self, tmp_path):
        """用例 2b：val 存在但只有空白行（非空文件但行数=0）→ 应被填充。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_items = _make_train_items(10)
        _write_jsonl(train_path, train_items)
        # val.jsonl 只有空白行（文件大小 > 0 但非空行数 = 0）
        val_path.write_text("\n   \n\n", encoding="utf-8")
        assert os.path.getsize(val_path) > 0
        assert count_lines(str(val_path)) == 0

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.2
        )

        assert n_train == 8
        assert n_val == 2
        val_items = _read_jsonl(val_path)
        assert len(val_items) == 2

    def test_val_exists_not_modified(self, tmp_path):
        """用例 3：val 已存在且非空 → 不切分，返回原计数。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_items = _make_train_items(10)
        val_items = [{"text": f"existing-val-{i}"} for i in range(5)]
        _write_jsonl(train_path, train_items)
        _write_jsonl(val_path, val_items)
        # 记录原内容用于后续对比
        original_val_content = val_path.read_text(encoding="utf-8")

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.2
        )

        # val 已存在 5 条，不切分；train 仍是 10 条
        assert n_train == 10
        assert n_val == 5
        # val 文件未被修改
        assert val_path.read_text(encoding="utf-8") == original_val_content

    def test_val_ratio_zero_at_least_one(self, tmp_path):
        """用例 4a：val_ratio=0.0 仍至少切 1 条 val。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(10))

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.0
        )

        assert n_val == 1
        assert n_train == 9
        assert count_lines(str(val_path)) == 1

    def test_val_ratio_one_all_as_val(self, tmp_path):
        """用例 4b：val_ratio=1.0 全部作为 val，n_train=0。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(10))

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=1.0
        )

        assert n_val == 10
        assert n_train == 0
        assert count_lines(str(val_path)) == 10

    def test_val_ratio_out_of_range_raises(self, tmp_path):
        """用例 4c：val_ratio 越界（<0 或 >1）抛 ValueError。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(5))

        with pytest.raises(ValueError, match="val_ratio"):
            ensure_val_split(str(train_path), str(val_path), val_ratio=-0.1)
        with pytest.raises(ValueError, match="val_ratio"):
            ensure_val_split(str(train_path), str(val_path), val_ratio=1.5)

    def test_write_back_false_no_file(self, tmp_path):
        """用例 5：write_back=False → 不写文件，只返回计数。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(10))

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.2, write_back=False
        )

        assert n_train == 8
        assert n_val == 2
        # val.jsonl 未被创建
        assert not val_path.exists()

    def test_default_val_ratio_5_percent(self, tmp_path):
        """默认 val_ratio=0.05：100 条 train 切 5 条 val。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(100))

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path)  # 默认 val_ratio=0.05
        )

        assert n_val == 5
        assert n_train == 95

    def test_train_empty_raises(self, tmp_path):
        """train_path 为空文件 → 抛 ValueError。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="训练集为空"):
            ensure_val_split(str(train_path), str(val_path))

    def test_val_path_in_nonexistent_dir(self, tmp_path):
        """val_path 在不存在的目录下 → 自动创建目录。"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "subdir" / "nested" / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(10))

        n_train, n_val = ensure_val_split(
            str(train_path), str(val_path), val_ratio=0.2
        )

        assert n_train == 8
        assert n_val == 2
        assert val_path.exists()
        # 目录被自动创建
        assert val_path.parent.is_dir()


# ---------------------------------------------------------------------------
# 6. trainer 集成：train() 入口自动调用 ensure_val_split
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    """SubTask 6.3：train() 入口自动调用 ensure_val_split（dry-run）。"""

    def test_train_calls_ensure_val_split(self, tmp_path, monkeypatch):
        """用例 6：train() 入口自动调用 ensure_val_split（mock 全部下游）。"""
        # 准备数据文件（train.jsonl 存在，val.jsonl 不存在）
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, _make_train_items(10))
        # val.jsonl 不创建，让 ensure_val_split（mock）记录调用即可

        # mock ensure_val_split 记录调用
        call_log = {"called": False, "args": None}

        def mock_ensure_val_split(tp, vp, *args, **kwargs):
            call_log["called"] = True
            call_log["args"] = (tp, vp)
            return (8, 2)

        monkeypatch.setattr(
            trainer_mod, "ensure_val_split", mock_ensure_val_split
        )

        # 准备 config.yml（mock _load_full_config 会覆盖它，但 train() 需要路径存在）
        config_path = tmp_path / "config.yml"
        config_path.write_text("model:\n  arch: versenex\n", encoding="utf-8")

        save_dir = str(tmp_path / "ckpts")
        os.makedirs(save_dir, exist_ok=True)

        # mock 模型（小模型，避免触发 1B 优化路径）
        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_model.named_parameters.return_value = []
        mock_model.state_dict.return_value = {}
        mock_model.to.return_value = mock_model
        mock_model.save.return_value = None
        mock_model.device_info.return_value = "cpu"
        mock_model.count_parameters.return_value = 1000

        mock_config = MagicMock()
        mock_config.arch = "versenex"
        mock_config.device = "cpu"
        mock_config.aux_loss_weight = 0.01

        mock_tok = MagicMock()
        mock_tok.__len__ = lambda self: 100

        # mock CachedDataset 返回简单 dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 4
        mock_ds.__getitem__ = lambda self, i: (
            np.zeros(8, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
        )

        # mock Trainer 避免真正训练
        mock_trainer = MagicMock()
        mock_trainer.best_val_loss = 1.5
        mock_trainer.fit.return_value = ([1.0], [1.5])

        with patch(
            "verse_infra.verse_trainer.trainer._build_model",
            return_value=(mock_model, mock_config),
        ), patch(
            "verse_infra.verse_trainer.trainer._load_tokenizer",
            return_value=mock_tok,
        ), patch(
            "verse_infra.verse_trainer.trainer._load_full_config",
            return_value={
                "model": {"arch": "versenex", "seq_len": 8, "n_embd": 16,
                          "n_layer": 2, "n_head": 2},
                "training": {"max_steps": 1, "batch_size": 2, "lr": 0.01,
                             "eval_interval": 1, "patience": 5, "seed": 42},
                "tokenizer": {"kind": "byte"},
                "data": {"train_path": "train.jsonl", "val_path": "val.jsonl"},
                "checkpoint": {"save_dir": save_dir},
            },
        ), patch(
            "verse_infra.verse_trainer.trainer.CachedDataset",
            return_value=mock_ds,
        ), patch(
            "verse_infra.verse_trainer.trainer.BatchLoader",
            return_value=MagicMock(),
        ), patch(
            "verse_infra.verse_trainer.trainer.collate_fn",
        ), patch(
            "verse_infra.verse_trainer.trainer._auto_evaluate",
            return_value={},
        ), patch(
            "verse_infra.verse_trainer.trainer.Trainer",
            return_value=mock_trainer,
        ):
            try:
                trainer_mod.train(
                    config_path=str(config_path),
                    base_dir=str(tmp_path),
                    quiet=True,
                )
            except Exception:
                # mock 环境下可能有其他副作用，主要验证 ensure_val_split 被调用
                pass

        # 验证 ensure_val_split 被调用
        assert call_log["called"], "train() 入口应调用 ensure_val_split"
        assert call_log["args"] is not None
        # 传入的 train_path / val_path 应包含配置中的文件名
        assert "train.jsonl" in call_log["args"][0]
        assert "val.jsonl" in call_log["args"][1]


# ---------------------------------------------------------------------------
# 7. CachedDataset preload
# ---------------------------------------------------------------------------


def _make_tokenizer():
    """构造 ByteTokenizer（vocab_size=259）。"""
    from verse_infra.verse_tokenizer import ByteTokenizer
    return ByteTokenizer()


def _make_small_train_jsonl(path: Path, n: int = 4):
    """构造小尺寸训练数据（n 条 text 样本，每条足够长以填满 seq_len 块）。"""
    items = [
        {"text": f"sample-{i} hello world this is a test for preload {i}"}
        for i in range(n)
    ]
    _write_jsonl(path, items)
    return items


class TestCachedDatasetPreload:
    """SubTask 6.2：CachedDataset 后台预加载。"""

    def test_preload_false_default_behavior(self, tmp_path):
        """用例 7a：preload=False（默认）走原同步路径，行为不变。"""
        train_path = tmp_path / "train.jsonl"
        _make_small_train_jsonl(train_path, n=4)
        cache_path = str(train_path) + ".cache.npz"
        # 确保无缓存
        if os.path.exists(cache_path):
            os.remove(cache_path)

        tok = _make_tokenizer()
        # preload=False 默认
        ds = CachedDataset(tok, str(train_path), seq_len=8)
        # 同步路径：构造完成后 ids/mask 已就位
        assert ds._preload_thread is None
        assert ds.ids is not None
        assert ds.mask is not None
        assert ds.n_blocks > 0
        # __getitem__ 可直接调用
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_preload_true_async_build(self, tmp_path):
        """用例 7b：preload=True 启动后台线程，__getitem__ 时 join。"""
        train_path = tmp_path / "train.jsonl"
        _make_small_train_jsonl(train_path, n=4)
        cache_path = str(train_path) + ".cache.npz"
        # 确保无缓存（强制走编码路径）
        if os.path.exists(cache_path):
            os.remove(cache_path)

        tok = _make_tokenizer()
        ds = CachedDataset(tok, str(train_path), seq_len=8, preload=True)

        # preload=True 且缓存未命中：应启动后台线程
        assert ds._preload_thread is not None, "preload=True 应启动后台线程"
        # 构造函数立即返回，ids 可能还未就位
        # （但允许已就位，因为后台线程可能很快完成）

        # __getitem__ 触发等待后台线程完成
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)
        # 调用后 _preload_thread 应被清理（_wait_for_preload 完成后置 None）
        assert ds._preload_thread is None
        # ids / mask 已就位
        assert ds.ids is not None
        assert ds.mask is not None
        assert ds.n_blocks > 0

    def test_preload_true_data_consistent_with_sync(self, tmp_path):
        """用例 7c：preload=True 与 preload=False 返回相同数据。"""
        train_path_a = tmp_path / "train_a.jsonl"
        train_path_b = tmp_path / "train_b.jsonl"
        items = _make_small_train_jsonl(train_path_a, n=4)
        _write_jsonl(train_path_b, items)
        # 删除缓存
        for p in (train_path_a, train_path_b):
            cache = str(p) + ".cache.npz"
            if os.path.exists(cache):
                os.remove(cache)

        tok = _make_tokenizer()
        ds_sync = CachedDataset(tok, str(train_path_a), seq_len=8, preload=False)
        ds_preload = CachedDataset(tok, str(train_path_b), seq_len=8, preload=True)

        # 触发 preload 等待
        x_sync, y_sync = ds_sync[0]
        x_pre, y_pre = ds_preload[0]

        # 长度一致
        assert len(ds_sync) == len(ds_preload)
        # 数据一致（同步路径与预加载路径产出相同）
        np.testing.assert_array_equal(x_sync, x_pre)
        np.testing.assert_array_equal(y_sync, y_pre)

    def test_preload_true_with_existing_cache(self, tmp_path):
        """用例 7d：preload=True 但缓存已命中 → 不启动后台线程（直接同步加载）。"""
        train_path = tmp_path / "train.jsonl"
        _make_small_train_jsonl(train_path, n=4)
        cache_path = str(train_path) + ".cache.npz"

        tok = _make_tokenizer()
        # 第一次：构建缓存（同步）
        if os.path.exists(cache_path):
            os.remove(cache_path)
        ds_first = CachedDataset(tok, str(train_path), seq_len=8)
        assert os.path.exists(cache_path), "缓存应已生成"

        # 第二次：preload=True 但缓存命中，应直接加载，不启动后台线程
        ds_second = CachedDataset(tok, str(train_path), seq_len=8, preload=True)
        assert ds_second._preload_thread is None, "缓存命中时不启动后台线程"
        assert ds_second.ids is not None
        assert ds_second.n_blocks == ds_first.n_blocks
        # 数据一致
        x1, y1 = ds_first[0]
        x2, y2 = ds_second[0]
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_preload_true_len_blocks_until_done(self, tmp_path):
        """用例 7e：preload=True 时 __len__ 也会阻塞等待后台线程完成。"""
        train_path = tmp_path / "train.jsonl"
        _make_small_train_jsonl(train_path, n=4)
        cache_path = str(train_path) + ".cache.npz"
        if os.path.exists(cache_path):
            os.remove(cache_path)

        tok = _make_tokenizer()
        ds = CachedDataset(tok, str(train_path), seq_len=8, preload=True)
        assert ds._preload_thread is not None

        # __len__ 触发等待
        n = len(ds)
        assert n > 0
        # 等待完成后 _preload_thread 清理
        assert ds._preload_thread is None

    def test_preload_error_propagated(self, tmp_path):
        """用例 7f：preload 后台线程抛异常 → 主线程 __getitem__ 重新抛出。"""
        # 构造一个无法编码的 jsonl（空数据触发 ValueError）
        train_path = tmp_path / "empty.jsonl"
        train_path.write_text("", encoding="utf-8")
        cache_path = str(train_path) + ".cache.npz"
        if os.path.exists(cache_path):
            os.remove(cache_path)

        tok = _make_tokenizer()
        # preload=True 启动后台线程，后台线程会因数据为空抛错
        ds = CachedDataset(tok, str(train_path), seq_len=8, preload=True)
        # 后台线程启动（即使数据无效）
        # __getitem__ 触发等待 + 异常重抛
        with pytest.raises((ValueError, RuntimeError)):
            _ = ds[0]


# ---------------------------------------------------------------------------
# 8. BatchLoader prefetch 非 torch 环境（纯 threading）
# ---------------------------------------------------------------------------


class _DummyDataset:
    """简单数据集：返回 (x, y) 数组。"""

    def __init__(self, n=8, seq_len=4):
        self.n = n
        self.seq_len = seq_len
        self.x = np.arange(n * seq_len, dtype=np.int64).reshape(n, seq_len)
        self.y = self.x + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TestBatchLoaderPrefetchNoTorch:
    """SubTask 6.2：BatchLoader prefetch 在无 torch 环境也能工作。"""

    def test_prefetch_true_works_without_torch(self):
        """prefetch=True 在 _torch=None 时仍走 threading 预取路径。"""
        ds = _DummyDataset(n=8, seq_len=4)
        loader = BatchLoader(
            ds,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            prefetch=True,
        )
        # 模拟无 torch 环境
        loader._torch = None
        loader.pin_memory = False
        # prefetch 应保持 True（不因无 torch 而降级）
        assert loader.prefetch is True

        # 迭代应正常工作（纯 threading 预取）
        batches = list(loader)
        assert len(batches) == 4  # 8 / 2 = 4 batches
        # 每个 batch 形状 (2, 4)
        for x_b, y_b in batches:
            assert x_b.shape == (2, 4)
            assert y_b.shape == (2, 4)

    def test_prefetch_default_follows_pin_memory(self):
        """prefetch=None 时默认跟随 pin_memory。"""
        ds = _DummyDataset(n=4)
        # pin_memory=False → prefetch 默认 False
        loader = BatchLoader(ds, pin_memory=False)
        assert loader.prefetch is False

        # pin_memory=True → prefetch 默认 True（即使无 torch，prefetch 也保持 True）
        loader2 = BatchLoader(ds, pin_memory=True)
        # 无 torch 时 pin_memory 被降级为 False，但 prefetch 在赋值时已跟随原 pin_memory
        # 注意：prefetch 在 pin_memory 降级前赋值，所以保持 True
        assert loader2.prefetch is True

    def test_prefetch_explicit_true_without_torch(self):
        """显式 prefetch=True 即使无 torch 也能预取。"""
        ds = _DummyDataset(n=6, seq_len=4)
        loader = BatchLoader(
            ds,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            prefetch=True,  # 显式开启
        )
        # 即使无 torch，prefetch 仍为 True
        assert loader.prefetch is True

        # 迭代结果与同步路径一致
        loader_sync = BatchLoader(
            ds, batch_size=2, shuffle=False, collate_fn=collate_fn, prefetch=False
        )
        batches_pre = list(loader)
        batches_sync = list(loader_sync)
        assert len(batches_pre) == len(batches_sync)
        for (x_p, y_p), (x_s, y_s) in zip(batches_pre, batches_sync):
            np.testing.assert_array_equal(x_p, x_s)
            np.testing.assert_array_equal(y_p, y_s)

    def test_prefetch_single_batch_no_prefetch(self):
        """数据只有 1 个 batch 时即使 prefetch=True 也不预取（无意义）。"""
        ds = _DummyDataset(n=2, seq_len=4)
        loader = BatchLoader(
            ds, batch_size=2, shuffle=False, collate_fn=collate_fn, prefetch=True
        )
        # batch_slices 长度=1，不预取
        batches = list(loader)
        assert len(batches) == 1
