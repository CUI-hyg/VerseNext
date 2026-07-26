"""tests/test_checkpoint_atomic.py

Part5K1.3 Task 2: CheckpointManager 原子写 + format 参数 + 扩展签名 测试。

覆盖 SubTask 2.1 ~ 2.6:
- 2.1 原子写：save_best / save_last 后 .tmp 不存在，文件存在且可读
- 2.1 中断不损坏：mock 写入失败，旧 best.pt 未被破坏，.tmp 已清理
- 2.2 format 参数：auto / vn / pt 选择 + invalid 抛 ValueError
- 2.3 扩展签名：training_state / optimizer_state / extra_state / step 参数
- 2.4 load_best 向后兼容 + load_best_full 返回统一 dict（缺失字段为 None）
- 2.5 use_vmpc 强制 .vn：format="pt" + use_vmpc=True 抛 ValueError

运行方式：
    cd /workspace && python -m pytest tests/test_checkpoint_atomic.py -v
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch.training import CheckpointManager  # noqa: E402


# ---------------------------------------------------------------------------
# SubTask 2.1: 原子写
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """原子写：save_best / save_last 后 .tmp 不存在，文件可读。"""

    def test_save_best_no_tmp_left(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_best({"step": 1, "val": 0.5})

        assert (tmp_path / "best.pt").exists()
        assert not (tmp_path / "best.pt.tmp").exists()

        loaded = ckpt.load_best()
        assert loaded["step"] == 1
        assert loaded["val"] == pytest.approx(0.5)

    def test_save_last_no_tmp_left(self, tmp_path):
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_last({"step": 2, "val": 0.6})

        assert (tmp_path / "last.pt").exists()
        assert not (tmp_path / "last.pt.tmp").exists()

        loaded = ckpt.load_last()
        assert loaded["step"] == 2

    def test_save_best_overwrites_atomically(self, tmp_path):
        """多次覆盖写不会残留 .tmp，最终内容为最后一次写入。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_best({"version": 1})
        ckpt.save_best({"version": 2})
        ckpt.save_best({"version": 3})

        assert not (tmp_path / "best.pt.tmp").exists()
        assert ckpt.load_best()["version"] == 3

    def test_interrupt_does_not_corrupt_best(self, tmp_path):
        """mock pickle.dump 失败：旧 best.pt 未被破坏，.tmp 已清理。"""
        ckpt = CheckpointManager(str(tmp_path))
        # 先写一个有效的 best.pt
        ckpt.save_best({"old": "data", "step": 50})
        assert ckpt.load_best()["old"] == "data"

        # mock pickle.dump 抛 OSError（模拟磁盘满 / 写入中断）
        with patch(
            "verse_torch.training.pickle.dump",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError, match="disk full"):
                ckpt.save_best({"new": "data", "step": 100})

        # 旧 best.pt 未被破坏
        loaded = ckpt.load_best()
        assert loaded["old"] == "data"
        assert loaded["step"] == 50
        assert "new" not in loaded

        # .tmp 已清理
        assert not (tmp_path / "best.pt.tmp").exists()

    def test_interrupt_does_not_corrupt_last(self, tmp_path):
        """同上，但针对 save_last。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_last({"old": "last_data", "step": 30})

        with patch(
            "verse_torch.training.pickle.dump",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError):
                ckpt.save_last({"new": "data"})

        loaded = ckpt.load_last()
        assert loaded["old"] == "last_data"
        assert loaded["step"] == 30
        assert "new" not in loaded
        assert not (tmp_path / "last.pt.tmp").exists()

    def test_interrupt_leaves_existing_file_intact(self, tmp_path):
        """写入失败时不仅 .tmp 被清理，原 best.pt 的内容字节完全不变。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_best({"keep": True})

        # 记录原文件字节
        original_bytes = (tmp_path / "best.pt").read_bytes()

        with patch(
            "verse_torch.training.pickle.dump",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError):
                ckpt.save_best({"should_not_be_written": True})

        # 原文件字节完全一致
        assert (tmp_path / "best.pt").read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# SubTask 2.2 + 2.5: format 参数 + use_vmpc 强制 .vn
# ---------------------------------------------------------------------------


class TestFormatParameter:
    """format 参数解析 + use_vmpc 强制 .vn 校验。"""

    def test_format_auto_no_vmpc_selects_pt(self, tmp_path):
        """format="auto" + use_vmpc=False → 选 "pt"，写 best.pt。"""
        ckpt = CheckpointManager(str(tmp_path), format="auto", use_vmpc=False)
        assert ckpt.format == "pt"
        ckpt.save_best({"a": 1})
        assert (tmp_path / "best.pt").exists()
        assert not (tmp_path / "best.vn").exists()

    def test_format_auto_with_vmpc_selects_vn(self, tmp_path):
        """format="auto" + use_vmpc=True → 选 "vn"，默认路径 best.vn。"""
        ckpt = CheckpointManager(str(tmp_path), format="auto", use_vmpc=True)
        assert ckpt.format == "vn"
        assert ckpt.best_path == tmp_path / "best.vn"
        assert ckpt.last_path == tmp_path / "last.vn"

    def test_format_pt_explicit(self, tmp_path):
        """format="pt" → 写 best.pt。"""
        ckpt = CheckpointManager(str(tmp_path), format="pt")
        assert ckpt.format == "pt"
        ckpt.save_best({"a": 1})
        assert (tmp_path / "best.pt").exists()
        assert not (tmp_path / "best.vn").exists()

    def test_format_vn_explicit(self, tmp_path):
        """format="vn" → 选 "vn"，默认路径 best.vn（Part5K1.3 Task 4 原生 VNFileWriter）。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        assert ckpt.format == "vn"
        assert ckpt.best_path == tmp_path / "best.vn"
        # 原生 VNFileWriter 写入 .vn 路径（需提供 ndarray 权重）
        weights = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
        ckpt.save_best(weights)
        assert (tmp_path / "best.vn").exists()
        assert not (tmp_path / "best.vn.tmp").exists()
        # 可正常 load（VNFileReader 读取）
        loaded = ckpt.load_best()
        np.testing.assert_allclose(loaded["model_state_dict"]["weight"], [1.0, 2.0])

    def test_format_invalid_raises(self, tmp_path):
        """format="invalid" → 抛 ValueError。"""
        with pytest.raises(ValueError, match="format"):
            CheckpointManager(str(tmp_path), format="invalid")

    def test_format_invalid_other_value_raises(self, tmp_path):
        with pytest.raises(ValueError, match="format"):
            CheckpointManager(str(tmp_path), format="safetensors")

    def test_use_vmpc_with_format_pt_raises(self, tmp_path):
        """format="pt" + use_vmpc=True → 抛 ValueError（强制 .vn）。"""
        with pytest.raises(ValueError, match="use_vmpc"):
            CheckpointManager(str(tmp_path), format="pt", use_vmpc=True)

    def test_default_no_format_args_backward_compat(self, tmp_path):
        """不传 format/use_vmpc → 默认 auto + use_vmpc=False → pt（向后兼容）。"""
        ckpt = CheckpointManager(str(tmp_path))
        assert ckpt.format == "pt"
        assert ckpt.use_vmpc is False
        assert ckpt.best_path == tmp_path / "best.pt"
        assert ckpt.last_path == tmp_path / "last.pt"

    def test_custom_path_overrides_format_default(self, tmp_path):
        """自定义 best_path / last_path 优先于 format 默认扩展名。"""
        custom_best = tmp_path / "my_best.pt"
        custom_last = tmp_path / "my_last.pt"
        ckpt = CheckpointManager(
            str(tmp_path),
            best_path=custom_best,
            last_path=custom_last,
            format="pt",
        )
        assert ckpt.best_path == custom_best
        assert ckpt.last_path == custom_last
        ckpt.save_best({"a": 1})
        assert custom_best.exists()


# ---------------------------------------------------------------------------
# SubTask 2.3: 扩展 save_best / save_last 签名
# ---------------------------------------------------------------------------


class TestExtendedSignature:
    """save_best / save_last 扩展签名（training_state / optimizer_state / extra_state / step）。"""

    def test_save_best_extended_signature_pt(self, tmp_path):
        """format="pt" 时 save_best 接受 training_state / optimizer_state / extra_state（忽略）。"""
        ckpt = CheckpointManager(str(tmp_path), format="pt")
        ckpt.save_best(
            {"weight": np.array([1.0, 2.0])},
            training_state={"step": 100, "epoch": 5},
            optimizer_state={"m": np.zeros(2), "v": np.zeros(2), "step": 100},
            extra_state={"custom": "state"},
            step=100,
        )
        # format="pt" 仅 pickle state，忽略额外参数
        loaded = ckpt.load_best()
        np.testing.assert_allclose(loaded["weight"], [1.0, 2.0])
        # 额外参数未写入
        assert "training_state" not in loaded
        assert "optimizer_state" not in loaded

    def test_save_last_extended_signature_pt(self, tmp_path):
        """format="pt" 时 save_last 接受扩展参数（忽略）。"""
        ckpt = CheckpointManager(str(tmp_path), format="pt")
        ckpt.save_last(
            {"weight": np.array([3.0])},
            training_state={"step": 50},
            optimizer_state={"m": np.zeros(1)},
            extra_state=None,
            step=50,
        )
        loaded = ckpt.load_last()
        np.testing.assert_allclose(loaded["weight"], [3.0])

    def test_save_best_extended_signature_vn_native(self, tmp_path):
        """format="vn" 时 save_best 接受扩展参数（Part5K1.3 Task 4 原生 VNFileWriter）。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        # 原生 VNFileWriter 写入完整状态
        ckpt.save_best(
            {"model_state_dict": {"weight": np.array([1.0], dtype=np.float32)}},
            training_state={"step": 100, "epoch": 5},
            optimizer_state={"m": np.zeros(1, dtype=np.float32), "step": 100},
            extra_state={"custom": "state"},
            step=100,
        )
        # 加载（VNFileReader 读取）
        loaded = ckpt.load_best()
        np.testing.assert_allclose(loaded["model_state_dict"]["weight"], [1.0])
        assert loaded["training_state"]["step"] == 100
        assert loaded["training_state"]["epoch"] == 5
        np.testing.assert_allclose(loaded["optimizer_state"]["m"], [0.0])
        assert loaded["optimizer_state"]["step"] == 100
        assert loaded["extra_state"]["custom"] == "state"

    def test_save_best_default_none_backward_compat(self, tmp_path):
        """旧调用 save_best(state_dict) 仍工作（默认 None）。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_best({"a": 1})
        assert ckpt.load_best()["a"] == 1

    def test_save_last_default_none_backward_compat(self, tmp_path):
        """旧调用 save_last(state_dict) 仍工作（默认 None）。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_last({"b": 2})
        assert ckpt.load_last()["b"] == 2

    def test_save_best_with_only_step(self, tmp_path):
        """仅传 step 参数（其他默认 None）。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_best({"weights": [1, 2, 3]}, step=42)
        assert ckpt.load_best()["weights"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# SubTask 2.4: load_best 向后兼容
# ---------------------------------------------------------------------------


class TestLoadBestBackwardCompat:
    """load_best / load_last 向后兼容旧 .pt 文件。"""

    def test_load_best_old_pt_file(self, tmp_path):
        """旧 best.pt 文件（直接 pickle 写入）可被 load_best 读取。"""
        best_path = tmp_path / "best.pt"
        old_state = {
            "step": 200,
            "val_loss": 0.123,
            "weights": np.array([1.0, 2.0]),
        }
        with open(best_path, "wb") as f:
            pickle.dump(old_state, f)

        ckpt = CheckpointManager(str(tmp_path))
        loaded = ckpt.load_best()
        assert loaded["step"] == 200
        assert loaded["val_loss"] == pytest.approx(0.123)
        np.testing.assert_allclose(loaded["weights"], [1.0, 2.0])

    def test_load_last_old_pt_file(self, tmp_path):
        """旧 last.pt 文件可被 load_last 读取。"""
        last_path = tmp_path / "last.pt"
        old_state = {"step": 99, "data": "hello"}
        with open(last_path, "wb") as f:
            pickle.dump(old_state, f)

        ckpt = CheckpointManager(str(tmp_path))
        loaded = ckpt.load_last()
        assert loaded["step"] == 99
        assert loaded["data"] == "hello"

    def test_load_best_roundtrip_preserves_tensor(self, tmp_path):
        """Tensor 字段经序列化往返仍可还原（保持原 test_training 行为）。"""
        from verse_torch.tensor import Tensor

        ckpt = CheckpointManager(str(tmp_path))
        state = {
            "step": 10,
            "tensor_in_state": Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        }
        ckpt.save_best(state)
        loaded = ckpt.load_best()
        assert loaded["step"] == 10
        assert isinstance(loaded["tensor_in_state"], Tensor)
        np.testing.assert_allclose(
            loaded["tensor_in_state"].data,
            state["tensor_in_state"].data,
        )


# ---------------------------------------------------------------------------
# SubTask 2.4: load_best_full / load_last_full 返回统一 dict
# ---------------------------------------------------------------------------


class TestLoadBestFull:
    """load_best_full / load_last_full 返回统一 dict（缺失字段为 None）。"""

    def test_load_best_full_extracts_standard_fields(self, tmp_path):
        """save_best 保存含标准字段的 dict，load_best_full 提取。"""
        ckpt = CheckpointManager(str(tmp_path))
        state = {
            "model_state_dict": {"weight": np.array([1.0, 2.0])},
            "training_state": {"step": 100, "epoch": 5},
            "optimizer_state": {"m": np.zeros(2), "step": 100},
            "extra_state": {"custom": "state"},
            "step": 100,
            "best_val_loss": 0.456,
        }
        ckpt.save_best(state)
        full = ckpt.load_best_full()

        assert set(full.keys()) == {
            "model_state_dict",
            "training_state",
            "optimizer_state",
            "extra_state",
            "step",
            "best_val_loss",
        }
        np.testing.assert_allclose(full["model_state_dict"]["weight"], [1.0, 2.0])
        assert full["training_state"]["step"] == 100
        assert full["training_state"]["epoch"] == 5
        assert full["optimizer_state"]["step"] == 100
        np.testing.assert_allclose(full["optimizer_state"]["m"], [0.0, 0.0])
        assert full["extra_state"]["custom"] == "state"
        assert full["step"] == 100
        assert full["best_val_loss"] == pytest.approx(0.456)

    def test_load_best_full_missing_fields_are_none(self, tmp_path):
        """旧 .pt 文件无 training_state 等字段时返回 None。"""
        ckpt = CheckpointManager(str(tmp_path))
        # 仅保存 model_state_dict + step + val_loss（旧风格）
        ckpt.save_best({
            "model_state_dict": {"weight": np.array([1.0])},
            "step": 50,
            "val_loss": 0.789,
        })
        full = ckpt.load_best_full()

        assert full["model_state_dict"] is not None
        np.testing.assert_allclose(full["model_state_dict"]["weight"], [1.0])
        assert full["training_state"] is None
        assert full["optimizer_state"] is None
        assert full["extra_state"] is None
        assert full["step"] == 50
        # val_loss 作为 best_val_loss 的别名
        assert full["best_val_loss"] == pytest.approx(0.789)

    def test_load_best_full_no_standard_fields(self, tmp_path):
        """保存的 dict 完全不含标准字段，load_best_full 全部返回 None。"""
        ckpt = CheckpointManager(str(tmp_path))
        ckpt.save_best({"random_key": "random_value"})
        full = ckpt.load_best_full()

        assert full["model_state_dict"] is None
        assert full["training_state"] is None
        assert full["optimizer_state"] is None
        assert full["extra_state"] is None
        assert full["step"] is None
        assert full["best_val_loss"] is None

    def test_load_last_full_extracts_standard_fields(self, tmp_path):
        """load_last_full 同 load_best_full 但读 last 文件。"""
        ckpt = CheckpointManager(str(tmp_path))
        state = {
            "model_state_dict": {"weight": np.array([3.0])},
            "step": 200,
            "best_val_loss": 0.321,
        }
        ckpt.save_last(state)
        full = ckpt.load_last_full()

        np.testing.assert_allclose(full["model_state_dict"]["weight"], [3.0])
        assert full["step"] == 200
        assert full["best_val_loss"] == pytest.approx(0.321)
        assert full["training_state"] is None
        assert full["optimizer_state"] is None
        assert full["extra_state"] is None

    def test_load_best_full_backward_compat_with_old_pt(self, tmp_path):
        """旧 best.pt 文件（直接 pickle 写入）可被 load_best_full 读取。"""
        best_path = tmp_path / "best.pt"
        old_state = {
            "model_state_dict": {"weight": np.array([5.0])},
            "step": 10,
            "val_loss": 0.111,
        }
        with open(best_path, "wb") as f:
            pickle.dump(old_state, f)

        ckpt = CheckpointManager(str(tmp_path))
        full = ckpt.load_best_full()
        np.testing.assert_allclose(full["model_state_dict"]["weight"], [5.0])
        assert full["step"] == 10
        assert full["best_val_loss"] == pytest.approx(0.111)
        assert full["training_state"] is None
        assert full["optimizer_state"] is None
        assert full["extra_state"] is None

    def test_load_best_full_and_load_best_independent(self, tmp_path):
        """load_best_full 不影响 load_best 的旧调用（返回不同结构）。"""
        ckpt = CheckpointManager(str(tmp_path))
        state = {
            "model_state_dict": {"weight": np.array([1.0])},
            "step": 50,
            "val_loss": 0.5,
            "extra_field": "hello",
        }
        ckpt.save_best(state)

        # load_best 返回原始 dict（含 extra_field）
        loaded = ckpt.load_best()
        assert loaded["extra_field"] == "hello"
        assert loaded["step"] == 50

        # load_best_full 返回统一 dict（不含 extra_field）
        full = ckpt.load_best_full()
        assert "extra_field" not in full
        assert full["step"] == 50
        assert full["best_val_loss"] == pytest.approx(0.5)
