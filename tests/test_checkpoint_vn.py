"""tests/test_checkpoint_vn.py

Part5K1.3 Task 4.5: CheckpointManager 原生 .vn 支持（集成 v2）测试。

覆盖 SubTask 4.5 要求的 5 个测试要点：
1. vn checkpoint 含 optimizer state：save_best 写入 optimizer_state（AdamW m/v 矩阵），
   load_best_full 读取后 m/v 数值一致（float32 吻合 1e-7）
2. 训练中断后 load_best 恢复：save_best 写入完整状态（model + optimizer + step +
   best_val_loss），新建 CheckpointManager 实例 load_best_full 恢复全部字段
3. use_vmpc 强制 .vn：format="pt" + use_vmpc=True 抛 ValueError；
   format="auto" + use_vmpc=True → 选 "vn"
4. v1 向后兼容：手工构造 v1 .vn 文件，CheckpointManager format="vn" 加载，
   load_best_full 的 training_state / optimizer_state / extra_state 返回 None
5. 旧 API 向后兼容：save_best(state_dict) 不传扩展参数仍工作；
   load_best() 返回含 model_state_dict 的 dict

运行方式：
    cd /workspace && python -m pytest tests/test_checkpoint_vn.py -v
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verse_torch.training import CheckpointManager  # noqa: E402
from verse_torch.vn_format import VNFileReader, VN_FORMAT_VERSION  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助：手工构造 v1 .vn 文件（模拟 Part5K1 写出的 v1 格式）
# ---------------------------------------------------------------------------


def _state_dict_to_npz_bytes(state_dict: dict) -> bytes:
    """把 state_dict 序列化为 npz 字节流（与 vn_format._npz_to_bytes 一致）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, arr in state_dict.items():
            arr = np.ascontiguousarray(arr)
            npy_buf = io.BytesIO()
            np.lib.format.write_array(npy_buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", npy_buf.getvalue())
    return buf.getvalue()


def _write_v1_vn_file(
    vn_path: str,
    state_dict: dict,
    config: dict | None = None,
):
    """手工构造 v1 .vn 文件（模拟 Part5K1 写出的 v1 格式）。

    v1 meta.json 不含 vn_format_version 字段（或显式 =1），也不含
    has_training_state / has_optimizer_state / has_extra_state 字段。
    权重走 npz 路径以避免依赖 safetensors。
    """
    if config is None:
        config = {"arch": "versenex", "n_layer": 2}
    npz_buf = _state_dict_to_npz_bytes(state_dict)
    meta = {
        "arch": config.get("arch", "versenex"),
        "weight_format": "npz",
        "compression_info": None,
        "created_at": "2024-01-01T00:00:00",
        "weight_count": len(state_dict),
    }
    with zipfile.ZipFile(vn_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.npz", npz_buf)
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr(
            "config.yml",
            f"arch: {config.get('arch', 'versenex')}\n"
            f"n_layer: {config.get('n_layer', 2)}\n",
        )


# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state_dict():
    """带点号参数名的 state_dict（模拟真实模型）。"""
    rng = np.random.default_rng(42)
    return {
        "blocks.0.attn.q.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "blocks.0.attn.q.bias": rng.standard_normal(8).astype(np.float32),
        "tok_emb.weight": rng.standard_normal((16, 8)).astype(np.float32),
    }


@pytest.fixture
def sample_optimizer_state():
    """AdamW 优化器状态：exp_avg / exp_avg_sq（numpy 数组）+ step + scheduler。"""
    rng = np.random.default_rng(7)
    return {
        "exp_avg": {
            "blocks.0.attn.q.weight": rng.standard_normal((8, 8)).astype(np.float32),
            "blocks.0.attn.q.bias": rng.standard_normal(8).astype(np.float32),
        },
        "exp_avg_sq": {
            "blocks.0.attn.q.weight": rng.standard_normal((8, 8)).astype(np.float32),
        },
        "step": 1234,
        "scheduler_state": {
            "base_lrs": [3e-4, 1e-4],
            "last_epoch": -1,
            "_step_count": 1234,
            "last_lr": 1.5e-4,
        },
    }


@pytest.fixture
def sample_training_state():
    """典型 training_state：JSON-able dict（断点续训所需字段）。"""
    return {
        "step": 1234,
        "epoch": 7,
        "best_val_loss": 0.321,
        "patience_count": 2,
        "rng_state_hex": "a1b2c3d4e5f6",
    }


@pytest.fixture
def sample_extra_state():
    """用户自定义 extra_state。"""
    return {
        "user_metadata": {"experiment_id": "exp-001", "tags": ["baseline", "v2"]},
        "checkpoint_reason": "manual",
    }


# ---------------------------------------------------------------------------
# 1. vn checkpoint 含 optimizer state
# ---------------------------------------------------------------------------


class TestVnCheckpointWithOptimizer:
    """vn checkpoint 含 optimizer state：save_best 写入 optimizer_state，
    load_best_full 读取后 m/v 矩阵数值一致。"""

    def test_save_and_load_optimizer_state(
        self, tmp_path, sample_state_dict, sample_optimizer_state,
    ):
        """save_best 写入 optimizer_state，load_best_full 读取后字段一致。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        state = {
            "model_state_dict": sample_state_dict,
            "step": 1234,
            "best_val_loss": 0.321,
        }
        ckpt.save_best(
            state,
            optimizer_state=sample_optimizer_state,
            step=1234,
        )

        # 验证 .vn 文件存在且无 .tmp 残留
        assert (tmp_path / "best.vn").exists()
        assert not (tmp_path / "best.vn.tmp").exists()

        full = ckpt.load_best_full()
        # optimizer_state 完整还原
        assert full["optimizer_state"] is not None
        assert full["optimizer_state"]["step"] == 1234
        assert full["optimizer_state"]["scheduler_state"]["last_lr"] == 1.5e-4

    def test_optimizer_m_v_matrix_consistency(
        self, tmp_path, sample_state_dict, sample_optimizer_state,
    ):
        """optimizer m/v 矩阵数值一致（float32 吻合 1e-7）。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        ckpt.save_best(
            {"model_state_dict": sample_state_dict},
            optimizer_state=sample_optimizer_state,
        )

        full = ckpt.load_best_full()
        opt = full["optimizer_state"]

        # exp_avg 矩阵数值一致（float32 吻合 1e-7）
        for key, expected in sample_optimizer_state["exp_avg"].items():
            np.testing.assert_allclose(
                opt["exp_avg"][key], expected, atol=1e-7, rtol=1e-7,
            )

        # exp_avg_sq 矩阵数值一致
        for key, expected in sample_optimizer_state["exp_avg_sq"].items():
            np.testing.assert_allclose(
                opt["exp_avg_sq"][key], expected, atol=1e-7, rtol=1e-7,
            )

    def test_vn_meta_version_is_v2(self, tmp_path, sample_state_dict):
        """CheckpointManager 写出的 .vn 文件 meta 版本为 v2。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        ckpt.save_best({"model_state_dict": sample_state_dict})

        vn_path = str(tmp_path / "best.vn")
        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["vn_format_version"] == 2
            assert meta["vn_format_version"] == VN_FORMAT_VERSION
            # 写入 optimizer_state 前 has_optimizer_state=False
            assert meta.get("has_optimizer_state", False) is False

    def test_has_optimizer_state_flag_in_meta(
        self, tmp_path, sample_state_dict, sample_optimizer_state,
    ):
        """写入 optimizer_state 后 meta.json 的 has_optimizer_state=True。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        ckpt.save_best(
            {"model_state_dict": sample_state_dict},
            optimizer_state=sample_optimizer_state,
        )

        vn_path = str(tmp_path / "best.vn")
        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["has_optimizer_state"] is True


# ---------------------------------------------------------------------------
# 2. 训练中断后 load_best 恢复
# ---------------------------------------------------------------------------


class TestVnCheckpointResume:
    """训练中断后 load_best 恢复：模拟训练中断 → 新建 CheckpointManager →
    load_best_full 恢复 model + optimizer + step + best_val_loss。"""

    def test_resume_full_state_recovery(
        self, tmp_path, sample_state_dict, sample_optimizer_state,
        sample_training_state, sample_extra_state,
    ):
        """模拟训练中断后恢复：model + optimizer + step + best_val_loss 全部还原。"""
        # 第一步：模拟训练中断前保存 checkpoint
        ckpt1 = CheckpointManager(str(tmp_path), format="vn")
        state = {
            "model_state_dict": sample_state_dict,
            "step": 1234,
            "best_val_loss": 0.321,
        }
        ckpt1.save_best(
            state,
            training_state=sample_training_state,
            optimizer_state=sample_optimizer_state,
            extra_state=sample_extra_state,
            step=1234,
        )

        # 第二步：新建 CheckpointManager（模拟重启），load_best_full 恢复
        ckpt2 = CheckpointManager(str(tmp_path), format="vn")
        full = ckpt2.load_best_full()

        # model_state_dict 权重一致
        assert full["model_state_dict"] is not None
        for key, expected in sample_state_dict.items():
            np.testing.assert_allclose(
                full["model_state_dict"][key], expected, atol=1e-7,
            )

        # training_state 字段一致
        assert full["training_state"] is not None
        assert full["training_state"]["step"] == 1234
        assert full["training_state"]["epoch"] == 7
        assert full["training_state"]["best_val_loss"] == pytest.approx(0.321)

        # optimizer_state 字段一致
        assert full["optimizer_state"] is not None
        assert full["optimizer_state"]["step"] == 1234
        for key, expected in sample_optimizer_state["exp_avg"].items():
            np.testing.assert_allclose(
                full["optimizer_state"]["exp_avg"][key], expected, atol=1e-7,
            )

        # extra_state 字段一致
        assert full["extra_state"] is not None
        assert full["extra_state"]["user_metadata"]["experiment_id"] == "exp-001"

        # step / best_val_loss 顶层字段
        assert full["step"] == 1234
        assert full["best_val_loss"] == pytest.approx(0.321)

    def test_resume_via_load_best_old_style_access(
        self, tmp_path, sample_state_dict, sample_optimizer_state,
    ):
        """旧式 load_best()["step"] / ["model_state_dict"] 访问仍工作。"""
        ckpt1 = CheckpointManager(str(tmp_path), format="vn")
        ckpt1.save_best(
            {"model_state_dict": sample_state_dict, "step": 500, "val_loss": 0.456},
            optimizer_state=sample_optimizer_state,
            step=500,
        )

        # 新建 CheckpointManager（模拟重启）
        ckpt2 = CheckpointManager(str(tmp_path), format="vn")
        loaded = ckpt2.load_best()

        # 旧式访问：load_best()["step"] / ["model_state_dict"] / ["val_loss"]
        assert loaded["step"] == 500
        assert loaded["val_loss"] == pytest.approx(0.456)
        assert loaded["model_state_dict"] is not None
        np.testing.assert_allclose(
            loaded["model_state_dict"]["blocks.0.attn.q.weight"],
            sample_state_dict["blocks.0.attn.q.weight"],
            atol=1e-7,
        )
        # optimizer_state 也可通过 load_best() 访问
        assert loaded["optimizer_state"] is not None
        assert loaded["optimizer_state"]["step"] == 1234

    def test_save_last_and_resume_from_last(
        self, tmp_path, sample_state_dict, sample_optimizer_state,
    ):
        """save_last 写入 last.vn，load_last_full 恢复。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        ckpt.save_last(
            {"model_state_dict": sample_state_dict, "step": 999},
            optimizer_state=sample_optimizer_state,
            step=999,
        )

        assert (tmp_path / "last.vn").exists()
        full = ckpt.load_last_full()
        assert full["step"] == 999
        assert full["optimizer_state"] is not None
        assert full["optimizer_state"]["step"] == 1234

    def test_atomic_write_no_corruption_on_interrupt(
        self, tmp_path, sample_state_dict,
    ):
        """原子写：写入失败时旧 .vn 未被破坏，.tmp 已清理。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        # 先写一个有效的 best.vn
        ckpt.save_best({"model_state_dict": sample_state_dict, "step": 50})

        # mock VNFileWriter.__init__ 抛异常（模拟写入失败）
        from unittest.mock import patch
        from verse_torch.vn_format import VNFileWriter as _RealWriter

        def _failing_init(self, *args, **kwargs):
            raise OSError("disk full")

        with patch.object(_RealWriter, "__init__", _failing_init):
            with pytest.raises(OSError, match="disk full"):
                ckpt.save_best(
                    {"model_state_dict": sample_state_dict, "step": 100},
                )

        # 旧 best.vn 未被破坏
        full = ckpt.load_best_full()
        assert full["step"] == 50

        # .tmp 已清理
        assert not (tmp_path / "best.vn.tmp").exists()


# ---------------------------------------------------------------------------
# 3. use_vmpc 强制 .vn
# ---------------------------------------------------------------------------


class TestUseVmpcForcesVn:
    """use_vmpc 强制 .vn：format="pt" + use_vmpc=True 抛 ValueError；
    format="auto" + use_vmpc=True → 选 "vn"。"""

    def test_use_vmpc_with_format_pt_raises(self, tmp_path):
        """format="pt" + use_vmpc=True → 抛 ValueError。"""
        with pytest.raises(ValueError, match="use_vmpc"):
            CheckpointManager(str(tmp_path), format="pt", use_vmpc=True)

    def test_use_vmpc_auto_selects_vn(self, tmp_path):
        """format="auto" + use_vmpc=True → 选 "vn"，默认路径 best.vn。"""
        ckpt = CheckpointManager(str(tmp_path), format="auto", use_vmpc=True)
        assert ckpt.format == "vn"
        assert ckpt.best_path == tmp_path / "best.vn"
        assert ckpt.last_path == tmp_path / "last.vn"

    def test_no_vmpc_auto_selects_pt(self, tmp_path):
        """format="auto" + use_vmpc=False → 选 "pt"（向后兼容）。"""
        ckpt = CheckpointManager(str(tmp_path), format="auto", use_vmpc=False)
        assert ckpt.format == "pt"
        assert ckpt.best_path == tmp_path / "best.pt"

    def test_use_vmpc_with_format_vn_ok(self, tmp_path):
        """format="vn" + use_vmpc=True → 正常构造（不抛异常）。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn", use_vmpc=True)
        assert ckpt.format == "vn"
        assert ckpt.use_vmpc is True

    def test_vmpc_writes_vn_file(self, tmp_path, sample_state_dict):
        """use_vmpc=True + format="auto" → 实际写出 .vn 文件。"""
        ckpt = CheckpointManager(str(tmp_path), format="auto", use_vmpc=True)
        ckpt.save_best({"model_state_dict": sample_state_dict})
        assert (tmp_path / "best.vn").exists()
        assert not (tmp_path / "best.pt").exists()


# ---------------------------------------------------------------------------
# 4. v1 向后兼容
# ---------------------------------------------------------------------------


class TestV1BackwardCompat:
    """v1 向后兼容：手工构造 v1 .vn 文件，CheckpointManager format="vn" 加载，
    load_best_full 的 training_state / optimizer_state / extra_state 返回 None。"""

    def test_load_v1_vn_file_returns_none_for_v2_fields(
        self, tmp_path, sample_state_dict,
    ):
        """v1 .vn 文件：load_best_full 返回 model_state_dict，
        training_state / optimizer_state / extra_state 为 None。"""
        # 手工构造 v1 .vn 文件
        v1_path = tmp_path / "best.vn"
        _write_v1_vn_file(str(v1_path), sample_state_dict)

        # 用 CheckpointManager format="vn" 加载
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        full = ckpt.load_best_full()

        # v1 文件有权重
        assert full["model_state_dict"] is not None
        for key, expected in sample_state_dict.items():
            np.testing.assert_allclose(
                full["model_state_dict"][key], expected, atol=1e-7,
            )

        # v1 文件无 v2 字段 → 全部 None
        assert full["training_state"] is None
        assert full["optimizer_state"] is None
        assert full["extra_state"] is None
        assert full["step"] is None
        assert full["best_val_loss"] is None

    def test_load_v1_via_load_best_old_style(self, tmp_path, sample_state_dict):
        """v1 .vn 文件：load_best() 旧式访问返回 model_state_dict。"""
        v1_path = tmp_path / "best.vn"
        _write_v1_vn_file(str(v1_path), sample_state_dict)

        ckpt = CheckpointManager(str(tmp_path), format="vn")
        loaded = ckpt.load_best()

        # v1 文件有权重
        assert loaded["model_state_dict"] is not None
        np.testing.assert_allclose(
            loaded["model_state_dict"]["blocks.0.attn.q.weight"],
            sample_state_dict["blocks.0.attn.q.weight"],
            atol=1e-7,
        )
        # v1 文件无 v2 字段
        assert loaded["training_state"] is None
        assert loaded["optimizer_state"] is None

    def test_v1_vn_meta_version_is_1(self, tmp_path, sample_state_dict):
        """v1 .vn 文件的 VNFileReader.meta.vn_format_version == 1。"""
        v1_path = tmp_path / "best.vn"
        _write_v1_vn_file(str(v1_path), sample_state_dict)

        with VNFileReader(str(v1_path)) as r:
            assert r.vn_format_version == 1
            assert r.read_training_state() is None
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() is None


# ---------------------------------------------------------------------------
# 5. 旧 API 向后兼容
# ---------------------------------------------------------------------------


class TestOldApiBackwardCompat:
    """旧 API 向后兼容：save_best(state_dict) 不传扩展参数仍工作；
    load_best() 返回含 model_state_dict 的 dict。"""

    def test_save_best_pure_state_dict_vn(self, tmp_path, sample_state_dict):
        """save_best(纯 state_dict) 不传 model_state_dict 键也工作（format="vn"）。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        # 纯 state_dict（所有值为 ndarray）
        ckpt.save_best(sample_state_dict)
        assert (tmp_path / "best.vn").exists()

        loaded = ckpt.load_best()
        # 纯 state_dict 被识别为 model weights
        assert loaded["model_state_dict"] is not None
        for key, expected in sample_state_dict.items():
            np.testing.assert_allclose(
                loaded["model_state_dict"][key], expected, atol=1e-7,
            )

    def test_save_best_default_none_args_vn(self, tmp_path, sample_state_dict):
        """save_best 不传 training_state / optimizer_state / extra_state 仍工作。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        ckpt.save_best({"model_state_dict": sample_state_dict})
        full = ckpt.load_best_full()
        # 权重正常
        assert full["model_state_dict"] is not None
        # 缺失字段为 None
        assert full["optimizer_state"] is None
        assert full["extra_state"] is None

    def test_format_pt_path_unchanged(self, tmp_path):
        """format="pt" 路径行为不变（pickle 读写）。"""
        ckpt = CheckpointManager(str(tmp_path), format="pt")
        state = {"step": 42, "data": "hello", "arr": np.array([1.0, 2.0])}
        ckpt.save_best(state)

        assert (tmp_path / "best.pt").exists()
        loaded = ckpt.load_best()
        # .pt 路径返回原始 dict（经 pickle 往返）
        assert loaded["step"] == 42
        assert loaded["data"] == "hello"
        np.testing.assert_allclose(loaded["arr"], [1.0, 2.0])

    def test_default_no_format_args_backward_compat(self, tmp_path):
        """不传 format/use_vmpc → 默认 auto + use_vmpc=False → pt（向后兼容）。"""
        ckpt = CheckpointManager(str(tmp_path))
        assert ckpt.format == "pt"
        assert ckpt.use_vmpc is False
        assert ckpt.best_path == tmp_path / "best.pt"

    def test_custom_path_overrides_format_default_vn(self, tmp_path, sample_state_dict):
        """自定义 best_path 优先于 format 默认扩展名（format="vn" 时）。"""
        custom_best = tmp_path / "my_best.vn"
        ckpt = CheckpointManager(
            str(tmp_path),
            best_path=custom_best,
            format="vn",
        )
        assert ckpt.best_path == custom_best
        ckpt.save_best({"model_state_dict": sample_state_dict})
        assert custom_best.exists()

    def test_save_last_load_last_vn_roundtrip(self, tmp_path, sample_state_dict):
        """save_last / load_last 在 format="vn" 下完整往返。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        ckpt.save_last(
            {"model_state_dict": sample_state_dict, "step": 777},
            step=777,
        )
        full = ckpt.load_last_full()
        assert full["step"] == 777
        assert full["model_state_dict"] is not None
        np.testing.assert_allclose(
            full["model_state_dict"]["tok_emb.weight"],
            sample_state_dict["tok_emb.weight"],
            atol=1e-7,
        )

    def test_resolve_path_vn(self, tmp_path):
        """_resolve_path 方法：format="vn" → save_dir/{name}.vn。"""
        ckpt = CheckpointManager(str(tmp_path), format="vn")
        assert ckpt._resolve_path("best") == tmp_path / "best.vn"
        assert ckpt._resolve_path("last") == tmp_path / "last.vn"

    def test_resolve_path_pt(self, tmp_path):
        """_resolve_path 方法：format="pt" → save_dir/{name}.pt。"""
        ckpt = CheckpointManager(str(tmp_path), format="pt")
        assert ckpt._resolve_path("best") == tmp_path / "best.pt"
        assert ckpt._resolve_path("last") == tmp_path / "last.pt"
