""".vn v2 格式测试（Part5K1.3 Task 3.9）。

覆盖 SubTask 3.9 要求的 4 个测试要点：
1. v2 完整读写：weights + chat_template + training_state + optimizer_state + extra_state
   + optimizer m/v 矩阵数值一致（float32 吻合 1e-7）
2. v1 向后兼容：Part5K1 写出的 v1 文件用 v2 VNFileReader 读取，
   read 新方法返回 None，read_weights() 正常
3. 缺失字段优雅处理：v2 文件但仅写 weights，read 新方法返回 None
4. 复杂 Python 对象：optimizer_state 含 AdamW exp_avg/exp_avg_sq（numpy 数组）
   + step（int）+ scheduler state（嵌套 dict），pickle 序列化/反序列化无损

运行方式：
    cd /workspace && python -m pytest tests/test_vn_v2_format.py -v
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

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

from verse_torch import (  # noqa: E402
    VNFileReader,
    VNFileWriter,
    VN_FORMAT_VERSION,
    VN_ENTRY_TRAINING_STATE,
    VN_ENTRY_OPTIMIZER_STATE,
    VN_ENTRY_EXTRA_STATE,
)


# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """临时目录 fixture。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_state_dict():
    """带点号参数名的 state_dict（模拟真实模型）。"""
    rng = np.random.default_rng(42)
    return {
        "blocks.0.attn.q.weight": rng.standard_normal((8, 8)).astype(np.float32),
        "blocks.0.attn.q.bias": rng.standard_normal(8).astype(np.float32),
        "tok_emb.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "lm_head.bias": np.zeros(8, dtype=np.float32),
    }


@pytest.fixture
def sample_config():
    """样本模型配置。"""
    return {
        "arch": "versenex",
        "n_layer": 2,
        "n_embd": 64,
        "vocab_size": 256,
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
        "config_snapshot_hash": "sha256:deadbeef",
    }


@pytest.fixture
def sample_optimizer_state():
    """AdamW 优化器状态：exp_avg / exp_avg_sq（numpy 数组）+ step + scheduler。"""
    rng = np.random.default_rng(7)
    return {
        "exp_avg": {
            "blocks.0.attn.q.weight": rng.standard_normal((8, 8)).astype(np.float32),
            "blocks.0.attn.q.bias": rng.standard_normal(8).astype(np.float32),
            "tok_emb.weight": rng.standard_normal((16, 8)).astype(np.float32),
        },
        "exp_avg_sq": {
            "blocks.0.attn.q.weight": rng.standard_normal((8, 8)).astype(np.float32),
            "tok_emb.weight": rng.standard_normal((16, 8)).astype(np.float32),
        },
        "step": 1234,
        "scheduler_state": {
            "base_lrs": [3e-4, 1e-4],
            "last_epoch": -1,
            "_step_count": 1234,
            "last_lr": 1.5e-4,
            "nested": {"alpha": 0.9, "beta": [1, 2, 3]},
        },
    }


@pytest.fixture
def sample_extra_state():
    """用户自定义 extra_state：dict + 嵌套结构 + 自定义类型。"""
    return {
        "user_metadata": {"experiment_id": "exp-001", "tags": ["baseline", "v2"]},
        "gradient_history": [0.1, 0.2, 0.3, 0.4],
        "checkpoint_reason": "manual",
    }


# ---------------------------------------------------------------------------
# 辅助：手工构造 v1 .vn 文件（模拟 Part5K1 写出的 v1 格式）
# ---------------------------------------------------------------------------


def _write_v1_file(
    vn_path: str,
    state_dict: dict,
    config: dict,
    chat_template: str = None,
    tokenizer: dict = None,
):
    """手工构造 v1 .vn 文件。

    v1 meta.json 不含 vn_format_version 字段（或显式 =1），也不含
    has_training_state / has_optimizer_state / has_extra_state 字段。
    权重走 npz 路径以避免依赖 safetensors。
    """
    # 构造 npz 字节流（与 vn_format._npz_to_bytes 一致的逻辑）
    npz_buf = _state_dict_to_npz_bytes(state_dict)

    # v1 meta（不含 vn_format_version，模拟 Part5K1 写出的旧文件）
    meta = {
        "arch": "versenex",
        "weight_format": "npz",
        "compression_info": None,
        "created_at": "2024-01-01T00:00:00",
        "weight_count": len(state_dict),
    }

    with zipfile.ZipFile(vn_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.npz", npz_buf)
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        zf.writestr("config.yml", f"arch: {config['arch']}\nn_layer: {config['n_layer']}\n")
        if chat_template is not None:
            zf.writestr("chat_template.jinja", chat_template)
        if tokenizer is not None:
            zf.writestr("tokenizer.json", json.dumps(tokenizer, ensure_ascii=False))


def _state_dict_to_npz_bytes(state_dict: dict) -> bytes:
    """把 state_dict 序列化为 npz 字节流（与 vn_format._npz_to_bytes 一致）。"""
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, arr in state_dict.items():
            arr = np.ascontiguousarray(arr)
            npy_buf = io.BytesIO()
            np.lib.format.write_array(npy_buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", npy_buf.getvalue())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. v2 完整读写
# ---------------------------------------------------------------------------


class TestV2FullWriteRead:
    """v2 完整写入 + 读取 round-trip。"""

    def test_v2_full_roundtrip(
        self, tmp_dir, sample_state_dict, sample_config,
        sample_training_state, sample_optimizer_state, sample_extra_state,
    ):
        """写入 weights + chat_template + training_state + optimizer_state + extra_state，
        读取后所有字段一致。"""
        vn_path = os.path.join(tmp_dir, "full_v2.vn")
        chat_tmpl = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
        tokenizer = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}}

        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_chat_template(chat_tmpl)
            w.write_tokenizer(tokenizer)
            w.write_training_state(sample_training_state)
            w.write_optimizer_state(sample_optimizer_state)
            w.write_extra_state(sample_extra_state)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            # v2 版本号
            assert meta["vn_format_version"] == 2
            assert meta["vn_format_version"] == VN_FORMAT_VERSION
            # v2 新增 has_* 字段全为 True
            assert meta["has_training_state"] is True
            assert meta["has_optimizer_state"] is True
            assert meta["has_extra_state"] is True
            # vn_format_version 属性
            assert r.vn_format_version == 2

            # 权重正常
            sd = r.read_weights()
            assert set(sd.keys()) == set(sample_state_dict.keys())
            for k, v in sample_state_dict.items():
                assert np.array_equal(sd[k], v), f"权重不一致: {k}"

            # chat_template + tokenizer
            assert r.read_chat_template() == chat_tmpl
            assert r.read_tokenizer() == tokenizer

            # training_state：JSON dict 完全相等
            ts = r.read_training_state()
            assert ts == sample_training_state
            assert ts["step"] == 1234
            assert ts["best_val_loss"] == pytest.approx(0.321)

            # extra_state：完全相等
            es = r.read_extra_state()
            assert es == sample_extra_state

    def test_v2_optimizer_state_matrix_consistency(
        self, tmp_dir, sample_state_dict, sample_config, sample_optimizer_state,
    ):
        """optimizer m/v 矩阵数值一致（float32 吻合 1e-7）。"""
        vn_path = os.path.join(tmp_dir, "opt_v2.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_optimizer_state(sample_optimizer_state)

        with VNFileReader(vn_path) as r:
            opt_state = r.read_optimizer_state()

        # exp_avg / exp_avg_sq 矩阵数值一致（float32 吻合 1e-7）
        for k, expected in sample_optimizer_state["exp_avg"].items():
            actual = opt_state["exp_avg"][k]
            assert actual.dtype == np.float32
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected, atol=1e-7, rtol=0), \
                f"exp_avg 不一致: {k}"

        for k, expected in sample_optimizer_state["exp_avg_sq"].items():
            actual = opt_state["exp_avg_sq"][k]
            assert actual.dtype == np.float32
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected, atol=1e-7, rtol=0), \
                f"exp_avg_sq 不一致: {k}"

        # 标量 + 嵌套 dict 也一致
        assert opt_state["step"] == 1234
        assert opt_state["scheduler_state"]["base_lrs"] == [3e-4, 1e-4]
        assert opt_state["scheduler_state"]["last_lr"] == pytest.approx(1.5e-4)
        assert opt_state["scheduler_state"]["nested"] == {"alpha": 0.9, "beta": [1, 2, 3]}

    def test_v2_meta_has_flags_default_false(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """v2 文件仅写 weights 时，has_* 字段全为 False。"""
        vn_path = os.path.join(tmp_dir, "weights_only.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["vn_format_version"] == 2
            assert meta["has_training_state"] is False
            assert meta["has_optimizer_state"] is False
            assert meta["has_extra_state"] is False

    def test_v2_zip_entry_names(
        self, tmp_dir, sample_state_dict, sample_config,
        sample_training_state, sample_optimizer_state, sample_extra_state,
    ):
        """v2 文件 ZIP 内包含新增条目（training_state.json / optimizer_state.pkl / extra_state.pkl）。"""
        vn_path = os.path.join(tmp_dir, "entries.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_training_state(sample_training_state)
            w.write_optimizer_state(sample_optimizer_state)
            w.write_extra_state(sample_extra_state)

        with zipfile.ZipFile(vn_path, "r") as zf:
            names = set(zf.namelist())

        assert VN_ENTRY_TRAINING_STATE in names
        assert VN_ENTRY_OPTIMIZER_STATE in names
        assert VN_ENTRY_EXTRA_STATE in names
        assert "meta.json" in names
        assert "config.yml" in names


# ---------------------------------------------------------------------------
# 2. v1 向后兼容
# ---------------------------------------------------------------------------


class TestV1BackwardCompat:
    """v1 文件用 v2 VNFileReader 读取，新方法返回 None，权重正常。"""

    def test_v1_file_read_by_v2_reader(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """Part5K1 写出的 v1 文件（无 vn_format_version 字段）被 v2 reader 加载。"""
        vn_path = os.path.join(tmp_dir, "v1.vn")
        _write_v1_file(vn_path, sample_state_dict, sample_config)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            # v1 文件无 vn_format_version 字段，默认 1
            assert meta.get("vn_format_version", 1) == 1
            assert r.vn_format_version == 1

            # 新方法返回 None
            assert r.read_training_state() is None
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() is None

            # 权重正常读取
            sd = r.read_weights()
            assert set(sd.keys()) == set(sample_state_dict.keys())
            for k, v in sample_state_dict.items():
                assert np.array_equal(sd[k], v), f"v1 权重读取不一致: {k}"

    def test_v1_file_with_explicit_version_1(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """v1 文件 meta 显式声明 vn_format_version=1 也能被读取。"""
        vn_path = os.path.join(tmp_dir, "v1_explicit.vn")
        # 修改 _write_v1_file 输出，显式设置 vn_format_version=1
        npz_buf = _state_dict_to_npz_bytes(sample_state_dict)
        meta = {
            "vn_format_version": 1,  # 显式声明 v1
            "arch": "versenex",
            "weight_format": "npz",
            "compression_info": None,
            "created_at": "2024-01-01T00:00:00",
            "weight_count": len(sample_state_dict),
        }
        with zipfile.ZipFile(vn_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.npz", npz_buf)
            zf.writestr("meta.json", json.dumps(meta))
            zf.writestr("config.yml", f"arch: {sample_config['arch']}\n")

        with VNFileReader(vn_path) as r:
            assert r.read_meta()["vn_format_version"] == 1
            assert r.vn_format_version == 1
            assert r.read_training_state() is None
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() is None
            # 权重正常
            sd = r.read_weights()
            for k, v in sample_state_dict.items():
                assert np.array_equal(sd[k], v)

    def test_v1_file_with_chat_template_and_tokenizer(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """v1 文件含 chat_template + tokenizer 也能被 v2 reader 读取。"""
        vn_path = os.path.join(tmp_dir, "v1_extras.vn")
        chat_tmpl = "{{ prompt }}"
        tokenizer = {"vocab": ["a", "b"]}
        _write_v1_file(
            vn_path, sample_state_dict, sample_config,
            chat_template=chat_tmpl, tokenizer=tokenizer,
        )

        with VNFileReader(vn_path) as r:
            assert r.read_meta().get("vn_format_version", 1) == 1
            assert r.read_chat_template() == chat_tmpl
            assert r.read_tokenizer() == tokenizer
            # v1 文件新方法返回 None
            assert r.read_training_state() is None
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() is None


# ---------------------------------------------------------------------------
# 3. 缺失字段优雅处理
# ---------------------------------------------------------------------------


class TestMissingFieldsGraceful:
    """v2 文件但仅写 weights（未调 write_training_state 等），read 新方法返回 None。"""

    def test_v2_weights_only_new_methods_return_none(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """v2 文件仅写 weights，read_training_state/optimizer/extra 返回 None。"""
        vn_path = os.path.join(tmp_dir, "weights_only.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["vn_format_version"] == 2
            assert meta["has_training_state"] is False
            assert meta["has_optimizer_state"] is False
            assert meta["has_extra_state"] is False

            # 新方法全部返回 None
            assert r.read_training_state() is None
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() is None

            # 权重正常
            sd = r.read_weights()
            for k, v in sample_state_dict.items():
                assert np.array_equal(sd[k], v)

    def test_v2_partial_state_only_training(
        self, tmp_dir, sample_state_dict, sample_config, sample_training_state,
    ):
        """v2 文件只写 training_state，read_optimizer/extra 返回 None。"""
        vn_path = os.path.join(tmp_dir, "partial.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_training_state(sample_training_state)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["has_training_state"] is True
            assert meta["has_optimizer_state"] is False
            assert meta["has_extra_state"] is False

            assert r.read_training_state() == sample_training_state
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() is None

    def test_v2_partial_state_only_optimizer(
        self, tmp_dir, sample_state_dict, sample_config, sample_optimizer_state,
    ):
        """v2 文件只写 optimizer_state，read_training/extra 返回 None。"""
        vn_path = os.path.join(tmp_dir, "partial_opt.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_optimizer_state(sample_optimizer_state)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["has_training_state"] is False
            assert meta["has_optimizer_state"] is True
            assert meta["has_extra_state"] is False

            assert r.read_training_state() is None
            opt = r.read_optimizer_state()
            assert opt is not None
            assert opt["step"] == sample_optimizer_state["step"]
            assert r.read_extra_state() is None

    def test_v2_partial_state_only_extra(
        self, tmp_dir, sample_state_dict, sample_config, sample_extra_state,
    ):
        """v2 文件只写 extra_state，read_training/optimizer 返回 None。"""
        vn_path = os.path.join(tmp_dir, "partial_extra.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_extra_state(sample_extra_state)

        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["has_training_state"] is False
            assert meta["has_optimizer_state"] is False
            assert meta["has_extra_state"] is True

            assert r.read_training_state() is None
            assert r.read_optimizer_state() is None
            assert r.read_extra_state() == sample_extra_state


# ---------------------------------------------------------------------------
# 4. 复杂 Python 对象 pickle 序列化
# ---------------------------------------------------------------------------


class TestComplexPythonObjects:
    """optimizer_state / extra_state 含复杂 Python 对象，pickle 序列化/反序列化无损。"""

    def test_adamw_full_state_pickle_roundtrip(
        self, tmp_dir, sample_state_dict, sample_config, sample_optimizer_state,
    ):
        """AdamW exp_avg / exp_avg_sq（numpy 数组）+ step + scheduler state 全部能 pickle。"""
        vn_path = os.path.join(tmp_dir, "complex_opt.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_optimizer_state(sample_optimizer_state)

        with VNFileReader(vn_path) as r:
            opt_state = r.read_optimizer_state()

        # 检查 numpy 数组类型 + 数值
        for k in sample_optimizer_state["exp_avg"]:
            assert isinstance(opt_state["exp_avg"][k], np.ndarray)
            assert opt_state["exp_avg"][k].dtype == np.float32
            assert np.allclose(
                opt_state["exp_avg"][k],
                sample_optimizer_state["exp_avg"][k],
                atol=1e-7,
            )

        # 标量类型保留
        assert isinstance(opt_state["step"], int)
        assert opt_state["step"] == 1234

        # 嵌套 dict 完整
        sched_expected = sample_optimizer_state["scheduler_state"]
        sched_actual = opt_state["scheduler_state"]
        assert sched_actual["base_lrs"] == sched_expected["base_lrs"]
        assert sched_actual["last_epoch"] == sched_expected["last_epoch"]
        assert sched_actual["_step_count"] == sched_expected["_step_count"]
        assert sched_actual["last_lr"] == pytest.approx(sched_expected["last_lr"])
        assert sched_actual["nested"] == sched_expected["nested"]

    def test_extra_state_arbitrary_objects(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """extra_state 承载任意 Python 对象（list / dict / tuple / 自定义类实例）。"""
        # 定义一个局部类（pickle 能序列化模块级类，局部类不行）
        # 用模块级 dict / tuple / list / 嵌套结构测试
        extra = {
            "tuple_value": (1, "two", 3.0),
            "nested_list": [[1, 2], [3, 4], {"deep": [5, 6]}],
            "none_value": None,
            "bool_values": [True, False, True],
            "unicode_str": "你好，世界 🌍",
            "frozenset_value": frozenset([1, 2, 3]),
            "large_array": np.arange(100, dtype=np.int64),
            "float_array": np.linspace(0, 1, 50, dtype=np.float64),
        }
        vn_path = os.path.join(tmp_dir, "complex_extra.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_extra_state(extra)

        with VNFileReader(vn_path) as r:
            loaded = r.read_extra_state()

        # tuple 类型保留
        assert loaded["tuple_value"] == (1, "two", 3.0)
        assert isinstance(loaded["tuple_value"], tuple)

        # 嵌套 list / dict
        assert loaded["nested_list"] == [[1, 2], [3, 4], {"deep": [5, 6]}]

        # None / bool
        assert loaded["none_value"] is None
        assert loaded["bool_values"] == [True, False, True]

        # Unicode
        assert loaded["unicode_str"] == "你好，世界 🌍"

        # frozenset
        assert loaded["frozenset_value"] == frozenset([1, 2, 3])
        assert isinstance(loaded["frozenset_value"], frozenset)

        # numpy 数组
        assert isinstance(loaded["large_array"], np.ndarray)
        assert loaded["large_array"].dtype == np.int64
        assert np.array_equal(loaded["large_array"], np.arange(100, dtype=np.int64))

        assert isinstance(loaded["float_array"], np.ndarray)
        assert np.allclose(loaded["float_array"], np.linspace(0, 1, 50))

    def test_training_state_json_serialization(
        self, tmp_dir, sample_state_dict, sample_config, sample_training_state,
    ):
        """training_state 用 JSON 序列化，所有字段类型正确。"""
        vn_path = os.path.join(tmp_dir, "ts.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_training_state(sample_training_state)

        with VNFileReader(vn_path) as r:
            ts = r.read_training_state()

        # 类型 + 值
        assert isinstance(ts["step"], int)
        assert ts["step"] == 1234
        assert isinstance(ts["epoch"], int)
        assert ts["epoch"] == 7
        assert isinstance(ts["best_val_loss"], float)
        assert ts["best_val_loss"] == pytest.approx(0.321)
        assert isinstance(ts["patience_count"], int)
        assert isinstance(ts["rng_state_hex"], str)
        assert isinstance(ts["config_snapshot_hash"], str)

        # 验证 training_state.json 是 JSON 文本（不是 pickle）
        with zipfile.ZipFile(vn_path, "r") as zf:
            raw = zf.read(VN_ENTRY_TRAINING_STATE).decode("utf-8")
        parsed = json.loads(raw)
        assert parsed == sample_training_state


# ---------------------------------------------------------------------------
# 5. 重复写入保护 + 错误处理
# ---------------------------------------------------------------------------


class TestWriteProtection:
    """write_training_state / write_optimizer_state / write_extra_state 不可重复调用。"""

    def test_duplicate_write_training_state_raises(
        self, tmp_dir, sample_state_dict, sample_config, sample_training_state,
    ):
        """重复 write_training_state 抛 RuntimeError。"""
        vn_path = os.path.join(tmp_dir, "dup_ts.vn")
        w = VNFileWriter(vn_path, arch="versenex", config=sample_config)
        try:
            w.write_weights(sample_state_dict)
            w.write_training_state(sample_training_state)
            with pytest.raises(RuntimeError, match="training_state 已写入"):
                w.write_training_state(sample_training_state)
        finally:
            w.close()

    def test_duplicate_write_optimizer_state_raises(
        self, tmp_dir, sample_state_dict, sample_config, sample_optimizer_state,
    ):
        """重复 write_optimizer_state 抛 RuntimeError。"""
        vn_path = os.path.join(tmp_dir, "dup_opt.vn")
        w = VNFileWriter(vn_path, arch="versenex", config=sample_config)
        try:
            w.write_weights(sample_state_dict)
            w.write_optimizer_state(sample_optimizer_state)
            with pytest.raises(RuntimeError, match="optimizer_state 已写入"):
                w.write_optimizer_state(sample_optimizer_state)
        finally:
            w.close()

    def test_duplicate_write_extra_state_raises(
        self, tmp_dir, sample_state_dict, sample_config, sample_extra_state,
    ):
        """重复 write_extra_state 抛 RuntimeError。"""
        vn_path = os.path.join(tmp_dir, "dup_extra.vn")
        w = VNFileWriter(vn_path, arch="versenex", config=sample_config)
        try:
            w.write_weights(sample_state_dict)
            w.write_extra_state(sample_extra_state)
            with pytest.raises(RuntimeError, match="extra_state 已写入"):
                w.write_extra_state(sample_extra_state)
        finally:
            w.close()

    def test_write_training_state_type_check(
        self, tmp_dir, sample_state_dict, sample_config,
    ):
        """training_state 必须 dict，否则抛 TypeError。"""
        vn_path = os.path.join(tmp_dir, "type_check.vn")
        w = VNFileWriter(vn_path, arch="versenex", config=sample_config)
        try:
            w.write_weights(sample_state_dict)
            with pytest.raises(TypeError, match="training_state 必须是 dict"):
                w.write_training_state([1, 2, 3])  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="optimizer_state 必须是 dict"):
                w.write_optimizer_state("not a dict")  # type: ignore[arg-type]
        finally:
            w.close()


# ---------------------------------------------------------------------------
# 6. v1/v2 混合场景：read_meta 多次调用缓存
# ---------------------------------------------------------------------------


class TestMetaCacheAndVersion:
    """read_meta 缓存 + vn_format_version 属性。"""

    def test_read_meta_cached(
        self, tmp_dir, sample_state_dict, sample_config, sample_training_state,
    ):
        """多次调用 read_meta 返回同一对象（缓存生效）。"""
        vn_path = os.path.join(tmp_dir, "cache.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_training_state(sample_training_state)

        with VNFileReader(vn_path) as r:
            meta1 = r.read_meta()
            meta2 = r.read_meta()
            assert meta1 is meta2  # 同一对象
            assert r.vn_format_version == 2

    def test_unknown_version_rejected(self, tmp_dir, sample_state_dict, sample_config):
        """版本号 999 仍然被拒绝（v1/v2 之外的版本）。"""
        vn_path = os.path.join(tmp_dir, "v999.vn")
        # 手工构造版本 999 文件
        npz_buf = _state_dict_to_npz_bytes(sample_state_dict)
        meta = {
            "vn_format_version": 999,
            "arch": "versenex",
            "weight_format": "npz",
            "compression_info": None,
            "created_at": "2024-01-01T00:00:00",
            "weight_count": len(sample_state_dict),
        }
        with zipfile.ZipFile(vn_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.npz", npz_buf)
            zf.writestr("meta.json", json.dumps(meta))
            zf.writestr("config.yml", f"arch: {sample_config['arch']}\n")

        with VNFileReader(vn_path) as r:
            with pytest.raises(ValueError, match="不支持的 .vn 格式版本"):
                r.read_meta()
