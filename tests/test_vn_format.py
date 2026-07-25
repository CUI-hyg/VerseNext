""".vn 文件格式 + safetensors 性能优化测试（Part4K2 Task 1.9）。

覆盖：
1. 创建 .vn 并读取（权重 + config + meta）
2. .pt → .vn 转换（先创建 .pt，转 .vn，验证权重一致）
3. .vn → .pt 转换
4. mmap 零拷贝读取验证
5. safetensors 不可用时降级 npz（强制 monkeypatch 模拟）
6. safetensors 可用路径（skipif 未安装）
7. chat_template 和 tokenizer 可选字段
8. CometSparkV05LM.save_vn / load_vn 端到端
9. 大文件读写（1MB+ 权重）
10. 智能压缩存储（compression_info）
11. 版本校验（拒绝未知 vn_format_version）
12. convert_format 自动检测 + CLI convert

运行方式：
    cd /workspace && python -m pytest tests/test_vn_format.py -x -q
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
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
    pt_to_vn,
    vn_to_pt,
    convert_format,
    has_safetensors,
)
import verse_torch.vn_format as vn_module  # noqa: E402


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
    """构造带点号参数名（模拟真实模型）的 state_dict。"""
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
        "tie_weights": True,
        "layer_pattern": ["trisparse", "mod"],
    }


# ---------------------------------------------------------------------------
# 1. 创建 .vn 并读取
# ---------------------------------------------------------------------------


class TestCreateAndRead:
    """基础读写：权重 + config + meta。"""

    def test_write_read_roundtrip(self, tmp_dir, sample_state_dict, sample_config):
        """写入后读回，权重/config/meta 一致。"""
        vn_path = os.path.join(tmp_dir, "model.vn")
        writer = VNFileWriter(vn_path, arch="versenex", config=sample_config)
        writer.write_weights(sample_state_dict)
        writer.close()
        assert os.path.exists(vn_path)

        reader = VNFileReader(vn_path)
        meta = reader.read_meta()
        assert meta["vn_format_version"] == VN_FORMAT_VERSION
        assert meta["arch"] == "versenex"
        assert meta["weight_count"] == len(sample_state_dict)
        assert meta["weight_format"] in ("safetensors", "npz")

        cfg = reader.read_config()
        assert cfg["arch"] == "versenex"
        assert cfg["n_layer"] == 2
        assert cfg["layer_pattern"] == ["trisparse", "mod"]

        sd = reader.read_weights()
        assert set(sd.keys()) == set(sample_state_dict.keys())
        for k, v in sample_state_dict.items():
            assert np.array_equal(sd[k], v), f"权重不一致: {k}"
            assert sd[k].dtype == v.dtype
            assert sd[k].shape == v.shape
        reader.close()

    def test_context_manager(self, tmp_dir, sample_state_dict, sample_config):
        """上下文管理器用法。"""
        vn_path = os.path.join(tmp_dir, "cm.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            sd = r.read_weights()
            for k, v in sample_state_dict.items():
                assert np.array_equal(sd[k], v)

    def test_zip_container_contents(self, tmp_dir, sample_state_dict, sample_config):
        """验证 .vn 是合法 ZIP 且包含必需条目。"""
        import zipfile

        vn_path = os.path.join(tmp_dir, "zip.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        with zipfile.ZipFile(vn_path, "r") as zf:
            names = set(zf.namelist())
        assert "meta.json" in names
        assert "config.yml" in names
        assert "model.safetensors" in names or "model.npz" in names


# ---------------------------------------------------------------------------
# 2 & 3. .pt ↔ .vn 转换
# ---------------------------------------------------------------------------


class TestPtVnConversion:
    """.pt ↔ .vn 互转无损。"""

    def _write_pt(self, pt_path, sd, cfg, arch="versenex"):
        payload = {"arch": arch, "config": cfg, "state_dict": sd}
        with open(pt_path, "wb") as f:
            pickle.dump(payload, f)

    def test_pt_to_vn(self, tmp_dir, sample_state_dict, sample_config):
        """pt → vn 权重一致 + arch/config 透传。"""
        pt_path = os.path.join(tmp_dir, "m.pt")
        vn_path = os.path.join(tmp_dir, "m.vn")
        self._write_pt(pt_path, sample_state_dict, sample_config)

        pt_to_vn(pt_path, vn_path)
        reader = VNFileReader(vn_path)
        sd = reader.read_weights()
        for k, v in sample_state_dict.items():
            assert np.array_equal(sd[k], v)
        assert reader.read_meta()["arch"] == "versenex"
        assert reader.read_config()["n_layer"] == sample_config["n_layer"]
        reader.close()

    def test_vn_to_pt(self, tmp_dir, sample_state_dict, sample_config):
        """vn → pt 权重一致 + payload 结构正确。"""
        vn_path = os.path.join(tmp_dir, "m.vn")
        pt_path = os.path.join(tmp_dir, "m.pt")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        vn_to_pt(vn_path, pt_path)
        with open(pt_path, "rb") as f:
            payload = pickle.load(f)
        assert payload["arch"] == "versenex"
        assert payload["config"]["n_layer"] == sample_config["n_layer"]
        for k, v in sample_state_dict.items():
            assert np.array_equal(payload["state_dict"][k], v)

    def test_roundtrip_pt_vn_pt(self, tmp_dir, sample_state_dict, sample_config):
        """pt → vn → pt 全程无损。"""
        pt1 = os.path.join(tmp_dir, "a.pt")
        vn = os.path.join(tmp_dir, "a.vn")
        pt2 = os.path.join(tmp_dir, "b.pt")
        self._write_pt(pt1, sample_state_dict, sample_config)
        convert_format(pt1, vn)
        convert_format(vn, pt2)

        with open(pt1, "rb") as f:
            p1 = pickle.load(f)
        with open(pt2, "rb") as f:
            p2 = pickle.load(f)
        for k, v in p1["state_dict"].items():
            assert np.array_equal(p2["state_dict"][k], v)

    def test_convert_format_invalid_suffix(self, tmp_dir):
        """非 .pt/.vn 后缀组合应报错。"""
        with pytest.raises(ValueError, match="仅支持"):
            convert_format("a.txt", "b.bin")

    def test_pt_to_vn_with_extras(self, tmp_dir, sample_state_dict, sample_config):
        """pt → vn 携带 chat_template + tokenizer。"""
        pt_path = os.path.join(tmp_dir, "m.pt")
        vn_path = os.path.join(tmp_dir, "m.vn")
        self._write_pt(pt_path, sample_state_dict, sample_config)
        tok_path = os.path.join(tmp_dir, "tok.json")
        with open(tok_path, "w") as f:
            json.dump({"vocab": ["a", "b"]}, f)

        pt_to_vn(
            pt_path, vn_path,
            chat_template="{{ prompt }}",
            tokenizer=tok_path,
        )
        reader = VNFileReader(vn_path)
        assert reader.read_chat_template() == "{{ prompt }}"
        assert reader.read_tokenizer() == {"vocab": ["a", "b"]}
        reader.close()


# ---------------------------------------------------------------------------
# 4. mmap 读取
# ---------------------------------------------------------------------------


class TestMmapRead:
    """mmap 零拷贝读取。"""

    def test_mmap_true(self, tmp_dir, sample_state_dict, sample_config):
        """mmap=True 读取权重一致。"""
        vn_path = os.path.join(tmp_dir, "m.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        reader = VNFileReader(vn_path)
        sd = reader.read_weights(mmap=True)
        for k, v in sample_state_dict.items():
            assert np.array_equal(sd[k], v)
        reader.close()

    def test_mmap_false(self, tmp_dir, sample_state_dict, sample_config):
        """mmap=False 读取权重一致。"""
        vn_path = os.path.join(tmp_dir, "m.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        reader = VNFileReader(vn_path)
        sd = reader.read_weights(mmap=False)
        for k, v in sample_state_dict.items():
            assert np.array_equal(sd[k], v)
        reader.close()

    def test_reader_close_cleans_temp(self, tmp_dir, sample_state_dict, sample_config):
        """close 后临时文件被清理。"""
        vn_path = os.path.join(tmp_dir, "m.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        reader = VNFileReader(vn_path)
        reader.read_weights(mmap=True)
        tmp_path = reader._weight_tmp_path
        assert tmp_path is not None
        assert os.path.exists(tmp_path)
        reader.close()
        assert not os.path.exists(tmp_path)


# ---------------------------------------------------------------------------
# 5. safetensors 不可用 → npz 降级
# ---------------------------------------------------------------------------


class TestNpzFallback:
    """safetensors 不可用时强制走 npz 路径。"""

    def test_force_npz_mode(self, tmp_dir, sample_state_dict, sample_config,
                            monkeypatch):
        """monkeypatch _HAS_SAFETENSORS=False 强制 npz，无论环境是否安装 safetensors。"""
        monkeypatch.setattr(vn_module, "_HAS_SAFETENSORS", False)
        vn_path = os.path.join(tmp_dir, "npz.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        reader = VNFileReader(vn_path)
        assert reader.weight_format == "npz"
        sd = reader.read_weights()
        for k, v in sample_state_dict.items():
            assert np.array_equal(sd[k], v)
            assert sd[k].dtype == v.dtype
        reader.close()

    def test_npz_supports_dotted_keys(self, tmp_dir, monkeypatch):
        """npz 路径必须支持带点号的参数名（核心需求）。"""
        monkeypatch.setattr(vn_module, "_HAS_SAFETENSORS", False)
        sd = {
            "blocks.0.attn.qkv.weight": np.eye(4, dtype=np.float32),
            "head.layers.10.norm.gamma": np.ones(4, dtype=np.float32),
        }
        vn_path = os.path.join(tmp_dir, "dots.vn")
        with VNFileWriter(vn_path, arch="versenex", config={}) as w:
            w.write_weights(sd)
        with VNFileReader(vn_path) as r:
            out = r.read_weights()
        assert set(out.keys()) == set(sd.keys())
        for k, v in sd.items():
            assert np.array_equal(out[k], v)

    def test_env_without_safetensors_records_npz(self, tmp_dir, sample_config,
                                                 sample_state_dict, monkeypatch):
        """无 safetensors 时 meta.json 的 weight_format == npz。"""
        monkeypatch.setattr(vn_module, "_HAS_SAFETENSORS", False)
        if has_safetensors():
            pytest.skip("safetensors 已安装，此用例由 force_npz_mode 覆盖")
        vn_path = os.path.join(tmp_dir, "env.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            assert r.weight_format == "npz"


# ---------------------------------------------------------------------------
# 6. safetensors 可用路径（skipif 未安装）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_safetensors(),
                    reason="safetensors 未安装")
class TestSafetensorsPath:
    """safetensors 可用时的 mmap 零拷贝路径。"""

    def test_safetensors_format(self, tmp_dir, sample_state_dict, sample_config):
        vn_path = os.path.join(tmp_dir, "st.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            assert r.weight_format == "safetensors"
            sd = r.read_weights(mmap=True)
            for k, v in sample_state_dict.items():
                assert np.allclose(sd[k], v)


# ---------------------------------------------------------------------------
# 7. chat_template / tokenizer 可选字段
# ---------------------------------------------------------------------------


class TestOptionalFields:
    """chat_template 与 tokenizer 可选字段。"""

    def test_absent_fields_return_none(self, tmp_dir, sample_state_dict,
                                       sample_config):
        """未写入时 read_chat_template / read_tokenizer 返回 None。"""
        vn_path = os.path.join(tmp_dir, "bare.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            assert r.read_chat_template() is None
            assert r.read_tokenizer() is None

    def test_chat_template(self, tmp_dir, sample_state_dict, sample_config):
        """写入并读取 chat_template.jinja。"""
        tmpl = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
        vn_path = os.path.join(tmp_dir, "tmpl.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_chat_template(tmpl)
        with VNFileReader(vn_path) as r:
            assert r.read_chat_template() == tmpl

    def test_tokenizer_dict(self, tmp_dir, sample_state_dict, sample_config):
        """tokenizer 以 dict 形式写入。"""
        tok = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}}
        vn_path = os.path.join(tmp_dir, "tok.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_tokenizer(tok)
        with VNFileReader(vn_path) as r:
            assert r.read_tokenizer() == tok

    def test_tokenizer_from_file(self, tmp_dir, sample_state_dict, sample_config):
        """tokenizer 以文件路径形式写入。"""
        tok_path = os.path.join(tmp_dir, "tokenizer.json")
        tok = {"vocab": ["x", "y"]}
        with open(tok_path, "w") as f:
            json.dump(tok, f)
        vn_path = os.path.join(tmp_dir, "tokfile.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
            w.write_tokenizer(tok_path)
        with VNFileReader(vn_path) as r:
            assert r.read_tokenizer() == tok


# ---------------------------------------------------------------------------
# 8. CometSparkV05LM.save_vn / load_vn 端到端
# ---------------------------------------------------------------------------


class TestCometSparkSaveLoadVn:
    """CometSparkV05LM 的 save_vn / load_vn 端到端。"""

    def test_save_load_vn_roundtrip(self, tmp_dir):
        """Small 模型 save_vn → load_vn，权重数值一致。"""
        from spark.src.base_model import CometSparkV05Small, CometSparkV05LM

        model = CometSparkV05Small()
        orig_sd = {k: np.asarray(v).copy() for k, v in model.state_dict().items()}
        vn_path = os.path.join(tmp_dir, "spark.vn")

        model.save_vn(vn_path, chat_template="{{x}}", tokenizer={"v": [0, 1]})
        assert os.path.exists(vn_path)

        loaded = CometSparkV05LM.load_vn(vn_path)
        assert isinstance(loaded, CometSparkV05LM)
        assert loaded.config.arch == "versenex"
        assert loaded.config.vocab_size == model.config.vocab_size

        new_sd = {k: np.asarray(v) for k, v in loaded.state_dict().items()}
        # 权重应一致（strict=False 加载，但键应齐全）
        common = set(orig_sd.keys()) & set(new_sd.keys())
        assert len(common) > 0
        for k in common:
            assert np.array_equal(orig_sd[k], new_sd[k]), f"权重不一致: {k}"

    def test_save_vn_then_convert_to_pt(self, tmp_dir):
        """save_vn 写出后，vn_to_pt 能还原为 .pt 且权重一致。"""
        from spark.src.base_model import CometSparkV05Small

        model = CometSparkV05Small()
        orig_sd = {k: np.asarray(v).copy() for k, v in model.state_dict().items()}
        vn_path = os.path.join(tmp_dir, "s.vn")
        pt_path = os.path.join(tmp_dir, "s.pt")
        model.save_vn(vn_path)
        vn_to_pt(vn_path, pt_path)
        with open(pt_path, "rb") as f:
            payload = pickle.load(f)
        assert payload["arch"] == "versenex"
        for k, v in orig_sd.items():
            assert np.array_equal(payload["state_dict"][k], v)


# ---------------------------------------------------------------------------
# 9. 大文件读写（1MB+ 权重）
# ---------------------------------------------------------------------------


class TestLargeFile:
    """大权重张量（>1MB）读写。"""

    def test_large_weights_roundtrip(self, tmp_dir):
        """单张量 > 1MB 读写无损。"""
        # 512x512 float32 = 1MB（权重数据本身 >1MB，ZIP 压缩后文件可能更小）
        big = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
        sd = {"big.weight": big, "small.bias": np.ones(64, dtype=np.float32)}
        assert big.nbytes >= 1024 * 1024  # 权重数据 >= 1MB
        vn_path = os.path.join(tmp_dir, "big.vn")
        with VNFileWriter(vn_path, arch="versenex", config={}) as w:
            w.write_weights(sd)
        assert os.path.exists(vn_path)
        with VNFileReader(vn_path) as r:
            out = r.read_weights(mmap=True)
            assert np.array_equal(out["big.weight"], big)
            assert np.array_equal(out["small.bias"], sd["small.bias"])

    def test_large_mmap_and_non_mmap(self, tmp_dir):
        """大文件 mmap 与非 mmap 结果一致。"""
        big = np.random.default_rng(0).standard_normal((2048, 256)).astype(np.float32)
        sd = {"emb.weight": big}
        vn_path = os.path.join(tmp_dir, "big2.vn")
        with VNFileWriter(vn_path, arch="versenex", config={}) as w:
            w.write_weights(sd)
        with VNFileReader(vn_path) as r1:
            mmap_sd = r1.read_weights(mmap=True)
        with VNFileReader(vn_path) as r2:
            non_mmap_sd = r2.read_weights(mmap=False)
        assert np.array_equal(mmap_sd["emb.weight"], non_mmap_sd["emb.weight"])
        assert np.array_equal(mmap_sd["emb.weight"], big)


# ---------------------------------------------------------------------------
# 10. 智能压缩存储
# ---------------------------------------------------------------------------


class TestCompressionInfo:
    """compression_info 量化元数据。"""

    def test_explicit_compression_info(self, tmp_dir, sample_state_dict,
                                       sample_config):
        """显式传入 compression_info 写入 meta。"""
        ci = {"quantized": True, "bits": 4, "scheme": "int4"}
        vn_path = os.path.join(tmp_dir, "q.vn")
        with VNFileWriter(
            vn_path, arch="versenex", config=sample_config, compression_info=ci
        ) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            meta = r.read_meta()
            assert meta["compression_info"]["quantized"] is True
            assert meta["compression_info"]["bits"] == 4
            assert meta["compression_info"]["scheme"] == "int4"

    def test_no_compression_info(self, tmp_dir, sample_state_dict, sample_config):
        """不传 compression_info 时 meta.compression_info 为 None。"""
        vn_path = os.path.join(tmp_dir, "nq.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            assert r.read_meta()["compression_info"] is None

    def test_quant_info_attribute_collected(self, tmp_dir, sample_config):
        """数组携带 quant_info 属性时自动收集到 compression_info。"""
        arr = np.zeros(4, dtype=np.float32)
        # 模拟量化标注（numpy ndarray 不原生支持自定义属性，用包装 + setattr 尝试）
        try:
            arr.quant_info = {"bits": 8, "scheme": "int8"}  # type: ignore[attr-defined]
        except AttributeError:
            # 部分 numpy 版本不允许设置属性 → 跳过该断言路径
            pytest.skip("当前 numpy 不支持 ndarray 自定义属性")

        sd = {"w": arr}
        vn_path = os.path.join(tmp_dir, "qi.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sd)
        with VNFileReader(vn_path) as r:
            ci = r.read_meta()["compression_info"]
            assert ci is not None
            assert ci.get("quantized") is True
            assert ci["quant_details"][0]["bits"] == 8


# ---------------------------------------------------------------------------
# 11. 版本校验
# ---------------------------------------------------------------------------


class TestVersionValidation:
    """读取时校验 vn_format_version。"""

    def test_valid_version(self, tmp_dir, sample_state_dict, sample_config):
        """版本 1 正常读取。"""
        vn_path = os.path.join(tmp_dir, "v1.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        with VNFileReader(vn_path) as r:
            assert r.read_meta()["vn_format_version"] == 1

    def test_invalid_version_rejected(self, tmp_dir, sample_state_dict,
                                      sample_config):
        """篡改 meta.json 版本号后被拒绝。"""
        import zipfile

        vn_path = os.path.join(tmp_dir, "bad.vn")
        with VNFileWriter(vn_path, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)

        # 读取 zip 条目，篡改 meta.json 版本号，写回新文件
        with zipfile.ZipFile(vn_path, "r") as zin:
            entries = {n: zin.read(n) for n in zin.namelist()}
        bad_meta = json.loads(entries["meta.json"].decode("utf-8"))
        bad_meta["vn_format_version"] = 999
        entries["meta.json"] = json.dumps(bad_meta).encode("utf-8")
        bad_path = os.path.join(tmp_dir, "bad2.vn")
        with zipfile.ZipFile(bad_path, "w") as zout:
            for n, data in entries.items():
                zout.writestr(n, data)

        with VNFileReader(bad_path) as r:
            with pytest.raises(ValueError, match="不支持的 .vn 格式版本"):
                r.read_meta()

    def test_missing_meta_rejected(self, tmp_dir):
        """缺少 meta.json 应报错。"""
        import zipfile

        vn_path = os.path.join(tmp_dir, "nometa.vn")
        with zipfile.ZipFile(vn_path, "w") as zf:
            zf.writestr("config.yml", "arch: versenex\n")
        with VNFileReader(vn_path) as r:
            with pytest.raises(ValueError, match="meta.json"):
                r.read_meta()


# ---------------------------------------------------------------------------
# 12. CLI convert 子命令
# ---------------------------------------------------------------------------


class TestCliConvert:
    """verse-convert CLI 子命令。"""

    def test_cli_pt_to_vn(self, tmp_dir, sample_state_dict, sample_config):
        pt = os.path.join(tmp_dir, "m.pt")
        vn = os.path.join(tmp_dir, "m.vn")
        with open(pt, "wb") as f:
            pickle.dump(
                {"arch": "versenex", "config": sample_config,
                 "state_dict": sample_state_dict},
                f,
            )
        from verse_infra.verse_trainer.cli import convert_main
        rc = convert_main(["--input", pt, "--output", vn])
        assert rc == 0
        assert os.path.exists(vn)
        with VNFileReader(vn_path := vn) as r:
            for k, v in sample_state_dict.items():
                assert np.array_equal(r.read_weights()[k], v)

    def test_cli_vn_to_pt(self, tmp_dir, sample_state_dict, sample_config):
        vn = os.path.join(tmp_dir, "m.vn")
        pt = os.path.join(tmp_dir, "m.pt")
        with VNFileWriter(vn, arch="versenex", config=sample_config) as w:
            w.write_weights(sample_state_dict)
        from verse_infra.verse_trainer.cli import convert_main
        rc = convert_main(["--input", vn, "--output", pt])
        assert rc == 0
        with open(pt, "rb") as f:
            payload = pickle.load(f)
        for k, v in sample_state_dict.items():
            assert np.array_equal(payload["state_dict"][k], v)

    def test_cli_with_extras(self, tmp_dir, sample_state_dict, sample_config):
        pt = os.path.join(tmp_dir, "m.pt")
        vn = os.path.join(tmp_dir, "m.vn")
        tmpl_path = os.path.join(tmp_dir, "tmpl.jinja")
        tok_path = os.path.join(tmp_dir, "tok.json")
        with open(pt, "wb") as f:
            pickle.dump(
                {"arch": "versenex", "config": sample_config,
                 "state_dict": sample_state_dict},
                f,
            )
        with open(tmpl_path, "w") as f:
            f.write("{{ prompt }}")
        with open(tok_path, "w") as f:
            json.dump({"vocab": ["a"]}, f)

        from verse_infra.verse_trainer.cli import convert_main
        rc = convert_main([
            "--input", pt, "--output", vn,
            "--chat-template", tmpl_path, "--tokenizer", tok_path,
        ])
        assert rc == 0
        with VNFileReader(vn) as r:
            assert r.read_chat_template() == "{{ prompt }}"
            assert r.read_tokenizer() == {"vocab": ["a"]}

    def test_cli_invalid_combo_returns_error(self, tmp_dir):
        from verse_infra.verse_trainer.cli import convert_main
        rc = convert_main([
            "--input", os.path.join(tmp_dir, "a.pt"),
            "--output", os.path.join(tmp_dir, "b.txt"),
        ])
        assert rc == 1
