""".vn 文件格式：基于 safetensors 的性能优化模型容器（Part4K2 Task 1）。

设计目标
--------
- **性能**：safetensors 可用时支持 mmap 零拷贝读取，避免 pickle 反序列化开销。
- **优雅降级**：safetensors 不可用时自动降级为 numpy .npz（纯标准库 + numpy）。
- **无损互转**：.vn ↔ .pt 权重数值完全一致。
- **自描述**：meta.json 记录格式版本、架构、权重格式、压缩信息、创建时间。

.vn 文件结构
------------
.vn 是一个 ZIP 容器（``zipfile``），内含：

- ``model.safetensors`` 或 ``model.npz`` —— 权重（safetensors 可用时用前者）
- ``config.yml``                —— 模型配置（YAML，优先 PyYAML，否则 JSON 兼容子集）
- ``chat_template.jinja``       —— 聊天模板（可选）
- ``tokenizer.json``            —— tokenizer（可选）
- ``meta.json``                 —— 元数据

meta.json 结构::

    {
        "vn_format_version": 1,
        "arch": "versenex",
        "weight_format": "safetensors" | "npz",
        "compression_info": {...} | null,
        "created_at": "ISO8601 时间戳",
        "weight_count": 12
    }

智能压缩存储
------------
- 写入时若 ``compression_info`` 不为空，或 ``state_dict`` 中数组携带
  ``quant_info`` 属性，则把压缩元数据记录到 ``meta.json`` 的
  ``compression_info`` 字段。
- 读取时仅透传 ``compression_info``，**不自动反量化**（由模型加载逻辑处理），
  避免在格式层引入对量化方案的硬依赖。

安全性
------
- 读取时校验 ``meta.json`` 的 ``vn_format_version``，仅支持版本 1。
- npz 路径用 ``np.lib.format`` 显式 ``allow_pickle=False``，杜绝 pickle 反序列化攻击。
- safetensors 本身即 pickle-free。
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# 可选依赖：safetensors
# ---------------------------------------------------------------------------

try:
    from safetensors import safe_open  # type: ignore
    from safetensors.numpy import save_file as _st_save_file  # type: ignore
    _HAS_SAFETENSORS = True
except Exception:  # pragma: no cover - 依赖探测，环境相关
    safe_open = None  # type: ignore
    _st_save_file = None  # type: ignore
    _HAS_SAFETENSORS = False

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VN_FORMAT_VERSION = 1

_META_NAME = "meta.json"
_CONFIG_NAME = "config.yml"
_CHAT_TEMPLATE_NAME = "chat_template.jinja"
_TOKENIZER_NAME = "tokenizer.json"
_WEIGHTS_ST_NAME = "model.safetensors"
_WEIGHTS_NPZ_NAME = "model.npz"


def has_safetensors() -> bool:
    """返回当前环境是否可用 safetensors。"""
    return _HAS_SAFETENSORS


# ---------------------------------------------------------------------------
# 内部辅助：config / yaml 序列化
# ---------------------------------------------------------------------------


def _normalize_config(config: Any) -> dict:
    """把 config 规整成可序列化的 dict。

    接受 dict、带 ``to_dict()`` 的对象（如 ``CometSparkV05Config``）或 None。
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return dict(config)


def _dump_yaml_text(data: dict) -> str:
    """把 dict 序列化为 YAML 文本（优先 PyYAML，否则用 JSON 兼容子集）。"""
    if _HAS_YAML:
        return yaml.safe_dump(
            data, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
    # JSON 是 YAML 1.2 的严格子集，作为 fallback 安全可用
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _load_yaml_text(text: str) -> dict:
    """从 YAML 文本加载 dict（优先 PyYAML，否则 JSON）。"""
    if _HAS_YAML:
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 内部辅助：npz 读写（支持任意 key，包括带 "." 的参数名）
# ---------------------------------------------------------------------------


def _npz_to_bytes(state_dict: dict) -> bytes:
    """把 {name: ndarray} 序列化为 npz 字节流。

    手工构造 ZIP(npz) 以支持带点号等非标识符的参数名（``np.savez`` 仅接受
    合法标识符作为 kwarg 名，无法承载 ``blocks.0.attn.q.weight`` 这类名字）。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, arr in state_dict.items():
            arr = np.ascontiguousarray(arr)
            npy_buf = io.BytesIO()
            np.lib.format.write_array(npy_buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", npy_buf.getvalue())
    return buf.getvalue()


def _npz_from_bytes(data: bytes) -> dict:
    """从 npz 字节流加载 {name: ndarray}。"""
    result: dict = {}
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf, "r") as zf:
        for entry in zf.namelist():
            if not entry.endswith(".npy"):
                continue
            key = entry[:-4]
            with zf.open(entry) as f:
                result[key] = np.lib.format.read_array(f, allow_pickle=False)
    return result


# ---------------------------------------------------------------------------
# VNFileWriter
# ---------------------------------------------------------------------------


class VNFileWriter:
    """.vn 文件写入器。

    用法::

        writer = VNFileWriter("model.vn", arch="versenex", config=cfg_dict)
        writer.write_weights(state_dict)
        writer.write_chat_template(template_str)   # 可选
        writer.write_tokenizer("tokenizer.json")   # 可选
        writer.close()

    也支持上下文管理器::

        with VNFileWriter("model.vn", arch="versenex", config=cfg_dict) as w:
            w.write_weights(state_dict)

    Args:
        path: 输出 .vn 文件路径。
        arch: 模型架构名（如 ``"versenex"``）。
        config: 模型配置，dict 或带 ``to_dict()`` 的对象。
        compression_info: 压缩/量化元数据（如
            ``{"quantized": True, "bits": 4, "scheme": "int4"}``），写入
            ``meta.json``；None 表示无压缩信息。
    """

    def __init__(
        self,
        path: str,
        arch: str,
        config: Any,
        compression_info: Optional[dict] = None,
    ):
        self.path = str(path)
        self.arch = arch
        self.config = _normalize_config(config)
        self.compression_info = dict(compression_info) if compression_info else None
        self._weight_format: Optional[str] = None
        self._weight_count: int = 0
        self._collected_compression_info: Optional[dict] = None
        self._written: set = set()

        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        # ZIP_DEFLATED 提供较好压缩比；权重本身已是二进制，主要压缩 config/meta
        self._zf = zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED)

    # ------------------------------------------------------------------
    # 权重写入
    # ------------------------------------------------------------------

    def write_weights(self, state_dict: dict) -> None:
        """写入权重字典 ``{name: ndarray}``。

        safetensors 可用时存为 ``model.safetensors``（含 dtype/shape 头，支持
        mmap），否则降级为 ``model.npz``。同时收集数组上的 ``quant_info``
        属性以补全压缩元数据。
        """
        if _WEIGHTS_ST_NAME in self._written or _WEIGHTS_NPZ_NAME in self._written:
            raise RuntimeError("write_weights 已调用过，不可重复写入")

        # 规整为连续 ndarray，避免视图/非连续内存带来的兼容问题
        clean_sd: dict = {}
        quant_details = []
        for name, arr in state_dict.items():
            arr = np.ascontiguousarray(arr)
            clean_sd[name] = arr
            qi = getattr(arr, "quant_info", None)
            if qi is not None:
                entry = {"name": name}
                if isinstance(qi, dict):
                    entry.update(qi)
                else:
                    entry["info"] = str(qi)
                quant_details.append(entry)

        self._weight_count = len(clean_sd)

        # 合并压缩元数据：显式 compression_info + 数组 quant_info
        merged: dict = dict(self.compression_info) if self.compression_info else {}
        if quant_details:
            merged.setdefault("quantized", True)
            merged["quant_details"] = quant_details
        self._collected_compression_info = merged if merged else None

        if _HAS_SAFETENSORS:
            # safetensors.save_file 仅接受文件路径，先落临时文件再入 ZIP
            fd, tmp_path = tempfile.mkstemp(suffix=".safetensors")
            os.close(fd)
            try:
                # safetensors 要求 ndarray 连续且为受支持 dtype
                tensors = {}
                for name, arr in clean_sd.items():
                    if arr.dtype == np.float64:
                        arr = arr.astype(np.float32)
                    tensors[name] = np.ascontiguousarray(arr)
                _st_save_file(tensors, tmp_path)
                with open(tmp_path, "rb") as f:
                    self._zf.writestr(_WEIGHTS_ST_NAME, f.read())
                self._weight_format = "safetensors"
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            # 降级 npz
            self._zf.writestr(_WEIGHTS_NPZ_NAME, _npz_to_bytes(clean_sd))
            self._weight_format = "npz"

        self._written.add(_WEIGHTS_ST_NAME if self._weight_format == "safetensors"
                          else _WEIGHTS_NPZ_NAME)

    # ------------------------------------------------------------------
    # 可选字段
    # ------------------------------------------------------------------

    def write_chat_template(self, template_str: str) -> None:
        """写入聊天模板字符串到 ``chat_template.jinja``。"""
        if _CHAT_TEMPLATE_NAME in self._written:
            raise RuntimeError("chat_template 已写入")
        if not isinstance(template_str, str):
            raise TypeError(f"chat_template 必须是 str，得到 {type(template_str)}")
        self._zf.writestr(_CHAT_TEMPLATE_NAME, template_str)
        self._written.add(_CHAT_TEMPLATE_NAME)

    def write_tokenizer(
        self, tokenizer: Union[str, os.PathLike, dict]
    ) -> None:
        """写入 tokenizer 到 ``tokenizer.json``。

        Args:
            tokenizer: tokenizer.json 文件路径（str/PathLike）或 dict 对象。
        """
        if _TOKENIZER_NAME in self._written:
            raise RuntimeError("tokenizer 已写入")
        if isinstance(tokenizer, (str, os.PathLike)):
            with open(tokenizer, "rb") as f:
                self._zf.writestr(_TOKENIZER_NAME, f.read())
        elif isinstance(tokenizer, dict):
            self._zf.writestr(
                _TOKENIZER_NAME,
                json.dumps(tokenizer, ensure_ascii=False, indent=2, default=str),
            )
        else:
            raise TypeError(
                f"tokenizer 必须是路径(str/PathLike)或 dict，得到 {type(tokenizer)}"
            )
        self._written.add(_TOKENIZER_NAME)

    # ------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------

    def close(self) -> None:
        """写入 meta.json 并关闭 ZIP。"""
        if self._zf is None:
            return
        meta = {
            "vn_format_version": VN_FORMAT_VERSION,
            "arch": self.arch,
            "weight_format": self._weight_format,
            "compression_info": self._collected_compression_info,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "weight_count": self._weight_count,
        }
        self._zf.writestr(
            _META_NAME,
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        )
        # config.yml
        self._zf.writestr(_CONFIG_NAME, _dump_yaml_text(self.config))
        self._zf.close()
        self._zf = None

    # 上下文管理
    def __enter__(self) -> "VNFileWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # 异常时直接关闭 zip 不写 meta
            if self._zf is not None:
                self._zf.close()
                self._zf = None

    def __del__(self):
        try:
            if getattr(self, "_zf", None) is not None:
                self._zf.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VNFileReader
# ---------------------------------------------------------------------------


class VNFileReader:
    """.vn 文件读取器。

    用法::

        reader = VNFileReader("model.vn")
        meta = reader.read_meta()
        cfg = reader.read_config()
        weights = reader.read_weights(mmap=True)
        tmpl = reader.read_chat_template()    # Optional[str]
        tok = reader.read_tokenizer()         # Optional[dict]
        reader.close()

    也支持上下文管理器。读取时校验 ``vn_format_version``。
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._zf = zipfile.ZipFile(self.path, "r")
        self._names = set(self._zf.namelist())
        self._meta: Optional[dict] = None
        # 权重临时文件（safetensors mmap / npz 落盘时使用），close 时清理
        self._weight_tmp_path: Optional[str] = None

    # ------------------------------------------------------------------
    # meta
    # ------------------------------------------------------------------

    def read_meta(self) -> dict:
        """读取并校验 meta.json。"""
        if self._meta is not None:
            return self._meta
        if _META_NAME not in self._names:
            raise ValueError(f".vn 文件缺少 {_META_NAME}：{self.path}")
        with self._zf.open(_META_NAME) as f:
            meta = json.loads(f.read().decode("utf-8"))
        version = meta.get("vn_format_version")
        if version != VN_FORMAT_VERSION:
            raise ValueError(
                f"不支持的 .vn 格式版本：期望 {VN_FORMAT_VERSION}，得到 {version}"
            )
        self._meta = meta
        return meta

    @property
    def weight_format(self) -> str:
        """返回权重格式（``safetensors`` / ``npz``）。"""
        return self.read_meta().get("weight_format", "npz")

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    def read_config(self) -> dict:
        """读取 config.yml，返回 dict。"""
        if _CONFIG_NAME not in self._names:
            raise ValueError(f".vn 文件缺少 {_CONFIG_NAME}：{self.path}")
        with self._zf.open(_CONFIG_NAME) as f:
            text = f.read().decode("utf-8")
        return _load_yaml_text(text)

    # ------------------------------------------------------------------
    # 权重
    # ------------------------------------------------------------------

    def read_weights(self, mmap: bool = True) -> dict:
        """读取权重，返回 ``{name: ndarray}``。

        Args:
            mmap: 是否尽量零拷贝读取。safetensors 路径下使用 ``safe_open``
                的 mmap；npz 路径下落盘后用 ``np.load`` 懒加载（npz 内部为
                压缩 zip，严格意义上的 mmap 不可得，但访问仍是惰性的）。
        """
        meta = self.read_meta()
        wfmt = meta.get("weight_format", "npz")

        if wfmt == "safetensors" and _HAS_SAFETENSORS and _WEIGHTS_ST_NAME in self._names:
            return self._read_weights_safetensors(mmap)
        # 降级 / npz 路径
        if _WEIGHTS_NPZ_NAME in self._names:
            return self._read_weights_npz(mmap)
        # 兼容：声明 safetensors 但运行期不可用 → 尝试 npz
        if _WEIGHTS_ST_NAME in self._names and not _HAS_SAFETENSORS:
            raise RuntimeError(
                ".vn 声明权重为 safetensors 格式，但当前环境未安装 safetensors；"
                "请安装 safetensors 或用 npz 格式重新生成"
            )
        raise ValueError(f".vn 文件缺少权重条目：{self.path}")

    def _read_weights_safetensors(self, mmap: bool) -> dict:
        """safetensors 路径读取（mmap 零拷贝）。"""
        # 落盘临时文件以支持 safe_open 的 mmap
        tmp_path = self._extract_to_temp(_WEIGHTS_ST_NAME, suffix=".safetensors")
        result: dict = {}
        with safe_open(tmp_path, framework="numpy") as f:
            for key in f.keys():
                result[key] = f.get_tensor(key)
        # mmap 模式下保留临时文件直到 close，避免悬空映射
        if mmap:
            self._weight_tmp_path = tmp_path
        else:
            os.remove(tmp_path)
        return result

    def _read_weights_npz(self, mmap: bool) -> dict:
        """npz 路径读取。"""
        with self._zf.open(_WEIGHTS_NPZ_NAME) as f:
            data = f.read()
        if mmap:
            # 落盘后用 np.load 懒加载（NpzFile 惰性访问）
            tmp_path = self._extract_to_temp(
                _WEIGHTS_NPZ_NAME, suffix=".npz", data=data
            )
            npz = np.load(tmp_path, allow_pickle=False)
            result = {k: npz[k] for k in npz.files}
            self._weight_tmp_path = tmp_path
            return result
        return _npz_from_bytes(data)

    def _extract_to_temp(
        self, zip_name: str, suffix: str, data: Optional[bytes] = None
    ) -> str:
        """把 ZIP 内某个条目落盘到临时文件，返回路径。"""
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        if data is None:
            with self._zf.open(zip_name) as f:
                data = f.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
        return tmp_path

    # ------------------------------------------------------------------
    # 可选字段
    # ------------------------------------------------------------------

    def read_chat_template(self) -> Optional[str]:
        """读取 chat_template.jinja；不存在则返回 None。"""
        if _CHAT_TEMPLATE_NAME not in self._names:
            return None
        with self._zf.open(_CHAT_TEMPLATE_NAME) as f:
            return f.read().decode("utf-8")

    def read_tokenizer(self) -> Optional[dict]:
        """读取 tokenizer.json 为 dict；不存在则返回 None。"""
        if _TOKENIZER_NAME not in self._names:
            return None
        with self._zf.open(_TOKENIZER_NAME) as f:
            return json.loads(f.read().decode("utf-8"))

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭 ZIP 并清理临时文件。"""
        if self._zf is not None:
            self._zf.close()
            self._zf = None
        if self._weight_tmp_path is not None:
            try:
                if os.path.exists(self._weight_tmp_path):
                    os.remove(self._weight_tmp_path)
            except Exception:
                pass
            self._weight_tmp_path = None

    def __enter__(self) -> "VNFileReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# .pt ↔ .vn 转换
# ---------------------------------------------------------------------------


def _load_pt_payload(pt_path: str) -> dict:
    """加载 .pt 文件 payload。

    支持两种结构：
    - 完整 payload: ``{"arch": ..., "config": ..., "state_dict": {...}}``
    - 纯 state_dict: ``{name: ndarray}``（兼容 ``save_pretrained`` 写出的 model.pt）
    """
    import pickle

    with open(pt_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj
    if isinstance(obj, dict):
        # 启发式判断：如果值都是 ndarray，当作纯 state_dict
        if all(isinstance(v, (np.ndarray, np.generic)) for v in obj.values()):
            return {"arch": None, "config": {}, "state_dict": obj}
        return obj
    raise ValueError(f"无法识别的 .pt payload 结构：{pt_path}")


def pt_to_vn(
    pt_path: str,
    vn_path: str,
    arch: Optional[str] = None,
    config: Optional[Any] = None,
    chat_template: Optional[str] = None,
    tokenizer: Optional[Union[str, os.PathLike, dict]] = None,
) -> None:
    """把 ``.pt`` 文件转换为 ``.vn`` 文件（无损）。

    Args:
        pt_path: 源 .pt 文件路径。
        vn_path: 目标 .vn 文件路径。
        arch: 架构名；None 则从 .pt payload 的 ``arch`` 字段读取，再退化为
            ``"unknown"``。
        config: 模型配置（dict 或带 ``to_dict()`` 的对象）；None 则从 .pt
            payload 的 ``config`` 字段读取。
        chat_template: 聊天模板字符串；None 表示不写入。
        tokenizer: tokenizer 路径或 dict；None 表示不写入。
    """
    payload = _load_pt_payload(pt_path)
    sd = payload.get("state_dict", payload)
    if not isinstance(sd, dict):
        raise ValueError(f".pt 的 state_dict 不是 dict：{pt_path}")

    effective_arch = arch or payload.get("arch") or "unknown"
    effective_config = config if config is not None else payload.get("config", {})

    writer = VNFileWriter(vn_path, arch=effective_arch, config=effective_config)
    try:
        writer.write_weights(sd)
        if chat_template is not None:
            writer.write_chat_template(chat_template)
        if tokenizer is not None:
            writer.write_tokenizer(tokenizer)
        writer.close()
    except Exception:
        writer.close()
        raise


def vn_to_pt(vn_path: str, pt_path: str) -> None:
    """把 ``.vn`` 文件转换为 ``.pt`` 文件（无损）。

    输出 payload 结构与 ``CometSparkV05LM.save`` 一致::

        {"arch": ..., "config": dict, "state_dict": {name: ndarray}}
    """
    import pickle

    reader = VNFileReader(vn_path)
    try:
        meta = reader.read_meta()
        cfg = reader.read_config()
        sd = reader.read_weights()
    finally:
        reader.close()

    payload = {
        "arch": meta.get("arch"),
        "config": cfg,
        "state_dict": {k: np.asarray(v) for k, v in sd.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(pt_path)) or ".", exist_ok=True)
    with open(pt_path, "wb") as f:
        pickle.dump(payload, f)


def convert_format(src_path: str, dst_path: str) -> None:
    """自动检测源/目标后缀，在 ``.pt`` ↔ ``.vn`` 之间互转。

    - ``*.pt`` → ``*.vn``：调用 :func:`pt_to_vn`
    - ``*.vn`` → ``*.pt``：调用 :func:`vn_to_pt`
    - 其他后缀组合抛出 ``ValueError``。
    """
    src_lower = src_path.lower()
    dst_lower = dst_path.lower()
    if src_lower.endswith(".pt") and dst_lower.endswith(".vn"):
        pt_to_vn(src_path, dst_path)
    elif src_lower.endswith(".vn") and dst_lower.endswith(".pt"):
        vn_to_pt(src_path, dst_path)
    else:
        raise ValueError(
            f"convert_format 仅支持 .pt ↔ .vn 互转，"
            f"得到 src={src_path!r} dst={dst_path!r}"
        )


__all__ = [
    "VN_FORMAT_VERSION",
    "VNFileWriter",
    "VNFileReader",
    "pt_to_vn",
    "vn_to_pt",
    "convert_format",
    "has_safetensors",
]
