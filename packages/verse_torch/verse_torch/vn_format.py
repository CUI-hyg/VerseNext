""".vn 文件格式：基于 safetensors 的性能优化模型容器（Part4K2 Task 1）。

设计目标
--------
- **性能**：safetensors 可用时支持 mmap 零拷贝读取，避免 pickle 反序列化开销。
- **优雅降级**：safetensors 不可用时自动降级为 numpy .npz（纯标准库 + numpy）。
- **无损互转**：.vn ↔ .pt 权重数值完全一致。
- **自描述**：meta.json 记录格式版本、架构、权重格式、压缩信息、创建时间。
- **Part5K1.1 多空间缓存**：``VNCachedWeights`` 提供内存/硬盘混合缓存，
  按访问优先级分层（高优先级放内存常驻，低优先级放硬盘 mmap 懒加载），
  内存不足时自动降级到硬盘，配合 VMPC V2 + VSC 实现高性能压缩感知布局。

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

多空间缓存（Part5K1.1）
-----------------------
``VNCachedWeights`` 提供 3 种策略：

- ``"hybrid"``（默认，推荐）：高优先级张量（小 + 频繁访问）放内存常驻，
  低优先级张量（大 experts 权重）放硬盘 mmap 懒加载。
- ``"memory"``：全部放内存（小模型 / 内存充足场景）。
- ``"disk"``：全部放硬盘 mmap（大模型 / 内存紧张场景）。

优先级判定（``_tensor_priority``）：

- 高优先级（``"high"``）：名称匹配 ``tok_emb`` / ``head`` / ``norm`` /
  ``wq`` / ``wk`` / ``wv`` / ``wo`` / ``pos_emb`` / ``router`` 等频繁访问张量，
  或体积 < ``small_tensor_threshold``（默认 1MB）。
- 低优先级（``"low"``）：其余大张量（典型为 MoD experts / FFN 中间权重）。

内存预算（``memory_budget_mb``）：

- 默认 ``None`` → 自动检测：``min(total_virtual_memory * 0.25, 4096MB)``。
- 高优先级张量超出预算时，按 LRU 降级到硬盘（释放内存）。
- 低优先级张量不占用内存预算（始终走 mmap 懒加载）。

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
# Part5K1.1: VNCachedWeights —— .vn 多空间缓存
# ---------------------------------------------------------------------------


# 高优先级张量名匹配模式（频繁访问的小张量 + 路由 / norm / embedding）
_HIGH_PRIORITY_PATTERNS = (
    "tok_emb", "head", "norm", "router", "gate",
    "wq", "wk", "wv", "wo", "pos_emb", "ln",  # attention 核心
    "final_norm", "embed", "lm_head",
)

# 默认小张量阈值：小于 1MB 的张量自动归为高优先级
_DEFAULT_SMALL_TENSOR_BYTES = 1 * 1024 * 1024


def _detect_memory_budget_mb() -> int:
    """自动检测内存预算：``min(total_virtual_memory * 0.25, 4096MB)``。

    失败时回退到 1024MB（保守默认值）。
    """
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        budget = int(vm.total * 0.25 / (1024 * 1024))
        return max(64, min(budget, 4096))
    except Exception:
        # 不可用时退到保守 1GB
        return 1024


def _tensor_priority(name: str, nbytes: int,
                     small_threshold: int = _DEFAULT_SMALL_TENSOR_BYTES) -> str:
    """判定张量的缓存优先级。

    Args:
        name: 张量名（如 ``blocks.0.attn.wq.weight``）
        nbytes: 张量字节数
        small_threshold: 小张量阈值（默认 1MB）

    Returns:
        ``"high"`` 或 ``"low"``
    """
    # 名称匹配高优先级模式
    name_lower = name.lower()
    for pat in _HIGH_PRIORITY_PATTERNS:
        if pat in name_lower:
            return "high"
    # 小张量自动归为高优先级
    if nbytes < small_threshold:
        return "high"
    return "low"


class VNCachedWeights:
    """.vn 多空间缓存（Part5K1.1）。

    在 :class:`VNFileReader` 之上提供内存/硬盘混合缓存能力：

    - **hybrid**（默认）：高优先级张量常驻内存，低优先级走硬盘 mmap 懒加载。
    - **memory**：全部常驻内存（小模型 / 内存充足场景）。
    - **disk**：全部走硬盘 mmap（大模型 / 内存紧张场景）。

    特性：
    - **按需加载**：首次访问张量时才真正读取（lazy load）。
    - **LRU 降级**：内存预算超限时，按 LRU 顺序把高优先级张量降级到硬盘 mmap。
    - **预取接口**：``prefetch(names)`` 主动把指定张量加载到内存。
    - **释放接口**：``release(names)`` / ``release_low_priority()`` 主动释放内存。
    - **零拷贝**：safetensors 可用时优先走 mmap，避免反序列化开销。

    用法::

        reader = VNFileReader("model.vn")
        cached = VNCachedWeights(
            reader, strategy="hybrid", memory_budget_mb=2048
        )
        # 访问张量（首次自动加载，后续从缓存读）
        w = cached["blocks.0.attn.wq.weight"]
        # 主动预取下一层
        cached.prefetch(["blocks.1.attn.wq.weight", ...])
        # 释放内存（如训练完一层后）
        cached.release_low_priority()
        cached.close()  # 关闭底层 reader

    Args:
        reader: :class:`VNFileReader` 实例（所有权转移到本类，close 时一并关闭）
        strategy: 缓存策略（``"hybrid"`` / ``"memory"`` / ``"disk"``）
        memory_budget_mb: 内存预算（MB）；``None`` 自动检测
        small_tensor_threshold: 小张量阈值（字节），小于此值自动归为高优先级
    """

    VALID_STRATEGIES = ("hybrid", "memory", "disk")

    def __init__(
        self,
        reader: "VNFileReader",
        strategy: str = "hybrid",
        memory_budget_mb: Optional[int] = None,
        small_tensor_threshold: int = _DEFAULT_SMALL_TENSOR_BYTES,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy 仅支持 {self.VALID_STRATEGIES}，得到 {strategy!r}"
            )
        self._reader = reader
        self._strategy = strategy
        self._small_threshold = int(small_tensor_threshold)
        self._memory_budget_bytes = (
            int(memory_budget_mb * 1024 * 1024) if memory_budget_mb is not None
            else _detect_memory_budget_mb() * 1024 * 1024
        )

        # 缓存状态
        # _memory_cache: name -> ndarray（常驻内存）
        # _disk_mmap_cache: name -> mmap-like lazy loader（safetensors 或 npz）
        # _priority: name -> "high" / "low"
        # _access_order: LRU 顺序（list of name，最近访问的在末尾）
        # _memory_used_bytes: 当前内存缓存占用
        self._memory_cache: dict = {}
        self._priority: dict = {}
        self._access_order: list = []
        self._memory_used_bytes = 0

        # 硬盘 mmap 懒加载所需的 safetensors handle / npz 落盘临时文件
        # （由 _ensure_disk_loader 按需初始化）
        self._disk_loader: Optional[_VNDiskLoader] = None

        # 张量名 / 体积信息（首次访问时从 reader 读取并缓存）
        self._tensor_names: Optional[list] = None

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """当前缓存策略。"""
        return self._strategy

    @property
    def memory_budget_mb(self) -> float:
        """内存预算（MB）。"""
        return self._memory_budget_bytes / (1024 * 1024)

    @property
    def memory_used_mb(self) -> float:
        """当前内存缓存占用（MB）。"""
        return self._memory_used_bytes / (1024 * 1024)

    def tensor_names(self) -> list:
        """返回所有张量名（lazy 探测）。"""
        if self._tensor_names is None:
            self._tensor_names = self._probe_tensor_names()
        return list(self._tensor_names)

    def _probe_tensor_names(self) -> list:
        """探测 .vn 内的张量名列表。

        优先用 safetensors 的 keys()；不可用时读 npz 的 files。
        """
        wfmt = self._reader.weight_format
        if wfmt == "safetensors" and _HAS_SAFETENSORS and _WEIGHTS_ST_NAME in self._reader._names:
            self._ensure_disk_loader()
            if self._disk_loader is not None:
                return list(self._disk_loader.keys())
        # npz 路径：读取一次得到 keys（成本较高，但仅在首次访问时发生）
        weights = self._reader.read_weights(mmap=False)
        return list(weights.keys())

    def priority_of(self, name: str) -> str:
        """返回张量优先级（``"high"`` / ``"low"``）。"""
        if name not in self._priority:
            # 通过 reader 探测单张量的 nbytes（避免全量读取）
            nbytes = self._tensor_nbytes(name)
            self._priority[name] = _tensor_priority(
                name, nbytes, self._small_threshold
            )
        return self._priority[name]

    def _tensor_nbytes(self, name: str) -> int:
        """估算张量的字节数（不实际读取数据）。"""
        # safetensors 路径：从 disk loader 拿到 dtype/shape 后估算
        self._ensure_disk_loader()
        if self._disk_loader is not None:
            info = self._disk_loader.tensor_info(name)
            if info is not None:
                return info["nbytes"]
        # npz 路径：无法在不读取的情况下拿到 size，返回大值（保守判为 low）
        return 10 * 1024 * 1024  # 10MB placeholder

    # ------------------------------------------------------------------
    # 硬盘 lazy loader 初始化
    # ------------------------------------------------------------------

    def _ensure_disk_loader(self) -> None:
        """按需初始化硬盘 lazy loader（safetensors mmap / npz 落盘）。"""
        if self._disk_loader is not None:
            return
        wfmt = self._reader.weight_format
        if wfmt == "safetensors" and _HAS_SAFETENSORS and _WEIGHTS_ST_NAME in self._reader._names:
            # safetensors mmap 零拷贝
            tmp_path = self._reader._extract_to_temp(
                _WEIGHTS_ST_NAME, suffix=".safetensors"
            )
            self._disk_loader = _VNSafetensorsLoader(tmp_path)
            # 临时文件由 disk_loader 持有，close 时清理
            self._reader._weight_tmp_path = tmp_path
        elif _WEIGHTS_NPZ_NAME in self._reader._names:
            # npz 落盘 + NpzFile 懒加载
            tmp_path = self._reader._extract_to_temp(
                _WEIGHTS_NPZ_NAME, suffix=".npz"
            )
            self._disk_loader = _VNNpzLoader(tmp_path)
            self._reader._weight_tmp_path = tmp_path

    # ------------------------------------------------------------------
    # 核心访问 API
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> np.ndarray:
        """访问张量（首次自动加载，后续从缓存读）。"""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        try:
            return name in self.tensor_names()
        except Exception:
            return False

    def get(self, name: str) -> np.ndarray:
        """读取张量。高优先级 + memory 策略走内存；其余走硬盘 mmap。"""
        # 1. 内存命中
        if name in self._memory_cache:
            self._touch_lru(name)
            return self._memory_cache[name]

        # 2. 决定是否进入内存
        strategy = self._strategy
        priority = self.priority_of(name)

        if strategy == "memory":
            target_memory = True
        elif strategy == "disk":
            target_memory = False
        else:  # hybrid
            target_memory = (priority == "high")

        # 3. 读取张量
        arr = self._load_from_disk(name)

        if target_memory:
            # 检查内存预算
            self._ensure_memory_budget(arr.nbytes)
            self._memory_cache[name] = arr
            self._memory_used_bytes += arr.nbytes
            self._touch_lru(name)

        return arr

    def _load_from_disk(self, name: str) -> np.ndarray:
        """从硬盘 lazy loader 读取张量。"""
        self._ensure_disk_loader()
        if self._disk_loader is not None:
            return self._disk_loader.get_tensor(name)
        # 兜底：用 reader 的 read_weights（成本较高，但保证可用）
        weights = self._reader.read_weights(mmap=True)
        return weights[name]

    # ------------------------------------------------------------------
    # LRU 管理
    # ------------------------------------------------------------------

    def _touch_lru(self, name: str) -> None:
        """更新 LRU 顺序（移到末尾）。"""
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)

    def _ensure_memory_budget(self, incoming_bytes: int) -> None:
        """确保内存预算足够容纳 incoming_bytes；不够则按 LRU 降级。"""
        target = self._memory_used_bytes + incoming_bytes
        if target <= self._memory_budget_bytes:
            return
        # 按 LRU 顺序（从头部开始）降级，直到预算足够
        while self._access_order and target > self._memory_budget_bytes:
            victim = self._access_order.pop(0)
            if victim not in self._memory_cache:
                continue
            arr = self._memory_cache.pop(victim)
            self._memory_used_bytes -= arr.nbytes
            target -= arr.nbytes

    # ------------------------------------------------------------------
    # 预取 / 释放
    # ------------------------------------------------------------------

    def prefetch(self, names) -> None:
        """主动把指定张量预取到内存（用于训练前预热 / 流水线预取）。"""
        for name in names:
            if name not in self._memory_cache:
                # 强制加载到内存（忽略 hybrid 策略下的优先级）
                arr = self._load_from_disk(name)
                self._ensure_memory_budget(arr.nbytes)
                self._memory_cache[name] = arr
                self._memory_used_bytes += arr.nbytes
                self._touch_lru(name)

    def release(self, names) -> None:
        """释放指定张量的内存缓存（降级到硬盘 mmap）。"""
        for name in names:
            if name in self._memory_cache:
                arr = self._memory_cache.pop(name)
                self._memory_used_bytes -= arr.nbytes
                if name in self._access_order:
                    self._access_order.remove(name)

    def release_low_priority(self) -> int:
        """释放所有低优先级张量的内存缓存。

        Returns:
            释放的张量数量
        """
        released = 0
        to_release = [n for n in list(self._memory_cache.keys())
                      if self.priority_of(n) == "low"]
        for name in to_release:
            arr = self._memory_cache.pop(name)
            self._memory_used_bytes -= arr.nbytes
            if name in self._access_order:
                self._access_order.remove(name)
            released += 1
        return released

    def clear(self) -> None:
        """清空所有内存缓存。"""
        self._memory_cache.clear()
        self._access_order.clear()
        self._memory_used_bytes = 0

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------

    def close(self) -> None:
        """清空缓存并关闭底层 reader。"""
        self.clear()
        if self._disk_loader is not None:
            try:
                self._disk_loader.close()
            except Exception:
                pass
            self._disk_loader = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    def __enter__(self) -> "VNCachedWeights":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 硬盘 lazy loader 内部实现
# ---------------------------------------------------------------------------


class _VNDiskLoader:
    """硬盘 lazy loader 基类（safetensors / npz 共同接口）。"""

    def get_tensor(self, name: str) -> np.ndarray:
        raise NotImplementedError

    def tensor_info(self, name: str) -> Optional[dict]:
        raise NotImplementedError

    def keys(self) -> list:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _VNSafetensorsLoader(_VNDiskLoader):
    """safetensors mmap 零拷贝 loader。"""

    def __init__(self, path: str):
        self._path = path
        self._handle = safe_open(path, framework="numpy")
        self._keys = list(self._handle.keys())
        # 缓存 tensor_info（name -> {dtype, shape, nbytes}）
        self._info_cache: dict = {}

    def get_tensor(self, name: str) -> np.ndarray:
        return self._handle.get_tensor(name)

    def tensor_info(self, name: str) -> Optional[dict]:
        if name not in self._info_cache:
            if name not in self._keys:
                return None
            # safetensors 没有 metadata() 接口，需要 get_slice 或 get_tensor 推断
            # 这里用 get_tensor 一次后缓存（适合懒加载场景）
            try:
                arr = self._handle.get_tensor(name)
                self._info_cache[name] = {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "nbytes": int(arr.nbytes),
                }
            except Exception:
                return None
        return self._info_cache[name]

    def keys(self) -> list:
        return list(self._keys)

    def close(self) -> None:
        # safetensors 的 safe_open handle 没有显式 close，靠 GC 回收
        self._handle = None


class _VNNpzLoader(_VNDiskLoader):
    """npz 落盘 + NpzFile 懒加载。"""

    def __init__(self, path: str):
        self._path = path
        self._npz = np.load(path, allow_pickle=False)
        self._keys = list(self._npz.files)
        self._info_cache: dict = {}

    def get_tensor(self, name: str) -> np.ndarray:
        return self._npz[name]

    def tensor_info(self, name: str) -> Optional[dict]:
        if name not in self._info_cache:
            if name not in self._keys:
                return None
            try:
                arr = self._npz[name]
                self._info_cache[name] = {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "nbytes": int(arr.nbytes),
                }
            except Exception:
                return None
        return self._info_cache[name]

    def keys(self) -> list:
        return list(self._keys)

    def close(self) -> None:
        try:
            self._npz.close()
        except Exception:
            pass
        self._npz = None


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
    "VNCachedWeights",
    "pt_to_vn",
    "vn_to_pt",
    "convert_format",
    "has_safetensors",
]
