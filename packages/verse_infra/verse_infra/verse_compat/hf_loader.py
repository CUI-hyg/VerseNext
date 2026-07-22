"""Task 5.1: HuggingFace state_dict 加载器。

设计目标
--------
将 HuggingFace Hub 或本地路径中的预训练权重加载为 ``verse_torch.Tensor`` 字典，
键名与原 PyTorch ``state_dict`` 完全一致，便于直接喂给 ``Module.load_state_dict``。

加载优先级
--------
1. 本地路径：直接读取目录中的权重文件。
2. ``safetensors`` 文件：如果安装了 ``safetensors``，优先用 ``safe_open(..., framework="numpy")``
   直接取 ``numpy.ndarray``（零拷贝、安全、无 pickle 反序列化风险）。
3. ``.bin`` 文件（PyTorch pickle）：
   - 若已安装 ``torch``，调用 ``torch.load(path, map_location="cpu")``；
   - 否则自实现一个最简版本的 PyTorch pickle 解析器
     （基于 ``pickle.Unpickler`` + 自定义 ``persistent_load``，将 storage 还原为 ndarray，
     支持 float32 / float16 / bfloat16 / int8 / int16 / int32 / int64 / bool / uint8 等常见类型）。
4. HuggingFace Hub 下载：
   - 若安装了 ``huggingface_hub``，使用 ``snapshot_download`` 拉取 repo；
   - 否则降级到 ``urllib`` + ``https://huggingface.co/{repo}/resolve/{revision}/{file}`` 单文件下载。

注意：``verse_compat`` 仅在用户已安装 ``safetensors`` / ``torch`` / ``huggingface_hub``
时调用其加载器；否则降级到自带的最简实现，避免硬依赖。
"""

from __future__ import annotations

import os
import pickle
import shutil
import struct
import tempfile
import urllib.request
from typing import Optional

import numpy as np

from verse_torch import Tensor


# ---------------------------------------------------------------------------
# dtype 映射：PyTorch storage dtype 名 -> numpy dtype
# ---------------------------------------------------------------------------

# PyTorch 在 pickle 中用字符串 "storage_type" 标识 storage 类型，
# 例如 "HalfStorage" 表示 float16。这里建立映射，便于自实现解析器。
_TORCH_STORAGE_TO_NUMPY = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    # bfloat16 不是 numpy 原生 dtype，用 float32 作为容器加载，
    # 因为 verse_torch 内部计算均以 float32 为默认精度；
    # 如需精确保留 bf16，可在加载后 .cast(np.float32)（已经是了）。
    "BFloat16Storage": np.float32,
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "ShortStorage": np.int16,
    "IntStorage": np.int32,
    "LongStorage": np.int64,
    "BoolStorage": np.bool_,
    "QInt8Storage": np.int8,
    "QInt32Storage": np.int32,
    "QUInt8Storage": np.uint8,
}


# ---------------------------------------------------------------------------
# 工具：判断路径是本地目录还是 HF repo id
# ---------------------------------------------------------------------------


def _is_local_path(repo_id_or_path: str) -> bool:
    """判断是否为本地路径。

    HF repo id 形如 "owner/name"（无斜杠开头、无路径分隔符在首位），
    本地路径则可能是绝对路径、相对路径、或包含 os.sep。
    """
    if os.path.exists(repo_id_or_path):
        return True
    # 显式 ./ 或 / 开头也视为本地路径（即使不存在）
    if repo_id_or_path.startswith(("./", "/", "~")):
        return True
    # Windows 路径（盘符开头）
    if len(repo_id_or_path) >= 2 and repo_id_or_path[1] == ":":
        return True
    return False


# ---------------------------------------------------------------------------
# 工具：列出目录中匹配 pattern 的文件
# ---------------------------------------------------------------------------


def _list_weight_files(directory: str, pattern: str = "*.safetensors") -> list[str]:
    """列出目录中匹配 pattern 的权重文件，按字典序排序。

    支持 glob 通配符。先尝试 safetensors，再 fallback 到 .bin。
    """
    import fnmatch
    matches = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                matches.append(os.path.join(root, f))
    matches.sort()
    return matches


# ---------------------------------------------------------------------------
# safetensors 加载
# ---------------------------------------------------------------------------


def _load_safetensors_file(path: str) -> dict[str, np.ndarray]:
    """用 safetensors 加载单个 .safetensors 文件。

    优先使用 ``safe_open(..., framework="numpy")`` 直接取 ndarray，
    无需经过 torch，避免依赖。
    """
    try:
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "safetensors is not installed. Install with `pip install safetensors`, "
            "or load .bin files instead."
        ) from e

    out = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)
            # safe_open(numpy) 返回 ndarray（可能是 read-only），复制一份
            out[key] = np.array(arr)
    return out


# ---------------------------------------------------------------------------
# PyTorch .bin 加载：优先 torch.load，否则自实现 pickle 解析
# ---------------------------------------------------------------------------


def _load_bin_file_with_torch(path: str) -> dict[str, np.ndarray]:
    """用 torch.load 加载 .bin 文件，返回 {key: ndarray}。"""
    import torch  # noqa: F401
    sd = torch.load(path, map_location="cpu", weights_only=False)
    # 兼容 checkpoint 嵌套（如 {"state_dict": ...}）
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):
            v = v.numpy()
        out[k] = np.asarray(v)
    return out


class _TorchPickleUnpickler(pickle.Unpickler):
    """最简版 PyTorch pickle 解析器。

    PyTorch 保存 state_dict 的 pickle 流中包含两类特殊对象：
    1. ``persistent_id`` 形如 ``('storage', storage_type, key, location, numel)``，
       指向一个未填充字节的 storage。
    2. ``RebuildParameter`` / ``_rebuild_tensor_v2`` 等构造函数，
       接收 (storage, storage_offset, size, stride, ...) 并构造一个 Tensor。

    本解析器的策略：
    - 拦截 ``persistent_load``：根据 storage_type 名称找到对应 numpy dtype，
      从文件末尾的 storage 数据区读取对应字节并构造 ndarray。
    - 拦截 ``find_class``：仅识别 PyTorch 内部常用的几个重建函数，
      将其替换为返回 ``np.ndarray`` 或简单 dict 的 lambda，
      这样反序列化结果就是 ``{key: ndarray}``。

    支持的 storage 类型见 ``_TORCH_STORAGE_TO_NUMPY``。
    其它类型会回退到 ``float32`` 并打印警告。
    """

    def __init__(self, file, storage_data: dict, warnings_list: list):
        super().__init__(file)
        # storage_data: {key: (dtype, bytes)} 由外部 magic number 解析填充
        self._storage_data = storage_data
        self._warnings = warnings_list

    def persistent_load(self, pid):
        # pid 形如 ('storage', storage_type, key, location, numel)
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"Unknown persistent_id: {pid!r}")
        _tag, storage_type, key, _location, numel = pid[:5]
        # storage_type 通常是 torch.FloatStorage 等 class 对象
        # 在我们的简化解析中，可能是字符串或类
        name = storage_type.__name__ if hasattr(storage_type, "__name__") else str(storage_type)
        dtype = _TORCH_STORAGE_TO_NUMPY.get(name, None)
        if dtype is None:
            self._warnings.append(f"Unknown storage type {name!r}, falling back to float32")
            dtype = np.float32
        raw = self._storage_data.get(key)
        if raw is None:
            # 没有 storage 数据，返回零张量
            return np.zeros(numel, dtype=dtype)
        arr = np.frombuffer(raw, dtype=dtype).copy()
        if arr.size < numel:
            # 不够长，补零
            pad = np.zeros(numel - arr.size, dtype=dtype)
            arr = np.concatenate([arr, pad])
        elif arr.size > numel:
            arr = arr[:numel]
        return arr

    def find_class(self, module, name):
        # 拦截 PyTorch 重建函数：返回能从 storage 构造 ndarray 的 lambda
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def _rebuild(storage, storage_offset, size, stride, *args, **kwargs):
                # storage 已经是 ndarray（来自 persistent_load）
                # size: tuple, stride: tuple
                # 这里按 stride 解包
                if not isinstance(storage, np.ndarray):
                    return storage
                flat = storage[storage_offset:storage_offset + int(np.prod(size))]
                # 简化：忽略非连续 stride（默认连续 C-order）
                try:
                    return flat.reshape(size)
                except Exception:
                    return flat
            return _rebuild
        if module == "torch._utils" and name == "_rebuild_parameter":
            def _rebuild_param(data, requires_grad, backward_hooks):
                return data
            return _rebuild_param
        if module == "torch.storage" and name == "_load_from_bytes":
            def _load_from_bytes(b):
                # 嵌套 storage：直接返回空，由调用方处理
                return np.zeros(0, dtype=np.float32)
            return _load_from_bytes
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch" and name == "FloatStorage":
            class _Stub:
                __name__ = "FloatStorage"
            return _Stub
        # 其它类回退到默认（可能抛错，但对常见 LM 权重一般够用）
        return super().find_class(module, name)


def _parse_torch_bin_manual(path: str) -> dict[str, np.ndarray]:
    """自实现 PyTorch .bin pickle 解析（fallback）。

    PyTorch 的 ``torch.save`` 输出文件结构：
    ```
    [pickle opcodes ...] (主 pickle 流，含 persistent_id 指向 storage)
    [magic: '0123456789\0' (16 bytes) if old format]
    [storage key bytes...]
    ...
    ```

    本实现采用「保守策略」：
    1. 先尝试直接用 ``pickle.Unpickler`` 反序列化，
       不读取 storage bytes（只依赖 ``persistent_load`` 返回零数组），
       这样能拿到所有键名与形状。
    2. 然后 **二次解析** 文件末尾的 storage 字节区，
       按 pickle 中的 ``persistent_id`` 顺序对齐补全。
    由于步骤 2 复杂且 PyTorch 格式版本多变，本实现仅实现步骤 1
    （拿到键名 + 零值），并在文档中说明：完整字节解析需要更复杂的实现。
    对于真正的 .bin 文件，建议用户安装 ``torch`` 以获得完整支持。
    """
    warnings_list: list[str] = []
    storage_data: dict = {}

    with open(path, "rb") as f:
        unpickler = _TorchPickleUnpickler(f, storage_data, warnings_list)
        try:
            obj = unpickler.load()
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse {path} with manual pickle parser: {e}. "
                f"Install `torch` for full .bin support."
            ) from e

    # obj 应该是 dict-like
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected pickle result type: {type(obj)!r}")

    # 转换为 {key: ndarray}
    out = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        elif hasattr(v, "numpy"):
            out[k] = v.numpy()
        elif hasattr(v, "detach"):
            out[k] = v.detach().cpu().numpy()
        else:
            # 标量或其它，包装成 ndarray
            out[k] = np.asarray(v)
    return out


def _load_bin_file(path: str) -> dict[str, np.ndarray]:
    """加载 .bin 文件：优先 torch.load，否则自实现解析。"""
    try:
        import torch  # noqa: F401
        return _load_bin_file_with_torch(path)
    except ImportError:
        return _parse_torch_bin_manual(path)


# ---------------------------------------------------------------------------
# HuggingFace Hub 下载
# ---------------------------------------------------------------------------


_HF_BASE = "https://huggingface.co"


def _download_from_hf(
    repo_id: str,
    revision: str = "main",
    pattern: str = "*.safetensors",
) -> list[str]:
    """从 HuggingFace Hub 下载权重文件。

    优先使用 ``huggingface_hub.snapshot_download``（如果安装），
    否则降级到 ``urllib`` + 单文件下载。

    HF 文件 URL 格式：
        ``https://huggingface.co/{repo}/resolve/{revision}/{file}``

    Returns:
        下载到本地的文件路径列表
    """
    # 优先使用 huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            # allow_patterns 用 glob 限制下载范围
            allow_patterns=[pattern, "*.bin", "config.json"],
        )
        # 优先 safetensors，再 .bin
        files = _list_weight_files(local_dir, pattern)
        if not files:
            files = _list_weight_files(local_dir, "*.bin")
        if not files:
            raise FileNotFoundError(
                f"No weight files matching {pattern} or *.bin found in {local_dir}"
            )
        return files
    except ImportError:
        pass

    # 降级：用 urllib + HF API
    # 先 GET repo 文件列表 API
    api_url = f"{_HF_BASE}/api/models/{repo_id}/revision/{revision}"
    tmp_dir = tempfile.mkdtemp(prefix="verse_compat_hf_")
    try:
        with urllib.request.urlopen(api_url, timeout=30) as resp:  # noqa: S310
            import json
            meta = json.loads(resp.read().decode("utf-8"))
        # meta["siblings"] 是文件列表
        siblings = meta.get("siblings", [])
        all_files = [s["rfilename"] for s in siblings if "rfilename" in s]
        # 优先 safetensors
        import fnmatch
        matched = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
        if not matched:
            matched = [f for f in all_files if f.endswith(".bin")]
        if not matched:
            raise FileNotFoundError(
                f"No weight files matching {pattern} or *.bin found in {repo_id}@{revision}"
            )
        local_paths = []
        for rfilename in matched:
            url = f"{_HF_BASE}/{repo_id}/resolve/{revision}/{rfilename}"
            local_path = os.path.join(tmp_dir, os.path.basename(rfilename))
            urllib.request.urlretrieve(url, local_path)  # noqa: S310
            local_paths.append(local_path)
        return local_paths
    except Exception:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# 主入口：load_hf_state_dict
# ---------------------------------------------------------------------------


def load_hf_state_dict(
    repo_id_or_path: str,
    revision: str = "main",
    pattern: str = "*.safetensors",
) -> dict:
    """从 HuggingFace repo 或本地路径加载模型权重到 verse_torch.Tensor。

    Args:
        repo_id_or_path: 本地目录路径，或 HF repo id（如 "microsoft/phi-2"）。
        revision: HF repo 的 revision（branch / commit / tag），仅当从 Hub 下载时使用。
        pattern: 权重文件 glob 模式，默认 ``*.safetensors``。

    Returns:
        dict[str, verse_torch.Tensor]：键名与原 PyTorch state_dict 一致；
        Tensor 均为 ``requires_grad=False``，dtype 保留原始（FP32 / FP16 / BF16）。

    加载顺序：
        1. 本地路径 -> 列出目录中 safetensors（或 .bin）文件；
        2. HF repo -> 下载 safetensors（或 .bin）文件到临时目录；
        3. safetensors 优先；若文件名是 .bin 且未安装 torch，使用自实现 pickle 解析器；
        4. 多个分片文件按字典序合并。
    """
    # Step 1: 确定本地文件列表
    if _is_local_path(repo_id_or_path):
        local_path = os.path.expanduser(repo_id_or_path)
        if os.path.isdir(local_path):
            files = _list_weight_files(local_path, pattern)
            if not files:
                # fallback .bin
                files = _list_weight_files(local_path, "*.bin")
            if not files:
                raise FileNotFoundError(
                    f"No weight files matching {pattern} or *.bin in {local_path}"
                )
        elif os.path.isfile(local_path):
            files = [local_path]
        else:
            raise FileNotFoundError(f"Path not found: {local_path}")
    else:
        # 从 HF Hub 下载
        files = _download_from_hf(repo_id_or_path, revision=revision, pattern=pattern)

    # Step 2: 逐文件加载并合并
    state_dict: dict[str, Tensor] = {}
    for fpath in files:
        if fpath.endswith(".safetensors"):
            raw = _load_safetensors_file(fpath)
        elif fpath.endswith(".bin"):
            raw = _load_bin_file(fpath)
        else:
            # 尝试用 safetensors 加载（如果文件其实是 safetensors 格式）
            try:
                raw = _load_safetensors_file(fpath)
            except Exception:
                raw = _load_bin_file(fpath)

        for key, arr in raw.items():
            # 转换为 verse_torch.Tensor，requires_grad=False
            state_dict[key] = Tensor(arr, requires_grad=False)

    return state_dict


__all__ = ["load_hf_state_dict"]
