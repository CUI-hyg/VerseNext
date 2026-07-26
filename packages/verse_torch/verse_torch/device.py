"""VerseTorch: 设备抽象层与后端工厂。

设计要点
========
- ``DeviceBackend`` 抽象基类定义 ``matmul`` / ``linear`` / ``softmax`` /
  ``layernorm`` / ``rmsnorm`` / ``rope`` / ``attention`` 等算子接口，
  以及只读 ``device_type`` 属性。
- ``NumpyBackend``：默认 CPU 后端，所有算子用 NumPy 实现（封装现有操作，
  保持与 Tensor 自研 autograd 路径一致的行为）。
- ``TorchBackend``：PyTorch 委托后端，支持 ``cuda`` / ``mps`` / ``npu``
  / ``rocm``（``npu`` 通过 ``torch_npu`` 扩展支持，CANN 为其底层计算库；
  ``rocm`` 走 PyTorch ROCm build，HIP-on-ROCm 复用 cuda API）。
  算子委托给 ``torch.Tensor``，CUDA/HIP kernel 走 PyTorch 原生实现，
  **不自研 kernel**（含 ROCm / CANN）。
- ``get_backend(device)`` 工厂函数：根据 device 字符串返回对应 backend 实例。
- ``has_torch()`` / ``has_torch_npu()``：检测 PyTorch / torch_npu 是否可用。
- ``has_rocm()`` / ``get_rocm_version()`` / ``has_cann()`` / ``get_cann_version()``
  （Part5K1.3 Task 9）：探测 AMD ROCm / 华为 NPU CANN 环境，便于运维诊断。
- 模块级常量 ``DEFAULT_DEVICE = "cpu"``。

向后兼容
========
无 PyTorch 环境下，本模块仍可正常 import；只有当请求 GPU/NPU/ROCm 后端时
才会抛 ``RuntimeError``。
"""

from __future__ import annotations

import abc
import os
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# PyTorch 可用性检测
# ---------------------------------------------------------------------------


def _try_import_torch():
    """尝试导入 torch 与 torch_npu，返回 (torch, torch_npu)。

    任意一步失败均返回 (None, None)，保证本模块在无 torch 环境下可独立 import。
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None, None
    torch_npu = None
    try:
        import torch_npu  # type: ignore  noqa: F401
        torch_npu = torch_npu
    except Exception:
        pass
    return torch, torch_npu


# 模块级缓存：torch 与 torch_npu 的导入结果（None 表示不可用）
_TORCH, _TORCH_NPU = _try_import_torch()


def has_torch() -> bool:
    """检测 PyTorch 是否可用。"""
    return _TORCH is not None


def has_torch_npu() -> bool:
    """检测 torch_npu（华为 NPU 支持）是否可用。"""
    return _TORCH_NPU is not None


def get_torch_module():
    """返回已缓存的 torch 模块（不可用时返回 None）。"""
    return _TORCH


def get_torch_npu_module():
    """返回已缓存的 torch_npu 模块（不可用时返回 None）。"""
    return _TORCH_NPU


# ---------------------------------------------------------------------------
# ROCm / CANN 探测（Part5K1.3 Task 9）
# ---------------------------------------------------------------------------


def has_rocm() -> bool:
    """检测当前 PyTorch 是否为 ROCm build（HIP-on-ROCm）。

    ROCm build 的 PyTorch ``torch.version.hip`` 非 None；同时
    ``torch.cuda.is_available()`` 也返回 True（HIP 暴露为 cuda API）。

    无 PyTorch 环境下返回 ``False``，不抛异常（lazy 检测）。
    """
    torch_mod = get_torch_module()
    if torch_mod is None:
        return False
    try:
        return getattr(torch_mod.version, "hip", None) is not None
    except AttributeError:
        return False


def get_rocm_version() -> Optional[str]:
    """返回 ROCm 版本字符串（如 ``"6.1.0"``），非 ROCm 环境返回 ``None``。

    版本来自 ``torch.version.hip``（PyTorch ROCm build 的 HIP 运行时版本）。
    无 PyTorch / 非 ROCm build 时返回 ``None``，不抛异常。
    """
    torch_mod = get_torch_module()
    if torch_mod is None:
        return None
    try:
        hip = getattr(torch_mod.version, "hip", None)
        return str(hip) if hip is not None else None
    except AttributeError:
        return None


def _try_get_cann_version() -> Optional[str]:
    """尝试读取 CANN 版本（torch_npu.cann_version / torch_npu.version）。

    torch_npu 的版本 API 名因版本差异（``torch_npu.version`` /
    ``torch_npu.cann_version``），此处依次尝试。无 torch_npu 时返回 ``None``。
    """
    torch_npu = get_torch_npu_module()
    if torch_npu is None:
        return None
    try:
        for attr in ("cann_version", "version"):
            v = getattr(torch_npu, attr, None)
            if v is not None:
                return str(v)
        return None
    except Exception:
        return None


def has_cann() -> bool:
    """检测 torch_npu 是否可用 + CANN 版本可读。

    torch_npu 是华为 NPU 的 PyTorch 扩展，CANN（Compute Architecture for
    Neural Networks）是其底层计算库。本函数检测 torch_npu 已被 import
    （由模块级 ``_TORCH_NPU`` 缓存）且能读取到版本信息。

    无 torch_npu 环境下返回 ``False``，不抛异常（lazy 检测）。
    """
    if not has_torch_npu():
        return False
    return _try_get_cann_version() is not None


def get_cann_version() -> Optional[str]:
    """返回 CANN 版本字符串（如 ``"8.0.0"``），无 NPU 环境返回 ``None``。

    版本来自 ``torch_npu.cann_version`` / ``torch_npu.version``（因 torch_npu
    版本差异做兜底尝试）。无 torch_npu 时返回 ``None``，不抛异常。
    """
    return _try_get_cann_version()


# ---------------------------------------------------------------------------
# 默认设备与设备字符串解析
# ---------------------------------------------------------------------------

#: 默认设备字符串（CPU-first 引擎）
DEFAULT_DEVICE = "cpu"


def _parse_device(device) -> str:
    """规范化 device 字符串。

    接受 ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``"mps"`` / ``"npu"`` /
    ``"npu:0"`` / ``"rocm"`` / ``"rocm:0"`` 等形式，返回小写的 device type
    （``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"npu"`` / ``"rocm"``）。

    注意：``"rocm"`` 对外保留为独立 type（用于诊断，见 :func:`has_rocm` /
    :func:`get_rocm_version`）；在 :func:`get_backend` / :func:`empty_cache` /
    :func:`get_memory_info` 等内部路径中，``"rocm"`` 会被映射到 ``"cuda"``
    （HIP-on-ROCm 走 PyTorch cuda 路径，不自研 kernel）。
    """
    if device is None:
        return "cpu"
    s = str(device).lower()
    if s.startswith(("cuda", "npu", "mps", "rocm")):
        return s.split(":")[0]
    return "cpu"


def _to_torch_device_str(device) -> str:
    """把外部 device 字符串映射到 PyTorch 认识的 device 字符串。

    ``"rocm"`` / ``"rocm:N"`` → ``"cuda"`` / ``"cuda:N"``（HIP-on-ROCm 走
    PyTorch cuda 路径，PyTorch 不识别 "rocm" 字符串但 ROCm build 通过 HIP
    暴露 cuda API）；其他原样返回小写字符串。

    用于 :func:`get_memory_info` 等需要把 device 索引传给 ``torch.cuda`` 的
    场景，以及 ``backend_torch._torch_device`` 构造 ``torch.device`` 时。
    """
    s = str(device).lower()
    if s == "rocm":
        return "cuda"
    if s.startswith("rocm:"):
        return "cuda:" + s.split(":", 1)[1]
    return s


def is_cpu_device(device) -> bool:
    """判断 device 是否为 CPU（含 None / "cpu" / "cpu:0"）。"""
    return _parse_device(device) == "cpu"


# ---------------------------------------------------------------------------
# DeviceBackend 抽象基类
# ---------------------------------------------------------------------------


class DeviceBackend(abc.ABC):
    """设备后端抽象基类。

    定义一组算子接口，子类（``NumpyBackend`` / ``TorchBackend``）实现具体逻辑。
    所有算子接受与返回 ``np.ndarray`` 或 ``torch.Tensor``（取决于后端）。
    实现方应保证算子语义与 PyTorch 等价，以便上层 Tensor 在不同后端间切换。
    """

    @property
    @abc.abstractmethod
    def device_type(self) -> str:
        """返回后端设备类型字符串（``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"npu"`` / ``"rocm"``）。"""

    @abc.abstractmethod
    def matmul(self, a, b):
        """矩阵乘法 ``a @ b``。"""

    @abc.abstractmethod
    def linear(self, x, weight, bias=None):
        """全连接层：``y = x @ weight.T + bias``。"""

    @abc.abstractmethod
    def softmax(self, x, dim: int = -1):
        """沿 ``dim`` 做 softmax（数值稳定）。"""

    @abc.abstractmethod
    def layernorm(self, x, weight, bias, eps: float = 1e-5):
        """LayerNorm：沿最后一维归一化后仿射变换。"""

    @abc.abstractmethod
    def rmsnorm(self, x, weight, eps: float = 1e-6):
        """RMSNorm：用 RMS = ``sqrt(mean(x^2))`` 归一化后缩放。"""

    @abc.abstractmethod
    def rope(self, x, cos=None, sin=None):
        """Rotary Position Embedding 应用（GPT-NeoX 风格 rotate_half）。

        ``cos`` / ``sin`` 形状应与 ``x`` 后两维匹配或可广播。
        若 ``cos`` / ``sin`` 为 ``None``，原样返回 ``x``（占位）。
        """

    def attention(self, q, k, v, mask=None):
        """Scaled dot-product attention: ``softmax(q@k.T/sqrt(d)) @ v``。

        默认实现（NumPy 风格），子类可覆盖以调用底层 fused kernel。
        """
        d = q.shape[-1]
        scores = self.matmul(q, _swap_last_two(k)) / (d ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = self.softmax(scores, dim=-1)
        return self.matmul(attn, v)


def _swap_last_two(x):
    """交换 x 最后两维（NumPy 与 torch 通用）。"""
    if _TORCH is not None and isinstance(x, _TORCH.Tensor):
        return x.transpose(-1, -2)
    return np.swapaxes(x, -1, -2)


# ---------------------------------------------------------------------------
# NumpyBackend
# ---------------------------------------------------------------------------


class NumpyBackend(DeviceBackend):
    """默认 CPU 后端，所有算子用 NumPy 实现。

    与 ``Tensor`` 自研 autograd 路径完全等价，仅是把这些操作封装成
    DeviceBackend 接口形式，便于上层代码以统一接口调用。
    """

    @property
    def device_type(self) -> str:
        return "cpu"

    def matmul(self, a, b):
        return np.matmul(a, b)

    def linear(self, x, weight, bias=None):
        out = np.matmul(x, np.swapaxes(weight, -1, -2))
        if bias is not None:
            out = out + bias
        return out

    def softmax(self, x, dim: int = -1):
        x_max = np.max(x, axis=dim, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=dim, keepdims=True)

    def layernorm(self, x, weight, bias, eps: float = 1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        return normed * weight + bias

    def rmsnorm(self, x, weight, eps: float = 1e-6):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        rms = np.sqrt(ms + eps)
        return x / rms * weight

    def rope(self, x, cos=None, sin=None):
        # GPT-NeoX 风格 rotate_half：把最后一维拆成两半旋转
        if cos is None or sin is None:
            return x
        d = x.shape[-1]
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        # cos/sin 形状 (..., T, D) 或 (T, D/2)，统一广播到 x 的后两维
        cos_b = cos[..., :half] if cos.shape[-1] >= d else cos
        sin_b = sin[..., :half] if sin.shape[-1] >= d else sin
        rotated = np.concatenate(
            [x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], axis=-1
        )
        return rotated


# ---------------------------------------------------------------------------
# 工厂函数与缓存
# ---------------------------------------------------------------------------

# backend 实例缓存：device 字符串 -> DeviceBackend 实例
_BACKEND_CACHE: dict = {}


def get_backend(device=None) -> DeviceBackend:
    """根据 device 字符串返回对应 backend 实例。

    Args:
        device: ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``"mps"`` /
            ``"npu"`` / ``"npu:0"`` / ``"rocm"`` / ``"rocm:0"`` 等，
            ``None`` 等价于 ``"cpu"``。

    Returns:
        ``DeviceBackend`` 实例（``NumpyBackend`` 或 ``TorchBackend``）。
        ``"rocm"`` / ``"rocm:N"`` 返回 ``TorchBackend``，内部由
        ``_torch_device`` 映射到 ``torch.device("cuda:N")``（HIP-on-ROCm
        走 PyTorch cuda 路径），对外 ``device_type`` 保留 ``"rocm"`` 诊断用。

    Raises:
        RuntimeError: 请求 GPU/NPU/ROCm 但 PyTorch 不可用，或请求 NPU 但
            ``torch_npu`` 不可用。
    """
    dtype = _parse_device(device)
    if dtype == "cpu":
        if "cpu" not in _BACKEND_CACHE:
            _BACKEND_CACHE["cpu"] = NumpyBackend()
        return _BACKEND_CACHE["cpu"]

    # 非 CPU：必须依赖 torch（rocm / cuda / mps / npu 均走 TorchBackend）
    if not has_torch():
        raise RuntimeError(
            f"未安装 PyTorch，无法使用 device '{device}'（仅支持 CPU）"
        )
    if dtype == "npu" and not has_torch_npu():
        raise RuntimeError(
            f"未安装 torch_npu，无法使用 NPU 设备 '{device}'"
        )
    # 延迟导入 TorchBackend，避免 device.py 硬依赖 torch
    from .backend_torch import TorchBackend
    key = str(device).lower()
    if key not in _BACKEND_CACHE:
        # "rocm:N" 透传给 TorchBackend，由 _torch_device 内部映射到 cuda:N
        # （保留原 "rocm" 字符串用于诊断，见 TorchBackend.device_type）
        _BACKEND_CACHE[key] = TorchBackend(device=key)
    return _BACKEND_CACHE[key]


def clear_backend_cache() -> None:
    """清空 backend 缓存（主要用于测试隔离）。"""
    _BACKEND_CACHE.clear()


# ---------------------------------------------------------------------------
# GPU / NPU 显存管理（CPU 时优雅降级为 no-op）
# ---------------------------------------------------------------------------


def empty_cache(device="cpu") -> None:
    """清理 GPU/NPU/ROCm 显存缓存（CPU 时 no-op）。

    Args:
        device: 设备字符串；``"cpu"`` / ``None`` 时为 no-op，
            ``"cuda"`` / ``"cuda:0"`` 调用 ``torch.cuda.empty_cache()``，
            ``"rocm"`` / ``"rocm:0"`` 同样走 ``torch.cuda.empty_cache()``
            （PyTorch ROCm build 通过 HIP 暴露 cuda API），
            ``"npu"`` / ``"npu:0"`` 调用 ``torch_npu.npu.empty_cache()``。
    """
    dtype = _parse_device(device)
    if dtype == "cpu" or _TORCH is None:
        return
    try:
        if dtype in ("cuda", "rocm"):
            # ROCm build 的 PyTorch 通过 HIP 暴露 cuda API，
            # 故 rocm 走 torch.cuda.empty_cache()
            _TORCH.cuda.empty_cache()
        elif dtype == "mps":
            # MPS 后端没有 empty_cache API，跳过（no-op）
            return
        elif dtype == "npu":
            torch_npu = get_torch_npu_module()
            if torch_npu is not None and hasattr(torch_npu, "npu"):
                torch_npu.npu.empty_cache()
    except Exception:
        # 显存清理失败不应影响训练
        pass


def get_memory_info(device="cpu") -> dict:
    """获取设备内存信息。

    Args:
        device: 设备字符串

    Returns:
        dict: ``{"total": bytes, "used": bytes, "free": bytes}``。
        - CUDA / ROCm: 调用 ``torch.cuda.mem_get_info``（PyTorch >= 1.11）
          或 ``torch.cuda.memory_allocated`` / ``torch.cuda.get_device_properties``
          组合估算。ROCm build 的 PyTorch 通过 HIP 暴露 cuda API，故 rocm
          走同一路径。
        - NPU: 依次尝试 ``torch_npu.npu.mem_get_info`` /
          ``torch.npu.mem_get_info`` / ``torch_npu.npu.memory_allocated`` /
          ``torch.npu.memory_allocated``（CANN 版本 API 名差异兜底）。
        - CPU: 优先用 psutil（可选依赖），不可用时返回 0 占位 dict。
    """
    dtype = _parse_device(device)
    if dtype == "cpu":
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            return {
                "total": int(vm.total),
                "used": int(vm.used),
                "free": int(vm.available),
            }
        except Exception:
            return {"total": 0, "used": 0, "free": 0}
    if _TORCH is None:
        return {"total": 0, "used": 0, "free": 0}
    try:
        if dtype in ("cuda", "rocm"):
            # ROCm build 的 PyTorch 通过 HIP 暴露 cuda API，故 rocm 走
            # torch.cuda.mem_get_info() 等同路径
            # 优先用 mem_get_info（PyTorch >= 1.11）
            if hasattr(_TORCH.cuda, "mem_get_info"):
                free, total = _TORCH.cuda.mem_get_info()
                return {
                    "total": int(total),
                    "used": int(total - free),
                    "free": int(free),
                }
            # 回退：用 memory_allocated + device_properties
            # rocm:N → cuda:N 用于设备索引解析
            torch_device_str = _to_torch_device_str(device)
            idx = 0
            if ":" in str(torch_device_str):
                try:
                    idx = int(str(torch_device_str).split(":")[1])
                except ValueError:
                    pass
            props = _TORCH.cuda.get_device_properties(idx)
            total = int(props.total_memory)
            used = int(_TORCH.cuda.memory_allocated(idx))
            return {"total": total, "used": used, "free": max(0, total - used)}
        if dtype == "npu":
            # CANN 版本 API 名差异兜底：依次尝试
            #   1. torch_npu.npu.mem_get_info  / torch.npu.mem_get_info
            #   2. torch_npu.npu.memory_allocated / torch.npu.memory_allocated
            # 收集候选 NPU 模块（torch_npu.npu 优先，torch.npu 兜底）
            npu_candidates = []
            torch_npu_mod = get_torch_npu_module()
            if torch_npu_mod is not None:
                npu_mod = getattr(torch_npu_mod, "npu", None)
                if npu_mod is not None:
                    npu_candidates.append(npu_mod)
            torch_npu_attr = getattr(_TORCH, "npu", None)
            if torch_npu_attr is not None and torch_npu_attr not in npu_candidates:
                npu_candidates.append(torch_npu_attr)
            # 优先尝试 mem_get_info（返回 free, total）
            for mod in npu_candidates:
                if hasattr(mod, "mem_get_info"):
                    try:
                        free, total = mod.mem_get_info()
                        return {
                            "total": int(total),
                            "used": int(total - free),
                            "free": int(free),
                        }
                    except Exception:
                        continue
            # 兜底：memory_allocated（无 total 信息时返回 0 total）
            for mod in npu_candidates:
                if hasattr(mod, "memory_allocated"):
                    try:
                        used = int(mod.memory_allocated())
                        return {"total": 0, "used": used, "free": 0}
                    except Exception:
                        continue
        # 其他设备（如 mps）：返回 0 占位
        return {"total": 0, "used": 0, "free": 0}
    except Exception:
        return {"total": 0, "used": 0, "free": 0}


def memory_usage(device="cpu") -> float:
    """获取当前已用内存百分比（0.0-1.0）。

    基于 :func:`get_memory_info`，返回 ``used / total``。
    若 total 为 0（无法获取），返回 0.0。
    """
    info = get_memory_info(device)
    total = info.get("total", 0)
    if total <= 0:
        return 0.0
    return float(info.get("used", 0)) / float(total)


# ---------------------------------------------------------------------------
# CPU BLAS 线程优化
# ---------------------------------------------------------------------------

# BLAS 线程环境变量列表（OMP / OpenBLAS / MKL / NumPy / Veclib 等）
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# 模块级缓存：auto_tune_threads 设置过的值（避免重复设置）
_auto_tuned_threads: Optional[int] = None


def set_num_threads(n: int) -> None:
    """设置 CPU BLAS 线程数。

    通过设置多个环境变量（``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
    ``MKL_NUM_THREADS`` 等）影响 NumPy 底层 BLAS 库的并行度。
    若 PyTorch 可用，同时调用 ``torch.set_num_threads`` 同步线程数。

    Args:
        n: 线程数；<=0 时表示"不限制"（清除环境变量）
    """
    global _auto_tuned_threads
    if n is None:
        return
    n_int = int(n)
    if n_int <= 0:
        for var in _BLAS_THREAD_ENV_VARS:
            os.environ.pop(var, None)
        _auto_tuned_threads = None
    else:
        for var in _BLAS_THREAD_ENV_VARS:
            os.environ[var] = str(n_int)
        _auto_tuned_threads = n_int
    # 若 PyTorch 可用，同步设置 torch 线程数
    if _TORCH is not None:
        try:
            _TORCH.set_num_threads(max(1, n_int) if n_int > 0 else
                                   max(1, os.cpu_count() or 1))
        except Exception:
            pass


def get_num_threads() -> int:
    """获取当前 BLAS 线程数。

    优先读 ``OMP_NUM_THREADS`` 环境变量；不存在则读 ``OPENBLAS_NUM_THREADS`` /
    ``MKL_NUM_THREADS``；都不存在时返回 ``os.cpu_count()``。
    若 PyTorch 可用且环境变量未设置，回退到 ``torch.get_num_threads()``。
    """
    for var in _BLAS_THREAD_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            try:
                return max(1, int(val))
            except ValueError:
                continue
    if _TORCH is not None:
        try:
            return max(1, int(_TORCH.get_num_threads()))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def auto_tune_threads(model_size_hint: int = 0) -> int:
    """自动调优线程数（根据 CPU 核心数和模型大小）。

    启发式策略：
    - CPU 核心数 <= 4：直接用全部核心数（小机器 / CI 环境）
    - 模型较大（model_size_hint > 10_000_000 参数量）：使用 75% 核心，
      留出余量给数据加载等其他线程
    - 否则使用全部核心数

    Args:
        model_size_hint: 模型参数量提示（0 表示不指定）

    Returns:
        实际设置的线程数
    """
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 4:
        n = cpu_count
    elif model_size_hint and model_size_hint > 10_000_000:
        n = max(1, int(cpu_count * 0.75))
    else:
        n = cpu_count
    set_num_threads(n)
    return n


__all__ = [
    "DeviceBackend",
    "NumpyBackend",
    "get_backend",
    "has_torch",
    "has_torch_npu",
    "get_torch_module",
    "get_torch_npu_module",
    "DEFAULT_DEVICE",
    "_parse_device",
    "is_cpu_device",
    # ROCm / CANN 探测（Part5K1.3 Task 9）
    "has_rocm",
    "get_rocm_version",
    "has_cann",
    "get_cann_version",
    # 显存管理
    "empty_cache",
    "get_memory_info",
    "memory_usage",
    # BLAS 线程
    "set_num_threads",
    "get_num_threads",
    "auto_tune_threads",
]
