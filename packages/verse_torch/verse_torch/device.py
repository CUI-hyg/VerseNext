"""VerseTorch: 设备抽象层与后端工厂。

设计要点
========
- ``DeviceBackend`` 抽象基类定义 ``matmul`` / ``linear`` / ``softmax`` /
  ``layernorm`` / ``rmsnorm`` / ``rope`` / ``attention`` 等算子接口，
  以及只读 ``device_type`` 属性。
- ``NumpyBackend``：默认 CPU 后端，所有算子用 NumPy 实现（封装现有操作，
  保持与 Tensor 自研 autograd 路径一致的行为）。
- ``TorchBackend``：PyTorch 委托后端，支持 ``cuda`` / ``mps`` / ``npu``
  （``npu`` 通过 ``torch_npu`` 扩展支持）。算子委托给 ``torch.Tensor``，
  CUDA kernel 走 PyTorch 原生实现，**不自研 kernel**。
- ``get_backend(device)`` 工厂函数：根据 device 字符串返回对应 backend 实例。
- ``has_torch()`` / ``has_torch_npu()``：检测 PyTorch / torch_npu 是否可用。
- 模块级常量 ``DEFAULT_DEVICE = "cpu"``。

向后兼容
========
无 PyTorch 环境下，本模块仍可正常 import；只有当请求 GPU/NPU 后端时
才会抛 ``RuntimeError``。
"""

from __future__ import annotations

import abc
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
# 默认设备与设备字符串解析
# ---------------------------------------------------------------------------

#: 默认设备字符串（CPU-first 引擎）
DEFAULT_DEVICE = "cpu"


def _parse_device(device) -> str:
    """规范化 device 字符串。

    接受 ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``"mps"`` / ``"npu"`` /
    ``"npu:0"`` 等形式，返回小写的 device type（``"cpu"`` / ``"cuda"`` /
    ``"mps"`` / ``"npu"``）。
    """
    if device is None:
        return "cpu"
    s = str(device).lower()
    if s.startswith(("cuda", "npu", "mps")):
        return s.split(":")[0]
    return "cpu"


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
        """返回后端设备类型字符串（``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"npu"``）。"""

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
            ``"npu"`` / ``"npu:0"`` 等，``None`` 等价于 ``"cpu"``。

    Returns:
        ``DeviceBackend`` 实例（``NumpyBackend`` 或 ``TorchBackend``）。

    Raises:
        RuntimeError: 请求 GPU/NPU 但 PyTorch 不可用，或请求 NPU 但
            ``torch_npu`` 不可用。
    """
    dtype = _parse_device(device)
    if dtype == "cpu":
        if "cpu" not in _BACKEND_CACHE:
            _BACKEND_CACHE["cpu"] = NumpyBackend()
        return _BACKEND_CACHE["cpu"]

    # 非 CPU：必须依赖 torch
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
        _BACKEND_CACHE[key] = TorchBackend(device=key)
    return _BACKEND_CACHE[key]


def clear_backend_cache() -> None:
    """清空 backend 缓存（主要用于测试隔离）。"""
    _BACKEND_CACHE.clear()


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
]
