"""VerseTorch: PyTorch 委托后端实现。

``TorchBackend`` 把算子委托给 ``torch.Tensor``，CUDA kernel 走 PyTorch 原生
实现，NPU 通过 ``torch_npu`` 扩展支持（CANN 为其底层计算库），ROCm 走
PyTorch ROCm build（HIP-on-ROCm，复用 cuda API）。本模块**仅在 PyTorch
可用时**被实例化（由 ``device.get_backend`` 工厂延迟导入）。

主要导出
========
- ``TorchBackend``：继承 ``DeviceBackend``，实现 ``matmul`` / ``linear`` /
  ``attention`` / ``softmax`` / ``layernorm`` / ``rmsnorm`` / ``rope`` 等算子
  的 torch 委托。接受 ``"cuda"`` / ``"mps"`` / ``"npu"`` / ``"rocm"`` /
  ``"rocm:N"`` 设备字符串（``"rocm"`` 内部映射到 ``torch.device("cuda")``，
  原字符串保留在 ``_device_str`` 用于诊断）。
- ``autocast``：fp16 混合精度上下文管理器（CPU 时 no-op；ROCm 映射到
  ``device_type="cuda"``；NPU 路径首次启用时打印 CANN + PyTorch + torch_npu
  版本日志，便于运维诊断）。
- ``to_torch`` / ``to_numpy``：``ndarray`` <-> ``torch.Tensor`` 转换工具。

设计原则
========
- **不自研 CUDA / ROCm / CANN kernel**：所有 GPU/NPU 计算走 PyTorch 原生
  算子（含 ``torch.nn.functional.scaled_dot_product_attention`` 等 fused
  路径）；ROCm 走 PyTorch ROCm build 的 MIOpen / rocBLAS / FlashAttention-ROCm；
  NPU 走 ``torch_npu`` 的 HCCL / hccl_kernel。
- **NPU 走 torch_npu**：``torch_npu`` 注册后，``torch.device("npu")`` 可用，
  其余 API 与 cuda 一致。
- **ROCm 走 PyTorch cuda API**：PyTorch 不识别 ``"rocm"`` 字符串，但 ROCm
  build 通过 HIP 暴露 cuda API，故 ``"rocm"`` / ``"rocm:N"`` 内部映射到
  ``torch.device("cuda")`` / ``torch.device("cuda:N")``。
- **autograd 委托**：本后端只提供前向算子，反向传播由 ``torch.Tensor`` 自身
  autograd 机制处理（``Tensor.backward`` 在 GPU 路径调用 ``data.backward()``）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np

from .device import (
    DeviceBackend,
    has_torch,
    has_torch_npu,
    has_cann,
    get_torch_module,
    get_torch_npu_module,
    get_cann_version,
    _parse_device,
)

# 仅在 torch 可用时绑定到模块级变量，便于本文件内部直接使用
if has_torch():
    import torch  # type: ignore
else:  # pragma: no cover - 无 torch 环境下兜底，避免 NameError
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# 转换工具
# ---------------------------------------------------------------------------


def _torch_device(device):
    """把字符串 device 转成 ``torch.device`` 实例。

    Args:
        device: ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ``"npu"`` / ``"mps"``
            / ``"rocm"`` / ``"rocm:0"`` ...

    Returns:
        ``torch.device`` 实例

    Note:
        ``"rocm"`` / ``"rocm:N"`` 会被映射到 ``torch.device("cuda")`` /
        ``torch.device("cuda:N")``：PyTorch 不识别 "rocm" 字符串，但 ROCm
        build 的 PyTorch 通过 HIP 暴露 cuda API，故走 cuda 路径
        （HIP-on-ROCm，不自研 kernel）。原 "rocm" 字符串保留在
        ``TorchBackend._device_str`` 中用于诊断。

    Raises:
        RuntimeError: 未安装 torch；或 NPU 设备但未安装 torch_npu。
    """
    if torch is None:
        raise RuntimeError("未安装 PyTorch，无法构造 torch.device")
    dtype = _parse_device(device)
    if dtype == "npu" and not has_torch_npu():
        raise RuntimeError(
            f"未安装 torch_npu，无法使用 NPU 设备 '{device}'"
        )
    # ROCm (HIP-on-ROCm) 走 PyTorch cuda 路径：rocm / rocm:N → cuda / cuda:N
    dev_str = str(device).lower()
    if dev_str == "rocm":
        dev_str = "cuda"
    elif dev_str.startswith("rocm:"):
        dev_str = "cuda:" + dev_str.split(":", 1)[1]
    return torch.device(dev_str)


def to_torch(ndarray, device: str = "cpu", dtype=None):
    """把 numpy ndarray 或 Python 标量转成 ``torch.Tensor``。

    Args:
        ndarray: ``np.ndarray`` / 标量 / 已是 ``torch.Tensor`` 的对象
        device: 目标设备字符串（``"cpu"`` / ``"cuda"`` / ``"npu"`` / ...）
        dtype: 可选 dtype（``torch.dtype`` 或 numpy dtype）

    Returns:
        ``torch.Tensor``（位于指定 device 上）

    Raises:
        RuntimeError: 未安装 PyTorch。
    """
    if torch is None:
        raise RuntimeError("未安装 PyTorch，无法调用 to_torch()")
    if isinstance(ndarray, torch.Tensor):
        t = ndarray
    elif isinstance(ndarray, np.ndarray):
        # from_numpy 保留原 dtype；非连续 / 非标准 layout 由 torch 处理
        t = torch.from_numpy(ndarray)
    else:
        t = torch.as_tensor(ndarray)
    if dtype is not None:
        # 兼容传入 numpy dtype 的情况
        if isinstance(dtype, np.dtype):
            dtype = _numpy_dtype_to_torch(dtype)
        t = t.to(dtype=dtype)
    return t.to(_torch_device(device))


def to_numpy(torch_tensor):
    """把 ``torch.Tensor`` 转成 numpy ndarray（先 detach 再 cpu）。

    非 ``torch.Tensor`` 输入直接 ``np.asarray``。
    """
    if torch is None:
        raise RuntimeError("未安装 PyTorch，无法调用 to_numpy()")
    if isinstance(torch_tensor, torch.Tensor):
        return torch_tensor.detach().cpu().numpy()
    return np.asarray(torch_tensor)


def _numpy_dtype_to_torch(np_dtype):
    """把 numpy dtype 映射到 torch dtype（常见类型）。"""
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }
    return mapping.get(np.dtype(np_dtype), None)


# ---------------------------------------------------------------------------
# autocast 上下文管理器
# ---------------------------------------------------------------------------

# NPU autocast 首次启用日志标记：避免每个 autocast 块都打印 CANN 版本日志
# （Part5K1.3 Task 10.4：仅首次启用时打印，便于运维诊断）
_npu_autocast_logged: bool = False


@contextmanager
def autocast(device=None, dtype=None, enabled: bool = True):
    """混合精度 autocast 上下文管理器。

    GPU（cuda / mps / npu / rocm）设备下启用 ``torch.autocast``（默认 fp16）；
    CPU 设备下为 **no-op**（按需求"CPU 时验证 no-op"）。
    无 PyTorch 时同样 no-op。

    Part4K2 Task 5.5: NPU 后端完善 —— 当 device 为 ``"npu"`` 且
    ``torch_npu`` 可用时，``torch.autocast(device_type="npu", ...)`` 由
    ``torch_npu`` 注册后可正常工作。

    Part5K1.3 Task 10:
    - **ROCm 兼容**：``device_type="rocm"`` 等价 ``device_type="cuda"``
      （PyTorch ROCm build 原生支持 fp16 autocast via HIP，在
      ``torch.autocast(device_type="cuda")`` 下生效）。内部把 ``"rocm"``
      映射为 ``"cuda"`` 传给 ``torch.autocast``（PyTorch 不识别 "rocm"
      字符串，但 ROCm build 通过 HIP 暴露 cuda API，不自研 kernel）。
    - **NPU CANN 日志**：NPU 路径首次启用 autocast 时打印 CANN +
      PyTorch + torch_npu 版本日志（便于运维诊断，仅首次打印）。

    Args:
        device: 设备字符串或 ``torch.device``；``None`` 时自动用 ``"cpu"``。
        dtype: 计算 dtype（cuda 默认 ``torch.float16``）。
        enabled: 是否启用 autocast；``False`` 时直接 yield（no-op）。
    """
    global _npu_autocast_logged
    if not enabled or torch is None:
        yield
        return
    dev_str = str(device) if device is not None else "cpu"
    dtype_str = _parse_device(dev_str)
    if dtype_str == "cpu":
        # CPU 时 no-op
        yield
        return
    if dtype is None:
        dtype = torch.float16
    elif isinstance(dtype, np.dtype):
        dtype = _numpy_dtype_to_torch(dtype)
    # NPU 检测：torch_npu 必须可用才能 autocast
    if dtype_str == "npu" and not has_torch_npu():
        # torch_npu 不可用时降级为 no-op（避免训练中断）
        yield
        return
    # ROCm (HIP-on-ROCm) 走 PyTorch cuda 路径：rocm → cuda
    # PyTorch 不识别 "rocm" 字符串，但 ROCm build 通过 HIP 暴露 cuda API
    torch_device_type = "cuda" if dtype_str == "rocm" else dtype_str
    # NPU 路径首次启用时打印 CANN 版本日志（便于运维诊断，仅首次打印）
    if dtype_str == "npu" and not _npu_autocast_logged:
        try:
            cann_version = get_cann_version() if has_cann() else None
            torch_version = getattr(torch, "__version__", "unknown")
            torch_npu_mod = get_torch_npu_module()
            torch_npu_version = (
                getattr(torch_npu_mod, "__version__", "unknown")
                if torch_npu_mod is not None
                else "unavailable"
            )
            print(
                f"[autocast] NPU CANN version={cann_version}, "
                f"torch={torch_version}, torch_npu={torch_npu_version}"
            )
        except Exception:
            # 日志打印失败不应影响训练
            pass
        finally:
            _npu_autocast_logged = True
    with torch.autocast(device_type=torch_device_type, dtype=dtype, enabled=enabled):
        yield


# ---------------------------------------------------------------------------
# TorchBackend
# ---------------------------------------------------------------------------


class TorchBackend(DeviceBackend):
    """PyTorch 委托后端。

    算子委托给 ``torch.Tensor``，CUDA kernel 走 PyTorch 原生实现，
    NPU 通过 ``torch_npu`` 扩展支持，ROCm 通过 PyTorch ROCm build
    （HIP-on-ROCm，暴露为 cuda API）支持。

    Args:
        device: 设备字符串（如 ``"cuda:0"`` / ``"npu:1"`` / ``"mps"`` /
            ``"rocm"`` / ``"rocm:0"``）。``"rocm"`` / ``"rocm:N"`` 内部
            映射到 ``torch.device("cuda")`` / ``torch.device("cuda:N")``
            （PyTorch 不识别 "rocm" 字符串，但 ROCm build 通过 HIP 暴露
            cuda API），原 ``"rocm"`` 字符串保留在 ``_device_str`` 中用于
            诊断（见 :attr:`device_type`）。

    Raises:
        RuntimeError: 未安装 PyTorch；或 NPU 设备但未安装 torch_npu。
    """

    def __init__(self, device: str = "cuda"):
        if torch is None:
            raise RuntimeError("未安装 PyTorch，无法实例化 TorchBackend")
        # 保留原 device 字符串（如 "rocm:0"）用于诊断；rocm → cuda 的映射
        # 在 _torch_device 内部完成（不自研 kernel，走 PyTorch 原生 HIP 路径）
        self._device_str = str(device).lower()
        self._torch_device = _torch_device(self._device_str)

    @property
    def device_type(self) -> str:
        """返回后端设备类型（用于诊断）。

        - ``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"npu"``：原值
        - ``"rocm"``：``_device_str`` 以 ``"rocm"`` 开头时返回 ``"rocm"``
          （诊断用，底层 ``_torch_device`` 已映射到 ``cuda``）
        """
        # split(":")[0] 对 "rocm:0" 返回 "rocm"，对 "cuda:0" 返回 "cuda"
        # 满足"rocm 开头返回 rocm，否则按原逻辑返回 cuda/npu/cpu"的要求
        return self._device_str.split(":")[0]

    @property
    def torch_device(self):
        """返回底层 ``torch.device`` 实例。"""
        return self._torch_device

    @property
    def torch_device_str(self) -> str:
        """返回底层 device 字符串（小写）。"""
        return self._device_str

    # ---- 转换工具 ----

    def from_numpy(self, x):
        """把 ndarray 或标量转到本后端 device 上的 ``torch.Tensor``。

        若 ``x`` 已是 ``torch.Tensor``，迁移到本后端 device。
        """
        if isinstance(x, torch.Tensor):
            return x.to(self._torch_device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self._torch_device)
        return torch.as_tensor(x, device=self._torch_device)

    def to_numpy(self, t):
        """把 ``torch.Tensor`` 转回 numpy ndarray。"""
        return to_numpy(t)

    # ---- 算子 ----

    def matmul(self, a, b):
        """矩阵乘法 ``a @ b``。"""
        return torch.matmul(a, b)

    def linear(self, x, weight, bias=None):
        """全连接层 ``y = x @ weight.T + bias``，委托 ``F.linear``。"""
        return torch.nn.functional.linear(x, weight, bias)

    def softmax(self, x, dim: int = -1):
        """沿 ``dim`` 做 softmax。"""
        return torch.softmax(x, dim=dim)

    def layernorm(self, x, weight, bias, eps: float = 1e-5):
        """LayerNorm，委托 ``F.layer_norm``（weight/bias 需在相同 device 上）。"""
        return torch.nn.functional.layer_norm(
            x,
            normalized_shape=tuple(weight.shape),
            weight=weight,
            bias=bias,
            eps=eps,
        )

    def rmsnorm(self, x, weight, eps: float = 1e-6):
        """RMSNorm（与 ``NumpyBackend`` 数值等价）。"""
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + eps)
        return x / rms * weight

    def rope(self, x, cos=None, sin=None):
        """RoPE 委托实现（GPT-NeoX 风格 rotate_half，与 NumpyBackend 一致）。"""
        if cos is None or sin is None:
            return x
        d = x.shape[-1]
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos_b = cos[..., :half] if cos.shape[-1] >= d else cos
        sin_b = sin[..., :half] if sin.shape[-1] >= d else sin
        return torch.cat(
            [x1 * cos_b - x2 * sin_b, x1 * sin_b + x2 * cos_b], dim=-1
        )

    def attention(self, q, k, v, mask=None):
        """Scaled dot-product attention。

        优先委托 ``F.scaled_dot_product_attention``（含 fused kernel）；
        当 ``mask`` 形状不符合 SDPA 要求时回退到手动 softmax 实现。
        """
        # 若 q/k/v 形状兼容且 mask 为 None / additive，走 SDPA fused 路径
        try:
            if mask is None:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=False
                )
            # 仅当 mask 形状可作为 attn_mask 时才走 SDPA
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False
            )
        except Exception:
            # 回退到手动实现
            d = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
            if mask is not None:
                scores = scores + mask
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)


__all__ = [
    "TorchBackend",
    "autocast",
    "to_torch",
    "to_numpy",
    "_torch_device",
]
