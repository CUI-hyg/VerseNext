"""VerseTorch: 核心张量类与 autograd 引擎。

设计要点：
- Tensor 包装 NumPy ndarray，默认 float32。
- 动态计算图：每次操作产生新 Tensor，记录 `_backward` 闭包和 `_prev` 父节点集合。
- 反向模式 autograd：`backward()` 通过 DFS 拓扑排序所有节点，逆序调用 `_backward()` 闭包。
- 梯度累积：闭包内对 `self.grad` 使用 `+=`（PyTorch 语义）。
- broadcasting-aware 反向：`unbroadcast` 工具函数把上游梯度形状还原到目标 shape。
- 全局 `_GRAD_ENABLED` 标志，配合 `no_grad()` / `enable_grad()` 上下文管理器。
"""

from __future__ import annotations

import numpy as np

from .device import (
    has_torch,
    has_torch_npu,
    get_torch_module,
    _parse_device,
    is_cpu_device,
    DEFAULT_DEVICE,
)

# 模块级缓存 torch 模块（无 torch 时为 None）
_TORCH = get_torch_module()
#: PyTorch 是否可用（用于快速短路判断）
_TORCH_AVAILABLE = has_torch()


def _is_torch_data(x) -> bool:
    """判断 ``x`` 是否为 ``torch.Tensor``（无 torch 时恒为 False）。

    作为 Tensor 的内部标记，用于在算子入口区分 CPU/NumPy 路径与 GPU/Torch 路径。
    """
    return _TORCH is not None and isinstance(x, _TORCH.Tensor)


def _np_dtype_to_torch_dtype(np_dtype):
    """把 numpy dtype 映射到 torch dtype（仅常见类型；未匹配返回 None）。"""
    if _TORCH is None:
        return None
    mapping = {
        np.dtype(np.float32): _TORCH.float32,
        np.dtype(np.float64): _TORCH.float64,
        np.dtype(np.float16): _TORCH.float16,
        np.dtype(np.int64): _TORCH.int64,
        np.dtype(np.int32): _TORCH.int32,
        np.dtype(np.int16): _TORCH.int16,
        np.dtype(np.int8): _TORCH.int8,
        np.dtype(np.uint8): _TORCH.uint8,
        np.dtype(np.bool_): _TORCH.bool,
    }
    return mapping.get(np.dtype(np_dtype), None)


# ---------------------------------------------------------------------------
# 全局梯度开关 (Task 1.12)
# ---------------------------------------------------------------------------

_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    """返回当前是否启用梯度构建。"""
    return _GRAD_ENABLED


def set_grad_enabled(mode: bool) -> None:
    """全局设置是否构建计算图。"""
    global _GRAD_ENABLED
    _GRAD_ENABLED = bool(mode)


class no_grad:
    """上下文管理器：在 with 块内禁用计算图构建。

    用法:
        with no_grad():
            y = model(x)  # 不构建计算图
    """

    def __init__(self):
        self.prev = None

    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev
        return False


class enable_grad:
    """上下文管理器：在 with 块内强制启用计算图构建。"""

    def __init__(self):
        self.prev = None

    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev
        return False


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """把广播后的梯度 `grad` 还原回 `target_shape`。

    NumPy 广播规则：
    1. 维度从右对齐，缺失的维度通过在最前面添加大小为 1 的维度补齐。
    2. 大小为 1 的维度会被扩展到目标大小。

    反向：先 sum 掉所有多余的前导轴，再对 `target_shape` 中为 1 的轴做 keepdims sum。
    """
    # 1. reduce 多余的前导轴
    ndim_extra = grad.ndim - len(target_shape)
    if ndim_extra > 0:
        # 将前 ndim_extra 个轴全部求和
        grad = grad.sum(axis=tuple(range(ndim_extra)))
    # 2. 对 target_shape 中为 1 的轴（但 grad 中大于 1）做 keepdims sum
    # 注意：此时 grad.ndim == len(target_shape)
    axes_to_sum = tuple(
        i for i, dim in enumerate(target_shape)
        if dim == 1 and grad.shape[i] != 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    # 3. 最终 reshape 以确保形状精确（应对 axes_to_sum 之后仍可能的形状差异）
    grad = grad.reshape(target_shape)
    return grad


def _as_array(x, dtype=None) -> np.ndarray:
    """把 Python 标量/列表/ndarray 转成 np.ndarray。"""
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x, dtype=dtype)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _default_float_dtype():
    return np.float32


# ---------------------------------------------------------------------------
# Tensor 类 (Task 1.1 - 1.6)
# ---------------------------------------------------------------------------


class Tensor:
    """NumPy 后端的自动微分张量。

    字段：
        data: np.ndarray (默认 float32)
        grad: np.ndarray 或 None，反向传播后填充
        requires_grad: bool，是否需要梯度
        _backward: 闭包，调用时把上游梯度累加到父节点的 grad
        _prev: set of Tensor，父节点集合（用于拓扑排序）
        _op: str，操作名（仅用于调试）
    """

    __array_priority__ = 1000  # 让 numpy 把反向算子优先转给 Tensor

    def __init__(self, data, requires_grad: bool = False, dtype=None, _children=(), _op="",
                 device=None):
        # GPU 路径：data 已是 torch.Tensor（委托 PyTorch autograd）
        if _is_torch_data(data):
            if dtype is not None:
                t_dtype = dtype if _TORCH is not None and isinstance(dtype, _TORCH.dtype) \
                    else _np_dtype_to_torch_dtype(dtype)
                if t_dtype is not None:
                    data = data.to(t_dtype)
            self.data = data
            self.requires_grad = bool(requires_grad)
            # 让 torch 接管 requires_grad（仅在叶子节点上设置）
            if requires_grad and data.is_leaf and not data.requires_grad:
                data.requires_grad_(True)
            self._device = str(data.device) if device is None else str(device)
        elif isinstance(data, Tensor):
            # 复制语义：从 Tensor 构造时拷贝 data 与 requires_grad
            arr = data.data
            if dtype is not None:
                if _is_torch_data(arr):
                    t_dtype = _np_dtype_to_torch_dtype(dtype)
                    if t_dtype is not None:
                        arr = arr.to(t_dtype)
                else:
                    arr = arr.astype(dtype, copy=False)
            self.data = arr
            self.requires_grad = bool(requires_grad)
            self._device = data._device if device is None else str(device)
        else:
            if dtype is None:
                # 类型推断策略：
                # - 如果是 numpy ndarray 或 numpy scalar，保留其 dtype
                #   （用户可能显式指定 float64 用于梯度检查；reduction 也返回 numpy scalar）
                # - 如果是 Python 标量/list，默认 float32（避免 NumPy 把 Python float 升级到 float64）
                # - 整型保持整型（用于索引）
                if isinstance(data, np.ndarray):
                    arr = data
                elif isinstance(data, np.generic):
                    # numpy scalar（如 np.float64, np.int32）
                    arr = np.asarray(data)
                else:
                    arr = np.asarray(data)
                    # Python int / list of ints -> int64 (NumPy default); 保留
                    # Python float / list of floats -> float64 (NumPy default); 改为 float32
                    if arr.dtype == np.float64:
                        arr = arr.astype(np.float32)
            else:
                arr = np.asarray(data, dtype=dtype)
            self.data = arr
            self.requires_grad = bool(requires_grad)
            self._device = "cpu" if device is None else str(device)
        self.grad = None
        # 闭包：调用时将上游梯度 * 链式因子累加到父节点
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # --- 工厂方法 (Task 1.1) ---

    @staticmethod
    def _ensure_float(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind != "f":
            return arr.astype(np.float32)
        if arr.dtype == np.float64:
            return arr.astype(np.float32)
        return arr

    @classmethod
    def zeros(cls, *shape, dtype=np.float32, requires_grad=False) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype=np.float32, requires_grad=False) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def rand(cls, *shape, dtype=np.float32, requires_grad=False) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, dtype=np.float32, requires_grad=False) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

    @classmethod
    def empty(cls, *shape, dtype=np.float32, requires_grad=False) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(np.empty(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def full(cls, shape, fill_value, dtype=np.float32, requires_grad=False) -> "Tensor":
        if isinstance(shape, int):
            shape = (shape,)
        return cls(np.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def arange(cls, start=0, end=None, step=1, dtype=None, requires_grad=False) -> "Tensor":
        # 兼容 arange(end) 与 arange(start, end, step)
        if end is None:
            start, end = 0, start
        arr = np.arange(start, end, step, dtype=dtype)
        if dtype is None:
            arr = cls._ensure_float(arr)
        return cls(arr, requires_grad=requires_grad)

    @classmethod
    def eye(cls, n: int, dtype=np.float32, requires_grad=False) -> "Tensor":
        return cls(np.eye(n, dtype=dtype), requires_grad=requires_grad)

    # --- 基本属性 (Task 1.1) ---

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    @property
    def is_leaf(self):
        # requires_grad 且没有 _prev（即由用户直接创建）的节点是叶子节点
        # 闭包 _backward 默认 lambda: None；若 _prev 为空则视为叶子
        return not self._prev

    # --- 设备相关 (Task 1.2 device 升级) ---

    @property
    def device(self) -> str:
        """返回当前 Tensor 所在的设备字符串。

        CPU 路径返回 ``"cpu"``；GPU/NPU 路径返回 torch device 字符串
        （如 ``"cuda:0"`` / ``"npu:0"`` / ``"mps"``）。
        """
        # 与 self._device 保持同步：若底层 data 是 torch.Tensor，以 data.device 为准
        if _is_torch_data(self.data):
            return str(self.data.device)
        return self._device

    @property
    def _is_torch_tensor(self) -> bool:
        """内部标记：底层 data 是否为 ``torch.Tensor``（GPU/Torch 路径）。

        无 PyTorch 时恒为 False。算子入口用此属性快速分流 CPU/GPU 实现。
        """
        return _is_torch_data(self.data)

    @property
    def is_cuda(self) -> bool:
        """是否在 CUDA 设备上（torch 路径下委托 ``data.is_cuda``）。"""
        if not _is_torch_data(self.data):
            return False
        return bool(self.data.is_cuda)

    def to(self, device, dtype=None) -> "Tensor":
        """迁移到指定 device（可选同时转换 dtype），返回新的 Tensor。

        - ``device="cpu"``：把 torch.Tensor 转回 numpy ndarray，回到自研 autograd 路径。
        - ``device`` 为 GPU/NPU：把 ndarray 转成 torch.Tensor 并迁移到目标设备，
          autograd 委托 PyTorch。
        - 同 device 同 dtype 时直接返回 self（避免无谓拷贝）。

        Args:
            device: 目标设备字符串（``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` /
                ``"npu"`` / ``"mps"`` 等）。
            dtype: 可选 dtype（numpy dtype 或 torch dtype）。

        Returns:
            迁移后的 Tensor（device 与 dtype 满足要求）。

        Raises:
            RuntimeError: 请求 GPU/NPU 但 PyTorch 不可用。
        """
        target_dev = str(device) if device is not None else "cpu"
        target_dtype = _parse_device(target_dev)
        # 同 device 且不要求改 dtype：直接返回 self
        if (self.device == target_dev or
                (_parse_device(self.device) == target_dtype and target_dtype != "cpu")):
            if dtype is None:
                return self
            # 同 device 但要改 dtype
            if _is_torch_data(self.data):
                t_dtype = dtype if isinstance(dtype, _TORCH.dtype) else _np_dtype_to_torch_dtype(dtype)
                if t_dtype is None or self.data.dtype == t_dtype:
                    return self
                new_data = self.data.to(t_dtype)
                return Tensor(new_data, requires_grad=self.requires_grad,
                              _children=(self,) if self.requires_grad else (), _op="to_dtype",
                              device=target_dev)
            else:
                np_dt = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
                if self.data.dtype == np_dt:
                    return self
                return Tensor(self.data.astype(np_dt), requires_grad=self.requires_grad,
                              device="cpu")

        # CPU -> GPU/NPU
        if target_dtype != "cpu":
            if not _TORCH_AVAILABLE:
                raise RuntimeError("未安装 PyTorch，无法使用 GPU")
            if target_dtype == "npu" and not has_torch_npu():
                raise RuntimeError("未安装 torch_npu，无法使用 NPU 设备")
            torch_device = _TORCH.device(target_dev)
            if _is_torch_data(self.data):
                new_data = self.data.to(torch_device)
            else:
                # ndarray -> torch.Tensor（保留 dtype；非 float 用 from_numpy）
                new_data = _TORCH.from_numpy(np.ascontiguousarray(self.data)).to(torch_device)
            if dtype is not None:
                t_dtype = dtype if isinstance(dtype, _TORCH.dtype) else _np_dtype_to_torch_dtype(dtype)
                if t_dtype is not None:
                    new_data = new_data.to(t_dtype)
            # requires_grad 由调用方决定（保持原 requires_grad）
            requires_grad = self.requires_grad
            if requires_grad and new_data.is_leaf and not new_data.requires_grad:
                new_data.requires_grad_(True)
            return Tensor(new_data, requires_grad=requires_grad, device=target_dev)

        # GPU/NPU -> CPU：把 torch.Tensor 转回 ndarray，回到自研 autograd 路径
        if _is_torch_data(self.data):
            arr = self.data.detach().cpu().numpy()
            if dtype is not None:
                np_dt = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
                arr = arr.astype(np_dt)
            return Tensor(arr, requires_grad=self.requires_grad, device="cpu")
        # CPU -> CPU：仅可能改 dtype（前面已处理 dtype 一致的情况）
        if dtype is not None:
            np_dt = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
            return Tensor(self.data.astype(np_dt), requires_grad=self.requires_grad, device="cpu")
        return self

    def cuda(self, device: str = "cuda") -> "Tensor":
        """迁移到 CUDA 设备。

        Args:
            device: CUDA 设备字符串（默认 ``"cuda"``，等价于默认 GPU）。

        Raises:
            RuntimeError: 未安装 PyTorch。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("未安装 PyTorch，无法使用 GPU")
        return self.to(device)

    def npu(self, device: str = "npu") -> "Tensor":
        """迁移到 NPU（华为昇腾）设备。

        Args:
            device: NPU 设备字符串（默认 ``"npu"``）。

        Raises:
            RuntimeError: 未安装 PyTorch 或 torch_npu。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("未安装 PyTorch，无法使用 NPU")
        if not has_torch_npu():
            raise RuntimeError("未安装 torch_npu，无法使用 NPU 设备")
        return self.to(device)

    def cpu(self) -> "Tensor":
        """迁移到 CPU（返回 ndarray 路径的 Tensor）。"""
        return self.to("cpu")

    def numpy(self) -> np.ndarray:
        if self._is_torch_tensor:
            return self.data.detach().cpu().numpy()
        return self.data

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})\n{self.data}"

    # --- 内部辅助：构造带 requires_grad 传播的结果 ---

    def _result(self, out_data, _children, _op, requires_grad=None):
        """构造一个新的 Tensor 结果节点，自动根据 grad enabled 与父节点 requires_grad 设置。"""
        if requires_grad is None:
            requires_grad = _GRAD_ENABLED and any(c.requires_grad for c in _children)
        # 当不需要梯度时，不记录父节点（避免无谓的计算图构建与内存占用）
        if not requires_grad:
            _children = ()
        out = Tensor(out_data, requires_grad=requires_grad, _children=_children, _op=_op)
        return out

    # -----------------------------------------------------------------
    # GPU 路径算子委托 (Task 1.2 device 升级)
    # -----------------------------------------------------------------

    def _torch_apply(self, op_fn, *others, _op="op", requires_grad=None) -> "Tensor":
        """GPU 路径通用算子委托。

        把 ``self.data`` 与 ``others`` 的 data（对齐 device 后）传给 ``op_fn``，
        返回的新 ``torch.Tensor`` 包装成 Tensor。autograd 完全委托 PyTorch，
        不再设置 ``_backward`` 闭包。

        Args:
            op_fn: 接收 ``(self_data, *other_data)`` 的可调用对象，返回 ``torch.Tensor``。
            *others: Tensor 或标量（自动取 .data）。
            _op: 操作名（仅用于调试）。
            requires_grad: 显式指定是否需要梯度；None 时按父节点自动推断。
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("未安装 PyTorch，无法在 GPU 路径执行算子")
        a = self.data
        other_data = []
        other_children = []
        for o in others:
            if isinstance(o, Tensor):
                b = o.data
                # 若两边都是 torch.Tensor 但 device 不同，对齐到 self 的 device
                if _is_torch_data(a) and _is_torch_data(b) and a.device != b.device:
                    b = b.to(a.device)
                other_data.append(b)
                other_children.append(o)
            else:
                other_data.append(o)
        out_data = op_fn(a, *other_data)
        if requires_grad is None:
            requires_grad = _GRAD_ENABLED and (
                self.requires_grad or any(c.requires_grad for c in other_children)
            )
        children = (self,) + tuple(other_children) if requires_grad else ()
        out = Tensor(out_data, requires_grad=requires_grad, _children=children, _op=_op,
                     device=str(out_data.device))
        return out

    def _accumulate_grad(self, grad: np.ndarray):
        """把梯度累加到 self.grad（PyTorch 语义）。

        梯度 dtype 与 self.data.dtype 对齐（PyTorch 行为）。
        """
        target_dtype = self.data.dtype
        if self.grad is None:
            self.grad = grad.astype(target_dtype, copy=True)
        else:
            self.grad = self.grad + grad.astype(target_dtype, copy=False)

    # -----------------------------------------------------------------
    # 元素级算子 (Task 1.2)
    # -----------------------------------------------------------------

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        # GPU 路径委托 torch（autograd 自动构建）
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: a + b, other, _op="+")
        out_data = self.data + other.data

        def _backward():
            if self.requires_grad:
                g = unbroadcast(out.grad, self.shape)
                self._accumulate_grad(g)
            if other.requires_grad:
                g = unbroadcast(out.grad, other.shape)
                other._accumulate_grad(g)

        out = self._result(out_data, (self, other), "+")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: a - b, other, _op="-")
        out_data = self.data - other.data

        def _backward():
            if self.requires_grad:
                g = unbroadcast(out.grad, self.shape)
                self._accumulate_grad(g)
            if other.requires_grad:
                g = unbroadcast(-out.grad, other.shape)
                other._accumulate_grad(g)

        out = self._result(out_data, (self, other), "-")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __rsub__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: a * b, other, _op="*")
        out_data = self.data * other.data

        def _backward():
            if self.requires_grad:
                g = unbroadcast(out.grad * other.data, self.shape)
                self._accumulate_grad(g)
            if other.requires_grad:
                g = unbroadcast(out.grad * self.data, other.shape)
                other._accumulate_grad(g)

        out = self._result(out_data, (self, other), "*")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: a / b, other, _op="/")
        out_data = self.data / other.data

        def _backward():
            if self.requires_grad:
                g = unbroadcast(out.grad / other.data, self.shape)
                self._accumulate_grad(g)
            if other.requires_grad:
                # d(a/b)/db = -a/b^2
                g = unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
                other._accumulate_grad(g)

        out = self._result(out_data, (self, other), "/")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)

    def __pow__(self, power) -> "Tensor":
        # 支持标量 power；如果 power 是 Tensor，按元素幂
        if isinstance(power, Tensor):
            other = power
            # GPU 路径委托 torch
            if self._is_torch_tensor or other._is_torch_tensor:
                return self._torch_apply(lambda a, b: a ** b, other, _op="**")
            out_data = self.data ** other.data

            def _backward():
                if self.requires_grad:
                    g = unbroadcast(
                        out.grad * other.data * (self.data ** (other.data - 1)),
                        self.shape,
                    )
                    self._accumulate_grad(g)
                if other.requires_grad:
                    # d(a^b)/db = a^b * ln(a)
                    safe_log = np.where(self.data > 0, np.log(np.abs(self.data) + 1e-30), 0.0)
                    g = unbroadcast(out.grad * out_data * safe_log, other.shape)
                    other._accumulate_grad(g)

            out = self._result(out_data, (self, other), "**")
            if out.requires_grad:
                out._backward = _backward
            return out
        # 标量 power
        p = float(power)
        # GPU 路径委托 torch
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: a ** p, _op=f"**{p}")
        out_data = self.data ** p

        def _backward():
            if self.requires_grad:
                g = out.grad * (p * (self.data ** (p - 1)))
                self._accumulate_grad(g)

        out = self._result(out_data, (self,), f"**{p}")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        # GPU 路径委托 torch
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: -a, _op="neg")
        return self.__mul__(-1.0)

    # --- 数学函数 ---

    def exp(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.exp(a), _op="exp")
        out_data = np.exp(self.data)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * out_data)

        out = self._result(out_data, (self,), "exp")
        if out.requires_grad:
            out._backward = _backward
        return out

    def log(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.log(a), _op="log")
        out_data = np.log(self.data)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad / self.data)

        out = self._result(out_data, (self,), "log")
        if out.requires_grad:
            out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.sqrt(a), _op="sqrt")
        out_data = np.sqrt(self.data)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * 0.5 / np.sqrt(self.data + 1e-30))

        out = self._result(out_data, (self,), "sqrt")
        if out.requires_grad:
            out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.relu(a), _op="relu")
        out_data = np.maximum(self.data, 0)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0).astype(out.grad.dtype))

        out = self._result(out_data, (self,), "relu")
        if out.requires_grad:
            out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        if self._is_torch_tensor:
            # 委托 torch.nn.functional.gelu（tanh 近似，与 NumPy 路径一致）
            return self._torch_apply(
                lambda a: _TORCH.nn.functional.gelu(a, approximate="tanh"), _op="gelu"
            )
        # 使用 GELU tanh 近似
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        out_data = 0.5 * x * (1.0 + np.tanh(inner))

        def _backward():
            if self.requires_grad:
                # d/dx [0.5 x (1 + tanh(inner))]
                # = 0.5 (1 + tanh(inner)) + 0.5 x * sech^2(inner) * c * (1 + 3*0.044715*x^2)
                t = np.tanh(inner)
                dt = 1.0 - t * t
                d_inner = c * (1.0 + 3 * 0.044715 * x * x)
                grad = 0.5 * (1.0 + t) + 0.5 * x * dt * d_inner
                self._accumulate_grad(out.grad * grad)

        out = self._result(out_data, (self,), "gelu")
        if out.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.sigmoid(a), _op="sigmoid")
        # 数值稳定 sigmoid：用等价公式 0.5 * (1 + tanh(x/2))。
        # 旧版 np.where(x>=0, 1/(1+exp(-x)), exp(x)/(1+exp(x))) 虽然
        # 结果正确，但 NumPy 语义下两个分支都会被计算，导致
        # x 为大负数时 exp(-x) 溢出、大正数时 exp(x) 溢出，产生 RuntimeWarning。
        # tanh 是数值稳定的，无 overflow 风险。
        x = self.data
        out_data = 0.5 * (1.0 + np.tanh(0.5 * x))

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * out_data * (1.0 - out_data))

        out = self._result(out_data, (self,), "sigmoid")
        if out.requires_grad:
            out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.tanh(a), _op="tanh")
        out_data = np.tanh(self.data)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1.0 - out_data * out_data))

        out = self._result(out_data, (self,), "tanh")
        if out.requires_grad:
            out._backward = _backward
        return out

    def silu(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.nn.functional.silu(a), _op="silu")
        # SiLU = x * sigmoid(x)
        # sigmoid 用 0.5 * (1 + tanh(x/2)) 等价公式避免 overflow warning
        x = self.data
        s = 0.5 * (1.0 + np.tanh(0.5 * x))
        out_data = x * s

        def _backward():
            if self.requires_grad:
                # d(silu)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                grad = s + x * s * (1.0 - s)
                self._accumulate_grad(out.grad * grad)

        out = self._result(out_data, (self,), "silu")
        if out.requires_grad:
            out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: _TORCH.abs(a), _op="abs")
        out_data = np.abs(self.data)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * np.sign(self.data))

        out = self._result(out_data, (self,), "abs")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __abs__(self) -> "Tensor":
        return self.abs()

    def maximum(self, other) -> "Tensor":
        """逐元素最大值。"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: _TORCH.maximum(a, b), other, _op="maximum")
        a = self.data
        b = other.data
        out_data = np.maximum(a, b)

        def _backward():
            if self.requires_grad:
                mask = (a >= b).astype(out.grad.dtype)
                g = unbroadcast(out.grad * mask, self.shape)
                self._accumulate_grad(g)
            if other.requires_grad:
                mask = (b > a).astype(out.grad.dtype)
                g = unbroadcast(out.grad * mask, other.shape)
                other._accumulate_grad(g)

        out = self._result(out_data, (self, other), "maximum")
        if out.requires_grad:
            out._backward = _backward
        return out

    def minimum(self, other) -> "Tensor":
        """逐元素最小值。"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: _TORCH.minimum(a, b), other, _op="minimum")
        a = self.data
        b = other.data
        out_data = np.minimum(a, b)

        def _backward():
            if self.requires_grad:
                mask = (a <= b).astype(out.grad.dtype)
                g = unbroadcast(out.grad * mask, self.shape)
                self._accumulate_grad(g)
            if other.requires_grad:
                mask = (b < a).astype(out.grad.dtype)
                g = unbroadcast(out.grad * mask, other.shape)
                other._accumulate_grad(g)

        out = self._result(out_data, (self, other), "minimum")
        if out.requires_grad:
            out._backward = _backward
        return out

    def clamp(self, low=None, high=None) -> "Tensor":
        """逐元素 clamp 到 [low, high]。"""
        if self._is_torch_tensor:
            # torch.clamp 不允许 low/high 同时为 None
            if low is None and high is None:
                return self
            t_low = low.data if isinstance(low, Tensor) else low
            t_high = high.data if isinstance(high, Tensor) else high
            return self._torch_apply(
                lambda a: _TORCH.clamp(a, min=t_low, max=t_high), _op="clamp"
            )
        out_data = self.data
        if low is not None:
            out_data = np.maximum(out_data, low)
        if high is not None:
            out_data = np.minimum(out_data, high)

        def _backward():
            if self.requires_grad:
                grad = out.grad.copy()
                if low is not None:
                    grad = grad * (self.data >= low)
                if high is not None:
                    grad = grad * (self.data <= high)
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "clamp")
        if out.requires_grad:
            out._backward = _backward
        return out

    def softmax(self, dim: int = -1) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(
                lambda a: _TORCH.softmax(a, dim=dim), _op="softmax"
            )
        # 数值稳定 softmax
        x = self.data
        x_max = np.max(x, axis=dim, keepdims=True)
        e = np.exp(x - x_max)
        out_data = e / np.sum(e, axis=dim, keepdims=True)

        def _backward():
            if self.requires_grad:
                # Jacobian-vector product:
                # dx = (s - onehot(argmax) * sum) * grad  -> 通用公式
                # dx_i = s_i * (grad_i - sum_j(s_j * grad_j))
                s = out_data
                dot = np.sum(out.grad * s, axis=dim, keepdims=True)
                grad = s * (out.grad - dot)
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "softmax")
        if out.requires_grad:
            out._backward = _backward
        return out

    def log_softmax(self, dim: int = -1) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(
                lambda a: _TORCH.log_softmax(a, dim=dim), _op="log_softmax"
            )
        x = self.data
        x_max = np.max(x, axis=dim, keepdims=True)
        shifted = x - x_max
        log_sum = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
        out_data = shifted - log_sum

        def _backward():
            if self.requires_grad:
                # softmax
                s = np.exp(out_data)
                # dx_i = grad_i - s_i * sum_j(grad_j)
                sum_grad = np.sum(out.grad, axis=dim, keepdims=True)
                grad = out.grad - s * sum_grad
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "log_softmax")
        if out.requires_grad:
            out._backward = _backward
        return out

    # -----------------------------------------------------------------
    # shape 算子 (Task 1.3)
    # -----------------------------------------------------------------

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if self._is_torch_tensor:
            return self._torch_apply(
                lambda a: a.reshape(shape), _op="reshape"
            )
        out_data = self.data.reshape(shape)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))

        out = self._result(out_data, (self,), "reshape")
        if out.requires_grad:
            out._backward = _backward
        return out

    def view(self, *shape) -> "Tensor":
        # view 是 reshape 的别名（NumPy 默认连续，所以可直接 reshape）
        return self.reshape(*shape)

    def transpose(self, dim0=None, dim1=None) -> "Tensor":
        if self._is_torch_tensor:
            if dim0 is None and dim1 is None:
                # 完全反转所有轴
                return self._torch_apply(lambda a: a.permute(*reversed(range(a.ndim))), _op="transpose")
            if dim0 is None:
                dim0 = -2
            if dim1 is None:
                dim1 = -1
            return self._torch_apply(lambda a: a.transpose(dim0, dim1), _op="transpose")
        if dim0 is None and dim1 is None:
            # 完全反转所有轴
            out_data = self.data.T
        else:
            if dim0 is None:
                dim0 = -2
            if dim1 is None:
                dim1 = -1
            out_data = np.swapaxes(self.data, dim0, dim1)

        def _backward():
            if self.requires_grad:
                if dim0 is None and dim1 is None:
                    self._accumulate_grad(out.grad.T)
                else:
                    self._accumulate_grad(np.swapaxes(out.grad, dim0, dim1))

        out = self._result(out_data, (self,), "transpose")
        if out.requires_grad:
            out._backward = _backward
        return out

    def permute(self, *dims) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        dims = tuple(int(d) for d in dims)
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: a.permute(*dims), _op="permute")
        out_data = np.transpose(self.data, dims)

        def _backward():
            if self.requires_grad:
                # 反向 permute：argsort(dims)
                inv = np.argsort(dims)
                self._accumulate_grad(np.transpose(out.grad, inv))

        out = self._result(out_data, (self,), "permute")
        if out.requires_grad:
            out._backward = _backward
        return out

    def squeeze(self, dim=None) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: a.squeeze(dim), _op="squeeze")
        out_data = np.squeeze(self.data, axis=dim)

        def _backward():
            if self.requires_grad:
                # 反向是 reshape 回原 shape
                self._accumulate_grad(out.grad.reshape(self.shape))

        out = self._result(out_data, (self,), "squeeze")
        if out.requires_grad:
            out._backward = _backward
        return out

    def unsqueeze(self, dim: int) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(lambda a: a.unsqueeze(dim), _op="unsqueeze")
        out_data = np.expand_dims(self.data, axis=dim)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(np.squeeze(out.grad, axis=dim))

        out = self._result(out_data, (self,), "unsqueeze")
        if out.requires_grad:
            out._backward = _backward
        return out

    def expand(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # -1 表示保持原维度大小
        new_shape = list(shape)
        # 从右对齐
        cur_shape = list(self.shape)
        # 在前面补 1
        while len(cur_shape) < len(new_shape):
            cur_shape = [1] + cur_shape
        for i, s in enumerate(new_shape):
            if s == -1:
                new_shape[i] = cur_shape[i]
        out_data = np.broadcast_to(self.data, tuple(new_shape)).copy()

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(unbroadcast(out.grad, self.shape))

        out = self._result(out_data, (self,), "expand")
        if out.requires_grad:
            out._backward = _backward
        return out

    def broadcast_to(self, shape) -> "Tensor":
        if isinstance(shape, int):
            shape = (shape,)
        return self.expand(*shape)

    def contiguous(self) -> "Tensor":
        # NumPy 默认连续；如果非连续（如 transpose 后）则返回 copy
        if self.data.flags["C_CONTIGUOUS"]:
            out_data = self.data
        else:
            out_data = np.ascontiguousarray(self.data)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))

        out = self._result(out_data, (self,), "contiguous")
        if out.requires_grad:
            out._backward = _backward
        return out

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        ndim = self.data.ndim
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        new_shape = (
            self.data.shape[:start_dim]
            + (int(np.prod(self.data.shape[start_dim:end_dim + 1])),)
            + self.data.shape[end_dim + 1:]
        )
        out_data = self.data.reshape(new_shape)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))

        out = self._result(out_data, (self,), "flatten")
        if out.requires_grad:
            out._backward = _backward
        return out

    def __getitem__(self, idx) -> "Tensor":
        """支持 int/slice/tuple/None/boolean mask 索引。

        反向是 scatter / padding：把上游梯度按相同索引放回原 shape 的零张量对应位置。
        GPU 路径委托 torch 自动微分（索引操作可微）。
        """
        if self._is_torch_tensor:
            # 把 Tensor 形式的 idx 转成底层 data
            t_idx = idx.data if isinstance(idx, Tensor) else idx
            return self._torch_apply(lambda a: a[t_idx], _op="getitem")
        out_data = self.data[idx]

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                # 用 np.add.at 处理重复索引（如 integer array indexing）
                np.add.at(grad, idx, out.grad)
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "getitem")
        if out.requires_grad:
            out._backward = _backward
        return out

    # -----------------------------------------------------------------
    # reduction 算子 (Task 1.4)
    # -----------------------------------------------------------------

    def sum(self, dim=None, keepdim: bool = False) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(
                lambda a: a.sum(dim=dim, keepdim=keepdim), _op="sum"
            )
        out_data = self.data.sum(axis=dim, keepdims=keepdim)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # 如果 keepdim=False 且 dim 不是 None，需要 expand 回去
                if dim is not None and not keepdim:
                    if isinstance(dim, int):
                        dims_tuple = (dim,) if dim >= 0 else (self.data.ndim + dim,)
                    else:
                        dims_tuple = tuple(d if d >= 0 else self.data.ndim + d for d in dim)
                    shape = list(self.shape)
                    for d in dims_tuple:
                        shape[d] = 1
                    grad = grad.reshape(shape)
                elif dim is None:
                    grad = np.broadcast_to(grad, self.shape).copy()
                grad = np.broadcast_to(grad, self.shape)
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "sum")
        if out.requires_grad:
            out._backward = _backward
        return out

    def mean(self, dim=None, keepdim: bool = False) -> "Tensor":
        if self._is_torch_tensor:
            return self._torch_apply(
                lambda a: a.mean(dim=dim, keepdim=keepdim), _op="mean"
            )
        out_data = self.data.mean(axis=dim, keepdims=keepdim)
        if dim is None:
            n = self.data.size
        elif isinstance(dim, int):
            n = self.data.shape[dim]
        else:
            n = int(np.prod([self.data.shape[d] for d in dim]))

        def _backward():
            if self.requires_grad:
                grad = out.grad / n
                if dim is not None and not keepdim:
                    if isinstance(dim, int):
                        dims_tuple = (dim,) if dim >= 0 else (self.data.ndim + dim,)
                    else:
                        dims_tuple = tuple(d if d >= 0 else self.data.ndim + d for d in dim)
                    shape = list(self.shape)
                    for d in dims_tuple:
                        shape[d] = 1
                    grad = grad.reshape(shape)
                grad = np.broadcast_to(grad, self.shape)
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "mean")
        if out.requires_grad:
            out._backward = _backward
        return out

    def max(self, dim=None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            out_data = self.data.max()
            out_data = np.asarray(out_data)

            def _backward():
                if self.requires_grad:
                    grad = out.grad * (self.data == out_data).astype(out.grad.dtype)
                    # 如果有多个最大值，按数量平分
                    count = (self.data == out_data).sum()
                    grad = grad / max(count, 1)
                    self._accumulate_grad(grad)

            out = self._result(out_data, (self,), "max")
            if out.requires_grad:
                out._backward = _backward
            return out

        # 沿 dim 求 max
        out_data = np.max(self.data, axis=dim, keepdims=keepdim)
        argmax = np.argmax(self.data, axis=dim)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data, dtype=out.grad.dtype)
                # 创建 one-hot mask
                if keepdim:
                    # argmax shape: (... 1 ...) where dim is 1
                    # 需要把它 squeeze 在 dim 上，然后 scatter
                    mask = np.zeros_like(self.data, dtype=out.grad.dtype)
                    # 把 argmax 的 axis dim 移到最后做 one_hot
                    # 简单做法：用 np.put_along_axis
                    np.put_along_axis(mask, argmax, 1.0, axis=dim)
                    grad = mask * np.broadcast_to(out.grad, self.shape)
                else:
                    mask = np.zeros_like(self.data, dtype=out.grad.dtype)
                    # 把 argmax 扩展一维（在 dim 位置），用 put_along_axis 需要 keepdims 的索引
                    idx = np.expand_dims(argmax, axis=dim)
                    np.put_along_axis(mask, idx, 1.0, axis=dim)
                    # out.grad 形状是 reduced shape，broadcast 到 self.shape
                    grad_expanded = np.broadcast_to(
                        np.expand_dims(out.grad, axis=dim), self.shape
                    )
                    grad = mask * grad_expanded
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "max")
        if out.requires_grad:
            out._backward = _backward
        return out

    def min(self, dim=None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            out_data = self.data.min()
            out_data = np.asarray(out_data)

            def _backward():
                if self.requires_grad:
                    grad = out.grad * (self.data == out_data).astype(out.grad.dtype)
                    count = (self.data == out_data).sum()
                    grad = grad / max(count, 1)
                    self._accumulate_grad(grad)

            out = self._result(out_data, (self,), "min")
            if out.requires_grad:
                out._backward = _backward
            return out

        out_data = np.min(self.data, axis=dim, keepdims=keepdim)
        argmin = np.argmin(self.data, axis=dim)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data, dtype=out.grad.dtype)
                mask = np.zeros_like(self.data, dtype=out.grad.dtype)
                if keepdim:
                    np.put_along_axis(mask, argmin, 1.0, axis=dim)
                    grad = mask * np.broadcast_to(out.grad, self.shape)
                else:
                    idx = np.expand_dims(argmin, axis=dim)
                    np.put_along_axis(mask, idx, 1.0, axis=dim)
                    grad_expanded = np.broadcast_to(
                        np.expand_dims(out.grad, axis=dim), self.shape
                    )
                    grad = mask * grad_expanded
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "min")
        if out.requires_grad:
            out._backward = _backward
        return out

    def argmax(self, dim=None) -> "Tensor":
        # argmax 不可微（subgradient），我们返回不参与梯度的 Tensor
        out_data = np.argmax(self.data, axis=dim)
        out = Tensor(out_data, requires_grad=False, _children=(self,), _op="argmax")
        return out

    def norm(self, p: int = 2) -> "Tensor":
        # 默认 L2 范数，返回标量
        out_data = np.linalg.norm(self.data.flatten(), ord=p)

        def _backward():
            if self.requires_grad:
                x = self.data
                if p == 2:
                    n = np.linalg.norm(x.flatten()) + 1e-30
                    grad = (x / n) * out.grad
                else:
                    # 一般情形 |x|^p 求和再开 p 次方
                    absx = np.abs(x)
                    n = np.sum(absx ** p) ** (1.0 / p) + 1e-30
                    grad = out.grad * (np.sign(x) * (absx ** (p - 1)) / (n ** (p - 1)))
                self._accumulate_grad(grad)

        out = self._result(np.asarray(out_data), (self,), f"norm{p}")
        if out.requires_grad:
            out._backward = _backward
        return out

    def var(self, dim=None, unbiased: bool = True) -> "Tensor":
        if dim is None:
            n = self.data.size
            ddof = 1 if unbiased and n > 1 else 0
            mean_val = self.data.mean()
            # 保留输入 dtype（不强制 float32，便于梯度检查）
            out_data = np.asarray(((self.data - mean_val) ** 2).sum() / (n - ddof))
        else:
            if isinstance(dim, int):
                n = self.data.shape[dim]
            else:
                n = int(np.prod([self.data.shape[d] for d in dim]))
            ddof = 1 if unbiased and n > 1 else 0
            out_data = self.data.var(axis=dim, ddof=ddof)

        def _backward():
            if self.requires_grad:
                # dvar/dx_i = 2 (x_i - mean) / (n - ddof)
                # 注意：mean 项的梯度恰好相互抵消（sum (x-mean)=0），故公式简化为上式
                if dim is None:
                    mean_val = self.data.mean()
                    grad = out.grad * 2.0 * (self.data - mean_val) / (n - ddof)
                else:
                    mean_val = self.data.mean(axis=dim, keepdims=True)
                    # out.grad 形状是 reduced shape（keepdim=False 时），需要 reshape 才能广播
                    if isinstance(dim, int):
                        dims_tuple = (dim,) if dim >= 0 else (self.data.ndim + dim,)
                    else:
                        dims_tuple = tuple(d if d >= 0 else self.data.ndim + d for d in dim)
                    shape = list(self.shape)
                    for d in dims_tuple:
                        shape[d] = 1
                    g = out.grad.reshape(shape)
                    grad = g * 2.0 * (self.data - mean_val) / (n - ddof)
                self._accumulate_grad(grad)

        out = self._result(out_data, (self,), "var")
        if out.requires_grad:
            out._backward = _backward
        return out

    def std(self, dim=None) -> "Tensor":
        # std = sqrt(var)
        v = self.var(dim=dim)
        return v.sqrt()

    # -----------------------------------------------------------------
    # matmul (Task 1.5)
    # -----------------------------------------------------------------

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        # GPU 路径委托 torch.matmul（autograd 自动构建）
        if self._is_torch_tensor or other._is_torch_tensor:
            return self._torch_apply(lambda a, b: _TORCH.matmul(a, b), other, _op="@")
        out_data = self.data @ other.data

        def _backward():
            if self.requires_grad:
                g = out.grad
                a = self.data
                b = other.data
                if a.ndim == 1 and b.ndim == 1:
                    # 1D x 1D -> 标量；dx = g * b, db = g * a
                    self._accumulate_grad(g * b)
                elif a.ndim == 1:
                    # 1D x 2D+ -> 把 a 视作 (1, M)，结果 (..., N)
                    # dx = (g @ b.swapaxes(-1, -2)).sum(-1) 取最后
                    ga = g[..., None, :] @ b.swapaxes(-1, -2)
                    ga = ga.reshape(a.shape)
                    self._accumulate_grad(ga)
                elif b.ndim == 1:
                    # 2D+ x 1D -> a[..., M] x b[M] -> result (...)
                    # dx = outer(g, b)
                    ga = np.expand_dims(g, -1) * b
                    self._accumulate_grad(ga)
                else:
                    ga = g @ b.swapaxes(-1, -2)
                    ga = unbroadcast(ga, a.shape)
                    self._accumulate_grad(ga)
            if other.requires_grad:
                g = out.grad
                a = self.data
                b = other.data
                if a.ndim == 1 and b.ndim == 1:
                    other._accumulate_grad(g * a)
                elif a.ndim == 1:
                    # 1D x 2D+ -> b shape (M, N) 或 (..., M, N)
                    # gb = a[..., None] x g[..., None, :]
                    gb = a[..., None] @ np.expand_dims(g, -2)
                    gb = unbroadcast(gb, b.shape)
                    other._accumulate_grad(gb)
                elif b.ndim == 1:
                    # 2D+ x 1D -> b shape (M,)
                    # gb = sum over all but last: sum_{...} a[..., :, :] * g[..., None]
                    gb = (a.swapaxes(-1, -2) @ np.expand_dims(g, -1))
                    # 形状 (..., M, 1)，需要 sum 掉前面的 batch 维度并 squeeze
                    gb = gb.reshape(-1, b.shape[0]).sum(axis=0)
                    other._accumulate_grad(gb)
                else:
                    gb = a.swapaxes(-1, -2) @ g
                    gb = unbroadcast(gb, b.shape)
                    other._accumulate_grad(gb)

        out = self._result(out_data, (self, other), "@")
        if out.requires_grad:
            out._backward = _backward
        return out

    def matmul(self, other) -> "Tensor":
        return self.__matmul__(other)

    # -----------------------------------------------------------------
    # backward (Task 1.6)
    # -----------------------------------------------------------------

    def backward(self, grad=None) -> None:
        """反向传播：拓扑排序后逆序调用每个节点的 _backward 闭包。

        GPU/Torch 路径下委托 PyTorch autograd（调用 ``data.backward()``），
        并把叶子节点的 ``data.grad`` 同步到 ``Tensor.grad``。

        参数:
            grad: 上游梯度。如果 self 是标量，默认为 1.0。
        """
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad and cannot call backward().")

        # GPU/Torch 路径：委托 PyTorch autograd
        if self._is_torch_tensor:
            if not self.data.requires_grad:
                raise RuntimeError(
                    "底层 torch.Tensor 未启用 requires_grad，无法 backward()"
                )
            if grad is None:
                if self.data.numel() != 1:
                    raise RuntimeError(
                        f"grad can only be implicitly created for scalar outputs "
                        f"(got shape {tuple(self.shape)})"
                    )
                grad_t = _TORCH.ones_like(self.data)
            elif isinstance(grad, Tensor):
                grad_t = grad.data if _is_torch_data(grad.data) else \
                    _TORCH.as_tensor(grad.data, dtype=self.data.dtype, device=self.data.device)
            else:
                grad_t = _TORCH.as_tensor(grad, dtype=self.data.dtype, device=self.data.device)
                if tuple(grad_t.shape) != tuple(self.shape):
                    grad_t = grad_t.expand(self.shape)
            # 调用 torch autograd 反向传播
            self.data.backward(grad_t)
            # 同步叶子节点的 .grad 到 Tensor.grad（便于优化器读取）
            visited = set()
            stack = [self]
            while stack:
                node = stack.pop()
                if id(node) in visited:
                    continue
                visited.add(id(node))
                nd = node.data
                if _is_torch_data(nd) and nd.is_leaf and nd.requires_grad and nd.grad is not None:
                    node.grad = nd.grad
                for child in node._prev:
                    if id(child) not in visited:
                        stack.append(child)
            return

        # CPU/NumPy 路径（原有实现）
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    f"grad can only be implicitly created for scalar outputs (got shape {self.shape})"
                )
            grad = np.ones_like(self.data)
        elif isinstance(grad, Tensor):
            grad = grad.data
        else:
            grad = np.asarray(grad, dtype=self.data.dtype)
            if grad.shape != self.shape:
                grad = np.broadcast_to(grad, self.shape).copy()

        # 梯度 dtype 与 self.data 对齐
        if grad.dtype != self.data.dtype:
            grad = grad.astype(self.data.dtype, copy=False)
        self.grad = grad

        # 拓扑排序：迭代式 DFS（显式栈替代递归，避免深度计算图触发
        # `RecursionError: maximum recursion depth exceeded`）。
        # 原递归实现在多层 Mamba2 / Transformer 计算图（每层几十个算子）
        # 下深度可达 1000+，超过 Python 默认递归上限。
        topo = []
        visited = set()
        # 栈元素：(node, child_iter)；child_iter 用于"续迭"剩余子节点
        stack = [(self, iter(self._prev))]
        visited.add(id(self))
        while stack:
            node, child_iter = stack[-1]
            advanced = False
            for child in child_iter:
                if id(child) not in visited:
                    visited.add(id(child))
                    stack.append((child, iter(child._prev)))
                    advanced = True
                    break
            if not advanced:
                # 当前节点所有子节点已处理完毕，输出到 topo
                topo.append(node)
                stack.pop()

        # 逆序调用 _backward
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """清空当前 Tensor 的 grad。"""
        if self._is_torch_tensor and self.data.is_leaf and self.data.requires_grad:
            self.data.grad = None
        self.grad = None

    def detach(self) -> "Tensor":
        """返回一个脱离计算图的副本（共享 data）。"""
        if self._is_torch_tensor:
            return Tensor(self.data.detach(), requires_grad=False,
                          device=str(self.data.device))
        return Tensor(self.data, requires_grad=False)

    def clone(self) -> "Tensor":
        """返回一个深拷贝（数据复制）。"""
        if self._is_torch_tensor:
            return Tensor(self.data.clone(), requires_grad=self.requires_grad,
                          device=str(self.data.device))
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    # 注意：不重载 __eq__ 与 __hash__，使用 Python 默认的 id-based 语义，
    # 这样 set(Tensor) 去重与拓扑排序行为正确。
    # 用户如需逐元素比较，可用 (a.data == b.data) 或 (a - b).abs() 等。

    # 便捷属性
    def cast(self, dtype) -> "Tensor":
        """类型转换（不可微，但保留 requires_grad）。"""
        if self._is_torch_tensor:
            t_dtype = dtype if isinstance(dtype, _TORCH.dtype) else _np_dtype_to_torch_dtype(dtype)
            return self._torch_apply(lambda a: a.to(t_dtype), _op="cast")
        out_data = self.data.astype(dtype)

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.astype(self.data.dtype))

        out = self._result(out_data, (self,), "cast")
        if out.requires_grad:
            out._backward = _backward
        return out

    def float(self) -> "Tensor":
        if self._is_torch_tensor:
            return self.cast(_TORCH.float32)
        return self.cast(np.float32)

    def long(self) -> "Tensor":
        # 整型转换：不可微
        if self._is_torch_tensor:
            return Tensor(self.data.to(_TORCH.int64), requires_grad=False,
                          device=str(self.data.device))
        return Tensor(self.data.astype(np.int64), requires_grad=False)

    def int(self) -> "Tensor":
        if self._is_torch_tensor:
            return Tensor(self.data.to(_TORCH.int32), requires_grad=False,
                          device=str(self.data.device))
        return Tensor(self.data.astype(np.int32), requires_grad=False)

    def bool(self) -> "Tensor":
        if self._is_torch_tensor:
            return Tensor(self.data.to(_TORCH.bool), requires_grad=False,
                          device=str(self.data.device))
        return Tensor(self.data.astype(bool), requires_grad=False)
