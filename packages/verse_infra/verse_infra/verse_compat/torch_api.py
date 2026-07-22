"""Task 5.2: PyTorch-compatible API aliases for easy migration.

提供与 PyTorch 兼容的别名与工厂函数，便于将现有 ``import torch`` 的代码
以最小改动迁移到 ``verse_torch`` 后端。

用法
----
    from verse_compat import torch_api as torch
    # 或：from verse_compat.torch_api import Tensor, nn, optim

    x = torch.randn(2, 3)             # -> verse_torch.Tensor
    linear = torch.nn.Linear(3, 4)
    out = linear(x)

设计要点
--------
- 不重新实现算子，全部透传到 ``verse_torch``；
- 仅做命名别名，不引入新行为；
- dtype 用字符串表示（PyTorch 用 ``torch.float32`` 对象，这里用 ``"float32"``），
  调用 ``Tensor.cast(dtype)`` 时也接受字符串（``np.dtype`` 自动转换）。
"""

from __future__ import annotations

import numpy as np

from verse_torch import (
    Tensor,
    nn,
    optim,
    losses,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
)
from verse_torch.nn import (
    Linear,
    Embedding,
    LayerNorm,
    RMSNorm,
    Dropout,
    Module,
    Sequential,
    ModuleList,
)
from verse_torch.optim import SGD, Adam, AdamW
from verse_torch.losses import cross_entropy, mse_loss


# ---------------------------------------------------------------------------
# dtype 字符串别名（与 PyTorch 名称一致）
# ---------------------------------------------------------------------------

float16 = "float16"
float32 = "float32"
float64 = "float64"
bfloat16 = "bfloat16"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
bool = "bool"

# numpy dtype 转换表（字符串 -> numpy dtype）
_DTYPE_STR_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bfloat16": np.float32,  # numpy 无原生 bf16，回退到 float32
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def _to_numpy_dtype(dtype):
    """将字符串 / numpy dtype / None 转换为 numpy dtype。"""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        if dtype not in _DTYPE_STR_TO_NUMPY:
            raise ValueError(f"Unknown dtype string: {dtype!r}")
        return _DTYPE_STR_TO_NUMPY[dtype]
    return np.dtype(dtype)


# ---------------------------------------------------------------------------
# 工厂函数（与 torch.tensor / torch.zeros 等同义）
# ---------------------------------------------------------------------------


def tensor(data, requires_grad: bool = False, dtype=None) -> Tensor:
    """构造一个 Tensor（对应 ``torch.tensor``）。"""
    np_dtype = _to_numpy_dtype(dtype)
    return Tensor(data, requires_grad=requires_grad, dtype=np_dtype)


def zeros(*shape, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """全零 Tensor（对应 ``torch.zeros``）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=np_dtype), requires_grad=requires_grad)


def ones(*shape, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """全一 Tensor（对应 ``torch.ones``）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape, dtype=np_dtype), requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """标准正态分布 Tensor（对应 ``torch.randn``）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.random.randn(*shape).astype(np_dtype), requires_grad=requires_grad)


def rand(*shape, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """[0, 1) 均匀分布 Tensor（对应 ``torch.rand``）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.random.rand(*shape).astype(np_dtype), requires_grad=requires_grad)


def arange(*args, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """range Tensor（对应 ``torch.arange``）。"""
    np_dtype = _to_numpy_dtype(dtype)
    return Tensor.arange(*args, dtype=np_dtype, requires_grad=requires_grad)


def full(shape, fill_value, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """常数填充 Tensor（对应 ``torch.full``）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor(np.full(shape, fill_value, dtype=np_dtype), requires_grad=requires_grad)


def empty(*shape, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """未初始化 Tensor（对应 ``torch.empty``，实际用 zeros 占位）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    # NumPy 的 empty 不初始化内存，但出于确定性考虑用 zeros
    return Tensor(np.zeros(shape, dtype=np_dtype), requires_grad=requires_grad)


def eye(n: int, dtype=None, requires_grad: bool = False, **kwargs) -> Tensor:
    """单位矩阵（对应 ``torch.eye``）。"""
    np_dtype = _to_numpy_dtype(dtype) or np.float32
    return Tensor(np.eye(n, dtype=np_dtype), requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# 常用函数别名
# ---------------------------------------------------------------------------


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return x.softmax(dim=dim)


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def exp(x: Tensor) -> Tensor:
    return x.exp()


def log(x: Tensor) -> Tensor:
    return x.log()


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def cat(tensors, dim: int = 0) -> Tensor:
    """沿指定轴拼接多个 Tensor（对应 ``torch.cat``）。"""
    if not tensors:
        raise ValueError("cat: tensors list is empty")
    arrs = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    out_arr = np.concatenate(arrs, axis=dim)
    # 仅当所有输入都需要 grad 时才构建图（保守策略）
    requires_grad = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
    out = Tensor(out_arr, requires_grad=requires_grad)
    return out


def stack(tensors, dim: int = 0) -> Tensor:
    """沿新轴堆叠多个 Tensor（对应 ``torch.stack``）。"""
    if not tensors:
        raise ValueError("stack: tensors list is empty")
    arrs = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    out_arr = np.stack(arrs, axis=dim)
    requires_grad = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
    return Tensor(out_arr, requires_grad=requires_grad)


def no_grad_context():
    """与 ``torch.no_grad()`` 同义（已导出 ``no_grad`` 别名）。"""
    return no_grad()


# ---------------------------------------------------------------------------
# 导出清单
# ---------------------------------------------------------------------------

__all__ = [
    # 核心类
    "Tensor",
    "nn",
    "optim",
    "losses",
    # nn 类
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Module",
    "Sequential",
    "ModuleList",
    # optim 类
    "SGD",
    "Adam",
    "AdamW",
    # losses
    "cross_entropy",
    "mse_loss",
    # grad 控制
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    # 工厂函数
    "tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    "arange",
    "full",
    "empty",
    "eye",
    # 数学函数
    "softmax",
    "sigmoid",
    "relu",
    "gelu",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "matmul",
    "cat",
    "stack",
    # dtype 字符串
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
]
