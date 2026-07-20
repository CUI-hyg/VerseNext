"""VerseTorch: 神经网络模块（nn.Module 基类与核心层）。

设计参考 PyTorch `torch.nn`：
- Module 自动注册子模块与参数（通过 __setattr__ 拦截）。
- parameters() / state_dict() / load_state_dict() 支持模型持久化。
- train()/eval() 控制 training 模式（影响 Dropout、BatchNorm 等）。
- 核心层：Linear、Embedding、LayerNorm、RMSNorm、Dropout、Sequential、ModuleList。
- 初始化辅助：kaiming_uniform_、xavier_uniform_、normal_、zeros_、ones_。
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, no_grad, _GRAD_ENABLED


# ---------------------------------------------------------------------------
# 初始化辅助函数 (Task 1.8)
# ---------------------------------------------------------------------------


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Tensor:
    """Kaiming 均匀初始化（He initialization, uniform 版本）。

    参考 PyTorch `nn.init.kaiming_uniform_`。
    """
    fan_in = tensor.shape[-1] if tensor.ndim >= 2 else tensor.shape[0]
    fan_out = tensor.shape[0] if tensor.ndim >= 2 else tensor.shape[0]
    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode}")
    # gain = sqrt(2 / (1 + a^2))  for leaky_relu; for relu a=0
    gain = np.sqrt(2.0 / (1.0 + a * a))
    if nonlinearity == "linear" or nonlinearity == "sigmoid" or nonlinearity == "tanh":
        gain = 1.0
    elif nonlinearity == "relu":
        gain = np.sqrt(2.0)
    std = gain / np.sqrt(fan)
    # uniform: bound = sqrt(3) * std
    bound = np.sqrt(3.0) * std
    with no_grad():
        tensor.data = np.random.uniform(-bound, bound, size=tensor.shape).astype(np.float32)
    return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier / Glorot 均匀初始化。"""
    fan_in = tensor.shape[-1] if tensor.ndim >= 2 else tensor.shape[0]
    fan_out = tensor.shape[0] if tensor.ndim >= 2 else tensor.shape[0]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std
    with no_grad():
        tensor.data = np.random.uniform(-bound, bound, size=tensor.shape).astype(np.float32)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """正态分布初始化。"""
    with no_grad():
        tensor.data = (np.random.randn(*tensor.shape) * std + mean).astype(np.float32)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    with no_grad():
        tensor.data = np.zeros(tensor.shape, dtype=np.float32)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    with no_grad():
        tensor.data = np.ones(tensor.shape, dtype=np.float32)
    return tensor


def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    with no_grad():
        tensor.data = np.random.uniform(low, high, size=tensor.shape).astype(np.float32)
    return tensor


# ---------------------------------------------------------------------------
# Module 基类 (Task 1.7)
# ---------------------------------------------------------------------------


class Module:
    """所有神经网络模块的基类。

    通过 __setattr__ 自动注册 Tensor（requires_grad=True）参数和 Module 子模块。
    子类需要实现 forward() 方法。
    """

    def __init__(self):
        # 跳过自定义 __setattr__，直接设置内部字典
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        # 自动注册：Tensor（requires_grad=True）-> _parameters；Module -> _modules
        if isinstance(value, Tensor):
            # 即使 requires_grad=False，也作为参数保存（例如 Embedding 权重）
            # 但严格遵循 PyTorch：只有 requires_grad=True 的 Tensor 才是 parameter
            # 不过实践中我们更宽松：保存所有 Tensor 属性
            params = self.__dict__.get("_parameters", None)
            if params is None:
                object.__setattr__(self, "_parameters", {})
                params = self._parameters
            # 如果之前注册为 module，先移除
            self._modules.pop(name, None)
            params[name] = value
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            modules = self.__dict__.get("_modules", None)
            if modules is None:
                object.__setattr__(self, "_modules", {})
                modules = self._modules
            self._parameters.pop(name, None)
            modules[name] = value
            object.__setattr__(self, name, value)
        else:
            # 普通 Python 对象
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # 仅在 __dict__ 中找不到时调用，避免无限递归
        # 注意：__getattr__ 不会在正常属性查找时被调用
        _parameters = self.__dict__.get("_parameters", {})
        if name in _parameters:
            return _parameters[name]
        _modules = self.__dict__.get("_modules", {})
        if name in _modules:
            return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def parameters(self):
        """递归生成所有参数（requires_grad=True 的 Tensor）。"""
        for name, p in self._parameters.items():
            if p.requires_grad:
                yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self):
        seen = set()
        for name, p in self._parameters.items():
            if p.requires_grad:
                key = (id(self), name)
                if key not in seen:
                    seen.add(key)
                    yield name, p
        for mname, m in self._modules.items():
            for sub_name, p in m.named_parameters():
                yield f"{mname}.{sub_name}", p

    def modules(self):
        """递归生成所有子模块（包括 self）。"""
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_modules(self, prefix: str = ""):
        yield prefix if prefix else "", self
        for name, m in self._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from m.named_modules(sub_prefix)

    def children(self):
        """直接子模块（不递归）。"""
        return iter(self._modules.values())

    def zero_grad(self):
        """清空所有参数的 grad。"""
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> dict:
        """返回参数字典：{name: Tensor}。"""
        sd = {}
        for name, p in self._parameters.items():
            sd[name] = p.data.copy()
        for mname, m in self._modules.items():
            m_sd = m.state_dict()
            for k, v in m_sd.items():
                sd[f"{mname}.{k}"] = v
        return sd

    def load_state_dict(self, sd: dict, strict: bool = True):
        """加载参数。"""
        own_sd = dict(self.named_parameters_dict())
        if strict:
            missing = set(own_sd.keys()) - set(sd.keys())
            extra = set(sd.keys()) - set(own_sd.keys())
            if missing:
                raise KeyError(f"Missing keys: {missing}")
            if extra:
                raise KeyError(f"Unexpected keys: {extra}")
        for name, p in self.named_parameters_with_module():
            if name in sd:
                # 直接替换 data
                p.data = np.asarray(sd[name], dtype=p.data.dtype)
        return self

    def named_parameters_dict(self):
        """返回 {name: data ndarray}（用于 state_dict 对比）。"""
        d = {}
        for name, p in self.named_parameters():
            d[name] = p.data
        return d

    def named_parameters_with_module(self):
        """返回 {name: Tensor}，递归遍历。"""
        for name, p in self._parameters.items():
            if p.requires_grad:
                yield name, p
        for mname, m in self._modules.items():
            for sub_name, p in m.named_parameters_with_module():
                yield f"{mname}.{sub_name}", p

    def train(self, mode: bool = True):
        """设置 training 模式（影响 Dropout 等）。"""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        """切换到评估模式。"""
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward().")

    def to(self, dtype=None):
        """类型转换（保持 API 兼容；CPU-only 故 device 忽略）。"""
        if dtype is not None:
            for p in self._parameters.values():
                p.data = p.data.astype(dtype)
            for m in self._modules.values():
                m.to(dtype)
        return self

    def apply(self, fn):
        """对 self 和所有子模块应用 fn。"""
        fn(self)
        for m in self._modules.values():
            m.apply(fn)
        return self


# ---------------------------------------------------------------------------
# 核心层 (Task 1.8)
# ---------------------------------------------------------------------------


class Linear(Module):
    """全连接层: y = x @ W.T + b

    Args:
        in_features: 输入维度
        out_features: 输出维度
        bias: 是否使用偏置
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重 shape (out, in)
        weight = Tensor.empty(out_features, in_features, requires_grad=True)
        # Kaiming uniform 初始化（fan_in，适合 relu）
        kaiming_uniform_(weight, a=np.sqrt(5.0))
        self.weight = weight
        if bias:
            b = Tensor.empty(out_features, requires_grad=True)
            # bias 初始化：bound = 1/sqrt(fan_in)
            fan_in = in_features
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            with no_grad():
                b.data = np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
            self.bias = b
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) -> (..., out_features)
        # W: (out, in)，所以 W.T: (in, out)，x @ W.T: (..., out)
        out = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            # bias shape (out,)，broadcast 到 (..., out)
            out = out + self.bias
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
    """Token 嵌入层: y = weight[indices]

    Args:
        num_embeddings: 词表大小
        embedding_dim: 嵌入维度
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = Tensor.empty(num_embeddings, embedding_dim, requires_grad=True)
        # 标准正态初始化
        normal_(w, mean=0.0, std=1.0)
        self.weight = w

    def forward(self, indices) -> Tensor:
        # indices 可以是 int, list, np.ndarray 或 Tensor
        if isinstance(indices, Tensor):
            idx = indices.data
        else:
            idx = np.asarray(indices)
        # 用 __getitem__ 实现可微索引
        return self.weight[idx]


class LayerNorm(Module):
    """Layer Normalization.

    Args:
        normalized_shape: int 或 (int,)，归一化的最后若干维度
        eps: 数值稳定常数
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        # gamma 与 beta 形状与 normalized_shape 一致
        gamma = Tensor.ones(*self.normalized_shape, requires_grad=True)
        beta = Tensor.zeros(*self.normalized_shape, requires_grad=True)
        self.weight = gamma
        self.bias = beta

    def forward(self, x: Tensor) -> Tensor:
        # 沿最后 len(normalized_shape) 个轴做归一化
        ndim_norm = len(self.normalized_shape)
        dims = tuple(range(x.ndim - ndim_norm, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        # var 用 unbiased=False
        diff = x - mean
        var = (diff * diff).mean(dim=dims, keepdim=True)
        std_norm = (var + self.eps).sqrt()
        normed = diff / std_norm
        return normed * self.weight + self.bias


class RMSNorm(Module):
    """Root Mean Square Normalization.

    与 LayerNorm 区别：不减均值，用 RMS = sqrt(mean(x^2)) 作为分母。

    Args:
        normalized_shape: int 或 (int,)，归一化的最后若干维度
        eps: 数值稳定常数
    """

    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        gamma = Tensor.ones(*self.normalized_shape, requires_grad=True)
        self.weight = gamma

    def forward(self, x: Tensor) -> Tensor:
        ndim_norm = len(self.normalized_shape)
        dims = tuple(range(x.ndim - ndim_norm, x.ndim))
        # ms = mean(x^2)
        ms = (x * x).mean(dim=dims, keepdim=True)
        rms = (ms + self.eps).sqrt()
        return x / rms * self.weight


class Dropout(Module):
    """Dropout: 训练时按概率 p 随机置零，并按 1/(1-p) 缩放以保持期望。

    Args:
        p: dropout 概率，0 <= p < 1
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        # 生成 mask：每个元素以 (1-p) 概率保留
        mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        # mask 是 numpy，乘以 Tensor 得 Tensor；但 mask 不需要 grad
        # 用 Tensor 包装但 requires_grad=False
        mask_tensor = Tensor(mask, requires_grad=False)
        return x * mask_tensor * scale


class Sequential(Module):
    """顺序容器：按顺序调用子模块。"""

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)
        self._n = len(modules)

    def forward(self, x):
        for i in range(self._n):
            x = getattr(self, str(i))(x)
        return x

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return getattr(self, str(i))


class ModuleList(Module):
    """可索引的模块列表。"""

    def __init__(self, modules=None):
        super().__init__()
        if modules is None:
            modules = []
        for i, m in enumerate(modules):
            setattr(self, str(i), m)
        self._n = len(modules)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ModuleList([getattr(self, str(j)) for j in range(*i.indices(self._n))])
        if i < 0:
            i += self._n
        return getattr(self, str(i))

    def __iter__(self):
        for i in range(self._n):
            yield getattr(self, str(i))

    def append(self, m):
        setattr(self, str(self._n), m)
        self._n += 1
        return self

    def forward(self, x):
        # ModuleList 默认不实现 forward，需用户在外部遍历
        raise NotImplementedError("ModuleList does not implement forward(); iterate over it manually.")
