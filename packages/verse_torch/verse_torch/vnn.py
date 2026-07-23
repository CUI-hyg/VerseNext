"""VerseTorch.vnn（Part5K1：原 verse_torch.nn 重命名）。

设计参考 PyTorch `torch.nn`：
- Module 自动注册子模块与参数（通过 __setattr__ 拦截）。
- parameters() / state_dict() / load_state_dict() 支持模型持久化。
- train()/eval() 控制 training 模式（影响 Dropout、BatchNorm 等）。
- 核心层：Linear、Embedding、LayerNorm、RMSNorm、Dropout、Sequential、ModuleList。
- 初始化辅助：kaiming_uniform_、xavier_uniform_、normal_、zeros_、ones_。

Part5K1 BREAKING：本模块由 ``verse_torch.nn`` 重命名为 ``verse_torch.vnn``。
旧的 ``verse_torch.nn`` 路径仍作为薄壳保留以向后兼容，但 transformer 系
公开别名（``TransformerLM`` / ``TransformerBlock`` / ``GQASelfAttention``）
已移除——请改用 ``verse_nex`` 品牌入口或本模块的私有实现（``_`` 前缀）。
"""

from __future__ import annotations

import warnings

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
        """自动注册 Tensor 参数与 Module 子模块（普通属性走 object.__setattr__）。

        Part5K1 SubTask 2.3 去壳简化：原 if/elif/else 三分支改为单层 isinstance
        判断 + 早返回，逻辑等价但更清晰。

        注册规则（与 PyTorch 风格一致，但更宽松：所有 Tensor 都视为参数）：
        - ``Tensor``: 注册到 ``_parameters``，并从 ``_modules`` 移除同名旧值
        - ``Module``: 注册到 ``_modules``，并从 ``_parameters`` 移除同名旧值
        - 其他: 从 ``_parameters`` / ``_modules`` 移除同名旧值后走普通赋值

        所有分支都同步 ``object.__setattr__`` 以保证 ``self.X`` 直接访问、
        pickle 序列化等场景可用。``_parameters`` / ``_modules`` 字典在首次
        访问时惰性创建（兼容未调用 ``super().__init__()`` 的边缘场景）。
        """
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        # Tensor -> _parameters（无论 requires_grad，统一注册；兼容 Embedding 等冻结权重）
        if isinstance(value, Tensor):
            if params is None:
                params = {}
                object.__setattr__(self, "_parameters", params)
            if modules is not None:
                modules.pop(name, None)
            params[name] = value
            object.__setattr__(self, name, value)
            return
        # Module -> _modules
        if isinstance(value, Module):
            if modules is None:
                modules = {}
                object.__setattr__(self, "_modules", modules)
            if params is not None:
                params.pop(name, None)
            modules[name] = value
            object.__setattr__(self, name, value)
            return
        # 普通属性：清理同名旧注册后走默认赋值
        if params is not None:
            params.pop(name, None)
        if modules is not None:
            modules.pop(name, None)
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

    @property
    def device(self) -> str:
        """返回模块参数所在设备字符串。

        优先用第一个参数的 device；若没有任何参数则返回 ``"cpu"``。
        子模块的设备可能与父模块不同（极端情况下），但 ``.to(device)``
        会把所有子模块一并迁移。
        """
        for p in self._parameters.values():
            return p.device
        for m in self._modules.values():
            return m.device
        return "cpu"

    def to(self, device=None, dtype=None):
        """迁移模块参数到指定 device（可选同时转换 dtype）。

        - ``device="cpu"`` 或 ``None``：把所有 Tensor 参数转回 ndarray 路径。
        - ``device`` 为 GPU/NPU：把所有参数迁移到 torch.Tensor 路径。
        - ``dtype`` 非 None 时同时做类型转换。

        遍历所有参数（``_parameters``）与子模块（``_modules``），
        用 ``Tensor.to`` 替换原参数对象（保持 requires_grad）。

        Args:
            device: 目标设备字符串（``"cpu"`` / ``"cuda"`` / ``"npu"`` / ...）。
            dtype: 可选 dtype。

        Returns:
            self（链式调用）。
        """
        # 兼容旧 API：to(np.float32) 当第一个位置参数是 numpy dtype/type 时
        # 把它当作 dtype 而非 device 处理
        if device is not None and not isinstance(device, str):
            # np.float32 / np.float64 是 type；np.dtype(...) 是 np.dtype 实例
            if isinstance(device, np.dtype) or (isinstance(device, type) and issubclass(device, np.generic)):
                dtype = device
                device = None
        if device is None and dtype is not None and not isinstance(dtype, str):
            # 旧式 to(np.float32) 调用：仅做类型转换
            for name, p in self._parameters.items():
                new_p = p.to("cpu", dtype=dtype)
                # 保持引用一致（用 setattr 重新注册）
                setattr(self, name, new_p)
            for m in self._modules.values():
                m.to(dtype=dtype)
            return self

        target_dev = str(device) if device is not None else "cpu"
        # 迁移参数
        for name, p in list(self._parameters.items()):
            new_p = p.to(target_dev, dtype=dtype) if dtype is not None else p.to(target_dev)
            # 重新注册（保持名称与 requires_grad）
            # 直接覆盖 _parameters 字典 + 实例属性
            self._parameters[name] = new_p
            object.__setattr__(self, name, new_p)
        # 递归子模块
        for m in self._modules.values():
            m.to(device=target_dev, dtype=dtype)
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


# ---------------------------------------------------------------------------
# Part5K1 SubTask 2.4: 公共归一化内核
# ---------------------------------------------------------------------------


def _normalize_kernel(x, weight, bias, eps, mean_centered, fast_path=False):
    """公共归一化内核：``LayerNorm`` / ``RMSNorm`` / ``LayerNormFast`` 共用。

    把三个 Norm 类 forward 中重复的"归一化 + 仿射"逻辑提取到一处，消除重复代码。
    数值与原实现完全一致（autograd 路径走 Tensor 算子图，fast 路径走 numpy + 手动反向）。

    Args:
        x: 输入 ``Tensor``。
        weight: 仿射权重 ``Tensor``（gamma），shape 等于 ``normalized_shape``。
        bias: 仿射偏置 ``Tensor``（beta）；``RMSNorm`` 无 bias 时传 ``None``。
        eps: 数值稳定常数。
        mean_centered: ``True`` 走 LayerNorm（减均值 + 除标准差 + 仿射）；
            ``False`` 走 RMSNorm（只除 RMS + 仿射，不减均值）。
        fast_path: ``True`` 用向量化 numpy 操作 + 手动反向（``LayerNormFast`` CPU 路径）；
            ``False`` 用 Tensor autograd 算子（``LayerNorm`` / ``RMSNorm`` 路径）。

    Returns:
        归一化后的 ``Tensor``（与 ``x`` 同形状）。
    """
    # 推断归一化维度：weight.shape 的维度数 == len(normalized_shape)
    ndim_norm = len(weight.shape)
    dims = tuple(range(x.ndim - ndim_norm, x.ndim))

    if fast_path:
        return _normalize_fast(x, weight, bias, eps, mean_centered, dims)

    # Tensor autograd 路径（LayerNorm / RMSNorm）：通过算子重载自动构建计算图
    if mean_centered:
        # LayerNorm: 减均值 + 除标准差
        mean = x.mean(dim=dims, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=dims, keepdim=True)
        normed = diff / (var + eps).sqrt()
    else:
        # RMSNorm: 只除 RMS（不减均值）
        ms = (x * x).mean(dim=dims, keepdim=True)
        normed = x / (ms + eps).sqrt()
    out = normed * weight
    if bias is not None:
        out = out + bias
    return out


def _normalize_fast(x, weight, bias, eps, mean_centered, dims):
    """``fast_path`` 内核：numpy 向量化 + 手动反向（``LayerNormFast`` CPU 路径）。

    与 ``_normalize_kernel`` 的 autograd 路径数值等价，但直接操作 ``ndarray``
    并手写反向梯度，避免 autograd 图的多层 dispatch 开销。

    目前仅实现 ``mean_centered=True``（对应 ``LayerNormFast``）；
    ``mean_centered=False``（RMSNorm fast）暂无调用方，如需启用需补充对应反向。
    """
    if not mean_centered:
        # RMSNorm fast 路径暂未使用（RMSNorm 走 autograd 路径即可）
        raise NotImplementedError(
            "fast_path + mean_centered=False (RMSNorm fast) 暂未实现")

    # ----- LayerNormFast CPU 路径（mean_centered=True）-----
    mean = x.data.mean(axis=dims, keepdims=True)
    var = x.data.var(axis=dims, keepdims=True)
    normed = (x.data - mean) / np.sqrt(var + eps)
    out_data = normed * weight.data
    if bias is not None:
        out_data = out_data + bias.data

    # 非归一化维度（用于 weight / bias 梯度的 reduce-sum 轴）
    reduce_dims = tuple(d for d in range(x.ndim) if d not in dims)

    def _backward():
        if x.requires_grad:
            # 标准层归一化反向：dx = (1/σ) * (g - mean(g) - x̂ · mean(g·x̂))
            N = 1
            for d in dims:
                N *= x.data.shape[d]
            x_hat = (x.data - mean) / np.sqrt(var + eps)
            g = out.grad
            g_sum = g.sum(axis=dims, keepdims=True)
            gx_hat_sum = (g * x_hat).sum(axis=dims, keepdims=True)
            grad = (g - g_sum / N - x_hat * gx_hat_sum / N) / np.sqrt(var + eps)
            x._accumulate_grad(grad)
        if weight.requires_grad:
            gw = (out.grad * normed).sum(axis=reduce_dims)
            weight._accumulate_grad(gw.reshape(weight.shape))
        if bias is not None and bias.requires_grad:
            gb = out.grad.sum(axis=reduce_dims)
            bias._accumulate_grad(gb.reshape(bias.shape))

    children = [x, weight]
    if bias is not None:
        children.append(bias)
    out = Tensor(out_data,
                 requires_grad=any(c.requires_grad for c in children),
                 _children=tuple(c for c in children if c.requires_grad),
                 _op="layernorm_fast")
    if out.requires_grad:
        out._backward = _backward
    return out


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
        # 走公共归一化内核（Tensor autograd 路径，减均值 + 除标准差 + 仿射）
        return _normalize_kernel(x, self.weight, self.bias, self.eps,
                                 mean_centered=True, fast_path=False)


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
        # 走公共归一化内核（Tensor autograd 路径，只除 RMS + 仿射，无 bias）
        return _normalize_kernel(x, self.weight, None, self.eps,
                                 mean_centered=False, fast_path=False)


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


# ---------------------------------------------------------------------------
# 多层神经网络组件 (Stage 2 Task 1.1-1.4)
# ---------------------------------------------------------------------------


def _concat(tensors, dim: int = 1):
    """可微的 Tensor 拼接函数（沿指定维度）。

    用于 KV cache 在序列维度的拼接。支持反向传播。
    GPU 路径委托 ``torch.cat``（autograd 自动构建反向）。
    """
    from .tensor import _is_torch_data, _TORCH
    # GPU 路径：任一输入是 torch.Tensor 则委托 torch.cat
    if _TORCH is not None and any(_is_torch_data(t.data) for t in tensors):
        # 对齐 device：以第一个 torch 输入为基准
        ref = next(t for t in tensors if _is_torch_data(t.data))
        ref_dev = ref.data.device
        torch_tensors = []
        for t in tensors:
            if _is_torch_data(t.data):
                torch_tensors.append(t.data.to(ref_dev) if t.data.device != ref_dev else t.data)
            else:
                torch_tensors.append(_TORCH.from_numpy(np.ascontiguousarray(t.data)).to(ref_dev))
        out_data = _TORCH.cat(torch_tensors, dim=dim)
        requires_grad = _GRAD_ENABLED and any(t.requires_grad for t in tensors)
        children = tuple(t for t in tensors if t.requires_grad)
        return Tensor(out_data, requires_grad=requires_grad,
                      _children=children if requires_grad else (), _op="concat",
                      device=str(out_data.device))
    # CPU 路径（原有实现，自研 autograd）
    datas = [t.data for t in tensors]
    out_data = np.concatenate(datas, axis=dim)
    requires_grad = _GRAD_ENABLED and any(t.requires_grad for t in tensors)
    children = tuple(t for t in tensors if t.requires_grad)
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=children if requires_grad else (), _op="concat")
    if requires_grad:
        sizes = [t.shape[dim] for t in tensors]
        starts = [0]
        for s in sizes[:-1]:
            starts.append(starts[-1] + s)

        def _backward():
            grad = out.grad
            for t, start, size in zip(tensors, starts, sizes):
                if t.requires_grad:
                    idx = [slice(None)] * grad.ndim
                    idx[dim] = slice(start, start + size)
                    t._accumulate_grad(grad[tuple(idx)])

        out._backward = _backward
    return out


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """GQA 工具函数：在 head 维度（axis=2）上重复 n_rep 次。

    输入: (B, T, n_kv_head, head_dim)
    输出: (B, T, n_kv_head * n_rep, head_dim)

    每个 kv head 重复 n_rep 次相邻（与 HuggingFace repeat_kv 行为一致）。
    """
    if n_rep == 1:
        return x
    B, T, n_kv, D = x.shape
    out_data = np.repeat(x.data, n_rep, axis=2)
    requires_grad = _GRAD_ENABLED and x.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=(x,) if requires_grad else (), _op="repeat_kv")
    if requires_grad:
        def _backward():
            # 反向：将 n_rep 个相邻 head 的梯度求和
            grad = out.grad  # (B, T, n_kv * n_rep, D)
            grad_reshaped = grad.reshape(B, T, n_kv, n_rep, D)
            g = grad_reshaped.sum(axis=3)  # (B, T, n_kv, D)
            x._accumulate_grad(g)
        out._backward = _backward
    return out


class SwiGLUMLP(Module):
    """SwiGLU MLP: dropout(w_down( silu(w_gate(x)) * w_up(x) ))

    Args:
        d: 输入/输出维度
        dropout: dropout 概率
        hidden_multiple: 隐藏层维度倍数（默认 4）
        align: 隐藏层维度对齐（默认 64）
    """

    def __init__(self, d: int, dropout: float = 0.0, hidden_multiple: int = 4, align: int = 64):
        super().__init__()
        self.d = d
        self.align = align
        # hidden 计算：2/3 缩放后向上对齐到 align 的倍数
        hidden = int((hidden_multiple * d * 2 / 3 + align - 1) // align) * align
        self.hidden = hidden
        self.w_gate = Linear(d, hidden, bias=False)
        self.w_up = Linear(d, hidden, bias=False)
        self.w_down = Linear(hidden, d, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x).silu()
        up = self.w_up(x)
        h = gate * up
        h = self.w_down(h)
        h = self.dropout(h)
        return h


class _GQASelfAttention(Module):
    """Grouped Query Attention with RoPE and KV cache.

    Args:
        d: 模型维度
        n_head: query head 数量
        n_kv_head: key/value head 数量（默认 = n_head，即标准 MHA）
        dropout: dropout 概率
    """

    def __init__(self, d: int, n_head: int, n_kv_head: int = None, dropout: float = 0.0):
        super().__init__()
        if n_kv_head is None:
            n_kv_head = n_head
        assert d % n_head == 0, f"d({d}) 必须能被 n_head({n_head}) 整除"
        assert n_head % n_kv_head == 0, f"n_head({n_head}) 必须能被 n_kv_head({n_kv_head}) 整除"
        self.d = d
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d // n_head
        self.n_rep = n_head // n_kv_head
        # 四个投影矩阵，均 bias=False
        # wq: d → n_head * head_dim = d
        # wk/wv: d → n_kv_head * head_dim（GQA 下小于 d）
        # proj: n_head * head_dim = d → d
        kv_dim = n_kv_head * self.head_dim
        self.wq = Linear(d, n_head * self.head_dim, bias=False)
        self.wk = Linear(d, kv_dim, bias=False)
        self.wv = Linear(d, kv_dim, bias=False)
        self.proj = Linear(n_head * self.head_dim, d, bias=False)
        self.dropout = Dropout(dropout)
        # RoPE 预计算 cos/sin 表（避免依赖 verse_nex，自带实现以支持 position_offset）
        self._build_rope_table(self.head_dim, 32768)

    def _build_rope_table(self, head_dim: int, max_seq_len: int):
        half = head_dim // 2
        i = np.arange(half, dtype=np.float32)
        inv_freq = 1.0 / (10000.0 ** (2.0 * i / head_dim))
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions, inv_freq)  # (T, half)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (T, head_dim)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
        self._cos_table = cos
        self._sin_table = sin
        self._max_seq_len = max_seq_len

    def _apply_rope(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """对 x 应用 RoPE。

        Args:
            x: Tensor shape (B, T, H, D)
            position_offset: 位置偏移（用于 KV cache 场景下新 token 的起始位置）
        """
        B, T, H, D = x.shape
        if position_offset + T > self._max_seq_len:
            new_max = max(self._max_seq_len * 2, position_offset + T)
            self._build_rope_table(D, new_max)
        pos = position_offset + np.arange(T)
        cos = self._cos_table[pos]  # (T, D)
        sin = self._sin_table[pos]
        cos_b = cos.reshape(1, T, 1, D)
        sin_b = sin.reshape(1, T, 1, D)
        x_data = x.data
        half = D // 2
        # rotate_half(x) = concat(-x[half:], x[:half])
        rotate_half = np.concatenate([-x_data[..., half:], x_data[..., :half]], axis=-1)
        rotated = x_data * cos_b + rotate_half * sin_b

        requires_grad = _GRAD_ENABLED and x.requires_grad
        out = Tensor(rotated, requires_grad=requires_grad,
                     _children=(x,) if requires_grad else (), _op="rope")
        if requires_grad:
            def _backward():
                # dx = grad * cos + rotate_half(grad) * sin
                grad = out.grad
                g = grad * cos_b + np.concatenate(
                    [-grad[..., half:], grad[..., :half]], axis=-1
                ) * sin_b
                x._accumulate_grad(g)
            out._backward = _backward
        return out

    def forward(self, x: Tensor, kv_cache=None):
        B, T, d = x.shape
        # 1. 投影 q/k/v，shape (B, T, d)
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # 2. 计算 position_offset（KV cache 场景下，新 token 的起始位置 = cache 长度）
        position_offset = 0
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            position_offset = k_cache.shape[1]

        # 3. 应用 RoPE（仅 q, k；v 不应用）
        q = self._apply_rope(q, position_offset)
        k = self._apply_rope(k, position_offset)

        # 4. KV cache 拼接前缀
        if kv_cache is not None:
            k = _concat([k_cache, k], dim=1)
            v = _concat([v_cache, v], dim=1)
        # detach 后存入新 cache，避免梯度跨越 step 传播
        new_kv_cache = (k.detach(), v.detach())

        # 5. repeat_kv：将 kv head 重复 n_rep 次匹配 q head 数量
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 6. 转置为 (B, n_head, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 7. attention 计算
        # einsum 优化评估（Task 10.3）：
        #   QK^T 与 attn@V 均为 4D 批量矩阵乘法 (B, n_head, T, head_dim)，
        #   Tensor.__matmul__ 内部调用 np.matmul，已直接走 BLAS 批量 GEMM 路径，
        #   性能优于 np.einsum（einsum 多一层 contraction path 解释开销，
        #   且对标准 batched matmul 不会找到比 BLAS 更优的路径）。
        #   故此处保留 matmul 实现，不强行改 einsum。
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(-1, -2)) * scale  # (B, n_head, T_q, T_k)
        # causal mask: mask[i, j] = 0 if j <= i + offset else -1e9
        # offset = T_k - T_q（KV cache 场景下 q 只对应最后 T_q 个位置）
        T_q = q.shape[2]
        T_k = k.shape[2]
        offset = T_k - T_q
        i_idx = np.arange(T_q)[:, None]
        j_idx = np.arange(T_k)[None, :]
        mask_2d = np.where(j_idx <= i_idx + offset, 0.0, -1e9).astype(np.float32)
        mask = Tensor(mask_2d.reshape(1, 1, T_q, T_k), requires_grad=False)
        scores = scores + mask
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, n_head, T_q, head_dim)

        # 8. reshape 回 (B, T_q, d) 并投影
        out = out.transpose(1, 2).reshape(B, T_q, d)
        out = self.proj(out)
        return out, new_kv_cache


class _TransformerBlock(Module):
    """Pre-norm Transformer block.

    结构:
        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, d: int, n_head: int, n_kv_head: int = None, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = _GQASelfAttention(d, n_head, n_kv_head, dropout)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLUMLP(d, dropout)

    def forward(self, x: Tensor, kv_cache=None):
        attn_out, new_kv_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv_cache


class _TransformerLM(Module):
    """Transformer Language Model.

    结构: tok_emb → N × TransformerBlock → RMSNorm → head

    Args:
        vocab_size: 词表大小
        n_layer: Transformer block 层数
        n_head: attention head 数量
        n_embd: 嵌入维度
        seq_len: 最大序列长度
        dropout: dropout 概率
        n_kv_head: kv head 数量（None 表示 = n_head）
        tie_weights: 是否共享 tok_emb 与 head 的权重
    """

    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int,
                 seq_len: int = 128, dropout: float = 0.1,
                 n_kv_head: int = None, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.tie_weights = tie_weights

        self.tok_emb = Embedding(vocab_size, n_embd)
        self.blocks = ModuleList([
            _TransformerBlock(n_embd, n_head, n_kv_head, dropout)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embd)
        self.head = Linear(n_embd, vocab_size, bias=False)

        if tie_weights:
            # head.weight 与 tok_emb.weight 共享同一个 Tensor 对象
            self.head.weight = self.tok_emb.weight
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        # Linear / Embedding 用 normal_(std=0.02)
        for m in self.modules():
            if isinstance(m, Linear):
                normal_(m.weight, std=0.02)
                if m.bias is not None:
                    normal_(m.bias, std=0.02)
            elif isinstance(m, Embedding):
                normal_(m.weight, std=0.02)
        # 残差分支缩放：1/sqrt(2*n_layer)，保证训练稳定性
        scale = 1.0 / ((2 * self.n_layer) ** 0.5)
        for block in self.blocks:
            with no_grad():
                block.attn.proj.weight.data = (
                    block.attn.proj.weight.data * scale
                ).astype(np.float32)
                block.mlp.w_down.weight.data = (
                    block.mlp.w_down.weight.data * scale
                ).astype(np.float32)

    def forward(self, idx) -> Tensor:
        # idx: (B, T) int → emb (B, T, n_embd) → blocks → norm → head → logits (B, T, vocab)
        x = self.tok_emb(idx)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# 扩展模块 (Task 3.4): SlidingWindowAttention / ALiBi / DeepNorm
# ---------------------------------------------------------------------------


class SlidingWindowAttention(Module):
    """滑动窗口注意力（长上下文场景）。

    与 ``GQASelfAttention`` 类似，但 attention 矩阵只计算每个 query 对前 ``window_size`` 个 key，
    超出窗口的位置被 mask 为 ``-inf``。配合 causal mask 使用时，
    ``mask[i, j] = 0`` 当且仅当 ``j <= i`` 且 ``i - j < window_size``。

    论文参考: Longformer / Mistral（滑动窗口注意力）。

    Args:
        n_embd: 模型维度
        n_head: query head 数量
        window_size: 滑动窗口大小（每个 query 最多 attend 前 window_size 个 key）
        n_kv_head: key/value head 数量（默认 = n_head，即标准 MHA；小于 n_head 时为 GQA）
        dropout: dropout 概率
    """

    def __init__(self, n_embd: int, n_head: int, window_size: int,
                 n_kv_head: int = None, dropout: float = 0.1):
        super().__init__()
        if n_kv_head is None:
            n_kv_head = n_head
        assert n_embd % n_head == 0, (
            f"n_embd({n_embd}) 必须能被 n_head({n_head}) 整除"
        )
        assert n_head % n_kv_head == 0, (
            f"n_head({n_head}) 必须能被 n_kv_head({n_kv_head}) 整除"
        )
        assert window_size >= 1, f"window_size 必须为正整数，got {window_size}"
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.window_size = window_size
        self.head_dim = n_embd // n_head
        self.n_rep = n_head // n_kv_head
        kv_dim = n_kv_head * self.head_dim
        # 四个投影矩阵，均 bias=False（与 GQASelfAttention 一致）
        self.wq = Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = Linear(n_embd, kv_dim, bias=False)
        self.wv = Linear(n_embd, kv_dim, bias=False)
        self.proj = Linear(n_head * self.head_dim, n_embd, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor):
        """前向计算。

        Args:
            x: (B, T, n_embd)

        Returns:
            (B, T, n_embd) 的注意力输出
        """
        B, T, d = x.shape
        # 1. 投影 q/k/v，shape (B, T, n_*head, head_dim)
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # 2. repeat_kv：将 kv head 重复 n_rep 次匹配 q head 数量
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 3. 转置为 (B, n_head, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 4. attention scores: (B, n_head, T, T)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(-1, -2)) * scale

        # 5. 滑动窗口 + causal mask
        # mask[i, j] = 0 if (j <= i) and (i - j < window_size) else -inf
        i_idx = np.arange(T)[:, None]
        j_idx = np.arange(T)[None, :]
        in_window = (i_idx - j_idx) < self.window_size
        causal = j_idx <= i_idx
        mask_2d = np.where(in_window & causal, 0.0, -1e9).astype(np.float32)
        mask = Tensor(mask_2d.reshape(1, 1, T, T), requires_grad=False)
        scores = scores + mask

        # 6. softmax + dropout + 加权求和
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, n_head, T, head_dim)

        # 7. reshape 回 (B, T, n_embd) 并投影
        out = out.transpose(1, 2).reshape(B, T, d)
        out = self.proj(out)
        return out


class ALiBi(Module):
    """ALiBi (Attention with Linear Biases) 位置偏置。

    论文: https://arxiv.org/abs/2108.12409

    不学习位置嵌入，直接在 attention scores 上加线性偏置::

        bias[i, j] = -m_h * (i - j)   if i >= j  (causal)
        bias[i, j] = -inf             otherwise

    其中 ``m_h`` 是 head h 的斜率，按几何级数生成：``m_h = 1 / 2^(h / n_head)``，h = 1, ..., n_head。

    用法: 在 attention 计算中，对 ``qk_scores`` 调用 ``alibi(qk_scores)`` 加偏置后再 softmax。

    Args:
        n_head: head 数量
        max_seq_len: 预计算 bias 表的最大序列长度（默认 2048）
    """

    def __init__(self, n_head: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        # 斜率：几何级数 m_h = 1 / 2^(h/n_head)，h=1..n_head
        # 这样 m_1 = 1/2^(1/n_head) 接近 1，m_n = 1/2^1 = 0.5
        slopes = 1.0 / (2.0 ** (np.arange(1, n_head + 1, dtype=np.float64) / n_head))
        self.slopes = slopes.astype(np.float32)  # (n_head,)
        # 预计算 bias 表
        self._build_bias_table(max_seq_len)

    def _build_bias_table(self, max_seq_len: int):
        """预计算 (n_head, T, T) 的 bias 表。

        bias[h, i, j] = -slopes[h] * (i - j)  if j <= i  else -1e9
        """
        i_idx = np.arange(max_seq_len, dtype=np.float32)[:, None]  # (T, 1)
        j_idx = np.arange(max_seq_len, dtype=np.float32)[None, :]  # (1, T)
        dist = i_idx - j_idx  # (T, T)，下三角为正
        causal = (j_idx <= i_idx).astype(np.float32)  # (T, T)
        # 构造 (n_head, T, T) bias
        bias = np.zeros((self.n_head, max_seq_len, max_seq_len), dtype=np.float32)
        for h in range(self.n_head):
            bias[h] = np.where(
                causal > 0,
                -self.slopes[h] * dist,
                np.float32(-1e9),  # 大负数近似 -inf（softmax 后为 0）
            )
        self._bias_table = bias

    def forward(self, qk_scores: Tensor) -> Tensor:
        """对 qk_scores 加 ALiBi 偏置。

        Args:
            qk_scores: (B, n_head, T_q, T_k) 的 attention 分数

        Returns:
            加偏置后的 scores（同形状）
        """
        B, H, T_q, T_k = qk_scores.shape
        # 取对应大小的 bias（支持 T_q != T_k 的 KV cache 场景）
        bias = self._bias_table[:, :T_q, :T_k]  # (H, T_q, T_k)
        # 广播到 (1, H, T_q, T_k) 与 qk_scores 相加
        bias_t = Tensor(bias.reshape(1, H, T_q, T_k), requires_grad=False)
        return qk_scores + bias_t


class DeepNorm(Module):
    """DeepNorm: 深层 Transformer 用的归一化层。

    论文: https://arxiv.org/abs/2203.00555

    ``DeepNorm(x) = LayerNorm(x * alpha) + x``

    其中 ``alpha`` 是残差分支的缩放系数，通常 ``alpha = (2 * N)^(1/4)``，N 是 Transformer 层数。
    ``alpha`` 越大，残差分支的权重越大，训练越稳定（可训练上千层）。

    Args:
        normalized_shape: LayerNorm 的归一化形状
        alpha: 残差缩放系数（默认 1.0，等价于普通 LayerNorm + 残差）
        eps: LayerNorm 的数值稳定常数
    """

    def __init__(self, normalized_shape, alpha: float = 1.0, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.alpha = float(alpha)
        self.eps = eps
        # 复用 LayerNorm 的 gamma / beta 参数
        self.layernorm = LayerNorm(self.normalized_shape, eps=self.eps)

    def forward(self, x: Tensor) -> Tensor:
        # DeepNorm(x) = LayerNorm(x * alpha) + x
        return self.layernorm(x * self.alpha) + x


# ---------------------------------------------------------------------------
# Task 1.4: 新增组件（RotaryEmbedding / KVCache / GroupNorm / Conv1d / LayerNormFast）
# ---------------------------------------------------------------------------


class RotaryEmbedding(Module):
    """Rotary Position Embedding (RoPE) 独立类。

    实现 GPT-NeoX 风格的 rotate_half RoPE：
    ``x_rotated = x * cos + rotate_half(x) * sin``，其中 ``rotate_half`` 把
    最后一维拆成两半并取负交换。

    Args:
        dim: RoPE 作用的维度（通常等于 head_dim）
        max_seq_len: 预计算 cos/sin 表的最大序列长度
        base: 频率基数（默认 10000，与 LLaMA 一致）
        scaling: 可选的长度外推缩放因子（如 ``1 / rope_theta_scale``）
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0,
                 scaling: float = 1.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim 必须为偶数，got {dim}")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        self.scaling = float(scaling)
        # 预计算 inv_freq: (dim/2,)
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))
        # 序列位置（应用 scaling）
        t = np.arange(self.max_seq_len, dtype=np.float32) * self.scaling  # (T,)
        # freqs: (T, dim/2)
        freqs = np.outer(t, inv_freq)
        # cos/sin: (T, dim)，把每对频率复制一份（与 rotate_half 对齐）
        emb = np.concatenate([freqs, freqs], axis=-1)  # (T, dim)
        cos_np = np.cos(emb)
        sin_np = np.sin(emb)
        # 作为非梯度 buffer 存储（用 Tensor 但 requires_grad=False）
        # 通过 _parameters 注册以保证 Module.to(device) 能迁移
        self.cos = Tensor(cos_np, requires_grad=False)
        self.sin = Tensor(sin_np, requires_grad=False)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """GPT-NeoX rotate_half：把最后一维拆成两半，前半取负拼到后半。"""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        # 等价于 concat([-x2, x1])
        # 用现有算子实现，CPU/GPU 路径均自动分流
        return _concat([-x2, x1], dim=-1)

    def forward(self, x: Tensor, seq_len: int = None) -> Tensor:
        """对 ``x`` 应用 RoPE。

        Args:
            x: 形状 ``(..., T, dim)`` 的张量。
            seq_len: 可选，指定取 cos/sin 表前 ``seq_len`` 个位置；
                ``None`` 时取 ``x.shape[-2]``。

        Returns:
            旋转后的张量，形状与 ``x`` 相同。
        """
        T = x.shape[-2] if seq_len is None else int(seq_len)
        # cos/sin: (T, dim) -> 广播到 (..., T, dim)
        cos = self.cos[:T]  # (T, dim)
        sin = self.sin[:T]
        # 调整 cos/sin 的 ndim 与 x 对齐（前面补 1）
        # x.ndim 可能是 3 (B, T, D) 或 4 (B, H, T, D)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        # x * cos + rotate_half(x) * sin
        return x * cos + self.rotate_half(x) * sin


# ---------------------------------------------------------------------------
# KVCache 抽象与实现
# ---------------------------------------------------------------------------


class KVCache:
    """KV cache 抽象基类。

    推理时缓存每一层的 Key / Value，避免对历史 token 重复计算。
    不同实现（静态 / 动态）有不同的内存与扩展策略。

    子类应实现：
    - ``update(key, value, layer_idx)``：写入新一层的 K/V，返回更新后的 (K, V)
    - ``batch_update(keys, values, layer_idx)``：并行预测时批量更新 K/V
        （Part4K1 Task 3.3，speculative decoding 风格）
    - ``get(layer_idx)``：取出指定层的 (K, V)
    - ``reset()``：清空 cache
    - ``to(device)``：迁移到指定设备
    """

    def __init__(self, num_layers: int = 1):
        self.num_layers = int(num_layers)

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0):
        """写入新一层的 K/V，返回更新后的 (K, V)。"""
        raise NotImplementedError

    def batch_update(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int = 0,
    ):
        """并行批量更新 KV cache（Part4K1 Task 3.3）。

        用于 speculative decoding 风格的并行预测场景：一次前向预测 k 个
        token，把对应的 K/V 一次性写入 cache，避免 k 次串行 update。

        默认实现委托给 ``update``（子类可覆盖以提供更高效的批量实现）。

        Args:
            keys: (B, T_new, H, D) 新 K，T_new 通常 = 候选 token 数 k
            values: (B, T_new, H, D) 新 V
            layer_idx: 层索引
        Returns:
            更新后的 (K, V)
        """
        return self.update(keys, values, layer_idx=layer_idx)

    def get(self, layer_idx: int = 0):
        """取出指定层的 (K, V)。"""
        raise NotImplementedError

    def reset(self) -> None:
        """清空 cache。"""
        raise NotImplementedError

    @property
    def device(self) -> str:
        """返回 cache 当前所在设备（默认 cpu）。"""
        return "cpu"

    def to(self, device) -> "KVCache":
        """迁移 cache 到指定 device（子类实现）。"""
        raise NotImplementedError


class StaticCache(KVCache):
    """静态长度 KV cache。

    预分配固定大小的 ``[max_batch, max_seq, num_heads, head_dim]`` 缓冲区，
    每次更新把新 K/V 写入指定位置（in-place）。适合批处理推理与
    prefill / decode 分离场景。

    Args:
        num_layers: 层数
        max_batch: 最大 batch size
        max_seq: 最大序列长度
        num_heads: 头数
        head_dim: 每头维度
        dtype: 缓冲区 dtype（默认 float32）
    """

    def __init__(self, num_layers: int, max_batch: int, max_seq: int,
                 num_heads: int, head_dim: int, dtype=np.float32):
        super().__init__(num_layers=num_layers)
        self.max_batch = int(max_batch)
        self.max_seq = int(max_seq)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        # 预分配 K/V buffer: (num_layers, max_batch, max_seq, num_heads, head_dim)
        shape = (self.num_layers, self.max_batch, self.max_seq, self.num_heads, self.head_dim)
        self._k_buf = [Tensor(np.zeros(shape[1:], dtype=dtype), requires_grad=False)
                       for _ in range(self.num_layers)]
        self._v_buf = [Tensor(np.zeros(shape[1:], dtype=dtype), requires_grad=False)
                       for _ in range(self.num_layers)]
        # 记录每个 layer 已写入的序列长度
        self._seen = [0] * self.num_layers

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0):
        """写入新一层的 K/V，返回累积的 (K, V)。

        Args:
            key: (B, T_new, H, D) 新 K
            value: (B, T_new, H, D) 新 V
            layer_idx: 层索引
        """
        if layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}")
        start = self._seen[layer_idx]
        T_new = key.shape[1]
        if start + T_new > self.max_seq:
            raise RuntimeError(
                f"StaticCache 溢出：layer {layer_idx} 已存 {start}，"
                f"新增 {T_new}，超过 max_seq {self.max_seq}"
            )
        # 写入 buffer（用 __setitem__ 语义；这里简化为重建大 tensor）
        # 注意：当前 Tensor 不支持原地切片赋值，采用整体重建策略
        # 对推理场景（无梯度）影响有限
        k_old = self._k_buf[layer_idx]
        v_old = self._v_buf[layer_idx]
        # 用 numpy 拼接（或 torch 拼接，由 _concat 自动分流）
        if start == 0:
            new_k = key
            new_v = value
        else:
            # 取已缓存部分（无梯度）
            with no_grad():
                k_prev = k_old[:, :start]
                v_prev = v_old[:, :start]
                new_k = _concat([k_prev, key], dim=1)
                new_v = _concat([v_prev, value], dim=1)
        # 更新 buffer（保留 max_seq 长度，超出的部分用零填充）
        # 简化：直接保留累积结果，下次再拼
        self._k_buf[layer_idx] = new_k
        self._v_buf[layer_idx] = new_v
        self._seen[layer_idx] = start + T_new
        return new_k, new_v

    def get(self, layer_idx: int = 0):
        if layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}")
        return self._k_buf[layer_idx], self._v_buf[layer_idx]

    def reset(self) -> None:
        for i in range(self.num_layers):
            self._k_buf[i] = Tensor(np.zeros_like(self._k_buf[i].data), requires_grad=False)
            self._v_buf[i] = Tensor(np.zeros_like(self._v_buf[i].data), requires_grad=False)
            self._seen[i] = 0

    @property
    def device(self) -> str:
        if self._k_buf:
            return self._k_buf[0].device
        return "cpu"

    def to(self, device) -> "StaticCache":
        target = str(device)
        for i in range(self.num_layers):
            self._k_buf[i] = self._k_buf[i].to(target)
            self._v_buf[i] = self._v_buf[i].to(target)
        return self


class DynamicCache(KVCache):
    """动态增长 KV cache。

    每次更新把新 K/V 拼接到已有 cache 末尾（不预分配）。
    内存随序列长度线性增长；适合变长推理与生成。

    Args:
        num_layers: 层数
    """

    def __init__(self, num_layers: int = 1):
        super().__init__(num_layers=num_layers)
        self._k: list = [None] * self.num_layers
        self._v: list = [None] * self.num_layers
        self._seen = [0] * self.num_layers

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0):
        """追加新 K/V，返回累积的 (K, V)。"""
        if layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}")
        if self._k[layer_idx] is None:
            self._k[layer_idx] = key
            self._v[layer_idx] = value
        else:
            with no_grad():
                self._k[layer_idx] = _concat([self._k[layer_idx], key], dim=1)
                self._v[layer_idx] = _concat([self._v[layer_idx], value], dim=1)
        self._seen[layer_idx] = self._seen[layer_idx] + key.shape[1]
        return self._k[layer_idx], self._v[layer_idx]

    def get(self, layer_idx: int = 0):
        if layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}")
        return self._k[layer_idx], self._v[layer_idx]

    def reset(self) -> None:
        self._k = [None] * self.num_layers
        self._v = [None] * self.num_layers
        self._seen = [0] * self.num_layers

    @property
    def device(self) -> str:
        for k in self._k:
            if k is not None:
                return k.device
        return "cpu"

    def to(self, device) -> "DynamicCache":
        target = str(device)
        for i in range(self.num_layers):
            if self._k[i] is not None:
                self._k[i] = self._k[i].to(target)
            if self._v[i] is not None:
                self._v[i] = self._v[i].to(target)
        return self


# ---------------------------------------------------------------------------
# GroupNorm / Conv1d / LayerNormFast
# ---------------------------------------------------------------------------


class GroupNorm(Module):
    """Group Normalization。

    把通道维分成 ``num_groups`` 组，每组内做 mean/var 归一化（含通道与空间维）。
    与 BatchNorm 区别：归一化不依赖 batch 维，对小 batch / 序列模型更稳定。

    Args:
        num_groups: 分组数（必须能整除 num_channels）
        num_channels: 通道数
        eps: 数值稳定常数
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels({num_channels}) 必须能被 num_groups({num_groups}) 整除"
            )
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = eps
        self.weight = Tensor.ones(num_channels, requires_grad=True)
        self.bias = Tensor.zeros(num_channels, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """前向：``x`` 形状 ``(..., C, ...)``，归一化在 ``num_groups`` 组内进行。

        简化实现：假设 ``x`` 形状 ``(B, C, *)``，对 ``C`` 维分组后
        在每组 ``C/num_groups`` 个通道 + 所有空间维上做 mean/var 归一化。
        """
        # 形状: (B, C, *spatial)
        B = x.shape[0]
        C = x.shape[1]
        spatial_shape = x.shape[2:]
        # reshape: (B, num_groups, C // num_groups, *spatial)
        g = self.num_groups
        cg = C // g
        # 把 spatial 维 flatten 为一个 dim 以便 reduce
        x_r = x.reshape(B, g, cg, -1)  # (B, g, cg, L)
        # 在最后两维 (cg, L) 上求 mean/var
        mean = x_r.mean(dim=(2, 3), keepdim=True)  # (B, g, 1, 1)
        diff = x_r - mean
        var = (diff * diff).mean(dim=(2, 3), keepdim=True)
        normed = diff / ((var + self.eps).sqrt())
        # reshape 回 (B, C, *spatial)
        normed = normed.reshape(x.shape)
        # 仿射变换：weight/bias 形状 (C,)，需要广播到 (1, C, 1, ...)
        w = self.weight.reshape((1, C) + (1,) * len(spatial_shape))
        b = self.bias.reshape((1, C) + (1,) * len(spatial_shape))
        return normed * w + b


class Conv1d(Module):
    """一维卷积层。

    ``y[b, o, t] = sum_{i, c} x[b, c, t*stride + i* dilation - padding] * W[o, c, i] + b[o]``

    实现采用 im2col + matmul（CPU 路径用 NumPy，GPU 路径委托
    ``torch.nn.functional.conv1d``）。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长（默认 1）
        padding: padding 大小（默认 0）
        dilation: 膨胀系数（默认 1）
        bias: 是否使用偏置
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        # 权重 shape: (out, in, K)
        weight = Tensor.empty(out_channels, in_channels, kernel_size, requires_grad=True)
        kaiming_uniform_(weight, a=np.sqrt(5.0))
        self.weight = weight
        if bias:
            b = Tensor.empty(out_channels, requires_grad=True)
            bound = 1.0 / np.sqrt(in_channels * kernel_size) if in_channels * kernel_size > 0 else 0.0
            with no_grad():
                b.data = np.random.uniform(-bound, bound, size=(out_channels,)).astype(np.float32)
            self.bias = b
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """前向：``x`` 形状 ``(B, C_in, L)`` -> ``(B, C_out, L_out)``。"""
        # GPU 路径委托 torch.nn.functional.conv1d
        if x._is_torch_tensor or self.weight._is_torch_tensor:
            return self._forward_torch(x)
        # CPU 路径：im2col + matmul
        from .tensor import no_grad as _ng  # noqa
        B, C_in, L = x.shape
        K = self.kernel_size
        s = self.stride
        p = self.padding
        d = self.dilation
        # padding
        if p > 0:
            pad_width = ((0, 0), (0, 0), (p, p))
            x_padded = Tensor(np.pad(x.data, pad_width, mode="constant"),
                              requires_grad=x.requires_grad, _children=(x,) if x.requires_grad else (),
                              _op="pad")
            # 简化：pad 不可微在此处近似为常数 0，反向仅传给 x
            # 由于 np.pad 是可微的（梯度直接通过），我们用 result 包装
            # 这里偷懒：直接构造一个结果 Tensor 但 _backward 仅传给 x
            x_padded._backward = (lambda: x._accumulate_grad(
                unbroadcast(x_padded.grad[..., p:L + p] if p > 0 else x_padded.grad, x.shape)
            )) if x.requires_grad else None
        else:
            x_padded = x
        L_padded = x_padded.shape[-1]
        L_out = (L_padded - d * (K - 1) - 1) // s + 1
        # im2col: (B, C_in*K, L_out)
        cols = np.zeros((B, C_in * K, L_out), dtype=x.data.dtype)
        x_np = x_padded.data
        for k in range(K):
            offset = k * d
            # 取 x_padded[:, :, offset : offset + s*L_out : s]
            x_slice = x_np[:, :, offset: offset + s * L_out: s]  # (B, C_in, L_out)
            cols[:, k * C_in:(k + 1) * C_in, :] = x_slice
        # weight: (out, in, K) -> reshape to (out, in*K)
        W = self.weight.data.reshape(self.out_channels, -1)  # (out, in*K)
        # matmul: (B, out, L_out) = (out, in*K) @ (B, in*K, L_out)
        # 用 batched matmul：W (out, in*K) -> (1, out, in*K) 广播
        out_data = np.einsum("ok,bkl->bol", W, cols)  # (B, out, L_out)
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, -1, 1)
        # 反向：由于 im2col 索引较复杂，这里用闭包实现
        out = Tensor(out_data, requires_grad=x.requires_grad or self.weight.requires_grad,
                     _children=tuple(c for c in [x, self.weight, self.bias] if c is not None and (c.requires_grad if hasattr(c, 'requires_grad') else False)),
                     _op="conv1d")

        def _backward():
            g = out.grad  # (B, out, L_out)
            if self.bias is not None and self.bias.requires_grad:
                # grad_bias: sum over B, L_out
                gb = g.sum(axis=(0, 2))
                self.bias._accumulate_grad(gb)
            if self.weight.requires_grad:
                # grad_W: (out, in*K) = g @ cols^T (sum over B)
                gw = np.einsum("bol,bkl->ok", g, cols)  # (out, in*K)
                self.weight._accumulate_grad(gw.reshape(self.weight.shape))
            if x.requires_grad:
                # grad_x: 反 im2col
                gx = np.zeros_like(x_np)
                for k in range(K):
                    offset = k * d
                    g_slice = g @ W[:, k * C_in:(k + 1) * C_in]  # (B, C_in, L_out)? 实际是 (B, out, L_out) @ (out, C_in) -> (B, C_in, L_out)
                    # 反向 scatter 回 x_padded
                    for i in range(L_out):
                        gx[:, :, offset + i * s] += g_slice[:, :, i]
                if p > 0:
                    gx = gx[..., p:L + p]
                x._accumulate_grad(gx)

        if out.requires_grad:
            out._backward = _backward
        return out

    def _forward_torch(self, x: Tensor) -> Tensor:
        """GPU 路径委托 ``torch.nn.functional.conv1d``。"""
        from .tensor import _TORCH, _is_torch_data
        if _TORCH is None:
            raise RuntimeError("未安装 PyTorch")
        # 对齐 device：把 x / weight / bias 都迁到同一 device
        ref = x if x._is_torch_tensor else self.weight
        if not x._is_torch_tensor:
            x = x.to(ref.device)
        if not self.weight._is_torch_tensor:
            self.weight = self.weight.to(ref.device)
        if self.bias is not None and not self.bias._is_torch_tensor:
            self.bias = self.bias.to(ref.device)
        weight = self.weight.data
        bias = self.bias.data if self.bias is not None else None
        out_data = _TORCH.nn.functional.conv1d(
            x.data, weight, bias=bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
        )
        requires_grad = x.requires_grad or self.weight.requires_grad
        children = tuple(c for c in [x, self.weight, self.bias]
                         if c is not None and c.requires_grad)
        from .tensor import Tensor as _T
        return _T(out_data, requires_grad=requires_grad, _children=children,
                  _op="conv1d", device=str(out_data.device))


class LayerNormFast(Module):
    """LayerNorm 优化版。

    相比 ``LayerNorm``：
    - GPU 路径委托 ``torch.nn.functional.layer_norm``（含 fused kernel）
    - CPU 路径保持 NumPy 实现，但用 ``np.var`` 直接计算（避免重复 mean）
    - 数值上与 ``LayerNorm`` 等价

    Args:
        normalized_shape: 归一化形状（int 或 tuple）
        eps: 数值稳定常数
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Tensor.ones(*self.normalized_shape, requires_grad=True)
        self.bias = Tensor.zeros(*self.normalized_shape, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # GPU 路径委托 torch.nn.functional.layer_norm
        if x._is_torch_tensor or self.weight._is_torch_tensor:
            from .tensor import _TORCH, _is_torch_data
            # 对齐 device
            if not self.weight._is_torch_tensor:
                # 参数仍在 CPU，但 x 在 GPU：迁移参数（一次性）
                self.weight = self.weight.to(x.device)
                self.bias = self.bias.to(x.device)
            # 用 torch_apply 委托
            w_data = self.weight.data
            b_data = self.bias.data
            return x._torch_apply(
                lambda a: _TORCH.nn.functional.layer_norm(
                    a, normalized_shape=tuple(w_data.shape),
                    weight=w_data, bias=b_data, eps=self.eps,
                ),
                _op="layernorm_fast"
            )
        # CPU 优化路径：走公共归一化内核的 fast_path（numpy 向量化 + 手动反向）。
        # Part5K1 SubTask 2.4 去壳合并：原内联实现与 _normalize_fast 完全等价，已收敛到内核。
        return _normalize_kernel(x, self.weight, self.bias, self.eps,
                                 mean_centered=True, fast_path=True)
