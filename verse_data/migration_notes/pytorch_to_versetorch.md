# PyTorch → VerseTorch 迁移指南

> 本指南帮助用户将现有 PyTorch 代码迁移到 VerseTorch 后端，享受"纯 CPU / 零重型依赖 / 量化优先"的优势。
> 关联源码：[`torch_api.py`](../../packages/verse_compat/verse_compat/torch_api.py), [`hf_loader.py`](../../packages/verse_compat/verse_compat/hf_loader.py)
> 关联 ADR：[ADR-001 CPU 优先设计](../../docs/architecture/adr-001-cpu-first.md)
> 关联设计：[autograd_design.md](../designs/autograd_design.md)

## 1. 概览

### 1.1 VerseTorch 的定位

VerseTorch 是 **PyTorch 的纯 CPU 子集实现**，定位为：

- **PyTorch 替代品**（不是补充品）：API 与 PyTorch 子集对齐，便于代码迁移；
- **纯 CPU**：不依赖 CUDA / cuDNN / GPU，可在任何能跑 NumPy 的环境运行；
- **无 C++ 后端**：所有算子用 NumPy + Python 实现，可审计、可教学；
- **零重型依赖**：运行时仅依赖 NumPy（≥ 1.26）+ Python 标准库；
- **量化优先**：INT4 / 1.58-bit ternary 量化是一等公民，默认走量化路径。

### 1.2 API 兼容性目标

VerseTorch 实现 PyTorch 的 **核心子集**，覆盖：

- `Tensor` 类（包装 `np.ndarray`）；
- 元素级 / shape / reduction / matmul 算子；
- `nn.Module` 基类与 `Linear / Embedding / LayerNorm / RMSNorm / Dropout / Sequential / ModuleList`；
- `optim.SGD / Adam / AdamW` 与 `StepLR / ExponentialLR / CosineAnnealingLR`；
- `losses.cross_entropy / binary_cross_entropy / mse_loss / l1_loss / kl_div_loss`；
- `no_grad / enable_grad` 上下文管理器。

**不兼容**的部分见 [§10 不兼容项](#10-不兼容项)。

## 2. 导入替换

### 2.1 别名兼容（最小改动）

最简单的迁移方式：用 `verse_compat.torch_api` 提供的别名替换 `import torch`：

```python
# 原 PyTorch 代码
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(2, 3)
linear = torch.nn.Linear(3, 4)
out = linear(x)
loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1]))
```

```python
# 迁移到 VerseTorch
from verse_compat.torch_api import torch, nn, optim  # 别名兼容
# 或：
from verse_compat import torch_api as torch

x = torch.randn(2, 3)
linear = torch.nn.Linear(3, 4)  # 实际是 verse_torch.nn.Linear
out = linear(x)
# loss 用 verse_torch.losses.cross_entropy 而非 F.cross_entropy
from verse_torch import losses
loss = losses.cross_entropy(out, [0, 1])
```

源码定义见 [`torch_api.py`](file:///workspace/packages/verse_compat/verse_compat/torch_api.py)。

### 2.2 直接使用 verse_torch（推荐）

更清晰的方式是直接用 `verse_torch` 包：

```python
import verse_torch as vt
from verse_torch import Tensor, nn, optim, losses, no_grad

x = Tensor.randn(2, 3)  # 或 vt.Tensor(...)
linear = nn.Linear(3, 4)
out = linear(x)
loss = losses.cross_entropy(out, [0, 1])
```

### 2.3 安装

VerseTorch 通过 `pip install -e packages/verse_torch` 安装，或直接把 `packages/verse_torch` 加入 `sys.path`：

```python
import sys
sys.path.insert(0, "/path/to/verse/packages/verse_torch")
sys.path.insert(0, "/path/to/verse/packages/verse_compat")  # 如需 torch_api 别名
```

## 3. Tensor 创建

| PyTorch                              | VerseTorch                                       | 备注                          |
| ------------------------------------ | ------------------------------------------------ | ----------------------------- |
| `torch.tensor([1, 2, 3])`            | `Tensor([1, 2, 3])` 或 `vt.tensor([1, 2, 3])`    | Python list 默认 int64        |
| `torch.tensor([1.0, 2.0])`           | `Tensor([1.0, 2.0])`                             | Python float 默认 float32     |
| `torch.zeros(2, 3)`                  | `Tensor.zeros(2, 3)` 或 `vt.zeros(2, 3)`          |                               |
| `torch.ones(2, 3)`                   | `Tensor.ones(2, 3)`                              |                               |
| `torch.randn(2, 3)`                  | `Tensor.randn(2, 3)`                             | 标准正态                      |
| `torch.rand(2, 3)`                   | `Tensor.rand(2, 3)`                              | [0, 1) 均匀                   |
| `torch.empty(2, 3)`                  | `Tensor.empty(2, 3)`                              | 实际为 zeros（确定性考虑）    |
| `torch.arange(10)`                   | `Tensor.arange(10)`                              |                               |
| `torch.arange(0, 1, 0.1)`            | `Tensor.arange(0, 1, 0.1)`                       |                               |
| `torch.full((2, 3), 0.5)`            | `Tensor.full((2, 3), 0.5)`                        |                               |
| `torch.eye(3)`                       | `Tensor.eye(3)`                                  |                               |
| `torch.tensor(np_array)`             | `Tensor(np_array)`                                | 保留 numpy dtype              |
| `torch.from_numpy(np_array)`         | `Tensor(np_array)`                                | 共享内存（同 PyTorch）         |

工厂方法源码见 [`tensor.py` 第 188-237 行](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L188-L237)。

### 3.1 dtype 处理

```python
# PyTorch
x = torch.randn(2, 3, dtype=torch.float32)
y = x.to(torch.float64)

# VerseTorch
x = Tensor.randn(2, 3, dtype=np.float32)  # 或 dtype="float32"
y = x.cast(np.float64)  # 或 x.float() / x.long() / x.int() / x.bool()
```

dtype 字符串别名（来自 `torch_api`）：

```python
from verse_compat.torch_api import float32, float64, int32, int64, bool
# float16 / bfloat16 也定义了，但底层映射到 float32（NumPy 不原生支持 bf16）
```

## 4. Tensor 操作

### 4.1 基本算术

| PyTorch                  | VerseTorch                | 备注                |
| ------------------------ | ------------------------- | ------------------- |
| `a + b`                  | `a + b`                   | 元素级加            |
| `a - b`                  | `a - b`                   |                     |
| `a * b`                  | `a * b`                   | 元素级乘            |
| `a / b`                  | `a / b`                   |                     |
| `a ** 2`                 | `a ** 2`                  | 标量 power          |
| `a ** b`（Tensor b）    | `a ** b`                  | 元素级 power        |
| `-a`                     | `-a`                      |                     |
| `a @ b`                  | `a @ b`                   | 矩阵乘              |
| `torch.matmul(a, b)`     | `a @ b` 或 `a.matmul(b)`  |                     |
| `a.sum()`                | `a.sum()`                 |                     |
| `a.mean()`               | `a.mean()`                |                     |
| `a.max(dim=-1)`          | `a.max(dim=-1)`           | 返回 max 值（不返回 indices） |
| `a.argmax(dim=-1)`       | `a.argmax(dim=-1)`        |                     |
| `a.exp()`                | `a.exp()`                 |                     |
| `a.log()`                | `a.log()`                 |                     |
| `a.sqrt()`               | `a.sqrt()`                |                     |
| `a.relu()`               | `a.relu()`                |                     |
| `a.gelu()`               | `a.gelu()`                | tanh 近似           |
| `a.sigmoid()`            | `a.sigmoid()`            | 数值稳定             |
| `a.tanh()`               | `a.tanh()`                |                     |
| `a.silu()`               | `a.silu()`                |                     |
| `a.softmax(dim=-1)`      | `a.softmax(dim=-1)`       |                     |
| `a.log_softmax(dim=-1)`  | `a.log_softmax(dim=-1)`   |                     |
| `a.abs()`                | `a.abs()`                 |                     |
| `torch.abs(a)`           | `a.abs()`                 |                     |
| `torch.exp(a)`           | `a.exp()`                 |                     |
| `a.clamp(low, high)`     | `a.clamp(low, high)`      |                     |
| `torch.cat([a, b], dim=0)`| `vt.cat([a, b], dim=0)` 或 numpy `np.concatenate` | cat 在 torch_api 中实现 |
| `torch.stack([a, b], dim=0)` | `vt.stack([a, b], dim=0)` |                 |

### 4.2 Shape 操作

| PyTorch                            | VerseTorch                          | 备注                            |
| ---------------------------------- | ----------------------------------- | ------------------------------- |
| `a.reshape(2, 3)`                  | `a.reshape(2, 3)`                    |                                 |
| `a.view(2, 3)`                     | `a.view(2, 3)`                       | **与 reshape 等价**（见 §4.3） |
| `a.transpose(0, 1)`                | `a.transpose(0, 1)`                  |                                 |
| `a.T`                              | `a.T`                                | 完全反转所有轴                   |
| `a.permute(2, 0, 1)`               | `a.permute(2, 0, 1)`                 |                                 |
| `a.unsqueeze(0)`                   | `a.unsqueeze(0)`                     |                                 |
| `a.squeeze()`                      | `a.squeeze()`                        |                                 |
| `a.squeeze(0)`                     | `a.squeeze(0)`                       |                                 |
| `a.expand(2, 3, 4)`                | `a.expand(2, 3, 4)`                  | -1 表示保持原维度                |
| `a.broadcast_to((2, 3, 4))`        | `a.broadcast_to((2, 3, 4))`          | 内部调 `expand`                  |
| `a.contiguous()`                   | `a.contiguous()`                    | NumPy 默认连续；非连续时 copy   |
| `a.flatten()`                      | `a.flatten()`                       |                                 |
| `a.flatten(start_dim=1)`           | `a.flatten(start_dim=1)`             |                                 |
| `a[i]` / `a[i:j]` / `a[mask]`      | `a[i]` / `a[i:j]` / `a[mask]`        | int / slice / tuple / bool mask |

### 4.3 view 在 VerseTorch 中的语义

**重要差异**：在 PyTorch 中，`view` 要求 Tensor 是 contiguous 的（基于 stride），而 `reshape` 不要求（必要时 copy）。

VerseTorch 的 `view` 实现是 `reshape` 的别名（[tensor.py 第 692-694 行](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L692-L694)）：

```python
def view(self, *shape) -> "Tensor":
    # view 是 reshape 的别名（NumPy 默认连续，所以可直接 reshape）
    return self.reshape(*shape)
```

原因：NumPy 的 ndarray 默认是 C-contiguous，不存在 PyTorch 那种 stride-based 的非连续情况（除非 transpose / slice 后显式调用）。所以 `view` 与 `reshape` 行为一致。

**迁移提示**：PyTorch 代码中如果用 `a.view(...)` 但 Tensor 实际非 contiguous，会报错；VerseTorch 永远不会报这个错，但可能 silently copy。如果迁移后行为异常，把 `a.view(...)` 改成 `a.reshape(...)` 显式表达意图。

### 4.4 Reduction 操作

```python
# PyTorch
a.sum(dim=1, keepdim=True)
a.mean(dim=-1)
a.max(dim=0)  # 返回 (values, indices)
a.var(unbiased=True)
a.std()

# VerseTorch
a.sum(dim=1, keepdim=True)  # 一致
a.mean(dim=-1)               # 一致
a.max(dim=0)  # 只返回 values（不返回 indices）
a.var(unbiased=True)         # 一致
a.std()                      # 内部 = sqrt(var)
```

**注意**：`a.max(dim=0)` 在 PyTorch 中返回 namedtuple `(values, indices)`；VerseTorch 只返回 values。如需 indices，单独调用 `a.argmax(dim=0)`。

## 5. nn.Module

### 5.1 基类 API

```python
# PyTorch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.dropout(self.linear(x))

model = MyModel()
params = list(model.parameters())  # 生成器
sd = model.state_dict()             # OrderedDict
model.load_state_dict(sd)
model.zero_grad()
model.train()
model.eval()
model.to(torch.float32)
model.apply(init_fn)
```

```python
# VerseTorch
from verse_torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.dropout(self.linear(x))

model = MyModel()
params = list(model.parameters())  # 生成器
sd = model.state_dict()            # 普通 dict（非 OrderedDict）
model.load_state_dict(sd)
model.zero_grad()
model.train()
model.eval()
model.to(np.float32)               # 仅接受 dtype，不接受 device
model.apply(init_fn)
```

API 几乎完全一致。源码见 [`nn.py` Module 类](file:///workspace/packages/verse_torch/verse_torch/nn.py#L91-L259)。

### 5.2 核心层

| PyTorch                                       | VerseTorch                                       | 备注                          |
| --------------------------------------------- | ------------------------------------------------ | ----------------------------- |
| `nn.Linear(in, out, bias=True)`              | `nn.Linear(in, out, bias=True)`                  | Kaiming uniform 初始化        |
| `nn.Embedding(num, dim)`                       | `nn.Embedding(num, dim)`                          | normal_(0, 1) 初始化          |
| `nn.LayerNorm(dim, eps=1e-5)`                 | `nn.LayerNorm(dim, eps=1e-5)`                    |                               |
| `nn.RMSNorm(dim, eps=1e-6)`                   | `nn.RMSNorm(dim, eps=1e-6)`                      | PyTorch 2.4+ 引入              |
| `nn.Dropout(p=0.5)`                           | `nn.Dropout(p=0.5)`                              |                               |
| `nn.Sequential(*modules)`                     | `nn.Sequential(*modules)`                        |                               |
| `nn.ModuleList([...])`                        | `nn.ModuleList([...])`                           |                               |
| `nn.Conv2d(...)`                              | **不支持**                                       | 用 reshape + Linear 模拟      |
| `nn.BatchNorm1d / 2d`                        | **不支持**                                       | 用 LayerNorm 替代             |
| `nn.RNN / LSTM / GRU`                         | **不支持**                                       | 在 verse_awm/rssm.py 手写 GRU |
| `nn.MultiheadAttention`                       | **不支持**                                       | 在 verse_awm/jepa.py 手写     |

### 5.3 train() / eval() 模式

```python
# 与 PyTorch 完全一致
model.train()  # training=True, 启用 Dropout
model.eval()   # training=False, 禁用 Dropout

# 在 forward 中通过 self.training 判断
class MyModel(nn.Module):
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.1  # 训练时加噪声
        return x
```

### 5.4 初始化辅助

```python
from verse_torch.nn import kaiming_uniform_, xavier_uniform_, normal_, zeros_, ones_, uniform_

# 与 PyTorch 同名函数一致
kaiming_uniform_(linear.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
xavier_uniform_(linear.weight, gain=1.0)
normal_(linear.weight, mean=0.0, std=0.02)
zeros_(linear.bias)
ones_(embedding.weight)
```

## 6. Optimizer

```python
# PyTorch
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# VerseTorch
from verse_torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 训练循环完全一致
for inputs, targets in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

参数语义与 PyTorch 完全一致。源码见 [`optim.py`](file:///workspace/packages/verse_torch/verse_torch/optim.py)。

### 6.1 Scheduler

```python
# PyTorch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.99)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# VerseTorch
from verse_torch.optim import StepLR, ExponentialLR, CosineAnnealingLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.99)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 用法一致
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

注意：
- VerseTorch 不支持 `ReduceLROnPlateau` / `OneCycleLR` / `LambdaLR` 等高级调度器；
- `CosineAnnealingLR` 在 `last_epoch > T_max` 时返回 `eta_min`（与 PyTorch SGDR 重启语义略有差异）。

## 7. Loss

```python
# PyTorch
import torch.nn.functional as F

loss = F.cross_entropy(logits, targets)
loss = F.binary_cross_entropy(pred, target)
loss = F.binary_cross_entropy_with_logits(logits, target)
loss = F.mse_loss(pred, target)
loss = F.l1_loss(pred, target)
loss = F.kl_div(log_probs, target_probs)

# VerseTorch
from verse_torch import losses

loss = losses.cross_entropy(logits, targets)
loss = losses.binary_cross_entropy(pred, target)
loss = losses.binary_cross_entropy_with_logits(logits, target)
loss = losses.mse_loss(pred, target)
loss = losses.l1_loss(pred, target)
loss = losses.kl_div_loss(log_probs, target_probs)
```

源码见 [`losses.py`](file:///workspace/packages/verse_torch/verse_torch/losses.py)。

**注意**：
- VerseTorch 的 loss 是 **函数** 而非 `nn.Module`（与 `F.cross_entropy` 一致）；
- 不存在 `nn.CrossEntropyLoss` / `nn.MSELoss` 类，直接调用函数；
- `cross_entropy(logits, targets)` 的 logits 形状 `(N, C)`，targets 形状 `(N,)`（int 类别索引）。

## 8. state_dict 兼容

### 8.1 从 PyTorch 加载预训练权重

通过 `verse_compat.load_hf_state_dict` 可以直接加载 PyTorch 训练保存的 state_dict：

```python
from verse_compat import load_hf_state_dict

# 从 HuggingFace Hub 加载
state_dict = load_hf_state_dict("microsoft/phi-2")

# 或从本地路径加载
state_dict = load_hf_state_dict("/path/to/weights/")
# 或单个文件
state_dict = load_hf_state_dict("/path/to/model.safetensors")

# 加载到 VerseTorch 模型
model.load_state_dict(state_dict, strict=False)
```

支持的格式：
- `.safetensors`（推荐，零拷贝，需 `pip install safetensors`）；
- `.bin`（PyTorch pickle）：
  - 若已安装 `torch`，用 `torch.load`；
  - 否则用自实现的 pickle 解析器（支持 float32/64/16、bfloat16（→float32）、int8/16/32/64、bool、uint8）。

源码见 [`hf_loader.py`](file:///workspace/packages/verse_compat/verse_compat/hf_loader.py)。

### 8.2 state_dict 的键名匹配

VerseTorch 的 `state_dict()` 返回普通 `dict`（不是 `OrderedDict`），键名规则与 PyTorch 一致：

```
{
    "linear.weight": ndarray (out, in),
    "linear.bias": ndarray (out,),
    "embedding.weight": ndarray (num, dim),
    "layers.0.fc1.weight": ...,
    "layers.0.fc1.bias": ...,
    ...
}
```

`load_state_dict(sd, strict=True)` 严格匹配；`strict=False` 宽松匹配（仅覆盖能匹配的键）。

### 8.3 保存 VerseTorch 模型

```python
import numpy as np

# 保存
sd = model.state_dict()
np.savez("/path/to/model.npz", **sd)

# 加载
data = np.load("/path/to/model.npz")
sd = {k: data[k] for k in data.files}
model.load_state_dict(sd)
```

或用 `pickle` / `json`（需要先转 list）：

```python
import pickle

with open("/path/to/model.pkl", "wb") as f:
    pickle.dump({k: v.tolist() for k, v in model.state_dict().items()}, f)
```

## 9. 典型迁移示例

下面是一个 50 行的 PyTorch MNIST 训练代码，迁移到 VerseTorch 只需改 5 行：

### 9.1 原 PyTorch 代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 1. 数据
x_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

# 2. 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()

# 3. 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练循环
for epoch in range(5):
    for xb, yb in loader:
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# 5. 评估
model.eval()
with torch.no_grad():
    test_x = torch.randn(100, 784)
    preds = model(test_x).argmax(dim=-1)
    print(preds)
```

### 9.2 迁移到 VerseTorch

```python
import numpy as np
from verse_torch import Tensor, nn, optim, losses, no_grad  # ← 改动 1

# 1. 数据（用 numpy + 手写 batch）
x_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, (1000,))

def iterate_minibatches(X, y, batch_size, shuffle=True):
    n = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]

# 2. 模型（与 PyTorch 完全一致）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x).relu()  # ← 改动 2：用 .relu() 而非 F.relu
        return self.fc2(x)

model = MLP()

# 3. 优化器（与 PyTorch 完全一致）
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. 训练循环
for epoch in range(5):
    for xb, yb in iterate_minibatches(x_train, y_train, 32):
        x = Tensor(xb)                                    # ← 改动 3：np → Tensor
        logits = model(x)
        loss = losses.cross_entropy(logits, yb)           # ← 改动 4：losses.cross_entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# 5. 评估
model.eval()
with no_grad():                                          # ← 改动 5：no_grad
    test_x = Tensor(np.random.randn(100, 784).astype(np.float32))
    preds = model(test_x).data.argmax(axis=-1)            # ← 改动 6：.data.argmax
    print(preds)
```

总共 **6 处改动**，大部分是导入替换与 numpy ↔ Tensor 转换。

## 10. 不兼容项

迁移时必须注意以下 **不兼容** 的 PyTorch 特性：

### 10.1 硬件 / 后端

- **无 CUDA**：所有计算在 CPU 上，`device="cuda"` 参数被忽略；
- **无 DistributedDataParallel**：不支持多卡训练；
- **无 TensorRT / cuDNN 集成**：没有 GPU 加速后端；
- **无 AMP (Automatic Mixed Precision)**：不支持 float16/bfloat16 训练（仅推理时通过 `quantize` 模块做 INT4/INT8）。

### 10.2 自动微分

- **无 `autograd.Function`**：不能注册自定义算子的反向；新算子必须修改 `tensor.py`；
- **无 `torch.autograd.grad`**：只能用 `loss.backward()` 触发反向；
- **无高阶梯度**：`backward()` 不接受 `create_graph=True`；
- **无 in-place 操作**：所有 op 返回新 Tensor，`a += b` 等价于 `a = a + b`（破坏计算图，实际未实现）。

### 10.3 编译 / 部署

- **无 `torch.compile`**：不支持图编译优化；
- **无 `torch.jit`**：不支持 TorchScript 序列化；
- **无 `torch.fx`**：不支持图变换；
- **无 `torch.export`**：不支持 Export 模型格式。

### 10.4 Tensor 语义

- **`view` 与 `reshape` 等价**：因为 NumPy 默认连续，不存在 stride-based view；
- **`contiguous()` 是 no-op**：仅在非连续时 copy；
- **dtype 系统简化**：仅 float32/float64/int32/int64/bool；**无 float16/bfloat16**（`torch_api` 中的 `float16` 别名映射到 float32）；
- **`torch.Tensor` 不是 `np.ndarray` 子类**：用 `.data` 或 `.numpy()` 访问底层 ndarray；
- **不支持 sparse tensor**：所有 Tensor 都是 dense；
- **不支持 quantized tensor**：量化通过 `quantize.py` 单独的 API 处理。

### 10.5 nn 层

- **无 `nn.Conv1d/2d/3d`**：用 `reshape + Linear` 模拟；
- **无 `nn.BatchNorm*`**：用 `LayerNorm` 替代；
- **无 `nn.RNN/LSTM/GRU`**：在 `verse_awm/rssm.py` 手写 GRU；
- **无 `nn.MultiheadAttention`**：在 `verse_awm/jepa.py` 手写；
- **无 `nn.TransformerEncoder/Decoder`**：用 `ModuleList + TransformerBlock` 组合。

### 10.6 工具

- **无 `torch.utils.data.DataLoader`**：用 numpy + 手写 batch 迭代；
- **无 `torch.utils.tensorboard`**：用 stdout / 自定义日志；
- **无 `torch.distributed`**：不支持分布式训练；
- **无 `torch.cuda.amp`**：不支持混合精度。

## 11. 常见错误与解决方案（FAQ）

### Q1: `RuntimeError: Tensor does not require grad and cannot call backward()`

**原因**：调用 `backward()` 的 Tensor `requires_grad=False`。

**解决**：
```python
# 错误
x = Tensor(data)  # requires_grad=False
y = model(x)
y.backward()  # 报错

# 正确（1）：模型参数有 requires_grad=True，y 自动 requires_grad=True
x = Tensor(data, requires_grad=False)
y = model(x)
y.backward()  # OK，因为模型参数 requires_grad=True

# 正确（2）：用 enable_grad 强制启用
with enable_grad():
    y = model(x)
    y.backward()
```

### Q2: `RuntimeError: grad can only be implicitly created for scalar outputs`

**原因**：对非标量 Tensor 调用 `backward()` 但未传 `grad` 参数。

**解决**：
```python
# 错误
y = model(x)  # shape (B, C)
y.backward()  # 报错

# 正确（1）：先 sum 成标量
loss = y.sum()
loss.backward()

# 正确（2）：传 grad
y.backward(grad=np.ones_like(y.data))
```

### Q3: 训练时 loss 不下降

**可能原因**：
1. 忘记 `optimizer.zero_grad()`：梯度累积导致训练不稳定；
2. `requires_grad=False`：模型参数未启用梯度；
3. 在 `with no_grad():` 内训练：计算图未构建；
4. 学习率过大 / 过小。

**排查**：
```python
# 检查参数 requires_grad
for name, p in model.named_parameters():
    print(name, p.requires_grad, p.grad is not None)
```

### Q4: 数值精度问题（loss 异常大或 NaN）

**可能原因**：
1. 用了 `float32` 但 cumsum 长序列误差大 → 用 `float64`；
2. `log(0)` 未加 eps → 用 `np.log(x + 1e-12)`；
3. `1 / 0` 未保护 → 用 `np.where`；
4. softmax 没减 max → 用 `Tensor.softmax()` 方法（已稳定）。

### Q5: 速度比 PyTorch 慢

**预期行为**：CPU 上 VerseTorch 比 PyTorch CPU 慢 2-4 倍（无 fused kernel）。

**优化**：
1. 推理用 `with no_grad():` 跳过计算图构建；
2. 用 INT4 / ternary 量化（`verse_torch.quantize`）；
3. 增加 batch size 摊薄 Python 调用开销；
4. 关键路径用 numpy 向量化，避免 Python 循环。

### Q6: `view` 报错 "shape ... is invalid for input of size ..."

**原因**：PyTorch 中 view 要求 contiguous；VerseTorch 中 view 等价于 reshape。

**解决**：检查目标 shape 与元素数是否匹配；如果原代码依赖 view 的 contiguous 检查，迁移后这个检查消失，可能掩盖 bug。

### Q7: state_dict 加载失败

**原因**：
1. 模型结构不完全匹配 → 用 `strict=False`；
2. 键名前缀不一致（如 `model.` 前缀）→ 手动 strip；
3. dtype 不匹配 → 显式 cast。

**示例**：
```python
sd = load_hf_state_dict("/path/to/weights/")
# 去除 "model." 前缀
sd = {k.removeprefix("model."): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
```

### Q8: 如何调试 autograd？

VerseTorch 没有 `torch.autograd.gradcheck`，但可以手写有限差分：

```python
def numerical_grad(f, x, eps=1e-4):
    """对 f(x) 关于 x 做有限差分梯度检查。"""
    grad = np.zeros_like(x.data)
    for i in range(x.data.size):
        x.data.flat[i] += eps
        f_plus = float(f(x).data)
        x.data.flat[i] -= 2 * eps
        f_minus = float(f(x).data)
        x.data.flat[i] += eps
        grad.flat[i] = (f_plus - f_minus) / (2 * eps)
    return grad

# 与 autograd 结果对比
x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * x).sum()
y.backward()
print("autograd:", x.grad)
print("numerical:", numerical_grad(lambda t: (t * t).sum(), x))
```

差异应在 1e-5 量级。

## 12. 迁移清单（Checklist）

迁移一个 PyTorch 项目到 VerseTorch 时，按以下清单逐项检查：

- [ ] 导入：`import torch` → `import verse_torch as vt` 或 `from verse_compat.torch_api import torch`；
- [ ] Tensor 创建：`torch.tensor(...)` → `Tensor(...)` 或 `vt.tensor(...)`；
- [ ] dtype：`torch.float32` → `np.float32` 或 `"float32"`；
- [ ] 算子：`F.relu(x)` → `x.relu()`；`F.softmax(x, dim=-1)` → `x.softmax(dim=-1)`；
- [ ] Loss：`F.cross_entropy` → `losses.cross_entropy`；
- [ ] nn 层：检查是否用了不支持的层（Conv / BatchNorm / RNN / MultiheadAttention）；
- [ ] DataLoader：替换为 numpy 手写 batch 迭代；
- [ ] device：移除所有 `.to("cuda")` / `.cuda()` 调用；
- [ ] AMP：移除 `torch.cuda.amp.autocast`；
- [ ] DDP：移除 `DistributedDataParallel` 包装；
- [ ] `torch.compile` / `torch.jit`：移除；
- [ ] 自定义 autograd.Function：重写为 `Tensor` 方法的闭包，或修改 `tensor.py`；
- [ ] 模型保存：`torch.save(model.state_dict(), path)` → `np.savez(path, **model.state_dict())`；
- [ ] 模型加载：`torch.load(path)` → `load_hf_state_dict(path)` 或 `np.load(path)`；
- [ ] 数值检查：迁移后跑一遍，确认 loss 与 PyTorch 版本在 1e-5 量级一致。

## 13. 参考资源

- VerseTorch 源码：[`packages/verse_torch/`](../../packages/verse_torch/)；
- 设计文档：[autograd_design.md](../designs/autograd_design.md)；
- ADR-001 CPU 优先：[`docs/architecture/adr-001-cpu-first.md`](../../docs/architecture/adr-001-cpu-first.md)；
- 示例代码：[`examples/mnist_mlp.py`](../../examples/mnist_mlp.py)（MNIST 训练）、[`examples/minimal_lm.py`](../../examples/minimal_lm.py)（字符级 LM）、[`examples/cpu_inference_demo.py`](../../examples/cpu_inference_demo.py)（CPU 推理）；
- PyTorch 文档：[https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)（参考 API 行为）；
- micrograd：[https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)（autograd 教学参考）。
