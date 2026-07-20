# VerseCompat

> 中文定位：**HuggingFace / PyTorch 兼容适配层**（可选），仅在用户已安装 `torch` / `transformers` / `safetensors` / `huggingface_hub` 时启用，提供 `state_dict` 加载与 `torch` API 别名，便于把现有 PyTorch 模型代码以最小改动迁移到 VerseTorch 后端。

[返回主 README](../../README.md)

## 特性

- **HF `state_dict` 加载**：`load_hf_state_dict(repo_or_path)` 把 HuggingFace repo 或本地路径下的权重读为 `dict[str, verse_torch.Tensor]`
  - 支持 `.safetensors`（零拷贝，需安装 `safetensors`）
  - 支持 `.bin`（PyTorch pickle）：已装 `torch` 时走 `torch.load`；未装时回退到自实现的 `pickle.Unpickler` + `persistent_load` 解析器
  - 自动识别 HF repo id（如 `"microsoft/phi-2"`）vs 本地路径
  - 已安装 `huggingface_hub` 时走 `snapshot_download`，否则降级到 `urllib` + HF API
  - 多分片文件按字典序自动合并
- **`torch` API 别名**：`torch_api` 模块把常用 PyTorch 符号透传到 `verse_torch`，便于平滑迁移现有代码
  - `torch.nn.Linear → verse_torch.nn.Linear`
  - `torch.nn.Module → verse_torch.nn.Module`
  - `torch.Tensor → verse_torch.Tensor`
  - `torch.optim.AdamW → verse_torch.optim.AdamW`
  - 工厂函数：`tensor / zeros / ones / randn / rand / arange / full / empty / eye`
  - 数学函数：`softmax / sigmoid / relu / gelu / tanh / exp / log / sqrt / matmul / cat / stack`
  - 梯度控制：`no_grad / enable_grad / set_grad_enabled / is_grad_enabled`
  - dtype 字符串：`float16 / float32 / float64 / bfloat16 / int8 / int16 / int32 / int64 / uint8`
- **可选依赖**：`torch` / `transformers` / `safetensors` / `huggingface_hub` 未安装时自动降级，运行时**永远不强制要求**这些包

## 安装

```bash
pip install -e packages/verse_torch
pip install -e packages/verse_compat
```

要启用完整功能，按需安装可选依赖：

```bash
pip install "safetensors>=0.4"        # 加载 .safetensors 权重
pip install "huggingface_hub>=0.20"    # 从 HF Hub 下载
pip install torch                       # 完整 .bin 加载（自实现 fallback 仅支持常见 storage 类型）
```

## 模块导出

`verse_compat/__init__.py` 暴露以下符号（节选，完整列表见源文件）：

```python
from verse_compat import (
    load_hf_state_dict,             # 主入口
    # torch_api 别名
    Tensor, nn, optim, losses,
    Linear, Embedding, LayerNorm, RMSNorm, Dropout, Module, Sequential, ModuleList,
    SGD, Adam, AdamW,
    cross_entropy, mse_loss,
    no_grad, enable_grad, set_grad_enabled, is_grad_enabled,
    tensor, zeros, ones, randn, rand, arange, full, empty, eye,
    softmax, sigmoid, relu, gelu, tanh, exp, log, sqrt, matmul, cat, stack,
    # dtype 字符串别名
    float16, float32, float64, bfloat16,
    int8, int16, int32, int64, uint8,
)
```

## 模块详解

### `hf_loader.load_hf_state_dict`

```python
load_hf_state_dict(
    repo_id_or_path: str,
    revision: str = "main",
    pattern: str = "*.safetensors",
) -> dict[str, Tensor]
```

加载顺序：

1. 判断 `repo_id_or_path` 是本地路径还是 HF repo id：
   - 已存在 / `./` / `/` / `~/` 开头 / 含 Windows 盘符 → 本地
   - 否则视为 HF repo id（如 `"microsoft/phi-2"`）
2. 本地路径：列出目录下 `*.safetensors` 文件；若空则回退到 `*.bin`
3. HF repo id：
   - 优先 `huggingface_hub.snapshot_download`（如已安装）
   - 否则用 `urllib` + `https://huggingface.co/{repo}/resolve/{revision}/{file}` 单文件下载到临时目录
4. 逐文件加载并合并：
   - `.safetensors` → `safe_open(..., framework="numpy")` 零拷贝取 `ndarray`
   - `.bin` → 优先 `torch.load(path, map_location="cpu")`；未装 `torch` 时用自实现 pickle 解析器
5. 所有 `ndarray` 包装成 `Tensor(arr, requires_grad=False)`，键名与原 `state_dict` 完全一致

支持的 PyTorch storage 类型（自实现解析器映射表）：

| PyTorch storage | NumPy dtype |
|---|---|
| `FloatStorage` | `float32` |
| `DoubleStorage` | `float64` |
| `HalfStorage` | `float16` |
| `BFloat16Storage` | `float32`（容器，numpy 无原生 bf16） |
| `ByteStorage` / `CharStorage` / `ShortStorage` / `IntStorage` / `LongStorage` / `BoolStorage` | 对应整数类型 |
| `QInt8Storage` / `QInt32Storage` / `QUInt8Storage` | 量化专用整数 |

未知 storage 类型会回退到 `float32` 并打印警告。

### `torch_api`

把 PyTorch 风格的代码直接搬到 `verse_torch`：

```python
# 原代码：
import torch
x = torch.randn(2, 3)
linear = torch.nn.Linear(3, 4)
out = linear(x)

# 迁移后：
from verse_compat import torch_api as torch   # 仅这一行改动
x = torch.randn(2, 3)
linear = torch.nn.Linear(3, 4)
out = linear(x)
```

设计要点：

- **不重新实现算子**：全部透传到 `verse_torch`，仅做命名别名，不引入新行为
- **dtype 用字符串**：PyTorch 用 `torch.float32` 对象，本包用字符串 `"float32"`，调用 `Tensor.cast(dtype)` 时自动转换（`bfloat16` 会回退到 `float32`，因 NumPy 无原生 bf16）
- **工厂函数签名对齐**：`zeros(*shape, dtype=None, requires_grad=False)` 与 PyTorch 等价
- **`cat` / `stack` 保守策略**：仅当所有输入 `requires_grad=True` 时才构建反向图，避免不必要的计算图开销

## 快速开始

### 示例 1：加载 HF 模型权重到 verse_torch

```python
from verse_compat import load_hf_state_dict

# 从 HuggingFace Hub 下载（需安装 huggingface_hub 或走 urllib fallback）
state_dict = load_hf_state_dict("microsoft/phi-2")
print(f"加载 {len(state_dict)} 个参数张量")
for k, v in list(state_dict.items())[:3]:
    print(f"  {k}: shape={v.shape}, dtype={v.data.dtype}")

# 从本地路径加载（需 safetensors 或 torch 之一）
state_dict = load_hf_state_dict("./models/my_model/")
```

### 示例 2：用 torch_api 迁移 PyTorch 代码

```python
from verse_compat import torch_api as torch
from verse_compat.torch_api import Tensor, nn, optim

# 1. 构建一个最小 MLP（代码与原 PyTorch 几乎一致）
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())

model = MLP(10, 64, 5)

# 2. 训练循环
x = torch.randn(8, 10)
y = torch.randint(0, 5, (8,))
opt = optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10):
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)  # 注：本包 losses.cross_entropy 顶层导出
    loss.backward()
    opt.step()
```

### 示例 3：把 HF 权重喂给 verse_torch 自定义模型

```python
from verse_compat import load_hf_state_dict, nn, Module, Linear

# 假设 model 是 verse_torch.nn.Module 子类，键名与 HF 权重一致
model = MyTransformerLM(config)
state_dict = load_hf_state_dict("./models/my_lm/")

# strict=False 允许部分键未匹配（宽松模式）
model.load_state_dict(state_dict, strict=False)
```

## 设计原则

- **零运行时硬依赖**：`verse_compat` 仅在用户已安装 `safetensors` / `torch` / `huggingface_hub` 时调用其加载器，否则降级到自带的最简实现。这与 Verse 框架"运行时不强制要求 PyTorch"的整体设计一致。
- **键名完全对齐**：返回的 dict 键名与原 PyTorch `state_dict` 完全一致，便于直接喂给 `Module.load_state_dict`。
- **保守的 dtype 转换**：`bfloat16` 回退到 `float32`，因为 NumPy 无原生 bf16，且 VerseTorch 内部默认以 float32 计算。
- **自实现 pickle 解析器的局限**：仅支持常见的 storage 类型；遇到未知类型或非连续 stride 时可能失败。完整支持请安装 `torch`。

## 相关文档

- [PyTorch 迁移笔记](../../verse_data/migration_notes/pytorch_to_versetorch.md) —— 详细列出 `torch.X` → `verse_torch.X` 的对应关系与差异
- [VerseTorch README](../verse_torch/README.md) —— Tensor / nn / autograd 基础
- [VerseInference README](../verse_inference/README.md) —— 推理时的模型加载流程
- [主 README](../../README.md)
