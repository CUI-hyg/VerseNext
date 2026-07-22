# VerseTorch

> 纯 Python / NumPy 实现的张量与自动微分引擎，CPU-first，PyTorch 风格 API，运行时零重型依赖。

[返回主 README](../../README.md)

## 特性

- 纯 NumPy 实现，无 PyTorch / TensorFlow / JAX 运行时依赖
- 动态计算图 + 反向模式 autograd（拓扑排序）
- PyTorch 风格 API（`Tensor` / `nn.Module` / `optim` / `losses`）
- 完整训练栈：`Trainer` + 学习率调度器 + 早停 + 梯度累积 + checkpoint
- 量化支持：INT8 / INT4 (W4A16) / 1.58-bit ternary（BitNet b1.58 风格）
- CPU 并行：基于 `multiprocessing` 的 batch 维度并行
- 模型压缩：剪枝 + LoRA + 蒸馏 + 量化组合管线

### Part3K2 新增能力

- **对齐 PyTorch 的高级优化器**：`Lion` / `Adafactor`（`optim_extras`）
- **高级学习率调度**：`OneCycleLR` / `ReduceLROnPlateau` / `CosineRestartsLR`（`scheduler_extras`）
- **新增激活函数**：`GeGLU` / `Mish` / `SiLU`（`activations`）
- **新型注意力与归一化**：`SlidingWindowAttention` / `ALiBi` / `DeepNorm`
- **并行训练器 `ParallelTrainer`**：chunk 拆分 + 合并策略 + 基于 `val_loss` 的 BUG 修复
- **生成质量打分 `ScoringEvaluator`**：`exact_match` / `prefix_accuracy` / `char_f1` / `bleu` / `rouge_l`
- **`Trainer.inference`**：批量推理生成（支持字符串 prompt 与 token ID 序列）
- **`focal_loss`** + `cross_entropy` 现支持 `ignore_index` / `label_smoothing` 参数

### Part4 新增能力

- **VerseNex 训练工具链（`training_nex.py`）**：4 个训练器 + 2 个数据集，专为 VerseNex 原生架构设计：
  - `VerseNexTrainer`：aux_loss-aware 训练器，自动检测 `forward_with_aux`，`loss = cross_entropy + aux_loss_weight * aux`
  - `LoRATrainer`：LoRA-aware 训练器，自动 `lora_only` 包装 + `merge_lora` 合并回 base
  - `SFTTrainer`：监督微调训练器，chat 数据格式，`ignore_index=-100` 屏蔽非 assistant token
  - `DPOTrainer`：Direct Preference Optimization 训练器，reference model 冻结 + DPO loss
  - `SFTDataset` / `DPODataset`：jsonl 加载 + chat template 渲染
- **`ParallelTrainer` 升级**：自动检测 `forward_with_aux`，启用 aux_loss 路径与 `VerseNexTrainer` 协同
- **MoD Expert 压缩**：`compress_mod_experts` 函数，按 Expert 参数 L2 范数排序，丢弃低利用率 Expert，同步修改 router 权重与 `top_k`

### Part4K1 新增能力

- **GPU/NPU 设备抽象（`device.py` + `backend_torch.py`）**：
  - `DeviceBackend` 抽象基类 + `NumpyBackend`（默认 CPU，零依赖）+ `TorchBackend`（PyTorch 委托，支持 `cuda` / `mps` / `npu`，NPU 通过 `torch_npu` 扩展）
  - **不自研 CUDA kernel**，全部 GPU 算子走 PyTorch 原生（含 `F.scaled_dot_product_attention` fused 路径）
  - `get_backend(device)` 工厂函数 + backend 实例缓存 + 无 torch 环境下安全 import
- **Tensor / Module 设备迁移**：
  - `Tensor.device` 属性 + `.to(device)` / `.cuda()` / `.npu()` / `.cpu()` 方法
  - GPU 下 autograd 委托给 `torch.Tensor` 原生 autograd；CPU 下保持自研 autograd
  - `Module.to(device)` / `Module.device` 递归迁移所有参数与子模块
- **新组件（`nn.py`）**：`RotaryEmbedding` / `KVCache` / `StaticCache` / `DynamicCache` / `GroupNorm` / `Conv1d` / `LayerNorm` 优化版
- **新优化器（`optim.py`）**：`NAdamW`（NAdam + 解耦 weight decay）/ `RMSProp`
- **新损失（`losses.py`）**：`contrastive_loss`（RL/DPO 备选）/ `perplexity`
- **新训练设施（`training.py`）**：`DistributedTrainer` 占位接口（多卡数据并行 API 预留）/ `autocast` 混合精度上下文（GPU 后端，CPU 时 no-op）
- **可选依赖**：`pyproject.toml` 把 `torch` 声明为可选 extra（`pip install -e packages/verse_torch[gpu]`）

## 安装

```bash
# 基础安装（CPU-only，零重型依赖）
pip install -e packages/verse_torch

# GPU/NPU 加速（可选）：安装 PyTorch 后端
pip install -e packages/verse_torch[gpu]      # 等价于额外安装 torch
# NPU（华为昇腾）还需：pip install torch_npu
```

唯一运行时依赖是 `numpy>=1.26`；`matplotlib` 为可选依赖（缺失时 `plot_loss_curve` 自动降级为 ASCII 曲线）。`torch` 为 Part4K1 新增的可选依赖（仅在使用 `--device cuda/npu` 或 `autocast` 时需要）。

## 快速开始

### 1. 最小 autograd 示例

```python
import numpy as np
from verse_torch import Tensor

# 构造可微张量
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()         # y = x^2 的和 = 14
y.backward()              # 反向传播

print(y.data)             # 14.0
print(x.grad)             # [2., 4., 6.]  即 dy/dx = 2x
```

### 2. 一个 `nn.Module` 训练示例

```python
import numpy as np
from verse_torch import Tensor, nn, optim, losses

# 简单二分类：3 维输入 -> 1 维 logit
model = nn.Linear(3, 1)
opt = optim.SGD(model.parameters(), lr=0.1)

x = Tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]], requires_grad=True)
y = Tensor([[0.0], [1.0]])

for step in range(50):
    opt.zero_grad()
    logits = model(x)
    loss = losses.mse_loss(logits, y)
    loss.backward()
    opt.step()
print("final loss:", float(loss.data))
```

### 3. 完整训练循环（Trainer + TransformerLM）

```python
import numpy as np
from verse_torch import nn, optim, training

# 小型 Transformer 语言模型
model = nn.TransformerLM(
    vocab_size=128, n_layer=2, n_head=4, n_embd=64, seq_len=32
)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = optim.LambdaLR(
    optimizer, optim.warmup_cosine_lr(warmup_steps=20, total_steps=200)
)

# 假造的 (x, y) batch：x 是 token id 序列，y 是右移一位的目标
def make_loader(n_batches, batch=4, seq_len=32):
    for _ in range(n_batches):
        x = np.random.randint(0, 128, size=(batch, seq_len))
        y = np.roll(x, -1, axis=1)
        yield x, y

cfg = dict(
    max_steps=200, eval_interval=20, patience=5,
    save_dir="./ckpt", grad_accum=2, log_interval=20,
)
# 注意：val_loader 会被多次迭代（每次评估都遍历一遍），
# 所以要传可复用的容器（list），不能传一次性 generator。
val_batches = list(make_loader(8))
trainer = training.Trainer(
    model, train_loader=make_loader(40), val_loader=val_batches,
    optimizer=optimizer, scheduler=scheduler, cfg=cfg,
)
train_losses, val_losses = trainer.fit()
```

## 模块详解

### tensor — 张量与自动微分

`Tensor` 是核心数据结构，内部包装 NumPy `ndarray` 并维护 `data` / `grad` / `requires_grad` 三个属性。所有运算符都会构建计算图节点（`_children` + `_backward` 闭包），`backward()` 通过 DFS 拓扑排序逆序回传梯度。

**构造与工厂方法**

```python
Tensor(data, requires_grad=False)
Tensor.zeros(*shape)           Tensor.ones(*shape)
Tensor.rand(*shape)            Tensor.randn(*shape)
Tensor.full(shape, val)        Tensor.arange(start, end, step)
Tensor.eye(n)
```

**元素级运算**：`+ - * / **`、`neg`、`exp`、`log`、`sqrt`、`relu`、`gelu`、`sigmoid`、`tanh`、`silu`、`abs`、`maximum`、`minimum`、`clamp`、`softmax(dim=-1)`、`log_softmax(dim=-1)`。

**形状操作**：`reshape`、`view`、`transpose(dim0, dim1)`、`permute(*dims)`、`squeeze`、`unsqueeze`、`expand`、`broadcast_to`、`contiguous`、`flatten(start_dim, end_dim)`、`__getitem__`（支持 int / slice / tuple / boolean mask 索引，反向为 `np.add.at` scatter）。

**归约**：`sum(dim, keepdim)`、`mean`、`max`、`min`、`argmax(dim)`、`norm(p=2)`、`var(unbiased=True)`、`std`。

**矩阵乘法**：`a @ b` / `a.matmul(b)`，支持 2D 与 batched 3D（自动广播 batch 维）。

**反向传播**：标量 `Tensor.backward()` 触发整张图的反向；非标量需显式传入 `grad`。

**梯度上下文**：`no_grad()` / `enable_grad()` / `set_grad_enabled(mode)` / `is_grad_enabled()`。

**设备迁移（Part4K1）**：`Tensor.device`（属性，返回 `"cpu"` / `"cuda"` / `"npu"` / `"mps"`）、`Tensor.to(device)`、`Tensor.cuda(device=None)`、`Tensor.npu(device=None)`、`Tensor.cpu()`。CPU 后端保持自研 autograd（`data` 为 `np.ndarray`）；GPU/NPU 后端 `data` 切换为 `torch.Tensor`，反向传播委托 `torch.Tensor.backward()`，所有算子经 `DeviceBackend` 转发。无 PyTorch 时调用 `.cuda()` / `.npu()` 抛 `RuntimeError`，`.cpu()` 与 `.to("cpu")` 始终可用。

**`Parameter`**：`Parameter = Tensor` 别名（按 PyTorch 习惯，通过 `requires_grad=True` 标识为可训练参数）。

```python
from verse_torch import Tensor, no_grad

a = Tensor.randn(3, 3, requires_grad=True)
b = Tensor.randn(3, 3, requires_grad=True)
with no_grad():
    c = a @ b          # 不构建计算图
out = (a @ b).sum()
out.backward()
print(a.grad.shape)    # (3, 3)
```

### device — 设备抽象与后端工厂（Part4K1）

`device.py` 定义设备抽象层，把"算子在哪种设备上执行"与"算子怎么算"解耦。CPU-first 引擎默认走 `NumpyBackend`；当用户安装 PyTorch 并请求 `cuda` / `mps` / `npu` 时，工厂延迟导入 `TorchBackend`，所有算子委托给 `torch`，CUDA kernel 走 PyTorch 原生实现（**不自研 kernel**）。

| 组件 | 作用 |
|---|---|
| `DeviceBackend`（abc.ABC） | 抽象基类，定义 `matmul` / `linear` / `softmax` / `layernorm` / `rmsnorm` / `rope` / `attention` 等算子接口与只读 `device_type` 属性 |
| `NumpyBackend` | 默认 CPU 后端，所有算子用 NumPy 实现，与 `Tensor` 自研 autograd 路径行为一致 |
| `TorchBackend`（见 `backend_torch.py`） | PyTorch 委托后端，支持 `cuda` / `mps` / `npu`（NPU 经 `torch_npu` 扩展），算子委托 `torch.Tensor`，attention 优先走 `F.scaled_dot_product_attention` fused 路径 |
| `get_backend(device=None)` | 工厂函数：按 device 字符串返回 backend 实例（带缓存）；非 CPU 设备需 torch 可用，NPU 还需 `torch_npu` |
| `has_torch()` / `has_torch_npu()` | 检测 PyTorch / torch_npu 是否可用（模块级缓存，避免重复 import） |
| `DEFAULT_DEVICE` | 模块级常量 `"cpu"`（CPU-first 默认） |
| `_parse_device(device)` / `is_cpu_device(device)` | 设备字符串规范化（`"cuda:0"` → `"cuda"`）与 CPU 判断 |

**设备字符串**：`"cpu"` / `"cuda"` / `"cuda:0"` / `"cuda:1"` / `"mps"` / `"npu"` / `"npu:0"`。`None` 等价于 `"cpu"`。

**向后兼容**：无 PyTorch 环境下 `device.py` 仍可正常 import；只有调用 `get_backend("cuda")` 等请求非 CPU 设备时才抛 `RuntimeError("未安装 PyTorch，无法使用 device 'cuda'（仅支持 CPU）")`。详见 [ADR-005 GPU/NPU 后端抽象](../../docs/architecture/adr-005-gpu-npu-backend.md)。

```python
from verse_torch.device import get_backend, has_torch, DEFAULT_DEVICE

print(DEFAULT_DEVICE)               # "cpu"
print(has_torch())                  # False（未装 torch）或 True

backend = get_backend("cpu")        # NumpyBackend 实例
out = backend.matmul(a, b)          # 等价于 np.matmul(a, b)
out = backend.softmax(x, dim=-1)    # 数值稳定 softmax

# GPU 路径（需 torch）：
# backend = get_backend("cuda:0")   # TorchBackend 实例
# out = backend.attention(q, k, v)  # 走 F.scaled_dot_product_attention fused kernel
```

### backend_torch — PyTorch 委托后端（Part4K1）

`backend_torch.py` 仅在 PyTorch 可用时被 `device.get_backend` 延迟导入，避免 `device.py` 硬依赖 `torch`。提供 `TorchBackend` 类、`autocast` 上下文管理器与 `to_torch` / `to_numpy` 转换工具。

**`TorchBackend(device="cuda")`**：继承 `DeviceBackend`，构造时把字符串 device 解析为 `torch.device` 实例；算子委托 `torch.matmul` / `F.linear` / `torch.softmax` / `F.layer_norm` / 自实现 RMSNorm / GPT-NeoX 风格 RoPE / `F.scaled_dot_product_attention`（mask 形状不兼容时回退手动 softmax）。`from_numpy(x)` 把 `ndarray` / 标量 / `torch.Tensor` 迁到本后端 device；`to_numpy(t)` 反向转换（detach + cpu + numpy）。

**`autocast(device=None, dtype=None, enabled=True)`**：fp16 混合精度上下文管理器。GPU（cuda / mps / npu）下启用 `torch.autocast`（默认 `torch.float16`）；CPU 下为 no-op；无 PyTorch 同样 no-op。用法对齐 `torch.autocast`。

**`to_torch(ndarray, device="cpu", dtype=None)` / `to_numpy(torch_tensor)`**：`ndarray` ↔ `torch.Tensor` 互转，处理 dtype 映射与 device 迁移。

```python
from verse_torch.backend_torch import autocast, to_torch, to_numpy

# 混合精度训练（GPU 时启用 fp16，CPU 时 no-op）
with autocast(device="cuda", enabled=True):
    out = model(x)               # 内部走 torch.autocast

# ndarray -> torch.Tensor（GPU）
t = to_torch(np_array, device="cuda:0", dtype=np.float32)

# torch.Tensor -> ndarray（自动 detach + cpu）
arr = to_numpy(t)
```

### nn — 神经网络层

**`Module` 基类**：通过 `__setattr__` 自动注册 `Tensor` 参数与子 `Module`。
关键方法：`parameters()`、`named_parameters()`、`modules()` / `named_modules()`、`children()`、`state_dict()`、`load_state_dict(sd, strict=True)`、`train()` / `eval()`、`zero_grad()`、`apply(fn)`、`to(device)`（Part4K1，递归迁移所有参数与子模块到目标 device）、`device`（Part4K1 只读属性，返回首个参数所在设备或 `"cpu"`）。

**基础层**：`Linear(in, out, bias=True)`、`Embedding(num, dim)`、`LayerNorm(shape, eps=1e-5)`、`RMSNorm(shape, eps=1e-6)`、`Dropout(p=0.5)`、`Sequential(*modules)`、`ModuleList(list)`。

**LM 组件**：
- `SwiGLUMLP(d, dropout=0.0, hidden_multiple=4, align=64)`：SwiGLU 激活 + 2/3 缩放对齐到 `align`。
- `GQASelfAttention(d, n_head, n_kv_head=None, dropout=0.0)`：GQA + 内置 RoPE（预计算 32768 长度 cos/sin 表）+ 因果掩码 + KV cache，`forward(x, kv_cache=None) -> (out, new_kv_cache)`。
- `TransformerBlock(d, n_head, n_kv_head=None, dropout=0.0)`：pre-norm 残差结构（`RMSNorm` → attn → +x → `RMSNorm` → MLP → +x）。
- `TransformerLM(vocab_size, n_layer, n_head, n_embd, seq_len=128, dropout=0.1, n_kv_head=None, tie_weights=True)`：含 token embedding、`n_layer` 个 block、最终 `RMSNorm` 与共享权重的 `head`，初始化按 GPT-2 风格（`normal_(std=0.02)` + 残差分支 `1/sqrt(2*n_layer)` 缩放）。
- `repeat_kv(x, n_rep)`：把 KV head 复制到 query head 数（GQA 工具）。

**初始化辅助**（原地修改）：`kaiming_uniform_(t, a=0, mode="fan_in", nonlinearity="leaky_relu")`、`xavier_uniform_(t, gain=1.0)`、`normal_(t, mean=0, std=1)`、`zeros_(t)`、`ones_(t)`、`uniform_(t, low=0, high=1)`。

#### 新组件（Part4K1）

`nn.py` 在 Part4K1 补齐了与 PyTorch / HF 对齐的关键组件，支撑 VerseNex 原生架构与 KV cache 推理路径：

| 类 | 说明 |
|---|---|
| `RotaryEmbedding(dim, max_seq_len=2048, base=10000.0)` | 独立 RoPE 模块。预计算 `(max_seq_len, dim)` 的 `cos` / `sin` 表，`forward(x, position_ids=None)` 对 `x` 后两维做 GPT-NeoX 风格 `rotate_half`。与 `GQASelfAttention` 内置 RoPE 等价，但可作为独立 `nn.Module` 嵌入任意注意力实现 |
| `KVCache`（抽象基类） | KV cache 统一接口：`update(k, v)` / `get()` / `reset()` / `__len__`。子类实现具体存储策略 |
| `StaticCache(batch_size, n_kv_head, max_seq_len, head_dim)` | 预分配固定大小 cache（`np.zeros`），按位置写入；适合 batch_size 与 max_seq_len 已知场景，零分配开销 |
| `DynamicCache()` | 动态增长的 cache，每次 `update` 沿 seq 维度 `concat`；适合变长生成场景，与 HF `DynamicCache` 行为对齐 |
| `GroupNorm(num_groups, num_channels, eps=1e-5)` | Group Normalization：把 channels 拆成 `num_groups` 组，每组独立计算 mean/var 归一化后仿射。常用于 Conv / diffusion 模型 |
| `Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)` | 1D 卷积层（NumPy 实现，autograd 兼容），用于 Mamba / SSM 的 causal depthwise conv |
| `LayerNorm`（优化版） | 原有 `LayerNorm` 在 Part4K1 做了数值稳定性优化：方差计算用 unbiased=False（与 PyTorch 默认一致）、eps 加在 sqrt 外、`weight` / `bias` 默认值与形状校验对齐 PyTorch |

```python
from verse_torch import nn, Tensor
import numpy as np

# 独立 RoPE
rope = nn.RotaryEmbedding(dim=64, max_seq_len=512)
x = Tensor(np.random.randn(2, 32, 64).astype(np.float32))
x_rot = rope(x)                    # (2, 32, 64)，已应用 RoPE

# DynamicCache（变长生成）
cache = nn.DynamicCache()
cache.update(k_step, v_step)       # 单步写入
k_full, v_full = cache.get()       # 取累积 KV
cache.reset()                      # 清空

# StaticCache（定长 batch 推理）
static = nn.StaticCache(batch_size=4, n_kv_head=8, max_seq_len=512, head_dim=64)

# GroupNorm
gn = nn.GroupNorm(num_groups=4, num_channels=64)
out = gn(x)                        # (2, 32, 64)

# Conv1d（Mamba causal conv）
conv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
```

#### 新型注意力与归一化（Part3K2）

| 类 | 说明 |
|---|---|
| `SlidingWindowAttention(n_embd, n_head, window_size, n_kv_head=None, dropout=0.1)` | 滑动窗口注意力。每个 query 仅 attend 前 `window_size` 个 key，配合 causal mask：`mask[i,j]=0` 当且仅当 `j<=i` 且 `i-j<window_size`，其余位置 mask 为 `-inf`。支持 GQA（`n_kv_head < n_head`）。参考 Longformer / Mistral。 |
| `ALiBi(n_head, max_seq_len=2048)` | Attention with Linear Biases 位置偏置。不学习位置嵌入，直接在 attention scores 上加线性偏置 `bias[i,j] = -m_h*(i-j)`（causal），斜率 `m_h = 1/2^(h/n_head)` 按几何级数生成。`forward(qk_scores)` 接受 `(B, n_head, T_q, T_k)` 返回加偏置后的 scores；预计算 `(n_head, max_seq_len, max_seq_len)` bias 表，支持 KV cache 场景（`T_q != T_k`）。论文: https://arxiv.org/abs/2108.12409 |
| `DeepNorm(normalized_shape, alpha=1.0, eps=1e-5)` | DeepNorm 归一化：`DeepNorm(x) = LayerNorm(x * alpha) + x`。`alpha` 通常取 `(2*N)^(1/4)`（N 为 Transformer 层数），`alpha` 越大残差分支权重越大、训练越稳定（可训练上千层）。内部复用 `LayerNorm` 的 `gamma` / `beta` 参数。论文: https://arxiv.org/abs/2203.00555 |

```python
from verse_torch import nn, Tensor
import numpy as np

# 滑动窗口注意力（长上下文场景）
swa = nn.SlidingWindowAttention(n_embd=64, n_head=8, window_size=128, n_kv_head=4)
x = Tensor(np.random.randn(2, 32, 64).astype(np.float32))
out = swa(x)            # (2, 32, 64)

# ALiBi 位置偏置（在已有 attention scores 上叠加）
alibi = nn.ALiBi(n_head=8, max_seq_len=512)
scores = Tensor(np.random.randn(2, 8, 32, 32).astype(np.float32))
scores = alibi(scores)  # 同形状，已加 ALiBi 偏置

# DeepNorm：深层 Transformer 残差归一化
dn = nn.DeepNorm(normalized_shape=64, alpha=(2 * 12) ** 0.25)  # 12 层
out = dn(x)             # LayerNorm(x * alpha) + x
```

#### 扩展激活函数（activations）

`activations` 子模块提供 3 个 `nn.Module` 子类，可直接用于 `Sequential` / `ModuleList`，底层算子复用 `Tensor` 的可微方法（`exp` / `log` / `tanh` / `sigmoid` / `__mul__`），自动获得 autograd 支持。

| 类 | 公式 | 说明 |
|---|---|---|
| `SiLU()` | `x * sigmoid(x)` | SiLU / Swish 激活，与 `Tensor.silu()` 等价但封装为 Module |
| `Mish()` | `x * tanh(softplus(x))` | Mish 激活，`softplus(x) = log(1 + exp(x))`，数值稳定 |
| `GeGLU()` | `a * gelu(b)`（沿最后一维 split `(a, b)`） | GeGLU 激活，输入 `(..., 2*d)` 输出 `(..., d)`，GELU 用 `x * sigmoid(1.702 * x)` 近似（误差 < 0.001），与 SwiGLU 类似但用 GELU 代替 SiLU |

```python
from verse_torch import nn, activations, Tensor
import numpy as np

mlp = nn.Sequential(
    nn.Linear(64, 128),
    activations.GeGLU(),    # 输入 128 → 输出 64（沿最后一维 split 后 a * gelu(b)）
)
x = Tensor(np.random.randn(4, 64).astype(np.float32))
out = mlp(x)               # (4, 64)

# 单独使用
silu = activations.SiLU()
mish = activations.Mish()
```

```python
from verse_torch import nn

model = nn.TransformerLM(
    vocab_size=256, n_layer=4, n_head=8, n_embd=128,
    seq_len=128, dropout=0.1, n_kv_head=4,  # GQA：4 个 KV head
)
print("参数量:", sum(p.data.size for p in model.parameters()))
# 前向
import numpy as np
idx = np.random.randint(0, 256, size=(2, 32))
logits = model(idx)            # (2, 32, 256)
```

### optim — 优化器与调度器

**`Optimizer`**：基类，提供 `zero_grad()` / `step()` 接口；构造时接受 `params` 生成器、列表或 `Module`（自动调用 `.parameters()`）。

**优化器**：
- `SGD(params, lr=1e-2, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)` — 标准 momentum + 可选 Nesterov。
- `Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)` — 耦合式 weight decay（L2）。
- `AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` — 解耦 weight decay。
- `NAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`（Part4K1）— NAdam + 解耦 weight decay：用 Nesterov momentum 更新（先按 momentum 预测下一位置再计算梯度），其余与 AdamW 一致。在 Transformer / SSM 训练上常比 AdamW 稍稳定。
- `RMSProp(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False)`（Part4K1）— RMSProp：用指数加权 moving average of squared gradients 自适应学习率；`momentum>0` 加 RMSProp momentum；`centered=True` 用 centered variance（减去 mean of squared gradients）。

**学习率调度器**（基类 `LRScheduler(optimizer, last_epoch=-1)`，每 `step()` 后修改 `optimizer.lr`）：
- `StepLR(opt, step_size, gamma=0.1)` — 每 `step_size` 步乘以 `gamma`。
- `ExponentialLR(opt, gamma=0.99)` — 每步乘以 `gamma`。
- `CosineAnnealingLR(opt, T_max, eta_min=0.0)` — 余弦退火到 `eta_min`。
- `LambdaLR(opt, lr_lambda)` — 自定义 `lr = base_lr * lr_lambda(step)`。

**`warmup_cosine_lr(warmup_steps, total_steps)`**：返回一个闭包，可直接传给 `LambdaLR`，前 `warmup_steps` 线性升温、之后余弦衰减到 0。

```python
from verse_torch import optim

opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
sched = optim.LambdaLR(opt, optim.warmup_cosine_lr(warmup_steps=50, total_steps=1000))

for step in range(1000):
    train_step()
    sched.step()                 # 自动更新 opt.lr
```

#### 高级优化器（optim_extras，Part3K2）

`optim_extras` 子模块对齐 PyTorch 生态，提供两个高级优化器，均继承 `optim.Optimizer` 基类，复用 `zero_grad` / `param_groups` / `state` 机制，并通过 `self.lr` 与 `LRScheduler` 兼容。

| 优化器 | 签名 | 说明 |
|---|---|---|
| `Lion` | `Lion(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)` | Lion-eats-AdamW。**无二阶矩**（节省约 50% 优化器状态内存），用 `sign(m·β1 + g·(1-β1))` 作为更新方向，解耦 weight decay。`lr` 通常比 AdamW 小 3-10x。论文: https://arxiv.org/abs/2302.06675 |
| `Adafactor` | `Adafactor(params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30, eps2=1e-3, weight_decay=0.0)` | Factored 二阶矩：2D+ 参数用行/列统计近似二阶矩，把 `W (m,n)` 的二阶矩内存从 `O(mn)` 降到 `O(m+n)`；1D 参数（bias/norm）走 AdaGrad 风格。含 trust ratio clipping。`lr=None` 时默认 `1e-3`。论文: https://arxiv.org/abs/1804.04235 |

更新规则（Lion）：

```
update = m * β1 + g * (1 - β1)
p = p - lr * sign(update)
if weight_decay != 0:
    p = p - lr * weight_decay * p
m = m * β2 + g * (1 - β2)
```

```python
from verse_torch import nn, optim_extras

model = nn.TransformerLM(vocab_size=128, n_layer=2, n_head=4, n_embd=64)

# Lion：省内存、lr 比 AdamW 小 3-10x
opt_lion = optim_extras.Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)

# Adafactor：factored 二阶矩，适合大模型
opt_adafactor = optim_extras.Adafactor(model.parameters(), lr=1e-3, beta1=0.9, beta2=0.999)

for step in range(100):
    opt_lion.zero_grad()
    loss = ...           # 你的 loss
    loss.backward()
    opt_lion.step()
```

#### 高级学习率调度器（scheduler_extras，Part3K2）

`scheduler_extras` 子模块提供 3 个高级调度器。`OneCycleLR` / `CosineRestartsLR` 继承 `optim.LRScheduler` 基类（无参 `step()`）；`ReduceLROnPlateau` 不继承基类（需外部传入 metric）。

| 调度器 | 签名 | 说明 |
|---|---|---|
| `OneCycleLR` | `OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.25, div_factor=25.0, final_div_factor=1e4)` | 1cycle super-convergence。前 `pct_start` 比例步从 `max_lr/div_factor` 线性升到 `max_lr`，后段从 `max_lr` 余弦退火到 `max_lr/(div_factor*final_div_factor)`。论文: https://arxiv.org/abs/1708.07120 |
| `ReduceLROnPlateau` | `ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0, threshold=1e-4)` | 按 metric 自适应降 lr。监控指标连续 `patience` 次 epoch 无显著改善时，每个 param_group 的 lr 乘以 `factor`（不低于 `min_lr`）。`step(metric)` 接受当前 epoch 的 metric（如 `val_loss`）。 |
| `CosineRestartsLR` | `CosineRestartsLR(optimizer, T_0, T_mult=1, eta_min=0)` | SGDR 带热重启。每个周期内 lr 从 `base_lr` 余弦退火到 `eta_min`，周期结束后重新升温到 `base_lr`；周期长度按 `T_mult` 倍增：第 k 个周期长度 = `T_0 * T_mult^k`。论文: https://arxiv.org/abs/1608.03983 |

```python
from verse_torch import nn, optim, optim_extras, scheduler_extras

model = nn.TransformerLM(vocab_size=128, n_layer=2, n_head=4, n_embd=64)

# 1) OneCycleLR：super-convergence
opt = optim.AdamW(model.parameters(), lr=3e-4)
sched = scheduler_extras.OneCycleLR(opt, max_lr=3e-3, total_steps=1000, pct_start=0.3)
for step in range(1000):
    train_step()
    sched.step()

# 2) ReduceLROnPlateau：按 val_loss 自适应降 lr（注意要传 metric）
opt2 = optim.AdamW(model.parameters(), lr=1e-3)
sched2 = scheduler_extras.ReduceLROnPlateau(opt2, mode='min', factor=0.1, patience=5)
for epoch in range(n_epochs):
    train_one_epoch()
    val_loss = evaluate()
    sched2.step(val_loss)        # 传入当前 metric

# 3) CosineRestartsLR：SGDR 热重启
opt3 = optim.AdamW(model.parameters(), lr=1e-3)
sched3 = scheduler_extras.CosineRestartsLR(opt3, T_0=100, T_mult=2, eta_min=0.0)
for step in range(1000):
    train_step()
    sched3.step()
```

### losses — 损失函数

所有损失返回**标量 `Tensor`**，`requires_grad` 自动传播，可直接 `backward()`。

| 函数 | 公式 / 用途 |
|---|---|
| `cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0)` | softmax 交叉熵，`logits: (N, C)` 或 `(B, T, V)`（自动 reshape），`targets: (N,)` 或 `(B, T)` int。**Part3K2**：新增 `ignore_index`（屏蔽 padding，默认 -100，`None` 表示不屏蔽）与 `label_smoothing`（标签平滑系数，`>0` 时 `loss = (1-ε)·CE_hard + ε·CE_uniform`）。与 PyTorch / HF 行为对齐 |
| `nll_loss(log_probs, targets)` | 负对数似然，输入已是 `log_softmax` 结果 |
| `binary_cross_entropy(pred, target)` | BCE，输入为概率（已 sigmoid） |
| `binary_cross_entropy_with_logits(logits, target)` | BCE，输入为 logits（数值稳定版） |
| `mse_loss(pred, target)` | 均方误差 |
| `l1_loss(pred, target)` | 平均绝对误差 |
| `kl_div_loss(log_probs, target_probs)` | KL 散度 `sum(t*(log t - log_probs))` |
| `focal_loss(logits, targets, gamma=2.0, alpha=0.25, ignore_index=-100, label_smoothing=0.0)` | **Part3K2**。Focal Loss `FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)`，类别不均衡场景使用。`gamma` 越大对易分样本抑制越强；`alpha` 是类别平衡因子（`alpha=1.0` 退化为无加权）。支持 `ignore_index` mask 与 `label_smoothing`。论文: https://arxiv.org/abs/1708.02002 |
| `contrastive_loss(anchor, positive, negative, temperature=0.07, margin=0.0)` | **Part4K1**。对比学习 / RL 偏好学习损失（InfoNCE 风格）：`-log(exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + exp(sim(a,n)/τ)))`，`sim` 为点积或余弦相似度。`margin>0` 时加 triplet margin 项。可用于 RL/DPO 候选对训练、句嵌入对比学习 |
| `perplexity(logits, targets, ignore_index=-100)` | **Part4K1**。困惑度 `PPL = exp(CE)`，先调 `cross_entropy` 算平均 loss 再 `exp`。`ignore_index` 屏蔽 padding。返回标量 `Tensor`（非 loss 张量，仅供评估） |

```python
from verse_torch import Tensor, losses

logits = Tensor([[2.0, 1.0, 0.1],
                 [0.1, 2.0, 1.0]], requires_grad=True)
targets = [0, 1]
loss = losses.cross_entropy(logits, targets)
loss.backward()
print(logits.grad.shape)   # (2, 3)
```

**Part3K2：`ignore_index` + `label_smoothing` + `focal_loss` 用法**

```python
from verse_torch import Tensor, losses
import numpy as np

# (B=2, T=4, V=5) 的 logits，targets 含 padding（-100）
logits = Tensor(np.random.randn(2, 4, 5).astype(np.float32), requires_grad=True)
targets = np.array([[1, 2, -100, -100],   # 后两位是 padding
                    [0, 3, 2, -100]])

# 1) cross_entropy：ignore_index 屏蔽 padding + label_smoothing=0.1
loss = losses.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.1)
loss.backward()

# 2) focal_loss：类别不均衡场景，gamma=2.0 alpha=0.25（原论文默认）
loss_fl = losses.focal_loss(logits, targets, gamma=2.0, alpha=0.25,
                            ignore_index=-100)
loss_fl.backward()
```

### training — 训练基础设施

提供训练循环常用工具与高层 API，仅依赖 NumPy + 标准库。

- **`cross_entropy_loss(logits, targets, ignore_index=-100)`**：支持 `(B, T, V)` / `(N, V)` 自动 reshape，`ignore_index` 位置不计入 loss 与梯度（用于 padding token）。
- **`EarlyStopping(patience, min_delta=0.0)`**：`__call__(val_loss) -> bool`，连续 `patience` 次未显著下降则触发停止；`reset()` 重置状态。
- **`GradientAccumulator(micro_batch, effective_batch)`**：`accum_steps = effective_batch // micro_batch`；每次反向后 `step()`，`should_step()` 为 True 时执行 `optimizer.step()` 并自动重置。
- **`CheckpointManager(save_dir, best_path=None, last_path=None)`**：`save_best(state)` / `save_last(state)` / `load_best()` / `load_last()`，内部用 `pickle`，自动处理 `Tensor` 与 `ndarray` 序列化。
- **`compute_loss_rate(loss_window, window=50, min_delta=1e-4)`**：滑动窗口 loss 下降率（前半均值 - 后半均值）/ 前半均值，数据不足或已收敛返回 0.0。
- **`plot_loss_curve(train_losses, val_losses, save_path, eval_interval=1)`**：matplotlib 可用时输出 PNG（蓝实线 train + 红虚线 val），否则降级为 ASCII 文本图（保存到 `.txt`）。
- **`Trainer(model, train_loader, val_loader, optimizer, scheduler=None, cfg=None)`**：端到端训练器。

**`Trainer` 的 `cfg` 字段**（dict 或 dataclass 均可）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `max_steps` | 100 | 最大训练步数 |
| `eval_interval` | 10 | 每 N 步评估一次 |
| `patience` | 10 | 早停容忍轮数 |
| `save_dir` | `"./checkpoints"` | 检查点保存目录 |
| `grad_accum` | 1 | 梯度累积步数 |
| `log_interval` | 10 | 日志打印间隔 |
| `loss_rate_window` | 50 | loss 下降率滑动窗口 |

`Trainer` 会自动调用 `cross_entropy_loss`、`EarlyStopping`、`GradientAccumulator`、`CheckpointManager` 与 `plot_loss_curve`；`fit()` 返回 `(train_losses, val_losses)`，并落盘 `loss_history.json` 与 `loss_curve.png/.txt`。

```python
from verse_torch import nn, optim, training

model = nn.TransformerLM(vocab_size=128, n_layer=2, n_head=4, n_embd=64)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

def loader(n):
    import numpy as np
    for _ in range(n):
        x = np.random.randint(0, 128, size=(4, 16))
        y = np.roll(x, -1, axis=1)
        yield x, y

trainer = training.Trainer(
    model=model,
    train_loader=loader(40),
    val_loader=loader(8),
    optimizer=optimizer,
    cfg=dict(max_steps=100, eval_interval=20, patience=3,
             save_dir="./ckpt", grad_accum=1, log_interval=20),
)
train_losses, val_losses = trainer.fit()
best = trainer.checkpoint.load_best()        # 加载最佳权重
model.load_state_dict(best["model_state_dict"])
```

#### ParallelTrainer — 并行训练器（Part3K2）

`ParallelTrainer(model, train_dataset, val_dataset, optimizer_cls=None, optimizer_kwargs=None, cfg=None, loss_fn=None, collate_fn=None, checkpoint_mgr=None)`：把 `max_steps` 拆成 N 个 chunk 并行训练（CPU 串行实现，接口对齐并行），训练完后按 `train_loss + val_loss` 排序（**差的前、好的后**）串行重训，最后整体 fine-tune 若干步。

**关键设计**：
- `parallel_chunks` 拆分步数：`max_steps` 均分到 N 个 chunk（余数均摊到前几个），每个 chunk 用独立 `Trainer` 实例从同一初始状态训练。
- **合并策略**：chunk 训练完后按 `train_loss + val_loss` 排序，loss 大的（效果差）放前面、loss 小的（效果好）放后面，串行重训（每个 chunk 再训练 `chunk_steps // 4` 步）。
- **`_eval_full_val(model)`**：基于**完整** val 数据集更新 `val_loss`（修复旧实现只用单 batch 估算的漏洞——不同 chunk batch 不可比、单 batch 方差大）。chunk 内 `Trainer` 用 `BatchLoader` 包装 dataset。
- `merge_finetune_steps`：最后整体 fine-tune 若干步（默认 `max_steps // 10`）。
- 训练完成后加载最佳 `state_dict` 到 `self.model`，若提供 `checkpoint_mgr` 则保存 best。

**`cfg` 字段**（在 `Trainer.cfg` 基础上新增）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `parallel_chunks` | 4 | chunk 数量 N |
| `batch_size` | 8 | batch 大小 |
| `lr` | 0.003 | 学习率 |
| `warmup` | 20 | warmup 步数 |
| `grad_clip` | 0.0 | 梯度裁剪阈值 |
| `label_smoothing` | 0.0 | 标签平滑系数 |
| `merge_finetune_steps` | `max_steps // 10` | 整体 fine-tune 步数 |
| `seed` | 42 | 随机种子 |

```python
import numpy as np
from verse_torch import nn, training

model = nn.TransformerLM(vocab_size=128, n_layer=2, n_head=4, n_embd=64)

# 简易 dataset：实现 __getitem__ 与 __len__
class ToyDataset:
    def __init__(self, n):
        self.x = np.random.randint(0, 128, size=(n, 16))
        self.y = np.roll(self.x, -1, axis=1)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

trainer = training.ParallelTrainer(
    model=model,
    train_dataset=ToyDataset(200),
    val_dataset=ToyDataset(40),          # 必须传完整 val 数据集
    cfg=dict(parallel_chunks=4, max_steps=200, batch_size=8,
             lr=3e-3, warmup=20, eval_interval=20,
             merge_finetune_steps=20, seed=42),
)
history = trainer.fit()                   # {"train_loss": [...], "val_loss": [...], "steps": [...]}
print("best_val_loss:", trainer.best_val_loss)
```

#### Trainer.inference — 批量推理生成（Part3K2）

`Trainer.inference(prompts, temperature=1.0, top_k=None, top_p=None, max_tokens=30)`：批量推理生成，支持三种输入模式：

1. **字符串 prompt + tokenizer**：先用 `tokenizer.encode` 转 token ID，调用 `model.generate` 生成，再用 `tokenizer.decode` 转回字符串（返回 `list[str]`）。需要 `trainer.tokenizer = tok` 挂载 tokenizer。
2. **字符串 prompt + 无 tokenizer**：原样传给 `model.generate`，返回 `list[str]`。
3. **token ID 序列**（list / np.ndarray / Tensor）：直接传给 `model.generate`，返回 `list[list[int]]`（每条 prompt 对应的完整 ID 序列，含原始 prompt + 新生成部分）。

要求模型实现 `generate` 方法，否则抛 `NotImplementedError`。内部用 `no_grad()` 包裹，自动切到 `eval` 模式。

```python
from verse_torch import nn, optim, training

model = nn.TransformerLM(vocab_size=128, n_layer=2, n_head=4, n_embd=64)
# ... 训练 model ...

trainer = training.Trainer(
    model=model, train_loader=..., val_loader=...,
    optimizer=optim.AdamW(model.parameters(), lr=1e-3), cfg=dict(max_steps=1),
)
# 可选：挂载 tokenizer（字符串 prompt 模式需要）
# trainer.tokenizer = my_tokenizer

# token ID 序列模式（向后兼容）
import numpy as np
prompts = [np.random.randint(0, 128, size=(8,))]   # 每条 8 个 token
results = trainer.inference(prompts, temperature=1.0, top_k=5, max_tokens=30)
# results: list[list[int]]，每条是完整 ID 序列
```

#### DistributedTrainer + autocast — 分布式与混合精度（Part4K1）

`training.py` 在 Part4K1 预留了多卡数据并行 API 与 GPU 混合精度支持：

| 组件 | 说明 |
|---|---|
| `DistributedTrainer(model, world_size=1, rank=0, backend="gloo", ...)` | 多卡数据并行训练器**占位接口**。当前实现为单进程 fallback（参数 `world_size` / `rank` 仅作 API 预留，向后兼容 `Trainer` 行为）。未来接入 `torch.distributed` 后将启用真实 DDP 路径：每张卡持有完整模型副本，按 `DistributedSampler` 切分 batch，反向后 all-reduce 梯度 |
| `autocast(device=None, dtype=None, enabled=True)` | fp16 混合精度上下文（重导出自 `backend_torch`）。GPU（cuda / mps / npu）下启用 `torch.autocast`，CPU 下为 no-op，无 PyTorch 同样 no-op。配合 `GradScaler` 使用时需自行在 optimizer.step 前 unscale（当前未内置 scaler） |

```python
from verse_torch import nn, optim, training
from verse_torch.backend_torch import autocast

model = nn.TransformerLM(vocab_size=256, n_layer=4, n_head=8, n_embd=128)
model = model.to("cuda")               # 迁移到 GPU（需 torch）
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# 混合精度训练循环
for step in range(1000):
    optimizer.zero_grad()
    with autocast(device="cuda", enabled=True):
        logits = model(x)              # fp16 前向
        loss = training.cross_entropy_loss(logits, y)
    loss.backward()                    # 反向走 torch autograd
    optimizer.step()

# 分布式占位接口（当前 fallback 为单进程）
dist_trainer = training.DistributedTrainer(
    model=model, world_size=1, rank=0,
    train_loader=..., val_loader=...,
    optimizer=optimizer, cfg=dict(max_steps=100),
)
```

### scoring — 生成质量打分（Part3K2）

`scoring` 子模块提供生成文本质量打分，包含 5 个独立指标函数与一个聚合打分器 `ScoringEvaluator`，仅依赖 NumPy + 标准库（`collections.Counter` / `math`）。

| 函数 | 说明 |
|---|---|
| `exact_match(prediction, reference)` | 精确匹配率：1.0 完全相等（strip 后），0.0 不等 |
| `prefix_accuracy(prediction, reference)` | 前缀匹配率：prediction 前缀与 reference 重合比例（`common_len / len(ref)`），适合续写任务 |
| `char_f1(prediction, reference)` | 字符级 F1：把文本看作字符多重集，计算 precision / recall / F1 |
| `bleu(prediction, reference, max_n=4)` | BLEU-4（简化版，无 smoothing）：1-gram 到 4-gram precision 的几何平均 + brevity penalty |
| `rouge_l(prediction, reference)` | ROUGE-L：基于最长公共子序列（LCS）计算 F1 |

**`ScoringEvaluator(metrics=None)`**：聚合打分器。
- `metrics`：`list[str]`，要计算的指标名（默认全部 5 个）；传入未知指标名抛 `ValueError`。
- `evaluate(predictions, references) -> dict`：批量计算，返回 `{metric: avg_score, "n_samples": int, "per_sample": list[dict]}`。
- `report(score_dict) -> str`：生成可读报告字符串。
- `score_pair(prediction, reference) -> dict`：计算单个样本的所有指标。

```python
from verse_torch import scoring

evaluator = scoring.ScoringEvaluator()   # 默认计算全部 5 个指标
scores = evaluator.evaluate(
    predictions=["你好世界", "床前明月光"],
    references=["你好世界", "床前明月光，疑是地上霜"],
)
# scores: {"exact_match": 0.5, "prefix_accuracy": ..., "char_f1": ...,
#          "bleu": ..., "rouge_l": ..., "n_samples": 2, "per_sample": [...]}

print(evaluator.report(scores))
# ==================================================
# 评分报告
# ==================================================
# 样本数: 2
# --------------------------------------------------
#   exact_match         : 0.5000
#   prefix_accuracy     : 0.xxxx
#   ...

# 单独使用某个指标函数
f1 = scoring.char_f1("hello world", "hello werld")
em = scoring.exact_match("abc", "abc")     # 1.0
```

### quantize — 量化

提供三种权重量化方案，每种都给出 `quantize_*` / `dequantize_*` / `matmul_*` 三件套，其中 `matmul_*` 是 **fused 反量化-GEMM**（直接吃 packed 权重，避免物化完整 fp32 矩阵）。

| 方案 | 函数 | 精度 | bit/param | 打包 |
|---|---|---|---|---|
| INT8 | `quantize_int8` / `dequantize_int8` | per-channel | 8 | 原生 int8 |
| INT4 (W4A16) | `quantize_int4` / `dequantize_int4` / `matmul_int4` | per-channel | 4 | 2 个 nibble → 1 字节 |
| 1.58-bit ternary | `quantize_ternary` / `dequantize_ternary` / `matmul_ternary` | per-channel，`{-1, 0, +1}` | 2 | 4 个 2-bit code → 1 字节 |

**`QuantizedLinear(linear, qtype="int4", cache_fp32=True)`**：可热替换 `nn.Linear` 的推理专用层。
- `cache_fp32=True`（默认，推荐推理）：构造时一次性 `unpack → fp32 → * scale → transpose` 缓存为 contiguous `(in, out)` fp32，forward 只做一次 GEMM + bias，BLAS 走最优路径。
- `cache_fp32=False`：每次 forward 走 fused 反量化-GEMM，内存占用约为 fp32 的 1/4（int8 路径），但有额外 cast 与 scale 开销。

**压缩比**：相对 fp32，INT8 ≈ 4×、INT4 ≈ 8×、ternary ≈ 16×。

```python
import numpy as np
from verse_torch import nn, quantize

linear = nn.Linear(512, 512, bias=False)
x = np.random.randn(8, 512).astype(np.float32)

# INT4 量化
qlin = quantize.QuantizedLinear(linear, qtype="int4", cache_fp32=True)
out = qlin(x)                     # 与 linear(x) 数值近似
print("packed bytes:", qlin.packed.nbytes, "fp32 bytes:", linear.weight.data.nbytes)
```

### parallel — CPU 并行计算

基于 `multiprocessing.Pool`（Linux fork 启动方式），仅使用 NumPy + 标准库。默认 `n_workers = max(1, os.cpu_count() // 2)`。受限环境（CI、不可 pickle 的函数）下自动降级为串行。

- **`parallel_matmul(A, B, n_workers=None)`**：批量矩阵乘法并行；`A` 可为 `(M, K)` 或 `(B, M, K)`，`B` 可为 `(K, N)` 或 `(B, K, N)`。返回类型与输入中第一个 `Tensor` 一致。3D×2D 路径下 B 通过全局变量在 fork 模式下被子进程继承，避免重复 pickle。
- **`ParallelLinear(in_features, out_features, bias=True, n_workers=None, batch_threshold=16)`**：继承 `nn.Linear`；当 `x.shape[0] >= batch_threshold` 时启用并行前向，反向手动构建 `_backward` 闭包，与父类数值 1e-6 内一致。
- **`parallel_map(fn, iterable, n_workers=None)`**：通用并行 map，按原顺序返回 list；`fn` 推荐为顶层函数（picklable）。

参考 [ADR-004 CPU 并行](../../docs/architecture/adr-004-cpu-parallel.md)。

```python
import numpy as np
from verse_torch import parallel

A = np.random.randn(64, 128, 64).astype(np.float32)   # batch=64
B = np.random.randn(64, 64).astype(np.float32)
C = parallel.parallel_matmul(A, B, n_workers=4)        # (64, 128, 64)

results = parallel.parallel_map(lambda x: x * 2, range(100), n_workers=4)
```

### compress — 模型压缩 PoC

提供剪枝 + LoRA + 蒸馏 + 量化的组合管线，仅依赖 NumPy + 标准库。压缩比按 bit-level 精确计算（fp32=32、INT8=8、INT4=4、ternary=2 bit/param）。

- **`OutlierSafePruner(model, sparsity=0.3)`**：结构化剪枝（mask + 冻结策略，原结构保留）。
  - `GQASelfAttention`：按 head 维度剪（同时 mask `wq` 行与 `proj` 列）。
  - `SwiGLUMLP`：按 hidden 维度剪（同时 mask `w_gate`/`w_up` 行与 `w_down` 列）。
  - `Linear`：按 output channel 剪。
  - `Embedding` / `head`（tie_weights）：跳过避免破坏词表语义。
  - `apply() -> (model, report)`，`report` 是各模块的剪枝统计 dict。
- **`LoRALinear(d_in, d_out, r=8, alpha=16.0, base=None)`**：`forward = base(x) + (x @ A) @ B * (alpha / r)`，`base` 冻结，`A` 高斯初始化（std=0.01）、`B` 零初始化（保证初始 ΔW=0）。`merge()` 把 `A @ B * scaling` 加回 `base.weight` 返回新 `nn.Linear`（仅支持 `nn.Linear` base，`QLinear` base 不支持）。
- **`KnowledgeDistiller(teacher, student, T=2.0, alpha=0.5)`**：`Loss = alpha * T^2 * KL(teacher/T || student/T) + (1-alpha) * CE(student, hard)`；构造时自动冻结 teacher；`distill(train_loader, optimizer, max_steps=100, eval_fn=None, eval_every=0)` 跑蒸馏循环。
- **`QLinear(linear, qtype="int4", cache_fp32=True)`**：把 `QuantizedLinear` 包装为 `nn.Module`，可嵌入模型树并参与 `state_dict` / `parameters()`。
- **`compress_pipeline(model, target_ratio=0.1, eval_fn=None, sparsity=0.3, qtype="int4", lora_r=8, lora_alpha=16.0, use_lora=False)`**：端到端 pipeline（prune → quantize → 可选 lora_wrap）。返回 dict：
  ```
  {original_params, compressed_params, compressed_bits, original_bits,
   compression_ratio, original_loss, compressed_loss, loss_diff_pct, steps}
  ```
- **单技术函数**：`prune_only(model, sparsity=0.3) -> (model, report)`、`quantize_only(model, dtype="int4") -> model`、`lora_only(model, r=8, alpha=16.0) -> model`、`ternary_only(model) -> model`、`distill_only(teacher, student, train_loader, max_steps=100, T=2.0, alpha=0.5, lr=1e-3, eval_fn=None, eval_every=0) -> student`。
- **统计辅助**：`count_parameters(model)`、`count_nonzero_params(model)`、`compute_compressed_bits(model)`。

**PoC 验证**：1M 参数模型压缩比 **10.343×**，loss 差异 **0.33%**。参考 [压缩管线设计](../../verse_data/designs/compression_pipeline_design.md) 与 [压缩技术参考](../../docs/papers/compression_references.md)。

```python
from verse_torch import nn, compress
import numpy as np

model = nn.TransformerLM(vocab_size=128, n_layer=2, n_head=4, n_embd=64)

# 评估函数：返回在固定 batch 上的 loss
def eval_fn(m):
    x = np.random.randint(0, 128, size=(4, 16))
    y = np.roll(x, -1, axis=1)
    from verse_torch.training import cross_entropy_loss
    return float(cross_entropy_loss(m(x), y).data)

report = compress.compress_pipeline(
    model, target_ratio=0.1, eval_fn=eval_fn,
    sparsity=0.3, qtype="ternary",   # ternary 才能达到 10× 压缩
)
print(f"压缩比: {report['compression_ratio']:.2f}x")
print(f"loss 差异: {report['loss_diff_pct']:.2f}%")
```

## 设计原则

- **CPU-first**：所有运算基于 NumPy + BLAS，不依赖 GPU（参考 [ADR-001](../../docs/architecture/adr-001-cpu-first.md)）。
- **GPU/NPU 可选委托（Part4K1）**：CPU-first 不变，PyTorch 仅作为可选加速后端；GPU 路径委托 `torch` 原生算子，不自研 kernel（参考 [ADR-005](../../docs/architecture/adr-005-gpu-npu-backend.md)）。
- **零重型依赖**：运行时仅 NumPy + 标准库；`matplotlib` / `numba` / `torch` 均为可选加速。
- **PyTorch API 兼容**：`Tensor` / `nn.Module` / `optim` / `losses` 命名与签名贴近 PyTorch，降低迁移成本。
- **数值正确性**：所有算子均通过有限差分梯度检查（见 `tests/test_unit_operators.py`）；GPU 后端与 CPU 后端语义等价。

## 测试

| 文件 | 覆盖范围 |
|---|---|
| [test_nn_advanced.py](../../tests/test_nn_advanced.py) | SwiGLU / GQA / TransformerLM 前向 + 反向 |
| [test_training.py](../../tests/test_training.py) | Trainer / cross_entropy / 调度器 / EarlyStopping |
| [test_parallel.py](../../tests/test_parallel.py) | 并行 matmul 数值一致性 |
| [test_compression_poc.py](../../tests/test_compression_poc.py) | 压缩 PoC 端到端验证 |
| [test_unit_operators.py](../../tests/test_unit_operators.py) | 基础算子 + 有限差分梯度检查 |
| [test_optim_extras.py](../../tests/test_optim_extras.py) | **Part3K2** Lion / Adafactor 优化器 |
| [test_scheduler_extras.py](../../tests/test_scheduler_extras.py) | **Part3K2** OneCycleLR / ReduceLROnPlateau / CosineRestartsLR 调度器 |
| [test_parallel_trainer.py](../../tests/test_parallel_trainer.py) | **Part3K2** ParallelTrainer 端到端（chunk 拆分 / 合并 / val_loss 修复） |
| [test_scoring.py](../../tests/test_scoring.py) | **Part3K2** 5 个指标（exact_match / prefix_accuracy / char_f1 / bleu / rouge_l） |
| [test_device_backend.py](../../tests/test_device_backend.py) | **Part4K1** NumpyBackend / TorchBackend / Tensor 设备迁移 / 无 PyTorch 回退 / autocast |

运行：

```bash
python -m pytest tests/test_nn_advanced.py tests/test_training.py \
    tests/test_parallel.py tests/test_compression_poc.py -v

# Part3K2 新增测试
python -m pytest tests/test_optim_extras.py tests/test_scheduler_extras.py \
    tests/test_parallel_trainer.py tests/test_scoring.py -v

# Part4K1 新增测试
python -m pytest tests/test_device_backend.py -v
```

## 相关文档

- [ADR-001 CPU-first](../../docs/architecture/adr-001-cpu-first.md)
- [ADR-004 CPU 并行](../../docs/architecture/adr-004-cpu-parallel.md)
- [ADR-005 GPU/NPU 后端抽象（Part4K1）](../../docs/architecture/adr-005-gpu-npu-backend.md)
- [VerseInfra 总包 README](../verse_infra/README.md)
- [VerseNex README](../verse_nex/README.md)
- [CometSpark V0.5-1B README](../../spark/README.md)
- [压缩管线设计](../../verse_data/designs/compression_pipeline_design.md)
- [压缩技术参考](../../docs/papers/compression_references.md)
