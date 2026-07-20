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

## 安装

```bash
pip install -e packages/verse_torch
```

唯一运行时依赖是 `numpy>=1.26`；`matplotlib` 为可选依赖（缺失时 `plot_loss_curve` 自动降级为 ASCII 曲线）。

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

### nn — 神经网络层

**`Module` 基类**：通过 `__setattr__` 自动注册 `Tensor` 参数与子 `Module`。
关键方法：`parameters()`、`named_parameters()`、`modules()` / `named_modules()`、`children()`、`state_dict()`、`load_state_dict(sd, strict=True)`、`train()` / `eval()`、`zero_grad()`、`apply(fn)`。

**基础层**：`Linear(in, out, bias=True)`、`Embedding(num, dim)`、`LayerNorm(shape, eps=1e-5)`、`RMSNorm(shape, eps=1e-6)`、`Dropout(p=0.5)`、`Sequential(*modules)`、`ModuleList(list)`。

**LM 组件**：
- `SwiGLUMLP(d, dropout=0.0, hidden_multiple=4, align=64)`：SwiGLU 激活 + 2/3 缩放对齐到 `align`。
- `GQASelfAttention(d, n_head, n_kv_head=None, dropout=0.0)`：GQA + 内置 RoPE（预计算 32768 长度 cos/sin 表）+ 因果掩码 + KV cache，`forward(x, kv_cache=None) -> (out, new_kv_cache)`。
- `TransformerBlock(d, n_head, n_kv_head=None, dropout=0.0)`：pre-norm 残差结构（`RMSNorm` → attn → +x → `RMSNorm` → MLP → +x）。
- `TransformerLM(vocab_size, n_layer, n_head, n_embd, seq_len=128, dropout=0.1, n_kv_head=None, tie_weights=True)`：含 token embedding、`n_layer` 个 block、最终 `RMSNorm` 与共享权重的 `head`，初始化按 GPT-2 风格（`normal_(std=0.02)` + 残差分支 `1/sqrt(2*n_layer)` 缩放）。
- `repeat_kv(x, n_rep)`：把 KV head 复制到 query head 数（GQA 工具）。

**初始化辅助**（原地修改）：`kaiming_uniform_(t, a=0, mode="fan_in", nonlinearity="leaky_relu")`、`xavier_uniform_(t, gain=1.0)`、`normal_(t, mean=0, std=1)`、`zeros_(t)`、`ones_(t)`、`uniform_(t, low=0, high=1)`。

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

### losses — 损失函数

所有损失返回**标量 `Tensor`**，`requires_grad` 自动传播，可直接 `backward()`。

| 函数 | 公式 / 用途 |
|---|---|
| `cross_entropy(logits, targets)` | softmax 交叉熵，`logits: (N, C)`，`targets: (N,)` int |
| `nll_loss(log_probs, targets)` | 负对数似然，输入已是 `log_softmax` 结果 |
| `binary_cross_entropy(pred, target)` | BCE，输入为概率（已 sigmoid） |
| `binary_cross_entropy_with_logits(logits, target)` | BCE，输入为 logits（数值稳定版） |
| `mse_loss(pred, target)` | 均方误差 |
| `l1_loss(pred, target)` | 平均绝对误差 |
| `kl_div_loss(log_probs, target_probs)` | KL 散度 `sum(t*(log t - log_probs))` |

```python
from verse_torch import Tensor, losses

logits = Tensor([[2.0, 1.0, 0.1],
                 [0.1, 2.0, 1.0]], requires_grad=True)
targets = [0, 1]
loss = losses.cross_entropy(logits, targets)
loss.backward()
print(logits.grad.shape)   # (2, 3)
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
- **零重型依赖**：运行时仅 NumPy + 标准库；`matplotlib` / `numba` 均为可选加速。
- **PyTorch API 兼容**：`Tensor` / `nn.Module` / `optim` / `losses` 命名与签名贴近 PyTorch，降低迁移成本。
- **数值正确性**：所有算子均通过有限差分梯度检查（见 `tests/test_unit_operators.py`）。

## 测试

| 文件 | 覆盖范围 |
|---|---|
| [test_nn_advanced.py](../../tests/test_nn_advanced.py) | SwiGLU / GQA / TransformerLM 前向 + 反向 |
| [test_training.py](../../tests/test_training.py) | Trainer / cross_entropy / 调度器 / EarlyStopping |
| [test_parallel.py](../../tests/test_parallel.py) | 并行 matmul 数值一致性 |
| [test_compression_poc.py](../../tests/test_compression_poc.py) | 压缩 PoC 端到端验证 |
| [test_unit_operators.py](../../tests/test_unit_operators.py) | 基础算子 + 有限差分梯度检查 |

运行：

```bash
python -m pytest tests/test_nn_advanced.py tests/test_training.py \
    tests/test_parallel.py tests/test_compression_poc.py -v
```

## 相关文档

- [ADR-001 CPU-first](../../docs/architecture/adr-001-cpu-first.md)
- [ADR-004 CPU 并行](../../docs/architecture/adr-004-cpu-parallel.md)
- [压缩管线设计](../../verse_data/designs/compression_pipeline_design.md)
- [压缩技术参考](../../docs/papers/compression_references.md)
- [CometSpark 训练仓库](../../data/demo/README.md)
