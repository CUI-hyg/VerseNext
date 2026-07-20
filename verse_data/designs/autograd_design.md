# 设计草稿：VerseTorch 反向自动微分（autograd）

> 关联源码：[`tensor.py`](file:///workspace/packages/verse_torch/verse_torch/tensor.py)
> 关联 spec：[`.trae/specs/build-verse-framework/spec.md`](../../.trae/specs/build-verse-framework/spec.md)
> 关联 ADR：[ADR-001 CPU 优先](../../docs/architecture/adr-001-cpu-first.md)

## 1. 背景与动机

Verse 框架的核心约束之一是 **不依赖 PyTorch** 作为运行时（见 ADR-001）。这意味着：

- 不能用 `torch.autograd.Function` 注册自定义算子的反向；
- 不能用 `torch.Tensor.backward()` 触发反向传播；
- 不能用 `torch.autograd.grad` 计算高阶梯度。

但是深度学习训练的根本需求 —— **梯度下降** —— 必须通过某种方式满足。我们需要自实现一套反向自动微分引擎，覆盖：

- 元素级算子（add / sub / mul / div / pow / exp / log / relu / gelu / sigmoid / tanh / silu / softmax / log_softmax）；
- shape 算子（reshape / view / transpose / permute / squeeze / unsqueeze / expand / slice）；
- reduction 算子（sum / mean / max / min / var / std / norm）；
- matmul（含 batched 与 1D 情形）；
- broadcasting-aware 反向；
- 拓扑排序与梯度累积（PyTorch 语义）。

参考实现是 [micrograd](https://github.com/karpathy/micrograd)（Karpathy 的教学版 autograd）与 PyTorch 的早期 Python 版本。本设计与 micrograd 的关键差异：

- 用 NumPy ndarray 而非 Python 标量（支持 batch + 向量化）；
- 显式处理 broadcasting 反向；
- 拓扑排序基于 `set` + DFS 后序遍历，不依赖 `__eq__` 重写。

## 2. 核心数据结构：Tensor

`Tensor` 类是 autograd 的最小单元，定义在 [`tensor.py` 第 128-176 行](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L128-L176)：

```python
class Tensor:
    __array_priority__ = 1000  # 让 numpy 把反向算子优先转给 Tensor

    def __init__(self, data, requires_grad: bool = False, dtype=None,
                 _children=(), _op=""):
        # data: numpy ndarray
        self.data = ...                  # np.ndarray，默认 float32
        self.requires_grad = bool(requires_grad)
        self.grad = None                 # np.ndarray 或 None
        self._backward = lambda: None    # 闭包，反向时调用
        self._prev = set(_children)      # 父节点集合（用于拓扑排序）
        self._op = _op                   # 操作名（仅用于调试）
```

字段语义：
- `data`：底层 NumPy ndarray，所有前向计算直接操作它；
- `requires_grad`：是否需要梯度；只有 `True` 的 Tensor 会在 `backward()` 时被计算 grad；
- `grad`：累积梯度，初始为 None；`backward()` 后填充为 ndarray；
- `_backward`：闭包，调用时把上游梯度 `out.grad` 累加到父节点的 `grad`；
- `_prev`：父节点集合（set of Tensor），用于拓扑排序；用 set 而非 list 是为了去重；
- `_op`：操作名（如 `"+"`、`"relu"`、`"matmul"`），仅用于调试，不影响计算。

### 2.1 默认 dtype 策略

构造时类型推断（[第 142-176 行](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L142-L176)）：

| 输入 data 类型           | 默认 dtype              | 原因                          |
| ------------------------ | ----------------------- | ----------------------------- |
| `np.ndarray`             | 保留原 dtype            | 用户可能显式传 float64 做梯度检查 |
| `np.generic`（标量）     | 保留原 dtype            | 同上                          |
| Python `int` / `list[int]` | int64（NumPy 默认）     | 索引常用                      |
| Python `float` / `list[float]` | **float32**（强制降级）| 避免 NumPy 默认 float64 占内存 |

显式 `dtype=` 参数会覆盖上述策略。

### 2.2 不重写 `__eq__` / `__hash__` 的设计

[第 1196-1198 行](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L1196-L1198) 显式注释：

```python
# 注意：不重载 __eq__ 与 __hash__，使用 Python 默认的 id-based 语义，
# 这样 set(Tensor) 去重与拓扑排序行为正确。
# 用户如需逐元素比较，可用 (a.data == b.data) 或 (a - b).abs() 等。
```

如果重写 `__eq__` 让 `(a == b).all()` 返回 bool，会让 `set(a, b)` 把"值相等但不同对象"的 Tensor 视为同一个，破坏拓扑排序的去重逻辑。这是 PyTorch 也踩过的坑：`torch.Tensor.__eq__` 是逐元素比较，但 `torch.Tensor.__hash__` 不可用，所以不能把 Tensor 放进 set。

Verse 的选择：**保留 Python 默认 `id-based` 语义**，让 `_prev` 用 set 去重；用户需要逐元素比较时显式 `.data ==`。

## 3. VJP（vector-Jacobian product）模式

每个算子的反向传播通过 **闭包** 实现，而非注册表。模式：

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out_data = self.data + other.data

    def _backward():
        if self.requires_grad:
            g = unbroadcast(out.grad, self.shape)  # 处理 broadcasting
            self._accumulate_grad(g)
        if other.requires_grad:
            g = unbroadcast(out.grad, other.shape)
            other._accumulate_grad(g)

    out = self._result(out_data, (self, other), "+")
    if out.requires_grad:
        out._backward = _backward
    return out
```

关键点：
- **前向时记录闭包**：闭包内捕获 `self` / `other` / `out` 引用（Python 闭包默认按引用捕获，所以 `out.grad` 在反向时是最新的）；
- **不立即执行反向**：闭包只是定义，真正调用在 `backward()` 拓扑排序后；
- **梯度累积用 `_accumulate_grad`**：[第 293-302 行](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L293-L302)，PyTorch 语义 `self.grad += g`，dtype 对齐到 `self.data.dtype`。

### 3.1 PyTorch autograd.Function 的区别

PyTorch 用 `autograd.Function` 注册算子：

```python
class MyAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output
```

Verse **不提供** `autograd.Function`：
- 用户不能自定义新算子的反向；
- 所有反向逻辑写在 `Tensor` 方法内；
- 若需要新算子，要么用现有算子组合表达（推荐），要么直接修改 `tensor.py` 添加方法。

这是已知限制，权衡是：API 简单 + 实现可控 + 不引入 Python 元类机制。

## 4. unbroadcast：broadcasting 反向

NumPy 的 broadcasting 规则：维度从右对齐，大小为 1 的维度会被扩展。反向时需要把"广播后"的梯度 sum 回原始 shape。

[`unbroadcast` 函数](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L81-L105)：

```python
def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    # 1. reduce 多余的前导轴（grad.ndim > len(target_shape)）
    ndim_extra = grad.ndim - len(target_shape)
    if ndim_extra > 0:
        grad = grad.sum(axis=tuple(range(ndim_extra)))
    # 2. 对 target_shape 中为 1 的轴（但 grad 中大于 1）做 keepdims sum
    axes_to_sum = tuple(
        i for i, dim in enumerate(target_shape)
        if dim == 1 and grad.shape[i] != 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    # 3. reshape 以确保形状精确
    grad = grad.reshape(target_shape)
    return grad
```

举例：
- `a.shape = (3,)`, `b.shape = (4, 3)`，`c = a + b` 形状 `(4, 3)`；
- 反向 `a.grad` 时 `c.grad` 形状 `(4, 3)`，需要 `sum(axis=0)` → `(3,)`；
- 反向 `b.grad` 时 `c.grad` 形状 `(4, 3)`，与 `b.shape` 一致，无需 sum。

`unbroadcast` 处理所有 broadcasting 情形，被几乎所有元素级算子调用。

## 5. 拓扑排序：DFS 后序遍历

`backward()` 函数实现反向传播，关键是拓扑排序计算图：

[`backward` 方法](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L1139-L1182)：

```python
def backward(self, grad=None) -> None:
    if not self.requires_grad:
        raise RuntimeError("Tensor does not require grad and cannot call backward().")

    if grad is None:
        if self.data.size != 1:
            raise RuntimeError("grad can only be implicitly created for scalar outputs")
        grad = np.ones_like(self.data)
    elif isinstance(grad, Tensor):
        grad = grad.data
    else:
        grad = np.asarray(grad, dtype=self.data.dtype)

    if grad.dtype != self.data.dtype:
        grad = grad.astype(self.data.dtype, copy=False)
    self.grad = grad

    # 拓扑排序：DFS
    topo = []
    visited = set()

    def build(v):
        if id(v) in visited:
            return
        visited.add(id(v))
        for child in v._prev:
            build(child)
        topo.append(v)

    build(self)

    # 逆序调用 _backward
    for v in reversed(topo):
        v._backward()
```

关键点：

1. **入口 grad**：标量输出默认 `np.ones_like`；非标量必须显式传 grad（与 PyTorch 一致）。
2. **`visited` 用 `id(v)`**：避免依赖 `__eq__`，且对循环引用安全。
3. **DFS 后序**：先递归子节点，再 append 当前节点。这样 `topo[0]` 是叶子节点，`topo[-1]` 是输出节点（self）。
4. **逆序调用**：从输出（self）开始，逆拓扑序调用每个 `_backward` 闭包。每次闭包调用时，`out.grad` 已经被它的下游节点填充好。

### 5.1 梯度累积语义

PyTorch 语义：`p.grad += g`，不是 `p.grad = g`。这让用户可以多次 `backward()` 累积梯度（例如梯度累积训练）。

Verse 实现：[`_accumulate_grad`](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L293-L302)：

```python
def _accumulate_grad(self, grad: np.ndarray):
    target_dtype = self.data.dtype
    if self.grad is None:
        self.grad = grad.astype(target_dtype, copy=True)
    else:
        self.grad = self.grad + grad.astype(target_dtype, copy=False)
```

注意 `dtype` 强制对齐到 `self.data.dtype`，避免 float64 累积（PyTorch 行为）。

### 5.2 复杂度分析

- 拓扑排序：O(V + E)，V 是节点数，E 是边数；
- 每个 `_backward` 闭包：O(N) 其中 N 是该算子处理的元素数；
- 总复杂度：O(V·N)（最坏情况，所有节点同等大小）。

实际场景（MNIST MLP 训练）：
- V ≈ 6（每个 batch 的 forward 图节点数）；
- N ≈ 64·784 ≈ 50k；
- 总反向开销 < 1ms。

## 6. 全局梯度开关：`no_grad` / `enable_grad`

[`_GRAD_ENABLED` 全局变量](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L21-L73)：

```python
_GRAD_ENABLED = True

def is_grad_enabled() -> bool:
    return _GRAD_ENABLED

def set_grad_enabled(mode: bool) -> None:
    global _GRAD_ENABLED
    _GRAD_ENABLED = bool(mode)

class no_grad:
    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev
        return False
```

每个算子在构造结果时通过 `_result` 检查：

[`_result` 方法](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L283-L291)：

```python
def _result(self, out_data, _children, _op, requires_grad=None):
    if requires_grad is None:
        requires_grad = _GRAD_ENABLED and any(c.requires_grad for c in _children)
    if not requires_grad:
        _children = ()  # 不记录父节点，节省内存
    out = Tensor(out_data, requires_grad=requires_grad, _children=_children, _op=_op)
    return out
```

效果：在 `with no_grad():` 块内，所有算子构造的 Tensor `requires_grad=False` 且 `_prev` 为空，不构建计算图。

这是推理时省内存的关键。在 [`examples/mnist_mlp.py` 的 `accuracy` 函数](file:///workspace/examples/mnist_mlp.py#L150-L163) 中，评估时用 `with no_grad():` 包裹前向，避免 60000 测试样本的计算图被构建。

## 7. 与 PyTorch 的关键差异

| 维度              | PyTorch                                            | VerseTorch                                          |
| ----------------- | -------------------------------------------------- | --------------------------------------------------- |
| 后端              | C++ (aten) + CUDA + cuDNN                          | NumPy (CPU only)                                    |
| Tensor 实现        | `torch._C._TensorBase`（C 扩展）                   | Python class 包装 `np.ndarray`                      |
| autograd          | `torch.autograd.Function` 注册表 + C++ engine      | 闭包 + DFS 拓扑排序                                  |
| 自定义算子反向     | 支持 (`autograd.Function`)                         | **不支持**（需修改 `tensor.py`）                    |
| broadcasting 反向  | 内部 `at::Tensor::sum_to`                          | [`unbroadcast`](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L81-L105) |
| `__eq__` / `__hash__` | `__eq__` 是逐元素；`__hash__` 不可用                | **保留默认 id-based**，set 去重正确                  |
| dtype 系统         | float16/32/64, bfloat16, int8/16/32/64, bool       | float32/64, int32/64, bool（**无 float16/bfloat16**）|
| CUDA              | 一等公民                                           | **无 CUDA**（纯 CPU）                                |
| 反向模式           | reverse-mode + forward-mode（双模）                 | 仅 reverse-mode                                     |
| 高阶梯度           | 支持（`create_graph=True`）                        | **不支持**                                          |
| DDP / RPC          | 支持                                               | **不支持**                                          |
| `torch.compile`    | 支持                                                | **不支持**                                          |
| In-place 操作      | 支持（带 version 计数）                            | **不支持**（所有操作都返回新 Tensor）                |
| Sparse tensor      | 支持 (`torch.sparse`)                              | **不支持**                                          |

## 8. 性能考量

### 8.1 纯 NumPy 的开销

每个算子至少 2 次 ndarray 构造（前向 out_data + 反向 grad），加上闭包对象创建。相比 PyTorch 的 fused kernel：

| 算子      | PyTorch (CUDA) | PyTorch (CPU) | VerseTorch (NumPy) |
| --------- | -------------- | ------------- | ------------------- |
| `a + b` (1M 元素) | ~5 μs          | ~50 μs        | ~200 μs             |
| `a @ b` (1024×1024) | ~100 μs        | ~5 ms         | ~10 ms              |
| `a.sum()` (1M 元素) | ~10 μs         | ~100 μs       | ~300 μs             |

差距约 2-4 倍，对教学/原型足够，但大规模训练不可行。

### 8.2 无 fused kernel

PyTorch 的 `FusedLinear` / `FlashAttention` 等把多个算子合并成一个 kernel，减少内存读写。VerseTorch 没有融合：

- `Linear` 拆成 `matmul + add bias` 两次 op；
- `softmax` 拆成 `max + exp + sum + div` 四次 op；
- `LayerNorm` 拆成 `mean + var + sqrt + mul + add` 五次 op。

每次 op 都创建中间 Tensor，内存带宽是瓶颈。

### 8.3 缓解策略

- 推理时用 `with no_grad():` 跳过计算图构建；
- 训练时尽量 batch 大（摊薄 Python 调用开销）；
- 用 INT4/ternary 量化（`verse_torch.quantize`）降低内存带宽需求；
- 关键路径（如 selective scan）用纯 NumPy 向量化，避免 Python 循环。

## 9. 已知限制

1. **不支持高阶梯度**：`backward()` 不接受 `create_graph=True`，不能在梯度上再求梯度。
2. **不支持 in-place 操作**：所有操作都返回新 Tensor；`a += b` 等价于 `a = a + b`，会破坏计算图（实际未实现，会触发 Python 默认行为）。
3. **不支持自定义算子反向**：用户不能像 `autograd.Function` 那样注册新算子。需要新算子时只能修改 `tensor.py`。
4. **不支持 CUDA**：所有计算在 CPU 上。
5. **dtype 系统简化**：无 float16/bfloat16（NumPy 原生不支持 bf16）；INT8/INT4 仅在 `quantize.py` 中作为存储格式，不参与 autograd。
6. **拓扑排序递归深度限制**：Python 默认递归深度 1000，超深计算图（如 RNN 展开几千步）会栈溢出。可改用迭代版本（未实现）。
7. **`set` 去重基于 `id()`**：同一对象在多处引用会被去重为同一个节点；这是预期行为，但用户需要注意 `a + a` 会创建两个 `a` 的引用，反向时 `a.grad` 会被累积两次（正确行为）。
8. **不支持 `torch.autograd.grad`**：只能用 `loss.backward()` 触发反向；不能选择性计算部分梯度。

## 10. 源码引用汇总

- [`Tensor` 类定义](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L128-L176)：核心字段与构造逻辑；
- [`unbroadcast` 函数](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L81-L105)：broadcasting 反向；
- [元素级算子示例（`__add__`）](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L308-L323)：闭包模式；
- [matmul 反向](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L1076-L1130)：含 1D / 2D+ 多种情况；
- [`backward` 方法](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L1139-L1182)：拓扑排序 + 反向调用；
- [`_accumulate_grad` 方法](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L293-L302)：梯度累积 + dtype 对齐；
- [`no_grad` / `enable_grad` 上下文管理器](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L35-L73)：全局梯度开关；
- [`_result` 工厂方法](file:///workspace/packages/verse_torch/verse_torch/tensor.py#L283-L291)：自动 requires_grad 传播。
