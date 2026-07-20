# 示例：MNIST MLP 训练（VerseTorch 核心引擎验证）

> 对应脚本：[`mnist_mlp.py`](file:///workspace/examples/mnist_mlp.py)

## 目标

在 MNIST 手写数字数据集上训练一个 2 层 MLP，端到端验证 VerseTorch 核心引擎（Tensor + autograd + nn + optim + losses）的正确性：
- 自实现的反向自动微分能否在真实任务上稳定收敛；
- Adam 优化器与 cross_entropy 损失的协同行为是否与 PyTorch 对齐；
- 5 epoch 内测试集准确率是否 ≥ 95%。

## 模型架构

简单的 2 层 MLP：

```
输入 (N, 784)
   │
   ▼
Linear(784, 128)   # fc1
   │
   ▼
ReLU
   │
   ▼
Linear(128, 10)    # fc2
   │
   ▼
logits (N, 10)  →  cross_entropy(logits, targets)
```

参数量约 101k（128\*784 + 128 + 10\*128 + 10 = 101,770）。

源码定义见 [`mnist_mlp.py` 的 `MLP` 类](file:///workspace/examples/mnist_mlp.py#L116-L129)。

## 训练配置

| 项目             | 取值              |
| ---------------- | ----------------- |
| batch_size       | 64                |
| epochs           | 5                 |
| 优化器           | Adam              |
| 学习率           | 1e-3              |
| betas            | (0.9, 0.999) 默认 |
| weight_decay     | 0（未启用）       |
| 损失函数         | cross_entropy     |
| 输入归一化       | /255.0 → [0, 1]   |
| 数据 shuffle     | 每个 epoch 打乱   |

源码片段见 [训练循环](file:///workspace/examples/mnist_mlp.py#L218-L243)。

## 数据来源

- 数据集：MNIST（60000 训练 + 10000 测试）；
- 下载地址：`https://ossci-datasets.s3.amazonaws.com/mnist/`（PyTorch 官方镜像）；
- 缓存路径：[`/workspace/datasets/raw/mnist/`](file:///workspace/datasets/raw/mnist/)；
- 解析方式：手写 IDX 格式解析（gzip + struct.unpack + np.frombuffer），见 [`parse_idx_images` 与 `parse_idx_labels`](file:///workspace/examples/mnist_mlp.py#L68-L85)；
- 网络不可用时会自动 fallback 到合成数据（10 类高斯簇），见 [fallback 分支](file:///workspace/examples/mnist_mlp.py#L181-L202)。

## 运行方式

```bash
cd /workspace
python examples/mnist_mlp.py
```

不需要任何额外环境变量；脚本会自动把 `packages/verse_torch` 加入 `sys.path`（见 [sys.path 注入](file:///workspace/examples/mnist_mlp.py#L24-L26)），无需 `pip install`。

## 预期结果

5 epoch 后测试集准确率应 ≥ 95%。实测结果：

```
[1/4] Loading MNIST dataset...
  Train: (60000, 784), Test: (10000, 784)
[2/4] Building model...
  Model: MLP(784 -> 128 -> 10), params=101770
  Optimizer: Adam(lr=1e-3)
[3/4] Training 5 epochs, batch_size=64...
  Epoch 1/5 | loss=0.0451 | train_acc=0.9786 | test_acc=0.9706 | ...
  Epoch 2/5 | loss=0.0202 | train_acc=0.9839 | test_acc=0.9760 | ...
  Epoch 3/5 | loss=0.0141 | train_acc=0.9884 | test_acc=0.9792 | ...
  Epoch 4/5 | loss=0.0109 | train_acc=0.9916 | test_acc=0.9809 | ...
  Epoch 5/5 | loss=0.0086 | train_acc=0.9930 | test_acc=0.9766 | ...
[4/4] Final evaluation...
  Final test accuracy: 97.66%
  ✓ PASS: accuracy >= 95%
```

实测最终测试准确率 **97.66%**（与 spec 要求一致）。

## 关键代码引用

- [Tensor 工厂与算子调用](file:///workspace/examples/mnist_mlp.py#L28)：`from verse_torch import Tensor, nn, optim, losses, no_grad`；
- [MLP 模型定义](file:///workspace/examples/mnist_mlp.py#L116-L129)：`nn.Linear` + `x.relu()` 链式调用，全部走 VerseTorch 的可微算子；
- [训练步骤](file:///workspace/examples/mnist_mlp.py#L223-L233)：
  ```python
  x = Tensor(xb, requires_grad=False)
  logits = model(x)
  loss = losses.cross_entropy(logits, yb)
  optimizer.zero_grad()
  loss.backward()       # 触发拓扑排序 + 反向传播
  optimizer.step()
  ```
- [评估循环](file:///workspace/examples/mnist_mlp.py#L150-L163)：用 `with no_grad():` 上下文跳过计算图构建，提升推理速度并降低内存。

## 注意事项

1. **首次运行需联网下载 MNIST**：约 11 MB（4 个 gzip 文件）。下载后缓存到 `datasets/raw/mnist/`，后续运行离线可用。
2. **数据归一化**：仅做 `/255.0` 缩放到 `[0, 1]`，未做标准化（mean/std）。在 5 epoch 内已能达标；若进一步压缩到 1~2 epoch 可考虑加 `Normalize((0.1307,), (0.3081,))`。
3. **梯度累积**：`optimizer.zero_grad()` 必须显式调用，否则梯度会跨 batch 累积（PyTorch 语义）。
4. **未启用 Dropout**：MLP 太小（101k 参数），5 epoch 内不会过拟合到影响测试准确率。
5. **CPU 性能**：在 4 核 CPU 上单 epoch 约 30~60 秒，5 epoch 总耗时约 3~5 分钟。如需加速可考虑：
   - 用 `pip install numba` 后由 VerseTorch 内部加速（如有）；
   - 减小隐藏层到 64 维（仍可达到 95%）。
6. **`Tensor` 与 `np.ndarray` 互操作**：直接通过 `Tensor(np_array, requires_grad=False)` 构造；评估时取 logits 用 `logits.data` 拿到 `np.ndarray`，再用 `np.argmax`。注意 `_GRAD_ENABLED` 默认为 `True`，所以推理时务必 `with no_grad():`。
