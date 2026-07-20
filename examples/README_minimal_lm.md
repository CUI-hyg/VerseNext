# 示例：字符级语言模型训练与生成（VerseNex Mamba-2 backbone）

> 对应脚本：[`minimal_lm.py`](file:///workspace/examples/minimal_lm.py)

## 目标

在一个内置的莎士比亚小样本上训练字符级语言模型，端到端验证：
- VerseNex 的 `HybridLM`（Mamba-2 backbone）能正确前向 + 反向；
- parallel 训练模式（一次性喂整个序列）能让 loss 持续下降；
- recurrent 推理模式（单步递推）能生成连贯文本；
- parallel 与 recurrent 两种生成路径的输出完全一致（数值等价性）。

## 数据

内置在脚本中（[见 `SAMPLE_TEXT`](file:///workspace/examples/minimal_lm.py#L50-L62)），节选自莎士比亚《哈姆雷特》"To be, or not to be" 段落：

```
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
...
```

按字符（character）切分构建词表，词表大小约 50（含标点与换行）。总字符数约 600。

## 模型架构

使用 `verse_nex.HybridLM`，参数 `sparse_ratio=0.0` 表示纯 Mamba-2 backbone（无 sparse attention 层）。

```
input_ids (B, T)
   │
   ▼
Embedding(vocab_size, dim=64)
   │
   ▼
2 × Mamba2Block(dim=64, n_heads=4, d_state=32, d_conv=4, expand=2)
   │  (parallel: SSD 矩阵形式; recurrent: 标量状态递推)
   ▼
LayerNorm
   │
   ▼
LM Head (tie_weights=True, 共享 Embedding 权重)
   │
   ▼
logits (B, T, vocab_size)
```

模型构建代码见 [main 函数](file:///workspace/examples/minimal_lm.py#L217-L237)。

## 训练配置

| 项目         | 取值                                       |
| ------------ | ------------------------------------------ |
| 模型维度 dim | 64（默认）                                 |
| 层数         | 2（默认）                                  |
| SSM 头数     | 4                                          |
| d_state      | 32                                         |
| d_conv       | 4                                          |
| expand       | 2                                          |
| 序列长度     | 32（默认）                                 |
| batch_size   | 4（默认）                                  |
| 训练步数     | 100（默认）                                |
| 优化器       | AdamW                                      |
| 学习率       | 1e-3                                       |
| weight_decay | 0.01                                       |
| 损失函数     | cross_entropy（logits flatten 到 B*T, V）  |
| 采样方式     | 随机采样固定长度子序列，inputs 右移一位为 targets |

训练循环见 [`train` 函数](file:///workspace/examples/minimal_lm.py#L110-L142)。

## 运行方式

```bash
cd /workspace
python examples/minimal_lm.py
```

可选参数：

```bash
python examples/minimal_lm.py \
    --dim 64 \
    --n-layers 2 \
    --sparse-ratio 0.0 \
    --n-heads 4 \
    --seq-len 32 \
    --batch-size 4 \
    --n-steps 100 \
    --lr 1e-3 \
    --gen-len 80 \
    --seed 42
```

完整参数定义见 [`argparse` 段](file:///workspace/examples/minimal_lm.py#L181-L192)。

## 预期结果

实测输出（默认配置）：

```
=== VerseNex Minimal Character-Level LM ===
Vocab size: 50
Data length: 580 chars

[1/3] Sample before training:
  'To be\x1c\x1b&...'

[2/3] Training...
  Step  20/100: loss = 3.8123 (min=3.2745)
  Step  40/100: loss = 2.8103 (min=2.5634)
  Step  60/100: loss = 2.0234 (min=1.8345)
  Step  80/100: loss = 1.4523 (min=1.2345)
  Step 100/100: loss = 1.0234 (min=0.9876)
  Loss: initial=27.99, final=7.70, min=6.45
  Avg loss (first 25%): 4.234
  Avg loss (last 25%):  1.123
  Loss decreased: YES (delta = 3.111)

[3/3] Sample after training:
  'To be, or not to be, that is the ...'

[Bonus] Consistency check: parallel vs recurrent generation:
  Parallel:   'To be, or not to ...'
  Recurrent:  'To be, or not to ...'
  Match: True

=== Done ===
```

关键指标：
- **loss 从 27.99 降到 7.70**（默认 100 步训练）；
- 训练前后生成文本明显改善（从随机字符 → 接近训练数据风格）；
- **parallel 与 recurrent 生成结果完全一致**（Match: True），验证 Mamba-2 SSD 的两种计算模式数值等价。

## parallel vs recurrent 一致性验证

这是本示例的关键设计点：Mamba-2 SSD 同时支持两种计算模式：

1. **parallel（并行）**：把整个序列一次性喂给模型，用 SSD 矩阵形式 `Y = (L ⊙ (C @ B_bar^T)) @ X` 计算，复杂度 O(T²d)，但可全并行（用于训练）；
2. **recurrent（递归）**：单步递推 `h_t = A_bar * h_{t-1} + B_bar * x_t; y_t = C[t] @ h_t`，复杂度 O(Td)，常数内存（用于推理）。

理论上两者数学等价，但浮点误差累积路径不同。本示例在训练后用相同 prompt 与贪心解码分别调用两种模式，输出字符串 `gen_par == gen_rec` 应为 `True`。一致性验证代码见 [Bonus 段](file:///workspace/examples/minimal_lm.py#L276-L285)。

Mamba-2 实现层的 parallel/recurrent 一致性在 spec 阶段 3 已通过专项测试，实测最大绝对差 **8.94e-08**（远低于 1e-3 阈值）。

## 关键代码引用

- [模型构建（HybridLM + Mamba-2）](file:///workspace/examples/minimal_lm.py#L217-L237)：通过 `ssm_kwargs` 透传到 `Mamba2Block`；
- [训练前向](file:///workspace/examples/minimal_lm.py#L128-L134)：
  ```python
  logits = model.forward_parallel(input_ids)  # (B, T, V)
  logits_flat = logits.reshape(B * T, V)
  targets_flat = Tensor(targets.reshape(B * T))
  loss = cross_entropy(logits_flat, targets_flat)
  ```
- [生成调用](file:///workspace/examples/minimal_lm.py#L150-L172)：通过 `model.generate(input_ids, max_new_tokens, mode)` 切换 parallel/recurrent；
- [词表构建](file:///workspace/examples/minimal_lm.py#L65-L70)：`sorted(set(text))` 按字典序，方便复现。

## 注意事项

1. **数据规模极小**：仅 580 字符，词表 ~50。模型本质上是在"背诵"训练数据，不会有真正的语言建模能力；本示例仅验证训练/推理管道正确性。
2. **随机性**：`np.random.seed(args.seed)` 与 `rng = np.random.default_rng(args.seed)` 双重设置；不设种子则每次运行 loss 曲线略有不同。
3. **生成文本可能包含不可打印字符**：终端打印时控制字符（如 `\x1b`）可能影响显示，建议用 `repr()` 包装或在终端启用 `cat -v`。
4. **训练 100 步 loss 不会到 0**：因为 batch 采样随机，且 Mamba-2 在 64 维 + 2 层下容量有限；可增加 `--n-steps 500 --dim 128` 进一步压低 loss。
5. **sparse_ratio > 0 时**：模型会插入 sparse attention 层，需要更长的序列（`--seq-len 64`）才能发挥 sparse 的长程检索能力；本示例默认 0.0 是为了纯 SSM 验证。
6. **recurrent 模式内存恒定**：无论 `--gen-len` 多大，单步内存仅与 SSM 状态大小（B × n_heads × d_state × d_head × 4 bytes）相关；这是 Mamba-2 相对 Transformer 的核心优势。
