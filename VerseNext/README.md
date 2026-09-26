# VerseNext

高性能神经网络架构框架：提供与 Transformer 等价的能力，重点优化训练性能与模型架构，并内置 **RSI（递归自进化）** 支持。

## 特性

- **模型架构**（`verse-nn`）
  - 可插拔注意力：SDPA 稠密（GQA/MQA）/ **DSA** 稀疏（indexer+top-k）/ **KDA** 线性（门控 delta 规则）
  - RoPE 旋转位置编码、pre-norm 残差结构、SwiGLU FFN（gate/up 融合 GEMM）
  - 静态预分配 KV Cache（零拷贝解码）+ 融合 QKV 投影 + `logits_to_keep`
  - 梯度检查点（gradient checkpointing）支持
  - LoRA 参数高效微调（apply / merge / 适配器独立保存）
- **分词与对话模板**（`verse-tokenizer`）
  - byte-level BPE：训练 / 编码 / 解码 / 字节偏移，任意文本无 OOV
  - chat_template.jinja（HF 约定，内置 ChatML 模板），SFT 回答掩码
- **训练性能**（`verse-trainer`）
  - 混合精度训练（AMP / bf16/fp16 autocast）
  - 梯度累积、梯度裁剪、torch.compile 加速
  - warmup + cosine 学习率调度、fused AdamW
  - 断点 checkpoint / 恢复训练
  - CPU 优化：自动线程数、flush denormal、int8 动态量化推理
  - SFT 对话微调（`SFTDataset`，只对回答计算 loss）
- **底层算子与设备专项优化**（`verse-nn`）
  - 自研融合算子：残差加+RMSNorm / SwiGLU / RoPE / FlashAttention / 分块 CE /
    KDA chunkwise，三级回退派发（自研 → torch 内建 → 参考），绝不因算子中断训练
  - CUDA / ROCm 手写 kernel（`csrc/cuda`，ROCm 由 hipify 复用），CANN NPU 走
    CANN 原生融合算子 + 自研 AscendC（`csrc/ascend`）
  - 设备抽象（`devices/`）：能力探测 + 后端注册 + **50% 资源锁**（CPU 线程 /
    CUDA·ROCm·NPU 显存 / NPU AI Core）
  - CPU 并行（线程池）、int8 动态量化、bf16/fp16 混合精度推理
  - 详见 [`docs/kernels.md`](docs/kernels.md)
- **RSI 递归自进化**（`verse-rsi`）
  - 评估 → 选择 → 变异（超参 + 架构）→ 再训练的递归进化循环
  - 变异算子可插拔、可扩展，种群档案持久化
- **基础设施**（`verse-core`）
  - 组件注册中心（Registry）：模型/优化器/变异算子均可注册替换
  - 事件总线：trainer 与 RSI 解耦通信

## 包结构

```
src/
├── verse-core/      # 基础设施：registry / config / logging / events
├── verse-nn/        # 模型架构 + 算子/设备：attention / RoPE / SwiGLU / transformer
│                    #   / LoRA / kernels（CPU·CUDA·ROCm·CANN）/ devices / csrc
├── verse-tokenizer/ # 分词器：byte-level BPE / chat_template.jinja
├── verse-trainer/   # 训练优化：trainer / optim / sft / inference / callbacks
├── verse-rsi/       # 递归自进化：evaluator / mutator / population / evolve
└── verse-cli/       # CLI：train / evolve / bench / train-tokenizer / sft / generate
```

## 快速开始

```bash
# 使用 uv（推荐）
uv sync
# 或 pip
pip install -e src/verse-core -e src/verse-tokenizer -e src/verse-nn -e src/verse-trainer -e src/verse-rsi -e src/verse-cli
```

训练：

```bash
verse train --steps 200
```

启动自进化（自动调整超参与架构）：

```bash
verse evolve --generations 5 --population 4
```

详见 `docs/`（架构、注意力、后训练、自进化、[算子与设备](docs/kernels.md)）。

## Changelog

见 [Changelogs.md](Changelogs.md)。
