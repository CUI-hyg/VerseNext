# CometSpark V0.5-1B

Part4K1 Task 8：CometSpark V0.5-1B 模型迁移 + 完全重写。

## 目录结构

```
spark/
  config/
    cometspark_v05.yml       # 1B 默认配置（vocab 248320, n_embd=1024, n_layer=20）
    cometspark_v05_small.yml # 调试小配置（vocab 256, n_embd=64, n_layer=2）
  model/
    __init__.py
    config.py                # CometSparkV05Config
    model.py                  # CometSparkV05LM + CometSparkV05 / CometSparkV05Small
  src/
    __init__.py
    data_loader.py            # 委托 verse_infra.verse_trainer.data
    trainer.py                # 委托 verse_infra.verse_trainer
    evaluate.py               # 委托 verse_infra.verse_trainer.evaluate
    utils.py                  # set_seed / num_threads / load_qwen_tokenizer
  README.md
```

## 设计要点

- **不重造底层 block**：`CometSparkV05LM` 组合 `verse_nex.CometSparkNexLM`
  （内部 `VerseNexBlock` = TriSparse + MoD），本包只做"架构优化 + 工厂 + 持久化"。
- **1B 参数预算**：`CometSparkV05()` 通过 `n_embd=1024, n_layer=20,
  5 MoD + 15 trisparse, 4 DensePart × 4 Expert × top-2` + `tie_weights=True`
  + `embedding_scale=True` 达到 ≈ 1.12B 参数（落在 0.8B-1.2B 区间）。
- **解决胡乱输出**（Task 8.7）：
  - embedding scale：`tok_emb(idx) * sqrt(n_embd)`
  - tie_weights：`lm_head` 与 `tok_emb` 共享权重
  - temperature scaling：生成时 `logits / temperature`
  - 合理初始化（normal + 残差缩放）
- **全面接入新框架**：
  - `spark/src/trainer.py` 调用 `verse_infra.verse_trainer`（VerseTrainer / ParallelTrainerSafe）
  - tokenizer 用 Qwen3.5-35B-A3B（通过 `BPETokenizer.from_pretrained`）
  - 导入用 `from verse_infra.verse_trainer import ...` / `from verse_infra.verse_tokenizer import ...`

## 用法

### 构建模型

```python
from spark.model.model import CometSparkV05, CometSparkV05Small

# 1B 模型（≈ 1.12B 参数）
model = CometSparkV05()
print(f"参数量: {model.count_parameters() / 1e9:.2f}B")

# 调试小配置（≈ 0.1M 参数）
small = CometSparkV05Small()
```

### 训练（CLI）

```bash
# 预训练（CPU）
verse-train --config spark/config/cometspark_v05.yml --device cpu

# 调试小配置（快速跑通）
verse-train --config spark/config/cometspark_v05_small.yml --device cpu --max-steps 10

# 并行训练（chunks > 1）
verse-train --config spark/config/cometspark_v05.yml --parallel-chunks 4

# 断点续训
verse-train --config spark/config/cometspark_v05.yml --resume

# 混合精度（GPU）
verse-train --config spark/config/cometspark_v05.yml --device cuda --amp
```

### 微调 / 后训练（CLI）

```bash
# LoRA 微调
verse-finetune --config spark/config/cometspark_v05.yml --method lora --device cpu

# 全量微调
verse-finetune --config spark/config/cometspark_v05.yml --method full

# NexRL 后训练（强化学习）
verse-posttrain --config spark/config/cometspark_v05.yml --rl nexrl --device cpu

# SFT 后训练
verse-posttrain --config spark/config/cometspark_v05.yml --rl sft

# DPO 后训练
verse-posttrain --config spark/config/cometspark_v05.yml --rl dpo
```

### 评估 + 打分（CLI）

```bash
# 评估（生成示例文本）
verse-eval --config spark/config/cometspark_v05.yml --checkpoint checkpoints/cometspark.pt

# 打分模式（需 references 文件）
verse-eval --config spark/config/cometspark_v05.yml --score --references-file references.txt
```

### Tokenizer（CLI）

```bash
# 从 HuggingFace 下载 Qwen tokenizer
verse-tokenize --from-hf Qwen/Qwen3.5-35B-A3B --save spark/config/tokenizer.json
```

### Python API

```python
from spark.model.model import CometSparkV05
from spark.src.utils import load_qwen_tokenizer

# 加载 Qwen tokenizer
try:
    tok = load_qwen_tokenizer("Qwen/Qwen3.5-35B-A3B")
    print(f"vocab_size: {len(tok)}")  # 248320
except RuntimeError as e:
    print(f"网络不可用，跳过: {e}")

# 构建模型
model = CometSparkV05(vocab_size=len(tok) if tok else 256)

# 生成
import numpy as np
prompt_ids = np.array([[1, 2, 3]], dtype=np.int64)
out = model.generate(prompt_ids, max_new_tokens=32, temperature=1.0)
print(out.shape)  # (1, 35)
```

## 参数预算（1B 默认配置）

`cometspark_v05.yml` 默认配置：`n_embd=1024, n_layer=20, 5 MoD + 15 trisparse, 4 DensePart × 4 Expert × top-2, tie_weights=True, embedding_scale=True`。

| 组件 | 参数量 |
|------|--------|
| Embedding (tie, vocab=248320, d=1024) | 254M |
| 15 × trisparse 层 (qkv + proj + SwiGLU) + 5 × MoD 层 (4 parts × 4 experts × 2×1024×2688) | ~861M |
| **总** | **≈ 1115M ≈ 1.12B** |

## 依赖

- `verse_torch`（Tensor / nn / optim / training）
- `verse_nex`（CometSparkNexLM + VerseNexBlock + MoDLayer + TriSparseAttention）
- `verse_infra.verse_trainer`（VerseTrainer / ParallelTrainerSafe / RLTrainer）
- `verse_infra.verse_tokenizer`（BPETokenizer + Qwen tokenizer 加载）
