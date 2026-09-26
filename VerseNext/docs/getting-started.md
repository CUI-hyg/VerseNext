# 快速开始

## 安装

```bash
# uv（推荐）
uv sync

# 或 pip
pip install -e src/verse-core -e src/verse-nn -e src/verse-trainer -e src/verse-rsi -e src/verse-cli
```

## Python API 训练

```python
import torch
from verse_nn.transformer import VerseTransformer, VerseTransformerConfig
from verse_trainer.trainer import Trainer, TrainerConfig
from verse_trainer.data import DataConfig

config = TrainerConfig(
    model=VerseTransformerConfig(vocab_size=1000, d_model=256, num_layers=4, num_heads=4),
    data=DataConfig(batch_size=8, seq_len=128),
    steps=200,
)
trainer = Trainer(config)

# tokens: 一维 token id 序列（此处以随机数据演示）
tokens = torch.randint(0, 1000, (20000,)).tolist()
summary = trainer.train(tokens, eval_tokens=tokens[-8000:])
print(summary)
```

## Python API 自进化（RSI）

```python
from verse_rsi.evolve import EvolveConfig, Evolver
from verse_rsi.individual import Individual
from verse_nn.transformer import VerseTransformerConfig

evolver = Evolver(EvolveConfig(generations=3, population_size=3))
best = evolver.run(train_tokens=tokens, eval_tokens=tokens[-8000:])
print(best.model_config, best.fitness)
```

## CLI

```bash
# 训练
verse train --steps 200 --d-model 256

# 基准测速
verse bench --d-model 512 --num-layers 6

# RSI 自进化（进化期短训 60 步筛选，胜出者重训 300 步）
verse evolve --generations 3 --population 3 --refine-steps 300
```

## 真实数据接入

CLI 中的 `_make_tokens` 只是合成数据占位。接入真实数据：把 tokenizer 输出的
token id 序列传入 `Trainer.train(tokens, eval_tokens)`，或仿照
`TokenBatchIterator` 实现自己的数据迭代器。

## 后训练 / 微调 / 分词器

见 [post-training.md](post-training.md)：BPE 分词器训练、chat_template.jinja、
LoRA SFT 微调、对话生成与 CPU 推理优化。
