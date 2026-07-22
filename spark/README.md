# CometSpark V0.5-1B

Part4K1 Task 8：CometSpark V0.5-1B 模型迁移 + 完全重写。
Part4K2.5：新增 `run.py` CLI 快捷入口与 `_bootstrap.py` 统一路径引导。

## 目录结构

```
spark/
  _bootstrap.py              # 统一路径引导模块（Part4K2.5）
  run.py                     # CLI 快捷入口（7 子命令，Part4K2.5）
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

## spark/run.py 快捷入口（Part4K2.5 新增）

`spark/run.py` 是基于 VerseTrainer API 封装的命令行快捷入口，提供 7 个子命令，**所有命令都有合理默认值，最小化用户配置**。无需 `pip install` 即可直接 `python spark/run.py <子命令>` 运行。

### 7 个子命令

| 子命令 | 用途 | 示例 |
|---|---|---|
| `train` | 训练模型（训练后默认自动评估） | `python spark/run.py train --small` |
| `eval` | 评估 + 打分 | `python spark/run.py eval --checkpoint checkpoints/best.pt --score` |
| `generate` | 生成文本 | `python spark/run.py generate --prompt "你好世界"` |
| `chat` | 交互式聊天（`/quit` / `/clear` / `/save`） | `python spark/run.py chat` |
| `compress` | 压缩模型（prune/quantize/lora/ternary） | `python spark/run.py compress --checkpoint ck.pt --method prune,quantize` |
| `convert` | 模型格式互转（`.pt ↔ .vn`） | `python spark/run.py convert --input ck.pt --output m.vn` |
| `download` | 下载数据集（URL + HF + 自动转 .npz） | `python spark/run.py download --hf wikitext --split train` |

### 通用参数

- `--dry-run`：只打印将要执行的操作而不真正执行（所有子命令均支持）。
- `--config`：指定配置文件路径（`train` 不指定时用 1B 默认配置，`--small` 用小配置）。
- `--quiet` / `--verbose`：静默模式（仅打印最终结果）/ 详细日志模式。

### train 子命令参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--small` | False | 使用小配置（`cometspark_v05_small.yml`，快速调试） |
| `--config` | None | 配置文件路径（覆盖 `--small`） |
| `--max-steps` | None | 覆盖 config 的 max_steps |
| `--batch-size` | None | 覆盖 config 的 batch_size |
| `--device` | auto | `cpu` / `cuda` / `npu`（默认从 config 读取） |
| `--amp` | False | 启用混合精度 |
| `--resume` | False | 从 checkpoint 断点续训 |
| `--eval-after` / `--no-eval` | True | 训练后自动评估（默认开启，`--no-eval` 跳过） |

> **提示**：`spark/run.py` 与 VerseTrainer CLI（`verse-train` 等）功能等价，底层都调用 `verse_infra.verse_trainer.train()`，训练结果完全一致。区别在于前者零安装可用、子命令更精简、默认值更友好（`--small` 一键调试、`--eval-after` 默认开）。

## _bootstrap.py 路径引导（Part4K2.5 新增）

`spark/_bootstrap.py` 是统一的路径引导模块，在项目未做 `pip install` 时确保所有包都能正确导入。

### 工作原理

- 基于 `__file__` 推断 `<repo_root>`，不硬编码 `/workspace`。
- 幂等：多次调用 `ensure_paths()` 不会重复添加路径。
- 模块导入时自动执行一次 `ensure_paths()`，后续显式调用是安全的空操作。
- 仅注入项目自身包目录（`verse_torch` / `verse_nex` / `verse_infra` / `spark` / `data`），不污染 `sys.path` 与第三方包命名空间。

### 使用方式

```python
# 入口文件（spark/run.py、CLI 入口、测试）只需导入一次
import spark._bootstrap  # 自动设置好所有路径

# 或显式调用
from spark._bootstrap import ensure_paths
ensure_paths()
```

Part4K2.5 之前，`spark/__init__.py`、`spark/run.py`、CLI 入口等 6 处各自重复 `sys.path.insert` 路径自举；Task 2 把它们统一收敛为单次 `import spark._bootstrap`，消除 `sys.path` 膨胀与跨路径导入风险。

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
# 最快验证：spark/run.py 快捷入口（小配置，零安装可用）
python spark/run.py train --small

# 预训练（CPU）
verse-train --config spark/config/cometspark_v05.yml --device cpu

# 调试小配置（快速跑通）
verse-train --config spark/config/cometspark_v05_small.yml --device cpu --max-steps 10

# spark/run.py 等价命令（训练后默认自动评估，--no-eval 跳过）
python spark/run.py train --no-eval --max-steps 10

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

# spark/run.py 等价命令
python spark/run.py eval --checkpoint checkpoints/cometspark.pt

# 打分模式（需 references 文件）
verse-eval --config spark/config/cometspark_v05.yml --score --references-file references.txt

# spark/run.py 等价命令
python spark/run.py eval --checkpoint checkpoints/cometspark.pt --score --references-file references.txt
```

### 生成 / 聊天（CLI，spark/run.py）

```bash
# 生成文本
python spark/run.py generate --prompt "床前明月光，" --temperature 0.8

# 交互式聊天（支持 /quit /clear /save 命令）
python spark/run.py chat --checkpoint checkpoints/best.pt

# 模型格式互转（.pt ↔ .vn）
python spark/run.py convert --input checkpoints/best.pt --output model.vn

# 压缩模型（剪枝 + 量化）
python spark/run.py compress --checkpoint checkpoints/best.pt --method prune,quantize

# 下载数据集
python spark/run.py download --url https://example.com/data.jsonl --to-npz -o data/cached.npz
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
