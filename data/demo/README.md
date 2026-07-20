# CometSpark-v0.1

> 基于 VerseNext 的端到端语言模型训练仓库，纯 Python / 纯 CPU 一键训练。
>
> 这是 Verse 框架"第二次进化"阶段的产物，演示如何用 `verse_torch` + `verse_nex` + `verse_tokenizer` + `verse_inference` 拼装出一个完整可跑的 LM 训练 / 评估 / 生成流程。

## 一键运行

```bash
cd /workspace/data/demo
python run.py
```

可选参数：

```bash
python run.py --skip-build       # 跳过 tokenizer 构建
python run.py --skip-train       # 跳过训练，直接加载 best.pt 评估
python run.py --skip-eval        # 跳过评估
python run.py --config path/to/config.yml

# Task 7: 自定义 prompt / 采样参数
python run.py --prompt "床前明月光，,你好，"          # 逗号分隔多条 prompt
python run.py --prompts-file my_prompts.txt         # 每行一个 prompt（忽略空行与 # 注释）
python run.py --max-tokens 50                       # 每条 prompt 生成最大 token 数（默认 30）
python run.py --temperature 0.8 --top-k 10          # 采样温度 + top-k
python run.py --skip-train --prompt "你好" --max-tokens 50  # 用已训练模型评估自定义 prompt

# Task 9: 架构覆盖
python run.py --arch hybrid                         # 覆盖 config 的 model.arch 字段（transformer / hybrid）
python run.py --config config/config_medium.yml     # 使用 medium 配置（~850K 参数）
```

### 三套预设配置（Task 9）

`config/` 目录下提供三套预设配置，覆盖不同参数量与场景：

| 配置文件 | 参数量 | n_layer / n_embd / seq_len / batch_size | max_steps | wall-clock（5GB 沙箱） | 适用场景 |
|---|---|---|---|---|---|
| `config_small.yml` | ~131K | 2 / 64 / 64 / 8 | 200 | ~8 秒 | PoC 端到端验证（默认） |
| `config_medium.yml` | ~853K | 4 / 128 / 128 / 4 | 60 | ~14 秒 | 验证模型容量提升 |
| `config_large.yml` | ~3M | 6 / 192 / 128 / 4 | 200 | 5GB 沙箱下可能 OOM | 大内存 / GPU 环境；CPU 沙箱需配合 INT8 量化 |

参数量说明：
- `tie_weights=true` 时 embedding 与 lm_head 共享权重，参数量约为不共享时的 1/2 ~ 2/3
- `count_parameters()` 在 `CometSparkLM.__init__` 末尾自动打印（`[model] arch=xxx parameters: N`）
- 三套配置均默认 `arch=transformer`，可用 `--arch hybrid` 切换到 SSM + Sparse Attention 混合架构

medium 配置内存调优说明：
- 5GB 沙箱下原 `batch_size=8 max_steps=200` 会因 `verse_torch` 计算图累积在 step 80~100 触发 SIGKILL
- 默认调整为 `batch_size=4 max_steps=60`，可在 5GB 沙箱下 14 秒内完成训练 + 评估
- 如需更长训练，建议在更大内存环境运行，或降低 `batch_size` / `max_steps`

large 配置说明：
- 5GB 沙箱下默认参数会 OOM，建议配合 INT8 量化（`verse_compat`）使用
- 或降低 `batch_size=2` `max_steps=50` 后再运行

## 目录结构

```
data/demo/
├── run.py                # 一键入口：build_tokenizer → train → evaluate → visualize
├── config/
│   ├── config.yml        # 默认配置（同 small）
│   ├── config_small.yml  # ~131K 参数（PoC 验证）
│   ├── config_medium.yml # ~853K 参数（容量提升）
│   └── config_large.yml  # ~3M 参数（需大内存或量化）
├── model/
│   ├── config.py         # CometSparkConfig dataclass + from_yaml / to_yaml
│   ├── model.py          # CometSparkLM（支持 arch="hybrid" | "transformer"，含 count_parameters）
│   └── tokenizer.py      # build_tokenizer + load_tokenizer
├── src/
│   ├── utils.py          # set_seed / num_threads / ensure_dir
│   └── data_loader.py    # load_jsonl + TextDataset + collate_fn
├── train/
│   ├── trainer.py        # 包装 verse_torch.training.Trainer
│   ├── evaluate.py       # 加载 best.pt 生成示例文本（支持自定义 prompt / temperature / top_k）
│   └── visualize.py      # plot_loss_curve + ASCII fallback
├── data/
│   ├── train.jsonl       # 训练集（唐诗 + 简单问答 + 数字序列）
│   ├── val.jsonl         # 验证集
│   └── README.md         # 数据格式说明
└── checkpoints/          # 训练产物（自动生成）
    ├── tokenizer.json
    ├── best.pt           # 最佳验证 loss 模型
    ├── last.pt           # 最后一步模型
    ├── cometspark.pt     # 完整模型（config + state_dict）
    ├── loss_history.json # 逐 step 训练 / 验证 loss
    └── loss_curve.txt    # ASCII loss 曲线（matplotlib 不可用时 fallback）
```

## CLI 参数说明

`run.py` 支持以下参数：

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `config/config.yml` | 配置文件路径 |
| `--skip-build` | `False` | 跳过 tokenizer 构建（已有 tokenizer.json 时使用） |
| `--skip-train` | `False` | 跳过训练阶段（仅 build + eval） |
| `--skip-eval` | `False` | 跳过评估阶段 |
| `--force-build` | `False` | 强制重建 tokenizer（覆盖已有文件） |
| `--verbose` | `False` | 异常时打印完整 traceback（用于调试） |

### Task 7: 自定义 Prompt 支持

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--prompt` | `None` | 自定义评估 prompt，逗号分隔多条（如 `--prompt "床前明月光，,你好，"` → 2 条） |
| `--prompts-file` | `None` | 从文件读取 prompt（每行一个，忽略空行与 `#` 注释行） |
| `--max-tokens` | `30` | 每条 prompt 生成最大 token 数 |
| `--temperature` | `1.0` | 采样温度（1.0 等价 greedy；>1 增加随机性；<1 收敛） |
| `--top-k` | `None` | top-k 采样（None 表示 greedy） |

优先级：`--prompt` > `--prompts-file` > 默认 5 条（`床前明月光，` / `白日依山尽，` / `你好，` / `1+1=` / `春风`）

### Task 9: 模型架构覆盖

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--arch` | `None`（用 config 的值） | 覆盖 config 的 `model.arch` 字段（`transformer` / `hybrid`） |

`--arch` 通过创建临时 config 文件实现覆盖（不修改原 config.yml，运行后自动清理）。
hybrid 架构在 verse_nex Mamba2 数值溢出修复后已可启用（seq_len=128 下无 NaN）。

## 配置参数说明

`config/config.yml` 包含五个 section：

### `model`
| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `arch` | str | `transformer` | 架构，可选 `transformer` / `hybrid`（hybrid 已修复数值溢出，可启用） |
| `vocab_size` | int | `256` | 词表大小（byte tokenizer 为 256 字符 + 3 特殊 token） |
| `n_layer` | int | `2` | Transformer / Hybrid block 数量 |
| `n_head` | int | `4` | 注意力头数 |
| `n_embd` | int | `64` | 模型维度 |
| `seq_len` | int | `64` | 上下文长度 |
| `dropout` | float | `0.1` | dropout 概率 |
| `n_kv_head` | int | `2` | GQA 的 KV 头数（n_head // n_kv_head 为 repeat 因子） |
| `ssm_kind` | str | `mamba2` | hybrid 架构下的 SSM 种类 |
| `sparse_ratio` | float | `0.5` | hybrid block 中 sparse attention 比例 |
| `tie_weights` | bool | `true` | 是否共享 embedding 与 head 权重（影响参数量） |

### `training`
| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `batch_size` | int | `8` | 训练 batch |
| `micro_batch` | int | `4` | 梯度累积时的 micro batch |
| `lr` | float | `0.003` | 学习率 |
| `weight_decay` | float | `0.01` | AdamW 权重衰减 |
| `max_steps` | int | `200` | 训练步数 |
| `warmup` | int | `20` | warmup 步数 |
| `eval_interval` | int | `20` | 评估间隔 |
| `patience` | int | `5` | EarlyStopping patience |
| `grad_accum` | int | `1` | 梯度累积步数 |
| `log_interval` | int | `20` | 日志打印间隔 |
| `seed` | int | `42` | 随机种子 |

### `tokenizer`
| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `kind` | str | `byte` | `byte` / `bpe` / `hf` |
| `vocab_size` | int | `259` | BPE 词表大小（仅 kind=bpe 时生效） |

### `data`
| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `train_path` | str | `data/train.jsonl` | 训练集路径（相对 base_dir） |
| `val_path` | str | `data/val.jsonl` | 验证集路径（相对 base_dir） |

### `checkpoint`
| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `save_dir` | str | `checkpoints` | 检查点保存目录（相对 base_dir） |

## 数据格式

`train.jsonl` / `val.jsonl` 每行一个 JSON 对象，至少包含 `text` 字段：

```json
{"text": "床前明月光，疑是地上霜。"}
```

详见 `data/README.md`。

## 端到端验收（PoC 环境）

在 3 核 CPU / 5GB 内存沙箱中，默认 small 配置下：

| 指标 | 实测 |
|---|---|
| wall-clock | ~8 秒 |
| 参数量 | 131,776（~131K） |
| 初始 train loss | 5.59 |
| 最终 train loss | 2.44 |
| 最佳 val loss | 2.36 |
| 生成样本数 | 5 条 |
| checkpoints | best.pt / last.pt / cometspark.pt / loss_history.json / loss_curve.txt |

medium 配置（`config_medium.yml`）实测：

| 指标 | 实测 |
|---|---|
| wall-clock | ~14 秒 |
| 参数量 | 853,888（~853K） |
| 初始 train loss | 5.59 |
| 最终 train loss | 3.29 |
| 最佳 val loss | 3.34 |

## 依赖

仅依赖 `verse_torch` / `verse_nex` / `verse_tokenizer` / `verse_inference`（运行时不需要 PyTorch / TensorFlow / JAX / transformers）。
