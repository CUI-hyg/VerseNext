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
```

## 目录结构

```
data/demo/
├── run.py                # 一键入口：build_tokenizer → train → evaluate → visualize
├── config/
│   └── config.yml        # 模型 / 训练 / tokenizer / data / checkpoint 配置
├── model/
│   ├── config.py         # CometSparkConfig dataclass + from_yaml / to_yaml
│   ├── model.py          # CometSparkLM（支持 arch="hybrid" | "transformer"）
│   └── tokenizer.py      # build_tokenizer + load_tokenizer
├── src/
│   ├── utils.py          # set_seed / num_threads / ensure_dir
│   └── data_loader.py    # load_jsonl + TextDataset + collate_fn
├── train/
│   ├── trainer.py        # 包装 verse_torch.training.Trainer
│   ├── evaluate.py       # 加载 best.pt 生成示例文本
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

## 配置参数说明

`config/config.yml` 包含五个 section：

### `model`
| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `arch` | str | `transformer` | 架构，可选 `transformer` / `hybrid`（hybrid 待 verse_nex Mamba2 数值修复后启用） |
| `vocab_size` | int | `256` | 词表大小（byte tokenizer 为 256 字符 + 3 特殊 token） |
| `n_layer` | int | `2` | Transformer / Hybrid block 数量 |
| `n_head` | int | `4` | 注意力头数 |
| `n_embd` | int | `64` | 模型维度 |
| `seq_len` | int | `64` | 上下文长度 |
| `dropout` | float | `0.1` | dropout 概率 |
| `n_kv_head` | int | `2` | GQA 的 KV 头数（n_head // n_kv_head 为 repeat 因子） |
| `ssm_kind` | str | `mamba2` | hybrid 架构下的 SSM 种类 |
| `sparse_ratio` | float | `0.5` | hybrid block 中 sparse attention 比例 |
| `tie_weights` | bool | `true` | 是否共享 embedding 与 head 权重 |

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

在 3 核 CPU / 5GB 内存沙箱中，默认配置下：

| 指标 | 实测 |
|---|---|
| wall-clock | ~9 秒 |
| 初始 train loss | 5.61 |
| 最终 train loss | 2.16 |
| 最佳 val loss | 2.28 |
| 生成样本数 | 5 条 |
| checkpoints | best.pt / last.pt / cometspark.pt / loss_history.json / loss_curve.txt |

## 依赖

仅依赖 `verse_torch` / `verse_nex` / `verse_tokenizer` / `verse_inference`（运行时不需要 PyTorch / TensorFlow / JAX / transformers）。
