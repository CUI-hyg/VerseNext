# CometSpark

CometSpark 是基于 VerseNex 原生架构（TriSparse + MoD）的端到端 LM 训练仓库，承载 **small** 与 **mate** 双模型并行。

- **Part5K1.1**：目录精简为 `small/` + `mate/` + `src/` 三目录；VMPC 升级到 V2.0（VSC 三维空间压缩）；`run.py` 支持 `--model small|mate` 自动调用 + 表格化信息显示。

## 目录结构（Part5K1.1）

```
spark/
  _bootstrap.py              # 统一路径引导模块（Part4K2.5）
  run.py                     # CLI 快捷入口（--model small|mate 自动调用 + 表格显示）
  small/                     # 0.06zB 小模型（VMPC-small 预设，端侧 / 嵌入式 / 树莓派）
    config/
      cometspark_small.yml   # small 配置（vocab=256, n_embd=64, n_layer=2）
    model/
      __init__.py
      config.py              # CometSparkSmallConfig（含 VMPC V2.0 字段）
      model.py               # CometSparkSmallLM + CometSparkSmall 工厂
  mate/                      # 0.2zB 旗舰模型（VMPC-mate 预设，消费级 CPU / 单卡 GPU）
    config/
      cometspark_mate.yml    # mate 配置（vocab=248320, n_embd=1024, n_layer=20）
    model/
      __init__.py
      config.py              # CometSparkMateConfig（含 VMPC V2.0 字段）
      model.py               # CometSparkMateLM + CometSparkMate 工厂
  src/                       # 共享基础组件
    __init__.py
    base_config.py           # CometSparkV05Config 基类（MoD / TriSparse / layer_pattern 固化）
    base_model.py            # CometSparkV05LM 基类（save / load / VMPC 接口）
    data_loader.py           # 委托 verse_infra.verse_trainer.data
    trainer.py               # 委托 verse_infra.verse_trainer
    evaluate.py              # 委托 verse_infra.verse_trainer.evaluate
    utils.py                 # set_seed / num_threads / load_qwen_tokenizer
  README.md
```

> **Part5K1.1 弃用说明**：旧版 `spark/config/` 与 `spark/model/` 目录已删除，统一迁移到 `small/` + `mate/` + `src/` 三目录结构。

## 双模型定位

| 模型 | 目录 | 目标 | VMPC 预设 | 适用场景 |
|---|---|---|---|---|
| **small** | `spark/small/` | 0.06zB（1zB ≈ 1010B 等效能力） | ternary 量化 + 高稀疏 0.5 | 端侧 / 嵌入式 / 树莓派 |
| **mate** | `spark/mate/` | 0.2zB | int4 量化 + 中稀疏 0.3 + LoRA + 蒸馏 | 消费级 CPU / 单卡 GPU |

> **Part5K1.1 等效能力**：在 VMPC V2.0 支持下，1zB ≈ 1010B 模型能力（等效处理）。物理压缩占 40%（量化 + 剪枝），专项算法优化 + 训练补偿占 60%。

## VMPC V2.0 + VSC（Part5K1.1）

Part5K1.1 将 VMPC 全面升级到 V2.0，明确 **VMPC ≠ 单纯的量化 / 剪枝**，而是基于：

1. **VN 格式文件**：高吞吐、高速度、方便压缩的模型容器（多空间缓存）。
2. **传统技术**：量化 / 剪枝 / 蒸馏 / LoRA（作为 VSC 的底层算子）。
3. **VSC（VerseNext Space Compression）**：从三维空间角度（存储 / 算力 / 时间）对模型做特别压缩，保持「速度快、能力强、占用小」三维优势。

### 配置统一开关

所有配置通过 `vmpc.use_vmpc` 单项管理（默认开启）：

```yaml
vmpc:
  use_vmpc: true              # 开启 VMPC V2.0（默认）
  # use_vmpc=true 时：强制 *.vn 格式 + VSC 引擎 + 训练/推理自动接入
  # use_vmpc=false 时：走 legacy 模式（传统技术直通，允许 .pt）

  # legacy 字段（use_vmpc=false 时生效）
  profile: small
  prune_sparsity: 0.5
  quantize_dtype: ternary

  # V2.0 专属参数（VSC 三维空间压缩）
  target_ratio: 0.06          # 压到原大小 6%
  quantize_bits: 2            # ternary 量化
  storage_weight: 0.5         # 存储维度权重
  compute_weight: 0.3         # 算力维度权重
  time_weight: 0.2            # 时间维度权重
  force_vn_format: true       # 强制 .vn 格式
  enable_compensation: true   # 训练补偿（恢复压缩损失的能力）
```

### VN 格式多空间缓存

`.vn` 文件支持按需在内存与硬盘间自动缓存：

- **混合缓存**（内存充足 + 硬盘充足）：优先级高的放内存，优先级低的放硬盘。
- **硬盘优先**（内存不充足时）：主要放硬盘。
- 通过 `VNCacheManager` 实现 LRU 内存缓存 + mmap 硬盘懒加载。

## spark/run.py 快捷入口

`spark/run.py` 提供 7 个子命令，**指定 `--model small|mate` 时无需再指定配置 / 权重目录，自动调用**。所有命令支持 `--dry-run` 预览。

### 子命令

| 子命令 | 用途 | 示例 |
|---|---|---|
| `train` | 训练模型（训练后默认自动评估） | `python spark/run.py train --model small` |
| `eval` | 评估 + 打分 | `python spark/run.py eval --model small --score` |
| `generate` | 生成文本 | `python spark/run.py generate --model small --prompt "你好"` |
| `chat` | 交互式聊天（`/quit` / `/clear` / `/save`） | `python spark/run.py chat --model small` |
| `compress` | 压缩模型（VMPC V2.0 默认 / `--no-vmpc` legacy） | `python spark/run.py compress --model small` |
| `convert` | 模型格式互转（`.pt ↔ .vn`） | `python spark/run.py convert --input m.pt --output m.vn` |
| `download` | 下载数据集（URL + HF + 自动转 .npz） | `python spark/run.py download --hf wikitext --split train` |

### --model 自动调用（Part5K1.1）

```bash
# 训练 small（自动用 spark/small/config/cometspark_small.yml，checkpoint 写 mf_small/）
python spark/run.py train --model small

# 训练 mate（自动用 spark/mate/config/cometspark_mate.yml，checkpoint 写 mf_mate/）
python spark/run.py train --model mate

# 生成 / 聊天也自动查找对应目录的最新 checkpoint
python spark/run.py generate --model small --prompt "你好世界"
python spark/run.py chat --model mate
```

### 表格化信息显示（Part5K1.1）

`run.py` 使用 ASCII 表格快速展现配置与模型信息（无外部依赖）：

```
[spark] Configuration
+------------------------+-----------+
| Field                  | Value     |
+------------------------+-----------+
| arch                   | versenex  |
| vocab_size             | 256       |
| n_layer                | 2         |
| use_vmpc               | True      |
| vmpc_profile           | small     |
| ...                    | ...       |
+------------------------+-----------+
```

### compress 双路径（Part5K1.1）

```bash
# VMPC V2.0 路径（默认）：VSC 引擎三维空间压缩 + 强制 .vn 输出
python spark/run.py compress --model small

# 启用训练补偿（算法优化 60% 的核心环节）
python spark/run.py compress --model small --compensate

# Legacy 路径：传统技术直通（prune/quantize/lora/ternary）
python spark/run.py compress --model small --no-vmpc --method prune,quantize
```

## 用法

### 构建模型

```python
from spark.small.model import CometSparkSmall
from spark.mate.model import CometSparkMate

# small 模型（0.06zB 目标，≈ 0.2M 参数调试配置）
small = CometSparkSmall()
print(f"small 参数量: {small.count_parameters() / 1e6:.2f}M")

# mate 模型（0.2zB 目标，≈ 1.12B 参数）
mate = CometSparkMate()
print(f"mate 参数量: {mate.count_parameters() / 1e9:.2f}B")
```

### 训练（CLI）

```bash
# 最快验证：small 模型（约 10-30 秒完成，零安装可用）
python spark/run.py train --model small

# mate 旗舰训练
python spark/run.py train --model mate

# 训练后跳过自动评估
python spark/run.py train --model small --no-eval

# 自定义步数 / 设备
python spark/run.py train --model mate --max-steps 1000 --device cuda --amp

# 断点续训
python spark/run.py train --model small --resume
```

### 压缩（CLI）

```bash
# VMPC V2.0 压缩（默认，强制 .vn 输出）
python spark/run.py compress --model small

# 启用训练补偿
python spark/run.py compress --model mate --compensate

# Legacy 压缩（传统技术直通）
python spark/run.py compress --model small --no-vmpc --method prune,quantize --sparsity 0.5
```

### 评估 / 生成 / 聊天（CLI）

```bash
# 评估（自动从 mf_small/ 找最新 checkpoint）
python spark/run.py eval --model small --score

# 生成文本
python spark/run.py generate --model small --prompt "床前明月光，" --temperature 0.8

# 交互式聊天
python spark/run.py chat --model small
```

### Python API

```python
from spark.small.model import CometSparkSmall
from spark.mate.model import CometSparkMate

# VMPC V2.0 压缩（返回新模型实例，不修改原模型）
small = CometSparkSmall()
compressed = small.vmpc_compress_model(use_vmpc=True)
print(f"压缩比: {small.count_parameters() / compressed.count_parameters():.2f}x")

# 保存（use_vmpc=True 时强制 .vn 格式）
compressed.save("model_vmpc.vn", format="vn")
```

## 参数预算

| 模型 | 配置 | 参数量 | VMPC 目标 |
|---|---|---|---|
| small | n_embd=64, n_layer=2, vocab=256 | ≈ 0.2M | 0.06zB |
| mate | n_embd=1024, n_layer=20, vocab=248320, 5 MoD + 15 trisparse | ≈ 1.12B | 0.2zB |

## 依赖

- `verse_torch`（Tensor / vnn / optim / training / vmpc / vsc / vn_format）
- `verse_nex`（CometSparkNexLM + VerseNexBlock + MoDLayer + TriSparseAttention）
- `verse_infra.verse_trainer`（VerseTrainer / ParallelTrainerSafe / RLTrainer）
- `verse_infra.verse_tokenizer`（BPETokenizer + Qwen tokenizer 加载）
