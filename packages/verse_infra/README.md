# VerseInfra

> 总包：聚合 `verse_tokenizer` / `verse_compat` / `verse_inference` / `verse_trainer` 四个子模块为单一可安装单元，统一版本与依赖；旧顶层包通过 thin shim 转发 + `DeprecationWarning` 保持向后兼容。

[返回主 README](../../README.md)

## 设计动机

Part4K1 之前，`verse_tokenizer` / `verse_compat` / `verse_inference` / `verse_trainer` 是 4 个独立的顶层包，用户需要分别 `pip install -e` 多次，且版本/依赖容易漂移。Part4K1 将它们聚合为 `verse_infra` 的子模块：

- **单包安装**：`pip install -e packages/verse_infra` 一次安装四个子模块
- **版本对齐**：所有子模块共享 `verse_infra.__version__`
- **便捷重导出**：常用 API（`BPETokenizer` / `ModelLoader` / `train` / `RLTrainer` 等）可直接从 `verse_infra` 顶层导入
- **shim 兼容**：旧导入路径仍可工作（`from verse_tokenizer import ...` 仍可用），但发出 `DeprecationWarning`

`verse_torch` / `verse_nex` 保持独立未并入 VerseInfra（它们是底层后端，依赖关系不同）。

## 子模块结构

```
packages/verse_infra/
├── pyproject.toml                  # 总包元数据 + 依赖声明
└── verse_infra/
    ├── __init__.py                 # 便捷重导出 + __getattr__ 延迟导入
    ├── verse_tokenizer/            # BPE / Unigram / WordPiece 分词器
    │   └── README.md
    ├── verse_compat/               # HuggingFace / PyTorch 兼容层
    │   └── README.md
    ├── verse_inference/            # 模型加载 / 状态缓存 / 流式生成
    │   └── README.md
    └── verse_trainer/              # 训练 CLI（verse-train/finetune/posttrain/eval/tokenize）
```

## 安装

```bash
# 方式一：pip 可编辑安装
pip install -e packages/verse_infra

# 方式二：uv workspace（在仓库根目录执行）
uv sync
```

## 导入路径迁移指南

### 旧路径 → 新路径对照表

| 子模块 | 旧路径（已废弃，仍可工作） | 新路径（推荐） |
|---|---|---|
| `verse_tokenizer` | `from verse_tokenizer import BPETokenizer` | `from verse_infra.verse_tokenizer import BPETokenizer` |
| `verse_compat` | `from verse_compat import load_hf_state_dict` | `from verse_infra.verse_compat import load_hf_state_dict` |
| `verse_inference` | `from verse_inference import ModelLoader` | `from verse_infra.verse_inference import ModelLoader` |
| `verse_trainer` | （新模块，无旧路径） | `from verse_infra.verse_trainer import train` |

### 便捷重导出

`verse_infra` 顶层导出了各子模块的常用 API，可直接 `from verse_infra import X`：

```python
# tokenizer
from verse_infra import BPETokenizer, ByteTokenizer, WordPieceTokenizer
from verse_infra import SentencePieceUnigramTokenizer, NexTokenizerWrapper
from verse_infra import load_tokenizer, render_chat, render_prompt

# compat
from verse_infra import load_hf_state_dict

# inference
from verse_infra import ModelLoader, StateCache, Sampler, StreamingGenerator

# trainer
from verse_infra import train, ParallelTrainerSafe, LossOptimizer, RLTrainer
from verse_infra import CachedDataset, TextDataset, SingleSampleDataset
```

完整 `__all__` 列表见 [`verse_infra/__init__.py`](verse_infra/__init__.py)。

### 延迟导入

便捷重导出使用 `__getattr__` 延迟加载——`import verse_infra` 不会强制加载所有子包（特别是较重的 `verse_trainer`），仅在首次访问某个公共 API 时才加载对应子模块。这避免了"导入 `verse_infra` 就拖入训练栈"的副作用：

```python
import verse_infra            # 快：只导入 __init__.py，不加载子模块
from verse_infra import train # 首次访问触发 verse_trainer 加载
```

## shim 兼容说明

Part4K1 在原 `packages/verse_tokenizer/` / `packages/verse_compat/` / `packages/verse_inference/` 顶层位置保留了 thin shim：

- 旧路径 `from verse_tokenizer import BPETokenizer` 仍可工作
- 首次导入时发出 `DeprecationWarning: verse_tokenizer 已迁移到 verse_infra.verse_tokenizer 子模块，请改用 from verse_infra.verse_tokenizer import ...`
- shim 只保留一个版本，下次 major release 将删除

### 迁移建议

```python
# 旧代码（仍可用，但会发警告）
from verse_tokenizer import BPETokenizer       # ⚠ DeprecationWarning

# 新代码（推荐）
from verse_infra.verse_tokenizer import BPETokenizer   # ✅

# 或便捷重导出
from verse_infra import BPETokenizer                   # ✅
```

### 全项目已迁移

Part4K1 已将 `tests/` / `examples/` / `packages/verse_nex/` / `packages/verse_torch/` / `spark/` / `docs/` 中所有 `from verse_tokenizer/verse_compat/verse_inference import` 统一更新为 `from verse_infra.verse_xxx import`，详见 [SubTask 7.7](../../.trae/specs/part4k1-infra-model-upgrade/tasks.md)。

## 子模块文档

| 子模块 | 文档 | 主要能力 |
|---|---|---|
| `verse_tokenizer` | [README](verse_infra/verse_tokenizer/README.md) | BPE（并行 merge）/ Unigram / WordPiece / Qwen tokenizer / NexTokenizerWrapper |
| `verse_compat` | [README](verse_infra/verse_compat/README.md) | HuggingFace / PyTorch 兼容层（读 `state_dict` / `.bin` / `.safetensors`） |
| `verse_inference` | [README](verse_infra/verse_inference/README.md) | 模型加载 / 状态缓存 / 流式生成 / OpenAI 兼容 HTTP server |
| `verse_trainer` | （见下） | 预训练 / 微调 / 后训练 / 评估 CLI + CachedDataset + LossOptimizer |

## VerseTrainer 训练包

`verse_trainer` 是 Part4K1 新增的子模块，承载从 `data/demo/` 迁移并重构的训练栈。

### CLI 入口（5 个 console_scripts）

| 命令 | 用途 |
|---|---|
| `verse-train` | 预训练（`--device cpu/cuda/npu`、`--parallel-chunks N`、`--single-sample`、`--resume`、`--amp`、`--loss-optimizer`） |
| `verse-finetune` | 微调（`--method lora` / `--method full`） |
| `verse-posttrain` | 后训练（`--rl nexrl` / `--rl sft` / `--rl dpo`） |
| `verse-eval` | 评估 + 打分（`--score --references-file`） |
| `verse-tokenize` | tokenizer 训练 / 加载 / 转换（`--from-hf Qwen/Qwen3.5-35B-A3B`） |

### Python API

```python
from verse_infra.verse_trainer import (
    train,                    # 高层训练入口
    ParallelTrainerSafe,      # 并行训练 + OOM 兜底
    ChunkOOMError,            # chunk OOM 异常
    CachedDataset,            # 首次扫描缓存 .npz + 流式 lazy load
    TextDataset,              # JSONL 双格式（chat 数组 / prompt-completion）
    SingleSampleDataset,      # 单样本数据集（--single-sample 用）
    BatchLoader, collate_fn, load_jsonl,
    LossOptimizer,            # plateau 重走 + NaN/Inf 跳过 + LR × 0.3 + 重置 Adam 动量
    RLTrainer,                # NexRL 后训练
    evaluate, visualize,
)
```

### 训练流程示意

```bash
# 1. 预训练
verse-train --config spark/config/cometspark_v05.yml --device cpu

# 2. 并行训练（chunks > 1）
verse-train --config spark/config/cometspark_v05.yml --parallel-chunks 4

# 3. 断点续训
verse-train --config spark/config/cometspark_v05.yml --resume

# 4. 后训练（NexRL / SFT / DPO）
verse-posttrain --config spark/config/cometspark_v05.yml --rl nexrl
```

## 设计说明

- **不自研 kernel**：`verse_trainer` 调用 `verse_torch` / `verse_nex` 的训练栈，不重复实现 autograd / 注意力 / SSM
- **CPU 优先 + GPU 可选**：默认 CPU 训练，`--device cuda` / `--device npu` 通过 `verse_torch.device.DeviceBackend` 委托 PyTorch（详见 [ADR-005](../../docs/architecture/adr-005-gpu-npu-backend.md)）
- **`_safe_chunk_run`**：`ParallelTrainerSafe` 用子进程包裹每个 chunk，捕获异常 + 信号处理 + OOM 兜底，避免"莫名终止退出"
- **`CachedDataset`**：首次扫描训练数据集合并缓存为 `.npz`，后续启动直接 lazy load，解决大文件加载耗时
- **`LossOptimizer`**：监控 loss plateau（连续 N 步不降），自动回退到 `best_state_dict` + LR × 0.3 + 重置 Adam 动量 + 微扰；NaN/Inf 检测 + 跳过该 batch

## 相关文档

- [ADR-006 VerseInfra 总包聚合](../../docs/architecture/adr-006-verse-infra-aggregation.md)
- [Verse 训练指南](../../docs/training_guide.md)
- [Verse 性能调优](../../docs/performance_tuning.md)
- [CometSpark V0.5-1B](../../spark/README.md)
- [主 README](../../README.md)
