# VerseNext

> **VerseNext** —— 纯 Python / 纯 CPU 的深度学习与大语言模型框架（VerseTorch + VerseNex + VerseAWM），不强制依赖 PyTorch / Transformers，可在消费级 CPU、嵌入式设备与树莓派上开箱即用；可选 GPU/NPU 后端用于大规模训练。

VerseNext 的目标是用 **线性复杂度架构（SSM / Mamba / RWKV / Linear Attention）** 与 **VerseNex 原生架构（TriSparse + MoD）** 替代或混合 Transformer，根治自注意力 O(N²) 与 KV Cache 线性膨胀问题；并为下一代 **世界模型（JEPA、RSSM、H-JEPA）** 与端侧高能力 LLM 提供原生支撑，同时尽可能兼容 HuggingFace / PyTorch 生态以降低迁移成本。

## 包定位

| 包 | PyPI 名 | 定位 | 关键能力 |
|---|---|---|---|
| **VerseTorch** | `verse-torch` | 纯 Python + NumPy 的张量与自动微分引擎（PyTorch 替代） | `Tensor` 类、动态计算图、反向模式 autograd、`nn.Module`、优化器栈（SGD/Adam/AdamW/NAdamW/RMSProp）、INT4/INT8/1.58-bit 量化、可读 PyTorch `state_dict`、`DeviceBackend` 抽象（CPU/GPU/NPU） |
| **VerseNex** | `verse-nex` | Transformer 替代架构库 + VerseNex 原生架构 | Mamba-2 / RWKV-7 / RetNet Linear Attention / Sparse Attention / Hybrid / **TriSparseAttention** / **MoDLayer**（5 DensePart × 8 Expert × top-3）/ **VerseNexLM** / **NexRL** 强化学习 |
| **VerseAWM** | `verse-awm` | 世界模型专用包（Autonomous World Model） | I-JEPA / V-JEPA 潜在空间预测、RSSM（Dreamer 风格）、H-JEPA 层次化规划、EMA target encoder、energy-based loss |
| **VerseInfra** | `verse-infra` | 总包：聚合 verse_tokenizer / verse_compat / verse_inference / verse_trainer 四个子模块 | 单包安装 + 子模块结构 + 便捷重导出 + shim 兼容旧导入路径 |
| **CometSpark** | `spark/`（仓库内置） | CometSpark V0.5-1B 端到端 LM 训练仓库 | 基于 `VerseNexBlock` 的 1B 参数模型 + Qwen3.5-35B-A3B tokenizer + VerseTrainer CLI 一键训练 |

### VerseInfra（总包）

`verse_infra` 聚合了原本独立的四个辅助包为子模块，便于一键安装与版本对齐：

| 子模块 | 旧路径（已废弃，仍可工作） | 新路径（推荐） |
|---|---|---|
| `verse_tokenizer` | `from verse_tokenizer import BPETokenizer` | `from verse_infra.verse_tokenizer import BPETokenizer` |
| `verse_compat` | `from verse_compat import load_hf_state_dict` | `from verse_infra.verse_compat import load_hf_state_dict` |
| `verse_inference` | `from verse_inference import ModelLoader` | `from verse_infra.verse_inference import ModelLoader` |
| `verse_trainer` | （新模块） | `from verse_infra.verse_trainer import train` |

旧路径仍可工作（通过 thin shim 转发 + `DeprecationWarning`），详见 [VerseInfra README](packages/verse_infra/README.md)。

### VerseTrainer（训练 CLI）

`verse_trainer` 提供 8 个 CLI 入口（注册为 console_scripts，`verse-continue` 通过统一分发入口 `python -m verse_infra.verse_trainer.cli verse-continue` 调用）：

| 命令 | 用途 |
|---|---|
| `verse-train` | 预训练（支持 `--device cpu/cuda/npu`、`--parallel-chunks N`、`--single-sample`、`--resume`、`--amp`、`--loss-optimizer`、`--partition-training`、`--quiet/--verbose`、`--parallel-strategy round_robin`） |
| `verse-continue` | **持续训练（Part4K2 新增）**：从 checkpoint 加载继续追加训练（`--checkpoint` / `--additional-steps`） |
| `verse-finetune` | 微调（`--method lora` / `--method full`） |
| `verse-posttrain` | 后训练（`--rl nexrl` / `--rl sft` / `--rl dpo`） |
| `verse-eval` | 评估 + 打分（`--score --references-file`，`--max-tokens` 默认 None=EOS 自然停止） |
| `verse-tokenize` | tokenizer 训练 / 加载 / 转换（`--from-hf Qwen/Qwen3.5-35B-A3B`） |
| `verse-download` | **数据集下载器（Part4K2 新增）**：任意 URL + HuggingFace datasets，断点续传 + 多线程 + 自动转 .npz（`--url` / `--hf` / `--to-npz`） |
| `verse-convert` | **模型格式转换（Part4K2 新增）**：`.pt ↔ .vn` 互转（`--input` / `--output` / `--chat-template` / `--tokenizer`） |

### NexRL（RL 算法）

`verse_nex.nexrl` 子包实现 VerseNex 强化学习算法，五要素抽象：

- `NexAgent`：策略网络（VerseNexLM）+ 参考网络（KL 约束）
- `NexEnv`：任务环境（ChatEnv / MathEnv / CodeEnv）
- `NexState`：RL 状态（prompt + tokens + KV cache）
- `NexAction`：动作采样（ε-greedy / softmax / nucleus）+ 探索衰减 schedule + 重复惩罚
- `NexReward`：多维奖励（correctness + fluency + safety + length_penalty）+ 归一化 + reward shaping

训练组件：`ParallelRolloutCollector`（并行 rollout）+ `NexTrainer`（PPO clipped surrogate + GAE + KL 自适应 + value function）。详见 [ADR-007 NexRL 设计](docs/architecture/adr-007-nexrl-design.md)。

### CometSpark V0.5-1B

`spark/` 目录承载基于 VerseNex 原生架构的 1B 参数 LM 训练仓库：

- **CometSparkV05LM**：组合 `verse_nex.CometSparkNexLM`（内部 `VerseNexBlock` = TriSparse + MoD），不重造底层 block
- **1B 参数预算**：`n_embd=1024, n_layer=20, 5 MoD + 15 trisparse, 4 DensePart × 4 Expert × top-2` + `tie_weights=True` + `embedding_scale=True` ≈ 1.12B 参数
- **Qwen tokenizer**：`vocab=248320`（Qwen3.5-35B-A3B）
- **解决胡乱输出**：embedding scale + tie_weights + temperature scaling + 合理初始化
- **训练/推理 CLI**：`verse-train --config spark/config/cometspark_v05.yml`，详见 [spark/README.md](spark/README.md)

### GPU/NPU 支持（DeviceBackend）

`verse_torch.device.DeviceBackend` 提供设备后端抽象：

- `NumpyBackend`（默认 CPU）：所有算子用 NumPy 实现，与自研 autograd 完全等价
- `TorchBackend`（PyTorch 委托）：CUDA kernel 走 PyTorch 原生实现，NPU 通过 `torch_npu` 扩展支持；**不自研 kernel**
- `autocast`：fp16 混合精度上下文管理器（CPU 时 no-op）
- `Tensor.device` / `.to(device)` / `.cuda()` / `.npu()` / `.cpu()`：API 与 PyTorch 一致
- `Module.to(device)`：迁移所有参数到目标设备
- 无 PyTorch 环境下 `Tensor.cuda()` 抛 `RuntimeError`，CPU 路径完全不变

详见 [ADR-005 GPU/NPU 后端抽象](docs/architecture/adr-005-gpu-npu-backend.md)。

### .vn 文件格式（Part4K2 新增）

`verse_torch.vn_format` 定义基于 **safetensors 性能优化**的模型容器格式 `.vn`，取代传统 pickle `.pt`：

- **ZIP 容器**：内含 `model.safetensors`（或降级 `model.npz`）+ `config.yml` + `chat_template.jinja`（可选）+ `tokenizer.json`（可选）+ `meta.json`
- **mmap 零拷贝**：safetensors 可用时通过 `safe_open` 零拷贝读取权重；npz 路径落盘懒加载
- **无损互转**：`pt_to_vn` / `vn_to_pt` / `convert_format`，权重数值完全一致
- **自描述**：`meta.json` 记录 `vn_format_version` / `arch` / `weight_format` / `compression_info` / `created_at`
- **安全性**：npz 路径强制 `allow_pickle=False`，safetensors 本身 pickle-free
- **CLI**：`verse-convert --input model.pt --output model.vn`（详见 [ADR-009](docs/architecture/adr-009-vn-format.md)）

```python
from verse_torch import VNFileReader, VNFileWriter

# 写入
with VNFileWriter("model.vn", arch="versenex", config=cfg_dict) as w:
    w.write_weights(model.state_dict())
    w.write_chat_template(template_str)   # 可选
    w.write_tokenizer("tokenizer.json")   # 可选

# 读取（mmap 零拷贝）
with VNFileReader("model.vn") as r:
    meta = r.read_meta()
    cfg = r.read_config()
    sd = r.read_weights(mmap=True)
```

### jinja2 聊天模板（Part4K2 新增）

`verse_tokenizer.chat_template` 升级为 **Qwen3 ChatML 风格**，jinja2 为可选依赖（不可用时降级为 f-string，输出完全等价）：

- **ChatML 模板**：`<|im_start|>{role}\n{content}<|im_end|>\n` 循环拼接 + `add_generation_prompt`
- **工具调用**：Qwen3 官方 `<tool_call>{"name":...,"arguments":...}</tool_call>` 格式，含 `tools` 声明 system 段 + assistant tool_calls + tool 角色返回
- **模板常量**：`CHATML_TEMPLATE` / `CHATML_TEMPLATE_WITH_TOOLS` / `CHATML_TEMPLATE_WITH_TOOL_CALLS`
- **解析**：`extract_tool_calls_qwen3` 从生成文本提取工具调用（与渲染互逆）
- **tokenizer.json 内嵌**：`chat_template.jinja` 可内嵌到 tokenizer.json
- **apply_chat_template 升级**：支持 `tools` 参数 + `add_generation_prompt`（详见 [ADR-010](docs/architecture/adr-010-jinja2-chat-template.md)）

```python
from verse_infra.verse_tokenizer import (
    render_chat_qwen, render_chat_qwen_with_tools, extract_tool_calls_qwen3,
)

# 基础 ChatML
text = render_chat_qwen([{"role": "user", "content": "你好"}], add_generation_prompt=True)

# 工具调用
out = render_chat_qwen_with_tools(
    messages=[{"role": "user", "content": "北京天气"},
              {"role": "assistant", "content": "",
               "tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]}],
    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}],
)
calls = extract_tool_calls_qwen3(generated_text)  # 解析模型生成的工具调用
```

### 智能分区训练（LayerWiseTrainer，Part4K2 新增）

`verse_torch.layerwise_trainer.LayerWiseTrainer` 把模型按 layer 分组训练，训完一组卸载到硬盘 `.vn` 分片，**保持统一实体**（对外表现为完整模型训练）：

- **按 layer 拆分**：transformer blocks 按 `partition_size` 分组，embedding/lm_head 始终在内存
- **.vn 分片卸载**：训完一组用 `VNFileWriter` 写到 `partition_{idx}.vn`，内存超阈值时自动卸载已训练的非当前组
- **统一实体**：训练过程中模型对象不变，参数在内存/硬盘间备份；对外接口与普通 `Trainer` 一致
- **内存监控**：调用 `get_memory_info`，超过 `memory_threshold_mb` 自动触发卸载
- **合并 + fine-tune**：全部组训练完成后合并所有分片为完整模型，可选整体 fine-tune
- **CLI**：`verse-train --partition-training --partition-size N --offload-dir DIR`（详见 [ADR-011](docs/architecture/adr-011-layerwise-training.md)）

```python
from verse_torch import LayerWiseTrainer

trainer = LayerWiseTrainer(
    model, config={"lr": 1e-3, "finetune_steps": 20},
    partition_size=2, memory_threshold_mb=512,
)
train_losses, val_losses = trainer.fit(train_loader, val_loader, max_steps=1000)
```

### 压缩技术 V1.3（以小博大，Part4K2 新增）

`verse_torch.compress` 推出 V1.3 压缩流水线，组合 **剪枝 + 量化 + 知识蒸馏 + LoRA** 实现大模型→小模型能力转移：

- **KnowledgeDistiller V1.3**：三重损失（软标签 KL + 硬标签 CE + 中间层特征 MSE）+ 自适应温度退火
- **compress_pipeline V1.3**：流程重排为 `prune → quantize → distill → lora`，吞吐率优化（fused matmul + batch 量化）
- **VerseNex 集成**：`CometSparkNexLM.compress_v13()` / `distill_from(teacher, train_data)`
- **压缩报告**：`compression_report(model, compressed)` 返回参数量/压缩比/吞吐率提升估算
- **teacher 便捷入口**：`config` 顶层可直接放 `teacher_model` / `train_loader`（详见 [ADR-012](docs/architecture/adr-012-compression-v13.md)）

```python
from verse_torch.compress import compress_pipeline, compression_report

config = {
    "prune":    {"sparsity": 0.3},
    "quantize": {"bits": 4},
    "distill":  {"teacher": teacher_model, "epochs": 3, "lr": 1e-3, "temperature": 4.0},
    "lora":     {"rank": 8, "alpha": 16},
}
compressed, stats = compress_pipeline(model, config, version="1.3", return_stats=True)
print(compression_report(model, compressed))
```

### 数据集下载器（Part4K2 新增）

`data/downloader.py` 的 `DatasetDownloader`（经 `verse_infra` 顶层导出）支持任意 URL + HuggingFace datasets 下载，自动转 `.npz` 缓存：

- **任意 URL**：`download_url` 支持 HTTP/HTTPS，大文件（≥10MB）自动多线程分块下载
- **断点续传**：基于已下载字节数 + `Range` header，从中断点继续
- **HuggingFace**：`download_hf` 调用 `datasets` 库（可选依赖，缺失时提示安装）
- **自动转 .npz**：`.json` / `.jsonl` / `.csv` / `.txt` / `.parquet` → `.npz`（含 `ids` / `mask` / `seq_len`，与 `CachedDataset` 对齐）
- **一站式**：`download_and_cache(url_or_repo)` 自动判断 URL/HF + 转 npz
- **CLI**：`verse-download --url ... --to-npz` / `verse-download --hf wikitext --split train`

```python
from verse_infra import DatasetDownloader

dl = DatasetDownloader(cache_dir="data/datasets", num_workers=4)
npz_path = dl.download_and_cache("https://example.com/data.jsonl",
                                  output_path="data/datasets/cached.npz")
# 或从 HuggingFace
dir_ = dl.download_hf("wikitext", subset="wikitext-2-raw-v1", split="train")
```

### 持续训练 verse-continue（Part4K2 新增）

`verse-continue` 在训练完成后从 checkpoint 继续追加训练（与 `--resume` 的"中断恢复"语义不同）：

- 自动继承之前的 `best_val_loss`（不从头比较）
- 支持 `--device cuda --amp` GPU 加速
- 通过统一分发入口调用：`python -m verse_infra.verse_trainer.cli verse-continue ...`

```bash
python -m verse_infra.verse_trainer.cli verse-continue \
    --checkpoint checkpoints/best.pt --additional-steps 1000 \
    --config spark/config/cometspark_v05.yml --device cuda --amp
```

## 安装

需要 Python ≥ 3.10、NumPy ≥ 1.26。本仓库采用 uv/pip workspace 多包布局，每个 package 都是独立的可编辑安装单元。

```bash
# 1) 克隆仓库
git clone <repo-url> verse && cd verse

# 2a) 方式一：pip 可编辑安装（按需选择包）
pip install -e packages/verse_torch \
            -e packages/verse_nex \
            -e packages/verse_awm \
            -e packages/verse_infra

# 2b) 方式二：uv workspace 一次性安装全部成员
uv sync

# 3) 可选运行时依赖（按需安装）
pip install "verse-nex[speed]"  # 安装 numba 加速 selective scan（推荐）
pip install "safetensors>=0.4"   # 加载 .safetensors 权重 + .vn 格式 mmap 零拷贝
pip install "jinja2>=3.0"        # ChatML 聊天模板渲染（不可用时降级为 f-string）
pip install "fastapi>=0.110"     # OpenAI 兼容 HTTP server
pip install "torch>=2.2"         # GPU/NPU 后端（CUDA kernel + autocast）
pip install "torch_npu>=2.2"     # 华为 NPU 后端（昇腾设备）
pip install "datasets>=2.18"     # verse-download --hf 下载 HuggingFace datasets
```

> **numba 加速说明**：`verse-nex[speed]` 会安装 `numba>=0.60`，对 Mamba-2 / Hybrid 的 selective scan 递推循环做 JIT 编译，recurrent 模式生成吞吐量提升约 1.8× ~ 3.2×。numba 是可选依赖——不安装也能运行，只是 `@njit` 装饰器退化为 no-op。详见 [性能调优指南](docs/performance_tuning.md)。
>
> **GPU/NPU 说明**：`torch` 是可选依赖——CPU 路径无需安装；仅在调用 `Tensor.cuda()` / `Tensor.npu()` / `autocast` / `--device cuda` 时才会触发 PyTorch 委托后端。无 torch 环境下 CPU 路径完全不变（向后兼容）。

## 快速开始

### 最小 autograd 示例

```python
from verse_torch import Tensor

x = Tensor([1.0, 2.0], requires_grad=True)
y = (x * x).sum()        # 1 + 4 = 5
y.backward()
print(y)                 # 5.0
print(x.grad)            # [2. 4.]  与 PyTorch 一致
```

实测：上述代码与 PyTorch 数值一致到 1e-6（已通过 786 项单元测试 + 有限差分梯度检查）。

### 推荐导入路径

```python
# 1. VerseTorch：张量 / autograd / nn / optim
from verse_torch import Tensor, nn, optim, losses

# 2. VerseNex：原生架构
from verse_nex import VerseNexLM, VerseNexAttention, MoDLayer

# 3. VerseInfra：tokenizer / inference / compat / trainer（推荐）
from verse_infra.verse_tokenizer import BPETokenizer
from verse_infra.verse_inference import ModelLoader, StreamingGenerator
from verse_infra.verse_trainer import train

# 4. NexRL：强化学习
from verse_nex.nexrl import NexAgent, NexTrainer, NexReward

# 5. CometSpark：1B 模型工厂
from spark.model.model import CometSparkV05, CometSparkV05Small

# 6. Part4K2 新增：.vn 格式 + 智能分区训练
from verse_torch import VNFileReader, VNFileWriter, LayerWiseTrainer

# 7. Part4K2 新增：数据集下载器（经 verse_infra 顶层导出）
from verse_infra import DatasetDownloader
```

### 一键训练 CometSpark V0.5-1B

```bash
# CPU 预训练（小配置，快速验证）
verse-train --config spark/config/cometspark_v05_small.yml --device cpu --max-steps 10

# 1B 模型预训练（CPU / GPU / NPU）
verse-train --config spark/config/cometspark_v05.yml --device cpu
verse-train --config spark/config/cometspark_v05.yml --device cuda --amp
verse-train --config spark/config/cometspark_v05.yml --device npu

# 并行训练（chunks > 1）
verse-train --config spark/config/cometspark_v05.yml --parallel-chunks 4

# 断点续训
verse-train --config spark/config/cometspark_v05.yml --resume

# 后训练（NexRL / SFT / DPO）
verse-posttrain --config spark/config/cometspark_v05.yml --rl nexrl
verse-posttrain --config spark/config/cometspark_v05.yml --rl sft
verse-posttrain --config spark/config/cometspark_v05.yml --rl dpo
```

### Part4K2 新增命令快速开始

```bash
# 模型格式互转：.pt → .vn（safetensors 性能优化版，可附加 chat_template / tokenizer）
verse-convert --input checkpoints/best.pt --output model.vn \
    --chat-template chat_template.jinja --tokenizer tokenizer.json --arch versenex
verse-convert --input model.vn --output model.pt   # .vn → .pt 无损回转

# 数据集下载：任意 URL（多线程 + 断点续传 + 自动转 .npz）
verse-download --url https://example.com/data.jsonl --to-npz -o data/cached.npz
verse-download --hf wikitext --split train          # HuggingFace datasets

# 智能分区训练（按 layer 分组训练 + .vn 分片卸载，低内存跑大模型）
verse-train --config spark/config/cometspark_v05.yml \
    --partition-training --partition-size 2 --max-steps 1000

# 持续训练（从 checkpoint 继续追加训练）
python -m verse_infra.verse_trainer.cli verse-continue \
    --checkpoint checkpoints/best.pt --additional-steps 1000 \
    --config spark/config/cometspark_v05.yml --device cuda --amp
```

更多示例见 [`examples/`](examples/)：

- [`mnist_mlp.py`](examples/mnist_mlp.py) —— MNIST MLP，5 epoch 准确率 97.66%
- [`minimal_lm.py`](examples/minimal_lm.py) —— 字符级 LM（Mamba-2 backbone），parallel vs recurrent 一致
- [`jepa_demo.py`](examples/jepa_demo.py) —— I-JEPA 自监督预训练
- [`cpu_inference_demo.py`](examples/cpu_inference_demo.py) —— 纯 CPU 流式生成（715 tokens/s，峰值 RSS 44.5MB）

## 关键技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 张量后端 | **NumPy** + 可选 Numba/Cython + DeviceBackend（CPU/GPU/NPU 抽象） | CPU 优先、零重型依赖、可选 PyTorch 委托 GPU/NPU |
| Autograd | 反向模式 VJP、动态计算图；GPU 下委托 PyTorch autograd | 与 PyTorch API 一致、易于审计与调试 |
| 主推架构 | **Mamba-2 + RWKV-7 + Hybrid + VerseNex**（SSM + Sparse Attention + TriSparse + MoD） | 已公开、工业验证（MiniMax-01、混元 T1、Nemotron-H、RWKV-X） |
| 原生注意力 | **TriSparseAttention**（SWA + Global sink + ALiBi 三路并行稀疏，sigmoid gate 融合） | 线性/亚二次复杂度 + 长程依赖兼顾，CPU 友好 |
| 原生 FFN | **MoD（Mixture of Dense Parts）**：5 DensePart × N Experts × top-k + aux loss | 灵感来源于大脑分区，结构化稀疏 + 双层门控 |
| 量化默认 | **INT4 (W4A16) + 1.58-bit ternary** | BitNet.cpp 在 CPU 上已验证 6.17× 提速；端侧友好 |
| 兼容策略 | 初期可读 PyTorch `state_dict`，运行时无 PyTorch 依赖；GPU/NPU 走可选委托后端 | 生态友好但运行时零依赖；GPU 路径 API 一致 |
| 世界模型主线 | **JEPA（非生成式）+ RSSM（生成式）** | 兼顾 LeCun 路线与 Dreamer 路线 |
| GPU/NPU 后端 | `DeviceBackend` 抽象（PyTorch 委托 + NumpyBackend 回退） | CUDA kernel 走 PyTorch 原生实现；NPU 走 `torch_npu`；详见 [ADR-005](docs/architecture/adr-005-gpu-npu-backend.md) |
| 后训练路线 | **NexRL**（PPO + GAE + KL 自适应）+ SFT + DPO | 五要素抽象 + 并行 rollout；详见 [ADR-007](docs/architecture/adr-007-nexrl-design.md) |
| 加速路线 | **超稀疏并行注意力 + Speculative Decoding** | 多 chunk 并行 + verify-then-commit；详见 [ADR-008](docs/architecture/adr-008-parallel-sparse-attention.md) |
| 模型交付格式 | **.vn**（ZIP 容器 + safetensors + 自描述 meta） | mmap 零拷贝 + pickle-free 安全 + config/template/tokenizer 内嵌；详见 [ADR-009](docs/architecture/adr-009-vn-format.md) |
| 聊天模板 | **ChatML + jinja2 可选依赖**（缺失降级 f-string） | 与 Qwen3 工具调用格式对齐 + tokenizer.json 内嵌；详见 [ADR-010](docs/architecture/adr-010-jinja2-chat-template.md) |
| 大模型低内存训练 | **LayerWiseTrainer 智能分区训练**（按 layer 分组 + .vn 分片卸载） | 内存超阈值自动卸载 + 统一实体 + 合并 fine-tune；详见 [ADR-011](docs/architecture/adr-011-layerwise-training.md) |
| 大→小模型能力转移 | **压缩 V1.3**（prune → quantize → distill → lora） | 三重损失知识蒸馏 + 温度退火 + 吞吐率优化；详见 [ADR-012](docs/architecture/adr-012-compression-v13.md) |

详细架构决策记录见 [`docs/architecture/`](docs/architecture/)。

## 详细文档

### 训练指南

[Verse 训练指南](docs/training_guide.md) —— 从零训练 LM 的完整流程：数据准备 → tokenizer → 模型 → 训练 → 评估 → 压缩 → 推理。

### 性能调优

[Verse 性能调优指南](docs/performance_tuning.md) —— numba JIT 加速、BLAS 配置、batch_size 选择、CPU 线程数、量化加速、并行计算六个维度的 CPU 调优手册。

### 各包文档

| 包 | 文档 | 定位 |
|---|---|---|
| VerseTorch | [README](packages/verse_torch/README.md) | 张量 / autograd / nn / optim / losses / training / quantize / parallel / compress / DeviceBackend（GPU/NPU）/ **.vn 格式（Part4K2）** / **LayerWiseTrainer（Part4K2）** / **compress V1.3（Part4K2）** |
| VerseNex | [README](packages/verse_nex/README.md) | Mamba-2 / RWKV-7 / RetNet / Sparse Attention / Hybrid / TriSparseAttention / MoDLayer / VerseNexLM / NexRL / **compress_v13 / distill_from（Part4K2）** |
| VerseAWM | [README](packages/verse_awm/README.md) | I-JEPA / V-JEPA / H-JEPA / RSSM 世界模型 |
| VerseInfra | [README](packages/verse_infra/README.md) | 总包结构 / 导入路径迁移指南 / shim 兼容（Part4K1）/ **DatasetDownloader（Part4K2）** |
| ├ verse_tokenizer | [README](packages/verse_infra/verse_infra/verse_tokenizer/README.md) | BPE / Unigram / WordPiece / Qwen tokenizer / NexTokenizerWrapper / **ChatML jinja2 模板（Part4K2）** |
| ├ verse_inference | [README](packages/verse_infra/verse_infra/verse_inference/README.md) | 模型加载 / 状态缓存 / 流式生成 / HTTP server |
| └ verse_compat | [README](packages/verse_infra/verse_infra/verse_compat/README.md) | HuggingFace / PyTorch 兼容层 |
| CometSpark | [README](spark/README.md) | CometSpark V0.5-1B 模型说明 + 配置 + 训练 CLI（Part4K1）/ **持续训练 verse-continue（Part4K2）** |

### 设计文档

- [压缩管线设计（trillion → billion 路线图）](verse_data/designs/compression_pipeline_design.md)
- [Autograd 设计](verse_data/designs/autograd_design.md)
- [SSM scan 设计](verse_data/designs/ssm_scan_design.md)
- [JEPA EMA 设计](verse_data/designs/jepa_ema_design.md)
- [PyTorch 迁移笔记](verse_data/migration_notes/pytorch_to_versetorch.md)

### 基准测试

- [CPU 并行基准](docs/benchmarks/parallel_benchmark.md)
- [模型压缩 PoC](docs/benchmarks/compression_poc.md)
- [量化基准](docs/benchmarks/quantize_benchmark.md)
- [v0.1 整体基准](docs/benchmarks/benchmark-v0.1.md)

### 架构决策记录（ADR）

- [ADR-001 CPU 优先](docs/architecture/adr-001-cpu-first.md)
- [ADR-002 线性复杂度架构](docs/architecture/adr-002-linear-complexity.md)
- [ADR-003 世界模型路线](docs/architecture/adr-003-world-model-route.md)
- [ADR-004 CPU 并行](docs/architecture/adr-004-cpu-parallel.md)
- [ADR-005 GPU/NPU 后端抽象](docs/architecture/adr-005-gpu-npu-backend.md)（**Part4K1 新增**）
- [ADR-006 VerseInfra 总包聚合](docs/architecture/adr-006-verse-infra-aggregation.md)（**Part4K1 新增**）
- [ADR-007 NexRL 设计](docs/architecture/adr-007-nexrl-design.md)（**Part4K1 新增**）
- [ADR-008 超稀疏并行注意力](docs/architecture/adr-008-parallel-sparse-attention.md)（**Part4K1 新增**）
- [ADR-009 .vn 文件格式](docs/architecture/adr-009-vn-format.md)（**Part4K2 新增**）
- [ADR-010 jinja2 聊天模板](docs/architecture/adr-010-jinja2-chat-template.md)（**Part4K2 新增**）
- [ADR-011 智能分区训练](docs/architecture/adr-011-layerwise-training.md)（**Part4K2 新增**）
- [ADR-012 压缩技术 V1.3](docs/architecture/adr-012-compression-v13.md)（**Part4K2 新增**）

### Part4K1 重大升级摘要

Part4K1 在 Part4 基础上完成 8 大升级，正式推出 **VerseInfra 总包**、**VerseTrainer CLI**、**NexRL 强化学习**、**CometSpark V0.5-1B** 与 **GPU/NPU 后端**：

- **VerseTorch GPU/NPU 设备抽象**：新增 `device.py`（`DeviceBackend` 抽象 + `NumpyBackend` 默认 + `TorchBackend` PyTorch 委托）+ `backend_torch.py`（CUDA kernel 走 PyTorch 原生 + NPU via `torch_npu` + `autocast`）；`Tensor.device` / `.to(device)` / `.cuda()` / `.npu()`；`Module.to(device)`；新组件 `RotaryEmbedding` / `KVCache` / `StaticCache` / `DynamicCache` / `GroupNorm` / `Conv1d` / `NAdamW` / `RMSProp` / `contrastive_loss` / `perplexity` / `DistributedTrainer`。
- **VerseNex 品牌落地**：`TransformerLM` → `VerseNexLM`、`GQASelfAttention` → `VerseNexAttention`（旧名作为 `DeprecationWarning` 别名）；`MoDLayer` 完善（5 DensePart × 8 Expert × top-3 + `load_balance_loss` + router z-loss）；`config.yml` `arch` 字段统一为 `versenex`（旧值映射 + 警告）；`HybridBlock` / `HybridLM` deprecated。
- **超稀疏并行注意力**：`tri_sparse_attn.py` 多 query chunk 并行；`speculative.py`（`SpeculativeDecoder` Medusa 风格 draft + verify-then-commit）；`kv_cache_parallel.py`（`ParallelKVCache`）。
- **NexRL 强化学习包**：`verse_nex/nexrl/` 五要素（`NexAgent` / `NexEnv` / `NexState` / `NexAction` / `NexReward`）+ `ParallelRolloutCollector` + `NexTrainer`（PPO clipped + GAE + KL 自适应 + value function）。
- **VerseTokenizer 升级**：BPE 并行 merge + `WordPieceTokenizer` + `BatchEncoding`（`add_bos`/`add_eos` + truncation/padding）+ `BPETokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")`（vocab 248320）+ `NexTokenizerWrapper`（reward-weighted token preference）。
- **VerseTrainer 独立包**：CLI（`verse-train` / `verse-finetune` / `verse-posttrain` / `verse-eval` / `verse-tokenize`）+ `CachedDataset` + `LossOptimizer`（plateau 重走 + NaN/Inf 跳过）+ `_safe_chunk_run` + 断点续训 + `RLTrainer`。
- **VerseInfra 总包聚合**：`verse_tokenizer` / `verse_compat` / `verse_inference` / `verse_trainer` 四包聚合为子模块；旧路径通过 thin shim 转发 + `DeprecationWarning`；全项目导入路径更新为 `from verse_infra.verse_xxx import`。
- **CometSpark V0.5-1B**：`spark/` 目录承载基于 `VerseNexBlock` 的 1B 参数 LM（n_embd=1024, n_layer=20, 5 MoD + 15 trisparse）；`CometSparkV05Config` / `CometSparkV05LM` / 工厂 `CometSparkV05()` / `CometSparkV05Small()`；`cometspark_v05.yml` 默认配置 + `cometspark_v05_small.yml` 调试配置；解决胡乱输出（embedding scale + tie_weights + temperature scaling）。
- **删除 `data/demo/`**：训练能力迁入 VerseTrainer，模型能力迁入 spark/，data/demo/ 整个目录删除。
- 全量测试 786 passed。

### Part4K2 重大升级摘要

Part4K2 在 Part4K1 基础上完成 8 大升级，重点解决 **模型交付格式标准化 / 聊天模板工程化 / 大模型低内存训练 / 大模型→小模型能力转移 / 数据集工程化 / 持续训练闭环** 六大方向：

- **.vn 文件格式（Task 1）**：`verse_torch.vn_format` 定义基于 safetensors 的 ZIP 容器格式 `.vn`（model.safetensors + config.yml + chat_template.jinja + tokenizer.json + meta.json），mmap 零拷贝读取 + pickle-free 安全性；`VNFileReader` / `VNFileWriter` / `pt_to_vn` / `vn_to_pt` / `convert_format`；CLI `verse-convert` 实现 .pt ↔ .vn 无损互转（safetensors 不可用时自动降级 npz）。
- **jinja2 聊天模板（Task 2）**：`verse_tokenizer.chat_template` 升级为 ChatML Qwen3 风格，jinja2 为可选依赖（缺失时降级 f-string，输出完全等价）；`CHATML_TEMPLATE` / `CHATML_TEMPLATE_WITH_TOOLS` / `CHATML_TEMPLATE_WITH_TOOL_CALLS` 三个模板常量；`render_chat_qwen` / `render_chat_qwen_with_tools` / `extract_tool_calls_qwen3`；支持 Qwen3 工具调用 `<tool_call>...</tool_call>` 格式 + tokenizer.json 内嵌 chat_template.jinja。
- **生成输出优化（Task 3）**：`evaluate.py` `max_new_tokens` 默认 None（EOS 自然停止），避免截断有效内容；`stop_strings` 支持字符串停止条件；生成日志打印优化。
- **智能分区训练（Task 4）**：`verse_torch.layerwise_trainer.LayerWiseTrainer` 按 layer 分组训练，训完一组卸载到 `partition_{idx}.vn` 分片，内存超阈值（默认 512MB）自动卸载已训练的非当前组；对外保持统一实体（模型对象不变）；合并后可选整体 fine-tune；CLI `verse-train --partition-training --partition-size N`。
- **资源利用优化（Task 5）**：`verse_torch.device` 完善 `empty_cache` / `get_memory_info` / `memory_usage` / `set_num_threads` / `get_num_threads` / `auto_tune_threads`；`backend_torch.py` autocast 支持 NPU；`VerseNexBlock` 激活检查点开关 `use_checkpoint`（GPU 大模型训练节省显存，CPU 自动降级）。
- **压缩技术 V1.3（Task 6）**：`KnowledgeDistiller` V1.3 三重损失（软标签 KL + 硬标签 CE + 中间层特征 MSE）+ 自适应温度退火；`compress_pipeline` V1.3 流程重排为 `prune → quantize → distill → lora` + 吞吐率优化（fused matmul + batch 量化）；`CometSparkNexLM.compress_v13()` / `distill_from(teacher, train_data)`；`compression_report` 返回参数量/压缩比/吞吐率提升估算。
- **VerseTrainer 优化（Task 7）**：CLI 新增 `verse-convert` / `verse-download` / `verse-continue`（统一分发入口 `python -m verse_infra.verse_trainer.cli verse-continue`）三个子命令；`continue_train()` 通过 `train(continue_from=checkpoint)` 实现持续训练；`--partition-training` 智能分区训练开关。
- **数据集下载器（Task 8）**：`data/downloader.py` 的 `DatasetDownloader`（经 `verse_infra` 顶层 `__getattr__` 懒加载导出）支持任意 URL + HuggingFace datasets 下载，断点续传 + 多线程（≥10MB 自动分块）+ 自动转 .npz（json/jsonl/csv/txt/parquet → npz，含 ids/mask/seq_len）；CLI `verse-download --url/--hf --to-npz`。

### 第二次进化 / Part3K2 / Part4 摘要

详见 [审计报告](audit_report.md) 与 [Part4 升级报告](docs/part4_upgrade_report.md)。

## 仓库结构

```
/workspace/
├── packages/
│   ├── verse_torch/        # 张量与 autograd 引擎（含 device.py / backend_torch.py GPU/NPU 后端）
│   │   └── verse_torch/
│   │       ├── vn_format.py        # .vn 文件格式（VNFileReader/Writer，Part4K2）
│   │       ├── layerwise_trainer.py # 智能分区训练（LayerWiseTrainer，Part4K2）
│   │       └── compress.py         # 压缩管线 V1.3（prune→quantize→distill→lora，Part4K2）
│   ├── verse_nex/          # 线性复杂度架构库 + VerseNex 原生架构 + NexRL
│   │   └── verse_nex/
│   │       └── nexrl/      # NexRL 强化学习包（Part4K1）
│   ├── verse_awm/          # 世界模型包
│   └── verse_infra/        # 总包：聚合 verse_tokenizer/verse_compat/verse_inference/verse_trainer
│       └── verse_infra/
│           ├── verse_tokenizer/  # BPE/Unigram/WordPiece 分词器 + ChatML jinja2 模板（Part4K2）
│           ├── verse_compat/     # HF/PyTorch 兼容层
│           ├── verse_inference/  # 推理引擎
│           └── verse_trainer/    # 训练 CLI（verse-train/finetune/posttrain/eval/tokenize/convert/download/continue）
├── data/
│   └── downloader.py       # DatasetDownloader（任意 URL + HF + 断点续传 + 自动转 .npz，Part4K2）
├── spark/                  # CometSpark V0.5-1B 端到端 LM 训练仓库（Part4K1）
│   ├── config/             # cometspark_v05.yml / cometspark_v05_small.yml
│   ├── model/              # CometSparkV05LM + CometSparkV05Config
│   └── src/                # data_loader / trainer / evaluate / utils
├── datasets/               # raw / cleaned / tokenizer
├── docs/                   # papers / architecture / benchmarks
│   └── architecture/       # ADR-001 ~ ADR-012
├── verse_data/             # designs / experiments / migration_notes（内部材料）
├── tests/                  # 单元测试 + 数值梯度检查 + 端到端用例（786 passed）
├── examples/               # MNIST / 最小 LM / CPU 推理 demo / JEPA demo
├── pyproject.toml          # workspace 级配置（PEP 621 + uv workspace）
└── README.md
```

## 参考资料链接

更完整的清单（含微信公众号文章）见 [`docs/papers/`](docs/papers/)。

### 论文（arXiv / 会议）

- [The End of Transformers? Sub-Quadratic Architectures](https://arxiv.org/html/2510.05364v1) —— 线性/亚二次架构综述
- [RWKV-X: A Linear Complexity Hybrid Language Model](https://arxiv.org/html/2504.21463v2) —— SSM + Sparse Attention 混合
- [LongMamba (ICLR 2025)](https://www.jankautz.com/publications/LongMamba_ICLR25.pdf) —— Mamba 长上下文增强
- [On the Length Generalization of Mamba (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/file/1bfc9f74afa91b9b8add5a97a97001a1-Paper-Conference.pdf)
- [Characterizing SSM/Hybrid LM Performance with Long Context](https://arxiv.org/html/2507.12442v4)
- [Achilles' Heel of Mamba (NeurIPS 2025 Spotlight)](https://arxiv.org/html/2509.17514v1)
- Nemotron-H: Hybrid Mamba-Transformer Models（NVIDIA, 2025-03）

### 工程参考

- [llama.cpp](https://github.com/ggml-org/llama.cpp) —— CPU 端侧 LLM 推理事实标准
- [BitNet.cpp](https://github.com/microsoft/BitNet) —— 1.58-bit ternary CPU 推理
- [lm.c](https://github.com/oderoi/lm.c) —— 极简 LM 训练参考
- [tinygrad](https://docs.tinygrad.org/) —— 轻量 autograd 引擎参考
- [PureML (JOSS paper)](https://joss.theoj.org/papers/10.21105/joss.09631.pdf)
- [micrograd](https://github.com/karpathy/micrograd) —— 教学级 autograd

### 世界模型

- [V-JEPA 2](https://aiwiki.ai/wiki/v_jepa_2) —— Meta 视频世界模型
- [JEPA Deep Dive](https://dgallitelli.github.io/blog/world-models-series/part2-jepa-deep-dive/)
- [Beyond Next-Token Prediction: World Models and JEPA](https://ai.briqmind.com/research_en/14-world-models-jepa/)
- LeCun, "A Path Towards Autonomous Machine Intelligence"（2022）

## License

[MIT License](LICENSE) © 2026 CometFuture.

---
