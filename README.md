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

### CometSpark 双模型（Part5K1 / Part5K1.1）

`spark/` 目录承载基于 VerseNex 原生架构的双模型 LM 训练仓库（Part5K1.1 精简为 `small/` + `mate/` + `src/` 三目录）：

- **CometSparkSmallLM / CometSparkMateLM**：组合 `verse_nex.CometSparkNexLM`（内部 `VerseNexBlock` = TriSparse + MoD），不重造底层 block
- **small**：0.06zB 目标（1zB ≈ 1010B 等效能力），`n_embd=64, n_layer=2, vocab=256`，端侧部署
- **mate**：0.2zB 目标，`n_embd=1024, n_layer=20, 5 MoD + 15 trisparse` + `tie_weights=True` + `embedding_scale=True` ≈ 1.12B 参数
- **Qwen tokenizer**：`vocab=248320`（Qwen3.5-35B-A3B，mate 模型）
- **解决胡乱输出**：embedding scale + tie_weights + temperature scaling + 合理初始化
- **MoD V1.2（Part5K1.1）**：温度可调路由 + aux loss EMA 平滑 + 熵正则化，修复训练时 MoD 显示不稳定问题
- **训练/推理 CLI**：`python spark/run.py train --model small|mate`，详见 [spark/README.md](spark/README.md)

### spark/run.py 快捷入口（Part4K2.5 / Part5K1.1）

`spark/run.py` 是基于 VerseTrainer API 封装的命令行快捷入口，提供 7 个子命令，**所有命令都有合理默认值，最小化用户配置**。Part5K1.1 新增 `--model small|mate` 自动调用（无需指定配置 / 权重目录）+ 表格化信息显示。无需安装即可直接 `python spark/run.py <子命令>` 运行，路径自举由 `spark/_bootstrap.py` 统一完成。

| 子命令 | 用途 | 关键参数 |
|---|---|---|
| `train` | 训练模型（训练后默认自动评估） | `--model small\|mate`（自动调用对应配置）/ `--config` / `--max-steps` / `--device cpu\|cuda\|npu` / `--amp` / `--resume` / `--eval-after` / `--no-eval` / `--dry-run` / `--quiet` / `--verbose` |
| `eval` | 评估 + 打分 | `--model` / `--checkpoint` / `--max-tokens`（默认不限，EOS 自然停止）/ `--temperature` / `--score` / `--references-file` |
| `generate` | 生成文本 | `--model` / `--prompt` / `--max-tokens` / `--temperature` / `--top-k` |
| `chat` | 交互式聊天（支持 `/quit` / `/clear` / `/save`） | `--model` / `--checkpoint` / `--max-tokens`（默认 512）/ `--temperature` |
| `compress` | 压缩模型（VMPC V2.0 默认 / `--no-vmpc` legacy） | `--model` / `--checkpoint` / `--no-vmpc` / `--compensate` / `--method` / `--sparsity` / `--qtype` / `--output` |
| `convert` | 模型格式互转（`.pt ↔ .vn`） | `--input` / `--output` / `--chat-template` / `--tokenizer` / `--arch` |
| `download` | 下载数据集（URL + HuggingFace + 自动转 .npz） | `--url` / `--hf` / `--split` / `--to-npz` / `-o` / `--workers` / `--no-resume` |

```bash
# 快速训练（small 模型，自动调用 spark/small/config/cometspark_small.yml）
python spark/run.py train --model small

# mate 旗舰训练
python spark/run.py train --model mate

# 跳过训练后自动评估
python spark/run.py train --model small --no-eval

# 生成文本（自动从 mf_small/ 找最新 checkpoint）
python spark/run.py generate --model small --prompt "你好世界"

# 交互式聊天
python spark/run.py chat --model small

# 压缩模型（VMPC V2.0 默认 + 训练补偿）
python spark/run.py compress --model small --compensate

# Legacy 压缩（传统技术直通）
python spark/run.py compress --model small --no-vmpc --method prune,quantize

# 模型格式互转
python spark/run.py convert --input checkpoints/best.pt --output model.vn

# 下载数据集
python spark/run.py download --url https://example.com/data.jsonl --to-npz -o data/cached.npz
```

> **提示**：所有子命令均支持 `--dry-run` 只打印将要执行的操作而不真正执行。Part5K1.1 起 `--model small|mate` 自动调用对应配置与 checkpoint 目录，并用 ASCII 表格展示配置 / 模型信息（无外部依赖）。

### GPU/NPU 支持（DeviceBackend）

`verse_torch.device.DeviceBackend` 提供设备后端抽象：

- `NumpyBackend`（默认 CPU）：所有算子用 NumPy 实现，与自研 autograd 完全等价
- `TorchBackend`（PyTorch 委托）：CUDA kernel 走 PyTorch 原生实现，NPU 通过 `torch_npu` 扩展支持；**不自研 kernel**
- `autocast`：fp16 混合精度上下文管理器（CPU 时 no-op）
- `Tensor.device` / `.to(device)` / `.cuda()` / `.npu()` / `.cpu()`：API 与 PyTorch 一致
- `Module.to(device)`：迁移所有参数到目标设备
- 无 PyTorch 环境下 `Tensor.cuda()` 抛 `RuntimeError`，CPU 路径完全不变

详见 [ADR-005 GPU/NPU 后端抽象](docs/architecture/adr-005-gpu-npu-backend.md)。

### .vn 文件格式（Part4K2 / Part5K1.1 多空间缓存）

`verse_torch.vn_format` 定义基于 **safetensors 性能优化**的模型容器格式 `.vn`，取代传统 pickle `.pt`。Part5K1.1 新增**多空间缓存**支持：

- **ZIP 容器**：内含 `model.safetensors`（或降级 `model.npz`）+ `config.yml` + `chat_template.jinja`（可选）+ `tokenizer.json`（可选）+ `meta.json`
- **mmap 零拷贝**：safetensors 可用时通过 `safe_open` 零拷贝读取权重；npz 路径落盘懒加载
- **无损互转**：`pt_to_vn` / `vn_to_pt` / `convert_format`，权重数值完全一致
- **自描述**：`meta.json` 记录 `vn_format_version` / `arch` / `weight_format` / `compression_info` / `created_at`
- **安全性**：npz 路径强制 `allow_pickle=False`，safetensors 本身 pickle-free
- **多空间缓存（Part5K1.1）**：`VNCacheManager` 按需在内存与硬盘间自动缓存——内存充足 + 硬盘充足时优先混合缓存（优先级高的放内存，优先级低的放硬盘），内存不充足时主要放硬盘。为 VMPC V2.0 提供高吞吐、高速度的底层支撑。
- **VMPC 强制格式**：`use_vmpc=True`（默认）时所有模型文件必须使用 `*.vn` 格式，不可替换修改。
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

### VMPC V2.0 压缩技术（Part5K1.1 大升级）

Part5K1.1 将 **VMPC（VerseNext Model Parameters Compression）** 全面升级到 V2.0，**全面抛弃 V1.5 的技术栈**，明确 **VMPC ≠ 单纯的量化 / 剪枝**。VMPC V2.0 基于三大支柱：

1. **VN 格式文件**：高吞吐、高速度、方便压缩的模型容器（支持多空间缓存：内存 LRU + 硬盘 mmap 混合调度）。
2. **传统技术**：量化 / 剪枝 / 蒸馏 / LoRA（作为 VSC 的底层算子，不再是独立流程）。
3. **VSC（VerseNext Space Compression）**：VerseNext 空间压缩技术，从**三维空间角度**（存储 / 算力 / 时间）对模型做特别压缩，保持「速度快、能力强、占用小」三维优势。

> **等效能力**：在 VMPC V2.0 支持下，**1zB ≈ 1010B 模型能力**（等效处理）。物理压缩占 40%（量化 + 剪枝），专项算法优化 + 训练补偿占 60%。

- **统一开关**：所有配置通过 `vmpc.use_vmpc` 单项管理（默认开启）。`use_vmpc=true` 时强制 `*.vn` 格式 + VSC 引擎；`use_vmpc=false` 走 legacy 模式（传统技术直通，允许 `.pt`）。
- **VSC 引擎**：`verse_torch.vsc.VSCEngine` 独立算法引擎，按 `storage_weight` / `compute_weight` / `time_weight` 三维权重优化压缩策略。
- **训练补偿**：`VMPCV2.compensate()` 通过 `train_fn` 恢复压缩损失的能力（算法优化 60% 的核心环节）。
- **VN 多空间缓存**：`VNCacheManager` 实现内存 LRU + 硬盘 mmap 混合缓存，按优先级自动调度（内存充足时混合缓存，内存不足时硬盘优先）。
- **贯穿全流程**：VMPC 作为独立组件 API 贯穿训练 / 微调 / 后训练 / 推理，但不过度耦合。
- **彻底替换旧架构**：V1.5 的 `compress_pipeline(version="1.5")` / `contrastive_distill` / `logit_calibration` 等接口已不保留；legacy 路径仅保留 `compress_pipeline` 传统技术直通（`--no-vmpc`）。

```python
from verse_torch.vmpc import vmpc_compress, VMPCV2
from verse_torch.vsc import VSCEngine

# 一键 VMPC V2.0 压缩（返回 (compressed_model, stats) 元组）
compressed, stats = vmpc_compress(
    model, profile="small", use_vmpc=True,
    compensate_fn=train_fn,  # 可选：训练补偿
)

# 模型自带接口（CometSparkSmallLM / CometSparkMateLM）
new_model = model.vmpc_compress_model(use_vmpc=True)
new_model.save("model_vmpc.vn", format="vn")  # use_vmpc=True 时强制 .vn

# Legacy 路径（传统技术直通，允许 .pt）
compressed, stats = vmpc_compress(model, use_vmpc=False)
```

### 双模型并行 small / mate（Part5K1 / Part5K1.1）

Part5K1 正式开始双模型并行，`spark/small/` 与 `spark/mate/` 分别承载两个定位差异明显的模型。Part5K1.1 精简目录为 `small/` + `mate/` + `src/` 三目录（弃用旧 `config/` / `model/`）。详见 [ADR-014](docs/architecture/adr-014-dual-model-small-mate.md)。

- **spark/small/**：0.06zB 小模型（1zB ≈ 1010B 等效能力），面向端侧 / 嵌入式 / 树莓派。VMPC-small 预设：ternary 量化 + 高稀疏 0.5。
- **spark/mate/**：0.2zB 旗舰模型，面向消费级 CPU / 单卡 GPU。VMPC-mate 预设：int4 量化 + 中稀疏 0.3 + LoRA + 蒸馏。
- **统一基座**：双模型均基于 `verse_nex.CometSparkNexLM`，VMPC V2.0 适配通过 config 传入。
- **checkpoint 目录**：`mf_small/` / `mf_mate/`（替代旧 `checkpoints_XXX/`），`mf_` 前缀统一。
- **模型文件强制 `.vn`**：`use_vmpc=True`（默认）时强制 `*.vn` 格式（不可替换修改）；legacy 模式允许 `.pt`。
- **`spark/run.py` 支持 `--model small|mate` 自动调用**：指定 `--model` 后无需再指定配置 / 权重目录，自动查找对应配置与 checkpoint；表格化显示配置与模型信息。

```bash
# 训练 small 模型（0.06zB，端侧部署）—— 自动用 spark/small/config/cometspark_small.yml
python spark/run.py train --model small

# 训练 mate 模型（0.2zB，旗舰能力）—— 自动用 spark/mate/config/cometspark_mate.yml
python spark/run.py train --model mate

# 生成 / 聊天也自动查找对应目录的最新 checkpoint
python spark/run.py generate --model small --prompt "你好世界"
python spark/run.py chat --model mate

# 压缩（VMPC V2.0 默认 / --no-vmpc legacy）
python spark/run.py compress --model small --compensate
```

### VMT 完整三档策略（Part5K1 新增）

Part5K1 推出 **VMT（Versenext Memory-aware Training）** 完整三档智能分区训练，`VMTTrainer` 继承 `LayerWiseTrainer`，支持按层位置差异化分配训练策略。详见 [ADR-015](docs/architecture/adr-015-vmt-full-strategy.md)。

- **三档策略**：
  - `unload`：卸载到硬盘 `.vn` 分片（已有，LayerWiseTrainer 基础能力）。
  - `freeze`：INT4 量化 + `requires_grad=False`（压缩冻结），训练后从 fp32 备份精确恢复。
  - `optimize`：层融合 + 梯度累积（高频训练专项优化），前向走 `_fused_forward_blocks`。
- **vmt_strategy 解析**：
  - `"auto"`（默认）：前 1/3 freeze、中间 1/3 optimize、后 1/3 unload。
  - 显式语法：`"layers[0:8]=freeze, layers[8:56]=optimize, layers[56:]=unload"`。
- **LayerWiseTrainer 保留为简化版**（仅 unload），向后兼容；`VMTTrainer` 是超集。

```python
from verse_torch.layerwise_trainer import VMTTrainer

trainer = VMTTrainer(
    model, config={"lr": 1e-3},
    vmt_strategy="auto",         # 或显式语法
    partition_size=2,
)
trainer.fit(train_loader, val_loader, max_steps=1000)
```

### verse_torch.nn → verse_torch.vnn 重命名（Part5K1 新增，BREAKING）

Part5K1 将 `verse_torch.nn` 重命名为 `verse_torch.vnn`，统一核心类命名与 VerseNex / VMPC / VMT 品牌体系。详见 [ADR-016](docs/architecture/adr-016-nn-to-vnn-rename.md)。

- **`verse_torch.nn` → `verse_torch.vnn`（BREAKING）**：核心类（`Module` / `Linear` / `Embedding` / `VerseNexLM` / `VerseNexBlock` 等）迁移到 `vnn.py`。
- **`nn.py` 降级为 thin shim**：`from .vnn import *`，旧路径 `from verse_torch.nn import Module` 仍可工作（已废弃）。
- **transformer 系旧名抛 ImportError**：`TransformerLM` / `TransformerBlock` / `GQASelfAttention` 从 DeprecationWarning 升级为抛 ImportError（引导迁移到 VerseNex 命名）。
- **`__init__.py` 中 `nn = vnn` 别名**：`from verse_torch import nn` 仍可工作（PyTorch 用户友好）。

```python
# 推荐（新）
from verse_torch.vnn import Module, Linear, VerseNexLM

# 兼容（仍可工作）
from verse_torch import nn          # nn 是 vnn 的别名
from verse_torch.nn import Module   # 经 shim 转发（已废弃）

# 旧名抛 ImportError（BREAKING）
# from verse_torch.nn import TransformerLM  # ❌ ImportError，改用 VerseNexLM
```

### JSONL 自修复（Part5K1 新增）

`verse_torch.jsonl_repair` 提供 JSONL 文件自动修复与标准化，解决训练数据 JSONL 格式损坏（截断 / 引号未闭合 / 多行 JSON 拼接等）导致的加载失败问题。

- **自动修复**：识别并修复截断的 JSON 行、未闭合的引号 / 括号、多行 JSON 拼接等常见损坏模式。
- **标准化输出**：修复后输出标准 JSONL（每行一个合法 JSON 对象），与 `CachedDataset` / `TextDataset` 对齐。
- **val.json 自动生成**：训练数据准备阶段自动从 train.jsonl 切分验证集（Part5K1 Task 6）。
- **数据预加载**：`CachedDataset` 启动时预加载 + 缓存，跳过重复 tokenize 开销。

```python
from verse_torch.jsonl_repair import repair_jsonl

# 修复损坏的 JSONL 文件
repair_jsonl("data/train_corrupted.jsonl", "data/train.jsonl")
```

### 64+ 层训练加速（Part5K1 新增）

Part5K1 针对 64+ 层大模型训练推出层融合 + chunked_forward 加速（Task 7）。详见 [ADR-015](docs/architecture/adr-015-vmt-full-strategy.md)。

- **层融合（`_fused_forward_blocks`）**：把多个 VerseNexBlock 的前向融合为单次调用，减少 Python 循环开销，数值与逐块前向严格一致。
- **chunked_forward**：`n_layer >= 64` 时自动启用 chunked_forward，把长序列按 chunk 分块前向，降低峰值内存。
- **VMT optimize 档集成**：层融合路径由 VMTTrainer 的 optimize 档调用，配合梯度累积支持大 batch 等效训练。

```python
# 自动启用：n_layer >= 64 的模型训练时自动走 chunked_forward
# 手动调用层融合前向（VMT optimize 档内部使用）
logits = model._fused_forward_blocks(x, block_indices=range(0, 64))
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

# 5. CometSpark：双模型工厂（Part5K1.1）
from spark.small.model import CometSparkSmall, CometSparkSmallLM
from spark.mate.model import CometSparkMate, CometSparkMateLM

# 6. Part4K2 新增：.vn 格式 + 智能分区训练
from verse_torch import VNFileReader, VNFileWriter, LayerWiseTrainer

# 7. Part4K2 新增：数据集下载器（经 verse_infra 顶层导出）
from verse_infra import DatasetDownloader

# 8. Part5K1.1 新增：VMPC V2.0 + VSC
from verse_torch.vmpc import vmpc_compress, VMPCV2
from verse_torch.vsc import VSCEngine
```

### 一键训练 CometSpark 双模型

```bash
# 最快验证：spark/run.py 快捷入口（small 模型，约 10-30 秒完成，零安装可用）
python spark/run.py train --model small

# CPU 预训练（mate 旗舰，CPU / GPU / NPU）
python spark/run.py train --model mate --device cpu
python spark/run.py train --model mate --device cuda --amp
python spark/run.py train --model mate --device npu

# 并行训练（chunks > 1）
python spark/run.py train --model mate --parallel-chunks 4

# 断点续训
python spark/run.py train --model small --resume

# 后训练（NexRL / SFT / DPO）
python spark/run.py posttrain --model mate --rl nexrl
python spark/run.py posttrain --model mate --rl sft
python spark/run.py posttrain --model mate --rl dpo

# VMPC V2.0 压缩（默认 + 训练补偿）
python spark/run.py compress --model small --compensate
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
| 大→小模型能力转移 | **VMPC V2.0**（VN 格式 + 传统技术 + VSC 三维空间压缩，1zB ≈ 1010B 等效能力） | 物理压缩 40% + 算法优化 60%；`use_vmpc` 统一开关 + 强制 .vn + 训练补偿；详见 [ADR-013](docs/architecture/adr-013-vmpc-naming-v15.md) |
| 内存感知训练 | **VMT 三档策略**（unload / freeze / optimize） | VMTTrainer 继承 LayerWiseTrainer，按层位置差异化分配策略；详见 [ADR-015](docs/architecture/adr-015-vmt-full-strategy.md) |
| 双模型并行 | **spark/small（0.06zB）+ spark/mate（0.2zB）** | VMPC-small / VMPC-mate 预设 + `--model` 参数 + `mf_` checkpoint；详见 [ADR-014](docs/architecture/adr-014-dual-model-small-mate.md) |
| 核心类命名 | **verse_torch.vnn**（nn → vnn 重命名，BREAKING） | 品牌统一 + nn.py thin shim + nn=vnn 别名；详见 [ADR-016](docs/architecture/adr-016-nn-to-vnn-rename.md) |

详细架构决策记录见 [`docs/architecture/`](docs/architecture/)。

## 详细文档

### 训练指南

[Verse 训练指南](docs/training_guide.md) —— 从零训练 LM 的完整流程：数据准备 → tokenizer → 模型 → 训练 → 评估 → 压缩 → 推理。

### 性能调优

[Verse 性能调优指南](docs/performance_tuning.md) —— numba JIT 加速、BLAS 配置、batch_size 选择、CPU 线程数、量化加速、并行计算六个维度的 CPU 调优手册。

### 各包文档

| 包 | 文档 | 定位 |
|---|---|---|
| VerseTorch | [README](packages/verse_torch/README.md) | 张量 / autograd / vnn / optim / losses / training / quantize / parallel / compress / DeviceBackend（GPU/NPU）/ **.vn 格式（Part4K2）** / **LayerWiseTrainer（Part4K2）** / **compress V1.3（Part4K2）** / **vnn 重命名 + VMPC V1.5 + VMTTrainer + jsonl_repair（Part5K1）** / **VMPC V2.0 + VSC + VN 多空间缓存（Part5K1.1）** |
| VerseNex | [README](packages/verse_nex/README.md) | Mamba-2 / RWKV-7 / RetNet / Sparse Attention / Hybrid / TriSparseAttention / MoDLayer / VerseNexLM / NexRL / **compress_v13 / distill_from（Part4K2）** / **VMPC V1.5 适配（Part5K1）** / **MoD V1.2 路由优化（Part5K1.1）** |
| VerseAWM | [README](packages/verse_awm/README.md) | I-JEPA / V-JEPA / H-JEPA / RSSM 世界模型 |
| VerseInfra | [README](packages/verse_infra/README.md) | 总包结构 / 导入路径迁移指南 / shim 兼容（Part4K1）/ **DatasetDownloader（Part4K2）** |
| ├ verse_tokenizer | [README](packages/verse_infra/verse_infra/verse_tokenizer/README.md) | BPE / Unigram / WordPiece / Qwen tokenizer / NexTokenizerWrapper / **ChatML jinja2 模板（Part4K2）** |
| ├ verse_inference | [README](packages/verse_infra/verse_infra/verse_inference/README.md) | 模型加载 / 状态缓存 / 流式生成 / HTTP server |
| └ verse_compat | [README](packages/verse_infra/verse_infra/verse_compat/README.md) | HuggingFace / PyTorch 兼容层 |
| CometSpark | [README](spark/README.md) | CometSpark 双模型说明 + 配置 + 训练 CLI（Part4K1）/ **持续训练 verse-continue（Part4K2）** / **双模型并行 small/mate（Part5K1）** / **目录精简 + --model 自动调用 + 表格显示 + VMPC V2.0（Part5K1.1）** |

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
- [ADR-013 VMPC 命名 + V1.5 设计](docs/architecture/adr-013-vmpc-naming-v15.md)（**Part5K1 新增**）
- [ADR-014 双模型并行 small / mate](docs/architecture/adr-014-dual-model-small-mate.md)（**Part5K1 新增**）
- [ADR-015 VMT 完整三档策略](docs/architecture/adr-015-vmt-full-strategy.md)（**Part5K1 新增**）
- [ADR-016 verse_torch.nn → vnn 重命名](docs/architecture/adr-016-nn-to-vnn-rename.md)（**Part5K1 新增**）

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

### Part4K2.5 重大升级摘要

Part4K2.5 在 Part4K2 基础上完成 6 项紧急优化，重点解决 **易用性 / 包导入稳定性 / 训练可视化正确性 / 训练后验证闭环 / 并行训练健壮性** 五个方向：

- **spark/run.py CLI 快捷方式（Task 1）**：基于 VerseTrainer API 封装的 7 子命令入口（train/eval/generate/chat/compress/convert/download），所有命令有合理默认值，零安装可用 `python spark/run.py <子命令>`；统一 `--dry-run` 预览参数。
- **包导入修复（Task 2）**：新增 `spark/_bootstrap.py` 统一路径引导模块（基于 `__file__` 推断，幂等注入 sys.path），简化 6 处重复的 `sys.path.insert` 路径自举为单次 `import spark._bootstrap`；删除孤儿 `.pyc` 缓存。
- **loss 图表修复（Task 3）**：`plot_loss_curve` x 轴对齐修复（按 `eval_interval` 步长取点，避免偏移）；ASCII 降级模式正确显示 val 线；`visualize` 统计增强。
- **训练后自动评估（Task 4）**：`train()` 新增 `eval_after` 参数（默认 `True`），训练完成后调用 `_auto_evaluate` 对 best checkpoint 做 5 指标打分（exact_match / prefix_accuracy / char_f1 / bleu / rouge_l），结果写入返回 dict 的 `eval_result` 字段；`spark/run.py train --no-eval` 可跳过。
- **小错误修复 + 性能优化（Task 5）**：见 [性能调优指南](docs/performance_tuning.md) 的 Part4K2.5 优化清单。
- **并行训练修复（Task 6）**：`ParallelTrainer` chunk 间状态重置（每个 chunk 创建独立优化器，避免状态泄漏）；`round_robin` 数据分配策略使 chunk 间数据均匀不重复；Phase 2 重训在 `chunk_steps < 4` 时跳过（避免步数为 0 崩溃）；非 tty 环境 tqdm 自动降级为简洁打印（CI / 重定向场景不再输出 `\r` 垃圾字符）。

### Part5K1 重大升级摘要

Part5K1 在 Part4K2.5 基础上完成 12 项功能任务 + 1 项文档任务，重点完成 **VMPC 命名 + V1.5 算法升级 / 双模型并行 / VMT 完整三档策略 / vnn 重命名 / VerseTorch 底层去壳** 五大方向：

- **verse_torch.nn → vnn 重命名（Task 1，BREAKING）**：`verse_torch.nn` → `verse_torch.vnn`，核心类迁移到 `vnn.py`；`nn.py` 降级为 thin shim（`from .vnn import *`）；transformer 系旧名（TransformerLM / TransformerBlock / GQASelfAttention）从 DeprecationWarning 升级为抛 ImportError；`__init__.py` 中 `nn = vnn` 别名（向后兼容 `from verse_torch import nn`）。详见 [ADR-016](docs/architecture/adr-016-nn-to-vnn-rename.md)。
- **VerseTorch 底层去壳与合并（Task 2）**：VerseTorch 底层依赖去壳，模块合并精简，减少冗余封装层。
- **VMPC 命名 + V1.5 门面（Task 3）**：`verse_torch.vmpc` 作为压缩 / 量化 / 蒸馏 / 剪枝统一门面（re-export `compress.py` 核心对象）；`VMPCRegularizer`（防过拟合 + 压缩感知稀疏收紧，val_loss 平台期自动收紧 sparsity）；`vmpc_compress(model, profile="small"|"mate")` 一键预设。详见 [ADR-013](docs/architecture/adr-013-vmpc-naming-v15.md)。
- **VMPC V1.5 算法（Task 4）**：`contrastive_distill`（对比蒸馏，margin ranking loss 保证 student/teacher top-k 排序一致）+ `logit_calibration`（推理 logits 校准，temperature-aware sharpening）+ outlier-aware 反量化（outlier channel 保留 fp16，其余 int4）+ `vmpc_version` 元数据。`compress.py` 默认 `version="1.5"`，V1.3 仍可通过 `version="1.3"` 访问。
- **JSONL 自修复（Task 5）**：`verse_torch.jsonl_repair` 提供 JSONL 文件自动修复与标准化（截断 / 引号未闭合 / 多行 JSON 拼接等损坏模式）。
- **val.json 自动生成 + 数据预加载（Task 6）**：训练数据准备阶段自动从 train.jsonl 切分验证集；`CachedDataset` 启动时预加载 + 缓存。
- **64+ 层训练加速（Task 7）**：层融合（`_fused_forward_blocks`，数值与逐块前向严格一致）+ chunked_forward（`n_layer >= 64` 自动启用，降低峰值内存）。
- **VMT 完整三档策略（Task 8）**：`VMTTrainer` 继承 `LayerWiseTrainer`，支持 unload / freeze / optimize 三档（freeze = INT4 量化冻结，optimize = 层融合 + 梯度累积）；`vmt_strategy` 解析（"auto" 或显式语法 `layers[0:8]=freeze, ...`）。详见 [ADR-015](docs/architecture/adr-015-vmt-full-strategy.md)。
- **spark 双模型并行（Task 9）**：`spark/small/`（0.06zB 小模型，VMPC-small 预设：ternary + 高稀疏 0.5）+ `spark/mate/`（0.2zB 旗舰模型，VMPC-mate 预设：int4 + 中稀疏 0.3 + LoRA + 蒸馏）。详见 [ADR-014](docs/architecture/adr-014-dual-model-small-mate.md)。
- **checkpoint 重命名（Task 10）**：`mf_XXX/` 目录命名（替代旧 `checkpoints_XXX/`）+ `.vn` 默认输出格式。
- **spark/run.py 训练模式补齐（Task 11）**：`finetune` / `posttrain` / `continue` 子命令补齐 + `--model small|mate` 参数支持。
- **VerseNex 精简（Task 12）**：VerseNex 包结构精简，去除冗余模块。
- **文档与代码注释升级（Task 13）**：新增 ADR-013 ~ ADR-016；更新 README / training_guide / performance_tuning；代码注释统一到 VMPC 术语。

### Part5K1.1 重大升级摘要

Part5K1.1 在 Part5K1 基础上完成模型升级 + 架构升级 + 修复，重点完成 **VMPC V2.0 + VSC / VN 多空间缓存 / MoD V1.2 / 目录精简 / run.py 体验优化** 五大方向：

- **Bug 修复**：修复 `run.py` 加载模型时 `KeyError: 'config'`（`.pt` payload 容错）；修复 MoD 训练时显示不稳定（`layer_pattern` 固化到 config，保证 save/load 一致）。
- **MoD V1.2 升级**：温度可调路由（`router_temperature`）+ aux loss EMA 平滑（`aux_loss_ema_decay`）+ 熵正则化（`entropy_reg_weight`），提升路由能力与稳定性。
- **VMPC V2.0 + VSC**：全面抛弃 V1.5 技术栈，基于 **VN 格式 + 传统技术 + VSC（VerseNext Space Compression）** 三大支柱；`use_vmpc` 统一开关（默认开启）+ 强制 `*.vn` 格式；VSC 从存储 / 算力 / 时间三维空间角度压缩；训练补偿恢复能力（物理压缩 40% + 算法优化 60%，1zB ≈ 1010B 等效能力）。
- **VN 多空间缓存**：`VNCacheManager` 实现内存 LRU + 硬盘 mmap 混合缓存，按优先级自动调度（内存充足时混合缓存，内存不足时硬盘优先），为 VMPC V2.0 提供高吞吐底层支撑。
- **目录精简**：`spark/` 弃用旧 `config/` / `model/` 目录，仅保留 `small/` + `mate/` + `src/` 三目录。
- **run.py 体验优化**：`--model small|mate` 自动调用对应配置与 checkpoint 目录（无需指定配置 / 权重目录）+ ASCII 表格化显示配置 / 模型信息（无外部依赖）+ compress 双路径（VMPC V2.0 默认 / `--no-vmpc` legacy）。
- **全面升级文档与代码注释**：README / spark/README / docs / 代码注释统一到 VMPC V2.0 + VSC 术语。

### 第二次进化 / Part3K2 / Part4 摘要

详见 [审计报告](audit_report.md) 与 [Part4 升级报告](docs/part4_upgrade_report.md)。

## 仓库结构

```
/workspace/
├── packages/
│   ├── verse_torch/        # 张量与 autograd 引擎（含 device.py / backend_torch.py GPU/NPU 后端）
│   │   └── verse_torch/
│   │       ├── vnn.py             # 核心神经网络类（Module/Linear/VerseNexLM，Part5K1 重命名）
│   │       ├── nn.py              # thin shim → vnn（向后兼容，Part5K1）
│   │       ├── vn_format.py        # .vn 文件格式 + VNCacheManager 多空间缓存（Part4K2/Part5K1.1）
│   │       ├── layerwise_trainer.py # 智能分区训练（LayerWiseTrainer + VMTTrainer，Part4K2/Part5K1）
│   │       ├── compress.py         # 传统压缩管线（legacy 路径：prune→quantize→distill→lora）
│   │       ├── vmpc.py            # VMPC V2.0（vmpc_compress + VMPCV2 + 训练补偿，Part5K1.1）
│   │       ├── vsc.py             # VSC 三维空间压缩引擎（VSCEngine，Part5K1.1）
│   │       └── jsonl_repair.py    # JSONL 自修复与标准化（Part5K1）
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
├── spark/                  # CometSpark 端到端 LM 训练仓库（Part4K1，Part5K1 双模型，Part5K1.1 目录精简）
│   ├── _bootstrap.py       # 统一路径引导模块（Part4K2.5）
│   ├── run.py              # CLI 快捷入口（--model small|mate 自动调用 + 表格显示，Part5K1.1）
│   ├── small/              # 0.06zB 小模型（VMPC-small 预设 + V2.0 字段，Part5K1/Part5K1.1）
│   ├── mate/               # 0.2zB 旗舰模型（VMPC-mate 预设 + V2.0 字段，Part5K1/Part5K1.1）
│   └── src/                # 共享基础组件（base_config/base_model/data_loader/trainer/evaluate/utils）
├── datasets/               # raw / cleaned / tokenizer
├── docs/                   # papers / architecture / benchmarks
│   └── architecture/       # ADR-001 ~ ADR-016
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
