# CometSpark-V0.2 数据下载与处理脚本

> 用于在本地下载公开数据集、处理为统一 schema、构建 Qwen3-32B tokenizer，为 CometSpark-V0.2（32 层 VerseNex，0.5B 参数）训练做准备。

## 目录结构

```
data/demo/scripts/
├── download_datasets.py     # 下载公开数据集脚本
├── process_datasets.py      # 数据处理脚本（raw/ -> train.jsonl）
├── build_tokenizer.py       # 下载并构建 Qwen3-32B tokenizer
├── README.md                # 本文档
└── raw/                     # 下载的原始数据（运行后生成，每源一个子目录）
    ├── belle_chat/
    │   └── data.jsonl
    ├── firefly_zh/
    │   └── data.jsonl
    ├── code_alpaca/
    │   └── data.jsonl
    ├── ...
    └── _hf_cache/           # build_tokenizer 的 HuggingFace 缓存
```

## 数据源列表与许可证

| 源名称 | HuggingFace repo | 语言 | 任务 | 许可证 | 默认启用 |
|---|---|---|---|---|---|
| `wiki_zh` | `wikiann` (subset=zh) | zh | ner（可作 LM） | CC-BY-SA 3.0 | ✓ |
| `belle_chat` | `BelleGroup/train_3.5M_CN` | zh | sft | Apache 2.0（仅限研究用途） | ✓ |
| `firefly_zh` | `YeungNLP/firefly-train-1.1M` | zh | sft | Apache 2.0 | ✓ |
| `code_alpaca` | `sahil2801/code-alpaca` | code | code | Apache 2.0 | ✓ |
| `math_qa_zh` | `BelleGroup/train_2M_CN` | zh | sft | Apache 2.0 | ✓ |
| `cmrc2018` | `cmrc2018` | zh | qa | CC-BY-SA 4.0 | ✓ |
| `alpaca_en` | `tatsu-lab/alpaca` | en | sft | CC-BY-NC 4.0（仅研究） | ✗（默认关闭） |

> **许可证注意**：`belle_chat` / `firefly_zh` / `alpaca_en` / `cmrc2018` 仅用于研究用途；商用前请检查各自最新许可声明。`code_alpaca` 与 `wikiann` 为 Apache 2.0 / CC-BY-SA，相对宽松。

## 运行步骤

### 步骤 0：安装依赖（可选，未安装会自动降级）

```bash
pip install huggingface_hub datasets transformers
# parquet 转换用（sahil2801/code-alpaca 下载为 parquet）：
pip install pyarrow    # 或 pip install pandas
```

### 步骤 1：下载原始数据集

```bash
cd /workspace/data/demo/scripts

# 1.1 列出可用数据源
python download_datasets.py --list

# 1.2 下载所有 enabled=True 的源
python download_datasets.py

# 1.3 仅下载指定的源（推荐先小规模试跑）
python download_datasets.py --only belle_chat firefly_zh

# 1.4 自定义 raw 目录
python download_datasets.py --raw-dir /path/to/raw
```

每个源下载到独立子目录 `raw/<source_name>/data.jsonl`，下载失败的源跳过，不影响其他源。

### 步骤 2：处理为统一 schema

```bash
cd /workspace/data/demo/scripts

# 2.1 默认处理：输出预训练 text 格式到 ../data/train.jsonl
python process_datasets.py

# 2.2 限制每源最多 1000 条（调试用）
python process_datasets.py --max-per-source 1000

# 2.3 输出 SFT messages 格式（供 SFTTrainer 使用）
python process_datasets.py --format messages

# 2.4 输出混合格式（text + messages 都有，最灵活）
python process_datasets.py --format both

# 2.5 自定义 raw / output 路径
python process_datasets.py --raw-dir ./raw --output /path/to/train.jsonl

# 2.6 仅处理指定源
python process_datasets.py --only belle_chat firefly_zh
```

### 步骤 3：构建 Qwen3-32B tokenizer

```bash
cd /workspace/data/demo/scripts

# 3.1 默认下载到 ../checkpoints_verse_nex/
python build_tokenizer.py

# 3.2 指定输出目录
python build_tokenizer.py --output-dir /path/to/tokenizer_dir

# 3.3 指定模型 repo（默认按 Qwen3-32B → Qwen2.5-32B → Qwen2.5-14B → Qwen2.5-7B 顺序尝试）
python build_tokenizer.py --repo-id Qwen/Qwen3-32B

# 3.4 强制重新下载
python build_tokenizer.py --force
```

### 步骤 4：合并到训练用的 `train.jsonl`

`process_datasets.py` 默认输出到 `../data/train.jsonl`，即 `config_verse_nex.yml` 中 `data.train_path` 的默认值。

若要单独保留原始 `train.jsonl`，可输出到自定义路径后手动合并：

```bash
python process_datasets.py --output /tmp/new_train.jsonl --format both
cat /tmp/new_train.jsonl >> ../data/train.jsonl
```

## 配置开关说明

### `download_datasets.py` 的 `ENABLED_SOURCES`

脚本顶部 `ENABLED_SOURCES` 字典控制每个源的开关与参数：

```python
ENABLED_SOURCES = {
    "belle_chat": {
        "enabled": True,             # 是否启用（False 时跳过下载）
        "repo_id": "BelleGroup/train_3.5M_CN",   # HuggingFace 数据集 ID
        "url": "https://...",        # urllib 兜底用的直接 URL
        "format": "jsonl",           # 输出格式：jsonl / parquet / json / txt
        "lang": "zh",                # 语言：zh / en / code
        "task_type": "sft",          # 任务类型：sft / lm / qa / ner / code
        "split": "train",            # load_dataset 的 split
        "description": "...",        # 描述（仅日志）
    },
    # ...
}
```

可通过 CLI 参数 `--only <name1> <name2>` 临时指定要下载的源，无需修改代码。

### `process_datasets.py` 的源元信息

`process_datasets.py` 通过两种方式确定每个源的 `(lang, task_type)`：

1. **优先**：读取 `raw/<src>/meta.json`（用户可手动创建覆盖默认）
2. **兜底**：使用 `DEFAULT_SOURCE_META` 字典中的默认映射

如需为某个源指定不同的 `lang` / `task_type`，创建 `raw/<src>/meta.json`：

```json
{"lang": "zh", "task_type": "sft"}
```

## 输出格式说明

### `--format text`（默认，预训练用）

```json
{"text": "<|user|>你好<|assistant|>你好！<|endoftext|>", "source": "belle_chat", "lang": "zh", "task_type": "sft"}
```

### `--format messages`（SFT 训练用）

```json
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}], "source": "belle_chat", "lang": "zh", "task_type": "sft"}
```

### `--format both`（混合，两种用法都支持）

```json
{"text": "<|user|>你好<|assistant|>你好！<|endoftext|>", "messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}], "source": "belle_chat", "lang": "zh", "task_type": "sft"}
```

### 通用文本类（wiki_zh / LM）

assistant 为空，仅渲染 `<|user|>{text}` 不带 `<|endoftext|>`：

```json
{"text": "<|user|>北京是中国的首都...", "source": "wiki_zh", "lang": "zh", "task_type": "lm"}
```

## 过滤规则

| 规则 | 阈值 | 说明 |
|---|---|---|
| 最小长度 | 50 字符 | `< --min-len` 丢弃 |
| 最大长度 | 8192 字符 | `> --max-len` 截断到该长度 |
| 重复文本 | SHA1 去重 | 跨源全局去重（同一 text hash 只保留一条） |
| 特殊 token | 含 `<\|user\|>` / `<\|imb\|>` / `<\|im_start\|>` 等 | 丢弃（避免破坏 chat template 渲染） |

可通过 CLI 参数调整：

```bash
python process_datasets.py --min-len 100 --max-len 4096
```

## 统计输出

`process_datasets.py` 运行结束会打印：

- 每个源的输入条数、保留条数、过滤短文本数、过滤特殊 token 数、去重数、平均长度
- 全局总输入 / 总保留 / 总过滤 / 总去重
- 全局总字符数 / 平均长度 / 最小长度 / 最大长度
- 字符分布：中文占比 / 英文占比 / 数字占比 / 其他占比

## 常见问题

### Q1: 下载失败，提示 "huggingface_hub 未安装" 或网络超时

**原因**：未安装 `huggingface_hub` 或 HuggingFace Hub 在你的网络环境不可达。

**解决方案**：

1. 安装依赖：`pip install huggingface_hub datasets`
2. 配置 HuggingFace 镜像（中国大陆推荐）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python download_datasets.py
   ```
3. 使用代理：
   ```bash
   export HTTPS_PROXY=http://127.0.0.1:7890
   export HTTP_PROXY=http://127.0.0.1:7890
   python download_datasets.py
   ```
4. 手动下载数据集后放到 `raw/<source_name>/data.jsonl`，再运行 `process_datasets.py`。

### Q2: 磁盘空间不足

**原因**：`BelleGroup/train_3.5M_CN` 解压后约 6GB+，全部源下载可能需要 20GB+ 空间。

**解决方案**：

1. 先下载小规模源试跑：
   ```bash
   python download_datasets.py --only firefly_zh code_alpaca
   ```
2. 处理时限制每源样本数：
   ```bash
   python process_datasets.py --max-per-source 50000
   ```
3. 清理 HuggingFace 缓存：`rm -rf ~/.cache/huggingface`

### Q3: parquet 文件无法解析

**原因**：未安装 `pyarrow` 或 `pandas`。

**解决方案**：

```bash
pip install pyarrow
# 或
pip install pandas
```

### Q4: 处理后 train.jsonl 为空

**可能原因**：

1. `raw/` 下没有任何源子目录 → 检查下载步骤
2. 所有样本都被过滤（长度 < 50 或含特殊 token）→ 调整 `--min-len` 或检查数据格式
3. 源的 `task_type` 与实际数据不匹配 → 创建 `raw/<src>/meta.json` 覆盖

**调试方式**：

```bash
# 查看某个源的原始数据格式
head -n 3 raw/belle_chat/data.jsonl

# 用极小的 min-len 测试
python process_datasets.py --min-len 1 --max-per-source 10 --only belle_chat
```

### Q5: build_tokenizer.py 失败

**原因**：Qwen3-32B 是 gated repo，可能需要 HF token；或网络问题。

**解决方案**：

1. 登录 HuggingFace：
   ```bash
   pip install huggingface_hub
   huggingface-cli login  # 输入你的 HF token
   ```
2. 使用备选 repo（不需要登录）：
   ```bash
   python build_tokenizer.py --repo-id Qwen/Qwen2.5-7B-Instruct
   ```
3. 手动下载：
   - 从 https://huggingface.co/Qwen/Qwen3-32B/tree/main 下载 `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` 等文件
   - 复制到 `data/demo/checkpoints_verse_nex/`
   - 运行 `python build_tokenizer.py` 验证

### Q6: 处理速度太慢

**原因**：Belle 3.5M / Firefly 1.1M 数据量大，纯 Python 处理慢。

**解决方案**：

1. 限制每源样本数：`--max-per-source 100000`
2. 多次运行，每次处理一个源：`--only <src>`（避免一次加载所有源）
3. 在大内存机器上运行（Belle 3.5M 全量处理约需 8GB+ 内存）

### Q7: 想增加新的数据源

1. 在 `download_datasets.py` 顶部的 `ENABLED_SOURCES` 中添加新条目
2. 在 `process_datasets.py` 的 `DEFAULT_SOURCE_META` 中添加 `(lang, task_type)` 映射
3. 运行 `python download_datasets.py --only <new_source_name>` 下载
4. 运行 `python process_datasets.py --only <new_source_name>` 处理

## 与训练流程的对接

完成上述步骤后：

1. `data/demo/data/train.jsonl` —— 处理后的训练数据（统一 schema）
2. `data/demo/checkpoints_verse_nex/tokenizer.json` —— Qwen3-32B tokenizer
3. 修改 `config/config_verse_nex.yml`：

```yaml
tokenizer:
  kind: qwen                      # 使用 Qwen3-32B tokenizer
  vocab_size: 151936

data:
  train_path: data/train.jsonl    # 指向处理后的数据
  val_path: data/val.jsonl        # 验证集（可单独从 train.jsonl 切分）
```

4. 运行训练：

```bash
cd /workspace/data/demo
python run.py --config config/config_verse_nex.yml
```
