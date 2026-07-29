# Part5K1.8 gigatoken 兼容修复 + 训练数据优化 + 交互式 tokenizer 初始化 Spec

## Why

Part5K1.7 完成后，实际运行 small 模型训练时暴露出 gigatoken 兼容组件的致命错误：
`RuntimeError: 加载 tokenizer 失败: kind=giga, path=.../mf_small/tokenizer.json,
error=Repo id must be in the form 'repo_name' or 'namespace/repo_name'`。

根因是 `GigaTokenizerWrapper.__init__` 收到 `.json` 文件路径（元信息文件）时，
错误地走 `AutoTokenizer.from_pretrained(json_path)` 分支，而 HuggingFace 的
`from_pretrained` 不接受 `.json` 文件路径（要求 repo_id 或目录）。

同时 small 模型缺少真实训练数据（当前 `_auto_generate_test_data` 仅生成 100 条
`{"text":"..."}` 重复文本），且 tokenizer 文件不存在时无交互式选择，用户体验差。

本变更修复上述三类问题，并补充 `UPDATE.MD` 项目结构分析，为下一阶段优化提供依据。

## What Changes

### 1. gigatoken 兼容组件修复（giga.py + bpe.py）

- **`GigaTokenizerWrapper.__init__`**：当 `model_id_or_tokenizer` 为字符串、
  以 `.json` 结尾且文件存在时，**不再走 `AutoTokenizer.from_pretrained`**，
  改为先用默认 `DEFAULT_GIGA_MODEL` 构造空实例，再调用 `self.load(path)`
  走元信息加载路径（复用已有 `load()` 方法，不重复造轮子）。
- **`load_tokenizer` 工厂函数**（bpe.py）：`kind="giga"` 分支增加路径类型判断：
  - `path` 为 `.json` 文件 → 创建空 wrapper 后调用 `load(path)`
  - `path` 为目录 → 走原 `GigaTokenizerWrapper(model_id_or_tokenizer=path)`
  - `path` 为 repo_id 字符串 → 走原构造路径
  - `path` 为 None → 走默认 `DEFAULT_GIGA_MODEL`

### 2. small 模型训练数据优化（40000 条 prompt-completion）

- **新建 `spark/small/data/generate_train_data.py`**：脚本化生成 40000 条
  `{"prompt":"...", "completion":"..."}` 格式训练数据，覆盖 8 大类别：
  - 问答（常识 / 科学 / 地理 / 历史）
  - 翻译（中英互译）
  - 代码（Python 基础 / 算法 / 数据结构）
  - 数学（算术 / 代数 / 几何）
  - 对话（日常 / 情感 / 建议）
  - 续写（诗词 / 故事 / 描述）
  - 指令（格式化 / 转换 / 摘要）
  - 知识（定义 / 解释 / 对比）
  - 使用模板 + 词表组合生成，保证多样性与低重复率
- **生成 `spark/small/data/train.jsonl`（40000 条）+ `val.jsonl`（500 条）**
- **修改 `_auto_generate_test_data`**：默认格式改为 `{"prompt":"", "completion":""}`，
  保留 `{"text":"..."}` 兼容（单样本模式仍可用）
- **修改 `spark/small/config/cometspark_small.yml`**：`data.train_path` 指向
  `../data/train.jsonl`（相对 config 目录），`val_path` 指向 `../data/val.jsonl`

### 3. 交互式 tokenizer 初始化（trainer.py）

- **`_load_tokenizer` 新增交互分支**：当 `tokenizer.json` 不存在且 `kind != "byte"`
  时，调用新函数 `_prompt_tokenizer_action(tok_cfg, save_dir, base_dir)`：
  - **TTY 环境**：打印选项界面，等待用户输入
    - `y` → 复制路径：从 `tok_cfg.tokenizer_repo` 或 `tok_cfg.from_hf` 下载并保存
    - `n` → 自行构建：调用 `_auto_build_tokenizer`
  - **非 TTY（CI=true）**：打印警告，默认走自行构建（n 路径），避免阻塞
- **统一 `tokenizer_repo` 读取**：`_prompt_tokenizer_action` 优先从 `tok_cfg` 读
  `tokenizer_repo`，兜底从 `model_cfg.tokenizer_repo` 读（兼容 small 配置）
- **覆盖所有训练入口**：`train / finetune / posttrain / continue` 均经过
  `_load_tokenizer`，自动获得交互能力（无需每个入口重复实现）

### 4. UPDATE.MD 补充项目结构分析

- **新增"七、verse_infra 审计"**：覆盖 `verse_tokenizer/giga.py`、`bpe.py`、
  `verse_trainer/trainer.py` 的中危问题（含本次修复的 giga .json 路径问题）
- **新增"八、spark/run.py 审计"**：覆盖 `_load_tokenizer_for_config` 与
  `trainer._load_tokenizer` 的重复逻辑、cmd_* 子命令的健壮性问题
- **更新摘要表**：中危 +N，低危 +M，合计更新

## Impact

- **Affected specs**: part5k1.7-high-risk-fix（giga 加载链路相关）
- **Affected code**:
  - `packages/verse_infra/verse_infra/verse_tokenizer/giga.py`（.json 路径处理）
  - `packages/verse_infra/verse_infra/verse_tokenizer/bpe.py`（工厂函数路径分发）
  - `packages/verse_infra/verse_infra/verse_trainer/trainer.py`（交互式初始化 + 数据格式）
  - `spark/small/config/cometspark_small.yml`（数据路径）
  - `spark/small/data/generate_train_data.py`（新建）
  - `spark/small/data/train.jsonl` + `val.jsonl`（新建）
  - `.agent_work/UPDATE.MD`（补充审计）
- **BREAKING**: 无（`_auto_generate_test_data` 保留 `{"text":"..."}` 兼容）

## ADDED Requirements

### Requirement: gigatoken .json 元信息文件加载

`GigaTokenizerWrapper` SHALL 支持以 `.json` 元信息文件路径作为构造参数，
通过复用 `load()` 方法重建 tokenizer，不再错误地走 `AutoTokenizer.from_pretrained`。

#### Scenario: 传入 .json 文件路径
- **WHEN** 调用 `GigaTokenizerWrapper(model_id_or_tokenizer="/path/to/tokenizer.json")`
  且该文件存在
- **THEN** wrapper 读取元信息 json，按 `native` 字段选择加载路径，
  返回可用的 tokenizer 实例，不抛 `Repo id must be in the form...` 错误

#### Scenario: 传入目录路径
- **WHEN** 调用 `GigaTokenizerWrapper(model_id_or_tokenizer="/path/to/tok_dir/")`
  且该目录存在
- **THEN** 走原兼容模式（`AutoTokenizer.from_pretrained(dir)` + `gt.Tokenizer.as_hf()`）

### Requirement: 40000 条 prompt-completion 训练数据

系统 SHALL 提供 `spark/small/data/generate_train_data.py` 脚本，生成 40000 条
`{"prompt":"...", "completion":"..."}` 格式的训练数据，覆盖 8 大类别，多样性丰富。

#### Scenario: 生成训练数据
- **WHEN** 运行 `python spark/small/data/generate_train_data.py`
- **THEN** 在 `spark/small/data/` 下生成 `train.jsonl`（40000 条）+ `val.jsonl`（500 条）
- **AND** 每条均为 `{"prompt":"...", "completion":"..."}` 格式
- **AND** 重复率 < 1%（相同 prompt+completion 组合）
- **AND** 覆盖问答 / 翻译 / 代码 / 数学 / 对话 / 续写 / 指令 / 知识 8 大类

### Requirement: 交互式 tokenizer 初始化

`_load_tokenizer` SHALL 在 `tokenizer.json` 不存在且 `kind != "byte"` 时，
提示用户选择"复制（y）"或"自行构建（n）"。

#### Scenario: TTY 环境 + 用户选 y
- **WHEN** `tokenizer.json` 不存在 + `kind="giga"` + TTY + 用户输入 `y`
- **THEN** 从 `tokenizer_repo` / `from_hf` 下载 tokenizer 并 save 到 `save_dir/tokenizer.json`
- **AND** 返回可用的 tokenizer 实例

#### Scenario: TTY 环境 + 用户选 n
- **WHEN** `tokenizer.json` 不存在 + `kind="giga"` + TTY + 用户输入 `n`
- **THEN** 调用 `_auto_build_tokenizer` 自行构建（对 giga 降级为 byte）
- **AND** 返回可用的 tokenizer 实例

#### Scenario: 非 TTY（CI=true）
- **WHEN** `tokenizer.json` 不存在 + `kind="giga"` + 非 TTY
- **THEN** 打印警告"非交互环境，默认自行构建"，调用 `_auto_build_tokenizer`
- **AND** 不阻塞等待输入

## MODIFIED Requirements

### Requirement: `_auto_generate_test_data` 默认格式

`_auto_generate_test_data` SHALL 默认生成 `{"prompt":"...", "completion":"..."}`
格式（与 `SingleSampleDataset` 的 prompt-completion 模式对齐），保留 `{"text":"..."}`
作为兼容格式。

#### Scenario: 自动生成测试数据
- **WHEN** 训练数据文件不存在 + `_is_test_config` 返回 True
- **THEN** 生成 `{"prompt":"...", "completion":"..."}` 格式的 JSONL
- **AND** `CachedDataset` 能正确解析两种格式

## REMOVED Requirements

无（本次不删除任何已有功能）
