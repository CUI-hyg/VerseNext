# Part3: 架构优化升级 + 模型优化 Spec

## Why

Verse 框架在第二次进化（CometSpark-v0.1）后已跑通端到端 LM 训练流程，但存在多个阻塞性 Bug 与体验问题：
- `run.py` 出现 `maximum recursion depth exceeded`，导致评估阶段崩溃，端到端流程不可用
- hybrid 模式（SSM + Sparse Attention）在 `seq_len >= 64` 时 `np.exp(log_decay)` 数值溢出为 NaN，迫使 CometSpark 退化为纯 transformer，丧失线性复杂度优势
- 生成文本频繁出现乱码（U+FFFD），中文场景下尤其严重
- CLI 缺少自定义 prompt / 推理参数，错误处理薄弱，`server.py` 相对导入失败
- 极简 YAML 解析器不支持 list / 多行字符串，限制了配置表达能力
- `val_loss` 曲线在可视化中视觉丢失（数据未丢，但标记不明显）
- 模型参数量仅 ~130K，远低于参考项目 Gpt_teacher-3.37M-cn 的 3.37M

本 spec 旨在修复上述阻塞问题，升级 tokenizer 架构向先进模型看齐，替换极简 YAML 为 PyYAML，优化模型架构提升参数量，并增强框架依赖与性能。

## What Changes

### 架构优化升级
- **升级 tokenizer**：重构 `verse_tokenizer`，提供标准化 Tokenizer 接口（向先进模型看齐），支持 NFKC 正规化、字节对齐解码、BPE/Byte/Char 统一抽象基类
- **解决乱码问题**：`ByteTokenizer.decode` 增加字节对齐检查，BPE merge 强制 UTF-8 字节边界，生成时强制 eos，参考 Gpt_teacher-3.37M-cn 的处理方法
- **升级架构**：删除 `data/demo/model/config.py` 中的极简 YAML 解析器，改为直接使用 PyYAML 库（`pyyaml>=6.0` 列入依赖），保留无 PyYAML 时的 fallback warning
- **解决 hybrid 数值溢出**：在 `verse_nex/mamba2.py` 和 `rwkv7.py` 对 `log_decay` 加 `np.clip(-50, 0)`，对 `A_log` 参数化约束（`A = -softplus(A_log) - eps`），对 `dt` 加上界
- **修复 BUG 与错误**：修复 `run.py` 递归溢出、`server.py` 相对导入、CLI 错误处理

### 模型优化
- **支持自定义 Prompt**：`run.py` 增加 `--prompt` / `--prompts-file` / `--max-tokens` / `--temperature` / `--top-k` 参数，参考 Gpt_teacher-3.37M-cn 的交互模式
- **修复 val_loss 曲线丢失**：`plot_loss_curve` 给 val 曲线加显著标记（marker/linewidth），ASCII 模式独立行绘制，额外生成 `val_losses.txt`
- **修复 CLI 意外错误**：`server.py` 改绝对导入，`run.py` 统一 try/except + traceback，补充 `--help` 示例
- **优化模型架构提升参数量**：提供 small/medium/large 三套预设配置，目标支持 1-3M 参数（CPU 5GB 约束下），实现 `count_parameters()` 工具
- **优化框架**：`verse_nex` 增加 numba 可选加速，`verse_torch` 用 `np.einsum` 优化热点，文档补充性能调优指南
- **解决 run.py 递归溢出**：定位 `CometSparkLM.generate` → `HybridLM.forward_recurrent` 的递归调用链，修复循环引用，添加 `sys.setrecursionlimit` 缓解

## Impact

- **Affected specs**：
  - `evolve2-cometspark`（CometSpark-v0.1 训练仓库）—— config.py / evaluate.py / run.py 大幅修改
  - `build-verse-framework`（verse_torch / verse_nex / verse_tokenizer）—— 核心模块修改
- **Affected code**：
  - `/workspace/packages/verse_nex/verse_nex/mamba2.py` — 数值溢出修复
  - `/workspace/packages/verse_nex/verse_nex/rwkv7.py` — 数值溢出修复
  - `/workspace/packages/verse_nex/verse_nex/hybrid.py` — generate 递归修复
  - `/workspace/packages/verse_tokenizer/verse_tokenizer/bpe.py` — Tokenizer 架构升级 + 乱码修复
  - `/workspace/packages/verse_torch/verse_torch/training.py` — val_loss 曲线修复
  - `/workspace/packages/verse_inference/verse_inference/server.py` — CLI 修复
  - `/workspace/data/demo/run.py` — 递归修复 + CLI 扩展
  - `/workspace/data/demo/model/config.py` — PyYAML 替换
  - `/workspace/data/demo/model/model.py` — generate 递归修复
  - `/workspace/data/demo/train/evaluate.py` — 自定义 prompt 支持
  - `/workspace/data/demo/config/config.yml` — 参数量提升
  - `/workspace/pyproject.toml` — 依赖增强（pyyaml / numba 可选）

## ADDED Requirements

### Requirement: 标准化 Tokenizer 接口
系统 SHALL 提供标准化的 Tokenizer 抽象基类，统一 BPE / Byte / Char 三种实现接口，支持 NFKC 正规化与预处理，向先进模型（如 GPT-4 / Llama）的 tokenizer 设计看齐。

#### Scenario: 标准化接口一致
- **WHEN** 用户使用 `BPETokenizer` / `ByteTokenizer` / `CharTokenizer` 任一实现
- **THEN** 三者都继承自 `BaseTokenizer`，提供统一的 `encode(text) -> List[int]` / `decode(ids) -> str` / `save(path)` / `load(path)` / `__len__()` 接口

#### Scenario: NFKC 正规化
- **WHEN** 输入文本含全角字符或 Unicode 组合形式
- **THEN** Tokenizer 在 encode 前自动做 NFKC 正规化，确保相同语义字符映射到相同 token

### Requirement: 字节对齐解码（防乱码）
系统 SHALL 在 `ByteTokenizer.decode` 中实现字节对齐检查，丢弃末尾不完整的多字节 UTF-8 序列，避免生成 U+FFFD 乱码字符。

#### Scenario: 截断字节安全解码
- **WHEN** 生成的 token 序列在多字节 UTF-8 字符中间结束（如中文 3 字节只生成 2 字节）
- **THEN** decode 丢弃末尾不完整字节，返回完整字符组成的字符串，不出现 U+FFFD

#### Scenario: BPE merge 字节边界
- **WHEN** BPE 训练生成 merge 规则
- **THEN** 所有 merge 产出的 token 对应的字节序列均为合法 UTF-8 字节边界，避免解码时跨字符截断

### Requirement: 自定义 Prompt 支持
系统 SHALL 在 `run.py` CLI 中支持用户输入自定义 prompt 进行生成测试。

#### Scenario: 命令行指定 prompt
- **WHEN** 用户执行 `python run.py --prompt "床前明月光，"`
- **THEN** 评估阶段使用该 prompt 生成文本，而非默认 5 条 prompt

#### Scenario: 从文件加载 prompt
- **WHEN** 用户执行 `python run.py --prompts-file my_prompts.txt`
- **THEN** 从文件按行读取 prompt 列表，逐条生成并输出

#### Scenario: 推理参数可配置
- **WHEN** 用户执行 `python run.py --temperature 0.8 --top-k 40 --max-tokens 50`
- **THEN** 生成使用指定的温度 / top-k / 最大 token 数，覆盖 config 默认值

### Requirement: 多套模型预设配置
系统 SHALL 提供 small / medium / large 三套预设配置，支持在 CPU 5GB 约束下训练 1-3M 参数模型。

#### Scenario: 配置切换
- **WHEN** 用户执行 `python run.py --config config/config_medium.yml`
- **THEN** 加载 medium 预设（约 1M 参数），训练与生成正常工作

#### Scenario: 参数量报告
- **WHEN** 模型构建完成
- **THEN** 自动打印参数量（如 `[model] parameters: 1,024,384`），方便用户评估

### Requirement: PyYAML 配置解析
系统 SHALL 使用 PyYAML 库解析配置文件，支持完整 YAML 语法（list / 多行字符串 / 引号转义），删除极简自实现解析器。

#### Scenario: 完整 YAML 支持
- **WHEN** 配置文件含 list（如 `prompts: ["a", "b"]`）或多行字符串
- **THEN** PyYAML 正确解析为对应 Python 类型

#### Scenario: 无 PyYAML 时降级
- **WHEN** 环境未安装 PyYAML
- **THEN** 打印 warning 并 fallback 到极简解析器（仅支持标量 + 两层嵌套），不崩溃

### Requirement: numba 可选加速
系统 SHALL 将 numba 列为可选依赖，在 `verse_nex` 热点函数（selective scan）上加 `@njit` 装饰器，安装 numba 后自动加速。

#### Scenario: 无 numba 时正常运行
- **WHEN** 环境未安装 numba
- **THEN** 热点函数走纯 Python 路径，功能正常但速度较慢

#### Scenario: 有 numba 时加速
- **WHEN** 环境安装了 numba
- **THEN** selective scan 等热点函数自动使用 JIT 编译，训练速度提升

## MODIFIED Requirements

### Requirement: hybrid 模式数值稳定性
`verse_nex` 的 Mamba-2 / RWKV-7 实现 SHALL 在 `seq_len >= 64` 时保持数值稳定，不出现 NaN / Inf。

#### Scenario: 长序列稳定
- **WHEN** 使用 hybrid arch 训练，`seq_len = 128`
- **THEN** forward 输出无 NaN / Inf，loss 正常下降

#### Scenario: log_decay 裁剪
- **WHEN** `np.exp(log_decay)` 计算时
- **THEN** `log_decay` 被 clip 到 `[-50, 0]`，避免 exp 溢出

### Requirement: run.py 端到端流程稳定性
`run.py` SHALL 在全流程（build → train → eval → visualize）中不出现 `maximum recursion depth exceeded` 错误。

#### Scenario: 评估阶段不崩溃
- **WHEN** 训练完成后进入评估阶段，调用 `model.generate`
- **THEN** 生成正常完成，不触发递归上限

#### Scenario: 错误处理友好
- **WHEN** 任意阶段发生异常
- **THEN** 打印友好的错误信息（含阶段名 + 异常类型 + 消息），`--verbose` 时打印完整 traceback

### Requirement: val_loss 曲线可视化
`plot_loss_curve` SHALL 在 matplotlib 和 ASCII 两种模式下都清晰显示 val_loss 曲线，不出现视觉丢失。

#### Scenario: matplotlib 模式清晰标记
- **WHEN** 使用 matplotlib 绘制 loss 曲线
- **THEN** val 曲线用显著标记（`marker='o', markersize=8, linewidth=2.5`），图例标注 "val (every N steps)"

#### Scenario: ASCII 模式独立行
- **WHEN** matplotlib 不可用，fallback 到 ASCII
- **THEN** val 点用独立符号（如 `V`）绘制，不被 train 曲线覆盖

#### Scenario: val 数据文件
- **WHEN** 训练完成
- **THEN** 额外生成 `val_losses.txt` 纯文本列表，方便用户直接查看

### Requirement: CLI 错误处理
`run.py` 和 `server.py` SHALL 提供友好的错误处理与完整的 CLI 参数。

#### Scenario: server.py 直接执行
- **WHEN** 用户执行 `python server.py`
- **THEN** 使用绝对导入，不出现 `ImportError: attempted relative import`

#### Scenario: run.py 完整参数
- **WHEN** 用户执行 `python run.py --help`
- **THEN** 显示所有参数（含 `--prompt` / `--prompts-file` / `--max-tokens` / `--temperature` / `--top-k` / `--arch` / `--verbose`）及示例

## REMOVED Requirements

### Requirement: 极简 YAML 解析器
**Reason**: 自实现解析器仅支持标量 + 两层嵌套，无法表达 list / 多行字符串，限制了配置表达能力，且易静默返回错误类型
**Migration**: 改用 PyYAML 库直接解析，无 PyYAML 时 fallback 到极简解析器并打印 warning（保留 fallback 代码作为降级路径，不彻底删除）
