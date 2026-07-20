# Part3: 架构优化升级 + 模型优化 规范

## 目标

基于 Part1/Part2 已完成的 VerseTorch + VerseNex + VerseAWM + VerseTokenizer + CometSpark-v0.1 demo，进行架构与模型两方面的系统性优化。

参考项目：`/tmp/GPT_teacher`（GPT_teacher-3.37M-cn）

## 范围

### 一、架构优化升级

#### 1.1 升级 tokenizer（向先进模型看齐）

- 在 `verse_tokenizer` 包中新增 `StandardTokenizer`，统一包装：
  - HuggingFace `tokenizers` 库（若已安装）
  - 本仓库 `BPETokenizer`（已有）
  - `ByteTokenizer`（已有）
- 暴露统一属性：`bos_id` / `eos_id` / `pad_id` / `unk_id` / `vocab_size`，对齐 GPT_teacher 风格
- `encode(text, add_special_tokens=False) -> List[int]`：返回 id 列表，默认不附加 special token
- `decode(ids, errors="ignore") -> str`：默认 `errors="ignore"` 丢弃无效字节，避免乱码
- 升级 `load_tokenizer(kind, path)` 工厂，支持 `kind="standard"` 返回 `StandardTokenizer`

#### 1.2 解决乱码问题

- 在 `CometSparkLM.generate()` 中：
  - 屏蔽 `pad_id` / `bos_id` / `unk_id` 的 logit 设为 `-inf`
  - 支持 `min_tokens` 参数：在 step < min_tokens 时屏蔽 `eos_id`
  - 支持 `repetition_penalty`：对历史 token 的 logit 除以 penalty
  - 支持 `stop_strings`：在生成文本以 stop_string 结尾时停止
- 在 `model/tokenizer.py` 的 `_safe_decode` 中，强制使用 `errors="ignore"`
- 后处理 `_trim_leading_punct`：去除前导标点/空白

#### 1.3 删除极简 YAML 处理，改用 PyYAML

- `data/demo/model/config.py`：
  - 删除 `_parse_yaml` / `_dump_yaml` / `_parse_scalar` / `_format_scalar` ~100 行手写解析器
  - `load_full_config` 改为 `yaml.safe_load`
  - `save_full_config` 改为 `yaml.safe_dump`
  - `CometSparkConfig.from_yaml` / `to_yaml` 同样改用 PyYAML
  - 添加 fallback：若 PyYAML 不可用，保留极简解析器作为降级（保证向后兼容）
- 在 `verse_torch` 包的 `pyproject.toml` 中将 `pyyaml` 加入可选依赖 `extras`
- 在仓库根 `pyproject.toml` 中提示 `pip install pyyaml matplotlib` 即可获得完整体验

#### 1.4 解决 hybrid 模式下数值溢出

- `verse_nex/mamba2.py` 的 `forward_parallel`：
  - 当前问题：`log_decay = cs_i - cs_j`（cumsum 累积后做差），长序列下 `cs_i` 可能极负，`exp(log_decay)` 对远处位置 underflow 为 0，且 cumsum 数值范围过大可能产生 NaN
  - 修复方案：对 `log_decay` 进行 `np.clip(log_decay, -50.0, 0.0)` 后再 `np.exp`，确保：
    - 下界 -50：`exp(-50) ≈ 1.9e-22`，小于 float32 epsilon，等价于 0 但不产生 denormal
    - 上界 0：`exp(0) = 1`（对角线位置 i==j 的最大值）
  - 同时对 `L_data` 做 `np.nan_to_num` 兜底，防止极端情况下产生 NaN/Inf
  - 在 `_prepare_parallel` 与 `_prepare_recurrent` 中同步修复
- 验证：seq_len=128、256、512 时 forward_parallel 输出无 NaN/Inf

#### 1.5 解决部分 BUG 与错误

- `verse_torch/training.py` 的 `plot_loss_curve`：
  - val_x 对齐逻辑修复（详见 2.2）
  - ASCII fallback 中 val 与 train 使用一致的 x 轴范围
- `data/demo/train/evaluate.py` 的 `_safe_decode` 增加 `errors="ignore"` 默认
- `data/demo/run.py` 的 stage 失败处理：
  - build/train 失败不应静默吞掉，但应保证可视化仍能执行（如果有 loss_history）

### 二、模型优化

#### 2.1 支持用户输入自定义 Prompt

- `data/demo/run.py`：
  - 新增 CLI 参数：`--prompt <text>` / `--max-new-tokens <int>` / `--temperature <float>` / `--top-k <int>` / `--top-p <float>` / `--repetition-penalty <float>` / `--stop-strings <str...>` / `--min-tokens <int>`
  - 当指定 `--prompt` 时：跳过训练，直接加载模型并交互生成
  - 支持单次 prompt 或交互式循环（stdin）
- `data/demo/train/evaluate.py`：
  - `evaluate()` 接受 `prompts` 列表 + 完整生成参数（temperature/top_k/top_p/repetition_penalty/stop_strings/min_tokens）
  - 透传给 `CometSparkLM.generate()`

#### 2.2 解决可视化 Loss 中 val_loss 曲线丢失问题

- `verse_torch/training.py` 的 `plot_loss_curve`：
  - **根因 1**：`val_x = [i * eval_interval for i in range(len(val_losses))]`，但 `Trainer.fit()` 在 step=0 时也触发评估（`step % eval_interval == 0` 在 step=0 时为 True），导致 `val_x[0]=0` 与 `train_x[0]=0` 重合
  - **根因 2**：`if val_x and val_x[-1] >= len(train_losses): val_x = [min(x, len(train_losses) - 1) for x in val_x]` 强制 clamp，导致 val 曲线被压缩
  - **根因 3**：ASCII fallback 中 `put_curve(val_losses, max(n_train, n_val), "V")` 的 n_total 与 train 的 `n_train` 不一致，导致两条曲线 x 轴不一致
  - **修复**：
    - val_x 用 `step % eval_interval == 0` 触发评估的实际 step 值（即 `[i * eval_interval for i in range(len(val_losses))]`，但跳过 step=0 的首次评估）
    - 或更稳健：在 `Trainer.fit()` 中改为 `step > 0 and step % eval_interval == 0`，避免 step=0 触发评估
    - val_x 不再 clamp，而是使用实际 step 值
    - ASCII fallback 中两条曲线统一使用 `n_total = max(n_train, n_val)` 作为分母
- `verse_torch/training.py` 的 `Trainer.fit()`：
  - 评估条件改为 `step > 0 and step % self.eval_interval == 0`（避免 step=0 的无意义评估）

#### 2.3 解决 CLI 出现意外错误的问题

- `data/demo/run.py`：
  - 当 `--skip-train` 且无 checkpoint 时，给出明确错误提示并退出码 1（而非栈追溯）
  - 当 evaluate 阶段失败时，打印可读错误信息（而非原始异常）
  - 添加 `--verbose` 标志，默认情况下只打印 INFO 级别日志
- `data/demo/train/evaluate.py`：
  - 当 tokenizer 加载失败时，给出明确提示
  - 当模型文件未找到时，给出可读错误（已实现）
  - 单条 prompt 生成失败时 catch 并继续（已实现）

#### 2.4 优化模型架构，提升参数量

- `data/demo/config/config.yml`：
  - 默认配置升级：
    - `n_layer: 2 → 4`
    - `n_embd: 64 → 96`（保持 3 核 CPU 可承受）
    - `n_head: 4 → 6`（n_embd % n_head == 0）
    - `n_kv_head: 2 → 2`（GQA 3:1）
    - `seq_len: 64 → 96`
    - `max_steps: 200 → 300`
- 在 `CometSparkConfig` 中新增 `ffn_mult` 字段（默认 4），控制 FFN 中间维度
- 在 `verse_torch/nn.py` 的 `TransformerLM` 中支持 `ffn_mult` 参数

#### 2.5 优化框架，增强 VerseNext 的优化与依赖

- `packages/verse_nex/pyproject.toml`：
  - 添加 `optional-dependencies = { full = ["pyyaml", "matplotlib"] }`
- `packages/verse_torch/pyproject.toml`：
  - 添加 `optional-dependencies = { viz = ["matplotlib"], yaml = ["pyyaml"] }`
- `packages/verse_tokenizer/pyproject.toml`：
  - 添加 `optional-dependencies = { full = ["tokenizers>=0.15"] }`
- `README.md`：
  - 在「详细文档」章节新增「可选依赖」小节，说明 `pip install pyyaml matplotlib tokenizers`

## 验收标准

1. 所有现有测试不回归（`pytest tests/` 132 PASS）
2. 新增测试覆盖：
   - `tests/test_part3_tokenizer.py`：StandardTokenizer 接口测试
   - `tests/test_part3_yaml.py`：PyYAML 升级后的配置加载测试
   - `tests/test_part3_mamba2_stable.py`：长序列下 mamba2 forward_parallel 无 NaN
   - `tests/test_part3_generate.py`：自定义 prompt + repetition_penalty + stop_strings + min_tokens
   - `tests/test_part3_val_loss.py`：val_loss 曲线坐标对齐
3. `data/demo/run.py --skip-train --prompt "你好"` 端到端可运行
4. `data/demo/run.py`（全流程）能完成训练 + 评估 + 可视化
5. 输出文件包含完整 val_loss 曲线（PNG 或 TXT）

## 依赖

- 新增可选依赖：`pyyaml>=6.0`、`matplotlib>=3.5`、`tokenizers>=0.15`
- 无新增硬依赖：所有新功能在依赖缺失时降级到原行为
