# Part3 任务分解

## 阶段 1: 架构修复（无相互依赖，可并行）

### Task 3.1 升级 tokenizer（StandardTokenizer）

**文件**：
- `packages/verse_tokenizer/verse_tokenizer/standard.py`（新建）
- `packages/verse_tokenizer/verse_tokenizer/__init__.py`（导出 StandardTokenizer）
- `packages/verse_tokenizer/verse_tokenizer/bpe.py`（更新 load_tokenizer 支持 "standard"）

**要点**：
- StandardTokenizer 包装三种后端（hf_tokenizers / BPETokenizer / ByteTokenizer）
- 统一属性：bos_id / eos_id / pad_id / unk_id / vocab_size
- encode(text, add_special_tokens=False) -> List[int]
- decode(ids, errors="ignore") -> str
- 参考 `/tmp/GPT_teacher/src/tokenizer.py` 的接口风格

### Task 3.2 修复 hybrid 模式下 mamba2 数值溢出

**文件**：
- `packages/verse_nex/verse_nex/mamba2.py`

**要点**：
- `forward_parallel` 第 378-381 行：`log_decay = cs_i - cs_j`，`L_data = np.exp(log_decay) * mask`
- 修复：`log_decay = np.clip(log_decay, -50.0, 0.0)`，然后 `L_data = np.exp(log_decay) * mask`
- 添加 `np.nan_to_num(L_data, nan=0.0, posinf=1.0, neginf=0.0)` 兜底
- 验证：seq_len=128/256/512 forward_parallel 输出无 NaN

### Task 3.3 删除极简 YAML 处理，改用 PyYAML

**文件**：
- `data/demo/model/config.py`

**要点**：
- 删除 `_parse_yaml` / `_dump_yaml` / `_parse_scalar` / `_format_scalar`
- `load_full_config` 改为：先尝试 `import yaml; yaml.safe_load`，失败时回退到极简解析器（保留为内部 _fallback_parse_yaml）
- `CometSparkConfig.from_yaml` / `to_yaml` 同样改用 PyYAML
- 更新 `config/config.yml` 顶部注释（删除「本环境无 PyYAML」说明）

### Task 3.4 修复 val_loss 曲线丢失

**文件**：
- `packages/verse_torch/verse_torch/training.py`

**要点**：
- `Trainer.fit()` 第 592 行：`if self.eval_interval > 0 and step % self.eval_interval == 0` → 改为 `if self.eval_interval > 0 and step > 0 and step % self.eval_interval == 0`
- `plot_loss_curve` 中 val_x 计算：
  - 删除 `if val_x and val_x[-1] >= len(train_losses): val_x = [min(x, len(train_losses) - 1) for x in val_x]` 的 clamp
  - val_x 直接用 `[i * eval_interval for i in range(len(val_losses))]`
- `_plot_ascii` 中 `put_curve` 调用统一使用 `n_total = max(n_train, n_val)`：
  - `put_curve(train_losses, max(n_train, n_val), "T")`
  - `put_curve(val_losses, max(n_train, n_val), "V")`

## 阶段 2: 模型优化（依赖阶段 1 部分完成）

### Task 3.5 支持自定义 Prompt + 生成参数

**文件**：
- `data/demo/run.py`（新增 CLI 参数 + 交互模式）
- `data/demo/train/evaluate.py`（透传生成参数）
- `data/demo/model/model.py`（CometSparkLM.generate 支持 repetition_penalty / stop_strings / min_tokens + 特殊 token 屏蔽）

**要点**：
- `CometSparkLM.generate()` 新增参数：`repetition_penalty=1.0` / `stop_strings=None` / `min_tokens=0` / `top_p=None` / `pad_id=None` / `bos_id=None` / `eos_id=None` / `unk_id=None`
- 生成时屏蔽 pad/bos/unk 的 logit 设为 -inf
- step < min_tokens 时屏蔽 eos_id
- repetition_penalty > 1.0 时对最近 32 个 token 的 logit 除以 penalty
- stop_strings：生成文本以任意 stop_string 结尾时停止
- `_generate_with_logits` 同步实现上述逻辑
- 参考 `/tmp/GPT_teacher/src/infer.py` 的 `generate()` 函数

### Task 3.6 优化模型架构，提升参数量

**文件**：
- `data/demo/config/config.yml`（升级默认配置）
- `data/demo/model/config.py`（新增 ffn_mult 字段）
- `packages/verse_torch/verse_torch/nn.py`（TransformerLM 支持 ffn_mult）

**要点**：
- config.yml：
  - n_layer: 2 → 4
  - n_embd: 64 → 96
  - n_head: 4 → 6
  - n_kv_head: 2 → 2（GQA 3:1）
  - seq_len: 64 → 96
  - max_steps: 200 → 300
- CometSparkConfig 新增 `ffn_mult: int = 4` 字段
- TransformerLM 支持 `ffn_mult` 参数，FFN 中间维度 = `n_embd * ffn_mult`

### Task 3.7 CLI 错误处理优化

**文件**：
- `data/demo/run.py`

**要点**：
- `--skip-train` 且无 checkpoint 时：明确错误提示并退出码 1
- evaluate 失败时打印可读错误（非原始 traceback）
- 添加 `--verbose` 标志（默认 False，仅打印 INFO；True 时打印 DEBUG）
- 单条 prompt 生成失败 catch 并继续（已在 evaluate.py 实现）

### Task 3.8 框架依赖优化

**文件**：
- `packages/verse_torch/pyproject.toml`
- `packages/verse_nex/pyproject.toml`
- `packages/verse_tokenizer/pyproject.toml`
- `README.md`

**要点**：
- verse_torch：新增 `optional-dependencies = { viz = ["matplotlib>=3.5"], yaml = ["pyyaml>=6.0"] }`
- verse_nex：新增 `optional-dependencies = { full = ["pyyaml>=6.0", "matplotlib>=3.5"] }`
- verse_tokenizer：新增 `optional-dependencies = { full = ["tokenizers>=0.15"] }`
- README：新增「可选依赖」小节

## 阶段 3: 测试与端到端验证

### Task 3.9 新增测试

**文件**：
- `tests/test_part3_tokenizer.py`：StandardTokenizer 接口
- `tests/test_part3_yaml.py`：PyYAML 配置加载
- `tests/test_part3_mamba2_stable.py`：长序列 mamba2 无 NaN
- `tests/test_part3_generate.py`：自定义 prompt + 生成参数
- `tests/test_part3_val_loss.py`：val_loss 曲线坐标

### Task 3.10 端到端验证

- `python -m pytest tests/ -v` 全部 PASS
- `python data/demo/run.py --skip-train --prompt "你好"` 可运行
- `python data/demo/run.py`（全流程）能完成训练 + 评估 + 可视化
- val_loss 曲线在 loss_curve.png 中清晰可见

## 并行执行计划

| 阶段 | 任务 | 并行组 |
|------|------|--------|
| 1 | 3.1 tokenizer 升级 | A |
| 1 | 3.2 mamba2 数值溢出 | A |
| 1 | 3.3 PyYAML 升级 | A |
| 1 | 3.4 val_loss 曲线修复 | A |
| 2 | 3.5 自定义 Prompt + 生成参数 | B（依赖 3.1） |
| 2 | 3.6 模型架构优化 | B |
| 2 | 3.7 CLI 错误处理 | B |
| 2 | 3.8 框架依赖优化 | B |
| 3 | 3.9 新增测试 | C（依赖 1+2） |
| 3 | 3.10 端到端验证 | C（依赖 3.9） |
