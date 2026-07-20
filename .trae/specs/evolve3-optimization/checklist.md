# Part3 验收检查清单

## 阶段 1: 架构修复

### Task 3.1 升级 tokenizer

- [ ] 新建 `packages/verse_tokenizer/verse_tokenizer/standard.py`
- [ ] 实现 `StandardTokenizer` 类，包装 hf_tokenizers / BPETokenizer / ByteTokenizer
- [ ] 暴露统一属性：bos_id / eos_id / pad_id / unk_id / vocab_size
- [ ] `encode(text, add_special_tokens=False) -> List[int]`
- [ ] `decode(ids, errors="ignore") -> str` 默认 errors="ignore"
- [ ] `__init__.py` 导出 `StandardTokenizer`
- [ ] `load_tokenizer` 工厂支持 `kind="standard"`
- [ ] `tests/test_part3_tokenizer.py` 全部 PASS

### Task 3.2 修复 hybrid 模式下 mamba2 数值溢出

- [ ] `mamba2.py` 的 `forward_parallel` 中 `log_decay` 添加 `np.clip(-50, 0)`
- [ ] `L_data` 添加 `np.nan_to_num` 兜底
- [ ] seq_len=128 时 forward_parallel 输出无 NaN/Inf
- [ ] seq_len=256 时 forward_parallel 输出无 NaN/Inf
- [ ] seq_len=512 时 forward_parallel 输出无 NaN/Inf
- [ ] `tests/test_part3_mamba2_stable.py` 全部 PASS

### Task 3.3 删除极简 YAML 处理，改用 PyYAML

- [ ] `data/demo/model/config.py` 删除 `_parse_yaml` / `_dump_yaml` / `_parse_scalar` / `_format_scalar`（或保留为 _fallback）
- [ ] `load_full_config` 优先使用 `yaml.safe_load`
- [ ] `save_full_config` 优先使用 `yaml.safe_dump`
- [ ] `CometSparkConfig.from_yaml` / `to_yaml` 改用 PyYAML
- [ ] PyYAML 不可用时降级到原极简解析器（向后兼容）
- [ ] `config/config.yml` 注释更新
- [ ] `tests/test_part3_yaml.py` 全部 PASS

### Task 3.4 修复 val_loss 曲线丢失

- [ ] `Trainer.fit()` 评估条件改为 `step > 0 and step % self.eval_interval == 0`
- [ ] `plot_loss_curve` 中 val_x 不再 clamp
- [ ] `_plot_ascii` 中 train/val 统一使用 `n_total = max(n_train, n_val)`
- [ ] `tests/test_part3_val_loss.py` 全部 PASS

## 阶段 2: 模型优化

### Task 3.5 支持自定义 Prompt + 生成参数

- [ ] `CometSparkLM.generate()` 新增参数：repetition_penalty / stop_strings / min_tokens / top_p
- [ ] 生成时屏蔽 pad/bos/unk 的 logit 设为 -inf
- [ ] step < min_tokens 时屏蔽 eos_id
- [ ] repetition_penalty > 1.0 时对历史 token 的 logit 除以 penalty
- [ ] stop_strings 命中时停止生成
- [ ] `_generate_with_logits` 同步实现
- [ ] `data/demo/run.py` 新增 CLI 参数：--prompt / --max-new-tokens / --temperature / --top-k / --top-p / --repetition-penalty / --stop-strings / --min-tokens
- [ ] `--prompt` 指定时跳过训练，直接交互生成
- [ ] `data/demo/train/evaluate.py` 透传所有生成参数
- [ ] `tests/test_part3_generate.py` 全部 PASS

### Task 3.6 优化模型架构，提升参数量

- [ ] `config/config.yml`：n_layer=4, n_embd=96, n_head=6, n_kv_head=2, seq_len=96, max_steps=300
- [ ] `CometSparkConfig` 新增 `ffn_mult: int = 4` 字段
- [ ] `TransformerLM` 支持 `ffn_mult` 参数
- [ ] 现有 test_nn_advanced.py 仍 PASS
- [ ] 端到端训练能在 3 核 CPU 上完成（< 5 分钟）

### Task 3.7 CLI 错误处理优化

- [ ] `--skip-train` 且无 checkpoint 时给出明确错误提示
- [ ] evaluate 失败时打印可读错误（非原始 traceback）
- [ ] 新增 `--verbose` 标志
- [ ] `data/demo/run.py --skip-train --prompt "你好"` 可运行

### Task 3.8 框架依赖优化

- [ ] `packages/verse_torch/pyproject.toml` 新增 optional-dependencies
- [ ] `packages/verse_nex/pyproject.toml` 新增 optional-dependencies
- [ ] `packages/verse_tokenizer/pyproject.toml` 新增 optional-dependencies
- [ ] `README.md` 新增「可选依赖」小节

## 阶段 3: 测试与端到端验证

### Task 3.9 新增测试

- [ ] `tests/test_part3_tokenizer.py`
- [ ] `tests/test_part3_yaml.py`
- [ ] `tests/test_part3_mamba2_stable.py`
- [ ] `tests/test_part3_generate.py`
- [ ] `tests/test_part3_val_loss.py`

### Task 3.10 端到端验证

- [ ] `python -m pytest tests/ -v` 全部 PASS（不回归）
- [ ] `python data/demo/run.py --skip-train --prompt "你好"` 可运行
- [ ] `python data/demo/run.py`（全流程）能完成训练 + 评估 + 可视化
- [ ] val_loss 曲线在 loss_curve.png 中清晰可见
- [ ] 长序列（seq_len>=128）hybrid 模式无 NaN

## 总验收

- [ ] 10 项任务全部完成
- [ ] 现有 132 个测试不回归
- [ ] 5 个新测试文件全部 PASS
- [ ] 端到端流程跑通
- [ ] 乱码问题修复（生成结果无乱码）
- [ ] val_loss 曲线正常显示
- [ ] CLI 自定义 prompt 可用
- [ ] 模型参数量提升
