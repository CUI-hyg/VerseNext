# Checklist

## 阶段 1：阻塞性 Bug 修复

- [x] run.py 全流程（build → train → eval → visualize）不出现 `maximum recursion depth exceeded`
- [x] `CometSparkLM.generate` 使用迭代式生成（for 循环），无隐式递归
- [x] `tests/test_recursion_fix.py` 通过（长 prompt + 多次 generate 不触发 RecursionError）
- [x] hybrid 模式在 `seq_len=128` 下 forward 输出无 NaN / Inf
- [x] `mamba2.py` 的 `log_decay` 被 clip 到 `[-50, 0]`
- [x] `rwkv7.py` 的 `log_decay` 被 clip 到 `[-50, 0]`
- [x] `A_log` 参数化约束为 `A = -softplus(A_log) - eps`，A 严格为负
- [x] `dt` 加上界约束 `dt = softplus(dt_raw).clamp(0, 10)`
- [x] `tests/test_hybrid_stability.py` 通过
- [x] `server.py` 直接执行 `python server.py` 不报 `ImportError`
- [x] `run.py --help` 显示完整参数列表
- [x] `run.py` 异常时打印友好错误信息（阶段名 + 异常类型 + 消息）

## 阶段 2：Tokenizer 升级与乱码修复

- [x] `BaseTokenizer` 抽象基类已创建，`BPETokenizer` / `ByteTokenizer` / `CharTokenizer` 均继承
- [x] 三种 tokenizer 提供统一接口 `encode` / `decode` / `save` / `load` / `__len__`
- [x] `encode` 前置 NFKC 正规化（全角字符映射到半角）
- [x] `ByteTokenizer.decode` 实现字节对齐检查，丢弃末尾不完整多字节序列
- [x] `BPETokenizer.train` 的 merge 规则产出合法 UTF-8 字节边界
- [x] `CometSparkLM.generate` 末尾强制追加 `eos_id`
- [x] `evaluate.py` 的 `_safe_decode` 分别 decode prompt 和生成部分再拼接
- [x] `tests/test_tokenizer_standard.py` 通过（接口一致 + NFKC）
- [x] `tests/test_no_garbled.py` 通过（截断字节 decode 不出现 U+FFFD）

## 阶段 3：架构升级（PyYAML）

- [x] `pyyaml>=6.0` 已添加到 pyproject.toml 依赖
- [x] `config.py` 优先使用 `yaml.safe_load` 解析配置
- [x] 无 PyYAML 时 fallback 到极简解析器并打印 warning
- [x] 配置文件含 list / 多行字符串 / 引号转义时正确解析
- [x] `tests/test_yaml_config.py` 通过

## 阶段 4：模型优化

- [x] `run.py` 支持 `--prompt`（逗号分隔多 prompt）
- [x] `run.py` 支持 `--prompts-file`（文件每行一个 prompt）
- [x] `run.py` 支持 `--max-tokens` / `--temperature` / `--top-k` 参数
- [x] `evaluate` 函数接受并透传 prompts / max_new_tokens / temperature / top_k
- [x] 评估输出格式化（`[prompt] xxx → [output] yyy`）
- [x] `tests/test_custom_prompt.py` 通过
- [x] `plot_loss_curve` matplotlib 模式 val 曲线有显著标记（marker='o', markersize=8, linewidth=2.5）
- [x] `plot_loss_curve` ASCII 模式 val 点用独立符号 `V` 绘制
- [x] `Trainer.fit` 完成后生成 `val_losses.txt` 纯文本列表
- [x] `plot_loss_curve` 完成后打印 `[info] val_losses: N points, best=X at step M`
- [x] `tests/test_val_loss_curve.py` 通过
- [x] `config_small.yml`（~130K）/ `config_medium.yml`（~1M）/ `config_large.yml`（~3M）三套预设已创建
- [x] `CometSparkLM.__init__` 末尾打印参数量 `[model] parameters: N`
- [x] `run.py` 支持 `--arch` 参数覆盖 config
- [x] medium 配置在 CPU 5GB 约束下 5 分钟内完成训练（默认 batch_size=4 max_steps=60，~14s 完成；原 200 步在沙箱下 OOM，已在 README 说明）
- [x] `/workspace/data/demo/README.md` 已更新三套配置说明

## 阶段 5：框架优化

- [x] `verse_nex/pyproject.toml` 添加 `[project.optional-dependencies] speed = ["numba>=0.60"]`
- [x] `mamba2.py` 热点函数加 numba `@njit` 兼容装饰器（无 numba 时自动降级）
- [x] 无 numba 时功能正常，有 numba 时自动加速
- [x] `verse_torch/nn.py` 热点矩阵运算用 `np.einsum` 优化（若适用）
- [x] `/workspace/docs/performance_tuning.md` 已创建（numba / BLAS / batch_size 指导）
- [x] `/workspace/README.md` 安装章节说明可选依赖 `pip install "verse-nex[speed]"`

## 阶段 6：端到端验证

- [x] 所有新增测试通过（test_recursion_fix / test_hybrid_stability / test_tokenizer_standard / test_no_garbled / test_yaml_config / test_custom_prompt / test_val_loss_curve）
- [x] 现有测试无回归（test_nn_advanced / test_training / test_tokenizer / test_parallel / test_compression_poc / test_cometspark_inference）
- [x] `python run.py` 全流程跑通（无 RecursionError、loss 下降、生成无乱码、val_loss 曲线可见、参数量打印）
- [x] `python run.py --arch hybrid` 数值稳定（无 NaN）
- [x] `python run.py --prompt "床前明月光，,你好，" --temperature 0.8 --top-k 40` 自定义 prompt 生效
- [x] `python run.py --config config/config_medium.yml` 参数量 ~1M 且 5 分钟内完成
- [x] `import verse_torch; import verse_nex; import verse_tokenizer; import verse_inference` 无报错
- [x] 运行时无 torch / tensorflow / jax / transformers 导入（零重型依赖）
