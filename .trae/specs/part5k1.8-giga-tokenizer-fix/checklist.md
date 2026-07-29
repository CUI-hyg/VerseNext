# Checklist

## 阶段一：gigatoken 兼容组件修复

- [x] `giga.py` `GigaTokenizerWrapper.__init__` 对 `.json` 文件路径走 `self.load(path)`，不再走 `AutoTokenizer.from_pretrained`
- [x] `giga.py` 保留目录路径与 repo_id 字符串的原有处理逻辑
- [x] `giga.py` 顶部 docstring 说明 `.json` 元信息文件加载行为
- [x] `GigaTokenizerWrapper("/path/to/tokenizer.json")` 不再抛 `Repo id must be in the form...`
- [x] `bpe.py` `load_tokenizer(kind="giga", path=...)` 按 path 类型分发（.json 文件 / 目录 / repo_id / None）
- [x] `bpe.py` 保留 `ImportError` 时降级到 `VerseTokenizer` 的逻辑
- [x] `bpe.py` `load_tokenizer` docstring 说明 path 的三种形式

## 阶段二：small 模型训练数据优化

- [x] `spark/small/data/generate_train_data.py` 脚本存在
- [x] 脚本实现 8 大类数据生成器（问答 / 翻译 / 代码 / 数学 / 对话 / 续写 / 指令 / 知识）
- [x] 每条格式为 `{"prompt":"...", "completion":"..."}`
- [x] 脚本支持 `--num-train` / `--num-val` 参数（默认 40000 / 500）
- [x] `spark/small/data/train.jsonl` 存在且行数 = 40000
- [x] `spark/small/data/val.jsonl` 存在且行数 = 500
- [x] 重复率 < 1%（相同 prompt+completion 组合，实测 0.000%）
- [x] `trainer.py` `_auto_generate_test_data` 默认生成 `{"prompt":"", "completion":""}` 格式
- [x] `CachedDataset` 能解析新格式（同时兼容旧 `{"text":"..."}`）
- [x] `cometspark_small.yml` `data.train_path` 指向 `../data/train.jsonl`
- [x] `cometspark_small.yml` `data.val_path` 指向 `../data/val.jsonl`
- [x] `_resolve_path(base_dir, "../data/train.jsonl")` 解析到 `spark/small/data/train.jsonl`

## 阶段三：交互式 tokenizer 初始化

- [x] `trainer.py` 新增 `_prompt_tokenizer_action(tok_cfg, model_cfg, save_dir, base_dir)` 函数
- [x] TTY 检测使用 `sys.stdin.isatty()`
- [x] TTY 时打印选项界面："复制后输 y, 自行构建输 n"
- [x] `y` 路径：从 `tokenizer_repo` / `from_hf` 加载并 save 到 `save_dir/tokenizer.json`
- [x] `n` 路径：调用 `_auto_build_tokenizer`
- [x] 非 TTY 时打印警告 + 默认走 `n` 路径（不阻塞）
- [x] 输入异常时重新提示，最多 3 次后默认 `n`
- [x] `_load_tokenizer` 签名增加 `model_cfg` 参数（默认 None 向后兼容）
- [x] `_load_tokenizer` 在 step 4 前插入交互分支
- [x] `train()` 调用 `_load_tokenizer` 时传入 `model_cfg`
- [x] `cli.py` / `evaluate.py` 中所有 `_load_tokenizer` 调用点已更新
- [x] `train / finetune / posttrain / continue` 均经过交互分支

## 阶段四：UPDATE.MD 补充项目结构分析

- [x] `UPDATE.MD` 新增"七、verse_infra 审计"章节
- [x] 记录 giga.py `.json` 路径处理问题（标记 ✅ 已修复）
- [x] 记录 `bpe.py load_tokenizer` 路径分发缺失（标记 ✅ 已修复）
- [x] 记录 `trainer.py _load_tokenizer_for_config` 与 `_load_tokenizer` 重复逻辑（中危）
- [x] 记录 `_auto_generate_test_data` 格式陈旧（标记 ✅ 已修复）
- [x] 记录 `trainer.py _auto_build_tokenizer` 不支持 giga/verse（中危）
- [x] `UPDATE.MD` 新增"八、spark/run.py 审计"章节
- [x] 记录 `_load_tokenizer_for_config` 与 `trainer._load_tokenizer` 重复（中危）
- [x] 记录 `cmd_finetune / cmd_posttrain` 未做 tokenizer.json 检查（中危，本次统一修复）
- [x] 记录 `small/config` 中 `tokenizer_repo` 放在 `model` 段而非 `tokenizer` 段（低危）
- [x] 摘要表更新：中危 35→39，低危 25，合计 60→64
- [x] "按文件分布"表增加 verse_infra / spark/run.py 行
- [x] "按类型分布"表更新（Bug 38→39，冲突 15→16，蜘蛛丝 20→22）
- [x] P1/P2/P3 修复优先级建议更新

## 阶段五：测试与验证

- [x] `tests/test_giga_tokenizer.py` 增加 `TestGigaJsonMetaLoading` 类（4 个测试）
- [x] `tests/test_train_data.py` 验证生成的 train.jsonl 格式 + 行数（9 个测试）
- [x] `tests/test_trainer_tokenizer_init.py` 增加交互式初始化测试（15 个测试）
- [x] `python -m pytest tests/test_giga_tokenizer.py tests/test_train_data.py tests/test_trainer_tokenizer_init.py tests/test_mod_complete.py tests/test_nexrl.py tests/test_spark_run.py tests/test_checkpoint_vn.py -x -q` 通过（242 passed, 2 skipped）
- [x] small 模型训练 dry-run 能正常解析新数据路径
- [x] 非 TTY 环境下 `_load_tokenizer` 不阻塞（CI=true 验证）
