# Tasks

## 阶段一：gigatoken 兼容组件修复

- [x] Task 1: 修复 `GigaTokenizerWrapper.__init__` 对 `.json` 文件路径的处理
  - [x] SubTask 1.1: 在 `giga.py` 的 `__init__` 中，对字符串参数判断：以 `.json` 结尾 + 文件存在 → 走"先构造默认实例再 `self.load(path)`"路径
  - [x] SubTask 1.2: 保留目录路径与 repo_id 字符串的原有处理逻辑（不破坏兼容）
  - [x] SubTask 1.3: 更新 `giga.py` 顶部 docstring，说明 `.json` 元信息文件加载行为
  - [x] SubTask 1.4: 验证 `GigaTokenizerWrapper("/path/to/tokenizer.json")` 不再抛 `Repo id must be in the form...`

- [x] Task 2: 修复 `load_tokenizer` 工厂函数的 `kind="giga"` 路径分发
  - [x] SubTask 2.1: 在 `bpe.py` 的 `load_tokenizer` 中，`kind="giga"` 分支按 path 类型分发：`.json` 文件 → `wrapper.load(path)`；目录 → 原构造；repo_id → 原构造；None → 默认
  - [x] SubTask 2.2: 保留 `ImportError` 时降级到 `VerseTokenizer` 的逻辑
  - [x] SubTask 2.3: 更新 `load_tokenizer` docstring，说明 path 的三种形式
  - [x] SubTask 2.4: 验证 `load_tokenizer(kind="giga", path="/path/to/tokenizer.json")` 正常工作

## 阶段二：small 模型训练数据优化

- [x] Task 3: 新建 `spark/small/data/generate_train_data.py` 脚本
  - [x] SubTask 3.1: 创建 `spark/small/data/` 目录
  - [x] SubTask 3.2: 实现 8 大类数据生成器：问答 / 翻译 / 代码 / 数学 / 对话 / 续写 / 指令 / 知识
  - [x] SubTask 3.3: 每类用模板 + 词表组合生成，保证多样性（每类 5000 条，共 40000 条）
  - [x] SubTask 3.4: 生成 `train.jsonl`（40000 条）+ `val.jsonl`（500 条，从训练集抽样）
  - [x] SubTask 3.5: 每条格式为 `{"prompt":"...", "completion":"..."}`
  - [x] SubTask 3.6: 添加重复率检查（相同 prompt+completion 组合 < 1%）
  - [x] SubTask 3.7: 脚本支持 `--num-train` / `--num-val` 参数（默认 40000 / 500）

- [x] Task 4: 运行脚本生成实际数据文件
  - [x] SubTask 4.1: 执行 `python spark/small/data/generate_train_data.py` 生成 train.jsonl + val.jsonl
  - [x] SubTask 4.2: 验证 train.jsonl 行数 = 40000，val.jsonl 行数 = 500
  - [x] SubTask 4.3: 抽样 10 条验证格式正确 + 内容多样性

- [x] Task 5: 修改 `_auto_generate_test_data` 默认格式
  - [x] SubTask 5.1: `trainer.py` 中 `_auto_generate_test_data` 改为生成 `{"prompt":"", "completion":""}` 格式
  - [x] SubTask 5.2: 扩充 `_TEST_TEXTS` 为 prompt-completion 对（不再用纯文本）
  - [x] SubTask 5.3: 验证 `CachedDataset` 能解析新格式（同时兼容旧 `{"text":"..."}`）

- [x] Task 6: 修改 `cometspark_small.yml` 数据路径
  - [x] SubTask 6.1: `data.train_path` 改为 `../data/train.jsonl`（相对 config 目录）
  - [x] SubTask 6.2: `data.val_path` 改为 `../data/val.jsonl`
  - [x] SubTask 6.3: 验证 `_resolve_path(base_dir, "../data/train.jsonl")` 解析到 `spark/small/data/train.jsonl`

## 阶段三：交互式 tokenizer 初始化

- [x] Task 7: 实现 `_prompt_tokenizer_action` 交互函数
  - [x] SubTask 7.1: 在 `trainer.py` 中新增 `_prompt_tokenizer_action(tok_cfg, model_cfg, save_dir, base_dir) -> tokenizer`
  - [x] SubTask 7.2: 实现 TTY 检测（`sys.stdin.isatty()`）
  - [x] SubTask 7.3: TTY 时打印选项界面："复制后输 y, 自行构建输 n"
  - [x] SubTask 7.4: `y` 路径：从 `tok_cfg.tokenizer_repo` 或 `model_cfg.tokenizer_repo` / `from_hf` 加载并 save 到 `save_dir/tokenizer.json`
  - [x] SubTask 7.5: `n` 路径：调用 `_auto_build_tokenizer`
  - [x] SubTask 7.6: 非 TTY 时打印警告 + 默认走 `n` 路径（不阻塞）
  - [x] SubTask 7.7: 输入异常（空 / 非 y/n）时重新提示，最多 3 次后默认 `n`

- [x] Task 8: 在 `_load_tokenizer` 中接入交互分支
  - [x] SubTask 8.1: 修改 `_load_tokenizer` 签名，增加 `model_cfg` 参数（用于读取 `tokenizer_repo`）
  - [x] SubTask 8.2: 在 step 4（其他 kind 自动构建）前插入交互分支
  - [x] SubTask 8.3: 更新 `train()` 中调用 `_load_tokenizer` 的地方，传入 `model_cfg`
  - [x] SubTask 8.4: 更新 `cli.py` / `evaluate.py` 中所有 `_load_tokenizer` 调用点
  - [x] SubTask 8.5: 验证 `train / finetune / posttrain / continue` 均经过交互分支

## 阶段四：UPDATE.MD 补充项目结构分析

- [x] Task 9: 补充 verse_infra 审计
  - [x] SubTask 9.1: 新增"七、verse_infra 审计"章节
  - [x] SubTask 9.2: 记录 giga.py `.json` 路径处理问题（本次修复，标记 ✅）
  - [x] SubTask 9.3: 记录 `bpe.py load_tokenizer` 路径分发缺失（本次修复，标记 ✅）
  - [x] SubTask 9.4: 记录 `trainer.py _load_tokenizer_for_config` 与 `_load_tokenizer` 重复逻辑（中危）
  - [x] SubTask 9.5: 记录 `_auto_generate_test_data` 格式陈旧（本次修复，标记 ✅）
  - [x] SubTask 9.6: 记录 `trainer.py _auto_build_tokenizer` 不支持 giga/verse（中危）

- [x] Task 10: 补充 spark/run.py 审计
  - [x] SubTask 10.1: 新增"八、spark/run.py 审计"章节
  - [x] SubTask 10.2: 记录 `_load_tokenizer_for_config` 与 `trainer._load_tokenizer` 重复（中危）
  - [x] SubTask 10.3: 记录 `cmd_finetune / cmd_posttrain` 未做 tokenizer.json 检查（中危，本次通过 `_load_tokenizer` 统一修复）
  - [x] SubTask 10.4: 记录 `small/config` 中 `tokenizer_repo` 放在 `model` 段而非 `tokenizer` 段（低危，命名不一致）

- [x] Task 11: 更新 UPDATE.MD 摘要表
  - [x] SubTask 11.1: 更新摘要表：中危 +4，低危 +0，合计 60→64
  - [x] SubTask 11.2: 更新"按文件分布"表：增加 verse_infra / spark/run.py 行
  - [x] SubTask 11.3: 更新"按类型分布"表（Bug 38→39，冲突 15→16，蜘蛛丝 20→22）
  - [x] SubTask 11.4: 更新 P1/P2/P3 修复优先级建议

## 阶段五：测试与验证

- [x] Task 12: 编写/更新测试用例
  - [x] SubTask 12.1: 在 `tests/test_giga_tokenizer.py` 增加 `TestGigaJsonMetaLoading` 类（4 个测试）
  - [x] SubTask 12.2: 创建 `tests/test_train_data.py` 验证生成的 train.jsonl 格式 + 行数（9 个测试）
  - [x] SubTask 12.3: 创建 `tests/test_trainer_tokenizer_init.py` 验证交互式初始化（15 个测试）

- [x] Task 13: 运行核心测试套件
  - [x] SubTask 13.1: `python -m pytest tests/test_giga_tokenizer.py tests/test_train_data.py tests/test_trainer_tokenizer_init.py tests/test_mod_complete.py tests/test_nexrl.py tests/test_spark_run.py tests/test_checkpoint_vn.py -x -q` → 242 passed, 2 skipped
  - [x] SubTask 13.2: 验证 small 模型训练 dry-run 能正常解析新数据路径
  - [x] SubTask 13.3: 验证非 TTY 环境下 `_load_tokenizer` 不阻塞（CI=true）

# Task Dependencies

- Task 2 依赖 Task 1（工厂函数复用 wrapper 的 .json 加载能力）
- Task 4 依赖 Task 3（先生成脚本再运行）
- Task 5 独立，可与 Task 3 并行
- Task 6 依赖 Task 4（数据文件存在后才能配置路径）
- Task 8 依赖 Task 7（先实现交互函数再接入）
- Task 9-11 独立，可与 Task 1-8 并行（基于代码分析，不依赖运行时）
- Task 12 依赖 Task 1-8 完成
- Task 13 依赖 Task 12
