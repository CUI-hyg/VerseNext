# Part4K2：架构升级与模型优化

## 概述
在 Part4K1 基础设施全面升级的基础上，Part4K2 聚焦于：
1. 训练资源利用效率（CPU/GPU/NPU）
2. 内存优化（智能分区训练）
3. Tokenizer jinja2 聊天模板规范
4. 生成输出优化（不限制 token 数）
5. 压缩技术 V1.3（以小博大）
6. .vn 文件格式（safetensors 性能优化版）
7. 持续训练 + 数据集下载器

## 关键决策（已与用户确认）

| 决策项 | 选择 |
|--------|------|
| 聊天模板 | ChatML (Qwen 风格) + jinja2 引擎 |
| 工具调用 | 完整支持，Qwen3 官方格式 |
| .vn 格式 | 基于 safetensors 的性能优化版，兼容 .pt 互转 |
| 分区训练 | 按 layer 自动拆分 + 硬盘卸载 |
| 1zB 含义 | 以小博大（压缩后小模型等效大模型能力） |
| 输出限制 | 移除默认限制，EOS 自然停止 + 安全上限 100K |
| 数据下载器 | 任意 URL + HF datasets + .npz 缓存 + 断点续传 |

## 任务分解（10 个任务）

### Task 1: .vn 文件格式 + safetensors 性能优化
- 基于 safetensors 的 .vn 格式（ZIP 容器：safetensors 权重 + config.yml + chat_template.jinja + tokenizer.json）
- .pt ↔ .vn 互转工具
- 高效读写（mmap 零拷贝 + 分片加载）
- 智能压缩存储（量化参数内联标记）

### Task 2: Verse_Tokenizer jinja2 + ChatML + 工具调用
- jinja2 引擎集成（动态模板渲染）
- ChatML 模板（Qwen 风格，`<|im_start|>role\ncontent<|im_end|>`）
- 工具调用支持（Qwen3 官方格式：tool 角色 + assistant JSON 工具调用 + tool 返回）
- tokenizer.json 内嵌 chat_template.jinja 字段
- apply_chat_template 全量升级

### Task 3: 生成输出优化（不限制 token 数）
- 移除 max_new_tokens 默认值（CometSpark + VerseTrainer）
- EOS 自然停止 + 安全上限 100K
- jinja 模板完整输出（按模板格式生成完整内容）
- StreamingGenerator 适配

### Task 4: 智能分区训练（LayerWiseTrainer）
- 按 transformer layer 自动拆分
- 训完一组卸载到硬盘 .vn 分片
- 统一实体（对外表现为完整模型训练）
- 内存监控 + 自动卸载/加载

### Task 5: CPU/GPU/NPU 资源利用优化
- autocast 完善（GPU 混合精度训练一致性）
- GPU 显存管理（empty_cache + 梯度累积 + activation checkpointing）
- CPU BLAS 线程优化
- 设备亲和性（数据预取 + pin_memory）
- NPU 后端完善

### Task 6: 压缩技术 V1.3（以小博大）
- 集成到 VerseNex & VerseTorch
- 知识蒸馏增强（大模型→小模型能力转移）
- 量化+剪枝+蒸馏组合优化
- 吞吐率优化
- compress_pipeline V1.3

### Task 7: VerseTrainer tqdm + 持续训练 CLI + 1B 模型优化
- 并行训练 tqdm 支持（统一进度条）
- 简化输出（重点突出数据，不杂乱）
- 持续训练 CLI（verse-continue 命令）
- 1B 模型 CPU/GPU 亲和优化
- 并行训练进一步支持

### Task 8: 数据集下载器
- data/ 下 downloader.py
- 任意 URL 下载 + HF datasets 支持
- 断点续传 + 多线程
- 自动转 .npz 缓存
- CLI: verse-download

### Task 9: 文档更新
- README + ADR + training_guide + perf_tuning
- 压缩技术文档
- .vn 格式文档

### Task 10: 测试 + 验收
- 全量测试
- audit_report 更新

## 依赖关系
- Task 1（.vn 格式）→ Task 4（分区训练卸载用 .vn 分片）
- Task 2（tokenizer）→ Task 3（生成输出用 jinja 模板）
- Task 5（资源优化）独立
- Task 6（压缩）最好在 Task 1 之后
- Task 7 依赖 Task 2/3
- Task 8（下载器）独立
- Task 9/10 依赖所有

## 并行策略
- 第一批：Task 1 + Task 2 + Task 5 + Task 8（独立任务）
- 第二批：Task 3 + Task 4 + Task 6
- 第三批：Task 7
- 第四批：Task 9 + Task 10
