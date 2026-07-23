# Part4K2 任务清单

## Task 1: .vn 文件格式 + safetensors 性能优化
- [x] 1.1 创建 verse_torch/vn_format.py：VNFileReader/VNFileWriter
- [x] 1.2 .vn 格式定义（ZIP 容器：safetensors 权重 + config.yml + chat_template.jinja + tokenizer.json + meta.json）
- [x] 1.3 .pt → .vn 转换函数（pt_to_vn）
- [x] 1.4 .vn → .pt 转换函数（vn_to_pt）
- [x] 1.5 mmap 零拷贝读取支持
- [x] 1.6 智能压缩存储（量化参数内联标记，读写时自动量化/反量化）
- [x] 1.7 CometSparkV05LM.save_vn / load_vn 方法
- [x] 1.8 CLI: verse-convert 命令（.pt ↔ .vn 互转）
- [x] 1.9 测试

## Task 2: Verse_Tokenizer jinja2 + ChatML + 工具调用
- [x] 2.1 jinja2 引擎集成（可选依赖，缺失时降级为字符串拼接）
- [x] 2.2 ChatML 模板定义（Qwen 风格 jinja2 模板字符串）
- [x] 2.3 工具调用模板（Qwen3 官方格式：tool 角色 + assistant JSON 工具调用 + tool 返回）
- [x] 2.4 tokenizer.json 内嵌 chat_template.jinja 字段
- [x] 2.5 apply_chat_template 全量升级（支持 tools 参数 + add_generation_prompt）
- [x] 2.6 BaseTokenizer/BPETokenizer/VerseTokenizer 统一升级
- [x] 2.7 测试

## Task 3: 生成输出优化（不限制 token 数）
- [x] 3.1 CometSparkV05LM.generate 移除 max_new_tokens 默认值，改为 EOS 自然停止 + 安全上限 100K
- [x] 3.2 CometSparkNexLM.generate 同步修改
- [x] 3.3 StreamingGenerator 适配（移除默认 max_tokens 限制）
- [x] 3.4 VerseTrainer 生成调用适配
- [x] 3.5 verse-eval CLI 适配
- [x] 3.6 测试

## Task 4: 智能分区训练（LayerWiseTrainer）
- [x] 4.1 LayerWiseTrainer 类（按 layer 自动拆分）
- [x] 4.2 训完一组卸载到硬盘 .vn 分片
- [x] 4.3 统一实体（对外表现为完整模型训练）
- [x] 4.4 内存监控 + 自动卸载/加载
- [x] 4.5 CLI: --partition-training 选项
- [x] 4.6 测试

## Task 5: CPU/GPU/NPU 资源利用优化
- [x] 5.1 autocast 完善（GPU 混合精度训练一致性）
- [x] 5.2 GPU 显存管理（empty_cache + 梯度累积 + activation checkpointing）
- [x] 5.3 CPU BLAS 线程优化
- [x] 5.4 设备亲和性（数据预取 + pin_memory 实装）
- [x] 5.5 NPU 后端完善
- [x] 5.6 测试

## Task 6: 压缩技术 V1.3（以小博大）
- [x] 6.1 知识蒸馏增强（大模型→小模型能力转移）
- [x] 6.2 量化+剪枝+蒸馏组合优化（compress_pipeline V1.3）
- [x] 6.3 集成到 VerseNex（CometSparkNexLM.compress_v13）
- [x] 6.4 集成到 VerseTorch（compress.py V1.3）
- [x] 6.5 吞吐率优化
- [x] 6.6 测试

## Task 7: VerseTrainer tqdm + 持续训练 CLI + 1B 模型优化
- [x] 7.1 并行训练 tqdm 支持（统一进度条）
- [x] 7.2 简化输出（重点突出数据，不杂乱）
- [x] 7.3 持续训练 CLI（verse-continue 命令）
- [x] 7.4 1B 模型 CPU/GPU 亲和优化
- [x] 7.5 并行训练进一步支持
- [x] 7.6 测试

## Task 8: 数据集下载器
- [x] 8.1 data/downloader.py（任意 URL 下载 + HF datasets 支持）
- [x] 8.2 断点续传 + 多线程分块下载
- [x] 8.3 自动转 .npz 缓存
- [x] 8.4 CLI: verse-download 命令
- [x] 8.5 测试

## Task 9: 文档更新
- [x] 9.1 README 更新（.vn 格式 + jinja2 模板 + 分区训练 + 持续训练 + 下载器）
- [x] 9.2 ADR（.vn 格式设计 + jinja2 模板 + 分区训练 + 压缩 V1.3）
- [x] 9.3 training_guide + performance_tuning 更新
- [x] 9.4 压缩技术文档

## Task 10: 测试 + 验收
- [x] 10.1 全量测试零失败
- [x] 10.2 关键导入验证
- [x] 10.3 CLI 端到端验证
- [x] 10.4 audit_report 更新
- [x] 10.5 checklist 综合验收
