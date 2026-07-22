# Part4K2 验收清单

## Task 1: .vn 文件格式
- [x] .vn 格式可读写（safetensors 权重 + 元数据）
- [x] .pt → .vn 转换正常
- [x] .vn → .pt 转换正常
- [x] mmap 零拷贝读取可用
- [x] 智能压缩存储可用
- [x] CometSparkV05LM.save_vn / load_vn 可用
- [x] verse-convert CLI 可用

## Task 2: Tokenizer jinja2
- [x] jinja2 引擎集成（可选依赖）
- [x] ChatML 模板渲染正确
- [x] 工具调用模板渲染正确（Qwen3 格式）
- [x] tokenizer.json 内嵌 chat_template.jinja
- [x] apply_chat_template 支持 tools 参数
- [x] 所有 tokenizer 类统一升级

## Task 3: 生成输出优化
- [x] CometSparkV05LM.generate 无默认 max_new_tokens 限制
- [x] EOS 自然停止可用
- [x] 安全上限 100K 生效
- [x] VerseTrainer 生成调用适配
- [x] verse-eval CLI 适配

## Task 4: 智能分区训练
- [x] LayerWiseTrainer 按 layer 拆分可用
- [x] 硬盘卸载/加载可用
- [x] 统一实体（训练一致性）
- [x] 内存监控可用
- [x] CLI --partition-training 可用

## Task 5: 资源利用优化
- [x] autocast GPU 混合精度一致
- [x] GPU 显存管理可用
- [x] CPU BLAS 线程优化
- [x] 设备亲和性（pin_memory 实装）
- [x] NPU 后端完善

## Task 6: 压缩技术 V1.3
- [x] 知识蒸馏增强可用
- [x] compress_pipeline V1.3 可用
- [x] 集成到 VerseNex
- [x] 集成到 VerseTorch
- [x] 吞吐率优化

## Task 7: VerseTrainer 优化
- [x] 并行训练 tqdm 可用
- [x] 输出简化清晰
- [x] 持续训练 CLI 可用
- [x] 1B 模型优化
- [x] 并行训练进一步支持

## Task 8: 数据集下载器
- [x] 任意 URL 下载可用
- [x] HF datasets 下载可用
- [x] 断点续传可用
- [x] 自动转 .npz 缓存
- [x] verse-download CLI 可用

## Task 9: 文档
- [x] README 更新
- [ ] ADR 新增
- [ ] training_guide + perf_tuning 更新
- [ ] 压缩技术文档

## Task 10: 综合验收
- [x] pytest tests/ 全量测试零失败
- [x] 关键导入全部成功
- [x] CLI 端到端验证通过
- [x] .vn ↔ .pt 互转验证
- [x] audit_report 更新
- [x] 无回归问题
