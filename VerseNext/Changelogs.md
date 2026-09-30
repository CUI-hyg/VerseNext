# Changelogs

本文件记录 VerseNext 的所有重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- **底层算子与设备专项优化**（`verse-nn` / `verse-trainer`）
  - **设备抽象层**（`verse-nn/devices/`）：`DeviceCaps` 能力描述（bf16/tf32/
    flash 支持、warp 宽度、共享内存、架构串）+ `Backend` 注册表
    （CPU / CUDA / ROCm / CANN NPU / XPU / MPS），统一 `lock_resources`
    **50% 资源锁**：CPU 限线程、CUDA/ROCm 限显存 + TF32/benchmark、NPU 限显存
    并记录 AI Core 占比目标（`ai_cores_target`）。后端不可用自动回退 CPU。
  - **CPU 专项优化**（`kernels/cpu_ops.py`）：融合残差加+RMSNorm、SwiGLU、
    分块 CE（跨 chunk 复用 logits 缓冲，训练自动避开 `out=` 保可微）；
    批内独立计算（KDA 按 batch×head 切片）经 `parallel_map` 线程池并行。
  - **CPU 推理专项**（`kernels/quant.py` + `inference.py`）：int8 动态量化
    （自研 `Int8Linear` per-channel 权重 + per-token 激活 + `torch._int_mm`，
    形状不满足自动回退 fp32）、`torch.inference_mode`、KV cache 复用。
  - **混合精度推理**：`Generator` 按设备能力自动选 bf16/fp16 autocast
    （CPU 仅在 AVX512-BF16/AMX 上启用），统一走 `Backend.autocast`。
  - **CUDA / ROCm 自研算子**（`csrc/cuda/`）：手写 6 个前向 kernel——融合
    RMSNorm、SwiGLU、RoPE、分块在线 softmax FlashAttention（不物化 S×T 矩阵）、
    分块 CE（不物化 B×S×V logits）、KDA chunkwise；注册到 `torch.ops.verse_nn`。
    ROCm 由 hipify 从同一份 `.cu` 生成，无需维护两套源码；反向统一用参考实现
    在 `enable_grad` 下重放，保证梯度数学严格正确。
  - **CANN NPU 适配**（`kernels/npu_ops.py` + `csrc/ascend/`）：优先 CANN 原生
    融合算子（`npu_add_rms_norm` / `npu_swiglu` / `npu_rotary_mul` /
    `npu_fusion_attention`），其次自研 AscendC 算子（op_host tiling + op_kernel +
    ACLNN 适配层，`PrivateUse1`）。已在 Ascend 910_9362 上验证（见 Fixed）。
  - **三级回退派发**（`kernels/_dispatch.py`）：自研扩展 → torch 内建融合 →
    参考实现；任一级失败静默降级（记 debug 日志），**绝不因算子问题中断训练**。
    `KernelPolicy` / `set_policy` 提供全局开关用于 A/B 基准。
  - **构建入口**：`python -m verse_nn.kernels.build --backend {cuda,rocm,npu}`
    / `--info`，产物落到 `kernels/_build/<backend>/`，import 期只探测不编译。
- **测试**：`test_kernels.py`（参考/CPU 数值对拍、梯度、派发回退）、
  `test_devices.py`（能力探测 + 50% 资源锁）、`test_quant.py`（int8 精度/回退）、
  `test_kernel_build.py`（构建探测 + Python 引用名与 C++ 注册名的静态一致性）。

- **断点续训（full resume）**（`verse-trainer`）
  - checkpoint 现在保存**完整训练状态**：优化器/缩放器、调度元信息、RNG
    （python/torch/numpy/cuda + 数据采样器）、监控与早停进度
    （`best_loss`/`bad_evals`/`early_stopped`）、累计 token 数、loss 历史、
    数据指纹与 resolved 配置；`Trainer.load_checkpoint(resume_training=True)`
    一键恢复。实测续训后每一步 loss 与不间断训练**逐位一致**。
  - `TrainerConfig.save_last`：每个 `ckpt_every` 覆盖保存
    `{model_name}_last.pt`，配合 `save_best_only` 提供崩溃续训兜底
    （finalize 时随中间 checkpoint 一并清理）。
  - `verse_trainer.data.fingerprint_source`：token 流/SFT 样本集的轻量内容
    指纹（超长时首尾采样），用于检测「是否换了数据」。
- **资源智能分配与占用上限**（`verse-trainer/resources.py`）
  - `ResourceConfig`（`cpu_fraction`/`mem_fraction`/`gpu_mem_fraction`，默认各 0.5）、
    `resolve_threads`（按 `sched_getaffinity` 感知的可用核数取比例，修复容器内
    `os.cpu_count()` 偏小导致的算力欠用）、`apply_resource_limits`（线程 +
    flush denormal + `torch.cuda.set_per_process_memory_fraction`）、
    `MemoryGuard`（RSS 软上限告警）。
  - `TrainerConfig.resources` 接入训练循环；`inference.optimize_cpu` 复用同一套。
- **训练数据 / 文件格式支持**（`verse-trainer/data_formats.py`）
  - 文本：jsonl/json/txt/md/csv/tsv/parquet（`text_column`/`text_columns`/
    `text_template`/`delimiter` 控制字段抽取，问答对 query+response 自动识别）。
  - 预分词：npy/bin/tokens（`load_token_stream`，dtype 可选、支持 memmap）。
  - `SFTDataset.from_file` 与 `load_conversations`：对话数据支持 jsonl/json/csv/parquet，
    兼容 messages/conversations/system+user+assistant/JSON 字符串等多种结构。
- **生成质量与稳定性**（`verse-nn`）
  - `generate()` 新增 `top_p`（核采样）、`repetition_penalty`（CTRL 式重复惩罚）、
    `no_repeat_ngram`（禁重复 n-gram）、`stop_token_ids`；输入/输出 id 一律
    clamp 到 `[0, vocab)`（越界 id 是乱码的常见来源）；全屏蔽时回退均匀分布避免 NaN。
  - `GenerateConfig` 同步暴露上述参数，默认 `temperature=0.7, top_p=0.9,
    repetition_penalty=1.1, no_repeat_ngram=3`。
- **checkpoint 生命周期与阶段分级**（`verse-trainer/trainer.py`）
  - `TrainerConfig.stage`（pretrain/finetune/sft/posttrain）、`cleanup_intermediate`
    （训练完成只保留 final）、`final_keep_optimizer`（final 是否保留优化器状态）、
    `plot_every_eval`（每个 eval 刷新 loss 曲线）、`loss_chunk_size`。
  - `Trainer.finalize(stage, path)` / `cleanup_intermediate(keep)`；
    checkpoint 内写入 `stage`/`param_count`/`state_numel`/`tie_weights` 元数据。
- **统一参数口径**：`count_parameters(model) -> (unique, total)`，按底层张量
  去重（权重绑定只计一次）；加载 checkpoint 时校验 `param_count`，架构不匹配
  立即报错而非静默「参数量减半」。

### Changed

- `VerseTransformer.forward` 新增 `return_logits` / `loss_chunk_size`：训练时
  可分块计算 lm_head + CE，避免物化 `(B,S,vocab)` logits 峰值（数值等价）。
- 分词器性能（`verse-tokenizer/bpe.py`）：模块级正则 + 词级 BPE LRU 缓存 +
  token→bytes 缓存 + `encode_batch`；`eos_token_id` 正确解析 `<|endoftext|>`；
  `add_special_token` 以实际最大 id 递增，避免越界。
- `Trainer` 训练/评估走分块 CE 路径；因果掩码按 `(seq,total,device)` 缓存。

### Fixed

- **昇腾 NPU 路径在真实硬件上跑通**（Ascend 910_9362 / CANN 9.1.0 /
  torch_npu 2.10）：`kernels/npu_ops.py` 此前用的是不存在的 `torch.npu.npu_*`，
  原生融合算子全部静默回退到参考实现（训练照跑、零加速）。已改为 `torch_npu`
  模块级命名空间，并修正各算子契约（`npu_add_rms_norm` 返回 `(y, rstd, res)`、
  `npu_rotary_mul` 需 `(1,1,S,D)` 系数、`npu_fusion_attention` 用显式因果掩码 +
  `sparse_mode=0`）。
- **自研 AscendC 算子编译并验证**：`verse_add_rms_norm` / `verse_swiglu` 的
  op_host tiling、op_kernel 与 ACLNN 适配层全部跑通。三处关键修复：自定义算子名
  加 `Verse` 前缀避开 CANN 内建算子（`AddRmsNorm` / `SwiGlu`）；kernel 入口补
  `REGISTER_TILING_DEFAULT`（否则框架按骨架的 4 字节占位结构分配 tiling data，
  `GetTilingData<T>()` 返回 nullptr，报 561002）；单入口二进制下 `SetTilingKey(0)`
  （否则报 361001）。dtype 分派改用构建期注入的 `DTYPE_*` 宏。
- **`ascend/build.sh` 重写**：自动探测芯片 → msopgen → 编译安装算子包 → 编
  torch 适配层一条命令跑通；适配层改为直接调 CANN ACL C API，绕开 `EXEC_NPU_CMD`
  依赖的 torch_npu 未导出符号。尾块改用 `DataCopyPad`，非 32B 对齐的 shape 不再越界。
- **NPU 环境下的 6 个测试失败**：RNG state / LoRA 参数 / 输入 batch 未随设备
  迁移（`trainer.py`）；跨 checkpoint 恢复在加速器上的非确定性
  （`test_resume.py` 在加速器上放宽为 1e-4，CPU 仍要求逐位相等）。
- **算子扩展可用性探测的算子名不一致**：`kernels/_ext.py` 的 `_REQUIRED_OPS`
  用的是 `fused_add_rms_norm`/`fused_swiglu`/`flash_attn`，而 C++ 实际注册的是
  `add_rms_norm_fwd`/`swiglu_fwd`/`flash_attn_fwd`；`npu_ops.py` 的 AscendC 分支
  也引用了不存在的 `*_npu` 名字。此前会导致**编译成功的 `.so` 仍被判为不可用**，
  自研 CUDA/ROCm/AscendC 算子全部静默回退到参考实现。已统一为注册名，并在
  `test_kernel_build.py` 增加 Python 引用名 ↔ C++ 注册名的静态一致性校验防回归。
- `ascend/bindings_npu.cpp` 补充 `TORCH_LIBRARY` schema 定义：NPU 构建只编译
  该文件（不编 `cuda/bindings.cpp`），缺 schema 时 `torch.ops.verse_nn.*` 不存在，
  `TORCH_LIBRARY_IMPL` 的实现无算子可挂载。
- **梯度检查点路径**：`torch.utils.checkpoint.checkpoint` 未导入导致
  `gradient_checkpointing=True` 时前向报 `AttributeError`（0.6B Dense 配置启用后暴露）。
- 数据加载：jsonl 未识别文本字段时不再静默训练序列化 JSON（乱码根因），
  改为告警 + 回退，并支持 `query`/`response` 问答对。

## [0.3.1] - 2026-09-19

### Added

- **KDA chunkwise 并行内核**（`verse-nn`）
  - 推导并实现门控 delta 规则注意力的 chunkwise 形式：利用门控行缩放与
    右侧状态转移可交换的性质做 chunk 内归一化（`S̃ = Λ⁻¹S`，log 空间
    cumsum），chunk 内递归展开为严格上三角线性系统（WY 表示），批量
    `solve_triangular` 求解——chunk 内全并行、chunk 间顺序传状态。
  - 预填充/训练走 chunkwise（`kda_chunk_size`，默认 64，含非整倍长度
    padding 路径）；单步解码保持递归（常数开销）。
  - 实测 CPU 512-token 前向 **10.5x**（1421ms → 136ms，b=2/4 层）；
    与递归基准 3 种子等价性验证（差异 < 1.4e-6）。
  - `VerseTransformerConfig.kda_chunk_size` 配置项；docs/attention.md
    补充算法说明与数值边界。
- **tests**：chunkwise/递归精确等价、padding 路径、chunk 预填充 + 递归
  解码的贪心一致性（累计 31 项测试）。

## [0.3.0] - 2026-09-19

### Added

- **注意力机制升级**（`verse-nn`）
  - `attention.py` 重构为可插拔注册式（`attention/<name>` + `build_attention` 工厂）。
  - **DSA**（`DSAttention`，DeepSeek 稀疏注意力简化参考实现）：轻量 indexer 打分 +
    top-k token 选择 + 自身强制保留；解码期 gather 选中 KV 直接计算注意力
    （免去全长度掩码构建与 SDPA 内部掩码转换）。
  - **KDA**（`KDAAttention`，Kimi Delta Attention 简化参考实现）：细粒度门控
    （逐通道 sigmoid 衰减）+ delta 规则递归状态 `S ∈ (d_k, d_v)`；
    状态与序列长度无关，1024 上下文缓存显存约为 KV 缓存的 1/32。
  - `VerseTransformerConfig` 新增 `attn_type` / `dsa_top_k` / `dsa_index_head_dim`
    （含合法性校验）。
- **算子与缓存优化**（`verse-nn`）
  - 融合 QKV 投影：单 GEMM 产出 q/k/v（替代 q/kv 两次投影）。
  - 融合 SwiGLU：gate/up 合并为单 GEMM（`gate_up_proj`，输出 2×hidden 切分）。
  - `kv_cache.py`：静态预分配 KV Cache（原地写入 + 零拷贝切片视图，
    消除逐步 `torch.cat` 的 O(t²) 分配/拷贝）；`RecurrentStateCache`（KDA）；
    单步解码免因果掩码路径（单 query 因果约束平凡满足）。
  - `generate()`：改用静态缓存 + `logits_to_keep=1`（prefill 只投影最后位置）
    + `torch.inference_mode`；实测 CPU 对旧动态缓存 1.3x、对全量重算 2.6x。
  - `forward()` 新增 `layer_caches`（静态）与 `logits_to_keep` 参数。
- **docs/attention.md**：注意力类型、性能优化清单与自定义注意力指南。
- **tests**：DSA/KDA 前向与缓存一致性、注意力注册工厂、配置校验、
  KDA 常量显存优势（累计 29 项测试）。

### Changed

- **破坏性**：模型权重命名变更（`q_proj`/`kv_proj` → 融合 `qkv_proj`；
  `gate_proj`/`up_proj` → `gate_up_proj`），0.2.0 及之前的 checkpoint 不兼容；
  LoRA 默认目标模块同步更新。

### Fixed

- DSA 缓存解码与全量重算的选择一致性（单步解码强制保留自身位置；
  indexer 分数并列受 GEMM 归约顺序噪声影响属语义合法行为，测试中以
  确定性分数验证选择逻辑）。

## [0.2.0] - 2026-09-19

### Added

- **verse-tokenizer** 新包：byte-level BPE 分词器
  - `bpe.py`：BPE 训练/编码/解码；256 字节基础词表（任意文本无 OOV）；
    特殊 token（`<|bos|>` 等）；文本中 `<|...|>` 标记自动注册；
    `encode(return_offsets=True)` 返回字节级偏移；HF 风格 `save_pretrained`/`from_pretrained`。
  - `chat_template.py`：`chat_template.jinja` 支持（jinja2，HF 约定上下文），
    内置 ChatML 默认模板；`encode_chat` 输出 SFT 掩码 labels（哨兵定位 + 字节偏移交集）。
  - `masking.py`：labels_from_spans 掩码工具。
- **CPU 推理/训练性能优化**
  - `verse-nn`：KV 缓存（attention/block/transformer 全链路，GQA 压缩缓存，
    RoPE 位置偏移，缓存路径显式因果掩码）；`generate` 支持 eos 提前停止。
  - `verse-trainer/inference.py`：int8 动态量化（`dynamic_quantize`）、
    `optimize_cpu`（线程数 + flush denormal）、`Generator` 推理引擎
    （chat / complete 生成，`torch.inference_mode`）。
  - `verse-trainer/trainer.py`：CPU profile（自动线程数 + flush denormal）。
- **后训练 / 微调支持**
  - `verse-nn/lora.py`：LoRALinear（B=0 初始化）、`apply_lora` /
    `mark_only_lora_trainable` / `lora_state_dict` / `load_lora_state_dict` / `merge_lora`。
  - `verse-trainer/sft.py`：`SFTDataset`（对话编码 + padding + 右移掩码 targets，
    实现 `next_batch` 复用 Trainer 循环）；`TrainerConfig.lora` 一键启用 LoRA 微调。
  - `Trainer.train` 支持任意实现 `next_batch()` 的数据源。
  - `Trainer.save_adapter`：适配器独立保存（<1% 主干体积）。
- **verse-cli** 新子命令：`verse train-tokenizer` / `verse sft` / `verse generate`。
- **docs/post-training.md**：后训练全流程文档（分词器 → SFT → 生成 + CPU 优化清单）。
- **tests**：`test_tokenizer_sft.py`、`test_kv_lora_inference.py`（累计 24 项测试）。

### Fixed

- `Generator.from_pretrained` 识别 LoRA checkpoint 并重建适配器结构。
- tokenizer 偏移量在多空格/特殊 token 边界的对齐问题。

## [0.1.0] - 2026-09-19

### Added

- **仓库架构**：uv workspace 多包结构，代码位于 `src/`，文档位于 `docs/`。
- **verse-core** 基础设施包
  - `registry.py`：全局组件注册中心（`Registry`），支持模型/优化器/调度器/变异算子的注册与创建。
  - `config.py`：dataclass 配置基类，支持 YAML 加载/保存与嵌套覆盖。
  - `events.py`：事件总线（`EventBus`），trainer 与 RSI 之间解耦通信。
  - `logging.py`：统一日志工具。
- **verse-nn** 模型架构包
  - `attention.py`：SDPA 注意力，支持 GQA/MQA、因果掩码、KV 缓存接口预留。
  - `rope.py`：RoPE 旋转位置编码。
  - `blocks.py`：pre-norm Transformer Block 与 SwiGLU FFN。
  - `transformer.py`：完整 decoder-only Transformer（`VerseTransformer`），支持梯度检查点与 `torch.compile`。
  - 模型通过 `Registry` 注册为 `model/verse_transformer`。
- **verse-trainer** 训练优化包
  - `trainer.py`：训练循环，支持 AMP 混合精度（bf16/fp16）、梯度累积、梯度裁剪、`torch.compile`、checkpoint 保存/恢复。
  - `optim.py`：AdamW（fused 可用时自动启用）+ warmup-cosine 学习率调度器。
  - `data.py`：token 流式批迭代器（`TokenBatchIterator`）。
  - `callbacks.py`：基于 `EventBus` 的回调（日志、checkpoint、评估）。
- **verse-rsi** 递归自进化包
  - `evaluator.py`：FitnessEvaluator，在验证集上评估个体适应度（loss / tokens-per-second 综合分）。
  - `mutator.py`：可插拔变异算子——超参变异（lr、batch size）、架构变异（层数/头数/隐藏维度），基于 `Registry` 注册。
  - `population.py`：种群档案，支持持久化到 JSON。
  - `evolve.py`：递归自进化主循环（评估 → 选择 → 变异 → 再训练），进化历史通过事件总线记录。
- **verse-cli** 命令行入口
  - `verse train`：训练一个模型。
  - `verse evolve`：启动 RSI 自进化循环。
  - `verse bench`：基准测速。
- **docs/**：`architecture.md`、`getting-started.md`、`rsi.md`。
- **tests/**：端到端冒烟测试。

[Unreleased]: https://example.com/versenext/compare/0.1.0...HEAD
[0.1.0]: https://example.com/versenext/releases/tag/0.1.0
