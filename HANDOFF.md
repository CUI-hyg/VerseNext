# HANDOFF — VerseNext_Exp

工作区交接文档。最后更新：2026-09-26。

## 1. 这是什么

`VerseNext_Exp/` 是一个双项目工作区：

| 目录 | 说明 |
|---|---|
| `VerseNext/` | 框架本体。uv workspace，多包结构在 `src/`（verse-core / verse-nn / verse-tokenizer / verse-trainer / verse-rsi / verse-cli） |
| `CometSpark/` | 下游实验模型项目。复用 `verse_nn.transformer.VerseTransformer`（RoPE + GQA + SwiGLU + RMSNorm），版本 0.1 / 0.2 / 0.3（0.3 ≈ 0.6B，Dense sdpa） |

远端：<https://github.com/CUI-hyg/VerseNext>（默认分支 `main`）。

## 2. 当前主线：底层算子 & 设备专项优化

任务要求：
1. **CPU**：并行计算、混合精度推理、推理专项优化。
2. **GPU**：针对 CUDA & ROCm 编写自研训练&推理底层算子，面向 VerseNext + CometSpark
   做专项优化，瞄准高吞吐。
3. **华为 CANN NPU**：适配，资源锁 50%，性能目标与 GPU 同级。

### 已完成

- **设备抽象层**（`VerseNext/src/verse-nn/verse_nn/devices/`）
  - `caps.py`：`DeviceCaps` 能力描述（bf16/tf32/flash、warp 宽度、共享内存、架构串）。
  - `backend.py`：`Backend` 注册表（CPU / CUDA / ROCm / CANN NPU / XPU / MPS），
    统一 `lock_resources` **50% 资源锁**（CPU 线程 / CUDA·ROCm 显存 + TF32 /
    NPU 显存 + AI Core 占比目标）；后端不可用自动回退 CPU。
- **CPU 专项**（`kernels/cpu_ops.py`）
  - 融合残差加+RMSNorm、SwiGLU、分块 CE（跨 chunk 复用 logits 缓冲）。
  - `parallel_map` 线程池：KDA 按 batch×head 切片并行。
  - 推理：int8 动态量化（`kernels/quant.py`，`Int8Linear` per-channel 权重 +
    per-token 激活 + `torch._int_mm`）、`torch.inference_mode`、KV cache 复用。
  - 混合精度推理：`Generator` 按设备能力自动选 bf16/fp16 autocast。
- **CUDA / ROCm 自研算子**（`VerseNext/src/verse-nn/verse_nn/csrc/cuda/`）
  - 6 个前向 kernel：`fused_rmsnorm` / `fused_swiglu` / `fused_rope` /
    `flash_attn`（分块在线 softmax）/ `chunked_ce` / `kda_chunk`。
  - ROCm 由 hipify 复用同一份 `.cu`；反向用参考实现重放保证梯度正确。
  - 注册到 `torch.ops.verse_nn.*_fwd`（`bindings.cpp`）。
- **CANN NPU 适配**（`kernels/npu_ops.py` + `csrc/ascend/`）
  - 优先 CANN 原生融合算子（`npu_rms_norm` / `npu_swiglu` / `npu_rotary_mul` /
    `npu_fusion_attention` / `npu_fused_infer_attention_score`）。
  - 其次自研 AscendC 算子（op_host tiling + op_kernel + ACLNN 适配层
    `bindings_npu.cpp`，`PrivateUse1` dispatch key）。
- **三级回退派发**（`kernels/_dispatch.py`）：自研扩展 → torch 内建 → 参考实现，
  任一级失败静默降级，绝不因算子问题中断训练。`KernelPolicy` 提供全局开关做 A/B。
- **文档**：`VerseNext/docs/kernels.md`（算子与设备分层、构建、正确性契约、已知限制）。

### 本次修复的关键 Bug（已修）

- `kernels/_ext.py` 的 `_REQUIRED_OPS` 与 C++ 注册名不一致
  （`fused_add_rms_norm` vs `add_rms_norm_fwd` 等），导致**编译成功的 `.so` 仍被判为
  不可用**、自研算子全部静默回退。已统一为注册名。
- `kernels/npu_ops.py` 的 AscendC 分支引用了不存在的 `*_npu` 算子名，已改为 `*_fwd`。
- `csrc/ascend/bindings_npu.cpp` 缺 `TORCH_LIBRARY` schema 定义（NPU 构建只编此文件），
  已补上。
- `tests/test_kernel_build.py` 新增静态一致性校验，防止 Python 引用名与 C++ 注册名
  再次漂移。

### 验证方式

```bash
cd VerseNext
export PYTHONPATH=$(ls -d src/*/ | tr '\n' ':')   # 包未 pip 安装，必须设 PYTHONPATH
python -m pytest tests/ -q                         # 150 passed, 1 skipped（skip 为无 NPU）
```

## 3. 未完成 / 待办

1. **CUDA / ROCm / CANN 三条路径均未编译验证**。本机是 `torch 2.3.1+cpu`，无
   nvcc / hipcc / CANN 工具链，只能做静态一致性检查与「优雅降级」测试。
   需在对应硬件上实际编译跑通；`bindings_npu.cpp` 的 ACLNN 头文件签名可能要按
   CANN 版本微调（见 `csrc/ascend/build.sh`）。
2. **无性能基准**。任务要求 NPU 性能与 GPU 同级，目前没有任何实测数据。
   `flash_attn.cu` 自述为未用 tensor core 的 SIMT 实现，训练吞吐仍有优化空间。
3. **CometSpark 端到端回归**未在 GPU/NPU 上跑过，需确认融合算子接入后的真实吞吐。

## 4. 环境与注意事项

- **本机无 GPU / NPU**：`torch 2.3.1+cpu`，无 CUDA / ROCm / CANN。
- **凭据安全**：`.codebuddy/models.json`（工作区根与 `VerseNext/` 下各一份）含
  **明文 API key**，已在 `.gitignore` 中排除，切勿提交。建议轮换该 key。
- **模型权重不入库**：`CometSpark/checkpoints/` 下两个 `.pt` 各约 2.4GB
  （共 4.5GB），超出 GitHub 单文件 100MB 限制，已在 `.gitignore` 排除。
  需要权重请另行分发（对象存储 / Release / Git LFS）。
- **网络**：本环境 `github.com` 不可达（git 协议超时），仅 `api.github.com` 可达；
  且未配置 GitHub 凭据。推送与建 PR 需在有凭据/网络的机器上完成，或提供 PAT 走 API。

## 5. 常用命令

```bash
# 测试（务必先设 PYTHONPATH）
export PYTHONPATH=$(ls -d src/*/ | tr '\n' ':') && python -m pytest tests/ -q

# 编译自研算子
python -m verse_nn.kernels.build --backend cuda   # 或 rocm / npu
python -m verse_nn.kernels.build --info           # 查看各后端可用性
bash src/verse-nn/verse_nn/csrc/build_cuda.sh     # 等价封装
```

详见 `VerseNext/docs/kernels.md`、`VerseNext/Changelogs.md`。
