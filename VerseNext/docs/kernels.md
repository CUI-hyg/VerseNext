# 算子与设备专项优化

本文说明 VerseNext 的底层算子层（`verse-nn/kernels`）、设备抽象层
（`verse-nn/devices`）以及 CPU / CUDA / ROCm / 华为 CANN NPU 四条执行路径。
目标：**在任意设备上都能跑、都尽量快，且数值口径完全一致**。

## 分层结构

```
上层模型（attention.py / blocks.py / transformer.py）
        │  只调用设备无关 API
        ▼
verse_nn.kernels._dispatch          三级回退派发 + KernelPolicy 开关
   ├── cuda_ops.py   ── torch.ops.verse_nn.*_fwd (CUDA/ROCm .so)
   ├── npu_ops.py    ── CANN 原生算子 → AscendC → 参考
   └── cpu_ops.py    ── ATen 优化 + 线程池并行 + int8
        │
        ▼
verse_nn.kernels.reference          纯 PyTorch 基准 / 最终兜底
        ▲
verse_nn.devices                    能力探测（DeviceCaps）+ 后端注册 + 50% 资源锁
```

公开 API 全部是设备无关的：

```python
from verse_nn.kernels import add_rms_norm, swiglu, apply_rope
from verse_nn.kernels import flash_attention, chunked_cross_entropy, kda_chunkwise
```

## 三级回退派发

`kernels/_dispatch.py` 按 **自研扩展 → torch 内建融合 → 参考实现** 的顺序尝试，
任一级失败都静默降级（记 debug 日志），**绝不因算子问题中断训练**：

| 算子 | CUDA / ROCm | CANN NPU | CPU |
|---|---|---|---|
| `add_rms_norm` | `add_rms_norm_fwd` | CANN `npu_add_rms_norm` → `verse_add_rms_norm` | 融合 ATen |
| `swiglu` | `swiglu_fwd` | CANN `npu_swiglu` → `verse_swiglu` | ATen |
| `apply_rope` | `rope_fwd` | CANN `npu_rotary_mul` | 参考（保可微） |
| `flash_attention` | `flash_attn_fwd` | `npu_fusion_attention` / `npu_fused_infer_attention_score` | SDPA |
| `chunked_cross_entropy` | `chunked_ce_fwd` | 参考（AscendC 版未验证） | 复用 logits 缓冲 |
| `kda_chunkwise` | `kda_chunk_fwd` | 参考（AscendC 版未验证） | 线程池并行 |

`KernelPolicy`（`set_policy(...)`）可全局关闭融合做 A/B 对照；`available_ops()`
返回各后端当前实际生效的实现，便于诊断。

## CPU 专项

CPU 没有共享内存 / Cube 单元，优化的真实着力点是**减少分配与访存**和**并行**：

- **融合算子**（`kernels/cpu_ops.py`）：残差加 + RMSNorm 只分配一次；
  分块 CE 跨 chunk 复用 `(B*chunk, V)` 的 logits 缓冲（训练时自动避开不支持
  自动微分的 `out=` 路径）。
- **并行计算**：大算子交给 ATen intra-op 线程池；批内独立的小算子
  （KDA 按 batch×head 切片、三角求解）用 `parallel_map` 在 `ThreadPoolExecutor`
  里并行，规避小算子的调度开销。
- **混合精度**：bf16 在支持 AVX512-BF16 / AMX 的 CPU 上吞吐翻倍；dtype 由
  调用方按 `DeviceCaps` 决策。
- **推理专项**（`kernels/quant.py` + `verse-trainer/inference.py`）：
  int8 动态量化（`Int8Linear`：per-channel 权重 + per-token 激活 + `torch._int_mm`，
  形状不满足自动回退 fp32）、`torch.inference_mode`、KV cache 复用。
  混合精度推理按设备能力自动选 bf16/fp16 autocast。

## CUDA / ROCm 自研算子

源码位于 `verse-nn/verse_nn/csrc/cuda/`，共 6 个前向 kernel：

- `fused_rmsnorm.cu`、`fused_swiglu.cu`、`fused_rope.cu`；
- `flash_attn.cu`：分块在线 softmax，显存从 O(S·T) 降到 O(S·D)；
- `chunked_ce.cu`：不物化 `(B,S,V)` logits；
- `kda_chunk.cu`：chunk 间顺序传状态、chunk 内并行。

设计要点：

- **ROCm 复用同一份源码**：PyTorch 的 `cpp_extension` 在 ROCm 构建下自动 hipify
  `.cu` → HIP 并调 hipcc，无需维护两套代码；warp 级原语按 `__HIP_PLATFORM_AMD__`
  / wavefront=64 分支。
- **反向用参考实现重放**（`cuda_ops.py` 的 `_RefBackward`）：前向走自研 kernel，
  反向在 `enable_grad` 下重放参考前向取梯度，**数学上严格正确**，不会因手写
  反向有 bug 而静默训坏。
- **不可用即返回 `None`**，由派发层回退。

编译：

```bash
python -m verse_nn.kernels.build --backend cuda   # 或 rocm
python -m verse_nn.kernels.build --info           # 查看各后端可用性
# 或 bash src/verse-nn/verse_nn/csrc/build_cuda.sh
```

产物落在 `verse_nn/kernels/_build/<backend>/verse_nn_kernels.so`，
`_ext.py` 在 import 时按此路径探测并 `load_library`（import 期绝不触发编译）。

## 华为 CANN NPU 适配

`kernels/npu_ops.py` 按「CANN 原生融合算子 → 自研 AscendC → 参考」三级尝试：

1. **CANN 原生融合算子**（经 `torch_npu`）：`npu_add_rms_norm` / `npu_swiglu` /
   `npu_rotary_mul` / `npu_fusion_attention`——昇腾上性能最好的路径。
2. **自研 AscendC 算子**（`csrc/ascend/`）：`verse_add_rms_norm`、`verse_swiglu`，
   含 `op_host` tiling、`op_kernel` AscendC 实现，以及注册到 `PrivateUse1`
   dispatch key 的 ACLNN 适配层 `bindings_npu.cpp`。
3. **参考实现**：派发层兜底。

CANN 各版本 API 签名差异较大，所有候选调用都包在 try/except 里，任一失败只
「换下一个」，绝不中断训练。

### 已在硬件上验证（Ascend 910_9362 / CANN 9.1.0 / torch_npu 2.10）

| 算子 | 实现 | 状态 |
|---|---|---|
| `add_rms_norm` | CANN `npu_add_rms_norm` | ✅ 与参考一致（fp32 ~1e-6，fp16 ~4e-3） |
| `swiglu` | CANN `npu_swiglu` | ✅ 一致 |
| `apply_rope` | CANN `npu_rotary_mul`（half 模式） | ✅ 一致 |
| `flash_attention` | CANN `npu_fusion_attention` | ✅ 一致（显式因果掩码 + `sparse_mode=0`） |
| `add_rms_norm` | 自研 `verse_add_rms_norm` | ✅ 编译/加载/数值全部通过 |
| `swiglu` | 自研 `verse_swiglu` | ✅ 同上（含非 32B 对齐尾块） |
| `chunked_cross_entropy` | — | 走参考实现（AscendC 版本未编译验证） |
| `kda_chunkwise` | — | 走参考实现（AscendC 版本未编译验证） |

`tests/test_npu_ops.py` 分两层验证：既**直接**调用 `torch_npu.npu_*` /
`torch.ops.verse_nn.*` 证明设备算子真的可用，也把参考实现打桩成会抛异常的
函数、确认结果确实来自设备路径而非静默回退（历史上出过「算子名写错导致全量
静默回退、训练照跑但零加速」的事故）。无 NPU 时整个模块 skip。

### 踩过的坑（改 `csrc/ascend` 前必读）

1. **算子名不能与 CANN 内建重名**。内建已有 `AddRmsNorm` / `SwiGlu`；同名不但
   会命中内建的 tiling，还会让 `torch_npu.npu_add_rms_norm` 意外解析到自研实现。
   故自定义算子统一加 `Verse` 前缀。
2. **kernel 入口必须 `REGISTER_TILING_DEFAULT(<Tiling>)`**。它把 `sizeof(Tiling)`
   写进 kernel 二进制的 `.ascendc_tiling.default` 段，框架据此分配 tiling data
   缓冲区；漏掉时框架按 msopgen 骨架里的 4 字节占位结构分配，host 侧
   `GetTilingData<T>()` 返回 nullptr，tiling 报 `561002`。
3. **单入口二进制下 `SetTilingKey` 必须为 0**。`TILING_KEY_IS(k)` 只是运行期比较
   （`g_tilingKey == k`），并不生成多个 kernel 入口；下发非 0 会报
   `BinaryGetFunctionByEntry ... funcEntry is invalid`（ret `361001`）。
4. **dtype 分派靠编译期宏**。msopgen 会按 dtype 组合分别编译同一份 kernel 源码，
   并注入 `-DDTYPE_<输入名大写>=half|float`，kernel 里直接
   `Kernel<DTYPE_X>` 实例化即可。**不要**用 `TILING_KEY_IS` 做 dtype 分派。
5. **`ops-info.ini` 的文件名来自 `AddConfig(<soc>)`**。`build.sh` 会在覆盖源码
   *之前* 从 msopgen 骨架里读出该值（如 `Ascend910_9362` → `ascend910_93`），
   顺序错了会让 build 找不到 `aic-<soc>-ops-info.ini`。
6. **Vector 指令不支持 bf16**（`Muls`/`Exp`/`Sigmoid` 等会静态断言失败）。
   自研算子只声明 fp16/fp32；bf16 输入由 CANN 原生算子覆盖。
7. **尾块要用 `DataCopyPad`**。`DataCopy` 要求长度 32B 对齐，`d` 不是对齐粒度
   整数倍时会越界读写。

编译（需 CANN 工具链 + torch_npu）：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
python -m verse_nn.kernels.build --backend npu
# 或 bash src/verse-nn/verse_nn/csrc/ascend/build.sh
```

`build.sh` 会自动探测芯片（`torch_npu.npu.get_device_name` → msopgen 的
compute unit），跑完 msopgen → 编译 → 安装算子包 → 编译 torch 适配层，
产物落在 `verse_nn/kernels/_build/npu/verse_nn_kernels.so`。

## 资源锁（默认 50%）

`verse_nn.devices.Backend.lock_resources` 每个后端一套实现，由
`verse_trainer.resources.apply_resource_limits` 统一调用：

| 后端 | 50% 锁的对象 |
|---|---|
| CPU | 线程数 = affinity 可用核 × `cpu_fraction`；flush denormal |
| CUDA / ROCm | `set_per_process_memory_fraction`；TF32 + cudnn.benchmark |
| CANN NPU | `torch.npu.set_per_process_memory_fraction` + 记录 AI Core 占比目标 |

NPU 的 AI Core 占比（`ai_core_fraction` / `ai_cores_target`）单卡内按核切分需
平台侧 `npu-smi` 配额，本层记录并告警，进程侧通过 `ASCEND_RT_VISIBLE_DEVICES`
限制可见设备。内存另有 `MemoryGuard` 软上限告警。

## 正确性契约与测试

**所有后端实现都必须与 `kernels/reference.py` 在容差内一致**，测试直接对拍：

- `tests/test_kernels.py`：参考实现 vs 仓库原实现、CPU 优化实现 vs 参考、
  梯度回传、策略开关与降级。
- `tests/test_devices.py`：能力探测、后端注册、50% 资源锁。
- `tests/test_npu_ops.py`：昇腾设备算子对拍（CANN 原生 + 自研 AscendC），
  并验证派发**没有**静默回退到参考实现。无 NPU 时整个模块 skip。
- `tests/test_quant.py`：int8 数值正确性、层替换范围、回退行为。
- `tests/test_kernel_build.py`：构建探测 + **Python 引用名与 C++ 注册名的
  静态一致性校验**（防止 `.so` 编译成功却因名字不匹配被判为不可用）。

## 已知限制

- `flash_attn.cu` 为正确的 SIMT 实现，**未使用 tensor core**（WMMA/MFMA）；
  后续可在 D 维做 16×16 分块交给张量核心，预期再获 2~4× 吞吐。
- CUDA / ROCm 路径需在对应硬件与工具链上编译验证；纯 CPU 环境下仅验证到
  「优雅降级」这一层。
- 昇腾侧：CANN 原生算子路径与自研 `verse_add_rms_norm` / `verse_swiglu`
  已在 Ascend 910_9362 上验证；`flash_attn` / `chunked_ce` / `kda_chunk`
  三个 AscendC kernel 的源码仍在仓库里（`op_kernel/*_ascendc.cpp`），
  但**未编译验证**，也未打包进算子包，当前由 CANN 原生算子 / 参考实现覆盖。
  它们尚未按上面「踩过的坑」改造（缺 `REGISTER_TILING_DEFAULT`、用
  `TILING_KEY_IS` 做 dtype 分派），接入前需先修。
