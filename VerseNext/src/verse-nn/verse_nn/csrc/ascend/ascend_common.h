/*
 * VerseNext 自研算子 —— 华为昇腾 CANN / AscendC 公共头
 *
 * AscendC 编程模型要点（与 CUDA 差异很大，务必注意）：
 *   - 三级存储：GM（全局/显存）→ L1/UB（片上）；计算在 AI Core 的
 *     Cube（矩阵）与 Vector（逐元素）单元上完成；
 *   - 用 TPipe/TQue 管理流水：DataCopy 搬入 → Compute 计算 → DataCopy 搬出，
 *     流水线由 CANN 调度器自动重叠（double buffer）；
 *   - 核内并行度由 block_dim 决定，每核通过 GetBlockIdx() 取核号；
 *   - 资源锁 50%：显存由 torch.npu.set_per_process_memory_fraction 限制，
 *     AI 核比例通过 block_dim 与可见设备数控制（见 devices/backend.py）。
 *
 * 编译：本目录用 CANN 的 msopgen/自定义算子工程编译（见 build.sh），
 *       产出 ACLNN 自定义算子包，再由 bindings_npu.cpp 经 aclnn 接口调用。
 *
 * 验证状态（Ascend 910_9362 / CANN 9.1.0）：本头文件与
 * ``op_kernel/verse_add_rms_norm_ascendc.cpp``、``op_kernel/verse_swiglu_ascendc.cpp``
 * 已编译、加载并在硬件上数值对拍通过；``flash_attn`` / ``chunked_ce`` /
 * ``kda_chunk`` 三个 kernel **尚未编译验证**，接入前请先读 docs/kernels.md
 * 的「踩过的坑」。
 */

#ifndef VERSE_ASCEND_COMMON_H
#define VERSE_ASCEND_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

namespace verse_nn {
namespace ascend {

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr int32_t kUbBytes = 192 * 1024;          // 910B Unified Buffer ≈ 192KB
constexpr int32_t kUbReserveBytes = 8 * 1024;     // 预留给系统/同步
constexpr int32_t kMaxBlockDim = 24;              // 910B AI Core 数
constexpr int32_t kDefaultTileElems = 8192;       // 单次 UB 处理元素数（fp32）

// 32B 对齐是 DataCopy 的高效粒度（一个 block 的搬运单位）
constexpr int32_t kAlignBytes = 32;

// ---------------------------------------------------------------------------
// 对齐工具
// ---------------------------------------------------------------------------
__aicore__ inline int32_t AlignUp(int32_t v, int32_t align) {
  return (v + align - 1) / align * align;
}

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// 标量工具
//   设备侧（AscendC kernel）不提供 Min/Max 的标量版本，且没有 sqrtf/expf/logf
//   这类 C 数学库符号；这里统一提供，避免各 kernel 各写一份。
// ---------------------------------------------------------------------------
__aicore__ inline int32_t MinI(int32_t a, int32_t b) { return a < b ? a : b; }
__aicore__ inline int32_t MaxI(int32_t a, int32_t b) { return a > b ? a : b; }
__aicore__ inline float MinS(float a, float b) { return a < b ? a : b; }
__aicore__ inline float MaxS(float a, float b) { return a > b ? a : b; }
// 设备侧没有 exp/log 的标量符号（连 sqrtf 也没有，只有 sqrt），需要指数/对数
// 的 kernel 请用 Vector 指令（AscendC::Exp / Log）在 LocalTensor 上算，
// 不要在这里补 C 数学库符号。
__aicore__ inline float SqrtS(float x) { return sqrt(x); }

// ---------------------------------------------------------------------------
// Tiling 读取
//   host 用 ``context->GetTilingData<T>()`` 写出的结构体，会作为 kernel 的
//   ``tiling`` GM 指针传入；GM_ADDR 是 ``__gm__ uint8_t*``，直接 reinterpret
//   到非 __gm__ 指针会被编译器拒绝，必须保留 __gm__ 限定。
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline const __gm__ T* ReadTiling(GM_ADDR tiling) {
  return reinterpret_cast<const __gm__ T*>(tiling);
}

// ---------------------------------------------------------------------------
// 按 dtype 选择 UB 中的 Tile 元素数
//    fp32 -> 4B, fp16/bf16 -> 2B；保证 tile 不超过 UB 预算
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline int32_t TileElemsForDtype() {
  constexpr int32_t elem_bytes = sizeof(T);
  const int32_t budget = kUbBytes - kUbReserveBytes;
  return AlignUp(budget / elem_bytes / 4, 256);   // 4 个缓冲（in/out/临时/权重）
}

// ---------------------------------------------------------------------------
// GM <-> UB 搬运
//   普通 DataCopy 要求长度按 32B 对齐，尾块（d 不是对齐粒度整数倍时）会
//   越界读写。DataCopyPad 支持任意字节长度，因此这里统一用它做搬运：
//   - 搬入时按需右侧补零，保证 UB 里没有脏数据参与计算；
//   - 搬出时只写回 count 个元素，绝不越界。
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CopyInPad(const LocalTensor<T>& dst,
                                 const GlobalTensor<T>& src, int32_t count) {
  DataCopyExtParams params(1, static_cast<uint32_t>(count) * sizeof(T), 0, 0, 0);
  DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
  DataCopyPad(dst, src, params, pad);
}

template <typename T>
__aicore__ inline void CopyOutPad(const GlobalTensor<T>& dst,
                                  const LocalTensor<T>& src, int32_t count) {
  DataCopyExtParams params(1, static_cast<uint32_t>(count) * sizeof(T), 0, 0, 0);
  DataCopyPad(dst, src, params);
}

// ---------------------------------------------------------------------------
// Vector -> Scalar 同步
//   ReduceSum 等 Vector 指令把结果写进 UB 后，标量单元（GetValue）读取前
//   必须显式同步，否则读到的是未完成的中间值。官方样例均显式 Set/Wait。
// ---------------------------------------------------------------------------
__aicore__ inline void SyncVectorToScalar() {
  event_t event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(event_id);
  WaitFlag<HardEvent::V_S>(event_id);
}

__aicore__ inline void SyncScalarToVector() {
  event_t event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  SetFlag<HardEvent::S_V>(event_id);
  WaitFlag<HardEvent::S_V>(event_id);
}

// ---------------------------------------------------------------------------
// 把 LocalTensor 转成 fp32 再求平方，结果写入 sq。
//   fp32 输入直接乘；fp16 输入先用 Cast 升精度（work 作中转），
//   这样归约全程 fp32，避免 fp16 平方和溢出/精度损失。
//   用重载而不是 Cast(src, dst 同类型)——同类型 Cast 在部分版本会断言失败。
// ---------------------------------------------------------------------------
__aicore__ inline void SquareToFp32(const LocalTensor<float>& sq,
                                    const LocalTensor<float>& work,
                                    const LocalTensor<float>& src, int32_t count) {
  Mul(sq, src, src, count);
}

__aicore__ inline void SquareToFp32(const LocalTensor<float>& sq,
                                    const LocalTensor<float>& work,
                                    const LocalTensor<half>& src, int32_t count) {
  Cast(work, src, RoundMode::CAST_NONE, count);
  Mul(sq, work, work, count);
}

}  // namespace ascend
}  // namespace verse_nn

#endif  // VERSE_ASCEND_COMMON_H
