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
 * 编译：本目录需用 CANN 的 msopgen/自定义算子工程编译（见 build.sh），
 *       产出 ACLNN 自定义算子包，再由 bindings_npu.cpp 经 aclnn 接口调用。
 *
 * ⚠ 本目录代码在无 CANN 工具链的机器上**未经编译验证**，属"待硬件验证"交付，
 *   详见 docs/kernels.md。API 用法遵循 AscendC 官方惯例，但具体版本
 *   （CANN 7.x/8.x）签名可能有差异，首次编译时需按报错微调。
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
// 逐元素一元/二元算子（Vector 单元）
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void SiluMul(const LocalTensor<T>& dst,
                               const LocalTensor<T>& gate,
                               const LocalTensor<T>& up,
                               int32_t count) {
  // silu(g) * u = g * sigmoid(g) * u
  LocalTensor<T> sig = dst;                        // 复用 dst 暂存 sigmoid
  Sigmoid(sig, gate, count);
  Mul(sig, sig, gate, count);
  Mul(dst, sig, up, count);
}

// ---------------------------------------------------------------------------
// RMSNorm 的核内归约：对每个 (row, tile) 求平方和
//   AscendC 的 ReduceSum 支持按 last-dim 归约
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void SumOfSquares(const LocalTensor<float>& dst,
                                    const LocalTensor<T>& src,
                                    const LocalTensor<float>& work,
                                    int32_t count) {
  Cast(work, src, RoundMode::CAST_NONE, count);
  Mul(work, work, work, count);
  ReduceSum(dst, work, work, count);
}

}  // namespace ascend
}  // namespace verse_nn

#endif  // VERSE_ASCEND_COMMON_H
