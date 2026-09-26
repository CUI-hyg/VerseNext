// VerseNext 自研算子 —— CUDA / ROCm 公共头
//
// 本头文件被 csrc/cuda/*.cu 与 csrc/cuda/bindings.cpp 共同包含，负责抹平
// NVIDIA CUDA 与 AMD ROCm（HIP）的差异，使同一份源码可被 hipify 复用。
//
// 主要差异与处理：
//   - warp 宽度：NVIDIA 32，AMD wavefront 64  -> VERSE_WARP_SIZE
//   - warp 级归约：__shfl_down_sync(NVIDIA) vs __shfl_down(AMD) -> VERSE_SHFL_DOWN
//   - bf16：统一用 c10::BFloat16（torch 的 AT_DISPATCH 就是传这个类型），
//     避免 __nv_bfloat16 / hip_bfloat16 两套类型分支
//
// 编译：
//   NVIDIA: nvcc -gencode arch=compute_80,code=sm_80 -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE ...
//   AMD   : hipcc -D__HIP_PLATFORM_AMD__ --offload-arch=gfx90a ...
//
// 注意：所有 kernel 均为 **前向** 实现。反向由 Python 侧
// verse_nn/kernels/cuda_ops.py 用参考实现重放求梯度（数学严格正确），
// 避免手写反向引入静默错误。后续可在此目录补充 *_bwd kernel 替换。

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

// ---------------------------------------------------------------------------
// 平台判定与 warp 宽度
// ---------------------------------------------------------------------------
#ifdef __HIP_PLATFORM_AMD__
  #define VERSE_ROCM 1
  #define VERSE_WARP_SIZE 64          // AMD wavefront
#else
  #define VERSE_ROCM 0
  #define VERSE_WARP_SIZE 32          // NVIDIA warp
#endif

#define VERSE_WARP_MASK (VERSE_WARP_SIZE - 1)
#define VERSE_MAX_THREADS 1024

// warp 内下移/异或归约：HIP 无 _sync 后缀
#if VERSE_ROCM
  #define VERSE_SHFL_DOWN(v, off) __shfl_down(v, off)
  #define VERSE_SHFL_XOR(v, off)  __shfl_xor(v, off)
#else
  #define VERSE_SHFL_DOWN(v, off) __shfl_down_sync(0xffffffff, v, off)
  #define VERSE_SHFL_XOR(v, off)  __shfl_xor_sync(0xffffffff, v, off)
#endif

// ---------------------------------------------------------------------------
// 错误检查
// ---------------------------------------------------------------------------
#define VERSE_CHECK_CUDA(call)                                                   \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    TORCH_CHECK(err__ == cudaSuccess, "VerseNext CUDA 错误: ",                   \
                cudaGetErrorString(err__), " (", __FILE__, ":", __LINE__, ")");  \
  } while (0)

// ---------------------------------------------------------------------------
// 标量 <-> fp32 转换
//
// AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16, ...) 会实例化
// scalar_t ∈ {double, float, at::Half, c10::BFloat16}，四者都必须有定义，
// 否则 ptxas 会报 "Unresolved extern function"。
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);          // float / double
}

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<c10::BFloat16>(c10::BFloat16 v) {
  return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  return static_cast<T>(v);              // float / double
}

template <>
__device__ __forceinline__ at::Half from_float<at::Half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ c10::BFloat16 from_float<c10::BFloat16>(float v) {
  return c10::BFloat16(v);
}

// ---------------------------------------------------------------------------
// 块内归约（sum），结果广播到全 block
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float block_reduce_sum(T val, float* shared) {
  const int tid = threadIdx.x;
  const int lane = tid & VERSE_WARP_MASK;
  const int wid = tid / VERSE_WARP_SIZE;
  float v = to_float<T>(val);
#pragma unroll
  for (int off = VERSE_WARP_SIZE / 2; off > 0; off >>= 1) {
    v += VERSE_SHFL_DOWN(v, off);
  }
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  const int n_warps = (blockDim.x + VERSE_WARP_SIZE - 1) / VERSE_WARP_SIZE;
  if (wid == 0) {
    v = (lane < n_warps) ? shared[lane] : 0.0f;
#pragma unroll
    for (int off = VERSE_WARP_SIZE / 2; off > 0; off >>= 1) {
      v += VERSE_SHFL_DOWN(v, off);
    }
    if (lane == 0) shared[0] = v;
  }
  __syncthreads();
  return shared[0];
}

// ---------------------------------------------------------------------------
// 启动配置辅助：给定元素数选择 grid
// ---------------------------------------------------------------------------
inline int verse_num_blocks(int64_t n, int threads) {
  return static_cast<int>((n + threads - 1) / threads);
}
