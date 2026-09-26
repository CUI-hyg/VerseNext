// 融合 SwiGLU（前向）：out = silu(gate) * up
//
// silu(x) = x * sigmoid(x)。原本需要 sigmoid 中间张量 + 两次逐元素 kernel，
// 融合为一次访存：读 gate/up，写 out。
//
// 访存优化：float 路径走 float4 向量化（16B/线程/次），其余类型走标量
// grid-stride 循环 + 展开。该 kernel 是纯 memory-bound，向量化即可接近带宽上限。

#include "common/cuda_common.h"
#include "common/ops.h"

namespace verse_nn {

template <typename T>
__device__ __forceinline__ T swiglu_op(T g, T u) {
  const float gf = to_float<T>(g);
  const float silu = gf / (1.0f + expf(-gf));
  return from_float<T>(silu * to_float<T>(u));
}

// 标量 grid-stride（fp16/bf16/fp64 及非对齐尾部）
template <typename T>
__global__ void swiglu_scalar_kernel(
    const T* __restrict__ gate, const T* __restrict__ up,
    T* __restrict__ out, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    out[i] = swiglu_op<T>(gate[i], up[i]);
  }
}

// float4 向量化路径
__global__ void swiglu_vec4_kernel(
    const float4* __restrict__ gate, const float4* __restrict__ up,
    float4* __restrict__ out, int64_t n4) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    const float4 g = gate[i];
    const float4 u = up[i];
    float4 o;
    o.x = (g.x / (1.0f + expf(-g.x))) * u.x;
    o.y = (g.y / (1.0f + expf(-g.y))) * u.y;
    o.z = (g.z / (1.0f + expf(-g.z))) * u.z;
    o.w = (g.w / (1.0f + expf(-g.w))) * u.w;
    out[i] = o;
  }
}

torch::Tensor swiglu_fwd(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "swiglu_fwd 需要 CUDA 张量");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate 与 up 形状必须一致");
  auto g = gate.contiguous();
  auto u = up.contiguous();
  auto out = torch::empty_like(g);
  const int64_t n = g.numel();
  if (n == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS = 256;

  if (g.scalar_type() == at::kFloat &&
      (n % 4 == 0) &&
      (reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(u.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)) {
    const int64_t n4 = n / 4;
    const int blocks = verse_num_blocks(n4, THREADS);
    swiglu_vec4_kernel<<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(g.data_ptr<float>()),
        reinterpret_cast<const float4*>(u.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, g.scalar_type(),
        "swiglu_fwd", [&] {
          const int blocks = verse_num_blocks(n, THREADS);
          swiglu_scalar_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
              g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(), n);
        });
  }
  VERSE_CHECK_CUDA(cudaGetLastError());
  return out;
}

}  // namespace verse_nn
