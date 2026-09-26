// 融合 RoPE（前向），语义与 verse_nn/rope.py 的 apply_rope 严格一致：
//
//   out = x * cos + rotate_half(x) * sin
//   rotate_half(x) = cat([-x2, x1]),  x1 = x[..., :D/2], x2 = x[..., D/2:]
//
// 展开后（c/s 只取前一半，因为 RotaryEmbedding 的 cos/sin 两半相同）：
//   out[i]     = x1[i] * c[i] - x2[i] * s[i]
//   out[i+H]   = x2[i] * c[i] + x1[i] * s[i]      (H = D/2)
//
// 融合收益：原本 q*cos、rotate_half、两次乘加、相加共 4~5 次逐元素 kernel，
// 融合为一次读 q/k、一次写 q/k，中间量全部留在寄存器。
//
// 一个线程处理一个 (b, h, s, i)，i ∈ [0, D/2)；一次读写覆盖 D/2 个位置。

#include "common/cuda_common.h"
#include "common/ops.h"

namespace verse_nn {

template <typename T>
__global__ void rope_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ cos, const T* __restrict__ sin,
    T* __restrict__ q_out, T* __restrict__ k_out,
    int64_t total,          // B*H*S*(D/2)
    int S, int D,
    int64_t cos_stride) {   // cos/sin 的最后一维（D 或 D/2）
  const int half = D / 2;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int i = static_cast<int>(idx % half);
    const int64_t row = idx / half;          // b*H*S + h*S + s
    const int s = static_cast<int>(row % S);
    const int64_t base = row * D;

    const float c = to_float<T>(cos[s * cos_stride + i]);
    const float sn = to_float<T>(sin[s * cos_stride + i]);

    // q
    {
      const float x1 = to_float<T>(q[base + i]);
      const float x2 = to_float<T>(q[base + i + half]);
      q_out[base + i] = from_float<T>(x1 * c - x2 * sn);
      q_out[base + i + half] = from_float<T>(x2 * c + x1 * sn);
    }
    // k
    {
      const float x1 = to_float<T>(k[base + i]);
      const float x2 = to_float<T>(k[base + i + half]);
      k_out[base + i] = from_float<T>(x1 * c - x2 * sn);
      k_out[base + i + half] = from_float<T>(x2 * c + x1 * sn);
    }
  }
}

std::vector<torch::Tensor> rope_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda(),
              "rope_fwd 需要 CUDA 张量");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k 必须是 (B,H,S,D)");
  TORCH_CHECK(q.sizes() == k.sizes(), "q 与 k 形状必须一致");
  const int64_t D = q.size(3);
  TORCH_CHECK(D % 2 == 0, "head_dim 必须为偶数");
  const int64_t S = q.size(2);
  TORCH_CHECK(cos.numel() >= S * (D / 2), "cos/sin 尺寸不足");
  // cos/sin 可能是 (S, D)（仓库默认）或 (S, D/2)
  const int64_t cos_stride = cos.size(-1);
  TORCH_CHECK(cos_stride == D || cos_stride == D / 2,
              "cos/sin 最后一维应为 D 或 D/2，实际 ", cos_stride);

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto cc = cos.contiguous();
  auto sc = sin.contiguous();
  auto q_out = torch::empty_like(qc);
  auto k_out = torch::empty_like(kc);

  const int64_t total = qc.numel() / 2;      // B*H*S*(D/2)
  if (total == 0) return {q_out, k_out};

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS = 256;
  const int blocks = verse_num_blocks(total, THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
      "rope_fwd", [&] {
        rope_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
            qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(),
            cc.data_ptr<scalar_t>(), sc.data_ptr<scalar_t>(),
            q_out.data_ptr<scalar_t>(), k_out.data_ptr<scalar_t>(),
            total, static_cast<int>(S), static_cast<int>(D), cos_stride);
      });
  VERSE_CHECK_CUDA(cudaGetLastError());
  return {q_out, k_out};
}

}  // namespace verse_nn
