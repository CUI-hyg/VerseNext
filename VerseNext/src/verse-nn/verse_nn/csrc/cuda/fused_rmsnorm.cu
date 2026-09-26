// 融合残差加 + RMSNorm（前向）
//
// 语义：res_out = x + residual; out = rms_norm(res_out) * weight
//   rms_norm(v) = v / sqrt(mean(v^2) + eps)
//
// 融合收益：原本需要 2 次全量读写（残差加一次、归一化一次），融合后
// res_out 写入后立即在片上复用，省掉一次全局往返；配合块内归约，
// 每行只需一次 read(x) + read(residual) + write(res_out) + write(out)。
//
// 布局：(rows, D) 连续，weight 为 (D,)。一行一个 block。
// 精度：平方和与归一化在 fp32 下累加，避免 fp16/bf16 的归约溢出与精度损失。

#include "common/cuda_common.h"
#include "common/ops.h"

namespace verse_nn {

template <typename T, int THREADS>
__global__ void add_rms_norm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ out,
    T* __restrict__ res_out,
    int64_t rows,
    int D,
    float eps) {
  __shared__ float sdata[32];
  const int64_t row = blockIdx.x;
  if (row >= rows) return;

  const T* x_row = x + row * D;
  const T* r_row = residual + row * D;
  T* o_row = out + row * D;
  T* ro_row = res_out + row * D;

  // 1) 残差相加（写回 res_out）并累加平方和
  float local = 0.0f;
  for (int i = threadIdx.x; i < D; i += THREADS) {
    const float xv = to_float<T>(x_row[i]);
    const float rv = to_float<T>(r_row[i]);
    const float s = xv + rv;
    ro_row[i] = from_float<T>(s);
    local = fmaf(s, s, local);
  }
  const float sum_sq = block_reduce_sum<float>(local, sdata);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(D) + eps);

  // 2) 归一化并乘 weight（复用刚写入的 res_out，命中 L2）
  for (int i = threadIdx.x; i < D; i += THREADS) {
    const float s = to_float<T>(ro_row[i]);
    const float w = to_float<T>(weight[i]);
    o_row[i] = from_float<T>(s * inv_rms * w);
  }
}

std::vector<torch::Tensor> add_rms_norm_fwd(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(),
              "add_rms_norm_fwd 需要 CUDA 张量");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x 与 residual 形状必须一致");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(), "x 与 residual dtype 必须一致");

  auto xc = x.contiguous();
  auto rc = residual.contiguous();
  auto wc = weight.contiguous();
  const int64_t D = xc.size(-1);
  const int64_t rows = xc.numel() / D;
  TORCH_CHECK(wc.numel() == D, "weight 长度必须等于最后一维");

  auto out = torch::empty_like(xc);
  auto res_out = torch::empty_like(xc);
  if (rows == 0) return {out, res_out};

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS = 256;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, xc.scalar_type(),
      "add_rms_norm_fwd", [&] {
        add_rms_norm_kernel<scalar_t, THREADS><<<rows, THREADS, 0, stream>>>(
            xc.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
            wc.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            res_out.data_ptr<scalar_t>(), rows, static_cast<int>(D),
            static_cast<float>(eps));
      });
  VERSE_CHECK_CUDA(cudaGetLastError());
  return {out, res_out};
}

}  // namespace verse_nn
