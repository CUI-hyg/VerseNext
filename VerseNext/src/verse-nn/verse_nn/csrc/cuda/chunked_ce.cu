// 融合 lm_head + 交叉熵（前向）
//
// 语义：loss = mean_over_valid( CE(hidden @ weight^T, targets) )
//
// 融合收益：标准做法先物化 (B, S, V) 的 logits（V=32k 时单个样本就上百 MB），
// 再算 CE。本 kernel 把「vocab 投影」与「logsumexp」融合，**从不物化 logits**：
//   - 每行 hidden 的 vocab 投影在块内流式完成；
//   - 用在线 logsumexp 维护 (running max, running sum)，跨线程做标准合并；
//   - 目标位置的 logit 单独记录，最后 loss = m + log(sum) - logit_target。
// 显存占用从 O(B·S·V) 降到 O(B·S)。
//
// 组织：一个 block 负责一行（B*S 行）；hidden 行放入共享内存供全块复用。
// 说明：本版本是 SIMT GEMV 式实现（未用 tensor core）。进一步优化方向是
// 「分块 GEMM + 分块 softmax」的 2D 分块版本，见 docs/kernels.md。

#include "common/cuda_common.h"
#include "common/ops.h"

namespace verse_nn {

template <typename T>
__global__ void chunked_ce_kernel(
    const T* __restrict__ hidden,
    const T* __restrict__ weight,
    const int64_t* __restrict__ targets,
    float* __restrict__ row_loss,
    float* __restrict__ row_valid,
    int64_t rows, int V, int D, int64_t ignore_index) {
  extern __shared__ char smem_raw[];
  float* sH = reinterpret_cast<float*>(smem_raw);          // D
  float* sM = sH + D;                                       // blockDim.x
  float* sS = sM + blockDim.x;                              // blockDim.x
  float* sTgt = sS + blockDim.x;                            // 1

  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;

  // hidden 行常驻共享内存：块内所有线程反复读取
  const T* h_row = hidden + row * D;
  for (int d = tid; d < D; d += blockDim.x) {
    sH[d] = to_float<T>(h_row[d]);
  }
  if (tid == 0) sTgt[0] = 0.0f;
  __syncthreads();

  const int64_t tgt = targets[row];
  float m = -INFINITY;
  float s = 0.0f;

  for (int v = tid; v < V; v += blockDim.x) {
    const T* w_row = weight + static_cast<int64_t>(v) * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot = fmaf(sH[d], to_float<T>(w_row[d]), dot);
    }
    if (v == tgt) sTgt[0] = dot;            // 唯一写者：只有 v==tgt 的线程
    // 在线 logsumexp：首项时 m=-inf，scale 取 0 避免 (-inf) - (-inf) = NaN
    const float m_new = fmaxf(m, dot);
    const float scale = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
    s = s * scale + __expf(dot - m_new);
    m = m_new;
  }

  sM[tid] = m;
  sS[tid] = s;
  __syncthreads();

  if (tid == 0) {
    float M = -INFINITY;
    for (int i = 0; i < blockDim.x; ++i) M = fmaxf(M, sM[i]);
    float S = 0.0f;
    if (M != -INFINITY) {
      for (int i = 0; i < blockDim.x; ++i) {
        if (sM[i] != -INFINITY) S += sS[i] * __expf(sM[i] - M);
      }
    }
    if (tgt != ignore_index && M != -INFINITY && S > 0.0f) {
      row_loss[row] = M + logf(S) - sTgt[0];
      row_valid[row] = 1.0f;
    } else {
      row_loss[row] = 0.0f;
      row_valid[row] = 0.0f;
    }
  }
}

torch::Tensor chunked_ce_fwd(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor targets,
    int64_t chunk_size, int64_t ignore_index) {
  TORCH_CHECK(hidden.is_cuda() && weight.is_cuda() && targets.is_cuda(),
              "chunked_ce_fwd 需要 CUDA 张量");
  TORCH_CHECK(hidden.dim() == 3, "hidden 必须是 (B, S, D)");
  TORCH_CHECK(weight.dim() == 2, "weight 必须是 (V, D)");

  const int64_t B = hidden.size(0), S = hidden.size(1), D = hidden.size(2);
  const int64_t V = weight.size(0);
  TORCH_CHECK(weight.size(1) == D, "weight 第二维必须等于 hidden 的 D");

  auto h = hidden.contiguous().view({B * S, D});
  auto w = weight.contiguous();
  auto t = targets.contiguous().view({-1}).to(torch::kLong);
  TORCH_CHECK(t.numel() == B * S, "targets 元素数必须等于 B*S");

  const int64_t rows = B * S;
  if (rows == 0) return torch::zeros({}, hidden.options().dtype(torch::kFloat32));

  auto opts_f = hidden.options().dtype(torch::kFloat32);
  auto row_loss = torch::empty({rows}, opts_f);
  auto row_valid = torch::empty({rows}, opts_f);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS = 256;
  const size_t smem = static_cast<size_t>(D) * sizeof(float) + 2 * THREADS * sizeof(float)
                      + sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, h.scalar_type(),
      "chunked_ce_fwd", [&] {
        VERSE_CHECK_CUDA(cudaFuncSetAttribute(
            chunked_ce_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        chunked_ce_kernel<scalar_t><<<rows, THREADS, smem, stream>>>(
            h.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            t.data_ptr<int64_t>(), row_loss.data_ptr<float>(),
            row_valid.data_ptr<float>(), rows, static_cast<int>(V),
            static_cast<int>(D), ignore_index);
      });
  VERSE_CHECK_CUDA(cudaGetLastError());

  // 与参考实现一致的加权平均：sum(loss) / sum(valid)
  const auto valid = row_valid.sum();
  return row_loss.sum() / valid.clamp_min(1.0);
}

}  // namespace verse_nn
