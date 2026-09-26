// KDA（Kimi Delta Attention）融合前向
//
// 逐位置递归定义：
//     S ← diag(g_t) · S                              # 逐通道门控衰减（q 空间行）
//     S ← S − β_t (S k_t) k_tᵀ + β_t v_t k_tᵀ        # delta 规则：写入并修正记忆
//     o_t = Sᵀ q_t
//
// 融合收益：Python/chunkwise 参考实现每步要启动多个 kernel 并反复读写状态
// S（D×D）；本 kernel **一次 launch 跑完整条序列**，状态 S 全程驻留共享内存，
// 中间量不出片。相比 chunkwise 的 O(S·C·D) 额外中间张量，这里显存为 O(D²)。
//
// 组织：一个 block 负责一个 (batch, head)，threads = head_dim。
//   - 更新阶段：线程 j 拥有 S 的第 j 行（行内 dot 在寄存器内完成，无需跨线程归约）；
//   - 输出阶段：线程 r 计算 o[r] = Σ_j S[j][r] q[j]（直接读共享内存的列）。
// 每步 3 次 __syncthreads（载入后 / 更新后 / 输出后），无原子操作。
//
// 与 chunkwise 参考实现的数值关系：chunkwise 用「chunk 内累计门控归一化 + 三角
// 求解」等价改写递归，二者在 fp32 下相对误差 ~1e-4（见 reference.kda_chunkwise
// 的 docstring）；本 kernel 直接计算递归定义式，数值上更接近 kda_recurrent。

#include "common/cuda_common.h"
#include "common/ops.h"

#include <algorithm>

namespace verse_nn {

template <typename T>
__global__ void kda_chunk_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const T* __restrict__ gate,
    const T* __restrict__ beta,
    T* __restrict__ out,
    float* __restrict__ state_out,
    int S, int D) {
  extern __shared__ char smem_raw[];
  float* sS = reinterpret_cast<float*>(smem_raw);   // D * D（状态，行主序 S[j][r]）
  float* sQ = sS + D * D;                           // D
  float* sK = sQ + D;                               // D
  float* sV = sK + D;                               // D
  float* sG = sV + D;                               // D
  float* sB = sG + D;                               // 1

  const int tid = threadIdx.x;
  const int64_t bh = blockIdx.x;                    // b*H + h
  const int64_t stride = static_cast<int64_t>(S) * D;

  const T* q_base = q + bh * stride;
  const T* k_base = k + bh * stride;
  const T* v_base = v + bh * stride;
  const T* g_base = gate + bh * stride;
  const T* b_base = beta + bh * static_cast<int64_t>(S);   // beta 为 (B,H,S,1)

  // 状态清零
  for (int i = tid; i < D * D; i += blockDim.x) sS[i] = 0.0f;

  T* o_base = out + bh * stride;

  for (int t = 0; t < S; ++t) {
    const int64_t off = static_cast<int64_t>(t) * D;
    for (int d = tid; d < D; d += blockDim.x) {
      sQ[d] = to_float<T>(q_base[off + d]);
      sK[d] = to_float<T>(k_base[off + d]);
      sV[d] = to_float<T>(v_base[off + d]);
      sG[d] = to_float<T>(g_base[off + d]);
    }
    if (tid == 0) sB[0] = to_float<T>(b_base[t]);
    __syncthreads();

    const float bta = sB[0];

    // ---- 更新阶段：线程 j 负责 S 的第 j 行 ----
    for (int j = tid; j < D; j += blockDim.x) {
      float* srow = sS + static_cast<int64_t>(j) * D;
      const float vj = sV[j];
      const float gj = sG[j];
      // (S k)_j = Σ_r S[j][r] k[r]
      float dot = 0.0f;
      for (int r = 0; r < D; ++r) dot = fmaf(srow[r], sK[r], dot);
      const float coef = bta * dot;
      const float write = bta * vj;
      // S ← g·S + (β·v_j − β·(Sk)_j) · k_r
      const float add = write - coef;
      for (int r = 0; r < D; ++r) {
        srow[r] = fmaf(gj, srow[r], add * sK[r]);
      }
    }
    __syncthreads();

    // ---- 输出阶段：线程 r 计算 o[r] = Σ_j S[j][r] q[j] ----
    for (int r = tid; r < D; r += blockDim.x) {
      float o = 0.0f;
      for (int j = 0; j < D; ++j) {
        o = fmaf(sS[static_cast<int64_t>(j) * D + r], sQ[j], o);
      }
      o_base[off + r] = from_float<T>(o);
    }
    __syncthreads();
  }

  // 写出末态（B,H,D,D），供推理续算/校验
  float* st = state_out + bh * static_cast<int64_t>(D) * D;
  for (int i = tid; i < D * D; i += blockDim.x) st[i] = sS[i];
}

std::vector<torch::Tensor> kda_chunk_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor gate, torch::Tensor beta, int64_t chunk_size) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && gate.is_cuda() && beta.is_cuda(),
              "kda_chunk_fwd 需要 CUDA 张量");
  TORCH_CHECK(q.dim() == 4, "q 必须是 (B,H,S,D)");
  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "k/v 形状必须与 q 一致");
  TORCH_CHECK(gate.sizes() == q.sizes(), "gate 形状必须与 q 一致");
  TORCH_CHECK(beta.numel() == B * H * S, "beta 元素数必须为 B*H*S");
  TORCH_CHECK(D <= 128, "kda_chunk_fwd 当前支持 head_dim<=128，实际 ", D);

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto gc = gate.contiguous();
  auto bc = beta.contiguous();

  auto out = torch::empty_like(qc);
  auto state = torch::empty({B, H, D, D}, qc.options().dtype(torch::kFloat32));
  if (qc.numel() == 0) return {out, state};

  const int threads = static_cast<int>(std::min<int64_t>(D, 256));
  const size_t smem = static_cast<size_t>(D * D + 4 * D + 1) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
      "kda_chunk_fwd", [&] {
        VERSE_CHECK_CUDA(cudaFuncSetAttribute(
            kda_chunk_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        kda_chunk_kernel<scalar_t><<<B * H, threads, smem, stream>>>(
            qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(),
            gc.data_ptr<scalar_t>(), bc.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), state.data_ptr<float>(),
            static_cast<int>(S), static_cast<int>(D));
      });
  VERSE_CHECK_CUDA(cudaGetLastError());
  return {out, state};
}

}  // namespace verse_nn
