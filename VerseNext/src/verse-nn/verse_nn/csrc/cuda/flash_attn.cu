// 融合注意力（前向）：分块在线 softmax（FlashAttention 思路）
//
// 核心收益：不物化 (S, T) 的注意力矩阵。标准实现需要 S*T*sizeof(T) 的中间
// 显存（长上下文下 O(S²) 爆炸）并来回读写；本实现把 K/V 分块搬入共享内存，
// 用「在线 softmax」（维护 running max m 与 running sum l）逐块累积输出，
// 显存占用从 O(S·T) 降到 O(S·D)。
//
// 线程组织：一个 block 负责 Br 个 query 行 × 一个 (batch, head)；
// 每个线程拥有 1 个 query 行，q 与输出累加 acc 常驻寄存器，K/V 走共享内存。
//
// 说明：本版本为**正确的 SIMT 实现**，未使用 tensor core（WMMA/MFMA）。
// 下一步优化方向：把 D 维按 16×16 分块交给 WMMA/MFMA 做 q·k^T 与 p·v，
// 可再获 2~4× 吞吐（见 docs/kernels.md 的「已知限制」）。
//
// 精度：softmax 统计与累加全程 fp32；输入可为 fp32/fp16/bf16。

#include "common/cuda_common.h"
#include "common/ops.h"

namespace verse_nn {

template <typename T, int D, int Br, int Bc>
__global__ void flash_attn_fwd_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ out,
    int S, int T_len, float scale, bool causal) {
  extern __shared__ char smem_raw[];
  T* sK = reinterpret_cast<T*>(smem_raw);
  T* sV = sK + Bc * D;

  const int tid = threadIdx.x;                       // 0..Br-1
  const int64_t bh = blockIdx.y;                     // b*H + h
  const int q_row = blockIdx.x * Br + tid;

  const T* q_base = q + bh * static_cast<int64_t>(S) * D;
  const T* k_base = k + bh * static_cast<int64_t>(T_len) * D;
  const T* v_base = v + bh * static_cast<int64_t>(T_len) * D;
  T* o_base = out + bh * static_cast<int64_t>(S) * D;

  float q_reg[D];
  float acc[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    q_reg[d] = 0.0f;
    acc[d] = 0.0f;
  }
  const bool valid = q_row < S;
  if (valid) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      q_reg[d] = to_float<T>(q_base[static_cast<int64_t>(q_row) * D + d]);
    }
  }

  float m_i = -INFINITY;    // running max（用 -inf 表示「尚无有效项」）
  float l_i = 0.0f;         // running sum of exp
  const int n_tiles = (T_len + Bc - 1) / Bc;

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int kv0 = tile * Bc;

    // ---- 协作加载 K/V 分块到共享内存（越界补 0）----
    for (int idx = tid; idx < Bc * D; idx += Br) {
      const int r = idx / D;
      const int d = idx % D;
      const int kvr = kv0 + r;
      if (kvr < T_len) {
        sK[idx] = k_base[static_cast<int64_t>(kvr) * D + d];
        sV[idx] = v_base[static_cast<int64_t>(kvr) * D + d];
      } else {
        sK[idx] = from_float<T>(0.0f);
        sV[idx] = from_float<T>(0.0f);
      }
    }
    __syncthreads();

    if (valid) {
      // ---- 1) 计算本块的 scores 并求块内最大值 ----
      float scores[Bc];
      float m_new = -INFINITY;
#pragma unroll
      for (int j = 0; j < Bc; ++j) {
        const int kvr = kv0 + j;
        float dot = -INFINITY;
        if (kvr < T_len && (!causal || kvr <= q_row)) {
          float acc_dot = 0.0f;
#pragma unroll
          for (int d = 0; d < D; ++d) {
            acc_dot = fmaf(q_reg[d], to_float<T>(sK[j * D + d]), acc_dot);
          }
          dot = acc_dot * scale;
        }
        scores[j] = dot;
        m_new = fmaxf(m_new, dot);
      }

      // ---- 2) 在线 softmax：重标定历史累加 ----
      const float alpha = (m_new == -INFINITY) ? 1.0f : __expf(m_i - m_new);
      float l_new = 0.0f;
#pragma unroll
      for (int j = 0; j < Bc; ++j) {
        if (scores[j] != -INFINITY) {
          l_new += __expf(scores[j] - m_new);
        }
      }
#pragma unroll
      for (int d = 0; d < D; ++d) {
        acc[d] *= alpha;
      }

      // ---- 3) 累加本块的 p·V ----
#pragma unroll
      for (int j = 0; j < Bc; ++j) {
        const int kvr = kv0 + j;
        if (kvr < T_len && scores[j] != -INFINITY) {
          const float p = __expf(scores[j] - m_new);
#pragma unroll
          for (int d = 0; d < D; ++d) {
            acc[d] = fmaf(p, to_float<T>(sV[j * D + d]), acc[d]);
          }
        }
      }
      m_i = m_new;
      l_i = l_i * alpha + l_new;
    }
    __syncthreads();   // 防止下一轮覆盖 sK/sV 时仍有线程在读
  }

  if (valid) {
    const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d) {
      o_base[static_cast<int64_t>(q_row) * D + d] = from_float<T>(acc[d] * inv);
    }
  }
}

namespace {

// 按 head_dim 选择模板实例；不支持的 head_dim 抛异常（Python 侧回退 SDPA）
template <typename T>
void launch_flash(const at::Tensor& qc, const at::Tensor& kc, const at::Tensor& vc,
                  at::Tensor& out, int B, int H, int S, int T_len, int D,
                  float scale, bool causal, cudaStream_t stream) {
  constexpr int Br = 64;
  constexpr int Bc = 32;
  const int n_q_tiles = (S + Br - 1) / Br;
  dim3 grid(n_q_tiles, B * H);
  dim3 block(Br);

  auto* qp = qc.data_ptr<T>();
  auto* kp = kc.data_ptr<T>();
  auto* vp = vc.data_ptr<T>();
  auto* op = out.data_ptr<T>();

#define VERSE_LAUNCH_FLASH(DIM)                                                    \
  do {                                                                             \
    const size_t smem = 2 * Bc * (DIM) * sizeof(T);                                \
    VERSE_CHECK_CUDA(cudaFuncSetAttribute(                                         \
        flash_attn_fwd_kernel<T, DIM, Br, Bc>,                                     \
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));     \
    flash_attn_fwd_kernel<T, DIM, Br, Bc><<<grid, block, smem, stream>>>(          \
        qp, kp, vp, op, S, T_len, scale, causal);                                  \
  } while (0)

  switch (D) {
    case 32: VERSE_LAUNCH_FLASH(32); break;
    case 64: VERSE_LAUNCH_FLASH(64); break;
    case 128: VERSE_LAUNCH_FLASH(128); break;
    default:
      TORCH_CHECK(false, "flash_attn 暂不支持 head_dim=", D,
                  "（支持 32/64/128）；请回退 SDPA");
  }
#undef VERSE_LAUNCH_FLASH
}

}  // namespace

torch::Tensor flash_attn_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "flash_attn_fwd 需要 CUDA 张量");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v 必须是 (B,H,S,D)");
  const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  const int T_len = k.size(2);
  TORCH_CHECK(k.size(0) == B && k.size(1) == H && v.size(0) == B && v.size(1) == H,
              "q/k/v 的 batch/head 维必须一致");
  TORCH_CHECK(v.size(3) == D, "v 的 head_dim 必须与 q 一致");
  TORCH_CHECK(k.size(3) == D, "k 的 head_dim 必须与 q 一致");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto out = torch::empty_like(qc);
  if (qc.numel() == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
      "flash_attn_fwd", [&] {
        launch_flash<scalar_t>(qc, kc, vc, out, B, H, S, T_len, D,
                               static_cast<float>(scale), causal, stream);
      });
  VERSE_CHECK_CUDA(cudaGetLastError());
  return out;
}

}  // namespace verse_nn
