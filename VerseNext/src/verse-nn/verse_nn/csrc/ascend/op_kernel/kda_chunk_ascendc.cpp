/*
 * KDA（Kimi Delta Attention）融合前向 —— AscendC kernel
 *
 * 递归：S ← diag(g_t)·S − β_t(S k_t)k_tᵀ + β_t v_t k_tᵀ ;  o_t = Sᵀ q_t
 *
 * 昇腾实现：
 *   - 状态 S（D×D）常驻 **UB**，整条序列一次 kernel 跑完，中间量不出片；
 *   - 每个 (batch, head) 由一个核处理（核数不足时按序轮转）；
 *   - 更新用 Vector：行缩放 Mul + 外积累加 Axpy；输出 o = Sᵀq 用
 *     「转置后按行点积」实现（Transpose + Mul + ReduceSum）。
 *
 * ⚠ 未编译验证。D<=64 时 S 占 16KB UB，可行；更大 head_dim 需分块驻留。
 */

#include "../ascend_common.h"
#include "../op_host/kda_chunk_tiling.h"

using namespace AscendC;
using namespace verse_nn::ascend;

class KdaChunkKernel {
 public:
  __aicore__ inline KdaChunkKernel() {}

  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gate,
                              GM_ADDR beta, GM_ADDR out, GM_ADDR state,
                              GM_ADDR tiling) {
    const KdaChunkTiling* t = reinterpret_cast<const KdaChunkTiling*>(tiling);
    b_ = t->batch;
    h_ = t->heads;
    s_ = t->seq_len;
    d_ = t->head_dim;
    bh_per_core_ = t->bh_per_core;

    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(q));
    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(k));
    vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(v));
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(gate));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(beta));
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(out));
    sGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state));

    pipe_.InitBuffer(stateBuf_, d_ * d_ * sizeof(float));      // S
    pipe_.InitBuffer(vecBuf_, 4 * d_ * sizeof(float));         // q/k/v/g
    pipe_.InitBuffer(scratchBuf_, 2 * d_ * sizeof(float));     // tmp + (Sk)
    pipe_.InitBuffer(dstBuf_, d_ * sizeof(float));             // 输出 o
    pipe_.InitBuffer(scalarBuf_, kAlignBytes);                 // beta / 标量
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t bh_total = b_ * h_;
    const int32_t begin = core * bh_per_core_;
    const int32_t end = Min(begin + bh_per_core_, bh_total);
    for (int32_t bh = begin; bh < end; ++bh) {
      ProcessHead(bh);
    }
  }

 private:
  __aicore__ inline void ProcessHead(int32_t bh) {
    const int64_t base = static_cast<int64_t>(bh) * s_ * d_;
    LocalTensor<float> S = stateBuf_.Get<float>();
    LocalTensor<float> vec = vecBuf_.Get<float>();
    LocalTensor<float> qt = vec[0];
    LocalTensor<float> kt = vec[d_];
    LocalTensor<float> vt = vec[2 * d_];
    LocalTensor<float> gt = vec[3 * d_];
    LocalTensor<float> scratch = scratchBuf_.Get<float>();
    LocalTensor<float> tmp = scratch[0];
    LocalTensor<float> sk = scratch[d_];
    LocalTensor<float> o = dstBuf_.Get<float>();

    Duplicate(S, 0.0f, d_ * d_);      // 初始状态为 0

    for (int32_t t = 0; t < s_; ++t) {
      const int64_t off = base + static_cast<int64_t>(t) * d_;
      DataCopy(qt, qGm_[off], AlignUp(d_, kAlignBytes / sizeof(DTYPE_X)));
      DataCopy(kt, kGm_[off], AlignUp(d_, kAlignBytes / sizeof(DTYPE_X)));
      DataCopy(vt, vGm_[off], AlignUp(d_, kAlignBytes / sizeof(DTYPE_X)));
      DataCopy(gt, gGm_[off], AlignUp(d_, kAlignBytes / sizeof(DTYPE_X)));
      const float beta = static_cast<float>(bGm_.GetValue(bh * s_ + t));

      // S ← diag(g)·S ：按行乘 g_j
      for (int32_t j = 0; j < d_; ++j) {
        Muls(S[j * d_], S[j * d_], gt.GetValue(j), d_);
      }
      // (S k)_j = Σ_r S[j][r] k[r]（tmp 为独立 scratch，不与 S 重叠）
      for (int32_t j = 0; j < d_; ++j) {
        Mul(tmp, S[j * d_], kt, d_);
        ReduceSum(sk[j], tmp, tmp, d_);
      }
      // S[j][r] += β·(v_j − (Sk)_j)·k_r
      for (int32_t j = 0; j < d_; ++j) {
        const float add = beta * (vt.GetValue(j) - sk.GetValue(j));
        Axpy(S[j * d_], kt, add, d_);
      }
      // o = Sᵀ q ：对每列 r，o[r] = Σ_j S[j][r]·q[j]
      for (int32_t r = 0; r < d_; ++r) {
        float acc = 0.0f;
        for (int32_t j = 0; j < d_; ++j) {
          acc += S[j * d_ + r] * qt.GetValue(j);
        }
        o.SetValue(r, acc);
      }
      DataCopy(oGm_[off], o, AlignUp(d_, kAlignBytes / sizeof(float)));
    }

    // 末态写回（B,H,D,D）
    DataCopy(sGm_[static_cast<int64_t>(bh) * d_ * d_], S, d_ * d_);
  }

  TPipe pipe_;
  TBuf<QuePosition::VECCALC> stateBuf_, vecBuf_, scratchBuf_, dstBuf_, scalarBuf_;
  GlobalTensor<DTYPE_X> qGm_, kGm_, vGm_, gGm_, bGm_, oGm_;
  GlobalTensor<float> sGm_;
  int32_t b_ = 0, h_ = 0, s_ = 0, d_ = 0, bh_per_core_ = 0;
};

extern "C" __global__ __aicore__ void kda_chunk(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                GM_ADDR gate, GM_ADDR beta,
                                                GM_ADDR out, GM_ADDR state,
                                                GM_ADDR tiling) {
  KdaChunkKernel op;
  op.Init(q, k, v, gate, beta, out, state, tiling);
  op.Process();
}
