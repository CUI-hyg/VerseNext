/*
 * 融合注意力 —— AscendC kernel（分块在线 softmax）
 *
 * 结构（与 CUDA 版同构，但用昇腾的 Cube/Vector 分工）：
 *   - QK^T 与 PV 走 **Cube（矩阵）单元**：AscendC 的 matmul::Matmul API，
 *     由 CANN 自动做分形（fractal）排布与 L0C/L1 调度；
 *   - softmax 的 exp / 行 max / 行 sum 走 **Vector 单元**；
 *   - K/V 按 tile 搬入 UB/L1，输出用在线 softmax 累积，避免物化 (S,T) 矩阵。
 *
 * ⚠ 未编译验证：本实现依赖 matmul::Matmul 的 tiling（TCubeTiling 由 host 下发），
 *   CANN 7.x/8.x 的 API 名称与参数可能有差异。首次在昇腾上编译时请按报错调整，
 *   参考 CANN 官方 "FlashAttention 算子样例"。
 *
 * 接入前必读：docs/kernels.md 的「踩过的坑」——本文件尚未按那几条改造
 * （缺 REGISTER_TILING_DEFAULT、用 TILING_KEY_IS 做 dtype 分派），
 * 且未纳入 build.sh 的 OPS 列表。
 *
 * 资源锁 50%：block_dim 由 host 按 AI 核数 × 0.5 下发。
 */

#include "../ascend_common.h"
#include "../op_host/flash_attn_tiling.h"

#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;
using namespace verse_nn::ascend;

class FlashAttnKernel {
 public:
  using A_T = DTYPE_X;
  using B_T = DTYPE_X;
  using C_T = float;

  __aicore__ inline FlashAttnKernel() {}

  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                              GM_ADDR tiling, GM_ADDR tiling_matmul) {
    const __gm__ FlashAttnTiling* t = ReadTiling<FlashAttnTiling>(tiling);
    b_ = t->batch;
    h_ = t->heads;
    s_ = t->seq_len;
    t_len_ = t->kv_len;
    d_ = t->head_dim;
    br_ = t->q_tile;
    bc_ = t->kv_tile;
    scale_ = t->scale;
    causal_ = t->causal != 0;
    q_tiles_per_core_ = t->q_tiles_per_core;

    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(q));
    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(k));
    vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(v));
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(out));

    // 在线 softmax 的中间量：scores(Br×Bc)、行 max/sum(Br)、输出累加(Br×D)
    pipe_.InitBuffer(scoreBuf_, br_ * bc_ * sizeof(float));
    pipe_.InitBuffer(rowBuf_, 2 * br_ * sizeof(float));
    pipe_.InitBuffer(accBuf_, br_ * d_ * sizeof(float));
    pipe_.InitBuffer(kvQue_, 1, 2 * bc_ * d_ * sizeof(B_T));

    REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mmQK_, tiling_matmul);
    REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mmPV_, tiling_matmul);
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t bh_total = b_ * h_;
    const int32_t q_tiles = CeilDiv(s_, br_);
    // 以 (bh, q_tile) 为工作单元，按核切分
    const int32_t total_units = bh_total * q_tiles;
    const int32_t unit_begin = core * q_tiles_per_core_;
    const int32_t unit_end = MinI(unit_begin + q_tiles_per_core_, total_units);

    for (int32_t unit = unit_begin; unit < unit_end; ++unit) {
      const int32_t bh = unit / q_tiles;
      const int32_t qt = unit % q_tiles;
      ProcessTile(bh, qt);
    }
  }

 private:
  __aicore__ inline void ProcessTile(int32_t bh, int32_t qt) {
    const int32_t b = bh / h_;
    const int32_t h = bh % h_;
    const int64_t q_base = (static_cast<int64_t>(b) * h_ + h) * s_ * d_;
    const int64_t kv_base = (static_cast<int64_t>(b) * h_ + h) * t_len_ * d_;
    const int32_t q0 = qt * br_;
    const int32_t n_kv_tiles = CeilDiv(t_len_, bc_);

    LocalTensor<float> acc = accBuf_.Get<float>();
    LocalTensor<float> row = rowBuf_.Get<float>();
    LocalTensor<float> m_i = row[0];
    LocalTensor<float> l_i = row[br_];

    Duplicate(acc, 0.0f, br_ * d_);
    Duplicate(m_i, -3.0e38f, br_);
    Duplicate(l_i, 0.0f, br_);

    for (int32_t kt = 0; kt < n_kv_tiles; ++kt) {
      const int32_t kv0 = kt * bc_;
      // 载入 K/V tile（bc × d）到 UB
      LocalTensor<B_T> kv = kvQue_.AllocTensor<B_T>();
      DataCopy(kv, kGm_[kv_base + static_cast<int64_t>(kv0) * d_],
               AlignUp(bc_ * d_, kAlignBytes / sizeof(B_T)));

      // QK^T -> scores(Br × Bc)，Cube
      LocalTensor<float> scores = scoreBuf_.Get<float>();
      mmQK_.SetTensorA(qGm_[q_base + static_cast<int64_t>(q0) * d_], false);
      mmQK_.SetTensorB(kv, true);
      mmQK_.IterateAll(scores);
      mmQK_.End();

      // ---- softmax（Vector）：scale、因果掩码、行 max、exp、行 sum ----
      Muls(scores, scores, scale_, br_ * bc_);
      if (causal_) ApplyCausalMask(scores, q0, kv0);
      SoftmaxOnline(scores, m_i, l_i, br_, bc_);

      // ---- PV：scores(Br×Bc) @ V(Bc×D) -> acc(Br×D)，Cube ----
      // V 需按 bc×d 排布；PV 的 B 矩阵为 V
      mmPV_.SetTensorA(scores, false);
      mmPV_.SetTensorB(kv, false);
      mmPV_.IterateAll(acc);
      mmPV_.End();

      kvQue_.FreeTensor(kv);
    }

    // acc /= l_i
    ApplyRowScale(acc, l_i, br_, d_);
    DataCopy(oGm_[q_base + static_cast<int64_t>(q0) * d_], acc,
             AlignUp(br_ * d_, kAlignBytes / sizeof(float)));
  }

  // 因果掩码：q 行 i 只能看 kv 位置 <= (q0 + i)
  __aicore__ inline void ApplyCausalMask(const LocalTensor<float>& scores,
                                         int32_t q0, int32_t kv0) {
    for (int32_t i = 0; i < br_; ++i) {
      const int32_t limit = q0 + i - kv0;
      for (int32_t j = 0; j < bc_; ++j) {
        if (j > limit) scores.SetValue(i * bc_ + j, -3.0e38f);
      }
    }
  }

  // 在线 softmax：更新 running max/sum，并按新 max 重标定历史
  __aicore__ inline void SoftmaxOnline(const LocalTensor<float>& scores,
                                       const LocalTensor<float>& m_i,
                                       const LocalTensor<float>& l_i,
                                       int32_t rows, int32_t cols) {
    LocalTensor<float> tmp = rowBuf_.Get<float>();   // 复用（行数 <= br_）
    for (int32_t i = 0; i < rows; ++i) {
      const int32_t off = i * cols;
      ReduceMax(tmp[i], scores[off], tmp, cols);
      const float m_new = MaxS(tmp.GetValue(i), m_i.GetValue(i));
      const float alpha = ExpS(m_i.GetValue(i) - m_new);
      Exp(scores[off], scores[off], cols);
      ReduceSum(tmp[i], scores[off], tmp, cols);
      l_i.SetValue(i, l_i.GetValue(i) * alpha + tmp.GetValue(i));
      m_i.SetValue(i, m_new);
    }
  }

  __aicore__ inline void ApplyRowScale(const LocalTensor<float>& acc,
                                       const LocalTensor<float>& l_i,
                                       int32_t rows, int32_t cols) {
    for (int32_t i = 0; i < rows; ++i) {
      const float inv = l_i.GetValue(i) > 0.0f ? 1.0f / l_i.GetValue(i) : 0.0f;
      Muls(acc[i * cols], acc[i * cols], inv, cols);
    }
  }

  TPipe pipe_;
  TBuf<QuePosition::VECCALC> scoreBuf_, rowBuf_, accBuf_;
  TQue<QuePosition::VECIN, 1> kvQue_;
  Matmul<A_T, B_T, C_T> mmQK_, mmPV_;
  GlobalTensor<A_T> qGm_, oGm_;
  GlobalTensor<B_T> kGm_, vGm_;
  int32_t b_ = 0, h_ = 0, s_ = 0, t_len_ = 0, d_ = 0, br_ = 0, bc_ = 0;
  int32_t q_tiles_per_core_ = 0, causal_ = 0;
  float scale_ = 0.0f;
};

extern "C" __global__ __aicore__ void flash_attn(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                 GM_ADDR out, GM_ADDR tiling,
                                                 GM_ADDR tiling_matmul) {
  FlashAttnKernel op;
  op.Init(q, k, v, out, tiling, tiling_matmul);
  op.Process();
}
