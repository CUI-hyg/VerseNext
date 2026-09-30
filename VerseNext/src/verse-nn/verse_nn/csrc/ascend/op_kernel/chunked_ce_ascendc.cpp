/*
 * 融合 lm_head + 交叉熵 —— AscendC kernel
 *
 * 语义：loss = sum(CE(hidden @ weight^T, targets)) / valid_count
 *
 * 结构：
 *   - vocab 投影走 Cube（matmul::Matmul），按 vocab 维分 tile，**不物化整块 logits**；
 *   - 每个 tile 的 logits 在 UB 内做在线 logsumexp（Vector 单元：ReduceMax/Exp/ReduceSum）；
 *   - 目标位置的 logit 单独记录，最终 loss = m + log(sum) - logit[target]。
 *
 * 收益：显存从 O(B·S·V) 降到 O(B·S)，与 CUDA 版一致。
 * ⚠ 未编译验证（同 flash_attn，依赖 CANN Matmul tiling）。
 *
 * 接入前必读：docs/kernels.md 的「踩过的坑」——本文件尚未按那几条改造
 * （缺 REGISTER_TILING_DEFAULT、用 TILING_KEY_IS 做 dtype 分派），
 * 且未纳入 build.sh 的 OPS 列表。
 */

#include "../ascend_common.h"
#include "../op_host/chunked_ce_tiling.h"

#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;
using namespace verse_nn::ascend;

class ChunkedCeKernel {
 public:
  __aicore__ inline ChunkedCeKernel() {}

  __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR weight, GM_ADDR targets,
                              GM_ADDR loss_out, GM_ADDR tiling, GM_ADDR tiling_matmul) {
    const __gm__ ChunkedCeTiling* t = ReadTiling<ChunkedCeTiling>(tiling);
    rows_ = t->rows;
    vocab_ = t->vocab;
    d_ = t->hidden_dim;
    v_tile_ = t->vocab_tile;
    ignore_index_ = t->ignore_index;
    rows_per_core_ = t->rows_per_core;

    hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(hidden));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(weight));
    tGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(targets));
    lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(loss_out));

    pipe_.InitBuffer(logitBuf_, v_tile_ * sizeof(float));
    pipe_.InitBuffer(rowBuf_, 2 * 64 * sizeof(float));
    pipe_.InitBuffer(hBuf_, d_ * sizeof(DTYPE_X));

    REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm_, tiling_matmul);
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t r_begin = core * rows_per_core_;
    const int32_t r_end = MinI(r_begin + rows_per_core_, rows_);
    for (int32_t row = r_begin; row < r_end; ++row) {
      ProcessRow(row);
    }
  }

 private:
  __aicore__ inline void ProcessRow(int32_t row) {
    const int64_t h_base = static_cast<int64_t>(row) * d_;
    const int32_t tgt = tGm_.GetValue(row);
    const int32_t n_tiles = CeilDiv(vocab_, v_tile_);

    LocalTensor<float> logits = logitBuf_.Get<float>();
    LocalTensor<float> rowtmp = rowBuf_.Get<float>();
    LocalTensor<float> m_i = rowtmp[0];
    LocalTensor<float> l_i = rowtmp[64];

    Duplicate(m_i, -3.0e38f, 1);
    Duplicate(l_i, 0.0f, 1);
    float tgt_logit = 0.0f;

    for (int32_t t = 0; t < n_tiles; ++t) {
      const int32_t v0 = t * v_tile_;
      const int32_t cnt = MinI(v_tile_, vocab_ - v0);

      // logits_tile = hidden_row(1×D) @ W_tile^T(D×cnt)  -> (1 × cnt)
      mm_.SetTensorA(hGm_[h_base], false);
      mm_.SetTensorB(wGm_[static_cast<int64_t>(v0) * d_], true);
      mm_.IterateAll(logits);
      mm_.End();

      // 目标 logit
      if (tgt >= v0 && tgt < v0 + cnt) {
        tgt_logit = logits.GetValue(tgt - v0);
      }
      // 在线 logsumexp
      LocalTensor<float> tmax = rowtmp[1];
      ReduceMax(tmax, logits, rowtmp[63], cnt);
      const float tile_max = tmax.GetValue(0);
      const float m_new = MaxS(tile_max, m_i.GetValue(0));
      const float alpha = (m_i.GetValue(0) <= -3.0e37f) ? 0.0f
                                                        : ExpS(m_i.GetValue(0) - m_new);
      Exp(logits, logits, cnt);
      LocalTensor<float> tsum = rowtmp[2];
      ReduceSum(tsum, logits, rowtmp[63], cnt);
      l_i.SetValue(0, l_i.GetValue(0) * alpha + tsum.GetValue(0));
      m_i.SetValue(0, m_new);
    }

    float loss = 0.0f;
    if (tgt != ignore_index_ && l_i.GetValue(0) > 0.0f) {
      loss = m_i.GetValue(0) + LogS(l_i.GetValue(0)) - tgt_logit;
    }
    lGm_.SetValue(row, loss);
  }

  TPipe pipe_;
  TBuf<QuePosition::VECCALC> logitBuf_, rowBuf_, hBuf_;
  Matmul<DTYPE_X, DTYPE_X, float> mm_;
  GlobalTensor<DTYPE_X> hGm_, wGm_;
  GlobalTensor<int32_t> tGm_;
  GlobalTensor<float> lGm_;
  int32_t rows_ = 0, vocab_ = 0, d_ = 0, v_tile_ = 0;
  int32_t ignore_index_ = -100, rows_per_core_ = 0;
};

extern "C" __global__ __aicore__ void chunked_ce(GM_ADDR hidden, GM_ADDR weight,
                                                 GM_ADDR targets, GM_ADDR loss_out,
                                                 GM_ADDR tiling, GM_ADDR tiling_matmul) {
  ChunkedCeKernel op;
  op.Init(hidden, weight, targets, loss_out, tiling, tiling_matmul);
  op.Process();
}
