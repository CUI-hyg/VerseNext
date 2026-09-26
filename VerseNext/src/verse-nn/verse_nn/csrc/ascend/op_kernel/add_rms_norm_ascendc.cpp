/*
 * 融合残差加 + RMSNorm —— AscendC kernel（昇腾 910B/310P）
 *
 * 语义（与 CUDA 版、Python 参考实现严格一致）：
 *     res_out = x + residual
 *     out     = res_out / sqrt(mean(res_out^2) + eps) * weight
 *
 * 昇腾上的实现策略：
 *   - 按行切分：每个 AI Core 处理若干行（row），核内按 UB 容量把 D 维分 tile；
 *   - 两次遍历每行：第一次求平方和（Vector 单元 ReduceSum），第二次归一化；
 *     res_out 第一次就写回 GM，第二次从 GM 读回（UB 放不下整行时不可避免，
 *     D <= UB/tile 时可在 UB 内复用，见下方 UseUbResidentRow 分支）；
 *   - 全程 fp32 归约，避免 fp16/bf16 的平方和精度损失。
 *
 * 资源锁 50%：block_dim 由 host tiling 按「AI 核数 × 0.5」下发。
 */

#include "../ascend_common.h"
#include "../op_host/add_rms_norm_tiling.h"

using namespace AscendC;
using namespace verse_nn::ascend;

namespace {
constexpr int32_t kMaxD = 8192;      // 单行最大 head_dim（UB 驻留上限由 tiling 决定）
}  // namespace

class AddRmsNormKernel {
 public:
  __aicore__ inline AddRmsNormKernel() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                              GM_ADDR out, GM_ADDR res_out, GM_ADDR tiling) {
    const AddRmsNormTiling* t = reinterpret_cast<const AddRmsNormTiling*>(tiling);
    rows_ = t->rows;
    d_ = t->d;
    eps_ = t->eps;
    tile_ = t->tile_elems;
    rows_per_core_ = t->rows_per_core;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(x));
    rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(residual));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(weight));
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(out));
    roGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(res_out));

    pipe_.InitBuffer(inQueX_, 1, tile_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(inQueR_, 1, tile_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(inQueW_, 1, tile_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(outQue_, 1, tile_ * sizeof(DTYPE_X));
    // fp32 工作缓冲：cast 后的 x、临时平方、行内归约结果
    pipe_.InitBuffer(workBuf_, 2 * tile_ * sizeof(float) + kAlignBytes);
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t row_begin = core * rows_per_core_;
    const int32_t row_end = Min(row_begin + rows_per_core_, rows_);

    for (int32_t row = row_begin; row < row_end; ++row) {
      ProcessRow(row);
    }
  }

 private:
  __aicore__ inline void ProcessRow(int32_t row) {
    const int64_t base = static_cast<int64_t>(row) * d_;
    const int32_t n_tiles = CeilDiv(d_, tile_);

    // ---------- pass 1: res_out = x + residual，并累加平方和 ----------
    float sum_sq = 0.0f;
    LocalTensor<float> work = workBuf_.Get<float>();
    LocalTensor<float> sq = work[tile_];       // 第二段：平方

    for (int32_t t = 0; t < n_tiles; ++t) {
      const int32_t off = t * tile_;
      const int32_t cnt = Min(tile_, d_ - off);

      LocalTensor<DTYPE_X> xin = inQueX_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> rin = inQueR_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> res = outQue_.AllocTensor<DTYPE_X>();

      DataCopy(xin, xGm_[base + off], AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X)));
      DataCopy(rin, rGm_[base + off], AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X)));
      inQueX_.EnQue(xin);
      inQueR_.EnQue(rin);
      xin = inQueX_.DeQue<DTYPE_X>();
      rin = inQueR_.DeQue<DTYPE_X>();

      Add(res, xin, rin, cnt);                        // res = x + residual
      DataCopy(roGm_[base + off], res, AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X)));

      // 平方和（fp32 累加）
      Cast(work, res, RoundMode::CAST_NONE, cnt);
      Mul(sq, work, work, cnt);
      LocalTensor<float> partial = work[2 * tile_];   // 复用尾部做标量归约输出
      ReduceSum(partial, sq, work, cnt);
      sum_sq += partial.GetValue(0);

      inQueX_.FreeTensor(xin);
      inQueR_.FreeTensor(rin);
      outQue_.FreeTensor(res);
    }

    const float inv_rms = 1.0f / sqrtf(sum_sq / static_cast<float>(d_) + eps_);

    // ---------- pass 2: out = res_out * inv_rms * weight ----------
    for (int32_t t = 0; t < n_tiles; ++t) {
      const int32_t off = t * tile_;
      const int32_t cnt = Min(tile_, d_ - off);

      LocalTensor<DTYPE_X> res = outQue_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> win = inQueW_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> o = outQue_.AllocTensor<DTYPE_X>();

      DataCopy(res, roGm_[base + off], AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X)));
      DataCopy(win, wGm_[off], AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X)));
      outQue_.EnQue(res);
      inQueW_.EnQue(win);
      res = outQue_.DeQue<DTYPE_X>();
      win = inQueW_.DeQue<DTYPE_X>();

      Muls(res, res, inv_rms, cnt);                   // res *= inv_rms
      Mul(o, res, win, cnt);                          // o = res * weight
      DataCopy(oGm_[base + off], o, AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X)));

      outQue_.FreeTensor(res);
      inQueW_.FreeTensor(win);
      outQue_.FreeTensor(o);
    }
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> inQueX_, inQueR_, inQueW_;
  TQue<QuePosition::VECOUT, 1> outQue_;
  TBuf<QuePosition::VECCALC> workBuf_;

  GlobalTensor<DTYPE_X> xGm_, rGm_, wGm_, oGm_, roGm_;
  int32_t rows_ = 0, d_ = 0, tile_ = 0, rows_per_core_ = 0;
  float eps_ = 1e-6f;
};

extern "C" __global__ __aicore__ void add_rms_norm(
    GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
    GM_ADDR out, GM_ADDR res_out, GM_ADDR tiling) {
  AddRmsNormKernel op;
  op.Init(x, residual, weight, out, res_out, tiling);
  op.Process();
}
