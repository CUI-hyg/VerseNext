/*
 * 融合 SwiGLU —— AscendC kernel
 *
 * 语义：out = silu(gate) * up = gate * sigmoid(gate) * up
 *
 * 昇腾实现：纯 Vector 单元逐元素算子，按 UB tile 流水处理。
 * 与 CUDA 版不同，昇腾不需要手工做向量化——DataCopy 与 Vector 指令
 * 由 CANN 自动按 32B/256B 粒度展开，关键是**保证 double buffer 流水**：
 * 用两个 TQue 交替 Alloc/EnQue/DeQue/Free，让搬运与计算重叠。
 */

#include "../ascend_common.h"
#include "../op_host/swiglu_tiling.h"

using namespace AscendC;
using namespace verse_nn::ascend;

class SwigluKernel {
 public:
  __aicore__ inline SwigluKernel() {}

  __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR out, GM_ADDR tiling) {
    const SwigluTiling* t = reinterpret_cast<const SwigluTiling*>(tiling);
    total_ = t->total;
    tile_ = t->tile_elems;
    tiles_per_core_ = t->tiles_per_core;

    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(gate));
    uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(up));
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(out));

    // 双缓冲：两个队列各 2 块，搬运与计算重叠
    pipe_.InitBuffer(gQue_, 2, tile_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(uQue_, 2, tile_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(oQue_, 2, tile_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(tmpBuf_, tile_ * sizeof(DTYPE_X));
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t t_begin = core * tiles_per_core_;
    const int32_t t_end = Min(t_begin + tiles_per_core_, CeilDiv(total_, tile_));

    for (int32_t t = t_begin; t < t_end; ++t) {
      const int64_t off = static_cast<int64_t>(t) * tile_;
      const int32_t cnt = Min(tile_, static_cast<int32_t>(total_ - off));
      if (cnt <= 0) break;
      const int32_t aligned = AlignUp(cnt, kAlignBytes / sizeof(DTYPE_X));

      LocalTensor<DTYPE_X> g = gQue_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> u = uQue_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> o = oQue_.AllocTensor<DTYPE_X>();
      LocalTensor<DTYPE_X> tmp = tmpBuf_.Get<DTYPE_X>();

      DataCopy(g, gGm_[off], aligned);
      DataCopy(u, uGm_[off], aligned);
      gQue_.EnQue(g);
      uQue_.EnQue(u);
      g = gQue_.DeQue<DTYPE_X>();
      u = uQue_.DeQue<DTYPE_X>();

      // tmp = sigmoid(g); tmp = tmp * g; o = tmp * u
      Sigmoid(tmp, g, cnt);
      Mul(tmp, tmp, g, cnt);
      Mul(o, tmp, u, cnt);
      DataCopy(oGm_[off], o, aligned);

      gQue_.FreeTensor(g);
      uQue_.FreeTensor(u);
      oQue_.FreeTensor(o);
    }
  }

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, 2> gQue_, uQue_;
  TQue<QuePosition::VECOUT, 2> oQue_;
  TBuf<QuePosition::VECCALC> tmpBuf_;

  GlobalTensor<DTYPE_X> gGm_, uGm_, oGm_;
  int32_t total_ = 0, tile_ = 0, tiles_per_core_ = 0;
};

extern "C" __global__ __aicore__ void swiglu(GM_ADDR gate, GM_ADDR up, GM_ADDR out,
                                             GM_ADDR tiling) {
  SwigluKernel op;
  op.Init(gate, up, out, tiling);
  op.Process();
}
