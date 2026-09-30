/*
 * 融合 SwiGLU —— AscendC kernel
 *
 * 语义：out = silu(gate) * up = gate * sigmoid(gate) * up
 *
 * 昇腾实现：纯 Vector 单元逐元素算子，按 UB tile 流水处理。
 * 与 CUDA 版不同，昇腾不需要手工做向量化——DataCopy 与 Vector 指令
 * 由 CANN 自动按 32B/256B 粒度展开，关键是**保证 double buffer 流水**：
 * 用两个 TQue 交替 Alloc/EnQue/DeQue/Free，让搬运与计算重叠。
 *
 * dtype 分派：构建系统按 dtype 组合分别编译本文件，并注入
 * ``-DDTYPE_GATE=half|float``（输入名大写加前缀），因此入口直接
 * ``SwigluKernel<DTYPE_GATE>`` 实例化即可，无需 TILING_KEY。
 */

#include "../ascend_common.h"
#include "../op_host/verse_swiglu_tiling.h"

using namespace AscendC;
using namespace verse_nn::ascend;

// 由构建系统注入（-DDTYPE_GATE=...）；单独语法检查时给个兜底。
#ifndef DTYPE_GATE
#define DTYPE_GATE half
#endif

template <typename T>
class SwigluKernel {
 public:
  __aicore__ inline SwigluKernel() {}

  __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR out, GM_ADDR tiling) {
    const __gm__ SwigluTiling* t = ReadTiling<SwigluTiling>(tiling);
    total_ = t->total;
    tile_ = t->tile_elems;
    tiles_per_core_ = t->tiles_per_core;

    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gate));
    uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(up));
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    // 双缓冲：两个队列各 2 块，搬运与计算重叠
    pipe_.InitBuffer(gQue_, 2, tile_ * sizeof(T));
    pipe_.InitBuffer(uQue_, 2, tile_ * sizeof(T));
    pipe_.InitBuffer(oQue_, 2, tile_ * sizeof(T));
    pipe_.InitBuffer(tmpBuf_, tile_ * sizeof(T));
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t t_begin = core * tiles_per_core_;
    const int32_t t_end = MinI(t_begin + tiles_per_core_, CeilDiv(total_, tile_));

    for (int32_t t = t_begin; t < t_end; ++t) {
      const int64_t off = static_cast<int64_t>(t) * tile_;
      const int32_t cnt = MinI(tile_, static_cast<int32_t>(total_ - off));
      if (cnt <= 0) break;

      LocalTensor<T> g = gQue_.AllocTensor<T>();
      LocalTensor<T> u = uQue_.AllocTensor<T>();
      LocalTensor<T> o = oQue_.AllocTensor<T>();
      LocalTensor<T> tmp = tmpBuf_.Get<T>();

      CopyInPad(g, gGm_[off], cnt);
      CopyInPad(u, uGm_[off], cnt);
      gQue_.EnQue(g);
      uQue_.EnQue(u);
      g = gQue_.DeQue<T>();
      u = uQue_.DeQue<T>();

      // tmp = sigmoid(g); tmp = tmp * g; o = tmp * u
      Sigmoid(tmp, g, cnt);
      Mul(tmp, tmp, g, cnt);
      Mul(o, tmp, u, cnt);
      oQue_.EnQue(o);                                 // 保证 Vector -> MTE3 同步
      o = oQue_.DeQue<T>();
      CopyOutPad(oGm_[off], o, cnt);

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

  GlobalTensor<T> gGm_, uGm_, oGm_;
  int32_t total_ = 0, tile_ = 0, tiles_per_core_ = 0;
};

// 入口签名由 CANN 算子框架约定：所有输入、输出之后是 workspace 与 tiling
// 两个 GM 指针（workspace 在无 Matmul 的算子里未使用）。
//
// REGISTER_TILING_DEFAULT 必须放在入口里：它把 ``sizeof(SwigluTiling)`` 写进
// kernel 二进制的 ``.ascendc_tiling.default`` 段，框架据此决定 tiling data
// 缓冲区容量。漏掉它时框架按骨架里的占位结构（4B）分配，host 侧
// ``GetTilingData<SwigluTiling>()`` 会返回 nullptr，tiling 报错 561002。
extern "C" __global__ __aicore__ void verse_swiglu(GM_ADDR gate, GM_ADDR up, GM_ADDR out,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(SwigluTiling);
  SwigluKernel<DTYPE_GATE> op;
  op.Init(gate, up, out, tiling);
  op.Process();
}
