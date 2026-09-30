/*
 * 融合残差加 + RMSNorm —— AscendC kernel（昇腾 910B/910_93）
 *
 * 语义（与 CUDA 版、Python 参考实现严格一致）：
 *     res_out = x + residual
 *     out     = res_out / sqrt(mean(res_out^2) + eps) * weight
 *
 * 昇腾上的实现策略：
 *   - 按行切分：每个 AI Core 处理若干行（row），核内按 UB 容量把 D 维分 tile；
 *   - 两次遍历每行：第一次求平方和（Vector 单元 ReduceSum）并写回 res_out，
 *     第二次从 GM 读回 res_out 做归一化；
 *   - 全程 fp32 归约，避免 fp16 的平方和精度损失。
 *
 * dtype 分派：
 *   msopgen 的自定义算子工程会**按 dtype 组合分别编译**同一份 kernel 源码，
 *   编译时通过 ``-DDTYPE_X=half|float`` 注入具体类型（见构建产物里的
 *   ``add_rms_norm.py``：``input_names=['x','residual','weight']`` 大写后加
 *   ``DTYPE_`` 前缀）。运行期框架按 dtype 选择对应二进制，因此 kernel 里
 *   **不需要** TILING_KEY 分派，直接用 ``DTYPE_X`` 实例化模板即可。
 *
 *   注意：TILING_KEY_IS(k) 只是 ``g_tilingKey == k`` 的**运行期比较**，并不会
 *   生成多个 kernel 入口；单入口二进制下 host 侧必须下发 tilingKey=0，否则
 *   会报 ``BinaryGetFunctionByEntry ... funcEntry is invalid``（ret=361001）。
 *
 * 资源锁 50%：block_dim 由 host tiling 按「AI 核数 × 0.5」下发。
 */

#include "../ascend_common.h"
#include "../op_host/verse_add_rms_norm_tiling.h"

using namespace AscendC;
using namespace verse_nn::ascend;

// 由构建系统注入（-DDTYPE_X=...）；单独语法检查时给个兜底。
#ifndef DTYPE_X
#define DTYPE_X half
#endif

template <typename T>
class AddRmsNormKernel {
 public:
  __aicore__ inline AddRmsNormKernel() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                              GM_ADDR out, GM_ADDR res_out, GM_ADDR tiling) {
    const __gm__ AddRmsNormTiling* t = ReadTiling<AddRmsNormTiling>(tiling);
    rows_ = t->rows;
    d_ = t->d;
    eps_ = t->eps;
    tile_ = t->tile_elems;
    rows_per_core_ = t->rows_per_core;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residual));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    roGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(res_out));

    // 每个队列 1 块（深度 1），不同用途必须用不同队列——同一队列连续
    // AllocTensor 两次会拿到同一个/非法 buffer，导致 device 异常。
    pipe_.InitBuffer(inQueX_, 1, tile_ * sizeof(T));
    pipe_.InitBuffer(inQueR_, 1, tile_ * sizeof(T));
    pipe_.InitBuffer(inQueW_, 1, tile_ * sizeof(T));
    pipe_.InitBuffer(inQueRes_, 1, tile_ * sizeof(T));
    pipe_.InitBuffer(outQue_, 1, tile_ * sizeof(T));
    // fp32 工作缓冲：cast 后的值、平方、归约输出（3 段 + 对齐余量）
    pipe_.InitBuffer(workBuf_, 3 * tile_ * sizeof(float) + kAlignBytes);
  }

  __aicore__ inline void Process() {
    const int32_t core = GetBlockIdx();
    const int32_t row_begin = core * rows_per_core_;
    const int32_t row_end = MinI(row_begin + rows_per_core_, rows_);

    for (int32_t row = row_begin; row < row_end; ++row) {
      ProcessRow(row);
    }
  }

 private:
  __aicore__ inline void ProcessRow(int32_t row) {
    const int64_t base = static_cast<int64_t>(row) * d_;
    const int32_t n_tiles = CeilDiv(d_, tile_);

    float sum_sq = 0.0f;
    LocalTensor<float> work = workBuf_.Get<float>();
    LocalTensor<float> sq = work[tile_];            // 第二段：平方
    LocalTensor<float> partial = work[2 * tile_];   // 第三段：归约输出

    // ---------- pass 1: res_out = x + residual，并累加平方和 ----------
    for (int32_t t = 0; t < n_tiles; ++t) {
      const int32_t off = t * tile_;
      const int32_t cnt = MinI(tile_, d_ - off);

      LocalTensor<T> xin = inQueX_.AllocTensor<T>();
      LocalTensor<T> rin = inQueR_.AllocTensor<T>();
      LocalTensor<T> res = outQue_.AllocTensor<T>();

      CopyInPad(xin, xGm_[base + off], cnt);
      CopyInPad(rin, rGm_[base + off], cnt);
      inQueX_.EnQue(xin);
      inQueR_.EnQue(rin);
      xin = inQueX_.DeQue<T>();
      rin = inQueR_.DeQue<T>();

      Add(res, xin, rin, cnt);                        // res = x + residual
      outQue_.EnQue(res);                             // 保证 Vector -> MTE3 同步
      res = outQue_.DeQue<T>();
      CopyOutPad(roGm_[base + off], res, cnt);

      // 平方和（fp32 累加）
      SquareToFp32(sq, work, res, cnt);
      ReduceSum(partial, sq, work, cnt);
      SyncVectorToScalar();
      sum_sq += partial.GetValue(0);
      SyncScalarToVector();

      inQueX_.FreeTensor(xin);
      inQueR_.FreeTensor(rin);
      outQue_.FreeTensor(res);
    }

    const float inv_rms = 1.0f / SqrtS(sum_sq / static_cast<float>(d_) + eps_);

    // ---------- pass 2: out = res_out * inv_rms * weight ----------
    for (int32_t t = 0; t < n_tiles; ++t) {
      const int32_t off = t * tile_;
      const int32_t cnt = MinI(tile_, d_ - off);

      LocalTensor<T> res = inQueRes_.AllocTensor<T>();
      LocalTensor<T> win = inQueW_.AllocTensor<T>();
      LocalTensor<T> o = outQue_.AllocTensor<T>();

      CopyInPad(res, roGm_[base + off], cnt);
      CopyInPad(win, wGm_[off], cnt);
      inQueRes_.EnQue(res);
      inQueW_.EnQue(win);
      res = inQueRes_.DeQue<T>();
      win = inQueW_.DeQue<T>();

      Muls(res, res, static_cast<T>(inv_rms), cnt);   // res *= inv_rms
      Mul(o, res, win, cnt);                          // o = res * weight
      outQue_.EnQue(o);
      o = outQue_.DeQue<T>();
      CopyOutPad(oGm_[base + off], o, cnt);

      inQueRes_.FreeTensor(res);
      inQueW_.FreeTensor(win);
      outQue_.FreeTensor(o);
    }
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> inQueX_, inQueR_, inQueW_, inQueRes_;
  TQue<QuePosition::VECOUT, 1> outQue_;
  TBuf<QuePosition::VECCALC> workBuf_;

  GlobalTensor<T> xGm_, rGm_, wGm_, oGm_, roGm_;
  int32_t rows_ = 0, d_ = 0, tile_ = 0, rows_per_core_ = 0;
  float eps_ = 1e-6f;
};

// 入口签名由 CANN 算子框架约定：输入、输出之后是 workspace 与 tiling。
// 类型由构建系统按 dtype 注入的 DTYPE_X 决定，单入口即可覆盖全部 dtype。
//
// REGISTER_TILING_DEFAULT 必须放在入口里：它把 ``sizeof(AddRmsNormTiling)``
// 写进 kernel 二进制的 ``.ascendc_tiling.default`` 段，框架据此决定 tiling
// data 缓冲区的容量。漏掉它时框架按 msopgen 骨架里的占位结构
// （``AddRmsNormTilingData{uint32_t}`` = 4B）分配，host 侧
// ``GetTilingData<AddRmsNormTiling>()`` 会因容量不足返回 nullptr，
// tiling 报错 561002。
extern "C" __global__ __aicore__ void verse_add_rms_norm(
    GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
    GM_ADDR out, GM_ADDR res_out, GM_ADDR workspace, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(AddRmsNormTiling);
  AddRmsNormKernel<DTYPE_X> op;
  op.Init(x, residual, weight, out, res_out, tiling);
  op.Process();
}
