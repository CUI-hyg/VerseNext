/*
 * VerseSwiglu —— 算子定义 + Tiling（host 侧）
 *
 * 纯逐元素算子：tiling 只需把元素总数按 UB tile 与 AI 核数切分。
 * 资源锁 50%：block_dim = AI 核数 × 0.5。
 */

#include "verse_swiglu_tiling.h"
#include "ascend_soc.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

namespace optiling {
namespace {
constexpr float kCoreFraction = 0.5f;
constexpr int32_t kUbBytes = 192 * 1024;
constexpr int32_t kUbReserve = 8 * 1024;
constexpr int32_t kAlign = 32;
constexpr int32_t kMaxBlockDim = 24;   // 910B 系列 AI Core 上限（平台查询失败时兜底）
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* shape = context->GetInputShape(0);
  if (shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  int64_t total = 1;
  for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
    total *= shape->GetStorageShape().GetDim(i);
  }

  const int32_t budget = kUbBytes - kUbReserve;
  // 3 个输入/输出队列各 2 块 + 1 个临时缓冲 => 按 6 份估算
  int32_t tile = (budget / static_cast<int32_t>(sizeof(float)) / 6) / kAlign * kAlign;
  tile = std::max(tile, kAlign);
  if (tile > total) tile = static_cast<int32_t>(total);

  // AI 核数与 50% 锁。自定义算子在部分 CANN 版本上拿不到 platform info，
  // 此时退化为「最大核数的一半」，保证 tiling 不因空指针崩掉。
  int32_t block_dim = std::max(1, static_cast<int32_t>(kMaxBlockDim * kCoreFraction));
  if (context->GetPlatformInfo() != nullptr) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t aic_total = static_cast<int32_t>(platform.GetCoreNumAiv());
    if (aic_total > 0) {
      block_dim = std::max(1, static_cast<int32_t>(aic_total * kCoreFraction));
    }
  }

  const int32_t n_tiles = static_cast<int32_t>((total + tile - 1) / tile);
  const int32_t tiles_per_core = (n_tiles + block_dim - 1) / block_dim;

  SwigluTiling* tiling = context->GetTilingData<SwigluTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->total = static_cast<int32_t>(total);
  tiling->tile_elems = tile;
  tiling->tiles_per_core = tiles_per_core;

  // dtype 分派在 kernel 编译期完成（见 op_kernel/verse_swiglu_ascendc.cpp），
  // 单入口二进制下 tilingKey 必须为 0。
  context->SetTilingKey(0);
  context->SetBlockDim(block_dim);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// 形状/类型推导：逐元素算子，输出与 gate 同形同类型。
// 本文件会覆盖 msopgen 生成的 op_host/verse_swiglu.cpp，因此这些函数必须自带
// （否则 OpDef 里没有可注册的 InferShape/InferDataType）。
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* in = context->GetInputShape(0);
  gert::Shape* out = context->GetOutputShape(0);
  *out = *in;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class VerseSwiglu : public OpDef {
 public:
  explicit VerseSwiglu(const char* name) : OpDef(name) {
    this->Input("gate").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("up").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig(VERSE_ASCEND_SOC);
  }
};

OP_ADD(VerseSwiglu);
}  // namespace ops
