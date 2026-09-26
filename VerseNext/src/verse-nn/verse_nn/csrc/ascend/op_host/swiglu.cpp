/*
 * Swiglu —— 算子定义 + Tiling（host 侧）
 *
 * 纯逐元素算子：tiling 只需把元素总数按 UB tile 与 AI 核数切分。
 * 资源锁 50%：block_dim = AI 核数 × 0.5。
 */

#include "swiglu_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

namespace optiling {
namespace {
constexpr float kCoreFraction = 0.5f;
constexpr int32_t kUbBytes = 192 * 1024;
constexpr int32_t kUbReserve = 8 * 1024;
constexpr int32_t kAlign = 32;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, shape);

  int64_t total = 1;
  for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
    total *= shape->GetStorageShape().GetDim(i);
  }

  const int32_t budget = kUbBytes - kUbReserve;
  // 3 个输入/输出队列各 2 块 + 1 个临时缓冲 => 按 6 份估算
  int32_t tile = (budget / static_cast<int32_t>(sizeof(float)) / 6) / kAlign * kAlign;
  tile = std::max(tile, kAlign);
  if (tile > total) tile = static_cast<int32_t>(total);

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const int32_t aic_total = static_cast<int32_t>(platform.GetCoreNumAiv());
  const int32_t block_dim = std::max(1, static_cast<int32_t>(aic_total * kCoreFraction));

  const int32_t n_tiles = static_cast<int32_t>((total + tile - 1) / tile);
  const int32_t tiles_per_core = (n_tiles + block_dim - 1) / block_dim;

  SwigluTiling* tiling = context->GetTilingData<SwigluTiling>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling);
  tiling->total = static_cast<int32_t>(total);
  tiling->tile_elems = tile;
  tiling->tiles_per_core = tiles_per_core;

  context->SetBlockDim(block_dim);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
class Swiglu : public OpDef {
 public:
  explicit Swiglu(const char* name) : OpDef(name) {
    this->Input("gate").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("up").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b").AddConfig("ascend310p");
  }
};

OP_ADD(Swiglu);
}  // namespace ops
