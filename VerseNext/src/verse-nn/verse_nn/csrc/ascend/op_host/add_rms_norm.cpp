/*
 * AddRmsNorm —— 算子定义 + Tiling（host 侧）
 *
 * CANN 自定义算子工程中，host 侧负责：
 *   1. 用 OP_ADD/REG_OP 声明算子的输入/输出/属性（供图编译器与 aclnn 使用）；
 *   2. TilingFunc 里按输入 shape、UB 容量、AI 核数算出切分方案并下发。
 *
 * 资源锁 50%：block_dim 取「AI 核数 × 0.5」。910B 有 24 个 AI Core，
 * 则默认下发 12 个，保证与 GPU/NPU 的资源口径一致（50% 占用上限）。
 */

#include "add_rms_norm_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

#include <algorithm>

namespace optiling {
namespace {
// 资源占比：与 ResourceConfig.cpu_fraction 默认值保持一致
constexpr float kCoreFraction = 0.5f;
constexpr int32_t kUbBytes = 192 * 1024;
constexpr int32_t kUbReserve = 8 * 1024;
constexpr int32_t kAlign = 32;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  const int64_t d = x_shape->GetStorageShape().GetDim(
      x_shape->GetStorageShape().GetDimNum() - 1);
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < x_shape->GetStorageShape().GetDimNum(); ++i) {
    rows *= x_shape->GetStorageShape().GetDim(i);
  }

  // 从算子属性取 eps
  float eps = 1e-6f;
  if (context->GetAttrs() != nullptr && context->GetAttrs()->GetAttrNum() > 0) {
    const float* attr = context->GetAttrs()->GetAttrPointer<float>(0);
    if (attr != nullptr) eps = *attr;
  }

  // UB 单次可处理的元素数：fp32 计算，预留 4 份缓冲
  const int32_t budget = kUbBytes - kUbReserve;
  int32_t tile = (budget / static_cast<int32_t>(sizeof(float)) / 4) / kAlign * kAlign;
  tile = std::max(tile, kAlign);
  tile = std::min<int32_t>(tile, static_cast<int32_t>(d));

  // AI 核数与 50% 锁
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const int32_t aic_total = static_cast<int32_t>(ascendc_platform.GetCoreNumAiv());
  const int32_t block_dim = std::max(1, static_cast<int32_t>(aic_total * kCoreFraction));

  const int32_t rows_per_core = static_cast<int32_t>((rows + block_dim - 1) / block_dim);

  AddRmsNormTiling* tiling = context->GetTilingData<AddRmsNormTiling>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling);
  tiling->rows = static_cast<int32_t>(rows);
  tiling->d = static_cast<int32_t>(d);
  tiling->tile_elems = tile;
  tiling->rows_per_core = rows_per_core;
  tiling->eps = eps;

  context->SetBlockDim(block_dim);
  size_t* workspaces = context->GetWorkspaceSizes(1);
  workspaces[0] = 0;   // 本算子不需要额外 workspace
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
class AddRmsNorm : public OpDef {
 public:
  explicit AddRmsNorm(const char* name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("residual").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("weight").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("res_out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("eps").AttrType(OPTIONAL).Float(1e-6f);
    this->AICore().AddConfig("ascend910b").AddConfig("ascend310p");
  }
};

OP_ADD(AddRmsNorm);
}  // namespace ops
