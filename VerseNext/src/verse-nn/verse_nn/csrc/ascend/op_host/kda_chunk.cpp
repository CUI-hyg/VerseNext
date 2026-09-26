/*
 * KdaChunk —— 算子定义 + Tiling（host 侧）
 *
 * 状态 S 为 head_dim×head_dim 的 fp32，需常驻 UB：head_dim=64 时 16KB，
 * 在 910B 的 192KB UB 内可行。host 侧据此校验并给出核数分配。
 * 资源锁 50%：block_dim = AI 核数 × 0.5。
 */

#include "kda_chunk_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

namespace optiling {
namespace {
constexpr float kCoreFraction = 0.5f;
constexpr int32_t kUbBytes = 192 * 1024;
constexpr int32_t kMaxHeadDim = 128;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* q_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, q_shape);
  const auto& qs = q_shape->GetStorageShape();
  const int32_t batch = static_cast<int32_t>(qs.GetDim(0));
  const int32_t heads = static_cast<int32_t>(qs.GetDim(1));
  const int32_t seq_len = static_cast<int32_t>(qs.GetDim(2));
  const int32_t head_dim = static_cast<int32_t>(qs.GetDim(3));

  // UB 校验：状态 D*D*4B + q/k/v/g 4*D*4B + scratch 2*D*4B + o D*4B
  const int64_t need = static_cast<int64_t>(head_dim) * head_dim * 4
                       + 4 * head_dim * 4 + 2 * head_dim * 4 + head_dim * 4;
  if (head_dim > kMaxHeadDim || need > kUbBytes) {
    // 超出 UB：需分块驻留状态（后续优化），此处直接失败以提示回退
    return ge::GRAPH_FAILED;
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const int32_t aic_total = static_cast<int32_t>(platform.GetCoreNumAiv());
  const int32_t block_dim = std::max(1, static_cast<int32_t>(aic_total * kCoreFraction));

  const int32_t bh_total = batch * heads;
  const int32_t bh_per_core = (bh_total + block_dim - 1) / block_dim;

  KdaChunkTiling* tiling = context->GetTilingData<KdaChunkTiling>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling);
  tiling->batch = batch;
  tiling->heads = heads;
  tiling->seq_len = seq_len;
  tiling->head_dim = head_dim;
  tiling->chunk_size = 64;
  tiling->bh_per_core = bh_per_core;

  context->SetBlockDim(block_dim);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
class KdaChunk : public OpDef {
 public:
  explicit KdaChunk(const char* name) : OpDef(name) {
    this->Input("q").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("k").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("v").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gate").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("beta").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("state").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(KdaChunk);
}  // namespace ops
