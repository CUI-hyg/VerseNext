/*
 * FlashAttn —— 算子定义 + Tiling（host 侧）
 *
 * 除了常规 tiling，本算子还需要给两个 Matmul（QK^T、PV）下发 TCubeTiling：
 * 用 matmul_tiling::MultiCoreMatmulTiling 按 (M, N, K) 与 dtype 生成，
 * 再写进 kernel 的 tiling 结构尾部。这是 AscendC Matmul 的标准做法。
 *
 * 资源锁 50%：block_dim = AI 核数 × 0.5。
 */

#include "flash_attn_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <algorithm>
#include <cmath>

namespace optiling {
namespace {
constexpr float kCoreFraction = 0.5f;
constexpr int32_t kBr = 64;
constexpr int32_t kBc = 64;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* q_shape = context->GetInputShape(0);
  const gert::StorageShape* k_shape = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, q_shape);
  OPS_CHECK_NULL_WITH_CONTEXT(context, k_shape);

  const auto& qs = q_shape->GetStorageShape();
  const auto& ks = k_shape->GetStorageShape();
  // q/k 均为 (B, H, S, D)
  const int32_t batch = static_cast<int32_t>(qs.GetDim(0));
  const int32_t heads = static_cast<int32_t>(qs.GetDim(1));
  const int32_t seq_len = static_cast<int32_t>(qs.GetDim(2));
  const int32_t head_dim = static_cast<int32_t>(qs.GetDim(3));
  const int32_t kv_len = static_cast<int32_t>(ks.GetDim(2));

  bool causal = true;
  if (context->GetAttrs() != nullptr && context->GetAttrs()->GetAttrNum() > 0) {
    const bool* attr = context->GetAttrs()->GetAttrPointer<bool>(0);
    if (attr != nullptr) causal = *attr;
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const int32_t aic_total = static_cast<int32_t>(platform.GetCoreNumAiv());
  const int32_t block_dim = std::max(1, static_cast<int32_t>(aic_total * kCoreFraction));

  const int32_t q_tiles = (seq_len + kBr - 1) / kBr;
  const int32_t total_units = batch * heads * q_tiles;
  const int32_t q_tiles_per_core = (total_units + block_dim - 1) / block_dim;

  FlashAttnTiling* tiling = context->GetTilingData<FlashAttnTiling>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, tiling);
  tiling->batch = batch;
  tiling->heads = heads;
  tiling->seq_len = seq_len;
  tiling->kv_len = kv_len;
  tiling->head_dim = head_dim;
  tiling->q_tile = kBr;
  tiling->kv_tile = kBc;
  tiling->q_tiles_per_core = q_tiles_per_core;
  tiling->causal = causal ? 1 : 0;
  tiling->scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // ---- 两个 Matmul 的 TCubeTiling ----
  matmul_tiling::MultiCoreMatmulTiling qk_tiling(platform);
  qk_tiling.SetDim(block_dim);
  qk_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16, false);
  qk_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16, true);
  qk_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
  qk_tiling.SetShape(kBr, kBc, head_dim);
  qk_tiling.SetOrgShape(kBr, kBc, head_dim);
  qk_tiling.SetBias(false);
  qk_tiling.SetBufferSpace(-1, -1, -1);
  if (qk_tiling.GetTiling(tiling->mm_qk_tiling) == -1) {
    return ge::GRAPH_FAILED;
  }

  matmul_tiling::MultiCoreMatmulTiling pv_tiling(platform);
  pv_tiling.SetDim(block_dim);
  pv_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT, false);
  pv_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16, false);
  pv_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
  pv_tiling.SetShape(kBr, head_dim, kBc);
  pv_tiling.SetOrgShape(kBr, head_dim, kBc);
  pv_tiling.SetBias(false);
  pv_tiling.SetBufferSpace(-1, -1, -1);
  if (pv_tiling.GetTiling(tiling->mm_pv_tiling) == -1) {
    return ge::GRAPH_FAILED;
  }

  context->SetBlockDim(block_dim);
  context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
class FlashAttn : public OpDef {
 public:
  explicit FlashAttn(const char* name) : OpDef(name) {
    this->Input("q").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("k").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("v").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("causal").AttrType(OPTIONAL).Bool(true);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(FlashAttn);
}  // namespace ops
