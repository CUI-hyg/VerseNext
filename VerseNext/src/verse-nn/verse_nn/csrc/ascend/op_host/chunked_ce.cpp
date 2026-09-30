/*
 * ChunkedCe —— 算子定义 + Tiling（host 侧）
 *
 * 资源锁 50%：block_dim = AI 核数 × 0.5。
 * vocab 维分块长度取 UB 预算内可容纳的最大值，保证在线 logsumexp 的
 * tile 不溢出；Matmul 的 M 维固定为 1（单行），K = D，N = vocab_tile。
 */

#include "chunked_ce_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <algorithm>

namespace optiling {
namespace {
constexpr float kCoreFraction = 0.5f;
constexpr int32_t kUbBytes = 192 * 1024;
constexpr int32_t kUbReserve = 16 * 1024;
constexpr int32_t kAlign = 32;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* h_shape = context->GetInputShape(0);
  const gert::StorageShape* w_shape = context->GetInputShape(1);
  if (h_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (w_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const auto& hs = h_shape->GetStorageShape();
  const auto& ws = w_shape->GetStorageShape();
  const int32_t vocab = static_cast<int32_t>(ws.GetDim(0));
  const int32_t d = static_cast<int32_t>(ws.GetDim(1));
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < hs.GetDimNum(); ++i) rows *= hs.GetDim(i);

  int32_t ignore_index = -100;
  if (context->GetAttrs() != nullptr && context->GetAttrs()->GetAttrNum() > 1) {
    const int64_t* attr = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    if (attr != nullptr) ignore_index = static_cast<int32_t>(*attr);
  }

  // vocab tile：UB 预算 / fp32，留 3 份（logits + 临时 + 归约）
  const int32_t budget = kUbBytes - kUbReserve;
  int32_t v_tile = (budget / static_cast<int32_t>(sizeof(float)) / 3) / kAlign * kAlign;
  v_tile = std::max(v_tile, kAlign);
  v_tile = std::min(v_tile, vocab);

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const int32_t aic_total = static_cast<int32_t>(platform.GetCoreNumAiv());
  const int32_t block_dim = std::max(1, static_cast<int32_t>(aic_total * kCoreFraction));
  const int32_t rows_per_core = static_cast<int32_t>((rows + block_dim - 1) / block_dim);

  ChunkedCeTiling* tiling = context->GetTilingData<ChunkedCeTiling>();
  if (tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->rows = static_cast<int32_t>(rows);
  tiling->vocab = vocab;
  tiling->hidden_dim = d;
  tiling->vocab_tile = v_tile;
  tiling->rows_per_core = rows_per_core;
  tiling->ignore_index = ignore_index;

  matmul_tiling::MultiCoreMatmulTiling mm(platform);
  mm.SetDim(block_dim);
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT16, false);
  mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT16, true);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
              matmul_tiling::DataType::DT_FLOAT);
  mm.SetShape(1, v_tile, d);
  mm.SetOrgShape(1, v_tile, d);
  mm.SetBias(false);
  mm.SetBufferSpace(-1, -1, -1);
  if (mm.GetTiling(tiling->mm_tiling) == -1) {
    return ge::GRAPH_FAILED;
  }

  context->SetBlockDim(block_dim);
  context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
class ChunkedCe : public OpDef {
 public:
  explicit ChunkedCe(const char* name) : OpDef(name) {
    this->Input("hidden").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("targets").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("loss").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("chunk_size").AttrType(OPTIONAL).Int(0);
    this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(ChunkedCe);
}  // namespace ops
