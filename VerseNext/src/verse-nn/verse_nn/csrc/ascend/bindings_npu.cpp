/*
 * 自研算子注册到 torch 库 ``verse_nn`` —— 昇腾 NPU 侧
 *
 * 与 CUDA 侧的 bindings.cpp 对称：把 csrc/ascend 编译出的 ACLNN 自定义算子
 * 包装成 torch 算子，注册到 dispatch key ``PrivateUse1``（torch_npu 的设备
 * key），使 Python 侧 ``torch.ops.verse_nn.add_rms_norm_fwd`` 在 NPU 张量上
 * 自动走 CANN 实现。
 *
 * 为什么不直接用 torch_npu 的 ``EXEC_NPU_CMD`` 宏？
 *   ``EXEC_NPU_CMD`` 依赖 ``at_npu::native::OpPreparation::unsafe_empty_workspace``
 *   与 ``torch_npu::NPUStorageImpl`` 的 typeinfo，这两个符号在
 *   ``libtorch_npu.so`` 里**未导出**（visibility=hidden），外部扩展链接后
 *   dlopen 会报 undefined symbol。因此这里直接走 CANN 的 ACL C API：
 *     1. ``aclCreateTensor`` 把 at::Tensor 包成 aclTensor（共享 device 指针）；
 *     2. ``aclnnXxxGetWorkspaceSize`` 拿 workspace 大小（用 torch 的 NPU
 *        分配器分配 workspace）；
 *     3. ``aclnnXxx`` 在当前 NPU 流上执行。
 *   只有 ``c10_npu::getCurrentNPUStream()`` 从 libtorch_npu 取（该符号是导出的）。
 *
 * 算子名/schema 必须与 ``csrc/cuda/bindings.cpp`` 逐字一致
 * （tests/test_kernel_build.py 有静态校验）。
 *
 * 目前实现的自研 AscendC 算子：``add_rms_norm_fwd``、``swiglu_fwd``。
 * 其余 schema（rope/flash_attn/chunked_ce/kda_chunk）保留定义以保持两侧一致；
 * rope 与 flash_attn 在 NPU 上由 CANN 原生融合算子覆盖（见 kernels/npu_ops.py）。
 */

#include <torch/extension.h>

#ifdef VERSE_HAS_CANN

#include <tuple>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "acl/acl_base.h"
#include "aclnn/acl_meta.h"

// 由 CANN 自定义算子工程生成并随算子包安装（命名规则：aclnn_<op>.h）
#include "aclnn_verse_add_rms_norm.h"
#include "aclnn_verse_swiglu.h"

namespace {

aclDataType ToAclDtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
      return ACL_FLOAT;
    case at::kHalf:
      return ACL_FLOAT16;
    case at::kBFloat16:
      return ACL_BF16;
    case at::kInt:
      return ACL_INT32;
    case at::kLong:
      return ACL_INT64;
    default:
      TORCH_CHECK(false, "verse_nn/npu: 不支持的 dtype ", t);
  }
}

// at::Tensor -> aclTensor 的 RAII 包装（只借用 device 指针，不拷贝数据）。
class AclTensor {
 public:
  explicit AclTensor(const at::Tensor& src) {
    auto sizes = src.sizes();
    auto strides = src.strides();
    dims_.assign(sizes.begin(), sizes.end());
    strides_.assign(strides.begin(), strides.end());
    // 仅支持连续张量：非连续张量请调用方先 .contiguous()
    TORCH_CHECK(src.is_contiguous(), "verse_nn/npu: 需要连续张量");
    tensor_ = aclCreateTensor(dims_.data(), dims_.size(), ToAclDtype(src.scalar_type()),
                              strides_.data(), 0, ACL_FORMAT_ND,
                              dims_.data(), dims_.size(), src.data_ptr());
    TORCH_CHECK(tensor_ != nullptr, "verse_nn/npu: aclCreateTensor 失败");
  }
  ~AclTensor() {
    if (tensor_ != nullptr) {
      aclDestroyTensor(tensor_);
    }
  }
  AclTensor(const AclTensor&) = delete;
  AclTensor& operator=(const AclTensor&) = delete;
  aclTensor* get() const { return tensor_; }

 private:
  aclTensor* tensor_ = nullptr;
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;
};

// 通用两段式 aclnn 调用：GetWorkspaceSize -> 分配 workspace -> 执行。
template <typename SizeFn, typename RunFn>
void AclnnRun(const char* name, SizeFn size_fn, RunFn run_fn, const at::Tensor& like) {
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  uint64_t ws_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = size_fn(&ws_size, &executor);
  TORCH_CHECK(ret == 0, name, " GetWorkspaceSize 失败, ret=", static_cast<int>(ret));

  at::Tensor workspace;
  void* ws_ptr = nullptr;
  if (ws_size > 0) {
    workspace = at::empty({static_cast<int64_t>(ws_size)}, like.options().dtype(at::kByte));
    ws_ptr = workspace.data_ptr();
  }
  ret = run_fn(ws_ptr, ws_size, executor, stream);
  TORCH_CHECK(ret == 0, name, " 执行失败, ret=", static_cast<int>(ret));
}

}  // namespace

namespace verse_nn {

// 残差加 + RMSNorm：aclnnAddRmsNorm 一次产出 (out, rstd, res_out)，取 (out, res_out)。
std::tuple<torch::Tensor, torch::Tensor> add_rms_norm_fwd_npu(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  auto xc = x.contiguous();
  auto rc = residual.contiguous();
  auto wc = weight.contiguous();
  auto out = at::empty_like(xc);
  auto res_out = at::empty_like(xc);

  AclTensor ax(xc), ar(rc), aw(wc), ao(out), ares(res_out);
  AclnnRun(
      "VerseAddRmsNorm",
      [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnVerseAddRmsNormGetWorkspaceSize(ax.get(), ar.get(), aw.get(), eps,
                                               ao.get(), ares.get(), ws, ex);
      },
      [&](void* ws, uint64_t n, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseAddRmsNorm(ws, n, ex, s);
      },
      xc);
  return std::make_tuple(out, res_out);
}

// 融合 SwiGLU：out = silu(gate) * up（gate/up 同形，逐元素）。
torch::Tensor swiglu_fwd_npu(torch::Tensor gate, torch::Tensor up) {
  auto gc = gate.contiguous();
  auto uc = up.contiguous();
  auto out = at::empty_like(gc);

  AclTensor ag(gc), au(uc), ao(out);
  AclnnRun(
      "VerseSwiglu",
      [&](uint64_t* ws, aclOpExecutor** ex) {
        return aclnnVerseSwigluGetWorkspaceSize(ag.get(), au.get(), ao.get(), ws, ex);
      },
      [&](void* ws, uint64_t n, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseSwiglu(ws, n, ex, s);
      },
      gc);
  return out;
}

}  // namespace verse_nn

// 算子 schema 必须在此定义：NPU 构建只编译本文件（见 ascend/build.sh），
// 不像 CUDA 侧会一起编译 cuda/bindings.cpp。缺少 schema 时下面的
// TORCH_LIBRARY_IMPL 没有可挂载的算子，torch.ops.verse_nn.* 将不存在。
TORCH_LIBRARY(verse_nn, m) {
  m.def("add_rms_norm_fwd(Tensor x, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
  m.def("swiglu_fwd(Tensor gate, Tensor up) -> Tensor");
  m.def("rope_fwd(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
  m.def("flash_attn_fwd(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
  m.def("chunked_ce_fwd(Tensor hidden, Tensor weight, Tensor targets, "
        "int chunk_size, int ignore_index) -> Tensor");
  m.def("kda_chunk_fwd(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
        "int chunk_size) -> (Tensor, Tensor)");
}

// NPU 张量的 dispatch key 为 PrivateUse1（torch_npu）
TORCH_LIBRARY_IMPL(verse_nn, PrivateUse1, m) {
  m.impl("add_rms_norm_fwd", &verse_nn::add_rms_norm_fwd_npu);
  m.impl("swiglu_fwd", &verse_nn::swiglu_fwd_npu);
}

#endif  // VERSE_HAS_CANN
