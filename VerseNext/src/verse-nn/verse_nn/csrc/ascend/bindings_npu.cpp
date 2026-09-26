/*
 * 自研算子注册到 torch 库 ``verse_nn`` —— 昇腾 NPU 侧
 *
 * 与 CUDA 侧的 bindings.cpp 对称：这里把 csrc/ascend 编译出的 ACLNN 自定义
 * 算子包装成 torch 算子，注册到 dispatch key ``PrivateUse1``（torch_npu 的
 * 设备 key），使 Python 侧 ``torch.ops.verse_nn.add_rms_norm_fwd`` 在 NPU 张量
 * 上自动走 CANN 实现。
 *
 * 调用约定（CANN 自定义算子标准两段式）：
 *   1. aclnn<Op>GetWorkspaceSize(...) 计算 workspace 大小、拿到执行器；
 *   2. aclnn<Op>(workspace, workspaceSize, executor, stream) 真正执行。
 *
 * ⚠ 未编译验证：需要 CANN 工具链 + torch_npu，且 ACLNN 头文件由自定义算子
 *   工程生成（见 build.sh 的 msopgen 流程）。首次编译请按 CANN 版本调整
 *   头文件名与函数签名。整个文件用 VERSE_HAS_CANN 守卫，未定义时退化为空
 *   注册（Python 侧自动回退 CANN 原生算子或参考实现）。
 */

#include <torch/extension.h>

#ifdef VERSE_HAS_CANN

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

// 由 CANN 自定义算子工程生成（命名规则：aclnn<OpName>.h）
#include "aclnn_verse_add_rms_norm.h"
#include "aclnn_verse_swiglu.h"
#include "aclnn_verse_flash_attn.h"
#include "aclnn_verse_chunked_ce.h"
#include "aclnn_verse_kda_chunk.h"

namespace {

// 通用执行器：跑完 GetWorkspaceSize + 执行两段式，并检查返回码
template <typename SizeFn, typename RunFn>
void aclnn_run(const char* op_name, SizeFn size_fn, RunFn run_fn) {
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = size_fn(&workspace_size, &executor, stream);
  TORCH_CHECK(ret == ACLNN_SUCCESS, op_name, " GetWorkspaceSize 失败, ret=", ret);

  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));
    workspace_ptr = workspace.data_ptr();
  }
  ret = run_fn(workspace_ptr, workspace_size, executor, stream);
  TORCH_CHECK(ret == ACLNN_SUCCESS, op_name, " 执行失败, ret=", ret);
}

}  // namespace

// 算子 schema 必须在此定义：NPU 构建只编译本文件（见 ascend/build.sh），
// 不像 CUDA 侧会一起编译 cuda/bindings.cpp。缺少 schema 时下面的
// TORCH_LIBRARY_IMPL 没有可挂载的算子，torch.ops.verse_nn.* 将不存在。
// 名称与签名必须与 cuda/bindings.cpp 逐字一致（tests/test_kernel_build.py 校验）。
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

namespace verse_nn {

std::vector<torch::Tensor> add_rms_norm_fwd_npu(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
  auto out = torch::empty_like(x);
  auto res_out = torch::empty_like(x);
  aclnn_run(
      "AddRmsNorm",
      [&](uint64_t* ws, aclOpExecutor** ex, aclrtStream s) {
        return aclnnVerseAddRmsNormGetWorkspaceSize(
            x, residual, weight, static_cast<float>(eps), out, res_out, ws, ex);
      },
      [&](void* ws, uint64_t ws_size, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseAddRmsNorm(ws, ws_size, ex, s);
      });
  return {out, res_out};
}

torch::Tensor swiglu_fwd_npu(torch::Tensor gate, torch::Tensor up) {
  auto out = torch::empty_like(gate);
  aclnn_run(
      "Swiglu",
      [&](uint64_t* ws, aclOpExecutor** ex, aclrtStream s) {
        return aclnnVerseSwigluGetWorkspaceSize(gate, up, out, ws, ex);
      },
      [&](void* ws, uint64_t ws_size, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseSwiglu(ws, ws_size, ex, s);
      });
  return out;
}

torch::Tensor flash_attn_fwd_npu(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale) {
  auto out = torch::empty_like(q);
  aclnn_run(
      "FlashAttn",
      [&](uint64_t* ws, aclOpExecutor** ex, aclrtStream s) {
        return aclnnVerseFlashAttnGetWorkspaceSize(
            q, k, v, causal, static_cast<float>(scale), out, ws, ex);
      },
      [&](void* ws, uint64_t ws_size, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseFlashAttn(ws, ws_size, ex, s);
      });
  return out;
}

torch::Tensor chunked_ce_fwd_npu(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor targets,
    int64_t chunk_size, int64_t ignore_index) {
  auto row_loss = torch::empty({hidden.size(0) * hidden.size(1)},
                               hidden.options().dtype(at::kFloat));
  aclnn_run(
      "ChunkedCe",
      [&](uint64_t* ws, aclOpExecutor** ex, aclrtStream s) {
        return aclnnVerseChunkedCeGetWorkspaceSize(
            hidden, weight, targets, chunk_size, ignore_index, row_loss, ws, ex);
      },
      [&](void* ws, uint64_t ws_size, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseChunkedCe(ws, ws_size, ex, s);
      });
  // 与参考实现一致的加权平均（有效位置已在 kernel 内置 0，此处按非零计数）
  auto valid = (row_loss != 0).sum().clamp_min(1);
  return row_loss.sum() / valid;
}

std::vector<torch::Tensor> kda_chunk_fwd_npu(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor gate, torch::Tensor beta, int64_t chunk_size) {
  auto out = torch::empty_like(q);
  auto state = torch::empty({q.size(0), q.size(1), q.size(3), q.size(3)},
                            q.options().dtype(at::kFloat));
  aclnn_run(
      "KdaChunk",
      [&](uint64_t* ws, aclOpExecutor** ex, aclrtStream s) {
        return aclnnVerseKdaChunkGetWorkspaceSize(
            q, k, v, gate, beta, chunk_size, out, state, ws, ex);
      },
      [&](void* ws, uint64_t ws_size, aclOpExecutor* ex, aclrtStream s) {
        return aclnnVerseKdaChunk(ws, ws_size, ex, s);
      });
  return {out, state};
}

}  // namespace verse_nn

// NPU 张量的 dispatch key 为 PrivateUse1（torch_npu）
TORCH_LIBRARY_IMPL(verse_nn, PrivateUse1, m) {
  m.impl("add_rms_norm_fwd", &verse_nn::add_rms_norm_fwd_npu);
  m.impl("swiglu_fwd", &verse_nn::swiglu_fwd_npu);
  m.impl("flash_attn_fwd", &verse_nn::flash_attn_fwd_npu);
  m.impl("chunked_ce_fwd", &verse_nn::chunked_ce_fwd_npu);
  m.impl("kda_chunk_fwd", &verse_nn::kda_chunk_fwd_npu);
}

#endif  // VERSE_HAS_CANN
