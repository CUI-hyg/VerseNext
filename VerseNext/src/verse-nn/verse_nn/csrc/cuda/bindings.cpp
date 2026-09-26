// 自研算子注册到 torch 库 ``verse_nn``
//
// 注册后即可在 Python 侧通过 ``torch.ops.verse_nn.<name>`` 调用，例如
//     torch.ops.verse_nn.add_rms_norm_fwd(x, residual, weight, 1e-6)
//
// 仅注册 CUDA 实现（CPU 侧优化走 Python/ATen，见 verse_nn/kernels/cpu_ops.py）。
// CUDA 与 ROCm 共用本实现：HIP 版由 hipify 从同一份 .cu 生成，算子名不变。
//
// 注意：TORCH_LIBRARY 的 ``m.def`` 只有 (schema, func[, tags]) 形式，**不接受
// 文档字符串**——算子语义写在 schema 名与上方注释里。

#include <torch/extension.h>

#include "common/ops.h"

TORCH_LIBRARY(verse_nn, m) {
  // 融合残差加 + RMSNorm：res = x + residual; out = rms_norm(res) * weight
  m.def("add_rms_norm_fwd(Tensor x, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
  // 融合 SwiGLU：out = silu(gate) * up
  m.def("swiglu_fwd(Tensor gate, Tensor up) -> Tensor");
  // 融合 RoPE（rotate_half 语义，与 rope.apply_rope 一致）
  m.def("rope_fwd(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
  // 融合注意力：分块在线 softmax，不物化 (S,T) 注意力矩阵
  m.def("flash_attn_fwd(Tensor q, Tensor k, Tensor v, bool causal, float scale) -> Tensor");
  // 融合 lm_head + 交叉熵：不物化 (B,S,V) logits
  m.def("chunked_ce_fwd(Tensor hidden, Tensor weight, Tensor targets, "
        "int chunk_size, int ignore_index) -> Tensor");
  // KDA 融合前向：状态驻留共享内存，一次 launch 跑完整条序列
  m.def("kda_chunk_fwd(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
        "int chunk_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(verse_nn, CUDA, m) {
  m.impl("add_rms_norm_fwd", &verse_nn::add_rms_norm_fwd);
  m.impl("swiglu_fwd", &verse_nn::swiglu_fwd);
  m.impl("rope_fwd", &verse_nn::rope_fwd);
  m.impl("flash_attn_fwd", &verse_nn::flash_attn_fwd);
  m.impl("chunked_ce_fwd", &verse_nn::chunked_ce_fwd);
  m.impl("kda_chunk_fwd", &verse_nn::kda_chunk_fwd);
}
