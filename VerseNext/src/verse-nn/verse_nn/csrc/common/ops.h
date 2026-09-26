// VerseNext 自研算子 —— 前向算子宿主函数声明
//
// 各 .cu 文件实现这些函数，bindings.cpp 统一注册到 torch 库 ``verse_nn``。
// 命名约定：``*_fwd`` 只做前向；反向由 Python 侧参考实现重放（见 cuda_ops.py）。

#pragma once

#include <torch/extension.h>
#include <vector>

namespace verse_nn {

// 融合残差加 + RMSNorm：res_out = x + residual; out = rms_norm(res_out) * weight
std::vector<torch::Tensor> add_rms_norm_fwd(
    torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps);

// 融合 SwiGLU：out = silu(gate) * up
torch::Tensor swiglu_fwd(torch::Tensor gate, torch::Tensor up);

// 融合 RoPE（旋转半维，GPT-NeoX 风格）
std::vector<torch::Tensor> rope_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);

// 融合注意力（分块在线 softmax）
torch::Tensor flash_attn_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double scale);

// 融合 lm_head + 分块交叉熵：返回标量损失
torch::Tensor chunked_ce_fwd(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor targets,
    int64_t chunk_size, int64_t ignore_index);

// KDA 融合前向（chunk 间状态传递，chunk 内并行）
std::vector<torch::Tensor> kda_chunk_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor gate, torch::Tensor beta, int64_t chunk_size);

}  // namespace verse_nn
