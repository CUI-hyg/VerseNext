#ifndef VERSE_ASCEND_FLASH_ATTN_TILING_H
#define VERSE_ASCEND_FLASH_ATTN_TILING_H

#include <cstdint>

#include "tiling/tiling_api.h"

struct FlashAttnTiling {
  int32_t batch;
  int32_t heads;
  int32_t seq_len;         // q 序列长度
  int32_t kv_len;          // k/v 序列长度
  int32_t head_dim;
  int32_t q_tile;          // Br
  int32_t kv_tile;         // Bc
  int32_t q_tiles_per_core;
  int32_t causal;          // 1 = 因果
  float scale;             // 1/sqrt(D)

  // 两个 Matmul（QK^T / PV）的 Cube 侧 tiling，由 host 用
  // matmul_tiling::MultiCoreMatmulTiling 生成后下发
  TCubeTiling mm_qk_tiling;
  TCubeTiling mm_pv_tiling;
};

#endif  // VERSE_ASCEND_FLASH_ATTN_TILING_H
