#ifndef VERSE_ASCEND_KDA_CHUNK_TILING_H
#define VERSE_ASCEND_KDA_CHUNK_TILING_H

#include <cstdint>

struct KdaChunkTiling {
  int32_t batch;
  int32_t heads;
  int32_t seq_len;
  int32_t head_dim;
  int32_t chunk_size;     // 保留：chunkwise 并行版本用（当前为递归实现）
  int32_t bh_per_core;    // 每核处理的 (batch*head) 数
};

#endif  // VERSE_ASCEND_KDA_CHUNK_TILING_H
