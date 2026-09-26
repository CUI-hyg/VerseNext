#ifndef VERSE_ASCEND_CHUNKED_CE_TILING_H
#define VERSE_ASCEND_CHUNKED_CE_TILING_H

#include <cstdint>

#include "tiling/tiling_api.h"

struct ChunkedCeTiling {
  int32_t rows;            // B*S
  int32_t vocab;           // V
  int32_t hidden_dim;      // D
  int32_t vocab_tile;      // vocab 维分块长度
  int32_t rows_per_core;
  int32_t ignore_index;
  TCubeTiling mm_tiling;   // hidden(1×D) @ W_tile^T(D×cnt) 的 Cube tiling
};

#endif  // VERSE_ASCEND_CHUNKED_CE_TILING_H
