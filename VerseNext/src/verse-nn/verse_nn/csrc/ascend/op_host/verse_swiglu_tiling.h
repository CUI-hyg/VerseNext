#ifndef VERSE_ASCEND_SWIGLU_TILING_H
#define VERSE_ASCEND_SWIGLU_TILING_H

#include <cstdint>

struct SwigluTiling {
  int32_t total;          // 元素总数
  int32_t tile_elems;     // UB 单次处理元素数
  int32_t tiles_per_core; // 每核处理的 tile 数
};

#endif  // VERSE_ASCEND_SWIGLU_TILING_H
