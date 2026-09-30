/*
 * AddRmsNorm 的 Tiling 结构（host 与 kernel 共享）
 *
 * Tiling = 把「怎么切分数据、每核干多少、UB 里放多大一块」从 host 侧算好，
 * 通过 tiling GM 传给 kernel，避免 kernel 里做昂贵的动态规划。
 */

#ifndef VERSE_ASCEND_ADD_RMS_NORM_TILING_H
#define VERSE_ASCEND_ADD_RMS_NORM_TILING_H

#include <cstdint>

struct AddRmsNormTiling {
  int32_t rows;          // 总行数 = numel / d
  int32_t d;             // 归一化维度（head_dim）
  int32_t tile_elems;    // UB 单次处理的元素数
  int32_t rows_per_core; // 每核处理的行数
  float eps;             // RMSNorm 的 eps
};

#endif  // VERSE_ASCEND_ADD_RMS_NORM_TILING_H
