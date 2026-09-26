#!/usr/bin/env bash
# 编译 CUDA 自研算子（等价于 python -m verse_nn.kernels.build --backend cuda）
#
# 环境要求：CUDA 版 PyTorch + nvcc（CUDA 11.8+ 建议，本仓库按 12.x 开发）
# 可选环境变量：
#   TORCH_CUDA_ARCH_LIST  目标架构，默认 "8.0;8.6;8.9;9.0"
#   VERSE_NN_JIT          设 1 时运行期按需编译
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # verse_nn/
export PYTHONPATH="${PKG_DIR}/..:${PYTHONPATH:-}"   # 让 python 能 import verse_nn

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

echo "== 编译 CUDA 自研算子 (arch: ${TORCH_CUDA_ARCH_LIST}) =="
python -m verse_nn.kernels.build --backend cuda --clean "$@"
python -m verse_nn.kernels.build --info
