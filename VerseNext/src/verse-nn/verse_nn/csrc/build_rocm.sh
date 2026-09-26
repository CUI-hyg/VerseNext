#!/usr/bin/env bash
# 编译 ROCm（AMD HIP）自研算子
#
# 关键点：**无需手工维护 HIP 源码**。PyTorch 的 cpp_extension 在 ROCm 构建下
# 会自动把 .cu hipify 成 HIP 并调用 hipcc；kernel 里的 warp 级原语已按
# VERSE_WARP_SIZE（AMD=64）与 __HIP_PLATFORM_AMD__ 分支适配。
#
# 环境要求：ROCm 版 PyTorch + hipcc（ROCm 5.7+）
# 可选环境变量：
#   PYTORCH_ROCM_ARCH  目标架构，默认 "gfx90a;gfx942"（MI200/MI300）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PKG_DIR}/..:${PYTHONPATH:-}"

export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a;gfx942}"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "错误：未找到 hipcc，请确认 ROCm 环境（source /opt/rocm/bin/env.sh）。" >&2
  exit 1
fi

echo "== 编译 ROCm 自研算子 (arch: ${PYTORCH_ROCM_ARCH}) =="
echo "   PyTorch 会在编译时自动 hipify csrc/cuda/*.cu"
python -m verse_nn.kernels.build --backend rocm --clean "$@"
python -m verse_nn.kernels.build --info
