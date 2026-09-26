#!/usr/bin/env bash
# 编译华为昇腾 CANN 自研算子
#
# 环境要求：CANN 工具链 + torch_npu
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 可选环境变量：
#   SOC_VERSION   ascend910b（默认）/ ascend310p
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PKG_DIR}/..:${PYTHONPATH:-}"

export SOC_VERSION="${SOC_VERSION:-ascend910b}"

if [[ -z "${ASCEND_HOME_PATH:-}" && ! -d /usr/local/Ascend ]]; then
  echo "错误：未检测到 CANN，请先 source set_env.sh。" >&2
  exit 1
fi

echo "== 编译昇腾 CANN 自研算子 (SOC: ${SOC_VERSION}) =="
bash "${SCRIPT_DIR}/ascend/build.sh"
python -m verse_nn.kernels.build --info
