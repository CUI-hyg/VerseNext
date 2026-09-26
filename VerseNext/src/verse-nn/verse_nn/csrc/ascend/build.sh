#!/usr/bin/env bash
# =============================================================================
# VerseNext 自研算子 —— 华为昇腾 CANN 构建脚本
#
# 流程（CANN 自定义算子的标准四步）：
#   1. msopgen gen  —— 用算子描述 json 生成算子工程骨架（op_host/op_kernel/framework）
#   2. 覆盖骨架     —— 把本目录的 op_host/*.cpp 与 op_kernel/*_ascendc.cpp 拷进去
#   3. bash build.sh —— 编译生成自定义算子包 *.run
#   4. 编译 torch 适配层 bindings_npu.cpp（链接已安装的 ACLNN 头/库）
#
# 前置条件：
#   - 已安装 CANN（默认 /usr/local/Ascend/ascend-toolkit/latest）
#   - source ${ASCEND_HOME_PATH}/set_env.sh
#   - 环境变量 SOC_VERSION 指定芯片（ascend910b / ascend310p），默认 ascend910b
#
# 用法：
#   bash build.sh                    # 默认 910b
#   SOC_VERSION=ascend310p bash build.sh
#
# ⚠ 未在无 CANN 环境验证；首次运行请按 msopgen 报错调整 json 与 SOC_VERSION。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${VERSE_OUT_DIR:-${SCRIPT_DIR}/../../kernels/_build/npu}"
SOC_VERSION="${SOC_VERSION:-ascend910b}"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
WORK_DIR="${SCRIPT_DIR}/_cann_work"
OPP_DIR="${SCRIPT_DIR}/_cann_work/verse_ops"

echo "== VerseNext CANN 构建 =="
echo "  SOC_VERSION : ${SOC_VERSION}"
echo "  CANN_HOME   : ${CANN_HOME}"
echo "  OUT_DIR     : ${OUT_DIR}"

if [[ ! -d "${CANN_HOME}" ]]; then
  echo "错误：未找到 CANN（${CANN_HOME}）。请 source set_env.sh 或设置 ASCEND_HOME_PATH。" >&2
  exit 1
fi

if ! command -v msopgen >/dev/null 2>&1; then
  echo "错误：未找到 msopgen，请确认 CANN 环境已 source。" >&2
  exit 1
fi

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
mkdir -p "${OUT_DIR}"

# -----------------------------------------------------------------------------
# 1) 生成算子描述 json（msopgen 的输入）
#    每个算子一个 json：输入/输出名、dtype、格式、属性
# -----------------------------------------------------------------------------
gen_json() {
  local name="$1" out="$2" inputs="$3" outputs="$4" attrs="$5"
  cat > "${out}" <<EOF
[
  {
    "op": "${name}",
    "input_desc": [${inputs}],
    "output_desc": [${outputs}],
    "attr": [${attrs}]
  }
]
EOF
}

DTYPES='"float16","float","bfloat16"'

gen_json "AddRmsNorm" "${WORK_DIR}/add_rms_norm.json" \
  '{"name":"x","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"residual","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"weight","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"res_out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"eps","param_type":"optional","type":"float","default_value":1e-6}'

gen_json "Swiglu" "${WORK_DIR}/swiglu.json" \
  '{"name":"gate","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"up","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' ''

gen_json "FlashAttn" "${WORK_DIR}/flash_attn.json" \
  '{"name":"q","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"k","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"v","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"causal","param_type":"optional","type":"bool","default_value":true}'

gen_json "ChunkedCe" "${WORK_DIR}/chunked_ce.json" \
  '{"name":"hidden","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"weight","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"targets","param_type":"required","format":["ND"],"type":["int32"]}' \
  '{"name":"loss","param_type":"required","format":["ND"],"type":["float"]}' \
  '{"name":"chunk_size","param_type":"optional","type":"int","default_value":0},
   {"name":"ignore_index","param_type":"optional","type":"int","default_value":-100}'

gen_json "KdaChunk" "${WORK_DIR}/kda_chunk.json" \
  '{"name":"q","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"k","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"v","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"gate","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"beta","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"state","param_type":"required","format":["ND"],"type":["float"]}' \
  '{"name":"chunk_size","param_type":"optional","type":"int","default_value":64}'

# -----------------------------------------------------------------------------
# 2) msopgen 生成工程骨架，并把我们的实现覆盖进去
# -----------------------------------------------------------------------------
SOC_SHORT="$(echo "${SOC_VERSION}" | sed 's/ascend//')"
CORE_TYPE="ai_core-Ascend${SOC_SHORT}"

echo "-- msopgen 生成算子工程骨架 (${CORE_TYPE})"
msopgen gen -i "${WORK_DIR}/add_rms_norm.json" -c "${CORE_TYPE}" -lan cpp -out "${OPP_DIR}"
for j in swiglu flash_attn chunked_ce kda_chunk; do
  msopgen gen -i "${WORK_DIR}/${j}.json" -c "${CORE_TYPE}" -lan cpp -out "${OPP_DIR}"
done

echo "-- 覆盖 op_host / op_kernel 实现"
cp "${SCRIPT_DIR}"/op_host/*.cpp "${SCRIPT_DIR}"/op_host/*.h "${OPP_DIR}/op_host/" 2>/dev/null || true
cp "${SCRIPT_DIR}"/op_kernel/*.cpp "${OPP_DIR}/op_kernel/" 2>/dev/null || true
# ascend_common.h 位于上一级，kernel 用 ../ 引用，需要一起拷进去
cp "${SCRIPT_DIR}/ascend_common.h" "${OPP_DIR}/ascend_common.h"

# -----------------------------------------------------------------------------
# 3) 编译算子包
# -----------------------------------------------------------------------------
echo "-- 编译 CANN 自定义算子"
cd "${OPP_DIR}"
bash build.sh

# -----------------------------------------------------------------------------
# 4) 安装算子包并编译 torch 适配层
# -----------------------------------------------------------------------------
RUN_PKG="$(find "${OPP_DIR}/build_out" -name "*.run" | head -1 || true)"
if [[ -n "${RUN_PKG}" ]]; then
  echo "-- 安装算子包: ${RUN_PKG}"
  bash "${RUN_PKG}" --quiet || echo "警告：算子包安装失败（可能已安装）"
fi

echo "-- 编译 torch 适配层 bindings_npu.cpp"
TORCH_INC="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"include"))')"
PY_INC="$(python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
NPU_INC="$(python -c 'import torch_npu,os;print(os.path.dirname(torch_npu.__file__))' 2>/dev/null || echo "")"

g++ -shared -fPIC -std=c++17 -O3 \
  -DVERSE_HAS_CANN=1 \
  -I"${SCRIPT_DIR}" \
  -I"${TORCH_INC}" -I"${TORCH_INC}/torch/csrc/api/include" \
  -I"${PY_INC}" \
  ${NPU_INC:+-I"${NPU_INC}/include"} \
  -I"${CANN_HOME}/include" \
  -I"${CANN_HOME}/include/aclnn" \
  "${SCRIPT_DIR}/bindings_npu.cpp" \
  -L"${TORCH_INC}/../lib" -ltorch -ltorch_cpu -lc10 \
  -L"${CANN_HOME}/lib64" -lascendcl -lnnopbase \
  -o "${OUT_DIR}/verse_nn_kernels.so"

echo "== 构建完成: ${OUT_DIR}/verse_nn_kernels.so =="
