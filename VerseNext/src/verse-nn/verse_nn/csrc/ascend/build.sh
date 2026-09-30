#!/usr/bin/env bash
# =============================================================================
# VerseNext 自研算子 —— 华为昇腾 CANN 构建脚本
#
# 流程（CANN 自定义算子的标准四步）：
#   1. msopgen gen     —— 用算子描述 json 生成算子工程骨架（op_host/op_kernel/framework）
#   2. 覆盖骨架        —— 把本目录的 op_host/*.cpp 与 op_kernel/*_ascendc.cpp 拷进去
#   3. bash build.sh   —— 编译生成自定义算子包 *.run 并安装到 CANN 的 vendors/customize
#   4. 编译 torch 适配层 bindings_npu.cpp（链接已安装的 ACLNN 头/库）
#
# 前置条件：
#   - 已安装 CANN（默认取 $ASCEND_HOME_PATH）
#   - source ${ASCEND_HOME_PATH}/set_env.sh
#
# 用法：
#   bash build.sh
#   SOC_NAME=Ascend910B1 bash build.sh        # 手工指定芯片（默认自动探测）
#
# 说明（踩过的坑，改动前务必读）：
#   - 自定义算子名必须避开 CANN 内建算子（内建已有 AddRmsNorm / SwiGlu）。
#     同名会命中内建的 tiling/op_api，还会让 ``torch_npu`` 的
#     ``npu_add_rms_norm`` 意外解析到我们的实现。故一律加 ``Verse`` 前缀。
#   - kernel 入口必须调用 ``REGISTER_TILING_DEFAULT(<Tiling>)``，否则框架按
#     msopgen 骨架里的占位结构（4 字节）分配 tiling data，host 侧
#     ``GetTilingData<T>()`` 返回 nullptr，tiling 报 561002。
#   - 单入口 kernel 二进制下 host 必须 ``SetTilingKey(0)``；下发非 0 会报
#     ``BinaryGetFunctionByEntry ... funcEntry is invalid``（ret=361001）。
#   - dtype 分派靠构建期注入的 ``DTYPE_<INPUTNAME>`` 宏（msopgen 自动加），
#     不要用 TILING_KEY_IS 做 dtype 分派——那只是运行期比较，不产生多入口。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${VERSE_OUT_DIR:-${SCRIPT_DIR}/../../kernels/_build/npu}"
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
WORK_DIR="${SCRIPT_DIR}/_cann_work"
OPP_DIR="${WORK_DIR}/verse_ops"

# 本目录里真正参与 CANN 编译的算子（json 名 == kernel 源文件名）。
# 其余 op_host/op_kernel 源文件（flash_attn / chunked_ce / kda_chunk）尚未在
# 硬件上验证，暂不打包；NPU 上由 CANN 原生算子或参考实现覆盖。
OPS="verse_add_rms_norm verse_swiglu"

echo "== VerseNext CANN 构建 =="
echo "  CANN_HOME : ${CANN_HOME}"
echo "  OUT_DIR   : ${OUT_DIR}"

if [[ ! -d "${CANN_HOME}" ]]; then
  echo "错误：未找到 CANN（${CANN_HOME}）。请 source set_env.sh 或设置 ASCEND_HOME_PATH。" >&2
  exit 1
fi

if ! command -v msopgen >/dev/null 2>&1; then
  echo "错误：未找到 msopgen，请确认 CANN 环境已 source。" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# 0) 探测芯片型号 -> msopgen 的 compute unit
#    msopgen 的 -c 形如 ai_core-<SOC 名>，SOC 名即 platform_config/*.ini 的名字
#    （如 Ascend910_9362）。优先用 torch_npu 的 get_device_name，失败则回退 910B。
# -----------------------------------------------------------------------------
detect_soc_name() {
  if [[ -n "${SOC_NAME:-}" ]]; then echo "${SOC_NAME}"; return; fi
  local name
  name="$(python -c '
import torch, torch_npu  # noqa: F401
print(torch_npu.npu.get_device_name(0))
' 2>/dev/null | tr -d "[:space:]" | tail -c 64 || true)"
  if [[ -n "${name}" ]]; then echo "${name}"; else echo "Ascend910B1"; fi
}

SOC_NAME="$(detect_soc_name)"
CORE_TYPE="ai_core-${SOC_NAME}"
echo "  SOC_NAME  : ${SOC_NAME}  (compute unit: ${CORE_TYPE})"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
mkdir -p "${OUT_DIR}"

# -----------------------------------------------------------------------------
# 1) 生成算子描述 json（msopgen 的输入）
#    算子名 / 输入输出名必须与 op_host、op_kernel 中的声明一致：
#    msopgen 由输入名大写推导编译期 dtype 宏（DTYPE_X / DTYPE_GATE ...）。
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

# 只声明 fp16 / fp32：AscendC 的 Vector 指令（Muls/Exp/Sigmoid...）不支持 bf16，
# bf16 输入在 Python 侧由 CANN 原生算子覆盖（见 kernels/npu_ops.py）。
DTYPES='"float16","float"'

gen_json "VerseAddRmsNorm" "${WORK_DIR}/verse_add_rms_norm.json" \
  '{"name":"x","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"residual","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"weight","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"res_out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"eps","param_type":"optional","type":"float","default_value":1e-6}'

gen_json "VerseSwiglu" "${WORK_DIR}/verse_swiglu.json" \
  '{"name":"gate","param_type":"required","format":["ND"],"type":['"${DTYPES}"']},
   {"name":"up","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' \
  '{"name":"out","param_type":"required","format":["ND"],"type":['"${DTYPES}"']}' ''

# -----------------------------------------------------------------------------
# 2) msopgen 生成工程骨架，并把我们的实现覆盖进去
#    第一个算子用默认模式建工程，后续算子用 -m 1 追加（否则报 project not empty）。
# -----------------------------------------------------------------------------
echo "-- msopgen 生成算子工程骨架 (${CORE_TYPE})"
first=1
for op in ${OPS}; do
  if [[ ${first} -eq 1 ]]; then
    msopgen gen -i "${WORK_DIR}/${op}.json" -c "${CORE_TYPE}" -lan cpp -out "${OPP_DIR}"
    first=0
  else
    msopgen gen -i "${WORK_DIR}/${op}.json" -c "${CORE_TYPE}" -lan cpp -out "${OPP_DIR}" -m 1
  fi
done

echo "-- 读取骨架里的 AICore 配置名（必须在覆盖源码之前读）"
# 骨架里的 AddConfig(<soc>) 是 msopgen 按 compute unit 推出的（如
# Ascend910_9362 -> ascend910_93），直接复用，避免自己维护这张映射表。
# 这个值还会决定后续生成的 aic-<soc>-ops-info.ini 文件名，读错会导致
# build 找不到 ops-info.ini 而失败。
GEN_OP_HOST="${OPP_DIR}/op_host/$(echo "${OPS}" | awk '{print $1}').cpp"
SOC_CONFIG="$(sed -n 's/.*AddConfig("\([^"]*\)").*/\1/p' "${GEN_OP_HOST}" | head -1)"
SOC_CONFIG="${SOC_CONFIG:-ascend910b}"
echo "  AICore config: ${SOC_CONFIG}"

echo "-- 覆盖 op_host / op_kernel 实现"
for op in ${OPS}; do
  cp "${SCRIPT_DIR}/op_host/${op}.cpp" "${OPP_DIR}/op_host/${op}.cpp"
  cp "${SCRIPT_DIR}/op_kernel/${op}_ascendc.cpp" "${OPP_DIR}/op_kernel/${op}.cpp"
done
cp "${SCRIPT_DIR}"/op_host/*_tiling.h "${OPP_DIR}/op_host/"
# ascend_common.h 位于上一级，kernel 用 ../ 引用，需要一起拷进去
cp "${SCRIPT_DIR}/ascend_common.h" "${OPP_DIR}/ascend_common.h"

cat > "${OPP_DIR}/op_host/ascend_soc.h" <<EOF
#ifndef VERSE_ASCEND_SOC_H
#define VERSE_ASCEND_SOC_H
// 由 build.sh 按 msopgen 的 compute unit 自动生成；勿手工改。
#define VERSE_ASCEND_SOC "${SOC_CONFIG}"
#endif
EOF

# -----------------------------------------------------------------------------
# 3) 编译并安装算子包
# -----------------------------------------------------------------------------
echo "-- 编译 CANN 自定义算子（耗时较长）"
cd "${OPP_DIR}"
bash build.sh

RUN_PKG="$(find "${OPP_DIR}/build_out" -name "*.run" | head -1 || true)"
if [[ -n "${RUN_PKG}" ]]; then
  echo "-- 安装算子包: ${RUN_PKG}"
  bash "${RUN_PKG}" --quiet || echo "警告：算子包安装失败（可能已安装）"
fi

# -----------------------------------------------------------------------------
# 4) 编译 torch 适配层 bindings_npu.cpp
#    直接走 CANN 的 ACL C API（不依赖 torch_npu 未导出的 EXEC_NPU_CMD 符号），
#    因此需要链 torch_npu（取 c10_npu::getCurrentNPUStream）与自定义算子
#    op_api 库（取 aclnnVerseXxx）。
# -----------------------------------------------------------------------------
echo "-- 编译 torch 适配层 bindings_npu.cpp"
TORCH_INC="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"include"))')"
PY_INC="$(python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
NPU_INC="$(python -c 'import torch_npu,os;print(os.path.dirname(torch_npu.__file__))' 2>/dev/null || echo "")"
VENDOR_API="${CANN_HOME}/opp/vendors/customize/op_api"

g++ -shared -fPIC -std=c++17 -O3 \
  -DVERSE_HAS_CANN=1 \
  -I"${SCRIPT_DIR}" \
  -I"${TORCH_INC}" -I"${TORCH_INC}/torch/csrc/api/include" \
  -I"${PY_INC}" \
  ${NPU_INC:+-I"${NPU_INC}/include"} \
  -I"${CANN_HOME}/include" \
  -I"${CANN_HOME}/include/aclnn" \
  -I"${VENDOR_API}/include" \
  "${SCRIPT_DIR}/bindings_npu.cpp" \
  -L"${TORCH_INC}/../lib" -ltorch -ltorch_cpu -lc10 \
  -L"${CANN_HOME}/lib64" -lascendcl -lnnopbase \
  ${NPU_INC:+-L"${NPU_INC}/lib" -ltorch_npu} \
  -L"${VENDOR_API}/lib" -lcust_opapi \
  -Wl,-rpath,"${VENDOR_API}/lib" \
  -o "${OUT_DIR}/verse_nn_kernels.so"

echo "== 构建完成: ${OUT_DIR}/verse_nn_kernels.so =="
