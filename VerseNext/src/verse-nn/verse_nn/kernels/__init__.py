"""VerseNext 自研算子库：融合算子 + 设备专项优化 + 多后端派发。

公开 API（所有函数都是「设备无关」的，内部自动派发）::

    from verse_nn.kernels import add_rms_norm, swiglu, apply_rope
    from verse_nn.kernels import flash_attention, chunked_cross_entropy, kda_chunkwise

    out, res = add_rms_norm(x, residual, weight, eps=1e-6)

派发顺序：自研扩展（CUDA/ROCm/NPU）→ torch 内建融合 → 参考实现。
用 :func:`set_policy` 可全局关闭融合做 A/B 对照。

后端覆盖
--------
- **CPU**：:mod:`verse_nn.kernels.cpu_ops`（减少分配 + 线程池并行 + int8 量化）；
- **CUDA/ROCm**：:mod:`verse_nn.kernels.cuda_ops`（``csrc/cuda`` 手写 kernel，hipify 复用）；
- **CANN NPU**：:mod:`verse_nn.kernels.npu_ops`（CANN 原生融合算子 + 自研 AscendC）。

量化推理见 :mod:`verse_nn.kernels.quant`；设备能力见 :mod:`verse_nn.devices`。
"""

from verse_nn.kernels._dispatch import (
    KernelPolicy,
    add_rms_norm,
    apply_rope,
    available_ops,
    backend_of,
    chunked_cross_entropy,
    flash_attention,
    get_policy,
    kda_chunkwise,
    set_policy,
    swiglu,
)
from verse_nn.kernels._ext import extension_available, extension_info, get_ops, load_extension
from verse_nn.kernels.quant import (
    Int8Linear,
    int8_gemm_available,
    quantize_dynamic_int8,
    quantize_linear,
    quantize_model_int8,
)
from verse_nn.kernels import reference

__all__ = [
    # 融合算子
    "add_rms_norm",
    "swiglu",
    "apply_rope",
    "flash_attention",
    "chunked_cross_entropy",
    "kda_chunkwise",
    # 策略
    "KernelPolicy",
    "get_policy",
    "set_policy",
    "backend_of",
    "available_ops",
    # 扩展管理
    "extension_available",
    "extension_info",
    "get_ops",
    "load_extension",
    # 量化
    "Int8Linear",
    "int8_gemm_available",
    "quantize_linear",
    "quantize_model_int8",
    "quantize_dynamic_int8",
    # 参考实现
    "reference",
]
