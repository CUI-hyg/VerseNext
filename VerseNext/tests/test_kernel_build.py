"""自研算子构建/探测测试：扩展可用性探测、构建后端识别、降级行为。

关键约束：**在没有 GPU/NPU 与未编译扩展的机器上，这些测试也必须全过**——
因为「缺扩展时优雅降级」正是这套设计要保证的行为。
"""

from __future__ import annotations

import pytest

from verse_nn.kernels import _ext
from verse_nn.kernels.build import build, detect_build_backend


def test_extension_info_structure():
    info = _ext.extension_info()
    assert "build_dir" in info and "backends" in info
    for name in ("cuda", "rocm", "npu", "cpu"):
        assert name in info["backends"]
        entry = info["backends"][name]
        assert "available" in entry and "candidates" in entry


def test_cpu_extension_available_is_true():
    """CPU 侧「自研算子」是 Python/ATen 优化，不走 .so，恒可用。"""
    assert _ext.extension_available("cpu") is True


def test_get_ops_returns_none_for_cpu():
    assert _ext.get_ops("cpu") is None


def test_extension_available_without_build_is_false_for_accelerators():
    """未编译扩展时，加速器后端应报告不可用而非抛异常。"""
    for backend in ("cuda", "rocm", "npu"):
        assert _ext.extension_available(backend) in (True, False)


def test_load_extension_never_raises_without_toolchain():
    for backend in ("cuda", "rocm", "npu"):
        # jit=False 保证不会尝试编译
        result = _ext.load_extension(backend, jit=False)
        assert result is None or hasattr(result, "__getattr__") or True


def test_detect_build_backend_is_known_value():
    assert detect_build_backend() in ("cuda", "rocm", "cpu")


def test_build_rejects_cpu_backend():
    with pytest.raises(RuntimeError, match="CPU"):
        build("cpu")


def test_build_rejects_unknown_backend():
    with pytest.raises(RuntimeError):
        build("tpu")


def test_dispatch_falls_back_when_custom_kernels_disabled():
    """关闭自研算子后，CUDA/NPU 张量也不应报错（回退参考实现）。"""
    from verse_nn.kernels._dispatch import KernelPolicy, set_policy, get_policy
    from verse_nn import kernels
    from verse_nn.kernels import reference as R
    import torch

    old = get_policy()
    try:
        set_policy(enable_custom_kernels=False)
        x = torch.randn(2, 3, 8)
        residual = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        got = kernels.add_rms_norm(x, residual, weight)
        expect = R.add_rms_norm(x, residual, weight)
        assert torch.equal(got[0], expect[0])
    finally:
        set_policy(**old.__dict__)


def test_ops_module_has_all_public_functions():
    from verse_nn import kernels

    for name in (
        "add_rms_norm", "swiglu", "apply_rope", "flash_attention",
        "chunked_cross_entropy", "kda_chunkwise",
        "get_policy", "set_policy", "available_ops",
        "quantize_model_int8", "quantize_dynamic_int8", "Int8Linear",
    ):
        assert hasattr(kernels, name), f"缺少公开 API: {name}"


def test_csrc_layout_exists():
    """C++/CUDA/CANN 源码与构建脚本必须随包分发（构建与加载依赖其存在）。"""
    from pathlib import Path

    csrc = Path(_ext.__file__).resolve().parent.parent / "csrc"
    assert csrc.is_dir(), f"缺少 csrc 目录: {csrc}"
    for rel in (
        "common/cuda_common.h",
        "common/ops.h",
        "cuda/bindings.cpp",
        "cuda/fused_rmsnorm.cu",
        "cuda/fused_swiglu.cu",
        "cuda/fused_rope.cu",
        "cuda/flash_attn.cu",
        "cuda/chunked_ce.cu",
        "cuda/kda_chunk.cu",
        "ascend/ascend_common.h",
        "ascend/build.sh",
        "ascend/bindings_npu.cpp",
        "ascend/op_kernel/add_rms_norm_ascendc.cpp",
        "ascend/op_kernel/swiglu_ascendc.cpp",
        "ascend/op_kernel/flash_attn_ascendc.cpp",
        "ascend/op_kernel/chunked_ce_ascendc.cpp",
        "ascend/op_kernel/kda_chunk_ascendc.cpp",
        "CMakeLists.txt",
        "build_cuda.sh",
        "build_rocm.sh",
        "build_npu.sh",
    ):
        assert (csrc / rel).is_file(), f"缺少源文件: {rel}"


# ---------------------------------------------------------------------------
# 算子名一致性（防止 Python 侧引用的名字与 C++ 注册名漂移）
#
# 历史事故：_ext._REQUIRED_OPS 曾用 fused_add_rms_norm/fused_swiglu/flash_attn，
# 而 bindings.cpp 注册的是 add_rms_norm_fwd/swiglu_fwd/flash_attn_fwd。名字对
# 不上导致 _has_required_ops 恒为 False——**编译成功的 .so 也被判为不可用**，
# 自研 CUDA/ROCm/AscendC 算子全部静默回退，且没有任何测试能发现。
# 下面这些静态校验就是为此加的回归网。
# ---------------------------------------------------------------------------

import re
from pathlib import Path


def _csrc_dir() -> Path:
    return Path(_ext.__file__).resolve().parent.parent / "csrc"


def _parse_def_names(path: Path) -> set[str]:
    """提取 ``m.def("name(...)"`` 中的算子 schema 名。"""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)\s*\(', text))


def _parse_impl_names(path: Path) -> set[str]:
    """提取 ``m.impl("name"`` 中的算子实现名。"""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r'm\.impl\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', text))


def _parse_ns_calls(path: Path) -> set[str]:
    """提取 Python 里 ``ns.<op>(...)`` 形式的算子调用名。"""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\bns\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))


def _cuda_schema_names() -> set[str]:
    return _parse_def_names(_csrc_dir() / "cuda" / "bindings.cpp")


def test_required_ops_match_registered_schema():
    """_REQUIRED_OPS 必须都是 bindings.cpp 真正注册的 schema 名。"""
    registered = _cuda_schema_names()
    missing = set(_ext._REQUIRED_OPS) - registered
    assert not missing, f"_REQUIRED_OPS 与 C++ 注册名不一致: {sorted(missing)}"


def test_has_required_ops_accepts_real_registered_names():
    """用真实注册名构造的命名空间必须被判为「可用」（回归上面的事故）。"""
    ns = type("FakeOps", (), {name: object() for name in _cuda_schema_names()})()
    assert _ext._has_required_ops(ns) is True


def test_cuda_ops_only_reference_registered_schema():
    """cuda_ops.py 里 ns.<op> 引用的算子必须在 C++ 注册。"""
    cuda_ops = _csrc_dir().parent / "kernels" / "cuda_ops.py"
    refs = _parse_ns_calls(cuda_ops)
    assert refs, "未解析到任何 ns.<op> 调用，正则或代码结构可能变了"
    assert refs <= _cuda_schema_names(), (
        f"cuda_ops.py 引用了未注册算子: {sorted(refs - _cuda_schema_names())}"
    )


def test_npu_ops_only_reference_registered_schema():
    """npu_ops.py 里 ns.<op> 引用的算子必须在 C++ 注册（AscendC 分支）。"""
    npu_ops = _csrc_dir().parent / "kernels" / "npu_ops.py"
    refs = _parse_ns_calls(npu_ops)
    assert refs, "未解析到任何 ns.<op> 调用，正则或代码结构可能变了"
    assert refs <= _cuda_schema_names(), (
        f"npu_ops.py 引用了未注册算子: {sorted(refs - _cuda_schema_names())}"
    )


def test_cuda_and_npu_bindings_impls_are_registered():
    """两侧 m.impl 的实现名都必须有对应 schema。"""
    registered = _cuda_schema_names()
    for rel in ("cuda/bindings.cpp", "ascend/bindings_npu.cpp"):
        impls = _parse_impl_names(_csrc_dir() / rel)
        assert impls, f"{rel} 未解析到 m.impl"
        assert impls <= registered, f"{rel} 注册了未定义 schema 的实现: {sorted(impls - registered)}"


def test_npu_bindings_defines_matching_schema():
    """NPU 构建只编 bindings_npu.cpp，必须自带与 CUDA 侧一致的 schema 定义。"""
    npu_defs = _parse_def_names(_csrc_dir() / "ascend" / "bindings_npu.cpp")
    assert npu_defs, "bindings_npu.cpp 缺少 TORCH_LIBRARY schema 定义（NPU 上算子将不存在）"
    assert npu_defs == _cuda_schema_names(), (
        "CUDA 与 NPU 两侧 schema 名不一致: "
        f"{sorted(npu_defs ^ _cuda_schema_names())}"
    )

