"""自研算子扩展的加载与可用性探测。

自研 kernel 编译产物按后端分目录存放::

    verse_nn/kernels/_build/
        cuda/verse_nn_kernels.so
        rocm/verse_nn_kernels.so
        npu/verse_nn_kernels.so

查找顺序：
1. 环境变量 ``VERSE_NN_KERNEL_LIB``（显式指定单个 ``.so``）；
2. 包内 ``_build/<backend>/``；
3. JIT 编译（仅当 ``VERSE_NN_JIT=1``，默认关闭，避免 import 期卡住）。

**关键约束**：本模块在 import 期绝不触发编译、绝不因缺扩展而抛异常——
没有自研算子时返回 ``None``，派发层自然回退到 torch 内建/参考实现。
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

from verse_core import get_logger

logger = get_logger("kernels.ext")

__all__ = [
    "extension_available",
    "load_extension",
    "get_ops",
    "extension_info",
    "BUILD_DIR",
]

BUILD_DIR = Path(__file__).resolve().parent / "_build"

#: 后端 -> 已加载的算子命名空间（None 表示不可用，未探测则为「未缓存」）
_LOADED: dict[str, object | None] = {}
#: 后端 -> 加载失败原因（诊断用）
_ERRORS: dict[str, str] = {}

# 后端别名：cuda/rocm 共用 torch.device("cuda")，但编译产物不同
_ALIASES = {"cuda": ("cuda",), "rocm": ("rocm",), "npu": ("npu",), "xpu": ("xpu",)}


def _candidates(backend: str) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("VERSE_NN_KERNEL_LIB")
    if env:
        out.append(Path(env))
    for alias in _ALIASES.get(backend, (backend,)):
        out.extend(sorted((BUILD_DIR / alias).glob("verse_nn_kernels*.so")))
    return [p for p in out if p.exists()]


def _torch_ops_ns():
    """返回 ``torch.ops.verse_nn`` 命名空间（未注册返回 None）。"""
    try:
        return torch.ops.verse_nn
    except AttributeError:
        return None


def load_extension(backend: str, *, jit: bool | None = None):
    """加载指定后端的自研算子扩展。

    Args:
        backend: ``cuda`` / ``rocm`` / ``npu`` / ``xpu`` / ``cpu``。
        jit: 是否允许 JIT 编译；None 时读 ``VERSE_NN_JIT`` 环境变量。

    Returns:
        算子命名空间（``torch.ops.verse_nn``）或 None。
    """
    if backend in _LOADED:
        return _LOADED[backend]

    if jit is None:
        jit = os.environ.get("VERSE_NN_JIT", "0") == "1"

    # 1) 先看是否已通过别的路径注册（例如用户 import 了预编译 wheel）
    ns = _torch_ops_ns()
    if ns is not None and _has_required_ops(ns):
        _LOADED[backend] = ns
        logger.info("检测到已注册的自研算子（backend=%s）", backend)
        return ns

    # 2) 加载预编译 .so
    for path in _candidates(backend):
        try:
            torch.ops.load_library(str(path))
            ns = _torch_ops_ns()
            if ns is not None and _has_required_ops(ns):
                _LOADED[backend] = ns
                logger.info("已加载自研算子扩展: %s", path)
                return ns
            _ERRORS[backend] = f"{path} 已加载但缺少必需算子"
        except Exception as exc:  # pragma: no cover - 平台相关
            _ERRORS[backend] = f"加载 {path} 失败: {exc}"
            logger.warning("加载自研算子扩展失败 %s: %s", path, exc)

    # 3) JIT 编译（默认关闭）
    if jit:
        try:
            ns = _jit_build(backend)
            if ns is not None:
                _LOADED[backend] = ns
                return ns
        except Exception as exc:  # pragma: no cover - 需要工具链
            _ERRORS[backend] = f"JIT 编译失败: {exc}"
            logger.warning("自研算子 JIT 编译失败(%s): %s", backend, exc)

    _LOADED[backend] = None
    _ERRORS.setdefault(backend, "未找到预编译扩展（设置 VERSE_NN_JIT=1 可尝试即时编译）")
    return None


def _jit_build(backend: str):
    """按需 JIT 编译（需要本机有 nvcc/hipcc/CANN 工具链）。"""
    try:
        module = importlib.import_module("verse_nn.kernels.build")
    except Exception as exc:
        _ERRORS[backend] = f"无法导入构建模块: {exc}"
        return None
    module.build(backend=backend, verbose=True)
    ns = _torch_ops_ns()
    return ns if (ns is not None and _has_required_ops(ns)) else None


#: 判定「扩展可用」所需的最小算子集合（缺任一即视为不可用）。
#: 名字必须与 ``csrc/cuda/bindings.cpp`` / ``csrc/ascend/bindings_npu.cpp``
#: 里 ``m.def`` 注册的 schema 名严格一致（tests/test_kernel_build.py 有静态校验）。
_REQUIRED_OPS = (
    "add_rms_norm_fwd",
    "swiglu_fwd",
    "flash_attn_fwd",
)


def _has_required_ops(ns) -> bool:
    for name in _REQUIRED_OPS:
        if not hasattr(ns, name):
            return False
    return True


def _flag_override(backend: str) -> bool | None:
    """读取 devices 层的显式覆盖（测试/构建脚本用，见 ``set_custom_kernels_available``）。

    派发层用 :func:`extension_available` 决定是否走自研算子，因此覆盖标志必须在这里
    生效，否则「强制可用/不可用」对实际派发无效。惰性 import 避免循环依赖。
    """
    try:
        from verse_nn.devices.backend import _CUSTOM_KERNELS_FLAG
    except Exception:
        return None
    if backend in _CUSTOM_KERNELS_FLAG:
        return _CUSTOM_KERNELS_FLAG[backend]
    return _CUSTOM_KERNELS_FLAG.get("__all__")


def extension_available(backend: str | None = None) -> bool:
    """自研算子是否可用（不触发编译）。

    ``backend=None`` 时按当前设备推断；供 :class:`DeviceCaps.has_custom_kernels`
    使用，因此必须**廉价且无副作用**。
    """
    if backend is None:
        backend = _current_backend_name()
    override = _flag_override(backend)
    if override is not None:
        return override
    if backend == "cpu":
        # CPU 侧「自研算子」是 Python/ATen 级优化，不走 .so，恒可用
        return True
    if backend in _LOADED:
        return _LOADED[backend] is not None
    # 只探测预编译产物，不做 JIT
    if _candidates(backend):
        return load_extension(backend, jit=False) is not None
    ns = _torch_ops_ns()
    if ns is not None and _has_required_ops(ns):
        _LOADED[backend] = ns
        return True
    _LOADED[backend] = None
    return False


def _current_backend_name() -> str:
    """推断当前默认设备的后端名（避免与 devices 包循环 import）。"""
    try:
        from verse_nn.devices import detect_backend

        return detect_backend().name
    except Exception:
        return "cpu"


def get_ops(backend: str | None = None):
    """取算子命名空间；不可用返回 None（调用方须回退）。"""
    if backend is None:
        backend = _current_backend_name()
    if backend == "cpu":
        return None
    return load_extension(backend)


def extension_info() -> dict:
    """诊断信息：各后端扩展状态与失败原因（CLI ``kernels --info`` 用）。"""
    info: dict = {"build_dir": str(BUILD_DIR), "backends": {}}
    for backend in ("cuda", "rocm", "npu", "xpu", "cpu"):
        candidates = [str(p) for p in _candidates(backend)]
        info["backends"][backend] = {
            "available": extension_available(backend),
            "candidates": candidates,
            "error": _ERRORS.get(backend),
        }
    return info
