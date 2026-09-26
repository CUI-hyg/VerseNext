"""自研算子的编译入口（CUDA / ROCm / CANN NPU）。

用法::

    python -m verse_nn.kernels.build --backend cuda        # 或 rocm / npu
    python -m verse_nn.kernels.build --info                # 查看各后端可用性

产物落在 ``verse_nn/kernels/_build/<backend>/verse_nn_kernels.so``，
:mod:`verse_nn.kernels._ext` 会在 import 时按此路径探测并 ``load_library``。

三个后端三条路径
----------------
- **CUDA**：torch ``cpp_extension.load`` 直接编译 ``csrc/cuda/*.cu`` +
  ``bindings.cpp``，用 nvcc，按 ``TORCH_CUDA_ARCH_LIST`` 生成对应 sm 的 cubin。
- **ROCm**：同一份源码；PyTorch 的 cpp_extension 在 ROCm 构建下会**自动
  hipify** ``.cu``（生成 HIP 并调 hipcc），因此无需手工维护两份代码。
- **NPU**：昇腾的自定义算子必须走 CANN 的算子工程（``msopgen`` + ``build.sh``）
  产出 ACLNN 包，再由 ``csrc/ascend/bindings_npu.cpp`` 经 aclnn 调用。
  本模块负责调用 ``csrc/ascend/build.sh`` 并在成功后编出 torch 适配层。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from verse_core import get_logger

logger = get_logger("kernels.build")

CSRC = Path(__file__).resolve().parent.parent / "csrc"
BUILD_ROOT = Path(__file__).resolve().parent / "_build"

_CUDA_SOURCES = [
    "cuda/fused_rmsnorm.cu",
    "cuda/fused_swiglu.cu",
    "cuda/fused_rope.cu",
    "cuda/flash_attn.cu",
    "cuda/chunked_ce.cu",
    "cuda/kda_chunk.cu",
    "cuda/bindings.cpp",
]


def detect_build_backend() -> str:
    """按当前 torch 构建推断可编译的后端。"""
    import torch

    if getattr(torch.version, "hip", None) is not None:
        return "rocm"
    if torch.version.cuda is not None:
        return "cuda"
    # 未编译 CUDA 的 torch：仍可能只是缺运行时，交给调用方显式指定
    return "cpu"


def build(backend: str | None = None, *, verbose: bool = False, clean: bool = False) -> Path:
    """编译指定后端并返回产物 ``.so`` 路径。

    Args:
        backend: ``cuda`` / ``rocm`` / ``npu``；None 时自动探测。
        verbose: 打印编译日志。
        clean: 先清空该后端的 build 目录（排查编译缓存问题时用）。
    """
    backend = backend or detect_build_backend()
    out_dir = BUILD_ROOT / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

    if backend in ("cuda", "rocm"):
        return _build_cuda_like(backend, verbose=verbose)
    if backend == "npu":
        return _build_npu(verbose=verbose)
    raise RuntimeError(
        f"后端 {backend} 不支持编译自研算子；CPU 侧优化走 Python（verse_nn/kernels/cpu_ops.py）"
    )


def _build_cuda_like(backend: str, *, verbose: bool) -> Path:
    """CUDA 与 ROCm 共用：cpp_extension 在 ROCm 构建下自动 hipify。"""
    import torch
    from torch.utils.cpp_extension import load

    if backend == "cuda" and torch.version.cuda is None:
        raise RuntimeError(
            "当前 torch 非 CUDA 构建，无法编译 CUDA 算子。\n"
            "请在 CUDA 版 torch 环境执行，或安装 CUDA 版 PyTorch 后重试。"
        )
    if backend == "rocm" and getattr(torch.version, "hip", None) is None:
        raise RuntimeError("当前 torch 非 ROCm 构建，无法编译 ROCm 算子。")

    sources = [str(CSRC / s) for s in _CUDA_SOURCES]
    missing = [s for s in sources if not Path(s).exists()]
    if missing:
        raise FileNotFoundError(f"缺少源文件: {missing}")

    build_dir = str(BUILD_ROOT / backend / "build")
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    extra_cflags = ["-O3"]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        # 展开 __shfl_* 的常量参数、放宽 constexpr 使用
        "--expt-relaxed-constexpr",
    ]
    if backend == "cuda":
        # 未设置时按常见架构生成（Ampere/Ada/Hopper）
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")
    else:
        # ROCm：CDNA2/3 常见架构
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a;gfx942")

    logger.info("开始编译 %s 自研算子（%d 个源文件）...", backend, len(sources))
    module = load(
        name="verse_nn_kernels",
        sources=sources,
        extra_include_paths=[str(CSRC)],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=build_dir,
        verbose=verbose,
    )
    # load() 已把扩展 import 进当前进程（TORCH_LIBRARY 随之注册）；
    # 同时把 .so 复制到 _build/<backend>/，供后续进程用 load_library 直接加载。
    produced = Path(module.__file__)
    target = BUILD_ROOT / backend / "verse_nn_kernels.so"
    shutil.copy2(produced, target)
    logger.info("编译完成: %s", target)
    return target


def _build_npu(*, verbose: bool) -> Path:
    """NPU：先跑 CANN 算子工程，再编 torch 适配层。"""
    script = CSRC / "ascend" / "build.sh"
    if not script.exists():
        raise FileNotFoundError(f"缺少 CANN 构建脚本: {script}")
    if not os.environ.get("ASCEND_HOME_PATH") and not Path("/usr/local/Ascend").exists():
        raise RuntimeError(
            "未检测到 CANN 工具链（ASCEND_HOME_PATH / /usr/local/Ascend）。\n"
            "请在已安装 CANN 的昇腾环境执行，并 source set_env.sh。"
        )
    logger.info("调用 CANN 算子工程构建: %s", script)
    env = dict(os.environ, VERSE_OUT_DIR=str(BUILD_ROOT / "npu"))
    subprocess.run(["bash", str(script)], check=True, env=env,
                   stdout=None if verbose else subprocess.DEVNULL)
    target = BUILD_ROOT / "npu" / "verse_nn_kernels.so"
    if not target.exists():
        raise RuntimeError(
            "CANN 构建完成但未找到 verse_nn_kernels.so；请检查 csrc/ascend/build.sh 的输出路径"
        )
    logger.info("NPU 自研算子编译完成: %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="编译 VerseNext 自研算子")
    parser.add_argument("--backend", choices=["cuda", "rocm", "npu", "auto"], default="auto")
    parser.add_argument("--info", action="store_true", help="打印各后端可用性与编译产物状态")
    parser.add_argument("--clean", action="store_true", help="编译前清理缓存")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.info:
        from verse_nn.kernels._ext import extension_info

        info = extension_info()
        print(f"build_dir: {info['build_dir']}")
        for name, st in info["backends"].items():
            mark = "OK " if st["available"] else "-- "
            print(f"  [{mark}] {name:5s} candidates={st['candidates']} error={st['error']}")
        print(f"detected build backend: {detect_build_backend()}")
        return 0

    backend = None if args.backend == "auto" else args.backend
    path = build(backend, verbose=args.verbose, clean=args.clean)
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
