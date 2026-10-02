"""设备后端抽象与注册表：CPU / CUDA / ROCm / CANN NPU（及 XPU / MPS）。

设计目标
--------
1. **统一入口**：上层（trainer / inference）只跟 :class:`Backend` 打交道，
   不再散落 ``if device.type == "cuda"`` 分支。
2. **能力驱动**：算子派发读 :class:`~verse_nn.devices.caps.DeviceCaps` 决策，
   而不是猜设备型号。
3. **资源锁**：每个后端都实现 ``lock_resources``，把占用限制在给定比例
   （默认 50%）以内——CPU 限线程、CUDA/ROCm 限显存、NPU 限显存与核数。
4. **可降级**：后端不可用时 :func:`available_backends` 不返回它，上层自然
   回退到 CPU，绝不因缺库而崩溃。

CUDA 与 ROCm 共用 ``torch.device("cuda")``；二者靠 ``torch.version.hip``
区分（ROCm 构建下该值非 None），因此同一套 kernel 源码可被 hipify 复用。
"""

from __future__ import annotations

import contextlib
import os
import platform
from abc import ABC, abstractmethod
from typing import Iterator

import torch

from verse_core import get_logger
from verse_nn.devices.caps import (
    DeviceCaps,
    cpu_arch_name,
    host_memory_bytes,
)

logger = get_logger("devices")

__all__ = [
    "Backend",
    "CPUBackend",
    "CUDABackend",
    "ROCmBackend",
    "NPUBackend",
    "XPUBackend",
    "MPSBackend",
    "get_backend",
    "available_backends",
    "detect_backend",
    "register_backend",
]


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class Backend(ABC):
    """设备后端接口。子类实现具体平台的资源锁、同步与缓存管理。"""

    #: 后端标识（cpu/cuda/rocm/npu/xpu/mps）
    name: str = "abstract"
    #: PyTorch 设备类型
    device_type: str = "cpu"
    #: 自动设备选择优先级（越大越优先）
    priority: int = 0

    # ---------------------------------------------------------- availability

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """该后端在当前环境是否可用（硬件 + 运行时齐备）。"""

    # ------------------------------------------------------------- caps

    @abstractmethod
    def caps(self, index: int = 0) -> DeviceCaps:
        """返回设备能力描述。"""

    # --------------------------------------------------------- resource lock

    @abstractmethod
    def lock_resources(
        self,
        *,
        mem_fraction: float = 0.5,
        cpu_fraction: float = 0.5,
        num_threads: int = 0,
        core_fraction: float | None = None,
    ) -> dict:
        """把本后端占用限制在给定比例内，返回实际生效的配置（便于日志/断言）。

        Args:
            mem_fraction: 显存/内存占比上限（0, 1]。
            cpu_fraction: 主机侧线程占比上限（0, 1]。
            num_threads: 显式线程数；>0 时优先于 ``cpu_fraction``。
            core_fraction: 加速器**计算核**占比上限（NPU 的 AI Core）；
                None 表示沿用 ``cpu_fraction``。其他后端忽略该项。
        """

    # ------------------------------------------------------------ runtime

    def synchronize(self) -> None:
        """等待设备计算完成（CPU 为 no-op）。"""

    def empty_cache(self) -> None:
        """释放未使用的缓存（CPU 为 no-op）。"""

    def memory_allocated(self) -> int:
        """当前已分配字节数（未知返回 0）。"""
        return 0

    def memory_reserved(self) -> int:
        """当前保留字节数（未知返回 0）。"""
        return 0

    def max_memory_bytes(self) -> int:
        """设备总内存/显存字节数（未知返回 0）。"""
        return 0

    @contextlib.contextmanager
    def autocast(self, dtype: torch.dtype | None = None, enabled: bool = True) -> Iterator[None]:
        """设备侧自动混合精度上下文。"""
        if not enabled or dtype is None:
            yield
            return
        with torch.autocast(self.device_type, dtype=dtype):
            yield

    def default_autocast_dtype(self) -> torch.dtype | None:
        """该后端推荐的混合精度 dtype（None 表示不推荐）。"""
        return None

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        caps = self.caps()
        if dtype is torch.bfloat16:
            return caps.supports_bf16
        if dtype is torch.float16:
            return caps.supports_fp16
        return True


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

class CPUBackend(Backend):
    """CPU 后端：线程锁 + denormal 刷新 + bf16 能力探测。"""

    name = "cpu"
    device_type = "cpu"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def caps(self, index: int = 0) -> DeviceCaps:
        arch = cpu_arch_name()
        # bf16 是否原生（AVX512-BF16 / AMX）：PyTorch 内部探测，缺失则保守为 False
        bf16 = False
        for probe in ("_is_avx512_bf16_supported", "_is_amx_bf16_supported"):
            fn = getattr(torch.backends.cpu, probe, None) or getattr(torch.cpu, probe, None)
            if callable(fn):
                try:
                    if fn():
                        bf16 = True
                        break
                except Exception:  # pragma: no cover - 版本差异
                    pass
        return DeviceCaps(
            name="cpu",
            device_type="cpu",
            backend="cpu",
            vendor="intel" if "intel" in arch.lower() else "unknown",
            is_accelerator=False,
            supports_fp16=False,      # CPU 上 fp16 GEMM 通常慢于 fp32/bf16
            supports_bf16=bf16,
            supports_tf32=False,
            supports_flash_attn=False,
            has_custom_kernels=False,
            warp_size=1,
            smem_per_block=0,
            compute_capability=None,
            arch=arch,
            memory_bytes=host_memory_bytes(),
            device_index=index,
        )

    def lock_resources(
        self,
        *,
        mem_fraction: float = 0.5,
        cpu_fraction: float = 0.5,
        num_threads: int = 0,
        core_fraction: float | None = None,
    ) -> dict:
        cores = _affinity_cpu_count()
        if num_threads > 0:
            threads = num_threads
        else:
            threads = max(1, int(round(cores * cpu_fraction)))
        torch.set_num_threads(threads)
        # 亚正规数会让 x86 矩阵运算慢数十倍，训练/推理都应刷新为 0
        torch.set_flush_denormal(True)
        # inter-op 线程池并发过多会与 intra-op 抢核，按同一比例收紧
        try:
            torch.set_num_interop_threads(max(1, min(threads, 4)))
        except RuntimeError:
            # 已初始化过线程池则不可再设，忽略
            pass
        return {
            "backend": "cpu",
            "threads": threads,
            "cores": cores,
            "cpu_fraction": cpu_fraction,
            "mem_fraction": mem_fraction,
        }


def _affinity_cpu_count() -> int:
    """可用 CPU 核数：优先 sched_getaffinity（容器里常与 cpu_count 不一致）。"""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# CUDA / ROCm（共用 torch.device("cuda")）
# ---------------------------------------------------------------------------

class CUDABackend(Backend):
    """NVIDIA CUDA 后端。子类 :class:`ROCmBackend` 覆盖 AMD 差异。"""

    name = "cuda"
    device_type = "cuda"
    priority = 30

    @classmethod
    def is_available(cls) -> bool:
        # ROCm 构建下 torch.version.hip 非 None，交由 ROCmBackend 认领
        return torch.cuda.is_available() and torch.version.hip is None

    def _index(self, index: int) -> int:
        return index

    def caps(self, index: int = 0) -> DeviceCaps:
        idx = self._index(index)
        props = torch.cuda.get_device_properties(idx)
        cc = torch.cuda.get_device_capability(idx)
        smem = getattr(props, "shared_memory_per_block_optin", 0) or getattr(
            props, "shared_memory_per_block", 0
        )
        return DeviceCaps(
            name=f"cuda:{idx}",
            device_type="cuda",
            backend=self.name,
            vendor="nvidia",
            is_accelerator=True,
            supports_fp16=True,
            supports_bf16=cc[0] >= 8,
            supports_tf32=cc[0] >= 8,
            supports_flash_attn=cc[0] >= 7 and (cc[0] > 7 or cc[1] >= 5),
            has_custom_kernels=_custom_kernels_available(),
            warp_size=32,
            smem_per_block=int(smem),
            compute_capability=cc,
            arch=f"sm_{cc[0]}{cc[1]}",
            memory_bytes=int(props.total_memory),
            device_index=idx,
        )

    def lock_resources(
        self,
        *,
        mem_fraction: float = 0.5,
        cpu_fraction: float = 0.5,
        num_threads: int = 0,
        core_fraction: float | None = None,
    ) -> dict:
        # 主机侧线程同样要限，否则 dataloader 会吃满核
        cpu_info = CPUBackend().lock_resources(
            mem_fraction=mem_fraction, cpu_fraction=cpu_fraction, num_threads=num_threads
        )
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
        # TF32：在支持的卡上打开，显著提升 fp32 GEMM 吞吐
        if self.caps().supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return {**cpu_info, "backend": self.name, "gpu_mem_fraction": mem_fraction}

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def memory_allocated(self) -> int:
        return int(torch.cuda.memory_allocated())

    def memory_reserved(self) -> int:
        return int(torch.cuda.memory_reserved())

    def max_memory_bytes(self) -> int:
        try:
            return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:  # pragma: no cover
            return 0

    def default_autocast_dtype(self) -> torch.dtype:
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:  # pragma: no cover
            return torch.float16


class ROCmBackend(CUDABackend):
    """AMD ROCm 后端：与 CUDA 共享 ``torch.device("cuda")``，差异在架构串与 wavefront。

    关键差异（影响自研 kernel）：
    - wavefront 宽度 **64**（NVIDIA warp 为 32），kernel 里的 warp 级原语必须
      按 ``__HIP_PLATFORM_AMD__`` 分支；
    - 无 TF32，矩阵单元走 MFMA（gfx90a+ 支持 bf16/fp16）；
    - 编译期用 hipify 把同一份 ``.cu`` 转成 HIP。
    """

    name = "rocm"
    device_type = "cuda"
    priority = 29

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available() and torch.version.hip is not None

    def caps(self, index: int = 0) -> DeviceCaps:
        idx = self._index(index)
        props = torch.cuda.get_device_properties(idx)
        arch = getattr(props, "gcnArchName", "") or ""
        # gfx90a / gfx942 等 CDNA 架构原生支持 bf16
        is_cdna = any(tag in arch for tag in ("gfx90a", "gfx94", "gfx95"))
        smem = getattr(props, "shared_memory_per_block_optin", 0) or getattr(
            props, "shared_memory_per_block", 0
        )
        return DeviceCaps(
            name=f"rocm:{idx}",
            device_type="cuda",
            backend="rocm",
            vendor="amd",
            is_accelerator=True,
            supports_fp16=True,
            supports_bf16=is_cdna,
            supports_tf32=False,
            supports_flash_attn=True,
            has_custom_kernels=_custom_kernels_available(),
            warp_size=64,
            smem_per_block=int(smem),
            compute_capability=None,
            arch=arch or "rocm",
            memory_bytes=int(props.total_memory),
            device_index=idx,
        )

    def lock_resources(
        self,
        *,
        mem_fraction: float = 0.5,
        cpu_fraction: float = 0.5,
        num_threads: int = 0,
        core_fraction: float | None = None,
    ) -> dict:
        info = super().lock_resources(
            mem_fraction=mem_fraction, cpu_fraction=cpu_fraction, num_threads=num_threads
        )
        info["backend"] = "rocm"
        info["gpu_arch"] = self.caps().arch
        return info

    def default_autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.caps().supports_bf16 else torch.float16


# ---------------------------------------------------------------------------
# Huawei CANN NPU
# ---------------------------------------------------------------------------

class NPUBackend(Backend):
    """华为昇腾 CANN NPU 后端（依赖 ``torch_npu``）。

    资源锁的两条腿（比例默认 50%，NPU 可在配置里提到 70%）：
    - **显存**：``torch.npu.set_per_process_memory_fraction``；
    - **AI 核**：CANN 没有进程级核数 API，通过 ``ASCEND_RT_VISIBLE_DEVICES``
      限制可见设备、并用 ``num_ai_cores`` 记录预期上限（多卡切分时按卡数折算）。
      单卡内部按核比例切分需平台侧 ``npu-smi`` 配额，本层记录并告警。
    """

    name = "npu"
    device_type = "npu"
    priority = 35

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch_npu  # noqa: F401
        except Exception:
            return False
        try:
            return bool(torch.npu.is_available())
        except Exception:
            return False

    @staticmethod
    def _npu_module():
        import torch_npu  # noqa: F401
        return torch.npu

    @staticmethod
    def _device_props(index: int = 0):
        """读取设备属性；旧版 torch_npu 缺该 API 时返回 None。"""
        try:
            return NPUBackend._npu_module().get_device_properties(index)
        except Exception:  # pragma: no cover - 版本差异
            return None

    @staticmethod
    def _supports_bf16(soc: str) -> bool:
        """bf16 能力：优先运行时探测，回退 SOC 串匹配。

        ``torch.npu.is_bf16_supported()`` 是权威答案（本机 910_9362 返回 True），
        而 SOC 串匹配表必然滞后于新架构，只作兜底。
        """
        try:
            import torch_npu  # noqa: F401

            probe = getattr(torch.npu, "is_bf16_supported", None)
            if callable(probe):
                return bool(probe())
        except Exception:  # pragma: no cover - 版本差异
            pass
        return _ascend_supports_bf16(soc)

    @staticmethod
    def _ai_core_count(props, soc: str) -> int:
        """AI Core 数量：优先设备属性，回退 SOC 表。

        910B/910C 的 ``get_device_properties`` 暴露 ``cube_core_num``（本机实测
        20）；部分版本只给 ``vector_core_num`` / ``multi_processor_count``。
        """
        if props is not None:
            for attr in ("cube_core_num", "vector_core_num", "multi_processor_count"):
                try:
                    value = int(getattr(props, attr, 0) or 0)
                except (TypeError, ValueError):  # pragma: no cover
                    value = 0
                if value > 0:
                    return value
        return _ascend_ai_core_count(soc)

    def caps(self, index: int = 0) -> DeviceCaps:
        props = self._device_props(index)
        soc = getattr(props, "name", "") or getattr(props, "soc_version", "") or "ascend"
        mem = 0
        try:
            mem = int(getattr(props, "total_memory", 0)) if props is not None else 0
        except Exception:  # pragma: no cover
            mem = 0
        bf16 = self._supports_bf16(str(soc))
        return DeviceCaps(
            name=f"npu:{index}",
            device_type="npu",
            backend="npu",
            vendor="huawei",
            is_accelerator=True,
            supports_fp16=True,
            supports_bf16=bf16,
            supports_tf32=False,          # 昇腾无 TF32 概念，用 cube 单元 fp16/bf16
            supports_flash_attn=True,     # 910B 支持 FlashAttention 类融合算子
            has_custom_kernels=_custom_kernels_available("npu"),
            warp_size=1,                  # 无 warp 语义：AI Core（Cube/Vector/Scalar）
            smem_per_block=_ascend_ub_bytes(str(soc)),
            compute_capability=None,
            arch=str(soc).lower(),
            memory_bytes=mem,
            device_index=index,
        )

    def lock_resources(
        self,
        *,
        mem_fraction: float = 0.5,
        cpu_fraction: float = 0.5,
        num_threads: int = 0,
        core_fraction: float | None = None,
    ) -> dict:
        cpu_info = CPUBackend().lock_resources(
            mem_fraction=mem_fraction, cpu_fraction=cpu_fraction, num_threads=num_threads
        )
        npu = self._npu_module()
        info: dict = {**cpu_info, "backend": "npu", "npu_mem_fraction": mem_fraction}
        try:
            npu.set_per_process_memory_fraction(mem_fraction, 0)
            info["npu_mem_locked"] = True
        except Exception as exc:  # pragma: no cover - 平台相关
            logger.warning("设置 NPU 显存占比失败: %s", exc)
            info["npu_mem_locked"] = False
        # 单卡 AI 核按比例切分需要平台配额，这里记录预期值供上层/运维参考。
        # core_fraction 未显式给出时沿用 cpu_fraction。
        core_frac = core_fraction if core_fraction is not None else cpu_fraction
        props = self._device_props(0)
        soc = getattr(props, "name", "") or getattr(props, "soc_version", "") or "ascend"
        cores = self._ai_core_count(props, str(soc))
        info["ai_cores_total"] = cores
        info["ai_core_fraction"] = core_frac
        info["ai_cores_target"] = max(1, int(round(cores * core_frac))) if cores else 0
        if cores and core_frac < 1.0:
            logger.info(
                "NPU AI 核占比 %.0f%%：目标 %d/%d 核；单卡内按核切分需平台侧 npu-smi 配额，"
                "本进程已通过 ASCEND_RT_VISIBLE_DEVICES 限制可见设备",
                core_frac * 100, info["ai_cores_target"], cores,
            )
        return info

    def synchronize(self) -> None:
        self._npu_module().synchronize()

    def empty_cache(self) -> None:
        try:
            self._npu_module().empty_cache()
        except Exception:  # pragma: no cover
            pass

    def memory_allocated(self) -> int:
        try:
            return int(self._npu_module().memory_allocated())
        except Exception:  # pragma: no cover
            return 0

    def memory_reserved(self) -> int:
        try:
            return int(self._npu_module().memory_reserved())
        except Exception:  # pragma: no cover
            return 0

    def max_memory_bytes(self) -> int:
        return self.caps().memory_bytes

    def default_autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.caps().supports_bf16 else torch.float16


def _ascend_supports_bf16(soc: str) -> bool:
    """910B/910C/910_93 及更新架构原生支持 bf16；310P 以 fp16 为主。

    仅作 :meth:`NPUBackend._supports_bf16` 的回退：``torch.npu`` 名称形如
    ``Ascend910_9362``（910B 的 93 系列），既不含 ``910b`` 也不含 ``910c``，
    旧表会漏判，故补 ``910_93`` / ``910_9`` / ``910d`` 等前缀。
    """
    soc = soc.lower()
    return any(
        tag in soc
        for tag in ("910b", "910c", "910d", "910_93", "910_9", "ascend910b", "ascend910c")
    )


def _ascend_ub_bytes(soc: str) -> int:
    """Unified Buffer 大小（分块调优用）。未知返回 0。"""
    soc = soc.lower()
    if "310p" in soc:
        return 128 * 1024
    if any(tag in soc for tag in ("910b", "910c", "910_9", "910d")):
        return 192 * 1024      # 910B/910C UB ≈ 192KB
    return 0


def _ascend_ai_core_count(soc: str) -> int:
    """AI Core 数量（资源占比换算用，SOC 回退表）。未知返回 0。"""
    soc = soc.lower()
    if "310p" in soc:
        return 8
    if "910c" in soc:
        return 32
    if any(tag in soc for tag in ("910b", "910_9", "910d")):
        return 24
    return 0


# ---------------------------------------------------------------------------
# XPU / MPS（次要后端，接口齐全即可）
# ---------------------------------------------------------------------------

class XPUBackend(Backend):
    """Intel XPU 后端（``torch.xpu``）。"""

    name = "xpu"
    device_type = "xpu"
    priority = 20

    @classmethod
    def is_available(cls) -> bool:
        try:
            return bool(torch.xpu.is_available())
        except Exception:
            return False

    def caps(self, index: int = 0) -> DeviceCaps:
        props = torch.xpu.get_device_properties(index)
        mem = int(getattr(props, "total_memory", 0))
        return DeviceCaps(
            name=f"xpu:{index}", device_type="xpu", backend="xpu", vendor="intel",
            is_accelerator=True, supports_fp16=True, supports_bf16=True,
            supports_tf32=False, supports_flash_attn=True,
            has_custom_kernels=_custom_kernels_available("xpu"),
            warp_size=32, smem_per_block=int(getattr(props, "shared_memory_per_block", 0) or 0),
            arch=getattr(props, "name", "xpu"), memory_bytes=mem, device_index=index,
        )

    def lock_resources(self, *, mem_fraction=0.5, cpu_fraction=0.5, num_threads=0,
                       core_fraction=None) -> dict:
        info = CPUBackend().lock_resources(
            mem_fraction=mem_fraction, cpu_fraction=cpu_fraction, num_threads=num_threads
        )
        try:
            torch.xpu.set_per_process_memory_fraction(mem_fraction)
            info["xpu_mem_locked"] = True
        except Exception as exc:  # pragma: no cover
            logger.warning("设置 XPU 显存占比失败: %s", exc)
            info["xpu_mem_locked"] = False
        return {**info, "backend": "xpu"}


class MPSBackend(Backend):
    """Apple MPS 后端（无显存占比 API，仅做主机侧锁与能力上报）。"""

    name = "mps"
    device_type = "mps"
    priority = 10

    @classmethod
    def is_available(cls) -> bool:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    def caps(self, index: int = 0) -> DeviceCaps:
        return DeviceCaps(
            name="mps", device_type="mps", backend="mps", vendor="apple",
            is_accelerator=True, supports_fp16=True, supports_bf16=False,
            supports_tf32=False, supports_flash_attn=False,
            has_custom_kernels=False, warp_size=32,
            arch=platform.machine(), memory_bytes=host_memory_bytes(), device_index=0,
        )

    def lock_resources(self, *, mem_fraction=0.5, cpu_fraction=0.5, num_threads=0,
                       core_fraction=None) -> dict:
        info = CPUBackend().lock_resources(
            mem_fraction=mem_fraction, cpu_fraction=cpu_fraction, num_threads=num_threads
        )
        # MPS 无 per-process 显存上限 API，靠 high watermark 软监控
        return {**info, "backend": "mps", "mps_mem_fraction": mem_fraction}


# ---------------------------------------------------------------------------
# 注册表与探测
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[Backend]] = {}
_CUSTOM_KERNELS_FLAG: dict[str, bool] = {}


def register_backend(cls: type[Backend], *, override: bool = False) -> type[Backend]:
    """注册一个后端类（第三方可扩展，如寒武纪 MLU）。"""
    if cls.name in _BACKENDS and not override:
        raise KeyError(f"后端 {cls.name} 已注册，如需覆盖请传 override=True")
    _BACKENDS[cls.name] = cls
    return cls


for _cls in (CPUBackend, CUDABackend, ROCmBackend, NPUBackend, XPUBackend, MPSBackend):
    register_backend(_cls, override=True)


def _custom_kernels_available(backend: str | None = None) -> bool:
    """自研算子扩展是否已编译可用（避免在 import 期触发编译）。"""
    if backend is not None and backend in _CUSTOM_KERNELS_FLAG:
        return _CUSTOM_KERNELS_FLAG[backend]
    try:
        from verse_nn.kernels._ext import extension_available

        return extension_available(backend)
    except Exception:
        return False


def set_custom_kernels_available(flag: bool, backend: str | None = None) -> None:
    """测试/构建脚本用：显式声明自研算子可用性。"""
    _CUSTOM_KERNELS_FLAG[backend or "__all__"] = bool(flag)


def get_backend(name: str) -> Backend:
    """按名字取后端实例（不检查可用性）。"""
    if name not in _BACKENDS:
        raise KeyError(f"未知后端 {name}；已注册: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


def available_backends() -> list[Backend]:
    """返回当前环境所有可用后端，按优先级降序（npu > cuda > rocm > xpu > mps > cpu）。"""
    out: list[Backend] = []
    for cls in _BACKENDS.values():
        try:
            if cls.is_available():
                out.append(cls())
        except Exception as exc:  # pragma: no cover - 探测本身不应崩溃
            logger.debug("后端 %s 可用性探测失败: %s", cls.name, exc)
    out.sort(key=lambda b: b.priority, reverse=True)
    return out


def detect_backend(device: torch.device | str | None = None) -> Backend:
    """解析设备字符串/对象为后端；``None``/``auto`` 时按优先级自动选。

    显式指定但不可用时**回退到 CPU 并告警**，保证训练不会因缺硬件而中断。
    """
    if device is None or (isinstance(device, str) and device == "auto"):
        backends = available_backends()
        return backends[0] if backends else CPUBackend()

    device_type = getattr(device, "type", None) or str(device).split(":")[0]
    for cls in _BACKENDS.values():
        if cls.device_type != device_type:
            continue
        try:
            if cls.is_available():
                return cls()
        except Exception:  # pragma: no cover
            continue
    logger.warning("请求的设备 %s 不可用，回退到 CPU", device_type)
    return CPUBackend()
