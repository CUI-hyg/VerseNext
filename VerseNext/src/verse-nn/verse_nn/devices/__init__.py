"""设备抽象层：CPU / CUDA / ROCm / CANN NPU（含 XPU / MPS）。

对外主入口：

.. code-block:: python

    from verse_nn.devices import detect_backend, apply_device_limits

    backend = detect_backend("auto")          # 按优先级选最优可用后端
    info = apply_device_limits(backend, mem_fraction=0.5, cpu_fraction=0.5)
    print(backend.caps().as_dict())           # 能力上报
    with backend.autocast():                  # 设备侧混合精度
        ...

本层不依赖 verse-trainer；trainer/resources 反向调用它，避免循环依赖。
"""

from verse_nn.devices.backend import (
    Backend,
    CPUBackend,
    CUDABackend,
    NPUBackend,
    ROCmBackend,
    XPUBackend,
    MPSBackend,
    available_backends,
    detect_backend,
    get_backend,
    register_backend,
    set_custom_kernels_available,
)
from verse_nn.devices.caps import DeviceCaps, cpu_arch_name, host_memory_bytes

__all__ = [
    "Backend",
    "CPUBackend",
    "CUDABackend",
    "ROCmBackend",
    "NPUBackend",
    "XPUBackend",
    "MPSBackend",
    "DeviceCaps",
    "available_backends",
    "detect_backend",
    "get_backend",
    "register_backend",
    "set_custom_kernels_available",
    "apply_device_limits",
    "device_summary",
    "cpu_arch_name",
    "host_memory_bytes",
]


def apply_device_limits(
    backend: Backend | None = None,
    *,
    mem_fraction: float = 0.5,
    cpu_fraction: float = 0.5,
    num_threads: int = 0,
    core_fraction: float | None = None,
) -> dict:
    """把设备占用锁在给定比例内（默认 50%），返回实际生效配置。

    这是 CPU/GPU/NPU 统一的资源锁入口——trainer 与 CometSpark 都调它，
    避免各平台分支散落。

    Args:
        backend: 目标后端；None 时自动探测。
        mem_fraction: 显存/内存占比上限。
        cpu_fraction: 主机线程占比上限。
        num_threads: 显式线程数（>0 时优先）。
        core_fraction: 加速器计算核占比（NPU AI Core）；None 沿用 ``cpu_fraction``。
    """
    backend = backend or detect_backend()
    if not 0.0 < mem_fraction <= 1.0:
        raise ValueError(f"mem_fraction 必须在 (0, 1]，当前 {mem_fraction}")
    if not 0.0 < cpu_fraction <= 1.0:
        raise ValueError(f"cpu_fraction 必须在 (0, 1]，当前 {cpu_fraction}")
    if core_fraction is not None and not 0.0 < core_fraction <= 1.0:
        raise ValueError(f"core_fraction 必须在 (0, 1]，当前 {core_fraction}")
    return backend.lock_resources(
        mem_fraction=mem_fraction,
        cpu_fraction=cpu_fraction,
        num_threads=num_threads,
        core_fraction=core_fraction,
    )


def device_summary() -> list[dict]:
    """列出所有可用后端及其能力（CLI/日志用）。"""
    return [b.caps().as_dict() for b in available_backends()]
