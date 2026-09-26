"""设备能力描述（DeviceCaps）。

把「这台机器/这块卡能干什么」抽象成不可变数据，供算子派发与调优决策使用：
调度器据此选择融合算子实现、决定是否启用 bf16/tf32、以及分块大小。

注意：本模块**不依赖 verse-trainer**（依赖方向是 trainer -> nn），只依赖
``torch`` 与标准库，保证 verse-nn 可独立使用。
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

__all__ = ["DeviceCaps", "read_sysfs_int", "cpu_arch_name"]


def read_sysfs_int(path: str) -> int:
    """读取 sysfs 中的整数（失败返回 0），用于 cgroup 内存上限探测。"""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read().strip()
        if not raw or raw == "max":
            return 0
        return int(raw)
    except (OSError, ValueError):
        return 0


def cpu_arch_name() -> str:
    """返回 CPU 微架构名（尽量精确，失败回退 machine）。"""
    machine = platform.machine()
    if machine not in ("x86_64", "AMD64"):
        return machine
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return machine


@dataclass(frozen=True)
class DeviceCaps:
    """一块设备的静态能力描述（构造后不再变化）。

    Args:
        name: 人类可读名称，如 ``"cpu"`` / ``"cuda:0"`` / ``"rocm:0"`` / ``"npu:0"``。
        device_type: PyTorch 设备类型（``cpu`` / ``cuda`` / ``npu`` / ``xpu`` / ``mps``）。
        backend: 后端标识（``cpu`` / ``cuda`` / ``rocm`` / ``npu`` / ``xpu`` / ``mps``）。
        vendor: 厂商（``intel`` / ``amd`` / ``nvidia`` / ``huawei`` / ``apple`` / ``unknown``）。
        is_accelerator: 是否为加速器（决定是否需要显式同步/清缓存）。
        supports_fp16 / supports_bf16: 是否原生支持对应低精度。
        supports_tf32: 是否支持 TF32（CUDA cc>=8.0）。
        supports_flash_attn: 是否具备跑融合注意力的硬件条件。
        has_custom_kernels: 本仓库自研算子是否已编译并可用。
        warp_size: 硬件执行单元宽度（CUDA 32 / ROCm wavefront 64 / 标量 1）。
        smem_per_block: 片上共享内存/统一缓冲字节数（用于分块调优），未知为 0。
        compute_capability: CUDA 计算能力 (major, minor)，非 CUDA 为 None。
        arch: 架构串（``sm_80`` / ``gfx90a`` / ``ascend910b`` / CPU 型号）。
        memory_bytes: 该设备可用内存/显存总量（0 表示未知）。
        device_index: 设备序号。
    """

    name: str
    device_type: str
    backend: str
    vendor: str = "unknown"
    is_accelerator: bool = False
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False
    supports_flash_attn: bool = False
    has_custom_kernels: bool = False
    warp_size: int = 1
    smem_per_block: int = 0
    compute_capability: tuple[int, int] | None = None
    arch: str = ""
    memory_bytes: int = 0
    device_index: int = 0

    @property
    def is_cuda_like(self) -> bool:
        """CUDA 与 ROCm 共享 ``torch.device("cuda")`` 语义与同一套 kernel 源码。"""
        return self.backend in ("cuda", "rocm")

    @property
    def is_npu(self) -> bool:
        return self.backend == "npu"

    def as_dict(self) -> dict:
        """便于日志/事件总线序列化。"""
        return {
            "name": self.name,
            "device_type": self.device_type,
            "backend": self.backend,
            "vendor": self.vendor,
            "is_accelerator": self.is_accelerator,
            "fp16": self.supports_fp16,
            "bf16": self.supports_bf16,
            "tf32": self.supports_tf32,
            "flash_attn": self.supports_flash_attn,
            "custom_kernels": self.has_custom_kernels,
            "warp_size": self.warp_size,
            "smem": self.smem_per_block,
            "arch": self.arch,
            "memory_gb": round(self.memory_bytes / 1e9, 2) if self.memory_bytes else 0.0,
        }


def host_memory_bytes() -> int:
    """宿主机/容器可用内存上限（优先 cgroup，回退物理内存）。"""
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        value = read_sysfs_int(path)
        if value > 0:
            return value
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        return 0
