"""资源智能分配与占用上限。

目标：把 CPU / GPU / NPU / 内存占用统一控制在可配置比例以内（CPU/CUDA 默认
50%，NPU 默认 70%），同时在上限内充分利用资源（不空转）。

设备相关的实际加锁动作已下沉到 :mod:`verse_nn.devices`（每个后端一套实现），
本模块只负责「读配置 → 调后端 → 记录/告警」，避免平台分支散落各处：

- **CPU**：线程数按 ``sched_getaffinity`` 感知的可用核数 × ``cpu_fraction``
  计算。容器里 ``os.cpu_count()`` 常与 affinity 不一致（本机实测 8 vs 64），
  直接用它会严重欠用算力，故优先 affinity。
- **CUDA / ROCm**：``torch.cuda.set_per_process_memory_fraction`` 限制显存占比，
  并在支持的卡上打开 TF32 与 cudnn.benchmark。
- **CANN NPU**：``torch.npu.set_per_process_memory_fraction`` 限显存；
  AI 核占比记录为 ``ai_cores_target``（单卡内按核切分需平台配额）。
- **内存**：``MemoryGuard`` 周期性采样 RSS，超过 ``mem_fraction × 上限``
  时告警（CPU 侧无法硬性限制，只能软性监控）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from verse_core import get_logger
from verse_core.config import BaseConfig

logger = get_logger("resources")


def available_cpu_count() -> int:
    """可用 CPU 核数（优先 cgroup/affinity 感知）。"""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def total_memory_bytes() -> int:
    """内存上限：优先 cgroup v2/v1 限制，回退 psutil/物理内存；未知返回 0。"""
    from verse_nn.devices.caps import host_memory_bytes

    return host_memory_bytes()


def current_rss_bytes() -> int:
    """当前进程 RSS（psutil 可选，回退 /proc/self/statm）。"""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


@dataclass
class ResourceConfig(BaseConfig):
    """资源占用比例配置（各值均为「占总量的比例」，上限 1.0）。"""

    enabled: bool = True
    cpu_fraction: float = 0.5
    """CPU 线程占可用核数比例；0.5 表示最多用一半核。"""
    mem_fraction: float = 0.5
    """内存软上限比例（RSS 超过即告警）。"""
    gpu_mem_fraction: float = 0.5
    """CUDA/ROCm 进程显存占比上限（仅 CUDA 系生效）。"""
    npu_mem_fraction: float = 0.7
    """CANN NPU 进程显存占比上限（torch.npu.set_per_process_memory_fraction）。

    昇腾单卡独占场景下显存充足（本机实测 ≈66GB），默认给到 70% 以充分利用
    Cube/Vector 流水所需的 batch/seq_len，比 CUDA 侧的 50% 更激进。
    """
    npu_core_fraction: float = 0.7
    """NPU AI Core 占比目标（单卡内按核切分需平台配额，本层记录并告警）。"""

    def validate(self) -> None:
        for name in (
            "cpu_fraction",
            "mem_fraction",
            "gpu_mem_fraction",
            "npu_mem_fraction",
            "npu_core_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 必须在 (0, 1] 区间内，当前 {value}")


def resolve_threads(num_threads: int = 0, config: ResourceConfig | None = None) -> int:
    """解析 CPU 线程数：显式值优先，否则按 ``cpu_fraction`` 计算。"""
    if num_threads > 0:
        return num_threads
    cfg = config or ResourceConfig()
    cores = available_cpu_count()
    return max(1, int(round(cores * cfg.cpu_fraction)))


def apply_resource_limits(
    num_threads: int = 0,
    config: ResourceConfig | None = None,
    device: torch.device | str | None = None,
) -> int:
    """按配置对目标设备加资源锁，返回最终 CPU 线程数。

    所有平台的加锁细节由 :class:`verse_nn.devices.Backend` 实现；本函数只做
    「选后端 + 传比例 + 回读生效值」。``config.enabled=False`` 时用满可用核。
    """
    from verse_nn.devices import apply_device_limits, detect_backend

    cfg = config or ResourceConfig()
    backend = detect_backend(device)

    if not cfg.enabled:
        threads = num_threads if num_threads > 0 else available_cpu_count()
        torch.set_num_threads(threads)
        torch.set_flush_denormal(True)
        logger.info("资源限制已关闭：使用全部 %d 线程，device=%s", threads, backend.name)
        return threads

    # NPU 的显存/核比例单独配置；其余后端用 gpu_mem_fraction
    mem_fraction = cfg.npu_mem_fraction if backend.name == "npu" else cfg.gpu_mem_fraction
    info = apply_device_limits(
        backend,
        mem_fraction=mem_fraction,
        cpu_fraction=cfg.cpu_fraction,
        num_threads=num_threads,
        core_fraction=cfg.npu_core_fraction if backend.name == "npu" else None,
    )
    logger.info(
        "资源锁已应用(%s)：线程 %s/%s，mem %.0f%%",
        backend.name,
        info.get("threads"),
        info.get("cores"),
        mem_fraction * 100,
    )
    return int(info.get("threads", resolve_threads(num_threads, cfg)))


def describe_devices() -> list[dict]:
    """列出可用设备及其能力（CLI/日志用）。"""
    from verse_nn.devices import device_summary

    return device_summary()


class MemoryGuard:
    """周期性 RSS 采样，超过 ``mem_fraction × 上限`` 时告警（不阻断训练）。"""

    def __init__(self, config: ResourceConfig | None = None) -> None:
        self.config = config or ResourceConfig()
        self.limit = total_memory_bytes()
        self.soft_limit = int(self.limit * self.config.mem_fraction) if self.limit else 0
        self.peak = 0
        self._warned = False

    def check(self, step: int | None = None) -> int:
        """采样一次 RSS，返回当前 RSS；超限时告警（每个实例只告警一次）。"""
        rss = current_rss_bytes()
        self.peak = max(self.peak, rss)
        if (
            self.soft_limit
            and rss > self.soft_limit
            and not self._warned
        ):
            self._warned = True
            where = f"step {step}" if step is not None else "startup"
            logger.warning(
                "内存占用超软上限(%s): RSS %.2fGB > %.2fGB (%.0f%% of %.2fGB)，"
                "建议减小 batch/seq_len 或开启 gradient_checkpointing",
                where, rss / 1e9, self.soft_limit / 1e9,
                self.config.mem_fraction * 100, self.limit / 1e9,
            )
        return rss

    @property
    def peak_gb(self) -> float:
        return self.peak / 1e9
