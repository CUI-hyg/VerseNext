"""资源智能分配测试（CPU/CUDA 默认 50%，NPU 默认 70% 上限）。"""

import torch

from verse_trainer.resources import (
    MemoryGuard,
    ResourceConfig,
    apply_resource_limits,
    available_cpu_count,
    resolve_threads,
    total_memory_bytes,
)


def test_available_cpu_count_positive():
    assert available_cpu_count() >= 1


def test_resolve_threads_explicit_wins():
    assert resolve_threads(3, ResourceConfig(cpu_fraction=0.5)) == 3


def test_resolve_threads_fraction():
    cores = available_cpu_count()
    cfg = ResourceConfig(cpu_fraction=0.5)
    expected = max(1, int(round(cores * 0.5)))
    assert resolve_threads(0, cfg) == expected
    assert resolve_threads(0, ResourceConfig(cpu_fraction=1.0)) == max(1, cores)


def test_resource_config_validation():
    import pytest

    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            ResourceConfig(cpu_fraction=bad).validate()
    ResourceConfig().validate()


def test_apply_resource_limits_sets_threads():
    threads = apply_resource_limits(2, ResourceConfig(), "cpu")
    assert threads == 2
    assert torch.get_num_threads() == 2
    # 恢复一个安全值，避免影响后续测试
    torch.set_num_threads(max(1, available_cpu_count() // 2))


def test_apply_resource_limits_disabled():
    cfg = ResourceConfig(enabled=False)
    threads = apply_resource_limits(0, cfg, "cpu")
    assert threads == available_cpu_count()


def test_memory_guard_soft_limit():
    guard = MemoryGuard(ResourceConfig(mem_fraction=0.5))
    total = total_memory_bytes()
    if total:
        assert guard.soft_limit == int(total * 0.5)
    rss = guard.check(step=1)
    assert rss >= 0
    assert guard.peak >= rss
