"""设备抽象层测试：能力探测、后端注册、50% 资源锁。

这些测试必须**在任意设备上都能过**（CPU-only / CUDA / NPU 机器），
因此对「有哪些后端可用」只做存在性断言，不做具体值断言。
"""

from __future__ import annotations

import os

import pytest
import torch

from verse_nn.devices import (
    CPUBackend,
    DeviceCaps,
    available_backends,
    detect_backend,
    device_summary,
    get_backend,
    register_backend,
)
from verse_nn.devices.backend import Backend, _affinity_cpu_count


# ---------------------------------------------------------------------------
# 探测与注册
# ---------------------------------------------------------------------------

def test_cpu_backend_always_available():
    assert CPUBackend.is_available() is True


def test_available_backends_non_empty_and_sorted():
    backends = available_backends()
    assert backends, "至少应有 CPU 后端"
    names = [b.name for b in backends]
    assert "cpu" in names
    # 按优先级降序
    prios = [b.priority for b in backends]
    assert prios == sorted(prios, reverse=True)


def test_detect_backend_auto_returns_available_backend():
    backend = detect_backend("auto")
    assert backend.name in {b.name for b in available_backends()}


def test_detect_backend_none_equals_auto():
    assert detect_backend(None).name == detect_backend("auto").name


def test_detect_backend_unavailable_falls_back_to_cpu():
    # "meta" 不是已注册后端 -> 回退 CPU（确定性，与硬件无关）
    backend = detect_backend("meta")
    assert backend.name == "cpu"


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_backend("no_such_backend")


def test_register_backend_duplicate_raises():
    class FakeBackend(Backend):
        name = "cpu"          # 与已注册的 CPU 冲突
        device_type = "cpu"

        @classmethod
        def is_available(cls):
            return False

        def caps(self, index=0):
            raise NotImplementedError

        def lock_resources(self, **kwargs):
            return {}

    with pytest.raises(KeyError):
        register_backend(FakeBackend)
    # override=True 允许替换（测试后需还原，这里用独立名字避免污染）
    class AnotherBackend(FakeBackend):
        name = "test_backend_xyz"

    register_backend(AnotherBackend, override=True)
    assert get_backend("test_backend_xyz").name == "test_backend_xyz"


# ---------------------------------------------------------------------------
# 能力描述
# ---------------------------------------------------------------------------

def test_cpu_caps_fields_are_sane():
    caps = CPUBackend().caps()
    assert isinstance(caps, DeviceCaps)
    assert caps.device_type == "cpu"
    assert caps.backend == "cpu"
    assert caps.is_accelerator is False
    assert caps.warp_size == 1
    assert caps.supports_tf32 is False
    assert caps.arch                      # CPU 型号非空
    assert caps.memory_bytes >= 0
    assert caps.is_cuda_like is False
    assert caps.is_npu is False


def test_caps_as_dict_serializable():
    d = CPUBackend().caps().as_dict()
    assert d["backend"] == "cpu"
    assert "memory_gb" in d and "bf16" in d


def test_device_summary_returns_list_of_dicts():
    summary = device_summary()
    assert isinstance(summary, list) and summary
    assert all(isinstance(item, dict) for item in summary)
    assert any(item["backend"] == "cpu" for item in summary)


# ---------------------------------------------------------------------------
# 资源锁（50% 口径）
# ---------------------------------------------------------------------------

def test_cpu_lock_threads_is_half_of_affinity_cores():
    backend = CPUBackend()
    info = backend.lock_resources(cpu_fraction=0.5, mem_fraction=0.5)
    expected = max(1, int(round(_affinity_cpu_count() * 0.5)))
    assert info["threads"] == expected
    assert torch.get_num_threads() == expected
    assert info["cores"] == _affinity_cpu_count()


def test_cpu_lock_explicit_num_threads_wins():
    info = CPUBackend().lock_resources(num_threads=3, cpu_fraction=0.5)
    assert info["threads"] == 3
    assert torch.get_num_threads() == 3


def test_apply_device_limits_rejects_bad_fractions():
    from verse_nn.devices import apply_device_limits

    with pytest.raises(ValueError):
        apply_device_limits(CPUBackend(), mem_fraction=0.0)
    with pytest.raises(ValueError):
        apply_device_limits(CPUBackend(), mem_fraction=1.5)
    with pytest.raises(ValueError):
        apply_device_limits(CPUBackend(), cpu_fraction=0.0)
    with pytest.raises(ValueError):
        apply_device_limits(CPUBackend(), core_fraction=2.0)


def test_npu_core_fraction_recorded_when_npu_absent():
    """无 NPU 时不应崩溃；有 NPU 时 ai_core_fraction 应被记录。"""
    from verse_nn.devices import NPUBackend

    if not NPUBackend.is_available():
        pytest.skip("无 NPU")
    info = NPUBackend().lock_resources(mem_fraction=0.5, cpu_fraction=0.5,
                                       core_fraction=0.25)
    assert info["ai_core_fraction"] == 0.25
    assert info["ai_cores_target"] <= info["ai_cores_total"]


def test_ascend_soc_helpers_recognize_910_93():
    """``Ascend910_9362``（910B 的 93 系列）必须被认出来。

    旧匹配表只有 ``910b``/``910c`` 子串，而 ``torch.npu`` 报告的名称里既没有
    ``910b`` 也没有 ``910c``，导致 bf16/AI 核/UB 三项全部探测失败——历史上
    直接造成 NPU 上 fp16 训练不带 loss scaling。
    """
    from verse_nn.devices.backend import (
        _ascend_ai_core_count,
        _ascend_supports_bf16,
        _ascend_ub_bytes,
    )

    soc = "Ascend910_9362"
    assert _ascend_supports_bf16(soc) is True
    assert _ascend_ub_bytes(soc) == 192 * 1024
    assert _ascend_ai_core_count(soc) > 0
    # 310P 以 fp16 为主，不应被误判为支持 bf16
    assert _ascend_supports_bf16("Ascend310P") is False


def test_npu_ai_core_count_prefers_device_properties():
    """AI Core 数优先读 ``cube_core_num``，缺失才回退 SOC 表。"""
    from verse_nn.devices import NPUBackend

    class _Props:
        cube_core_num = 20
        vector_core_num = 40
        multi_processor_count = 40

    assert NPUBackend._ai_core_count(_Props(), "ascend910_9362") == 20

    class _PropsNoCube:
        vector_core_num = 40

    assert NPUBackend._ai_core_count(_PropsNoCube(), "ascend910_9362") == 40
    # 属性全无时回退 SOC 表
    assert NPUBackend._ai_core_count(None, "ascend910_9362") > 0


def test_npu_bf16_probe_returns_true_for_910_93():
    """两条路径都要给出 bf16=True。

    有 NPU 时走 ``torch.npu.is_bf16_supported()``；无 NPU 时 import 失败回退
    SOC 串匹配。因此该断言在 CPU 机与 NPU 机上同样成立。
    """
    from verse_nn.devices import NPUBackend

    assert NPUBackend._supports_bf16("ascend910_9362") is True


def test_autocast_context_is_noop_when_disabled():
    backend = CPUBackend()
    with backend.autocast(torch.bfloat16, enabled=False):
        x = torch.ones(2, 2) + 1
    assert float(x.sum()) == 8.0          # 2x2 全为 2


def test_supports_dtype_reflects_caps():
    backend = CPUBackend()
    caps = backend.caps()
    assert backend.supports_dtype(torch.bfloat16) == caps.supports_bf16
    assert backend.supports_dtype(torch.float32) is True


def test_custom_kernels_flag_override():
    from verse_nn.devices import backend as backend_mod
    from verse_nn.kernels._ext import extension_available

    backend_mod.set_custom_kernels_available(True, "cuda")
    try:
        assert extension_available("cuda") is True
    finally:
        # 还原：清掉缓存，避免污染其他测试
        from verse_nn.kernels import _ext

        _ext._LOADED.pop("cuda", None)
        backend_mod._CUSTOM_KERNELS_FLAG.pop("cuda", None)


def test_resource_config_npu_fields_validate():
    from verse_trainer.resources import ResourceConfig

    cfg = ResourceConfig()
    cfg.validate()
    # NPU 默认 70%：单卡独占时显存充足，比 CPU/CUDA 的 50% 更激进
    assert cfg.npu_mem_fraction == 0.7
    assert cfg.npu_core_fraction == 0.7
    assert cfg.cpu_fraction == 0.5 and cfg.gpu_mem_fraction == 0.5
    # BaseConfig.__post_init__ 会立即 validate，故非法值在构造时即抛错
    with pytest.raises(ValueError):
        ResourceConfig(npu_core_fraction=0.0)


def test_apply_resource_limits_returns_threads_and_locks():
    from verse_trainer.resources import ResourceConfig, apply_resource_limits

    threads = apply_resource_limits(0, ResourceConfig(cpu_fraction=0.5), "cpu")
    assert threads == max(1, int(round(_affinity_cpu_count() * 0.5)))
    assert torch.get_num_threads() == threads


def test_apply_resource_limits_disabled_uses_all_cores():
    from verse_trainer.resources import ResourceConfig, apply_resource_limits

    threads = apply_resource_limits(0, ResourceConfig(enabled=False), "cpu")
    assert threads == _affinity_cpu_count()
