"""ROCm / NPU CANN 兼容性测试（Part5K1.3 Task 11）。

覆盖内容（对应 SubTask 11.3 测试要点）：
1. device 字符串识别（_parse_device 支持 rocm / rocm:0 / ROCM:2）
2. ROCm/CANN 探测 API（has_rocm / has_cann / get_rocm_version / get_cann_version）
3. autocast 等价性（autocast(device="rocm") 与 autocast(device="cuda") 等价，
   无 torch 时 skip；非 ROCm + 无 CUDA 时验证 rocm→cuda 映射路径）
4. spark/run.py CLI --device 参数支持 rocm（argparse 解析 + dry-run smoke）
5. CANN 兜底（get_memory_info("npu") 在无 torch_npu 时不抛异常）
6. _print_device_info 在各种 device 下不抛异常（Part5K1.3 Task 11.2）
7. get_backend("rocm") 工厂行为（无 torch 抛 RuntimeError，有 torch 返回 TorchBackend）

运行方式：
    python3 -m pytest tests/test_rocm_npu_compat.py -v
    python3 tests/test_rocm_npu_compat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path 注入：让 tests/ 目录能 import verse_torch / spark
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_torch", "verse_nex", "verse_infra"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verse_torch.device import (
    _parse_device,
    has_rocm,
    has_cann,
    get_rocm_version,
    get_cann_version,
    has_torch,
    has_torch_npu,
    get_memory_info,
    get_backend,
    clear_backend_cache,
)


# ===========================================================================
# 1. device 字符串识别（SubTask 11.3 测试要点 1）
# ===========================================================================


class TestDeviceParsing:
    """_parse_device 支持 rocm 系列（Part5K1.3 Task 9.1）。"""

    def test_parse_rocm(self):
        """'rocm' 解析为 'rocm'。"""
        assert _parse_device("rocm") == "rocm"

    def test_parse_rocm_with_index(self):
        """'rocm:0' 解析为 'rocm'（剥掉索引）。"""
        assert _parse_device("rocm:0") == "rocm"

    def test_parse_rocm_uppercase(self):
        """'ROCM:2' 解析为 'rocm'（大小写无关）。"""
        assert _parse_device("ROCM:2") == "rocm"

    def test_parse_existing_devices_preserved(self):
        """保留 cpu / cuda / npu / mps 支持（向后兼容）。"""
        assert _parse_device("cpu") == "cpu"
        assert _parse_device("cuda") == "cuda"
        assert _parse_device("cuda:0") == "cuda"
        assert _parse_device("CUDA:1") == "cuda"
        assert _parse_device("npu") == "npu"
        assert _parse_device("npu:1") == "npu"
        assert _parse_device("mps") == "mps"

    def test_parse_none_returns_cpu(self):
        """None 解析为 'cpu'。"""
        assert _parse_device(None) == "cpu"

    def test_parse_unknown_returns_cpu(self):
        """未知字符串兜底为 'cpu'（_parse_device 的 fallback 语义）。"""
        assert _parse_device("foo") == "cpu"
        assert _parse_device("") == "cpu"


# ===========================================================================
# 2. ROCm / CANN 探测 API（SubTask 11.3 测试要点 2）
# ===========================================================================


class TestProbeAPI:
    """has_rocm / has_cann / get_rocm_version / get_cann_version 不抛异常。"""

    def test_has_rocm_returns_bool(self):
        """has_rocm() 返回 bool，不抛异常。"""
        assert isinstance(has_rocm(), bool)

    def test_has_cann_returns_bool(self):
        """has_cann() 返回 bool，不抛异常。"""
        assert isinstance(has_cann(), bool)

    def test_get_rocm_version_type(self):
        """get_rocm_version() 返回 str 或 None。"""
        v = get_rocm_version()
        assert v is None or isinstance(v, str)

    def test_get_cann_version_type(self):
        """get_cann_version() 返回 str 或 None。"""
        v = get_cann_version()
        assert v is None or isinstance(v, str)

    def test_rocm_version_consistency(self):
        """若 has_rocm() 为 True，get_rocm_version() 应返回非空 str。"""
        if has_rocm():
            v = get_rocm_version()
            assert isinstance(v, str) and len(v) > 0
        else:
            # 非 ROCm 环境：版本应为 None
            assert get_rocm_version() is None

    def test_cann_version_consistency(self):
        """若 has_cann() 为 True，get_cann_version() 应返回非空 str。"""
        if has_cann():
            v = get_cann_version()
            assert isinstance(v, str) and len(v) > 0
        else:
            # 非 CANN 环境：版本应为 None
            assert get_cann_version() is None

    def test_has_torch_npu_returns_bool(self):
        """has_torch_npu() 返回 bool，不抛异常。"""
        assert isinstance(has_torch_npu(), bool)

    def test_has_torch_returns_bool(self):
        """has_torch() 返回 bool，不抛异常。"""
        assert isinstance(has_torch(), bool)


# ===========================================================================
# 3. autocast 等价性（SubTask 11.3 测试要点 3，需要 torch，无则 skip）
# ===========================================================================


@pytest.mark.skipif(not has_torch(), reason="需要 PyTorch")
class TestAutocastEquivalent:
    """autocast(device='rocm') 与 autocast(device='cuda') 行为等价。

    Part5K1.3 Task 10.3: device_type='rocm' 等价 device_type='cuda'
    （PyTorch ROCm build 原生支持 fp16 autocast via HIP）。
    """

    def test_autocast_rocm_equivalent_to_cuda(self):
        """验证 autocast(device='rocm') 与 autocast(device='cuda') 行为等价。

        - ROCm 环境：真正启用 autocast（不抛异常）
        - 非 ROCm + CUDA 可用：autocast(device='rocm') 内部映射到 cuda，应能启用
        - 非 ROCm + 无 CUDA：autocast(device='rocm') 抛 RuntimeError 提到 cuda
          （验证 rocm → cuda 路径映射生效，等价性体现在路径映射上）
        """
        import torch
        from verse_torch.backend_torch import autocast

        if has_rocm() or torch.cuda.is_available():
            # 可以实际启用 autocast（rocm 内部映射到 cuda，行为等价）
            with autocast(device="rocm", enabled=True):
                x = torch.randn(3, 4, device="cuda")
                y = x * 2
            assert y.shape == (3, 4)
        else:
            # 无 GPU：autocast(device='rocm') 应抛 RuntimeError 提到 cuda
            # （证明 rocm → cuda 映射生效，路径等价）
            with pytest.raises(RuntimeError) as exc_info:
                with autocast(device="rocm", enabled=True):
                    pass
            assert "cuda" in str(exc_info.value).lower()

    def test_torch_device_rocm_maps_to_cuda(self):
        """_torch_device('rocm') 内部映射到 torch.device('cuda')。

        Part5K1.3 Task 10.1: rocm / rocm:N → cuda / cuda:N 映射。
        """
        from verse_torch.backend_torch import _torch_device
        import torch
        if not torch.cuda.is_available():
            pytest.skip("无 CUDA/ROCm 设备，无法验证 _torch_device('rocm') 映射")
        rocm_dev = _torch_device("rocm")
        cuda_dev = _torch_device("cuda")
        assert str(rocm_dev) == str(cuda_dev)
        assert rocm_dev.type == "cuda"

    def test_torch_device_rocm_with_index_maps_to_cuda(self):
        """_torch_device('rocm:1') 内部映射到 torch.device('cuda:1')。"""
        from verse_torch.backend_torch import _torch_device
        import torch
        if not torch.cuda.is_available():
            pytest.skip("无 CUDA/ROCm 设备")
        if torch.cuda.device_count() < 2:
            pytest.skip("仅 1 个 GPU，无法验证 rocm:1 映射")
        rocm_dev = _torch_device("rocm:1")
        cuda_dev = _torch_device("cuda:1")
        assert str(rocm_dev) == str(cuda_dev)

    def test_autocast_cpu_noop(self):
        """CPU autocast 为 no-op，不抛异常。"""
        from verse_torch.backend_torch import autocast
        import torch
        with autocast(device="cpu", enabled=True):
            x = torch.randn(3, 4)
            y = x * 2
        assert y.shape == (3, 4)

    def test_autocast_disabled_noop(self):
        """autocast enabled=False 时为 no-op（无论 device 是 rocm 还是 cuda）。"""
        from verse_torch.backend_torch import autocast
        import torch
        with autocast(device="rocm", enabled=False):
            x = torch.randn(3, 4)
        assert x.shape == (3, 4)

    def test_autocast_rocm_runs_on_rocm_env(self):
        """若 has_rocm() 为 True（真实 ROCm 环境），autocast(device='rocm') 实际启用。"""
        if not has_rocm():
            pytest.skip("非 ROCm 环境")
        from verse_torch.backend_torch import autocast
        import torch
        # 真实 ROCm 环境下 autocast 应能实际启用
        with autocast(device="rocm", enabled=True):
            x = torch.randn(3, 4, device="cuda")
            y = x * 2
        assert y.shape == (3, 4)

    def test_autocast_rocm_runs_on_cuda_env(self):
        """非 ROCm 但有 CUDA 时，autocast(device='rocm') 内部映射到 cuda，
        应能实际启用（验证映射正确，与 autocast(device='cuda') 行为等价）。"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("无 CUDA/ROCm 设备")
        if has_rocm():
            pytest.skip("ROCm 环境（由 test_autocast_rocm_runs_on_rocm_env 覆盖）")
        from verse_torch.backend_torch import autocast
        # cuda 可用，autocast(device='rocm') 内部映射到 cuda，应能启用
        with autocast(device="rocm", enabled=True):
            x = torch.randn(3, 4, device="cuda")
            y = x * 2
        assert y.shape == (3, 4)


# ===========================================================================
# 4. spark/run.py CLI --device 参数支持（SubTask 11.3 测试要点 4）
# ===========================================================================


class TestCLIDeviceArg:
    """spark/run.py --device 支持 rocm 系列（Part5K1.3 Task 11.1）。

    无 torch 环境时仅验证 device 字符串识别（argparse 解析），
    不实际启动训练（用 --dry-run 避免 verse_trainer.train 调用）。
    """

    def test_train_device_rocm(self):
        """--device rocm 可被 argparse 解析。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "rocm"])
        assert args.device == "rocm"

    def test_train_device_rocm_with_index(self):
        """--device rocm:0 可被 argparse 解析（含冒号）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "rocm:0"])
        assert args.device == "rocm:0"

    def test_train_device_rocm_uppercase(self):
        """--device ROCM:2 可被 argparse 解析（大小写无关）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "ROCM:2"])
        assert args.device == "ROCM:2"

    def test_train_device_cuda_backward_compat(self):
        """--device cuda 仍可被解析（向后兼容）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "cuda"])
        assert args.device == "cuda"

    def test_train_device_cuda_with_index_backward_compat(self):
        """--device cuda:0 仍可被解析（向后兼容）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "cuda:0"])
        assert args.device == "cuda:0"

    def test_train_device_npu_backward_compat(self):
        """--device npu 仍可被解析（向后兼容）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "npu"])
        assert args.device == "npu"

    def test_train_device_cpu_backward_compat(self):
        """--device cpu 仍可被解析（向后兼容）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "cpu"])
        assert args.device == "cpu"

    def test_train_device_mps(self):
        """--device mps 可被解析（新支持）。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["train", "--device", "mps"])
        assert args.device == "mps"

    def test_train_device_invalid_rejected(self):
        """--device foo 被 argparse 拒绝（exit 2）。"""
        from spark.run import build_parser
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train", "--device", "foo"])

    def test_train_device_invalid_index_rejected(self):
        """--device rocm:abc 被 argparse 拒绝（冒号后非整数）。"""
        from spark.run import build_parser
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train", "--device", "rocm:abc"])

    def test_finetune_device_rocm(self):
        """finetune --device rocm 可被解析。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["finetune", "--device", "rocm"])
        assert args.device == "rocm"

    def test_continue_device_rocm(self):
        """continue --device rocm 可被解析。"""
        from spark.run import build_parser
        args = build_parser().parse_args([
            "continue", "--checkpoint", "ck.pt", "--device", "rocm",
        ])
        assert args.device == "rocm"

    def test_posttrain_device_rocm(self):
        """posttrain --device rocm 可被解析。"""
        from spark.run import build_parser
        args = build_parser().parse_args(["posttrain", "--device", "rocm"])
        assert args.device == "rocm"

    def test_train_dry_run_rocm(self):
        """train --device rocm --dry-run 不报错（端到端 smoke）。

        无 torch 环境时仅验证 device 字符串识别 + _print_device_info 不抛异常，
        不实际启动训练（dry-run 直接返回 0）。
        """
        from spark.run import build_parser, cmd_train
        args = build_parser().parse_args([
            "train", "--model", "small", "--device", "rocm", "--dry-run",
        ])
        ret = cmd_train(args)
        assert ret == 0

    def test_train_dry_run_rocm_with_index(self):
        """train --device rocm:0 --dry-run 不报错。"""
        from spark.run import build_parser, cmd_train
        args = build_parser().parse_args([
            "train", "--model", "small", "--device", "rocm:0", "--dry-run",
        ])
        ret = cmd_train(args)
        assert ret == 0

    def test_train_dry_run_cpu_no_regression(self):
        """train --device cpu --dry-run 不报错（向后兼容 smoke）。"""
        from spark.run import build_parser, cmd_train
        args = build_parser().parse_args([
            "train", "--model", "small", "--device", "cpu", "--dry-run",
        ])
        ret = cmd_train(args)
        assert ret == 0

    def test_train_dry_run_no_device_no_regression(self):
        """train（不传 --device）--dry-run 不报错（默认 None 路径）。"""
        from spark.run import build_parser, cmd_train
        args = build_parser().parse_args([
            "train", "--model", "small", "--dry-run",
        ])
        # args.device 应为 None（默认值）
        assert args.device is None
        ret = cmd_train(args)
        assert ret == 0

    def test_validate_device_function(self):
        """_validate_device 直接调用：合法 device 返回原值，非法抛 ArgumentTypeError。"""
        from spark.run import _validate_device
        import argparse
        # 合法 device（原值返回，不修改大小写）
        for dev in ("cpu", "cuda", "cuda:0", "npu", "npu:1", "mps",
                    "rocm", "rocm:0", "ROCM:2"):
            assert _validate_device(dev) == dev
        # 非法 device 抛 ArgumentTypeError
        for bad in ("foo", "rocm:abc", "cuda:x", "gpu", ""):
            with pytest.raises(argparse.ArgumentTypeError):
                _validate_device(bad)


# ===========================================================================
# 5. CANN 兜底 + get_memory_info（SubTask 11.3 测试要点 5）
# ===========================================================================


class TestCANNFallback:
    """get_memory_info 在无 torch_npu / 无 GPU 时不抛异常。"""

    def test_npu_memory_info_no_raise(self):
        """get_memory_info('npu') 不抛异常（无 torch_npu 时返回 0 占位 dict）。"""
        info = get_memory_info("npu")
        assert isinstance(info, dict)
        # dict 应包含 total / used / free 三个键
        assert "total" in info
        assert "used" in info
        assert "free" in info
        # 无 torch_npu 时应为 0 占位（或非负数）
        assert info["total"] >= 0
        assert info["used"] >= 0
        assert info["free"] >= 0

    def test_rocm_memory_info_no_raise(self):
        """get_memory_info('rocm') 不抛异常。"""
        info = get_memory_info("rocm")
        assert isinstance(info, dict)
        assert "total" in info
        assert "used" in info
        assert "free" in info
        assert info["total"] >= 0

    def test_cpu_memory_info(self):
        """get_memory_info('cpu') 返回 dict（可能为 0 占位）。"""
        info = get_memory_info("cpu")
        assert isinstance(info, dict)
        assert "total" in info

    def test_npu_memory_info_dict_keys(self):
        """get_memory_info('npu') 返回 dict 含 total/used/free 三键。"""
        info = get_memory_info("npu")
        assert set(info.keys()) >= {"total", "used", "free"}

    def test_npu_memory_info_without_torch_npu(self):
        """无 torch_npu 时 get_memory_info('npu') 返回 0 占位（不抛异常）。"""
        if has_torch_npu():
            pytest.skip("torch_npu 可用，跳过无 torch_npu 兜底测试")
        info = get_memory_info("npu")
        assert info == {"total": 0, "used": 0, "free": 0}


# ===========================================================================
# 6. get_backend 工厂测试（额外覆盖）
# ===========================================================================


class TestGetBackend:
    """get_backend('rocm') 工厂行为测试。"""

    def test_get_backend_rocm_without_torch_raises(self):
        """无 torch 时请求 ROCm backend 抛 RuntimeError（与 cuda/npu 一致）。"""
        if has_torch():
            pytest.skip("PyTorch 可用，跳过无 torch 回退测试")
        clear_backend_cache()
        with pytest.raises(RuntimeError) as exc_info:
            get_backend("rocm")
        # 错误信息应提到 PyTorch 未安装
        assert "PyTorch" in str(exc_info.value) or "torch" in str(exc_info.value).lower()

    def test_get_backend_rocm_with_torch(self):
        """有 torch 时 get_backend('rocm') 返回 TorchBackend，device_type='rocm'。

        Part5K1.3 Task 9.6: get_backend("rocm") 内部映射到 TorchBackend(device="cuda")，
        对外暴露 'rocm' device_type 用于诊断。
        """
        if not has_torch():
            pytest.skip("无 PyTorch")
        clear_backend_cache()
        try:
            backend = get_backend("rocm")
            # device_type 应保留 'rocm' 用于诊断
            assert backend.device_type == "rocm"
        except RuntimeError as e:
            # 如果报错，不应是 "torch 不可用" 错误（说明 torch 已安装）
            assert "未安装 PyTorch" not in str(e)
            pytest.skip(f"构造 TorchBackend 失败（可能无 GPU）：{e}")

    def test_get_backend_rocm_with_index(self):
        """get_backend('rocm:0') 返回 TorchBackend，device_type='rocm'。"""
        if not has_torch():
            pytest.skip("无 PyTorch")
        clear_backend_cache()
        try:
            backend = get_backend("rocm:0")
            assert backend.device_type == "rocm"
        except RuntimeError as e:
            if "未安装 PyTorch" in str(e):
                raise
            pytest.skip(f"构造 TorchBackend 失败（可能无 GPU）：{e}")


# ===========================================================================
# 7. _print_device_info 不抛异常（Part5K1.3 Task 11.2 额外覆盖）
# ===========================================================================


class TestPrintDeviceInfo:
    """_print_device_info 在各种 device 下都不抛异常。"""

    @pytest.mark.parametrize("device", [
        "cpu", "cuda", "cuda:0", "npu", "npu:0", "mps",
        "rocm", "rocm:0", "ROCM:2", None,
    ])
    def test_no_raise(self, device, capsys):
        """_print_device_info(device) 不抛异常，且至少打印一行 [device]。"""
        from spark.run import _print_device_info
        _print_device_info(device)
        captured = capsys.readouterr()
        assert "[device]" in captured.out

    def test_print_device_info_cpu_shows_cpu_mode(self, capsys):
        """_print_device_info('cpu') 打印 CPU mode。"""
        from spark.run import _print_device_info
        _print_device_info("cpu")
        captured = capsys.readouterr()
        assert "CPU mode" in captured.out
        assert "[device] target device: cpu" in captured.out

    def test_print_device_info_rocm_shows_cuda_or_rocm(self, capsys):
        """_print_device_info('rocm') 打印 CUDA 或 ROCm 版本信息。"""
        from spark.run import _print_device_info
        _print_device_info("rocm")
        captured = capsys.readouterr()
        # 应至少打印 "NVIDIA CUDA" 或 "AMD ROCm version"
        assert "CUDA" in captured.out or "ROCm" in captured.out

    def test_print_device_info_none_shows_auto(self, capsys):
        """_print_device_info(None) 打印 'auto'（cmd_train 默认从 config 推断）。"""
        from spark.run import _print_device_info
        _print_device_info(None)
        captured = capsys.readouterr()
        assert "auto" in captured.out

    def test_print_device_info_npu_shows_cann_or_memory(self, capsys):
        """_print_device_info('npu') 打印 CANN 版本或 NPU 显存信息。"""
        from spark.run import _print_device_info
        _print_device_info("npu")
        captured = capsys.readouterr()
        # 应打印 NPU memory（无 torch_npu 时为 0 占位 dict）
        assert "NPU memory" in captured.out or "CANN" in captured.out


# ===========================================================================
# 主入口
# ===========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
