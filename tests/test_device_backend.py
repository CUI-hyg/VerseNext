"""设备抽象层与 GPU/NPU 后端单元测试（Task 1.9）。

覆盖内容：
1. NumpyBackend 算子：matmul / linear / softmax / layernorm / rmsnorm / rope / attention
2. TorchBackend 算子（无 torch 时 skip）：与 NumpyBackend 数值等价
3. get_backend 工厂：CPU 返回 NumpyBackend，GPU 返回 TorchBackend（需 torch）
4. 无 torch 时请求 GPU 抛 RuntimeError
5. Tensor.device 属性 + .to() / .cuda() / .npu() / .cpu() 迁移
6. autocast 在 CPU 时为 no-op
7. Module.to(device) 参数迁移
8. 向后兼容：无 torch 时纯 NumPy 路径正常工作
9. 新优化器 NAdamW / RMSProp CPU 路径
10. 新损失 contrastive_loss / perplexity
11. 新 nn 组件 RotaryEmbedding / KVCache / StaticCache / DynamicCache / GroupNorm / Conv1d / LayerNormFast

运行方式：
    python3 -m pytest tests/test_device_backend.py -v
    python3 tests/test_device_backend.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import Tensor
from verse_torch.device import (
    DeviceBackend,
    NumpyBackend,
    get_backend,
    has_torch,
    has_torch_npu,
    _parse_device,
    is_cpu_device,
    DEFAULT_DEVICE,
    clear_backend_cache,
)


def _set_seed(seed: int = 42):
    np.random.seed(seed)


# ===========================================================================
# 1. NumpyBackend 算子测试
# ===========================================================================


class TestNumpyBackend(unittest.TestCase):
    """NumpyBackend 各算子的正确性测试。"""

    def setUp(self):
        _set_seed(42)
        self.backend = NumpyBackend()

    def test_device_type(self):
        """NumpyBackend.device_type 应为 'cpu'。"""
        self.assertEqual(self.backend.device_type, "cpu")

    def test_matmul(self):
        """matmul: (2,3) @ (3,4) -> (2,4)。"""
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        out = self.backend.matmul(a, b)
        expected = a @ b
        np.testing.assert_allclose(out, expected, atol=1e-5)
        self.assertEqual(out.shape, (2, 4))

    def test_linear(self):
        """linear: y = x @ W.T + b。"""
        x = np.random.randn(4, 3).astype(np.float32)
        w = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(2).astype(np.float32)
        out = self.backend.linear(x, w, b)
        expected = x @ w.T + b
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_linear_no_bias(self):
        """linear 无 bias 时 y = x @ W.T。"""
        x = np.random.randn(4, 3).astype(np.float32)
        w = np.random.randn(2, 3).astype(np.float32)
        out = self.backend.linear(x, w, None)
        expected = x @ w.T
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_softmax(self):
        """softmax: 沿最后一维归一化，和为 1。"""
        x = np.random.randn(3, 5).astype(np.float32)
        out = self.backend.softmax(x, dim=-1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)
        # 数值稳定：大数不溢出
        x_large = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        out_large = self.backend.softmax(x_large, dim=-1)
        self.assertTrue(np.all(np.isfinite(out_large)))

    def test_layernorm(self):
        """layernorm: 归一化后均值≈0，方差≈1。"""
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        out = self.backend.layernorm(x, w, b, eps=1e-5)
        # 归一化后均值≈0（eps 影响）
        mean = out.mean(axis=-1)
        np.testing.assert_allclose(mean, 0.0, atol=1e-4)
        # 带 weight/bias 的仿射
        w2 = np.full(8, 2.0, dtype=np.float32)
        b2 = np.full(8, 1.0, dtype=np.float32)
        out2 = self.backend.layernorm(x, w2, b2, eps=1e-5)
        # 缩放后方差≈4
        var2 = out2.var(axis=-1)
        np.testing.assert_allclose(var2, 4.0, atol=0.1)

    def test_rmsnorm(self):
        """rmsnorm: 用 RMS 归一化。"""
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        out = self.backend.rmsnorm(x, w, eps=1e-6)
        # RMS(x_normed) ≈ 1
        rms = np.sqrt((out * out).mean(axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_rope_no_cos_sin(self):
        """rope 无 cos/sin 时原样返回。"""
        x = np.random.randn(2, 4, 8).astype(np.float32)
        out = self.backend.rope(x)
        np.testing.assert_allclose(out, x)

    def test_rope_with_cos_sin(self):
        """rope 带 cos/sin 时做旋转。"""
        B, T, D = 2, 4, 8
        x = np.random.randn(B, T, D).astype(np.float32)
        half = D // 2
        cos = np.random.randn(B, T, half).astype(np.float32)
        sin = np.random.randn(B, T, half).astype(np.float32)
        out = self.backend.rope(x, cos=cos, sin=sin)
        self.assertEqual(out.shape, x.shape)
        # 验证旋转不改变模长（近似）
        # x1 * cos - x2 * sin, x1 * sin + x2 * cos
        x1, x2 = x[..., :half], x[..., half:]
        expected = np.concatenate(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
        )
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_attention(self):
        """attention: SDPA 前向 shape 正确。"""
        B, T, D = 2, 4, 8
        q = np.random.randn(B, T, D).astype(np.float32)
        k = np.random.randn(B, T, D).astype(np.float32)
        v = np.random.randn(B, T, D).astype(np.float32)
        out = self.backend.attention(q, k, v)
        self.assertEqual(out.shape, (B, T, D))


# ===========================================================================
# 2. get_backend 工厂测试
# ===========================================================================


class TestGetBackend(unittest.TestCase):
    """get_backend 工厂函数测试。"""

    def test_get_cpu_backend(self):
        """get_backend('cpu') 返回 NumpyBackend。"""
        clear_backend_cache()
        backend = get_backend("cpu")
        self.assertIsInstance(backend, NumpyBackend)

    def test_get_none_backend(self):
        """get_backend(None) 等价于 get_backend('cpu')。"""
        clear_backend_cache()
        backend = get_backend(None)
        self.assertIsInstance(backend, NumpyBackend)

    def test_get_cpu_caching(self):
        """get_backend('cpu') 缓存返回同一实例。"""
        clear_backend_cache()
        b1 = get_backend("cpu")
        b2 = get_backend("cpu")
        self.assertIs(b1, b2)

    def test_gpu_without_torch_raises(self):
        """无 torch 时请求 GPU 抛 RuntimeError。"""
        if has_torch():
            self.skipTest("PyTorch 可用，跳过无 torch 回退测试")
        clear_backend_cache()
        with self.assertRaises(RuntimeError):
            get_backend("cuda")
        with self.assertRaises(RuntimeError):
            get_backend("npu")

    def test_npu_without_torch_npu_raises(self):
        """有 torch 但无 torch_npu 时请求 NPU 抛 RuntimeError。"""
        if not has_torch():
            self.skipTest("PyTorch 不可用")
        if has_torch_npu():
            self.skipTest("torch_npu 可用，跳过")
        clear_backend_cache()
        with self.assertRaises(RuntimeError):
            get_backend("npu")


# ===========================================================================
# 3. 设备字符串解析测试
# ===========================================================================


class TestDeviceParsing(unittest.TestCase):
    """_parse_device / is_cpu_device 测试。"""

    def test_parse_cpu(self):
        self.assertEqual(_parse_device("cpu"), "cpu")
        self.assertEqual(_parse_device(None), "cpu")

    def test_parse_cuda(self):
        self.assertEqual(_parse_device("cuda"), "cuda")
        self.assertEqual(_parse_device("cuda:0"), "cuda")
        self.assertEqual(_parse_device("CUDA:1"), "cuda")

    def test_parse_npu(self):
        self.assertEqual(_parse_device("npu"), "npu")
        self.assertEqual(_parse_device("npu:0"), "npu")

    def test_parse_mps(self):
        self.assertEqual(_parse_device("mps"), "mps")

    def test_is_cpu_device(self):
        self.assertTrue(is_cpu_device("cpu"))
        self.assertTrue(is_cpu_device(None))
        self.assertFalse(is_cpu_device("cuda"))
        self.assertFalse(is_cpu_device("npu:0"))

    def test_default_device(self):
        self.assertEqual(DEFAULT_DEVICE, "cpu")


# ===========================================================================
# 4. Tensor 设备属性与迁移测试
# ===========================================================================


class TestTensorDevice(unittest.TestCase):
    """Tensor.device / .to() / .cuda() / .cpu() 测试。"""

    def setUp(self):
        _set_seed(42)

    def test_tensor_default_device(self):
        """新建 Tensor 默认 device 为 'cpu'。"""
        t = Tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.device, "cpu")

    def test_tensor_to_cpu_same_device(self):
        """Tensor.to('cpu') 在已是 CPU 时返回 self（短路）。"""
        t = Tensor([1.0, 2.0])
        t2 = t.to("cpu")
        self.assertIs(t2, t)

    def test_tensor_to_cpu_with_dtype(self):
        """Tensor.to('cpu', dtype) 转换 dtype。"""
        t = Tensor([1.0, 2.0], dtype=np.float32)
        t2 = t.to("cpu", dtype=np.float64)
        self.assertEqual(t2.dtype, np.float64)

    def test_tensor_cuda_without_torch_raises(self):
        """无 torch 时 .cuda() 抛 RuntimeError。"""
        if has_torch():
            self.skipTest("PyTorch 可用，跳过无 torch 测试")
        t = Tensor([1.0, 2.0])
        with self.assertRaises(RuntimeError):
            t.cuda()

    def test_tensor_npu_without_torch_raises(self):
        """无 torch 时 .npu() 抛 RuntimeError。"""
        if has_torch():
            self.skipTest("PyTorch 可用，跳过无 torch 测试")
        t = Tensor([1.0, 2.0])
        with self.assertRaises(RuntimeError):
            t.npu()

    def test_tensor_is_cuda_property(self):
        """Tensor.is_cuda 在 CPU 上为 False。"""
        t = Tensor([1.0, 2.0])
        self.assertFalse(t.is_cuda)

    def test_tensor_to_cpu_preserves_data(self):
        """Tensor.to('cpu') 数据不变。"""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        t2 = t.to("cpu")
        np.testing.assert_allclose(t2.data, t.data)


# ===========================================================================
# 5. autocast 测试
# ===========================================================================


class TestAutocast(unittest.TestCase):
    """autocast 上下文管理器测试。"""

    def test_autocast_cpu_noop(self):
        """autocast 在 CPU 上为 no-op，不改变计算结果。"""
        from verse_torch.training import _get_autocast
        from contextlib import nullcontext
        ctx = _get_autocast("cpu", enabled=True)
        self.assertIsInstance(ctx, type(nullcontext()))

    def test_autocast_disabled_noop(self):
        """autocast enabled=False 时为 no-op。"""
        from verse_torch.training import _get_autocast
        from contextlib import nullcontext
        ctx = _get_autocast("cuda", enabled=False)
        self.assertIsInstance(ctx, type(nullcontext()))

    def test_autocast_none_device_noop(self):
        """autocast device=None 时为 no-op。"""
        from verse_torch.training import _get_autocast
        from contextlib import nullcontext
        ctx = _get_autocast(None, enabled=True)
        self.assertIsInstance(ctx, type(nullcontext()))

    def test_autocast_cpu_runs_without_error(self):
        """在 CPU autocast 下计算不报错。"""
        from verse_torch.training import _get_autocast
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        with _get_autocast("cpu", enabled=True):
            y = (t * t).sum()
        y.backward()
        np.testing.assert_allclose(t.grad, [2.0, 4.0, 6.0], atol=1e-5)


# ===========================================================================
# 6. Module.to(device) 测试
# ===========================================================================


class TestModuleToDevice(unittest.TestCase):
    """Module.to(device) 参数迁移测试。"""

    def setUp(self):
        _set_seed(42)
        from verse_torch.nn import Linear
        self.Linear = Linear

    def test_module_device_default(self):
        """Module 默认 device 为 'cpu'。"""
        m = self.Linear(4, 2)
        self.assertEqual(m.device, "cpu")

    def test_module_to_cpu(self):
        """Module.to('cpu') 迁移参数到 CPU（仍为 ndarray）。"""
        m = self.Linear(4, 2)
        m2 = m.to("cpu")
        self.assertIs(m2, m)  # to 返回 self
        # 参数仍为 ndarray（CPU 路径）
        self.assertIsInstance(m.weight.data, np.ndarray)

    def test_module_to_dtype_backward_compat(self):
        """Module.to(np.float64) 旧式 dtype 转换仍工作。"""
        m = self.Linear(4, 2)
        m.to(np.float64)
        self.assertEqual(m.weight.data.dtype, np.float64)

    def test_module_to_cpu_preserves_values(self):
        """Module.to('cpu') 参数值不变。"""
        m = self.Linear(4, 2)
        w_before = m.weight.data.copy()
        m.to("cpu")
        np.testing.assert_allclose(m.weight.data, w_before)


# ===========================================================================
# 7. 向后兼容测试
# ===========================================================================


class TestBackwardCompat(unittest.TestCase):
    """无 torch 时纯 NumPy 路径的向后兼容测试。"""

    def setUp(self):
        _set_seed(42)

    def test_tensor_ops_cpu(self):
        """Tensor 基本运算在 CPU 路径正常工作。"""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad, [4.0, 5.0, 6.0], atol=1e-5)
        np.testing.assert_allclose(b.grad, [1.0, 2.0, 3.0], atol=1e-5)

    def test_matmul_cpu(self):
        """Tensor @ 在 CPU 路径正确。"""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        np.testing.assert_allclose(c.data, expected, atol=1e-5)

    def test_softmax_cpu(self):
        """Tensor.softmax 在 CPU 路径和为 1。"""
        t = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        s = t.softmax(dim=-1)
        np.testing.assert_allclose(s.data.sum(axis=-1), 1.0, atol=1e-6)

    def test_no_grad_context(self):
        """no_grad 上下文不构建计算图。"""
        from verse_torch import no_grad
        with no_grad():
            t = Tensor([1.0, 2.0], requires_grad=True)
            y = t * 2
            # no_grad 下不构建计算图
            self.assertFalse(y.requires_grad)

    def test_existing_optimizers_work(self):
        """现有优化器 SGD/Adam/AdamW 仍工作。"""
        from verse_torch.optim import SGD, Adam, AdamW
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        for OptCls in [SGD, Adam, AdamW]:
            t2 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
            opt = OptCls([t2], lr=0.01)
            loss = (t2 * t2).sum()
            loss.backward()
            opt.step()
            # 参数应更新
            self.assertFalse(np.allclose(t2.data, [1.0, 2.0, 3.0]))


# ===========================================================================
# 8. 新优化器 NAdamW / RMSProp 测试
# ===========================================================================


class TestNewOptimizers(unittest.TestCase):
    """NAdamW / RMSProp 优化器测试。"""

    def setUp(self):
        _set_seed(42)

    def test_nadamw_step(self):
        """NAdamW 单步更新不报错且参数变化。"""
        from verse_torch.optim import NAdamW
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        opt = NAdamW([t], lr=0.01)
        loss = (t * t).sum()
        loss.backward()
        before = t.data.copy()
        opt.step()
        # 参数应更新
        self.assertFalse(np.allclose(t.data, before))

    def test_nadamw_multiple_steps(self):
        """NAdamW 多步更新收敛到 0（minimize x^2）。"""
        from verse_torch.optim import NAdamW
        t = Tensor([3.0, -2.0], requires_grad=True)
        opt = NAdamW([t], lr=0.1, weight_decay=0.0)
        for _ in range(200):
            opt.zero_grad()
            loss = (t * t).sum()
            loss.backward()
            opt.step()
        np.testing.assert_allclose(t.data, [0.0, 0.0], atol=0.1)

    def test_nadamw_weight_decay(self):
        """NAdamW weight_decay > 0 时额外衰减参数。"""
        from verse_torch.optim import NAdamW
        t = Tensor([1.0], requires_grad=True)
        opt = NAdamW([t], lr=0.0, weight_decay=0.1)
        # grad=0，只有 weight_decay 生效
        t.grad = np.zeros_like(t.data)
        before = t.data.copy()
        opt.step()
        # p *= (1 - lr * wd) = 1 * (1 - 0 * 0.1) = 1，lr=0 不更新
        # 但 m/v 被初始化
        np.testing.assert_allclose(t.data, before)

    def test_rmsprop_step(self):
        """RMSProp 单步更新不报错且参数变化。"""
        from verse_torch.optim import RMSProp
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        opt = RMSProp([t], lr=0.01)
        loss = (t * t).sum()
        loss.backward()
        before = t.data.copy()
        opt.step()
        self.assertFalse(np.allclose(t.data, before))

    def test_rmsprop_multiple_steps(self):
        """RMSProp 多步更新收敛到 0。"""
        from verse_torch.optim import RMSProp
        t = Tensor([3.0, -2.0], requires_grad=True)
        opt = RMSProp([t], lr=0.1)
        for _ in range(300):
            opt.zero_grad()
            loss = (t * t).sum()
            loss.backward()
            opt.step()
        np.testing.assert_allclose(t.data, [0.0, 0.0], atol=0.2)

    def test_rmsprop_with_momentum(self):
        """RMSProp 带 momentum 不报错。"""
        from verse_torch.optim import RMSProp
        t = Tensor([1.0, 2.0], requires_grad=True)
        opt = RMSProp([t], lr=0.01, momentum=0.9)
        loss = (t * t).sum()
        loss.backward()
        opt.step()
        # 第二步（momentum 累积）
        opt.zero_grad()
        loss = (t * t).sum()
        loss.backward()
        opt.step()

    def test_rmsprop_centered(self):
        """RMSProp centered=True 不报错。"""
        from verse_torch.optim import RMSProp
        t = Tensor([1.0, 2.0], requires_grad=True)
        opt = RMSProp([t], lr=0.01, centered=True)
        loss = (t * t).sum()
        loss.backward()
        opt.step()

    def test_nadamw_param_groups(self):
        """NAdamW 支持参数组。"""
        from verse_torch.optim import NAdamW
        t1 = Tensor([1.0], requires_grad=True)
        t2 = Tensor([2.0], requires_grad=True)
        opt = NAdamW([
            {"params": [t1], "weight_decay": 0.0},
            {"params": [t2], "weight_decay": 0.1},
        ], lr=0.01)
        t1.grad = np.array([1.0])
        t2.grad = np.array([1.0])
        opt.step()
        # 两组都应更新
        self.assertEqual(len(opt.param_groups), 2)


# ===========================================================================
# 9. 新损失函数测试
# ===========================================================================


class TestNewLosses(unittest.TestCase):
    """contrastive_loss / perplexity 测试。"""

    def setUp(self):
        _set_seed(42)

    def test_perplexity_basic(self):
        """perplexity = exp(CE)，对均匀 logits 应为 V。"""
        from verse_torch.losses import perplexity
        # 均匀 logits -> CE = log(V) -> PPL = V
        V = 10
        logits = Tensor(np.zeros((4, V), dtype=np.float32), requires_grad=True)
        targets = np.array([0, 1, 2, 3])
        ppl = perplexity(logits, targets)
        # PPL ≈ V = 10
        self.assertAlmostEqual(float(ppl.data), V, places=1)

    def test_perplexity_perfect(self):
        """完美预测（target logit 远大于其他）时 PPL≈1。"""
        from verse_torch.losses import perplexity
        logits_data = np.full((2, 5), -100.0, dtype=np.float32)
        logits_data[0, 0] = 100.0  # target=0
        logits_data[1, 1] = 100.0  # target=1
        logits = Tensor(logits_data, requires_grad=True)
        targets = np.array([0, 1])
        ppl = perplexity(logits, targets)
        self.assertLess(float(ppl.data), 1.1)

    def test_perplexity_backward(self):
        """perplexity 支持 backward。"""
        from verse_torch.losses import perplexity
        logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        targets = np.array([0, 1, 2, 3])
        ppl = perplexity(logits, targets)
        ppl.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, (4, 10))

    def test_perplexity_ignore_index(self):
        """perplexity 支持 ignore_index。"""
        from verse_torch.losses import perplexity
        logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        targets = np.array([0, -100, 2, -100])
        ppl = perplexity(logits, targets, ignore_index=-100)
        # 不报错且为有限值
        self.assertTrue(np.isfinite(float(ppl.data)))

    def test_contrastive_loss_inbatch(self):
        """contrastive_loss 无 negatives（in-batch）返回正值且可 backward。"""
        from verse_torch.losses import contrastive_loss
        B, D = 8, 16
        a = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        p = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        loss = contrastive_loss(a, p, temperature=0.07)
        # loss 应为正（负 log 概率）
        self.assertGreater(float(loss.data), 0)
        loss.backward()
        self.assertIsNotNone(a.grad)

    def test_contrastive_loss_explicit_negatives(self):
        """contrastive_loss 带显式 negatives (N, D) 不报错。"""
        from verse_torch.losses import contrastive_loss
        B, D, N = 4, 8, 6
        a = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        p = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        n = Tensor(np.random.randn(N, D).astype(np.float32), requires_grad=False)
        loss = contrastive_loss(a, p, negatives=n, temperature=0.1)
        self.assertGreater(float(loss.data), 0)
        loss.backward()

    def test_contrastive_loss_per_anchor_negatives(self):
        """contrastive_loss 带 (B, N, D) negatives 不报错。"""
        from verse_torch.losses import contrastive_loss
        B, D, N = 4, 8, 3
        a = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        p = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        n = Tensor(np.random.randn(B, N, D).astype(np.float32), requires_grad=False)
        loss = contrastive_loss(a, p, negatives=n, temperature=0.1)
        self.assertGreater(float(loss.data), 0)
        loss.backward()

    def test_contrastive_loss_dot_sim_fn(self):
        """contrastive_loss sim_fn='dot' 不报错。"""
        from verse_torch.losses import contrastive_loss
        B, D = 4, 8
        a = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        p = Tensor(np.random.randn(B, D).astype(np.float32), requires_grad=True)
        loss = contrastive_loss(a, p, sim_fn="dot")
        loss.backward()

    def test_contrastive_loss_low_temp_higher_loss(self):
        """温度越低 loss 越高（对比越尖锐）。"""
        from verse_torch.losses import contrastive_loss
        B, D = 8, 16
        np.random.seed(42)
        a_data = np.random.randn(B, D).astype(np.float32)
        p_data = np.random.randn(B, D).astype(np.float32)
        loss_high_temp = float(contrastive_loss(
            Tensor(a_data), Tensor(p_data), temperature=10.0).data)
        loss_low_temp = float(contrastive_loss(
            Tensor(a_data), Tensor(p_data), temperature=0.01).data)
        self.assertGreater(loss_low_temp, loss_high_temp)


# ===========================================================================
# 10. 新 nn 组件测试
# ===========================================================================


class TestNewNNComponents(unittest.TestCase):
    """RotaryEmbedding / KVCache / GroupNorm / Conv1d / LayerNormFast 测试。"""

    def setUp(self):
        _set_seed(42)

    def test_rotary_embedding_shape(self):
        """RotaryEmbedding 前向输出 shape 正确。"""
        from verse_torch.nn import RotaryEmbedding
        rope = RotaryEmbedding(dim=8, max_seq_len=32)
        x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
        out = rope(x)
        self.assertEqual(out.shape, (2, 4, 8))

    def test_rotary_embedding_seq_len(self):
        """RotaryEmbedding 指定 seq_len 不报错。"""
        from verse_torch.nn import RotaryEmbedding
        rope = RotaryEmbedding(dim=8, max_seq_len=64)
        x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
        out = rope(x, seq_len=4)
        self.assertEqual(out.shape, (2, 4, 8))

    def test_kvcache_is_abstract(self):
        """KVCache 方法未实现时抛 NotImplementedError。"""
        from verse_torch.nn import KVCache
        cache = KVCache()
        # 基类方法应抛 NotImplementedError
        with self.assertRaises(NotImplementedError):
            cache.update(Tensor([1.0]), Tensor([1.0]), 0)
        with self.assertRaises(NotImplementedError):
            cache.get(0)

    def test_static_cache_basic(self):
        """StaticCache 基本读写。"""
        from verse_torch.nn import StaticCache
        cache = StaticCache(num_layers=1, max_batch=2, max_seq=8,
                            num_heads=2, head_dim=4)
        k = Tensor(np.random.randn(2, 3, 2, 4).astype(np.float32))
        v = Tensor(np.random.randn(2, 3, 2, 4).astype(np.float32))
        cache.update(k, v, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)
        # 前缀应为写入的 3 步
        self.assertEqual(k_out.shape[1], 3)

    def test_dynamic_cache_basic(self):
        """DynamicCache 基本追加读写。"""
        from verse_torch.nn import DynamicCache
        cache = DynamicCache()
        k1 = Tensor(np.random.randn(2, 3, 2, 4).astype(np.float32))
        v1 = Tensor(np.random.randn(2, 3, 2, 4).astype(np.float32))
        cache.update(k1, v1, layer_idx=0)
        # 追加
        k2 = Tensor(np.random.randn(2, 2, 2, 4).astype(np.float32))
        v2 = Tensor(np.random.randn(2, 2, 2, 4).astype(np.float32))
        cache.update(k2, v2, layer_idx=0)
        k_out, v_out = cache.get(layer_idx=0)
        # 总长度应为 3+2=5
        self.assertEqual(k_out.shape[1], 5)

    def test_groupnorm_shape(self):
        """GroupNorm 前向输出 shape 正确。"""
        from verse_torch.nn import GroupNorm
        gn = GroupNorm(num_groups=2, num_channels=8)
        x = Tensor(np.random.randn(2, 8, 4).astype(np.float32), requires_grad=True)
        out = gn(x)
        self.assertEqual(out.shape, (2, 8, 4))

    def test_groupnorm_normalization(self):
        """GroupNorm 归一化后每组均值≈0。"""
        from verse_torch.nn import GroupNorm
        gn = GroupNorm(num_groups=2, num_channels=8, eps=1e-6)
        # weight=1, bias=0（默认初始化）
        x = Tensor(np.random.randn(1, 8, 4).astype(np.float32))
        out = gn(x)
        # 每组 4 channels，reshape 到 (1, 2, 4, 4) 后均值≈0
        out_data = out.data.reshape(1, 2, 4, 4)
        means = out_data.mean(axis=(2, 3))
        np.testing.assert_allclose(means, 0.0, atol=1e-3)

    def test_conv1d_shape(self):
        """Conv1d 前向输出 shape 正确。"""
        from verse_torch.nn import Conv1d
        conv = Conv1d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        # (B, C_in, L)
        x = Tensor(np.random.randn(2, 3, 8).astype(np.float32), requires_grad=True)
        out = conv(x)
        self.assertEqual(out.shape, (2, 4, 8))

    def test_conv1d_no_padding(self):
        """Conv1d 无 padding 时长度缩减。"""
        from verse_torch.nn import Conv1d
        conv = Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=0)
        x = Tensor(np.random.randn(1, 2, 8).astype(np.float32))
        out = conv(x)
        self.assertEqual(out.shape, (1, 4, 6))  # L - k + 1 = 8 - 3 + 1 = 6

    def test_layernorm_fast_shape(self):
        """LayerNormFast 前向输出 shape 正确。"""
        from verse_torch.nn import LayerNormFast
        ln = LayerNormFast(normalized_shape=8)
        x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
        out = ln(x)
        self.assertEqual(out.shape, (2, 4, 8))

    def test_layernorm_fast_normalization(self):
        """LayerNormFast 归一化后均值≈0。"""
        from verse_torch.nn import LayerNormFast
        ln = LayerNormFast(normalized_shape=8, eps=1e-6)
        x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
        out = ln(x)
        means = out.data.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-3)

    def test_layernorm_fast_matches_layernorm(self):
        """LayerNormFast 与 LayerNorm 数值一致。"""
        from verse_torch.nn import LayerNorm, LayerNormFast
        np.random.seed(42)
        x_data = np.random.randn(2, 4, 8).astype(np.float32)
        ln1 = LayerNorm(normalized_shape=8)
        ln2 = LayerNormFast(normalized_shape=8)
        # 复制权重
        ln2.weight = ln1.weight
        ln2.bias = ln1.bias
        x1 = Tensor(x_data.copy())
        x2 = Tensor(x_data.copy())
        out1 = ln1(x1)
        out2 = ln2(x2)
        np.testing.assert_allclose(out1.data, out2.data, atol=1e-5)

    def test_module_to_migrates_rotary(self):
        """Module.to('cpu') 迁移 RotaryEmbedding 的 cos/sin。"""
        from verse_torch.nn import RotaryEmbedding
        rope = RotaryEmbedding(dim=8, max_seq_len=16)
        # cos/sin 应为 Tensor
        self.assertTrue(hasattr(rope, "cos"))
        rope.to("cpu")
        # 迁移后仍可前向
        x = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        out = rope(x)
        self.assertEqual(out.shape, (1, 4, 8))


# ===========================================================================
# 11. TorchBackend 测试（无 torch 时 skip）
# ===========================================================================


class TestTorchBackend(unittest.TestCase):
    """TorchBackend 算子测试（需要 PyTorch）。"""

    def setUp(self):
        if not has_torch():
            self.skipTest("PyTorch 不可用，跳过 TorchBackend 测试")
        _set_seed(42)
        import torch
        self.torch = torch
        from verse_torch.backend_torch import TorchBackend
        self.TorchBackend = TorchBackend

    def test_torch_backend_device_type(self):
        """TorchBackend.device_type 正确。"""
        backend = self.TorchBackend(device="cpu")
        self.assertEqual(backend.device_type, "cpu")

    def test_torch_backend_matmul(self):
        """TorchBackend.matmul 与 NumPy 等价。"""
        backend = self.TorchBackend(device="cpu")
        a = self.torch.randn(2, 3)
        b = self.torch.randn(3, 4)
        out = backend.matmul(a, b)
        expected = (a @ b).numpy()
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

    def test_torch_backend_softmax(self):
        """TorchBackend.softmax 和为 1。"""
        backend = self.TorchBackend(device="cpu")
        x = self.torch.randn(3, 5)
        out = backend.softmax(x, dim=-1)
        sums = out.sum(dim=-1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_torch_backend_layernorm(self):
        """TorchBackend.layernorm 归一化。"""
        backend = self.TorchBackend(device="cpu")
        x = self.torch.randn(4, 8)
        w = self.torch.ones(8)
        b = self.torch.zeros(8)
        out = backend.layernorm(x, w, b, eps=1e-5)
        means = out.mean(dim=-1).numpy()
        np.testing.assert_allclose(means, 0.0, atol=1e-4)

    def test_torch_backend_rmsnorm(self):
        """TorchBackend.rmsnorm RMS≈1。"""
        backend = self.TorchBackend(device="cpu")
        x = self.torch.randn(4, 8)
        w = self.torch.ones(8)
        out = backend.rmsnorm(x, w, eps=1e-6)
        rms = (out * out).mean(dim=-1).sqrt().numpy()
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_torch_backend_attention(self):
        """TorchBackend.attention shape 正确。"""
        backend = self.TorchBackend(device="cpu")
        B, T, D = 2, 4, 8
        q = self.torch.randn(B, T, D)
        k = self.torch.randn(B, T, D)
        v = self.torch.randn(B, T, D)
        out = backend.attention(q, k, v)
        self.assertEqual(out.shape, (B, T, D))

    def test_torch_to_numpy_conversion(self):
        """to_torch / to_numpy 转换正确。"""
        from verse_torch.backend_torch import to_torch, to_numpy
        arr = np.random.randn(3, 4).astype(np.float32)
        t = to_torch(arr, device="cpu")
        self.assertIsInstance(t, self.torch.Tensor)
        arr2 = to_numpy(t)
        np.testing.assert_allclose(arr2, arr)

    def test_autocast_gpu_on_cpu(self):
        """autocast 在 CPU 设备上为 no-op（不报错）。"""
        from verse_torch.backend_torch import autocast
        # CPU autocast 应为 no-op
        with autocast(device="cpu", enabled=True):
            x = self.torch.randn(3, 4)
            y = x * 2
        self.assertTrue(y.shape == (3, 4))

    def test_get_backend_cuda_torch(self):
        """get_backend('cuda') 在有 torch 时返回 TorchBackend（无 GPU 设备则报错）。"""
        if not self.torch.cuda.is_available():
            # 有 torch 但无 GPU：get_backend('cuda') 应尝试构造 TorchBackend
            # 可能因设备不存在而失败，这里只验证不抛 RuntimeError(torch 不可用)
            try:
                clear_backend_cache()
                backend = get_backend("cuda")
                self.assertIsNotNone(backend)
            except RuntimeError as e:
                # 设备不可用是合理的（不是 "torch 不可用"）
                self.assertNotIn("未安装 PyTorch", str(e))
        else:
            clear_backend_cache()
            backend = get_backend("cuda")
            self.assertIsNotNone(backend)


# ===========================================================================
# 12. Trainer device 参数测试
# ===========================================================================


class TestTrainerDevice(unittest.TestCase):
    """Trainer device 参数与 autocast 集成测试。"""

    def setUp(self):
        _set_seed(42)

    def test_trainer_default_device(self):
        """Trainer 默认 device 为 'cpu'。"""
        from verse_torch import Trainer
        from verse_torch.nn import Linear
        from verse_torch.optim import SGD
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, [], [], opt)
        self.assertEqual(trainer.device, "cpu")

    def test_trainer_device_param(self):
        """Trainer 接受 device 参数。"""
        from verse_torch import Trainer
        from verse_torch.nn import Linear
        from verse_torch.optim import SGD
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, [], [], opt, device="cpu")
        self.assertEqual(trainer.device, "cpu")
        self.assertFalse(trainer.use_autocast)

    def test_trainer_autocast_cfg(self):
        """Trainer cfg autocast=True 启用 autocast。"""
        from verse_torch import Trainer
        from verse_torch.nn import Linear
        from verse_torch.optim import SGD
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, [], [], opt, cfg={"autocast": True})
        self.assertTrue(trainer.use_autocast)

    def test_distributed_trainer_basic(self):
        """DistributedTrainer 基本构造。"""
        from verse_torch import DistributedTrainer
        from verse_torch.nn import Linear
        from verse_torch.optim import SGD
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        trainer = DistributedTrainer(
            model, [], [], opt, device="cpu",
            world_size=1, rank=0,
        )
        self.assertEqual(trainer.world_size, 1)
        self.assertEqual(trainer.rank, 0)
        self.assertTrue(trainer.is_main_process)

    def test_distributed_trainer_barrier_noop(self):
        """DistributedTrainer.barrier() 单进程时 no-op。"""
        from verse_torch import DistributedTrainer
        from verse_torch.nn import Linear
        from verse_torch.optim import SGD
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.01)
        trainer = DistributedTrainer(
            model, [], [], opt, device="cpu",
            world_size=1, rank=0,
        )
        # 不应报错
        trainer.barrier()
        trainer.init_process_group()


# ===========================================================================
# 主入口
# ===========================================================================


if __name__ == "__main__":
    unittest.main(verbosity=2)
