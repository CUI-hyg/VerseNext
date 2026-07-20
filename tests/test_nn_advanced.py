"""verse_torch.nn 多层神经网络组件单元测试（阶段 1 / Task 1.5）。

覆盖：
1. SwiGLUMLP forward shape (B, T, d) 正确
2. SwiGLUMLP 有限差分梯度检查（数值梯度 vs autograd 梯度，相对误差 < 1e-3）
3. GQASelfAttention forward shape (B, T, d) 正确
4. GQASelfAttention KV cache：分批前向应等价于一次性前向的后 T2 步
5. TransformerLM forward shape (B, T, vocab) 正确
6. TransformerLM 参数量计算正确（手动数一遍）
7. tie_weights=True 时 tok_emb 与 head 权重共享（修改一个影响另一个）

运行方式：
    python3 -m pytest tests/test_nn_advanced.py -v
    python3 tests/test_nn_advanced.py
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
from verse_torch.nn import (
    SwiGLUMLP,
    GQASelfAttention,
    TransformerBlock,
    TransformerLM,
)


def _set_seed(seed: int = 42):
    np.random.seed(seed)


def _count_parameters(model) -> int:
    """统计模型参数量（按 Tensor 对象 id 去重，处理 tie_weights 共享）。"""
    seen = set()
    count = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            count += p.data.size
    return count


class TestSwiGLUMLP(unittest.TestCase):
    """Task 1.5 SubTask 1.5.1: SwiGLUMLP forward shape + 有限差分梯度。"""

    def test_forward_shape(self):
        """forward 输入 (B, T, d) 输出应同形状。"""
        _set_seed(0)
        B, T, d = 2, 3, 16
        mlp = SwiGLUMLP(d, dropout=0.0)
        mlp.eval()
        x = Tensor(np.random.randn(B, T, d).astype(np.float32), requires_grad=True)
        y = mlp(x)
        self.assertEqual(y.shape, (B, T, d))
        # hidden 维度对齐 64
        self.assertEqual(mlp.hidden % 64, 0)
        # 三个 Linear（bias=False）
        self.assertEqual(mlp.w_gate.weight.shape, (mlp.hidden, d))
        self.assertEqual(mlp.w_up.weight.shape, (mlp.hidden, d))
        self.assertEqual(mlp.w_down.weight.shape, (d, mlp.hidden))
        # bias 必须为 None
        self.assertIsNone(mlp.w_gate.bias)
        self.assertIsNone(mlp.w_up.bias)
        self.assertIsNone(mlp.w_down.bias)

    def test_hidden_dim_formula(self):
        """hidden = ((4*d*2/3 + align-1) // align) * align，对齐到 64。"""
        for d in [16, 32, 64, 96, 128, 256]:
            mlp = SwiGLUMLP(d, dropout=0.0, hidden_multiple=4, align=64)
            expected = int((4 * d * 2 / 3 + 64 - 1) // 64) * 64
            self.assertEqual(mlp.hidden, expected)
            self.assertEqual(mlp.hidden % 64, 0)

    def test_gradient_check(self):
        """有限差分梯度检查：autograd vs 数值梯度，相对误差 < 1e-3。"""
        _set_seed(123)
        d = 8
        mlp = SwiGLUMLP(d, dropout=0.0)
        mlp.eval()  # 关闭 dropout
        B, T = 2, 3
        # 用 float64 计算以避免 float32 精度问题
        # （模型权重是 float32，但输入用 float64 后整个计算图会升级到 float64）
        x_data = (np.random.randn(B, T, d) * 0.5).astype(np.float64)
        x = Tensor(x_data, requires_grad=True)

        # autograd 前向 + 反向（取 sum 作为标量损失）
        y = mlp(x).sum()
        y.backward()
        grad_autograd = x.grad.copy()

        # 数值梯度（中心差分）
        eps = 1e-4
        grad_num = np.zeros_like(x_data, dtype=np.float64)
        x_arr = x.data  # float64，原地修改
        for i in range(x_arr.size):
            orig = float(x_arr.flat[i])
            x_arr.flat[i] = orig + eps
            y_plus = float(mlp(x).sum().data)
            x_arr.flat[i] = orig - eps
            y_minus = float(mlp(x).sum().data)
            x_arr.flat[i] = orig
            grad_num.flat[i] = (y_plus - y_minus) / (2 * eps)

        # 相对误差
        grad_auto = grad_autograd.astype(np.float64)
        denom = np.maximum(np.abs(grad_auto) + np.abs(grad_num), 1e-8)
        rel_err = np.abs(grad_auto - grad_num) / denom
        max_err = float(rel_err.max())
        self.assertLess(max_err, 1e-3, f"梯度检查失败，最大相对误差 {max_err}")


class TestGQASelfAttention(unittest.TestCase):
    """Task 1.5 SubTask 1.5.2: GQASelfAttention forward shape + KV cache。"""

    def test_forward_shape_mha(self):
        """标准 MHA（n_kv_head=None）：forward 输出 (B, T, d)。"""
        _set_seed(0)
        B, T, d, n_head = 2, 5, 16, 4
        attn = GQASelfAttention(d, n_head)
        attn.eval()
        x = Tensor(np.random.randn(B, T, d).astype(np.float32), requires_grad=True)
        out, kv_cache = attn(x)
        self.assertEqual(out.shape, (B, T, d))
        # cache 是 (k, v)，每个 shape (B, T, n_kv_head, head_dim)
        k_cache, v_cache = kv_cache
        self.assertEqual(k_cache.shape, (B, T, n_head, d // n_head))
        self.assertEqual(v_cache.shape, (B, T, n_head, d // n_head))

    def test_forward_shape_gqa(self):
        """GQA（n_kv_head < n_head）：forward 输出 (B, T, d)。"""
        _set_seed(0)
        B, T, d, n_head, n_kv_head = 2, 5, 16, 4, 2
        attn = GQASelfAttention(d, n_head, n_kv_head=n_kv_head)
        attn.eval()
        self.assertEqual(attn.n_rep, 2)
        x = Tensor(np.random.randn(B, T, d).astype(np.float32), requires_grad=True)
        out, kv_cache = attn(x)
        self.assertEqual(out.shape, (B, T, d))
        # cache 是 (k, v)，每个 shape (B, T, n_kv_head, head_dim)
        k_cache, v_cache = kv_cache
        self.assertEqual(k_cache.shape, (B, T, n_kv_head, d // n_head))
        self.assertEqual(v_cache.shape, (B, T, n_kv_head, d // n_head))

    def test_kv_cache_equivalence(self):
        """KV cache 等价性：分批前向的后 T2 步 ≈ 一次性前向的后 T2 步。"""
        _set_seed(7)
        B, T1, T2, d, n_head = 2, 4, 3, 16, 4
        attn = GQASelfAttention(d, n_head, dropout=0.0)
        attn.eval()

        x_full = Tensor(np.random.randn(B, T1 + T2, d).astype(np.float32), requires_grad=False)
        x1 = Tensor(x_full.data[:, :T1].copy(), requires_grad=False)
        x2 = Tensor(x_full.data[:, T1:].copy(), requires_grad=False)

        # 一次性前向
        out_full, _ = attn(x_full)
        # 分批前向：第一次 + cache，第二次用 cache
        out1, cache1 = attn(x1)
        out2, cache2 = attn(x2, kv_cache=cache1)

        # out2 应与 out_full[:, T1:, :] 数值一致
        out_full_tail = out_full.data[:, T1:, :]
        out2_data = out2.data
        np.testing.assert_allclose(
            out2_data, out_full_tail, atol=1e-4, rtol=1e-4,
            err_msg="KV cache 分批前向与一次性前向不一致",
        )

        # 第二次后的 cache 长度应等于 T1 + T2
        k2, v2 = cache2
        self.assertEqual(k2.shape[1], T1 + T2)
        self.assertEqual(v2.shape[1], T1 + T2)

    def test_kv_cache_equivalence_gqa(self):
        """GQA 下 KV cache 等价性。"""
        _set_seed(11)
        B, T1, T2, d, n_head, n_kv = 2, 4, 3, 16, 4, 2
        attn = GQASelfAttention(d, n_head, n_kv_head=n_kv, dropout=0.0)
        attn.eval()

        x_full = Tensor(np.random.randn(B, T1 + T2, d).astype(np.float32), requires_grad=False)
        x1 = Tensor(x_full.data[:, :T1].copy(), requires_grad=False)
        x2 = Tensor(x_full.data[:, T1:].copy(), requires_grad=False)

        out_full, _ = attn(x_full)
        out1, cache1 = attn(x1)
        out2, cache2 = attn(x2, kv_cache=cache1)

        np.testing.assert_allclose(
            out2.data, out_full.data[:, T1:, :], atol=1e-4, rtol=1e-4,
            err_msg="GQA KV cache 分批前向与一次性前向不一致",
        )

    def test_causal_mask(self):
        """causal mask: 每个 query 只能 attend 到自身及之前的 key。"""
        _set_seed(0)
        B, T, d, n_head = 1, 4, 8, 2
        attn = GQASelfAttention(d, n_head, dropout=0.0)
        attn.eval()
        # 构造 x：每个时间步是一个独热向量
        x_data = np.eye(T, dtype=np.float32)[None, :, :]  # (1, T, T)
        # 投影到 d 维：用线性变换让 x 通过 wq 等
        # 简单做法：直接构造 d 维输入，每个时间步是基向量
        x_data = np.zeros((B, T, d), dtype=np.float32)
        for t in range(T):
            x_data[0, t, t % d] = 1.0
        x = Tensor(x_data, requires_grad=False)
        out, _ = attn(x)
        # 由于 causal mask，query t 不应受 t' > t 的 key 影响。
        # 我们用扰动法验证：把 t' > t 的 v 改成大值，out[:, t] 不应变。
        # 但因为 v 通过 wv 投影，简单做法是验证 out[:, 0] 仅依赖 x[:, 0]。
        # 这里我们采用另一种验证：修改 x[:, t+1:] 不影响 out[:, :t+1]
        x2_data = x_data.copy()
        x2_data[0, 2:, :] += 100.0  # 大幅扰动 t=2,3
        x2 = Tensor(x2_data, requires_grad=False)
        out2, _ = attn(x2)
        # out[:, 0] 和 out[:, 1] 应不变（causal: 只 attend 到 t<=1）
        np.testing.assert_allclose(out.data[:, :2, :], out2.data[:, :2, :], atol=1e-4,
                                    err_msg="causal mask 失效：未来 token 影响了历史输出")
        # out[:, 2:] 应变化
        self.assertFalse(np.allclose(out.data[:, 2:, :], out2.data[:, 2:, :], atol=1e-4),
                         "causal mask 失效：扰动未来 token 应改变后缀输出")


class TestTransformerLM(unittest.TestCase):
    """Task 1.5 SubTask 1.5.3: TransformerLM forward shape + 参数量 + tie_weights。"""

    def test_forward_shape(self):
        """TransformerLM forward: (B, T) int → (B, T, vocab)。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 100, 2, 4, 16
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0)
        model.eval()
        B, T = 3, 7
        idx = np.random.randint(0, vocab, size=(B, T)).astype(np.int64)
        logits = model(idx)
        self.assertEqual(logits.shape, (B, T, vocab))

    def test_parameter_count_tied(self):
        """tie_weights=True 时参数量正确（head 与 tok_emb 共享，不重复计）。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 100, 2, 4, 16
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0, tie_weights=True)
        # 手动计算参数量
        head_dim = n_embd // n_head
        hidden = int((4 * n_embd * 2 / 3 + 64 - 1) // 64) * 64
        # tok_emb
        p_tok = vocab * n_embd
        # 每个 block
        p_block = (
            2 * n_embd  # norm1 + norm2 (RMSNorm weight)
            + 4 * (n_embd * n_embd)  # wq + wk + wv + proj
            + 3 * (n_embd * hidden)  # w_gate + w_up + w_down
        )
        p_blocks = n_layer * p_block
        # final norm
        p_norm = n_embd
        # head 与 tok_emb 共享，不重复计算
        expected = p_tok + p_blocks + p_norm
        actual = _count_parameters(model)
        self.assertEqual(actual, expected,
                         f"参数量不匹配：期望 {expected}，实际 {actual}")

    def test_parameter_count_untied(self):
        """tie_weights=False 时参数量正确（head 独立）。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 100, 2, 4, 16
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0, tie_weights=False)
        head_dim = n_embd // n_head
        hidden = int((4 * n_embd * 2 / 3 + 64 - 1) // 64) * 64
        p_tok = vocab * n_embd
        p_block = (
            2 * n_embd
            + 4 * (n_embd * n_embd)
            + 3 * (n_embd * hidden)
        )
        p_blocks = n_layer * p_block
        p_norm = n_embd
        p_head = vocab * n_embd  # head 独立
        expected = p_tok + p_blocks + p_norm + p_head
        actual = _count_parameters(model)
        self.assertEqual(actual, expected,
                         f"参数量不匹配：期望 {expected}，实际 {actual}")

    def test_tie_weights_shared(self):
        """tie_weights=True 时，修改 tok_emb.weight.data 应同步影响 head.weight.data。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 50, 1, 2, 8
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0, tie_weights=True)
        # 验证：head.weight 与 tok_emb.weight 是同一个 Tensor 对象
        self.assertIs(model.head.weight, model.tok_emb.weight,
                      "tie_weights=True 时 head.weight 应与 tok_emb.weight 是同一对象")
        # 修改 tok_emb.weight.data，head.weight.data 应同步变化
        original = model.tok_emb.weight.data.copy()
        new_data = np.ones_like(original) * 0.123
        model.tok_emb.weight.data = new_data
        np.testing.assert_array_equal(model.head.weight.data, new_data,
                                       err_msg="修改 tok_emb.weight.data 未同步到 head.weight.data")
        # 也验证从 head 修改反向同步
        new_data2 = np.ones_like(original) * 0.456
        model.head.weight.data = new_data2
        np.testing.assert_array_equal(model.tok_emb.weight.data, new_data2,
                                       err_msg="修改 head.weight.data 未同步到 tok_emb.weight.data")

    def test_tie_weights_off_independent(self):
        """tie_weights=False 时，head 与 tok_emb 独立。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 50, 1, 2, 8
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0, tie_weights=False)
        self.assertIsNot(model.head.weight, model.tok_emb.weight,
                         "tie_weights=False 时 head.weight 应与 tok_emb.weight 独立")
        # 保存 head 与 tok_emb 各自的原始值
        head_original = model.head.weight.data.copy()
        tok_original = model.tok_emb.weight.data.copy()
        # 修改 tok_emb，head 不应变
        model.tok_emb.weight.data = np.ones_like(tok_original) * 0.789
        np.testing.assert_array_equal(model.head.weight.data, head_original,
                                       err_msg="tie_weights=False 时修改 tok_emb 不应影响 head")
        # 修改 head，tok_emb 不应变
        model.head.weight.data = np.ones_like(head_original) * 0.456
        np.testing.assert_array_equal(model.tok_emb.weight.data, np.ones_like(tok_original) * 0.789,
                                       err_msg="tie_weights=False 时修改 head 不应影响 tok_emb")

    def test_residual_scaling(self):
        """残差分支（attn.proj, mlp.w_down）权重应被 1/sqrt(2*n_layer) 缩放。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 100, 4, 4, 16
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0, tie_weights=True)
        # 由于残差缩放在 _init_weights 中应用，且 normal_(std=0.02) 先应用再缩放，
        # 验证：权重 std 应 ≈ 0.02 * 1/sqrt(2*n_layer)
        # 由于样本 std 有随机波动，用 10% 相对容差
        expected_std = 0.02 / np.sqrt(2 * n_layer)
        for block in model.blocks:
            proj_std = float(np.std(block.attn.proj.weight.data))
            wdown_std = float(np.std(block.mlp.w_down.weight.data))
            self.assertAlmostEqual(proj_std, expected_std, delta=expected_std * 0.15,
                                    msg=f"attn.proj 权重 std 异常: {proj_std} vs {expected_std}")
            self.assertAlmostEqual(wdown_std, expected_std, delta=expected_std * 0.15,
                                    msg=f"mlp.w_down 权重 std 异常: {wdown_std} vs {expected_std}")

    def test_forward_grad_flow(self):
        """TransformerLM 反向梯度能流到 tok_emb。"""
        _set_seed(0)
        vocab, n_layer, n_head, n_embd = 50, 2, 2, 8
        model = TransformerLM(vocab, n_layer, n_head, n_embd, dropout=0.0)
        model.eval()
        B, T = 2, 4
        idx = np.random.randint(0, vocab, size=(B, T)).astype(np.int64)
        logits = model(idx)
        loss = logits.sum()
        loss.backward()
        # tok_emb.weight 应有梯度
        self.assertIsNotNone(model.tok_emb.weight.grad,
                             "tok_emb.weight.grad 为 None，梯度未回流")
        self.assertEqual(model.tok_emb.weight.grad.shape, model.tok_emb.weight.shape)
        # head.weight 与 tok_emb.weight 共享，grad 也应一致（同一对象）
        if model.tie_weights:
            self.assertIs(model.head.weight, model.tok_emb.weight)


class TestTransformerBlock(unittest.TestCase):
    """TransformerBlock 集成测试。"""

    def test_forward_shape(self):
        """TransformerBlock forward 输入 (B, T, d) 输出同形状。"""
        _set_seed(0)
        B, T, d, n_head = 2, 5, 16, 4
        block = TransformerBlock(d, n_head, dropout=0.0)
        block.eval()
        x = Tensor(np.random.randn(B, T, d).astype(np.float32), requires_grad=True)
        out, kv = block(x)
        self.assertEqual(out.shape, (B, T, d))

    def test_forward_with_kv_cache(self):
        """TransformerBlock 支持 kv_cache 输入。"""
        _set_seed(0)
        B, T1, T2, d, n_head = 2, 3, 2, 16, 4
        block = TransformerBlock(d, n_head, dropout=0.0)
        block.eval()
        x1 = Tensor(np.random.randn(B, T1, d).astype(np.float32), requires_grad=False)
        x2 = Tensor(np.random.randn(B, T2, d).astype(np.float32), requires_grad=False)
        out1, cache1 = block(x1)
        out2, cache2 = block(x2, kv_cache=cache1)
        self.assertEqual(out1.shape, (B, T1, d))
        self.assertEqual(out2.shape, (B, T2, d))
        k2, v2 = cache2
        self.assertEqual(k2.shape[1], T1 + T2)


def run_all():
    """允许直接 python tests/test_nn_advanced.py 运行。"""
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all())
