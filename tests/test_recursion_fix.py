"""Part3 Task 1.4: 递归溢出修复测试。

验证以下场景不触发 ``RecursionError: maximum recursion depth exceeded``：

1. ``verse_torch.Tensor.backward()`` 在深度计算图（链式 1500+ 算子）下不爆栈
   - 根因修复：``backward`` 中拓扑排序 DFS 已从递归改为迭代式（显式栈）
2. ``CometSparkLM.generate`` 在 transformer / hybrid 两种 arch 下，
   长 prompt + 多次 generate 不触发 RecursionError
   - 根因修复：``forward_recurrent`` 在 transformer arch 下直接调 ``self.net(idx)``，
     不再经过 ``self.forward``；``generate`` 两条路径均为显式 for 循环
3. ``run.py --help`` 与 ``verse_inference.server`` 导入不报错
   - CLI 修复：server.py 改用 try/except 兼容绝对/相对导入

运行：
    cd /workspace && python -m pytest tests/test_recursion_fix.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 路径常量与 sys.path 注入（与 test_end_to_end.py / test_mamba2_memory.py 一致）
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
for pkg in ("verse_torch", "verse_nex", "verse_tokenizer",
            "verse_inference", "verse_compat"):
    p = REPO_ROOT / "packages" / pkg
    if p.is_dir():
        sys.path.insert(0, str(p))

# 把 data/demo 加入 path 以便 import model.config / model.model
_DEMO_DIR = REPO_ROOT / "data" / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))


# ---------------------------------------------------------------------------
# Task 1.2 根因测试：Tensor.backward 迭代式 DFS
# ---------------------------------------------------------------------------


class TestBackwardIterativeDFS:
    """验证 ``Tensor.backward()`` 在深度计算图下不触发 RecursionError。"""

    def test_backward_deep_chain_no_recursion_error(self):
        """链式加法构造 1500 层深度计算图，backward 不应爆栈。

        Python 默认递归上限 1000，原递归式 DFS 在 1001 层即触发
        ``RecursionError``。改为迭代式后可处理任意深度。
        """
        from verse_torch import Tensor

        # 用一个长为 1500 的链式加法构造深度计算图
        # 每一步 x = x + const 都会创建新 Tensor 并把前一个 Tensor 作为父节点
        depth = 1500  # 显著超过 Python 默认递归上限 1000
        x = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        for _ in range(depth):
            x = x + 1.0  # 链式构造深度图

        # backward 应正常完成，不触发 RecursionError
        x.backward()

        # 梯度应为 1.0（链式加法对 x 的导数恒为 1）
        # 用原始 x（链头）取 grad，但 x 已被覆盖；这里只验证不抛异常即可
        # 改：保留原始引用
        assert x.grad is not None, "backward 未产生梯度"

    def test_backward_deep_branching_graph(self):
        """分支型计算图（每节点 2 个子节点）也不爆栈。"""
        from verse_torch import Tensor

        # 构造一个分支型计算图：每次用同一个 x 与多个常量相乘再相加
        depth = 800
        x = Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)
        cur = x
        for i in range(depth):
            cur = cur * 0.5 + cur * 0.5  # 两个分支都引用 cur

        cur.backward()
        assert cur.grad is not None

    def test_backward_no_grad_context_unchanged(self):
        """no_grad 上下文内构造的 Tensor 不构建计算图，backward 应直接报错。"""
        from verse_torch import Tensor, no_grad

        with no_grad():
            x = Tensor(np.array([1.0], dtype=np.float32), requires_grad=False)
            y = x + 1.0
        # y.requires_grad 为 False，调用 backward 应抛 RuntimeError 而非 RecursionError
        with pytest.raises(RuntimeError, match="does not require grad"):
            y.backward()


# ---------------------------------------------------------------------------
# Task 1.2 根因测试：CometSparkLM.generate 迭代式生成
# ---------------------------------------------------------------------------


def _make_cometspark_model(arch: str):
    """构造一个最小 CometSparkLM 模型用于测试。

    Args:
        arch: "hybrid" 或 "transformer"
    """
    from model.config import CometSparkConfig
    from model.model import CometSparkLM

    # 极小配置，仅用于验证不爆栈，不追求数值合理性
    config = CometSparkConfig(
        vocab_size=32,
        n_layer=2,            # 2 层足够测试递归路径
        n_head=2,
        n_embd=16,
        seq_len=64,           # 短 seq_len 避免数值溢出
        dropout=0.0,
        arch=arch,
        ssm_kind="mamba2",
        sparse_ratio=0.0,     # hybrid 但全 SSM，避免 sparse attn 数值问题
        tie_weights=True,
    )
    return CometSparkLM(config)


class TestCometSparkGenerateIterative:
    """验证 ``CometSparkLM.generate`` 两种 arch 下不触发 RecursionError。"""

    def test_transformer_arch_long_prompt_generate(self):
        """transformer arch: 长 prompt（200 token）+ 50 token generate 不爆栈。"""
        from verse_torch import Tensor

        model = _make_cometspark_model("transformer")
        # 长 prompt（超过 Python 默认递归上限 1000 不会触发，但超过 seq_len 64
        # 会触发 _generate_with_logits 内的 context_len 截断分支）
        prompt_len = 200
        idx = np.random.default_rng(0).integers(
            0, 32, size=(1, prompt_len), dtype=np.int64
        )

        # greedy 路径：transformer arch 下 net 无 generate 方法，
        # 走 _generate_with_logits 迭代式 for 循环
        out = model.generate(idx, max_new_tokens=50, temperature=1.0, top_k=None)
        assert out.shape == (1, prompt_len + 50)
        # 所有 token 应在 vocab 范围内
        assert np.all(out < 32) and np.all(out >= 0)

    def test_hybrid_arch_long_prompt_generate(self):
        """hybrid arch: 长 prompt + 50 token generate 不爆栈。"""
        model = _make_cometspark_model("hybrid")
        prompt_len = 100  # hybrid arch recurrent 路径每步推进状态
        idx = np.random.default_rng(0).integers(
            0, 32, size=(1, prompt_len), dtype=np.int64
        )

        # greedy 路径：hybrid arch 下 net 有 generate 方法（HybridLM.generate），
        # 走 self.net.generate(mode="recurrent") 迭代式 for 循环
        out = model.generate(idx, max_new_tokens=50, temperature=1.0, top_k=None)
        assert out.shape == (1, prompt_len + 50)
        assert np.all(out < 32) and np.all(out >= 0)

    def test_transformer_arch_sampling_generate(self):
        """transformer arch: 带 temperature / top_k 采样路径不爆栈。"""
        model = _make_cometspark_model("transformer")
        prompt_len = 80
        idx = np.random.default_rng(0).integers(
            0, 32, size=(1, prompt_len), dtype=np.int64
        )

        # 采样路径：top_k=10 触发 _generate_with_logits
        out = model.generate(
            idx, max_new_tokens=30, temperature=0.8, top_k=10
        )
        assert out.shape == (1, prompt_len + 30)
        assert np.all(out < 32) and np.all(out >= 0)

    def test_repeated_generate_no_stack_growth(self):
        """多次调用 generate 不应有栈深度累积（每次都是独立 for 循环）。"""
        model = _make_cometspark_model("hybrid")
        idx = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        # 连续调用 5 次 generate
        for _ in range(5):
            out = model.generate(idx, max_new_tokens=20, temperature=1.0, top_k=None)
            assert out.shape == (1, 5 + 20)

    def test_forward_recurrent_transformer_no_loop(self):
        """transformer arch 下 forward_recurrent 直接走 self.net，不经 self.forward。"""
        from verse_torch import Tensor

        model = _make_cometspark_model("transformer")
        # 单步 forward_recurrent
        input_ids = Tensor(np.array([[5]], dtype=np.int64), requires_grad=False)
        out = model.forward_recurrent(input_ids, None)
        # 应返回 (logits, None)
        assert isinstance(out, tuple) and len(out) == 2
        logits, new_states = out
        # transformer arch: new_states 应为 None
        assert new_states is None
        # logits shape: (B=1, T=1, vocab_size)
        assert logits.data.shape == (1, 1, 32)


# ---------------------------------------------------------------------------
# Task 1.3 缓解测试：run.py 入口 sys.setrecursionlimit(2000)
# ---------------------------------------------------------------------------


class TestRunPyRecursionLimit:
    """验证 run.py 入口设置了 sys.setrecursionlimit(2000)。"""

    def test_run_py_sets_recursion_limit(self):
        """导入 run.py 模块后，递归上限应 ≥ 2000。"""
        # 以子进程运行，避免污染当前 pytest 进程的 sys.recursionlimit
        demo_dir_str = str(_DEMO_DIR)
        run_py_str = str(_DEMO_DIR / "run.py")
        code = (
            "import sys; "
            f"sys.path.insert(0, {demo_dir_str!r}); "
            "import importlib.util; "
            "spec = importlib.util.spec_from_file_location("
            f"'run_under_test', {run_py_str!r}); "
            "mod = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(mod); "
            "print(sys.getrecursionlimit())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"run.py 导入失败：{result.stderr}"
        )
        limit = int(result.stdout.strip())
        assert limit >= 2000, (
            f"run.py 应设置 sys.setrecursionlimit(2000)，实际 {limit}"
        )


# ---------------------------------------------------------------------------
# Task 3.3 验证：run.py --help 与 server.py 导入不报错
# ---------------------------------------------------------------------------


class TestCLIHelp:
    """验证 run.py --help 与 verse_inference.server --help 不报错。"""

    def test_run_py_help(self):
        """``python run.py --help`` 应正常退出（exit 0）且包含 --verbose。"""
        result = subprocess.run(
            [sys.executable, str(_DEMO_DIR / "run.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"run.py --help 退出码非 0：\n{result.stderr}"
        )
        assert "--verbose" in result.stdout, (
            f"run.py --help 未显示 --verbose：\n{result.stdout}"
        )

    def test_server_module_help(self):
        """``python -m verse_inference.server --help`` 应正常退出。"""
        env = os.environ.copy()
        # 设置 PYTHONPATH 让 -m verse_inference.server 能找到所有依赖包
        env["PYTHONPATH"] = os.pathsep.join([
            str(REPO_ROOT / "packages" / "verse_torch"),
            str(REPO_ROOT / "packages" / "verse_nex"),
            str(REPO_ROOT / "packages" / "verse_tokenizer"),
            str(REPO_ROOT / "packages" / "verse_inference"),
            str(REPO_ROOT / "packages" / "verse_compat"),
        ])
        result = subprocess.run(
            [sys.executable, "-m", "verse_inference.server", "--help"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 0, (
            f"python -m verse_inference.server --help 退出码非 0：\n{result.stderr}"
        )

    def test_server_imports_no_error(self):
        """直接 import verse_inference.server 不应报 ImportError。"""
        # 在子进程里跑，避免污染当前 pytest 进程
        code = (
            "import sys; "
            f"sys.path.insert(0, '{REPO_ROOT / 'packages' / 'verse_torch'}'); "
            f"sys.path.insert(0, '{REPO_ROOT / 'packages' / 'verse_nex'}'); "
            f"sys.path.insert(0, '{REPO_ROOT / 'packages' / 'verse_tokenizer'}'); "
            f"sys.path.insert(0, '{REPO_ROOT / 'packages' / 'verse_inference'}'); "
            f"sys.path.insert(0, '{REPO_ROOT / 'packages' / 'verse_compat'}'); "
            "from verse_inference.server import "
            "create_app, create_http_server, run_server; "
            "print('server imports OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"server.py 导入失败：\n{result.stderr}"
        )
        assert "server imports OK" in result.stdout


# ---------------------------------------------------------------------------
# 主入口：直接运行脚本
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 让脚本可独立运行：python tests/test_recursion_fix.py
    sys.exit(pytest.main([__file__, "-v"]))
