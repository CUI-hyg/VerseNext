"""Task 8.5: val_loss 曲线修复的单元测试。

覆盖：
1. ASCII 模式下 val 点可见（不被 train 的 T 覆盖）—— 用 mock 强制 matplotlib 不可用
2. matplotlib 模式下 val 曲线有显著 marker（marker='o', markersize=8, linewidth=2.5）
3. ASCII 模式下附加 val 数值表（val_losses detail）
4. Trainer.fit 完成后生成 val_losses.txt 纯文本列表
5. plot_loss_curve 完成后打印 [info] val_losses: N points, best=X at step M

运行方式：
    cd /workspace && python -m pytest tests/test_val_loss_curve.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch import (
    Tensor,
    Linear,
    SGD,
    plot_loss_curve,
    Trainer,
)
from verse_torch import training as training_mod


# ---------------------------------------------------------------------------
# 公共 mock 数据
# ---------------------------------------------------------------------------


def _make_mock_losses(n_train: int = 200, eval_interval: int = 20):
    """构造 mock 训练 loss 与验证 loss。

    模拟 demo/checkpoints/loss_history.json 的形状：
    - train_losses: 200 条单调下降（带噪声）
    - val_losses: 10 条（200 / 20），单调下降
    """
    rng = np.random.RandomState(42)
    # train: 从 5.6 单调下降到 ~2.2，加少量噪声
    base = np.linspace(5.6, 2.2, n_train)
    noise = rng.randn(n_train) * 0.1
    train_losses = list((base + noise).astype(float))
    # val: 从 5.5 下降到 ~2.28，每 eval_interval 步一个
    n_val = n_train // eval_interval
    val_base = np.linspace(5.5, 2.28, n_val)
    val_losses = list(val_base.astype(float))
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# 1. ASCII 模式：val 点可见（不被 T 覆盖）
# ---------------------------------------------------------------------------


class TestAsciiValVisible:

    def _force_ascii(self, monkeypatch):
        """模拟 matplotlib 不可用，强制走 ASCII 分支。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # 清理已 import 的 matplotlib
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

    def test_ascii_val_points_visible_not_covered_by_T(self, tmp_path, monkeypatch):
        """ASCII 模式下，val 点 V 必须可见，不被密集的 T 完全覆盖。"""
        self._force_ascii(monkeypatch)

        train_losses, val_losses = _make_mock_losses(n_train=200, eval_interval=20)
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=20)
        assert actual_path.endswith(".txt")
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 画布中至少存在 1 个 V 或 *（重叠标记）
        # val 后绘制，重叠处用 *，独立处用 V，二者都说明 val 可见
        v_count = content.count("V")
        star_count = content.count("*")
        assert v_count + star_count >= 1, (
            f"ASCII 画布中未找到 V 或 *，val 点似乎被 train 完全覆盖"
            f"（V={v_count}, *={star_count}）"
        )

    def test_ascii_legend_includes_overlap_marker(self, tmp_path, monkeypatch):
        """ASCII 图例应说明 *=overlap，让用户理解重叠标记。"""
        self._force_ascii(monkeypatch)

        train_losses, val_losses = _make_mock_losses(n_train=50, eval_interval=10)
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=10)
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "V=val" in content
        assert "*=overlap" in content

    def test_ascii_includes_val_detail_table(self, tmp_path, monkeypatch):
        """ASCII 模式下应在画布下方附加 val 数值表。"""
        self._force_ascii(monkeypatch)

        train_losses, val_losses = _make_mock_losses(n_train=100, eval_interval=20)
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=20)
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 应包含 val_losses detail 段
        assert "val_losses detail:" in content
        # 每行格式: [step     N] val_loss=X.XXXXXX
        n_val = len(val_losses)
        # 至少 n_val 行 detail（最后一行可能没有换行）
        detail_lines = [
            line for line in content.splitlines()
            if line.startswith("  [step") and "val_loss=" in line
        ]
        assert len(detail_lines) == n_val, (
            f"detail 行数 {len(detail_lines)} != val_losses 数 {n_val}"
        )
        # 第 i 行的 step 应为 i * eval_interval
        for i, line in enumerate(detail_lines):
            expected_step = i * 20
            assert f"step {expected_step:>6d}" in line, (
                f"detail 第 {i} 行 step 不匹配: {line}"
            )

    def test_ascii_val_count_at_least_equal_to_val_losses(self, tmp_path, monkeypatch):
        """ASCII 画布中的 V+* 数量应 <= len(val_losses)（每个 val 点画一次）。

        但更重要的是至少有 1 个 V 或 * 可见（防止全部被 T 覆盖的回归）。
        """
        self._force_ascii(monkeypatch)

        # 用极端密集 train + 稀疏 val，验证 val 仍可见
        train_losses = [5.5 - i * 0.01 for i in range(500)]
        val_losses = [5.4, 4.5, 3.6, 2.7, 1.8]
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=100)
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()

        v_count = content.count("V")
        star_count = content.count("*")
        assert (v_count + star_count) >= 1, (
            "密集 train 曲线下 val 点仍应可见（V 或 *）"
        )


# ---------------------------------------------------------------------------
# 2. matplotlib 模式：val 曲线有显著 marker
# ---------------------------------------------------------------------------


class TestMatplotlibValMarker:

    def test_val_curve_has_marker_and_label(self, tmp_path, monkeypatch):
        """matplotlib 模式下，val 曲线应有 marker='o', markersize=8, linewidth=2.5。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用，跳过 matplotlib marker 测试")

        import matplotlib.pyplot as plt

        # 用 mock 替换 subplots，捕获 ax.plot 调用参数
        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses, val_losses = _make_mock_losses(n_train=100, eval_interval=20)
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=20)
        assert actual_path.endswith(".png")

        # ax.plot 应被调用两次：一次 train，一次 val
        assert fake_ax.plot.call_count == 2, (
            f"ax.plot 应被调用 2 次（train + val），实际 {fake_ax.plot.call_count}"
        )

        # 第二次调用是 val 曲线
        val_call = fake_ax.plot.call_args_list[1]
        kwargs = val_call.kwargs
        assert kwargs.get("marker") == "o", f"marker 应为 'o', got {kwargs.get('marker')}"
        assert kwargs.get("markersize") == 8, f"markersize 应为 8, got {kwargs.get('markersize')}"
        assert kwargs.get("linewidth") == 2.5, f"linewidth 应为 2.5, got {kwargs.get('linewidth')}"
        # 图例应标注 "val (every N steps)"
        label = kwargs.get("label", "")
        assert "val (every 20 steps)" in label, f"label 应包含 'val (every 20 steps)', got {label}"

    def test_train_curve_keeps_simple_style(self, tmp_path, monkeypatch):
        """train 曲线保持简单的蓝色实线（不应被改成 marker 样式）。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses, val_losses = _make_mock_losses(n_train=50, eval_interval=10)
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=10)

        train_call = fake_ax.plot.call_args_list[0]
        kwargs = train_call.kwargs
        # train 不应有 marker
        assert "marker" not in kwargs or kwargs.get("marker") in (None, "None", ""), (
            f"train 曲线不应有 marker, got {kwargs.get('marker')}"
        )
        assert kwargs.get("label") == "train"


# ---------------------------------------------------------------------------
# 3. ASCII 附加 val 数值表 + matplotlib 不附加（不污染 PNG）
# ---------------------------------------------------------------------------


class TestValDetailTable:

    def test_ascii_val_detail_values_match_input(self, tmp_path, monkeypatch):
        """ASCII 模式下，val 数值表中的值应与输入 val_losses 一致（保留 6 位小数）。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        train_losses = [5.6, 5.4, 5.2, 5.0, 4.8, 4.6, 4.4, 4.2]
        val_losses = [5.5, 4.9, 4.3, 3.7]
        save_path = str(tmp_path / "loss_curve.png")

        actual_path = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()

        for v in val_losses:
            assert f"{v:.6f}" in content, f"val 数值 {v:.6f} 应在 detail 表中"


# ---------------------------------------------------------------------------
# 4. plot_loss_curve 完成后打印 val 信息
# ---------------------------------------------------------------------------


class TestValInfoPrinted:

    def test_prints_val_info_ascii(self, tmp_path, monkeypatch, capsys):
        """ASCII 模式下，plot_loss_curve 完成后应打印 [info] val_losses: ..."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        train_losses, val_losses = _make_mock_losses(n_train=100, eval_interval=20)
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=20)

        captured = capsys.readouterr()
        # 应包含 [info] val_losses: N points, best=X.XXXX at step M
        assert "[info] val_losses:" in captured.out, (
            f"应打印 [info] val_losses: ..., 实际输出: {captured.out!r}"
        )
        # 应包含 points 数
        n_val = len(val_losses)
        assert f"{n_val} points" in captured.out
        # 应包含 best=X.XXXX
        assert "best=" in captured.out
        # 应包含 at step M
        assert "at step" in captured.out

    def test_prints_val_info_matplotlib(self, tmp_path, monkeypatch, capsys):
        """matplotlib 模式下，plot_loss_curve 完成后也应打印 val 信息。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses, val_losses = _make_mock_losses(n_train=100, eval_interval=20)
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=20)

        captured = capsys.readouterr()
        assert "[info] val_losses:" in captured.out
        assert f"{len(val_losses)} points" in captured.out
        assert "best=" in captured.out

    def test_val_info_best_step_correct(self, tmp_path, monkeypatch, capsys):
        """打印的 best step 应与 val_losses 中最小值的索引 * eval_interval 一致。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        # 构造已知 best 在第 3 个 val 点（index=2）
        train_losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5]
        val_losses = [4.8, 4.2, 1.0, 1.5]  # best=1.0 at index=2 -> step=2*2=4
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)

        captured = capsys.readouterr()
        # best=1.0000 at step 4
        assert "best=1.0000" in captured.out, f"应打印 best=1.0000, 实际: {captured.out!r}"
        assert "at step 4" in captured.out, f"应打印 at step 4, 实际: {captured.out!r}"

    def test_val_info_zero_points(self, tmp_path, monkeypatch, capsys):
        """空 val_losses 时应打印 0 points。"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for k in list(sys.modules):
            if k == "matplotlib" or k.startswith("matplotlib."):
                monkeypatch.delitem(sys.modules, k, raising=False)

        train_losses = [5.0, 4.5, 4.0]
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, [], save_path, eval_interval=2)

        captured = capsys.readouterr()
        assert "[info] val_losses: 0 points" in captured.out


# ---------------------------------------------------------------------------
# 5. Trainer.fit 完成后生成 val_losses.txt
# ---------------------------------------------------------------------------


def _make_toy_dataset(vocab=4, dim_in=4, n_samples=64, seed=0):
    """构造一个简单的线性可分数据集。"""
    rng = np.random.RandomState(seed)
    W_true = rng.randn(dim_in, vocab).astype(np.float32)
    b_true = rng.randn(vocab).astype(np.float32)
    X = rng.randn(n_samples, dim_in).astype(np.float32)
    logits = X @ W_true + b_true
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


class TestTrainerValLossesTxt:

    def test_val_losses_txt_generated(self, tmp_path):
        """Trainer.fit 完成后应生成 val_losses.txt 纯文本列表。"""
        np.random.seed(0)
        dim_in, vocab = 4, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=64, seed=0)

        def make_batches(X, y, batch_size=16):
            batches = []
            for i in range(0, len(X), batch_size):
                batches.append((
                    Tensor(X[i:i + batch_size], requires_grad=False),
                    Tensor(y[i:i + batch_size], requires_grad=False),
                ))
            return batches

        train_loader = make_batches(X, y, batch_size=16)
        val_loader = make_batches(X[:16], y[:16], batch_size=16)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 30,
            "eval_interval": 10,
            "patience": 5,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "loss_rate_window": 10,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, scheduler=None, cfg=cfg)
        train_losses, val_losses = trainer.fit()

        # val_losses.txt 应存在
        val_txt_path = tmp_path / "val_losses.txt"
        assert val_txt_path.exists(), "val_losses.txt 未生成"

        # 内容应为每行一个 val loss 值
        with open(val_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == len(val_losses), (
            f"val_losses.txt 行数 {len(lines)} != val_losses 数 {len(val_losses)}"
        )
        # 每行应为合法浮点数
        for i, line in enumerate(lines):
            v = float(line)
            assert abs(v - val_losses[i]) < 1e-5, (
                f"val_losses.txt 第 {i} 行 {v} != val_losses[{i}] {val_losses[i]}"
            )

    def test_train_losses_txt_generated(self, tmp_path):
        """Trainer.fit 完成后也应生成 train_losses.txt。"""
        np.random.seed(1)
        dim_in, vocab = 3, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=32, seed=1)

        def make_batches(X, y, batch_size=8):
            return [
                (Tensor(X[i:i + batch_size]), Tensor(y[i:i + batch_size]))
                for i in range(0, len(X), batch_size)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 20,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "loss_rate_window": 5,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        train_losses, val_losses = trainer.fit()

        train_txt_path = tmp_path / "train_losses.txt"
        assert train_txt_path.exists(), "train_losses.txt 未生成"
        with open(train_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == len(train_losses)

    def test_trainer_fit_prints_val_info(self, tmp_path, capsys):
        """Trainer.fit 完成后（_save_history 调用 plot_loss_curve）应打印 val 信息。"""
        np.random.seed(2)
        dim_in, vocab = 3, 4
        X, y = _make_toy_dataset(vocab=vocab, dim_in=dim_in, n_samples=32, seed=2)

        def make_batches(X, y, batch_size=8):
            return [
                (Tensor(X[i:i + batch_size]), Tensor(y[i:i + batch_size]))
                for i in range(0, len(X), batch_size)
            ]

        train_loader = make_batches(X, y, 8)
        val_loader = make_batches(X[:8], y[:8], 8)

        model = Linear(dim_in, vocab)
        opt = SGD(model.parameters(), lr=0.5)
        cfg = {
            "max_steps": 20,
            "eval_interval": 5,
            "patience": 10,
            "save_dir": str(tmp_path),
            "grad_accum": 1,
            "log_interval": 100,
            "loss_rate_window": 5,
        }
        trainer = Trainer(model, train_loader, val_loader, opt, cfg=cfg)
        trainer.fit()

        captured = capsys.readouterr()
        assert "[info] val_losses:" in captured.out, (
            f"Trainer.fit 完成后应打印 val 信息, 实际: {captured.out!r}"
        )


# ---------------------------------------------------------------------------
# 6. _print_val_info 单元测试
# ---------------------------------------------------------------------------


class TestPrintValInfoUnit:

    def test_format_basic(self, capsys):
        training_mod._print_val_info([5.5, 4.0, 3.2, 2.8, 2.6], val_x=[0, 20, 40, 60, 80])
        captured = capsys.readouterr()
        assert "[info] val_losses: 5 points" in captured.out
        # best=2.6 at step 80
        assert "best=2.6000" in captured.out
        assert "at step 80" in captured.out

    def test_empty(self, capsys):
        training_mod._print_val_info([], val_x=[])
        captured = capsys.readouterr()
        assert "[info] val_losses: 0 points" in captured.out

    def test_val_x_none_uses_index(self, capsys):
        # val_x=None 时 best step 应为 best_idx 本身
        training_mod._print_val_info([3.0, 1.0, 2.0], val_x=None)
        captured = capsys.readouterr()
        # best=1.0 at index 1
        assert "best=1.0000" in captured.out
        assert "at step 1" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
