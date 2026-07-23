"""Part4K2.6 Task 1: val_loss 曲线画反问题修复的单元测试。

覆盖场景：
1. ParallelTrainer per-chunk 场景：train/val 等长 → val_x 1:1 对齐（不堆叠）
2. per-step 场景：val_x 不截断到最后一个 train step
3. eval_interval=1：等长 → val_x = train_x
4. 空 val_losses：不报错
5. 空 train_losses：不报错
6. val_x 超出 train 范围：不截断
7. ASCII 图有 y 轴标签（y_max / y_mid / y_min）
8. matplotlib 图 y 轴标签包含 "lower is better"
9. matplotlib 图标注 best_val_loss 位置（annotate）

运行方式：
    cd /workspace && python -m pytest tests/test_val_loss_curve_fix.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 让 tests/ 目录能 import verse_torch
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))

from verse_torch.training import _compute_loss_x, plot_loss_curve


# ---------------------------------------------------------------------------
# 辅助：强制 ASCII 模式（matplotlib 不可用）
# ---------------------------------------------------------------------------


def _force_ascii(monkeypatch):
    """模拟 matplotlib 不可用，强制走 ASCII 分支。"""
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


# ---------------------------------------------------------------------------
# 1. _compute_loss_x 单元测试
# ---------------------------------------------------------------------------


class TestComputeLossX:
    """_compute_loss_x 辅助函数的单元测试。"""

    def test_parallel_trainer_per_chunk(self):
        """ParallelTrainer 场景：train/val 等长 → val_x 1:1 对齐。

        旧 bug：val_x = [min(0*50,3), min(1*50,3), min(2*50,3), min(3*50,3)]
               = [0, 3, 3, 3]（3 个点堆叠在 x=3，曲线画反）
        修复后：val_x = [0, 1, 2, 3]（1:1 对齐）
        """
        train_losses = [5, 3, 2, 1]
        val_losses = [4, 3, 2, 2]
        train_x, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=50)

        assert val_x == [0, 1, 2, 3], (
            f"ParallelTrainer per-chunk 场景 val_x 应为 [0,1,2,3], got {val_x}"
        )
        assert train_x == [0, 1, 2, 3]

    def test_per_step_no_truncation(self):
        """per-step 场景：val_x 不截断到最后一个 train step。

        100 步训练，eval_interval=10，10 个 val 点。
        旧 bug：val_x = [0,10,20,...,90]（此例恰好不截断，但若 11 个点则截断）
        修复后：val_x = [0,10,20,...,90]（不截断）
        """
        train_losses = list(range(100))
        val_losses = list(range(10))
        train_x, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=10)

        assert val_x == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], (
            f"per-step val_x 应为 [0,10,...,90], got {val_x}"
        )
        # 特别验证没有截断到 99
        assert 99 not in val_x, "val_x 不应包含截断值 99"

    def test_val_x_exceeds_train_range(self):
        """val_x 超出 train 范围时不截断（11 个 val 点，最后一个 step=100 > 99）。"""
        train_losses = list(range(100))
        val_losses = list(range(11))
        _, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=10)

        assert val_x == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], (
            f"val_x 应为 [0,10,...,100], got {val_x}"
        )
        assert val_x[-1] == 100, "最后一个 val_x 不应截断到 99"

    def test_eval_interval_1_equal_length(self):
        """eval_interval=1 且 train/val 等长 → val_x = train_x。"""
        train_losses = [5, 4, 3, 2, 1]
        val_losses = [4.5, 3.5, 2.5, 1.5, 0.5]
        train_x, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=1)

        assert val_x == [0, 1, 2, 3, 4]
        assert val_x == train_x, "eval_interval=1 等长时 val_x 应等于 train_x"

    def test_eval_interval_1_shorter_val(self):
        """eval_interval=1 但 val 更短 → val_x 用索引（0..n_val-1）。"""
        train_losses = list(range(10))
        val_losses = [4.5, 3.5, 2.5]
        _, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=1)

        assert val_x == [0, 1, 2]

    def test_empty_val_losses(self):
        """空 val_losses → val_x 为空列表，不报错。"""
        train_losses = [5, 4, 3]
        _, val_x = _compute_loss_x(train_losses, [], eval_interval=10)

        assert val_x == []

    def test_empty_train_losses(self):
        """空 train_losses → val_x 用索引，不报错。"""
        val_losses = [4, 3, 2]
        train_x, val_x = _compute_loss_x([], val_losses, eval_interval=10)

        assert train_x == []
        assert val_x == [0, 1, 2]

    def test_both_empty(self):
        """两者都为空 → 两个空列表，不报错。"""
        train_x, val_x = _compute_loss_x([], [], eval_interval=10)

        assert train_x == []
        assert val_x == []

    def test_eval_interval_less_than_1(self):
        """eval_interval < 1 时视为 1。"""
        train_losses = [5, 4, 3, 2]
        val_losses = [4, 3, 2, 1]
        _, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=0)

        # eval_interval=0 → 视为 1，等长 → 1:1 对齐
        assert val_x == [0, 1, 2, 3]

    def test_negative_eval_interval(self):
        """eval_interval 为负数时视为 1。"""
        train_losses = [5, 4, 3, 2]
        val_losses = [4, 3, 2, 1]
        _, val_x = _compute_loss_x(train_losses, val_losses, eval_interval=-5)

        assert val_x == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 2. plot_loss_curve 集成测试（matplotlib mock）
# ---------------------------------------------------------------------------


class TestPlotLossCurveValX:
    """通过 plot_loss_curve 验证 val_x 正确传递给 matplotlib。"""

    def test_parallel_trainer_val_x_matplotlib(self, tmp_path, monkeypatch):
        """ParallelTrainer 场景（per-chunk）：val_x 1:1 对齐，不堆叠。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = [5, 3, 2, 1]
        val_losses = [4, 3, 2, 2]
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)

        # 第二次 ax.plot 是 val 曲线
        val_call = fake_ax.plot.call_args_list[1]
        val_x = val_call.args[0]
        assert val_x == [0, 1, 2, 3], (
            f"ParallelTrainer 场景 val_x 应为 [0,1,2,3], got {val_x}"
        )
        # 确保不是旧 bug 的 [0, 3, 3, 3]
        assert val_x != [0, 3, 3, 3], "val_x 不应是旧 bug 的 [0,3,3,3]"

    def test_per_step_val_x_no_truncation_matplotlib(self, tmp_path, monkeypatch):
        """per-step 场景：val_x 不截断。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = list(range(100))
        val_losses = list(range(10))
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=10)

        val_call = fake_ax.plot.call_args_list[1]
        val_x = val_call.args[0]
        assert val_x == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert 99 not in val_x, "val_x 不应截断到 99"

    def test_val_x_exceeds_train_range_matplotlib(self, tmp_path, monkeypatch):
        """val_x 超出 train 范围时不截断（matplotlib 自动扩展 x 轴）。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = list(range(100))
        val_losses = list(range(11))
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=10)

        val_call = fake_ax.plot.call_args_list[1]
        val_x = val_call.args[0]
        assert val_x[-1] == 100, f"最后一个 val_x 应为 100（不截断）, got {val_x[-1]}"

    def test_empty_val_losses_no_error_matplotlib(self, tmp_path, monkeypatch):
        """空 val_losses 时 matplotlib 分支不报错。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = [5, 4, 3, 2, 1]
        save_path = str(tmp_path / "loss_curve.png")

        # 不应抛异常
        actual = plot_loss_curve(train_losses, [], save_path, eval_interval=2)
        assert actual.endswith(".png")
        # val_losses 为空时 ax.plot 只调用一次（train）
        assert fake_ax.plot.call_count == 1

    def test_empty_train_losses_no_error_matplotlib(self, tmp_path, monkeypatch):
        """空 train_losses 时 matplotlib 分支不报错。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        val_losses = [4, 3, 2]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve([], val_losses, save_path, eval_interval=2)
        assert actual.endswith(".png")


# ---------------------------------------------------------------------------
# 3. ASCII 图 y 轴标签测试
# ---------------------------------------------------------------------------


class TestAsciiYAxisLabels:
    """ASCII 图 y 轴刻度标签测试。"""

    def test_ascii_has_y_max_y_min_labels(self, tmp_path, monkeypatch):
        """ASCII 图应包含 y_max 和 y_min 的 y 轴标签。"""
        _force_ascii(monkeypatch)

        train_losses = [5.0, 3.0, 2.0, 1.0]
        val_losses = [4.0, 3.0, 2.0, 2.0]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # y_max=5.0 应作为 y 轴标签出现（行首格式：右对齐数值 + " |"）
        y_max_str = f"{5.0:.4f}"
        y_max_lines = [
            line for line in content.splitlines()
            if line.strip().startswith(f"{y_max_str} |")
        ]
        assert len(y_max_lines) >= 1, (
            f"y_max={y_max_str} 应作为 y 轴标签出现，"
            f"内容:\n{content}"
        )

        # y_min=1.0 应作为 y 轴标签出现
        y_min_str = f"{1.0:.4f}"
        y_min_lines = [
            line for line in content.splitlines()
            if line.strip().startswith(f"{y_min_str} |")
        ]
        assert len(y_min_lines) >= 1, (
            f"y_min={y_min_str} 应作为 y 轴标签出现，"
            f"内容:\n{content}"
        )

    def test_ascii_has_y_mid_label(self, tmp_path, monkeypatch):
        """ASCII 图应包含 y_mid（中间值）的 y 轴标签。"""
        _force_ascii(monkeypatch)

        train_losses = [5.0, 3.0, 2.0, 1.0]
        val_losses = [4.0, 3.0, 2.0, 2.0]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # y_mid = (5.0 + 1.0) / 2 = 3.0
        y_mid_str = f"{3.0:.4f}"
        y_mid_lines = [
            line for line in content.splitlines()
            if line.strip().startswith(f"{y_mid_str} |")
        ]
        assert len(y_mid_lines) >= 1, (
            f"y_mid={y_mid_str} 应作为 y 轴标签出现"
        )

    def test_ascii_y_axis_label_width_le_10(self, tmp_path, monkeypatch):
        """y 轴标签宽度不超过 10 字符（数值 + ' |'）。"""
        _force_ascii(monkeypatch)

        train_losses = [5.0, 3.0, 2.0, 1.0]
        val_losses = [4.0, 3.0, 2.0, 2.0]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # 找到 y 轴标签行（以数值开头的行，后跟 " |"）
        label_lines = [
            line for line in content.splitlines()
            if line.strip() and line.strip()[0].isdigit()
            and " |" in line
        ]
        assert len(label_lines) >= 3, (
            f"应至少有 3 行 y 轴标签（y_max/y_mid/y_min）"
        )

        # 每行标签部分（到第一个 "|"）宽度不超过 10
        for line in label_lines:
            label_part = line[:line.index("|") + 1]
            assert len(label_part) <= 10, (
                f"y 轴标签宽度应 <= 10, got {len(label_part)}: {label_part!r}"
            )

    def test_ascii_has_step_arrow(self, tmp_path, monkeypatch):
        """ASCII 图底部应有 'step →' 标注 x 轴方向。"""
        _force_ascii(monkeypatch)

        train_losses = [5.0, 3.0, 2.0, 1.0]
        val_losses = [4.0, 3.0, 2.0, 2.0]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        assert "step →" in content, "ASCII 图应包含 'step →' 标注 x 轴方向"


# ---------------------------------------------------------------------------
# 4. matplotlib 图表增强测试
# ---------------------------------------------------------------------------


class TestMatplotlibEnhancements:
    """matplotlib 图表增强测试。"""

    def test_ylabel_includes_lower_is_better(self, tmp_path, monkeypatch):
        """matplotlib y 轴标签应包含 'lower is better'。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = [5.0, 4.0, 3.0, 2.0, 1.0]
        val_losses = [4.5, 3.5, 2.5]
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)

        # 检查 set_ylabel 被调用且包含 "lower is better"
        ylabel_call = fake_ax.set_ylabel.call_args
        assert ylabel_call is not None, "set_ylabel 应被调用"
        ylabel_arg = ylabel_call.args[0] if ylabel_call.args else ""
        assert "lower is better" in str(ylabel_arg), (
            f"y 轴标签应包含 'lower is better', got {ylabel_arg!r}"
        )

    def test_annotate_called_for_best_val(self, tmp_path, monkeypatch):
        """matplotlib 应调用 annotate 标注 best_val_loss 位置。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = [5.0, 4.0, 3.0, 2.0, 1.0]
        val_losses = [4.5, 3.5, 2.5]
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, val_losses, save_path, eval_interval=2)

        # annotate 应被调用（标注 best_val_loss）
        assert fake_ax.annotate.called, (
            "应调用 ax.annotate 标注 best_val_loss 位置"
        )

        # 检查 annotate 的参数包含 best= 前缀
        annotate_call = fake_ax.annotate.call_args
        text_arg = annotate_call.args[0] if annotate_call.args else ""
        assert "best=" in str(text_arg), (
            f"annotate 文本应包含 'best=', got {text_arg!r}"
        )

    def test_no_annotate_when_val_empty(self, tmp_path, monkeypatch):
        """val_losses 为空时不调用 annotate。"""
        try:
            import matplotlib  # noqa: F401
        except Exception:
            pytest.skip("matplotlib 不可用")

        import matplotlib.pyplot as plt

        fake_fig = MagicMock()
        fake_ax = MagicMock()
        monkeypatch.setattr(plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))

        train_losses = [5.0, 4.0, 3.0, 2.0, 1.0]
        save_path = str(tmp_path / "loss_curve.png")

        plot_loss_curve(train_losses, [], save_path, eval_interval=2)

        # val_losses 为空时不应调用 annotate
        assert not fake_ax.annotate.called, (
            "val_losses 为空时不应调用 annotate"
        )


# ---------------------------------------------------------------------------
# 5. ASCII 模式 val_x 对齐测试
# ---------------------------------------------------------------------------


class TestAsciiValXAlignment:
    """ASCII 模式下 val_x 对齐测试。"""

    def test_ascii_parallel_trainer_val_x_in_detail(self, tmp_path, monkeypatch):
        """ParallelTrainer 场景：ASCII detail 表中 val step 应 1:1 对齐。"""
        _force_ascii(monkeypatch)

        train_losses = [5, 3, 2, 1]
        val_losses = [4, 3, 2, 2]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # detail 表中 val step 应为 [0, 1, 2, 3]（不是 [0, 50, 100, 150]）
        detail_lines = [
            line for line in content.splitlines()
            if line.startswith("  [step") and "val_loss=" in line
        ]
        assert len(detail_lines) == 4
        for i, line in enumerate(detail_lines):
            expected_step = i  # 1:1 对齐，step = index
            assert f"step {expected_step:>6d}" in line, (
                f"detail 第 {i} 行 step 应为 {expected_step}（1:1 对齐）, got: {line}"
            )

    def test_ascii_per_step_val_x_in_detail(self, tmp_path, monkeypatch):
        """per-step 场景：ASCII detail 表中 val step 按 eval_interval 对齐。"""
        _force_ascii(monkeypatch)

        train_losses = list(range(100))
        val_losses = list(range(10))
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(
            train_losses, val_losses, save_path, eval_interval=10)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        detail_lines = [
            line for line in content.splitlines()
            if line.startswith("  [step") and "val_loss=" in line
        ]
        assert len(detail_lines) == 10
        for i, line in enumerate(detail_lines):
            expected_step = i * 10
            assert f"step {expected_step:>6d}" in line, (
                f"detail 第 {i} 行 step 应为 {expected_step}, got: {line}"
            )

    def test_ascii_parallel_trainer_val_points_spread(self, tmp_path, monkeypatch):
        """ParallelTrainer 场景：val 点应在画布中分散，不堆叠。"""
        _force_ascii(monkeypatch)

        train_losses = [5, 3, 2, 1]
        val_losses = [4, 3, 2, 2]
        save_path = str(tmp_path / "loss_curve.png")

        actual = plot_loss_curve(train_losses, val_losses, save_path, eval_interval=50)
        with open(actual, "r", encoding="utf-8") as f:
            content = f.read()

        # val 点（V 或 *）应至少有 2 个（4 个 val 点分散在画布中）
        # 旧 bug 中 3 个 val 点堆叠在 x=3，只有 1-2 个 V/* 可见
        v_count = content.count("V")
        star_count = content.count("*")
        # 减去图例中的 "V=val" 和 "*=overlap" 各 1 个
        canvas_v = v_count - 1  # 去掉 "V=val" 中的 V
        canvas_star = star_count - 1  # 去掉 "*=overlap" 中的 *
        assert canvas_v + canvas_star >= 2, (
            f"ParallelTrainer 场景 val 点应分散（至少 2 个 V/* 可见），"
            f"实际 canvas V={canvas_v}, *={canvas_star}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
