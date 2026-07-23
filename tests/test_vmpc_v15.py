"""Part5K1 Task 4：VMPC V1.5 命中与质量优化测试.

覆盖：
1. ``KnowledgeDistiller.contrastive_loss`` 排序一致性 + 可微 + 3D flatten
2. ``compute_loss`` 含对比蒸馏项（distill_contrastive=True 时 loss 增大）
3. ``QLinear`` outlier-aware 反量化（outlier 通道精度恢复）
4. ``compress_pipeline`` 默认版本为 V1.5（版本路由）
5. V1.3 仍可正常工作（向后兼容）
6. V1.5 与 V1.3 数值稳定性（禁用 V1.5 增强后压缩比一致）
7. V1.5 stats 包含 logit_calib_factor / contrastive_distill / vmpc_version 字段
8. 压缩模型元数据（_vmpc_version / _vmpc_compressed / logit_calib_factor）
9. ``compression_report`` 包含 vmpc_version + V1.5 专属字段
10. ``CometSparkNexLM._apply_logit_calibration`` 推理校准
11. ``CometSparkNexLM.generate`` 在压缩模型上不崩溃

运行方式：
    cd /workspace
    python -m pytest tests/test_vmpc_v15.py -x -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_torch / verse_nex
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_torch"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "verse_nex"))

from verse_torch import Tensor, nn, no_grad
from verse_torch.compress import (
    KnowledgeDistiller,
    QLinear,
    compress_pipeline,
    compression_report,
    count_parameters,
)
from verse_nex.cometspark import CometSparkNexLM


SEED = 42


# ---------------------------------------------------------------------------
# 辅助：构造小模型 / 数据
# ---------------------------------------------------------------------------


def _build_tlm(vocab=64, n_layer=2, n_head=4, n_embd=32, seed=SEED):
    """构造测试用 TransformerLM（固定 seed）。"""
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return nn._TransformerLM(
            vocab_size=vocab, n_layer=n_layer, n_head=n_head,
            n_embd=n_embd, seq_len=32, dropout=0.0, tie_weights=True,
        )


def _build_tiny_cometspark(vocab=64, dim=32, seed=SEED):
    """构造 tiny CometSparkNexLM（trisparse-only，快速）。"""
    np.random.seed(seed)
    return CometSparkNexLM(
        vocab_size=vocab, dim=dim, n_layer=2, n_head=4, n_kv_head=2,
        layer_pattern=["trisparse", "trisparse"], max_seq_len=64,
        use_alibi=True, use_rope=False, tie_weights=True,
    )


def _rand_batch(vocab=64, B=4, T=8, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab, size=(B, T)), rng.randint(0, vocab, size=(B, T))


# ---------------------------------------------------------------------------
# 1. contrastive_loss 排序一致性 + 可微
# ---------------------------------------------------------------------------


def test_contrastive_loss_ranking_consistency():
    """student 与 teacher 排序一致时 loss 小于排序不一致时。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    distiller = KnowledgeDistiller(
        teacher, student, temperature=4.0, alpha=0.7,
        distill_contrastive=True, contrastive_loss_weight=0.5,
        contrastive_margin=0.5, contrastive_top_k=10,
    )

    x, _ = _rand_batch(seed=3)
    with no_grad():
        teacher_logits = teacher(x)  # (B, T, V)

    # 场景 A：student logits = teacher logits（排序完全一致）
    student_matched = Tensor(teacher_logits.data.copy(), requires_grad=True)
    loss_matched = distiller.contrastive_loss(
        student_matched, teacher_logits, margin=0.5, top_k=10)

    # 场景 B：student logits = -teacher logits（排序完全反转）
    student_reversed = Tensor(-teacher_logits.data.copy(), requires_grad=True)
    loss_reversed = distiller.contrastive_loss(
        student_reversed, teacher_logits, margin=0.5, top_k=10)

    assert float(loss_matched.data) < float(loss_reversed.data), (
        f"排序一致时 loss 应更小：matched={loss_matched.data} "
        f"reversed={loss_reversed.data}"
    )
    # 排序一致时 loss 应接近 0（margin 内的 hinge 全为 0）
    assert float(loss_matched.data) >= 0.0, "contrastive_loss 应非负"
    # 排序反转时 loss 应严格大于 0
    assert float(loss_reversed.data) > 0.0, "排序反转时 loss 应 > 0"


def test_contrastive_loss_differentiable():
    """contrastive_loss 可微：backward 后 student 参数有梯度。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    distiller = KnowledgeDistiller(
        teacher, student, distill_contrastive=True,
    )
    x, _ = _rand_batch(seed=3)
    with no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)  # 构建计算图
    loss = distiller.contrastive_loss(student_logits, teacher_logits,
                                      margin=0.5, top_k=10)
    assert isinstance(loss, Tensor)
    loss.backward()
    has_grad = any(p.grad is not None for p in student.parameters())
    assert has_grad, "contrastive_loss 反向后 student 应有梯度"


def test_contrastive_loss_3d_flatten():
    """3D 输入 (B, T, V) 自动 flatten 到 (B*T, V) 处理，不报错。"""
    np.random.seed(0)
    B, T_, V = 2, 3, 16
    teacher_data = np.random.randn(B, T_, V).astype(np.float32)
    student_data = np.random.randn(B, T_, V).astype(np.float32)
    teacher_logits = Tensor(teacher_data, requires_grad=False)
    student_logits = Tensor(student_data, requires_grad=True)

    teacher = _build_tlm(vocab=V, seed=1)
    student = _build_tlm(vocab=V, seed=2)
    distiller = KnowledgeDistiller(teacher, student, distill_contrastive=True)
    loss = distiller.contrastive_loss(student_logits, teacher_logits,
                                      margin=0.5, top_k=5)
    assert isinstance(loss, Tensor)
    assert loss.data.shape == (), "loss 应为标量"
    assert np.isfinite(float(loss.data))


def test_contrastive_loss_too_small_vocab():
    """V < 2 时返回 0 标量（无 pairwise 关系）。"""
    teacher = _build_tlm(vocab=64, seed=1)
    student = _build_tlm(vocab=64, seed=2)
    distiller = KnowledgeDistiller(teacher, student, distill_contrastive=True)
    # V=1，actual_k=min(top_k, 1)=1 < 2 → 返回 0
    student_logits = Tensor(np.array([[0.5]], dtype=np.float32), requires_grad=True)
    teacher_logits = Tensor(np.array([[0.3]], dtype=np.float32), requires_grad=False)
    loss = distiller.contrastive_loss(student_logits, teacher_logits,
                                      margin=0.5, top_k=10)
    assert float(loss.data) == 0.0


# ---------------------------------------------------------------------------
# 2. compute_loss 含对比蒸馏项
# ---------------------------------------------------------------------------


def test_compute_loss_with_contrastive():
    """distill_contrastive=True 时 loss 大于 False（student ≠ teacher）。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    distiller = KnowledgeDistiller(
        teacher, student, temperature=4.0, alpha=0.7,
        distill_contrastive=True, contrastive_loss_weight=0.5,
    )
    x, y = _rand_batch(seed=3)
    with no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)

    loss_no_contrastive = distiller.compute_loss(
        student_logits, teacher_logits, labels=y, distill_contrastive=False)
    loss_with_contrastive = distiller.compute_loss(
        student_logits, teacher_logits, labels=y, distill_contrastive=True)

    assert float(loss_with_contrastive.data) > float(loss_no_contrastive.data), (
        f"启用对比蒸馏时 loss 应更大：with={loss_with_contrastive.data} "
        f"without={loss_no_contrastive.data}"
    )


# ---------------------------------------------------------------------------
# 3. QLinear outlier-aware 反量化
# ---------------------------------------------------------------------------


def test_qlinear_outlier_aware_residual():
    """outlier_aware=True 识别 outlier 通道并保存 residual。"""
    np.random.seed(0)
    n_out, n_in = 32, 16
    lin = nn.Linear(n_in, n_out, bias=True)
    # 正常权重（std≈0.1）
    w = np.random.randn(n_out, n_in).astype(np.float32) * 0.1
    # 制造一个 outlier 通道（行 3 的权重远大于其他行）
    w[3, :] = np.random.randn(n_in).astype(np.float32) * 5.0
    lin.weight.data = w

    # 不启用 outlier-aware
    qlin_normal = QLinear(lin, qtype="int4", outlier_aware=False)
    assert qlin_normal._outlier_residual is None
    assert qlin_normal._outlier_idx is None

    # 启用 outlier-aware
    qlin_outlier = QLinear(lin, qtype="int4", outlier_aware=True,
                           outlier_threshold=3.0)
    assert qlin_outlier._outlier_residual is not None, (
        "应检测到 outlier 通道并保存 residual"
    )
    assert qlin_outlier._outlier_idx is not None
    # outlier 通道应包含行 3
    assert 3 in qlin_outlier._outlier_idx.tolist(), (
        f"行 3 应被识别为 outlier，实际 outlier idx={qlin_outlier._outlier_idx}"
    )


def test_qlinear_outlier_aware_accuracy():
    """outlier-aware 模式下 outlier 通道精度恢复到 fp32 级别。"""
    np.random.seed(0)
    n_out, n_in = 32, 16
    lin = nn.Linear(n_in, n_out, bias=True)
    w = np.random.randn(n_out, n_in).astype(np.float32) * 0.1
    w[3, :] = np.random.randn(n_in).astype(np.float32) * 5.0
    lin.weight.data = w

    qlin_normal = QLinear(lin, qtype="int4", outlier_aware=False)
    qlin_outlier = QLinear(lin, qtype="int4", outlier_aware=True,
                           outlier_threshold=3.0)

    x_np = np.random.randn(4, n_in).astype(np.float32)
    x = Tensor(x_np)

    # 原始 Linear 输出（ground truth）
    with no_grad():
        y_orig = lin(x).data  # (4, n_out)
    y_normal = qlin_normal(x).data  # (4, n_out)
    y_outlier = qlin_outlier(x).data  # (4, n_out)

    # 在 outlier 通道（行 3）上，outlier-aware 应更接近原始
    err_normal_outlier = np.abs(y_normal[:, 3] - y_orig[:, 3])
    err_outlier_outlier = np.abs(y_outlier[:, 3] - y_orig[:, 3])
    assert np.mean(err_outlier_outlier) < np.mean(err_normal_outlier), (
        f"outlier 通道上 outlier-aware 应更精确："
        f"normal_err={np.mean(err_normal_outlier):.6f} "
        f"outlier_err={np.mean(err_outlier_outlier):.6f}"
    )
    # outlier 通道应几乎完全恢复（residual 修正 → fp32 级别精度）
    assert np.all(err_outlier_outlier < 1e-3), (
        f"outlier 通道应恢复到 fp32 精度：max_err={np.max(err_outlier_outlier)}"
    )

    # 非 outlier 通道上两者应相同（outlier 修正不影响其他通道）
    non_outlier_mask = np.ones(n_out, dtype=bool)
    non_outlier_mask[qlin_outlier._outlier_idx] = False
    err_non_outlier = np.abs(y_normal[:, non_outlier_mask] -
                             y_outlier[:, non_outlier_mask])
    assert np.max(err_non_outlier) < 1e-6, (
        f"非 outlier 通道上 normal 与 outlier-aware 应相同："
        f"max_diff={np.max(err_non_outlier)}"
    )


def test_qlinear_outlier_aware_no_outliers():
    """无 outlier 通道时 outlier_aware=True 不报错（residual=None）。"""
    np.random.seed(0)
    n_out, n_in = 32, 16
    lin = nn.Linear(n_in, n_out, bias=True)
    # 所有权重均匀分布（无 outlier）
    lin.weight.data = np.random.randn(n_out, n_in).astype(np.float32) * 0.1
    qlin = QLinear(lin, qtype="int4", outlier_aware=True,
                   outlier_threshold=3.0)
    # 无 outlier 时 residual 为 None（或不影响 forward）
    if qlin._outlier_residual is not None:
        # 如果检测到 outlier（统计上可能），forward 仍应正常
        pass
    x = Tensor(np.random.randn(2, n_in).astype(np.float32))
    out = qlin(x)
    assert out.shape == (2, n_out), f"forward 应正常返回 (2, {n_out})"


# ---------------------------------------------------------------------------
# 4. 版本路由（默认 V1.5）
# ---------------------------------------------------------------------------


def test_default_version_is_v15():
    """compress_pipeline 不指定 version 时默认走 V1.5。"""
    model = _build_tlm(seed=1)
    new_model, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True,
    )
    assert stats["vmpc_version"] == "1.5"
    assert stats["version"] == "1.5"
    assert getattr(new_model, "_vmpc_version", None) == "1.5"


def test_explicit_version_v15():
    """显式指定 version='1.5' 走 V1.5 路径。"""
    model = _build_tlm(seed=1)
    new_model, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True, version="1.5",
    )
    assert stats["vmpc_version"] == "1.5"


def test_version_tuple_routing():
    """版本号元组比较：'1.5.0' 等价于 '1.5'，都能走 V1.5 路径。"""
    model = _build_tlm(seed=1)
    new_model, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True, version="1.5.0",
    )
    assert stats["vmpc_version"] == "1.5"


# ---------------------------------------------------------------------------
# 5. V1.3 仍可正常工作
# ---------------------------------------------------------------------------


def test_v13_still_functional():
    """V1.3 路径仍正常工作，vmpc_version='1.3'。"""
    model = _build_tlm(seed=1)
    new_model, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True, version="1.3",
    )
    assert stats["vmpc_version"] == "1.3"
    assert getattr(new_model, "_vmpc_version", None) == "1.3"
    # V1.3 不设置 logit_calib_factor
    assert not hasattr(new_model, "logit_calib_factor") or \
        getattr(new_model, "logit_calib_factor", None) is None
    # V1.3 stats 不含 V1.5 专属字段
    assert "logit_calib_factor" not in stats
    # forward 仍正常
    x = np.random.randint(0, 64, size=(2, 8))
    out = new_model(Tensor(x))
    assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 6. V1.5 与 V1.3 数值稳定性
# ---------------------------------------------------------------------------


def test_v15_disabled_features_match_v13():
    """V1.5 禁用 contrastive_distill + logit_calibration 后压缩比与 V1.3 一致。"""
    model = _build_tlm(seed=1)
    cfg = {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4},
           "lora": {"rank": 4}}

    _, stats_v13 = compress_pipeline(
        model, cfg, return_stats=True, version="1.3")
    _, stats_v15 = compress_pipeline(
        model, dict(cfg, distill_contrastive=False, logit_calibration=False),
        return_stats=True, version="1.5")

    # 压缩比一致（模型结构相同）
    assert stats_v13["compression_ratio"] == pytest.approx(
        stats_v15["compression_ratio"], rel=1e-6), (
        f"禁用 V1.5 增强后压缩比应一致：v13={stats_v13['compression_ratio']} "
        f"v15={stats_v15['compression_ratio']}"
    )
    assert stats_v13["compressed_bits"] == stats_v15["compressed_bits"]
    # V1.5 仍标记版本为 1.5
    assert stats_v15["vmpc_version"] == "1.5"
    # 禁用后 logit_calib_factor 应为 1.0（默认值）
    assert stats_v15["logit_calib_factor"] == 1.0
    assert stats_v15["contrastive_distill"] is False


def test_v15_stats_all_finite():
    """V1.5 默认配置产生的 stats 全部为有限值（无数值异常）。"""
    model = _build_tlm(seed=1)
    _, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True, version="1.5",
    )
    for key in ("compression_ratio", "bits", "sparsity", "logit_calib_factor"):
        val = stats.get(key)
        if val is not None:
            assert np.isfinite(val), f"stats[{key!r}]={val} 非有限值"


# ---------------------------------------------------------------------------
# 7. V1.5 stats 包含新字段
# ---------------------------------------------------------------------------


def test_v15_stats_fields():
    """V1.5 stats 包含 logit_calib_factor / contrastive_distill / vmpc_version。"""
    model = _build_tlm(seed=1)
    _, stats = compress_pipeline(
        model, {"quantize": {"bits": 4}}, return_stats=True, version="1.5",
    )
    assert "vmpc_version" in stats
    assert stats["vmpc_version"] == "1.5"
    assert "logit_calib_factor" in stats
    assert isinstance(stats["logit_calib_factor"], float)
    assert "contrastive_distill" in stats
    assert isinstance(stats["contrastive_distill"], bool)
    # steps 中应包含 logit_calibration 步骤（默认启用）
    step_names = [s["step"] for s in stats["steps"]]
    assert "logit_calibration" in step_names, (
        f"V1.5 默认应包含 logit_calibration 步骤，实际 steps={step_names}"
    )


# ---------------------------------------------------------------------------
# 8. 压缩模型元数据
# ---------------------------------------------------------------------------


def test_v15_model_metadata():
    """V1.5 压缩后模型携带 _vmpc_version / _vmpc_compressed / logit_calib_factor。"""
    model = _build_tlm(seed=1)
    new_model = compress_pipeline(
        model, {"quantize": {"bits": 4}}, version="1.5",
    )
    assert getattr(new_model, "_vmpc_version", None) == "1.5"
    assert getattr(new_model, "_vmpc_compressed", False) is True
    assert hasattr(new_model, "logit_calib_factor")
    assert isinstance(getattr(new_model, "logit_calib_factor"), float)
    assert getattr(new_model, "logit_calib_factor") > 0.0


def test_v15_with_teacher_metadata():
    """V1.5 + teacher（无 train_loader）：contrastive_distill 元数据正确。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    new_model, stats = compress_pipeline(
        student,
        {"quantize": {"bits": 4}, "distill": {"teacher": teacher}},
        return_stats=True, version="1.5",
    )
    # 默认 distill_contrastive=True，但无 train_loader → contrastive_actually_used=True
    assert stats["contrastive_distill"] is True
    assert getattr(new_model, "_vmpc_contrastive_distill", False) is True
    # teacher 被冻结
    assert all(not p.requires_grad for p in teacher.parameters())


# ---------------------------------------------------------------------------
# 9. compression_report 包含 vmpc_version + V1.5 专属字段
# ---------------------------------------------------------------------------


def test_compression_report_v15():
    """V1.5 压缩报告包含 vmpc_version + logit_calib_factor + contrastive_distill。"""
    model = _build_tlm(seed=1)
    new_model = compress_pipeline(
        model, {"quantize": {"bits": 4}}, version="1.5",
    )
    report = compression_report(model, new_model)
    assert "vmpc_version" in report
    assert report["vmpc_version"] == "1.5"
    assert report["version"] == "1.5"
    # V1.5 专属字段
    assert "logit_calib_factor" in report
    assert isinstance(report["logit_calib_factor"], float)
    assert "contrastive_distill" in report
    assert isinstance(report["contrastive_distill"], bool)


def test_compression_report_v13_backward_compat():
    """V1.3 压缩报告 vmpc_version='1.3'，无 V1.5 专属字段。"""
    model = _build_tlm(seed=1)
    new_model = compress_pipeline(
        model, {"quantize": {"bits": 4}}, version="1.3",
    )
    report = compression_report(model, new_model)
    assert report["vmpc_version"] == "1.3"
    # V1.3 不应包含 V1.5 专属字段
    assert "logit_calib_factor" not in report
    assert "contrastive_distill" not in report


# ---------------------------------------------------------------------------
# 10. CometSparkNexLM._apply_logit_calibration 推理校准
# ---------------------------------------------------------------------------


def test_apply_logit_calibration_non_compressed():
    """非压缩模型 _apply_logit_calibration 原样返回。"""
    model = _build_tiny_cometspark(seed=1)
    logits = np.random.randn(2, 64).astype(np.float32)
    result = model._apply_logit_calibration(logits)
    np.testing.assert_array_equal(result, logits), (
        "非压缩模型应原样返回 logits"
    )


def test_apply_logit_calibration_compressed():
    """压缩模型 _apply_logit_calibration 对 logits 做方差归一化。"""
    model = _build_tiny_cometspark(seed=1)
    # 模拟 V1.5 压缩后设置元数据
    object.__setattr__(model, "_vmpc_compressed", True)
    object.__setattr__(model, "logit_calib_factor", 2.0)

    np.random.seed(0)
    logits = np.random.randn(2, 64).astype(np.float32) * 3.0
    result = model._apply_logit_calibration(logits)

    # 验证校准公式：logits / sqrt(var + eps) * calib_factor
    var = np.var(logits, axis=-1, keepdims=True)
    expected = logits / np.sqrt(var + 1e-8) * 2.0
    np.testing.assert_allclose(result, expected, atol=1e-6), (
        "校准结果应与公式 logits/sqrt(var+eps)*calib_factor 一致"
    )
    # 校准后 shape 不变
    assert result.shape == logits.shape


def test_apply_logit_calibration_default_factor():
    """压缩模型但 calib_factor 缺省为 1.0 时仍做方差归一化。"""
    model = _build_tiny_cometspark(seed=1)
    object.__setattr__(model, "_vmpc_compressed", True)
    # 不设置 logit_calib_factor → getattr 默认 1.0
    np.random.seed(0)
    logits = np.random.randn(2, 64).astype(np.float32)
    result = model._apply_logit_calibration(logits)
    var = np.var(logits, axis=-1, keepdims=True)
    expected = logits / np.sqrt(var + 1e-8) * 1.0
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. CometSparkNexLM.generate 在压缩模型上不崩溃
# ---------------------------------------------------------------------------


def test_generate_on_compressed_model_sampling_path():
    """压缩模型 generate（采样路径）不崩溃，输出 shape 正确。"""
    model = _build_tiny_cometspark(seed=1)
    # 用 V1.5 压缩（设置 _vmpc_compressed + logit_calib_factor）
    compressed = compress_pipeline(
        model, {"quantize": {"bits": 4}}, version="1.5",
    )
    assert getattr(compressed, "_vmpc_compressed", False) is True

    idx = np.array([[1, 2, 3, 4]], dtype=np.int64)
    # 采样路径（temperature=0.5）→ _generate_with_logits → forward
    out = compressed.generate(idx, max_new_tokens=3, temperature=0.5)
    assert out.shape[0] == 1
    assert out.shape[1] >= 4  # prompt + 至少部分生成 token


def test_generate_on_compressed_model_greedy_path():
    """压缩模型 generate（greedy 路径）不崩溃，输出 shape 正确。"""
    model = _build_tiny_cometspark(seed=1)
    compressed = compress_pipeline(
        model, {"quantize": {"bits": 4}}, version="1.5",
    )
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    # greedy 路径（temperature=1.0, top_k=None）→ _generate_recurrent
    out = compressed.generate(idx, max_new_tokens=2)
    assert out.shape[0] == 1
    assert out.shape[1] >= 3


def test_generate_on_non_compressed_model_unchanged():
    """非压缩模型 generate 行为不变（不应用 logit 校准）。"""
    model = _build_tiny_cometspark(seed=1)
    assert not getattr(model, "_vmpc_compressed", False)
    idx = np.array([[1, 2, 3]], dtype=np.int64)
    out = model.generate(idx, max_new_tokens=2)
    assert out.shape[0] == 1
    assert out.shape[1] >= 3


# ---------------------------------------------------------------------------
# 附加：V1.5 端到端压缩 + forward 一致性
# ---------------------------------------------------------------------------


def test_v15_compress_forward_consistent():
    """V1.5 压缩后模型 forward 输出 shape 正确，原模型不变。"""
    model = _build_tlm(seed=1)
    orig_sd = {k: v.copy() for k, v in model.state_dict().items()}

    new_model = compress_pipeline(
        model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
        version="1.5",
    )
    # 原模型不变
    for k in orig_sd:
        assert np.array_equal(orig_sd[k], model.state_dict()[k])
    # forward 正常
    x = np.random.randint(0, 64, size=(2, 8))
    out = new_model(Tensor(x))
    assert out.shape == (2, 8, 64)


def test_v15_outlier_aware_quantize_in_pipeline():
    """V1.5 流水线中 quantize.outlier_aware=True 生效。"""
    model = _build_tlm(seed=1)
    new_model, stats = compress_pipeline(
        model,
        {"quantize": {"bits": 4, "outlier_aware": True}},
        return_stats=True, version="1.5",
    )
    quant_step = next(s for s in stats["steps"] if s["step"] == "quantize")
    assert quant_step["outlier_aware"] is True
    # 模型中应有 QLinear 且 outlier_aware=True
    has_outlier_qlin = False
    for _, m in new_model.named_modules():
        if isinstance(m, QLinear) and m.outlier_aware:
            has_outlier_qlin = True
            break
    assert has_outlier_qlin, "应存在 outlier_aware=True 的 QLinear"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
