"""Part4K2 Task 6: 压缩技术 V1.3（以小博大）测试.

覆盖：
1. ``KnowledgeDistiller`` V1.3 ``compute_loss`` 含特征匹配项 + 可微
2. 自适应温度调度（temperature annealing）
3. ``compress_pipeline`` V1.3 流程（prune → quantize → distill → lora）
4. ``CometSparkNexLM.distill_from``（teacher → student 能力转移，loss 下降）
5. 吞吐率优化（``QuantizedLinear.forward_fused`` + ``benchmark_throughput`` + ``quantize_batch``）
6. ``compression_report`` 生成
7. ``CometSparkNexLM.compress_v13`` 端到端
8. 蒸馏后小模型能力不大幅下降（简单数据集验证）
9. 不指定 teacher_model 时仅 prune+quantize+lora（兼容）+ V1.0 旧 API 兼容

运行方式：
    cd /workspace
    python -m pytest tests/test_compress_v13.py -x -q
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

from verse_torch import Tensor, nn
from verse_torch.compress import (
    KnowledgeDistiller,
    compress_pipeline,
    compression_report,
    QLinear,
    LoRALinear,
    count_parameters,
    count_nonzero_params,
    compute_compressed_bits,
)
from verse_torch.quantize import (
    QuantizedLinear,
    quantize_batch,
    benchmark_throughput,
)
from verse_nex.cometspark import CometSparkNexLM


SEED = 42


# ---------------------------------------------------------------------------
# 辅助：构造小模型 / 数据 / 特征提取
# ---------------------------------------------------------------------------


def _build_tlm(vocab=64, n_layer=2, n_head=4, n_embd=32, seed=SEED):
    """构造测试用 TransformerLM（固定 seed）。"""
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return nn.TransformerLM(
            vocab_size=vocab, n_layer=n_layer, n_head=n_head,
            n_embd=n_embd, seq_len=32, dropout=0.0, tie_weights=True,
        )


def _rand_batch(vocab=64, B=4, T=8, seed=SEED):
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab, size=(B, T)), rng.randint(0, vocab, size=(B, T))


def _lm_feature_extractor(model, idx):
    """通用中间层特征提取：返回 (logits, [hidden])。

    hidden 为 norm 之后、head 之前的隐藏态 ``(B, T, D)``。
    对 TransformerLM 与 CometSparkNexLM 均适用（两者都有 tok_emb/blocks/norm/head，
    且 block 前向返回 ``(out, state)``）。
    """
    if not isinstance(idx, Tensor):
        idx = Tensor(np.asarray(idx, dtype=np.int64))
    elif idx.data.dtype != np.int64:
        idx = Tensor(idx.data.astype(np.int64))
    x = model.tok_emb(idx)
    for block in model.blocks:
        x, _ = block(x)
    x = model.norm(x)
    logits = model.head(x)
    return logits, [x]


def _build_tiny_cometspark(vocab=64, dim=32, seed=SEED):
    """构造 tiny CometSparkNexLM（trisparse-only，快速）。"""
    np.random.seed(seed)
    return CometSparkNexLM(
        vocab_size=vocab, dim=dim, n_layer=2, n_head=4, n_kv_head=2,
        layer_pattern=["trisparse", "trisparse"], max_seq_len=64,
        use_alibi=True, use_rope=False, tie_weights=True,
    )


# ---------------------------------------------------------------------------
# 1. KnowledgeDistiller V1.3 compute_loss 含特征匹配项
# ---------------------------------------------------------------------------


def test_compute_loss_with_feature_matching():
    """compute_loss 包含特征匹配项：带特征的 loss > 不带特征的 loss（且可微）。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    distiller = KnowledgeDistiller(
        teacher, student, temperature=4.0, alpha=0.7,
        feature_loss_weight=0.3,
    )

    x, y = _rand_batch(seed=3)
    # teacher 特征（no_grad）
    from verse_torch import no_grad
    with no_grad():
        teacher_logits, (teacher_feat,) = _lm_feature_extractor(teacher, x)
    # student 特征（构建计算图）
    student_logits, (student_feat,) = _lm_feature_extractor(student, x)

    # 不带特征
    loss_no_feat = distiller.compute_loss(
        student_logits, teacher_logits, labels=y)
    # 带特征
    loss_feat = distiller.compute_loss(
        student_logits, teacher_logits,
        student_features=[student_feat], teacher_features=[teacher_feat],
        labels=y,
    )

    assert isinstance(loss_feat, Tensor)
    assert loss_feat.data.shape == (), "loss 应为标量"
    # 特征匹配项为正（随机特征 MSE > 0），故带特征的 loss 更大
    assert float(loss_feat.data) > float(loss_no_feat.data), (
        f"带特征匹配的 loss 应更大：feat={loss_feat.data} vs no_feat={loss_no_feat.data}"
    )
    # 可微：反向后 student 参数应有梯度
    loss_feat.backward()
    has_grad = any(p.grad is not None for p in student.parameters())
    assert has_grad, "蒸馏损失反向后 student 应有梯度"


def test_compute_loss_without_labels():
    """labels=None 时跳过 CE 项，仅 soft loss（不报错）。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    distiller = KnowledgeDistiller(teacher, student, temperature=4.0)
    x, _ = _rand_batch(seed=3)
    from verse_torch import no_grad
    with no_grad():
        tl = teacher(x)
    sl = student(x)
    loss = distiller.compute_loss(sl, tl, labels=None)
    assert isinstance(loss, Tensor)
    assert np.isfinite(float(loss.data))


# ---------------------------------------------------------------------------
# 2. 自适应温度调度
# ---------------------------------------------------------------------------


def test_temperature_annealing():
    """自适应温度调度：训练后温度退火到 _T_min；关闭退火则保持不变。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    distiller = KnowledgeDistiller(teacher, student, temperature=8.0)
    assert distiller._T_init == 8.0
    assert distiller._T_min == pytest.approx(2.0)  # max(1.0, 8.0*0.25)

    x, y = _rand_batch(seed=3)
    loader = [(x, y)]

    # 开启退火：epochs=4，最后一个 epoch frac=1.0 → temperature=_T_min
    distiller.distill(loader, epochs=4, lr=1e-3, anneal_temperature=True)
    assert distiller.temperature == pytest.approx(distiller._T_min), (
        f"退火后温度应到 _T_min={distiller._T_min}，实际 {distiller.temperature}"
    )

    # 关闭退火：温度保持 _T_init
    distiller2 = KnowledgeDistiller(
        _build_tlm(seed=5), _build_tlm(seed=6), temperature=6.0)
    t_init = distiller2._T_init
    distiller2.distill(loader, epochs=3, lr=1e-3, anneal_temperature=False)
    assert distiller2.temperature == pytest.approx(t_init), (
        f"关闭退火时温度应保持 {t_init}，实际 {distiller2.temperature}"
    )


def test_T_backward_compat_alias():
    """V1.0 旧参数 T 仍可用作 temperature 别名。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    d1 = KnowledgeDistiller(teacher, student, T=3.0)
    assert d1.temperature == 3.0
    assert d1.T == 3.0
    # forward(student, teacher, hard_targets) V1.0 接口仍可用
    x, y = _rand_batch(seed=3)
    from verse_torch import no_grad
    with no_grad():
        tl = teacher(x)
    sl = student(x)
    loss = d1.forward(sl, tl, y)
    assert isinstance(loss, Tensor)


# ---------------------------------------------------------------------------
# 3. compress_pipeline V1.3 流程
# ---------------------------------------------------------------------------


def test_compress_pipeline_v13_flow():
    """V1.3 流程：prune → quantize → lora（无 teacher），含压缩报告。"""
    model = _build_tlm(seed=1)
    orig_sd = {k: v.copy() for k, v in model.state_dict().items()}

    new_model, stats = compress_pipeline(
        model,
        {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4},
         "lora": {"rank": 4}},
        return_stats=True,
    )
    step_names = [s["step"] for s in stats["steps"]]
    assert step_names == ["prune", "quantize", "lora"], (
        f"V1.3 步骤顺序应为 prune→quantize→lora，实际 {step_names}"
    )
    assert stats["version"] == "1.3"
    assert "compression_report" in stats
    assert stats["compression_report"]["version"] == "1.3"
    assert stats["compression_ratio"] > 1.0
    # 原模型不变
    for k in orig_sd:
        assert np.array_equal(orig_sd[k], model.state_dict()[k])
    # 新模型可 forward
    x = np.random.randint(0, 64, size=(2, 8))
    out = new_model(Tensor(x))
    assert out.shape == (2, 8, 64)


def test_compress_pipeline_v13_distill_order():
    """V1.3 重排：distill 在 quantize 之后、lora 之前（teacher 无 train_loader）。"""
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    new_model, stats = compress_pipeline(
        student,
        {"prune": {"sparsity": 0.2}, "quantize": {"bits": 4},
         "distill": {"teacher": teacher}, "lora": {"rank": 4}},
        return_stats=True,
    )
    step_names = [s["step"] for s in stats["steps"]]
    assert step_names == ["prune", "quantize", "distill", "lora"], (
        f"V1.3 含 teacher 时顺序应为 prune→quantize→distill→lora，实际 {step_names}"
    )
    # distill 步骤记录"无 train_loader"
    distill_step = next(s for s in stats["steps"] if s["step"] == "distill")
    assert "note" in distill_step


# ---------------------------------------------------------------------------
# 4. distill_from（teacher → student 能力转移，loss 下降）
# ---------------------------------------------------------------------------


def test_cometspark_distill_from_loss_decreases():
    """CometSparkNexLM.distill_from：loss 末值低于初值。"""
    teacher = _build_tiny_cometspark(seed=1)
    student = _build_tiny_cometspark(seed=2)
    x, y = _rand_batch(vocab=64, B=2, T=8, seed=3)
    data = [(x, y)] * 3

    losses = student.distill_from(
        teacher, data, config={"epochs": 3, "lr": 1e-2, "max_steps": 9})
    assert len(losses) == 9
    assert losses[-1] < losses[0], (
        f"蒸馏后 loss 应下降：first={losses[0]:.4f} last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. 吞吐率优化
# ---------------------------------------------------------------------------


def test_quantized_linear_forward_fused_matches_forward():
    """forward_fused 与 forward 数值等价。"""
    np.random.seed(0)
    lin = nn.Linear(16, 32, bias=True)
    qlin = QuantizedLinear(lin, qtype="int4", cache_fp32=False)
    x = np.random.randn(5, 16).astype(np.float32)
    out_fwd = qlin.forward(x)
    out_fused = qlin.forward_fused(x)
    assert out_fwd.shape == out_fused.shape == (5, 32)
    assert np.allclose(out_fwd, out_fused, atol=1e-5), (
        f"forward 与 forward_fused 应数值等价：max diff "
        f"{np.max(np.abs(out_fwd - out_fused))}"
    )


def test_benchmark_throughput_and_quantize_batch():
    """benchmark_throughput 返回有效吞吐率；quantize_batch 批量量化。"""
    np.random.seed(0)
    lin = nn.Linear(16, 32, bias=True)
    qlin = QuantizedLinear(lin, qtype="int4", cache_fp32=True)
    x = np.random.randn(8, 16).astype(np.float32)
    bench = benchmark_throughput(qlin, x, warmup=1, iters=5, unit="samples")
    assert bench["throughput"] > 0
    assert bench["iters"] == 5
    assert bench["unit"] == "samples"
    assert bench["latency_ms"] >= 0.0

    # batch 量化
    linears = [nn.Linear(8, 16), nn.Linear(16, 4)]
    qbatch = quantize_batch(linears, qtype="int8")
    assert len(qbatch) == 2
    assert all(isinstance(q, QuantizedLinear) for q in qbatch)
    assert qbatch[0].qtype == "int8"


# ---------------------------------------------------------------------------
# 6. compression_report
# ---------------------------------------------------------------------------


def test_compression_report():
    """compression_report 生成完整字段。"""
    model = _build_tlm(seed=1)
    new_model, _ = compress_pipeline(
        model, {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4}},
        return_stats=True,
    )
    report = compression_report(model, new_model)
    expected_keys = {
        "original_params", "compressed_params", "compression_ratio",
        "sparsity", "bits_per_param", "estimated_throughput_improvement",
        "version",
    }
    assert expected_keys.issubset(report.keys()), (
        f"compression_report 缺字段：{expected_keys - set(report.keys())}"
    )
    assert report["version"] == "1.3"
    assert report["original_params"] > 0
    assert report["compression_ratio"] > 1.0
    assert report["bits_per_param"] < 32.0
    assert report["estimated_throughput_improvement"] >= 1.0


# ---------------------------------------------------------------------------
# 7. CometSparkNexLM.compress_v13 端到端
# ---------------------------------------------------------------------------


def test_cometspark_compress_v13_endtoend():
    """CometSparkNexLM.compress_v13：返回新模型，原模型不变，可 forward。"""
    model = _build_tiny_cometspark(seed=1)
    orig_sd = {k: v.copy() for k, v in model.state_dict().items()}
    orig_params = model.count_parameters()

    compressed = model.compress_v13(
        {"prune": {"sparsity": 0.3}, "quantize": {"bits": 4},
         "lora": {"rank": 4}})

    assert compressed is not model
    assert model.count_parameters() == orig_params  # 原模型不变
    for k in orig_sd:
        assert np.array_equal(orig_sd[k], model.state_dict()[k])
    # 压缩后可 forward
    x = np.random.randint(0, 64, size=(1, 8))
    out = compressed(Tensor(x))
    assert out.shape == (1, 8, 64)
    # 缓存了 v13 stats
    stats = getattr(compressed, "_v13_stats", None)
    assert stats is not None and stats["version"] == "1.3"


def test_cometspark_compress_v13_with_teacher():
    """compress_v13 传入 teacher_model（无 train_loader）：仅冻结 teacher 做准备。"""
    teacher = _build_tiny_cometspark(seed=1)
    student = _build_tiny_cometspark(seed=2)
    compressed = student.compress_v13(
        {"quantize": {"bits": 4}}, teacher_model=teacher)
    stats = getattr(compressed, "_v13_stats", None)
    assert stats is not None
    step_names = [s["step"] for s in stats["steps"]]
    assert "distill" in step_names
    # teacher 被冻结
    assert all(not p.requires_grad for p in teacher.parameters())


# ---------------------------------------------------------------------------
# 8. 蒸馏后小模型能力不大幅下降
# ---------------------------------------------------------------------------


def test_distill_capability_not_degraded():
    """蒸馏后 student 在 teacher 定义的任务上 loss 下降（能力转移）。"""
    teacher = _build_tlm(vocab=32, n_layer=2, n_embd=32, seed=1)
    student = _build_tlm(vocab=32, n_layer=1, n_embd=32, seed=2)  # 更小
    from verse_torch import no_grad
    from verse_torch.losses import cross_entropy

    # 构造训练/评估数据：y = teacher 的 argmax（teacher 定义"正确"标签）
    rng = np.random.RandomState(10)
    train_batches = []
    for _ in range(3):
        xb = rng.randint(0, 32, size=(4, 8))
        with no_grad():
            tlog = teacher(xb)  # (B, T, V)
        yb = tlog.data.argmax(axis=-1)  # (B, T)
        train_batches.append((xb, yb))
    # 评估 batch（独立 seed）
    rng2 = np.random.RandomState(99)
    eval_x = rng2.randint(0, 32, size=(4, 8))
    with no_grad():
        eval_y = teacher(eval_x).data.argmax(axis=-1)

    # 蒸馏前 student 评估 loss
    with no_grad():
        before = float(cross_entropy(student(eval_x), eval_y).data)

    # 蒸馏（启用特征级蒸馏）
    distiller = KnowledgeDistiller(
        teacher, student, temperature=4.0, alpha=0.7,
        feature_loss_weight=0.3,
    )
    distiller.distill(
        train_batches, epochs=5, lr=5e-3,
        feature_extractor=_lm_feature_extractor,
    )

    # 蒸馏后 student 评估 loss
    with no_grad():
        after = float(cross_entropy(student(eval_x), eval_y).data)

    assert after < before, (
        f"蒸馏后能力应不下降（loss 应降低）：before={before:.4f} after={after:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. 兼容性：无 teacher 仅 prune+quantize+lora + V1.0 旧 API
# ---------------------------------------------------------------------------


def test_compress_pipeline_v13_no_teacher_compat():
    """不指定 teacher_model 时仅 prune+quantize+lora（无 distill 步骤）。"""
    model = _build_tlm(seed=1)
    new_model, stats = compress_pipeline(
        model,
        {"prune": {"sparsity": 0.3}, "quantize": {"bits": 8},
         "lora": {"rank": 4}},
        return_stats=True,
    )
    step_names = [s["step"] for s in stats["steps"]]
    assert step_names == ["prune", "quantize", "lora"]
    assert "distill" not in step_names
    assert stats["qtype"] == "int8"


def test_compress_pipeline_v1_old_api_compat():
    """V1.0 旧 API（非 dict config）仍原地修改并返回 stats dict。"""
    model = _build_tlm(seed=1)
    orig_params = count_parameters(model)
    stats = compress_pipeline(
        model, target_ratio=0.1, qtype="int4",
        sparsity=0.3, use_lora=False,
    )
    assert isinstance(stats, dict)
    assert "compression_ratio" in stats
    assert stats["original_params"] == orig_params
    # 旧 API 原地修改：模型已被量化（含 QLinear）
    has_qlinear = any(isinstance(m, QLinear) for _, m in model.named_modules())
    assert has_qlinear, "旧 API 应原地量化模型"


def test_distill_only_backward_compat():
    """distill_only 旧接口（max_steps + T + alpha）仍工作。"""
    from verse_torch.compress import distill_only
    teacher = _build_tlm(seed=1)
    student = _build_tlm(seed=2)
    x, y = _rand_batch(seed=3)
    result = distill_only(
        teacher, student, [(x, y)], max_steps=3, T=2.0, alpha=0.5, lr=1e-3)
    assert result is student


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
