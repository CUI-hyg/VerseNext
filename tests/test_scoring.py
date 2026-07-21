"""ScoringEvaluator 及 5 个指标函数的测试。

覆盖：
- exact_match：完全匹配 / 部分匹配 / 空白处理
- prefix_accuracy：完整前缀 / 部分前缀 / 空 reference 边界
- char_f1：完全相同 / 无重合 / 部分重合
- bleu：完美 / 无重合 / brevity penalty
- rouge_l：完全相同 / 子序列匹配 / 无重合
- ScoringEvaluator：批量评估 / 报告格式 / 指标子集 / 空输入 / 长度不一致
"""
from __future__ import annotations

import math

import pytest

from verse_torch import (
    ScoringEvaluator,
    exact_match,
    prefix_accuracy,
    char_f1,
    bleu,
    rouge_l,
)


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_full():
    """完全匹配返回 1.0。"""
    assert exact_match("你好世界", "你好世界") == 1.0
    assert exact_match("hello", "hello") == 1.0


def test_exact_match_partial():
    """部分匹配返回 0.0。"""
    assert exact_match("你好世界", "你好") == 0.0
    assert exact_match("hello", "world") == 0.0
    assert exact_match("abc", "abcd") == 0.0


def test_exact_match_whitespace():
    """空白不影响（strip）。"""
    assert exact_match("  你好  ", "你好") == 1.0
    assert exact_match("hello\n", " hello ") == 1.0
    assert exact_match("\thello\t", "hello") == 1.0


# ---------------------------------------------------------------------------
# prefix_accuracy
# ---------------------------------------------------------------------------


def test_prefix_accuracy_full():
    """完整前缀匹配 1.0。"""
    assert prefix_accuracy("你好世界", "你好世界") == 1.0
    # prediction 比 reference 长，但前缀完全匹配
    assert prefix_accuracy("你好世界啊", "你好世界") == 1.0


def test_prefix_accuracy_partial():
    """部分前缀匹配。"""
    # pred="你好ABC", ref="你好世界" → 公共前缀 "你好" 长度 2, ref 长度 4
    assert prefix_accuracy("你好ABC", "你好世界") == pytest.approx(0.5)
    # 完全无公共前缀
    assert prefix_accuracy("XYZ", "你好世界") == 0.0


def test_prefix_accuracy_empty_ref():
    """空 reference 边界。"""
    # ref 为空、pred 也为空 → 1.0
    assert prefix_accuracy("", "") == 1.0
    assert prefix_accuracy("   ", "") == 1.0  # strip 后 pred 也为空
    # ref 为空、pred 非空 → 0.0
    assert prefix_accuracy("hello", "") == 0.0
    # pred 为空、ref 非空 → 0.0
    assert prefix_accuracy("", "hello") == 0.0


# ---------------------------------------------------------------------------
# char_f1
# ---------------------------------------------------------------------------


def test_char_f1_identical():
    """完全相同 1.0。"""
    assert char_f1("你好世界", "你好世界") == 1.0
    assert char_f1("hello", "hello") == 1.0
    # 空白 strip 后相同
    assert char_f1("  hello  ", "hello") == 1.0


def test_char_f1_no_overlap():
    """无重合 0.0。"""
    assert char_f1("abc", "xyz") == 0.0
    assert char_f1("你好", "世界") == 0.0


def test_char_f1_partial():
    """部分重合。"""
    # pred="abc", ref="abcd"
    # common = {a:1, b:1, c:1}, num_common=3
    # precision = 3/3 = 1.0, recall = 3/4 = 0.75
    # F1 = 2*1.0*0.75 / (1.0+0.75) = 1.5/1.75 ≈ 0.8571
    assert char_f1("abc", "abcd") == pytest.approx(2 * 1.0 * 0.75 / 1.75, rel=1e-6)
    # pred="aabc", ref="abc" → common={a:1, b:1, c:1}, num_common=3
    # precision=3/4=0.75, recall=3/3=1.0, F1=2*0.75*1.0/1.75
    assert char_f1("aabc", "abc") == pytest.approx(2 * 0.75 * 1.0 / 1.75, rel=1e-6)


# ---------------------------------------------------------------------------
# bleu
# ---------------------------------------------------------------------------


def test_bleu_perfect():
    """完美 BLEU=1.0。"""
    assert bleu("你好世界", "你好世界") == pytest.approx(1.0)
    assert bleu("hello world", "hello world") == pytest.approx(1.0)


def test_bleu_no_overlap():
    """无重合 BLEU=0.0。"""
    assert bleu("abc", "xyz") == 0.0
    assert bleu("你好", "世界") == 0.0


def test_bleu_brevity_penalty():
    """短预测触发 brevity penalty。

    pred="你好世界" (4 字), ref="你好世界啊" (5 字)
    所有 n-gram precision 均为 1.0（pred 是 ref 的前缀），
    bp = exp(1 - 5/4) = exp(-0.25) ≈ 0.7788
    """
    pred = "你好世界"
    ref = "你好世界啊"
    score = bleu(pred, ref)
    # 应大于 0（precisions 非零）且小于 1（brevity penalty 生效）
    assert 0.0 < score < 1.0
    expected_bp = math.exp(1 - len(ref) / len(pred))
    assert score == pytest.approx(expected_bp, rel=1e-6)
    # 对比：长度相等时 BLEU=1.0
    assert bleu("你好世界", "你好世界") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------


def test_rouge_l_identical():
    """完全相同 1.0。"""
    assert rouge_l("你好世界", "你好世界") == 1.0
    assert rouge_l("hello", "hello") == 1.0


def test_rouge_l_subsequence():
    """子序列匹配。

    pred="abc", ref="axbxc" → LCS="abc" 长度 3
    precision=3/3=1.0, recall=3/5=0.6
    F1=2*1.0*0.6/(1.0+0.6)=1.2/1.6=0.75
    """
    score = rouge_l("abc", "axbxc")
    assert score == pytest.approx(0.75, rel=1e-6)


def test_rouge_l_no_overlap():
    """无重合 0.0。"""
    assert rouge_l("abc", "xyz") == 0.0
    assert rouge_l("你好", "世界") == 0.0


# ---------------------------------------------------------------------------
# ScoringEvaluator
# ---------------------------------------------------------------------------


def test_scoring_evaluator_evaluate():
    """批量评估返回所有指标。"""
    evaluator = ScoringEvaluator()
    predictions = ["你好世界", "床前明月光"]
    references = ["你好世界", "床前明月光，疑是地上霜"]
    scores = evaluator.evaluate(predictions, references)

    # 应包含所有指标键
    for m in ["exact_match", "prefix_accuracy", "char_f1", "bleu", "rouge_l"]:
        assert m in scores, f"缺少指标 {m}"
        assert isinstance(scores[m], float)
        assert 0.0 <= scores[m] <= 1.0

    # 元数据
    assert scores["n_samples"] == 2
    assert len(scores["per_sample"]) == 2
    # per_sample 每个元素应包含所有指标
    for sample in scores["per_sample"]:
        for m in ["exact_match", "prefix_accuracy", "char_f1", "bleu", "rouge_l"]:
            assert m in sample

    # 第一条完全匹配 → exact_match=1.0
    assert scores["per_sample"][0]["exact_match"] == 1.0
    # 整体 exact_match 平均 = (1.0 + 0.0) / 2 = 0.5
    assert scores["exact_match"] == pytest.approx(0.5)


def test_scoring_evaluator_report():
    """报告格式正确。"""
    evaluator = ScoringEvaluator()
    scores = evaluator.evaluate(["你好"], ["你好"])
    report = evaluator.report(scores)

    assert isinstance(report, str)
    assert "评分报告" in report
    assert "样本数" in report
    assert "exact_match" in report
    assert "prefix_accuracy" in report
    assert "char_f1" in report
    assert "bleu" in report
    assert "rouge_l" in report
    # 报告应包含分隔线
    assert "=" in report
    assert "-" in report


def test_scoring_evaluator_metrics_subset():
    """只计算部分指标。"""
    evaluator = ScoringEvaluator(metrics=["exact_match", "char_f1"])
    scores = evaluator.evaluate(["你好", "世界"], ["你好", "你好"])

    # 应只包含指定的 2 个指标
    assert "exact_match" in scores
    assert "char_f1" in scores
    assert "prefix_accuracy" not in scores
    assert "bleu" not in scores
    assert "rouge_l" not in scores

    # per_sample 也应只包含指定指标
    for sample in scores["per_sample"]:
        assert set(sample.keys()) == {"exact_match", "char_f1"}

    # exact_match = (1.0 + 0.0) / 2 = 0.5
    assert scores["exact_match"] == pytest.approx(0.5)


def test_scoring_evaluator_empty():
    """空输入边界。"""
    evaluator = ScoringEvaluator()
    scores = evaluator.evaluate([], [])

    assert scores["n_samples"] == 0
    assert scores["per_sample"] == []
    # 所有指标默认 0.0
    for m in ["exact_match", "prefix_accuracy", "char_f1", "bleu", "rouge_l"]:
        assert scores[m] == 0.0

    # 报告也应能正常生成
    report = evaluator.report(scores)
    assert "样本数: 0" in report


def test_scoring_evaluator_length_mismatch():
    """长度不一致抛 ValueError。"""
    evaluator = ScoringEvaluator()
    with pytest.raises(ValueError, match="长度不一致"):
        evaluator.evaluate(["a", "b"], ["a"])

    with pytest.raises(ValueError):
        evaluator.evaluate(["a"], ["a", "b"])
