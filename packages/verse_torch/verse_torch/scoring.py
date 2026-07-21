"""模型生成质量打分：exact_match / prefix_accuracy / char_f1 / bleu / rouge_l。"""
import math
from collections import Counter


def exact_match(prediction: str, reference: str) -> float:
    """精确匹配率：1.0 完全相等，0.0 不等。"""
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def prefix_accuracy(prediction: str, reference: str) -> float:
    """前缀匹配率：prediction 的前缀与 reference 重合的比例。

    适合续写任务：模型生成的开头有多少字符与参考答案一致。
    """
    pred = prediction.strip()
    ref = reference.strip()
    if not ref:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    # 找最长公共前缀
    common_len = 0
    for i in range(min(len(pred), len(ref))):
        if pred[i] == ref[i]:
            common_len += 1
        else:
            break
    return common_len / len(ref)


def char_f1(prediction: str, reference: str) -> float:
    """字符级 F1。

    把 prediction 和 reference 看作字符的多重集，
    计算 precision / recall / F1。
    """
    pred = prediction.strip()
    ref = reference.strip()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counter = Counter(pred)
    ref_counter = Counter(ref)
    common = pred_counter & ref_counter
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / sum(pred_counter.values())
    recall = num_common / sum(ref_counter.values())
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens, n):
    """生成 n-gram。"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """BLEU-4（简化版，无 smoothing）。

    Args:
        prediction: 预测文本
        reference: 参考文本
        max_n: 最大 n-gram（默认 4，即 BLEU-4）

    Returns:
        BLEU 分数（0-1）
    """
    pred_tokens = list(prediction.strip())
    ref_tokens = list(reference.strip())
    if not pred_tokens or not ref_tokens:
        return 0.0

    # 计算 1-gram 到 max_n-gram 的 precision
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        # clip
        clipped = sum(min(c, ref_counter.get(ng, 0)) for ng, c in pred_counter.items())
        total = sum(pred_counter.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    # 几何平均
    if min(precisions) == 0:
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n

    # brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else \
        math.exp(1 - len(ref_tokens) / len(pred_tokens))

    return bp * math.exp(log_avg)


def _lcs_length(a, b):
    """最长公共子序列长度。"""
    m, n = len(a), len(b)
    # dp[i][j] = LCS(a[:i], b[:j])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L（F1 of LCS）。

    基于最长公共子序列计算 F1。
    """
    pred = list(prediction.strip())
    ref = list(reference.strip())
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


class ScoringEvaluator:
    """模型生成质量打分器。

    支持 5 个指标：
    - exact_match: 精确匹配率
    - prefix_accuracy: 前缀匹配率
    - char_f1: 字符级 F1
    - bleu: BLEU-4（简化版）
    - rouge_l: ROUGE-L

    用法：
        evaluator = ScoringEvaluator()
        scores = evaluator.evaluate(
            predictions=["你好世界", "床前明月光"],
            references=["你好世界", "床前明月光，疑是地上霜"]
        )
        print(evaluator.report(scores))
    """

    METRICS = {
        "exact_match": exact_match,
        "prefix_accuracy": prefix_accuracy,
        "char_f1": char_f1,
        "bleu": bleu,
        "rouge_l": rouge_l,
    }

    def __init__(self, metrics=None):
        """初始化。

        Args:
            metrics: list[str]，要计算的指标名（默认全部）
        """
        if metrics is None:
            self.metrics = list(self.METRICS.keys())
        else:
            # 校验指标名合法
            for m in metrics:
                if m not in self.METRICS:
                    raise ValueError(
                        f"未知指标 {m!r}，可选：{list(self.METRICS.keys())}"
                    )
            self.metrics = list(metrics)

    def score_pair(self, prediction: str, reference: str) -> dict:
        """计算单个 (prediction, reference) 对的所有指标。"""
        return {m: self.METRICS[m](prediction, reference) for m in self.metrics}

    def evaluate(self, predictions, references) -> dict:
        """批量计算指标。

        Args:
            predictions: list[str]，模型生成文本列表
            references: list[str]，参考答案列表

        Returns:
            dict: {
                "exact_match": 0.4,
                "prefix_accuracy": 0.65,
                "char_f1": 0.72,
                "bleu": 0.32,
                "rouge_l": 0.55,
                "n_samples": 10,
                "per_sample": [...]
            }
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions({len(predictions)}) 与 references({len(references)}) "
                f"长度不一致"
            )

        if len(predictions) == 0:
            empty_result = {m: 0.0 for m in self.metrics}
            empty_result["n_samples"] = 0
            empty_result["per_sample"] = []
            return empty_result

        per_sample = []
        for pred, ref in zip(predictions, references):
            per_sample.append(self.score_pair(pred, ref))

        # 平均
        result = {}
        for m in self.metrics:
            values = [s[m] for s in per_sample]
            result[m] = sum(values) / len(values) if values else 0.0
        result["n_samples"] = len(predictions)
        result["per_sample"] = per_sample
        return result

    def report(self, score_dict: dict) -> str:
        """生成可读报告。"""
        lines = ["=" * 50, "评分报告", "=" * 50]
        lines.append(f"样本数: {score_dict.get('n_samples', 0)}")
        lines.append("-" * 50)
        for m in self.metrics:
            if m in score_dict:
                lines.append(f"  {m:20s}: {score_dict[m]:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


__all__ = [
    "ScoringEvaluator",
    "exact_match",
    "prefix_accuracy",
    "char_f1",
    "bleu",
    "rouge_l",
]
