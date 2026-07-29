"""验证 small 模型训练数据（Part5K1.8 Task 12.2）。

测试 ``spark/small/data/train.jsonl`` 与 ``val.jsonl`` 的：
1. 文件存在性
2. 行数（train=40000，val=500）
3. 每行格式（prompt + completion 字段）
4. 重复率（< 1%）
5. 内容多样性（覆盖问答 / 翻译 / 代码 / 数学 等类别）

运行方式：
    cd /workspace && python -m pytest tests/test_train_data.py -v
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 路径定位
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "spark" / "small" / "data"
TRAIN_PATH = _DATA_DIR / "train.jsonl"
VAL_PATH = _DATA_DIR / "val.jsonl"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list:
    """读取 jsonl 文件，返回 dict 列表。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


class TestTrainData:
    """验证 small 模型训练数据。"""

    def test_train_jsonl_exists(self):
        """train.jsonl 文件存在。"""
        assert TRAIN_PATH.is_file(), f"训练数据文件不存在：{TRAIN_PATH}"

    def test_val_jsonl_exists(self):
        """val.jsonl 文件存在。"""
        assert VAL_PATH.is_file(), f"验证数据文件不存在：{VAL_PATH}"

    def test_train_jsonl_count(self):
        """train.jsonl 行数 = 40000。"""
        if not TRAIN_PATH.is_file():
            pytest.skip("train.jsonl 不存在")
        items = _read_jsonl(TRAIN_PATH)
        assert len(items) == 40000, (
            f"train.jsonl 行数应为 40000，实际 {len(items)}"
        )

    def test_val_jsonl_count(self):
        """val.jsonl 行数 = 500。"""
        if not VAL_PATH.is_file():
            pytest.skip("val.jsonl 不存在")
        items = _read_jsonl(VAL_PATH)
        assert len(items) == 500, f"val.jsonl 行数应为 500，实际 {len(items)}"

    def test_format_prompt_completion(self):
        """抽样 10 条验证每条都有 prompt 和 completion 字段。"""
        if not TRAIN_PATH.is_file():
            pytest.skip("train.jsonl 不存在")
        items = _read_jsonl(TRAIN_PATH)
        assert len(items) > 0, "train.jsonl 为空"

        # 随机抽样 10 条（固定随机种子保证可复现）
        random.seed(42)
        sample_size = min(10, len(items))
        samples = random.sample(items, sample_size)

        for i, item in enumerate(samples):
            assert isinstance(item, dict), f"第 {i} 条不是 dict：{type(item)}"
            assert "prompt" in item, f"第 {i} 条缺少 prompt 字段"
            assert "completion" in item, f"第 {i} 条缺少 completion 字段"
            assert isinstance(item["prompt"], str), (
                f"第 {i} 条 prompt 不是字符串：{type(item['prompt'])}"
            )
            assert isinstance(item["completion"], str), (
                f"第 {i} 条 completion 不是字符串：{type(item['completion'])}"
            )
            assert len(item["prompt"]) > 0, f"第 {i} 条 prompt 为空"
            assert len(item["completion"]) > 0, f"第 {i} 条 completion 为空"

    def test_low_duplication(self):
        """验证重复率 < 1%（按 prompt+completion 完整对去重）。"""
        if not TRAIN_PATH.is_file():
            pytest.skip("train.jsonl 不存在")
        items = _read_jsonl(TRAIN_PATH)
        assert len(items) > 0, "train.jsonl 为空"

        # 计算重复率
        seen = set()
        dup_count = 0
        for item in items:
            key = (item.get("prompt", ""), item.get("completion", ""))
            if key in seen:
                dup_count += 1
            else:
                seen.add(key)

        dup_rate = dup_count / len(items)
        assert dup_rate < 0.01, (
            f"重复率 {dup_rate * 100:.2f}% 超过 1%（"
            f"重复 {dup_count} 条 / 总 {len(items)} 条）"
        )

    def test_content_diversity(self):
        """验证覆盖多个类别（至少包含问答 / 翻译 / 代码 / 数学 等关键词）。"""
        if not TRAIN_PATH.is_file():
            pytest.skip("train.jsonl 不存在")
        items = _read_jsonl(TRAIN_PATH)
        assert len(items) > 0, "train.jsonl 为空"

        # 把所有 prompt + completion 拼成大文本用于关键词检测
        # （只取前 5000 条样本，避免字符串过大）
        sample = items[:5000]
        all_text = " ".join(
            (it.get("prompt", "") + " " + it.get("completion", "")) for it in sample
        )

        # 类别关键词（与 generate_train_data.py 的 8 大类对应）
        # 问答 / 翻译 / 代码 / 数学 / 对话 / 续写 / 指令 / 知识
        category_keywords = {
            "问答": ["首都是", "化学式", "什么人", "哪一年", "谁", "哪里"],
            "翻译": ["翻译", "Translate", "译", "morning", "hello", "Hello"],
            "代码": ["def", "Python", "函数", "代码", "function", "lambda"],
            "数学": ["计算", "=", "+", "-", "×", "÷", "加", "减", "乘", "除"],
            "对话": ["用户：", "助手：", "回复", "回应", "聊天"],
            "续写": ["续写", "续", "春眠", "床前明月光"],
            "指令": ["转换", "格式化", "换算", "请执行", "任务"],
            "知识": ["什么是", "解释", "定义", "是什么"],
        }

        # 至少覆盖 4 个类别（保证多样性）
        covered = []
        for cat, keywords in category_keywords.items():
            if any(kw in all_text for kw in keywords):
                covered.append(cat)
        assert len(covered) >= 4, (
            f"内容多样性不足，仅覆盖 {len(covered)} 个类别（"
            f"{covered}），应至少覆盖 4 个类别"
        )

    def test_no_garbled_text(self):
        """抽样验证不含乱码（U+FFFD 替换字符）。"""
        if not TRAIN_PATH.is_file():
            pytest.skip("train.jsonl 不存在")
        items = _read_jsonl(TRAIN_PATH)
        # 抽样 100 条
        random.seed(123)
        sample = random.sample(items, min(100, len(items)))
        for i, item in enumerate(sample):
            text = item.get("prompt", "") + item.get("completion", "")
            assert "\ufffd" not in text, (
                f"第 {i} 条含 U+FFFD 替换字符（可能乱码）：{text[:50]}"
            )

    def test_val_is_subset_of_train_format(self):
        """验证 val.jsonl 与 train.jsonl 格式一致（都有 prompt + completion）。"""
        if not VAL_PATH.is_file():
            pytest.skip("val.jsonl 不存在")
        items = _read_jsonl(VAL_PATH)
        assert len(items) > 0, "val.jsonl 为空"
        for i, item in enumerate(items[:20]):  # 抽样前 20 条
            assert "prompt" in item, f"val 第 {i} 条缺少 prompt 字段"
            assert "completion" in item, f"val 第 {i} 条缺少 completion 字段"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
