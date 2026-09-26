#!/usr/bin/env python3
"""扩充 CometSpark 训练数据。

通过模板组合生成中英文语料（算术、事实句、日常句式等），与原有
train/val.jsonl 合并去重后写回。确定性输出（固定随机种子），可重复运行。

用法::

    python3 scripts/gen_data.py [--train-size 4500] [--val-size 200]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_existing(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(l)["text"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 语料模板

def _expand(rng: random.Random, items: list[str], n: int) -> list[str]:
    """返回恰好 n 条：候选不足 n 时循环复用并打乱（避免死循环）。"""
    items = list(dict.fromkeys(items))
    if len(items) >= n:
        return rng.sample(items, n)
    reps = n // len(items) + 1
    out = (items * reps)[:n]
    rng.shuffle(out)
    return out


def zh_fact_sentences() -> list[str]:
    countries = ["中国", "美国", "日本", "法国", "德国", "英国", "加拿大", "澳大利亚",
                 "韩国", "印度", "巴西", "俄罗斯", "意大利", "西班牙", "埃及"]
    capitals = ["北京", "华盛顿", "东京", "巴黎", "柏林", "伦敦", "渥太华", "堪培拉",
                "首尔", "新德里", "巴西利亚", "莫斯科", "罗马", "马德里", "开罗"]
    rows = ["北京是中国的首都。", "华盛顿是美国的首都。", "东京是日本的首都。",
            "巴黎是法国的首都。", "柏林是德国的首都。", "伦敦是英国的首都。",
            "渥太华是加拿大的首都。", "堪培拉是澳大利亚的首都。", "首尔是韩国的首都。",
            "新德里是印度的首都。", "巴西利亚是巴西的首都。", "莫斯科是俄罗斯的首都。",
            "罗马是意大利的首都。", "马德里是西班牙的首都。", "开罗是埃及的首都。"]
    return rows


def zh_daily_sentences(rng: random.Random, n: int) -> list[str]:
    names = ["小明", "小红", "爷爷", "奶奶", "老师", "医生", "猫咪", "小狗", "机器人", "姐姐", "弟弟"]
    places = ["在花园里", "在教室里", "在图书馆", "在公园里", "在厨房里", "在操场上", "在河边", "在树下"]
    verbs = ["安静地读书", "认真地写字", "快乐地唱歌", "慢慢地散步", "专心地画画",
             "仔细地观察", "努力地学习", "开心地跑步", "静静地思考"]
    times = ["清晨", "上午", "中午", "下午", "傍晚", "晚上", "周末", "假期里"]
    candidates = [f"{t}，{nm}{p}{v}。" for t in times for nm in names for p in places for v in verbs]
    return _expand(rng, candidates, n)


def zh_tech_sentences(rng: random.Random, n: int) -> list[str]:
    subjects = ["机器学习", "深度学习", "自然语言处理", "计算机视觉", "人工智能",
                "神经网络", "大语言模型", "强化学习", "数据挖掘", "算法"]
    predicates = [
        "是人工智能的重要分支。",
        "正在快速发展。",
        "改变着我们的生活方式。",
        "需要大量的数据和算力。",
        "吸引了众多研究者。",
        "已经应用于日常生活。",
        "仍在不断演进。",
        "让计算机变得更聪明。",
    ]
    candidates = [s + p for s in subjects for p in predicates]
    return _expand(rng, candidates, n)


def zh_color_facts() -> list[str]:
    facts = {
        "苹果": ["红色", "绿色"], "香蕉": ["黄色"], "西瓜": ["绿色", "黑色"],
        "天空": ["蓝色"], "草地": ["绿色"], "雪": ["白色"], "夜空": ["黑色"],
        "玫瑰": ["红色"], "柠檬": ["黄色"], "橘子": ["橙色"], "葡萄": ["紫色"],
        "大海": ["蓝色"], "火焰": ["红色"], "稻谷": ["金色"], "树叶": ["绿色"],
    }
    return [f"{k}是{c}的。" for k, cs in facts.items() for c in cs]


def zh_math(rng: random.Random, n: int) -> list[str]:
    pairs = [(a, b) for a in range(0, 100) for b in range(0, 100)]
    rng.shuffle(pairs)
    items = []
    for a, b in pairs[:n // 2]:
        items.append(f"{a}+{b}={a + b}" if rng.random() < 0.5 else f"{a}加{b}等于{a + b}。")
    for a, b in pairs[: n - len(items)]:
        items.append(f"{a}×{b}={a * b}" if rng.random() < 0.5 else f"{a}乘以{b}等于{a * b}。")
    return _expand(rng, items, n)


def zh_numbers(rng: random.Random, n: int) -> list[str]:
    items = [f"{x}的平方是{x * x}。" for x in range(1, 501)]
    items += [f"数字{k}是一个自然数。" for k in range(1, 1001)]
    return _expand(rng, items, n)


def en_daily_sentences(rng: random.Random, n: int) -> list[str]:
    colors = ["red", "black", "small", "clever", "happy", "young", "curious", "brave"]
    animals = ["cat", "dog", "bird", "fox", "rabbit", "horse", "bear", "monkey"]
    verbs = ["runs", "sleeps", "plays", "jumps", "walks", "sits", "eats", "rests"]
    places = ["in the garden", "on the farm", "in the forest", "near the river",
              "in the park", "under the tree", "on the hill", "in the yard"]
    candidates = [f"The {c} {a} {v} {p}." for c in colors for a in animals for v in verbs for p in places]
    return _expand(rng, candidates, n)


def en_tech_sentences(rng: random.Random, n: int) -> list[str]:
    subjects = ["Machine learning", "Deep learning", "Natural language processing",
                "Computer vision", "Artificial intelligence", "Neural networks",
                "Large language models", "Reinforcement learning", "Data science", "Python"]
    predicates = ["is a branch of AI.", "is widely used in industry.",
                  "learns patterns from data.", "requires large datasets.",
                  "is changing the world.", "is an important research topic.",
                  "helps computers understand humans.", "is a popular programming language."]
    candidates = [f"{s} {p}" for s in subjects for p in predicates]
    return _expand(rng, candidates, n)


def en_math(rng: random.Random, n: int) -> list[str]:
    pairs = [(a, b) for a in range(0, 100) for b in range(0, 100)]
    rng.shuffle(pairs)
    items = [f"{a} plus {b} equals {a + b}." for a, b in pairs[: n // 2]]
    small = [(c, d) for c in range(1, 10) for d in range(1, 10)]
    need = n - len(items)
    items += [f"{c} times {d} is {c * d}." for c, d in (small * (need // len(small) + 1))[:need]]
    return _expand(rng, items, n)


def generate_corpus(rng: random.Random, train_size: int, val_size: int) -> tuple[list[str], list[str]]:
    groups = [
        zh_daily_sentences(rng, 1200),
        en_daily_sentences(rng, 1200),
        zh_tech_sentences(rng, 500),
        en_tech_sentences(rng, 500),
        zh_math(rng, 800),
        en_math(rng, 800),
        zh_numbers(rng, 500),
    ]
    facts = zh_fact_sentences() + zh_color_facts()

    # 每组先留出验证集份额（held-out，避免 train/val 泄漏）
    n_groups = len(groups) + 1
    per_group = max(val_size // n_groups, 5)
    pool: list[str] = []
    val_pool: list[str] = []
    for group in groups:
        k = min(per_group, max(len(group) // 20, 5))
        val_part = rng.sample(group, k)
        val_pool.extend(val_part)
        pool.extend(s for s in group if s not in val_part)
    # 事实句短小、信息密度高，重复 2 次加入训练池
    pool.extend(facts * 2)
    val_pool.extend(facts)

    rng.shuffle(pool)
    rng.shuffle(val_pool)

    train_gen = pool[: max(0, train_size - 105)]
    val_gen = val_pool[: max(0, val_size - 5)]
    return train_gen, val_gen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=4500, help="扩充后训练集总条数")
    parser.add_argument("--val-size", type=int, default=200, help="扩充后验证集总条数")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    old_train = load_existing(DATA_DIR / "train.jsonl")
    old_val = load_existing(DATA_DIR / "val.jsonl")
    gen_train, gen_val = generate_corpus(rng, args.train_size, args.val_size)

    train_texts = list(dict.fromkeys(old_train + gen_train))
    val_texts = list(dict.fromkeys(old_val + gen_val))
    # 防止 train/val 泄漏
    leak = set(val_texts) & set(train_texts)
    if leak:
        train_texts = [t for t in train_texts if t not in leak]

    write_jsonl(DATA_DIR / "train.jsonl", train_texts)
    write_jsonl(DATA_DIR / "val.jsonl", val_texts)
    print(f"train.jsonl: {len(train_texts)} 条 (原 {len(old_train)} 条)")
    print(f"val.jsonl:   {len(val_texts)} 条 (原 {len(old_val)} 条)")


if __name__ == "__main__":
    main()
