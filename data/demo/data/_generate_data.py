"""生成 CometSpark PoC 训练 / 验证数据。

数据混合：
- 唐诗（5-7 言绝句）：50 条
- 简单问答（"问：xxx 答：yyy"）：50 条
- 数字序列（"1,2,3,4,5"）：50 条
- 简单英文短句：50 条
合计 200 条训练数据，验证集从同模板独立生成 50 条。

设计目标：让 LM 能在 200 步内学到明显模式（固定句式、数字递增）。
"""

from __future__ import annotations

import json
import os
import random


# ---------------------------------------------------------------------------
# 数据生成模板
# ---------------------------------------------------------------------------


def gen_tang_poems(n: int, rng: random.Random) -> list[str]:
    """唐诗 5-7 言绝句模板生成。

    用一组真实名句 + 模板拼接，让模型学到"五言/七言 + 句号"模式。
    """
    real_poems = [
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
        "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
        "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
        "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
        "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
        "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
        "墙角数枝梅，凌寒独自开。遥知不是雪，为有暗香来。",
        "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
        "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
        "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
        "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
        "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
    ]
    out = []
    for i in range(n):
        poem = real_poems[i % len(real_poems)]
        # 略加变换：在不同位置加序号或重复，增加多样性
        if i % 3 == 0:
            out.append(f"诗云：{poem}")
        elif i % 3 == 1:
            out.append(poem)
        else:
            out.append(f"唐诗一首：{poem}")
    rng.shuffle(out)
    return out


def gen_qa(n: int, rng: random.Random) -> list[str]:
    """简单问答模板。"""
    qa_pairs = [
        ("你好", "你好，很高兴见到你。"),
        ("你叫什么名字", "我叫 CometSpark，是一个语言模型。"),
        ("1+1等于几", "1+1=2。"),
        ("2+3等于几", "2+3=5。"),
        ("什么是人工智能", "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"),
        ("唐诗是什么", "唐诗是唐代诗歌的统称，是中国古典诗歌的高峰。"),
        ("床前明月光的下一句", "床前明月光，疑是地上霜。"),
        ("白日依山尽的下一句", "白日依山尽，黄河入海流。"),
        ("春天的特点", "春天气候温暖，万物复苏，百花盛开。"),
        ("Python 是什么", "Python 是一种解释型、高级编程语言。"),
        ("数学有什么用", "数学是研究数量、结构、变化、空间以及信息等概念的学科，是科学技术的基础。"),
        ("早上好", "早上好，祝你有美好的一天。"),
        ("谢谢", "不客气，有问题尽管问。"),
        ("再见", "再见，期待下次见面。"),
        ("今天天气如何", "抱歉，我无法获取实时天气，请查看天气应用。"),
    ]
    out = []
    for i in range(n):
        q, a = qa_pairs[i % len(qa_pairs)]
        out.append(f"问：{q} 答：{a}")
    rng.shuffle(out)
    return out


def gen_number_seqs(n: int, rng: random.Random) -> list[str]:
    """数字序列模板：让模型学到"逗号分隔、递增"的模式。"""
    out = []
    for i in range(n):
        start = rng.randint(1, 10)
        step = rng.choice([1, 1, 1, 2, 2, 3])
        length = rng.randint(5, 12)
        nums = [start + step * j for j in range(length)]
        out.append(",".join(str(x) for x in nums))
    rng.shuffle(out)
    return out


def gen_english(n: int, rng: random.Random) -> list[str]:
    """简单英文短句模板。"""
    templates = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and natural language processing.",
        "Python is a great programming language.",
        "The sun rises in the east and sets in the west.",
        "Knowledge is power.",
        "Practice makes perfect.",
        "A journey of a thousand miles begins with a single step.",
        "Time is money.",
        "Better late than never.",
        "Actions speak louder than words.",
        "The early bird catches the worm.",
        "Where there is a will, there is a way.",
        "Learning never exhausts the mind.",
        "Music is the universal language of mankind.",
    ]
    out = []
    for i in range(n):
        out.append(templates[i % len(templates)])
    rng.shuffle(out)
    return out


def build_dataset(n_per_class: int, seed: int) -> list[dict]:
    """生成完整数据集（n_per_class 条/类 × 4 类）。"""
    rng = random.Random(seed)
    items = []
    items.extend(gen_tang_poems(n_per_class, rng))
    items.extend(gen_qa(n_per_class, rng))
    items.extend(gen_number_seqs(n_per_class, rng))
    items.extend(gen_english(n_per_class, rng))
    rng.shuffle(items)
    return [{"text": t} for t in items]


def write_jsonl(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    # 训练集：每类 50 条 × 4 = 200 条
    train_items = build_dataset(n_per_class=50, seed=42)
    # 验证集：每类 ~13 条 × 4 ≈ 52 条
    val_items = build_dataset(n_per_class=13, seed=123)
    write_jsonl(os.path.join(base, "train.jsonl"), train_items)
    write_jsonl(os.path.join(base, "val.jsonl"), val_items)
    print(f"生成 train.jsonl: {len(train_items)} 条")
    print(f"生成 val.jsonl: {len(val_items)} 条")


if __name__ == "__main__":
    main()
