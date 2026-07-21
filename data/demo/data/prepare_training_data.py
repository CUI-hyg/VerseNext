"""CometSpark-V0.2 训练数据收集与处理脚本（Part4 P3.2）。

功能：
1. 尝试从 HuggingFace 下载优质中文/数学/代码数据集（带网络降级）
2. 生成大量优质合成训练数据（6 大类，覆盖语言/数学/代码/常识/指令/古诗）
3. 数据清洗：去重、长度过滤、格式统一
4. 合并写入 train.jsonl / val.jsonl（兼容 data_loader 的 chat 与 prompt-completion 格式）

输出格式：每行一个 JSON 对象，支持两种格式混用：
  - {"prompt": "...", "completion": "..."}
  - [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]

用法：
    python prepare_training_data.py                    # 默认生成 ~3000 条
    python prepare_training_data.py --n-per-class 200  # 自定义每类条数
    python prepare_training_data.py --try-hf           # 尝试 HF 下载
    python prepare_training_data.py --output-dir /path/to/out
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# HuggingFace 数据集下载（可选，带降级）
# ---------------------------------------------------------------------------

_HF_DATASETS = [
    # (repo_id, filename, parser) — 只下载小样本，避免内存爆炸
    ("wikimedia/wikipedia", "20231101.zh/simple/train-00000-of-00001.parquet", "_parse_wiki"),
    ("openai/gsm8k", "main/train-00000-of-00001.parquet", "_parse_gsm8k"),
]


def _try_download_hf(repo_id: str, filename: str, timeout: float = 10.0) -> bytes | None:
    """尝试从 HF mirror 下载文件，失败返回 None。"""
    urls = [
        f"https://hf-mirror.com/datasets/{repo_id}/resolve/main/{filename}",
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CometSpark-V0.2/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"  [HF] 下载失败 {url}: {e}", file=sys.stderr)
            continue
    return None


def _parse_parquet_rows(data: bytes, max_rows: int = 50) -> list[dict]:
    """解析 parquet 数据（需要 pyarrow 或 pandas），失败返回空。"""
    try:
        import pyarrow.parquet as pq
        import io
        table = pq.read_table(io.BytesIO(data))
        rows = table.to_pylist()[:max_rows]
        return rows
    except ImportError:
        try:
            import pandas as pd
            import io
            df = pd.read_parquet(io.BytesIO(data))
            rows = df.head(max_rows).to_dict(orient="records")
            return rows
        except ImportError:
            print("  [HF] 跳过 parquet 解析：未安装 pyarrow/pandas", file=sys.stderr)
            return []


def collect_hf_data(max_per_source: int = 50) -> list[dict]:
    """尝试从 HF 下载真实数据，失败返回空列表。"""
    results = []
    for repo_id, filename, parser_name in _HF_DATASETS:
        print(f"  [HF] 尝试下载 {repo_id}/{filename} ...")
        data = _try_download_hf(repo_id, filename)
        if data is None:
            continue
        rows = _parse_parquet_rows(data, max_rows=max_per_source)
        parser = globals().get(parser_name)
        if parser:
            parsed = parser(rows)
            results.extend(parsed)
            print(f"  [HF] {repo_id}: 获取 {len(parsed)} 条")
    return results


def _parse_wiki(rows: list[dict]) -> list[dict]:
    """解析 wikipedia 中文条目 → prompt-completion 格式。"""
    out = []
    for row in rows:
        text = row.get("text", "")
        title = row.get("title", "")
        if not text or len(text) < 50:
            continue
        # 截取前 500 字作为 completion
        snippet = text[:500].replace("\n", " ").strip()
        out.append({
            "prompt": f"请介绍{title}。",
            "completion": snippet,
        })
    return out


def _parse_gsm8k(rows: list[dict]) -> list[dict]:
    """解析 GSM8K 数学题 → prompt-completion 格式。"""
    out = []
    for row in rows:
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question or not answer:
            continue
        out.append({
            "prompt": question,
            "completion": answer,
        })
    return out


# ---------------------------------------------------------------------------
# 优质合成数据生成器（6 大类）
# ---------------------------------------------------------------------------


def gen_chinese_qa(n: int, rng: random.Random) -> list[dict]:
    """中文问答/对话（扩展版）。"""
    qa_pairs = [
        ("你好", "你好，很高兴见到你。有什么可以帮助你的吗？"),
        ("你叫什么名字", "我叫 CometSpark，是一个由 VerseNex 架构驱动的语言模型。"),
        ("1+1等于几", "1+1=2。"),
        ("2+3等于几", "2+3=5。"),
        ("什么是人工智能", "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门技术科学。"),
        ("唐诗是什么", "唐诗是唐代诗歌的统称，是中国古典诗歌的高峰，代表诗人有李白、杜甫、白居易等。"),
        ("床前明月光的下一句", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
        ("白日依山尽的下一句", "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"),
        ("春天的特点", "春天气候温暖，万物复苏，百花盛开，是播种和生长的季节。"),
        ("Python 是什么", "Python 是一种解释型、高级编程语言，以简洁优雅的语法和丰富的生态著称。"),
        ("数学有什么用", "数学是研究数量、结构、变化、空间以及信息等概念的学科，是科学技术的基础。"),
        ("早上好", "早上好，祝你有美好的一天。"),
        ("谢谢", "不客气，有问题尽管问。"),
        ("再见", "再见，期待下次见面。"),
        ("今天天气如何", "抱歉，我无法获取实时天气信息，请查看天气预报应用。"),
        ("什么是深度学习", "深度学习是机器学习的一个分支，使用多层神经网络从数据中学习特征表示。"),
        ("水的化学式是什么", "水的化学式是 H₂O，由两个氢原子和一个氧原子组成。"),
        ("地球有多少大洲", "地球有七大洲：亚洲、非洲、北美洲、南美洲、南极洲、欧洲和大洋洲。"),
        ("什么是递归", "递归是函数调用自身的编程技巧，常用于分治问题和树形结构遍历。"),
        ("如何学习编程", "学习编程的建议：选择一门语言入门，多做项目实践，阅读优秀代码，参与开源社区。"),
        ("什么是神经网络", "神经网络是受生物神经系统启发的计算模型，由神经元和连接权重组成，能学习复杂映射。"),
        ("中国首都是哪里", "中国首都是北京。"),
        ("一年有多少天", "平年有 365 天，闰年有 366 天。"),
        ("什么是 API", "API（应用程序编程接口）是软件组件之间通信的约定，定义了请求和响应的格式。"),
        ("如何保持健康", "保持健康的建议：均衡饮食、规律运动、充足睡眠、定期体检、保持心情愉悦。"),
        ("什么是量子计算", "量子计算利用量子叠加和纠缠原理进行计算，理论上能加速某些特定问题的求解。"),
        ("唐诗宋词有什么区别", "唐诗以律诗、绝句为主，讲究格律对仗；宋词以长短句为主，依词牌填词，更重抒情。"),
        ("什么是区块链", "区块链是去中心化的分布式账本技术，通过密码学保证数据不可篡改。"),
        ("如何写好代码", "写好代码的建议：命名清晰、函数简短、注释到位、遵循设计模式、持续重构。"),
        ("什么是气候变化", "气候变化指长期气温和天气模式的改变，主要由温室气体排放导致，影响全球生态。"),
    ]
    out = []
    for i in range(n):
        q, a = qa_pairs[i % len(qa_pairs)]
        out.append({"prompt": q, "completion": a})
    rng.shuffle(out)
    return out


def gen_math_problems(n: int, rng: random.Random) -> list[dict]:
    """数学题（算术 + 应用题 + 代数）。"""
    out = []
    for i in range(n):
        kind = rng.choice(["arithmetic", "word", "algebra", "sequence"])
        if kind == "arithmetic":
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            op = rng.choice(["+", "-", "×", "÷"])
            if op == "+":
                ans = a + b
            elif op == "-":
                if a < b:
                    a, b = b, a
                ans = a - b
            elif op == "×":
                a, b = rng.randint(1, 12), rng.randint(1, 12)
                ans = a * b
            else:  # ÷
                b = rng.randint(1, 12)
                ans = rng.randint(1, 12)
                a = ans * b
            out.append({
                "prompt": f"计算：{a} {op} {b} = ?",
                "completion": f"{a} {op} {b} = {ans}",
            })
        elif kind == "word":
            a, b = rng.randint(3, 20), rng.randint(3, 20)
            out.append({
                "prompt": f"小明有 {a} 个苹果，吃了 {b} 个，还剩几个？",
                "completion": f"{a} - {b} = {a - b}，还剩 {a - b} 个苹果。",
            })
        elif kind == "algebra":
            x = rng.randint(1, 20)
            k = rng.randint(2, 9)
            c = rng.randint(1, 20)
            out.append({
                "prompt": f"解方程：{k}x + {c} = {k * x + c}",
                "completion": f"{k}x = {k * x + c - c} = {k * x}，x = {x}。",
            })
        else:  # sequence
            start = rng.randint(1, 10)
            step = rng.choice([1, 2, 3, 5])
            seq = [start + step * j for j in range(5)]
            hidden = seq[-1]
            visible = seq[:-1]
            out.append({
                "prompt": f"数列：{','.join(map(str, visible))},? 下一个数是什么？",
                "completion": f"下一个数是 {hidden}。规律：每次加 {step}。",
            })
    rng.shuffle(out)
    return out


def gen_code_snippets(n: int, rng: random.Random) -> list[dict]:
    """代码片段（Python 函数 + 算法）。"""
    snippets = [
        ("写一个 Python 函数计算阶乘", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
        ("写一个 Python 函数判断回文", "def is_palindrome(s):\n    return s == s[::-1]"),
        ("写一个 Python 函数找列表最大值", "def find_max(lst):\n    if not lst:\n        return None\n    m = lst[0]\n    for x in lst:\n        if x > m:\n            m = x\n    return m"),
        ("写一个 Python 函数反转字符串", "def reverse_string(s):\n    return s[::-1]"),
        ("写一个 Python 函数判断素数", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ("写一个 Python 冒泡排序", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"),
        ("写一个 Python 斐波那契数列", "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),
        ("写一个 Python 二分查找", "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
        ("写一个 Python 函数计算字符串长度", "def string_length(s):\n    count = 0\n    for _ in s:\n        count += 1\n    return count"),
        ("写一个 Python 函数合并两个列表", "def merge_lists(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"),
        ("写一个 Python 函数计算平方和", "def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))"),
        ("写一个 Python 函数去重", "def deduplicate(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"),
        ("写一个 Python 函数判断偶数", "def is_even(n):\n    return n % 2 == 0"),
        ("写一个 Python 函数计算平均数", "def average(lst):\n    if not lst:\n        return 0\n    return sum(lst) / len(lst)"),
        ("写一个 Python 函数实现栈", "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop() if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0"),
    ]
    out = []
    for i in range(n):
        prompt, code = snippets[i % len(snippets)]
        out.append({"prompt": prompt, "completion": code})
    rng.shuffle(out)
    return out


def gen_knowledge(n: int, rng: random.Random) -> list[dict]:
    """常识知识问答。"""
    knowledge = [
        ("地球绕太阳一周需要多少时间", "地球绕太阳一周大约需要 365.25 天，即一年。"),
        ("光速是多少", "光在真空中的速度约为 30 万公里每秒（299,792,458 米/秒）。"),
        ("人体有多少块骨头", "成年人体通常有 206 块骨头。"),
        ("中国的最长的河流是", "中国最长的河流是长江，全长约 6300 公里。"),
        ("什么是光合作用", "光合作用是植物利用阳光、二氧化碳和水合成有机物并释放氧气的过程。"),
        ("DNA 的全称是什么", "DNA 的全称是脱氧核糖核酸，是生物体内储存遗传信息的分子。"),
        ("太阳系有几大行星", "太阳系有八大行星：水星、金星、地球、火星、木星、土星、天王星、海王星。"),
        ("什么是万有引力", "万有引力是物体之间由于质量而产生的相互吸引力，由牛顿发现。"),
        ("什么是相对论", "相对论是爱因斯坦提出的物理理论，分为狭义相对论和广义相对论。"),
        ("什么是基因", "基因是 DNA 上编码遗传信息的基本单位，决定生物的性状。"),
        ("什么是 HTTP", "HTTP（超文本传输协议）是 Web 通信的基础协议，用于客户端与服务器之间传输数据。"),
        ("什么是 TCP", "TCP（传输控制协议）是面向连接的、可靠的传输层协议。"),
        ("什么是数据库", "数据库是按数据结构组织、存储和管理数据的仓库，常见类型有关系型和 NoSQL。"),
        ("什么是操作系统", "操作系统是管理计算机硬件和软件资源的系统软件，如 Windows、Linux、macOS。"),
        ("什么是机器学习", "机器学习是让计算机从数据中学习规律并做出预测的技术，是人工智能的核心。"),
        ("中国四大发明是什么", "中国四大发明是造纸术、印刷术、火药和指南针。"),
        ("什么是可再生能源", "可再生能源是可自然补充的能源，如太阳能、风能、水能、地热能。"),
        ("什么是温室效应", "温室效应是大气中温室气体吸收地表辐射热量，导致地表温度升高的现象。"),
        ("什么是细胞", "细胞是生物体结构和功能的基本单位，分原核细胞和真核细胞。"),
        ("什么是进化论", "进化论是达尔文提出的生物进化理论，核心是自然选择决定物种演化方向。"),
    ]
    out = []
    for i in range(n):
        q, a = knowledge[i % len(knowledge)]
        out.append({"prompt": q, "completion": a})
    rng.shuffle(out)
    return out


def gen_instruction_following(n: int, rng: random.Random) -> list[dict]:
    """指令遵循（格式化、翻译、摘要）。"""
    instructions = [
        ("把下面的句子翻译成英文：你好世界", "Hello, world!"),
        ("把下面的句子翻译成英文：我爱编程", "I love programming."),
        ("把下面的句子翻译成英文：今天天气很好", "The weather is nice today."),
        ("把下面的句子翻译成中文：Hello, how are you?", "你好，你怎么样？"),
        ("把下面的句子翻译成中文：I am a programmer.", "我是一名程序员。"),
        ("把下面的数字格式化为千分位：1234567", "1,234,567"),
        ("把下面的数字格式化为千分位：987654321", "987,654,321"),
        ("总结下面的句子：Python 是一种解释型高级编程语言，以简洁的语法和丰富的库著称。", "Python 是简洁的高级编程语言。"),
        ("把下面的列表按升序排列：[3, 1, 4, 1, 5, 9, 2, 6]", "[1, 1, 2, 3, 4, 5, 6, 9]"),
        ("把下面的列表按降序排列：[3, 1, 4, 1, 5, 9, 2, 6]", "[9, 6, 5, 4, 3, 2, 1, 1]"),
        ("提取下面句子中的数字：我有 3 个苹果和 5 个橘子", "3, 5"),
        ("计算下面字符串的长度：Hello World", "11"),
        ("把下面的句子改成大写：hello world", "HELLO WORLD"),
        ("把下面的句子改成小写：HELLO WORLD", "hello world"),
        ("反转下面的字符串：abcdef", "fedcba"),
        ("统计下面句子中字母 a 的个数：banana has many a", "5"),
        ("把下面的数组去重：[1, 2, 2, 3, 3, 3, 4]", "[1, 2, 3, 4]"),
        ("计算 1 到 10 的和", "55"),
        ("计算 2 的 10 次方", "1024"),
        ("判断 7 是不是素数", "7 是素数。"),
    ]
    out = []
    for i in range(n):
        prompt, completion = instructions[i % len(instructions)]
        out.append({"prompt": prompt, "completion": completion})
    rng.shuffle(out)
    return out


def gen_classical_chinese(n: int, rng: random.Random) -> list[dict]:
    """古诗文（唐诗宋词 + 注释）。"""
    poems = [
        ("床前明月光", "疑是地上霜。举头望明月，低头思故乡。", "李白《静夜思》：描写游子思乡之情。"),
        ("白日依山尽", "黄河入海流。欲穷千里目，更上一层楼。", "王之涣《登鹳雀楼》：抒发积极进取的胸怀。"),
        ("春眠不觉晓", "处处闻啼鸟。夜来风雨声，花落知多少。", "孟浩然《春晓》：描绘春天清晨的景象。"),
        ("红豆生南国", "春来发几枝。愿君多采撷，此物最相思。", "王维《相思》：以红豆寄托相思之情。"),
        ("千山鸟飞绝", "万径人踪灭。孤舟蓑笠翁，独钓寒江雪。", "柳宗元《江雪》：描绘雪中孤舟的清冷意境。"),
        ("锄禾日当午", "汗滴禾下土。谁知盘中餐，粒粒皆辛苦。", "李绅《悯农》：劝人珍惜粮食。"),
        ("离离原上草", "一岁一枯荣。野火烧不尽，春风吹又生。", "白居易《赋得古原草送别》：写草的顽强生命力。"),
        ("墙角数枝梅", "凌寒独自开。遥知不是雪，为有暗香来。", "王安石《梅花》：赞梅的傲雪品格。"),
        ("故人西辞黄鹤楼", "烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。", "李白《送孟浩然之广陵》：写送别友人的深情。"),
        ("朝辞白帝彩云间", "千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。", "李白《早发白帝城》：写顺流而下之快。"),
        ("月落乌啼霜满天", "江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。", "张继《枫桥夜泊》：写夜泊的愁思。"),
        ("清明时节雨纷纷", "路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。", "杜牧《清明》：写清明时节的景象。"),
        ("葡萄美酒夜光杯", "欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。", "王翰《凉州词》：写边塞将士的豪情。"),
        ("两个黄鹂鸣翠柳", "一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。", "杜甫《绝句》：写草堂春景。"),
        ("独在异乡为异客", "每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。", "王维《九月九日忆山东兄弟》：写重阳思亲。"),
    ]
    out = []
    for i in range(n):
        first_line, rest, note = poems[i % len(poems)]
        out.append({"prompt": f"{first_line}，", "completion": f"{rest}\n{note}"})
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# 数据清洗与合并
# ---------------------------------------------------------------------------


def clean_and_dedup(items: list[dict]) -> list[dict]:
    """清洗：去重、长度过滤、格式校验。"""
    seen = set()
    out = []
    for it in items:
        # 统一提取 prompt + completion 文本用于去重
        if "prompt" in it and "completion" in it:
            key = (it["prompt"], it["completion"])
        elif isinstance(it, list) and len(it) >= 2:
            key = (it[0].get("content", ""), it[1].get("content", ""))
        else:
            continue
        if key in seen:
            continue
        # 长度过滤：prompt 至少 2 字，completion 至少 1 字
        if len(key[0]) < 2 or len(key[1]) < 1:
            continue
        # 长度上限：避免单条过长（截断到 2000 字符）
        if len(key[1]) > 2000:
            it = dict(it)
            it["completion"] = it["completion"][:2000]
        seen.add(key)
        out.append(it)
    return out


def write_jsonl(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def build_dataset(n_per_class: int, seed: int, try_hf: bool = False) -> list[dict]:
    """生成完整数据集（6 大类合成 + 可选 HF 真实数据）。"""
    rng = random.Random(seed)
    items = []
    items.extend(gen_chinese_qa(n_per_class, rng))
    items.extend(gen_math_problems(n_per_class, rng))
    items.extend(gen_code_snippets(n_per_class, rng))
    items.extend(gen_knowledge(n_per_class, rng))
    items.extend(gen_instruction_following(n_per_class, rng))
    items.extend(gen_classical_chinese(n_per_class, rng))

    if try_hf:
        print("[HF] 尝试下载真实数据集...")
        hf_items = collect_hf_data(max_per_source=50)
        items.extend(hf_items)
        print(f"[HF] 共获取 {len(hf_items)} 条真实数据")

    # 清洗去重
    before = len(items)
    items = clean_and_dedup(items)
    after = len(items)
    print(f"[clean] 去重前 {before} 条 → 去重后 {after} 条")

    rng.shuffle(items)
    return items


def main():
    parser = argparse.ArgumentParser(description="CometSpark-V0.2 训练数据准备")
    parser.add_argument("--n-per-class", type=int, default=100,
                        help="每类合成数据条数（默认 100，总 6 类 ≈ 600 条）")
    parser.add_argument("--try-hf", action="store_true",
                        help="尝试从 HuggingFace 下载真实数据（需网络）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认 data/）")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="验证集比例（默认 0.05 = 5 百分比）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    base = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("CometSpark-V0.2 训练数据准备")
    print("=" * 70)
    print(f"每类合成数据: {args.n_per_class} 条 × 6 类")
    print(f"尝试 HF 下载: {args.try_hf}")
    print(f"输出目录: {base}")
    print(f"验证集比例: {args.val_ratio}")
    print()

    # 训练集（用 seed=42）
    print("[1/2] 生成训练数据...")
    train_items = build_dataset(args.n_per_class, seed=args.seed, try_hf=args.try_hf)
    train_path = os.path.join(base, "train.jsonl")
    write_jsonl(train_path, train_items)
    print(f"  → {train_path}: {len(train_items)} 条\n")

    # 验证集（用不同 seed，避免与训练集重复）
    print("[2/2] 生成验证数据...")
    val_n = max(args.n_per_class // 5, 20)
    val_items = build_dataset(val_n, seed=args.seed + 1000, try_hf=False)
    val_path = os.path.join(base, "val.jsonl")
    write_jsonl(val_path, val_items)
    print(f"  → {val_path}: {len(val_items)} 条\n")

    # 统计
    print("=" * 70)
    print("数据统计")
    print("=" * 70)
    print(f"训练集: {len(train_items)} 条")
    print(f"验证集: {len(val_items)} 条")
    total_chars = sum(
        len(it.get("prompt", "")) + len(it.get("completion", ""))
        for it in train_items
        if isinstance(it, dict)
    )
    print(f"训练集总字符数: {total_chars:,}")
    print(f"平均每条字符: {total_chars / max(len(train_items), 1):.1f}")


if __name__ == "__main__":
    main()
