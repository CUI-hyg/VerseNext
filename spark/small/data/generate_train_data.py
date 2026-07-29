#!/usr/bin/env python3
"""生成 small 模型训练数据（prompt-completion 格式）。

生成 8 大类共 40000 条训练数据：
- 问答（常识 / 科学 / 地理 / 历史）
- 翻译（中英互译）
- 代码（Python 基础 / 算法 / 数据结构）
- 数学（算术 / 代数 / 几何）
- 对话（日常 / 情感 / 建议）
- 续写（诗词 / 故事 / 描述）
- 指令（格式化 / 转换 / 摘要）
- 知识（定义 / 解释 / 对比）

每类 5000 条，输出 JSONL 格式，每行：{"prompt": "...", "completion": "..."}

用法：
    python spark/small/data/generate_train_data.py
    python spark/small/data/generate_train_data.py --num-train 40000 --num-val 500

依赖：仅 Python 标准库（不依赖项目其他模块）。
"""

import argparse
import json
import os
import random
from typing import Dict, List

random.seed(42)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_PATH = os.path.join(_SCRIPT_DIR, "train.jsonl")
DEFAULT_VAL_PATH = os.path.join(_SCRIPT_DIR, "val.jsonl")
PER_CATEGORY = 5000


def _dedup_and_trim(items: List[Dict], target: int) -> List[Dict]:
    """对 items 去重并裁剪到 target 条。"""
    seen = set()
    unique = []
    for it in items:
        key = (it["prompt"], it["completion"])
        if key not in seen:
            seen.add(key)
            unique.append(it)
        if len(unique) >= target:
            break
    i = 0
    while len(unique) < target and items:
        cand = items[i % len(items)]
        key = (cand["prompt"], cand["completion"])
        if key not in seen:
            seen.add(key)
            unique.append(cand)
        i += 1
        if i > target * 10:
            break
    return unique[:target]


def _expand_pairs(pairs: List[tuple], templates: List[str], limit: int) -> List[Dict]:
    """对 (subject, completion) × 模板 笛卡尔积，shuffle 后取 limit 条。"""
    out = []
    for tpl in templates:
        for subj, comp in pairs:
            prompt = tpl.format(p=subj)
            out.append({"prompt": prompt, "completion": comp})
        if len(out) >= limit * 3:
            break
    random.shuffle(out)
    return out


_REPHRASE_TPL = [
    "{p}", "问：{p}", "请回答：{p}", "请问，{p}", "请告诉我，{p}",
    "你知道{p}", "我想知道{p}", "请简短回答：{p}", "请用一句话回答：{p}",
    "{p}请回答。", "请说明{p}", "请解释一下{p}", "麻烦解答：{p}",
    "求解答：{p}", "帮我回答{p}", "我想问{p}", "请教一下{p}",
    "请简要回答{p}", "请回答问题：{p}", "{p}答案是什么？",
    "请回答下面的问题：{p}", "答：{p}", "Q：{p}", "问题：{p}",
    "{p}请给出答案。",
]


# ===========================================================================
# 1. 问答（常识 / 科学 / 地理 / 历史）
# ===========================================================================

_COUNTRY_CAPITAL = [
    ("中国", "北京"), ("美国", "华盛顿特区"), ("日本", "东京"), ("英国", "伦敦"),
    ("法国", "巴黎"), ("德国", "柏林"), ("俄罗斯", "莫斯科"), ("意大利", "罗马"),
    ("西班牙", "马德里"), ("葡萄牙", "里斯本"), ("荷兰", "阿姆斯特丹"),
    ("比利时", "布鲁塞尔"), ("瑞士", "伯尔尼"), ("奥地利", "维也纳"),
    ("瑞典", "斯德哥尔摩"), ("挪威", "奥斯陆"), ("芬兰", "赫尔辛基"),
    ("丹麦", "哥本哈根"), ("波兰", "华沙"), ("捷克", "布拉格"),
    ("匈牙利", "布达佩斯"), ("希腊", "雅典"), ("土耳其", "安卡拉"),
    ("埃及", "开罗"), ("南非", "比勒陀利亚"), ("尼日利亚", "阿布贾"),
    ("肯尼亚", "内罗毕"), ("摩洛哥", "拉巴特"), ("加拿大", "渥太华"),
    ("墨西哥", "墨西哥城"), ("巴西", "巴西利亚"), ("阿根廷", "布宜诺斯艾利斯"),
    ("智利", "圣地亚哥"), ("秘鲁", "利马"), ("哥伦比亚", "波哥大"),
    ("澳大利亚", "堪培拉"), ("新西兰", "惠灵顿"), ("印度", "新德里"),
    ("巴基斯坦", "伊斯兰堡"), ("孟加拉国", "达卡"), ("泰国", "曼谷"),
    ("越南", "河内"), ("韩国", "首尔"), ("朝鲜", "平壤"),
    ("菲律宾", "马尼拉"), ("印度尼西亚", "雅加达"), ("马来西亚", "吉隆坡"),
    ("新加坡", "新加坡"), ("柬埔寨", "金边"), ("老挝", "万象"),
    ("缅甸", "内比都"), ("斯里兰卡", "科伦坡"), ("尼泊尔", "加德满都"),
    ("伊朗", "德黑兰"), ("伊拉克", "巴格达"), ("沙特阿拉伯", "利雅得"),
    ("阿联酋", "阿布扎比"), ("以色列", "耶路撒冷"), ("约旦", "安曼"),
    ("叙利亚", "大马士革"), ("阿富汗", "喀布尔"), ("蒙古", "乌兰巴托"),
    ("哈萨克斯坦", "阿斯塔纳"), ("乌兹别克斯坦", "塔什干"), ("乌克兰", "基辅"),
    ("罗马尼亚", "布加勒斯特"), ("保加利亚", "索菲亚"), ("塞尔维亚", "贝尔格莱德"),
    ("克罗地亚", "萨格勒布"), ("斯洛伐克", "布拉迪斯拉发"), ("斯洛文尼亚", "卢布尔雅那"),
    ("爱尔兰", "都柏林"), ("冰岛", "雷克雅未克"), ("爱沙尼亚", "塔林"),
    ("拉脱维亚", "里加"), ("立陶宛", "维尔纽斯"), ("白俄罗斯", "明斯克"),
    ("摩尔多瓦", "基希讷乌"), ("阿尔巴尼亚", "地拉那"), ("波黑", "萨拉热窝"),
    ("格鲁吉亚", "第比利斯"), ("亚美尼亚", "埃里温"), ("阿塞拜疆", "巴库"),
    ("利比亚", "的黎波里"), ("突尼斯", "突尼斯市"), ("阿尔及利亚", "阿尔及尔"),
    ("苏丹", "喀土穆"), ("埃塞俄比亚", "亚的斯亚贝巴"), ("坦桑尼亚", "多多马"),
    ("乌干达", "坎帕拉"), ("加纳", "阿克拉"), ("科特迪瓦", "亚穆苏克罗"),
    ("塞内加尔", "达喀尔"), ("喀麦隆", "雅温得"), ("安哥拉", "罗安达"),
    ("津巴布韦", "哈拉雷"), ("博茨瓦纳", "哈博罗内"), ("纳米比亚", "温得和克"),
    ("委内瑞拉", "加拉加斯"), ("厄瓜多尔", "基多"), ("玻利维亚", "苏克雷"),
    ("乌拉圭", "蒙得维的亚"), ("巴拉圭", "亚松森"),
]

_CAPITAL_TPL = [
    "{p}的首都是哪里？", "请问{p}的首都是什么城市？", "{p}的首都叫什么名字？",
    "你知道{p}的首都是哪座城市吗？", "请告诉我{p}的首都。",
    "{p}这个国家的首都是？", "{p}的首都是哪个城市？", "介绍一下{p}的首都。",
    "哪座城市是{p}的首都？", "我想知道{p}的首都。", "请说出{p}的首都名称。",
    "{p}的首都，请回答。", "你能告诉我{p}的首都是哪吗？", "请列举{p}的首都。",
    "请回答：{p}的首都是？", "{p}的首都是哪一座城市？",
    "谁来告诉我{p}的首都是哪？", "我想了解{p}的首都。",
    "请简要说明{p}的首都。", "请回答{p}的首都是什么。",
    "我想知道{p}首都是哪座城市？", "请告诉我，{p}的首都名称是什么？",
    "问：{p}的首都是？", "请说明{p}的首都是什么城市。", "{p}的首都叫什么？",
]

_SCIENCE_FACTS = [
    ("水的化学式是什么？", "水的化学式是 H₂O。"),
    ("光速大约是多少？", "光速大约是 3×10⁸ 米/秒。"),
    ("地球的半径大约是多少？", "地球的半径大约是 6371 千米。"),
    ("一年有多少天？", "一年有 365 天（闰年 366 天）。"),
    ("太阳系有几颗行星？", "太阳系有 8 颗行星。"),
    ("人体最大的器官是什么？", "人体最大的器官是皮肤。"),
    ("氧气占空气的百分比大约是多少？", "氧气约占空气的 21%。"),
    ("声音在空气中的传播速度大约是多少？", "声音在空气中传播速度约为 340 米/秒。"),
    ("铁的元素符号是什么？", "铁的元素符号是 Fe。"),
    ("金的元素符号是什么？", "金的元素符号是 Au。"),
    ("银的元素符号是什么？", "银的元素符号是 Ag。"),
    ("铜的元素符号是什么？", "铜的元素符号是 Cu。"),
    ("钠的元素符号是什么？", "钠的元素符号是 Na。"),
    ("钾的元素符号是什么？", "钾的元素符号是 K。"),
    ("钙的元素符号是什么？", "钙的元素符号是 Ca。"),
    ("碳的元素符号是什么？", "碳的元素符号是 C。"),
    ("氢的元素符号是什么？", "氢的元素符号是 H。"),
    ("氮的元素符号是什么？", "氮的元素符号是 N。"),
    ("氧的元素符号是什么？", "氧的元素符号是 O。"),
    ("汞的元素符号是什么？", "汞的元素符号是 Hg。"),
    ("人体的正常体温是多少？", "人体的正常体温约为 37℃。"),
    ("人体有多少块骨骼？", "成年人共有 206 块骨骼。"),
    ("DNA 的全称是什么？", "DNA 的全称是脱氧核糖核酸。"),
    ("光合作用主要发生在哪个细胞器？", "光合作用主要发生在叶绿体。"),
    ("电流的单位是什么？", "电流的单位是安培（A）。"),
    ("电压的单位是什么？", "电压的单位是伏特（V）。"),
    ("电阻的单位是什么？", "电阻的单位是欧姆（Ω）。"),
    ("功率的单位是什么？", "功率的单位是瓦特（W）。"),
    ("能量的单位是什么？", "能量的单位是焦耳（J）。"),
    ("力的单位是什么？", "力的单位是牛顿（N）。"),
    ("压强的单位是什么？", "压强的单位是帕斯卡（Pa）。"),
    ("频率的单位是什么？", "频率的单位是赫兹（Hz）。"),
    ("地球绕太阳一周需要多久？", "地球绕太阳一周大约需要 365.25 天。"),
    ("月亮绕地球一周需要多久？", "月亮绕地球一周大约需要 27.3 天。"),
    ("地球自转一周需要多久？", "地球自转一周大约需要 24 小时。"),
    ("太阳的主要成分是什么？", "太阳的主要成分是氢和氦。"),
    ("地球大气中含量最多的气体是什么？", "地球大气中含量最多的气体是氮气。"),
    ("水的沸点是多少？", "在标准大气压下，水的沸点是 100℃。"),
    ("水的冰点是多少？", "在标准大气压下，水的冰点是 0℃。"),
    ("光的三原色是什么？", "光的三原色是红、绿、蓝。"),
    ("颜料的三原色是什么？", "颜料的三原色是红、黄、蓝。"),
    ("声音的传播需要介质吗？", "是的，声音的传播需要介质，不能在真空中传播。"),
    ("电磁波的传播需要介质吗？", "电磁波可以在真空中传播，不需要介质。"),
    ("牛顿第一定律又称什么？", "牛顿第一定律又称惯性定律。"),
    ("万有引力是谁发现的？", "万有引力是牛顿发现的。"),
    ("相对论是谁提出的？", "相对论是爱因斯坦提出的。"),
    ("进化论是谁提出的？", "进化论是达尔文提出的。"),
    ("元素周期表是谁创立的？", "元素周期表是门捷列夫创立的。"),
    ("电话是谁发明的？", "电话是贝尔发明的。"),
    ("电灯是谁发明的？", "电灯是爱迪生发明的。"),
    ("蒸汽机是谁改良的？", "蒸汽机是瓦特改良的。"),
    ("青霉素是谁发现的？", "青霉素是弗莱明发现的。"),
    ("X 射线是谁发现的？", "X 射线是伦琴发现的。"),
    ("放射性是谁发现的？", "放射性是贝克勒尔发现的。"),
    ("镭是谁发现的？", "镭是居里夫人发现的。"),
    ("日心说是谁提出的？", "日心说是哥白尼提出的。"),
    ("浮力定律是谁发现的？", "浮力定律是阿基米德发现的。"),
    ("杠杆定律是谁发现的？", "杠杆定律是阿基米德发现的。"),
    ("勾股定理是谁证明的？", "勾股定理在中国由商高提出，毕达哥拉斯学派给出证明。"),
    ("圆周率大约是多少？", "圆周率 π 大约是 3.14159。"),
    ("自然对数的底 e 大约是多少？", "自然对数的底 e 大约是 2.71828。"),
    ("黄金分割比大约是多少？", "黄金分割比大约是 0.618。"),
    ("1 摩尔物质包含多少个粒子？", "1 摩尔物质包含约 6.02×10²³ 个粒子。"),
    ("绝对零度是多少？", "绝对零度是 -273.15℃，即 0K。"),
    ("人体血液的 pH 值大约是多少？", "人体血液的 pH 值大约是 7.35-7.45。"),
    ("植物进行光合作用的主要色素是什么？", "植物进行光合作用的主要色素是叶绿素。"),
    ("人脑大约由多少个神经元组成？", "人脑大约由 860 亿个神经元组成。"),
    ("成人一般有多少颗牙齿？", "成年人一般有 28-32 颗牙齿。"),
    ("人有多少对染色体？", "人类有 23 对（46 条）染色体。"),
    ("心脏有几个腔？", "心脏有 4 个腔：左心房、左心室、右心房、右心室。"),
    ("血液中运输氧气的细胞是什么？", "血液中运输氧气的是红细胞。"),
    ("人体最大的腺体是什么？", "人体最大的腺体是肝脏。"),
    ("昆虫有几条腿？", "昆虫有 6 条腿。"),
    ("蜘蛛是昆虫吗？", "蜘蛛不是昆虫，它属于蛛形纲。"),
    ("鸟类是恒温动物吗？", "是的，鸟类是恒温动物。"),
    ("哺乳动物的特征是什么？", "哺乳动物的特征是胎生和哺乳。"),
    ("种子植物分为哪两类？", "种子植物分为裸子植物和被子植物。"),
    ("植物光合作用产生什么气体？", "植物光合作用产生氧气。"),
    ("植物呼吸作用产生什么气体？", "植物呼吸作用产生二氧化碳。"),
    ("光合作用的原料是什么？", "光合作用的原料是二氧化碳和水。"),
    ("盐的化学式是什么？", "食盐的化学式是 NaCl。"),
    ("小苏打的化学式是什么？", "小苏打的化学式是 NaHCO₃。"),
    ("二氧化碳的化学式是什么？", "二氧化碳的化学式是 CO₂。"),
    ("甲烷的化学式是什么？", "甲烷的化学式是 CH₄。"),
    ("硫酸的化学式是什么？", "硫酸的化学式是 H₂SO₄。"),
    ("盐酸的化学式是什么？", "盐酸是 HCl 的水溶液。"),
    ("硝酸银的化学式是什么？", "硝酸银的化学式是 AgNO₃。"),
    ("氨水的化学式是什么？", "氨水的化学式是 NH₃·H₂O。"),
    ("双氧水的化学式是什么？", "双氧水的化学式是 H₂O₂。"),
    ("白糖的主要成分是什么？", "白糖的主要成分是蔗糖。"),
    ("石墨和金刚石由什么元素组成？", "石墨和金刚石都由碳元素组成。"),
    ("合金是化合物吗？", "合金不是化合物，是混合物。"),
    ("最轻的气体是什么？", "最轻的气体是氢气。"),
    ("空气中第二多的气体是什么？", "空气中第二多的气体是氧气。"),
    ("臭氧层的主要作用是什么？", "臭氧层主要吸收紫外线，保护地球生物。"),
    ("温室效应主要由哪种气体引起？", "温室效应主要由二氧化碳等气体引起。"),
    ("酸雨的主要成分是什么？", "酸雨的主要成分是硫酸和硝酸。"),
    ("大气层最底层叫什么？", "大气层最底层叫对流层。"),
    ("太阳系中最大的行星是什么？", "太阳系中最大的行星是木星。"),
    ("离太阳最近的行星是什么？", "太阳系中离太阳最近的行星是水星。"),
    ("离地球最近的恒星是什么？", "离地球最近的恒星是太阳。"),
    ("北极星属于哪个星座？", "北极星属于小熊座。"),
    ("银河系大约有多少颗恒星？", "银河系大约有 1000-4000 亿颗恒星。"),
]

_HISTORY_FACTS = [
    ("中国第一个统一的封建王朝是？", "中国第一个统一的封建王朝是秦朝。"),
    ("秦始皇的名字叫什么？", "秦始皇名叫嬴政。"),
    ("汉朝的建立者是谁？", "汉朝的建立者是刘邦。"),
    ("三国时期魏国的建立者是谁？", "三国时期魏国的建立者是曹丕。"),
    ("三国时期蜀汉的建立者是谁？", "三国时期蜀汉的建立者是刘备。"),
    ("三国时期吴国的建立者是谁？", "三国时期吴国的建立者是孙权。"),
    ("唐朝的建立者是谁？", "唐朝的建立者是李渊。"),
    ("唐太宗的名字叫什么？", "唐太宗名叫李世民。"),
    ("宋朝的建立者是谁？", "宋朝的建立者是赵匡胤。"),
    ("元朝的建立者是谁？", "元朝的建立者是忽必烈。"),
    ("明朝的建立者是谁？", "明朝的建立者是朱元璋。"),
    ("清朝的建立者是谁？", "清朝的建立者是努尔哈赤（后金）/ 皇太极（清）。"),
    ("辛亥革命发生在哪一年？", "辛亥革命发生在 1911 年。"),
    ("中华人民共和国成立于哪一年？", "中华人民共和国成立于 1949 年。"),
    ("鸦片战争发生在哪一年？", "第一次鸦片战争发生在 1840 年。"),
    ("五四运动发生在哪一年？", "五四运动发生在 1919 年。"),
    ("郑和下西洋始于哪一年？", "郑和下西洋始于 1405 年。"),
    ("丝绸之路开辟于哪个朝代？", "丝绸之路开辟于西汉。"),
    ("造纸术是谁发明的？", "造纸术是蔡伦改进的。"),
    ("活字印刷术是谁发明的？", "活字印刷术是毕昇发明的。"),
    ("指南针应用于航海始于哪个朝代？", "指南针应用于航海始于北宋。"),
    ("火药发明于哪个朝代？", "火药发明于唐代。"),
    ("《孙子兵法》的作者是谁？", "《孙子兵法》的作者是孙武。"),
    ("《史记》的作者是谁？", "《史记》的作者是司马迁。"),
    ("《资治通鉴》的作者是谁？", "《资治通鉴》的作者是司马光。"),
    ("《红楼梦》的作者是谁？", "《红楼梦》的作者是曹雪芹。"),
    ("《西游记》的作者是谁？", "《西游记》的作者是吴承恩。"),
    ("《水浒传》的作者是谁？", "《水浒传》的作者是施耐庵。"),
    ("《三国演义》的作者是谁？", "《三国演义》的作者是罗贯中。"),
    ("第一次世界大战开始于哪一年？", "第一次世界大战开始于 1914 年。"),
    ("第二次世界大战结束于哪一年？", "第二次世界大战结束于 1945 年。"),
    ("法国大革命爆发于哪一年？", "法国大革命爆发于 1789 年。"),
    ("美国独立宣言发表于哪一年？", "美国独立宣言发表于 1776 年。"),
    ("哥伦布发现新大陆是哪一年？", "哥伦布发现新大陆是 1492 年。"),
    ("麦哲伦环球航行始于哪一年？", "麦哲伦环球航行始于 1519 年。"),
    ("文艺复兴起源于哪个国家？", "文艺复兴起源于意大利。"),
    ("工业革命起源于哪个国家？", "工业革命起源于英国。"),
    ("马克思主义的创立者是谁？", "马克思主义的创立者是马克思和恩格斯。"),
    ("《共产党宣言》发表于哪一年？", "《共产党宣言》发表于 1848 年。"),
    ("古埃及金字塔建于什么时候？", "古埃及金字塔建于约公元前 2700-前 1500 年。"),
    ("《汉谟拉比法典》刻在哪里？", "《汉谟拉比法典》刻在一根黑色玄武岩石柱上。"),
    ("古希腊奥运会始于哪一年？", "古代奥运会始于公元前 776 年。"),
    ("罗马帝国分裂发生在哪一年？", "罗马帝国于公元 395 年分裂。"),
    ("拜占庭帝国灭亡于哪一年？", "拜占庭帝国灭亡于 1453 年。"),
    ("贞观之治是哪位皇帝的治世？", "贞观之治是唐太宗李世民的治世。"),
    ("开元盛世是哪位皇帝的治世？", "开元盛世是唐玄宗李隆基的治世。"),
    ("文景之治是哪个朝代的治世？", "文景之治是西汉文帝、景帝时期的治世。"),
    ("康乾盛世是哪个朝代的治世？", "康乾盛世是清朝康熙、雍正、乾隆时期的治世。"),
    ("贞观之治发生在哪个朝代？", "贞观之治发生在唐朝。"),
]


def gen_qa() -> List[Dict]:
    """问答类：常识 / 科学 / 地理 / 历史。"""
    items = []
    # 国家首都：100 国 × 25 模板
    for tpl in _CAPITAL_TPL:
        for country, capital in _COUNTRY_CAPITAL:
            prompt = tpl.format(p=country)
            completion = f"{country}的首都是{capital}。"
            items.append({"prompt": prompt, "completion": completion})
    # 科学常识：100 题 × 25 改写
    items.extend(_expand_pairs(_SCIENCE_FACTS, _REPHRASE_TPL, 2500))
    # 历史问答：50 题 × 25 改写
    items.extend(_expand_pairs(_HISTORY_FACTS, _REPHRASE_TPL, 1250))
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 2. 翻译（中英互译）
# ===========================================================================

_TRANSLATE_PAIRS = [
    ("你好世界", "Hello World"), ("早上好", "Good morning"),
    ("下午好", "Good afternoon"), ("晚上好", "Good evening"),
    ("晚安", "Good night"), ("谢谢你", "Thank you"),
    ("不客气", "You're welcome"), ("对不起", "I'm sorry"),
    ("没关系", "It doesn't matter"), ("再见", "Goodbye"),
    ("你好", "Hello"), ("很高兴认识你", "Nice to meet you"),
    ("请问怎么去火车站", "Excuse me, how can I get to the train station"),
    ("我饿了", "I'm hungry"), ("我渴了", "I'm thirsty"),
    ("我累了", "I'm tired"), ("我爱中国", "I love China"),
    ("北京是中国的首都", "Beijing is the capital of China"),
    ("今天天气很好", "The weather is nice today"),
    ("学习使我快乐", "Learning makes me happy"),
    ("知识就是力量", "Knowledge is power"),
    ("时间就是金钱", "Time is money"),
    ("一寸光阴一寸金", "An inch of time is an inch of gold"),
    ("失败是成功之母", "Failure is the mother of success"),
    ("熟能生巧", "Practice makes perfect"),
    ("入乡随俗", "When in Rome, do as the Romans do"),
    ("有志者事竟成", "Where there's a will, there's a way"),
    ("不劳无获", "No pain, no gain"),
    ("早起的鸟儿有虫吃", "The early bird catches the worm"),
    ("活到老学到老", "One is never too old to learn"),
    ("千里之行始于足下", "A journey of a thousand miles begins with a single step"),
    ("三思而后行", "Look before you leap"),
    ("物以类聚", "Birds of a feather flock together"),
    ("不经历风雨怎能见彩虹", "No rainbow without rain"),
    ("书籍是人类进步的阶梯", "Books are the ladder of human progress"),
    ("读万卷书行万里路", "Read ten thousand books, travel ten thousand miles"),
    ("学而时习之", "Learn and practice what you have learned"),
    ("温故而知新", "Review the old to learn the new"),
    ("三人行必有我师", "Among any three, there must be a teacher for me"),
    ("己所不欲勿施于人", "Do not do to others what you do not want done to yourself"),
    ("人无远虑必有近忧", "He who has no foresight will have immediate worries"),
    ("千里之堤毁于蚁穴", "A small leak will sink a great ship"),
    ("欲速则不达", "Haste makes waste"),
    ("自助者天助之", "God helps those who help themselves"),
    ("事实胜于雄辩", "Actions speak louder than words"),
    ("机不可失时不再来", "Opportunity knocks but once"),
    ("今日事今日毕", "Never put off until tomorrow what you can do today"),
    ("团结就是力量", "Unity is strength"),
    ("健康就是财富", "Health is wealth"),
    ("我爱你", "I love you"), ("我喜欢编程", "I like programming"),
    ("我爱学习", "I love learning"), ("我喜欢读书", "I like reading"),
    ("我喜欢音乐", "I like music"), ("我喜欢运动", "I like sports"),
    ("我喜欢旅行", "I like traveling"), ("我喜欢看电影", "I like watching movies"),
    ("我喜欢吃苹果", "I like eating apples"), ("我喜欢喝咖啡", "I like drinking coffee"),
    ("我喜欢喝茶", "I like drinking tea"), ("我喜欢吃面条", "I like eating noodles"),
    ("我喜欢吃米饭", "I like eating rice"), ("我喜欢吃饺子", "I like eating dumplings"),
    ("我喜欢吃火锅", "I like eating hotpot"), ("我喜欢春天", "I like spring"),
    ("我喜欢夏天", "I like summer"), ("我喜欢秋天", "I like autumn"),
    ("我喜欢冬天", "I like winter"), ("这是我的书", "This is my book"),
    ("那是你的笔", "That is your pen"), ("这本书很有趣", "This book is very interesting"),
    ("这部电影很精彩", "This movie is wonderful"), ("这首歌很好听", "This song sounds good"),
    ("这道菜很美味", "This dish is delicious"), ("这个城市很美", "This city is beautiful"),
    ("这个国家很大", "This country is big"), ("这栋楼很高", "This building is tall"),
    ("这条河很长", "This river is long"), ("这座山很高", "This mountain is high"),
    ("这个湖很清", "This lake is clear"), ("这朵花很香", "This flower smells good"),
    ("这棵树很大", "This tree is big"), ("这只猫很可爱", "This cat is cute"),
    ("这只狗很聪明", "This dog is smart"), ("这个孩子很乖", "This child is well-behaved"),
    ("这个学生很努力", "This student is hardworking"), ("这个老师很好", "This teacher is good"),
    ("这个医生很有经验", "This doctor is experienced"),
    ("这个工程师很专业", "This engineer is professional"),
    ("你在做什么", "What are you doing"), ("你来自哪里", "Where are you from"),
    ("你叫什么名字", "What is your name"), ("你多大了", "How old are you"),
    ("你住在哪里", "Where do you live"), ("你会说什么语言", "What languages do you speak"),
    ("你有兄弟姐妹吗", "Do you have brothers or sisters"),
    ("你结婚了吗", "Are you married"), ("你有孩子吗", "Do you have children"),
    ("你喜欢什么颜色", "What color do you like"),
    ("你喜欢什么动物", "What animal do you like"),
    ("你喜欢什么食物", "What food do you like"),
    ("你喜欢什么运动", "What sport do you like"),
    ("你喜欢什么音乐", "What music do you like"),
    ("你喜欢什么书", "What book do you like"),
    ("你喜欢什么电影", "What movie do you like"),
    ("你去过中国吗", "Have you been to China"),
    ("你去过北京吗", "Have you been to Beijing"),
    ("你去过上海吗", "Have you been to Shanghai"),
    ("你会说中文吗", "Can you speak Chinese"),
    ("你会说英语吗", "Can you speak English"),
    ("我会说中文", "I can speak Chinese"), ("我会说英语", "I can speak English"),
    ("我是一名学生", "I am a student"), ("我是一名老师", "I am a teacher"),
    ("我是一名医生", "I am a doctor"), ("我是一名工程师", "I am an engineer"),
    ("我是一名程序员", "I am a programmer"), ("我是一名设计师", "I am a designer"),
    ("我在学中文", "I am learning Chinese"), ("我在学英语", "I am learning English"),
    ("我在学编程", "I am learning programming"), ("我在学数学", "I am learning math"),
    ("我在学物理", "I am learning physics"), ("我在学化学", "I am learning chemistry"),
    ("我在学历史", "I am learning history"), ("我在学地理", "I am learning geography"),
    ("时间不早了", "It's getting late"), ("我得走了", "I have to go"),
    ("回头见", "See you later"), ("明天见", "See you tomorrow"),
    ("下周见", "See you next week"), ("保持联系", "Keep in touch"),
    ("注意安全", "Take care"), ("祝你好运", "Good luck"),
    ("生日快乐", "Happy birthday"), ("新年快乐", "Happy New Year"),
    ("圣诞快乐", "Merry Christmas"), ("节日快乐", "Happy holidays"),
    ("恭喜恭喜", "Congratulations"), ("一路平安", "Have a safe trip"),
    ("欢迎光临", "Welcome"), ("请进", "Please come in"),
    ("请坐", "Please sit down"), ("请喝茶", "Please have some tea"),
    ("请稍等", "Please wait a moment"), ("请再说一遍", "Please say it again"),
    ("请慢一点说", "Please speak slowly"), ("请大声一点", "Please speak louder"),
    ("请小声一点", "Please speak softly"), ("请帮帮我", "Please help me"),
    ("请问贵姓", "May I ask your surname"),
    ("我同意你的观点", "I agree with you"), ("我不同意", "I disagree"),
    ("你说得对", "You are right"), ("你说错了", "You are wrong"),
    ("我不确定", "I'm not sure"), ("我不知道", "I don't know"),
    ("让我想想", "Let me think"), ("等一下", "Wait a moment"),
    ("马上就好", "It will be ready soon"), ("没问题", "No problem"),
    ("当然可以", "Of course"), ("你真棒", "You are great"),
    ("干得好", "Well done"), ("继续努力", "Keep it up"),
    ("别放弃", "Don't give up"), ("相信自己", "Believe in yourself"),
    ("勇敢一点", "Be brave"), ("加油", "Come on"),
    ("辛苦了", "Thanks for your hard work"), ("合作愉快", "Happy cooperation"),
    ("旅途愉快", "Have a good trip"), ("用餐愉快", "Enjoy your meal"),
    ("周末愉快", "Have a good weekend"),
    ("好好学习天天向上", "Study hard and make progress every day"),
    ("世上无难事只怕有心人", "Nothing is impossible to a willing heart"),
    ("百闻不如一见", "Seeing is believing"),
    ("有备无患", "Better safe than sorry"),
    ("知足常乐", "Contentment brings happiness"),
    ("不要把所有鸡蛋放在一个篮子里", "Don't put all your eggs in one basket"),
    ("罗马不是一天建成的", "Rome was not built in a day"),
    ("条条大路通罗马", "All roads lead to Rome"),
    ("人生苦短", "Life is short"), ("活在当下", "Live in the moment"),
    ("不忘初心", "Never forget why you started"),
    ("方得始终", "Only then can you succeed"),
    ("天道酬勤", "Heaven rewards the diligent"),
    ("厚德载物", "Great virtue carries all things"),
    ("宁静致远", "Tranquility leads to far-reaching goals"),
    ("淡泊明志", "A simple life reveals high aspirations"),
    ("自强不息", "Strive unceasingly"), ("知行合一", "Unity of knowledge and action"),
    ("海纳百川", "The sea admits hundreds of rivers"),
    ("上善若水", "The highest good is like water"),
    ("大道至简", "Great truths are simple"),
    ("中庸之道", "The doctrine of the mean"),
    ("仁者爱人", "The benevolent love others"),
    ("礼尚往来", "Courtesy demands reciprocity"),
    ("博学之审问之", "Learn widely and inquire carefully"),
    ("格物致知", "Investigate things to extend knowledge"),
    ("以人为本", "People-oriented"), ("和而不同", "Harmony in diversity"),
    ("天下为公", "The world belongs to all"),
    ("民为贵", "The people are the most important"),
]

_ZH2EN_TPL = [
    "翻译成英文：{p}", "请将以下中文翻译成英文：{p}", "把这句中文翻成英文：{p}",
    "中文翻译英文：{p}", "请翻译为英语：{p}", "英译：{p}",
    "请给出英文翻译：{p}", "中→英：{p}", "请把这句话译成英语：{p}",
    "翻译（中译英）：{p}", '请将"{p}"译为英文', "求英文翻译：{p}",
    "请把下面中文翻译为英文：{p}", "请用英语表达：{p}", "英文怎么说：{p}",
    "请用英文表达下面这句话：{p}", "把这句译为英语：{p}",
    "请将此句翻译为英文：{p}", "请翻译下面的中文：{p}", "中翻英：{p}",
    "请把它翻译成英文：{p}", "请把它译成英语：{p}",
    "请把这段中文翻译为英文：{p}", "英译如下中文：{p}", "请译为英文：{p}",
]

_EN2ZH_TPL = [
    "翻译成中文：{p}", "请将以下英文翻译成中文：{p}", "把这句英文翻成中文：{p}",
    "英文翻译中文：{p}", "请翻译为汉语：{p}", "中译：{p}",
    "请给出中文翻译：{p}", "英→中：{p}", "请把这句话译成中文：{p}",
    "翻译（英译中）：{p}", '请将"{p}"译为中文', "求中文翻译：{p}",
    "请把下面英文翻译为中文：{p}", "请用汉语表达：{p}", "中文怎么说：{p}",
    "请用中文表达下面这句话：{p}", "把这句译为中文：{p}",
    "请将此句翻译为中文：{p}", "请翻译下面的英文：{p}", "英翻中：{p}",
    "请把它翻译成中文：{p}", "请把它译成汉语：{p}",
    "请把这段英文翻译为中文：{p}", "中译如下英文：{p}", "请译为中文：{p}",
]


def gen_translate() -> List[Dict]:
    """翻译类：中英互译。"""
    items = []
    # 中译英：200 对 × 25 模板
    for tpl in _ZH2EN_TPL:
        for zh, en in _TRANSLATE_PAIRS:
            prompt = tpl.format(p=zh)
            items.append({"prompt": prompt, "completion": en})
    # 英译中：200 对 × 25 模板
    for tpl in _EN2ZH_TPL:
        for zh, en in _TRANSLATE_PAIRS:
            prompt = tpl.format(p=en)
            items.append({"prompt": prompt, "completion": zh})
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 3. 代码（Python 基础 / 算法 / 数据结构）
# ===========================================================================

# 函数名 → (描述, 代码实现)
_CODE_FUNCS = {
    "add": ("求两个数的和", "def add(a, b):\n    return a + b"),
    "subtract": ("求两个数的差", "def subtract(a, b):\n    return a - b"),
    "multiply": ("求两个数的积", "def multiply(a, b):\n    return a * b"),
    "divide": ("求两个数的商", "def divide(a, b):\n    return a / b if b != 0 else None"),
    "square": ("求一个数的平方", "def square(x):\n    return x ** 2"),
    "cube": ("求一个数的立方", "def cube(x):\n    return x ** 3"),
    "is_even": ("判断是否为偶数", "def is_even(n):\n    return n % 2 == 0"),
    "is_odd": ("判断是否为奇数", "def is_odd(n):\n    return n % 2 != 0"),
    "is_positive": ("判断是否为正数", "def is_positive(n):\n    return n > 0"),
    "is_negative": ("判断是否为负数", "def is_negative(n):\n    return n < 0"),
    "max_of_two": ("求两个数的最大值", "def max_of_two(a, b):\n    return a if a > b else b"),
    "min_of_two": ("求两个数的最小值", "def min_of_two(a, b):\n    return a if a < b else b"),
    "abs_value": ("求绝对值", "def abs_value(n):\n    return abs(n)"),
    "factorial": ("求阶乘", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    "fibonacci": ("求斐波那契数", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"),
    "reverse_string": ("反转字符串", "def reverse_string(s):\n    return s[::-1]"),
    "count_vowels": ("统计元音字母", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    "is_palindrome": ("判断回文", "def is_palindrome(s):\n    return s == s[::-1]"),
    "sum_list": ("列表求和", "def sum_list(lst):\n    return sum(lst)"),
    "max_list": ("列表最大值", "def max_list(lst):\n    return max(lst) if lst else None"),
    "min_list": ("列表最小值", "def min_list(lst):\n    return min(lst) if lst else None"),
    "len_list": ("列表长度", "def len_list(lst):\n    return len(lst)"),
    "sort_list": ("列表排序", "def sort_list(lst):\n    return sorted(lst)"),
    "unique_list": ("列表去重", "def unique_list(lst):\n    return list(dict.fromkeys(lst))"),
    "flatten_list": ("扁平化列表", "def flatten_list(lst):\n    result = []\n    for x in lst:\n        if isinstance(x, list):\n            result.extend(flatten_list(x))\n        else:\n            result.append(x)\n    return result"),
    "count_occurrences": ("统计元素出现次数", "def count_occurrences(lst, x):\n    return lst.count(x)"),
    "merge_lists": ("合并两个列表", "def merge_lists(a, b):\n    return a + b"),
    "intersection": ("求两个列表的交集", "def intersection(a, b):\n    return list(set(a) & set(b))"),
    "union": ("求两个列表的并集", "def union(a, b):\n    return list(set(a) | set(b))"),
    "difference": ("求两个列表的差集", "def difference(a, b):\n    return list(set(a) - set(b))"),
    "is_prime": ("判断质数", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    "gcd": ("求最大公约数", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
    "lcm": ("求最小公倍数", "def lcm(a, b):\n    return a * b // gcd(a, b)"),
    "power": ("幂运算", "def power(base, exp):\n    return base ** exp"),
    "sqrt": ("求平方根", "def sqrt(n):\n    return n ** 0.5"),
    "celsius_to_fahrenheit": ("摄氏度转华氏度", "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32"),
    "fahrenheit_to_celsius": ("华氏度转摄氏度", "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9"),
    "km_to_mile": ("公里转英里", "def km_to_mile(km):\n    return km * 0.621371"),
    "mile_to_km": ("英里转公里", "def mile_to_km(mile):\n    return mile * 1.60934"),
    "kg_to_pound": ("千克转磅", "def kg_to_pound(kg):\n    return kg * 2.20462"),
    "pound_to_kg": ("磅转千克", "def pound_to_kg(pound):\n    return pound * 0.453592"),
    "capitalize": ("首字母大写", "def capitalize(s):\n    return s.capitalize()"),
    "to_upper": ("转大写", "def to_upper(s):\n    return s.upper()"),
    "to_lower": ("转小写", "def to_lower(s):\n    return s.lower()"),
    "count_words": ("统计单词数", "def count_words(s):\n    return len(s.split())"),
    "split_string": ("分割字符串", "def split_string(s, sep):\n    return s.split(sep)"),
    "join_strings": ("连接字符串", "def join_strings(lst, sep):\n    return sep.join(lst)"),
    "replace_char": ("替换字符", "def replace_char(s, old, new):\n    return s.replace(old, new)"),
    "strip_spaces": ("去除空格", "def strip_spaces(s):\n    return s.strip()"),
}

# 100 个代码片段（问题描述, 代码）
_CODE_SNIPPETS = [
    ("写一个 Python 函数计算列表平均值", "def average(lst):\n    return sum(lst) / len(lst) if lst else 0"),
    ("用 Python 实现冒泡排序", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"),
    ("用 Python 实现快速排序", "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"),
    ("用 Python 实现选择排序", "def selection_sort(arr):\n    for i in range(len(arr)):\n        min_idx = i\n        for j in range(i + 1, len(arr)):\n            if arr[j] < arr[min_idx]:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr"),
    ("用 Python 实现插入排序", "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr"),
    ("用 Python 实现二分查找", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
    ("用 Python 实现线性查找", "def linear_search(arr, target):\n    for i, x in enumerate(arr):\n        if x == target:\n            return i\n    return -1"),
    ("用 Python 实现栈", "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop() if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0"),
    ("用 Python 实现队列", "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n        return self.items.pop(0) if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0"),
    ("用 Python 实现链表节点", "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None"),
    ("用 Python 实现单链表", "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, data):\n        if not self.head:\n            self.head = Node(data)\n            return\n        cur = self.head\n        while cur.next:\n            cur = cur.next\n        cur.next = Node(data)"),
    ("用 Python 实现二叉树节点", "class TreeNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None"),
    ("用 Python 实现二叉树前序遍历", "def preorder(root):\n    if root:\n        print(root.val)\n        preorder(root.left)\n        preorder(root.right)"),
    ("用 Python 实现二叉树中序遍历", "def inorder(root):\n    if root:\n        inorder(root.left)\n        print(root.val)\n        inorder(root.right)"),
    ("用 Python 实现二叉树后序遍历", "def postorder(root):\n    if root:\n        postorder(root.left)\n        postorder(root.right)\n        print(root.val)"),
    ("用 Python 实现哈希表", "class HashTable:\n    def __init__(self, size=100):\n        self.size = size\n        self.table = [[] for _ in range(size)]\n    def put(self, key, value):\n        h = hash(key) % self.size\n        for i, (k, v) in enumerate(self.table[h]):\n            if k == key:\n                self.table[h][i] = (key, value)\n                return\n        self.table[h].append((key, value))"),
    ("用 Python 实现图（邻接表）", "class Graph:\n    def __init__(self):\n        self.adj = {}\n    def add_edge(self, u, v):\n        if u not in self.adj:\n            self.adj[u] = []\n        self.adj[u].append(v)"),
    ("用 Python 实现 BFS", "from collections import deque\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    visited.add(start)\n    while queue:\n        v = queue.popleft()\n        for n in graph.get(v, []):\n            if n not in visited:\n                visited.add(n)\n                queue.append(n)"),
    ("用 Python 实现 DFS", "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for n in graph.get(start, []):\n        if n not in visited:\n            dfs(graph, n, visited)\n    return visited"),
    ("用 Python 实现字典反转", "def invert_dict(d):\n    return {v: k for k, v in d.items()}"),
    ("用 Python 实现列表推导式生成平方数", "squares = [x ** 2 for x in range(10)]"),
    ("用 Python 实现斐波那契数列（迭代）", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
    ("用 Python 实现斐波那契数列（递归）", "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)"),
    ("用 Python 实现阶乘（递归）", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("用 Python 实现阶乘（迭代）", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
    ("用 Python 判断回文数", "def is_palindrome_number(n):\n    s = str(n)\n    return s == s[::-1]"),
    ("用 Python 实现字符串反转", "def reverse_string(s):\n    return s[::-1]"),
    ("用 Python 统计字符出现次数", "def count_char(s, ch):\n    return s.count(ch)"),
    ("用 Python 实现笛卡尔积", "from itertools import product\nresult = list(product([1, 2], [3, 4]))"),
    ("用 Python 实现排列", "from itertools import permutations\nresult = list(permutations([1, 2, 3]))"),
    ("用 Python 实现组合", "from itertools import combinations\nresult = list(combinations([1, 2, 3], 2))"),
    ("用 Python 实现计数器", "from collections import Counter\nc = Counter('abracadabra')"),
    ("用 Python 实现默认字典", "from collections import defaultdict\nd = defaultdict(list)"),
    ("用 Python 实现双端队列", "from collections import deque\ndq = deque([1, 2, 3])\ndq.appendleft(0)\ndq.append(4)"),
    ("用 Python 实现堆", "import heapq\nh = []\nheapq.heappush(h, 3)\nheapq.heappush(h, 1)\nheapq.heappop(h)"),
    ("用 Python 实现优先队列", "import heapq\nclass PriorityQueue:\n    def __init__(self):\n        self.h = []\n    def push(self, item):\n        heapq.heappush(self.h, item)\n    def pop(self):\n        return heapq.heappop(self.h)"),
    ("用 Python 实现两数交换", "a, b = b, a"),
    ("用 Python 实现三元运算", "result = x if x > 0 else 0"),
    ("用 Python 实现枚举遍历", "for i, v in enumerate(['a', 'b', 'c']):\n    print(i, v)"),
    ("用 Python 实现 zip 并行遍历", "for a, b in zip([1, 2, 3], ['a', 'b', 'c']):\n    print(a, b)"),
    ("用 Python 实现矩阵转置", "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]"),
    ("用 Python 实现矩阵乘法", "def matrix_mul(A, B):\n    return [[sum(a * b for a, b in zip(row, col))\n             for col in zip(*B)] for row in A]"),
    ("用 Python 实现字符串分割", "def split_str(s, sep):\n    return s.split(sep)"),
    ("用 Python 实现字符串连接", "def join_str(lst, sep):\n    return sep.join(lst)"),
    ("用 Python 实现首字母大写", "def capitalize(s):\n    return s.title()"),
    ("用 Python 实现去除重复元素", "def unique(lst):\n    return list(dict.fromkeys(lst))"),
    ("用 Python 实现列表扁平化", "def flatten(lst):\n    result = []\n    for x in lst:\n        if isinstance(x, list):\n            result.extend(flatten(x))\n        else:\n            result.append(x)\n    return result"),
    ("用 Python 实现字典合并", "def merge_dict(d1, d2):\n    return {**d1, **d2}"),
    ("用 Python 实现文件读取", "with open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()"),
    ("用 Python 实现文件写入", "with open('file.txt', 'w', encoding='utf-8') as f:\n    f.write('hello')"),
    ("用 Python 实现文件逐行读取", "with open('file.txt', 'r', encoding='utf-8') as f:\n    for line in f:\n        print(line.strip())"),
    ("用 Python 实现异常处理", "try:\n    x = 1 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error: {e}')"),
    ("用 Python 实现上下文管理器", "class MyContext:\n    def __enter__(self):\n        return self\n    def __exit__(self, *args):\n        pass"),
    ("用 Python 实现装饰器", "def log(func):\n    def wrapper(*args, **kwargs):\n        print(f'Calling {func.__name__}')\n        return func(*args, **kwargs)\n    return wrapper"),
    ("用 Python 实现生成器", "def counter(n):\n    i = 0\n    while i < n:\n        yield i\n        i += 1"),
    ("用 Python 实现列表推导式", "evens = [x for x in range(20) if x % 2 == 0]"),
    ("用 Python 实现字典推导式", "squares = {x: x ** 2 for x in range(5)}"),
    ("用 Python 实现集合推导式", "evens = {x for x in range(10) if x % 2 == 0}"),
    ("用 Python 实现字符串格式化", "name = 'World'\nmsg = f'Hello, {name}!'"),
    ("用 Python 实现 map 函数", "result = list(map(str, [1, 2, 3]))"),
    ("用 Python 实现 filter 函数", "result = list(filter(lambda x: x > 0, [-1, 0, 1, 2]))"),
    ("用 Python 实现 reduce 函数", "from functools import reduce\nresult = reduce(lambda x, y: x + y, [1, 2, 3, 4])"),
    ("用 Python 实现 lambda 表达式", "square = lambda x: x ** 2"),
    ("用 Python 实现闭包", "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder"),
    ("用 Python 实现类继承", "class Animal:\n    def speak(self):\n        pass\nclass Dog(Animal):\n    def speak(self):\n        return 'Woof'"),
    ("用 Python 实现多态", "class Cat(Animal):\n    def speak(self):\n        return 'Meow'"),
    ("用 Python 实现魔术方法 __str__", "class Person:\n    def __init__(self, name):\n        self.name = name\n    def __str__(self):\n        return f'Person({self.name})'"),
    ("用 Python 实现魔术方法 __len__", "class MyList:\n    def __init__(self, items):\n        self.items = items\n    def __len__(self):\n        return len(self.items)"),
    ("用 Python 实现魔术方法 __eq__", "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __eq__(self, other):\n        return self.x == other.x and self.y == other.y"),
    ("用 Python 实现静态方法", "class MathUtil:\n    @staticmethod\n    def add(a, b):\n        return a + b"),
    ("用 Python 实现类方法", "class Counter:\n    count = 0\n    @classmethod\n    def increment(cls):\n        cls.count += 1"),
    ("用 Python 实现属性装饰器", "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def area(self):\n        return 3.14 * self._r ** 2"),
    ("用 Python 实现抽象类", "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass"),
    ("用 Python 实现单例模式", "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"),
    ("用 Python 实现工厂模式", "class Dog:\n    def speak(self):\n        return 'Woof'\nclass Cat:\n    def speak(self):\n        return 'Meow'\ndef factory(kind):\n    if kind == 'dog':\n        return Dog()\n    elif kind == 'cat':\n        return Cat()"),
    ("用 Python 实现观察者模式", "class Subject:\n    def __init__(self):\n        self.observers = []\n    def notify(self):\n        for o in self.observers:\n            o.update()"),
    ("用 Python 实现迭代器", "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i < self.n:\n            self.i += 1\n            return self.i\n        raise StopIteration"),
    ("用 Python 实现协程", "async def hello():\n    print('hello')\n    await asyncio.sleep(1)"),
    ("用 Python 实现异步 IO", "import asyncio\nasync def main():\n    await asyncio.gather(hello(), hello())"),
    ("用 Python 实现线程", "import threading\ndef worker():\n    print('working')\nt = threading.Thread(target=worker)\nt.start()"),
    ("用 Python 实现进程", "import multiprocessing\ndef worker():\n    print('working')\np = multiprocessing.Process(target=worker)\np.start()"),
    ("用 Python 实现正则匹配", "import re\nresult = re.findall(r'\\d+', 'a1b22c333')"),
    ("用 Python 实现正则替换", "import re\nresult = re.sub(r'\\d+', '#', 'a1b22c333')"),
    ("用 Python 实现日期时间", "from datetime import datetime\nnow = datetime.now()"),
    ("用 Python 实现时间戳", "import time\ntimestamp = time.time()"),
    ("用 Python 实现随机数", "import random\nn = random.randint(1, 100)"),
    ("用 Python 实现随机选择", "import random\nchoice = random.choice(['a', 'b', 'c'])"),
    ("用 Python 实现随机打乱", "import random\nrandom.shuffle(lst)"),
    ("用 Python 实现数学函数", "import math\nprint(math.sqrt(16))"),
    ("用 Python 实现对数", "import math\nprint(math.log(10))"),
    ("用 Python 实现三角函数", "import math\nprint(math.sin(0))"),
    ("用 Python 实现圆周率", "import math\nprint(math.pi)"),
    ("用 Python 实现欧拉常数", "import math\nprint(math.e)"),
    ("用 Python 实现向上取整", "import math\nprint(math.ceil(3.2))"),
    ("用 Python 实现向下取整", "import math\nprint(math.floor(3.8))"),
    ("用 Python 实现绝对值", "print(abs(-5))"),
    ("用 Python 实现 max 函数", "print(max(1, 2, 3))"),
    ("用 Python 实现 min 函数", "print(min(1, 2, 3))"),
    ("用 Python 实现 sum 函数", "print(sum([1, 2, 3]))"),
    ("用 Python 实现排序", "print(sorted([3, 1, 2]))"),
    ("用 Python 实现反转", "print(list(reversed([1, 2, 3])))"),
    ("用 Python 实现类型转换", "n = int('123')"),
    ("用 Python 实现字符串转列表", "lst = list('hello')"),
    ("用 Python 实现列表转字符串", "s = ''.join(['a', 'b', 'c'])"),
    ("用 Python 实现字符串长度", "n = len('hello')"),
    ("用 Python 实现字符串包含", "result = 'a' in 'abc'"),
    ("用 Python 实现字符串切片", "s = 'hello'[1:3]"),
    ("用 Python 实现列表切片", "lst = [1, 2, 3][1:]"),
    ("写一个 Python 函数计算 1 到 n 的和", "def sum_to_n(n):\n    return n * (n + 1) // 2"),
    ("写一个 Python 函数判断 n 是否为完全数", "def is_perfect(n):\n    return sum(i for i in range(1, n) if n % i == 0) == n"),
    ("写一个 Python 函数返回 n 以内的所有质数", "def primes(n):\n    return [x for x in range(2, n + 1) if all(x % i for i in range(2, int(x ** 0.5) + 1))]"),
    ("写一个 Python 函数实现字符串去重", "def dedup(s):\n    return ''.join(dict.fromkeys(s))"),
    ("写一个 Python 函数实现字符串大小写互换", "def swap_case(s):\n    return s.swapcase()"),
    ("写一个 Python 函数统计字符串中每个字符出现的次数", "def char_count(s):\n    from collections import Counter\n    return dict(Counter(s))"),
    ("写一个 Python 函数判断数字是否为水仙花数", "def is_narcissistic(n):\n    s = str(n)\n    return n == sum(int(c) ** len(s) for c in s)"),
    ("写一个 Python 函数返回列表中第二大的数", "def second_max(lst):\n    uniq = sorted(set(lst), reverse=True)\n    return uniq[1] if len(uniq) > 1 else None"),
    ("写一个 Python 函数返回列表中第二小的数", "def second_min(lst):\n    uniq = sorted(set(lst))\n    return uniq[1] if len(uniq) > 1 else None"),
]

_CODE_FUNC_TPL = [
    "用 Python 写一个{name}函数，{desc}",
    "请用 Python 实现{name}：{desc}",
    "写一个 Python 函数 {name}，{desc}",
    "请用 Python 编写函数 {name}，要求{desc}",
    "用 Python 完成：{desc}，函数名使用 {name}",
    "请用 Python 实现 {name} 函数，{desc}",
    "用 Python 代码实现 {desc}（函数名 {name}）",
    "编写 Python 函数 {name}，{desc}",
    "Python：{desc}，函数名 {name}",
    "请实现 Python 函数 {name}，功能是{desc}",
]

_CODE_REPHRASE_TPL = [
    "{p}", "请{p}", "请用 Python 实现：{p}", "请帮我{p}", "请编写代码：{p}",
    "请写出代码：{p}", "请给出代码：{p}", "代码实现：{p}", "Python 代码：{p}",
    "请用代码完成：{p}", "请用 Python 编写：{p}", "请用 Python 完成：{p}",
    "请用代码实现：{p}", "请帮我写一段代码：{p}", "代码示例：{p}",
    "请提供代码：{p}", "请用代码解决：{p}", "请用 Python 解决：{p}",
    "请用 Python 实现以下功能：{p}", "请编写 Python 代码：{p}",
    "请编写一段 Python 代码：{p}", "请实现：{p}", "请完成：{p}",
    "请使用 Python：{p}", "用 Python 完成：{p}",
]


def gen_code() -> List[Dict]:
    """代码类：Python 基础 / 算法 / 数据结构。"""
    items = []
    # 50 函数 × 10 模板 = 500
    for name, (desc, code) in _CODE_FUNCS.items():
        for tpl in _CODE_FUNC_TPL:
            prompt = tpl.format(name=name, desc=desc)
            items.append({"prompt": prompt, "completion": code})
    # 100 片段 × 25 改写 = 2500
    items.extend(_expand_pairs(_CODE_SNIPPETS, _CODE_REPHRASE_TPL, 2500))
    # 用函数实现 × 改写模板补充
    func_pairs = [(f"用 Python 写一个{name}函数，{desc}", code)
                  for name, (desc, code) in _CODE_FUNCS.items()]
    items.extend(_expand_pairs(func_pairs, _CODE_REPHRASE_TPL, 2000))
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 4. 数学（算术 / 代数 / 几何）
# ===========================================================================

def _int2zh(n: int) -> str:
    """将整数转中文（支持 0-9999）。"""
    digits = "零一二三四五六七八九"
    if n == 0:
        return "零"
    if n < 0:
        return "负" + _int2zh(-n)
    if n < 10:
        return digits[n]
    if n < 20:
        return "十" + (digits[n - 10] if n - 10 != 0 else "")
    if n < 100:
        tens, ones = divmod(n, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        result = digits[hundreds] + "百"
        if rest == 0:
            return result
        if rest < 10:
            return result + "零" + digits[rest]
        return result + _int2zh(rest)
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        result = digits[thousands] + "千"
        if rest == 0:
            return result
        if rest < 100:
            return result + "零" + _int2zh(rest)
        return result + _int2zh(rest)
    return str(n)


_ADD_TPL = [
    "计算：{a} + {b}", "求 {a} 加 {b} 的和", "{a} + {b} 等于多少",
    "{a} 加上 {b} 等于", "请计算 {a} + {b}", "求和：{a} + {b}",
    "算一算 {a} + {b}", "{a} 加 {b} = ?", "请问 {a} + {b} 等于多少",
    "计算下面算式：{a} + {b}", "求 {a} 与 {b} 的和", "{a} 和 {b} 相加等于",
    "{a}+{b}=?", "口算：{a} + {b}", "请回答 {a} + {b}",
    "把 {a} 和 {b} 加起来", "{a} 加 {b} 是多少", "加法运算：{a} + {b}",
    "数学题：{a} + {b}", "{a} 增加多少等于 {a}+{b}",
]

_SUB_TPL = [
    "计算：{a} - {b}", "求 {a} 减 {b} 的差", "{a} - {b} 等于多少",
    "{a} 减去 {b} 等于", "请计算 {a} - {b}", "求差：{a} - {b}",
    "算一算 {a} - {b}", "{a} 减 {b} = ?", "请问 {a} - {b} 等于多少",
    "计算下面算式：{a} - {b}", "求 {a} 与 {b} 的差", "{a} 减 {b} 是多少",
    "{a}-{b}=?", "口算：{a} - {b}", "请回答 {a} - {b}",
    "从 {a} 中减去 {b}", "减法运算：{a} - {b}", "数学题：{a} - {b}",
    "{a} 比 {b} 多多少", "{a} 减 {b} 等于",
]

_MUL_TPL = [
    "计算：{a} × {b}", "求 {a} 乘 {b} 的积", "{a} × {b} 等于多少",
    "{a} 乘以 {b} 等于", "请计算 {a} × {b}", "求积：{a} × {b}",
    "算一算 {a} × {b}", "{a} × {b} = ?", "请问 {a} 乘 {b} 等于多少",
    "计算下面算式：{a} × {b}", "求 {a} 与 {b} 的积", "{a} 乘 {b} 是多少",
    "{a}×{b}=?", "口算：{a} × {b}", "请回答 {a} × {b}",
    "乘法运算：{a} × {b}", "数学题：{a} × {b}", "{a} 的 {b} 倍是多少",
    "{a} 乘 {b} 等于", "求 {a} 乘以 {b}",
]

_DIV_TPL = [
    "计算：{a} ÷ {b}", "求 {a} 除以 {b} 的商", "{a} ÷ {b} 等于多少",
    "{a} 除以 {b} 等于", "请计算 {a} ÷ {b}", "求商：{a} ÷ {b}",
    "算一算 {a} ÷ {b}", "{a} ÷ {b} = ?", "请问 {a} 除以 {b} 等于多少",
    "计算下面算式：{a} ÷ {b}", "求 {a} 除以 {b} 的商", "{a} 除以 {b} 是多少",
    "{a}÷{b}=?", "口算：{a} ÷ {b}", "请回答 {a} ÷ {b}",
    "除法运算：{a} ÷ {b}", "数学题：{a} ÷ {b}", "{a} 里有多少个 {b}",
    "{a} 除以 {b} 等于", "求 {a} 除以 {b}",
]

_GEO_ITEMS = [
    ("求边长为 3 的正方形面积", "正方形面积 = 3 × 3 = 9"),
    ("求边长为 4 的正方形周长", "正方形周长 = 4 × 4 = 16"),
    ("求长 5 宽 3 的矩形面积", "矩形面积 = 5 × 3 = 15"),
    ("求长 5 宽 3 的矩形周长", "矩形周长 = 2 × (5 + 3) = 16"),
    ("求半径为 2 的圆的面积", "圆面积 = π × 2² = 4π ≈ 12.57"),
    ("求半径为 3 的圆的周长", "圆周长 = 2π × 3 = 6π ≈ 18.85"),
    ("求底为 4 高为 3 的三角形面积", "三角形面积 = 4 × 3 ÷ 2 = 6"),
    ("求边长为 5 的等边三角形面积", "等边三角形面积 = (√3/4) × 5² ≈ 10.83"),
    ("求底为 6 高为 4 的平行四边形面积", "平行四边形面积 = 6 × 4 = 24"),
    ("求上底 3 下底 5 高 4 的梯形面积", "梯形面积 = (3 + 5) × 4 ÷ 2 = 16"),
    ("求棱长为 3 的正方体体积", "正方体体积 = 3³ = 27"),
    ("求棱长为 3 的正方体表面积", "正方体表面积 = 6 × 3² = 54"),
    ("求长 4 宽 3 高 2 的长方体体积", "长方体体积 = 4 × 3 × 2 = 24"),
    ("求长 4 宽 3 高 2 的长方体表面积", "长方体表面积 = 2 × (4×3 + 4×2 + 3×2) = 52"),
    ("求半径为 3 的球的体积", "球体积 = (4/3)π × 3³ = 36π ≈ 113.10"),
    ("求半径为 3 的球的表面积", "球表面积 = 4π × 3² = 36π ≈ 113.10"),
    ("求直角边为 3 和 4 的斜边长", "斜边 = √(3² + 4²) = 5"),
    ("求直角边为 6 和 8 的斜边长", "斜边 = √(6² + 8²) = 10"),
    ("求直角边为 5 和 12 的斜边长", "斜边 = √(5² + 12²) = 13"),
    ("求半径为 5 直径为 10 的圆的面积", "圆面积 = π × 5² = 25π ≈ 78.54"),
]


def gen_math() -> List[Dict]:
    """数学类：算术 / 代数 / 几何。"""
    items = []
    seen = set()

    # 1. 加法（约 1500 条，每对随机数用 1 个模板）
    add_pairs = set()
    while len(add_pairs) < 1500:
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        add_pairs.add((a, b))
    for a, b in add_pairs:
        tpl = random.choice(_ADD_TPL)
        prompt = tpl.format(a=a, b=b)
        completion = f"{a} + {b} = {a + b}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})

    # 2. 减法（800 条）
    sub_pairs = set()
    while len(sub_pairs) < 800:
        a = random.randint(100, 9999)
        b = random.randint(1, a)
        sub_pairs.add((a, b))
    for a, b in sub_pairs:
        tpl = random.choice(_SUB_TPL)
        prompt = tpl.format(a=a, b=b)
        completion = f"{a} - {b} = {a - b}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})

    # 3. 乘法（800 条）
    mul_pairs = set()
    while len(mul_pairs) < 800:
        a = random.randint(2, 99)
        b = random.randint(2, 99)
        mul_pairs.add((a, b))
    for a, b in mul_pairs:
        tpl = random.choice(_MUL_TPL)
        prompt = tpl.format(a=a, b=b)
        completion = f"{a} × {b} = {a * b}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})

    # 4. 除法（800 条，整除）
    div_pairs = set()
    while len(div_pairs) < 800:
        b = random.randint(2, 50)
        q = random.randint(2, 99)
        a = b * q
        div_pairs.add((a, b))
    for a, b in div_pairs:
        tpl = random.choice(_DIV_TPL)
        prompt = tpl.format(a=a, b=b)
        q = a // b
        completion = f"{a} ÷ {b} = {q}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})

    # 5. 解方程（500 条）
    eq_pairs = set()
    while len(eq_pairs) < 500:
        x = random.randint(1, 50)
        k = random.randint(2, 20)
        b = random.randint(1, 50)
        eq_pairs.add((k, b, x))
    for k, b, x in eq_pairs:
        rhs = k * x + b
        prompt = f"解方程：{k}x + {b} = {rhs}"
        completion = f"x = {x}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})

    # 6. 几何（20 题 × 多个改写）
    geo_tpl = ["{p}", "请{p}", "求解：{p}", "请计算：{p}", "数学题：{p}",
               "请回答：{p}", "请问{p}", "求{p}", "计算题：{p}", "请解答：{p}"]
    for q, a in _GEO_ITEMS:
        for tpl in geo_tpl:
            prompt = tpl.format(p=q)
            key = (prompt, a)
            if key not in seen:
                seen.add(key)
                items.append({"prompt": prompt, "completion": a})

    # 7. 数字转中文（300 条）
    num_seen = set()
    while len(num_seen) < 300:
        n = random.randint(1, 9999)
        if n in num_seen:
            continue
        num_seen.add(n)
        prompt = f"把数字 {n} 转成中文"
        completion = f"{n} 的中文是 {_int2zh(n)}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})

    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 5. 对话（日常 / 情感 / 建议）
# ===========================================================================

_DIALOG_PAIRS = [
    ("今天天气真好", "是的，适合出门散步。"),
    ("你好啊", "你好！很高兴见到你。"),
    ("最近怎么样", "还不错，谢谢关心！你呢？"),
    ("很高兴认识你", "我也很高兴认识你。"),
    ("好久不见", "是啊，好久不见了，最近还好吗？"),
    ("你吃饭了吗", "吃过了，你呢？"),
    ("早上好", "早上好！新的一天加油。"),
    ("晚安", "晚安，做个好梦。"),
    ("我要去上班了", "路上注意安全，工作顺利。"),
    ("下班了", "辛苦了，好好休息。"),
    ("我饿了", "那就去吃点东西吧。"),
    ("我渴了", "喝杯水吧。"),
    ("我累了", "休息一下吧，别太辛苦。"),
    ("我困了", "早点休息吧。"),
    ("我心情不好", "想聊聊吗？我会倾听的。"),
    ("我今天很开心", "太好了！愿快乐常伴你左右。"),
    ("我很伤心", "别难过，事情会好起来的。"),
    ("我很生气", "深呼吸，冷静一下会好些。"),
    ("我很紧张", "放轻松，相信你能行。"),
    ("我很焦虑", "试着深呼吸，一切都会过去的。"),
    ("我失恋了", "抱歉听到这个，给自己一些时间疗愈。"),
    ("我考试没考好", "一次失败不算什么，下次继续努力。"),
    ("我得到了这份工作", "恭喜你！真是太棒了。"),
    ("我要搬家了", "搬家辛苦了，新环境加油。"),
    ("我生病了", "好好休息，早日康复。"),
    ("我感冒了", "多喝水，注意保暖。"),
    ("我发烧了", "记得看医生，多休息。"),
    ("我头疼", "休息一下，必要时看医生。"),
    ("我肚子疼", "可能是吃坏了，注意饮食。"),
    ("我睡不着", "试试听些舒缓的音乐放松。"),
    ("我做噩梦了", "别怕，那只是梦，不是真的。"),
    ("我做了个好梦", "美好的梦境，愿你也能在现实中实现。"),
    ("我喜欢看书", "看书是很好的习惯，能增长见识。"),
    ("我喜欢运动", "运动有益健康，坚持下去。"),
    ("我喜欢音乐", "音乐能陶冶情操，真好。"),
    ("我喜欢画画", "艺术是表达情感的好方式。"),
    ("我喜欢旅行", "旅行能开阔眼界，真好。"),
    ("我喜欢做饭", "烹饪是一门艺术，享受其中。"),
    ("我喜欢摄影", "用镜头记录美好瞬间，很有意义。"),
    ("我喜欢宠物", "宠物能带来陪伴和快乐。"),
    ("我想学英语", "学习英语很有用，加油！"),
    ("我想学编程", "编程很有趣，从基础开始吧。"),
    ("我想学画画", "兴趣是最好的老师，加油。"),
    ("我想学钢琴", "音乐能丰富生活，加油。"),
    ("我想减肥", "合理饮食加运动，健康最重要。"),
    ("我想健身", "坚持锻炼，身体会更好。"),
    ("我想戒烟", "戒烟有益健康，加油。"),
    ("我想戒酒", "为了健康，加油。"),
    ("我想旅行", "去看看世界吧，会很精彩。"),
    ("我想回家", "家是温暖的港湾。"),
    ("我想你了", "我也想你。"),
    ("我爱你", "我也爱你。"),
    ("谢谢你的帮助", "不客气，能帮到你就好。"),
    ("非常感谢", "不用谢，应该的。"),
    ("你真厉害", "过奖了，互相学习。"),
    ("你真聪明", "谢谢夸奖。"),
    ("你真漂亮", "谢谢你的赞美。"),
    ("你真善良", "谢谢，你也很好。"),
    ("你真有趣", "和你聊天也很愉快。"),
    ("你真好", "你也是。"),
    ("再见", "再见，保重。"),
    ("回见", "回头见。"),
    ("明天见", "明天见。"),
    ("下次见", "好的，期待下次见面。"),
    ("保重", "你也是，保重。"),
    ("一路顺风", "谢谢，我会的。"),
    ("祝你成功", "谢谢你的祝福。"),
    ("祝你生日快乐", "谢谢你的祝福。"),
    ("祝你新年快乐", "新年快乐，万事如意。"),
    ("祝你圣诞快乐", "圣诞快乐。"),
    ("祝你节日快乐", "节日快乐。"),
    ("你在忙什么", "在处理一些事情，你呢？"),
    ("你在想什么", "在想一些人生的事。"),
    ("你在做什么", "在看书，你呢？"),
    ("你在听什么", "在听一些轻音乐。"),
    ("你在看什么", "在看一部电影。"),
    ("你周末有什么计划", "打算去爬山，你呢？"),
    ("你假期有什么安排", "准备回家看看家人。"),
    ("你喜欢吃什么", "我喜欢吃面条，你呢？"),
    ("你喜欢什么颜色", "我喜欢蓝色，很宁静。"),
    ("你喜欢什么季节", "我喜欢秋天，凉爽舒适。"),
    ("你喜欢什么动物", "我喜欢猫，很可爱。"),
    ("你最近在忙什么", "在工作，偶尔看书放松。"),
    ("你读过这本书吗", "读过，写得很不错。"),
    ("你看过这部电影吗", "看过，很精彩。"),
    ("你会说英语吗", "会一点，还在学习中。"),
    ("你会做饭吗", "会一些简单的菜。"),
    ("你会游泳吗", "会，夏天常去。"),
    ("你会开车吗", "会，有驾照。"),
    ("你打算去哪里", "还没定，可能去旅游。"),
    ("你来自哪里", "我来自中国。"),
    ("你能帮我吗", "当然，请说。"),
    ("你能教我吗", "可以，你想学什么？"),
    ("你能告诉我吗", "可以，请说。"),
    ("你能再说一遍吗", "好的，我再说一次。"),
    ("你能慢点说吗", "好的，我会慢一点。"),
    ("你能大声点吗", "好的，我会大声一点。"),
    ("你能给我建议吗", "当然，请告诉我具体情况。"),
    ("我应该怎么办", "别急，先冷静下来分析问题。"),
    ("我该选择哪个", "要看你的需求和优先级。"),
    ("我该怎么做", "先制定计划，再一步步执行。"),
    ("我该去哪里", "看你想做什么。"),
    ("我该学什么", "可以根据兴趣和职业规划选择。"),
    ("我该买什么", "看你有什么需求。"),
    ("我该读什么书", "可以从经典文学开始。"),
    ("我该看什么电影", "可以试试经典影片。"),
    ("我该听什么音乐", "看你心情，流行或古典都好。"),
    ("我该吃什么", "看你想吃什么，营养均衡即可。"),
    ("我该怎么减肥", "合理饮食和规律运动。"),
    ("我该怎么提高英语", "多听多说多读多写。"),
    ("我该怎么学编程", "从基础语法开始，多动手实践。"),
    ("我该怎么写作文", "先构思框架，再填充内容。"),
    ("我该怎么处理压力", "运动、冥想或与朋友倾诉都有帮助。"),
    ("我该怎么面对失败", "失败是成功之母，从中学习。"),
    ("我该怎么处理冲突", "冷静沟通，理解对方立场。"),
    ("我该怎么管理时间", "列出待办事项，按优先级处理。"),
    ("我该怎么提高效率", "专注一件事，减少干扰。"),
    ("我该怎么养成好习惯", "从小事做起，坚持 21 天。"),
    ("我该怎么改掉坏习惯", "用新习惯替代旧习惯。"),
    ("我该怎么存钱", "制定预算，理性消费。"),
    ("我该怎么投资", "先学习理财知识，分散风险。"),
    ("我该怎么找工作", "准备好简历，多投递多面试。"),
    ("我该怎么准备面试", "研究公司，练习常见问题。"),
    ("我该怎么写简历", "突出优势，简洁明了。"),
    ("我该怎么谈判", "了解对方需求，找到双赢点。"),
    ("我该怎么道歉", "真诚表达歉意，承担责任。"),
    ("我该怎么表达感谢", "真诚地说谢谢，必要时送小礼物。"),
    ("我该怎么拒绝别人", "礼貌但坚定地说明原因。"),
    ("我该怎么赞美别人", "真诚具体地表达欣赏。"),
    ("我该怎么安慰别人", "倾听陪伴，给予支持。"),
    ("我该怎么鼓励别人", "肯定对方优点，给予信心。"),
    ("我该怎么处理愤怒", "深呼吸，冷静后再处理。"),
    ("我该怎么处理悲伤", "允许自己悲伤，找人倾诉。"),
    ("我该怎么处理焦虑", "深呼吸，专注于当下。"),
    ("我该怎么处理孤独", "找朋友聊聊，培养兴趣。"),
    ("我该怎么处理无聊", "学点新东西，或出去走走。"),
    ("我该怎么处理疲倦", "好好休息，劳逸结合。"),
    ("我该怎么处理失望", "调整期望，重新出发。"),
    ("我该怎么处理嫉妒", "化嫉妒为动力，提升自己。"),
    ("我该怎么处理内疚", "承认错误，做出改变。"),
    ("我该怎么处理后悔", "接受过去，着眼未来。"),
    ("我该怎么处理恐惧", "面对恐惧，逐步克服。"),
    ("我该怎么处理迷茫", "理清目标，从小事做起。"),
    ("我该怎么处理挫折", "总结经验，继续前行。"),
    ("我该怎么处理成功", "保持谦逊，继续努力。"),
    ("我该怎么处理变化", "拥抱变化，适应新环境。"),
    ("我该怎么处理选择", "权衡利弊，听从内心。"),
    ("我该怎么处理失去", "接受现实，珍惜拥有。"),
    ("我该怎么处理得到", "感恩珍惜，不骄不躁。"),
    ("我该怎么处理矛盾", "换位思考，互相理解。"),
    ("我该怎么处理误会", "及时沟通，澄清事实。"),
    ("我该怎么处理分歧", "尊重差异，求同存异。"),
    ("我该怎么处理合作", "明确分工，互相支持。"),
    ("我该怎么处理竞争", "公平竞争，共同进步。"),
    ("我该怎么处理批评", "虚心接受，有则改之。"),
    ("我该怎么处理表扬", "谦虚接受，继续努力。"),
    ("我该怎么处理责任", "勇于担当，尽职尽责。"),
    ("我该怎么处理挑战", "勇敢面对，逐一解决。"),
    ("我该怎么处理机会", "抓住机会，全力以赴。"),
    ("我该怎么处理风险", "评估风险，谨慎决策。"),
    ("我该怎么处理复杂", "化繁为简，逐步解决。"),
    ("我该怎么处理简单", "认真对待，精益求精。"),
    ("我该怎么处理大事", "谋定后动，慎重决策。"),
    ("我该怎么处理小事", "注重细节，积少成多。"),
    ("我该怎么处理急事", "冷静应对，分清主次。"),
    ("我该怎么处理缓事", "提前规划，从容应对。"),
    ("我该怎么处理难事", "分解目标，逐步攻克。"),
    ("我该怎么处理易事", "认真完成，不可轻心。"),
    ("我该怎么处理新事", "学习了解，勇于尝试。"),
    ("我该怎么处理旧事", "总结经验，继往开来。"),
    ("我该怎么处理好事", "把握机会，乘势而上。"),
    ("我该怎么处理坏事", "沉着应对，转危为安。"),
    ("我该怎么处理喜事", "分享快乐，再接再厉。"),
    ("我该怎么处理忧事", "调整心态，寻求帮助。"),
    ("我该怎么处理苦事", "坚持不懈，苦尽甘来。"),
    ("我该怎么处理甜事", "珍惜美好，感恩生活。"),
    ("我该怎么处理未知", "保持好奇，勇敢探索。"),
    ("我该怎么学习新技能", "从基础学起，多加练习。"),
    ("我该怎么提升自己", "持续学习，反思总结。"),
    ("我该怎么保持健康", "合理饮食，规律作息，适度运动。"),
    ("我该怎么管理情绪", "觉察情绪，合理表达。"),
    ("我该怎么建立自信", "从小目标开始，积累成功体验。"),
    ("我该怎么克服拖延", "分解任务，立即行动。"),
    ("我该怎么提高专注力", "减少干扰，番茄工作法。"),
    ("我该怎么培养耐心", "接受过程，不急于求成。"),
    ("我该怎么保持积极", "关注美好，感恩生活。"),
    ("我该怎么处理人际关系", "真诚待人，换位思考。"),
    ("我该怎么提高沟通能力", "多倾听，表达清晰。"),
    ("我该怎么提高学习能力", "找到方法，持续练习。"),
    ("我该怎么提高创造力", "多接触新事物，发散思考。"),
    ("我该怎么提高记忆力", "理解记忆，反复复习。"),
    ("我该怎么提高阅读速度", "指读法，减少回视。"),
    ("我该怎么提高写作能力", "多读多写，模仿优秀作品。"),
    ("我该怎么提高口语", "多开口，不怕犯错。"),
    ("我该怎么提高听力", "多听原版材料，循序渐进。"),
]

_DIALOG_TPL = [
    "用户：{p}\n助手：", "我：{p}\n你：", "{p}",
    "对话——用户：{p}", "用户说：{p}\n请回应",
    "对方说：{p}\n你怎么回", "聊天气泡：{p}\n回复：",
    "输入：{p}\n输出：", "请回复：{p}", "对话：{p}",
    "我说：{p}\n你说：", "对方：{p}\n我：",
    "用户输入：{p}\n助手回复：", "{p}\n请给出回应",
    "消息：{p}\n回复：", "对话——{p}", "请回应对方：{p}",
    "{p}——回复", "对方说：{p}\n如何回应",
    "聊天：{p}\n回应：", "用户：{p}\nAI：",
    "用户:{p}\n助手:", "我:{p}\n你:",
    "{p}\n你怎么回", "{p}\n怎么回复",
]


def gen_dialog() -> List[Dict]:
    """对话类：日常 / 情感 / 建议。"""
    items = []
    for tpl in _DIALOG_TPL:
        for prompt_src, resp in _DIALOG_PAIRS:
            prompt = tpl.format(p=prompt_src)
            items.append({"prompt": prompt, "completion": resp})
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 6. 续写（诗词 / 故事 / 描述）
# ===========================================================================

_POETRY_PAIRS = [
    ("春眠不觉晓，", "处处闻啼鸟。"), ("床前明月光，", "疑是地上霜。"),
    ("举头望明月，", "低头思故乡。"), ("白日依山尽，", "黄河入海流。"),
    ("欲穷千里目，", "更上一层楼。"), ("两个黄鹂鸣翠柳，", "一行白鹭上青天。"),
    ("窗含西岭千秋雪，", "门泊东吴万里船。"), ("独在异乡为异客，", "每逢佳节倍思亲。"),
    ("遥知兄弟登高处，", "遍插茱萸少一人。"), ("君自故乡来，", "应知故乡事。"),
    ("来日绮窗前，", "寒梅著花未？"), ("空山不见人，", "但闻人语响。"),
    ("返景入深林，", "复照青苔上。"), ("红豆生南国，", "春来发几枝。"),
    ("愿君多采撷，", "此物最相思。"), ("竹喧归浣女，", "莲动下渔舟。"),
    ("随意春芳歇，", "王孙自可留。"), ("大漠孤烟直，", "长河落日圆。"),
    ("萧关逢候骑，", "都护在燕然。"), ("海上生明月，", "天涯共此时。"),
    ("情人怨遥夜，", "竟夕起相思。"), ("灭烛怜光满，", "披衣觉露滋。"),
    ("不堪盈手赠，", "还寝梦佳期。"), ("前不见古人，", "后不见来者。"),
    ("念天地之悠悠，", "独怆然而涕下。"), ("鹅鹅鹅，", "曲项向天歌。"),
    ("白毛浮绿水，", "红掌拨清波。"), ("锄禾日当午，", "汗滴禾下土。"),
    ("谁知盘中餐，", "粒粒皆辛苦。"), ("离离原上草，", "一岁一枯荣。"),
    ("野火烧不尽，", "春风吹又生。"), ("远芳侵古道，", "晴翠接荒城。"),
    ("又送王孙去，", "萋萋满别情。"), ("千山鸟飞绝，", "万径人踪灭。"),
    ("孤舟蓑笠翁，", "独钓寒江雪。"), ("李白乘舟将欲行，", "忽闻岸上踏歌声。"),
    ("桃花潭水深千尺，", "不及汪伦送我情。"), ("故人西辞黄鹤楼，", "烟花三月下扬州。"),
    ("孤帆远影碧空尽，", "唯见长江天际流。"), ("朝辞白帝彩云间，", "千里江陵一日还。"),
    ("两岸猿声啼不住，", "轻舟已过万重山。"), ("日照香炉生紫烟，", "遥看瀑布挂前川。"),
    ("飞流直下三千尺，", "疑是银河落九天。"), ("天门中断楚江开，", "碧水东流至此回。"),
    ("两岸青山相对出，", "孤帆一片日边来。"), ("月落乌啼霜满天，", "江枫渔火对愁眠。"),
    ("姑苏城外寒山寺，", "夜半钟声到客船。"), ("渭城朝雨浥轻尘，", "客舍青青柳色新。"),
    ("劝君更尽一杯酒，", "西出阳关无故人。"), ("空山新雨后，", "天气晚来秋。"),
    ("明月松间照，", "清泉石上流。"), ("独坐幽篁里，", "弹琴复长啸。"),
    ("深林人不知，", "明月来相照。"), ("山中相送罢，", "日暮掩柴扉。"),
    ("春草明年绿，", "王孙归不归？"), ("松下问童子，", "言师采药去。"),
    ("只在此山中，", "云深不知处。"), ("向晚意不适，", "驱车登古原。"),
    ("夕阳无限好，", "只是近黄昏。"), ("君问归期未有期，", "巴山夜雨涨秋池。"),
    ("何当共剪西窗烛，", "却话巴山夜雨时。"), ("远上寒山石径斜，", "白云生处有人家。"),
    ("停车坐爱枫林晚，", "霜叶红于二月花。"), ("折戟沉沙铁未销，", "自将磨洗认前朝。"),
    ("东风不与周郎便，", "铜雀春深锁二乔。"), ("烟笼寒水月笼沙，", "夜泊秦淮近酒家。"),
    ("商女不知亡国恨，", "隔江犹唱后庭花。"), ("银烛秋光冷画屏，", "轻罗小扇扑流萤。"),
    ("天阶夜色凉如水，", "坐看牵牛织女星。"), ("云母屏风烛影深，", "长河渐落晓星沉。"),
    ("嫦娥应悔偷灵药，", "碧海青天夜夜心。"), ("葡萄美酒夜光杯，", "欲饮琵琶马上催。"),
    ("醉卧沙场君莫笑，", "古来征战几人回。"), ("秦时明月汉时关，", "万里长征人未还。"),
    ("但使龙城飞将在，", "不教胡马度阴山。"), ("黄河远上白云间，", "一片孤城万仞山。"),
    ("羌笛何须怨杨柳，", "春风不度玉门关。"), ("寒雨连江夜入吴，", "平明送客楚山孤。"),
    ("洛阳亲友如相问，", "一片冰心在玉壶。"), ("青海长云暗雪山，", "孤城遥望玉门关。"),
    ("黄沙百战穿金甲，", "不破楼兰终不还。"), ("千里莺啼绿映红，", "水村山郭酒旗风。"),
    ("南朝四百八十寺，", "多少楼台烟雨中。"), ("朱雀桥边野草花，", "乌衣巷口夕阳斜。"),
    ("旧时王谢堂前燕，", "飞入寻常百姓家。"), ("山外青山楼外楼，", "西湖歌舞几时休。"),
    ("暖风熏得游人醉，", "直把杭州作汴州。"), ("毕竟西湖六月中，", "风光不与四时同。"),
    ("接天莲叶无穷碧，", "映日荷花别样红。"), ("泉眼无声惜细流，", "树阴照水爱晴柔。"),
    ("小荷才露尖尖角，", "早有蜻蜓立上头。"), ("半亩方塘一鉴开，", "天光云影共徘徊。"),
    ("问渠那得清如许？", "为有源头活水来。"), ("胜日寻芳泗水滨，", "无边光景一时新。"),
    ("等闲识得东风面，", "万紫千红总是春。"), ("迟日江山丽，", "春风花草香。"),
    ("泥融飞燕子，", "沙暖睡鸳鸯。"), ("江碧鸟逾白，", "山青花欲燃。"),
    ("今春看又过，", "何日是归年。"), ("千锤万凿出深山，", "烈火焚烧若等闲。"),
    ("粉骨碎身浑不怕，", "要留清白在人间。"), ("咬定青山不放松，", "立根原在破岩中。"),
    ("千磨万击还坚劲，", "任尔东西南北风。"), ("墙角数枝梅，", "凌寒独自开。"),
    ("遥知不是雪，", "为有暗香来。"),
]

_STORY_PAIRS = [
    ("从前有一座山，", "山上有一座庙，庙里有一个老和尚和一个小和尚。"),
    ("很久很久以前，", "有一个勇敢的少年踏上了冒险之旅。"),
    ("在一个遥远的国度，", "住着一位善良的公主。"),
    ("深夜里，", "月光洒在窗前，照亮了书桌上的书。"),
    ("春天来了，", "万物复苏，鸟语花香。"),
    ("夏天到了，", "蝉鸣声声，热浪滚滚。"),
    ("秋天来了，", "落叶纷飞，金黄一片。"),
    ("冬天到了，", "白雪皑皑，银装素裹。"),
    ("清晨，", "太阳从东方升起，照亮了大地。"),
    ("傍晚，", "夕阳西下，晚霞映红了天空。"),
    ("夜晚，", "星星点点的灯火在城市中闪烁。"),
    ("下雨了，", "细雨绵绵，滋润着大地。"),
    ("下雪了，", "雪花飘飘，覆盖了屋顶。"),
    ("起风了，", "树叶沙沙作响，凉爽宜人。"),
    ("天晴了，", "阳光明媚，万里无云。"),
    ("他走进房间，", "发现桌上放着一封神秘的信。"),
    ("她打开窗户，", "看到外面下起了大雨。"),
    ("小明起床后，", "刷了牙，洗了脸，准备吃早餐。"),
    ("小红放学回家，", "把书包放下，开始写作业。"),
    ("老师走进教室，", "同学们起立问好。"),
    ("火车缓缓驶出站台，", "窗外的风景开始向后退去。"),
    ("飞机起飞了，", "城市在脚下越来越小。"),
    ("轮船鸣笛，", "缓缓驶离港口。"),
    ("汽车启动，", "沿着公路向前行驶。"),
    ("自行车骑过小巷，", "发出清脆的铃声。"),
    ("书打开了，", "第一页写着一段引人入胜的话。"),
    ("音乐响起，", "优美的旋律在空中回荡。"),
    ("电影开始了，", "大银幕上出现了第一幕。"),
    ("游戏开始了，", "玩家们全神贯注地投入其中。"),
    ("比赛开始了，", "运动员们奋力拼搏。"),
    ("会议开始了，", "主持人走上台发言。"),
    ("课程开始了，", "老师在黑板上写下今天的主题。"),
    ("旅行开始了，", "行李装好，出发去机场。"),
    ("冒险开始了，", "勇士们踏入未知的森林。"),
    ("故事开始了，", "让我们一同进入这个奇妙的世界。"),
    ("旅程开始了，", "前方的路还很长。"),
    ("挑战开始了，", "困难接踵而至。"),
    ("梦想开始了，", "为它努力奋斗吧。"),
    ("新的开始，", "意味着新的希望。"),
    ("旧事结束，", "新的篇章即将展开。"),
    ("太阳升起，", "新的一天充满了可能。"),
    ("月亮升起，", "夜色中带着宁静与神秘。"),
    ("星星闪烁，", "点缀着深邃的夜空。"),
    ("云朵飘过，", "投下变幻的影子。"),
    ("鸟儿飞过，", "留下清脆的鸣叫。"),
    ("鱼儿游过，", "在水中划出优美的弧线。"),
    ("风吹过，", "带来远方的消息。"),
    ("雨落下，", "敲打着屋顶和窗棂。"),
    ("雪飘落，", "为大地披上银装。"),
    ("花开时，", "满园芬芳扑鼻而来。"),
]

_DESC_PAIRS = [
    ("天空是", "湛蓝的，几朵白云悠闲地飘浮其中。"),
    ("大海是", "辽阔的，波涛汹涌，一望无际。"),
    ("森林是", "深邃的，参天大树遮天蔽日。"),
    ("沙漠是", "广袤的，黄沙漫漫，一望无垠。"),
    ("草原是", "碧绿的，牛羊成群，生机盎然。"),
    ("高山是", "巍峨的，云雾缭绕，气势磅礴。"),
    ("河流是", "蜿蜒的，流水潺潺，源远流长。"),
    ("湖泊是", "宁静的，碧波荡漾，倒映着山色。"),
    ("花园是", "美丽的，百花齐放，香气扑鼻。"),
    ("城市是", "繁华的，高楼林立，车水马龙。"),
    ("乡村是", "宁静的，炊烟袅袅，田园如画。"),
    ("夜晚是", "神秘的，星光点点，万籁俱寂。"),
    ("白天是", "明亮的，阳光普照，生机勃勃。"),
    ("春天是", "温暖的，万物复苏，充满希望。"),
    ("夏天是", "炎热的，骄阳似火，蝉鸣不绝。"),
    ("秋天是", "凉爽的，硕果累累，金风送爽。"),
    ("冬天是", "寒冷的，白雪皑皑，银装素裹。"),
    ("清晨是", "清新的，露珠晶莹，鸟语花香。"),
    ("傍晚是", "迷人的，夕阳西下，余晖满地。"),
    ("月夜是", "静谧的，月光如水，万籁俱寂。"),
    ("星空是", "浩瀚的，繁星点点，深邃神秘。"),
    ("阳光是", "温暖的，照亮大地，给予生命。"),
    ("雨露是", "清新的，滋润万物，洗涤尘埃。"),
    ("风是", "自由的，无拘无束，随处可至。"),
    ("云是", "飘渺的，变化多端，自由自在。"),
    ("山是", "沉稳的，巍然不动，历经风雨。"),
    ("水是", "柔和的，滋润万物，却也能穿石。"),
    ("火是", "热烈的，给予温暖，也需谨慎。"),
    ("土是", "厚重的，承载万物，默默奉献。"),
    ("金是", "珍贵的，光芒闪耀，价值连城。"),
    ("木是", "蓬勃的，向上生长，充满生机。"),
    ("花是", "美丽的，芬芳扑鼻，赏心悦目。"),
    ("草是", "顽强的，野火烧不尽，春风吹又生。"),
    ("树是", "高大的，枝繁叶茂，遮风挡雨。"),
    ("叶是", "翠绿的，光合作用，滋养生命。"),
    ("果是", "甜美的，硕果累累，丰收喜悦。"),
    ("鸟是", "自由的，展翅高飞，鸣唱天籁。"),
    ("鱼是", "灵动的，水中嬉戏，自由自在。"),
    ("虫是", "微小的，却也构成生态的一环。"),
    ("兽是", "凶猛的，森林之王，威风凛凛。"),
    ("人是", "智慧的，创造文明，探索未知。"),
    ("家是", "温暖的，遮风挡雨，团圆幸福。"),
    ("国是", "伟大的，山河壮丽，历史悠久。"),
    ("世界是", "多彩的，文化多元，万象更新。"),
    ("宇宙是", "浩瀚的，星辰大海，无垠无际。"),
    ("时间是", "宝贵的，一去不复返，珍惜当下。"),
    ("生命是", "宝贵的，珍惜每一刻，活出精彩。"),
    ("爱是", "美好的，给予温暖，照亮人生。"),
    ("希望是", "光明的，指引方向，给予力量。"),
    ("梦想是", "美好的，努力追逐，终会实现。"),
]

_CONTINUE_TPL = [
    "续写：{p}", "请续写：{p}", "请接着写：{p}", "接下去：{p}",
    "续：{p}", "请继续：{p}", "请接着下面写：{p}",
    "请续写下面这句：{p}", "下面请续写：{p}",
    "请把下面这句续写下去：{p}", "请完成下面这段：{p}",
    "请补全：{p}", "请续写下面这句话：{p}", "请续写下面的内容：{p}",
    "请接着这句写：{p}", "请接着这句继续：{p}", "请把这句续写完整：{p}",
    "请继续往下写：{p}", "请把这段续写：{p}", "续写下面的句子：{p}",
    "请续写下面这句：{p}", "请续写接下来的内容：{p}",
    "请接着这段写：{p}", "请完成续写：{p}", "请续写这句：{p}",
]


def gen_continue() -> List[Dict]:
    """续写类：诗词 / 故事 / 描述。"""
    items = []
    # 诗词：100 × 25 模板 = 2500
    for tpl in _CONTINUE_TPL:
        for prev, nxt in _POETRY_PAIRS:
            prompt = tpl.format(p=prev)
            items.append({"prompt": prompt, "completion": nxt})
    # 故事：50 × 25 模板 = 1250
    for tpl in _CONTINUE_TPL:
        for prev, nxt in _STORY_PAIRS:
            prompt = tpl.format(p=prev)
            items.append({"prompt": prompt, "completion": nxt})
    # 描述：50 × 25 模板 = 1250
    for tpl in _CONTINUE_TPL:
        for prev, nxt in _DESC_PAIRS:
            prompt = tpl.format(p=prev)
            items.append({"prompt": prompt, "completion": nxt})
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 7. 指令（格式化 / 转换 / 摘要）
# ===========================================================================

_UNIT_CONVERT = [
    ("1 千米", "1000 米"), ("1 米", "100 厘米"), ("1 米", "1000 毫米"),
    ("1 厘米", "10 毫米"), ("1 千米", "0.621371 英里"), ("1 英里", "1.60934 千米"),
    ("1 米", "3.28084 英尺"), ("1 英尺", "0.3048 米"), ("1 千克", "1000 克"),
    ("1 克", "1000 毫克"), ("1 吨", "1000 千克"), ("1 千克", "2.20462 磅"),
    ("1 磅", "0.453592 千克"), ("1 小时", "60 分钟"), ("1 分钟", "60 秒"),
    ("1 小时", "3600 秒"), ("1 天", "24 小时"), ("1 天", "1440 分钟"),
    ("1 天", "86400 秒"), ("1 周", "7 天"), ("1 年", "365 天"),
    ("1 年", "12 月"), ("1 升", "1000 毫升"), ("1 立方米", "1000 升"),
    ("1 加仑", "3.78541 升"), ("1 平方米", "10000 平方厘米"),
    ("1 平方千米", "1000000 平方米"), ("1 公顷", "10000 平方米"),
    ("1 亩", "666.67 平方米"), ("1 平方千米", "100 公顷"),
]

# 指令类任务（问题描述, 完成结果）—— 100 条
_INSTR_TASKS = [
    ("把以下数字转成中文：123", "一百二十三"),
    ("把以下数字转成中文：456", "四百五十六"),
    ("把以下数字转成中文：789", "七百八十九"),
    ("把以下数字转成中文：100", "一百"),
    ("把以下数字转成中文：1000", "一千"),
    ("把以下数字转成中文：999", "九百九十九"),
    ("把以下数字转成中文：50", "五十"),
    ("把以下数字转成中文：88", "八十八"),
    ("把以下数字转成中文：666", "六百六十六"),
    ("把以下数字转成中文：888", "八百八十八"),
    ("把以下数字转成中文：2024", "二千零二十四"),
    ("把以下数字转成中文：520", "五百二十"),
    ("把以下数字转成中文：1314", "一千三百一十四"),
    ("把以下数字转成中文：365", "三百六十五"),
    ("把以下数字转成中文：24", "二十四"),
    ("把以下数字转成中文：60", "六十"),
    ("把以下数字转成中文：360", "三百六十"),
    ("把以下数字转成中文：10000", "一万"),
    ("把以下数字转成中文：99", "九十九"),
    ("把以下数字转成中文：11", "十一"),
    ("把以下中文数字转成阿拉伯数字：一百二十三", "123"),
    ("把以下中文数字转成阿拉伯数字：四百五十六", "456"),
    ("把以下中文数字转成阿拉伯数字：七百八十九", "789"),
    ("把以下中文数字转成阿拉伯数字：一百", "100"),
    ("把以下中文数字转成阿拉伯数字：一千", "1000"),
    ("把以下中文数字转成阿拉伯数字：九百九十九", "999"),
    ("把以下中文数字转成阿拉伯数字：五十", "50"),
    ("把以下中文数字转成阿拉伯数字：八十八", "88"),
    ("把以下中文数字转成阿拉伯数字：六百六十六", "666"),
    ("把以下中文数字转成阿拉伯数字：八百八十八", "888"),
    ("把列表 [3, 1, 2] 升序排列", "[1, 2, 3]"),
    ("把列表 [3, 1, 2] 降序排列", "[3, 2, 1]"),
    ("把列表 [5, 3, 8, 1, 9] 升序排列", "[1, 3, 5, 8, 9]"),
    ("把列表 [5, 3, 8, 1, 9] 降序排列", "[9, 8, 5, 3, 1]"),
    ("把列表 [10, 20, 30] 反转", "[30, 20, 10]"),
    ("把列表 [1, 2, 3, 4, 5] 去重", "[1, 2, 3, 4, 5]"),
    ("把列表 [1, 2, 2, 3, 3, 3] 去重", "[1, 2, 3]"),
    ("把列表 [1, 2, 3] 求和", "6"),
    ("把列表 [1, 2, 3, 4, 5] 求和", "15"),
    ("把列表 [10, 20, 30] 求和", "60"),
    ("把字符串 hello 转大写", "HELLO"),
    ("把字符串 HELLO 转小写", "hello"),
    ("把字符串 Hello World 转大写", "HELLO WORLD"),
    ("把字符串 Hello World 转小写", "hello world"),
    ("把字符串 hello world 首字母大写", "Hello World"),
    ("把字符串 hello 反转", "olleh"),
    ("把字符串 world 反转", "dlrow"),
    ("把字符串 hello world 按空格分割", "['hello', 'world']"),
    ("把列表 ['hello', 'world'] 用空格连接", "hello world"),
    ("把列表 ['a', 'b', 'c'] 用逗号连接", "a,b,c"),
    ("统计字符串 hello 中字母 l 的个数", "2"),
    ("统计字符串 abracadabra 中字母 a 的个数", "5"),
    ("统计字符串 mississippi 中字母 s 的个数", "4"),
    ("把 1 到 5 的数字用顿号连接", "1、2、3、4、5"),
    ("把 1 到 5 的数字用逗号连接", "1,2,3,4,5"),
    ("把 1 到 5 的数字用换行连接", "1\n2\n3\n4\n5"),
    ("把 1 到 5 的数字列表化", "[1, 2, 3, 4, 5]"),
    ("把单词 apple, banana, cherry 按字母顺序排列", "apple, banana, cherry"),
    ("把单词 cherry, apple, banana 按字母顺序排列", "apple, banana, cherry"),
    ("把单词 banana, cherry, apple 按字母顺序排列", "apple, banana, cherry"),
    ("把单词 dog, cat, bird 按字母顺序排列", "bird, cat, dog"),
    ("把单词 red, blue, green 按字母顺序排列", "blue, green, red"),
    ("摘要：人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
     "人工智能是模拟和扩展人类智能的技术科学。"),
    ("摘要：机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需明确编程。",
     "机器学习是让计算机从数据中学习的 AI 分支。"),
    ("摘要：深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。",
     "深度学习用多层神经网络模拟人脑学习。"),
    ("摘要：自然语言处理是人工智能和语言学的交叉领域，研究计算机如何理解和处理人类语言。",
     "自然语言处理研究计算机理解人类语言。"),
    ("摘要：计算机视觉是研究如何使计算机从图像或多维数据中获取理解的人工智能领域。",
     "计算机视觉让计算机从图像中获取理解。"),
    ("摘要：云计算是一种通过互联网提供计算服务的模式，包括服务器、存储、数据库、网络、软件等。",
     "云计算通过互联网提供计算服务。"),
    ("摘要：大数据是指无法用传统数据处理应用软件处理的数据集合，具有大量、高速、多样、低价值密度和真实性的特征。",
     "大数据是传统软件难以处理的海量数据。"),
    ("摘要：区块链是一种分布式数据存储、点对点传输、共识机制、加密算法等计算机技术的新型应用模式。",
     "区块链是分布式存储与加密技术的应用。"),
    ("摘要：物联网是指通过信息传感设备，按约定的协议，将任何物体与网络相连接，进行信息交换和通信。",
     "物联网将物体连接到网络进行通信。"),
    ("摘要：5G 是第五代移动通信技术，具有高速率、低时延、大连接的特点。",
     "5G 是高速率低时延的第五代移动通信。"),
    ("摘要：量子计算利用量子力学原理进行计算，在某些问题上比经典计算机快得多。",
     "量子计算利用量子力学实现超快计算。"),
    ("摘要：虚拟现实是一种可以创建和体验虚拟世界的计算机仿真系统。",
     "虚拟现实创建可体验的虚拟世界。"),
    ("摘要：增强现实将虚拟信息叠加到现实世界中，增强用户对现实的感知。",
     "增强现实将虚拟信息叠加到现实。"),
    ("把 JSON 字符串 {\"name\": \"Alice\", \"age\": 25} 格式化",
     '{\n  "name": "Alice",\n  "age": 25\n}'),
    ("把 JSON 字符串 {\"name\": \"Bob\", \"age\": 30} 格式化",
     '{\n  "name": "Bob",\n  "age": 30\n}'),
    ("把字典 {\"name\": \"Alice\", \"age\": 25} 转成 JSON 字符串",
     '{"name": "Alice", "age": 25}'),
    ("把列表 [1, 2, 3] 转成 JSON 字符串", "[1, 2, 3]"),
    ("把字符串 hello 转成 JSON 字符串", '"hello"'),
    ("把数字 42 转成 JSON 字符串", "42"),
    ("把布尔值 True 转成 JSON 字符串", "true"),
    ("把布尔值 False 转成 JSON 字符串", "false"),
    ("把 null 转成 JSON 字符串", "null"),
    ("把二进制 1010 转成十进制", "10"),
    ("把二进制 1111 转成十进制", "15"),
    ("把二进制 10000 转成十进制", "16"),
    ("把十进制 10 转成二进制", "1010"),
    ("把十进制 15 转成二进制", "1111"),
    ("把十进制 16 转成二进制", "10000"),
    ("把十进制 255 转成二进制", "11111111"),
    ("把十进制 10 转成十六进制", "a"),
    ("把十进制 15 转成十六进制", "f"),
    ("把十进制 16 转成十六进制", "10"),
    ("把十进制 255 转成十六进制", "ff"),
    ("把十六进制 a 转成十进制", "10"),
    ("把十六进制 f 转成十进制", "15"),
    ("把十六进制 10 转成十进制", "16"),
    ("把十六进制 ff 转成十进制", "255"),
    ("把八进制 12 转成十进制", "10"),
    ("把十进制 10 转成八进制", "12"),
    ("把字符串 2024-01-15 重新格式化为 2024/01/15", "2024/01/15"),
    ("把字符串 2024/01/15 重新格式化为 2024-01-15", "2024-01-15"),
    ("把字符串 2024年1月15日 重新格式化为 2024-01-15", "2024-01-15"),
    ("把字符串 2024-01-15 重新格式化为 2024年1月15日", "2024年1月15日"),
]

_INSTR_TPL = [
    "{p}", "请{p}", "请执行：{p}", "请处理：{p}", "请完成：{p}",
    "请帮忙：{p}", "请操作：{p}", "任务：{p}", "指令：{p}",
    "请按指令操作：{p}", "请按要求处理：{p}", "请执行以下操作：{p}",
    "请完成以下任务：{p}", "请处理以下内容：{p}", "请按以下指令操作：{p}",
    "请按以下要求处理：{p}", "请执行任务：{p}", "请完成指令：{p}",
    "请处理任务：{p}", "请操作以下：{p}", "请执行以下：{p}",
    "请完成以下：{p}", "请处理以下：{p}", "请按指令执行：{p}", "请按要求完成：{p}",
]

_UNIT_TPL = [
    "把{src}转换成目标单位", "请将{src}换算", "请转换：{src}",
    "{src}等于多少", "请把{src}换算", "请将{src}转换为对应单位",
    "请换算：{src}", "请计算{src}的换算结果", "请将{src}进行单位换算",
    "请回答：{src}是多少", "请换算{src}", "请将{src}换算为常用单位",
    "请计算{src}的等值", "请转换{src}", "请将{src}转换为基本单位",
    "请给出{src}的换算结果", "请将{src}进行转换", "请回答{src}的换算",
    "请算出{src}等于多少", "请把{src}转换为对应单位", "请换算下面这个：{src}",
    "请转换下面这个：{src}", "请给出{src}的转换结果", "请计算{src}的转换",
    "请将{src}转换",
]


def gen_instruction() -> List[Dict]:
    """指令类：格式化 / 转换 / 摘要。"""
    items = []
    # 1. 单位转换：30 项 × 25 模板 = 750
    for tpl in _UNIT_TPL:
        for src, dst in _UNIT_CONVERT:
            prompt = tpl.format(src=src)
            completion = f"{src} = {dst}"
            items.append({"prompt": prompt, "completion": completion})
    # 2. 指令任务：100 项 × 25 模板 = 2500
    for tpl in _INSTR_TPL:
        for q, a in _INSTR_TASKS:
            prompt = tpl.format(p=q)
            items.append({"prompt": prompt, "completion": a})
    # 3. 随机数字转中文（补充到 5000）
    seen = set()
    pair_set = {(p, c) for p, c in items}
    num_tpl = [
        "把数字 {n} 转成中文", "请将 {n} 转为中文数字", "数字 {n} 的中文是什么",
        "请把 {n} 写成中文", "请把阿拉伯数字 {n} 转成中文数字",
        "{n} 用中文怎么写", "请将阿拉伯数字 {n} 转换为中文",
        "请把 {n} 翻译成中文数字", "请将 {n} 转为中文写法",
        "请把数字 {n} 转换成中文表示",
    ]
    while len(items) < PER_CATEGORY * 2:
        n = random.randint(1, 9999)
        zh = _int2zh(n)
        tpl = random.choice(num_tpl)
        prompt = tpl.format(n=n)
        completion = f"{n} 的中文是 {zh}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})
        if len(items) >= PER_CATEGORY * 2:
            break
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 8. 知识（定义 / 解释 / 对比）
# ===========================================================================

_KNOWLEDGE_FACTS = [
    ("什么是人工智能？", "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"),
    ("什么是机器学习？", "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需明确编程。"),
    ("什么是深度学习？", "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。"),
    ("什么是自然语言处理？", "自然语言处理是人工智能和语言学的交叉领域，研究计算机如何理解和处理人类语言。"),
    ("什么是计算机视觉？", "计算机视觉是研究如何使计算机从图像或多维数据中获取理解的人工智能领域。"),
    ("什么是神经网络？", "神经网络是一种模仿生物神经网络结构和功能的计算模型，由大量神经元相互连接组成。"),
    ("什么是算法？", "算法是解决特定问题的一系列明确指令或步骤的有限序列。"),
    ("什么是数据结构？", "数据结构是计算机中存储、组织数据的方式，便于数据的访问和修改。"),
    ("什么是数据库？", "数据库是按照数据结构来存储和管理数据的仓库。"),
    ("什么是操作系统？", "操作系统是管理计算机硬件和软件资源的系统软件，是计算机系统的核心。"),
    ("什么是编程语言？", "编程语言是用来定义计算机程序的形式语言，用于人与计算机之间的通信。"),
    ("什么是编译器？", "编译器是将源代码翻译成目标代码（机器码）的程序。"),
    ("什么是解释器？", "解释器是逐行读取并执行源代码的程序，无需预先编译。"),
    ("什么是 API？", "API（应用程序编程接口）是一组定义和协议，用于构建和集成应用软件。"),
    ("什么是云计算？", "云计算是一种通过互联网提供计算服务的模式，包括服务器、存储、数据库等。"),
    ("什么是大数据？", "大数据是指无法用传统数据处理软件处理的海量数据集合。"),
    ("什么是区块链？", "区块链是一种分布式数据存储、点对点传输、共识机制、加密算法的技术。"),
    ("什么是物联网？", "物联网是通过信息传感设备将物体与网络连接，进行信息交换和通信的网络。"),
    ("什么是 5G？", "5G 是第五代移动通信技术，具有高速率、低时延、大连接的特点。"),
    ("什么是量子计算？", "量子计算利用量子力学原理进行计算，在某些问题上比经典计算机快得多。"),
    ("什么是虚拟现实？", "虚拟现实是一种可以创建和体验虚拟世界的计算机仿真系统。"),
    ("什么是增强现实？", "增强现实将虚拟信息叠加到现实世界中，增强用户对现实的感知。"),
    ("什么是机器学习中的监督学习？", "监督学习是从带标签的训练数据中学习映射关系的机器学习方法。"),
    ("什么是机器学习中的无监督学习？", "无监督学习是从无标签数据中发现模式的机器学习方法。"),
    ("什么是机器学习中的强化学习？", "强化学习是通过与环境交互、根据奖励信号学习策略的机器学习方法。"),
    ("什么是过拟合？", "过拟合是模型在训练数据上表现很好，但在新数据上表现差的现象。"),
    ("什么是欠拟合？", "欠拟合是模型在训练数据和新数据上都表现不好的现象。"),
    ("什么是梯度下降？", "梯度下降是一种优化算法，通过沿梯度反方向迭代更新参数来最小化损失函数。"),
    ("什么是反向传播？", "反向传播是神经网络中通过链式法则计算梯度的算法，用于训练网络。"),
    ("什么是损失函数？", "损失函数是衡量模型预测值与真实值差异的函数，用于指导模型优化。"),
    ("什么是激活函数？", "激活函数是神经网络中引入非线性的函数，如 ReLU、Sigmoid、Tanh 等。"),
    ("什么是卷积神经网络？", "卷积神经网络是一种专门处理网格结构数据（如图像）的深度学习模型。"),
    ("什么是循环神经网络？", "循环神经网络是一种处理序列数据的神经网络，具有记忆能力。"),
    ("什么是 Transformer？", "Transformer 是一种基于自注意力机制的神经网络架构，广泛应用于自然语言处理。"),
    ("什么是注意力机制？", "注意力机制是一种让模型关注输入中重要部分的技术，提升模型表现。"),
    ("什么是生成对抗网络？", "生成对抗网络由生成器和判别器组成，通过对抗训练生成逼真数据。"),
    ("什么是迁移学习？", "迁移学习是将一个任务上学到的知识应用到另一个相关任务的方法。"),
    ("什么是微调？", "微调是在预训练模型基础上，用特定任务数据继续训练以适应新任务的方法。"),
    ("什么是词向量？", "词向量是将词语映射为稠密向量表示的技术，捕捉词语间的语义关系。"),
    ("什么是 tokenizer？", "tokenizer 是将文本分割为 token（词或子词）的工具，是 NLP 的基础步骤。"),
    ("什么是 BPE？", "BPE（字节对编码）是一种子词分词算法，通过合并高频字节对构建词表。"),
    ("什么是 Python？", "Python 是一种高级、通用、解释型编程语言，以简洁易读的语法著称。"),
    ("什么是 Java？", "Java 是一种面向对象的高级编程语言，具有跨平台特性（一次编写，到处运行）。"),
    ("什么是 JavaScript？", "JavaScript 是一种动态、解释型编程语言，主要用于网页交互开发。"),
    ("什么是 C 语言？", "C 语言是一种通用的、过程式的编程语言，具有高效和灵活的特点。"),
    ("什么是 C++？", "C++ 是在 C 语言基础上发展而来的面向对象编程语言。"),
    ("什么是 Go 语言？", "Go 是 Google 开发的静态类型、编译型语言，具有并发编程优势。"),
    ("什么是 Rust？", "Rust 是一种系统编程语言，专注于内存安全和并发性能。"),
    ("什么是 HTML？", "HTML（超文本标记语言）是用于创建网页的标准标记语言。"),
    ("什么是 CSS？", "CSS（层叠样式表）用于描述 HTML 文档的呈现样式。"),
    ("什么是 SQL？", "SQL（结构化查询语言）是用于管理关系数据库的标准语言。"),
    ("什么是 NoSQL？", "NoSQL 是非关系型数据库的统称，适合处理大规模、灵活结构的数据。"),
    ("什么是 Git？", "Git 是一个分布式版本控制系统，用于跟踪文件变化和协作开发。"),
    ("什么是 Docker？", "Docker 是一个容器化平台，将应用及其依赖打包到容器中运行。"),
    ("什么是 Kubernetes？", "Kubernetes 是容器编排系统，用于自动化部署、扩展和管理容器化应用。"),
    ("什么是 Linux？", "Linux 是一种开源的类 Unix 操作系统内核，广泛用于服务器和嵌入式设备。"),
    ("什么是 HTTP？", "HTTP（超文本传输协议）是用于分布式、协作式超媒体信息系统的应用层协议。"),
    ("什么是 HTTPS？", "HTTPS 是 HTTP 的安全版本，通过 SSL/TLS 加密通信内容。"),
    ("什么是 TCP？", "TCP（传输控制协议）是一种面向连接的、可靠的传输层协议。"),
    ("什么是 UDP？", "UDP（用户数据报协议）是一种无连接的、不可靠但快速的传输层协议。"),
    ("什么是 IP 地址？", "IP 地址是分配给网络设备的数字标识，用于在网络中定位设备。"),
    ("什么是 DNS？", "DNS（域名系统）将域名转换为 IP 地址，便于网络访问。"),
    ("什么是 RESTful API？", "RESTful API 是遵循 REST 架构风格的 API，使用 HTTP 方法操作资源。"),
    ("什么是微服务？", "微服务是一种将应用拆分为小型、独立服务的架构风格。"),
    ("什么是 DevOps？", "DevOps 是开发与运维的结合，旨在缩短开发周期、提高交付质量。"),
    ("什么是 CI/CD？", "CI/CD 是持续集成和持续交付/部署的缩写，自动化软件发布流程。"),
    ("什么是敏捷开发？", "敏捷开发是一种迭代、增量的软件开发方法，强调快速响应变化。"),
    ("什么是测试驱动开发？", "测试驱动开发是先写测试再写代码的开发方法，确保代码质量。"),
    ("什么是设计模式？", "设计模式是软件设计中常见问题的可复用解决方案。"),
    ("什么是单例模式？", "单例模式确保一个类只有一个实例，并提供全局访问点。"),
    ("什么是工厂模式？", "工厂模式定义一个创建对象的接口，让子类决定实例化哪个类。"),
    ("什么是观察者模式？", "观察者模式定义对象间一对多的依赖关系，当对象状态变化时通知所有依赖者。"),
    ("什么是递归？", "递归是函数直接或间接调用自身的方法，常用于分治问题。"),
    ("什么是动态规划？", "动态规划通过将问题分解为子问题并存储结果来求解最优化问题。"),
    ("什么是贪心算法？", "贪心算法在每一步选择当前最优解，期望得到全局最优解。"),
    ("什么是分治算法？", "分治算法将问题分解为子问题，分别求解后合并结果。"),
    ("什么是回溯算法？", "回溯算法通过尝试所有可能并回退无效选择来搜索解空间。"),
    ("什么是时间复杂度？", "时间复杂度描述算法运行时间随输入规模增长的变化趋势。"),
    ("什么是空间复杂度？", "空间复杂度描述算法所需额外空间随输入规模增长的变化趋势。"),
    ("什么是大 O 表示法？", "大 O 表示法用于描述算法复杂度的上界，如 O(n)、O(log n)。"),
    ("什么是排序算法？", "排序算法是将一组数据按特定顺序排列的算法。"),
    ("什么是搜索算法？", "搜索算法是在数据集合中查找特定元素的算法。"),
    ("什么是哈希表？", "哈希表是通过哈希函数将键映射到值的数据结构，支持快速查找。"),
    ("什么是二叉树？", "二叉树是每个节点最多有两个子节点的树形数据结构。"),
    ("什么是图？", "图是由顶点和边组成的数据结构，用于表示对象间的关系。"),
    ("什么是栈？", "栈是后进先出（LIFO）的线性数据结构。"),
    ("什么是队列？", "队列是先进先出（FIFO）的线性数据结构。"),
    ("什么是链表？", "链表是通过指针连接节点的线性数据结构，支持高效插入删除。"),
    ("什么是数组？", "数组是连续存储相同类型元素的数据结构，支持随机访问。"),
    ("什么是堆？", "堆是满足堆性质的完全二叉树，常用于优先队列。"),
    ("什么是面向对象编程？", "面向对象编程是以对象为基础的编程范式，包含封装、继承、多态特性。"),
    ("什么是函数式编程？", "函数式编程是将计算视为函数求值的编程范式，强调不可变性和无副作用。"),
    ("什么是闭包？", "闭包是捕获外部变量并保持其引用的函数对象。"),
    ("什么是高阶函数？", "高阶函数是接受函数作为参数或返回函数的函数。"),
    ("什么是lambda表达式？", "lambda 表达式是创建匿名函数的简洁语法。"),
    ("什么是装饰器？", "装饰器是在不修改原函数的前提下扩展其功能的函数。"),
    ("什么是生成器？", "生成器是使用 yield 暂停执行并产生值的特殊函数。"),
    ("什么是迭代器？", "迭代器是实现了 __iter__ 和 __next__ 方法、可逐个访问元素的对象。"),
    ("什么是上下文管理器？", "上下文管理器通过 with 语句管理资源，确保资源正确释放。"),
    ("什么是异常处理？", "异常处理是通过 try-except 捕获和处理运行时错误的机制。"),
    ("什么是类型注解？", "类型注解是为变量和函数添加类型信息的方式，便于静态检查。"),
    ("什么是元编程？", "元编程是编写能够操作程序的代码的技术，如元类、装饰器。"),
    ("什么是并发？", "并发是多个任务在同一时间段内交替执行的机制。"),
    ("什么是并行？", "并行是多个任务在同一时刻同时执行的机制。"),
    ("什么是异步？", "异步是任务发起后不等待完成、通过回调或 await 处理结果的机制。"),
]

_KNOWLEDGE_TPL = [
    "{p}", "请解释：{p}", "请说明：{p}", "请简要解释{p}",
    "请定义：{p}", "请介绍一下{p}", "请解释一下{p}",
    "请简要说明{p}", "请详细解释{p}", "请回答：{p}",
    "请问{p}", "请简述{p}", "请描述{p}", "请阐述{p}",
    "请说明什么是{p}", "请解释什么是{p}", "请帮我解释{p}",
    "请帮我说明{p}", "请帮我回答{p}", "请帮我介绍一下{p}",
    "我想了解{p}", "我想知道{p}", "请教一下{p}",
    "麻烦解释一下{p}", "麻烦说明一下{p}",
]

# 将问题里的"什么是"去掉以适配模板（部分模板自带"什么是"）
_KNOWLEDGE_TPL_NO_PREFIX = [
    "什么是{p}？", "请解释什么是{p}", "请说明什么是{p}",
    "请介绍一下什么是{p}", "请解释一下什么是{p}",
    "请简要说明什么是{p}", "请详细解释什么是{p}",
    "请回答什么是{p}", "请问什么是{p}", "请简述什么是{p}",
    "请描述什么是{p}", "请阐述什么是{p}", "请帮我解释什么是{p}",
    "请帮我说明什么是{p}", "请帮我回答什么是{p}",
    "请帮我介绍一下什么是{p}", "我想了解什么是{p}",
    "我想知道什么是{p}", "请教一下什么是{p}",
    "麻烦解释一下什么是{p}", "麻烦说明一下什么是{p}",
    "请定义{p}", "请简要定义{p}", "请给出{p}的定义",
]


def gen_knowledge() -> List[Dict]:
    """知识类：定义 / 解释 / 对比。"""
    items = []
    for q, a in _KNOWLEDGE_FACTS:
        # 原始问题直接用
        items.append({"prompt": q, "completion": a})
        # 用 _KNOWLEDGE_TPL 改写（{p} = 原始问题）
        for tpl in _KNOWLEDGE_TPL:
            prompt = tpl.format(p=q)
            items.append({"prompt": prompt, "completion": a})
        # 用 _KNOWLEDGE_TPL_NO_PREFIX 改写（{p} = 去掉"什么是"后的主题）
        # 提取主题：去掉"什么是"前缀和"？"后缀
        if q.startswith("什么是") and q.endswith("？"):
            topic = q[3:-1]
            for tpl in _KNOWLEDGE_TPL_NO_PREFIX:
                prompt = tpl.format(p=topic)
                items.append({"prompt": prompt, "completion": a})
    random.shuffle(items)
    return _dedup_and_trim(items, PER_CATEGORY)


# ===========================================================================
# 主函数
# ===========================================================================

# 8 大类生成器
_GENERATORS = [
    ("问答", gen_qa),
    ("翻译", gen_translate),
    ("代码", gen_code),
    ("数学", gen_math),
    ("对话", gen_dialog),
    ("续写", gen_continue),
    ("指令", gen_instruction),
    ("知识", gen_knowledge),
]


def _gen_extra_math(count: int, seen: set) -> List[Dict]:
    """生成额外的数学题（加法/减法/乘法/除法）以补充总量不足。

    可无限生成唯一项（随机数组合），用于填补其他类目去重后的缺口。
    """
    items = []
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("×", lambda a, b: a * b),
    ]
    while len(items) < count:
        op_sym, op_fn = random.choice(ops)
        if op_sym == "+":
            a, b = random.randint(1, 9999), random.randint(1, 9999)
        elif op_sym == "-":
            a = random.randint(100, 99999)
            b = random.randint(1, a)
        else:
            a, b = random.randint(2, 999), random.randint(2, 999)
        result = op_fn(a, b)
        prompt = f"计算：{a} {op_sym} {b}"
        completion = f"{a} {op_sym} {b} = {result}"
        key = (prompt, completion)
        if key not in seen:
            seen.add(key)
            items.append({"prompt": prompt, "completion": completion})
    return items


def generate_all(num_train: int = 40000, num_val: int = 500,
                 train_path: str = DEFAULT_TRAIN_PATH,
                 val_path: str = DEFAULT_VAL_PATH) -> None:
    """生成全部训练数据并写入 JSONL 文件。"""
    print(f"[generate] 开始生成训练数据（目标 train={num_train}, val={num_val}）", flush=True)

    all_items: List[Dict] = []
    per_cat = num_train // len(_GENERATORS)
    remainder = num_train - per_cat * len(_GENERATORS)

    for idx, (name, gen_fn) in enumerate(_GENERATORS):
        target = per_cat + (1 if idx < remainder else 0)
        # 临时调整 PER_CATEGORY 不影响全局；直接调用生成器后裁剪
        items = gen_fn()
        # gen_fn 已按 PER_CATEGORY=5000 返回；若 target 不同则裁剪
        if target != len(items) and target <= len(items):
            items = items[:target]
        elif target > len(items):
            # 不够则循环补充（去重后）
            seen = {(it["prompt"], it["completion"]) for it in items}
            i = 0
            base = list(items)
            while len(items) < target and base:
                cand = base[i % len(base)]
                key = (cand["prompt"], cand["completion"])
                if key not in seen:
                    seen.add(key)
                    items.append(cand)
                i += 1
                if i > target * 5:
                    break
            items = items[:target]
        all_items.extend(items)
        print(f"[generate] {name}: {len(items)} 条", flush=True)

    # 全局去重
    seen = set()
    unique = []
    for it in all_items:
        key = (it["prompt"], it["completion"])
        if key not in seen:
            seen.add(key)
            unique.append(it)
    # 若去重后不足 num_train，用数学随机题补充（可无限生成唯一项）
    if len(unique) < num_train:
        shortfall = num_train - len(unique)
        print(f"[generate] 去重后 {len(unique)} 条，用数学随机题补充 {shortfall} 条",
              flush=True)
        extra = _gen_extra_math(shortfall * 2, seen)
        unique.extend(extra[:shortfall])
        seen.update((it["prompt"], it["completion"]) for it in unique[len(unique) - shortfall:])
    train_data = unique[:num_train]
    random.shuffle(train_data)

    # val 从 train 中随机抽样（去重后）
    val_pool = list(train_data)
    random.shuffle(val_pool)
    val_data = val_pool[:num_val]

    # 写入 train.jsonl
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写入 val.jsonl
    os.makedirs(os.path.dirname(val_path) or ".", exist_ok=True)
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    prompts = [it["prompt"] + "|" + it["completion"] for it in train_data]
    unique_count = len(set(prompts))
    print(f"[generate] 完成：train={len(train_data)} 条 → {train_path}", flush=True)
    print(f"[generate]       val={len(val_data)} 条 → {val_path}", flush=True)
    print(f"[generate] 唯一率: {unique_count}/{len(prompts)} = "
          f"{unique_count / len(prompts) * 100:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="生成 small 模型训练数据（prompt-completion 格式）"
    )
    parser.add_argument("--num-train", type=int, default=40000,
                        help="训练集条数（默认 40000）")
    parser.add_argument("--num-val", type=int, default=500,
                        help="验证集条数（默认 500，从 train 抽样）")
    parser.add_argument("--train-path", type=str, default=DEFAULT_TRAIN_PATH,
                        help=f"训练集输出路径（默认 {DEFAULT_TRAIN_PATH}）")
    parser.add_argument("--val-path", type=str, default=DEFAULT_VAL_PATH,
                        help=f"验证集输出路径（默认 {DEFAULT_VAL_PATH}）")
    args = parser.parse_args()

    generate_all(
        num_train=args.num_train,
        num_val=args.num_val,
        train_path=args.train_path,
        val_path=args.val_path,
    )


if __name__ == "__main__":
    main()
