"""CometSparkConfig: CometSpark-v0.1 模型配置 dataclass。

由于运行环境无 PyYAML，本模块自带一个支持 config.yml 子集的极简解析器：
- 顶层多段，每段形如 ``key:`` 换行缩进 ``  sub: value``
- 支持 int / float / bool / str / None 四种标量类型
- 不支持 list / 多行字符串 / 引号转义等高级特性（PoC 阶段足够）

如需更复杂 YAML 能力，可后续切换到 PyYAML 而无需改 API。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Any


# ---------------------------------------------------------------------------
# 极简 YAML 解析器（仅支持本配置所需的子集）
# ---------------------------------------------------------------------------


_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}


def _parse_scalar(text: str) -> Any:
    """把 YAML 标量字符串解析为 Python 对象。

    优先级：None -> bool -> int -> float -> str
    """
    s = text.strip()
    if s == "" or s.lower() == "null" or s == "~":
        return None
    low = s.lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    # int（支持负号）
    try:
        return int(s)
    except ValueError:
        pass
    # float
    try:
        return float(s)
    except ValueError:
        pass
    # 去掉可能存在的引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _parse_yaml(text: str) -> dict:
    """极简 YAML 解析器：支持两层嵌套（section + key: value）。

    语法示例：
        model:
          n_layer: 4
          arch: hybrid
        training:
          lr: 0.001

    返回 {section: {key: value}} 形式的 dict。
    """
    result: dict = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        # 去掉行尾换行与注释
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        # 顶层 section（无前导空格，以冒号结尾）
        if not line.startswith((" ", "\t")):
            if ":" in line:
                key = line.split(":", 1)[0].strip()
                value_part = line.split(":", 1)[1].strip()
                if value_part == "":
                    # 新 section
                    current_section = key
                    result[current_section] = {}
                else:
                    # 顶层标量 key: value
                    result[key] = _parse_scalar(value_part)
                    current_section = None
            continue
        # 子项（有前导空格）
        if current_section is None:
            # 跳过未在 section 下的缩进行
            continue
        if ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        value_part = line.split(":", 1)[1].strip()
        result[current_section][key] = _parse_scalar(value_part)
    return result


def _dump_yaml(data: dict) -> str:
    """把 dict 序列化为极简 YAML 文本（两层嵌套）。"""
    lines = []
    for top_key, top_val in data.items():
        if isinstance(top_val, dict):
            lines.append(f"{top_key}:")
            for k, v in top_val.items():
                lines.append(f"  {k}: {_format_scalar(v)}")
        else:
            lines.append(f"{top_key}: {_format_scalar(top_val)}")
    return "\n".join(lines) + "\n"


def _format_scalar(v: Any) -> str:
    """把 Python 值格式化为 YAML 标量字符串。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    # str
    return str(v)


# ---------------------------------------------------------------------------
# CometSparkConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class CometSparkConfig:
    """CometSpark-v0.1 模型配置。

    字段说明：
        vocab_size: 词表大小（实际由 tokenizer 决定，会覆盖）
        n_layer: 总层数（Transformer 或 Hybrid 均适用）
        n_head: 注意力头数（仅 arch="transformer" 时使用）
        n_embd: 模型隐藏维度
        seq_len: 训练序列长度
        dropout: dropout 概率
        n_kv_head: GQA 的 kv head 数；None 表示 = n_head
        arch: 架构选择，"hybrid" 或 "transformer"
        ssm_kind: 仅 arch="hybrid" 时生效，"mamba2" 或 "rwkv7"
        sparse_ratio: 仅 arch="hybrid" 时生效，Sparse Attention 层占比
        tie_weights: 是否共享 embedding 与 lm_head 权重
    """

    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    seq_len: int = 128
    dropout: float = 0.1
    n_kv_head: Optional[int] = None
    arch: str = "hybrid"
    ssm_kind: str = "mamba2"
    sparse_ratio: float = 0.5
    tie_weights: bool = True

    # ------------------------------------------------------------------
    # YAML 持久化
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "CometSparkConfig":
        """从 YAML 文件加载配置。

        仅读取 ``model`` 段下的字段；其他段（training/tokenizer/data/checkpoint）
        留给 TrainerConfig 等其他模块使用，本方法不解析。
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        full = _parse_yaml(text)
        model_cfg = full.get("model", {})
        # 过滤 None 字段以保留 dataclass 默认值
        kwargs = {k: v for k, v in model_cfg.items() if v is not None}
        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        """把当前配置写入 YAML 文件（仅写 model 段）。"""
        data = {"model": asdict(self)}
        text = _dump_yaml(data)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """返回字段字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CometSparkConfig":
        """从字典构造（忽略未知字段）。"""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# 辅助：完整配置（含 training / tokenizer / data / checkpoint）
# ---------------------------------------------------------------------------


def load_full_config(path: str) -> dict:
    """加载完整 YAML 配置（含所有段），返回嵌套 dict。

    供 run.py / trainer.py / evaluate.py 共用。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return _parse_yaml(text)


def save_full_config(config: dict, path: str) -> None:
    """把完整配置 dict 写回 YAML。"""
    text = _dump_yaml(config)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


__all__ = ["CometSparkConfig", "load_full_config", "save_full_config"]
