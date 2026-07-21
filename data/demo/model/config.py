"""CometSparkConfig: CometSpark-v0.1 模型配置 dataclass。

YAML 解析策略：
- 优先使用 ``PyYAML``（``yaml.safe_load`` / ``yaml.safe_dump``），完整支持
  YAML 语法：list / 多行字符串（``|``、``>``）/ 引号转义 / 锚点 / 数值类型等。
- 若运行环境未安装 PyYAML，则降级到模块内置的极简解析器（仅支持标量 + 两层嵌套），
  并打印 warning 提示安装 PyYAML 以获得完整能力。
- ``load_full_config`` / ``save_full_config`` / ``CometSparkConfig.to_yaml`` 均遵循此策略。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Any

# 探测 PyYAML 是否可用；不可用时降级到内置极简解析器
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover - 仅在缺少 PyYAML 的环境触发
    _HAS_YAML = False
    print(
        "[warning] PyYAML 未安装，使用极简解析器（不支持 list/多行字符串/引号转义）。"
        "建议 pip install pyyaml>=6.0"
    )


# ---------------------------------------------------------------------------
# 极简 YAML 解析器（fallback：仅支持标量 + 两层嵌套的最小子集）
# ---------------------------------------------------------------------------
# 限制：
# - 仅支持两层嵌套（顶层 section + key: value）
# - 不支持 list（``- item``）、多行字符串（``|`` / ``>``）、引号转义、锚点
# - 仅识别 int / float / bool / None / str 标量
# 用途：在无 PyYAML 的环境下作为降级路径，保证 config.yml 仍可加载。
# 完整 YAML 语法支持需安装 PyYAML（推荐）。
# ---------------------------------------------------------------------------


_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}


def _parse_scalar(text: str) -> Any:
    """把 YAML 标量字符串解析为 Python 对象（极简版）。

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
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # 去掉可能存在的单层引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _parse_yaml_fallback(text: str) -> dict:
    """极简 YAML 解析器（fallback）：仅支持两层嵌套标量子集。

    语法示例：
        model:
          n_layer: 4
          arch: hybrid
        training:
          lr: 0.001

    返回 {section: {key: value}} 形式的 dict。不支持 list / 多行字符串 / 引号转义。
    """
    result: dict = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        # 去掉行尾换行与注释
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        # 顶层 section（无前导空格，含冒号）
        if not line.startswith((" ", "\t")):
            if ":" in line:
                key = line.split(":", 1)[0].strip()
                value_part = line.split(":", 1)[1].strip()
                if value_part == "":
                    current_section = key
                    result[current_section] = {}
                else:
                    result[key] = _parse_scalar(value_part)
                    current_section = None
            continue
        # 子项（有前导空格）
        if current_section is None or ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        value_part = line.split(":", 1)[1].strip()
        result[current_section][key] = _parse_scalar(value_part)
    return result


def _dump_yaml_fallback(data: dict) -> str:
    """把 dict 序列化为极简 YAML 文本（两层嵌套，fallback 路径）。"""
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
    """把 Python 值格式化为 YAML 标量字符串（fallback 路径）。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


# ---------------------------------------------------------------------------
# CometSparkConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class CometSparkConfig:
    """CometSpark-v0.1 模型配置。

    字段说明：
        vocab_size: 词表大小（实际由 tokenizer 决定，会覆盖）
        n_layer: 总层数（Transformer / Hybrid / VerseNex 均适用）
        n_head: 注意力头数（transformer / versenex 时使用）
        n_embd: 模型隐藏维度（transformer / hybrid 使用；versenex 见 d_model）
        seq_len: 训练序列长度
        dropout: dropout 概率（通用，向后兼容）
        n_kv_head: GQA 的 kv head 数；None 表示 = n_head
        arch: 架构选择，"hybrid" / "transformer" / "versenex"
        ssm_kind: 仅 arch="hybrid" 时生效，"mamba2" 或 "rwkv7"
        sparse_ratio: 仅 arch="hybrid" 时生效，Sparse Attention 层占比
        tie_weights: 是否共享 embedding 与 lm_head 权重
        rope_theta: RoPE 基础频率（默认 10000.0，与 Llama/Mistral 一致）
        max_position_embeddings: 模型支持的最大上下文长度
            （与 seq_len 分离：seq_len 是训练时的实际长度，
            max_position_embeddings 是 RoPE 预计算缓存大小上限）
        attention_dropout: attention softmax 后的 dropout（独立于 dropout）
        hidden_dropout: MLP 中间层的 dropout（独立于 dropout）
        embedding_dropout: embedding 后的 dropout（独立于 dropout）

    Part4 P3.3 新增（仅 arch="versenex" 生效）：
        d_model: VerseNex 隐藏维度（默认与 n_embd 同步）
        attn_top_k: UltraSparse 注意力的 Top-K（0 = 全注意力，默认 64）
        mod_n_parts: MoD DensePart 数量（默认 4）
        mod_n_experts: 每个 DensePart 内的 Expert 数（默认 4）
        mod_top_k_parts: 每个 token 激活的 DensePart 数（默认 2）
        mod_top_k_experts: 每个 DensePart 内激活的 Expert 数（默认 2）
        mod_d_ff: Expert MLP 中间维度（默认 4 * n_embd）
        mod_aux_loss_weight: MoD load balancing loss 权重（默认 0.01）
        medusa_n_heads: Medusa 副头数量（0 = 不使用，默认 0）
        medusa_aux_weight: Medusa 副头 loss 权重（默认 0.5）
        use_position_embed: 是否使用可学习位置 embedding（默认 False 仅 RoPE）
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
    # Task 4.1: 新增字段（向后兼容，默认值不破坏现有配置）
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    embedding_dropout: float = 0.0
    # Part4 P3.3: VerseNex 原生架构字段
    d_model: Optional[int] = None  # None 时使用 n_embd
    attn_top_k: int = 64
    mod_n_parts: int = 4
    mod_n_experts: int = 4
    mod_top_k_parts: int = 2
    mod_top_k_experts: int = 2
    mod_d_ff: Optional[int] = None  # None 时使用 4 * n_embd
    mod_aux_loss_weight: float = 0.01
    medusa_n_heads: int = 0
    medusa_aux_weight: float = 0.5
    use_position_embed: bool = False

    # ------------------------------------------------------------------
    # YAML 持久化
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "CometSparkConfig":
        """从 YAML 文件加载配置。

        优先使用 PyYAML（``yaml.safe_load``）解析；无 PyYAML 时降级到极简解析器。
        仅读取 ``model`` 段下的字段；其他段（training/tokenizer/data/checkpoint）
        留给 TrainerConfig 等其他模块使用，本方法不解析。
        """
        full = load_full_config(path)
        model_cfg = full.get("model", {})
        # 过滤 None 字段以保留 dataclass 默认值
        kwargs = {k: v for k, v in model_cfg.items() if v is not None}
        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        """把当前配置写入 YAML 文件（仅写 model 段）。

        优先使用 PyYAML 序列化；无 PyYAML 时降级到极简序列化器。
        """
        data = {"model": asdict(self)}
        text = _dump_yaml(data)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # Task 4.1: HuggingFace 风格目录持久化
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, dir_path: str) -> "CometSparkConfig":
        """从目录加载配置（HuggingFace 风格）。

        期望目录结构：
            dir_path/
              config.yml   ← 必需，包含 model 段

        与 ``from_yaml`` 的区别：``from_yaml`` 接收完整文件路径，
        ``from_pretrained`` 接收目录路径并自动拼接 ``config.yml``。

        Args:
            dir_path: 配置目录

        Returns:
            :class:`CometSparkConfig` 实例
        """
        cfg_path = os.path.join(dir_path, "config.yml")
        return cls.from_yaml(cfg_path)

    def save_pretrained(self, dir_path: str) -> None:
        """保存配置到目录（HuggingFace 风格）。

        将当前配置以 ``config.yml`` 写入 ``dir_path`` 目录。
        如目录不存在会自动创建。

        Args:
            dir_path: 目标目录
        """
        os.makedirs(dir_path, exist_ok=True)
        cfg_path = os.path.join(dir_path, "config.yml")
        self.to_yaml(cfg_path)

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


def _dump_yaml(data: dict) -> str:
    """把 dict 序列化为 YAML 文本。

    优先使用 PyYAML（``yaml.safe_dump``，保留顺序与中文字符）；无 PyYAML 时
    降级到极简序列化器（仅支持两层嵌套标量）。
    """
    if _HAS_YAML:
        # allow_unicode=True 保留中文字符；sort_keys=False 保留插入顺序
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return _dump_yaml_fallback(data)


def load_full_config(path: str) -> dict:
    """加载完整 YAML 配置（含所有段），返回嵌套 dict。

    优先使用 PyYAML（``yaml.safe_load``）解析，支持完整 YAML 语法
    （list / 多行字符串 / 引号转义 / 数值类型等）。无 PyYAML 时降级到极简解析器
    （仅支持标量 + 两层嵌套，并已在模块加载时打印 warning）。

    供 run.py / trainer.py / evaluate.py 共用。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _HAS_YAML:
        loaded = yaml.safe_load(text)
        # yaml.safe_load 对空文件返回 None，统一为空 dict
        return loaded if isinstance(loaded, dict) else {}
    return _parse_yaml_fallback(text)


def save_full_config(config: dict, path: str) -> None:
    """把完整配置 dict 写回 YAML。

    优先使用 PyYAML 序列化；无 PyYAML 时降级到极简序列化器。
    """
    text = _dump_yaml(config)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


__all__ = ["CometSparkConfig", "load_full_config", "save_full_config"]
