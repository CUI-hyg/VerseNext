"""CometSpark-V0.5-1B 模型配置（Part4K1 Task 8.3）。

.. note::
    Part5K1.1 目录优化：本模块从 ``spark/model/config.py`` 迁移到
    ``spark/src/base_config.py``，作为 small / mate 双模型的公共基类配置。
    旧路径 ``spark.model.config`` 已删除，请改用 ``spark.src.base_config``。

基于 VerseNex 配置 + 1B 参数预算，针对 Qwen3.5-35B-A3B tokenizer（vocab 248320）
缩放。本配置类只做"配置承载 + 持久化"，真正的模型构建由
``spark/src/base_model.py`` 的 ``CometSparkV05LM`` 完成（基于 ``verse_nex`` 的
``CometSparkNexLM``，不重造底层 ``VerseNexBlock``）。

设计要点
--------
- ``arch`` 固定为 ``"versenex"``（VerseNex 原生架构，TriSparse + MoD）。
- 1B 参数预算通过 ``n_embd / n_layer / layer_pattern / MoD expert`` 控制：
  默认 ``n_embd=1024, n_layer=20, 5 MoD + 15 trisparse, 4 DensePart × 4 Expert × top-2``
  + ``tie_weights=True``，目标参数量 ≈ 1.12B（落在 0.8B-1.2B 区间）。
  其中 embedding(vocab=248320)≈254M + 15 trisparse+5 MoD 层 ≈ 861M。
- ``tokenizer_repo`` 指向 HuggingFace 上的 Qwen tokenizer，由
  ``verse_infra.verse_tokenizer.BPETokenizer.from_pretrained`` 加载。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, List

# 探测 PyYAML（ verse_trainer._load_full_config 也支持 fallback，但本类自带序列化）
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# 极简 YAML 序列化/反序列化（无 PyYAML 时的 fallback）
# ---------------------------------------------------------------------------


_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}


def _parse_scalar(text: str) -> Any:
    """把 YAML 标量字符串解析为 Python 对象（极简版）。"""
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
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _parse_yaml_fallback(text: str) -> dict:
    """极简 YAML 解析（仅支持两层嵌套标量 + 简单 list）。"""
    result: dict = {}
    current_section: Optional[str] = None
    current_list: Optional[list] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            if ":" in line:
                key = line.split(":", 1)[0].strip()
                value_part = line.split(":", 1)[1].strip()
                if value_part == "":
                    current_section = key
                    result[current_section] = {}
                    current_list = None
                else:
                    result[key] = _parse_scalar(value_part)
                    current_section = None
                    current_list = None
            continue
        # 子项
        if current_section is None:
            continue
        stripped = line.strip()
        # list 项：- xxx
        if stripped.startswith("- "):
            val = stripped[2:].strip()
            if current_list is None:
                current_list = []
                result[current_section].setdefault("__list__", None)
            # 取最后一个 key 作为 list 容器
            # 简化：只支持 layer_pattern 这类已知 list
            result[current_section].setdefault("_inline_list", []).append(_parse_scalar(val))
            continue
        if ":" in line:
            key = line.split(":", 1)[0].strip()
            value_part = line.split(":", 1)[1].strip()
            result[current_section][key] = _parse_scalar(value_part)
    # 把 _inline_list 提升为 list 字段
    for sec in result.values():
        if isinstance(sec, dict) and "_inline_list" in sec:
            sec["layer_pattern"] = sec.pop("_inline_list")
    return result


def _dump_yaml_fallback(data: dict) -> str:
    """把 dict 序列化为极简 YAML 文本。"""
    lines = []
    for top_key, top_val in data.items():
        if isinstance(top_val, dict):
            lines.append(f"{top_key}:")
            for k, v in top_val.items():
                if isinstance(v, list):
                    lines.append(f"  {k}:")
                    for item in v:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {k}: {_format_scalar(v)}")
        else:
            lines.append(f"{top_key}: {_format_scalar(top_val)}")
    return "\n".join(lines) + "\n"


def _format_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


def _dump_yaml(data: dict) -> str:
    """把 dict 序列化为 YAML 文本（优先 PyYAML）。"""
    if _HAS_YAML:
        return yaml.safe_dump(
            data, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
    return _dump_yaml_fallback(data)


def load_full_config(path: str) -> dict:
    """加载完整 YAML 配置（含所有段），返回嵌套 dict。

    优先使用 PyYAML；不可用时降级到极简解析器。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _HAS_YAML:
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    return _parse_yaml_fallback(text)


def save_full_config(config: dict, path: str) -> None:
    """把完整配置 dict 写回 YAML。"""
    text = _dump_yaml(config)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# CometSparkV05Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class CometSparkV05Config:
    """CometSpark-V0.5-1B 模型配置。

    基于 VerseNex 原生架构（``VerseNexBlock`` = TriSparse + MoD），目标 1B 参数。

    Args / 字段：
        arch: 架构名，固定为 ``"versenex"``；旧值 ``"transformer"`` /
            ``"hybrid"`` / ``"verse_nex"`` 在 ``__post_init__`` 中自动映射 +
            ``DeprecationWarning``。
        vocab_size: 词表大小。Qwen3.5-35B-A3B tokenizer 词表为 248320；
            调试小配置可用 256（ByteTokenizer）。
        n_layer: 总层数。1B 默认 20；调试 2。
        n_embd: 模型隐藏维度。1B 默认 1024；调试 64。
        n_head: 注意力头数。1B 默认 16；调试 4。
        n_kv_head: GQA 的 kv head 数（None 表示 = n_head）。1B 默认 8（2:1 GQA）。
        seq_len: 训练序列长度。
        dropout: dropout 概率（通用）。
        tie_weights: 是否共享 tok_emb 与 lm_head 权重（节省参数 + 稳定输出）。
        layer_pattern: ``list[str]``，每元素 ``"trisparse"`` 或 ``"mod"``；
            None 则按 ``mod_every`` 自动生成。
        mod_every: 自动生成 layer_pattern 时 MoD 层间隔（每 N 层一个 MoD）。
        num_dense_parts: MoD 的 DensePart 数量（1B 默认 4）。
        num_experts_per_part: 每个 DensePart 的 Expert 数（1B 默认 4）。
        top_k: 每个 token 选出的 Expert 数（1B 默认 2）。
        expert_hidden: Expert 隐藏层维度（None 自动）。
        window_size: TriSparse 滑动窗口大小。
        num_global_tokens: TriSparse 全局 sink token 数。
        use_alibi: TriSparse 是否启用 ALiBi 路径。
        use_rope: TriSparse 是否对 Q/K 应用 RoPE（1B 默认 True，配合 RoPE）。
        rope_theta: RoPE 基础频率。
        max_position_embeddings: 模型支持的最大上下文长度。
        aux_loss_weight: MoD aux loss 权重。
        tokenizer_repo: HuggingFace tokenizer repo（如
            ``"Qwen/Qwen3.5-35B-A3B"``）。由 verse_tokenizer 加载。
        embedding_scale: 是否对 embedding 输出乘以 sqrt(n_embd)（缓解
            训练初期 embedding 过小导致的梯度不平衡 + 生成胡乱输出）。
        temperature_scaling: 生成时 logits / temperature 的默认 temperature
            （<=0 表示禁用，由调用方传入 temperature）。
        init_std: 权重初始化标准差。
        device: 默认设备（``"cpu"`` / ``"cuda"`` / ``"npu"``）。
        parallel_chunks: 训练并行 chunk 数（1=标准 Trainer，>1=ParallelTrainer）。
    """

    # 架构
    arch: str = "versenex"
    vocab_size: int = 248320
    n_layer: int = 20
    n_embd: int = 1024
    n_head: int = 16
    n_kv_head: Optional[int] = 8
    seq_len: int = 2048
    dropout: float = 0.0
    tie_weights: bool = True

    # VerseNex / TriSparse / MoD
    layer_pattern: Optional[List[str]] = None
    mod_every: int = 4
    num_dense_parts: int = 4
    num_experts_per_part: int = 4
    top_k: int = 2
    expert_hidden: Optional[int] = None
    window_size: int = 1024
    num_global_tokens: int = 128
    use_alibi: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    aux_loss_weight: float = 0.01

    # tokenizer
    tokenizer_repo: str = "Qwen/Qwen3.5-35B-A3B"

    # Part4K1 Task 8.7：VerseNex 优化（解决胡乱输出）
    embedding_scale: bool = True
    temperature_scaling: float = 1.0
    init_std: float = 0.02

    # 训练 / 设备
    device: str = "cpu"
    parallel_chunks: int = 1

    # ------------------------------------------------------------------
    # arch 字段统一为 "versenex"
    # ------------------------------------------------------------------

    _ARCH_DEPRECATED_MAP: dict = field(
        default_factory=lambda: {
            "transformer": "versenex",
            "hybrid": "versenex",
            "verse_nex": "versenex",
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        """Part4K1: arch 字段统一为 ``"versenex"``。"""
        if self.arch in self._ARCH_DEPRECATED_MAP:
            new_arch = self._ARCH_DEPRECATED_MAP[self.arch]
            warnings.warn(
                f"arch={self.arch!r} 已废弃，自动映射为 {new_arch!r}。"
                f"原 transformer/hybrid 路径已由 VerseNexLM 统一接管，"
                f"请在 config.yml 中改用 arch: {new_arch}。",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "arch", new_arch)
        if self.arch != "versenex":
            raise ValueError(
                f"arch 必须为 'versenex'（旧值 transformer/hybrid/verse_nex "
                f"会自动映射 + DeprecationWarning），得到 {self.arch!r}"
            )

    # ------------------------------------------------------------------
    # YAML 持久化
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "CometSparkV05Config":
        """从 YAML 文件加载配置（仅读取 ``model`` 段）。"""
        full = load_full_config(path)
        model_cfg = full.get("model", {})
        # 过滤 None + 把 list 类字段还原
        kwargs = {k: v for k, v in model_cfg.items() if v is not None}
        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        """把当前配置写入 YAML 文件（仅写 model 段）。"""
        data = {"model": asdict(self)}
        # 移除内部字段
        data["model"].pop("_ARCH_DEPRECATED_MAP", None)
        text = _dump_yaml(data)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # HuggingFace 风格目录持久化
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, dir_path: str) -> "CometSparkV05Config":
        """从目录加载配置（HuggingFace 风格）。

        期望目录结构：
            dir_path/
              config.yml   ← 必需，包含 model 段
        """
        cfg_path = os.path.join(dir_path, "config.yml")
        return cls.from_yaml(cfg_path)

    def save_pretrained(self, dir_path: str) -> None:
        """保存配置到目录（HuggingFace 风格）。"""
        os.makedirs(dir_path, exist_ok=True)
        cfg_path = os.path.join(dir_path, "config.yml")
        self.to_yaml(cfg_path)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """返回字段字典（剔除内部字段）。"""
        d = asdict(self)
        d.pop("_ARCH_DEPRECATED_MAP", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CometSparkV05Config":
        """从字典构造（忽略未知字段）。"""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


__all__ = [
    "CometSparkV05Config",
    "load_full_config",
    "save_full_config",
]
