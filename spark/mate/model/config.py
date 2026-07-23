"""CometSpark Mate 模型配置（Part5K1 Task 9.4）。

基于 :class:`spark.model.config.CometSparkV05Config` 派生，新增 VMPC-mate 相关字段。
本配置类只做"配置承载 + 持久化"，真正的模型构建由
``spark/mate/model/model.py`` 的 :class:`CometSparkMateLM` 完成（基于
``verse_nex.CometSparkNexLM``，不重造底层 ``VerseNexBlock``）。

设计要点
--------
- ``arch`` 固定为 ``"versenex"``（继承自 V05Config，自动映射旧值）。
- 0.2zB 目标（旗舰）：默认值与 V05 1B 配置一致
  （``vocab_size=248320, n_embd=1024, n_layer=20``），通过 VMPC 压缩到 0.2zB。
- 新增 VMPC-mate 字段：``vmpc_profile`` / ``vmpc_prune_sparsity`` /
  ``vmpc_quantize_dtype`` / ``vmpc_use_lora`` / ``vmpc_distill``，
  从 YAML 的 ``vmpc:`` 段读取（也兼容 ``model:`` 段内联写法）。
- 新增 ``checkpoint_save_dir`` 字段，从 YAML 的 ``checkpoint:`` 段读取
  （Task 10 用，默认 ``mf_mate``）。
- 保留 from_yaml / to_yaml / from_pretrained / save_pretrained / to_dict / from_dict。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any

# 复用 V05Config 的基类 + YAML 工具函数（避免重复造轮子）
from spark.model.config import (
    CometSparkV05Config,
    load_full_config,
    _dump_yaml,
)


# ---------------------------------------------------------------------------
# VMPC 段字段映射：YAML vmpc.xxx ↔ dataclass vmpc_xxx
# ---------------------------------------------------------------------------

_VMPC_FIELD_MAP = {
    "profile": "vmpc_profile",
    "prune_sparsity": "vmpc_prune_sparsity",
    "quantize_dtype": "vmpc_quantize_dtype",
    "use_lora": "vmpc_use_lora",
    "distill": "vmpc_distill",
}


# ---------------------------------------------------------------------------
# CometSparkMateConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class CometSparkMateConfig(CometSparkV05Config):
    """CometSpark Mate（0.2zB 旗舰）VMPC-mate 预设配置。

    从 :class:`CometSparkV05Config` 派生，默认值与 V05 1B 配置一致
    （旗舰规模），通过 VMPC 压缩到 0.2zB。新增 VMPC-mate 相关字段。

    Args / 新增字段（VMPC-mate）：
        vmpc_profile: VMPC 预设名，固定为 ``"mate"``。
        vmpc_prune_sparsity: 剪枝稀疏度（mate 默认 0.3，中稀疏）。
        vmpc_quantize_dtype: 量化类型（mate 默认 ``"int4"``，4bit/值）。
        vmpc_use_lora: 是否包装 LoRA 适配器（mate 默认 True，旗舰微调友好）。
        vmpc_distill: 是否启用蒸馏（mate 默认 True，需 teacher_model）。
        checkpoint_save_dir: checkpoint 保存目录名（默认 ``"mf_mate"``，Task 10 用）。

    Note:
        架构字段默认值与 V05 1B 一致（旗舰规模）：
        ``vocab_size=248320, n_embd=1024, n_layer=20, n_head=16, n_kv_head=8,
        mod_every=4, num_dense_parts=4, num_experts_per_part=4, top_k=2,
        init_std=0.02``（mate 用中等 init_std，保持稳定收敛）。
        参数量 ≈ 1.12B，VMPC 压缩后目标 0.2zB。
    """

    # 覆盖父类默认值（mate 旗舰配置，与 V05 1B 一致）
    vocab_size: int = 248320
    n_layer: int = 20
    n_embd: int = 1024
    n_head: int = 16
    n_kv_head: int = 8
    seq_len: int = 2048
    max_position_embeddings: int = 4096
    mod_every: int = 4
    num_dense_parts: int = 4
    num_experts_per_part: int = 4
    top_k: int = 2
    window_size: int = 1024
    num_global_tokens: int = 128
    use_alibi: bool = False
    use_rope: bool = True
    # mate 用中等 init_std，保持稳定收敛
    init_std: float = 0.02

    # VMPC-mate 字段
    vmpc_profile: str = "mate"
    vmpc_prune_sparsity: float = 0.3
    vmpc_quantize_dtype: str = "int4"
    vmpc_use_lora: bool = True
    vmpc_distill: bool = True

    # checkpoint 目录（Task 10 用）
    checkpoint_save_dir: str = "mf_mate"

    # ------------------------------------------------------------------
    # YAML 持久化（读取 model + vmpc + checkpoint 段）
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "CometSparkMateConfig":
        """从 YAML 文件加载配置。

        读取 ``model`` 段（架构字段）+ ``vmpc`` 段（VMPC 预设）+
        ``checkpoint`` 段（save_dir）。也兼容 VMPC 字段直接写在 ``model`` 段
        的旧写法（``vmpc_profile`` 等带前缀字段名）。
        """
        full = load_full_config(path)
        model_cfg = dict(full.get("model", {}))
        vmpc_cfg = full.get("vmpc", {})
        ckpt_cfg = full.get("checkpoint", {})

        # 映射 vmpc 段字段到 dataclass 字段（加 vmpc_ 前缀）
        for yaml_key, field_name in _VMPC_FIELD_MAP.items():
            if yaml_key in vmpc_cfg:
                model_cfg[field_name] = vmpc_cfg[yaml_key]

        # 映射 checkpoint 段
        if "save_dir" in ckpt_cfg:
            model_cfg["checkpoint_save_dir"] = ckpt_cfg["save_dir"]

        # 过滤 None
        kwargs = {k: v for k, v in model_cfg.items() if v is not None}
        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        """把当前配置写入 YAML 文件（model + vmpc + checkpoint 段）。"""
        full = asdict(self)
        # 移除内部字段
        full.pop("_ARCH_DEPRECATED_MAP", None)

        # 提取 vmpc 字段到独立段
        vmpc_seg = {
            "profile": full.pop("vmpc_profile", "mate"),
            "prune_sparsity": full.pop("vmpc_prune_sparsity", 0.3),
            "quantize_dtype": full.pop("vmpc_quantize_dtype", "int4"),
            "use_lora": full.pop("vmpc_use_lora", True),
            "distill": full.pop("vmpc_distill", True),
        }
        # 提取 checkpoint 字段到独立段
        ckpt_seg = {"save_dir": full.pop("checkpoint_save_dir", "mf_mate")}

        data = {"model": full, "vmpc": vmpc_seg, "checkpoint": ckpt_seg}
        text = _dump_yaml(data)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # HuggingFace 风格目录持久化（继承父类，使用本类的 from_yaml/to_yaml）
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, dir_path: str) -> "CometSparkMateConfig":
        """从目录加载配置（HuggingFace 风格）。

        期望目录结构：
            dir_path/
              config.yml   ← 必需，包含 model + vmpc + checkpoint 段
        """
        cfg_path = os.path.join(dir_path, "config.yml")
        return cls.from_yaml(cfg_path)

    def save_pretrained(self, dir_path: str) -> None:
        """保存配置到目录（HuggingFace 风格）。"""
        os.makedirs(dir_path, exist_ok=True)
        cfg_path = os.path.join(dir_path, "config.yml")
        self.to_yaml(cfg_path)

    # ------------------------------------------------------------------
    # 便捷方法（继承父类 to_dict / from_dict，自动包含新字段）
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """返回字段字典（剔除内部字段，包含 VMPC 字段）。"""
        d = asdict(self)
        d.pop("_ARCH_DEPRECATED_MAP", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CometSparkMateConfig":
        """从字典构造（忽略未知字段，自动识别 VMPC 字段）。"""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


__all__ = ["CometSparkMateConfig"]
