"""CometSpark Mate 语言模型（Part5K1 Task 9.4）。

设计目标
--------
- **基于 ``verse_nex.CometSparkNexLM``**：本类继承自
  :class:`spark.model.model.CometSparkV05LM`（V05LM 内部组合
  ``CometSparkNexLM``），不重造底层 ``VerseNexBlock``。
- **0.2zB 目标（旗舰）**：通过 :class:`CometSparkMateConfig` 的旗舰默认值
  （``vocab_size=248320, n_embd=1024, n_layer=20`` + 更多 expert）达到 ≈ 1.12B
  参数，再经 VMPC 压缩到 0.2zB。
- **VMPC-mate 适配**：更多 expert + 中等 init_std + 蒸馏准备，通过 config 传入，
  不修改 ``CometSparkNexLM`` 本身。
- **VMPC 压缩便捷接口**：``vmpc_compress_model()`` 用配置中的
  ``vmpc_profile`` 一键压缩。

依赖
----
- ``verse_torch``（Tensor / no_grad）
- ``verse_nex``（``CometSparkNexLM``）—— 间接通过 V05LM 组合
- ``spark.model.model.CometSparkV05LM``（父类）
- ``spark.mate.model.config.CometSparkMateConfig``
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from spark.model.model import CometSparkV05LM

from .config import CometSparkMateConfig


# ---------------------------------------------------------------------------
# 路径引导：统一委托 spark._bootstrap（幂等，自动注入 verse_torch 等）
# ---------------------------------------------------------------------------
import spark._bootstrap  # noqa: F401 — 副作用导入：设置 sys.path


# ---------------------------------------------------------------------------
# CometSparkMateLM：Mate 模型（基于 CometSparkNexLM，继承 V05LM）
# ---------------------------------------------------------------------------


class CometSparkMateLM(CometSparkV05LM):
    """CometSpark Mate 语言模型（0.2zB 旗舰，VMPC-mate 预设）。

    继承自 :class:`CometSparkV05LM`（内部组合 ``verse_nex.CometSparkNexLM``），
    不重造底层 block。聚焦：

    - **mate 配置默认值**：通过 :class:`CometSparkMateConfig` 控制旗舰规模
      + VMPC-mate 预设（int4 + 中稀疏 + 蒸馏）。
    - **VMPC 适配微调**：更多 expert（``num_dense_parts=4, num_experts_per_part=4,
      top_k=2``）+ 中等 init_std（``0.02``）+ 蒸馏准备，通过 config 传入，
      不修改 ``CometSparkNexLM``。
    - **VMPC 压缩便捷接口**：:meth:`vmpc_compress_model` 用配置中的
      ``vmpc_profile`` 一键压缩。

    Args:
        config: :class:`CometSparkMateConfig` 实例。

    Attributes:
        config: 配置对象（CometSparkMateConfig）。
        net: 内部 :class:`verse_nex.CometSparkNexLM` 实例（由父类构造）。
    """

    # 覆盖父类 _config_class，使 from_pretrained / load_vn 用 Mate 配置（保留 VMPC 字段）
    _config_class = CometSparkMateConfig

    def __init__(self, config: CometSparkMateConfig):
        # 委托父类构造（V05LM 内部组合 CometSparkNexLM，config 驱动架构）
        # CometSparkMateConfig 是 CometSparkV05Config 的子类，字段完全兼容
        super().__init__(config)

    # ------------------------------------------------------------------
    # VMPC 压缩便捷接口
    # ------------------------------------------------------------------

    def vmpc_compress_model(self, profile: Optional[str] = None) -> "CometSparkMateLM":
        """用 VMPC 预设压缩模型，返回新的模型实例（**不修改原模型**）。

        使用 :func:`verse_torch.vmpc.vmpc_compress` 对内部 ``self.net``
        （``CometSparkNexLM``）做一键压缩，然后用压缩后的 net 构造新的
        :class:`CometSparkMateLM`。

        Args:
            profile: VMPC 预设名（``"small"`` / ``"mate"``）；None 则用
                ``self.config.vmpc_profile``。

        Returns:
            压缩后的新 :class:`CometSparkMateLM` 实例。
        """
        from verse_torch.vmpc import vmpc_compress

        prof = profile or self.config.vmpc_profile
        original_params = self.count_parameters()
        compressed_net = vmpc_compress(self.net, profile=prof)

        # 构造新的 CometSparkMateLM，替换内部 net
        new_model = type(self)(self.config)
        new_model.net = compressed_net
        object.__setattr__(new_model, "_pre_compress_param_count", original_params)
        return new_model

    # ------------------------------------------------------------------
    # compress 覆盖：用 type(self) 而非 V05LM（保证返回 MateLM 实例）
    # ------------------------------------------------------------------

    def compress(self, compress_config: dict) -> "CometSparkMateLM":
        """应用压缩管线，返回压缩后的新 :class:`CometSparkMateLM` 实例。"""
        from verse_torch.compress import compress_pipeline

        original_params = self.count_parameters()
        compressed_net, stats = compress_pipeline(
            self.net, compress_config, return_stats=True
        )
        new_model = type(self)(self.config)
        new_model.net = compressed_net
        object.__setattr__(new_model, "_pre_compress_param_count", original_params)
        object.__setattr__(new_model, "_compression_stats_cache", stats)
        return new_model

    # ------------------------------------------------------------------
    # save / save_pretrained 覆盖：支持 format="pt"|"vn"，默认 "vn"
    # （Part5K1 Task 10.1：双模型默认输出 .vn 性能优化格式）
    # ------------------------------------------------------------------

    def save(self, path: str, format: str = "vn", **kwargs) -> None:
        """保存模型。

        Args:
            path: 保存路径（不带扩展名或带都行）。
            format: ``"pt"`` 或 ``"vn"``（默认 ``"vn"``，性能优化格式）。
                - ``"vn"``：调用 :meth:`save_vn`，生成 ``.vn`` 文件
                  （基于 safetensors 的性能优化容器，支持 mmap 零拷贝）。
                - ``"pt"``：调用父类 :meth:`CometSparkV05LM.save`，生成
                  ``.pt`` 文件（pickle，兼容旧接口）。
            **kwargs: ``format="vn"`` 时透传给 :meth:`save_vn`
                （如 ``chat_template`` / ``tokenizer``）。
        """
        if format == "vn":
            # 确保 path 以 .vn 结尾
            if not path.endswith(".vn"):
                path = path + ".vn"
            self.save_vn(path, **kwargs)
        elif format == "pt":
            # 确保 path 以 .pt 结尾
            if not path.endswith(".pt"):
                path = path + ".pt"
            # 父类 save 不接受额外 kwargs，pt 格式仅支持 path
            super().save(path)
        else:
            raise ValueError(f"未知 format: {format}，支持 'pt' / 'vn'")

    def save_pretrained(self, dir_path: str, format: str = "vn", **kwargs) -> None:
        """保存到目录（HuggingFace 风格）。

        Args:
            dir_path: 输出目录路径。
            format: ``"pt"`` 或 ``"vn"``（默认 ``"vn"``）。
                - ``"vn"``：生成 ``config.yml`` + ``model.vn``
                - ``"pt"``：生成 ``config.yml`` + ``model.pt``（父类行为）
            **kwargs: ``format="vn"`` 时透传给 :meth:`save_vn`。
        """
        os.makedirs(dir_path, exist_ok=True)
        # 1. config.yml（委托 config 对象）
        self.config.save_pretrained(dir_path)
        if format == "vn":
            # 2. model.vn
            model_vn = os.path.join(dir_path, "model.vn")
            self.save_vn(model_vn, **kwargs)
        elif format == "pt":
            # 2. model.pt（父类行为）
            import pickle
            sd = {k: np.asarray(v) for k, v in self.state_dict().items()}
            model_pt = os.path.join(dir_path, "model.pt")
            with open(model_pt, "wb") as f:
                pickle.dump(sd, f)
        else:
            raise ValueError(f"未知 format: {format}，支持 'pt' / 'vn'")

    # ------------------------------------------------------------------
    # from_pretrained / load_vn：继承父类健壮实现（通过 _config_class 用 Mate 配置）
    # Part5K1.1：移除 buggy 覆盖，父类 from_pretrained 已支持
    #   .vn / .pt(含config) / 纯 state_dict(best.pt) / 目录，并自动回落 config。
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 工厂函数：CometSparkMate（0.2zB 旗舰）
# ---------------------------------------------------------------------------


def CometSparkMate(
    vocab_size: int = 248320,
    n_embd: int = 1024,
    n_layer: int = 20,
    n_head: int = 16,
    n_kv_head: int = 8,
    seq_len: int = 2048,
    max_position_embeddings: int = 4096,
    dropout: float = 0.0,
    mod_every: int = 4,
    num_dense_parts: int = 4,
    num_experts_per_part: int = 4,
    top_k: int = 2,
    expert_hidden: Optional[int] = None,
    window_size: int = 1024,
    num_global_tokens: int = 128,
    use_alibi: bool = False,
    use_rope: bool = True,
    rope_theta: float = 10000.0,
    aux_loss_weight: float = 0.01,
    tie_weights: bool = True,
    tokenizer_repo: str = "Qwen/Qwen3.5-35B-A3B",
    embedding_scale: bool = True,
    temperature_scaling: float = 1.0,
    init_std: float = 0.02,
    device: str = "cpu",
    parallel_chunks: int = 1,
    vmpc_profile: str = "mate",
    vmpc_prune_sparsity: float = 0.3,
    vmpc_quantize_dtype: str = "int4",
    vmpc_use_lora: bool = True,
    vmpc_distill: bool = True,
    checkpoint_save_dir: str = "mf_mate",
) -> CometSparkMateLM:
    """CometSpark Mate 工厂：0.2zB 旗舰，VMPC-mate 预设。

    默认配置（旗舰，与 V05 1B 一致）：
    - vocab_size=248320（Qwen3.5-35B-A3B tokenizer）
    - n_embd=1024, n_layer=20, n_head=16, n_kv_head=8 (GQA 2:1)
    - mod_every=4 → 每 4 层 1 个 MoD（共 5 MoD + 15 trisparse）
    - num_dense_parts=4, num_experts_per_part=4, top_k=2
    - init_std=0.02（mate 用中等 init_std，保持稳定收敛）
    - VMPC-mate：int4 量化 + 中稀疏（sparsity=0.3）+ LoRA + 蒸馏
    - checkpoint.save_dir=mf_mate（Task 10 用）

    参数量 ≈ 1.12B（VMPC 压缩后目标 0.2zB）。

    Warning:
        默认参数构建真实 1B 模型，在沙箱环境会 OOM。测试时请用小尺寸覆盖：
        ``CometSparkMate(vocab_size=256, n_embd=64, n_layer=2)``。

    Args:
        详见 :class:`CometSparkMateConfig` 字段说明。
        新增 VMPC 参数：vmpc_profile / vmpc_prune_sparsity / vmpc_quantize_dtype /
        vmpc_use_lora / vmpc_distill / checkpoint_save_dir。

    Returns:
        :class:`CometSparkMateLM` 实例。
    """
    config = CometSparkMateConfig(
        arch="versenex",
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        seq_len=seq_len,
        dropout=dropout,
        tie_weights=tie_weights,
        mod_every=mod_every,
        num_dense_parts=num_dense_parts,
        num_experts_per_part=num_experts_per_part,
        top_k=top_k,
        expert_hidden=expert_hidden,
        window_size=window_size,
        num_global_tokens=num_global_tokens,
        use_alibi=use_alibi,
        use_rope=use_rope,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        aux_loss_weight=aux_loss_weight,
        tokenizer_repo=tokenizer_repo,
        embedding_scale=embedding_scale,
        temperature_scaling=temperature_scaling,
        init_std=init_std,
        device=device,
        parallel_chunks=parallel_chunks,
        vmpc_profile=vmpc_profile,
        vmpc_prune_sparsity=vmpc_prune_sparsity,
        vmpc_quantize_dtype=vmpc_quantize_dtype,
        vmpc_use_lora=vmpc_use_lora,
        vmpc_distill=vmpc_distill,
        checkpoint_save_dir=checkpoint_save_dir,
    )
    return CometSparkMateLM(config)


__all__ = [
    "CometSparkMateLM",
    "CometSparkMate",
]
