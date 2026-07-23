"""CometSpark Small 语言模型（Part5K1 Task 9.3）。

设计目标
--------
- **基于 ``verse_nex.CometSparkNexLM``**：本类继承自
  :class:`spark.model.model.CometSparkV05LM`（V05LM 内部组合
  ``CometSparkNexLM``），不重造底层 ``VerseNexBlock``。
- **0.06zB 目标**：通过 :class:`CometSparkSmallConfig` 的极小默认值
  （``vocab_size=256, n_embd=64, n_layer=2`` + 极少 expert）达到 ≈ 100K-400K 参数。
- **VMPC-small 适配**：expert 数更少 + 更高 init_std，通过 config 传入，
  不修改 ``CometSparkNexLM`` 本身。
- **VMPC 压缩便捷接口**：``vmpc_compress_model()`` 用配置中的
  ``vmpc_profile`` 一键压缩。

依赖
----
- ``verse_torch``（Tensor / no_grad）
- ``verse_nex``（``CometSparkNexLM``）—— 间接通过 V05LM 组合
- ``spark.model.model.CometSparkV05LM``（父类）
- ``spark.small.model.config.CometSparkSmallConfig``
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from spark.model.model import CometSparkV05LM

from .config import CometSparkSmallConfig


# ---------------------------------------------------------------------------
# 路径引导：统一委托 spark._bootstrap（幂等，自动注入 verse_torch 等）
# ---------------------------------------------------------------------------
import spark._bootstrap  # noqa: F401 — 副作用导入：设置 sys.path


# ---------------------------------------------------------------------------
# CometSparkSmallLM：Small 模型（基于 CometSparkNexLM，继承 V05LM）
# ---------------------------------------------------------------------------


class CometSparkSmallLM(CometSparkV05LM):
    """CometSpark Small 语言模型（0.06zB 目标，VMPC-small 预设）。

    继承自 :class:`CometSparkV05LM`（内部组合 ``verse_nex.CometSparkNexLM``），
    不重造底层 block。聚焦：

    - **small 配置默认值**：通过 :class:`CometSparkSmallConfig` 控制极小规模
      + VMPC-small 预设（ternary + 高稀疏）。
    - **VMPC 适配微调**：更少 expert（``num_dense_parts=2, num_experts_per_part=2,
      top_k=1``）+ 更高 init_std（``0.04``），通过 config 传入，不修改
      ``CometSparkNexLM``。
    - **VMPC 压缩便捷接口**：:meth:`vmpc_compress_model` 用配置中的
      ``vmpc_profile`` 一键压缩。

    Args:
        config: :class:`CometSparkSmallConfig` 实例。

    Attributes:
        config: 配置对象（CometSparkSmallConfig）。
        net: 内部 :class:`verse_nex.CometSparkNexLM` 实例（由父类构造）。
    """

    def __init__(self, config: CometSparkSmallConfig):
        # 委托父类构造（V05LM 内部组合 CometSparkNexLM，config 驱动架构）
        # CometSparkSmallConfig 是 CometSparkV05Config 的子类，字段完全兼容
        super().__init__(config)

    # ------------------------------------------------------------------
    # VMPC 压缩便捷接口
    # ------------------------------------------------------------------

    def vmpc_compress_model(self, profile: Optional[str] = None) -> "CometSparkSmallLM":
        """用 VMPC 预设压缩模型，返回新的模型实例（**不修改原模型**）。

        使用 :func:`verse_torch.vmpc.vmpc_compress` 对内部 ``self.net``
        （``CometSparkNexLM``）做一键压缩，然后用压缩后的 net 构造新的
        :class:`CometSparkSmallLM`。

        Args:
            profile: VMPC 预设名（``"small"`` / ``"mate"``）；None 则用
                ``self.config.vmpc_profile``。

        Returns:
            压缩后的新 :class:`CometSparkSmallLM` 实例。
        """
        from verse_torch.vmpc import vmpc_compress

        prof = profile or self.config.vmpc_profile
        original_params = self.count_parameters()
        compressed_net = vmpc_compress(self.net, profile=prof)

        # 构造新的 CometSparkSmallLM，替换内部 net
        new_model = type(self)(self.config)
        new_model.net = compressed_net
        object.__setattr__(new_model, "_pre_compress_param_count", original_params)
        return new_model

    # ------------------------------------------------------------------
    # compress 覆盖：用 type(self) 而非 V05LM（保证返回 SmallLM 实例）
    # ------------------------------------------------------------------

    def compress(self, compress_config: dict) -> "CometSparkSmallLM":
        """应用压缩管线，返回压缩后的新 :class:`CometSparkSmallLM` 实例。"""
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
    # from_pretrained 覆盖：使用 CometSparkSmallConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str) -> "CometSparkSmallLM":
        """从目录或单文件加载完整模型。

        目录模式（HuggingFace 风格）：
            path/
              config.yml    ← CometSparkSmallConfig（model + vmpc + checkpoint 段）
              model.pt      ← state_dict (pickle)

        单文件模式：
            path.pt → {"arch": "versenex", "config": dict, "state_dict": dict}
        """
        if os.path.isdir(path):
            config = CometSparkSmallConfig.from_pretrained(path)
            model = cls(config)
            model_pt = os.path.join(path, "model.pt")
            if os.path.exists(model_pt):
                with open(model_pt, "rb") as f:
                    sd = pickle.load(f)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=False)
            return model

        # 单文件模式
        with open(path, "rb") as f:
            payload = pickle.load(f)
        cfg_dict = payload["config"]
        config = CometSparkSmallConfig.from_dict(cfg_dict)
        model = cls(config)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        model.load_state_dict(sd, strict=False)
        return model


# ---------------------------------------------------------------------------
# 工厂函数：CometSparkSmall（0.06zB 目标）
# ---------------------------------------------------------------------------


def CometSparkSmall(
    vocab_size: int = 256,
    n_embd: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    n_kv_head: int = 2,
    seq_len: int = 64,
    max_position_embeddings: int = 256,
    dropout: float = 0.0,
    mod_every: int = 2,
    num_dense_parts: int = 2,
    num_experts_per_part: int = 2,
    top_k: int = 1,
    expert_hidden: Optional[int] = None,
    window_size: int = 32,
    num_global_tokens: int = 4,
    use_alibi: bool = True,
    use_rope: bool = False,
    rope_theta: float = 10000.0,
    aux_loss_weight: float = 0.01,
    tie_weights: bool = True,
    tokenizer_repo: str = "Qwen/Qwen3.5-35B-A3B",
    embedding_scale: bool = True,
    temperature_scaling: float = 1.0,
    init_std: float = 0.04,
    device: str = "cpu",
    parallel_chunks: int = 1,
    vmpc_profile: str = "small",
    vmpc_prune_sparsity: float = 0.5,
    vmpc_quantize_dtype: str = "ternary",
    vmpc_use_lora: bool = False,
    vmpc_distill: bool = False,
    checkpoint_save_dir: str = "mf_small",
) -> CometSparkSmallLM:
    """CometSpark Small 工厂：0.06zB 目标，VMPC-small 预设。

    默认配置（极小，调试用）：
    - vocab_size=256（ByteTokenizer，无网络依赖）
    - n_embd=64, n_layer=2, n_head=4, n_kv_head=2
    - mod_every=2 + n_layer=2 → 1 mod + 1 trisparse（极小 MoD）
    - num_dense_parts=2, num_experts_per_part=2, top_k=1
    - init_std=0.04（small 用更高 init_std，加速小模型收敛）
    - VMPC-small：ternary 量化 + 高稀疏（sparsity=0.5）
    - checkpoint.save_dir=mf_small（Task 10 用）

    参数量 ≈ 100K-400K（调试小配置），3 核 CPU / 5GB 内存沙箱可快速跑通。

    Args:
        详见 :class:`CometSparkSmallConfig` 字段说明。
        新增 VMPC 参数：vmpc_profile / vmpc_prune_sparsity / vmpc_quantize_dtype /
        vmpc_use_lora / vmpc_distill / checkpoint_save_dir。

    Returns:
        :class:`CometSparkSmallLM` 实例。
    """
    config = CometSparkSmallConfig(
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
    return CometSparkSmallLM(config)


__all__ = [
    "CometSparkSmallLM",
    "CometSparkSmall",
]
