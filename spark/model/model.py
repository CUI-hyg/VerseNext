"""CometSpark-V0.5-1B 语言模型（Part4K1 Task 8.4 完全重写）。

设计目标
--------
- **不重造底层 block**：直接组合 ``verse_nex.CometSparkNexLM``（内部用
  ``VerseNexBlock`` = TriSparse + MoD），本类只做"架构优化 + 工厂 + 持久化"。
- **1B 参数预算**：``CometSparkV05()`` 通过 ``n_embd=1024, n_layer=20,
  5 MoD + 15 trisparse, 4 DensePart × 4 Expert × top-2`` + ``tie_weights=True``
  达到 ≈ 1.12B 参数（落在 0.8B-1.2B 区间）。
  实际 = 861M(层) + 254M(embedding, tie 共享 head)。
- **解决胡乱输出**（Task 8.7）：
    - embedding scale：``tok_emb(idx) * sqrt(n_embd)``
    - tie_weights：``lm_head`` 与 ``tok_emb`` 共享权重
    - temperature scaling：生成时 ``logits / temperature``
    - 合理初始化（normal + 残差缩放，由 ``CometSparkNexLM._init_weights`` 完成）
- **接口对齐**：``forward(idx)`` → logits / ``generate(idx, ...)`` → ndarray /
  ``save`` / ``load`` / ``from_pretrained`` / ``save_pretrained`` /
  ``count_parameters`` / ``state_dict`` / ``load_state_dict``。

依赖
----
- ``verse_torch``（Tensor / nn / no_grad）
- ``verse_nex``（``CometSparkNexLM`` + ``_build_v02_pattern``）
- ``spark.model.config.CometSparkV05Config``
"""

from __future__ import annotations

import json
import math
import os
import os as _os
import pickle
import sys as _sys
from typing import Optional, List

import numpy as np

from verse_torch import Tensor, no_grad

from .config import CometSparkV05Config


# ---------------------------------------------------------------------------
# 路径自举：确保 verse_torch / verse_nex 可被 import
# ---------------------------------------------------------------------------

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
# spark/model/ → spark/ → /workspace
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_THIS_DIR))
for _dep in ("verse_torch", "verse_nex", "verse_infra"):
    _dep_path = _os.path.join(_REPO_ROOT, "packages", _dep)
    if _os.path.isdir(_dep_path) and _dep_path not in _sys.path:
        _sys.path.insert(0, _dep_path)


def _import_cometspark_nex_lm():
    """延迟导入 CometSparkNexLM，避免 import spark 时强依赖 verse_nex。"""
    from verse_nex.cometspark import CometSparkNexLM
    return CometSparkNexLM


def _import_build_v02_pattern():
    """延迟导入 _build_v02_pattern。"""
    from verse_nex.cometspark import _build_v02_pattern
    return _build_v02_pattern


# ---------------------------------------------------------------------------
# CometSparkV05LM：顶层模型
# ---------------------------------------------------------------------------


class CometSparkV05LM:
    """CometSpark-V0.5-1B 语言模型（基于 VerseNex 原生架构）。

    本类**组合** ``verse_nex.CometSparkNexLM``（内部 ``VerseNexBlock`` =
    TriSparse + MoD），不重造底层 block。聚焦：

    - **layer_pattern / 规模 / 初始化**：通过 ``CometSparkV05Config`` 控制。
    - **embedding scale**：``forward`` 时 ``tok_emb(idx) * sqrt(n_embd)``，
      缓解训练初期 embedding 过小 + 生成胡乱输出。
    - **temperature scaling**：``generate`` 时 ``logits / temperature``。
    - **tie_weights**：``CometSparkNexLM`` 内部已实现 tok_emb 与 head 共享。

    Args:
        config: :class:`CometSparkV05Config` 实例。

    Attributes:
        config: 配置对象。
        net: 内部 :class:`verse_nex.CometSparkNexLM` 实例。
    """

    def __init__(self, config: CometSparkV05Config):
        self.config = config
        CometSparkNexLM = _import_cometspark_nex_lm()

        # 处理 layer_pattern：None 则按 mod_every 自动生成
        layer_pattern = config.layer_pattern
        if layer_pattern is None:
            _build_v02_pattern = _import_build_v02_pattern()
            layer_pattern = _build_v02_pattern(
                n_layer=config.n_layer,
                mod_every=config.mod_every,
            )

        # 构造内部 CometSparkNexLM（不重造底层）
        self.net = CometSparkNexLM(
            vocab_size=config.vocab_size,
            dim=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            layer_pattern=layer_pattern,
            window_size=config.window_size,
            num_global_tokens=config.num_global_tokens,
            use_alibi=config.use_alibi,
            use_rope=config.use_rope,
            max_seq_len=max(config.max_position_embeddings, config.seq_len),
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            num_dense_parts=config.num_dense_parts,
            num_experts_per_part=config.num_experts_per_part,
            top_k=config.top_k,
            expert_hidden=config.expert_hidden,
            aux_loss_weight=config.aux_loss_weight,
            dense_part_names=None,
            tie_weights=config.tie_weights,
            init_std=config.init_std,
        )

        # Task 8.7：embedding scale 缩放因子（sqrt(n_embd)）
        self._emb_scale = math.sqrt(config.n_embd) if config.embedding_scale else 1.0

    # ------------------------------------------------------------------
    # 参数量统计
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """统计可训练参数量。"""
        return self.net.count_parameters()

    # ------------------------------------------------------------------
    # forward（并行训练，可微）
    # ------------------------------------------------------------------

    def forward(self, idx) -> Tensor:
        """整序列并行前向，返回 logits。

        Args:
            idx: ``(B, T)`` 整数索引，Tensor / ndarray / list

        Returns:
            logits: Tensor, shape ``(B, T, vocab_size)``

        Note:
            Task 8.7：若 ``config.embedding_scale=True``，对 embedding 输出
            乘以 ``sqrt(n_embd)``。由于 ``CometSparkNexLM.forward`` 内部
            直接 ``self.tok_emb(idx)`` 后接 blocks，我们在外层手动重放
            该流程并插入 scale。
        """
        if self._emb_scale == 1.0:
            # 无 scale：直接委托
            return self.net.forward(idx)

        # 有 scale：手动重放 forward（tok_emb * sqrt(d) → blocks → norm → head）
        net = self.net
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        elif idx.data.dtype != np.int64:
            idx = Tensor(idx.data.astype(np.int64))

        x = net.tok_emb(idx)  # (B, T, D)
        # embedding scale：乘以 sqrt(n_embd)
        x = x * self._emb_scale
        for block in net.blocks:
            x, _ = block(x, position_offset=0, kv_cache=None)
        x = net.norm(x)
        logits = net.head(x)
        return logits

    def forward_with_aux(self, idx) -> tuple:
        """整序列并行前向，返回 (logits, total_aux_loss)。

        用于 SFT/RL 训练：total_aux_loss 是所有 MoD 层 aux_loss 的总和。
        """
        if self._emb_scale == 1.0:
            return self.net.forward_with_aux(idx)

        net = self.net
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        elif idx.data.dtype != np.int64:
            idx = Tensor(idx.data.astype(np.int64))

        x = net.tok_emb(idx) * self._emb_scale
        total_aux = None
        for block in net.blocks:
            x, layer_state = block(x, position_offset=0, kv_cache=None)
            aux = layer_state["aux"]
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
        x = net.norm(x)
        logits = net.head(x)
        if total_aux is None:
            total_aux = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)
        return logits, total_aux

    # ------------------------------------------------------------------
    # forward_recurrent（单步推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, input_ids, states: Optional[List] = None):
        """单步递推推理接口。

        Args:
            input_ids: ``(B, 1)`` 整数索引
            states: ``list[state]`` 每层一个 state，或 None（首次调用）

        Returns:
            logits: Tensor, shape ``(B, 1, vocab_size)``
            new_states: ``list[state]``
        """
        if self._emb_scale == 1.0:
            return self.net.forward_recurrent(input_ids, states)

        # 有 scale：手动重放 forward_recurrent
        net = self.net
        if not isinstance(input_ids, Tensor):
            idx = Tensor(np.asarray(input_ids, dtype=np.int64))
        elif input_ids.data.dtype != np.int64:
            idx = Tensor(input_ids.data.astype(np.int64))
        else:
            idx = input_ids

        x = net.tok_emb(idx) * self._emb_scale  # (B, 1, D)
        new_states: list = []
        for i, block in enumerate(net.blocks):
            layer_state = states[i] if states is not None else None
            x, new_layer_state = block.forward_recurrent(x, layer_state)
            new_states.append(new_layer_state)
        x = net.norm(x)
        logits = net.head(x)
        return logits, new_states

    # ------------------------------------------------------------------
    # generate（自回归生成，迭代式 for 循环）
    # ------------------------------------------------------------------

    def generate(
        self,
        idx,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> np.ndarray:
        """自回归生成。

        Args:
            idx: prompt 序列，shape (B, T_prompt) 或 (T_prompt,)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度；``temperature==1.0`` 且 ``top_k is None``
                走 greedy + recurrent（O(1) 单步）。
            top_k: top-k 采样；None 表示无限制。
            eos_id: 若指定且末尾非 eos，追加 eos 确保完整 UTF-8 边界。

        Returns:
            generated: ndarray, shape ``(B, T_prompt + max_new_tokens)``

        Note:
            Task 8.7：``config.temperature_scaling > 0`` 时，若调用方未传
            temperature（默认 1.0），用 ``config.temperature_scaling`` 作为
            默认温度；这避免 logits 数值过大导致 softmax 饱和 + 胡乱输出。
        """
        # 若调用方用默认 temperature=1.0 但 config 指定了 temperature_scaling，
        # 且无 top_k，则采用 config 的 temperature（让生成更平滑）。
        # 但为保持 greedy 路径的快速性，仅当 temperature_scaling != 1.0 时覆盖。
        eff_temp = temperature
        if (temperature == 1.0 and top_k is None
                and self.config.temperature_scaling > 0
                and self.config.temperature_scaling != 1.0):
            eff_temp = self.config.temperature_scaling

        return self.net.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=eff_temp,
            top_k=top_k,
            eos_id=eos_id,
        )

    # ------------------------------------------------------------------
    # state_dict / load_state_dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """返回模型参数字典（委托给内部 net）。"""
        return self.net.state_dict()

    def load_state_dict(self, sd: dict, strict: bool = True):
        """加载参数字典到内部 net。"""
        return self.net.load_state_dict(sd, strict=strict)

    def parameters(self):
        """返回模型参数迭代器。"""
        return self.net.parameters()

    def modules(self):
        """返回所有子模块迭代器。"""
        return self.net.modules()

    def named_parameters(self):
        """返回 (name, param) 迭代器。"""
        return self.net.named_parameters()

    def train(self, mode: bool = True):
        """设置训练/评估模式。"""
        return self.net.train(mode)

    def eval(self):
        """切换到评估模式。"""
        return self.net.eval()

    def to(self, device):
        """迁移到指定设备。"""
        return self.net.to(device)

    # ------------------------------------------------------------------
    # save / load（单文件 pickle，兼容旧 CometSparkLM 接口）
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存到 ``.pt`` 单文件（pickle）。

        Payload 结构：
            {
                "arch": "versenex",
                "config": dict,         # CometSparkV05Config.to_dict()
                "state_dict": {name: ndarray},
            }
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "arch": "versenex",
            "config": self.config.to_dict(),
            "state_dict": {k: np.asarray(v) for k, v in self.state_dict().items()},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> "CometSparkV05LM":
        """从 ``.pt`` 文件加载 state_dict 到当前模型（config 不变）。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        self.load_state_dict(sd, strict=False)
        return self

    @classmethod
    def from_pretrained(cls, path: str) -> "CometSparkV05LM":
        """从目录或单文件加载完整模型。

        目录模式（HuggingFace 风格）：
            path/
              config.yml    ← CometSparkV05Config（model 段）
              model.pt      ← state_dict (pickle)

        单文件模式：
            path.pt → {"arch": "versenex", "config": dict, "state_dict": dict}
        """
        if os.path.isdir(path):
            config = CometSparkV05Config.from_pretrained(path)
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
        config = CometSparkV05Config.from_dict(cfg_dict)
        model = cls(config)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        model.load_state_dict(sd, strict=False)
        return model

    def save_pretrained(self, dir_path: str) -> None:
        """保存到目录（HuggingFace 风格）。

        生成：
            dir_path/
              config.yml   ← CometSparkV05Config（model 段）
              model.pt     ← state_dict (pickle)
        """
        os.makedirs(dir_path, exist_ok=True)
        # 1. config.yml
        self.config.save_pretrained(dir_path)
        # 2. model.pt
        sd = {k: np.asarray(v) for k, v in self.state_dict().items()}
        model_pt = os.path.join(dir_path, "model.pt")
        with open(model_pt, "wb") as f:
            pickle.dump(sd, f)

    def get_config(self) -> dict:
        """返回可 ``json.dump`` 的构造参数 dict。"""
        return self.config.to_dict()

    # ------------------------------------------------------------------
    # 压缩接口（委托给内部 net，与旧 CometSparkLM 接口兼容）
    # ------------------------------------------------------------------

    def compress(self, compress_config: dict) -> "CometSparkV05LM":
        """应用压缩管线，返回压缩后的新模型实例（**不修改原模型**）。

        委托给 ``self.net.compress``（CometSparkNexLM 继承自 Module 的方法），
        然后用压缩后的 net 构造新的 CometSparkV05LM。

        Args:
            compress_config: 压缩配置 dict（prune/quantize/lora/ternary/distill 任意组合）

        Returns:
            压缩后的新 :class:`CometSparkV05LM` 实例
        """
        from verse_torch.compress import compress_pipeline

        original_params = self.count_parameters()
        # compress_pipeline 接受 Module（self.net 是 CometSparkNexLM = Module 子类）
        compressed_net, stats = compress_pipeline(
            self.net, compress_config, return_stats=True
        )
        # 构造新的 CometSparkV05LM，替换内部 net
        new_model = CometSparkV05LM(self.config)
        new_model.net = compressed_net
        object.__setattr__(new_model, "_pre_compress_param_count", original_params)
        object.__setattr__(new_model, "_compression_stats_cache", stats)
        return new_model

    def compression_stats(self) -> dict:
        """返回压缩统计信息（与旧 CometSparkLM 接口兼容）。"""
        cache = getattr(self, "_compression_stats_cache", None)
        if cache is not None:
            return cache
        # 无缓存时即时计算
        original = getattr(self, "_pre_compress_param_count", self.count_parameters())
        compressed = self.count_parameters()
        return {
            "original_params": original,
            "compressed_params": compressed,
            "sparsity": 0.0,
            "bits": 32.0,
            "compression_ratio": (original / compressed) if compressed > 0 else 1.0,
        }


# ---------------------------------------------------------------------------
# 工厂函数：CometSparkV05（≈1B） / CometSparkV05Small（调试）
# ---------------------------------------------------------------------------


def CometSparkV05(
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
) -> CometSparkV05LM:
    """CometSpark-V0.5-1B 工厂：目标参数量 ≈ 1.12B（落在 0.8B-1.2B 区间）。

    默认配置：
    - vocab_size=248320（Qwen3.5-35B-A3B tokenizer）
    - n_embd=1024, n_layer=20, n_head=16, n_kv_head=8 (GQA 2:1)
    - layer_pattern: 每 4 层 1 个 MoD（共 5 MoD + 15 trisparse）
    - num_dense_parts=4, num_experts_per_part=4, top_k=2
    - tie_weights=True, max_seq_len=4096
    - use_rope=True, use_alibi=False
    - embedding_scale=True, temperature_scaling=1.0

    参数预算（dim=1024, expert_hidden 自动 ≈ 2688）：
    - 15 个 trisparse 层 + 5 个 mod 层 ≈ 861M
    - Embedding (tie, vocab=248320) = 254M
    - **总 ≈ 1115M ≈ 1.12B**

    Args:
        详见 :class:`CometSparkV05Config` 字段说明。

    Returns:
        :class:`CometSparkV05LM` 实例。
    """
    config = CometSparkV05Config(
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
    )
    return CometSparkV05LM(config)


def CometSparkV05Small(
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
    init_std: float = 0.02,
    device: str = "cpu",
    parallel_chunks: int = 1,
) -> CometSparkV05LM:
    """CometSpark-V0.5 Small：调试小配置（~0.1-0.3M 参数）。

    用于 3 核 CPU / 5GB 内存沙箱下快速跑通端到端流程（训练 / 生成 / 打分）。
    使用纯 TriSparseAttention（mod_every=2 + n_layer=2 → 1 mod + 1 trisparse，
    专家数极少），保持轻量。

    Args:
        详见 :class:`CometSparkV05Config` 字段说明。

    Returns:
        :class:`CometSparkV05LM` 实例（小配置）。
    """
    config = CometSparkV05Config(
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
    )
    return CometSparkV05LM(config)


__all__ = [
    "CometSparkV05LM",
    "CometSparkV05",
    "CometSparkV05Small",
]
