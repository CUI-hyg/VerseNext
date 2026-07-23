"""VerseNex: Hybrid Block (Task 3.6) — **DEPRECATED**.

.. deprecated:: Part4K1 SubTask 2.5
    ``HybridBlock`` / ``HybridLM`` 已标记 deprecated。Part4K1 起
    ``config.yml`` 的 ``arch`` 字段统一为 ``"versenex"``（原 ``"hybrid"``
    自动映射为 ``"versenex"`` 并发 :class:`DeprecationWarning`）。
    原 transformer 路径已由 ``VerseNexLM``（``CometSparkNexLM``）统一接管。
    本模块保留只读兼容（类可正常实例化与使用，但每次实例化会发
    :class:`DeprecationWarning`），下个大版本将删除。

原设计：可配置 SSM : Sparse Attention 层数比例的 Hybrid 架构，参考
RWKV-X / Nemotron-H / Samba 的混合设计。

核心思想：
    纯 SSM 架构（Mamba-2, RWKV-7）有线性复杂度但长程检索能力有限；
    纯 Sparse Attention 长程检索强但开销略大。
    将两者混合：大部分层用 SSM（短程建模），少量层用 Sparse Attention（长程检索），
    兼顾效率与长上下文能力。

设计要点：
- HybridBlock: 单层封装，可选 SSM 或 Sparse Attention
- HybridLM: 完整的 LM 模型，由 embedding + N 个 HybridBlock + LM head 组成
- 层比例可配置，例如 90% SSM + 10% Sparse Attention（默认）
- parallel 模式：所有层用 forward_parallel（训练）
- recurrent 模式：所有层用 forward_recurrent（推理），状态在各层间传递
- 支持 Mamba-2 与 RWKV-7 作为 SSM 内核（默认 Mamba-2）
"""

from __future__ import annotations

import warnings

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.vnn import Linear, Embedding, LayerNorm, RMSNorm, Module, ModuleList

from .mamba2 import Mamba2Block
from .rwkv7 import RWKV7Block
from .sparse_attention import TopKChunkSparseAttention


# Part4K1 SubTask 2.5: 模块级 deprecation 提示
# 直接 import 本模块时发一次 DeprecationWarning（提示用户改用 VerseNexLM）
warnings.warn(
    "verse_nex.hybrid 模块已 deprecated（Part4K1 SubTask 2.5）。"
    "HybridBlock/HybridLM 保留只读兼容，"
    "请改用 verse_nex.CometSparkNexLM（VerseNexLM）作为顶层 LM，"
    "下个大版本将删除本模块。",
    DeprecationWarning,
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# HybridBlock: 单层封装
# ---------------------------------------------------------------------------


class HybridBlock(Module):
    """单层 Hybrid Block，可选 SSM 或 Sparse Attention 类型 — **DEPRECATED**.

    .. deprecated:: Part4K1 SubTask 2.5
        ``HybridBlock`` 已 deprecated。请在 ``config.yml`` 中使用
        ``arch: versenex``（原 ``hybrid`` 自动映射 + DeprecationWarning），
        原 hybrid 路径已由 VerseNexLM 统一接管。本类保留只读兼容，
        实例化时会发 :class:`DeprecationWarning`，下个大版本删除。

    Args:
        block_type: "ssm" 或 "sparse_attn"
        dim: 模型维度
        ssm_kind: "mamba2" 或 "rwkv7"（仅 block_type="ssm" 时有效）
        ssm_kwargs: SSM block 的额外参数
        sparse_kwargs: Sparse Attention 的额外参数
    """

    def __init__(
        self,
        block_type: str,
        dim: int,
        ssm_kind: str = "mamba2",
        ssm_kwargs: dict = None,
        sparse_kwargs: dict = None,
    ):
        # Part4K1 SubTask 2.5: 实例化时发 DeprecationWarning
        warnings.warn(
            "HybridBlock 已 deprecated（Part4K1 SubTask 2.5）。"
            "请改用 VerseNexBlock（TriSparse / MoD）或 arch='versenex'，"
            "下个大版本将删除本类。",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        if block_type not in ("ssm", "sparse_attn"):
            raise ValueError(f"block_type must be 'ssm' or 'sparse_attn', got {block_type!r}")
        if ssm_kind not in ("mamba2", "rwkv7"):
            raise ValueError(f"ssm_kind must be 'mamba2' or 'rwkv7', got {ssm_kind!r}")

        self.block_type = block_type
        self.ssm_kind = ssm_kind
        self.dim = dim

        ssm_kwargs = ssm_kwargs or {}
        sparse_kwargs = sparse_kwargs or {}

        if block_type == "ssm":
            if ssm_kind == "mamba2":
                self.block = Mamba2Block(dim=dim, **ssm_kwargs)
            else:
                self.block = RWKV7Block(dim=dim, **ssm_kwargs)
        else:
            self.block = TopKChunkSparseAttention(dim=dim, **sparse_kwargs)

        # 块前 LayerNorm（与 transformer 残差结构一致）
        self.norm = LayerNorm(dim)

    def forward_parallel(self, x: Tensor) -> Tensor:
        """整序列并行计算（训练）。"""
        x_norm = self.norm(x)
        out = self.block.forward_parallel(x_norm)
        return x + out

    def forward_recurrent(self, x: Tensor, state=None):
        """单步递推（推理）。

        state 格式取决于 block_type：
            - ssm (mamba2): (ssm_state, conv_state) 或 None
            - ssm (rwkv7): (time_mix_state, channel_mix_state) 或 None
            - sparse_attn: (kv_cache, position) 或 None
        """
        x_norm = self.norm(x)
        out, new_state = self.block.forward_recurrent(x_norm, state)
        return x + out, new_state

    def forward(self, x: Tensor, state=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(x)
        elif mode == "recurrent":
            out, new_state = self.forward_recurrent(x, state)
            object.__setattr__(out, "_state", new_state)
            return out
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent")


# ---------------------------------------------------------------------------
# HybridLM: 完整的语言模型
# ---------------------------------------------------------------------------


class HybridLM(Module):
    """Hybrid Language Model — **DEPRECATED**.

    .. deprecated:: Part4K1 SubTask 2.5
        ``HybridLM`` 已 deprecated。请在 ``config.yml`` 中使用
        ``arch: versenex``（原 ``hybrid`` 自动映射 + DeprecationWarning），
        原 hybrid 路径已由 VerseNexLM（``CometSparkNexLM``）统一接管。
        本类保留只读兼容，实例化时会发 :class:`DeprecationWarning`，
        下个大版本删除。

    结构: Embedding -> N × HybridBlock -> LayerNorm -> LM Head

    Args:
        vocab_size: 词表大小
        dim: 模型维度
        n_layers: 总层数
        sparse_ratio: Sparse Attention 层占比（0~1），如 0.1 表示 10% 层用 sparse
        ssm_kind: "mamba2" 或 "rwkv7"
        ssm_kwargs: SSM block 的额外参数
        sparse_kwargs: Sparse Attention 的额外参数
        sparse_placement: "spread" (均匀分布) 或 "last" (最后几层)
        tie_weights: 是否共享 embedding 与 lm_head 权重
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int = 4,
        sparse_ratio: float = 0.1,
        ssm_kind: str = "mamba2",
        ssm_kwargs: dict = None,
        sparse_kwargs: dict = None,
        sparse_placement: str = "spread",
        tie_weights: bool = False,
    ):
        # Part4K1 SubTask 2.5: 实例化时发 DeprecationWarning
        warnings.warn(
            "HybridLM 已 deprecated（Part4K1 SubTask 2.5）。"
            "请改用 verse_nex.CometSparkNexLM（VerseNexLM）"
            "或 config.yml 中 arch: versenex，"
            "下个大版本将删除本类。",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        if not 0.0 <= sparse_ratio <= 1.0:
            raise ValueError(f"sparse_ratio must be in [0, 1], got {sparse_ratio}")
        if sparse_placement not in ("spread", "last"):
            raise ValueError(f"sparse_placement must be 'spread' or 'last', got {sparse_placement!r}")

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.sparse_ratio = sparse_ratio
        self.ssm_kind = ssm_kind
        self.sparse_placement = sparse_placement
        self.tie_weights = tie_weights

        # Embedding
        self.embed = Embedding(vocab_size, dim)

        # 决定哪些层是 sparse attention
        n_sparse = int(round(n_layers * sparse_ratio))
        if sparse_placement == "spread" and n_sparse > 0:
            # 均匀分布：每隔 n_layers / n_sparse 层放一个 sparse
            step = max(1, n_layers // n_sparse)
            sparse_indices = set(
                min(n_layers - 1, i * step + step // 2) for i in range(n_sparse)
            )
            # 去重并排序
            sparse_indices = sorted(sparse_indices)
            # 如果去重后数量不足，补充
            while len(sparse_indices) < n_sparse:
                for i in range(n_layers):
                    if i not in sparse_indices:
                        sparse_indices.append(i)
                        break
                sparse_indices = sorted(set(sparse_indices))
        else:
            # last: 最后 n_sparse 层
            sparse_indices = list(range(n_layers - n_sparse, n_layers)) if n_sparse > 0 else []
        self.sparse_indices = sorted(set(sparse_indices))

        # 构建层
        layers = []
        for i in range(n_layers):
            if i in self.sparse_indices:
                block = HybridBlock(
                    block_type="sparse_attn",
                    dim=dim,
                    ssm_kind=ssm_kind,
                    ssm_kwargs=ssm_kwargs,
                    sparse_kwargs=sparse_kwargs,
                )
            else:
                block = HybridBlock(
                    block_type="ssm",
                    dim=dim,
                    ssm_kind=ssm_kind,
                    ssm_kwargs=ssm_kwargs,
                    sparse_kwargs=sparse_kwargs,
                )
            layers.append(block)
        self.layers = ModuleList(layers)

        # 最终 LayerNorm
        self.final_norm = LayerNorm(dim)

        # LM Head
        if tie_weights:
            self.lm_head = None  # 共享 embed.weight
        else:
            self.lm_head = Linear(dim, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Forward (parallel)
    # ------------------------------------------------------------------

    def forward_parallel(self, input_ids: Tensor) -> Tensor:
        """整序列并行计算（训练）。

        Args:
            input_ids: (B, T) 整数索引
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embed(input_ids)  # (B, T, D)
        for layer in self.layers:
            x = layer.forward_parallel(x)
        x = self.final_norm(x)
        if self.tie_weights:
            # 用 embedding 权重的转置作为输出投影
            logits = x @ self.embed.weight.transpose(-1, -2)
        else:
            logits = self.lm_head(x)
        return logits

    # ------------------------------------------------------------------
    # Forward (recurrent)
    # ------------------------------------------------------------------

    def forward_recurrent(self, input_ids: Tensor, states=None):
        """单步递推（推理）。

        Args:
            input_ids: (B, 1) 整数索引
            states: list of state，每层一个；None 表示初始化
        Returns:
            logits: (B, 1, vocab_size)
            new_states: list of state
        """
        if states is None:
            states = [None] * len(self.layers)
        elif isinstance(states, (list, tuple)):
            states = list(states)
        else:
            states = [states]

        x = self.embed(input_ids)  # (B, 1, D)
        new_states = []
        for i, layer in enumerate(self.layers):
            st = states[i] if i < len(states) else None
            x, new_st = layer.forward_recurrent(x, st)
            new_states.append(new_st)
        x = self.final_norm(x)
        if self.tie_weights:
            logits = x @ self.embed.weight.transpose(-1, -2)
        else:
            logits = self.lm_head(x)
        return logits, new_states

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor, states=None, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.forward_parallel(input_ids)
        elif mode == "recurrent":
            logits, new_states = self.forward_recurrent(input_ids, states)
            object.__setattr__(logits, "_state", new_states)
            return logits
        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected parallel/recurrent")

    # ------------------------------------------------------------------
    # 生成（greedy decoding）
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        mode: str = "recurrent",
    ) -> np.ndarray:
        """贪心生成。

        Args:
            input_ids: (B, T_prompt) 初始 token 序列
            max_new_tokens: 最大生成 token 数
            mode: "parallel" 每步重算整个序列，或 "recurrent" 用状态缓存
        Returns:
            generated: (B, T_prompt + max_new_tokens) ndarray
        """
        with no_grad():
            self.eval()
            B, T_prompt = input_ids.shape if isinstance(input_ids, Tensor) else (1, len(input_ids))
            if not isinstance(input_ids, Tensor):
                input_ids = Tensor(np.asarray(input_ids, dtype=np.int64))

            if mode == "parallel":
                # 简单实现：每步重算整个序列
                cur = input_ids
                for _ in range(max_new_tokens):
                    logits = self.forward_parallel(cur)  # (B, T, V)
                    next_tok = logits.data[:, -1, :].argmax(axis=-1)  # (B,)
                    next_tok_t = Tensor(next_tok[:, None])
                    cur = Tensor(np.concatenate([cur.data, next_tok_t.data], axis=1))
                return cur.data
            elif mode == "recurrent":
                # 用 recurrent 状态缓存：先跑一遍 prompt，再逐步生成
                states = None
                # 处理 prompt
                for t in range(T_prompt):
                    tok = Tensor(input_ids.data[:, t:t + 1])
                    logits, states = self.forward_recurrent(tok, states)
                # 生成新 token
                generated = [input_ids.data]
                last_tok = input_ids.data[:, -1]
                for _ in range(max_new_tokens):
                    tok = Tensor(last_tok[:, None])
                    logits, states = self.forward_recurrent(tok, states)
                    last_tok = logits.data[:, -1, :].argmax(axis=-1)
                    generated.append(last_tok[:, None])
                return np.concatenate(generated, axis=1)
            else:
                raise ValueError(f"Unknown mode: {mode!r}")


__all__ = ["HybridBlock", "HybridLM"]
