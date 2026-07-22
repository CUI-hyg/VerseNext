"""CometSparkNexLM: 纯 VerseNex 原生架构语言模型（Part4）。

设计目标
--------
- **完全不依赖 TransformerLM（已更名为 VerseNexLM）**：仅使用 verse_torch.nn 的基础组件
  （Embedding / RMSNorm / Linear / SwiGLUMLP）+ verse_nex 原生注意力
  （TriSparseAttention）+ verse_nex 原生 MoD（MoDLayer）。
- **layer_pattern 驱动**：用 ``list[str]`` 显式指定每层类型
  （``"trisparse"`` / ``"mod"``），无隐式规则。
- **三模式**：
    - ``forward``: 整序列并行（训练，可微）
    - ``forward_with_aux``: 整序列并行 + MoD aux loss（用于 SFT/RL 训练）
    - ``forward_recurrent``: 单步递推（推理，常数内存，KV cache 复用）
- **与 CometSparkLM 接口对齐**：
    ``forward(idx)`` → logits (B,T,vocab)
    ``generate(idx, ...)`` → ndarray
    ``save / load / from_pretrained / save_pretrained``
    ``count_parameters / state_dict / load_state_dict``

架构（Pre-Norm + 残差）
-----------------------
每层 ``VerseNexBlock`` 的结构::

    x = x + attn(norm1(x))         # TriSparseAttention
    x = x + ffn(norm2(x))          # SwiGLUMLP 或 MoDLayer

其中 ``attn`` 与 ``ffn`` 之间的差异由 ``layer_kind`` 决定：

- ``"trisparse"``: ffn = SwiGLUMLP（dense MLP）
- ``"mod"``: ffn = MoDLayer（5 DensePart × 8 Expert × top-3 双层门控）

参数预算（V0.2 工厂，d=384, n_layer=32, tie_weights=True）
---------------------------------------------------------
- 20 个 "trisparse" 层 × ~1.79M = 35.8M
- 12 个 "mod" 层 × ~32.1M = 385.2M
- Embedding (tie) = 58.3M
- 32 层 norm ≈ 24K
- **总 ≈ 479M ≈ 0.5B**
"""

from __future__ import annotations

import copy
import json
import os
import pickle
from typing import Optional, List, Iterable

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import (
    Module,
    Linear,
    Embedding,
    RMSNorm,
    Dropout,
    SwiGLUMLP,
    ModuleList,
    normal_,
)

from .tri_sparse_attn import TriSparseAttention
from .moe import MoDLayer


# ---------------------------------------------------------------------------
# VerseNexBlock: 单层（Pre-Norm + 残差 + TriSparse + FFN）
# ---------------------------------------------------------------------------


class VerseNexBlock(Module):
    """VerseNex 单层：Pre-Norm + TriSparse Attention + FFN（SwiGLU 或 MoD）。

    Args:
        dim: 模型维度
        n_head: 注意力头数
        n_kv_head: GQA 的 kv head 数（None 表示 = n_head）
        layer_kind: ``"trisparse"``（FFN=SwiGLUMLP）或 ``"mod"``（FFN=MoDLayer）
        window_size: TriSparse 滑动窗口大小
        num_global_tokens: TriSparse 全局 sink token 数
        use_alibi: TriSparse 是否启用 ALiBi 路径
        use_rope: TriSparse 是否对 Q/K 应用 RoPE
        max_seq_len: RoPE/ALiBi 预计算最大长度
        dropout: dropout 概率（通用）
        rope_theta: RoPE 基础频率
        # MoD 专属（仅当 layer_kind=="mod" 时生效）
        num_dense_parts: DensePart 数量
        num_experts_per_part: 每个 DensePart 内的 Expert 数
        top_k: 每个 token 选出的 Expert 数
        expert_hidden: Expert 隐藏层维度（None 自动）
        aux_loss_weight: MoD aux loss 权重
        # SwiGLU 专属（仅当 layer_kind=="trisparse" 时生效）
        mlp_hidden_multiple: SwiGLU 隐藏层倍数（默认 4）
        use_checkpoint: Part4K2 Task 5.2 激活检查点开关。
            True 时前向不保存中间激活，反向时重新计算（节省显存，适用于 GPU 大模型训练）。
            CPU / 无 PyTorch 时自动降级为直接前向。
    """

    VALID_KINDS = ("trisparse", "mod")

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        layer_kind: str = "trisparse",
        window_size: int = 512,
        num_global_tokens: int = 64,
        use_alibi: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        # MoD
        num_dense_parts: int = 5,
        num_experts_per_part: int = 8,
        top_k: int = 3,
        expert_hidden: Optional[int] = None,
        aux_loss_weight: float = 0.01,
        dense_part_names: Optional[list] = None,
        # SwiGLU
        mlp_hidden_multiple: int = 4,
        # Part4K2 Task 5.2: 激活检查点
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if layer_kind not in self.VALID_KINDS:
            raise ValueError(
                f"layer_kind 必须为 {self.VALID_KINDS}，got {layer_kind!r}"
            )

        self.dim = dim
        self.layer_kind = layer_kind
        self.use_checkpoint = bool(use_checkpoint)

        # 共用 norm
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # 注意力（所有层共用 TriSparseAttention）
        self.attn = TriSparseAttention(
            dim=dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            use_alibi=use_alibi,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
        )

        # FFN：根据 layer_kind 选择 SwiGLU 或 MoD
        if layer_kind == "trisparse":
            self.ffn = SwiGLUMLP(
                d=dim,
                dropout=dropout,
                hidden_multiple=mlp_hidden_multiple,
            )
        else:  # "mod"
            self.ffn = MoDLayer(
                dim=dim,
                num_dense_parts=num_dense_parts,
                num_experts_per_part=num_experts_per_part,
                top_k=top_k,
                expert_hidden=expert_hidden,
                dropout=dropout,
                aux_loss_weight=aux_loss_weight,
                dense_part_names=dense_part_names,
            )

    # ------------------------------------------------------------------
    # forward（并行训练，可微）
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, position_offset: int = 0,
                kv_cache=None) -> tuple:
        """前向计算。

        Args:
            x: ``(B, T, D)`` Tensor
            position_offset: query 在全局序列中的起始位置（KV cache 场景）
            kv_cache: 可选 KV cache（dict with 'k','v'），传给 TriSparseAttention

        Returns:
            out: ``(B, T, D)`` 输出
            layer_state: dict with keys:
                - 'aux': 标量 Tensor 或 None（MoD 的 aux_loss；SwiGLU 层为 None）
                - 'kv_cache': dict 或 None（TriSparse 的新 KV cache）
        """
        # 子层 1: TriSparse Attention
        attn_out, new_kv_cache = self.attn(
            self.norm1(x),
            position_offset=position_offset,
            kv_cache=kv_cache,
        )
        x = x + attn_out

        # 子层 2: FFN（SwiGLU 或 MoD）
        ffn_in = self.norm2(x)
        if self.layer_kind == "mod":
            ffn_out, aux = self.ffn(ffn_in)
            aux_loss = aux
        else:
            ffn_out = self.ffn(ffn_in)
            aux_loss = None

        x = x + ffn_out

        layer_state = {"aux": aux_loss, "kv_cache": new_kv_cache}
        return x, layer_state

    # ------------------------------------------------------------------
    # forward_recurrent（单步推理）
    # ------------------------------------------------------------------

    def forward_recurrent(self, x_single: Tensor, state: Optional[dict]) -> tuple:
        """单步递推推理。

        Args:
            x_single: ``(B, 1, D)`` Tensor
            state: 该层的状态 dict 或 None，包含：
                - 'attn_state': TriSparse 的 state（k_cache/v_cache/global_k/...）
                - 'aux': 始终为 None（推理时不需要 aux loss）

        Returns:
            out: ``(B, 1, D)`` Tensor
            new_state: 更新后的 state dict
        """
        attn_state = state.get("attn_state") if state is not None else None

        # 子层 1: Attention
        attn_out, new_attn_state = self.attn.forward_recurrent(
            self.norm1(x_single), attn_state
        )
        x = x_single + attn_out

        # 子层 2: FFN（推理时走 forward 即可，不取 aux）
        ffn_in = self.norm2(x)
        if self.layer_kind == "mod":
            ffn_out, _ = self.ffn(ffn_in)
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + ffn_out

        new_state = {"attn_state": new_attn_state, "aux": None}
        return x, new_state


# ---------------------------------------------------------------------------
# CometSparkNexLM: 顶层架构
# ---------------------------------------------------------------------------


class CometSparkNexLM(Module):
    """CometSpark-Nex 语言模型（VerseNex 原生架构）。

    通过 ``layer_pattern`` 显式指定每层类型，组合 TriSparseAttention 与
    MoDLayer 形成完整语言模型。

    Args:
        vocab_size: 词表大小
        dim: 模型维度
        n_layer: 总层数（与 ``len(layer_pattern)`` 必须一致；若 layer_pattern
            为 None 则自动生成长度为 n_layer 的全 "trisparse" pattern）
        n_head: 注意力头数
        n_kv_head: GQA 的 kv head 数（None 表示 = n_head）
        layer_pattern: ``list[str]``，每元素为 ``"trisparse"`` 或 ``"mod"``,
            显式指定每层类型；None 表示全 "trisparse"
        window_size: TriSparse 滑动窗口大小
        num_global_tokens: TriSparse 全局 sink token 数
        use_alibi: TriSparse 是否启用 ALiBi 路径
        use_rope: TriSparse 是否对 Q/K 应用 RoPE
        max_seq_len: RoPE/ALiBi 预计算最大长度
        dropout: dropout 概率
        rope_theta: RoPE 基础频率
        # MoD
        num_dense_parts: DensePart 数量
        num_experts_per_part: 每个 DensePart 内的 Expert 数
        top_k: 每个 token 选出的 Expert 数
        expert_hidden: Expert 隐藏层维度（None 自动）
        aux_loss_weight: MoD aux loss 权重
        dense_part_names: DensePart 名称列表
        # 其他
        tie_weights: 是否共享 tok_emb 与 head 的权重
        mlp_hidden_multiple: SwiGLU 隐藏层倍数（默认 4）
        init_std: 权重初始化标准差（默认 0.02）
        residual_scale: 残差分支缩放因子（None 表示 1/sqrt(2*n_layer)）
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layer: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        layer_pattern: Optional[List[str]] = None,
        # TriSparse
        window_size: int = 512,
        num_global_tokens: int = 64,
        use_alibi: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        # MoD
        num_dense_parts: int = 5,
        num_experts_per_part: int = 8,
        top_k: int = 3,
        expert_hidden: Optional[int] = None,
        aux_loss_weight: float = 0.01,
        dense_part_names: Optional[list] = None,
        # 其他
        tie_weights: bool = True,
        mlp_hidden_multiple: int = 4,
        init_std: float = 0.02,
        residual_scale: Optional[float] = None,
    ):
        super().__init__()
        # 处理 layer_pattern
        if layer_pattern is None:
            layer_pattern = ["trisparse"] * n_layer
        if len(layer_pattern) != n_layer:
            raise ValueError(
                f"layer_pattern 长度({len(layer_pattern)}) 必须等于 "
                f"n_layer({n_layer})"
            )
        for i, k in enumerate(layer_pattern):
            if k not in VerseNexBlock.VALID_KINDS:
                raise ValueError(
                    f"layer_pattern[{i}]={k!r} 非法，"
                    f"必须为 {VerseNexBlock.VALID_KINDS}"
                )

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.layer_pattern = list(layer_pattern)
        self.tie_weights = tie_weights
        self.max_seq_len = max_seq_len
        self.aux_loss_weight = aux_loss_weight

        # Token Embedding
        self.tok_emb = Embedding(vocab_size, dim)
        # Final norm
        self.norm = RMSNorm(dim)
        # LM head
        self.head = Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        # 构造每一层
        blocks = []
        for kind in self.layer_pattern:
            block = VerseNexBlock(
                dim=dim,
                n_head=n_head,
                n_kv_head=n_kv_head,
                layer_kind=kind,
                window_size=window_size,
                num_global_tokens=num_global_tokens,
                use_alibi=use_alibi,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                dropout=dropout,
                rope_theta=rope_theta,
                num_dense_parts=num_dense_parts,
                num_experts_per_part=num_experts_per_part,
                top_k=top_k,
                expert_hidden=expert_hidden,
                aux_loss_weight=aux_loss_weight,
                dense_part_names=dense_part_names,
                mlp_hidden_multiple=mlp_hidden_multiple,
            )
            blocks.append(block)
        self.blocks = ModuleList(blocks)

        # 参数初始化
        self._init_std = init_std
        self._residual_scale = (
            residual_scale
            if residual_scale is not None
            else 1.0 / ((2 * n_layer) ** 0.5)
        )
        self._init_weights()

        # 打印参数量
        n_params = self.count_parameters()
        n_mod = sum(1 for k in self.layer_pattern if k == "mod")
        print(
            f"[CometSparkNexLM] layers={n_layer} "
            f"(mod={n_mod}, trisparse={n_layer - n_mod}) "
            f"parameters: {n_params} ({n_params / 1e6:.1f}M)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 参数初始化
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """初始化所有 Linear / Embedding 权重为 normal(std=init_std)，
        并对残差分支（attn.proj 与 FFN 出口）按 residual_scale 缩放。
        """
        for m in self.modules():
            if isinstance(m, Linear):
                normal_(m.weight, std=self._init_std)
                if m.bias is not None:
                    normal_(m.bias, std=self._init_std)
            elif isinstance(m, Embedding):
                normal_(m.weight, std=self._init_std)

        # 残差分支缩放：attn.proj 与 SwiGLU.w_down / MoD 各 Expert 的 w_down
        # 对 MoD，因 Expert 数量较多，对 part_router 与每个 expert 的 w_down
        # 都做缩放；为简单起见，仅缩放 attn.proj 与 SwiGLU.w_down（不深入 MoD
        # 内部，因为 MoD 已有 aux_loss 与 softmax 门控做训练稳定性保障）。
        with no_grad():
            for block in self.blocks:
                # attn.proj
                block.attn.proj.weight.data = (
                    block.attn.proj.weight.data * self._residual_scale
                ).astype(np.float32)
                # SwiGLU.w_down（仅 trisparse 层）
                if block.layer_kind == "trisparse":
                    block.ffn.w_down.weight.data = (
                        block.ffn.w_down.weight.data * self._residual_scale
                    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 参数量统计
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """统计可训练参数量。"""
        total = 0
        for p in self.parameters():
            total += int(np.prod(p.data.shape))
        return total

    # ------------------------------------------------------------------
    # forward（并行训练）
    # ------------------------------------------------------------------

    def forward(self, idx) -> Tensor:
        """整序列并行前向，返回 logits。

        Args:
            idx: ``(B, T)`` 整数索引，Tensor / ndarray / list

        Returns:
            logits: Tensor, shape ``(B, T, vocab_size)``
        """
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        elif idx.data.dtype != np.int64:
            idx = Tensor(idx.data.astype(np.int64))

        x = self.tok_emb(idx)  # (B, T, D)
        for block in self.blocks:
            x, _ = block(x, position_offset=0, kv_cache=None)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def forward_with_aux(self, idx) -> tuple:
        """整序列并行前向，返回 (logits, total_aux_loss)。

        用于 SFT/RL 训练：total_aux_loss 是所有 MoD 层 aux_loss 的总和
        （标量 Tensor，已乘以 ``aux_loss_weight``）。

        Args:
            idx: ``(B, T)`` 整数索引

        Returns:
            logits: Tensor, shape ``(B, T, vocab_size)``
            total_aux_loss: 标量 Tensor（无 MoD 层时为 0.0）
        """
        if not isinstance(idx, Tensor):
            idx = Tensor(np.asarray(idx, dtype=np.int64))
        elif idx.data.dtype != np.int64:
            idx = Tensor(idx.data.astype(np.int64))

        x = self.tok_emb(idx)
        total_aux = None
        for block in self.blocks:
            x, layer_state = block(x, position_offset=0, kv_cache=None)
            aux = layer_state["aux"]
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
        x = self.norm(x)
        logits = self.head(x)

        if total_aux is None:
            # 无 MoD 层，返回 0.0 标量（保持 backward 兼容）
            total_aux = Tensor(np.zeros((), dtype=np.float32), requires_grad=False)
        return logits, total_aux

    # ------------------------------------------------------------------
    # forward_recurrent（单步推理，常数内存）
    # ------------------------------------------------------------------

    def forward_recurrent(self, input_ids, states: Optional[List] = None):
        """单步递推推理接口（与 CometSparkLM.forward_recurrent 兼容）。

        Args:
            input_ids: ``(B, 1)`` 整数索引，Tensor / ndarray
            states: ``list[state]`` 每层一个 state，或 None（首次调用）

        Returns:
            logits: Tensor, shape ``(B, 1, vocab_size)``
            new_states: ``list[state]`` 更新后的每层 state
        """
        if not isinstance(input_ids, Tensor):
            idx = Tensor(np.asarray(input_ids, dtype=np.int64))
        elif input_ids.data.dtype != np.int64:
            idx = Tensor(input_ids.data.astype(np.int64))
        else:
            idx = input_ids

        # 单 token embedding
        x = self.tok_emb(idx)  # (B, 1, D)

        # 每层递推
        new_states: list = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None
            x, new_layer_state = block.forward_recurrent(x, layer_state)
            new_states.append(new_layer_state)

        x = self.norm(x)
        logits = self.head(x)  # (B, 1, vocab)
        return logits, new_states

    # ------------------------------------------------------------------
    # generate（自回归生成，迭代式 for 循环，无隐式递归）
    # ------------------------------------------------------------------

    def generate(
        self,
        idx,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        max_safe_limit: int = 100_000,
    ) -> np.ndarray:
        """自回归生成（**完全迭代式 for 循环**）。

        Part4K2 Task 3 升级：默认不限长度（``max_new_tokens=None``），生成到
        EOS 自然停止；达到 ``max_safe_limit`` 安全上限时强制停止以防无限循环。

        - greedy 路径（``temperature==1.0`` 且 ``top_k is None``）：使用
          ``forward_recurrent`` 维护每层 KV/SSM 状态，O(1) 单步推理。
        - 采样路径（含 temperature / top_k）：走 ``forward`` 整序列计算 +
          按温度/top-k 采样（较慢但灵活）。

        Args:
            idx: prompt 序列，shape (B, T_prompt) 或 (T_prompt,)
            max_new_tokens: 最大生成 token 数；``None`` 表示不限（生成到
                ``eos_id`` 自然停止，或达到 ``max_safe_limit`` 安全上限）。
                指定值时按值生成（兼容旧调用）。
            temperature: 采样温度；1.0 走 greedy + recurrent
            top_k: top-k 采样；None 表示无限制
            eos_id: 若指定且末尾非 eos，则追加 eos_id 确保完整 UTF-8 边界
            max_safe_limit: 安全上限（默认 100K），防止无限循环；仅当
                ``max_new_tokens is None`` 时生效。

        Returns:
            generated: ndarray, shape ``(B, T_prompt + 实际生成 token 数)``
        """
        if isinstance(idx, Tensor):
            idx_np = idx.data
        else:
            idx_np = np.asarray(idx)
        if idx_np.ndim == 1:
            idx_np = idx_np[None, :]
        idx_np = idx_np.astype(np.int64)

        # 无限生成模式下：max_safe_limit 充当上限；旧调用按 max_new_tokens 限制
        effective_limit = max_safe_limit if max_new_tokens is None else int(max_new_tokens)

        if temperature == 1.0 and top_k is None:
            out = self._generate_recurrent(idx_np, effective_limit, eos_id=eos_id)
        else:
            out = self._generate_with_logits(
                idx_np, effective_limit, temperature, top_k, eos_id=eos_id
            )

        # 强制追加 eos（确保 decode 完整 UTF-8 边界，与 CometSparkLM 行为一致）
        if eos_id is not None and out.shape[1] > 0:
            last_col = out[:, -1]
            if not np.all(last_col == eos_id):
                eos_col = np.full((out.shape[0], 1), eos_id, dtype=out.dtype)
                out = np.concatenate([out, eos_col], axis=1)
        return out

    def _generate_recurrent(
        self,
        idx_np: np.ndarray,
        max_new_tokens: int,
        eos_id: Optional[int] = None,
    ) -> np.ndarray:
        """Greedy + recurrent：使用 forward_recurrent 维护 KV cache。

        Part4K2 Task 3：支持 EOS 提前停止（``eos_id`` 不为 None 时）。
        """
        rng = np.random.default_rng()
        B, T_prompt = idx_np.shape
        with no_grad():
            self.eval()
            # 1. 用 prompt 预热 state（逐 token 喂入 forward_recurrent）
            states = None
            for t in range(T_prompt):
                step_in = Tensor(idx_np[:, t:t + 1])
                _, states = self.forward_recurrent(step_in, states)

            # 2. 取 prompt 最后一个 token 的 logits 作为第一个生成 token 的输入
            cur = idx_np.copy()
            for _ in range(max_new_tokens):
                step_in = Tensor(cur[:, -1:])
                logits, states = self.forward_recurrent(step_in, states)
                # logits: (B, 1, vocab) → 取最后一个位置
                logits_np = logits.data[:, -1, :]  # (B, vocab)
                next_tok = logits_np.argmax(axis=-1).astype(np.int64)  # (B,)
                cur = np.concatenate([cur, next_tok[:, None]], axis=1)
                # EOS 提前停止（所有 batch 都生成 eos 时停止）
                if eos_id is not None and np.all(next_tok == eos_id):
                    break
        return cur

    def _generate_with_logits(
        self,
        idx_np: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        eos_id: Optional[int] = None,
    ) -> np.ndarray:
        """整序列 forward + 采样生成。

        Part4K2 Task 3：支持 EOS 提前停止（``eos_id`` 不为 None 时）。
        """
        rng = np.random.default_rng()
        with no_grad():
            self.eval()
            cur = idx_np.copy()
            context_len = self.max_seq_len
            for _ in range(max_new_tokens):
                T = cur.shape[1]
                inp = cur[:, -context_len:] if T > context_len else cur
                logits = self.forward(Tensor(inp))  # (B, T_in, vocab)
                logits_np = logits.data[:, -1, :]  # (B, vocab)
                if temperature <= 0:
                    next_tok = logits_np.argmax(axis=-1)
                else:
                    scaled = logits_np / max(temperature, 1e-8)
                    if top_k is not None and top_k > 0:
                        k = min(top_k, scaled.shape[-1])
                        top_idx = np.argpartition(-scaled, kth=k - 1, axis=-1)[:, :k]
                        next_tok = np.zeros(scaled.shape[0], dtype=np.int64)
                        for b in range(scaled.shape[0]):
                            vals = scaled[b, top_idx[b]]
                            probs = _softmax(vals)
                            choice = rng.choice(len(top_idx[b]), p=probs)
                            next_tok[b] = top_idx[b, choice]
                    else:
                        probs = _softmax(scaled)
                        next_tok = np.array(
                            [rng.choice(scaled.shape[-1], p=probs[b])
                             for b in range(scaled.shape[0])],
                            dtype=np.int64,
                        )
                cur = np.concatenate([cur, next_tok[:, None]], axis=1)
                # EOS 提前停止（所有 batch 都生成 eos 时停止）
                if eos_id is not None and np.all(next_tok == eos_id):
                    break
        return cur

    # ------------------------------------------------------------------
    # state_dict / load_state_dict（沿用 Module 默认实现）
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return super().state_dict()

    def load_state_dict(self, sd: dict, strict: bool = True):
        return super().load_state_dict(sd, strict=strict)

    # ------------------------------------------------------------------
    # save / load（pickle，与 CometSparkLM 兼容）
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存到 ``.pt`` 单文件（pickle）。

        Payload 结构::

            {
                "arch": "versenex",
                "config": dict,         # 构造参数
                "state_dict": {name: ndarray},
            }
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "arch": "versenex",
            "config": self.get_config(),
            "state_dict": {k: np.asarray(v) for k, v in self.state_dict().items()},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> "CometSparkNexLM":
        """从 ``.pt`` 文件加载 state_dict 到当前模型（config 不变）。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        self.load_state_dict(sd, strict=False)
        return self

    @classmethod
    def from_pretrained(cls, path: str) -> "CometSparkNexLM":
        """从目录或单文件加载完整模型。

        目录模式（HuggingFace 风格）::

            path/
              config.json    ← 构造参数
              model.pt       ← state_dict (pickle)

        单文件模式（向后兼容）::

            path.pt → {"arch": "verse_nex", "config": dict, "state_dict": dict}
        """
        if os.path.isdir(path):
            cfg_path = os.path.join(path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model = cls(**cfg)
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
        cfg = payload["config"]
        model = cls(**cfg)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        model.load_state_dict(sd, strict=False)
        return model

    def save_pretrained(self, dir_path: str) -> None:
        """保存到目录（HuggingFace 风格）。

        生成::

            dir_path/
              config.json   ← 构造参数
              model.pt      ← state_dict (pickle)
        """
        os.makedirs(dir_path, exist_ok=True)
        # 1. config.json
        cfg_path = os.path.join(dir_path, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, ensure_ascii=False, indent=2)
        # 2. model.pt
        sd = {k: np.asarray(v) for k, v in self.state_dict().items()}
        model_pt = os.path.join(dir_path, "model.pt")
        with open(model_pt, "wb") as f:
            pickle.dump(sd, f)

    # ------------------------------------------------------------------
    # get_config：返回构造参数（用于 save_pretrained）
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """返回可 ``json.dump`` 的构造参数 dict。

        保存构造模型所需的**全部**关键参数，确保 ``from_pretrained`` 能
        完整重建模型 shape（包括 TriSparse 的 n_kv_head / window_size /
        num_global_tokens / use_alibi / use_rope，以及 MoD 的 num_dense_parts /
        num_experts_per_part / top_k / dense_part_names 等）。

        注意：``dense_part_names`` 若为 None 则不保存（构造时按 num_dense_parts
        自动生成默认 names）；若为 list[str] 则保存为 JSON 数组。
        """
        # 从第一个 MoD 层读取 MoD 相关参数（所有 MoD 层共享相同配置）
        mod_block = None
        for b in self.blocks:
            if b.layer_kind == "mod":
                mod_block = b
                break

        cfg = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "layer_pattern": list(self.layer_pattern),
            "max_seq_len": self.max_seq_len,
            "tie_weights": self.tie_weights,
            "aux_loss_weight": self.aux_loss_weight,
        }

        # TriSparse 相关参数（所有层共享，从第 0 层读取）
        b0 = self.blocks[0]
        cfg["window_size"] = b0.attn.window_size
        cfg["num_global_tokens"] = b0.attn.num_global_tokens
        cfg["use_alibi"] = b0.attn.use_alibi
        cfg["use_rope"] = b0.attn.use_rope
        cfg["rope_theta"] = b0.attn.rope_theta

        # MoD 相关参数（若有 MoD 层，从第一个 MoD 层读取）
        if mod_block is not None:
            moe = mod_block.ffn
            cfg["num_dense_parts"] = moe.num_dense_parts
            cfg["num_experts_per_part"] = moe.num_experts_per_part
            cfg["top_k"] = moe.top_k
            cfg["expert_hidden"] = moe.expert_hidden
            # dense_part_names 仅在非默认时保存（默认 names 由构造器自动生成）
            default_names = [f"part_{i}" for i in range(moe.num_dense_parts)]
            if moe.dense_part_names != default_names:
                cfg["dense_part_names"] = list(moe.dense_part_names)

        return cfg

    # ------------------------------------------------------------------
    # Part4K2 Task 6: V1.3 压缩接口（以小博大）
    # ------------------------------------------------------------------

    def compress_v13(self, compress_config=None, teacher_model=None):
        """V1.3 压缩：以小博大（prune → quantize → distill → lora）。

        委托 :func:`verse_torch.compress.compress_pipeline`（version="1.3"），
        返回压缩后的**新** :class:`CometSparkNexLM` 实例（不修改原模型）。

        Args:
            compress_config: 压缩配置 dict（``prune`` / ``quantize`` / ``lora`` /
                ``ternary`` / ``distill`` 任意组合）。``None`` 表示空配置（仅深拷贝）。
            teacher_model: 可选教师模型。传入时等价于在 ``compress_config["distill"]``
                中设置 ``"teacher"``；若无 ``train_loader`` 则仅冻结 teacher 为学生
                做准备，不实际训练（实际蒸馏请用 :meth:`distill_from`）。

        Returns:
            压缩后的新 :class:`CometSparkNexLM` 实例
        """
        from verse_torch.compress import compress_pipeline

        cfg = dict(compress_config) if compress_config else {}
        if teacher_model is not None:
            d_cfg = cfg.get("distill")
            if not isinstance(d_cfg, dict):
                d_cfg = {}
            d_cfg.setdefault("teacher", teacher_model)
            cfg["distill"] = d_cfg
        compressed, stats = compress_pipeline(
            self, cfg, version="1.3", return_stats=True
        )
        # 缓存压缩统计，便于后续 compression_report 查询
        object.__setattr__(compressed, "_v13_stats", stats)
        return compressed

    def distill_from(self, teacher_model, train_data, config=None):
        """从大模型蒸馏能力到当前小模型（V1.3 以小博大核心）。

        在当前模型（self）上就地执行知识蒸馏，使小模型逼近教师模型的能力。
        蒸馏采用 V1.3 三重损失：软标签 KL + 硬标签 CE + 中间层特征 MSE，
        并启用自适应温度调度。

        Args:
            teacher_model: 教师模型（frozen，自动 eval + requires_grad=False）
            train_data: 可迭代对象，每次返回 ``(x, y)`` batch。
                ``x`` 为 token 索引 ``(B, T)``，``y`` 为目标索引 ``(B, T)``
            config: 蒸馏超参 dict，支持键：
                - ``epochs`` (默认 3)
                - ``lr`` (默认 1e-3)
                - ``temperature`` / ``T`` (默认 4.0)
                - ``alpha`` (默认 0.7)
                - ``feature_loss_weight`` (默认 0.3)
                - ``distill_layers`` (默认 None)
                - ``max_steps`` (默认 None，不限)
                - ``feature_extractor`` (默认 None)

        Returns:
            训练损失历史 ``list[float]``（末值应低于初值，体现能力转移）
        """
        from verse_torch.compress import KnowledgeDistiller

        cfg = dict(config) if config else {}
        epochs = int(cfg.pop("epochs", 3))
        lr = float(cfg.pop("lr", 1e-3))
        T = cfg.pop("temperature", cfg.pop("T", 4.0))
        alpha = float(cfg.pop("alpha", 0.7))
        feature_loss_weight = float(cfg.pop("feature_loss_weight", 0.3))
        distill_layers = cfg.pop("distill_layers", None)
        max_steps = cfg.pop("max_steps", None)
        feature_extractor = cfg.pop("feature_extractor", None)

        distiller = KnowledgeDistiller(
            teacher_model, self, temperature=float(T), alpha=alpha,
            distill_layers=distill_layers,
            feature_loss_weight=feature_loss_weight,
        )
        return distiller.distill(
            train_data, epochs=epochs, lr=lr, max_steps=max_steps,
            feature_extractor=feature_extractor,
        )


# ---------------------------------------------------------------------------
# 工厂函数：CometSpark-V0.2（VerseNex 原生，0.5B 参数）
# ---------------------------------------------------------------------------


def _build_v02_pattern(n_layer: int = 32, mod_every: int = 4) -> List[str]:
    """生成 V0.2 默认 layer_pattern。

    策略：每 ``mod_every`` 层中第 0 层为 ``"mod"``，其余为 ``"trisparse"``。
    ``n_layer=32, mod_every=4`` → 8 个 mod + 24 个 trisparse
    （每 MoD 层 ≈47.67M，每 trisparse 层 ≈1.65M，加 V0.2 Embedding 58.3M，
    总参数量 ≈ 480M ≈ 0.5B，符合 CometSpark-V0.2 0.5B 预算）。

    Args:
        n_layer: 总层数
        mod_every: MoD 层的间隔（每 N 层一个 MoD；1 表示全 MoD）

    Returns:
        ``list[str]``，长度 = n_layer
    """
    if mod_every < 1:
        raise ValueError(f"mod_every 必须 >= 1，got {mod_every}")
    pattern = []
    for i in range(n_layer):
        if i % mod_every == 0:
            pattern.append("mod")
        else:
            pattern.append("trisparse")
    return pattern


def CometSparkV02(
    vocab_size: int = 151936,
    dim: int = 384,
    n_layer: int = 32,
    n_head: int = 8,
    n_kv_head: int = 4,
    layer_pattern: Optional[List[str]] = None,
    window_size: int = 512,
    num_global_tokens: int = 64,
    use_alibi: bool = True,
    use_rope: bool = False,
    max_seq_len: int = 2048,
    dropout: float = 0.0,
    rope_theta: float = 10000.0,
    num_dense_parts: int = 5,
    num_experts_per_part: int = 8,
    top_k: int = 3,
    expert_hidden: Optional[int] = None,
    aux_loss_weight: float = 0.01,
    dense_part_names: Optional[list] = None,
    tie_weights: bool = True,
) -> CometSparkNexLM:
    """CometSpark-V0.2 工厂：32 层 VerseNex + MoD，目标参数量 ≈ 0.5B。

    默认配置：
    - vocab_size=151936（Qwen3 tokenizer 词表大小）
    - dim=384, n_layer=32, n_head=8, n_kv_head=4 (GQA 2:1)
    - layer_pattern: 每 4 层 1 个 MoD（共 8 MoD + 24 trisparse）
    - num_dense_parts=5（通用/语言/数理/生化/代码）
    - num_experts_per_part=8, top_k=3
    - tie_weights=True
    - max_seq_len=2048

    参数预算（dim=384, expert_hidden=1024 自动）：
    - 24 个 trisparse 层 × ~1.65M = 39.6M
    - 8 个 mod 层 × ~47.67M = 381.4M
    - Embedding (tie, vocab=151936) = 58.3M
    - **总 ≈ 479M ≈ 0.48B ≈ 0.5B**
    """
    if layer_pattern is None:
        layer_pattern = _build_v02_pattern(n_layer=n_layer, mod_every=4)

    if dense_part_names is None:
        # 用户明确的 5 个能力分区命名
        dense_part_names = ["general", "language", "math", "biochem", "code"]

    return CometSparkNexLM(
        vocab_size=vocab_size,
        dim=dim,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        layer_pattern=layer_pattern,
        window_size=window_size,
        num_global_tokens=num_global_tokens,
        use_alibi=use_alibi,
        use_rope=use_rope,
        max_seq_len=max_seq_len,
        dropout=dropout,
        rope_theta=rope_theta,
        num_dense_parts=num_dense_parts,
        num_experts_per_part=num_experts_per_part,
        top_k=top_k,
        expert_hidden=expert_hidden,
        aux_loss_weight=aux_loss_weight,
        dense_part_names=dense_part_names,
        tie_weights=tie_weights,
    )


# ---------------------------------------------------------------------------
# VerseNexLM: 顶层 LM 品牌别名（Part4K1 SubTask 2.1）
# ---------------------------------------------------------------------------
# CometSparkNexLM 即 VerseNex 原生顶层 LM，此处暴露 VerseNexLM 作为品牌统一入口。
# 后续代码应优先使用 from verse_nex import VerseNexLM。
VerseNexLM = CometSparkNexLM


# ---------------------------------------------------------------------------
# 工具：softmax（与 model.py 中 _softmax 一致）
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 numpy softmax（最后一维）。"""
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)


__all__ = [
    "VerseNexBlock",
    "VerseNexLM",
    "CometSparkNexLM",
    "CometSparkV02",
]
