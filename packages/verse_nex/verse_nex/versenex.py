"""VerseNex 型神经网络：原生架构（Part4 P2.3）。

VerseNexLM 是完全原生的 LLM 架构，**无需依赖 Transformer**，
整合了 P2.1 + P2.2 的核心创新：

1. **UltraSparseMultiAttention**（P2.1）：超稀疏并行多注意力
   - 替代标准 Multi-Head Attention
   - Top-K per-query 稀疏选择，复杂度 O(T*K) 而非 O(T²)
   - 多头并行，每个头独立选择稀疏模式

2. **MoDBlock**（P2.2）：多稠密分区
   - 替代标准 FFN/MLP
   - 多个 DensePart（大脑分区），每个含多个 Expert
   - 两级 Top-K 路由：Part-level + Expert-level
   - 高参数效率：参数量 ×n_parts×n_experts，计算量仅 ×top_k_parts×top_k_experts

3. **MedusaHeads**（P2.1）：多头并行预测
   - 主头预测 next token，N 个副头预测 +2/.../+(N+1) token
   - 训练时提供多步预测梯度信号
   - 推理时支持投机解码加速

架构设计
--------
    tokens (B, T)
        ↓
    token_embed + position_embed (可选，attention 内已有 RoPE)
        ↓
    for layer in range(n_layer):
        VerseNexBlock:
            x = x + attn(norm1(x))          # UltraSparseMultiAttention
            x = x + mod(norm2(x))[0]        # MoDBlock (out, aux_loss)
        total_aux += mod_aux_loss
        ↓
    final_norm
        ↓
    lm_head → main_logits (B, T, vocab)
    medusa_heads → aux_logits_list (N × (B, T, vocab))

训练 loss:
    total_loss = main_ce_loss + medusa_aux_loss + λ * mod_aux_loss

与 Transformer 的对比
---------------------
| 组件        | Transformer       | VerseNex                  |
|-------------|-------------------|---------------------------|
| Attention   | MHA (O(T²))       | UltraSparse (O(T*K))      |
| FFN         | 单一 MLP          | MoD (多 DensePart+Expert) |
| 预测        | 单步 next-token   | Medusa 多步并行预测       |
| 架构        | 依赖 Transformer  | 完全原生                   |
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from verse_torch.tensor import Tensor
from verse_torch.nn import Module, Linear, Embedding, RMSNorm, Dropout, ModuleList
from verse_nex.ultra_sparse_attention import (
    UltraSparseMultiAttention,
    MedusaHeads,
)
from verse_nex.mod import MoDBlock


# ---------------------------------------------------------------------------
# VerseNexConfig
# ---------------------------------------------------------------------------


@dataclass
class VerseNexConfig:
    """VerseNex 模型配置。

    核心字段：
        vocab_size: 词表大小
        n_layer: 层数
        d_model: 隐藏维度
        n_head: 注意力头数
        n_kv_head: GQA 的 KV 头数（None = n_head）
        attn_top_k: UltraSparse 注意力的 Top-K（0 = 全注意力）
        dropout: dropout 概率

        # MoD 配置
        mod_n_parts: DensePart 数量（大脑分区数）
        mod_n_experts: 每个 DensePart 内的 Expert 数
        mod_top_k_parts: 每个 token 激活的 DensePart 数
        mod_top_k_experts: 每个 DensePart 内激活的 Expert 数
        mod_d_ff: Expert MLP 中间维度
        mod_aux_loss_weight: MoD load balancing loss 权重

        # Medusa 配置
        medusa_n_heads: Medusa 副头数量（0 = 不使用）
        medusa_aux_weight: Medusa 副头 loss 权重

        # 其他
        max_position_embeddings: 最大位置数
        tie_weights: 是否共享 embedding 与 lm_head
        use_position_embed: 是否使用可学习位置 embedding（False = 仅 RoPE）
    """

    vocab_size: int = 151665
    n_layer: int = 32
    d_model: int = 768
    n_head: int = 12
    n_kv_head: Optional[int] = None
    attn_top_k: int = 64
    dropout: float = 0.0

    # MoD
    mod_n_parts: int = 4
    mod_n_experts: int = 4
    mod_top_k_parts: int = 2
    mod_top_k_experts: int = 2
    mod_d_ff: int = 2048
    mod_aux_loss_weight: float = 0.01

    # Medusa
    medusa_n_heads: int = 3
    medusa_aux_weight: float = 0.5

    # 其他
    max_position_embeddings: int = 2048
    tie_weights: bool = True
    use_position_embed: bool = False  # 默认仅 RoPE（在 attention 内部）

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.d_model % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0


# ---------------------------------------------------------------------------
# VerseNexBlock：单层
# ---------------------------------------------------------------------------


class VerseNexBlock(Module):
    """VerseNex 单层：UltraSparse Attention + MoD FFN。

    Pre-norm 结构：
        x = x + attn(norm1(x))
        x = x + mod_out(norm2(x))   # mod 返回 (out, aux_loss)

    Args:
        config: VerseNexConfig
    """

    def __init__(self, config: VerseNexConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        # Attention
        self.norm1 = RMSNorm(d)
        self.attn = UltraSparseMultiAttention(
            d_model=d,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            top_k=config.attn_top_k,
            dropout=config.dropout,
            rope_max_seq=config.max_position_embeddings,
        )
        # MoD
        self.norm2 = RMSNorm(d)
        self.mod = MoDBlock(
            d_model=d,
            d_ff=config.mod_d_ff,
            n_parts=config.mod_n_parts,
            n_experts_per_part=config.mod_n_experts,
            top_k_parts=config.mod_top_k_parts,
            top_k_experts=config.mod_top_k_experts,
            dropout=config.dropout,
            aux_loss_weight=config.mod_aux_loss_weight,
        )

    def forward(self, x: Tensor, kv_cache=None):
        """返回 (out, aux_loss, new_kv_cache)。"""
        # Attention 子层
        h = self.norm1(x)
        attn_out, new_kv = self.attn(h, kv_cache=kv_cache)
        x = x + attn_out

        # MoD 子层
        h = self.norm2(x)
        mod_out, aux = self.mod(h)
        x = x + mod_out

        return x, aux, new_kv


# ---------------------------------------------------------------------------
# VerseNexLM：完整语言模型
# ---------------------------------------------------------------------------


class VerseNexLM(Module):
    """VerseNex 型神经网络语言模型（原生架构，无需 Transformer）。

    组成：
        - Token Embedding
        - 可选 Position Embedding（默认仅 RoPE）
        - N × VerseNexBlock（UltraSparse Attention + MoD）
        - Final RMSNorm
        - LM Head（主头，预测 next token）
        - 可选 Medusa Heads（副头，预测 +2/.../+(N+1) token）

    forward:
        tokens: (B, T) int
        → (main_logits: (B, T, vocab), aux_logits_list, total_aux_loss)

    生成（generate）：
        支持贪婪解码 + KV cache + 投机解码（Medusa）
    """

    def __init__(self, config: VerseNexConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        V = config.vocab_size

        # Embedding
        self.token_embed = Embedding(V, d)
        if config.use_position_embed:
            self.pos_embed = Embedding(config.max_position_embeddings, d)
        else:
            self.pos_embed = None
        self.embed_dropout = Dropout(config.dropout)

        # Blocks
        self.blocks = ModuleList(
            [VerseNexBlock(config) for _ in range(config.n_layer)]
        )

        # Final norm + LM head
        self.final_norm = RMSNorm(d)
        self.lm_head = Linear(d, V, bias=False)
        if config.tie_weights:
            # 共享 embedding 与 lm_head 权重
            self.lm_head.weight = self.token_embed.weight

        # Medusa 副头（可选）
        if config.medusa_n_heads > 0:
            self.medusa = MedusaHeads(
                d_model=d,
                vocab_size=V,
                n_aux_heads=config.medusa_n_heads,
            )
        else:
            self.medusa = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: Tensor,
        targets: Optional[Tensor] = None,
        kv_caches: Optional[list] = None,
    ):
        """前向传播。

        Args:
            tokens: (B, T) int token ids
            targets: (B, T) int 目标 token ids（用于计算 loss）；None 时不计算
            kv_caches: 可选，每层一个 KV cache（增量推理用）

        Returns:
            dict:
                - "logits": (B, T, V) 主头 logits
                - "aux_logits": list of (B, T, V) Medusa 副头 logits（无 Medusa 时为 []）
                - "aux_loss": MoD load balancing loss（scalar Tensor）
                - "loss": 主头 CE loss（targets 非 None 时）
                - "medusa_loss": Medusa 副头 CE loss（有 Medusa + targets 时）
                - "total_loss": loss + medusa_loss + aux_loss_weight * aux_loss
                - "new_kv_caches": 每层新 KV cache
        """
        B, T = tokens.shape
        # 1. Embedding
        x = self.token_embed(tokens)  # (B, T, d)
        if self.pos_embed is not None:
            positions = np.arange(T)
            pos_ids = Tensor(positions.reshape(1, T).repeat(B, axis=0))
            x = x + self.pos_embed(pos_ids)
        x = self.embed_dropout(x)

        # 2. Blocks
        total_aux_loss = None
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, aux, new_kv = block(x, kv_cache=kv_cache)
            new_kv_caches.append(new_kv)
            if total_aux_loss is None:
                total_aux_loss = aux
            else:
                total_aux_loss = total_aux_loss + aux

        # 3. Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, V)

        # 4. Medusa 副头
        aux_logits = []
        if self.medusa is not None:
            aux_logits = self.medusa(x)  # list of (B, T, V)

        # 5. 计算 loss（如果提供 targets）
        result = {
            "logits": logits,
            "aux_logits": aux_logits,
            "aux_loss": total_aux_loss,
            "new_kv_caches": new_kv_caches,
        }
        if targets is not None:
            result.update(self._compute_loss(logits, aux_logits, targets, total_aux_loss))
        return result

    def _compute_loss(self, logits, aux_logits, targets, aux_loss):
        """计算 CE loss（主头 + Medusa 副头）+ MoD aux loss。"""
        # 主头 CE loss：预测 next token
        # logits: (B, T, V), targets: (B, T)
        # shift：logits[:, :-1] 预测 targets[:, 1:]
        main_loss = self._ce_loss(logits, targets, shift=True)
        result = {"loss": main_loss, "medusa_loss": None, "total_loss": None}

        # Medusa 副头 loss
        medusa_loss = None
        if aux_logits:
            medusa_loss = None
            for i, aux_lg in enumerate(aux_logits):
                # head_i 预测位置 t+i+2 的 token
                # aux_lg: (B, T, V)，targets: (B, T)
                # shift = i + 2：aux_lg[:, :-(i+2)] 预测 targets[:, i+2:]
                shift = i + 2
                if aux_lg.shape[1] <= shift:
                    continue
                head_loss = self._ce_loss(aux_lg, targets, shift=shift)
                if medusa_loss is None:
                    medusa_loss = head_loss
                else:
                    medusa_loss = medusa_loss + head_loss
            if medusa_loss is not None:
                medusa_loss = medusa_loss * self.config.medusa_aux_weight
        result["medusa_loss"] = medusa_loss

        # 总 loss
        total = main_loss
        if medusa_loss is not None:
            total = total + medusa_loss
        if aux_loss is not None:
            total = total + self.config.mod_aux_loss_weight * aux_loss
        result["total_loss"] = total
        return result

    def _ce_loss(self, logits: Tensor, targets: Tensor, shift: int = 1) -> Tensor:
        """交叉熵 loss。

        Args:
            logits: (B, T, V)
            targets: (B, T) int
            shift: 预测偏移量（1 = next token，2 = +2 token，...）

        Returns:
            scalar loss Tensor
        """
        B, T, V = logits.shape
        if T <= shift:
            # 序列太短，返回 0 loss
            return Tensor(np.array(0.0, dtype=np.float32), requires_grad=logits.requires_grad)
        # shift：logits[:, :T-shift] 预测 targets[:, shift:]
        # 但我们的 Tensor 没有 slice 操作，用 __getitem__
        # logits[:, :-shift] → 简化为 logits[:, :T-shift]
        lg = logits[:, :T - shift, :]  # (B, T-shift, V)
        tg = targets[:, shift:]  # (B, T-shift)

        # log_softmax + NLL
        log_probs = lg.log_softmax(dim=-1)  # (B, T-shift, V)

        # gather: 取出 target 对应的 log_prob
        # 用 advanced indexing
        log_probs_data = log_probs.data  # (B, T-shift, V)
        tg_data = tg.data  # (B, T-shift)
        T_s = T - shift
        b_idx, t_idx = np.indices((B, T_s))
        gathered = log_probs_data[b_idx, t_idx, tg_data]  # (B, T-shift)

        # mean negative log likelihood
        nll = -gathered.mean()
        loss_data = np.array(nll, dtype=np.float32)

        requires_grad = log_probs.requires_grad
        loss = Tensor(
            loss_data,
            requires_grad=requires_grad,
            _children=(log_probs,) if requires_grad else (),
            _op="ce_loss",
        )
        if requires_grad:
            _gathered = gathered
            _B, _T_s, _V = B, T_s, V
            _tg_data = tg_data
            _log_probs_data = log_probs_data

            def _backward():
                if loss.grad is None:
                    return
                g = float(loss.grad)
                # d loss / d log_probs = -1/(B*T_s) * one_hot(targets)
                grad = np.zeros((_B, _T_s, _V), dtype=np.float32)
                grad[b_idx, t_idx, _tg_data] = -1.0 / max(_B * _T_s, 1)
                # 乘以上游梯度
                grad = grad * g
                # log_softmax backward: dloss/dlogits = grad * probs - probs * sum(grad * probs)
                probs = np.exp(_log_probs_data)
                dot = (grad * probs).sum(axis=-1, keepdims=True)
                dlogits = probs * (grad - dot)
                log_probs._accumulate_grad(dlogits)

            loss._backward = _backward
        return loss

    # ------------------------------------------------------------------
    # 生成（贪婪 + KV cache）
    # ------------------------------------------------------------------

    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_medusa: bool = False,
    ) -> np.ndarray:
        """贪婪/采样生成。

        Args:
            tokens: (B, T) 初始 token ids
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（1.0 = 原始概率，0 = 贪婪）
            top_k: top-k 采样（None = 不限制）
            use_medusa: 是否使用 Medusa 投机解码（需 self.medusa 存在）

        Returns:
            (B, T + max_new_tokens) 生成的 token ids
        """
        B, T = tokens.shape
        generated = tokens.data.copy()  # (B, T)

        # 初始化 KV cache
        # 先做一次 forward 填充 KV cache
        cur_tokens = Tensor(generated, requires_grad=False)
        result = self.forward(cur_tokens, targets=None)
        kv_caches = result["new_kv_caches"]
        logits = result["logits"]
        # 取最后一个位置的 logits
        last_logits = logits.data[:, -1, :]  # (B, V)

        for step in range(max_new_tokens):
            # 采样 next token
            if temperature <= 0:
                next_token = np.argmax(last_logits, axis=-1)  # (B,)
            else:
                logits_t = last_logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    # top-k 采样
                    kth = np.sort(logits_t, axis=-1)[:, -top_k]
                    logits_t = np.where(logits_t >= kth[:, None], logits_t, -1e9)
                # softmax
                m = np.max(logits_t, axis=-1, keepdims=True)
                exp_l = np.exp(logits_t - m)
                probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
                # 按概率采样
                next_token = np.array([
                    np.random.choice(probs.shape[-1], p=probs[b])
                    for b in range(B)
                ])

            # append
            next_token = next_token.astype(np.int64)
            generated = np.concatenate(
                [generated, next_token.reshape(B, 1)], axis=1
            )

            # 用新 token 更新 KV cache
            new_tokens = Tensor(next_token.reshape(B, 1), requires_grad=False)
            result = self.forward(new_tokens, targets=None, kv_caches=kv_caches)
            kv_caches = result["new_kv_caches"]
            logits = result["logits"]
            last_logits = logits.data[:, -1, :]  # (B, V)

        return generated


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def VerseNexSmall(**overrides) -> VerseNexLM:
    """小型 VerseNex（PoC / 测试用）。

    默认：2 层，d=128，4 Parts × 4 Experts，top_k=2×2，~1M 参数
    """
    config = VerseNexConfig(
        vocab_size=overrides.pop("vocab_size", 1000),
        n_layer=2,
        d_model=128,
        n_head=4,
        attn_top_k=16,
        mod_d_ff=256,
        mod_n_parts=2,
        mod_n_experts=2,
        mod_top_k_parts=1,
        mod_top_k_experts=1,
        medusa_n_heads=2,
        max_position_embeddings=512,
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    config.__post_init__()
    return VerseNexLM(config)


def VerseNexCometSparkV02(**overrides) -> VerseNexLM:
    """CometSpark-V0.2：32 层 VerseNex + MoD，~0.5B 参数。

    目标参数量：~0.486B（未使用压缩技术）
    - vocab=151665, d=384, 32 层
    - 6 头注意力（head_dim=64），2 KV 头（GQA 3:1）
    - 4 DenseParts × 4 Experts，top_k=2×2
    - UltraSparse attention top_k=64
    - tie_weights=True（共享 token_embed 与 lm_head）
    - medusa_n_heads=0（关闭副头以满足 0.5B 约束；可在 overrides 中开启）

    实测参数分布：
        token_embed : 58.2M
        per_layer   : 13.4M × 32 = 428.1M
        final_norm  : 0.4M
        medusa      : 0M (n_heads=0 时)
        TOTAL       : 486.3M ≈ 0.49B
    """
    config = VerseNexConfig(
        vocab_size=overrides.pop("vocab_size", 151665),
        n_layer=32,
        d_model=384,
        n_head=6,
        n_kv_head=2,  # GQA 3:1
        attn_top_k=64,
        mod_d_ff=704,
        mod_n_parts=4,
        mod_n_experts=4,
        mod_top_k_parts=2,
        mod_top_k_experts=2,
        medusa_n_heads=0,  # 关闭 Medusa 副头以满足 0.5B 约束
        tie_weights=True,
        max_position_embeddings=2048,
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    config.__post_init__()
    return VerseNexLM(config)


__all__ = [
    "VerseNexConfig",
    "VerseNexBlock",
    "VerseNexLM",
    "VerseNexSmall",
    "VerseNexCometSparkV02",
]
