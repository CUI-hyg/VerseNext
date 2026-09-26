"""注意力机制集合：SDPA（稠密）/ DSA（稀疏）/ KDA（线性）。

所有注意力实现统一接口::

    forward(x, layer_cache=None, past_kv=None, use_cache=False)

- ``layer_cache``：静态预分配缓存（推理首选，见 ``kv_cache.py``）；
  KDA 对应 ``RecurrentStateCache``。
- ``past_kv`` / ``use_cache``：动态缓存路径（兼容与测试用）。
- ``CausalSelfAttention``（注册名 ``sdpa``）：标准缩放点积因果注意力，
  QKV 融合单 GEMM，支持 GQA/MQA。
- ``DSAttention``（``dsa``）：DeepSeek 稀疏注意力（简化参考实现）——
  轻量 indexer 打分 + top-k token 选择，注意力只作用于被选 token。
  训练期附带索引器辅助损失（``last_indexer_loss``）：top-k 选择不可导，
  索引器分布对齐主注意力分布（主分支 detach）以获得梯度，由模型侧按
  ``dsa_index_loss_weight`` 加权汇入总损失。
- ``KDAAttention``（``kda``）：Kimi Delta Attention（简化参考实现）——
  门控 delta 规则线性注意力，固定大小递归状态 S，显存与序列长度无关。

均通过 ``attention/<name>`` 注册，由 ``build_attention`` 工厂构建。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from verse_core import global_registry
from verse_nn import kernels
from verse_nn.rope import RotaryEmbedding, apply_rope


def _attention(q, k, v, *, attn_mask=None, is_causal=False, dropout_p=0.0):
    """统一的注意力计算入口：优先自研融合算子，带掩码时回退 SDPA。

    - 无掩码（等长 prefill / 单步解码）：走 ``kernels.flash_attention``，
      在 CUDA/ROCm/NPU 上命中自研 kernel（分块在线 softmax，不物化 S×T 矩阵）；
    - 有掩码（带偏移的因果掩码、DSA 稀疏掩码）：自研 kernel 暂不支持任意掩码，
      回退 ``F.scaled_dot_product_attention``，保证功能与数值一致。
    """
    if attn_mask is None:
        return kernels.flash_attention(
            q, k, v, is_causal=is_causal, dropout_p=dropout_p
        )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


# 因果掩码缓存：训练/预填每次 forward 都会重建 (seq, total) 的 bool 掩码，
# 按 (seq_len, total_len, device) 缓存可避免重复分配。掩码只读使用，安全共享。
_CAUSAL_MASK_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_CAUSAL_MASK_CACHE_MAX = 16


def _causal_mask_with_offset(seq_len: int, total_len: int, device) -> torch.Tensor:
    """带偏移的下三角掩码（解码时 q 只有最近 seq_len 个位置）。"""
    key = (seq_len, total_len, str(device))
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=device)
    mask = torch.tril(mask, diagonal=total_len - seq_len)
    if len(_CAUSAL_MASK_CACHE) >= _CAUSAL_MASK_CACHE_MAX:
        _CAUSAL_MASK_CACHE.clear()
    _CAUSAL_MASK_CACHE[key] = mask
    return mask


# ---------------------------------------------------------------------------
# SDPA 稠密注意力（默认）
# ---------------------------------------------------------------------------

@global_registry.register_attention("sdpa")
class CausalSelfAttention(nn.Module):
    """多头因果自注意力（融合 QKV 投影 + GQA/MQA + 静态/动态缓存）。

    Args:
        d_model: 模型隐藏维度。
        num_heads: Q 头数。
        num_kv_heads: KV 头数（GQA），None 表示标准 MHA。
        dropout: attention dropout（训练期）。
        rope_base: RoPE 频率基数。
    """

    index_head_dim = 0
    """>0 表示需要 indexer 缓冲（DSA 用）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须被 num_heads({num_heads}) 整除")
        num_kv_heads = num_kv_heads or num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads({num_heads}) 必须被 num_kv_heads({num_kv_heads}) 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads

        # 融合 QKV：单次 GEMM 产出 (h_q + 2*h_kv) * head_dim
        self.qkv_proj = nn.Linear(d_model, (num_heads + 2 * num_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.attn_dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def _split_qkv(self, x: torch.Tensor):
        qkv = self.qkv_proj(x)
        bsz, seq_len, _ = qkv.shape
        qkv = qkv.view(bsz, seq_len, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q = qkv[:, :, :self.num_heads].transpose(1, 2)
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_kv_heads].transpose(1, 2)
        v = qkv[:, :, self.num_heads + self.num_kv_heads:].transpose(1, 2)
        return q, k, v

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor):
        """GQA：KV 头广播到 Q 头数。"""
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        layer_cache=None,
        past_kv=None,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = x.shape
        q, k, v = self._split_qkv(x)
        position = layer_cache.pos if layer_cache is not None else (
            past_kv[0].shape[2] if past_kv is not None else 0
        )

        cos, sin = self.rope(x, seq_len, offset=position)
        q, k = kernels.apply_rope(q, k, cos, sin)

        present = None
        if layer_cache is not None:
            layer_cache.write(k, v)
            k_all, v_all = layer_cache.kv(0)
        elif past_kv is not None:
            k_all = torch.cat([past_kv[0], k], dim=2)
            v_all = torch.cat([past_kv[1], v], dim=2)
            present = (k_all, v_all) if use_cache else None
        else:
            k_all, v_all = k, v

        k_exp, v_exp = self._expand_kv(k_all, v_all)
        total_len = k_exp.shape[2]

        attn_mask = None
        is_causal = False
        if seq_len == 1:
            # 单步解码：单个 query 对全部缓存可见，因果约束平凡满足，免建掩码
            pass
        elif total_len != seq_len:
            attn_mask = _causal_mask_with_offset(seq_len, total_len, x.device)
        else:
            # 等长（prefill / 无缓存）：交给 SDPA 内建因果路径（最快）
            is_causal = True

        out = _attention(
            q, k_exp, v_exp,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.o_proj(out)
        return (out, present) if use_cache else out


# ---------------------------------------------------------------------------
# DSA：DeepSeek 稀疏注意力（简化参考实现）
# ---------------------------------------------------------------------------

@global_registry.register_attention("dsa")
class DSAttention(CausalSelfAttention):
    """DeepSeek Sparse Attention（简化版）。

    在标准因果注意力之上叠加「indexer + top-k 稀疏化」：
    1. indexer 用小投影计算 query-key 相关性分数
       ``score(t,i) = mean_h <ReLU(W_q x_t), ReLU(W_k x_i)>``
    2. 每个 query 只保留分数最高的 ``top_k`` 个因果合法位置（当前位置强制保留）
    3. 注意力在稀疏掩码上计算

    效果：解码阶段注意力 FLOPs 从 O(total) 降至 O(top_k)，长上下文收益显著。
    缓存：标准 KV 缓存 + indexer key 缓冲（``StaticKVCache(index_head_dim>0)``）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        index_head_dim: int = 32,
        top_k: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(d_model, num_heads, num_kv_heads, dropout, rope_base)
        self.index_head_dim = index_head_dim
        self.top_k = top_k
        self.index_q_proj = nn.Linear(d_model, self.num_kv_heads * index_head_dim, bias=False)
        self.index_k_proj = nn.Linear(d_model, self.num_kv_heads * index_head_dim, bias=False)
        self.last_indexer_loss: torch.Tensor | None = None
        """训练期索引器辅助损失（每个 forward 覆盖；推理期为 None）。"""

    def _indexer_aux_loss(
        self, q: torch.Tensor, k_exp: torch.Tensor, scores: torch.Tensor,
        seq_len: int, total_len: int,
    ) -> torch.Tensor:
        """索引器辅助损失（仅训练期调用）。

        top-k 选择不可导，索引器若只走选择路径将永远得不到梯度。这里以
        主注意力的因果分布（各 Q 头 softmax 的均值，detach）为软目标，对
        索引器分数分布做交叉熵——梯度 ``(p_idx − p_attn)`` 在两分布均接
        近均匀的初始化阶段依然显著，比 MSE 更适合冷启动。主分支梯度不受
        影响。fp32 计算 softmax 防止 fp16 溢出。
        """
        causal = _causal_mask_with_offset(seq_len, total_len, q.device)
        neg = float("-inf")
        qk = (q @ k_exp.transpose(-2, -1)) * self.head_dim ** -0.5      # (b, h, s, t)
        p_attn = torch.softmax(qk.masked_fill(~causal, neg).float(), dim=-1)
        p_attn = p_attn.mean(dim=1).detach()                            # (b, s, t)
        log_p_idx = torch.log_softmax(scores.masked_fill(~causal, neg).float(), dim=-1)
        # 非因果位置 p_attn=0 且 log_p_idx=-inf，直接相乘得 NaN，先屏蔽为 0
        log_p_idx = log_p_idx.masked_fill(~causal, 0.0)
        return -(p_attn * log_p_idx).sum(-1).mean()

    def _indexer_scores(self, x: torch.Tensor, idx_k_all: torch.Tensor) -> torch.Tensor:
        """返回 (b, seq_len, total_len) 的因果索引分数（未归一化）。"""
        bsz, seq_len, _ = x.shape
        idx_q = F.relu(self.index_q_proj(x))                          # (b, s, h_i, d)
        idx_q = idx_q.view(bsz, seq_len, self.num_kv_heads, self.index_head_dim)
        total_len = idx_k_all.shape[2]
        # (b, h_i, s, d) @ (b, h_i, d, t) -> (b, h_i, s, t)
        scores = torch.einsum("bshd,bhtd->bhst", idx_q, idx_k_all)
        return scores.mean(dim=1) / self.index_head_dim ** 0.5

    def _select_indices(self, scores: torch.Tensor, position: int) -> torch.Tensor:
        """top-k 选择索引 (b, seq_len, k)，保证每行包含自身位置。

        语义：top-k ∪ self；若自身不在 top-k 中则挤占分数最低的选中位。
        选择逻辑在掩码路径与 gather 路径间共享，保证两条路径严格一致。
        """
        bsz, seq_len, total_len = scores.shape
        device = scores.device
        k = min(self.top_k, total_len)
        if seq_len == 1:
            masked = scores  # 单步解码：全部历史因果合法
            self_col = torch.full((bsz, 1), position, dtype=torch.long, device=device)
        else:
            causal = _causal_mask_with_offset(seq_len, total_len, device)
            masked = scores.masked_fill(~causal, float("-inf"))
            self_col = (position + torch.arange(seq_len, device=device)).expand(bsz, seq_len)
        top_idx = masked.topk(k, dim=-1).indices                      # (b, s, k)
        # 自身强制保留：不在 top-k 中则替换分数最低的选中位（topk 末位）
        last = top_idx[:, :, -1]
        in_top = (top_idx == self_col.unsqueeze(-1)).any(-1)          # (b, s)
        last_new = torch.where(in_top, last, self_col)
        return torch.cat([top_idx[:, :, :-1], last_new.unsqueeze(-1)], dim=-1)

    def _sparse_mask(self, scores: torch.Tensor, seq_len: int, position: int) -> torch.Tensor:
        """top-k 稀疏掩码，返回 bool (b, 1, seq_len, total_len)。"""
        top_idx = self._select_indices(scores, position)
        bsz, seq_len, total_len = scores.shape
        mask = torch.zeros(bsz, seq_len, total_len, dtype=torch.bool, device=scores.device)
        mask.scatter_(-1, top_idx, True)
        return mask.unsqueeze(1)

    def forward(self, x, layer_cache=None, past_kv=None, use_cache: bool = False):
        bsz, seq_len, _ = x.shape
        if not self.training:
            self.last_indexer_loss = None  # 避免持有一张过期损失计算图
        q, k, v = self._split_qkv(x)
        idx_k_new = F.relu(self.index_k_proj(x))                        # (b, s, h_i, d)
        idx_k_new = idx_k_new.view(bsz, seq_len, self.num_kv_heads, self.index_head_dim)
        idx_k_new = idx_k_new.transpose(1, 2)                           # (b, h_i, s, d)
        position = layer_cache.pos if layer_cache is not None else (
            past_kv[0].shape[2] if past_kv is not None else 0
        )

        cos, sin = self.rope(x, seq_len, offset=position)
        q, k = kernels.apply_rope(q, k, cos, sin)

        present = None
        if layer_cache is not None:
            layer_cache.write(k, v)
            layer_cache.write_idx_k(idx_k_new)
            k_all, v_all = layer_cache.kv(0)
            idx_k_all = layer_cache.idx_k_view(0)
        elif past_kv is not None:
            k_all = torch.cat([past_kv[0], k], dim=2)
            v_all = torch.cat([past_kv[1], v], dim=2)
            idx_k_all = torch.cat([past_kv[2], idx_k_new], dim=2)
            present = (k_all, v_all, idx_k_all) if use_cache else None
        else:
            k_all, v_all, idx_k_all = k, v, idx_k_new

        total_len = k_all.shape[2]
        k_exp, v_exp = self._expand_kv(k_all, v_all)

        # top-k 覆盖全部历史时退化为稠密因果注意力（省去 indexer 开销）
        if self.top_k >= total_len and seq_len == total_len and not self.training:
            attn_mask, is_causal = None, True
        elif seq_len == 1 and not self.training:
            # 单步解码：gather 被选中的 top-k KV 直接计算注意力——
            # 免去全长度掩码的构建与 SDPA 内部的掩码转换
            scores = self._indexer_scores(x, idx_k_all)
            top_idx = self._select_indices(scores, position)          # (b, 1, k)
            n_sel = top_idx.shape[-1]
            gather_idx = top_idx.reshape(bsz, 1, n_sel, 1).expand(
                bsz, self.num_heads, n_sel, self.head_dim
            )
            k_sel = k_exp.gather(2, gather_idx)
            v_sel = v_exp.gather(2, gather_idx)
            out = _attention(
                q, k_sel, v_sel,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
            out = self.o_proj(out)
            return (out, present) if use_cache else out
        else:
            scores = self._indexer_scores(x, idx_k_all)
            attn_mask = self._sparse_mask(scores, seq_len, position)
            is_causal = False
            if self.training and seq_len > 1:
                self.last_indexer_loss = self._indexer_aux_loss(
                    q, k_exp, scores, seq_len, total_len
                )

        out = _attention(
            q, k_exp, v_exp,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.o_proj(out)
        return (out, present) if use_cache else out


# ---------------------------------------------------------------------------
# KDA：门控 delta 规则线性注意力（简化参考实现）
# ---------------------------------------------------------------------------

@global_registry.register_attention("kda")
class KDAAttention(nn.Module):
    """Kimi Delta Attention（门控 delta 规则线性注意力，chunkwise 并行实现）。

    状态更新（每头）::

        S ← diag(g_t) S                       # 细粒度门控衰减（q 空间行）
        S ← S (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ   # delta 规则（写入并修正记忆）
        o_t = Sᵀ q_t

    - ``g_t = sigmoid(W_g x_t)``：逐通道门控（KDA 的细粒度 gating）
    - ``β_t = sigmoid(W_β x_t)``：写入强度
    - 输出与显存与序列长度无关（状态 S 固定大小 (b, h, d_k, d_v)），
      长上下文推理的显存优势远大于 KV 缓存型注意力。

    **chunkwise 并行**（``chunk_size``，默认 64）：利用「门控行缩放与右侧
    状态转移可交换」的性质做 chunk 内归一化，将 chunk 内递归转化为严格
    上三角线性系统的批量求解（WY 表示）——chunk 内全并行，仅 chunk 间
    顺序传递状态。预填充/训练走 chunkwise；单步解码保持递归（常数开销）。

    数值说明：chunk 内用 exp(cumsum(log g)) 的逆做归一化（fp32、clamp 80），
    门控在 chunk 内极端衰减（累计 < ~1e-30）时旧状态的绝对贡献本就指数
    级衰减，精度损失与其语义一致。
    """

    index_head_dim = 0

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        chunk_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须被 num_heads({num_heads}) 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        # Q/K 同维（状态 S 为 head_dim x head_dim）；Q 独立投影以便单独缩放
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.q_scale = self.head_dim ** -0.5

    def _project(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(bsz, seq_len, h, d).transpose(1, 2) * self.q_scale
        kv = self.kv_proj(x).view(bsz, seq_len, 2, h, d)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x).view(bsz, seq_len, h, d).transpose(1, 2))
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(-1)  # (b,h,s,1)
        return q, k, v, gate, beta

    def forward(
        self,
        x: torch.Tensor,
        layer_cache=None,
        past_kv=None,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = x.shape
        q, k, v, gate, beta = self._project(x)

        # 初始状态：静态缓存 / 动态 past_kv / 从零
        if layer_cache is not None:
            S = layer_cache.state
        elif past_kv is not None:
            S = past_kv
        else:
            S = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim,
                            dtype=x.dtype, device=x.device)

        if seq_len == 1:
            O, S = self._recurrent_step(q, k, v, gate, beta, S)
        else:
            O, S = self._chunkwise_forward(q, k, v, gate, beta, S)

        present = S if use_cache else None
        if layer_cache is not None:
            layer_cache.state = S
            layer_cache.pos += seq_len

        out = self.dropout(O.transpose(1, 2).reshape(bsz, seq_len, -1))
        out = self.o_proj(out)
        return (out, present) if use_cache else out

    # ------------------------------------------------------------- 递归（单步）

    def _recurrent_step(self, q, k, v, gate, beta, S):
        """单步递归（解码用）：o_t = S_tᵀ q_t。"""
        S = S * gate[:, :, 0].unsqueeze(-1)
        kt = k[:, :, 0].unsqueeze(-1)
        vt = v[:, :, 0].unsqueeze(-1)
        bt = beta[:, :, 0].unsqueeze(-1)
        S = S - bt * (S @ kt) @ kt.transpose(-1, -2) + bt * (vt @ kt.transpose(-1, -2))
        o = torch.matmul(S.transpose(-1, -2), q[:, :, 0].unsqueeze(-1))[..., 0]
        return o.unsqueeze(2), S

    # ------------------------------------------------- chunkwise 并行（训练/预填）

    def _chunk_step(self, qc, kc, vc, gc, bc, S):
        """单个 chunk 的并行计算（WY 三角表示，精确等价于逐步递归）。

        归一化：S̃ = Λ^{-1} S（Λ = chunk 内累计门控，作用于 S 的行/q 空间）。
        归一化后递归变为无门控 delta 规则，其 chunk 内展开给出严格上三角
        系统 ``(I + β⊙Û) Mᵀ = ...`` 的批量解。
        """
        bct = bc.transpose(-1, -2)                                   # (b,h,1,C)
        log_lam = torch.cumsum(torch.log(gc.clamp_min(1e-6)), dim=2)  # (b,h,C,d)
        lam = torch.exp(log_lam)
        lam_inv = torch.exp((-log_lam).clamp(max=80.0))              # 防溢出（见 docstring）
        qn = qc * lam                                                # q̃_t = Λ_t ⊙ q_t
        vt = bc * vc * lam_inv                                       # Ṽ_t = β_t v_t ⊘ Λ_t

        # chunk 内 delta 修正：m_j = S̃_0 k_j + Σ_{l<j}(Ṽ_l − β_l m_l)(k_lᵀk_j)
        # → M (I + β⊙Û) = S̃_0 Kᵀ + Ṽᵀ Û（Û = K Kᵀ 严格上三角），批量三角求解
        U = (kc @ kc.transpose(-1, -2)).triu(1)
        Mtri = U * bc + torch.eye(self.chunk_size, device=qc.device, dtype=qc.dtype)
        RHS = S @ kc.transpose(-1, -2) + vt.transpose(-1, -2) @ U
        M = torch.linalg.solve_triangular(Mtri, RHS, upper=True, left=False)

        # 输出：o_t = S̃_0ᵀ q̃_t + Σ_{j≤t} k_j (Ṽ_jᵀ q̃_t − β_j m_jᵀ q̃_t)
        coef = torch.tril(qn @ vt.transpose(-1, -2) - (qn @ M) * bct, diagonal=0)
        O = qn @ S + coef @ kc

        # chunk 末状态（反归一化）：S_out = Λ_C ⊙ [S̃_0 + Σ_j (Ṽ_j − β_j m_j) k_jᵀ]
        W = vt - (M * bct).transpose(-1, -2)
        S_out = (S + W.transpose(-1, -2) @ kc) * lam[:, :, -1:, :].transpose(-1, -2)
        return O, S_out

    def _chunkwise_forward(self, q, k, v, gate, beta, S):
        bsz, num_heads, seq_len, head_dim = q.shape
        C = self.chunk_size
        pad = (-seq_len) % C
        if pad:
            pad_len = (0, 0, 0, pad)
            q = F.pad(q, pad_len)
            k = F.pad(k, pad_len)
            v = F.pad(v, pad_len)
            gate = F.pad(gate, pad_len, value=1.0)   # 填充步无衰减
            beta = F.pad(beta, (0, 0, 0, pad))       # 填充步无写入
        outs = []
        for c0 in range(0, seq_len + pad, C):
            O, S = self._chunk_step(
                q[:, :, c0:c0 + C], k[:, :, c0:c0 + C], v[:, :, c0:c0 + C],
                gate[:, :, c0:c0 + C], beta[:, :, c0:c0 + C], S,
            )
            outs.append(O)
        O = torch.cat(outs, dim=2)[:, :, :seq_len]
        return O, S


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def build_attention(
    attn_type: str,
    d_model: int,
    num_heads: int,
    num_kv_heads: int | None = None,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """按注册名构建注意力模块（``attention/<attn_type>``）。"""
    cls = global_registry.get("attention", attn_type)
    return cls(d_model, num_heads, num_kv_heads, dropout, **kwargs)
