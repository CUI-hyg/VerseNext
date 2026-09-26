"""VerseTransformer：decoder-only 因果语言模型（现代 GPT 架构）。

支持：
- 可插拔注意力：sdpa（默认）/ dsa（稀疏）/ kda（线性），经注册工厂构建
- RoPE + GQA + SwiGLU（gate/up 融合 GEMM）+ RMSNorm（pre-norm）
- 梯度检查点（省显存，换约 30% 计算开销）
- 静态预分配 KV Cache（推理零拷贝视图，避免 O(t²) 的逐步 cat）
- ``logits_to_keep``：仅对最后 n 个位置计算 lm_head（解码时省掉
  prefill 阶段对整段上下文的 vocab 投影——该项在长上下文下是显著开销）
- 权重绑定（embedding 与输出 head 共享，省参数）
- torch.compile（由 trainer 侧应用）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from verse_core import global_registry
from verse_core.config import BaseConfig
from verse_nn import kernels
from verse_nn.blocks import RMSNorm, TransformerBlock


@dataclass
class VerseTransformerConfig(BaseConfig):
    """VerseTransformer 的架构配置。

    RSI 的架构变异会直接生成该配置的新实例。
    """

    vocab_size: int = 32000
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int | None = None   # None = 标准 MHA；设为 1 即 MQA
    ffn_hidden_dim: int | None = None  # None = 自动 8/3*d_model 对齐 256
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_weights: bool = True          # embedding 与输出 head 权重绑定
    gradient_checkpointing: bool = False
    attn_type: str = "sdpa"           # sdpa / dsa / kda
    dsa_index_head_dim: int = 32      # DSA indexer 头维度
    dsa_top_k: int = 64               # DSA 每 query 保留的 token 数
    dsa_index_loss_weight: float = 0.1  # DSA 索引器辅助损失权重（0 = 关闭）
    kda_chunk_size: int = 64          # KDA chunkwise 并行的 chunk 长度

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model({self.d_model}) 必须被 num_heads({self.num_heads}) 整除")
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads 必须被 num_kv_heads 整除")
        if self.attn_type not in ("sdpa", "dsa", "kda"):
            raise ValueError(f"未知注意力类型: {self.attn_type}")
        if self.attn_type == "dsa" and self.dsa_top_k < 1:
            raise ValueError("dsa_top_k 必须 >= 1")
        if self.dsa_index_loss_weight < 0:
            raise ValueError("dsa_index_loss_weight 必须 >= 0")


@global_registry.register_model("verse_transformer")
class VerseTransformer(nn.Module):
    """Decoder-only Transformer。通过 ``VerseTransformerConfig`` 配置。"""

    config_cls = VerseTransformerConfig

    def __init__(self, config: VerseTransformerConfig | None = None, **overrides) -> None:
        super().__init__()
        if config is None:
            config = VerseTransformerConfig(**overrides)
        elif overrides:
            config = config.merge(**overrides)
        self.config = config

        attn_kwargs = {}
        if config.attn_type == "dsa":
            attn_kwargs = dict(
                index_head_dim=config.dsa_index_head_dim, top_k=config.dsa_top_k
            )
        elif config.attn_type == "kda":
            attn_kwargs = dict(chunk_size=config.kda_chunk_size)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
                attn_type=config.attn_type,
                **attn_kwargs,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.token_emb.weight.device

    def num_parameters(self, non_embedding: bool = False) -> int:
        """唯一参数数量（权重绑定只计一次）。

        与 ``state_dict`` 求和不同：绑定权重（``lm_head.weight`` 与
        ``token_emb.weight`` 共享）在 state_dict 中出现两次，直接求和会得到
        约 1.5–2× 的「虚高」数值。对外报告一律用本方法，保证预训练/微调/
        加载 checkpoint 各阶段口径一致。
        """
        unique, _ = count_parameters(self)
        if non_embedding and self.config.tie_weights:
            unique -= self.token_emb.weight.numel()
        return unique

    # ------------------------------------------------------------------ cache

    def create_caches(self, batch_size: int, max_len: int | None = None) -> list:
        """为每层创建静态推理缓存（SDPA/DSA: StaticKVCache；KDA: 递归状态）。"""
        from verse_nn.kv_cache import create_caches

        return create_caches(self, batch_size, max_len or self.config.max_seq_len)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        layer_caches: list | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        return_logits: bool = True,
        loss_chunk_size: int = 0,
    ):
        """前向计算。

        Args:
            input_ids: (batch, seq_len) 的 token id。
            targets: (batch, seq_len) 的目标 token id；提供时返回
                ``(logits, loss)``，否则仅返回 ``logits``。
            layer_caches: 每层静态缓存列表（推理首选）。
            past_key_values: 每层动态缓存 ``[(k, v, ...), ...]``（兼容路径）。
            use_cache: True 时返回值中附带更新后的动态缓存。
            logits_to_keep: >0 时只对最后 n 个位置计算 lm_head
                （解码/评估最后一步时大幅减少 vocab 投影开销）。
            return_logits: 训练时置 False 可跳过返回/保留整段 logits（返回 None），
                配合 ``loss_chunk_size`` 显著降低显存/内存峰值。
            loss_chunk_size: >0 时按序列维分块计算 lm_head + CE，
                避免一次性物化 (batch, seq, vocab) 的巨大 logits。
                与整段计算数值等价（按有效 token 数加权平均）。

        Returns:
            logits: (batch, seq_len[, logits_to_keep], vocab_size)（float32）；
                ``return_logits=False`` 且提供 targets 时为 None。
            loss: 可选的交叉熵损失。
            past_key_values: 可选的更新后动态缓存。
        """
        x = self.dropout(self.token_emb(input_ids))

        present: list = []
        use_checkpoint = (
            self.config.gradient_checkpointing and self.training
            and layer_caches is None and past_key_values is None
        )

        def _layer_forward(layer, hidden, cache, past):
            if cache is not None or past is not None or use_cache:
                hidden, pres = layer(hidden, layer_cache=cache, past_kv=past, use_cache=True)
                return hidden, pres
            return layer(hidden), None

        n_layers = len(self.layers)
        layer_caches = layer_caches or [None] * n_layers
        past_key_values = past_key_values or [None] * n_layers

        if use_checkpoint:
            for layer, cache, past in zip(self.layers, layer_caches, past_key_values):
                x = _grad_checkpoint(
                    _layer_forward, layer, x, cache, past, use_reentrant=False
                )[0]
        else:
            for layer, cache, past in zip(self.layers, layer_caches, past_key_values):
                x, pres = _layer_forward(layer, x, cache, past)
                if use_cache:
                    present.append(pres)

        x = self.final_norm(x)
        if logits_to_keep > 0:
            x = x[:, -logits_to_keep:]
            if targets is not None:
                targets = targets[:, -logits_to_keep:]

        need_logits = return_logits or targets is None
        logits = self.lm_head(x).float() if need_logits else None
        outputs = [logits]
        if targets is not None:
            if need_logits:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.reshape(-1).to(logits.device),
                    ignore_index=-100,
                )
            else:
                loss = self._chunked_cross_entropy(x, targets, loss_chunk_size)
            # DSA 索引器辅助损失：top-k 选择不可导，通过「索引器分布对齐主
            # 注意力分布」为 index_q/index_k 提供梯度。注意：该项依赖模块
            # 属性收集，与梯度检查点（先无梯前向、反向重算）不兼容，启用
            # gradient_checkpointing 时辅助损失退化为常数（主损失不受影响）。
            if self.training and self.config.attn_type == "dsa":
                aux = [
                    l.attn.last_indexer_loss for l in self.layers
                    if isinstance(getattr(l.attn, "last_indexer_loss", None), torch.Tensor)
                ]
                if aux and self.config.dsa_index_loss_weight > 0:
                    loss = loss + self.config.dsa_index_loss_weight * torch.stack(aux).mean()
            outputs.append(loss)
        if use_cache:
            outputs.append(present)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _chunked_cross_entropy(
        self, hidden: torch.Tensor, targets: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        """分块 lm_head + CE：按序列维切块，避免整段 (B,S,V) logits 峰值。

        走 :func:`verse_nn.kernels.chunked_cross_entropy` 派发：
        CUDA/ROCm/NPU 命中自研融合 kernel（投影与 logsumexp 融合、不物化
        logits），CPU 用复用缓冲的 ATen 优化路径；结果与整段 mean CE 等价
        （按有效 token 数加权平均）。

        ``lm_head`` 带 bias 时（本仓库默认 ``bias=False``，权重绑定）自研
        算子不支持，退回逐块计算以保证语义正确。
        """
        if self.lm_head.bias is not None:
            return self._chunked_cross_entropy_loop(hidden, targets, chunk_size)
        return kernels.chunked_cross_entropy(
            hidden, self.lm_head.weight, targets, chunk_size=chunk_size
        )

    def _chunked_cross_entropy_loop(
        self, hidden: torch.Tensor, targets: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        """逐块 lm_head + CE（带 bias 或需要逐块自定义时的回退实现）。"""
        seq_len = hidden.size(1)
        chunk = chunk_size if chunk_size > 0 else seq_len
        total = None
        valid = None
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            logits = self.lm_head(hidden[:, start:end]).float()
            tgt = targets[:, start:end].reshape(-1).to(logits.device)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt,
                ignore_index=-100, reduction="sum",
            )
            count = (tgt != -100).sum()
            total = loss if total is None else total + loss
            valid = count if valid is None else valid + count
        if total is None:
            return hidden.new_zeros(())
        return total / valid.clamp(min=1)

    # --------------------------------------------------------------- generate

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        greedy: bool = False,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | set[int] | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归采样生成。

        默认使用静态预分配 KV 缓存：prefill 一次（``logits_to_keep=1`` 只投影
        最后一个位置），之后每步原地写入 1 个 token、零拷贝读取历史。

        为降低「乱码 / 复读」概率，除常规采样外还支持：

        - ``top_p``：核采样（nucleus），与 ``top_k`` 可叠加。
        - ``repetition_penalty``：CTRL 式重复惩罚（>1 抑制已出现 token）。
        - ``no_repeat_ngram``：禁止重复出现的 n-gram（n>=2 时最常用）。
        - ``stop_token_ids``：额外的停止 token（如 ``<|im_end|>``）。
        - 输出 id 一律 clamp 到 ``[0, vocab_size)``，避免越界 id 触发
          embedding 未定义行为（这是产生乱码的常见来源之一）。
        """
        self.eval()
        vocab = self.config.vocab_size
        # 输入 id 先做边界防护：BPE 兜底或外部构造的越界 id 会让 embedding
        # 取到错误行，是乱码的典型来源。（out-of-place，避免改写调用方张量）
        ids = input_ids.clamp(0, vocab - 1)
        prompt = ids[:, -self.config.max_seq_len:]
        bsz = prompt.size(0)

        stop_ids: set[int] = set()
        if eos_token_id is not None:
            stop_ids.add(int(eos_token_id))
        if stop_token_ids:
            stop_ids.update(int(t) for t in stop_token_ids)

        if use_cache:
            caches = self.create_caches(bsz, self.config.max_seq_len)
            logits = self.forward(prompt, layer_caches=caches, logits_to_keep=1)
        else:
            caches = None
            logits = self.forward(prompt)

        finished = torch.zeros(bsz, dtype=torch.bool, device=ids.device)
        for _ in range(max_new_tokens):
            step_logits = logits[:, -1, :].float()
            # 惩罚/禁词只作用于「最近 max_seq_len 个」历史，控制长序列成本
            hist = ids[:, -self.config.max_seq_len:]
            if repetition_penalty and repetition_penalty != 1.0:
                step_logits = _apply_repetition_penalty(
                    step_logits, hist, repetition_penalty
                )
            if no_repeat_ngram and no_repeat_ngram > 0:
                step_logits = _ban_repeat_ngrams(step_logits, hist, no_repeat_ngram)

            if greedy or temperature <= 0:
                next_id = step_logits.argmax(dim=-1, keepdim=True)
            else:
                step_logits = step_logits / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    kth = torch.topk(
                        step_logits, min(top_k, step_logits.size(-1))
                    ).values[:, -1:]
                    step_logits = step_logits.masked_fill(step_logits < kth, float("-inf"))
                if top_p is not None and 0.0 < top_p < 1.0:
                    step_logits = _apply_top_p(step_logits, top_p)
                probs = torch.softmax(step_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                # 全部被屏蔽（如 no_repeat_ngram 禁掉所有候选）时退回均匀分布
                empty = probs.sum(dim=-1, keepdim=True) <= 0
                probs = torch.where(
                    empty, torch.full_like(probs, 1.0 / probs.size(-1)), probs
                )
                next_id = torch.multinomial(probs, num_samples=1)

            next_id = next_id.clamp_(0, vocab - 1)

            # 已完成的序列持续填充 pad（保持 batch 形状一致）
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, vocab - 1),
                next_id,
            )
            ids = torch.cat([ids, next_id], dim=1)
            if stop_ids:
                hit = torch.zeros(bsz, dtype=torch.bool, device=ids.device)
                for tid in stop_ids:
                    hit |= next_id.squeeze(1) == tid
                finished |= hit
                if bool(finished.all()):
                    break

            if use_cache:
                logits = self.forward(next_id, layer_caches=caches, logits_to_keep=1)
            else:
                logits = self.forward(ids[:, -self.config.max_seq_len:], logits_to_keep=1)
        return ids


def _apply_repetition_penalty(
    logits: torch.Tensor, ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL 式重复惩罚：正 logit 除以 penalty，负 logit 乘以 penalty。

    ``ids`` 为 (B, L) 历史 token；同一 token 出现多次只惩罚一次（scatter 去重）。
    """
    gathered = torch.gather(logits, 1, ids)
    gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    return logits.scatter(1, ids, gathered)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """核采样：保留累积概率刚超过 ``top_p`` 的最小 token 集合。"""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # 首个越过阈值的 token 保留（cumulative - prob <= top_p）
    remove = (cumulative - probs) > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return logits.scatter(1, sorted_idx, sorted_logits)


def _ban_repeat_ngrams(
    logits: torch.Tensor, ids: torch.Tensor, n: int
) -> torch.Tensor:
    """禁止重复 n-gram：若末尾 (n-1)-gram 在历史出现过，则屏蔽其后续 token。

    n=1 表示禁止任何已出现的 token；n<=0 时不做处理。
    """
    if n <= 0 or ids.size(1) < n:
        return logits
    for b in range(ids.size(0)):
        row = ids[b].tolist()
        banned: set[int] = set()
        if n == 1:
            banned.update(row)
        else:
            prefix = tuple(row[-(n - 1):])
            for i in range(len(row) - n + 1):
                if tuple(row[i:i + n - 1]) == prefix:
                    banned.add(row[i + n - 1])
        if banned:
            idx = torch.as_tensor(
                sorted(banned), device=logits.device, dtype=torch.long
            )
            logits[b].index_fill_(0, idx, float("-inf"))
    return logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """统计模型参数量，返回 ``(unique, total)``。

    - ``unique``：按底层张量（``data_ptr``）去重后的参数量。权重绑定时
      ``lm_head`` 与 ``token_emb`` 只计一次，是模型的「真实」规模。
    - ``total``：``state_dict`` 逐项求和。绑定权重会被重复计一次，故
      ``total >= unique``；仅用于诊断存储/对齐情况，不作为规模指标。

    该函数同时兼容 ``torch.compile`` 包装前后的模型（对 ``_orig_mod`` 亦可）。
    """
    seen: set[tuple[int, int]] = set()
    unique = 0
    for param in model.parameters():
        key = (param.data_ptr(), param.numel())
        if key in seen:
            continue
        seen.add(key)
        unique += param.numel()
    total = sum(value.numel() for value in model.state_dict().values())
    return unique, total


def build_model(model_name: str, config: dict | BaseConfig | None = None, **overrides):
    """通过注册中心构建模型：RSI 侧使用该入口动态构建变异后的结构。"""
    cls = global_registry.get("model", model_name)
    cfg_cls = getattr(cls, "config_cls", None)
    if cfg_cls is not None and isinstance(config, dict):
        config = cfg_cls.from_dict(config)
    return cls(config, **overrides) if cfg_cls is not None else cls(config, **overrides)
