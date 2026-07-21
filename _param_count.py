"""计算 CometSpark-V0.2 参数量，调整到 ~0.5B。"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/packages/verse_nex")
sys.path.insert(0, "/workspace/packages/verse_torch")

from verse_nex.versenex import VerseNexConfig


def count_params(config: VerseNexConfig) -> dict:
    """精确计算 VerseNexLM 参数量。"""
    d = config.d_model
    V = config.vocab_size
    n_layer = config.n_layer
    head_dim = d // config.n_head
    kv_dim = config.n_kv_head * head_dim
    d_ff = config.mod_d_ff

    # Embedding
    token_embed = V * d
    pos_embed = (config.max_position_embeddings * d) if config.use_position_embed else 0

    # Per layer
    # Attention
    wq = d * (config.n_head * head_dim)  # = d * d
    wk = d * kv_dim
    wv = d * kv_dim
    proj = (config.n_head * head_dim) * d  # = d * d
    attn_total = wq + wk + wv + proj

    # MoDBlock
    router = d * config.mod_n_parts
    # Each DensePart: inner_router + n_experts × (w_gate + w_up + w_down)
    expert_params = 3 * d * d_ff  # w_gate + w_up + w_down
    dense_part = d * config.mod_n_experts + config.mod_n_experts * expert_params
    mod_total = router + config.mod_n_parts * dense_part

    # Norms
    norm1 = d  # RMSNorm weight
    norm2 = d
    final_norm = d

    per_layer = attn_total + mod_total + norm1 + norm2
    all_layers = n_layer * per_layer

    # LM head
    lm_head = 0 if config.tie_weights else V * d

    # Medusa
    medusa = 0
    if config.medusa_n_heads > 0:
        # Each head: fc1 (d×d + d bias) + fc2 (d×V)
        medusa_per_head = d * d + d + d * V
        medusa = config.medusa_n_heads * medusa_per_head

    total = token_embed + pos_embed + all_layers + final_norm + lm_head + medusa

    return {
        "token_embed": token_embed,
        "pos_embed": pos_embed,
        "per_layer_attn": attn_total,
        "per_layer_mod": mod_total,
        "per_layer_total": per_layer,
        "all_layers": all_layers,
        "lm_head": lm_head,
        "medusa": medusa,
        "final_norm": final_norm,
        "total": total,
        "total_B": total / 1e9,
    }


# 测试不同配置
configs = [
    ("d=384, d_ff=768, medusa=0", VerseNexConfig(
        vocab_size=151665, n_layer=32, d_model=384, n_head=6, n_kv_head=2,
        mod_d_ff=768, mod_n_parts=4, mod_n_experts=4,
        mod_top_k_parts=2, mod_top_k_experts=2,
        medusa_n_heads=0, tie_weights=True,
    )),
    ("d=384, d_ff=640, medusa=0", VerseNexConfig(
        vocab_size=151665, n_layer=32, d_model=384, n_head=6, n_kv_head=2,
        mod_d_ff=640, mod_n_parts=4, mod_n_experts=4,
        mod_top_k_parts=2, mod_top_k_experts=2,
        medusa_n_heads=0, tie_weights=True,
    )),
    ("d=384, d_ff=704, medusa=0", VerseNexConfig(
        vocab_size=151665, n_layer=32, d_model=384, n_head=6, n_kv_head=2,
        mod_d_ff=704, mod_n_parts=4, mod_n_experts=4,
        mod_top_k_parts=2, mod_top_k_experts=2,
        medusa_n_heads=0, tie_weights=True,
    )),
    ("d=320, d_ff=640, medusa=0", VerseNexConfig(
        vocab_size=151665, n_layer=32, d_model=320, n_head=5, n_kv_head=1,
        mod_d_ff=640, mod_n_parts=4, mod_n_experts=4,
        mod_top_k_parts=2, mod_top_k_experts=2,
        medusa_n_heads=0, tie_weights=True,
    )),
]

for name, cfg in configs:
    params = count_params(cfg)
    print(f"\n{name}:")
    print(f"  token_embed:    {params['token_embed']/1e6:.1f}M")
    print(f"  per_layer_attn: {params['per_layer_attn']/1e6:.1f}M")
    print(f"  per_layer_mod:  {params['per_layer_mod']/1e6:.1f}M")
    print(f"  per_layer:      {params['per_layer_total']/1e6:.1f}M")
    print(f"  all_layers:     {params['all_layers']/1e6:.1f}M")
    print(f"  medusa:         {params['medusa']/1e6:.1f}M")
    print(f"  TOTAL:          {params['total_B']:.3f}B ({params['total']/1e6:.1f}M)")
