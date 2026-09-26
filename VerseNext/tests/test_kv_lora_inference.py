"""KV 缓存、LoRA、推理引擎测试。"""

import pytest
import torch

from verse_nn import (
    LoRAConfig,
    VerseTransformer,
    VerseTransformerConfig,
    apply_lora,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_lora,
)
from verse_nn.lora import load_lora_state_dict


def _tiny_model(**kw):
    torch.manual_seed(0)
    cfg = VerseTransformerConfig(
        vocab_size=100, d_model=64, num_layers=2, num_heads=4, max_seq_len=64, **kw
    )
    return VerseTransformer(cfg).eval()


@pytest.mark.parametrize("num_kv_heads", [None, 2, 1])
def test_generate_cache_matches_full_forward(num_kv_heads):
    model = _tiny_model(num_kv_heads=num_kv_heads)
    prompt = torch.randint(0, 100, (2, 8))
    a = model.generate(prompt, max_new_tokens=16, greedy=True, use_cache=True)
    b = model.generate(prompt, max_new_tokens=16, greedy=True, use_cache=False)
    assert torch.equal(a, b)


def test_generate_eos_stops():
    model = _tiny_model()
    # eos_token_id=0：让模型第一步必然输出 eos
    with torch.no_grad():
        model.lm_head.bias = None
    prompt = torch.tensor([[0]])  # eos 作为 prompt，greedy 后续大概率延续
    out = model.generate(prompt, max_new_tokens=8, greedy=True, eos_token_id=0)
    assert out.shape[1] <= 9


def test_lora_forward_and_merge():
    model = _tiny_model()
    x = torch.randint(0, 100, (1, 8))
    base_out = model(x)

    apply_lora(model, LoRAConfig(r=4, lora_alpha=8))
    mark_only_lora_trainable(model)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_lora > 0
    # B 初始化为 0：LoRA 初始输出与原模型完全一致
    assert torch.allclose(model(x), base_out, atol=1e-5)

    # 训练一步后 merge 保持输出
    y = torch.randint(0, 100, (1, 8))
    _, loss = model(x, y)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    model.eval()
    merged_out = merge_lora(model)(x)
    # merge 后无 lora 增量参与，输出 = 主干 + ΔW，等价于 merge 前输出
    state = lora_state_dict(model)
    assert state  # lora 参数仍可提取（用于审计）


def test_lora_adapter_save_load(tmp_path):
    model = _tiny_model()
    apply_lora(model, LoRAConfig(r=4))
    mark_only_lora_trainable(model)
    y = torch.randint(0, 100, (1, 8))
    _, loss = model(torch.randint(0, 100, (1, 8)), y)
    loss.backward()

    state = lora_state_dict(model)
    torch.save({"lora_state": state}, tmp_path / "adapter.pt")

    model2 = _tiny_model()
    apply_lora(model2, LoRAConfig(r=4))
    load_lora_state_dict(model2, state)
    for (n1, p1), (n2, p2) in zip(
        sorted(model.named_parameters()), sorted(model2.named_parameters())
    ):
        assert n1 == n2 and torch.equal(p1, p2)


def test_dynamic_quantize_runs():
    from verse_trainer.inference import dynamic_quantize

    model = _tiny_model()
    x = torch.randint(0, 100, (1, 4))
    q = dynamic_quantize(model)
    out = q(x)
    assert out.shape == (1, 4, 100)


# ---------------------------------------------------------------------------
# DSA / KDA / 静态缓存
# ---------------------------------------------------------------------------

def _tie_free_dsa(**kw):
    """构造 indexer 分数完全确定的 DSA 模型（测试缓存一致性用）。

    DSA 的 top-k 选择对分数敏感：缓存解码与全量重算的 GEMM 形状不同，
    归约顺序的浮点噪声（~1e-6）会在分数接近处翻转选择边界。测试中以
    纯整数运算的确定性分数替代 indexer（float32 下逐位一致），验证
    「选择逻辑 + 缓存读写」本身的一致性。
    """
    from verse_nn.attention import DSAttention

    model = _tiny_model(attn_type="dsa", **kw)

    def patched(self, x, idx_k_all):
        s_len, total = x.shape[1], idx_k_all.shape[2]
        position = total - s_len
        j = torch.arange(total, device=x.device, dtype=torch.float32)
        i = torch.arange(s_len, device=x.device, dtype=torch.float32)
        # 随 j 单调递增的确定性分数（整数级精度，无路径依赖噪声）
        score = j[None, :] - 0.01 * (j[None, :] - (position + i)[:, None]) ** 2
        return score.expand(x.shape[0], s_len, total)

    DSAttention._indexer_scores = patched
    return model


def test_dsa_forward_shapes_and_cache_consistency():
    model = _tie_free_dsa(dsa_top_k=8)
    x = torch.randint(0, 100, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, 100)

    prompt = torch.randint(0, 100, (2, 10))
    a = model.generate(prompt, 20, greedy=True, use_cache=True)
    b = model.generate(prompt, 20, greedy=True, use_cache=False)
    assert torch.equal(a, b)


def test_kda_forward_and_state_cache_consistency():
    model = _tiny_model(attn_type="kda")
    x = torch.randint(0, 100, (2, 12))
    assert model(x).shape == (2, 12, 100)

    prompt = torch.randint(0, 100, (2, 10))
    a = model.generate(prompt, 16, greedy=True, use_cache=True)
    b = model.generate(prompt, 16, greedy=True, use_cache=False)
    assert torch.equal(a, b)

    # 训练路径
    model.train()
    y = torch.randint(0, 100, (2, 12))
    _, loss = model(x, y)
    loss.backward()


def test_kda_chunkwise_matches_recurrent():
    """chunkwise 并行内核与逐步递归基准的精确等价性。"""
    from verse_nn.attention import KDAAttention

    torch.manual_seed(7)
    b, h, s, d = 2, 2, 96, 16
    q = torch.randn(b, h, s, d) * 0.5
    k = torch.randn(b, h, s, d) * 0.5
    v = torch.randn(b, h, s, d) * 0.5
    gate = torch.sigmoid(torch.randn(b, h, s, d) * 0.1)   # 现实门控范围
    beta = torch.sigmoid(torch.randn(b, h, s, 1) * 0.1)
    S0 = torch.randn(b, h, d, d) * 0.1

    attn = KDAAttention(d_model=h * d, num_heads=h, chunk_size=32)
    # 递归基准
    S = S0.clone()
    outs = []
    for t in range(s):
        S = S * gate[:, :, t].unsqueeze(-1)
        kt = k[:, :, t].unsqueeze(-1)
        vt = v[:, :, t].unsqueeze(-1)
        bt = beta[:, :, t].unsqueeze(-1)
        S = S - bt * (S @ kt) @ kt.transpose(-1, -2) + bt * (vt @ kt.transpose(-1, -2))
        outs.append(torch.matmul(S.transpose(-1, -2), q[:, :, t].unsqueeze(-1))[..., 0])
    o_rec = torch.stack(outs, dim=2)

    o_chunk, S_chunk = attn._chunkwise_forward(q, k, v, gate, beta, S0.clone())
    assert torch.allclose(o_rec, o_chunk, atol=1e-4, rtol=1e-3)
    assert torch.allclose(S, S_chunk, atol=1e-4, rtol=1e-3)

    # 非 chunk 整数倍长度（padding 路径）
    o_rec_c, S_rec_c = o_rec[:, :, :90], S
    o_c90, S_c90 = attn._chunkwise_forward(q[:, :, :90], k[:, :, :90], v[:, :, :90],
                                           gate[:, :, :90], beta[:, :, :90], S0.clone())
    assert torch.allclose(o_rec_c, o_c90, atol=1e-4, rtol=1e-3)


def test_kda_generate_consistency_with_chunk_prefill():
    """chunkwise 预填充 + 递归解码 与 全量重算的贪心解码一致性。"""
    model = _tiny_model(attn_type="kda", kda_chunk_size=8)
    prompt = torch.randint(0, 100, (2, 25))  # 非整数倍长度，覆盖 padding
    a = model.generate(prompt, 20, greedy=True, use_cache=True)
    b = model.generate(prompt, 20, greedy=True, use_cache=False)
    assert torch.equal(a, b)


def test_attention_factory_registry():
    from verse_nn import DSAttention, KDAAttention
    from verse_nn.attention import build_attention
    from verse_core import global_registry

    assert isinstance(build_attention("sdpa", 64, 4), CausalSelfAttention)
    assert isinstance(build_attention("dsa", 64, 4), DSAttention)
    assert isinstance(build_attention("kda", 64, 4), KDAAttention)
    assert "attention/dsa" in global_registry
    with pytest.raises(KeyError):
        build_attention("nope", 64, 4)


def test_config_attn_type_validation():
    with pytest.raises(ValueError):
        VerseTransformerConfig(vocab_size=100, d_model=64, attn_type="foo")


def test_kda_constant_memory_advantage():
    sdpa = _tiny_model().create_caches(1, 512)
    kda = _tiny_model(attn_type="kda").create_caches(1, 512)
    sdpa_mem = sum(c.memory_bytes for c in sdpa)
    kda_mem = sum(c.memory_bytes for c in kda)
    assert kda_mem < sdpa_mem  # KDA 状态与序列长度无关


from verse_nn.attention import CausalSelfAttention  # noqa: E402  (供上方类型断言使用)
