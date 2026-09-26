"""生成采样控制（降乱码 / 复读）测试。"""

import torch

from verse_nn.transformer import (
    VerseTransformer,
    VerseTransformerConfig,
    _apply_repetition_penalty,
    _apply_top_p,
    _ban_repeat_ngrams,
)


def _model(vocab: int = 64) -> VerseTransformer:
    return VerseTransformer(
        VerseTransformerConfig(vocab_size=vocab, d_model=32, num_layers=2,
                               num_heads=4, max_seq_len=32)
    )


def test_repetition_penalty_sign_aware():
    logits = torch.zeros(1, 10)
    logits[0, 3] = 2.0
    logits[0, 5] = -2.0
    ids = torch.tensor([[3, 5, 3]])
    out = _apply_repetition_penalty(logits.clone(), ids, 2.0)
    assert out[0, 3].item() == 1.0     # 正 logit / penalty
    assert out[0, 5].item() == -4.0    # 负 logit * penalty
    assert out[0, 0].item() == 0.0     # 未出现 token 不变


def test_top_p_keeps_minimal_set():
    logits = torch.tensor([[5.0, 1.0, 0.0, -1.0]])
    out = _apply_top_p(logits.clone(), 0.5)
    assert out[0, 0].item() == 5.0
    assert out[0, 1].item() == float("-inf")
    # top_p=1.0 不改变分布（由调用方保证不调用；此处直接验证边界语义）
    assert torch.equal(_apply_top_p(logits.clone(), 1.0), logits)


def test_ban_repeat_ngram():
    logits = torch.zeros(1, 8)
    logits[0, 3] = 9.0
    ids = torch.tensor([[1, 2, 3, 7, 1, 2]])
    out = _ban_repeat_ngrams(logits.clone(), ids, 2)
    assert out[0, 3].item() == float("-inf")   # 1,2 后已出现过 3
    assert out[0, 7].item() == 0.0


def test_ban_repeat_ngram_n1_bans_seen():
    logits = torch.zeros(1, 8)
    ids = torch.tensor([[1, 2, 3]])
    out = _ban_repeat_ngrams(logits.clone(), ids, 1)
    for tid in (1, 2, 3):
        assert out[0, tid].item() == float("-inf")
    assert out[0, 4].item() == 0.0


def test_generate_clamps_input_and_output():
    model = _model()
    bad = torch.tensor([[1, 2, 9999, -5]])
    out = model.generate(bad, max_new_tokens=6, greedy=True)
    assert out.min() >= 0 and out.max() < 64


def test_generate_stop_token_ids():
    model = _model()
    prompt = torch.tensor([[1, 2, 3]])
    full = model.generate(prompt, max_new_tokens=16, greedy=True)
    stop = full[0, 3].item()  # 贪心第一步必产出
    early = model.generate(prompt, max_new_tokens=16, greedy=True, stop_token_ids=[stop])
    assert early.shape[1] == 4  # 命中停止符即停


def test_generate_with_all_filters_no_nan():
    model = _model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    out = model.generate(
        prompt, max_new_tokens=20, temperature=0.7, top_k=10, top_p=0.9,
        repetition_penalty=1.2, no_repeat_ngram=3,
    )
    assert out.shape == (1, 24)
    assert out.min() >= 0 and out.max() < 64


def test_generate_does_not_mutate_input():
    model = _model()
    ids = torch.tensor([[1, 2, 9999]])
    original = ids.clone()
    model.generate(ids, max_new_tokens=3, greedy=True)
    assert torch.equal(ids, original)
