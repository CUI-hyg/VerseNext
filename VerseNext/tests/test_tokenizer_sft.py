"""Tokenizer / chat template / SFT 测试。"""

import json

import pytest

from verse_tokenizer import BPETokenizer, ChatTemplate
from verse_tokenizer.masking import labels_from_spans


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    corpus = [
        "hello world today", "good morning friend", "how can I help you",
        "the weather is nice", "tell me more about it", "I am a helpful assistant",
    ] * 60
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400)
    return tok


def test_bpe_roundtrip(tok):
    for text in ["hello world", "你好 world!", "hello\nworld\t good morning"]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_bpe_vocab_smaller_than_target(tok):
    assert tok.vocab_size <= 400
    assert tok.vocab_size >= 256


def test_special_tokens_roundtrip(tok):
    text = "hello <|im_start|>user<|im_end|> world"
    ids = tok.encode(text)
    assert tok.decode(ids, skip_special_tokens=False) == text
    # 特殊 token 各占一个 id
    assert tok.special_tokens["<|im_start|>"] in ids


def test_offsets_align_with_text(tok):
    text = "hello world good"
    ids, offsets = tok.encode(text, return_offsets=True)
    raw = text.encode("utf-8")
    assert len(ids) == len(offsets)
    for (s, e) in offsets:
        assert 0 <= s <= e <= len(raw)


def test_save_load_pretrained(tok, tmp_path):
    path = tok.save_pretrained(tmp_path / "tok")
    assert (path / "tokenizer.json").exists()
    tok2 = BPETokenizer.from_pretrained(path)
    text = "good morning world"
    assert tok2.encode(text) == tok.encode(text)


def test_chat_template_default_render():
    tpl = ChatTemplate.default()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    text = tpl.render(msgs, add_generation_prompt=True)
    assert text.startswith("<|bos|><|im_start|>system\nS<|im_end|>")
    assert text.endswith("<|im_start|>assistant\n")


def test_chat_template_custom_jinja(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{%- for m in messages -%}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}"
    )
    tpl = ChatTemplate.from_pretrained(tmp_path)
    assert tpl.render([{"role": "user", "content": "hi"}]) == "[user] hi"


def test_encode_chat_masking(tok):
    tpl = ChatTemplate.default()
    msgs = [
        {"role": "user", "content": "good morning"},
        {"role": "assistant", "content": "good morning world"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "how can I help you"},
    ]
    # 只训练最后一轮
    ids, labels = tpl.encode_chat(msgs, tok, train_on_last_turns=1)
    trained = tok.decode([l for l in labels if l != -100], skip_special_tokens=False)
    assert "how can I help you" in trained
    assert "good morning world" not in trained
    # 训练最近两轮
    _, labels2 = tpl.encode_chat(msgs, tok, train_on_last_turns=2)
    trained2 = tok.decode([l for l in labels2 if l != -100], skip_special_tokens=False)
    assert "good morning world" in trained2
    assert "good morning<|im_end|>" not in trained2 or "morning world" in trained2
    # user/system 内容永远不参与训练
    assert "ok" not in trained2.replace("how can I help you", "")


def test_labels_from_spans():
    ids = [1, 2, 3, 4]
    offsets = [(0, 2), (2, 5), (5, 8), (8, 10)]
    labels = labels_from_spans(ids, offsets, [(5, 8)])
    assert labels == [-100, -100, 3, -100]


def test_sft_dataset_and_lora_training(tmp_path, tok):
    from verse_nn import LoRAConfig, VerseTransformerConfig
    from verse_trainer import SFTConfig, SFTDataset, Trainer, TrainerConfig

    tpl = ChatTemplate.default()
    chats = [{"messages": [
        {"role": "user", "content": "good morning"},
        {"role": "assistant", "content": "good morning world"},
    ]}] * 6
    data_path = tmp_path / "chats.json"
    data_path.write_text(json.dumps(chats))

    tok_path = tok.save_pretrained(tmp_path / "tok")
    tok2 = BPETokenizer.from_pretrained(tok_path)
    ds = SFTDataset.from_json(data_path, tok2, tpl, SFTConfig(batch_size=2, max_seq_len=64))
    x, y = ds.next_batch()
    assert x.shape == y.shape
    assert (y != -100).any()  # 存在有效 target
    assert (y[:, 0] == -100).all()  # 序列首位永远是提示词，不训练

    cfg = TrainerConfig(
        model=VerseTransformerConfig(
            vocab_size=tok2.vocab_size, d_model=64, num_layers=2, num_heads=4, max_seq_len=64
        ),
        lora=LoRAConfig(r=4),
        steps=8, log_every=4, eval_every=0, ckpt_every=0,
    )
    trainer = Trainer(cfg)
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    trainer.train(ds)
    adapter = trainer.save_adapter(tmp_path / "adapter.pt")
    assert adapter.exists()
