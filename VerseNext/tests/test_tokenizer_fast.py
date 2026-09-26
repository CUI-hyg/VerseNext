"""分词器热路径与特殊 token 解析测试。"""

import pytest

from verse_tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    corpus = [
        "hello world today", "good morning friend", "how can I help you",
        "the weather is nice", "tell me more about it", "I am a helpful assistant",
        "机器学习 很有趣", "深度学习 模型 训练",
    ] * 40
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=500)
    return tok


def test_encode_batch_matches_single(tok):
    texts = ["hello world", "good morning friend", "机器学习 很有趣", ""]
    batch = tok.encode_batch(texts)
    assert batch == [tok.encode(t) for t in texts]


def test_encode_batch_roundtrip(tok):
    texts = ["hello world today", "深度学习 模型 训练", "how can I help you"]
    for ids, text in zip(tok.encode_batch(texts), texts):
        assert tok.decode(ids) == text


def test_eos_alias_priority(tok):
    # 默认 train() 会注册 <|eos|>，其优先级高于 <|endoftext|>
    assert "<|eos|>" in tok.special_tokens
    assert tok.eos_token_id == tok.special_tokens["<|eos|>"]
    # 只有 <|endoftext|> 时（CometSpark tokenizer 的情形）应解析到它
    tok2 = BPETokenizer()
    tok2.train(["a b c d e f g h"] * 30, vocab_size=300)
    eot = tok2.add_special_token("<|endoftext|>")
    tok2.special_tokens = {"<|endoftext|>": eot}
    tok2._invalidate_special_cache()
    assert tok2.eos_token_id == eot


def test_add_special_token_increments_max_id(tok):
    before = max(tok.vocab.values())
    new_id = tok.add_special_token("<|custom_special|>")
    assert new_id == before + 1
    # 幂等：重复注册返回同一 id
    assert tok.add_special_token("<|custom_special|>") == new_id


def test_special_token_roundtrip(tok):
    tok.add_special_token("<|im_start|>")
    tok.add_special_token("<|im_end|>")
    text = "hi<|im_start|>user<|im_end|>"
    ids = tok.encode(text)
    assert tok.decode(ids, skip_special_tokens=False) == text
    # 跳过特殊 token 时只剩可见文本（"user" 是普通文本，保留）
    assert tok.decode(ids, skip_special_tokens=True) == "hiuser"


def test_encode_without_special_regex_trigger(tok):
    # 不含 "<|" 的文本不应触发特殊 token 注册/切分逻辑
    text = "plain text with no special markers"
    assert tok.decode(tok.encode(text)) == text
