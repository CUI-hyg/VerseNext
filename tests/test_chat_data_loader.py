"""Task 1.4: TextDataset 新格式（chat 数组 + prompt-completion + text）单元测试。

覆盖用例：
1. test_load_chat_format: 加载 chat 数组格式
2. test_load_prompt_completion_format: 加载 prompt-completion 格式
3. test_text_format_supported: {"text":"..."} 格式支持（Part4 紧急更新）
   test_single_sample_prompt_only: 单样本只存在 prompt
   test_single_sample_completion_only: 单样本只存在 completion
4. test_loss_mask: loss mask 屏蔽 prompt 部分（y=-100），保留 completion 部分
5. test_mixed_formats: 混合格式加载
6. test_dataset_len: __len__ 返回 n_blocks
7. test_getitem_shapes: __getitem__ 返回 (x, y) 形状均为 (seq_len,)
8. test_collate_fn: collate_fn 堆叠后形状 (B, seq_len)

运行方式：
    cd /workspace && python -m pytest tests/test_chat_data_loader.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 目录能 import verse_infra.verse_tokenizer（ByteTokenizer）
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("verse_infra", "verse_torch"):
    _p = _REPO_ROOT / "packages" / _pkg
    if _p.is_dir():
        sys.path.insert(0, str(_p))

# Part4K1 Task 8.9: 从 spark/src 导入 data_loader（替代 data/demo/src）
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verse_infra.verse_tokenizer import ByteTokenizer  # noqa: E402
from spark.src.data_loader import (  # noqa: E402
    TextDataset,
    collate_fn,
    BatchLoader,
    _detect_format,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, items):
    """把 list 写入 jsonl 文件，每行一个 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _make_tokenizer():
    """构造 ByteTokenizer（vocab_size=259，含 bos/eos/pad/unk）。"""
    return ByteTokenizer()


# ---------------------------------------------------------------------------
# 1. test_load_chat_format
# ---------------------------------------------------------------------------


def test_load_chat_format(tmp_path):
    """加载 chat 数组格式：TextDataset 构造成功且 __len__ > 0。"""
    items = [
        [{"role": "user", "content": "你好"},
         {"role": "assistant", "content": "你好，很高兴见到你。"}],
        [{"role": "user", "content": "再见"},
         {"role": "assistant", "content": "再见，祝你一切顺利。"}],
    ]
    path = tmp_path / "chat.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    ds = TextDataset(tok, str(path), seq_len=16)
    assert len(ds) > 0, "chat 格式加载后 __len__ 应 > 0"
    # 检查 ids 已被填充
    assert ds.ids.shape[0] == len(ds) * 16


# ---------------------------------------------------------------------------
# 2. test_load_prompt_completion_format
# ---------------------------------------------------------------------------


def test_load_prompt_completion_format(tmp_path):
    """加载 prompt-completion 格式：TextDataset 构造成功且 __len__ > 0。"""
    items = [
        {"prompt": "床前明月光，", "completion": "疑是地上霜。"},
        {"prompt": "白日依山尽，", "completion": "黄河入海流。"},
        {"prompt": "1+1=", "completion": "2"},
    ]
    path = tmp_path / "pc.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    ds = TextDataset(tok, str(path), seq_len=16)
    assert len(ds) > 0
    # prompt + completion 都应该被编码进来
    # ids 长度 = n_blocks * seq_len
    assert ds.ids.shape[0] == len(ds) * 16


# ---------------------------------------------------------------------------
# 3. test_text_format_supported
# ---------------------------------------------------------------------------


def test_text_format_supported(tmp_path):
    """{"text":"..."} 格式现重新支持，所有 token 参与 loss。

    Part4 紧急更新：text 格式用于预训练/续训，单样本（只存在 prompt 或
    completion）也允许，存在的字段当作纯文本全部参与 loss。
    """
    items = [
        {"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
    ]
    path = tmp_path / "text.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    ds = TextDataset(tok, str(path), seq_len=16)
    assert len(ds) > 0
    # text 格式所有 token 参与 loss → y 中应有非 -100 的值
    x, y = ds[0]
    assert (y != -100).any(), "text 格式应有 token 参与 loss"


def test_single_sample_prompt_only(tmp_path):
    """单样本：只存在 prompt 时，prompt 当作纯文本全部参与 loss。"""
    items = [{"prompt": "只有 prompt 的样本，应当作为纯文本训练。"}]
    path = tmp_path / "prompt_only.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    ds = TextDataset(tok, str(path), seq_len=16)
    assert len(ds) > 0
    x, y = ds[0]
    assert (y != -100).any(), "prompt_only 应有 token 参与 loss"


def test_single_sample_completion_only(tmp_path):
    """单样本：只存在 completion 时，completion 当作纯文本全部参与 loss。"""
    items = [{"completion": "只有 completion 的样本，应当作为纯文本训练。"}]
    path = tmp_path / "completion_only.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    ds = TextDataset(tok, str(path), seq_len=16)
    assert len(ds) > 0
    x, y = ds[0]
    assert (y != -100).any(), "completion_only 应有 token 参与 loss"


# ---------------------------------------------------------------------------
# 4. test_loss_mask
# ---------------------------------------------------------------------------


def test_loss_mask(tmp_path):
    """loss mask：prompt 部分 y=-100，completion/assistant 部分 y=token_id。

    构造一条 prompt-completion 样本，渲染后：
        <|user|>{prompt}<|assistant|>{completion}<|eos|>
    prompt 部分（含 <|user|>...<|assistant|>）mask=0 → y=-100
    completion + <|eos|> 部分 mask=1 → y=真实 token id

    用较长的 completion 确保 completion 部分落在第一个 block 内，
    以便在同一 block 内同时观察到 mask=0 与 mask=1 的位置。
    """
    prompt = "Q"
    # completion 足够长，使得 prompt+completion 总字节数 > seq_len
    # <|user|>(8) + Q(1) + <|assistant|>(13) = 22 字节 prompt
    # <|eos|>(7) + \n(1) = 8 字节额外
    # 用 16 字节 completion 使 total = 22+16+8 = 46 字节
    completion = "ABCDEFGHabcdefgh"  # 16 字节
    items = [{"prompt": prompt, "completion": completion}]
    path = tmp_path / "mask.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    # seq_len=32：第一个 block [0:32] 含 22 字节 prompt + 10 字节 completion
    # y = [1:33] 含 21 字节 prompt mask=0 + 11 字节 completion mask=1
    seq_len = 32
    ds = TextDataset(tok, str(path), seq_len=seq_len)
    assert len(ds) >= 1

    # 取第一个 block
    x, y = ds[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)

    # 统计 y 中 -100 与非 -100 的数量
    n_ignored = int(np.sum(y == -100))
    n_valid = int(np.sum(y != -100))
    # 必须同时存在 mask=0 与 mask=1 的位置
    assert n_ignored > 0, "prompt 部分应有 y=-100"
    assert n_valid > 0, "completion 部分应有 y=token_id"

    # 验证 completion 部分（非 -100）至少包含 completion 的第一个字符 id
    # completion[0] = 'A' -> byte id 65
    # y 中应该出现 65（completion 首字符作为 next-token target）
    completion_first_byte = ord(completion[0])  # 'A' -> 65
    assert completion_first_byte in set(y.tolist()), (
        f"completion 首字符 byte id {completion_first_byte} 应在 y 中作为 target"
    )


# ---------------------------------------------------------------------------
# 5. test_mixed_formats
# ---------------------------------------------------------------------------


def test_mixed_formats(tmp_path):
    """混合格式加载：chat 数组 + prompt-completion 在同一文件中。"""
    items = [
        [{"role": "user", "content": "你好"},
         {"role": "assistant", "content": "你好。"}],
        {"prompt": "1+1=", "completion": "2"},
        [{"role": "user", "content": "再见"},
         {"role": "assistant", "content": "再见。"}],
        {"prompt": "春眠不觉晓，", "completion": "处处闻啼鸟。"},
    ]
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    ds = TextDataset(tok, str(path), seq_len=16)
    assert len(ds) > 0

    # 任意取一个 block 验证形状
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)


# ---------------------------------------------------------------------------
# 6. test_dataset_len
# ---------------------------------------------------------------------------


def test_dataset_len(tmp_path):
    """__len__ 返回 n_blocks = total_tokens // seq_len。"""
    items = [
        {"prompt": "abc", "completion": "def"},
        {"prompt": "xyz", "completion": "uvw"},
    ]
    path = tmp_path / "len.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    seq_len = 8
    ds = TextDataset(tok, str(path), seq_len=seq_len)

    # 验证 len(ds) == n_blocks，且 ids 长度 = n_blocks * seq_len
    expected_blocks = ds.ids.shape[0] // seq_len
    assert len(ds) == expected_blocks
    assert ds.ids.shape[0] == expected_blocks * seq_len
    assert ds.mask.shape[0] == ds.ids.shape[0]


# ---------------------------------------------------------------------------
# 7. test_getitem_shapes
# ---------------------------------------------------------------------------


def test_getitem_shapes(tmp_path):
    """__getitem__ 返回 (x, y)，形状均为 (seq_len,)。"""
    items = [
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "world"}],
    ]
    path = tmp_path / "shape.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    seq_len = 32
    ds = TextDataset(tok, str(path), seq_len=seq_len)
    assert len(ds) >= 1

    # 测试所有 block 的形状
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (seq_len,), f"block {i}: x.shape={x.shape}"
        assert y.shape == (seq_len,), f"block {i}: y.shape={y.shape}"
        # x dtype int64
        assert x.dtype == np.int64
        assert y.dtype == np.int64

    # 测试越界
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    with pytest.raises(IndexError):
        _ = ds[-1]


# ---------------------------------------------------------------------------
# 8. test_collate_fn
# ---------------------------------------------------------------------------


def test_collate_fn(tmp_path):
    """collate_fn 堆叠 batch：返回 (x_batch, y_batch)，形状 (B, seq_len)。"""
    items = [
        {"prompt": "床前明月光，", "completion": "疑是地上霜。"},
        {"prompt": "白日依山尽，", "completion": "黄河入海流。"},
        {"prompt": "春眠不觉晓，", "completion": "处处闻啼鸟。"},
        {"prompt": "1+1=", "completion": "2"},
    ]
    path = tmp_path / "collate.jsonl"
    _write_jsonl(path, items)

    tok = _make_tokenizer()
    seq_len = 16
    ds = TextDataset(tok, str(path), seq_len=seq_len)

    # 取 4 个样本（若 ds 不足 4，取全部）
    n = min(4, len(ds))
    batch = [ds[i] for i in range(n)]
    x_batch, y_batch = collate_fn(batch)

    assert x_batch.shape == (n, seq_len)
    assert y_batch.shape == (n, seq_len)
    assert x_batch.dtype == np.int64
    assert y_batch.dtype == np.int64

    # 验证 collate_fn 后内容与单样本一致
    x0, y0 = ds[0]
    np.testing.assert_array_equal(x_batch[0], x0)
    np.testing.assert_array_equal(y_batch[0], y0)

    # 也测试一下 BatchLoader 集成
    loader = BatchLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    n_batches = len(loader)
    assert n_batches == (len(ds) + 1) // 2
    for x_b, y_b in loader:
        assert x_b.shape[1] == seq_len
        assert y_b.shape[1] == seq_len
        break  # 仅验证第一个 batch


# ---------------------------------------------------------------------------
# 额外：_detect_format 单元测试（顺带覆盖）
# ---------------------------------------------------------------------------


def test_detect_format():
    """_detect_format 对各种输入返回正确格式名。"""
    assert _detect_format([{"role": "user", "content": "x"}]) == "chat"
    assert _detect_format({"prompt": "x", "completion": "y"}) == "prompt_completion"
    # Part4 紧急更新：text 格式重新支持，单样本（prompt_only/completion_only）允许
    assert _detect_format({"text": "x"}) == "text"
    assert _detect_format({"prompt": "x"}) == "prompt_only"
    assert _detect_format({"completion": "x"}) == "completion_only"
    assert _detect_format({"foo": "bar"}) == "unknown"
    assert _detect_format([1, 2, 3]) == "unknown"
    assert _detect_format("string") == "unknown"
