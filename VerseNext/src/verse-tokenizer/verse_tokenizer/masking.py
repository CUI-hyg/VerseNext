"""SFT loss 掩码工具：根据内容字节区间生成 labels（区间外 -100）。"""

from __future__ import annotations


def labels_from_spans(
    ids: list[int],
    offsets: list[tuple[int, int]],
    train_byte_spans: list[tuple[int, int]],
) -> list[int]:
    """生成 SFT labels：token 的字节范围与任一训练区间有交集则保留，否则 -100。

    Args:
        ids: token id 序列。
        offsets: 每个 token 的 ``(byte_start, byte_end)`` 偏移。
        train_byte_spans: 参与训练的字节区间列表 ``[(start, end), ...]``
            （通常为 assistant 回答 + 结束标记）。
    """
    labels = list(ids)
    for i, (s, e) in enumerate(offsets):
        if s == e:
            # 空偏移（如补加的 bos）不参与训练
            labels[i] = -100
            continue
        in_span = any(s < span_end and e > span_start for span_start, span_end in train_byte_spans)
        if not in_span:
            labels[i] = -100
    return labels
