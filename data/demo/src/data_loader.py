"""数据加载：load_jsonl / TextDataset / collate_fn。

仅依赖 NumPy + 标准库。返回的 (x, y) 是 ``np.ndarray``，verse_torch.training.Trainer
内部会用 ``_as_tensor`` 自动转为 Tensor。
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[dict]:
    """读取 JSONL 文件，每行一个 JSON 对象。

    Args:
        path: JSONL 文件路径
    Returns:
        List[dict]：每行解析为的 dict
    """
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 解析失败：{path} 第 {line_no} 行 - {e}"
                ) from e
    return items


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------


class TextDataset:
    """基于 jsonl + tokenizer 的 next-token prediction 数据集。

    流程：
        1. 加载 jsonl，提取 ``text`` 字段
        2. 把所有 text 拼接（用换行分隔），用 tokenizer.encode 编码为 id 序列
        3. 把 id 序列切成长度 seq_len 的块
        4. __getitem__(i) 返回 (x, y)，y 为 x 左移一位（next-token 目标）

    Args:
        tok: tokenizer 对象，需有 ``encode(text)`` 方法
        jsonl_path: jsonl 文件路径
        seq_len: 单个样本长度
        min_tokens: 最少需要的 token 数；不足则抛出错误
    """

    def __init__(
        self,
        tok,
        jsonl_path: str,
        seq_len: int,
        min_tokens: int = 0,
    ):
        self.tok = tok
        self.seq_len = int(seq_len)
        self.jsonl_path = jsonl_path

        # 1. 加载 jsonl
        items = load_jsonl(jsonl_path)
        texts = []
        for it in items:
            if isinstance(it, dict) and "text" in it:
                texts.append(str(it["text"]))
            elif isinstance(it, str):
                texts.append(it)
        if not texts:
            raise ValueError(f"jsonl 中没有有效 text 字段: {jsonl_path}")

        # 2. 编码为 id 序列（每条 text 之间用换行分隔）
        all_ids: List[int] = []
        for text in texts:
            ids = self._safe_encode(text)
            all_ids.extend(ids)
            # 文本之间插入换行 token（如果有）
            nl_ids = self._safe_encode("\n")
            all_ids.extend(nl_ids)

        ids_np = np.asarray(all_ids, dtype=np.int64)
        # 3. 截断为 seq_len 的整数倍
        n_blocks = len(ids_np) // self.seq_len
        if n_blocks == 0:
            # 数据太少：用滑动窗口 padding 也能至少产出 1 个样本
            # 但 PoC 阶段直接抛错让用户调整数据
            raise ValueError(
                f"token 数 {len(ids_np)} 不足 seq_len={self.seq_len}，"
                f"无法构造任何样本。请增加数据量或减小 seq_len。"
            )
        # 保留前 n_blocks * seq_len 个 token
        self.ids = ids_np[: n_blocks * self.seq_len]
        self.n_blocks = n_blocks

    def _safe_encode(self, text: str) -> List[int]:
        """对 tokenizer.encode 兼容不同签名。"""
        try:
            return list(self.tok.encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return list(self.tok.encode(text))
            except Exception:
                return []

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        """返回 (x, y)，shape (seq_len,)。

        x = ids[i*seq_len : (i+1)*seq_len]
        y = ids[i*seq_len+1 : (i+1)*seq_len+1]
        若 y 长度不足（仅最后一块可能发生），用 0 padding。
        """
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"index {i} 超出范围 [0, {self.n_blocks})")
        start = i * self.seq_len
        end = start + self.seq_len
        x = self.ids[start:end]
        # y = x 左移一位：取 [start+1, end+1]
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = self.ids[y_start:y_end]
        else:
            # 边界情况：用 0 padding 末尾
            y_avail = self.ids[y_start: len(self.ids)]
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
        return x, y


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


def collate_fn(batch, pad_id: int = 0):
    """把 list of (x, y) 堆叠为 batched ndarray。

    假设所有 (x, y) 长度一致（TextDataset 保证），直接 stack。

    Args:
        batch: List[Tuple[ndarray, ndarray]]
        pad_id: padding id（PoC 数据集已保证等长，此参数保留兼容）
    Returns:
        (x_batch, y_batch)：均为 np.ndarray，shape (B, seq_len)
    """
    xs = [np.asarray(b[0], dtype=np.int64) for b in batch]
    ys = [np.asarray(b[1], dtype=np.int64) for b in batch]
    x_batch = np.stack(xs, axis=0)
    y_batch = np.stack(ys, axis=0)
    return x_batch, y_batch


# ---------------------------------------------------------------------------
# DataLoader：把 TextDataset + collate_fn 组装成可迭代的 batch 生成器
# ---------------------------------------------------------------------------


class BatchLoader:
    """简易 batch loader：可迭代，每次返回一个 batch。

    与 PyTorch DataLoader 接口对齐（仅 PoC 所需的最小集）。

    Args:
        dataset: TextDataset 或任何实现 __len__ / __getitem__ 的对象
        batch_size: batch 大小
        shuffle: 是否每轮打乱顺序
        collate_fn: 批处理函数
        drop_last: 是否丢弃最后不足 batch_size 的样本
        seed: 随机种子（仅 shuffle=True 时生效）
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.collate_fn = collate_fn or (lambda b: tuple(np.stack(x) for x in zip(*b)))
        self.drop_last = bool(drop_last)
        self.rng = np.random.RandomState(seed)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)
        for s in range(0, n, self.batch_size):
            batch_idx = indices[s: s + self.batch_size]
            if len(batch_idx) == 0:
                continue
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            batch = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(batch)


__all__ = ["load_jsonl", "TextDataset", "collate_fn", "BatchLoader"]
