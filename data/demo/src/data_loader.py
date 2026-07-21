"""数据加载：支持 chat 数组 + prompt-completion 双格式。

仅依赖 NumPy + 标准库。返回的 (x, y) 是 ``np.ndarray``，verse_torch.training.Trainer
内部会用 ``_as_tensor`` 自动转为 Tensor。

支持的样本格式：
    1. chat 数组：``[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]``
       → 逐消息编码为 ``<|role|>content``，assistant content 参与 loss
    2. prompt-completion：``{"prompt":"...","completion":"..."}``
       → ``<|user|>prompt<|assistant|>completion<|eos|>``，completion 参与 loss

旧版 ``{"text":"..."}`` 格式已废弃，会抛出 ``ValueError``。
"""

from __future__ import annotations

import json
from typing import List

import numpy as np

# 尝试导入 chat_template 常量（verse_tokenizer 已升级，含此模块）
try:
    from verse_tokenizer.chat_template import EOS_TOKEN as _EOS_STR
except Exception:  # pragma: no cover - verse_tokenizer 不可用时的降级
    _EOS_STR = "<|eos|>"


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[dict]:
    """读取 JSONL 文件，每行一个 JSON 对象（或 JSON 数组）。

    Args:
        path: JSONL 文件路径
    Returns:
        List：每行解析为的 dict 或 list
    """
    items: List = []
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
# 格式检测
# ---------------------------------------------------------------------------


def _detect_format(item) -> str:
    """检测样本格式。

    Returns:
        ``"chat"`` | ``"prompt_completion"`` | ``"legacy_text"`` | ``"unknown"``
    """
    if isinstance(item, list):
        # chat 数组：每个元素是 {"role":..., "content":...}
        if all(isinstance(x, dict) and "role" in x and "content" in x for x in item):
            return "chat"
        return "unknown"
    if isinstance(item, dict):
        if "prompt" in item and "completion" in item:
            return "prompt_completion"
        if "text" in item:
            return "legacy_text"
    return "unknown"


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------


class TextDataset:
    """基于 jsonl + tokenizer 的 next-token prediction 数据集。

    支持两种样本格式：
        1. chat 数组：``[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]``
           → 逐消息编码为 ``<|role|>content``，assistant content 参与 loss
        2. prompt-completion：``{"prompt":"...","completion":"..."}``
           → ``<|user|>prompt<|assistant|>completion<|eos|>``，completion 参与 loss

    旧版 ``{"text":"..."}`` 格式已废弃，会抛出 ``ValueError``。

    Args:
        tok: tokenizer 对象（需有 ``encode`` 方法；若有
            ``apply_prompt_template`` 则优先使用）
        jsonl_path: jsonl 文件路径
        seq_len: 单个样本长度
        min_tokens: 最少需要的 token 数（保留接口，当前未使用）
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
        self.min_tokens = int(min_tokens)

        # 1. 加载 jsonl
        items = load_jsonl(jsonl_path)

        # 2. 逐样本编码为 (ids, loss_mask)
        #    loss_mask[i] = 1 表示该位置参与 loss，0 表示屏蔽（prompt 部分）
        all_ids: List[int] = []
        all_mask: List[int] = []

        for item in items:
            fmt = _detect_format(item)
            if fmt == "legacy_text":
                raise ValueError(
                    f"旧版 text 格式已废弃，请使用 chat 数组或 prompt-completion 格式。"
                    f"文件：{jsonl_path}"
                )
            if fmt == "unknown":
                # 跳过未知格式（不中断加载，但记录警告）
                continue

            if fmt == "chat":
                ids, mask = self._encode_chat(item)
            else:  # prompt_completion
                ids, mask = self._encode_prompt_completion(item)

            all_ids.extend(ids)
            all_mask.extend(mask)
            # 样本之间用换行分隔（不参与 loss）
            nl_ids = self._safe_encode("\n")
            all_ids.extend(nl_ids)
            all_mask.extend([0] * len(nl_ids))

        if not all_ids:
            raise ValueError(f"jsonl 中没有有效样本: {jsonl_path}")

        ids_np = np.asarray(all_ids, dtype=np.int64)
        mask_np = np.asarray(all_mask, dtype=np.int64)

        # 3. 截断为 seq_len 的整数倍
        n_blocks = len(ids_np) // self.seq_len
        if n_blocks == 0:
            raise ValueError(
                f"token 数 {len(ids_np)} 不足 seq_len={self.seq_len}，"
                f"无法构造任何样本。请增加数据量或减小 seq_len。"
            )
        self.ids = ids_np[: n_blocks * self.seq_len]
        self.mask = mask_np[: n_blocks * self.seq_len]
        self.n_blocks = n_blocks

    # ------------------------------------------------------------------
    # 编码方法
    # ------------------------------------------------------------------

    def _encode_chat(self, messages) -> tuple:
        """编码 chat 数组为 (ids, loss_mask)。

        逐消息编码：``<|role|>`` 标记 + content。
        assistant content 参与 loss（mask=1），其他不参与（mask=0）。
        末尾追加 ``<|eos|>``（参与 loss）。
        """
        ids: List[int] = []
        mask: List[int] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # 编码 role marker <|role|>（不参与 loss）
            marker = f"<|{role}|>"
            marker_ids = self._safe_encode(marker)
            ids.extend(marker_ids)
            mask.extend([0] * len(marker_ids))
            # 编码 content（assistant 参与 loss）
            content_ids = self._safe_encode(content)
            ids.extend(content_ids)
            if role == "assistant":
                mask.extend([1] * len(content_ids))
            else:
                mask.extend([0] * len(content_ids))
        # 末尾 <|eos|>（参与 loss）
        eos_ids = self._safe_encode(_EOS_STR)
        ids.extend(eos_ids)
        mask.extend([1] * len(eos_ids))
        return ids, mask

    def _encode_prompt_completion(self, item: dict) -> tuple:
        """编码 prompt-completion 为 (ids, loss_mask)。

        拼接为 ``<|user|>prompt<|assistant|>completion<|eos|>``。
        loss mask: completion + <|eos|> 部分为 1，prompt 部分为 0。
        """
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")

        # 优先用 apply_prompt_template 编码 prompt 部分（含 <|user|>...<|assistant|>）
        prompt_ids: List[int] = []
        if hasattr(self.tok, "apply_prompt_template"):
            try:
                prompt_ids = list(self.tok.apply_prompt_template(prompt))
            except Exception:
                prompt_ids = self._safe_encode(prompt)
        else:
            prompt_ids = self._safe_encode(prompt)

        completion_ids = self._safe_encode(completion)
        eos_ids = self._safe_encode(_EOS_STR)

        ids = list(prompt_ids) + completion_ids + eos_ids
        mask = (
            [0] * len(prompt_ids)
            + [1] * len(completion_ids)
            + [1] * len(eos_ids)
        )
        return ids, mask

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _safe_encode(self, text: str) -> List[int]:
        """对 tokenizer.encode 兼容不同签名。"""
        if not text:
            return []
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
        y = ids[i*seq_len+1 : (i+1)*seq_len+1]（next-token target）
        若某位置 mask=0，y 设为 -100（ignore_index，cross_entropy_loss 自动忽略）。
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
            y_mask = self.mask[y_start:y_end]
        else:
            # 边界情况：用 0 padding 末尾
            y_avail = self.ids[y_start: len(self.ids)]
            y_mask_avail = self.mask[y_start: len(self.ids)]
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        # 把 mask=0 的位置设为 -100（ignore_index）
        y = np.where(y_mask == 1, y, -100)
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
# BatchLoader：把 TextDataset + collate_fn 组装成可迭代的 batch 生成器
# ---------------------------------------------------------------------------


class BatchLoader:
    """简易 batch loader：可迭代，每次返回一个 batch。

    与 PyTorch DataLoader 接口对齐（仅 PoC 所需的最小集）。
    ``num_workers`` / ``pin_memory`` / ``persistent_workers`` 为占位参数
    （CPU-only 实现忽略，保留以匹配 PyTorch API）。

    Args:
        dataset: TextDataset 或任何实现 __len__ / __getitem__ 的对象
        batch_size: batch 大小
        shuffle: 是否每轮打乱顺序
        collate_fn: 批处理函数
        drop_last: 是否丢弃最后不足 batch_size 的样本
        seed: 随机种子（仅 shuffle=True 时生效）
        num_workers: 占位参数（CPU-only，默认 0，忽略）
        pin_memory: 占位参数（CPU-only，默认 False，忽略）
        persistent_workers: 占位参数（默认 False，忽略）
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.collate_fn = collate_fn or (lambda b: tuple(np.stack(x) for x in zip(*b)))
        self.drop_last = bool(drop_last)
        self.rng = np.random.RandomState(seed)
        self._epoch = 0
        # 占位参数（CPU-only 实现忽略）
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

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
