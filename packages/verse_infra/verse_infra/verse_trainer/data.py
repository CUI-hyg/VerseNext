"""数据加载：支持 chat 数组 + prompt-completion + text 三种格式。

Part4K1 Task 6.3 升级：
- 保留原 :class:`TextDataset`（向后兼容 ``data/demo`` 旧入口）。
- 新增 :class:`CachedDataset`：首次扫描数据集后缓存为 ``.npz``
  （token ids + loss mask 数组），后续启动毫秒级加载。
  支持流式 lazy load（大数据集不全量载入内存，按 batch 读取）。
- 保留原 :func:`collate_fn` / :class:`BatchLoader` 兼容。

支持的样本格式：
    1. chat 数组：``[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]``
       → 逐消息编码为 ``<|role|>content``，assistant content 参与 loss
    2. prompt-completion：``{"prompt":"...","completion":"..."}``
       → ``<|user|>prompt<|assistant|>completion<|eos|>``，completion 参与 loss
       （允许单样本：只存在 prompt 或只存在 completion 时，存在字段当作纯文本，全部参与 loss）
    3. text：``{"text":"..."}``
       → 纯文本，所有 token 参与 loss（适用于预训练 / 续训）
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

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
        ``"chat"`` | ``"prompt_completion"`` | ``"prompt_only"`` |
        ``"completion_only"`` | ``"text"`` | ``"unknown"``
    """
    if isinstance(item, list):
        # chat 数组：每个元素是 {"role":..., "content":...}
        if all(isinstance(x, dict) and "role" in x and "content" in x for x in item):
            return "chat"
        return "unknown"
    if isinstance(item, dict):
        if "prompt" in item and "completion" in item:
            return "prompt_completion"
        # 单样本：只存在 prompt 或只存在 completion
        if "prompt" in item:
            return "prompt_only"
        if "completion" in item:
            return "completion_only"
        if "text" in item:
            return "text"
    return "unknown"


# ---------------------------------------------------------------------------
# 编码辅助（TextDataset 与 CachedDataset 共用）
# ---------------------------------------------------------------------------


def _safe_encode(tok, text: str) -> List[int]:
    """对 tokenizer.encode 兼容不同签名。"""
    if not text:
        return []
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        try:
            return list(tok.encode(text))
        except Exception:
            return []


def _encode_chat(tok, messages) -> tuple:
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
        marker = f"<|{role}|>"
        marker_ids = _safe_encode(tok, marker)
        ids.extend(marker_ids)
        mask.extend([0] * len(marker_ids))
        content_ids = _safe_encode(tok, content)
        ids.extend(content_ids)
        if role == "assistant":
            mask.extend([1] * len(content_ids))
        else:
            mask.extend([0] * len(content_ids))
    eos_ids = _safe_encode(tok, _EOS_STR)
    ids.extend(eos_ids)
    mask.extend([1] * len(eos_ids))
    return ids, mask


def _encode_prompt_completion(tok, item: dict) -> tuple:
    """编码 prompt-completion 为 (ids, loss_mask)。

    拼接为 ``<|user|>prompt<|assistant|>completion<|eos|>``。
    loss mask: completion + <|eos|> 部分为 1，prompt 部分为 0。
    """
    prompt = item.get("prompt", "")
    completion = item.get("completion", "")

    prompt_ids: List[int] = []
    if hasattr(tok, "apply_prompt_template"):
        try:
            prompt_ids = list(tok.apply_prompt_template(prompt))
        except Exception:
            prompt_ids = _safe_encode(tok, prompt)
    else:
        prompt_ids = _safe_encode(tok, prompt)

    completion_ids = _safe_encode(tok, completion)
    eos_ids = _safe_encode(tok, _EOS_STR)

    ids = list(prompt_ids) + completion_ids + eos_ids
    mask = (
        [0] * len(prompt_ids)
        + [1] * len(completion_ids)
        + [1] * len(eos_ids)
    )
    return ids, mask


def _encode_text(tok, text: str) -> tuple:
    """编码纯文本为 (ids, loss_mask)，所有 token 参与 loss。

    末尾追加 ``<|eos|>``（参与 loss），作为样本边界标记。
    """
    ids = _safe_encode(tok, text)
    eos_ids = _safe_encode(tok, _EOS_STR)
    ids = list(ids) + eos_ids
    mask = [1] * len(ids)
    return ids, mask


def _encode_item(tok, item) -> Optional[tuple]:
    """编码单个样本为 (ids, mask)，未知格式返回 None。"""
    fmt = _detect_format(item)
    if fmt == "unknown":
        return None
    if fmt == "chat":
        return _encode_chat(tok, item)
    if fmt == "prompt_completion":
        return _encode_prompt_completion(tok, item)
    # text / prompt_only / completion_only：存在的字段当作纯文本，全部参与 loss
    text = item.get("text") or item.get("prompt") or item.get("completion") or ""
    return _encode_text(tok, text)


# ---------------------------------------------------------------------------
# TextDataset（原 data/demo/src/data_loader.py，保持向后兼容）
# ---------------------------------------------------------------------------


class TextDataset:
    """基于 jsonl + tokenizer 的 next-token prediction 数据集。

    支持三种样本格式：
        1. chat 数组：``[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]``
           → 逐消息编码为 ``<|role|>content``，assistant content 参与 loss
        2. prompt-completion：``{"prompt":"...","completion":"..."}``
           → ``<|user|>prompt<|assistant|>completion<|eos|>``，completion 参与 loss
           （允许单样本：只存在 prompt 或只存在 completion 时，存在字段当作纯文本，全部参与 loss）
        3. text：``{"text":"..."}``
           → 纯文本，所有 token 参与 loss（适用于预训练 / 续训）

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
        all_ids: List[int] = []
        all_mask: List[int] = []

        for item in items:
            encoded = _encode_item(tok, item)
            if encoded is None:
                continue
            ids, mask = encoded
            all_ids.extend(ids)
            all_mask.extend(mask)
            # 样本之间用换行分隔（不参与 loss）
            nl_ids = _safe_encode(tok, "\n")
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
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = self.ids[y_start:y_end]
            y_mask = self.mask[y_start:y_end]
        else:
            y_avail = self.ids[y_start: len(self.ids)]
            y_mask_avail = self.mask[y_start: len(self.ids)]
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        y = np.where(y_mask == 1, y, -100)
        return x, y


# ---------------------------------------------------------------------------
# CachedDataset：.npz 缓存 + 流式 lazy load
# ---------------------------------------------------------------------------


class CachedDataset:
    """带 ``.npz`` 缓存的数据集（Part4K1 Task 6.3）。

    在 :class:`TextDataset` 之上新增：
    1. **首次扫描缓存**：编码完成后把 ``ids`` / ``mask`` 数组以 ``.npz``
       保存到 ``cache_path``（默认 ``<data_path>.cache.npz``）。
       后续启动直接 ``np.load`` 缓存，跳过 jsonl 解析 + tokenizer 编码，
       毫秒级加载。
    2. **流式 lazy load**：``lazy=True`` 时不在 ``__init__`` 全量载入
       内存，而是通过 ``np.load(..., mmap_mode='r')`` 内存映射 .npz 解包
       后的临时文件，按 batch 读取。仅当 ``__getitem__`` 被调用时才
       触发对应区间的磁盘读取。

    缓存文件包含：
    - ``ids`` : int64 数组（全部 token id 顺序拼接）
    - ``mask`` : int64 数组（与 ids 等长，loss mask）
    - ``seq_len`` : int 标量
    - ``n_blocks`` : int 标量
    - ``source`` : 字符串（原始 jsonl 路径，便于校验）

    若 ``data_path`` 的 mtime 比缓存文件新，则缓存自动失效重建。

    Args:
        tok: tokenizer 对象
        data_path: jsonl 文件路径
        seq_len: 单个样本长度
        cache_path: 缓存文件路径；None 则默认 ``<data_path>.cache.npz``
        lazy: 是否启用流式 lazy load（默认 False，全量载入内存）
        min_tokens: 最少 token 数（保留接口）
    """

    def __init__(
        self,
        tok,
        data_path: str,
        seq_len: int,
        cache_path: Optional[str] = None,
        lazy: bool = False,
        min_tokens: int = 0,
    ):
        self.tok = tok
        self.data_path = data_path
        self.seq_len = int(seq_len)
        self.min_tokens = int(min_tokens)
        self.lazy = bool(lazy)

        if cache_path is None:
            cache_path = f"{data_path}.cache.npz"
        self.cache_path = cache_path

        # 尝试加载缓存；若缓存不存在 / 过期 / 损坏，则重建
        loaded = self._try_load_cache()
        if not loaded:
            self._build_and_save_cache()
            # 重建后再加载一次（确保 ids/mask 已就位）
            if not self._try_load_cache():
                raise RuntimeError(
                    f"CachedDataset 缓存重建后仍无法加载：{self.cache_path}"
                )

    # ------------------------------------------------------------------
    # 缓存加载 / 重建
    # ------------------------------------------------------------------

    def _try_load_cache(self) -> bool:
        """尝试从 cache_path 加载 ids/mask。

        Returns:
            True 表示成功加载（self.ids / self.mask / self.n_blocks 已就位）；
            False 表示缓存不存在 / 过期 / 损坏，需要重建。
        """
        if not os.path.exists(self.cache_path):
            return False
        # 缓存过期检测：源文件比缓存新则失效
        try:
            src_mtime = os.path.getmtime(self.data_path)
            cache_mtime = os.path.getmtime(self.cache_path)
            if src_mtime > cache_mtime:
                return False
        except OSError:
            return False

        try:
            if self.lazy:
                # mmap_mode='r' 内存映射：不占内存，按需读盘
                archive = np.load(self.cache_path, allow_pickle=False, mmap_mode='r')
                self.ids = archive["ids"]
                self.mask = archive["mask"]
                self._archive = archive  # 持有引用避免文件关闭
            else:
                archive = np.load(self.cache_path, allow_pickle=False)
                self.ids = np.asarray(archive["ids"], dtype=np.int64)
                self.mask = np.asarray(archive["mask"], dtype=np.int64)
                self._archive = None
            self.n_blocks = int(archive["n_blocks"])
            cached_seq_len = int(archive["seq_len"])
            if cached_seq_len != self.seq_len:
                # seq_len 变了，缓存失效
                self._close_archive()
                return False
            self._cached_source = str(archive["source"]) if "source" in archive.files else ""
            return True
        except Exception as e:
            print(f"[CachedDataset] 缓存加载失败，将重建：{e}", flush=True)
            self._close_archive()
            return False

    def _close_archive(self) -> None:
        """关闭 mmap archive（若有）。"""
        arc = getattr(self, "_archive", None)
        if arc is not None:
            try:
                arc.close()
            except Exception:
                pass
            self._archive = None

    def _build_and_save_cache(self) -> None:
        """扫描 jsonl + 编码 → 保存到 cache_path。"""
        print(f"[CachedDataset] 首次扫描数据集 {self.data_path}，构建缓存...",
              flush=True)
        items = load_jsonl(self.data_path)

        all_ids: List[int] = []
        all_mask: List[int] = []
        for item in items:
            encoded = _encode_item(self.tok, item)
            if encoded is None:
                continue
            ids, mask = encoded
            all_ids.extend(ids)
            all_mask.extend(mask)
            nl_ids = _safe_encode(self.tok, "\n")
            all_ids.extend(nl_ids)
            all_mask.extend([0] * len(nl_ids))

        if not all_ids:
            raise ValueError(f"jsonl 中没有有效样本: {self.data_path}")

        ids_np = np.asarray(all_ids, dtype=np.int64)
        mask_np = np.asarray(all_mask, dtype=np.int64)
        n_blocks = len(ids_np) // self.seq_len
        if n_blocks == 0:
            raise ValueError(
                f"token 数 {len(ids_np)} 不足 seq_len={self.seq_len}，"
                f"无法构造任何样本。请增加数据量或减小 seq_len。"
            )
        ids_np = ids_np[: n_blocks * self.seq_len]
        mask_np = mask_np[: n_blocks * self.seq_len]

        # 保存缓存（覆盖旧文件）
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(
            self.cache_path,
            ids=ids_np,
            mask=mask_np,
            seq_len=np.int64(self.seq_len),
            n_blocks=np.int64(n_blocks),
            source=np.asarray(self.data_path),
        )
        print(f"[CachedDataset] 缓存已保存到 {self.cache_path} "
              f"(n_blocks={n_blocks}, tokens={len(ids_np)})", flush=True)

    # ------------------------------------------------------------------
    # __len__ / __getitem__（与 TextDataset 接口一致）
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        """返回 (x, y)，shape (seq_len,)。

        与 :class:`TextDataset.__getitem__` 完全一致；mask=0 的位置 y=-100。
        lazy 模式下 self.ids 是 mmap 数组，切片会触发磁盘读取并返回
        普通 ndarray（拷贝），不持有 mmap 引用。
        """
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"index {i} 超出范围 [0, {self.n_blocks})")
        start = i * self.seq_len
        end = start + self.seq_len
        # 切片：mmap 数组切片自动转普通 ndarray（拷贝到内存）
        x = np.array(self.ids[start:end], dtype=np.int64)
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = np.array(self.ids[y_start:y_end], dtype=np.int64)
            y_mask = np.array(self.mask[y_start:y_end], dtype=np.int64)
        else:
            y_avail = np.array(self.ids[y_start: len(self.ids)], dtype=np.int64)
            y_mask_avail = np.array(self.mask[y_start: len(self.ids)], dtype=np.int64)
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        y = np.where(y_mask == 1, y, -100)
        return x, y


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


def collate_fn(batch, pad_id: int = 0):
    """把 list of (x, y) 堆叠为 batched ndarray。

    假设所有 (x, y) 长度一致（TextDataset / CachedDataset 保证），直接 stack。

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
# BatchLoader：把 dataset + collate_fn 组装成可迭代的 batch 生成器
# ---------------------------------------------------------------------------


class BatchLoader:
    """简易 batch loader：可迭代，每次返回一个 batch。

    与 PyTorch DataLoader 接口对齐（仅 PoC 所需的最小集）。
    ``num_workers`` / ``pin_memory`` / ``persistent_workers`` 为占位参数
    （CPU-only 实现忽略，保留以匹配 PyTorch API）。

    Args:
        dataset: TextDataset / CachedDataset 或任何实现 __len__ / __getitem__ 的对象
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


# ---------------------------------------------------------------------------
# 单样本构造（Task 6.5：--single-sample 支持）
# ---------------------------------------------------------------------------


class SingleSampleDataset:
    """单样本数据集：把一条 prompt/completion 或单文件文本包装为 dataset。

    用于 ``--single-sample --prompt "..." --completion "..."`` 的快速训练/推理调试。
    内部复用 :func:`_encode_item` 编码逻辑，重复填充到至少 1 个 seq_len 块。

    Args:
        tok: tokenizer 对象
        prompt: prompt 文本（可空）
        completion: completion 文本（可空）
        text: 纯文本（与 prompt/completion 二选一）
        seq_len: 单个样本长度
        n_repeat: 把单样本重复多少次拼接到一起（保证至少 1 个 seq_len 块）
    """

    def __init__(
        self,
        tok,
        prompt: str = "",
        completion: str = "",
        text: str = "",
        seq_len: int = 32,
        n_repeat: int = 8,
    ):
        self.tok = tok
        self.seq_len = int(seq_len)

        if text:
            item = {"text": text}
        elif prompt and completion:
            item = {"prompt": prompt, "completion": completion}
        elif prompt:
            item = {"prompt": prompt}
        elif completion:
            item = {"completion": completion}
        else:
            item = {"text": ""}

        encoded = _encode_item(tok, item)
        if encoded is None or not encoded[0]:
            # 编码失败：用 0 token 兜底
            ids = [0]
            mask = [1]
        else:
            ids, mask = encoded

        # 重复 n_repeat 次拼接到一起，保证至少 1 个 seq_len 块
        all_ids = list(ids) * max(1, n_repeat)
        all_mask = list(mask) * max(1, n_repeat)

        ids_np = np.asarray(all_ids, dtype=np.int64)
        mask_np = np.asarray(all_mask, dtype=np.int64)
        n_blocks = max(1, len(ids_np) // self.seq_len)
        self.ids = ids_np[: n_blocks * self.seq_len]
        self.mask = mask_np[: n_blocks * self.seq_len]
        self.n_blocks = n_blocks

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"index {i} 超出范围 [0, {self.n_blocks})")
        start = i * self.seq_len
        end = start + self.seq_len
        x = self.ids[start:end]
        y_start = start + 1
        y_end = end + 1
        if y_end <= len(self.ids):
            y = self.ids[y_start:y_end]
            y_mask = self.mask[y_start:y_end]
        else:
            y_avail = self.ids[y_start: len(self.ids)]
            y_mask_avail = self.mask[y_start: len(self.ids)]
            pad_len = self.seq_len - len(y_avail)
            y = np.concatenate([y_avail, np.zeros(pad_len, dtype=np.int64)])
            y_mask = np.concatenate(
                [y_mask_avail, np.zeros(pad_len, dtype=np.int64)]
            )
        y = np.where(y_mask == 1, y, -100)
        return x, y


__all__ = [
    "load_jsonl",
    "TextDataset",
    "CachedDataset",
    "SingleSampleDataset",
    "collate_fn",
    "BatchLoader",
]
