"""数据：token 流式批迭代器。

设计为对 token id 序列（list/np/torch 均可）做随机窗口采样，避免为玩具级
训练引入 HF datasets 等重依赖；后续可在同一接口下接入真实数据集。

内存优化：token 流以 **int32** 保存（而非 int64），且 numpy/memmap 输入零拷贝
共享内存（``torch.from_numpy``）；仅在采样出 (B, seq) 窗口后才转 int64 送入
模型。大语料下 token 流内存占用约为原来的 1/2，并省去 list→tensor 的全量拷贝。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from verse_core.config import BaseConfig


def fingerprint_source(source, sample_limit: int = 5_000_000) -> dict | None:
    """对训练数据做轻量指纹，用于断点续训时检测「数据是否变化」。

    - token 流（``ndarray``/``Tensor``/带 ``.tokens`` 的迭代器）：按内容哈希
      （超长时对首尾各 ``sample_limit`` 个元素采样，避免大语料哈希过慢）。
    - 样本集（带 ``.examples`` 的 SFTDataset）：对编码后的 ids 哈希。

    Returns:
        ``{"kind", "n", "sha256", "sampled", ...}``；无法识别返回 ``None``。
    """
    if source is None:
        return None

    tokens = getattr(source, "tokens", None)
    if tokens is None and isinstance(source, (np.ndarray, torch.Tensor)):
        tokens = source
    if tokens is not None:
        arr = tokens.detach().cpu().numpy() if hasattr(tokens, "detach") else np.asarray(tokens)
        arr = np.ascontiguousarray(arr.reshape(-1))
        if arr.dtype.kind not in "iu":
            arr = arr.astype(np.int32)
        digest = hashlib.sha256()
        sampled = arr.size > sample_limit
        if sampled:
            digest.update(arr[:sample_limit].tobytes())
            digest.update(arr[-sample_limit:].tobytes())
        else:
            digest.update(arr.tobytes())
        return {
            "kind": "tokens",
            "n": int(arr.size),
            "dtype": str(arr.dtype),
            "sha256": digest.hexdigest()[:16],
            "sampled": bool(sampled),
        }

    examples = getattr(source, "examples", None)
    if examples is not None:
        digest = hashlib.sha256()
        for item in examples:
            ids = item[0] if isinstance(item, (tuple, list)) else item
            digest.update(np.asarray(ids, dtype=np.int32).tobytes())
        return {
            "kind": "examples",
            "n": len(examples),
            "sha256": digest.hexdigest()[:16],
            "sampled": False,
        }
    return None


@dataclass
class DataConfig(BaseConfig):
    """数据相关配置。"""

    batch_size: int = 8
    seq_len: int = 256
    """训练窗口长度（模型输入 token 数）。"""
    seed: int = 42
    drop_last: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError("batch_size / seq_len 必须为正数")


class TokenBatchIterator:
    """从一维 token 序列构造随机 (inputs, targets) 批次。

    targets 为 inputs 右移一位（next-token prediction）。

    Args:
        tokens: 一维 token id 序列。
        config: :class:`DataConfig`。
        device: 批次目标设备。
    """

    def __init__(
        self,
        tokens,
        config: DataConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        if len(tokens) < config.seq_len + 1:
            raise ValueError(
                f"token 序列长度({len(tokens)})不足 seq_len+1({config.seq_len + 1})"
            )
        self.tokens = self._as_int_tensor(tokens)
        self.config = config
        self.device = torch.device(device)
        self._pin = self.device.type == "cuda"
        self._generator = torch.Generator().manual_seed(config.seed)

    @staticmethod
    def _as_int_tensor(tokens) -> torch.Tensor:
        """统一为 int32 tensor；numpy/memmap 零拷贝，list 走一次性转换。"""
        if isinstance(tokens, torch.Tensor):
            return tokens if tokens.dtype in (torch.int32, torch.int64) else tokens.to(torch.int32)
        if isinstance(tokens, np.ndarray):
            if tokens.dtype != np.int32:
                tokens = tokens.astype(np.int32)
            return torch.from_numpy(tokens)  # 与 numpy/memmap 共享内存
        return torch.as_tensor(tokens, dtype=torch.int32)

    def __len__(self) -> int:
        """完整可切分的批次数（估计值）。"""
        n_windows = len(self.tokens) - self.config.seq_len - 1
        n_batches = n_windows // self.config.batch_size
        return max(n_batches, 1)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """采样一个随机批次，返回 (inputs, targets)。"""
        cfg = self.config
        max_start = len(self.tokens) - cfg.seq_len - 1
        starts = torch.randint(0, max_start + 1, (cfg.batch_size,), generator=self._generator)
        arange = torch.arange(cfg.seq_len + 1)
        # (batch, seq_len+1) 的窗口（int32，仅窗口级拷贝）
        windows = self.tokens[starts[:, None] + arange[None, :]]
        inputs, targets = windows[:, :-1], windows[:, 1:]
        if self._pin:
            inputs, targets = inputs.pin_memory(), targets.pin_memory()
        return (
            inputs.to(self.device, dtype=torch.long, non_blocking=self._pin),
            targets.to(self.device, dtype=torch.long, non_blocking=self._pin),
        )

    def batches(self, num_batches: int):
        """生成指定数量的批次（用于一轮训练/评估）。"""
        for _ in range(num_batches):
            yield self.next_batch()
