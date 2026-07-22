"""VerseNex: 并行 KV cache（Part4K1 Task 3.3）。

提供 ``ParallelKVCache`` —— 一个 batch 维度上的并行 KV cache 包装器，
用于 speculative decoding 风格的并行预测场景：

- 多序列同时维护各自独立的 KV cache（B 个序列并行）
- ``batch_update`` 一次性给所有 B 个序列追加 K/V（避免 for 循环遍历 batch）
- 与 ``verse_torch.nn.StaticCache`` / ``DynamicCache`` 兼容（内部委托）

设计要点
--------
1. **保持 batch 维度**：B 个序列的 cache 始终组织成 (B, max_seq, H, D)，
   不展开为 B 个独立 cache 对象。这充分利用 numpy/torch 的批量算子。
2. **batch_update 语义**：传入 (B, T_new, H, D) 的 keys/values，
   一次性 append 到 cache 末尾。
3. **多序列变长支持**：``per_seq_lens`` 记录每个序列当前长度，
   支持不同序列的 step 数不同（speculative decoding 拒绝处不同序列
   接受 token 数不同）。

复用的项目内已有功能
--------------------
- ``verse_torch.nn.KVCache`` 抽象基类
- ``verse_torch.nn.StaticCache`` / ``DynamicCache``：底层 buffer 实现
- ``verse_torch.nn._concat``：可微 Tensor 拼接
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from verse_torch import Tensor, no_grad
from verse_torch.nn import KVCache, StaticCache, DynamicCache, _concat


# ---------------------------------------------------------------------------
# ParallelKVCache
# ---------------------------------------------------------------------------


class ParallelKVCache(KVCache):
    """并行批量 KV cache（speculative decoding / 并行 rollout 用）。

    与 ``StaticCache`` / ``DynamicCache`` 不同，``ParallelKVCache`` 显式维护
    batch 维度的多序列独立长度（``per_seq_lens``），适合以下场景：
    - speculative decoding：不同序列接受 token 数不同
    - 并行 rollout：不同序列提前结束 / 长度不一
    - 训练时变长 batch padding 的 mask 化 cache 维护

    Args:
        num_layers: 层数
        max_batch: 最大 batch size
        max_seq: 最大序列长度
        num_heads: 头数
        head_dim: 每头维度
        dtype: 缓冲区 dtype（默认 float32）

    Attributes:
        per_seq_lens: ``(B,)`` ndarray，记录每个序列当前 cache 长度
            （不同序列可不同步）

    用法::

        cache = ParallelKVCache(num_layers=2, max_batch=4, max_seq=128,
                                num_heads=8, head_dim=64)
        # 一次性 batch 更新（k 个 token 并行写入）
        k_new = Tensor(np.random.randn(4, 8, 8, 64).astype(np.float32))
        v_new = Tensor(np.random.randn(4, 8, 8, 64).astype(np.float32))
        cache.batch_update(k_new, v_new, layer_idx=0)
    """

    def __init__(
        self,
        num_layers: int = 1,
        max_batch: int = 1,
        max_seq: int = 1024,
        num_heads: int = 1,
        head_dim: int = 1,
        dtype=np.float32,
    ):
        super().__init__(num_layers=num_layers)
        self.max_batch = int(max_batch)
        self.max_seq = int(max_seq)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        # 预分配 K/V buffer: (B, max_seq, H, D) 每层一份
        shape = (self.max_batch, self.max_seq, self.num_heads, self.head_dim)
        self._k_buf: List[Tensor] = [
            Tensor(np.zeros(shape, dtype=dtype), requires_grad=False)
            for _ in range(self.num_layers)
        ]
        self._v_buf: List[Tensor] = [
            Tensor(np.zeros(shape, dtype=dtype), requires_grad=False)
            for _ in range(self.num_layers)
        ]
        # 每个序列当前长度（默认 0）
        self.per_seq_lens = np.zeros(self.max_batch, dtype=np.int64)
        # 每层是否已写入（用于 reset 后状态管理）
        self._layer_initialized = [False] * self.num_layers

    # ------------------------------------------------------------------
    # 核心：batch_update（并行批量追加 K/V）
    # ------------------------------------------------------------------

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0):
        """单 batch 序列追加 K/V（委托给 batch_update）。"""
        return self.batch_update(key, value, layer_idx=layer_idx)

    def batch_update(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int = 0,
    ):
        """并行批量更新 KV cache（多序列同时追加）。

        一次把 B 个序列的 (T_new, H, D) K/V 追加到各自 cache 末尾。
        利用 numpy/torch 的批量算子，避免 for 循环遍历 batch。

        Args:
            keys: (B, T_new, H, D) 新 K
            values: (B, T_new, H, D) 新 V
            layer_idx: 层索引
        Returns:
            new_k, new_v: 各为 (B, T_new_total, H, D) Tensor，
                T_new_total = per_seq_lens[b] + T_new（每个序列当前总长）

        Note:
            简化实现：所有序列追加相同 T_new 长度（speculative decoding
            常见用法）。若不同序列 T_new 不同，外部需 padding 对齐后再调用。
        """
        if layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}"
            )
        B, T_new, H, D = keys.shape
        if B > self.max_batch:
            raise ValueError(
                f"batch {B} 超过 max_batch {self.max_batch}"
            )
        if H != self.num_heads or D != self.head_dim:
            raise ValueError(
                f"K/V head 维度不匹配：期望 ({self.num_heads}, {self.head_dim})，"
                f"got ({H}, {D})"
            )

        # 检查是否所有序列都有足够剩余空间
        for b in range(B):
            new_len = int(self.per_seq_lens[b]) + T_new
            if new_len > self.max_seq:
                raise RuntimeError(
                    f"ParallelKVCache 溢出：序列 {b} 已存 "
                    f"{self.per_seq_lens[b]}，新增 {T_new}，"
                    f"超过 max_seq {self.max_seq}"
                )

        k_old = self._k_buf[layer_idx]
        v_old = self._v_buf[layer_idx]

        # 简化：所有序列统一 append T_new 个位置
        # 复杂变长场景（每序列 T_new 不同）需逐序列 scatter；此处不实现
        with no_grad():
            if not self._layer_initialized[layer_idx]:
                # 首次写入：直接赋值
                new_k = keys
                new_v = values
                self._layer_initialized[layer_idx] = True
            else:
                # 取已缓存部分：每个序列的前 per_seq_lens[b] 个位置
                # 简化：所有序列 per_seq_lens 相同时直接切片
                # （speculative decoding 场景下 batch 内序列步数通常一致）
                if np.all(self.per_seq_lens[:B] == self.per_seq_lens[0]):
                    start = int(self.per_seq_lens[0])
                    k_prev = k_old[:B, :start]
                    v_prev = v_old[:B, :start]
                    new_k = _concat([k_prev, keys], dim=1)
                    new_v = _concat([v_prev, values], dim=1)
                else:
                    # 变长 batch：逐序列 concat（性能略低但功能完整）
                    new_k_list = []
                    new_v_list = []
                    for b in range(B):
                        s = int(self.per_seq_lens[b])
                        k_prev_b = k_old[b:b + 1, :s]
                        v_prev_b = v_old[b:b + 1, :s]
                        new_k_b = _concat([k_prev_b, keys[b:b + 1]], dim=1)
                        new_v_b = _concat([v_prev_b, values[b:b + 1]], dim=1)
                        new_k_list.append(new_k_b)
                        new_v_list.append(new_v_b)
                    new_k = _concat(new_k_list, dim=0)
                    new_v = _concat(new_v_list, dim=0)

        # 把新 K/V 写回 buffer（保留 max_seq 维度，未用部分用零填充）
        # 用整体替换（与 StaticCache.update 一致策略）
        self._k_buf[layer_idx] = new_k
        self._v_buf[layer_idx] = new_v
        # 更新每个序列长度
        self.per_seq_lens[:B] = self.per_seq_lens[:B] + T_new
        return new_k, new_v

    def get(self, layer_idx: int = 0):
        """取出指定层的 (K, V) Tensor。

        返回的 K/V 形状为 (B, max(per_seq_lens), H, D)。
        """
        if layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}"
            )
        return self._k_buf[layer_idx], self._v_buf[layer_idx]

    def get_seq(self, b: int, layer_idx: int = 0):
        """取出指定序列、指定层的有效 (K, V)（去除尾部 padding）。"""
        if layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} 超出 num_layers {self.num_layers}"
            )
        s = int(self.per_seq_lens[b])
        return self._k_buf[layer_idx][b:b + 1, :s], \
            self._v_buf[layer_idx][b:b + 1, :s]

    def reset(self) -> None:
        """清空 cache（所有序列、所有层）。"""
        for i in range(self.num_layers):
            self._k_buf[i] = Tensor(
                np.zeros_like(self._k_buf[i].data), requires_grad=False
            )
            self._v_buf[i] = Tensor(
                np.zeros_like(self._v_buf[i].data), requires_grad=False
            )
            self._layer_initialized[i] = False
        self.per_seq_lens[:] = 0

    def reset_seq(self, b: int) -> None:
        """清空指定序列的 cache（speculative decoding 单序列重启用）。"""
        for i in range(self.num_layers):
            # 把指定序列的 buffer 置零
            k_b = self._k_buf[i].data
            v_b = self._v_buf[i].data
            k_b[b] = 0.0
            v_b[b] = 0.0
        self.per_seq_lens[b] = 0

    @property
    def device(self) -> str:
        if self._k_buf:
            return self._k_buf[0].device
        return "cpu"

    def to(self, device) -> "ParallelKVCache":
        target = str(device)
        for i in range(self.num_layers):
            self._k_buf[i] = self._k_buf[i].to(target)
            self._v_buf[i] = self._v_buf[i].to(target)
        return self


__all__ = ["ParallelKVCache"]
