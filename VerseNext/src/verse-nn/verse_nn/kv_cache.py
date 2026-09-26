"""推理缓存：静态预分配 KV Cache 与 KDA 递归状态缓存。

性能要点：
- 静态 KV Cache 一次性预分配 ``(b, h, max_len, d)`` 缓冲，每步原地写入，
  以零拷贝切片视图供注意力读取——替代逐步 ``torch.cat``（后者每步分配
  O(t) 新内存，整个解码过程累计 O(t²) 拷贝与分配压力）。
- KDA（线性注意力）天然无需缓存 K/V 序列：其状态是固定大小的
  ``S ∈ (b, h, d_k, d_v)`` 矩阵，每步原地更新，显存与序列长度无关。
"""

from __future__ import annotations

import torch


class StaticKVCache:
    """单层静态 KV 缓存（可选携带 DSA indexer key 缓冲）。

    用法（每层一个实例，由模型统一管理 pos）::

        cache = StaticKVCache(b, h_kv, head_dim, max_len, device, dtype)
        k_all, v_all = cache.kv(n_new)   # 含新槽位的视图 (b,h,pos+n_new,d)
        cache.write(k_new, v_new)        # 原地写入并前移 pos
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        index_head_dim: int = 0,
    ) -> None:
        self.max_len = max_len
        self.pos = 0
        self.k = torch.zeros(batch_size, num_kv_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.idx_k = (
            torch.zeros(batch_size, num_kv_heads, max_len, index_head_dim, device=device, dtype=dtype)
            if index_head_dim > 0
            else None
        )

    def reset(self) -> None:
        self.pos = 0

    def kv(self, n_new: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回包含 ``n_new`` 个新槽位的 (k, v) 连续视图。"""
        end = self.pos + n_new
        if end > self.max_len:
            raise RuntimeError(f"KV 缓存溢出: {end} > max_len({self.max_len})")
        return self.k[:, :, :end], self.v[:, :, :end]

    def write(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        n = k_new.shape[2]
        self.k[:, :, self.pos:self.pos + n].copy_(k_new)
        self.v[:, :, self.pos:self.pos + n].copy_(v_new)
        self.pos += n

    def write_idx_k(self, idx_k_new: torch.Tensor) -> None:
        if self.idx_k is None:
            raise RuntimeError("该缓存未启用 indexer 缓冲")
        n = idx_k_new.shape[2]
        self.idx_k[:, :, self.pos:self.pos + n].copy_(idx_k_new)

    def idx_k_view(self, n_new: int) -> torch.Tensor:
        end = self.pos + n_new
        return self.idx_k[:, :, :end]

    @property
    def memory_bytes(self) -> int:
        n = self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
        if self.idx_k is not None:
            n += self.idx_k.numel() * self.idx_k.element_size()
        return n


class RecurrentStateCache:
    """KDA 递归状态缓存：每层固定大小 S (b, h, d_k, d_v)，与序列长度无关。"""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.state = torch.zeros(
            batch_size, num_heads, key_dim, value_dim, device=device, dtype=dtype
        )
        self.pos = 0

    def reset(self) -> None:
        self.state.zero_()
        self.pos = 0

    @property
    def memory_bytes(self) -> int:
        return self.state.numel() * self.state.element_size()


def create_caches(model, batch_size: int, max_len: int, index_head_dim: int = 0) -> list:
    """按模型每层的注意力类型创建缓存对象列表。"""
    from verse_nn.attention import CausalSelfAttention, DSAttention, KDAAttention

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    caches = []
    for layer in model.layers:
        attn = layer.attn
        if isinstance(attn, KDAAttention):
            caches.append(RecurrentStateCache(
                batch_size, attn.num_heads, attn.head_dim, attn.head_dim, device, dtype
            ))
        elif isinstance(attn, DSAttention):
            caches.append(StaticKVCache(
                batch_size, attn.num_kv_heads, attn.head_dim, max_len, device, dtype,
                index_head_dim=attn.index_head_dim,
            ))
        else:
            caches.append(StaticKVCache(
                batch_size, attn.num_kv_heads, attn.head_dim, max_len, device, dtype
            ))
    return caches
