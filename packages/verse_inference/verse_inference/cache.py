"""Task 5.4.2: StateCache - Mamba/RWKV 的递归状态缓存。

设计目标
--------
自回归生成时，每个 block 都有一个 recurrent state（SSM 状态、KV cache 等），
需要在每个时间步之间保持。``StateCache`` 提供一个统一的容器：

- 初始化为全零状态；
- ``get(layer)`` 取出第 ``layer`` 层的状态（ndarray）；
- ``set(layer, value)`` 写入新状态；
- ``reset()`` 重置为全零。

注意
----
本类是 **辅助工具**，主要用于：
- 用户手动管理递归循环时的状态容器；
- 调试与可视化单层状态。

实际上 ``StreamingGenerator`` 直接调用 ``HybridLM.forward_recurrent``，
后者内部会返回 ``new_states`` 列表，自动管理所有层状态；
``StateCache`` 仅在用户希望「绕过 HybridLM 自己管理状态」时使用。

支持的 state shape
------------------
不同架构的 state 形状不同：

- **Mamba-2**:
  - ssm_state: ``(B, n_heads, d_state, d_head)``
  - conv_state: ``(B, d_conv - 1, d_inner)`` 其中 ``d_inner = expand * dim``
  - 完整 state 是 tuple ``(ssm_state, conv_state)``
- **RWKV-7**:
  - time_mix state: ``(ssm_state, x_prev)`` 其中 ssm_state 为 ``(B, n_head, head_size, head_size)``
  - channel_mix state: ``(x_prev,)``
  - 完整 state 是 tuple ``(time_mix_state, channel_mix_state)``
- **Sparse Attention**:
  - kv_cache: list of (K, V) per past token
  - position: int

由于形状差异巨大，``StateCache`` 采用 **灵活容器** 设计：
- 构造时根据 ``arch`` 决定每层 state 的 shape；
- 内部用 ``list[ndarray]`` 保存（不要求所有层同形状）；
- 用户也可以直接 ``set(layer, anything)`` 存任意对象。

本实现的默认构造遵循规范签名 ``(n_layers, n_heads, head_dim, d_state, batch_size)``，
默认初始化为 Mamba-2 风格的 ssm_state（ ``(B, n_heads, d_state, d_head)``）。
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np


class StateCache:
    """Mamba/RWKV 的递归状态缓存。

    Args:
        n_layers: 层数
        n_heads: 头数（Mamba-2 SSD 头数，或 RWKV-7 time-mix 头数）
        head_dim: 每头维度（Mamba-2: d_head = d_inner // n_heads；RWKV-7: head_size）
        d_state: SSM 状态维度（Mamba-2: 默认 128；RWKV-7: 等于 head_size）
        batch_size: batch 大小（默认 1）
        arch: "mamba2" / "rwkv7" / "hybrid"（决定每层 state 形状；"hybrid" 默认按 mamba2 初始化）

    Attributes:
        states: list[np.ndarray]，长度 = n_layers，每层一个 state（初始化为全零 ndarray）
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        d_state: int,
        batch_size: int = 1,
        arch: str = "mamba2",
    ):
        if n_layers < 0:
            raise ValueError(f"n_layers must be >= 0, got {n_layers}")
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.batch_size = batch_size
        self.arch = arch

        # 初始化全零 state
        self.states: list[Any] = []
        for _ in range(n_layers):
            self.states.append(self._zero_state())

    def _zero_state(self):
        """根据 arch 创建一个零状态。

        - mamba2: (ssm_state, conv_state)
        - rwkv7: ((ssm_state, x_prev), (x_prev,))
        - hybrid: 默认按 mamba2（实际层类型混合时由用户在外部覆盖对应层）

        为简化实现，这里只返回 ssm_state 这个 ndarray（最常见的用例）。
        用户可以 ``set(layer, full_state)`` 写入完整 state（含 conv_state / x_prev 等）。
        """
        if self.arch == "rwkv7":
            # RWKV-7 time-mix ssm_state shape: (B, n_heads, head_dim, head_dim)
            return np.zeros(
                (self.batch_size, self.n_heads, self.head_dim, self.head_dim),
                dtype=np.float32,
            )
        # Mamba-2 ssm_state shape: (B, n_heads, d_state, head_dim)
        return np.zeros(
            (self.batch_size, self.n_heads, self.d_state, self.head_dim),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 主要接口
    # ------------------------------------------------------------------

    def get(self, layer: int):
        """取出第 ``layer`` 层的 state。"""
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"layer {layer} out of range [0, {self.n_layers})")
        return self.states[layer]

    def set(self, layer: int, value) -> None:
        """写入第 ``layer`` 层的 state。

        Args:
            layer: 层索引
            value: ndarray 或 tuple / list（完整 state 结构）
        """
        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"layer {layer} out of range [0, {self.n_layers})")
        # 若 value 是 Tensor，提取 .data
        if hasattr(value, "data") and hasattr(value, "requires_grad"):
            value = value.data
        self.states[layer] = value

    def reset(self) -> None:
        """重置所有层为零状态。"""
        for i in range(self.n_layers):
            self.states[i] = self._zero_state()

    # ------------------------------------------------------------------
    # 便捷：批量设置 / 拷贝
    # ------------------------------------------------------------------

    def set_all(self, new_states) -> None:
        """从 list/tuple 批量覆盖所有层。``new_states`` 长度应等于 ``n_layers``。"""
        if len(new_states) != self.n_layers:
            raise ValueError(
                f"new_states length {len(new_states)} != n_layers {self.n_layers}"
            )
        for i, s in enumerate(new_states):
            self.set(i, s)

    def to_list(self) -> list:
        """返回所有层 state 的浅拷贝列表（用于传给 HybridLM.forward_recurrent）。"""
        return list(self.states)

    def clone(self) -> "StateCache":
        """深拷贝一份。"""
        new = StateCache(
            self.n_layers, self.n_heads, self.head_dim, self.d_state,
            self.batch_size, self.arch,
        )
        new.states = [copy.deepcopy(s) for s in self.states]
        return new

    def __len__(self) -> int:
        return self.n_layers

    def __repr__(self) -> str:
        return (
            f"StateCache(n_layers={self.n_layers}, arch={self.arch!r}, "
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"d_state={self.d_state}, batch_size={self.batch_size})"
        )


__all__ = ["StateCache"]
