"""VerseTorch: CPU 量化与加速模块（阶段 2，Task 2.1-2.4）.

提供三种权重量化方案：
- INT8 对称量化（per-channel，参考 llama.cpp Q8_0 思路简化版）
- INT4 (W4A16) 对称量化（per-channel，2 个 int4 打包成 1 个 uint8）
- 1.58-bit ternary 量化（BitNet b1.58 风格 {-1, 0, +1}，2 bit 打包）

每种方案都提供：
- `quantize_*`：把 fp32 权重量化为紧凑 packed 形式 + scale
- `dequantize_*`：解包并反量化回 fp32
- `matmul_*`：fused 反量化-GEMM，直接用 packed 权重计算 `x @ dequantize(W).T`，
  避免在调用方物化完整 fp32 权重矩阵

参考实现：
- BitNet.cpp（1.58-bit ternary {-1, 0, +1}）
- llama.cpp（GGUF Q4_K / Q8_0）
- lm.c（纯 C 推理引擎）

约束：仅使用 NumPy + 标准库；Numba 可用时可选加速，不可用则降级为纯 NumPy。
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor
from . import nn

# 可选 Numba 加速（不可用则降级为纯 NumPy）
try:
    import numba as _numba  # noqa: F401
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# 内部辅助：INT4 打包 / 解包
# ---------------------------------------------------------------------------


def _pack_int4(q_int8: np.ndarray) -> np.ndarray:
    """把 int8 数组（每个值在 [-8, 7]）打包成 uint8，每字节存 2 个值。

    打包原理：
    - 4 bit 可表示 16 个值，对称量化后值域 [-8, 7] 共 16 个，恰好 fit。
    - 把有符号 int4 转成无符号 nibble：直接取低 4 位（二进制补码表示），
      即 ``v & 0x0F``：-8→8, -1→15, 0→0, 7→7。
    - 两个 nibble 拼成 1 字节：低 4 位存偶数下标值，高 4 位存奇数下标值。
    - 若最后一维长度为奇数，末尾补 0 凑偶。

    参数:
        q_int8: int8 数组，1D 或 2D，值域 [-8, 7]。
    返回:
        uint8 数组，最后一维长度为 ceil(in/2)。
    """
    if q_int8.ndim == 1:
        n = q_int8.size
        if n % 2 != 0:
            q_int8 = np.concatenate([q_int8, np.zeros(1, dtype=q_int8.dtype)])
        # 取低 4 位作为无符号 nibble（二进制补码）
        low = (q_int8[0::2].astype(np.uint8)) & 0x0F
        high = (q_int8[1::2].astype(np.uint8)) & 0x0F
        return (high << 4) | low
    # 2D：沿最后一维打包
    rows, cols = q_int8.shape
    if cols % 2 != 0:
        q_int8 = np.concatenate(
            [q_int8, np.zeros((rows, 1), dtype=q_int8.dtype)], axis=1
        )
        cols += 1
    low = (q_int8[:, 0::2].astype(np.uint8)) & 0x0F
    high = (q_int8[:, 1::2].astype(np.uint8)) & 0x0F
    return (high << 4) | low


def _unpack_int4(packed: np.ndarray, original_shape) -> np.ndarray:
    """解包 uint8 数组（每字节 2 个 int4）回 int8 数组。

    解包原理：
    - 低 4 位 → 偶数下标的值，高 4 位 → 奇数下标的值。
    - 把无符号 nibble 转回有符号：若 nibble >= 8，则减 16（二进制补码还原）。
    - 截断到 original_shape 的实际长度（去掉打包时的尾部 padding）。
    """
    if len(original_shape) == 1:
        n = int(original_shape[0])
        flat = packed.ravel().astype(np.uint8)
        low = (flat & 0x0F).astype(np.int16)
        high = ((flat >> 4) & 0x0F).astype(np.int16)
        # 无符号 nibble → 有符号 int4
        low = np.where(low >= 8, low - 16, low).astype(np.int8)
        high = np.where(high >= 8, high - 16, high).astype(np.int8)
        out = np.empty(flat.size * 2, dtype=np.int8)
        out[0::2] = low
        out[1::2] = high
        return out[:n]
    out_dim, in_dim = original_shape
    low = (packed & 0x0F).astype(np.int16)
    high = ((packed >> 4) & 0x0F).astype(np.int16)
    low = np.where(low >= 8, low - 16, low).astype(np.int8)
    high = np.where(high >= 8, high - 16, high).astype(np.int8)
    out = np.empty((out_dim, packed.shape[1] * 2), dtype=np.int8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :in_dim]


# ---------------------------------------------------------------------------
# 内部辅助：ternary 打包 / 解包
# ---------------------------------------------------------------------------


def _pack_ternary(q_int8: np.ndarray) -> np.ndarray:
    """把 ternary 值 {-1, 0, +1}（int8）打包成 uint8，每字节存 4 个值。

    打包原理：
    - 2 bit 可表示 4 个状态，ternary 只用 3 个 {-1, 0, +1}，编码为 {0, 1, 2}。
      编码方式：``code = value + 1``（-1→00, 0→01, +1→10，11 未用）。
    - 4 个 2-bit code 拼成 1 字节：
        bit 0-1: 第 0 个值
        bit 2-3: 第 1 个值
        bit 4-5: 第 2 个值
        bit 6-7: 第 3 个值
    - 若最后一维长度不是 4 的倍数，末尾补 0。
    """
    codes = (q_int8.astype(np.int16) + 1).astype(np.uint8)  # 0, 1, 2
    if codes.ndim == 1:
        n = codes.size
        pad = (-n) % 4
        if pad:
            codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
        c0 = codes[0::4]
        c1 = codes[1::4]
        c2 = codes[2::4]
        c3 = codes[3::4]
        return (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).astype(np.uint8)
    rows, cols = codes.shape
    pad = (-cols) % 4
    if pad:
        codes = np.concatenate(
            [codes, np.zeros((rows, pad), dtype=np.uint8)], axis=1
        )
    c0 = codes[:, 0::4]
    c1 = codes[:, 1::4]
    c2 = codes[:, 2::4]
    c3 = codes[:, 3::4]
    return (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).astype(np.uint8)


def _unpack_ternary(packed: np.ndarray, original_shape) -> np.ndarray:
    """解包 uint8 数组（每字节 4 个 ternary）回 int8 {-1, 0, +1}。

    解包原理：按 2-bit 分组提取 code，再 ``value = code - 1`` 还原 {-1, 0, +1}。
    """
    if len(original_shape) == 1:
        n = int(original_shape[0])
        flat = packed.ravel().astype(np.uint8)
        c0 = (flat & 0b11).astype(np.int8)
        c1 = ((flat >> 2) & 0b11).astype(np.int8)
        c2 = ((flat >> 4) & 0b11).astype(np.int8)
        c3 = ((flat >> 6) & 0b11).astype(np.int8)
        out = np.empty(flat.size * 4, dtype=np.int8)
        out[0::4] = c0 - 1
        out[1::4] = c1 - 1
        out[2::4] = c2 - 1
        out[3::4] = c3 - 1
        return out[:n]
    out_dim, in_dim = original_shape
    c0 = (packed & 0b11).astype(np.int8)
    c1 = ((packed >> 2) & 0b11).astype(np.int8)
    c2 = ((packed >> 4) & 0b11).astype(np.int8)
    c3 = ((packed >> 6) & 0b11).astype(np.int8)
    out = np.empty((out_dim, packed.shape[1] * 4), dtype=np.int8)
    out[:, 0::4] = c0 - 1
    out[:, 1::4] = c1 - 1
    out[:, 2::4] = c2 - 1
    out[:, 3::4] = c3 - 1
    return out[:, :in_dim]


def _scale_to_vec(scale: np.ndarray) -> np.ndarray:
    """把 scale 统一成可广播到 matmul 输出最后一维的 1D 向量（或标量）。"""
    if scale.ndim == 0:
        return scale
    return scale.reshape(-1)


# ---------------------------------------------------------------------------
# Task 2.1: INT8 对称量化 / 反量化
# ---------------------------------------------------------------------------


def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """INT8 对称量化。

    - 对 2D 权重 ``(out, in)`` 做 per-output-channel 量化：scale shape ``(out, 1)``。
    - 对 1D 权重做 per-tensor 量化：scale 为标量。
    - 公式：``scale = max(|w|) / 127``，``q = round(w / scale).clip(-127, 127)``。

    参数:
        w: float32 权重数组。
    返回:
        (q, scale)：q 是 int8 数组，scale 是 float32 标量或 (out, 1) 数组。
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 1:
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        if max_abs == 0.0:
            scale = np.float32(1.0)
            q = np.zeros_like(w, dtype=np.int8)
        else:
            scale = np.float32(max_abs / 127.0)
            q = np.round(w / scale).clip(-127, 127).astype(np.int8)
        return q, scale
    # 2D：per-output-channel（沿最后一维求 max，keepdims）
    max_abs = np.max(np.abs(w), axis=-1, keepdims=True)  # (out, 1)
    safe_max = np.where(max_abs > 0, max_abs, 1.0)
    scale = (safe_max / 127.0).astype(np.float32)
    q = np.round(w / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """INT8 反量化：``w ≈ q * scale``。

    支持标量 scale（per-tensor）与 (out, 1) per-channel scale（自动广播）。
    """
    q = q.astype(np.float32)
    return q * scale


# ---------------------------------------------------------------------------
# Task 2.2: INT4 (W4A16) 权重量化 + fused 反量化-GEMM
# ---------------------------------------------------------------------------


def quantize_int4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """INT4 (W4A16) 对称量化。

    - 对 2D 权重 ``(out, in)`` 做 per-output-channel 量化：scale shape ``(out, 1)``。
    - 对 1D 权重做 per-tensor 量化：scale 为标量。
    - 公式：``scale = max(|w|) / 7``，``q = round(w / scale).clip(-8, 7)``。
    - 打包：2 个 int4 → 1 个 uint8（高 4 位 + 低 4 位）。

    参数:
        w: float32 权重数组。
    返回:
        (packed_uint8, scale)：packed 沿最后一维长度为 ceil(in/2)。
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 1:
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        if max_abs == 0.0:
            scale = np.float32(1.0)
            q = np.zeros_like(w, dtype=np.int8)
        else:
            scale = np.float32(max_abs / 7.0)
            q = np.round(w / scale).clip(-8, 7).astype(np.int8)
        return _pack_int4(q), scale
    max_abs = np.max(np.abs(w), axis=-1, keepdims=True)
    safe_max = np.where(max_abs > 0, max_abs, 1.0)
    scale = (safe_max / 7.0).astype(np.float32)
    q = np.round(w / scale).clip(-8, 7).astype(np.int8)
    return _pack_int4(q), scale


def dequantize_int4(
    packed: np.ndarray, scale: np.ndarray, original_shape
) -> np.ndarray:
    """INT4 解包 + 反量化。

    参数:
        packed: uint8 打包数组。
        scale: 标量或 (out, 1) per-channel scale。
        original_shape: 原始权重 shape（用于截断 padding）。
    返回:
        float32 反量化权重。
    """
    q_int8 = _unpack_int4(packed, original_shape)
    return q_int8.astype(np.float32) * scale


def matmul_int4(
    x: np.ndarray, packed_w: np.ndarray, scale: np.ndarray, w_shape
) -> np.ndarray:
    """fused: ``x @ dequantize(packed_w, scale).T``。

    内部一次性解包 int4 → int8 → 临时 fp32，完成 matmul 后立即释放，
    避免在调用方持有完整 fp32 权重矩阵。等价于：
        ``y = x @ (dequantize_W).T``，其中 ``dequantize_W = unpack(packed_w) * scale``。

    参数:
        x: 输入，shape ``(..., in_dim)``，float32 或可转换类型。
        packed_w: 打包的 int4 权重，shape ``(out_dim, ceil(in_dim/2))``。
        scale: 标量或 ``(out_dim, 1)`` per-channel scale。
        w_shape: 原始权重 shape ``(out_dim, in_dim)``。
    返回:
        ``(..., out_dim)`` float32。
    """
    # 1) 解包 int4 → int8（仅为原权重大小的 1/4，比 fp32 小 4 倍）
    q_int8 = _unpack_int4(packed_w, w_shape)  # (out, in), int8
    # 2) cast 到 fp32（临时数组，函数返回后 GC；不在调用方持久化）
    q_f32 = q_int8.astype(np.float32)
    # 3) 输入展平到 2D 做一次大 GEMM
    x_arr = np.asarray(x)
    x_f32 = x_arr.astype(np.float32, copy=False)
    orig_shape = x_f32.shape
    x_2d = x_f32.reshape(-1, orig_shape[-1])
    # x_2d (N, in) @ q_f32.T (in, out) -> (N, out)
    out_2d = x_2d @ q_f32.T
    # 4) per-channel scale：每列乘以 scale[o]（in-place 避免额外分配）
    scale_vec = _scale_to_vec(scale)
    out_2d *= scale_vec
    # 5) reshape 回原始 batch 形状
    return out_2d.reshape(*orig_shape[:-1], w_shape[0])


# ---------------------------------------------------------------------------
# Task 2.3: 1.58-bit ternary 量化（BitNet 风格）
# ---------------------------------------------------------------------------


def quantize_ternary(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1.58-bit ternary 量化（BitNet b1.58 风格）。

    - ``w ≈ {-1, 0, +1} * scale``
    - scale 公式（BitNet b1.58）：``scale = mean(|w|) / 0.5 = 2 * mean(|w|)``
      对 2D 权重做 per-output-channel，1D 做 per-tensor。
    - 量化：``w_q = round(w / scale)``，clip 到 ``{-1, 0, +1}``。
    - 打包：2 bit per value，4 values per uint8。

    参数:
        w: float32 权重数组。
    返回:
        (packed_uint8, scale)。
    """
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 1:
        mean_abs = float(np.mean(np.abs(w))) if w.size else 0.0
        if mean_abs == 0.0:
            scale = np.float32(1.0)
            q = np.zeros_like(w, dtype=np.int8)
        else:
            # scale = mean(|w|) / 0.5
            scale = np.float32(mean_abs / 0.5)
            q = np.round(w / scale).clip(-1, 1).astype(np.int8)
        return _pack_ternary(q), scale
    # 2D per-output-channel
    mean_abs = np.mean(np.abs(w), axis=-1, keepdims=True)  # (out, 1)
    safe_mean = np.where(mean_abs > 0, mean_abs, 1.0)
    scale = (safe_mean / 0.5).astype(np.float32)
    q = np.round(w / scale).clip(-1, 1).astype(np.int8)
    return _pack_ternary(q), scale


def dequantize_ternary(
    packed: np.ndarray, scale: np.ndarray, original_shape
) -> np.ndarray:
    """ternary 解包 + 反量化。"""
    q_int8 = _unpack_ternary(packed, original_shape)
    return q_int8.astype(np.float32) * scale


def matmul_ternary(
    x: np.ndarray, packed_w: np.ndarray, scale: np.ndarray, w_shape
) -> np.ndarray:
    """fused: ``x @ dequantize(packed_w, scale).T``。

    与 ``matmul_int4`` 同构：解包 ternary → int8 → 临时 fp32，matmul，scale。
    """
    q_int8 = _unpack_ternary(packed_w, w_shape)
    q_f32 = q_int8.astype(np.float32)
    x_arr = np.asarray(x)
    x_f32 = x_arr.astype(np.float32, copy=False)
    orig_shape = x_f32.shape
    x_2d = x_f32.reshape(-1, orig_shape[-1])
    out_2d = x_2d @ q_f32.T
    scale_vec = _scale_to_vec(scale)
    out_2d *= scale_vec
    return out_2d.reshape(*orig_shape[:-1], w_shape[0])


# ---------------------------------------------------------------------------
# Task 2.4: QuantizedLinear 层
# ---------------------------------------------------------------------------


class QuantizedLinear:
    """可热替换 ``nn.Linear`` 的量化版本（推理专用，不支持训练）。

    构造时接受一个 ``nn.Linear``（或权重 + bias），执行权重量化并存储 packed 形式。
    forward 时使用 fused matmul（``matmul_int4`` / ``matmul_ternary`` / int8 路径）。

    API 与 ``nn.Linear.forward(x)`` 完全兼容：``y = x @ W.T + b``，W shape ``(out, in)``。
    x 可以是 ``verse_torch.Tensor`` 或 ``np.ndarray``；forward 内部转 np.ndarray 处理，
    输出保持输入类型。
    """

    def __init__(
        self,
        linear: "nn.Linear",
        qtype: str = "int4",
        cache_fp32: bool = True,
    ):
        """
        参数:
            linear: 一个已初始化的 ``nn.Linear`` 实例。
            qtype: 量化类型，``"int8"`` / ``"int4"`` / ``"ternary"``。
            cache_fp32: 是否在 load-time 一次性反量化并缓存 fp32 转置权重。

                - ``True``（默认，推荐推理场景）：构造时一次性完成
                  ``unpack → cast fp32 → * scale → transpose``，得到 contiguous
                  ``(in, out)`` fp32 矩阵；forward 仅做一次 ``x @ W_T`` + bias。
                  相比纯 NumPy 路径下每次 forward 的 astype + in-place scale 开销，
                  此路径加速稳定，且让 BLAS 直接处理 contiguous fp32 GEMM。
                  代价：内存占用回到 fp32 大小（但 ``self.packed`` 仍为量化形式，
                  可用于统计模型大小/持久化）。
                - ``False``：每次 forward 走 fused 反量化-GEMM 路径
                  （cast int8→fp32、matmul、in-place scale），内存占用为 int8
                  （约为 fp32 的 1/4），但每次 forward 有额外 astype + scale 开销。
        """
        if qtype not in ("int8", "int4", "ternary"):
            raise ValueError(f"Unknown qtype: {qtype!r}, expected int8/int4/ternary")
        self.qtype = qtype
        self.cache_fp32 = bool(cache_fp32)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        # 取出权重（copy 避免 alias 原始 Tensor 的 data）
        w = np.array(linear.weight.data, dtype=np.float32, copy=True)
        self.w_shape = w.shape  # (out, in)
        # 量化并存储 packed + scale（持久化与模型大小统计依据）
        if qtype == "int8":
            self.packed, self.scale = quantize_int8(w)
        elif qtype == "int4":
            self.packed, self.scale = quantize_int4(w)
        else:  # ternary
            self.packed, self.scale = quantize_ternary(w)
        # 复制 bias（如有）
        if linear.bias is not None:
            self.bias = np.array(linear.bias.data, dtype=np.float32, copy=True)
        else:
            self.bias = None
        # 预计算 1D scale 向量（便于广播到 matmul 输出最后一维）
        self._scale_vec = _scale_to_vec(self.scale)

        if self.cache_fp32:
            # load-time 一次性反量化：unpack → fp32 → * scale → contiguous transpose
            # 内存：与原 fp32 权重同等大小（约 1MB / 512x512），换取 forward 稳定加速
            if qtype == "int4":
                q_int8 = _unpack_int4(self.packed, self.w_shape)
            elif qtype == "ternary":
                q_int8 = _unpack_ternary(self.packed, self.w_shape)
            else:  # int8
                q_int8 = self.packed
            w_fp32 = q_int8.astype(np.float32) * self.scale  # (out, in) fp32
            # contiguous 转置：(in, out)，BLAS 直接走 transB 路径
            self._w_fp32_T = np.ascontiguousarray(w_fp32.T)
            # 释放中间 int8 缓存，节省内存
            self._q_int8_T = None
        else:
            # 缓存：解包后的 int8 权重（仍比 fp32 小 4×，避免每次 forward 重复解包）
            # 仅 int4/ternary 需要解包；int8 的 packed 已经是 int8
            if qtype == "int4":
                self._q_int8 = _unpack_int4(self.packed, self.w_shape)
            elif qtype == "ternary":
                self._q_int8 = _unpack_ternary(self.packed, self.w_shape)
            else:  # int8
                self._q_int8 = self.packed
            # 预计算 contiguous 转置 int8（加速 matmul：x @ q.T 时 BLAS 可直接用 transB）
            self._q_int8_T = np.ascontiguousarray(self._q_int8.T)  # (in, out), int8
            self._w_fp32_T = None

    def forward(self, x):
        """前向计算：``y = x @ W.T + b``（W 已量化）。

        参数:
            x: ``verse_torch.Tensor`` 或 ``np.ndarray``，shape ``(..., in_features)``。
        返回:
            与 x 同类型的输出，shape ``(..., out_features)``。
        """
        is_tensor = isinstance(x, Tensor)
        if is_tensor:
            x_np = x.data
        else:
            x_np = np.asarray(x)
        x_np = x_np.astype(np.float32, copy=False)
        orig_shape = x_np.shape
        # 展平到 2D：(N, in_features)
        x_2d = x_np.reshape(-1, orig_shape[-1])
        if self.cache_fp32:
            # 路径 A：load-time 已反量化，forward 只做一次 GEMM + bias
            # ``self._w_fp32_T`` 形状 (in, out) contiguous fp32，BLAS 走最优路径
            out_2d = x_2d @ self._w_fp32_T  # (N, out)
            if self.bias is not None:
                out_2d += self.bias  # in-place
        else:
            # 路径 B：fused 反量化-GEMM，每次 forward cast int8 → fp32 + in-place scale
            q_f32 = self._q_int8_T.astype(np.float32)  # (in, out) 临时 fp32
            out_2d = x_2d @ q_f32  # (N, out)
            out_2d *= self._scale_vec  # in-place
            if self.bias is not None:
                out_2d += self.bias  # in-place
        # reshape 回 batch 形状
        out = out_2d.reshape(*orig_shape[:-1], self.out_features)
        if is_tensor:
            return Tensor(out, requires_grad=False)
        return out

    def __call__(self, x):
        return self.forward(x)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qtype: str = "int4",
        cache_fp32: bool = True,
    ) -> "QuantizedLinear":
        """从 state_dict 加载权重并量化。

        参数:
            state_dict: 字典，需包含 ``"weight"``（shape ``(out, in)``），
                若 ``bias=True`` 还需 ``"bias"``（shape ``(out,)``）。
            in_features: 输入维度。
            out_features: 输出维度。
            bias: 是否有 bias。
            qtype: 量化类型。
            cache_fp32: 是否在 load-time 缓存反量化 fp32 权重（见 ``__init__``）。
        返回:
            ``QuantizedLinear`` 实例。
        """
        if "weight" not in state_dict:
            raise KeyError("state_dict must contain 'weight'")
        # 构造一个临时 Linear，加载权重后交给 __init__ 量化
        lin = nn.Linear(in_features, out_features, bias=bias)
        w = np.asarray(state_dict["weight"], dtype=np.float32)
        if w.shape != (out_features, in_features):
            w = w.reshape(out_features, in_features)
        lin.weight.data = w
        if bias:
            if "bias" in state_dict and state_dict["bias"] is not None:
                lin.bias.data = np.asarray(state_dict["bias"], dtype=np.float32)
            else:
                # 用户声明 bias=True 但 state_dict 没有 bias，保持 Linear 默认初始化的 bias
                pass
        return cls(lin, qtype=qtype, cache_fp32=cache_fp32)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, qtype={self.qtype}, "
            f"cache_fp32={self.cache_fp32}"
        )

    def __repr__(self) -> str:
        return f"QuantizedLinear({self.extra_repr()})"


__all__ = [
    "quantize_int8",
    "dequantize_int8",
    "quantize_int4",
    "dequantize_int4",
    "matmul_int4",
    "quantize_ternary",
    "dequantize_ternary",
    "matmul_ternary",
    "QuantizedLinear",
]
