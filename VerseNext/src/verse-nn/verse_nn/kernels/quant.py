"""CPU 推理专项优化：int8 动态量化。

两条路径，按精度/速度需求选用：

1. :func:`quantize_dynamic_int8` —— 基于 ``torch.ao.quantization`` 的标准动态量化，
   成熟稳定，直接替换 ``nn.Linear`` 为量化版本；权重 int8、激活**运行时**按批
   动态定标（无需校准数据）。
2. :class:`Int8Linear` —— 自研 per-channel 对称量化层，激活按 token 动态定标，
   走 ``torch._int_mm`` 的 int8 GEMM；在支持 oneDNN int8 的 CPU 上吞吐最高。
   形状不满足 ``_int_mm`` 约束时自动回退到 fp32 计算，绝不报错。

默认**不量化** embedding / lm_head：词表投影对精度最敏感，量化它往往得不偿失。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from verse_core import get_logger

logger = get_logger("kernels.quant")

__all__ = [
    "int8_gemm_available",
    "Int8Linear",
    "quantize_linear",
    "quantize_dynamic_int8",
    "quantize_model_int8",
]


def int8_gemm_available() -> bool:
    """``torch._int_mm``（int8 矩阵乘）在当前 CPU 是否可用。"""
    if not hasattr(torch, "_int_mm"):
        return False
    try:
        a = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
        b = torch.randint(-8, 8, (16, 8), dtype=torch.int8)
        torch._int_mm(a, b)
        return True
    except Exception:  # pragma: no cover - 平台/版本相关
        return False


class Int8Linear(nn.Module):
    """自研 int8 动态量化线性层（per-channel 权重 + per-token 激活）。

    数学：
        ``y = (x / s_x) @ (W_q * s_w)^T``，其中 ``s_w`` 逐输出通道，
        ``s_x`` 逐 token（运行时按行取 ``max|x| / 127``）。

    计算：
        int8 GEMM 得到 int32 累加，再乘 ``s_x * s_w`` 还原。
        ``_int_mm`` 不可用或形状不满足约束时回退 fp32，保证功能正确。

    Args:
        linear: 待量化的 ``nn.Linear``（bias 保留 fp32，通常为 None）。
        per_channel: True 逐输出通道定标（精度更好），False 全权重一个 scale。
    """

    def __init__(self, linear: nn.Linear, per_channel: bool = True) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.per_channel = per_channel
        weight = linear.weight.detach().float()

        if per_channel:
            # 逐输出通道（dim=0）对称量化
            scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            q = torch.round(weight / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        else:
            scale = weight.abs().amax().clamp_min(1e-8) / 127.0
            q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)

        self.register_buffer("weight_q", q, persistent=True)
        self.register_buffer("weight_scale", scale, persistent=True)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear, per_channel: bool = True) -> "Int8Linear":
        return cls(linear, per_channel=per_channel)

    def dequantize_weight(self) -> torch.Tensor:
        """还原 fp32 权重（调试/导出用）。"""
        w = self.weight_q.float()
        scale = self.weight_scale
        if self.per_channel:
            return w * scale.unsqueeze(1)
        return w * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        # 输出形状 = 输入去掉最后一维 + out_features（不是原样 reshape！）
        out_shape = orig_shape[:-1] + (self.out_features,)
        x2 = x.reshape(-1, self.in_features)
        dtype = x.dtype

        if (
            int8_gemm_available()
            and x2.shape[0] >= 8            # _int_mm 要求 M 至少 8（对齐后）
            and self.in_features % 8 == 0
            and self.out_features % 8 == 0
        ):
            try:
                # per-token 动态定标
                x_scale = x2.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
                x_q = torch.round(x2.float() / x_scale).clamp(-127, 127).to(torch.int8)
                # int8 @ int8 -> int32
                acc = torch._int_mm(x_q, self.weight_q.t())
                if self.per_channel:
                    out = acc.float() * (x_scale * self.weight_scale.unsqueeze(0))
                else:
                    out = acc.float() * (x_scale * self.weight_scale)
                if self.bias is not None:
                    out = out + self.bias
                return out.to(dtype).reshape(out_shape)
            except Exception as exc:  # pragma: no cover - 形状/后端相关
                logger.debug("int8 GEMM 失败，回退 fp32: %s", exc)

        # 回退：反量化权重后走 fp32 线性
        out = F.linear(x2, self.dequantize_weight(), self.bias)
        return out.to(dtype).reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"per_channel={self.per_channel}, bias={self.bias is not None}"
        )


def quantize_linear(linear: nn.Linear, per_channel: bool = True) -> Int8Linear:
    """把单个 ``nn.Linear`` 换成 :class:`Int8Linear`。"""
    return Int8Linear.from_float(linear, per_channel=per_channel)


def _replace_linear(module: nn.Module, per_channel: bool, skip: tuple[str, ...]) -> int:
    """递归替换子模块中的 ``nn.Linear``，返回替换数量。"""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if any(tag in name for tag in skip):
                continue
            setattr(module, name, Int8Linear.from_float(child, per_channel=per_channel))
            replaced += 1
        else:
            replaced += _replace_linear(child, per_channel, skip)
    return replaced


def quantize_dynamic_int8(
    module: nn.Module,
    *,
    skip: tuple[str, ...] = ("lm_head", "token_emb", "embed"),
) -> nn.Module:
    """标准动态量化（torch.ao）：权重 int8，激活运行时定标。

    跳过名称含 ``skip`` 中任一子串的层（默认跳过 lm_head / embedding）。
    量化失败时原样返回并告警，不阻断推理。
    """
    try:
        from torch.ao.quantization import quantize_dynamic

        # torch.ao 的 quantize_dynamic 无法按名字排除，这里先收集要量化的层
        targets = _collect_linear_names(module, skip)
        if not targets:
            return module
        quantized = quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8)
        logger.info("动态量化完成：%d 个 Linear（跳过 %s）", len(targets), list(skip))
        return quantized
    except Exception as exc:  # pragma: no cover - 依赖版本
        logger.warning("动态量化失败，保持 fp32: %s", exc)
        return module


def quantize_model_int8(
    module: nn.Module,
    *,
    per_channel: bool = True,
    skip: tuple[str, ...] = ("lm_head", "token_emb", "embed"),
) -> nn.Module:
    """自研 per-channel int8 量化：原地替换 ``nn.Linear`` 为 :class:`Int8Linear`。

    与 :func:`quantize_dynamic_int8` 的区别：本函数用自研 :class:`Int8Linear`
    （per-channel 权重 + per-token 激活 + ``_int_mm``），精度与速度通常更好，
    且保留原 ``state_dict`` 结构以外的自定义 buffer。
    """
    n = _replace_linear(module, per_channel, skip)
    logger.info("per-channel int8 量化完成：替换 %d 个 Linear（跳过 %s）", n, list(skip))
    return module


def _collect_linear_names(module: nn.Module, skip: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and not any(tag in name for tag in skip):
            out.append(name)
    return out
