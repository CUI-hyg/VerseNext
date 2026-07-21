"""VerseNex: Mo"""VerseNex: MoD（Mixture of Dense Parts）多稠"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD:"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 45"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, Sw"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_D"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：("""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits ="""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent ="""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs ="""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        #"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k -"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None,"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] ="""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-top"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter："""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes ="""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B *"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  #"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bin"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f /"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] ="""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i] = f_i / (B*T) （因为 P_i = mean(probs[:,i])"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i] = f_i / (B*T) （因为 P_i = mean(probs[:,i])）
                grad_p = np.full((B,"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i] = f_i / (B*T) （因为 P_i = mean(probs[:,i])）
                grad_p = np.full((B, T, num_routes),
                                 saved_f / max(saved_B_T, 1),
                                 dtype=np"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i] = f_i / (B*T) （因为 P_i = mean(probs[:,i])）
                grad_p = np.full((B, T, num_routes),
                                 saved_f / max(saved_B_T, 1),
                                 dtype=np.float32) * self.aux_loss_weight
                probs._accumulate_grad(grad_p)
            aux_loss._backward = _bw

        return aux_loss


# ---------------------------------------------------------------------------
# Expert: 单个 Sw"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i] = f_i / (B*T) （因为 P_i = mean(probs[:,i])）
                grad_p = np.full((B, T, num_routes),
                                 saved_f / max(saved_B_T, 1),
                                 dtype=np.float32) * self.aux_loss_weight
                probs._accumulate_grad(grad_p)
            aux_loss._backward = _bw

        return aux_loss


# ---------------------------------------------------------------------------
# Expert: 单个 SwiGLU MLP
# ---------------------------------------------------------------------------


class Expert(Module):
    """单个 Expert（Swi"""VerseNex: MoD（Mixture of Dense Parts）多稠密分区架构。

Part4 核心创新：受人大脑功能分区启发，将模型容量分为 5 个 DensePart
（通用 / 语言 / 数理 / 生化 / 代码），每个 DensePart 下有 8 个 Expert
（SwiGLU MLP），每个 token 通过双层门控路由：
- 第一层（DensePart Router）：soft routing，5 个 DensePart 加权融合
- 第二层（Expert Router）：每个 DensePart 内 top-3 Expert 硬路由

设计要点：
- 双层路由兼顾容量与负载均衡
- aux loss（Switch Transformer 风格）防止路由坍缩
- Expert 复用 SwiGLUMLP，参数预算可控
- 全程 float32，梯度路径完整

参数预算（CometSpark-V0.2, d_model=384, 32 层）：
- 每层 MoD: 5 × 8 Expert × (2 × 384²) ≈ 11.8M
- 32 层 MoD ≈ 378M
- 加 Embedding + Attn + Router ≈ 458M ≈ 0.5B ✓
"""

from __future__ import annotations

import numpy as np

from verse_torch import Tensor
from verse_torch.nn import Linear, SwiGLUMLP, Dropout, Module, ModuleList


# 默认 5 个能力分区（人大脑功能分区隐喻）
DEFAULT_DENSE_PART_NAMES = ["通用", "语言", "数理", "生化", "代码"]


# ---------------------------------------------------------------------------
# Router: Top-k Token 路由器 + Load Balancing Aux Loss
# ---------------------------------------------------------------------------


class Router(Module):
    """Top-k token 路由器，含 Switch Transformer 风格 load balancing aux loss。

    Args:
        dim: 输入维度
        num_routes: 路由目标数（DensePart 数 或 每个 DensePart 内 Expert 数）
        top_k: 每个 token 选择的路由数（必须 <= num_routes）
        aux_loss_weight: aux loss 权重系数
        jitter: 路由 logits 的噪声幅度（探索用，0 表示无噪声）
    """

    def __init__(
        self,
        dim: int,
        num_routes: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        jitter: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_routes, f"top_k({top_k}) 必须 <= num_routes({num_routes})"
        self.dim = dim
        self.num_routes = num_routes
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.jitter = jitter
        # 路由权重矩阵：(num_routes, dim)
        self.w = Linear(dim, num_routes, bias=False)
        # 初始化为小值，避免初期路由极端
        with self._no_grad_ctx():
            self.w.weight.data = (np.random.randn(num_routes, dim).astype(np.float32) * 0.02)

    def _no_grad_ctx(self):
        """避免在 Module 外部 import no_grad 的本地便捷包装。"""
        from verse_torch import no_grad
        return no_grad()

    def forward(self, x):
        """前向计算。

        Args:
            x: (B, T, D)

        Returns:
            indices: (B, T, top_k) int32，每个 token 选中的 route 索引
            weights: (B, T, top_k) float32，softmax 后的路由权重
            aux_loss: scalar Tensor，load balancing 辅助损失
        """
        # logits: (B, T, num_routes)
        logits = self.w(x)
        if self.jitter > 0 and self.training:
            noise = np.random.uniform(-self.jitter, self.jitter, size=logits.data.shape).astype(np.float32)
            logits_data = logits.data + noise
            logits = Tensor(logits_data, requires_grad=logits.requires_grad,
                            _children=(logits,) if logits.requires_grad else (),
                            _op="jitter")
            if logits.requires_grad:
                parent = logits
                def _bw():
                    if parent.requires_grad:
                        parent._accumulate_grad(logits.grad)
                logits._backward = _bw

        # softmax 得到路由概率
        probs = logits.softmax(dim=-1)  # (B, T, num_routes)

        # top-k 选择（用 numpy argpartition 加速，top_k 通常很小）
        probs_np = probs.data  # (B, T, num_routes)
        B, T, R = probs_np.shape
        k = self.top_k
        # argpartition 拿到 top-k 索引（未排序）
        top_idx = np.argpartition(-probs_np, kth=k - 1, axis=-1)[..., :k]  # (B, T, k)
        # 取对应概率值
        batch_idx = np.arange(B)[:, None, None]
        seq_idx = np.arange(T)[None, :, None]
        top_probs = probs_np[batch_idx, seq_idx, top_idx]  # (B, T, k)
        # 重新归一化 top-k 概率（使和为 1）
        top_probs = top_probs / np.maximum(top_probs.sum(axis=-1, keepdims=True), 1e-8)
        # 排序（按概率降序）
        sort_order = np.argsort(-top_probs, axis=-1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=-1)
        top_probs = np.take_along_axis(top_probs, sort_order, axis=-1)

        # 把 weights 包装为可微 Tensor（梯度流回 probs）
        # weights[b,t,i] = probs[b,t, top_idx[b,t,i]] / sum_j(probs[b,t,top_idx[b,t,j]])
        # 反向：d_probs[b,t,r] = sum_{i: top_idx[b,t,i]==r} (d_weights[b,t,i] * w[i] * (1 - w[i]) / s)
        # 简化为：d_probs[b,t,r] = (d_weights[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i])) 当 r 被选中
        # 这是一个简化版本，更精确的反向需要 softmax jacobian，但 top-k softmax 反向复杂
        # 这里采用直接从 probs gather 的方式（梯度路径完整但近似）
        weights = Tensor(
            top_probs.astype(np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="topk_weights",
        )
        if probs.requires_grad:
            saved_top_idx = top_idx.copy()
            saved_top_probs = top_probs.astype(np.float32)
            saved_probs_data = probs.data
            def _bw():
                # 简化的反向：把 d_weights 按位置 scatter 回 d_probs
                # 数学上：weights_i = probs[idx_i] / sum_j(probs[idx_j])
                # d_probs[r] = sum_i (d_weights_i * d_weights_i / d_probs[r]_full) 复杂
                # 采用近似：d_probs[r] = sum_{i: idx_i==r} d_weights_i * w_i * (1 - w_i)
                # 这是 softmax-over-topk 的对角近似（忽略交叉项）
                grad_w = weights.grad  # (B, T, k)
                grad_p = np.zeros_like(saved_probs_data)  # (B, T, num_routes)
                # scatter：对每个 (b,t,i)，把 grad_w[b,t,i] * top_probs[b,t,i] * (1 - top_probs[b,t,i]) 累加到 grad_p[b,t, idx]
                contrib = grad_w * saved_top_probs * (1.0 - saved_top_probs)  # (B, T, k)
                np.add.at(grad_p, (batch_idx, seq_idx, saved_top_idx), contrib)
                probs._accumulate_grad(grad_p)
            weights._backward = _bw

        indices = top_idx.astype(np.int32)

        # 计算 aux loss：Switch Transformer 风格 load balancing loss
        # f_i = (被路由到 route i 的 token 数) / (总 token 数 × top_k)
        # P_i = mean(probs[:, :, i])
        # aux_loss = num_routes * sum_i (f_i * P_i)
        aux_loss = self._compute_aux_loss(probs, indices)

        return indices, weights, aux_loss

    def _compute_aux_loss(self, probs, indices):
        """计算 load balancing aux loss。

        Args:
            probs: (B, T, num_routes) softmax 概率
            indices: (B, T, top_k) 选中的 route 索引

        Returns:
            aux_loss: scalar Tensor
        """
        num_routes = self.num_routes
        B, T, k = indices.shape
        total_tokens = B * T * k

        # f_i: 每个 route 被选中的 token 比例
        # 用 one-hot 统计
        indices_flat = indices.reshape(-1)  # (B*T*k,)
        counts = np.bincount(indices_flat, minlength=num_routes).astype(np.float32)
        f = counts / max(total_tokens, 1)  # (num_routes,)

        # P_i: 每个 route 的平均概率
        P = probs.data.reshape(-1, num_routes).mean(axis=0)  # (num_routes,)

        # aux_loss = num_routes * sum(f_i * P_i)
        aux_val = float(num_routes * np.sum(f * P))  # scalar

        # 包装为可微 Tensor（aux loss 关于 probs 的梯度 = num_routes * f_i / (B*T*num_routes) = f / (B*T)）
        aux_loss = Tensor(
            np.array(aux_val, dtype=np.float32),
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="aux_loss",
        )
        if probs.requires_grad:
            saved_f = f.copy()
            saved_B_T = B * T
            def _bw():
                # d_aux / d_probs[i] = num_routes * f_i * (1/(B*T*num_routes)) * num_routes
                # 简化：d_aux / d_probs[i] = f_i / (B*T) （因为 P_i = mean(probs[:,i])）
                grad_p = np.full((B, T, num_routes),
                                 saved_f / max(saved_B_T, 1),
                                 dtype=np.float32) * self.aux_loss_weight
                probs._accumulate_grad(grad_p)
            aux_loss._backward = _bw

        return aux_loss


# ---------------------------------------------------------------------------
# Expert: 单个 SwiGLU MLP
# ---------------------------------------------------------------------------


class Expert(Module):
    """单个 Expert（SwiGLU MLP）。

    Args:
        dim: 输入/输出维度
        hidden