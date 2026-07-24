"""VerseTorch: Pure-Python/NumPy tensor & autograd engine (CPU-first).

提供 PyTorch 风格的 API：
    >>> import numpy as np
    >>> from verse_torch import Tensor
    >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> y.backward()
    >>> x.grad  # 应为 [2.0, 4.0, 6.0]
"""

from .tensor import (
    Tensor,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
    unbroadcast,
)

# 设备抽象层 (Task 1.1 / 1.3)
from .device import (
    DeviceBackend,
    NumpyBackend,
    get_backend,
    has_torch,
    has_torch_npu,
    DEFAULT_DEVICE,
    # Part4K2 Task 5: 显存管理与 BLAS 线程优化
    empty_cache,
    get_memory_info,
    memory_usage,
    set_num_threads,
    get_num_threads,
    auto_tune_threads,
)
# TorchBackend 仅在 torch 可用时导入（无 torch 时跳过，避免硬依赖）
from .device import has_torch as _has_torch
if _has_torch():
    from .backend_torch import TorchBackend
else:
    TorchBackend = None

# Parameter 别名（与 PyTorch 习惯一致：Parameter = Tensor，通过 requires_grad=True 标识）
Parameter = Tensor
from . import nn
# Part5K1 Task 1：vnn 为推荐路径（原 nn 重命名）。nn 仍可用（薄壳 re-export）。
from . import vnn
# 让 `from verse_torch import nn` 指向 vnn（推荐路径），保证向后兼容不报错。
# 注意：`from verse_torch.nn import X` 仍走 nn.py 薄壳（sys.modules 保留），
# 以便 transformer 系旧名抛 ImportError 的拦截逻辑生效。
nn = vnn
from . import optim
from . import losses
from . import training
from . import quantize
from . import parallel
from . import compress
# Part5K1.1：VMPC V2.0 门面（VN + 传统技术 + VSC）+ VSC 引擎
from . import vmpc
from . import vsc
from .vmpc import (
    VMPCRegularizer,
    vmpc_compress,
    VMPCConfig,
    VMPCV2,
    VMPCStats,
    VMPC_PROFILE_SMALL,
    VMPC_PROFILE_MATE,
)
from .vsc import (
    VSCProfile,
    VSCPlan,
    VSCEngine,
    VSCStats,
    vsc_profile,
    vsc_plan,
    vsc_compress,
)
from .parallel import (
    parallel_matmul,
    ParallelLinear,
    parallel_map,
)
from .quantize import (
    quantize_int8,
    dequantize_int8,
    quantize_int4,
    dequantize_int4,
    matmul_int4,
    quantize_ternary,
    dequantize_ternary,
    matmul_ternary,
    QuantizedLinear,
    quantize_batch,
    benchmark_throughput,
)
from .nn import (
    Module,
    Linear,
    Embedding,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
    ModuleList,
    SwiGLUMLP,
    # Part4K1 SubTask 2.2: 旧名重命名为私有实现，此处直接导入私有名以避免
    # import verse_torch 时触发 DeprecationWarning（仅 from verse_torch.nn
    # import TransformerLM 等直接访问 nn 模块时才发警告）
    _GQASelfAttention as GQASelfAttention,
    _TransformerBlock as TransformerBlock,
    _TransformerLM as TransformerLM,
    SlidingWindowAttention,
    ALiBi,
    DeepNorm,
    RotaryEmbedding,
    KVCache,
    StaticCache,
    DynamicCache,
    GroupNorm,
    Conv1d,
    LayerNormFast,
    repeat_kv,
    kaiming_uniform_,
    xavier_uniform_,
    normal_,
    zeros_,
    ones_,
    uniform_,
)
from .optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    NAdamW,
    RMSProp,
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    LambdaLR,
    warmup_cosine_lr,
)
from .optim_extras import (
    Lion,
    Adafactor,
)
from .scheduler_extras import (
    OneCycleLR,
    ReduceLROnPlateau,
    CosineRestartsLR,
)
from .activations import (
    SiLU,
    Mish,
    GeGLU,
)
from .losses import (
    cross_entropy,
    nll_loss,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    mse_loss,
    l1_loss,
    kl_div_loss,
    focal_loss,
    contrastive_loss,
    perplexity,
)
from .training import (
    cross_entropy_loss,
    EarlyStopping,
    GradientAccumulator,
    CheckpointManager,
    compute_loss_rate,
    plot_loss_curve,
    Trainer,
    BatchLoader,
    clip_grad_norm,
    ParallelTrainer,
    DistributedTrainer,
    # Part4K2 Task 5: 资源利用优化
    GradScaler,
    activation_checkpoint,
)
from .training_nex import (
    VerseNexTrainer,
    LoRATrainer,
    SFTTrainer,
    DPOTrainer,
    SFTDataset,
    DPODataset,
    _sft_collate as sft_collate,
    _dpo_collate as dpo_collate,
    _dpo_loss as dpo_loss,
)
from .compress import (
    OutlierSafePruner,
    LoRALinear,
    KnowledgeDistiller,
    QLinear,
    compress_pipeline,
    compress_mod_experts,
    compression_report,
    prune_only,
    quantize_only,
    lora_only,
    ternary_only,
    distill_only,
    count_parameters as compress_count_parameters,
    count_nonzero_params,
    compute_compressed_bits,
)
from .scoring import (
    ScoringEvaluator,
    exact_match,
    prefix_accuracy,
    char_f1,
    bleu,
    rouge_l,
)
# Part4K2 Task 1: .vn 文件格式（safetensors 性能优化版）
# Part5K1.1: VNCacheManager 多空间缓存
from .vn_format import (
    VN_FORMAT_VERSION,
    VNFileReader,
    VNFileWriter,
    VNCacheManager,
    pt_to_vn,
    vn_to_pt,
    convert_format,
    has_safetensors,
)
# Part4K2 Task 4: 智能分区训练器（LayerWiseTrainer）
# Part5K1 Task 8: VMT 完整智能分区训练器（VMTTrainer）
from .layerwise_trainer import LayerWiseTrainer, VMTTrainer

__version__ = "0.1.0"

__all__ = [
    # Tensor 核心
    "Tensor",
    "Parameter",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "unbroadcast",
    # 设备抽象层 (Task 1.1 / 1.3)
    "DeviceBackend",
    "NumpyBackend",
    "TorchBackend",
    "get_backend",
    "has_torch",
    "has_torch_npu",
    "DEFAULT_DEVICE",
    # Part4K2 Task 5: 显存管理与 BLAS 线程优化
    "empty_cache",
    "get_memory_info",
    "memory_usage",
    "set_num_threads",
    "get_num_threads",
    "auto_tune_threads",
    # 子模块
    "nn",
    "vnn",
    "optim",
    "losses",
    "training",
    "quantize",
    "parallel",
    # nn 类
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
    "ModuleList",
    "SwiGLUMLP",
    "GQASelfAttention",
    "TransformerBlock",
    "TransformerLM",
    "SlidingWindowAttention",
    "ALiBi",
    "DeepNorm",
    "RotaryEmbedding",
    "KVCache",
    "StaticCache",
    "DynamicCache",
    "GroupNorm",
    "Conv1d",
    "LayerNormFast",
    "repeat_kv",
    "kaiming_uniform_",
    "xavier_uniform_",
    "normal_",
    "zeros_",
    "ones_",
    "uniform_",
    # optim 类
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "NAdamW",
    "RMSProp",
    "LRScheduler",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "LambdaLR",
    "warmup_cosine_lr",
    # optim_extras 类
    "Lion",
    "Adafactor",
    # scheduler_extras 类
    "OneCycleLR",
    "ReduceLROnPlateau",
    "CosineRestartsLR",
    # activations 类
    "SiLU",
    "Mish",
    "GeGLU",
    # losses 函数
    "cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "mse_loss",
    "l1_loss",
    "kl_div_loss",
    "focal_loss",
    "contrastive_loss",
    "perplexity",
    # training 模块
    "cross_entropy_loss",
    "EarlyStopping",
    "GradientAccumulator",
    "CheckpointManager",
    "compute_loss_rate",
    "plot_loss_curve",
    "Trainer",
    "BatchLoader",
    "clip_grad_norm",
    "ParallelTrainer",
    "DistributedTrainer",
    # Part4K2 Task 5: 资源利用优化
    "GradScaler",
    "activation_checkpoint",
    # training_nex 模块（Part4）
    "VerseNexTrainer",
    "LoRATrainer",
    "SFTTrainer",
    "DPOTrainer",
    "SFTDataset",
    "DPODataset",
    "sft_collate",
    "dpo_collate",
    "dpo_loss",
    # quantize 函数
    "quantize_int8",
    "dequantize_int8",
    "quantize_int4",
    "dequantize_int4",
    "matmul_int4",
    "quantize_ternary",
    "dequantize_ternary",
    "matmul_ternary",
    "QuantizedLinear",
    "quantize_batch",
    "benchmark_throughput",
    # parallel 函数/类
    "parallel_matmul",
    "ParallelLinear",
    "parallel_map",
    # compress 函数/类
    "compress",
    "OutlierSafePruner",
    "LoRALinear",
    "KnowledgeDistiller",
    "QLinear",
    "compress_pipeline",
    "compress_mod_experts",
    "compression_report",
    "prune_only",
    "quantize_only",
    "lora_only",
    "ternary_only",
    "distill_only",
    "compress_count_parameters",
    "count_nonzero_params",
    "compute_compressed_bits",
    # vmpc（Part5K1.1：VMPC V2.0 门面 = VN + 传统技术 + VSC）
    "vmpc",
    "VMPCRegularizer",
    "vmpc_compress",
    "VMPCConfig",
    "VMPCV2",
    "VMPCStats",
    "VMPC_PROFILE_SMALL",
    "VMPC_PROFILE_MATE",
    # vsc（Part5K1.1：VSC 空间压缩引擎）
    "vsc",
    "VSCProfile",
    "VSCPlan",
    "VSCEngine",
    "VSCStats",
    "vsc_profile",
    "vsc_plan",
    "vsc_compress",
    # scoring 函数/类
    "ScoringEvaluator",
    "exact_match",
    "prefix_accuracy",
    "char_f1",
    "bleu",
    "rouge_l",
    # vn_format（Part4K2 Task 1 / Part5K1.1 多空间缓存）
    "VN_FORMAT_VERSION",
    "VNFileReader",
    "VNFileWriter",
    "VNCacheManager",
    "pt_to_vn",
    "vn_to_pt",
    "convert_format",
    "has_safetensors",
    # layerwise_trainer（Part4K2 Task 4 / Part5K1 Task 8）
    "LayerWiseTrainer",
    "VMTTrainer",
]
