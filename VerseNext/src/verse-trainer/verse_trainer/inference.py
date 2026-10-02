"""推理优化与生成引擎（CPU / CUDA / ROCm / CANN NPU）。

优化手段按设备分层，全部通过 :mod:`verse_nn.devices` 与 :mod:`verse_nn.kernels`
统一派发，不在本模块里写平台分支：

- **通用**：KV 缓存解码（prefill 一次、每步 1 token，模型侧实现）、
  ``torch.inference_mode``（关闭版本计数与 autograd）、融合算子
  （attention / swiglu / rope 走自研 kernel）。
- **CPU**：int8 动态量化（按通道权重 + 动态激活，走 oneDNN/FBGEMM，
  典型 1.5-2x、体积约 1/4）；线程数与 flush denormal 由资源锁统一设置；
  支持 AVX512-BF16/AMX 上的 bf16 混合精度推理。
- **CUDA / ROCm**：bf16/fp16 autocast + TF32；自研 flash attention kernel。
- **CANN NPU**：bf16 autocast + ``npu_fusion_attention`` 等 CANN 融合算子。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from verse_core import get_logger
from verse_core.config import BaseConfig
from verse_nn import kernels
from verse_nn.devices import detect_backend
from verse_nn.transformer import VerseTransformer

logger = get_logger("inference")


def optimize_cpu(num_threads: int = 0, resources=None) -> int:
    """CPU 推理全局优化：线程数 + flush denormal。返回设置的线程数。

    线程数默认按 :class:`ResourceConfig` 的 ``cpu_fraction``（50%）从
    affinity 感知的可用核数计算，避免容器里 ``os.cpu_count()`` 偏小导致欠用。
    """
    from verse_trainer.resources import ResourceConfig, apply_resource_limits

    return apply_resource_limits(num_threads, resources or ResourceConfig(), "cpu")


def dynamic_quantize(model: torch.nn.Module, mode: str = "dynamic") -> torch.nn.Module:
    """int8 量化（CPU 专属优化），就地量化并返回模型。

    Args:
        mode: ``"dynamic"`` —— 走 torch.ao 的标准动态量化（成熟稳定）；
              ``"per_channel"`` —— 走自研 :class:`~verse_nn.kernels.Int8Linear`
              （per-channel 权重 + per-token 激活 + ``_int_mm``）。
    """
    if mode == "per_channel":
        try:
            return kernels.quantize_model_int8(model)
        except Exception as exc:  # pragma: no cover - 版本/后端相关
            logger.warning("per-channel 量化失败，回退 dynamic: %s", exc)
    result = kernels.quantize_dynamic_int8(model)
    logger.info("已应用 int8 动态量化")
    return result


@dataclass
class GenerateConfig(BaseConfig):
    """生成配置。"""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram: int = 3
    greedy: bool = False
    use_cache: bool = True
    num_threads: int = 0
    quantize: bool = False
    """启用 int8 量化（仅 CPU 推荐）。"""
    quantize_mode: str = "dynamic"
    """``dynamic``（torch.ao）或 ``per_channel``（自研 Int8Linear）。"""
    device: str = "auto"
    """推理设备：``auto`` / ``cpu`` / ``cuda`` / ``npu``。"""
    dtype: str = "auto"
    """混合精度：``auto``（按设备能力选 bf16/fp16）/ ``fp32`` / ``bf16`` / ``fp16``。"""


class Generator:
    """推理引擎：封装模型 + 分词器 + chat 模板。

    用法::

        gen = Generator.from_pretrained("my_model_ckpt/", "my_tokenizer/")
        text = gen.chat([{"role": "user", "content": "你好"}])
        text = gen.complete("Once upon a time")
    """

    def __init__(
        self,
        model: VerseTransformer,
        tokenizer=None,
        chat_template=None,
        config: GenerateConfig | None = None,
    ) -> None:
        self.config = config or GenerateConfig()
        self.tokenizer = tokenizer
        self.chat_template = chat_template

        # 设备与资源锁：统一走设备后端（CPU 限线程、CUDA/NPU 限显存）
        backend = detect_backend(self.config.device)
        self.backend = backend
        self.device = torch.device(backend.device_type)
        from verse_trainer.resources import ResourceConfig, apply_resource_limits

        threads = apply_resource_limits(
            self.config.num_threads, ResourceConfig(), self.device
        )

        # 混合精度：按设备能力选 dtype（CPU 上仅在支持 bf16 时启用）
        self.dtype = self._resolve_dtype(self.config.dtype, backend)
        self.autocast_enabled = self.dtype != torch.float32

        if self.config.quantize:
            if backend.name != "cpu":
                logger.warning(
                    "int8 量化仅推荐 CPU 使用，当前设备 %s，已跳过", backend.name
                )
            else:
                model = dynamic_quantize(model, self.config.quantize_mode)

        model = model.to(self.device)
        logger.info(
            "推理引擎就绪：device=%s(%s) dtype=%s 线程=%d quantize=%s",
            backend.name, self.device, self.dtype, threads, self.config.quantize,
        )

        self.model = model
        self.model.eval()

    #: 显式 dtype 名 -> torch dtype。不能写 ``getattr(torch, "fp16")``——torch 只有
    #: ``float16``/``bfloat16``，没有 ``fp16``/``bf16`` 别名。
    _DTYPE_MAP = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }

    @classmethod
    def _resolve_dtype(cls, dtype: str, backend) -> torch.dtype:
        """解析推理 dtype：显式优先，auto 时用设备后端推荐值。"""
        explicit = cls._DTYPE_MAP.get(str(dtype).lower())
        if explicit is not None:
            return explicit
        recommended = backend.default_autocast_dtype()
        if recommended is not None and backend.supports_dtype(recommended):
            return recommended
        return torch.float32

    @classmethod
    def from_pretrained(
        cls, model_path: str, tokenizer_dir: str | None = None, config: GenerateConfig | None = None
    ) -> "Generator":
        """从 checkpoint（及可选 tokenizer 目录）构建推理引擎。"""
        from verse_nn.transformer import VerseTransformerConfig

        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        raw_cfg = VerseTransformerConfig.from_dict(payload["model_config"])
        model = VerseTransformer(raw_cfg)
        if payload.get("lora_config"):
            # checkpoint 来自 LoRA 微调：先重建 LoRA 结构再载入
            from verse_nn.lora import LoRAConfig, apply_lora

            apply_lora(model, LoRAConfig.from_dict(payload["lora_config"]))
        state = payload.get("model_state") or payload.get("lora_state")
        model.load_state_dict(state, strict=payload.get("model_state") is not None)
        logger.info("已加载模型 %s (step %s)", model_path, payload.get("global_step"))

        tokenizer = chat_template = None
        if tokenizer_dir is not None:
            from verse_tokenizer import BPETokenizer, ChatTemplate

            tokenizer = BPETokenizer.from_pretrained(tokenizer_dir)
            chat_template = ChatTemplate.from_pretrained(tokenizer_dir)
        return cls(model, tokenizer, chat_template, config)

    @torch.inference_mode()
    def generate_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """token ids -> 续写 token ids（含 eos 提前停止）。"""
        eos = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        ids = ids.to(self.device)
        # 设备侧混合精度：推理只用 autocast，不做 GradScaler
        with self.backend.autocast(self.dtype, enabled=self.autocast_enabled):
            return self.model.generate(
                ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram=self.config.no_repeat_ngram,
                greedy=self.config.greedy,
                eos_token_id=eos,
                use_cache=self.config.use_cache,
            )

    def complete(self, prompt: str) -> str:
        """无模板自由续写。需要 tokenizer。"""
        if self.tokenizer is None:
            raise ValueError("需要 tokenizer（Generator.from_pretrained 传入 tokenizer_dir）")
        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        out = self.generate_ids(ids)[0].tolist()
        return self.tokenizer.decode(out)

    def chat(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """对话生成：渲染模板 -> 生成 -> 解码回答。需要 tokenizer + chat 模板。"""
        if self.tokenizer is None or self.chat_template is None:
            raise ValueError("需要 tokenizer 与 chat 模板")
        prompt = self.chat_template.render(messages, add_generation_prompt=add_generation_prompt)
        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        out = self.generate_ids(ids)[0].tolist()
        text = self.tokenizer.decode(out, skip_special_tokens=False)
        # 截断到第一个结束标记
        for marker in self.tokenizer.special_tokens:
            if marker.startswith("<|im_end") or marker in ("<|end|>", "<|eos|>"):
                idx = text.find(marker)
                if idx >= 0:
                    text = text[:idx]
                break
        return text.strip()
