"""Task 5.4.1: ModelLoader - 从 HF repo 或本地路径加载预训练 LM。

简化版策略
-----------
本实现 **不要求真正加载 HF 上的大模型**（网络下载可能失败 / 模型格式不匹配），
而是支持：

1. **自构建轻量 LM**：根据 ``arch`` 参数（``mamba2`` / ``rwkv7`` / ``hybrid``）
   用 ``verse_nex`` 内置的 ``HybridLM`` 构建一个 CPU 友好的小型 LM；
2. **可选权重覆盖**：若 ``repo_or_path`` 指向一个本地或 HF 路径，
   则用 ``verse_compat.load_hf_state_dict`` 加载权重并尝试匹配覆盖；
3. **严格模式 / 宽松模式**：默认宽松模式，仅覆盖能匹配上的键，未匹配键保持初始化值；
   严格模式（``strict=True``）会要求所有键匹配。
4. **CometSpark arch**（Stage 7 新增）：``arch="cometspark"`` 时从 pickle 文件
   加载 ``CometSparkLM``（含 config + state_dict），动态导入类避免硬依赖。

构建的 LM 接口
--------------
返回的 ``LanguageModel``（即 ``verse_nex.HybridLM`` 或 ``CometSparkLM``）暴露：

- ``forward(input_ids, states=None, mode="parallel"|"recurrent") -> Tensor``
  - parallel 模式：返回 (B, T, vocab_size) logits
  - recurrent 模式：把 (B, 1) input_ids 转为 (B, 1, vocab_size) logits，
    并把新状态挂在返回 Tensor 的 ``_state`` 属性上
- ``forward_recurrent(input_ids, states=None) -> (logits, new_states)``
- ``forward_parallel(input_ids) -> logits``

这样 ``StreamingGenerator`` 可以直接调用 ``model.forward_recurrent``。
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import sys
from typing import Optional

import numpy as np

from verse_torch import Tensor, no_grad
from verse_nex import HybridLM, Mamba2Block, RWKV7Block
from verse_nex.hybrid import HybridBlock


# ---------------------------------------------------------------------------
# 默认架构配置：CPU 友好的小型 LM
# ---------------------------------------------------------------------------

# 各 arch 的默认 kwargs（vocab / dim 等可由 config.json 或显式参数覆盖）
_DEFAULT_ARCH_CONFIGS = {
    "mamba2": {
        "ssm_kind": "mamba2",
        "dim": 128,
        "n_layers": 4,
        "sparse_ratio": 0.0,    # 纯 mamba2，无 sparse attention
        "ssm_kwargs": {"d_state": 64, "d_conv": 4, "expand": 2, "n_heads": 4},
        "sparse_kwargs": {},
    },
    "rwkv7": {
        "ssm_kind": "rwkv7",
        "dim": 128,
        "n_layers": 4,
        "sparse_ratio": 0.0,
        "ssm_kwargs": {"n_head": 4, "head_size": 32, "hidden": 256},
        "sparse_kwargs": {},
    },
    "hybrid": {
        "ssm_kind": "mamba2",
        "dim": 128,
        "n_layers": 4,
        "sparse_ratio": 0.25,   # 25% sparse attention
        "ssm_kwargs": {"d_state": 64, "d_conv": 4, "expand": 2, "n_heads": 4},
        "sparse_kwargs": {"n_heads": 4, "chunk_size": 16, "n_sliding_chunks": 1, "topk_chunks": 1},
    },
}


def _merge_config(arch: str, user_kwargs: dict) -> dict:
    """合并默认配置与用户提供的 kwargs。"""
    cfg = dict(_DEFAULT_ARCH_CONFIGS.get(arch, _DEFAULT_ARCH_CONFIGS["mamba2"]))
    cfg.update(user_kwargs)
    return cfg


# ---------------------------------------------------------------------------
# CometSparkLM 动态加载（Stage 7 / Part4K1 Task 8.9 迁移到 spark/）
# ---------------------------------------------------------------------------
# Part4K1 Task 8.9: CometSparkV05LM 定义在 spark/model/model.py，
# 替代原 data/demo/model/model.py（已删除）。采用「动态导入 + sys.path 注入」策略：
# 1. 优先尝试 ``from spark.model.model import CometSparkV05LM``（需要 /workspace 在 sys.path）；
# 2. 回退到旧路径 ``data.demo.model.model``（向后兼容，data/demo 已删除时会失败）；
# 3. 用户可通过 ``register_cometspark_path()`` 主动注册路径。

# 缓存已加载的 CometSparkLM 类，避免重复 import
_COMETSPARK_LM_CLASS = None
# 默认 spark 路径基于 __file__ 推断，避免硬编码 /workspace。
# 假设包结构：<repo_root>/packages/verse_infra/verse_infra/model_loader.py
# 推断 <repo_root> 后拼接 spark/。
# 用户可通过环境变量 COMETSPARK_SPARK_PATH 覆盖默认路径。
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))  # verse_inference/  (package dir)
        )  # verse_inference/  (project dir)
    )  # packages/
)  # <repo_root>/
_DEFAULT_COMETSPARK_SPARK_PATH = os.path.join(_REPO_ROOT, "spark")


def register_cometspark_path(demo_path: str) -> None:
    """注册 CometSpark 模型目录路径，便于动态加载 CometSparkV05LM 类。

    Part4K1 Task 8.9: 原 ``demo_path`` 指向 ``data/demo``，现在指向 ``spark/``。
    为向后兼容保留参数名 ``demo_path``。

    Args:
        demo_path: 指向 ``spark`` 目录的绝对路径（或旧 data/demo 路径，向后兼容）
    """
    global _COMETSPARK_LM_CLASS
    if demo_path and demo_path not in sys.path:
        sys.path.insert(0, demo_path)
    # 清除缓存，下次加载时用新路径重试
    _COMETSPARK_LM_CLASS = None


def _import_cometspark_lm():
    """动态加载并返回 CometSparkV05LM 类（Part4K1 Task 8.9 迁移到 spark/）。

    查找顺序：
        1. 已缓存（``_COMETSPARK_LM_CLASS``）
        2. ``from spark.model.model import CometSparkV05LM``
           （需要 /workspace 在 sys.path）
        3. 旧路径 ``from data.demo.model.model import CometSparkLM``
           （向后兼容，data/demo 已删除时会失败）
        4. 从默认路径 ``<repo>/spark`` 加载（可用环境变量覆盖）

    Returns:
        CometSparkV05LM 类（不是实例）
    """
    global _COMETSPARK_LM_CLASS
    if _COMETSPARK_LM_CLASS is not None:
        return _COMETSPARK_LM_CLASS

    # 确保 repo root 在 sys.path（spark 是顶层包）
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    # 2. 尝试 spark.model.model（Part4K1 Task 8.9 首选路径）
    try:
        mod = importlib.import_module("spark.model.model")
        _COMETSPARK_LM_CLASS = getattr(mod, "CometSparkV05LM")
        return _COMETSPARK_LM_CLASS
    except ImportError:
        pass

    # 3. 旧路径 data.demo.model.model（向后兼容，data/demo 已删除时会失败）
    try:
        mod = importlib.import_module("data.demo.model.model")
        _COMETSPARK_LM_CLASS = getattr(mod, "CometSparkLM")
        return _COMETSPARK_LM_CLASS
    except ImportError:
        pass

    # 4. 从默认 spark 路径加载（注入 sys.path 后重试）
    spark_path = os.environ.get(
        "COMETSPARK_SPARK_PATH", _DEFAULT_COMETSPARK_SPARK_PATH
    )
    if spark_path and spark_path not in sys.path:
        sys.path.insert(0, spark_path)
    try:
        mod = importlib.import_module("spark.model.model")
        _COMETSPARK_LM_CLASS = getattr(mod, "CometSparkV05LM")
        return _COMETSPARK_LM_CLASS
    except ImportError as e:
        raise ImportError(
            f"无法加载 CometSparkV05LM。请确保 spark/ 目录存在，"
            f"或调用 register_cometspark_path(spark_path) 注册路径。"
            f"当前 spark_path={spark_path!r}，错误：{e}"
        )


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------


class ModelLoader:
    """从 HF repo 或本地路径加载预训练 LM 到 VerseNex + VerseTorch。

    Args:
        arch: "mamba2" / "rwkv7" / "hybrid" / "cometspark"
            （"cometspark" 时从 pickle 文件加载完整 CometSparkLM）
        vocab_size: 词表大小（默认 256，适合 demo）
        dim: 模型维度（默认 128）
        n_layers: 层数（默认 4）

    用法
    ----
        loader = ModelLoader(arch="mamba2", vocab_size=256, dim=128, n_layers=4)
        model = loader.load()                       # 自构建 LM
        model = loader.load("/path/to/weights")     # 自构建 + 加载权重覆盖
        model = loader.load("owner/repo")           # 从 HF 下载并覆盖（需网络）

        # CometSpark 加载（Stage 7）：
        loader = ModelLoader(arch="cometspark")
        model = loader.load("/workspace/data/demo/checkpoints/cometspark.pt")

    简化版策略
    ----------
    不要求真正加载 HF 上的大模型，而是：
    1. 用 ``verse_nex.HybridLM`` 构建一个轻量 LM；
    2. 若提供了 ``repo_or_path``，则用 ``verse_compat.load_hf_state_dict``
       加载权重并尝试匹配覆盖（宽松模式：只覆盖能匹配的键）。
    3. ``arch="cometspark"`` 时走专用路径：从 pickle 文件加载完整 CometSparkLM
       （含 config + state_dict），动态导入 ``CometSparkLM`` 类避免硬依赖。
    """

    def __init__(
        self,
        arch: str = "mamba2",
        vocab_size: int = 256,
        dim: int = 128,
        n_layers: int = 4,
        **arch_kwargs,
    ):
        if arch not in ("mamba2", "rwkv7", "hybrid", "cometspark"):
            raise ValueError(
                f"arch must be one of 'mamba2' / 'rwkv7' / 'hybrid' / 'cometspark', got {arch!r}"
            )
        self.arch = arch
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.arch_kwargs = arch_kwargs
        # 构建时记录 config，便于后续从 config.json 加载时复用
        self.config = {
            "arch": arch,
            "vocab_size": vocab_size,
            "dim": dim,
            "n_layers": n_layers,
            **arch_kwargs,
        }

    # ------------------------------------------------------------------
    # 从 config.json 加载（如果路径中存在）
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(repo_or_path: str) -> Optional[dict]:
        """尝试从路径或 HF repo 加载 config.json，返回 arch 字段等关键信息。"""
        local_path = None
        if os.path.isdir(repo_or_path):
            local_path = os.path.join(repo_or_path, "config.json")
        elif os.path.isfile(repo_or_path) and repo_or_path.endswith(".json"):
            local_path = repo_or_path
        if local_path and os.path.isfile(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # 主入口：load
    # ------------------------------------------------------------------

    def load(
        self,
        repo_or_path: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ) -> HybridLM:
        """构建 LM 并可选加载权重覆盖。

        Args:
            repo_or_path: 可选。本地目录 / 文件 / HF repo id。
                若为 None，仅返回自构建的随机初始化 LM。
                ``arch="cometspark"`` 时必须提供，指向 .pt pickle 文件。
            strict: 是否要求加载的 state_dict 与模型完全匹配（默认 False，宽松）。
            **kwargs: 传给 HybridLM 的额外参数（覆盖构造时的 arch_kwargs）。

        Returns:
            HybridLM 实例（已 eval 模式，``requires_grad=False``）。
            ``arch="cometspark"`` 时返回 CometSparkLM 实例。
        """
        # CometSpark arch: 走专用加载路径（从 pickle 文件加载完整模型）
        if self.arch == "cometspark":
            return self._load_cometspark(repo_or_path, strict=strict, **kwargs)

        # 1. 合并 config
        config = _merge_config(self.arch, dict(self.arch_kwargs))
        # 显式参数覆盖
        config["vocab_size"] = kwargs.pop("vocab_size", self.vocab_size)
        config["dim"] = kwargs.pop("dim", self.dim)
        config["n_layers"] = kwargs.pop("n_layers", self.n_layers)
        config.update(kwargs)

        # 若路径中有 config.json，尝试从中读取 arch / dim / n_layers（覆盖默认）
        if repo_or_path:
            cfg_json = self._load_config(repo_or_path)
            if cfg_json:
                # 尝试从 config.json 推断 arch
                arch_hint = (
                    cfg_json.get("arch")
                    or cfg_json.get("model_type")
                    or cfg_json.get("architectures", [None])[0]
                )
                if isinstance(arch_hint, str):
                    arch_lower = arch_hint.lower()
                    if "mamba" in arch_lower:
                        config["ssm_kind"] = "mamba2"
                    elif "rwkv" in arch_lower:
                        config["ssm_kind"] = "rwkv7"
                    elif "hybrid" in arch_lower:
                        # hybrid 不改 ssm_kind，保留用户设定
                        pass
                if "vocab_size" in cfg_json:
                    config["vocab_size"] = int(cfg_json["vocab_size"])
                if "dim" in cfg_json or "hidden_size" in cfg_json:
                    config["dim"] = int(cfg_json.get("dim", cfg_json.get("hidden_size")))
                if "n_layers" in cfg_json or "num_hidden_layers" in cfg_json:
                    config["n_layers"] = int(
                        cfg_json.get("n_layers", cfg_json.get("num_hidden_layers"))
                    )

        # 2. 构建 LM
        model = HybridLM(
            vocab_size=int(config["vocab_size"]),
            dim=int(config["dim"]),
            n_layers=int(config["n_layers"]),
            sparse_ratio=float(config.get("sparse_ratio", 0.0)),
            ssm_kind=config.get("ssm_kind", "mamba2"),
            ssm_kwargs=config.get("ssm_kwargs"),
            sparse_kwargs=config.get("sparse_kwargs"),
            sparse_placement=config.get("sparse_placement", "spread"),
            tie_weights=bool(config.get("tie_weights", False)),
        )

        # 3. 可选加载权重覆盖
        if repo_or_path:
            try:
                from verse_infra.verse_compat import load_hf_state_dict  # 延迟导入，避免循环依赖
                sd = load_hf_state_dict(repo_or_path)
                self._apply_state_dict(model, sd, strict=strict)
            except Exception as e:
                # 加载失败不致命：保留自构建的随机初始化 LM，并打印警告
                import warnings
                warnings.warn(
                    f"load_hf_state_dict failed for {repo_or_path!r}: {e}. "
                    f"Returning randomly initialized model."
                )

        # 4. 切换到 eval 模式 + 关闭梯度
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # 5. 在 model 上挂载 config 与 arch 信息，便于 generator 访问
        object.__setattr__(model, "_arch", self.arch)
        object.__setattr__(model, "_loader_config", config)
        return model

    # ------------------------------------------------------------------
    # CometSpark 专用加载（Stage 7）
    # ------------------------------------------------------------------

    def _load_cometspark(self, repo_or_path: Optional[str], strict: bool = False, **kwargs):
        """从 pickle 文件加载完整 CometSparkLM（含 config + state_dict）。

        期望文件格式（由 ``CometSparkLM.save`` 产生）::

            {
                "config": dict,        # CometSparkConfig.to_dict()
                "state_dict": dict,    # {name: ndarray}
                "arch": str,           # "hybrid" / "transformer"
            }

        Args:
            repo_or_path: .pt 文件路径。必须提供。
            strict: state_dict 加载是否严格匹配（默认 False，宽松）。
            **kwargs: 预留，目前未使用。

        Returns:
            CometSparkLM 实例（已 eval，``requires_grad=False``），
            并挂载 ``_arch="cometspark"`` 与 ``_loader_config`` 属性。
        """
        if not repo_or_path:
            raise ValueError(
                "cometspark arch 需要提供 model_path（.pt 文件路径），"
                "例如 ModelLoader(arch='cometspark').load('/path/to/cometspark.pt')"
            )
        if not os.path.isfile(repo_or_path):
            raise FileNotFoundError(f"CometSpark 模型文件不存在：{repo_or_path!r}")

        # 1. 动态加载 CometSparkLM 类
        CometSparkLM = _import_cometspark_lm()

        # 2. 从 pickle 读取 payload
        with open(repo_or_path, "rb") as f:
            payload = pickle.load(f)

        # 兼容两种格式：
        #   (a) {"config": dict, "state_dict": dict, "arch": str}  ← CometSparkV05LM.save
        #   (b) {"model_state_dict": dict, ...}                    ← CheckpointManager
        if isinstance(payload, dict) and "config" in payload and "state_dict" in payload:
            # 格式 (a)：直接用 CometSparkV05LM.from_pretrained（含 config + 权重）
            model = CometSparkLM.from_pretrained(repo_or_path)
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            # 格式 (b)：仅 state_dict，需要外部 config
            # Part4K1 Task 8.9: 用 spark.model.config 构造默认 config
            from spark.model.config import CometSparkV05Config  # type: ignore
            config_dict = {
                "vocab_size": self.vocab_size,
                "n_layer": self.n_layers,
                "n_embd": self.dim,
                "arch": "versenex",
            }
            config = CometSparkV05Config.from_dict(config_dict)
            model = CometSparkLM(config)
            model.load_state_dict(payload["model_state_dict"], strict=False)
        else:
            raise ValueError(
                f"无法识别的 CometSpark 文件格式：{repo_or_path!r}。"
                f"期望 keys 包括 'config' + 'state_dict' 或 'model_state_dict'，"
                f"实际 keys={list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
            )

        # 3. 若 strict=True，重新严格加载一遍以校验完整性
        if strict and "state_dict" in payload:
            model.load_state_dict(payload["state_dict"], strict=True)

        # 4. 切换到 eval 模式 + 关闭梯度
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # 5. 挂载 config 与 arch 信息，便于 generator 访问
        object.__setattr__(model, "_arch", "cometspark")
        try:
            loader_config = model.config.to_dict()
        except Exception:
            loader_config = {"arch": "cometspark"}
        object.__setattr__(model, "_loader_config", loader_config)
        return model

    # ------------------------------------------------------------------
    # 权重覆盖：宽松匹配
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_state_dict(model: HybridLM, sd: dict, strict: bool = False) -> None:
        """把加载的 state_dict 应用到 model 上。

        策略：
        - 先尝试 ``model.load_state_dict(sd, strict=strict)``（精确匹配）；
        - 若 strict=False 且失败，则降级为「逐键软匹配」：
          遍历 model 的 named_parameters，对于每个 (name, param)，
          尝试在 sd 中找同名的 Tensor，若形状匹配则覆盖 data。
        """
        if strict:
            model.load_state_dict(sd, strict=True)
            return

        # 软匹配
        own_sd = dict(model.named_parameters_with_module())
        # 同时考虑 buffer（如 LayerNorm 的 running_mean；这里没有 BN，可忽略）
        matched = 0
        for name, param in own_sd.items():
            if name in sd:
                src = sd[name]
                src_data = src.data if hasattr(src, "data") else np.asarray(src)
                if src_data.shape == param.data.shape:
                    param.data = src_data.astype(param.data.dtype, copy=False)
                    matched += 1
        # 如果一个都没匹配上，尝试宽松的「忽略前缀」匹配
        if matched == 0:
            # 常见情况：HF 权重有 "backbone." 前缀，本地模型没有
            for name, param in own_sd.items():
                for prefix in ("backbone.", "model.", "module."):
                    alt = prefix + name
                    if alt in sd:
                        src = sd[alt]
                        src_data = src.data if hasattr(src, "data") else np.asarray(src)
                        if src_data.shape == param.data.shape:
                            param.data = src_data.astype(param.data.dtype, copy=False)
                            matched += 1
                            break


__all__ = ["ModelLoader", "register_cometspark_path"]
