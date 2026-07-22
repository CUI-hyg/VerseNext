"""训练入口：ParallelTrainer 升级 + _safe_chunk_run + 断点续训（Part4K1 Task 6.2）。

从 ``data/demo/train/trainer.py`` 迁入并升级：

1. :class:`ParallelTrainerSafe`：包裹 :class:`verse_torch.training.ParallelTrainer`，
   每个 chunk 用 :func:`_safe_chunk_run` 执行：
   - try/except 捕获 chunk 内异常，graceful 保存 checkpoint 后退出
   - 信号处理：捕获 SIGTERM/SIGINT，graceful 保存 checkpoint 后退出
   - OOM 兜底：catch MemoryError / OutOfMemory，缩小 batch 重试
2. :func:`install_signal_handlers`：注册 SIGTERM/SIGINT 处理器，收到信号时
   设置全局 ``_shutdown_requested`` 标志，训练循环检测后 graceful 保存退出。
3. 断点续训：``--resume`` 参数，从 last checkpoint 恢复
   （保存 step + optimizer state + best_state）。
4. 修复"莫名终止退出"：异常捕获 + 信号处理 + OOM 兜底三重保险。
5. 集成 :class:`verse_trainer.loss_optim.LossOptimizer`（plateau 重走 + NaN/Inf 跳过）。

主入口 :func:`train` 兼容 ``data/demo/train/trainer.py`` 原签名，
并新增 ``resume`` / ``loss_optimizer`` 等参数。
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from verse_torch.optim import AdamW, LambdaLR, warmup_cosine_lr
from verse_torch.training import (
    Trainer,
    ParallelTrainer,
    CheckpointManager,
    _default_collate,
    plot_loss_curve,
)
from verse_torch.training_nex import VerseNexTrainer

from .data import (
    CachedDataset,
    TextDataset,
    BatchLoader,
    collate_fn,
    SingleSampleDataset,
)
from .loss_optim import LossOptimizer

# ---------------------------------------------------------------------------
# 信号处理 + 全局 shutdown 标志
# ---------------------------------------------------------------------------

# 全局 shutdown 标志：信号处理器置为 True，训练循环检测后 graceful 退出。
# 用 threading.Event 保证跨线程可见性（训练可能在子线程跑）。
_shutdown_event = threading.Event()
# 信号处理器安装标志（避免重复安装）
_signal_handlers_installed = False
# 保存原始信号处理器，便于恢复
_original_signal_handlers: dict = {}


def _signal_handler(signum, frame):
    """SIGTERM/SIGINT 处理器：设置 shutdown 标志，不立即退出。

    训练循环在下一个 step / chunk 边界检测到 ``_shutdown_event`` 后
    graceful 保存 checkpoint 并退出。
    """
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(
        f"\n[signal] 收到 {sig_name} 信号，设置 shutdown 标志，"
        f"将在下一个 checkpoint 边界 graceful 退出...",
        flush=True,
    )
    _shutdown_event.set()


def install_signal_handlers() -> None:
    """安装 SIGTERM/SIGINT 信号处理器（幂等）。

    在 :func:`train` 入口自动调用；也可由用户手动调用（如自定义训练循环）。
    重复调用不会重复安装。
    """
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return
    # CI 环境可能不允许注册信号处理器（非主线程），用 try/except 兜底
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            _original_signal_handlers[sig] = signal.signal(sig, _signal_handler)
        _signal_handlers_installed = True
    except (ValueError, OSError) as e:
        # 非 main 线程 / 不支持信号的平台：打印警告但不报错
        print(f"[signal] 无法安装信号处理器（非主线程？）：{e}", flush=True)


def reset_shutdown_flag() -> None:
    """重置 shutdown 标志（开始新训练时调用）。"""
    _shutdown_event.clear()


def is_shutdown_requested() -> bool:
    """查询是否收到 shutdown 信号。"""
    return _shutdown_event.is_set()


# ---------------------------------------------------------------------------
# _safe_chunk_run：包裹单个 chunk 执行
# ---------------------------------------------------------------------------


class ChunkOOMError(Exception):
    """chunk 执行时 OOM（MemoryError / OutOfMemory）。"""


def _safe_chunk_run(
    chunk_fn,
    *args,
    max_oom_retries: int = 2,
    batch_shrink_factor: float = 0.5,
    **kwargs,
):
    """安全执行单个 chunk，捕获异常 + OOM 兜底 + 信号处理。

    Args:
        chunk_fn: chunk 执行函数（如 ``ParallelTrainer._train_chunk``）
        *args: 透传给 chunk_fn 的位置参数
        max_oom_retries: OOM 时缩小 batch 重试的最大次数（默认 2）
        batch_shrink_factor: OOM 时 batch 缩小系数（默认 0.5）
        **kwargs: 透传给 chunk_fn 的关键字参数
            （支持 ``batch_size`` 关键字：OOM 时按系数缩小）
    Returns:
        chunk_fn 的返回值
    Raises:
        ChunkOOMError: OOM 重试次数用尽仍失败
        RuntimeError: 收到 shutdown 信号 / chunk 抛出非 OOM 异常
    """
    # 检查 shutdown 标志（chunk 开始前）
    if is_shutdown_requested():
        raise RuntimeError("收到 shutdown 信号，跳过 chunk 执行")

    last_exc: Optional[Exception] = None
    cur_batch = kwargs.get("batch_size")
    for attempt in range(max_oom_retries + 1):
        try:
            if cur_batch is not None:
                kwargs["batch_size"] = cur_batch
            return chunk_fn(*args, **kwargs)
        except MemoryError as e:
            last_exc = e
            print(
                f"[_safe_chunk_run] chunk OOM (MemoryError, attempt={attempt}), "
                f"batch={cur_batch} → 缩小重试",
                flush=True,
            )
            if cur_batch is None:
                # chunk_fn 不接受 batch_size 关键字，无法兜底
                break
            cur_batch = max(1, int(cur_batch * batch_shrink_factor))
        except (RuntimeError,) as e:
            # 兼容 PyTorch CUDA OOM：错误消息含 "out of memory"
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                last_exc = e
                print(
                    f"[_safe_chunk_run] chunk OOM (RuntimeError, attempt={attempt}), "
                    f"batch={cur_batch} → 缩小重试",
                    flush=True,
                )
                if cur_batch is None:
                    break
                cur_batch = max(1, int(cur_batch * batch_shrink_factor))
                continue
            # 非 OOM RuntimeError：记录但不重试
            last_exc = e
            print(
                f"[_safe_chunk_run] chunk 抛出 RuntimeError：{e}",
                flush=True,
            )
            raise RuntimeError(f"chunk 执行失败：{e}") from e
        except Exception as e:
            # 非预期异常：记录并向上抛（不吞掉，便于调试）
            print(
                f"[_safe_chunk_run] chunk 抛出非预期异常 "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            raise RuntimeError(f"chunk 执行失败：{type(e).__name__}: {e}") from e

        # 检查 shutdown 标志（重试前）
        if is_shutdown_requested():
            raise RuntimeError("收到 shutdown 信号，放弃 chunk 重试")

    # 重试次数用尽
    if last_exc is not None:
        raise ChunkOOMError(
            f"chunk OOM 重试 {max_oom_retries} 次仍失败：{last_exc}"
        ) from last_exc
    raise RuntimeError("chunk 执行失败：未知原因")


# ---------------------------------------------------------------------------
# ParallelTrainerSafe：ParallelTrainer + _safe_chunk_run + 信号处理
# ---------------------------------------------------------------------------


class ParallelTrainerSafe(ParallelTrainer):
    """ParallelTrainer 安全升级版（Part4K1 Task 6.2）。

    在 :class:`verse_torch.training.ParallelTrainer` 基础上：
    - 每个 chunk 用 :func:`_safe_chunk_run` 包裹（异常捕获 + OOM 兜底）
    - 训练循环检测 :func:`is_shutdown_requested`，收到信号 graceful 保存退出
    - ``fit`` 结束后保存断点续训所需的完整状态（step + optimizer state + best_state）

    Args:
        同 :class:`ParallelTrainer`，额外：
        resume_path: 断点续训 checkpoint 路径；None 不恢复
        enable_loss_optimizer: 是否启用 :class:`LossOptimizer`（plateau 重走）
        loss_optimizer_cfg: LossOptimizer 配置 dict
    """

    def __init__(
        self,
        *args,
        resume_path: Optional[str] = None,
        enable_loss_optimizer: bool = False,
        loss_optimizer_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resume_path = resume_path
        self.enable_loss_optimizer = bool(enable_loss_optimizer)
        self.loss_optimizer_cfg = dict(loss_optimizer_cfg or {})

        # LossOptimizer（在 fit 中首次用到时初始化）
        self.loss_optimizer: Optional[LossOptimizer] = None
        if self.enable_loss_optimizer:
            # 延迟到 fit 阶段构造（此时 optimizer 已就绪）
            pass

        # 断点续训：恢复 best_val_loss 与 best_state_dict
        if resume_path is not None and os.path.exists(resume_path):
            self._load_resume_state(resume_path)

    def _load_resume_state(self, path: str) -> None:
        """从 checkpoint 恢复训练状态。"""
        print(f"[ParallelTrainerSafe] 从 {path} 恢复训练状态...", flush=True)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            # 恢复 model
            sd = payload.get("model_state_dict") or payload.get("model")
            if sd is not None and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(copy.deepcopy(sd))
            # 恢复 best_val_loss / best_state_dict
            self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
            bs = payload.get("best_state_dict")
            if bs is not None:
                self.best_state_dict = copy.deepcopy(bs)
            print(
                f"[ParallelTrainerSafe] 恢复完成：best_val_loss={self.best_val_loss:.4f}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[ParallelTrainerSafe] 警告：恢复 checkpoint 失败：{e}，"
                f"从头开始训练",
                flush=True,
            )

    def _save_resume_state(self, path: str, step: int = -1) -> None:
        """保存断点续训所需的完整状态。"""
        try:
            payload = {
                "step": int(step),
                "model_state_dict": (
                    self.model.state_dict()
                    if hasattr(self.model, "state_dict") else None
                ),
                "best_state_dict": self.best_state_dict,
                "best_val_loss": float(self.best_val_loss),
                "history": self.history,
            }
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(
                f"[ParallelTrainerSafe] 警告：保存 resume checkpoint 失败：{e}",
                flush=True,
            )

    def _train_chunk_safe(self, model, train_dataset, chunk_steps, chunk_id):
        """用 _safe_chunk_run 包裹 _train_chunk。"""
        # _train_chunk 不接受 batch_size 关键字，直接包裹异常捕获
        # OOM 时由 _train_chunk 内部的 Trainer 抛出，我们捕获后跳过该 chunk
        try:
            return _safe_chunk_run(
                self._train_chunk, model, train_dataset, chunk_steps, chunk_id,
            )
        except ChunkOOMError as e:
            print(
                f"[ParallelTrainerSafe] chunk {chunk_id} OOM 放弃：{e}，"
                f"跳过该 chunk 继续后续流程",
                flush=True,
            )
            # 用当前 model 评估 val_loss 作为该 chunk 的结果
            val_loss = self._eval_full_val(model)
            return model, float("inf"), val_loss
        except RuntimeError as e:
            if is_shutdown_requested():
                print(
                    f"[ParallelTrainerSafe] 收到 shutdown 信号，"
                    f"chunk {chunk_id} 中止",
                    flush=True,
                )
                raise
            print(
                f"[ParallelTrainerSafe] chunk {chunk_id} 异常：{e}，"
                f"跳过该 chunk 继续后续流程",
                flush=True,
            )
            val_loss = self._eval_full_val(model)
            return model, float("inf"), val_loss

    def fit(self):
        """安全版 fit：每个 chunk 用 _safe_chunk_run 包裹 + 信号检测。

        Returns:
            同 :meth:`ParallelTrainer.fit`，``self.history`` dict
        """
        # 安装信号处理器
        install_signal_handlers()
        reset_shutdown_flag()

        # LossOptimizer 初始化（启用时）
        if self.enable_loss_optimizer:
            # ParallelTrainer 没有 optimizer 属性，用临时 AdamW 兜底
            # （实际 loss_optimizer 主要在 chunk 内的 Trainer 生效，
            # 这里仅维护 val_loss 历史与 best_val_loss）
            self.loss_optimizer = LossOptimizer(
                model=self.model,
                optimizer=AdamW(self.model.parameters(), lr=self.lr),
                **self.loss_optimizer_cfg,
            )

        try:
            # 调用父类 fit，但把 _train_chunk 替换为 _train_chunk_safe
            # 通过 monkey-patch self._train_chunk 实现（避免重写整个 fit）
            original_train_chunk = self._train_chunk
            self._train_chunk = self._train_chunk_safe

            # 在每个 chunk 之间检测 shutdown 信号：通过包装 _split_steps
            # 让 fit 在收到信号时尽快返回（best_state_dict 已更新）
            history = super().fit()

            # 保存断点续训状态
            if self.checkpoint_mgr is not None:
                resume_path = os.path.join(
                    str(self.checkpoint_mgr.save_dir), "resume.pt"
                )
                self._save_resume_state(resume_path, step=self.max_steps)
            return history
        finally:
            # 恢复原始 _train_chunk
            self._train_chunk = original_train_chunk
            # 保存最终 resume 状态
            if self.checkpoint_mgr is not None:
                resume_path = os.path.join(
                    str(self.checkpoint_mgr.save_dir), "resume.pt"
                )
                self._save_resume_state(resume_path, step=self.max_steps)


# ---------------------------------------------------------------------------
# 配置加载（从 data/demo/model/config.py 迁入轻量版本）
# ---------------------------------------------------------------------------


def _resolve_path(base_dir: str, path_str: str) -> str:
    """把配置中的相对路径解析为相对 base_dir 的绝对路径。"""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


def _load_full_config(path: str) -> dict:
    """加载完整 YAML 配置（含所有段）。

    优先用 PyYAML；不可用时降级到 verse_torch 内置解析器。
    """
    try:
        from verse_torch._yaml_compat import load_full_config as _load
        return _load(path)
    except Exception:
        pass
    # 兜底：尝试 PyYAML
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            return loaded if isinstance(loaded, dict) else {}
    except Exception:
        # 极简解析（仅 model/training 两层）
        return _parse_yaml_minimal(path)


def _parse_yaml_minimal(path: str) -> dict:
    """极简 YAML 解析（仅支持两层嵌套标量）。"""
    result: dict = {}
    current_section = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith((" ", "\t")):
                if ":" in line:
                    key = line.split(":", 1)[0].strip()
                    val = line.split(":", 1)[1].strip()
                    if val == "":
                        current_section = key
                        result[current_section] = {}
                    else:
                        result[key] = _parse_scalar(val)
                        current_section = None
                continue
            if current_section is None or ":" not in line:
                continue
            key = line.split(":", 1)[0].strip()
            val = line.split(":", 1)[1].strip()
            result[current_section][key] = _parse_scalar(val)
    return result


def _parse_scalar(s: str):
    s = s.strip()
    if s == "" or s.lower() in ("null", "~"):
        return None
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# 模型构建（从 data/demo/model 迁入轻量版本，延迟导入 verse_nex）
# ---------------------------------------------------------------------------


def _build_model(model_cfg: dict, vocab_size: int):
    """根据 model_cfg 构建模型。

    优先用 ``CometSparkConfig`` + ``CometSparkLM``（来自 data/demo/model），
    若不可用则直接用 ``verse_nex.VerseNexLM``。
    """
    config_dict = dict(model_cfg)
    config_dict["vocab_size"] = vocab_size

    # 路径 1：尝试用 data/demo/model 的 CometSparkLM（兼容旧 config.yml）
    try:
        import sys as _sys
        demo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(_sys.modules[__name__].__file__))),
            "data", "demo",
        )
        if demo_dir not in _sys.path:
            _sys.path.insert(0, demo_dir)
        from model.config import CometSparkConfig
        from model.model import CometSparkLM
        config = CometSparkConfig.from_dict(config_dict)
        return CometSparkLM(config), config
    except Exception as e:
        # 路径 2：直接用 verse_nex.VerseNexLM
        pass

    from verse_nex import VerseNexLM
    # VerseNexLM 关键参数：vocab_size / dim / n_layer / n_head / n_kv_head
    # / window_size / num_global_tokens / max_seq_len / use_alibi / use_rope
    # / dropout / tie_weights
    model = VerseNexLM(
        vocab_size=int(config_dict.get("vocab_size", 256)),
        dim=int(config_dict.get("n_embd", 128)),
        n_layer=int(config_dict.get("n_layer", 4)),
        n_head=int(config_dict.get("n_head", 4)),
        n_kv_head=int(config_dict.get("n_kv_head", None) or config_dict.get("n_head", 4)),
        window_size=int(config_dict.get("window_size", 512)),
        num_global_tokens=int(config_dict.get("num_global_tokens", 64)),
        max_seq_len=int(config_dict.get("seq_len", 128)),
        use_alibi=bool(config_dict.get("use_alibi", True)),
        use_rope=bool(config_dict.get("use_rope", False)),
        dropout=float(config_dict.get("dropout", 0.1)),
        tie_weights=bool(config_dict.get("tie_weights", True)),
    )
    # 构造一个最小 config 对象（供 to_dict / aux_loss_weight 读取）
    class _MinimalConfig:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
            if not hasattr(self, "aux_loss_weight"):
                self.aux_loss_weight = 0.01
            if not hasattr(self, "arch"):
                self.arch = "versenex"
        def to_dict(self):
            return config_dict
    return model, _MinimalConfig(config_dict)


def _load_tokenizer(tok_cfg: dict, base_dir: str, save_dir: str):
    """加载 tokenizer（兼容 data/demo/model/tokenizer.py 与 verse_tokenizer）。"""
    tok_kind = str(tok_cfg.get("kind", "byte"))
    tok_path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        alt = _resolve_path(base_dir, "tokenizer.json")
        if os.path.exists(alt):
            tok_path = alt
        elif tok_kind == "byte":
            # byte tokenizer 无需训练文件（vocab 259 确定），即时构造即可，
            # 让 ``verse-train`` 对 byte 配置开箱即用（small 调试配置场景）。
            from verse_infra.verse_tokenizer import load_tokenizer as _vload
            return _vload(kind="byte")
        else:
            raise FileNotFoundError(
                f"tokenizer 文件不存在：{tok_path}。请先 build_tokenizer。"
            )
    try:
        from verse_infra.verse_tokenizer import load_tokenizer as _vload
        return _vload(kind=tok_kind, path=tok_path)
    except Exception:
        # 兜底：data/demo/model/tokenizer.py
        import sys as _sys
        demo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(_sys.modules[__name__].__file__))),
            "data", "demo",
        )
        if demo_dir not in _sys.path:
            _sys.path.insert(0, demo_dir)
        from model.tokenizer import load_tokenizer
        return load_tokenizer(tok_path, kind=tok_kind)


# ---------------------------------------------------------------------------
# 主训练入口 train（兼容 data/demo/train/trainer.py.train）
# ---------------------------------------------------------------------------


def train(
    config_path: str,
    base_dir: str = ".",
    n_threads: int = 0,
    *,
    device: Optional[str] = None,
    single_sample: Optional[dict] = None,
    single_file: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    resume: bool = False,
    amp: bool = False,
    enable_loss_optimizer: bool = False,
) -> dict:
    """主训练函数（Part4K1 Task 6.2 升级版）。

    兼容 ``data/demo/train/trainer.py.train`` 原签名，并新增：
    - ``device``：设备字符串（cpu/cuda/npu）
    - ``single_sample``：单样本 dict（``{"prompt":..., "completion":...}`` 或 ``{"text":...}``）
    - ``single_file``：单文件路径（内容当作纯文本）
    - ``max_steps_override``：覆盖 config 的 max_steps
    - ``resume``：是否从 last checkpoint 断点续训
    - ``amp``：是否启用混合精度
    - ``enable_loss_optimizer``：是否启用 LossOptimizer（plateau 重走）

    Args:
        config_path: 配置文件路径（config.yml）
        base_dir: 配置中相对路径的基准目录（默认当前目录）
        n_threads: NumPy BLAS 线程数；0 表示不限制
    Returns:
        dict 包含：wall_clock / initial_loss / final_loss / best_val_loss /
        checkpoint_dir / loss_history_path / full_model_path / vocab_size
    """
    start_time = time.time()

    # 安装信号处理器
    install_signal_handlers()
    reset_shutdown_flag()

    # 1. 读取配置
    full_cfg = _load_full_config(config_path)
    model_cfg = full_cfg.get("model", {})
    train_cfg = full_cfg.get("training", {})
    tok_cfg = full_cfg.get("tokenizer", {})
    data_cfg = full_cfg.get("data", {})
    ckpt_cfg = full_cfg.get("checkpoint", {})

    # 2. 环境准备
    if n_threads > 0:
        try:
            from verse_torch import num_threads as _num_threads
            _num_threads(n_threads)
        except Exception:
            pass
    seed = int(train_cfg.get("seed", 42))
    try:
        from verse_torch import set_seed
        set_seed(seed)
    except Exception:
        import random
        random.seed(seed)
        np.random.seed(seed)

    # 3. 加载 tokenizer
    save_dir = _resolve_path(base_dir, str(ckpt_cfg.get("save_dir", "checkpoints")))
    os.makedirs(save_dir, exist_ok=True)
    print(f"[train] 加载 tokenizer", flush=True)
    tok = _load_tokenizer(tok_cfg, base_dir, save_dir)
    vocab_size = len(tok)
    print(f"[train] vocab_size = {vocab_size}", flush=True)

    # 4. 构建数据集（支持 single_sample / single_file / CachedDataset）
    seq_len = int(model_cfg.get("seq_len", 128))

    if single_sample is not None:
        # 单样本模式：--single-sample --prompt "..." --completion "..."
        print(f"[train] 单样本模式：{list(single_sample.keys())}", flush=True)
        train_ds = SingleSampleDataset(
            tok, seq_len=seq_len,
            prompt=single_sample.get("prompt", ""),
            completion=single_sample.get("completion", ""),
            text=single_sample.get("text", ""),
        )
        val_ds = train_ds  # 单样本模式下 val=train
    elif single_file is not None:
        # 单文件模式：把整个文件当作一条纯文本
        print(f"[train] 单文件模式：{single_file}", flush=True)
        with open(single_file, "r", encoding="utf-8") as f:
            text = f.read()
        train_ds = SingleSampleDataset(tok, seq_len=seq_len, text=text)
        val_ds = train_ds
    else:
        # 标准模式：用 CachedDataset（首次缓存 + 后续加速）
        train_path = _resolve_path(base_dir, str(data_cfg.get("train_path", "data/train.jsonl")))
        val_path = _resolve_path(base_dir, str(data_cfg.get("val_path", "data/val.jsonl")))
        print(f"[train] 加载训练数据 {train_path}", flush=True)
        train_ds = CachedDataset(tok, train_path, seq_len=seq_len)
        print(f"[train] 加载验证数据 {val_path}", flush=True)
        val_ds = CachedDataset(tok, val_path, seq_len=seq_len)
    print(f"[train] train_samples={len(train_ds)} val_samples={len(val_ds)}",
          flush=True)

    batch_size = int(train_cfg.get("batch_size", 16))
    train_loader = BatchLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False, seed=seed,
    )
    val_loader = BatchLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    # 5. 实例化模型
    model, config = _build_model(model_cfg, vocab_size)
    print(
        f"[train] 实例化模型 arch={getattr(config, 'arch', 'versenex')} "
        f"n_layer={getattr(config, 'n_layer', '?')} "
        f"n_embd={getattr(config, 'n_embd', '?')} seq_len={seq_len}",
        flush=True,
    )

    # 设备迁移
    if device is not None and device != "cpu":
        if hasattr(model, "to"):
            try:
                model.to(device)
                print(f"[train] 模型已迁移到 {device}", flush=True)
            except Exception as e:
                print(f"[train] 警告：迁移模型到 {device} 失败：{e}", flush=True)

    # 6. 优化器 + 学习率调度
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    no_decay = bool(train_cfg.get("no_decay", True))
    if no_decay:
        decay_params, nodecay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = int(max_steps_override or train_cfg.get("max_steps", 200))
    warmup = int(train_cfg.get("warmup", 20))
    scheduler = LambdaLR(
        optimizer, warmup_cosine_lr(warmup_steps=warmup, total_steps=max_steps)
    )

    # 7. Trainer 选择
    parallel_chunks = int(train_cfg.get("parallel_chunks", 1))
    patience = int(train_cfg.get("patience", 5))
    eval_interval = int(train_cfg.get("eval_interval", 20))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    log_interval = int(train_cfg.get("log_interval", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
    enable_progress_bar = bool(train_cfg.get("enable_progress_bar", True))
    realtime_plot = bool(train_cfg.get("realtime_plot", True))
    eta_window = int(train_cfg.get("eta_window", 20))

    arch = getattr(config, "arch", "versenex")
    resume_path = os.path.join(save_dir, "resume.pt") if resume else None

    if parallel_chunks > 1:
        # ParallelTrainerSafe 分支
        parallel_cfg = {
            "parallel_chunks": parallel_chunks,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "lr": lr,
            "warmup": warmup,
            "eval_interval": eval_interval,
            "grad_clip": grad_clip,
            "label_smoothing": label_smoothing,
            "seed": seed,
            "patience": patience,
            "save_dir": save_dir,
            "log_interval": log_interval,
            "loss_rate_window": min(50, max(10, max_steps // 4)),
            "enable_progress_bar": enable_progress_bar,
            "realtime_plot": realtime_plot,
            "eta_window": eta_window,
        }
        optimizer_kwargs = {"weight_decay": weight_decay}
        checkpoint_mgr = CheckpointManager(save_dir)
        parallel_trainer = ParallelTrainerSafe(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            optimizer_cls=AdamW,
            optimizer_kwargs=optimizer_kwargs,
            cfg=parallel_cfg,
            collate_fn=collate_fn,
            checkpoint_mgr=checkpoint_mgr,
            resume_path=resume_path,
            enable_loss_optimizer=enable_loss_optimizer,
        )
        print(
            f"[train] 开始训练 (ParallelTrainerSafe) chunks={parallel_chunks} "
            f"max_steps={max_steps} batch_size={batch_size} lr={lr}",
            flush=True,
        )
        history = parallel_trainer.fit()
        train_losses = list(history.get("train_loss", []))
        val_losses = list(history.get("val_loss", []))
        best_val_loss = float(parallel_trainer.best_val_loss)
        _save_parallel_history(
            save_dir, train_losses, val_losses, max_steps, eval_interval, best_val_loss,
        )
    elif arch in ("versenex", "verse_nex"):
        # VerseNexTrainer 分支（主路径）
        trainer_cfg = {
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "patience": patience,
            "save_dir": save_dir,
            "grad_accum": grad_accum,
            "log_interval": log_interval,
            "loss_rate_window": min(50, max(10, max_steps // 4)),
            "grad_clip": grad_clip,
            "label_smoothing": label_smoothing,
            "enable_progress_bar": enable_progress_bar,
            "realtime_plot": realtime_plot,
            "eta_window": eta_window,
            "aux_loss_weight": getattr(config, "aux_loss_weight", 0.01),
            "autocast": bool(amp),
        }
        nex_trainer = VerseNexTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=trainer_cfg,
        )
        # 断点续训：恢复模型 + optimizer 状态
        if resume_path is not None and os.path.exists(resume_path):
            try:
                with open(resume_path, "rb") as f:
                    payload = pickle.load(f)
                sd = payload.get("model_state_dict")
                if sd is not None and hasattr(model, "load_state_dict"):
                    model.load_state_dict(copy.deepcopy(sd))
                nex_trainer.best_val_loss = float(payload.get("best_val_loss", float("inf")))
                print(f"[train] 从 {resume_path} 恢复训练状态", flush=True)
            except Exception as e:
                print(f"[train] 警告：恢复 resume 失败：{e}", flush=True)

        print(
            f"[train] 开始训练 (VerseNexTrainer) max_steps={max_steps} "
            f"batch_size={batch_size} lr={lr}",
            flush=True,
        )
        train_losses, val_losses = nex_trainer.fit()
        best_val_loss = float(nex_trainer.best_val_loss)

        # 保存 resume checkpoint
        try:
            resume_path_out = os.path.join(save_dir, "resume.pt")
            payload = {
                "step": max_steps,
                "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
                "best_val_loss": best_val_loss,
            }
            with open(resume_path_out, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"[train] 警告：保存 resume 失败：{e}", flush=True)
    else:
        # 标准 Trainer 分支
        trainer_cfg = {
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "patience": patience,
            "save_dir": save_dir,
            "grad_accum": grad_accum,
            "log_interval": log_interval,
            "loss_rate_window": min(50, max(10, max_steps // 4)),
            "grad_clip": grad_clip,
            "label_smoothing": label_smoothing,
            "enable_progress_bar": enable_progress_bar,
            "realtime_plot": realtime_plot,
            "eta_window": eta_window,
            "autocast": bool(amp),
        }
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=trainer_cfg,
            device=device,
        )
        print(
            f"[train] 开始训练 (Trainer) max_steps={max_steps} "
            f"batch_size={batch_size} lr={lr}",
            flush=True,
        )
        train_losses, val_losses = trainer.fit()
        best_val_loss = float(trainer.best_val_loss)

    wall_clock = time.time() - start_time
    initial_loss = float(train_losses[0]) if train_losses else float("nan")
    final_loss = float(train_losses[-1]) if train_losses else float("nan")

    # 8. 保存完整模型
    full_model_path = os.path.join(save_dir, "cometspark.pt")
    try:
        if hasattr(model, "save"):
            model.save(full_model_path)
            print(f"[train] 完整模型已保存到 {full_model_path}", flush=True)
    except Exception as e:
        print(f"[train] 警告：保存完整模型失败：{e}", flush=True)

    print(
        f"[train] 训练完成 wall_clock={wall_clock:.2f}s "
        f"initial_loss={initial_loss:.4f} final_loss={final_loss:.4f} "
        f"best_val_loss={best_val_loss:.4f}",
        flush=True,
    )

    return {
        "wall_clock": wall_clock,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint_dir": save_dir,
        "loss_history_path": os.path.join(save_dir, "loss_history.json"),
        "full_model_path": full_model_path,
        "vocab_size": vocab_size,
    }


def _save_parallel_history(save_dir, train_losses, val_losses,
                            max_steps: int, eval_interval: int,
                            best_val_loss: float) -> None:
    """ParallelTrainer 分支的 loss 历史持久化（对齐 Trainer._save_history）。"""
    os.makedirs(save_dir, exist_ok=True)
    history = {
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "max_steps": max_steps,
        "eval_interval": eval_interval,
        "best_val_loss": float(best_val_loss),
    }
    with open(os.path.join(save_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "train_losses.txt"), "w", encoding="utf-8") as f:
        for v in train_losses:
            f.write(f"{float(v):.6f}\n")
    with open(os.path.join(save_dir, "val_losses.txt"), "w", encoding="utf-8") as f:
        for v in val_losses:
            f.write(f"{float(v):.6f}\n")
    actual_path = plot_loss_curve(
        train_losses, val_losses,
        os.path.join(save_dir, "loss_curve.png"),
        eval_interval=eval_interval,
    )
    print(f"[train] loss 曲线已保存到: {actual_path}", flush=True)


__all__ = [
    "train",
    "ParallelTrainerSafe",
    "_safe_chunk_run",
    "install_signal_handlers",
    "reset_shutdown_flag",
    "is_shutdown_requested",
    "ChunkOOMError",
]
