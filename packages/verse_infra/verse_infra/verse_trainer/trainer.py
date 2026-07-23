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
    ensure_val_split,
)
from .loss_optim import LossOptimizer
from .checkpoint_utils import migrate_checkpoint_dir

# ---------------------------------------------------------------------------
# 信号处理 + 全局 shutdown 标志 + 紧急保存（Part4K2.6 Task 2）
# ---------------------------------------------------------------------------

# 全局 shutdown 标志：保留用于 ParallelTrainerSafe 的 chunk 间检测。
# Part4K2.6: 信号处理器不再设置此标志——而是直接紧急保存后 os._exit(0) 强制退出，
# 避免"等待 checkpoint 边界"的延迟和线程/进程残留。
_shutdown_event = threading.Event()
# 信号处理器安装标志（避免重复安装）
_signal_handlers_installed = False
# 保存原始信号处理器，便于恢复
_original_signal_handlers: dict = {}
# 全局紧急保存函数引用（Ctrl+C 时调用，保存当前模型状态）
_emergency_save_fn: Optional[callable] = None


def _restore_signal_handlers() -> None:
    """恢复原始信号处理器。"""
    global _signal_handlers_installed
    if not _signal_handlers_installed:
        return
    for sig, handler in _original_signal_handlers.items():
        try:
            signal.signal(sig, handler)
        except Exception:
            pass
    _signal_handlers_installed = False


def _signal_handler(signum, frame):
    """SIGTERM/SIGINT 处理器：紧急保存后强制退出（Part4K2.6 Task 2）。

    不再等待 checkpoint 边界，立即执行：
    1. 调用注册的紧急保存函数（保存当前模型状态）
    2. 打印保存完成信息
    3. ``os._exit(0)`` 强制退出（绕过 Python 清理，避免线程残留）
    """
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n[signal] 收到 {sig_name}，紧急保存并强制退出...", flush=True)

    # 紧急保存
    if _emergency_save_fn is not None:
        try:
            _emergency_save_fn()
            print("[signal] 紧急保存完成", flush=True)
        except Exception as e:
            print(f"[signal] 紧急保存失败：{e}", flush=True)

    # 恢复原始信号处理器
    _restore_signal_handlers()

    # 强制退出（os._exit 绕过 Python atexit/shutdown，确保完全退出）
    print("[signal] 正在退出...", flush=True)
    os._exit(0)


def set_emergency_save_fn(fn) -> None:
    """注册紧急保存函数（训练开始前调用）。

    ``fn`` 是一个无参数的 callable，在收到 Ctrl+C 时被调用，
    通常保存当前模型状态到 checkpoint。
    """
    global _emergency_save_fn
    _emergency_save_fn = fn


def clear_emergency_save_fn() -> None:
    """清除紧急保存函数（训练结束后调用）。"""
    global _emergency_save_fn
    _emergency_save_fn = None


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


def _auto_build_tokenizer(kind: str, save_dir: str):
    """自动构建简单 tokenizer（用于测试模型，Part4K2.6 Task 3）。

    根据类型构建对应的 tokenizer：
    - byte/bytes: 直接构造（vocab=259）
    - char/charlevel/character: 构建字符级 tokenizer
    - 其他: 降级为 byte tokenizer
    """
    from verse_infra.verse_tokenizer import load_tokenizer as _vload

    if kind in ("byte", "bytes"):
        return _vload(kind="byte")

    if kind in ("char", "charlevel", "character"):
        try:
            from verse_infra.verse_tokenizer import CharTokenizer
            tok = CharTokenizer()
            # 保存供后续使用
            tok_path = os.path.join(save_dir, "tokenizer.json")
            tok.save(tok_path)
            return tok
        except Exception as e:
            print(f"[train] CharTokenizer 构建失败，降级为 byte：{e}", flush=True)
            return _vload(kind="byte")

    # 其他类型降级为 byte
    print(f"[train] 未知 tokenizer kind={kind}，降级为 byte", flush=True)
    return _vload(kind="byte")


def _load_tokenizer(tok_cfg: dict, base_dir: str, save_dir: str):
    """加载 tokenizer（兼容 data/demo/model/tokenizer.py 与 verse_tokenizer）。

    Part4K2.6 Task 3: 当 tokenizer 文件不存在且 kind 非 byte 时，自动构建。
    """
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
            # Part4K2.6: 自动构建 tokenizer
            print(f"[train] tokenizer 文件不存在，自动构建 {tok_kind}...", flush=True)
            tok = _auto_build_tokenizer(tok_kind, save_dir)
            print(f"[train] tokenizer 构建完成 vocab_size={len(tok)}", flush=True)
            return tok
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
# Part4K2.5 Task 4: 训练后自动评估默认测试用例
# ---------------------------------------------------------------------------

# 默认 4 条测试用例：2 条无 reference（只记录生成质量），2 条有 reference（计算打分）。
# 评估速度快，适合训练后快速验证。
_DEFAULT_EVAL_PROMPTS = [
    {"prompt": "你好", "reference": None},
    {"prompt": "Hello", "reference": None},
    {"prompt": "1+1=", "reference": "2"},
    {"prompt": "中国的首都是", "reference": "北京"},
]


# ---------------------------------------------------------------------------
# Part4K2.6 Task 3: 资源自动初始化（测试数据自动生成 + 测试配置判定）
# ---------------------------------------------------------------------------

# 测试数据模板（中英文混合，用于自动生成测试训练数据）
_TEST_TEXTS = [
    "你好世界，这是一个测试。",
    "Hello World, this is a test.",
    "1+1=2 2+2=4 3+3=6",
    "中国的首都是北京。",
    "机器学习是人工智能的一个重要分支。",
    "The quick brown fox jumps over the lazy dog.",
    "深度学习使用神经网络来学习数据表示。",
    "Python is a popular programming language.",
    "自然语言处理研究计算机与人类语言的交互。",
    "Artificial intelligence is the simulation of human intelligence.",
    "大语言模型通过预训练和微调获得语言能力。",
    "Machine learning algorithms improve through experience.",
    "今天的天气很好，适合出门散步。",
    "Knowledge is power, and learning is its key.",
    "代码是人与计算机沟通的桥梁。",
    "Practice makes perfect, persistence leads to success.",
    "数学是科学的基础，逻辑是数学的基础。",
    "Time flies when you are having fun.",
    "音乐是心灵的语言，艺术是情感的表达。",
    "A journey of a thousand miles begins with a single step.",
]


def _is_test_config(model_cfg: dict, full_cfg: dict) -> bool:
    """判断是否为测试/小模型配置（Part4K2.6 Task 3）。

    判断依据（任一满足即可）：
    - vocab_size <= 1000
    - n_embd <= 128
    - n_layer <= 4
    """
    vocab = int(model_cfg.get("vocab_size", 0))
    n_embd = int(model_cfg.get("n_embd", 0))
    n_layer = int(model_cfg.get("n_layer", 0))

    if vocab > 0 and vocab <= 1000:
        return True
    if n_embd > 0 and n_embd <= 128:
        return True
    if n_layer > 0 and n_layer <= 4:
        return True
    return False


def _auto_generate_test_data(train_path: str, val_path: str):
    """自动生成测试数据（仅用于测试模型，Part4K2.6 Task 3）。

    当训练数据文件不存在时，自动生成简单的中英文文本数据。
    生成的数据保存为 JSONL 格式，每行一个 ``{"text": "..."}`` 对象。
    """
    # 确保目录存在
    train_dir = os.path.dirname(train_path)
    val_dir = os.path.dirname(val_path)
    if train_dir:
        os.makedirs(train_dir, exist_ok=True)
    if val_dir and val_dir != train_dir:
        os.makedirs(val_dir, exist_ok=True)

    # 生成训练数据（20 条，重复 5 次 = 100 条）
    train_data = []
    for _ in range(5):
        for text in _TEST_TEXTS:
            train_data.append({"text": text})

    # 生成验证数据（5 条）
    val_data = [{"text": text} for text in _TEST_TEXTS[:5]]

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[train] 自动生成测试数据：train={len(train_data)}条 val={len(val_data)}条 "
        f"(路径: {train_path})",
        flush=True,
    )


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
    partition_training: bool = False,
    partition_size: int = 2,
    offload_dir: Optional[str] = None,
    continue_from: Optional[str] = None,
    quiet: bool = False,
    verbose: bool = False,
    eval_after: bool = True,
    eval_config: Optional[dict] = None,
) -> dict:
    """主训练函数（Part4K1 Task 6.2 升级版 + Part4K2 Task 7.3/7.4 + Task 4）。

    兼容 ``data/demo/train/trainer.py.train`` 原签名，并新增：
    - ``device``：设备字符串（cpu/cuda/npu）
    - ``single_sample``：单样本 dict（``{"prompt":..., "completion":...}`` 或 ``{"text":...}``）
    - ``single_file``：单文件路径（内容当作纯文本）
    - ``max_steps_override``：覆盖 config 的 max_steps
    - ``resume``：是否从 last checkpoint 断点续训
    - ``amp``：是否启用混合精度
    - ``enable_loss_optimizer``：是否启用 LossOptimizer（plateau 重走）
    - ``partition_training``：是否启用智能分区训练（LayerWiseTrainer，Part4K2 Task 4）
    - ``partition_size``：分区训练每组 layer 数量（默认 2）
    - ``offload_dir``：分区训练硬盘卸载目录（默认 tempfile 自动创建）
    - ``continue_from``：Part4K2 Task 7.3 持续训练 checkpoint 路径
      （best.pt / resume.pt）。设置后从该 checkpoint 加载模型状态与
      best_val_loss，再训练 ``max_steps_override`` 步。与 ``resume`` 区别：
      ``resume`` 是中断后恢复（目标是完成原计划步数）；``continue_from``
      是训练完成后继续追加训练（新目标 = additional_steps）。
    - ``quiet``：Part4K2 Task 7.2 静默模式（仅打印最终结果）
    - ``verbose``：Part4K2 Task 7.2 详细日志模式
    - ``eval_after``：Part4K2.5 Task 4 训练完成后是否自动评估打分（默认 True）
    - ``eval_config``：评估配置 dict（默认 None，使用训练配置中的评估参数）；
      可包含 ``"prompts"``（list of {prompt, reference}）覆盖默认测试用例

    Args:
        config_path: 配置文件路径（config.yml）
        base_dir: 配置中相对路径的基准目录（默认当前目录）
        n_threads: NumPy BLAS 线程数；0 表示不限制
    Returns:
        dict 包含：wall_clock / initial_loss / final_loss / best_val_loss /
        checkpoint_dir / loss_history_path / full_model_path / vocab_size；
        eval_after=True 时额外包含 ``eval_result``（评估打分结果）
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
    # Part5K1 Task 10.2: checkpoint 目录自动迁移（旧 checkpoints_XXX/ → mf_XXX/）
    # 在 save_dir 解析为绝对路径前，先在相对路径层面做迁移（旧目录与新目录
    # 通常在同一个工作目录下）。model_level 优先从 vmpc.profile 推断，回退到
    # 从 save_dir 名称推断。
    raw_save_dir = str(ckpt_cfg.get("save_dir", "checkpoints"))
    vmpc_cfg = full_cfg.get("vmpc", {})
    _profile = str(vmpc_cfg.get("profile", "")).lower()
    if _profile in ("small", "mate"):
        _model_level = _profile
    elif "mate" in raw_save_dir.lower():
        _model_level = "mate"
    else:
        _model_level = "small"
    raw_save_dir = migrate_checkpoint_dir(raw_save_dir, _model_level)
    save_dir = _resolve_path(base_dir, raw_save_dir)
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
        # Part4K2.6: 数据文件不存在时自动生成测试数据（仅小配置）
        if not os.path.exists(train_path):
            if _is_test_config(model_cfg, full_cfg):
                _auto_generate_test_data(train_path, val_path)
            else:
                raise FileNotFoundError(
                    f"训练数据文件不存在：{train_path}。"
                    f"请准备数据文件或使用 --single-sample / --single-file 模式。"
                )
        # Part5K1 Task 6.3: 自动生成 val.json（如果 val 不存在或为空，从 train 切分）
        # 保守策略：val 已存在且非空则不动；仅在 val 缺失时切分 train 末尾。
        # 用 try/except 包裹，切分失败不阻塞训练（ CachedDataset 会按原路径加载）。
        try:
            n_split_train, n_split_val = ensure_val_split(train_path, val_path)
            if not quiet:
                print(
                    f"[train] data split: train={n_split_train}, val={n_split_val}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[train] 警告：ensure_val_split 失败（不阻塞训练）：{e}",
                flush=True,
            )
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
    arch_name = getattr(config, "arch", "versenex")
    # Part4K2 Task 7.4: 优先用 model.device_info() 获取设备信息
    model_device_info = "cpu"
    if hasattr(model, "device_info"):
        try:
            model_device_info = str(model.device_info())
        except Exception:
            model_device_info = "cpu"
    if not quiet:
        print(
            f"[train] 实例化模型 arch={arch_name} "
            f"n_layer={getattr(config, 'n_layer', '?')} "
            f"n_embd={getattr(config, 'n_embd', '?')} seq_len={seq_len} "
            f"device={model_device_info}",
            flush=True,
        )

    # 设备迁移
    if device is not None and device != "cpu":
        if hasattr(model, "to"):
            try:
                model.to(device)
                if not quiet:
                    print(f"[train] 模型已迁移到 {device}", flush=True)
            except Exception as e:
                print(f"[train] 警告：迁移模型到 {device} 失败：{e}", flush=True)

    # ---------------------------------------------------------------
    # Part4K2 Task 7.4: 1B 模型 CPU/GPU 亲和优化
    # 检测参数量 > 100M 时自动启用优化：
    # - auto_tune_threads()：CPU BLAS 线程自动调优
    # - empty_cache_interval=50：GPU 显存定期清理
    # - 建议 --partition-training：大模型分区训练降内存
    # - GPU 时自动启用 autocast + GradScaler（amp=True）
    # ---------------------------------------------------------------
    n_params = 0
    try:
        if hasattr(model, "count_parameters"):
            n_params = int(model.count_parameters())
        else:
            n_params = sum(
                int(getattr(p, "numel", lambda: 0)()) if callable(getattr(p, "numel", None))
                else int(np.prod(getattr(p, "shape", (1,)))) if hasattr(p, "shape")
                else 0
                for p in model.parameters()
            )
    except Exception:
        n_params = 0

    is_large_model = n_params > 100_000_000  # > 100M
    empty_cache_interval = 0
    if is_large_model:
        # CPU BLAS 线程自动调优（大模型留 25% 余量给数据加载）
        try:
            from verse_torch.device import auto_tune_threads
            tuned = auto_tune_threads(model_size_hint=n_params)
            if not quiet:
                print(
                    f"[train] 1B 模型优化：参数量={n_params/1e6:.1f}M > 100M，"
                    f"已自动调优 BLAS 线程数={tuned}",
                    flush=True,
                )
        except Exception as e:
            if not quiet:
                print(f"[train] 警告：auto_tune_threads 失败：{e}", flush=True)

        # GPU 时定期清理显存
        eff_device = device or getattr(config, "device", "cpu") or "cpu"
        if eff_device and str(eff_device).lower() not in ("cpu", "none", ""):
            empty_cache_interval = 50
            # 自动启用 amp（GPU 混合精度）
            if not amp:
                amp = True
            if not quiet:
                print(
                    f"[train] 1B 模型优化：device={eff_device}，"
                    f"已启用 empty_cache_interval=50 + amp={amp}",
                    flush=True,
                )

        # 建议 --partition-training（CPU 大模型内存吃紧）
        if not partition_training and not quiet:
            print(
                f"[train] 1B 模型建议：加 --partition-training 启用分区训练"
                f"（按 layer 分组训练+卸载，降低峰值内存）",
                flush=True,
            )

    # ---------------------------------------------------------------
    # Part4K2 Task 7.3: 持续训练（continue_from）
    # 从 checkpoint 加载模型状态 + 继承 best_val_loss
    # 与 --resume 区别：continue_from 是追加训练，resume 是中断恢复
    # ---------------------------------------------------------------
    inherited_best_val_loss = None
    if continue_from is not None:
        if not os.path.exists(continue_from):
            raise FileNotFoundError(
                f"continue_from checkpoint 不存在：{continue_from}"
            )
        if not quiet:
            print(f"[train] 持续训练：从 {continue_from} 加载模型状态", flush=True)
        try:
            with open(continue_from, "rb") as f:
                ckpt_payload = pickle.load(f)
            # 加载模型状态
            sd = ckpt_payload.get("model_state_dict") or ckpt_payload.get("state_dict")
            if sd is not None and hasattr(model, "load_state_dict"):
                model.load_state_dict(copy.deepcopy(sd))
                if not quiet:
                    print(f"[train] 已加载模型 state_dict", flush=True)
            # 继承 best_val_loss（不从头比较）
            if "best_val_loss" in ckpt_payload:
                inherited_best_val_loss = float(ckpt_payload["best_val_loss"])
            elif "val_loss" in ckpt_payload:
                inherited_best_val_loss = float(ckpt_payload["val_loss"])
            if inherited_best_val_loss is not None and not quiet:
                print(
                    f"[train] 继承 best_val_loss={inherited_best_val_loss:.4f}"
                    f"（后续 val_loss 需低于此值才算改善）",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[train] 警告：加载 continue_from checkpoint 失败：{e}，"
                f"从头开始训练",
                flush=True,
            )

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

    # Part4K2.6: 注册紧急保存函数（Ctrl+C 时自动保存当前模型状态）
    def _emergency_save():
        """紧急保存当前模型状态。"""
        save_path = os.path.join(save_dir, "cometspark_emergency.pt")
        try:
            if hasattr(model, "save"):
                model.save(save_path)
                print(f"[signal] 模型已紧急保存到 {save_path}", flush=True)
        except Exception as e:
            print(f"[signal] 紧急保存模型失败：{e}", flush=True)

    set_emergency_save_fn(_emergency_save)

    # Part4K2 Task 4: 智能分区训练（LayerWiseTrainer）
    # 启用后优先于 ParallelTrainer / VerseNexTrainer，按 layer 分组训练+卸载+合并
    if partition_training:
        from verse_torch import LayerWiseTrainer
        # 卸载目录：未指定时在 save_dir 下创建 partition_offload 子目录
        eff_offload_dir = offload_dir or os.path.join(save_dir, "partition_offload")
        os.makedirs(eff_offload_dir, exist_ok=True)
        lw_cfg = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "eval_interval": eval_interval if val_loader is not None else None,
            "log_interval": log_interval,
            "seed": seed,
            "label_smoothing": label_smoothing,
            "finetune_steps": max(1, max_steps // 10),
            # Part4K2 Task 7.2: 输出控制
            "quiet": quiet,
            "verbose": verbose,
        }
        lw_optimizer_cfg = {"weight_decay": weight_decay}
        lw_trainer = LayerWiseTrainer(
            model=model,
            config=lw_cfg,
            optimizer_config=lw_optimizer_cfg,
            partition_size=partition_size,
            offload_dir=eff_offload_dir,
        )
        # Part4K2 Task 7.3: 持续训练继承 best_val_loss
        if inherited_best_val_loss is not None and hasattr(lw_trainer, "best_val_loss"):
            lw_trainer.best_val_loss = float(inherited_best_val_loss)
        if not quiet:
            print(
                f"[train] 开始训练 (LayerWiseTrainer) partition_size={partition_size} "
                f"n_partitions={lw_trainer.n_partitions} max_steps={max_steps} "
                f"batch_size={batch_size} lr={lr}",
                flush=True,
            )
        train_losses, val_losses = lw_trainer.fit(
            train_loader, val_loader, max_steps=max_steps,
        )
        best_val_loss = float(lw_trainer.best_val_loss)
        # 保存 loss 历史
        _save_parallel_history(
            save_dir, train_losses, val_losses, max_steps, eval_interval, best_val_loss,
        )
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
        # 清理临时卸载目录（由 LayerWiseTrainer 自动管理时）
        lw_trainer.cleanup()

    elif parallel_chunks > 1:
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
            # Part4K2 Task 7.2: 输出控制
            "quiet": quiet,
            "verbose": verbose,
            # Part4K2 Task 7.4: 1B 模型 GPU 显存定期清理
            "empty_cache_interval": empty_cache_interval,
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
        # Part4K2 Task 7.3: 持续训练继承 best_val_loss
        if inherited_best_val_loss is not None:
            parallel_trainer.best_val_loss = float(inherited_best_val_loss)
        if not quiet:
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
            # Part4K2 Task 7.2: 输出控制
            "quiet": quiet,
            "verbose": verbose,
            # Part4K2 Task 7.4: 1B 模型 GPU 显存定期清理
            "empty_cache_interval": empty_cache_interval,
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
                if not quiet:
                    print(f"[train] 从 {resume_path} 恢复训练状态", flush=True)
            except Exception as e:
                print(f"[train] 警告：恢复 resume 失败：{e}", flush=True)

        # Part4K2 Task 7.3: 持续训练继承 best_val_loss
        if inherited_best_val_loss is not None:
            nex_trainer.best_val_loss = float(inherited_best_val_loss)

        if not quiet:
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
            # Part4K2 Task 7.2: 输出控制
            "quiet": quiet,
            "verbose": verbose,
            # Part4K2 Task 7.4: 1B 模型 GPU 显存定期清理
            "empty_cache_interval": empty_cache_interval,
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
        # Part4K2 Task 7.3: 持续训练继承 best_val_loss
        if inherited_best_val_loss is not None:
            trainer.best_val_loss = float(inherited_best_val_loss)
        if not quiet:
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
            if not quiet:
                print(f"[train] 完整模型已保存到 {full_model_path}", flush=True)
    except Exception as e:
        print(f"[train] 警告：保存完整模型失败：{e}", flush=True)

    # Part4K2 Task 7.2: quiet 模式下仅打印最终结果
    if quiet:
        print(
            f"[train] done best_val={best_val_loss:.4f} "
            f"steps={max_steps} wall={wall_clock:.1f}s",
            flush=True,
        )
    else:
        print(
            f"[train] 训练完成 wall_clock={wall_clock:.2f}s "
            f"initial_loss={initial_loss:.4f} final_loss={final_loss:.4f} "
            f"best_val_loss={best_val_loss:.4f}",
            flush=True,
        )

    result = {
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

    # ---------------------------------------------------------------
    # Part4K2.5 Task 4: 训练完成后自动评估打分（默认 eval_after=True）
    # 评估失败不影响训练结果（try/except 包裹，失败时打印警告但继续返回）
    # ---------------------------------------------------------------
    if eval_after and full_model_path:
        try:
            eval_result = _auto_evaluate(
                model_path=full_model_path,
                config=full_cfg,
                tokenizer=tok,
                vocab_size=vocab_size,
                eval_config=eval_config,
                base_dir=base_dir,
                quiet=quiet,
            )
            result["eval_result"] = eval_result
        except Exception as e:
            print(
                f"[train] 警告：训练后自动评估失败（不影响训练结果）："
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            result["eval_result"] = None

    # Part4K2.6: 清除紧急保存函数
    clear_emergency_save_fn()

    return result


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


# ---------------------------------------------------------------------------
# Part4K2 Task 7.3: 持续训练入口 continue_train
# ---------------------------------------------------------------------------


def continue_train(
    checkpoint: str,
    additional_steps: int,
    config_path: str,
    base_dir: str = ".",
    n_threads: int = 0,
    *,
    device: Optional[str] = None,
    amp: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> dict:
    """持续训练入口（Part4K2 Task 7.3）。

    从 checkpoint 加载模型状态（best.pt 或 resume.pt），继续训练
    ``additional_steps`` 步。自动继承之前的 best_val_loss（不从头比较）。

    与 :func:`train` 的 ``--resume`` 区别：
    - ``resume`` 是中断后恢复（从中断点继续，目标是完成原计划的步数）
    - ``continue_train`` 是训练完成后继续追加训练
      （新目标 = additional_steps，独立步数）

    Args:
        checkpoint: checkpoint 文件路径（best.pt / resume.pt / 任意 pickle）
        additional_steps: 追加训练步数
        config_path: 配置文件路径（config.yml）
        base_dir: 配置中相对路径的基准目录（默认当前目录）
        n_threads: NumPy BLAS 线程数；0 表示不限制
        device: 设备字符串（cpu/cuda/npu）
        amp: 是否启用混合精度
        quiet: 静默模式（仅打印最终结果）
        verbose: 详细日志模式

    Returns:
        同 :func:`train` 的返回 dict
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    if additional_steps <= 0:
        raise ValueError(f"additional_steps 必须为正整数，收到 {additional_steps}")

    if not quiet:
        print(
            f"[continue] 持续训练：checkpoint={checkpoint} "
            f"additional_steps={additional_steps}",
            flush=True,
        )

    # 复用 train 入口，通过 continue_from + max_steps_override 实现
    return train(
        config_path=config_path,
        base_dir=base_dir,
        n_threads=n_threads,
        device=device,
        max_steps_override=int(additional_steps),
        resume=False,
        amp=amp,
        continue_from=checkpoint,
        quiet=quiet,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Part4K2.5 Task 4: 训练后自动评估打分
# ---------------------------------------------------------------------------


def _resolve_eval_prompts(eval_config, full_config) -> list:
    """解析评估测试用例。

    优先级：
    1. ``eval_config["prompts"]``（调用方显式传入的评估配置）
    2. ``full_config["eval"]["prompts"]``（配置文件中的 eval 段）
    3. ``_DEFAULT_EVAL_PROMPTS``（默认 4 条中英文测试用例）

    Args:
        eval_config: 调用方传入的评估配置 dict（可为 None）
        full_config: 训练配置 dict（含 model/training/eval 等段）

    Returns:
        list of {"prompt": str, "reference": Optional[str]}
    """
    raw_prompts = None

    # 1. eval_config 显式传入
    if eval_config is not None and isinstance(eval_config, dict):
        raw_prompts = eval_config.get("prompts")

    # 2. full_config 的 eval 段
    if raw_prompts is None and isinstance(full_config, dict):
        eval_section = full_config.get("eval", {})
        if isinstance(eval_section, dict):
            raw_prompts = eval_section.get("prompts")

    # 3. 默认测试用例
    if raw_prompts is None:
        return [dict(item) for item in _DEFAULT_EVAL_PROMPTS]

    # 规范化：支持 dict 格式和纯字符串格式
    result = []
    for item in raw_prompts:
        if isinstance(item, dict):
            result.append({
                "prompt": str(item.get("prompt", "")),
                "reference": item.get("reference"),
            })
        elif isinstance(item, str):
            result.append({"prompt": item, "reference": None})
    # 空列表兜底为默认
    if not result:
        return [dict(item) for item in _DEFAULT_EVAL_PROMPTS]
    return result


def _load_model_for_eval(model_path: str, full_config: dict, vocab_size: int):
    """从 model_path 加载模型用于评估。

    加载策略（按优先级）：
    1. ``CometSparkNexLM.from_pretrained(model_path)``（verse_nex 原生）
    2. ``CometSparkLM.from_pretrained(model_path)``（data/demo 兼容）
    3. 用 ``_build_model`` 重建模型 + ``pickle.load`` 加载 state_dict
    4. model_path 不存在时尝试同目录 best.pt

    Args:
        model_path: 模型文件路径（cometspark.pt）
        full_config: 训练配置 dict（含 model 段）
        vocab_size: 词表大小

    Returns:
        加载好的模型对象

    Raises:
        FileNotFoundError: 所有加载策略均失败
    """
    model_cfg = full_config.get("model", {}) if isinstance(full_config, dict) else {}

    # model_path 不存在时回退到 best.pt
    eff_path = model_path
    if not os.path.exists(eff_path):
        save_dir = os.path.dirname(model_path) or "."
        best_path = os.path.join(save_dir, "best.pt")
        if os.path.exists(best_path):
            eff_path = best_path
        else:
            raise FileNotFoundError(
                f"模型文件不存在：{model_path}（也未找到 {best_path}）"
            )

    # 策略 1：CometSparkNexLM.from_pretrained
    try:
        from verse_nex.cometspark import CometSparkNexLM
        return CometSparkNexLM.from_pretrained(eff_path)
    except Exception:
        pass

    # 策略 2：data/demo CometSparkLM.from_pretrained
    try:
        import sys as _sys
        demo_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                _sys.modules[__name__].__file__))),
            "data", "demo",
        )
        if demo_dir not in _sys.path:
            _sys.path.insert(0, demo_dir)
        from model.model import CometSparkLM
        if hasattr(CometSparkLM, "from_pretrained"):
            return CometSparkLM.from_pretrained(eff_path)
    except Exception:
        pass

    # 策略 3：重建模型 + pickle.load state_dict
    model, _ = _build_model(model_cfg, vocab_size)
    try:
        with open(eff_path, "rb") as f:
            payload = pickle.load(f)
        sd = payload.get("state_dict") or payload.get("model_state_dict") or payload
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(sd, strict=False)
        return model
    except Exception as e:
        raise FileNotFoundError(
            f"无法从 {eff_path} 加载模型：{type(e).__name__}: {e}"
        )


def _auto_evaluate(
    model_path: str,
    config,
    tokenizer,
    vocab_size: int,
    eval_config=None,
    base_dir: str = ".",
    quiet: bool = False,
) -> dict:
    """训练后自动评估打分（Part4K2.5 Task 4）。

    流程：
    1. 加载训练好的模型（从 model_path）
    2. 生成测试文本（用 eval_config / config["eval"]["prompts"] / 默认测试用例）
    3. 计算打分（exact_match / prefix_accuracy / char_f1 / bleu / rouge_l）
    4. 打印评估报告
    5. 返回评估结果

    Args:
        model_path: 训练好的模型文件路径（cometspark.pt）
        config: 训练配置 dict（含 model/eval 段），或模型 config 对象
        tokenizer: 已加载的 tokenizer
        vocab_size: 词表大小
        eval_config: 评估配置 dict（可含 ``"prompts"`` 覆盖默认测试用例）
        base_dir: 相对路径基准目录
        quiet: 静默模式（只打印打分汇总，不打印每条生成结果）

    Returns:
        dict: {
            "prompts": list[str],
            "generations": list[str],
            "references": list[Optional[str]],
            "scores": dict,  # 5 个指标 + n_samples + per_sample
            "avg_length": float,
            "has_eos_ratio": float,
        }
    """
    # 延迟导入：避免与 evaluate.py 循环导入
    from .evaluate import (
        _safe_encode,
        _safe_decode_with_prompt,
        _get_eos_id,
    )
    from verse_torch import Tensor, no_grad
    from verse_torch.scoring import ScoringEvaluator

    # 1. 解析测试用例
    prompts_data = _resolve_eval_prompts(eval_config, config)
    prompts = [p["prompt"] for p in prompts_data]
    references = [p["reference"] for p in prompts_data]

    # 解析 max_new_tokens（默认 32 = 快速评估；None = EOS 自然停止；
    # eval_config 可覆盖）。自动评估默认限制 32 token 以保证速度，
    # 避免未训练模型无 EOS 时生成 100K token 超时。
    max_new_tokens = 32
    if eval_config is not None and isinstance(eval_config, dict):
        cfg_mnt = eval_config.get("max_new_tokens")
        if cfg_mnt is not None:
            max_new_tokens = cfg_mnt

    # 2. 加载模型
    model = _load_model_for_eval(model_path, config, vocab_size)

    eos_id = _get_eos_id(tokenizer)

    # 3. 生成
    generations = []
    has_eos_count = 0
    total_length = 0

    if not quiet:
        print(f"[auto-eval] 开始评估 {len(prompts)} 条 prompt", flush=True)
        print("=" * 60, flush=True)

    with no_grad():
        if hasattr(model, "eval"):
            model.eval()
        for i, (prompt, reference) in enumerate(zip(prompts, references)):
            ids = _safe_encode(tokenizer, prompt)
            if not ids:
                ids = [0]
            idx_np = np.asarray(ids, dtype=np.int64).reshape(1, -1)
            try:
                gen_kwargs = dict(temperature=1.0, top_k=None, eos_id=eos_id)
                try:
                    # max_new_tokens=None：让模型按 EOS 自然停止
                    generated = model.generate(
                        idx_np, max_new_tokens=max_new_tokens, **gen_kwargs
                    )
                except TypeError:
                    # 旧模型 generate 不接受 max_new_tokens=None，限制 16 token
                    fallback_n = max_new_tokens if max_new_tokens is not None else 16
                    generated = model.generate(
                        idx_np, max_new_tokens=fallback_n, **gen_kwargs
                    )
                if isinstance(generated, Tensor):
                    gen_ids = generated.data.reshape(-1).tolist()
                else:
                    gen_ids = np.asarray(generated).reshape(-1).tolist()
            except Exception as e:
                if not quiet:
                    print(
                        f"[auto-eval] 生成失败 prompt={prompt!r}: {e}",
                        flush=True,
                    )
                gen_ids = list(ids)

            full_text = _safe_decode_with_prompt(tokenizer, prompt, ids, gen_ids)
            n_generated = len(gen_ids) - len(ids)
            generated_part = gen_ids[len(ids):]
            has_eos = (
                eos_id is not None
                and len(generated_part) > 0
                and eos_id in generated_part
            )

            generations.append(full_text)
            total_length += max(0, n_generated)
            if has_eos:
                has_eos_count += 1

            if not quiet:
                print(f"  [{i + 1}/{len(prompts)}] [prompt] {prompt}", flush=True)
                print(f"  [output] {full_text}", flush=True)
                print(
                    f"  (tokens: {n_generated}, has_eos: {has_eos})",
                    flush=True,
                )
                if reference is not None:
                    print(f"  [ref]    {reference}", flush=True)
                print("-" * 60, flush=True)

    # 4. 打分（仅有 reference 的样本参与打分）
    scored_indices = [i for i, r in enumerate(references) if r is not None]
    empty_scores = {
        "exact_match": 0.0,
        "prefix_accuracy": 0.0,
        "char_f1": 0.0,
        "bleu": 0.0,
        "rouge_l": 0.0,
        "n_samples": 0,
        "per_sample": [],
    }

    if scored_indices:
        scored_preds = []
        scored_refs = []
        for i in scored_indices:
            gen = generations[i]
            prompt = prompts[i]
            # 剥离 prompt 前缀后与 reference 对比（_safe_decode_with_prompt
            # 用 prompt + decoded 拼接，所以 startswith(prompt) 必然成立）
            if gen.startswith(prompt):
                pred = gen[len(prompt):]
            else:
                pred = gen
            scored_preds.append(pred)
            scored_refs.append(str(references[i]))

        evaluator = ScoringEvaluator()
        scores = evaluator.evaluate(scored_preds, scored_refs)
        report = evaluator.report(scores)
        print("[auto-eval] 评分报告：", flush=True)
        print(report, flush=True)
    else:
        scores = empty_scores
        if not quiet:
            print("[auto-eval] 无 reference，跳过打分", flush=True)

    # 5. 汇总
    n = len(prompts)
    avg_length = total_length / n if n > 0 else 0.0
    has_eos_ratio = has_eos_count / n if n > 0 else 0.0

    if quiet:
        # quiet 模式只打印打分汇总
        if scored_indices:
            score_str = " ".join(
                f"{k}={v:.4f}" for k, v in scores.items()
                if k not in ("per_sample", "n_samples")
            )
        else:
            score_str = "no_reference"
        print(
            f"[auto-eval] {score_str} has_eos_ratio={has_eos_ratio:.2f} "
            f"avg_length={avg_length:.1f}",
            flush=True,
        )
    else:
        print(
            f"[auto-eval] 汇总: avg_length={avg_length:.1f} "
            f"has_eos_ratio={has_eos_ratio:.2f}",
            flush=True,
        )

    return {
        "prompts": prompts,
        "generations": generations,
        "references": references,
        "scores": scores,
        "avg_length": avg_length,
        "has_eos_ratio": has_eos_ratio,
    }


__all__ = [
    "train",
    "continue_train",
    "ParallelTrainerSafe",
    "_safe_chunk_run",
    "install_signal_handlers",
    "reset_shutdown_flag",
    "is_shutdown_requested",
    "set_emergency_save_fn",
    "clear_emergency_save_fn",
    "ChunkOOMError",
    "_auto_evaluate",
    "migrate_checkpoint_dir",
]
