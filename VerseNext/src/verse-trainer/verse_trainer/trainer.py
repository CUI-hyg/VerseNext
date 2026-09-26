"""Trainer：高性能训练循环。

性能优化清单：
- AMP 混合精度（bf16 优先，fallback fp16 + GradScaler）
- 梯度累积（等效大 batch，减少 optimizer step 次数）
- 梯度裁剪
- torch.compile（默认关闭，CPU/调试场景可关）
- fused AdamW（optim.py）
- TF32 matmul（CUDA 上自动启用）

扩展点：通过 ``EventBus`` 暴露 ``train/step_end`` 等事件，callbacks 与 RSI
无需侵入训练循环。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from verse_core import EventBus, get_logger
from verse_core.config import BaseConfig
from verse_nn.transformer import (
    VerseTransformer,
    VerseTransformerConfig,
    build_model,
    count_parameters,
)
from verse_trainer.data import DataConfig, TokenBatchIterator, fingerprint_source
from verse_trainer.optim import OptimizerConfig, build_lr_scheduler, build_optimizer
from verse_trainer.resources import MemoryGuard, ResourceConfig, apply_resource_limits

logger = get_logger("trainer")


@dataclass
class TrainerConfig(BaseConfig):
    """训练配置（含模型、数据、优化器子配置）。"""

    model: VerseTransformerConfig = field(default_factory=VerseTransformerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    model_name: str = "verse_transformer"
    """注册中心中的模型名；RSI 变异架构时可换用其他已注册模型。"""

    steps: int = 500
    log_every: int = 20
    eval_every: int = 100
    eval_batches: int = 10
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 500

    device: str = "auto"
    dtype: str = "auto"
    """``auto``: bf16 优先；也可显式 ``bf16``/``fp16``/``fp32``。"""
    compile: bool = False
    """是否启用 torch.compile（首次编译有耗时，长训建议开启）。"""
    grad_accum_steps: int = 1
    seed: int = 42
    num_threads: int = 0
    """CPU 训练线程数；0 = 自动（按 ``resources.cpu_fraction`` 取可用核数的比例）。"""
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    """CPU/GPU/内存占用比例配置（默认各 50% 上限）。"""
    lora: "LoRAConfig | None" = None
    """非 None 时对模型应用 LoRA 并冻结主干（参数高效微调）。"""
    loss_plot: bool = False
    """训练结束时在 ``ckpt_dir`` 下绘制 loss 曲线图（``loss_curve.png``）。"""
    save_best_only: bool = False
    """True 时周期 checkpoint 只在监控 loss 优于历史最优时保存（智能保存）。"""
    save_last: bool = False
    """``save_best_only=True`` 时，额外每个 ``ckpt_every`` 覆盖保存
    ``{model_name}_last.pt`` 作为崩溃续训兜底。

    ``save_best_only=False`` 时 ``step_N.pt`` 已是最新状态，本项不额外写盘。
    finalize 时该文件随中间 checkpoint 一并清理。
    """
    early_stop_patience: int = 0
    """监控 loss 连续 N 次评估未创新低则提前停止训练；0 = 关闭。"""

    stage: str = "pretrain"
    """checkpoint 阶段标签：``pretrain`` / ``finetune`` / ``sft`` / ``posttrain``。

    用于产物命名（``{model}_{stage}_final.pt``）与区分不同训练阶段，
    避免预训练/微调/后训练产物互相覆盖。
    """
    cleanup_intermediate: bool = True
    """训练结束调用 :meth:`Trainer.finalize` 时删除周期 checkpoint，只保留 final。"""
    final_keep_optimizer: bool = True
    """final checkpoint 是否保留优化器状态（True 可续训，体积约为仅权重的 2×）。"""
    plot_every_eval: bool = True
    """每次评估都刷新 loss 曲线图（默认开启，便于实时观察）。"""
    loss_chunk_size: int = 0
    """>0 时按序列维分块计算 lm_head + CE，降低大词表下的 logits 内存峰值。"""

    def validate(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps 必须 >= 1")


class Trainer:
    """训练执行器。

    用法::

        trainer = Trainer(TrainerConfig(steps=200))
        result = trainer.train(train_tokens, eval_tokens)

    ``train`` 的数据源既可以是 token id 列表（自动包装为随机窗口迭代器），
    也可以是实现了 ``next_batch() -> (inputs, targets)`` 的数据集对象
    （如 SFTDataset）。
    """

    def __init__(self, config: TrainerConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        self.autocast_enabled = self.dtype != torch.float32

        torch.manual_seed(config.seed)
        self.memory_guard = MemoryGuard(config.resources)
        self._apply_resource_optimizations()

        self.model = build_model(config.model_name, config.model.to_dict()).to(self.device)
        if config.lora is not None:
            from verse_nn.lora import LoRAConfig, apply_lora, mark_only_lora_trainable

            lora_cfg = config.lora
            if isinstance(lora_cfg, dict):
                lora_cfg = LoRAConfig.from_dict(lora_cfg)
            apply_lora(self.model, lora_cfg)
            mark_only_lora_trainable(self.model)
            logger.info(
                "LoRA 已应用：可训练参数 %.2fM / 总参数 %.2fM",
                sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6,
                self.model.num_parameters() / 1e6,
            )
        if config.compile:
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(self.model, config.optim)
        self.scheduler = build_lr_scheduler("warmup_cosine", self.optimizer, config.optim)
        self.grad_scaler = torch.amp.GradScaler(
            self.device.type, enabled=(self.device.type == "cuda" and self.dtype == torch.float16)
        )
        self.global_step = 0
        # loss 历史 / 智能保存与早停状态
        self._train_hist: list[tuple[int, float]] = []
        self._eval_hist: list[tuple[int, float]] = []
        self._best_loss = math.inf
        self._bad_evals = 0
        self.early_stopped = False
        # 断点续训状态：累计 token 数、数据指纹、最近数据迭代器、待恢复的数据 RNG
        self._tokens_seen = 0
        self._data_fingerprint: dict | None = None
        self._resume_config: dict | None = None
        self._train_iter = None
        self._pending_data_rng = None
        # 复用同一张 matplotlib 图，支持每个 eval 刷新 loss_curve.png
        self._plot_fig = None
        self._plot_ax = None
        self._plot_disabled = False

    def _apply_resource_optimizations(self) -> None:
        """应用资源上限：CPU 线程/denormal、CUDA 显存占比、内存软监控。"""
        threads = apply_resource_limits(
            self.config.num_threads, self.config.resources, self.device
        )
        soft = self.memory_guard.soft_limit
        logger.info(
            "资源限制: %d 线程 (cpu_fraction=%.0f%%), 内存软上限 %s",
            threads, self.config.resources.cpu_fraction * 100,
            f"{soft / 1e9:.2f}GB" if soft else "未知",
        )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """解析设备：``auto`` 时按 NPU > CUDA > ROCm > XPU > MPS > CPU 优先级选择。

        设备探测与优先级统一交给 :mod:`verse_nn.devices`，避免此处散落平台分支；
        显式指定但不可用时由 ``detect_backend`` 回退 CPU 并告警。
        """
        from verse_nn.devices import detect_backend

        if device != "auto":
            return torch.device(device)
        return torch.device(detect_backend("auto").device_type)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == "fp32":
            return torch.float32
        if dtype in ("bf16", "fp16"):
            return getattr(torch, dtype)
        # auto：由设备后端决定推荐的混合精度 dtype（bf16 优先，无 GradScaler 需求）
        from verse_nn.devices import detect_backend

        backend = detect_backend(self.device)
        recommended = backend.default_autocast_dtype()
        return recommended if recommended is not None else torch.float32

    def _autocast(self):
        from verse_nn.devices import detect_backend

        # 设备后端负责选对 autocast 设备类型（cuda / npu / xpu / cpu）
        return detect_backend(self.device).autocast(
            self.dtype, enabled=self.autocast_enabled
        )

    def _as_batch_source(self, source):
        """token 列表/数组 -> TokenBatchIterator；数据集对象原样返回。"""
        import numpy as np

        if isinstance(source, (list, tuple, torch.Tensor, np.ndarray)):
            return TokenBatchIterator(source, self.config.data, self.device)
        if not hasattr(source, "next_batch"):
            raise TypeError("数据源需为 token 列表或实现 next_batch() 的对象")
        return source

    # ------------------------------------------------------------- resume state

    def _rng_state(self) -> dict:
        """收集全局 RNG 状态（python/torch/numpy/cuda）与数据采样器状态。

        数据采样器的 ``torch.Generator`` 状态单独保存，续训时回填到新建的
        迭代器上，保证换数据前已确定的采样序列可复现（不折损）。
        """
        state: dict = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        try:
            state["numpy"] = np.random.get_state()
        except Exception:  # pragma: no cover - 极端环境下 numpy 不可用
            pass
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        generator = getattr(self._train_iter, "_generator", None)
        if generator is not None:
            state["data"] = generator.get_state()
        return state

    def _training_state(self) -> dict:
        """可续训的完整训练状态（除权重/优化器外的一切）。"""
        return {
            "scheduler_state": {
                "warmup_steps": self.config.optim.warmup_steps,
                "max_steps": self.config.optim.max_steps,
                "min_lr_ratio": self.config.optim.min_lr_ratio,
                "base_lrs": list(self.scheduler.base_lrs),
            },
            "rng_state": self._rng_state(),
            "train_state": {
                "best_loss": self._best_loss,
                "bad_evals": self._bad_evals,
                "early_stopped": self.early_stopped,
                "tokens_seen": self._tokens_seen,
                "train_hist": list(self._train_hist),
                "eval_hist": list(self._eval_hist),
            },
            "data_fingerprint": self._data_fingerprint,
            "config": self.config.to_dict(),
        }

    def _restore_training_state(self, payload: dict) -> None:
        """恢复监控/早停/历史/RNG/数据指纹等续训状态。"""
        train_state = payload.get("train_state") or {}
        self._best_loss = train_state.get("best_loss", math.inf)
        self._bad_evals = int(train_state.get("bad_evals", 0))
        self.early_stopped = bool(train_state.get("early_stopped", False))
        self._tokens_seen = int(train_state.get("tokens_seen", 0))
        self._train_hist = [tuple(x) for x in train_state.get("train_hist", [])]
        self._eval_hist = [tuple(x) for x in train_state.get("eval_hist", [])]
        self._data_fingerprint = payload.get("data_fingerprint")
        self._resume_config = payload.get("config")

        scheduler_state = payload.get("scheduler_state") or {}
        saved_max = scheduler_state.get("max_steps")
        if saved_max is not None and saved_max != self.config.optim.max_steps:
            logger.warning(
                "续训 LR 地平线不一致：checkpoint max_steps=%s，当前配置 max_steps=%s；"
                "将以当前配置为准（如需延续原曲线请对齐 optim.max_steps）",
                saved_max, self.config.optim.max_steps,
            )

        rng = payload.get("rng_state") or {}
        if "python" in rng:
            random.setstate(rng["python"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "cuda" in rng and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
        self._pending_data_rng = rng.get("data")
        logger.info(
            "已恢复训练状态：step=%d, best_loss=%s, bad_evals=%d, tokens_seen=%d",
            self.global_step, self._best_loss, self._bad_evals, self._tokens_seen,
        )

    def _bind_train_iter(self, train_iter) -> None:
        """绑定训练迭代器：记录引用并回填待恢复的数据采样 RNG。"""
        self._train_iter = train_iter
        generator = getattr(train_iter, "_generator", None)
        if generator is not None and self._pending_data_rng is not None:
            generator.set_state(self._pending_data_rng)
            logger.info("已恢复数据采样 RNG，续训数据顺序与断点一致")
        self._pending_data_rng = None
        self._data_fingerprint = fingerprint_source(train_iter)

    # ------------------------------------------------------------------ train

    def train(self, train_source, eval_source=None) -> dict:
        """执行训练，返回包含最终指标与最优步的摘要。

        Args:
            train_source: token id 列表（包装为随机窗口迭代器），
                或实现 ``next_batch() -> (inputs, targets)`` 的数据集对象
                （如 :class:`SFTDataset`）。
            eval_source: 同上；用于周期性评估。
        """
        cfg = self.config
        self.model.train()
        train_iter = self._as_batch_source(train_source)
        self._bind_train_iter(train_iter)
        eval_iter = self._as_batch_source(eval_source) if eval_source is not None else None

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        t0 = time.perf_counter()
        tokens_seen = self._tokens_seen      # 累计（跨续训）
        run_tokens = 0                        # 本次 run 新增

        for step in range(self.global_step, cfg.steps):
            if getattr(self, "_stop_requested", False):
                logger.info("收到停止请求，提前结束训练 (step %d)", self.global_step)
                break
            lr = self.scheduler.step(step)
            self.optimizer.zero_grad(set_to_none=True)

            loss_accum = 0.0
            for micro in range(cfg.grad_accum_steps):
                inputs, targets = train_iter.next_batch()
                with self._autocast():
                    # 训练不需要 logits：跳过返回 + 分块 CE，降低 logits 峰值
                    _, loss = self.model(
                        inputs, targets,
                        return_logits=False, loss_chunk_size=cfg.loss_chunk_size,
                    )
                (loss / cfg.grad_accum_steps).backward()
                loss_accum += loss.item() / cfg.grad_accum_steps
                run_tokens += inputs.numel()
                tokens_seen += inputs.numel()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.global_step = step + 1
            self._tokens_seen = tokens_seen

            if self.global_step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                tps = run_tokens / max(elapsed, 1e-8)
                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e | %.0f tokens/s",
                    self.global_step, cfg.steps, loss_accum, lr, tps,
                )
                self._train_hist.append((self.global_step, loss_accum))
                self.event_bus.emit(
                    "train/step_end", step=self.global_step, loss=loss_accum,
                    lr=lr, tokens_per_second=tps,
                )
                self.memory_guard.check(self.global_step)
                # 无评估集时退化为用训练 loss 做监控（智能保存/早停）
                if eval_iter is None:
                    self._update_monitor(loss_accum, self.global_step)

            if eval_iter is not None and cfg.eval_every > 0 \
                    and self.global_step % cfg.eval_every == 0:
                eval_loss = self.evaluate(eval_iter)
                logger.info("step %d | eval_loss %.4f", self.global_step, eval_loss)
                self._eval_hist.append((self.global_step, eval_loss))
                self.event_bus.emit("train/eval_end", step=self.global_step, eval_loss=eval_loss)
                self._update_monitor(eval_loss, self.global_step)
                # 每个 eval 检查点刷新 loss 图像（实时可见）
                if cfg.loss_plot and cfg.plot_every_eval:
                    self._render_loss_plot()
                if self.early_stopped:
                    break

            if cfg.ckpt_every > 0 and self.global_step % cfg.ckpt_every == 0:
                if not cfg.save_best_only:
                    self.save_checkpoint()
                elif cfg.save_last:
                    # 仅「智能保存」模式下需要额外兜底：始终保留最近状态
                    # （save_best_only=False 时 step_N.pt 已是最新，无需重复写）
                    self.save_checkpoint(
                        str(Path(cfg.ckpt_dir) / f"{cfg.model_name}_last.pt")
                    )

        plot_path = self._plot_loss() if cfg.loss_plot else None

        summary = {
            "final_loss": loss_accum,
            "best_loss": self._best_loss if math.isfinite(self._best_loss) else None,
            "best_eval_loss": min((l for _, l in self._eval_hist), default=None)
            if self._eval_hist else None,
            "early_stopped": self.early_stopped,
            "steps": self.global_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_run": run_tokens,
            "wall_time_s": time.perf_counter() - t0,
        }
        if plot_path is not None:
            summary["loss_plot"] = str(plot_path)
        self.event_bus.emit("train/end", **summary)
        return summary

    # ------------------------------------------------------- monitor / plot

    def _update_monitor(self, loss: float, step: int) -> None:
        """智能保存 + 早停的统一监控入口。

        - ``save_best_only``：loss 创新低才保存周期 checkpoint（否则跳过）；
        - ``early_stop_patience``：连续 N 次监控未创新低则置 ``early_stopped``。
        """
        cfg = self.config
        if loss < self._best_loss - 1e-6:  # 容差防浮点噪声导致的假改善
            self._best_loss = loss
            self._bad_evals = 0
            if cfg.save_best_only:
                path = self.save_checkpoint()  # 内部已发 train/checkpoint 事件
                logger.info("step %d | 监控 loss %.4f 创新低 -> %s", step, loss, path)
        else:
            self._bad_evals += 1
            if cfg.early_stop_patience > 0 and self._bad_evals >= cfg.early_stop_patience:
                self.early_stopped = True
                logger.info(
                    "监控 loss 连续 %d 次未低于 best %.4f，提前停止训练 (step %d)",
                    self._bad_evals, self._best_loss, step,
                )
                self.event_bus.emit(
                    "train/early_stop", step=step, best_loss=self._best_loss,
                    patience=self._bad_evals,
                )
                if cfg.loss_plot and cfg.plot_every_eval:
                    self._render_loss_plot()

    def _ensure_plot(self) -> bool:
        """惰性创建并复用 matplotlib Figure（避免每次 eval 新建/导入）。"""
        if self._plot_fig is not None:
            return True
        if self._plot_disabled:
            return False
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("未安装 matplotlib，跳过 loss 曲线绘制")
            self._plot_disabled = True
            return False
        self._plt = plt
        self._plot_fig, self._plot_ax = plt.subplots(figsize=(8, 5))
        return True

    def _render_loss_plot(self, *, close: bool = False) -> Path | None:
        """绘制/刷新 loss 曲线到 ``ckpt_dir/loss_curve.png``。

        ``close=False`` 时保留 Figure 供下次 eval 复用（``ax.clear()`` 重绘）。
        """
        if not (self._train_hist or self._eval_hist) or not self._ensure_plot():
            return None
        ax = self._plot_ax
        ax.clear()
        if self._train_hist:
            xs, ys = zip(*self._train_hist)
            ax.plot(xs, ys, label="train loss", color="#4C8BF5", alpha=0.85, lw=1.2)
        if self._eval_hist:
            xs, ys = zip(*self._eval_hist)
            ax.plot(xs, ys, label="eval loss", color="#F5533D", marker="o", ms=3, lw=1.2)
        best = self._best_loss if math.isfinite(self._best_loss) else None
        if best is not None:
            ax.axhline(best, ls="--", color="gray", lw=0.8, label=f"best {best:.4f}")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        path = Path(self.config.ckpt_dir) / "loss_curve.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._plot_fig.savefig(path, dpi=120, bbox_inches="tight")
        if close:
            self._plt.close(self._plot_fig)
            self._plot_fig = self._plot_ax = None
        return path

    def _plot_loss(self) -> Path | None:
        """训练结束时绘制并关闭图像（保留原接口）。"""
        path = self._render_loss_plot(close=True)
        if path is not None:
            logger.info("loss 曲线已保存: %s", path)
        return path

    @torch.no_grad()
    def evaluate(self, eval_iter: TokenBatchIterator) -> float:
        self.model.eval()
        losses = []
        for inputs, targets in eval_iter.batches(self.config.eval_batches):
            with self._autocast():
                _, loss = self.model(
                    inputs, targets,
                    return_logits=False, loss_chunk_size=self.config.loss_chunk_size,
                )
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    # ------------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: str | None = None) -> Path:
        cfg = self.config
        ckpt_path = Path(path) if path else Path(cfg.ckpt_dir) / f"step_{self.global_step}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)  # 剥离 torch.compile 包装
        unique, total = count_parameters(raw_model)
        payload = {
            "global_step": self.global_step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.grad_scaler.state_dict(),
            "model_config": raw_model.config.to_dict(),
            "model_name": cfg.model_name,
            "lora_config": cfg.lora.to_dict() if cfg.lora is not None else None,
            # 元数据：加载校验 + 阶段区分（旧 checkpoint 无这些键，加载时按兼容处理）
            "stage": cfg.stage,
            "param_count": unique,
            "state_numel": total,
            "tie_weights": getattr(raw_model.config, "tie_weights", None),
            # 断点续训所需的一切非权重状态（RNG/监控/历史/数据指纹/配置）
            **self._training_state(),
        }
        torch.save(payload, ckpt_path)
        self.event_bus.emit("train/checkpoint", path=str(ckpt_path), step=self.global_step)
        return ckpt_path

    @staticmethod
    def _verify_param_count(model: torch.nn.Module, payload: dict, path) -> None:
        """校验加载后的模型与 checkpoint 元数据参数量一致。

        旧 checkpoint 无 ``param_count`` 时跳过（仅告警）。存在元数据却不一致
        说明模型架构与 checkpoint 不匹配——此时静默加载会出现「参数量减半/
        权重对不上」的隐蔽问题，这里直接报错并给出唯一/state_dict 两种口径。
        """
        expected = payload.get("param_count")
        if expected is None:
            logger.debug("checkpoint %s 无 param_count 元数据，跳过参数量校验", path)
            return
        actual, total = count_parameters(model)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint 参数量不匹配: {path}\n"
                f"  期望(唯一) {expected:,}，实际(唯一) {actual:,}，"
                f"state_dict 合计 {total:,}\n"
                "  模型架构与 checkpoint 不一致，请使用 checkpoint 自带的 "
                "model_config 构建模型（CometSpark 会自动这么做）。"
            )

    def finalize(self, stage: str | None = None, path: str | Path | None = None) -> Path:
        """保存 final checkpoint 并清理中间 checkpoint。

        - 命名：默认 ``{ckpt_dir}/{model_name}_{stage}_final.pt``，可传 ``path`` 覆盖。
        - ``final_keep_optimizer=True`` 时保留优化器/缩放器状态（可续训）。
        - 保存成功后按 ``cleanup_intermediate`` 删除 ``step_*.pt`` 等周期
          checkpoint（只删中间件，保留 final 与已存在的 adapter/merged）。
        """
        cfg = self.config
        stage = stage or cfg.stage
        out = Path(path) if path else Path(cfg.ckpt_dir) / f"{cfg.model_name}_{stage}_final.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        unique, total = count_parameters(raw_model)
        payload = {
            "global_step": self.global_step,
            "model_state": raw_model.state_dict(),
            "model_config": raw_model.config.to_dict(),
            "model_name": cfg.model_name,
            "lora_config": cfg.lora.to_dict() if cfg.lora is not None else None,
            "stage": stage,
            "param_count": unique,
            "state_numel": total,
            "tie_weights": getattr(raw_model.config, "tie_weights", None),
            "final": True,
            # 续训状态（历史/指纹/RNG/配置）始终保留，优化器按需
            **self._training_state(),
        }
        if cfg.final_keep_optimizer:
            payload["optimizer_state"] = self.optimizer.state_dict()
            payload["scaler_state"] = self.grad_scaler.state_dict()
        torch.save(payload, out)
        self.event_bus.emit("train/checkpoint", path=str(out), step=self.global_step)
        logger.info("final checkpoint 已保存(%s): %s (%.2fM 参数)",
                    stage, out, unique / 1e6)
        if cfg.cleanup_intermediate:
            removed = self._cleanup_intermediate(out.parent, keep={out})
            if removed:
                logger.info("已清理 %d 个中间 checkpoint，仅保留 %s", removed, out.name)
        return out

    @staticmethod
    def _cleanup_intermediate(ckpt_dir: Path, keep: set[Path]) -> int:
        """删除周期 checkpoint（``step_*.pt``/``*_last.pt`` 等），返回删除数量。"""
        keep = {p.resolve() for p in keep}
        removed = 0
        for pattern in ("step_*.pt", "checkpoint_*.pt", "ckpt_*.pt", "*_last.pt"):
            for candidate in ckpt_dir.glob(pattern):
                if candidate.resolve() in keep:
                    continue
                try:
                    candidate.unlink()
                    removed += 1
                except OSError as exc:  # 权限/占用等：不阻断训练收尾
                    logger.warning("删除中间 checkpoint 失败 %s: %s", candidate, exc)
        return removed

    def cleanup_intermediate(self, keep: set[str | Path] | None = None) -> int:
        """删除 ``ckpt_dir`` 下的周期 checkpoint，``keep`` 中的文件保留。

        供 LoRA 等自行导出产物（adapter/merged）的场景复用 finalize 的清理逻辑。
        """
        keep_paths = {Path(p) for p in (keep or set())}
        return self._cleanup_intermediate(Path(self.config.ckpt_dir), keep=keep_paths)

    def save_adapter(self, path: str | Path) -> Path:
        """只保存 LoRA 适配器（LoRA 微调时体积 <1% 主干）。"""
        from verse_nn.lora import lora_state_dict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        unique, total = count_parameters(raw_model)
        payload = {
            "lora_state": lora_state_dict(raw_model),
            "model_config": raw_model.config.to_dict(),
            "lora_config": self.config.lora.to_dict() if self.config.lora is not None else None,
            "global_step": self.global_step,
            "stage": self.config.stage,
            "param_count": unique,
            "state_numel": total,
            "tie_weights": getattr(raw_model.config, "tie_weights", None),
        }
        torch.save(payload, path)
        logger.info("LoRA 适配器已保存: %s", path)
        return path

    def load_checkpoint(self, path: str | Path, resume_training: bool = True) -> None:
        """从 checkpoint 恢复模型（并默认恢复完整续训状态）。

        Args:
            path: checkpoint 路径。
            resume_training: True 时额外恢复优化器/缩放器状态、RNG、监控/早停
                进度、loss 历史、累计 token 数与数据指纹；False 时只恢复权重与
                步数（用于「只想拿权重、不想继承优化器状态」的场景）。
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = getattr(self.model, "_orig_mod", self.model)
        raw_model.load_state_dict(payload["model_state"])
        self._verify_param_count(raw_model, payload, path)
        self.global_step = payload.get("global_step", 0)
        if resume_training:
            # final checkpoint 可能按需省略优化器/缩放器状态
            if payload.get("optimizer_state") is not None:
                self.optimizer.load_state_dict(payload["optimizer_state"])
            if payload.get("scaler_state") is not None:
                self.grad_scaler.load_state_dict(payload["scaler_state"])
            self._restore_training_state(payload)
        logger.info("已恢复 checkpoint %s (step %d, stage=%s, resume=%s)",
                    path, self.global_step, payload.get("stage", "?"), resume_training)
