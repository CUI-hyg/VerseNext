"""基于 EventBus 的回调集合。"""

from __future__ import annotations

from pathlib import Path

from verse_core import EventBus, get_logger

logger = get_logger("callbacks")


class CheckpointCallback:
    """在指定事件上保存 checkpoint。"""

    def __init__(self, ckpt_dir: str | Path, event_name: str = "train/eval_end") -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.event_name = event_name
        self._bus: EventBus | None = None
        self._trainer = None

    def attach(self, bus: EventBus, trainer) -> None:
        self._bus = bus
        self._trainer = trainer
        bus.subscribe(self.event_name, self._on_event)

    def _on_event(self, event) -> None:
        if self._trainer is None:
            return
        path = self._trainer.save_checkpoint()
        logger.info("callback 已保存 checkpoint: %s", path)


class EarlyStoppingCallback:
    """验证 loss 连续 ``patience`` 次不下降时停止训练。

    通过设置 trainer 上的 ``_stop_requested`` 标志实现；Trainer.train
    在每个 step 边界检查该标志。
    """

    def __init__(self, patience: int = 5, event_name: str = "train/eval_end") -> None:
        self.patience = patience
        self.event_name = event_name
        self.best = float("inf")
        self.bad_epochs = 0

    def attach(self, bus: EventBus, trainer) -> None:
        self._trainer = trainer
        bus.subscribe(self.event_name, self._on_event)

    def _on_event(self, event) -> None:
        loss = event.payload.get("eval_loss")
        if loss is None:
            return
        if loss < self.best - 1e-5:
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                logger.info("early stopping 触发（%d 次未改善）", self.bad_epochs)
                setattr(self._trainer, "_stop_requested", True)
