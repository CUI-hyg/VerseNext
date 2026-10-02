"""事件总线：trainer 与 callbacks / RSI 之间的解耦通信。

典型事件（约定命名）：
- ``train/start``          一次 run 开始（payload 含 ``start_step``/``total_steps``/``stage``）。
  订阅方必须用它给出的**相对区间**（``step - start_step`` / ``total_steps - start_step``）
  计算进度与 ETA；直接用绝对 ``step`` 在断点续训/阶段链下会算错。
- ``train/step_end``       每 step 结束（payload 含 loss、lr、step 等）
- ``train/epoch_end``      每 epoch 结束
- ``train/eval_end``       评估结束（payload 含 eval_loss）
- ``train/checkpoint``     保存 checkpoint
- ``train/stage_start``    阶段链中某阶段开始（payload 含 index/total/name/start_step/end_step）
- ``train/stage_end``      阶段链中某阶段结束（payload 含 index/name/final_loss/checkpoint）
- ``rsi/generation_end``   RSI 每代结束
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

Handler = Callable[["Event"], None]


@dataclass
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """极简同步事件总线。"""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Handler) -> None:
        self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: Handler) -> None:
        self._handlers[event_name].remove(handler)

    def emit(self, event_name: str, **payload: Any) -> None:
        event = Event(name=event_name, payload=payload)
        for handler in list(self._handlers.get(event_name, [])):
            handler(event)

    def clear(self) -> None:
        self._handlers.clear()
