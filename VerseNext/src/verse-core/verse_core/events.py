"""事件总线：trainer 与 callbacks / RSI 之间的解耦通信。

典型事件（约定命名）：
- ``train/step_end``       每 step 结束（payload 含 loss、lr、step 等）
- ``train/epoch_end``      每 epoch 结束
- ``train/eval_end``       评估结束（payload 含 eval_loss）
- ``train/checkpoint``     保存 checkpoint
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
