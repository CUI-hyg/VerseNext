"""VerseNext 基础设施包：注册中心、配置、事件总线、日志。"""

from verse_core.registry import Registry, global_registry
from verse_core.config import BaseConfig
from verse_core.events import EventBus
from verse_core.logging import get_logger

__all__ = ["Registry", "global_registry", "BaseConfig", "EventBus", "get_logger"]
__version__ = "0.1.0"
