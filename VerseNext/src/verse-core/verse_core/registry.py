"""全局组件注册中心。

VerseNext 中所有可替换组件（模型、优化器、调度器、RSI 变异算子等）
都通过 Registry 注册，按 ``"类别/名称"`` 命名，例如 ``"model/verse_transformer"``。
RSI 的架构变异依赖此机制动态构建新结构。
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_CREATE_HOOK = Callable[[Any], Any]


class Registry:
    """线程安全的组件注册中心。

    命名规范：``"namespace/name"``，如 ``"model/verse_transformer"``、
    ``"mutator/hyperparam"``。
    """

    def __init__(self, name: str = "registry") -> None:
        self.name = name
        self._lock = threading.RLock()
        self._entries: dict[str, Any] = {}

    @staticmethod
    def _qualify(namespace: str, name: str) -> str:
        return f"{namespace}/{name}"

    def register(self, namespace: str, name: str, obj: Any, *, override: bool = False) -> None:
        key = self._qualify(namespace, name)
        with self._lock:
            if key in self._entries and not override:
                raise KeyError(f"{key} 已注册，如需覆盖请传 override=True")
            self._entries[key] = obj

    def get(self, namespace: str, name: str) -> Any:
        with self._lock:
            try:
                return self._entries[self._qualify(namespace, name)]
            except KeyError:
                raise KeyError(
                    f"未注册组件 {namespace}/{name}；可用: {self.list(namespace)}"
                ) from None

    def create(self, namespace: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """取出注册项并实例化/调用。"""
        obj = self.get(namespace, name)
        return obj(*args, **kwargs)

    def list(self, namespace: str | None = None) -> list[str]:
        with self._lock:
            keys = list(self._entries)
        if namespace is None:
            return keys
        prefix = namespace + "/"
        return [k for k in keys if k.startswith(prefix)]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    # ---------------------------------------------------- 便捷装饰器（实例方法）

    def register_scheduler(self, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        """学习率调度器装饰器：注册到 ``scheduler/<name>``。"""
        def _wrap(cls: type[T]) -> type[T]:
            self.register("scheduler", name, cls, override=override)
            return cls
        return _wrap

    def register_attention(self, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        """注意力类装饰器：注册到 ``attention/<name>``。"""
        def _wrap(cls: type[T]) -> type[T]:
            self.register("attention", name, cls, override=override)
            return cls
        return _wrap

    def register_model(self, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        """模型类装饰器：注册到 ``model/<name>``。"""
        def _wrap(cls: type[T]) -> type[T]:
            self.register("model", name, cls, override=override)
            return cls
        return _wrap

    def register_mutator(self, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        """RSI 变异算子装饰器：注册到 ``mutator/<name>``。"""
        def _wrap(cls: type[T]) -> type[T]:
            self.register("mutator", name, cls, override=override)
            return cls
        return _wrap

    def register_scheduler(self, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        """学习率调度器装饰器：注册到 ``scheduler/<name>``。"""
        def _wrap(cls: type[T]) -> type[T]:
            self.register("scheduler", name, cls, override=override)
            return cls
        return _wrap


global_registry = Registry("versenext")


def register_model(name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
    """模型类装饰器：注册到 ``model/<name>``。"""
    def _wrap(cls: type[T]) -> type[T]:
        global_registry.register("model", name, cls, override=override)
        return cls
    return _wrap


def register_mutator(name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
    """RSI 变异算子装饰器：注册到 ``mutator/<name>``。"""
    def _wrap(cls: type[T]) -> type[T]:
        global_registry.register("mutator", name, cls, override=override)
        return cls
    return _wrap


def register_scheduler(name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
    """学习率调度器装饰器：注册到 ``scheduler/<name>``。"""
    def _wrap(cls: type[T]) -> type[T]:
        global_registry.register("scheduler", name, cls, override=override)
        return cls
    return _wrap
