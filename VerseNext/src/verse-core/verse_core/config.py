"""配置基类：dataclass 驱动，支持 YAML 加载保存与嵌套覆盖。"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """所有配置的基类。

    子类用 ``@dataclass`` 声明字段即可获得：
    - ``to_dict()`` / ``from_dict()``：与嵌套 dict 互转
    - ``save(path)`` / ``load(path)``：YAML 持久化
    - ``merge(**overrides)``：返回覆盖后的新配置（RSI 变异的基础操作）
    """

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                out[f.name] = value.to_dict()
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                out[f.name] = dataclasses.asdict(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(f"{cls.__name__} 未知配置项: {key}")
            # 嵌套 BaseConfig 字段：dict -> 子配置实例
            if isinstance(value, dict):
                sub = cls._resolve_config_type(known[key])
                if sub is not None:
                    value = sub.from_dict(value)
            kwargs[key] = value
        return cls(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _resolve_config_type(f: dataclasses.Field) -> type["BaseConfig"] | None:
        """从字段解析出 BaseConfig 子类（支持类型注解与 default_factory）。"""
        candidates = []
        if f.default_factory is not None:
            candidates.append(f.default_factory)
        if isinstance(f.type, type):
            candidates.append(f.type)
        else:
            import typing

            for arg in typing.get_args(f.type):
                candidates.append(arg)
        for c in candidates:
            if isinstance(c, type) and issubclass(c, BaseConfig):
                return c
        return None

    def merge(self: T, **overrides: Any) -> T:
        """返回用 overrides 覆盖后的新配置实例（不变更原配置）。"""
        data = self.to_dict()
        data.update(overrides)
        return type(self).from_dict(data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False))

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def validate(self) -> None:
        """子类可覆写以校验配置合法性，默认无操作。"""

    def __post_init__(self) -> None:
        self.validate()
