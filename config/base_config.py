from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig(ABC, Generic[T]):
    def override(self: T, **kwargs: object) -> T:
        current = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        current.update(kwargs)
        return type(self)(**current)
