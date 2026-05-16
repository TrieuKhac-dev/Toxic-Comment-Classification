from abc import ABC
from dataclasses import dataclass, fields
from typing import Generic, TypeVar

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig(ABC, Generic[T]):
    def override(self: T, **kwargs: object) -> T:
        current = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        current.update(kwargs)
        return type(self)(**current)

    def _export_to_globals(self) -> None:
        """Export dataclass fields to global variables for DVC."""
        import inspect

        module = inspect.getmodule(self.__class__)
        if module is None:
            return

        for f in fields(self):
            value = getattr(self, f.name)
            setattr(module, f.name, value)
