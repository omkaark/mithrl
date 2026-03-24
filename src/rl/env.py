from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    def __init__(self) -> None:
        self._metadata: dict[str, Any] = {}
        self.done = False
        self.turn_count = 0
        self.reward = 0

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @property
    @abstractmethod
    def next_query(self) -> str:
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @abstractmethod
    def step(self, response: str) -> None:
        ...


class EnvironmentFactory(ABC):
    @abstractmethod
    def create(self, rollout_idx: int) -> Environment:
        ...
