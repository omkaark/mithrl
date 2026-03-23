from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..utils.config import MithrlConfig


class Algorithm(ABC):
    def __init__(self, config: MithrlConfig, **kwargs: Any) -> None:
        self.config = config
        self.kwargs = self.validate_kwargs(kwargs)

    @classmethod
    def validate_kwargs(cls, kwargs: dict[str, Any]) -> Any:
        return kwargs

    @abstractmethod
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        metadatas: list[dict],
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def loss(
        self,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        masks: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...
