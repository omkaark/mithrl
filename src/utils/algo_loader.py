from __future__ import annotations

import importlib

from ..algo.base import Algorithm
from .config import MithrlConfig
from .env_loader import _split_factory_path


def load_algorithm(config: MithrlConfig) -> Algorithm:
    factory_path = config.algo.factory
    module_path, symbol = _split_factory_path(factory_path)
    module = importlib.import_module(module_path)

    try:
        algo_cls = getattr(module, symbol)
    except AttributeError as exc:
        raise ImportError(
            f"Algorithm class '{symbol}' was not found in module '{module_path}'."
        ) from exc

    if not isinstance(algo_cls, type) or not issubclass(algo_cls, Algorithm):
        raise TypeError(
            f"Configured algorithm factory '{factory_path}' must be an Algorithm subclass."
        )

    return algo_cls(config=config, **config.algo.kwargs)
