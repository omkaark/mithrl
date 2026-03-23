from __future__ import annotations

import importlib

from ..algo.base import Algorithm
from ..train.env import EnvironmentFactory
from .config import MithrlConfig


def load_environment_factory(config: MithrlConfig) -> EnvironmentFactory:
    factory_path = config.env.factory
    module_path, symbol = _split_factory_path(factory_path)
    module = importlib.import_module(module_path)

    try:
        factory_cls = getattr(module, symbol)
    except AttributeError as exc:
        raise ImportError(
            f"Environment factory '{symbol}' was not found in module '{module_path}'."
        ) from exc

    if not isinstance(factory_cls, type) or not issubclass(factory_cls, EnvironmentFactory):
        raise TypeError(
            f"Configured environment factory '{factory_path}' must be an EnvironmentFactory subclass."  # noqa
        )

    return factory_cls(config=config, **config.env.kwargs)


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


def _split_factory_path(factory_path: str) -> tuple[str, str]:
    if ":" in factory_path:
        module_path, symbol = factory_path.split(":", 1)
    else:
        module_path, _, symbol = factory_path.rpartition(".")

    if not module_path or not symbol:
        raise ValueError(
            "env.factory must look like 'package.module:FactoryClass' or 'package.module.FactoryClass'."  # noqa
        )

    return module_path, symbol
