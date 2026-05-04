"""
Strategy registry for the unified platform.

Strategies self-register using the @register decorator.  The registry maps
each strategy's ``name`` attribute to its class, so the optimization pipeline
and dashboard can look them up by name without importing them directly.

Usage
-----
Registering a strategy::

    from strategy_platform.registry import register
    from strategy_platform.base_strategy import BaseStrategy

    @register
    class WickTest5M(BaseStrategy):
        name = "wicktest5m"
        ...

Looking up a strategy::

    from strategy_platform.registry import StrategyRegistry

    cls = StrategyRegistry.get("wicktest5m")
    strategy = cls(params={"profit": 50})

Listing all registered strategies::

    StrategyRegistry.list_strategies()   # -> ["wicktest5m", "goldbot7"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from strategy_platform.base_strategy import BaseStrategy


class StrategyRegistry:
    _strategies: Dict[str, Type["BaseStrategy"]] = {}

    @classmethod
    def register(cls, strategy_cls: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
        """Register a strategy class. Called automatically by the @register decorator."""
        if not strategy_cls.name:
            raise ValueError(
                f"Cannot register {strategy_cls.__name__}: 'name' class attribute is empty."
            )
        if strategy_cls.name in cls._strategies:
            raise ValueError(
                f"A strategy named '{strategy_cls.name}' is already registered. "
                f"Each strategy must have a unique name."
            )
        cls._strategies[strategy_cls.name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str) -> Type["BaseStrategy"]:
        """Return the strategy class registered under *name*."""
        if name not in cls._strategies:
            available = cls.list_strategies()
            raise KeyError(
                f"No strategy named '{name}' is registered. "
                f"Available strategies: {available}"
            )
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Return a sorted list of all registered strategy names."""
        return sorted(cls._strategies.keys())

    @classmethod
    def all(cls) -> Dict[str, Type["BaseStrategy"]]:
        """Return a copy of the full registry dict."""
        return dict(cls._strategies)


def register(strategy_cls: Type["BaseStrategy"]) -> Type["BaseStrategy"]:
    """
    Class decorator that registers a strategy with the global StrategyRegistry.

    Example::

        @register
        class GoldBot7(BaseStrategy):
            name = "goldbot7"
            ...
    """
    return StrategyRegistry.register(strategy_cls)
