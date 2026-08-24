"""Model registry for video action recognition models."""

from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a model class."""

    def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs: Any) -> Any:
    """Get model by name from registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())
