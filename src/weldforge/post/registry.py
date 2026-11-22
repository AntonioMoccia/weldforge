"""Postprocessor registry: look one up by key."""

from __future__ import annotations

from .base import PostOptions, Postprocessor

__all__ = ["available_targets", "get_postprocessor", "register"]

_REGISTRY: dict[str, type[Postprocessor]] = {}


def register(cls: type[Postprocessor]) -> type[Postprocessor]:
    """Class decorator adding a postprocessor to the registry."""
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a key")
    _REGISTRY[cls.key] = cls
    return cls


def available_targets() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def get_postprocessor(key: str, options: PostOptions | None = None) -> Postprocessor:
    _load_all()
    try:
        cls = _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown postprocessor {key!r}; available: {sorted(_REGISTRY)}") from None
    return cls(options)


def _load_all() -> None:
    """Import the built-in targets so their decorators run."""
    from . import fanuc, krl, rapid, ros2, urscript  # noqa: F401
