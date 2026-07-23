from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from ..config import Settings
from .base import ReconstructionBackend


BackendFactory = Callable[[Settings], ReconstructionBackend]

_REGISTRY: dict[str, BackendFactory] = {}


def _discover_backends() -> dict[str, BackendFactory]:
    discovered = dict(_REGISTRY)
    for entry_point in entry_points(group="gfm_serve.backends"):
        factory = entry_point.load()
        if entry_point.name in discovered and discovered[entry_point.name] is not factory:
            raise ValueError(f"Backend '{entry_point.name}' is installed more than once.")
        discovered[entry_point.name] = factory
    return discovered


def register_backend(backend_id: str, factory: BackendFactory) -> None:
    existing = _REGISTRY.get(backend_id)
    if existing is not None and existing is not factory:
        raise ValueError(f"Backend '{backend_id}' is already registered.")
    _REGISTRY[backend_id] = factory


def create_backend(settings: Settings) -> ReconstructionBackend:
    available = _discover_backends()
    if settings.backend is None:
        if len(available) != 1:
            known = ", ".join(sorted(available)) or "none"
            raise ValueError(
                "Exactly one backend must be installed when GFM_SERVE_BACKEND is unset. "
                f"Discovered: {known}."
            )
        return next(iter(available.values()))(settings)
    try:
        factory = available[settings.backend]
    except KeyError as exc:
        known = ", ".join(sorted(available)) or "none"
        raise ValueError(
            f"Backend '{settings.backend}' is not implemented. Registered backends: {known}."
        ) from exc
    return factory(settings)


def list_backends() -> tuple[str, ...]:
    return tuple(sorted(_discover_backends()))
