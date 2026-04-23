from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from .base import ReconstructionBackend


BackendFactory = Callable[[Settings], ReconstructionBackend]

_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(backend_id: str, factory: BackendFactory) -> None:
    existing = _REGISTRY.get(backend_id)
    if existing is not None and existing is not factory:
        raise ValueError(f"Backend '{backend_id}' is already registered.")
    _REGISTRY[backend_id] = factory


def create_backend(settings: Settings) -> ReconstructionBackend:
    try:
        factory = _REGISTRY[settings.backend]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(
            f"Backend '{settings.backend}' is not implemented. Registered backends: {known}."
        ) from exc
    return factory(settings)


def list_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
