from __future__ import annotations

from .base import BackendRunRequest, BackendRunResult, EmptyBackendOptions, ReconstructionBackend
from .registry import create_backend, list_backends, register_backend
from .vggt import VGGTBackend, VGGTBackendOptions


register_backend(VGGTBackend.backend_id, VGGTBackend)

__all__ = [
    "BackendRunRequest",
    "BackendRunResult",
    "EmptyBackendOptions",
    "ReconstructionBackend",
    "VGGTBackend",
    "VGGTBackendOptions",
    "create_backend",
    "list_backends",
    "register_backend",
]
