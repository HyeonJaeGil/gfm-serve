from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..storage import ArtifactDescriptor, PreparedImage


class EmptyBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True)
class BackendRunRequest:
    request_id: str
    run_dir: Path
    images: list[PreparedImage]
    backend_options: BaseModel


@dataclass(slots=True)
class BackendRunResult:
    camera_results: list[dict[str, Any]]
    artifacts: list[ArtifactDescriptor]
    produced_outputs: list[str]
    timings_ms: dict[str, int]


class ReconstructionBackend(ABC):
    backend_id: ClassVar[str]
    display_name: ClassVar[str]
    capabilities: ClassVar[tuple[str, ...]] = ()
    options_model: ClassVar[type[BaseModel]] = EmptyBackendOptions

    def validate_options(self, payload: dict[str, Any] | None) -> BaseModel:
        return self.options_model.model_validate(payload or {})

    @property
    @abstractmethod
    def device_description(self) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def last_error(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, request: BackendRunRequest) -> BackendRunResult:
        raise NotImplementedError
