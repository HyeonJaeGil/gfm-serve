from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..contracts import BackendDescriptor, SceneInput, ViewResult
from ..storage import ArtifactDescriptor, PreparedImage


class EmptyBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True)
class PreparedView:
    view_id: str
    upload_key: str
    image: PreparedImage


@dataclass(slots=True)
class BackendRunRequest:
    request_id: str
    run_dir: Path
    scene: SceneInput
    views: list[PreparedView]
    backend_options: BaseModel

    @property
    def images(self) -> list[PreparedImage]:
        return [view.image for view in self.views]


@dataclass(slots=True)
class BackendRunResult:
    view_results: list[ViewResult]
    artifacts: list[ArtifactDescriptor]
    produced_outputs: list[str]
    timings_ms: dict[str, int]
    warnings: list[str] = field(default_factory=list)


class ReconstructionBackend(ABC):
    backend_id: ClassVar[str]
    display_name: ClassVar[str]
    capabilities: ClassVar[tuple[str, ...]] = ()
    options_model: ClassVar[type[BaseModel]] = EmptyBackendOptions

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend=self.backend_id,
            model_id=self.display_name,
            inputs={"images": {"required": True}},
            outputs=list(self.capabilities),
            options_schema=self.options_model.model_json_schema(),
        )

    def validate_options(self, payload: dict[str, Any] | None) -> BaseModel:
        return self.options_model.model_validate(payload or {})

    def validate_request(self, scene: SceneInput, options: dict[str, Any] | None) -> BaseModel:
        if any(view.camera is not None for view in scene.views):
            raise ValueError(f"Backend '{self.backend_id}' does not support camera inputs.")
        return self.validate_options(options)

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
