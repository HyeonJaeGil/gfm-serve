from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .contracts import CameraResult, ImageSize, ViewResult


class InputSummary(BaseModel):
    scene_id: str | None = None
    image_count: int
    filenames: list[str]
    total_bytes: int


class TimingStats(BaseModel):
    validation: int = 0
    inference: int = 0
    postprocess: int = 0
    total: int = 0


class ArtifactInfo(BaseModel):
    name: str
    kind: str
    url: str
    content_type: str
    size_bytes: int


class ErrorInfo(BaseModel):
    code: str
    message: str


class ReconstructionResponse(BaseModel):
    backend: str
    request_id: str
    client_request_id: str | None = None
    status: Literal["succeeded", "failed"]
    input_summary: InputSummary | None = None
    timings_ms: TimingStats
    view_results: list[ViewResult] = Field(default_factory=list)
    camera_results: list[ViewResult] = Field(default_factory=list)
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    produced_outputs: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    ready: bool
    backend: str
    capabilities: list[str] = Field(default_factory=list)
    device: str | None = None
    error: str | None = None
