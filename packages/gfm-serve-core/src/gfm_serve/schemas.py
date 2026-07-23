from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .contracts import BackendDescriptor, CameraResult, ImageSize, ViewResult


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


class LegacyCameraResult(BaseModel):
    filename: str
    original_size: ImageSize
    cam_from_world: list[list[float]]
    intrinsics: list[list[float]]


class ArtifactInfo(BaseModel):
    name: str
    kind: str
    url: str
    content_type: str
    size_bytes: int
    metadata: dict[str, object] = Field(default_factory=dict)


class ErrorInfo(BaseModel):
    code: str
    message: str


class ReconstructionResponse(BaseModel):
    service_version: str = "0.2.0"
    result_schema_version: str = "1.0"
    backend: str
    model: BackendDescriptor
    request_id: str
    client_request_id: str | None = None
    status: Literal["succeeded", "failed"]
    input_summary: InputSummary | None = None
    timings_ms: TimingStats
    view_results: list[ViewResult] = Field(default_factory=list)
    camera_results: list[LegacyCameraResult] = Field(default_factory=list)
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    produced_outputs: list[str] = Field(default_factory=list)
    normalized_request: dict[str, object] | None = None
    input_coordinate_convention: str = "opencv"
    output_coordinate_convention: str = "opencv"
    warnings: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    ready: bool
    backend: str
    model_descriptor_url: str = "/v1/models/current"
    capabilities: list[str] = Field(default_factory=list)
    device: str | None = None
    error: str | None = None
