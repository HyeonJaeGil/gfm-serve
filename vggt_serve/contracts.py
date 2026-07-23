from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Matrix3x3 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
]
Matrix4x4 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
]


class CameraInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    convention: Literal["opencv"]
    world_to_camera: Matrix4x4 | None = None
    intrinsics: Matrix3x3 | None = None

    @model_validator(mode="after")
    def validate_matrices(self) -> CameraInput:
        for matrix in (self.world_to_camera, self.intrinsics):
            if matrix is not None and not all(math.isfinite(value) for row in matrix for value in row):
                raise ValueError("Camera matrices must contain only finite numbers.")
        return self


class ViewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1)
    upload_key: str = Field(min_length=1)
    camera: CameraInput | None = None


class SceneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str | None = None
    views: list[ViewInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> SceneInput:
        view_ids = [view.view_id for view in self.views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("view_id values must be unique.")
        upload_keys = [view.upload_key for view in self.views]
        if len(upload_keys) != len(set(upload_keys)):
            raise ValueError("upload_key values must be unique.")
        return self


class BackendInputField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool


class BackendDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    model_id: str
    model_revision: str | None = None
    inputs: dict[str, BackendInputField]
    outputs: list[str]
    options_schema: dict[str, object]


class CameraResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    convention: Literal["opencv"]
    world_to_camera: Matrix4x4 | None = None
    intrinsics: Matrix3x3 | None = None
    source: Literal["predicted", "provided", "aligned"]


class ImageSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ViewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str
    filename: str
    original_size: ImageSize
    camera: CameraResult | None = None
