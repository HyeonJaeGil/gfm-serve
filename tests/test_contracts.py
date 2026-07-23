from __future__ import annotations

import pytest
from pydantic import ValidationError

from vggt_serve.contracts import CameraInput, SceneInput


IDENTITY_4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
IDENTITY_3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_camera_input_accepts_canonical_matrices() -> None:
    camera = CameraInput(convention="opencv", world_to_camera=IDENTITY_4, intrinsics=IDENTITY_3)

    assert camera.world_to_camera is not None
    assert len(camera.world_to_camera) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_to_camera", [[1, 0], [0, 1]]),
        ("intrinsics", [[1, 0], [0, 1]]),
        ("world_to_camera", [[float("nan"), 0, 0, 0], *IDENTITY_4[1:]]),
        ("intrinsics", [[float("inf"), 0, 0], *IDENTITY_3[1:]]),
    ],
)
def test_camera_input_rejects_invalid_matrices(field: str, value: list[list[float]]) -> None:
    with pytest.raises(ValidationError):
        CameraInput.model_validate({"convention": "opencv", field: value})


def test_scene_rejects_duplicate_view_ids() -> None:
    with pytest.raises(ValidationError, match="view_id values must be unique"):
        SceneInput.model_validate(
            {
                "views": [
                    {"view_id": "same", "upload_key": "one"},
                    {"view_id": "same", "upload_key": "two"},
                ]
            }
        )


def test_scene_rejects_duplicate_upload_keys() -> None:
    with pytest.raises(ValidationError, match="upload_key values must be unique"):
        SceneInput.model_validate(
            {
                "views": [
                    {"view_id": "one", "upload_key": "same"},
                    {"view_id": "two", "upload_key": "same"},
                ]
            }
        )
