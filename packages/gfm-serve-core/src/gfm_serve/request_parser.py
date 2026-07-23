from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from .contracts import SceneInput
from .errors import ValidationApiError


class ReconstructionManifest(SceneInput):
    model_config = ConfigDict(extra="forbid")

    options: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class ParsedMultipartRequest:
    scene: SceneInput
    uploads: list[UploadFile]
    raw_options: dict[str, Any]
    client_request_id: str | None


def _optional_string(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationApiError(f"{key} must be a text form field.")
    return value


async def parse_reconstruction_multipart(request: Request) -> ParsedMultipartRequest:
    form = await request.form()
    file_items = [(key, value) for key, value in form.multi_items() if isinstance(value, UploadFile)]
    manifest_payload = _optional_string(form, "manifest")
    client_request_id = _optional_string(form, "client_request_id")
    if manifest_payload is None:
        raise ValidationApiError("A manifest form field is required.")

    text_keys = {
        key
        for key, value in form.multi_items()
        if not isinstance(value, UploadFile)
    }
    unexpected_text_keys = sorted(text_keys - {"manifest", "client_request_id"})
    if unexpected_text_keys:
        raise ValidationApiError(
            f"Unexpected form fields: {', '.join(unexpected_text_keys)}."
        )

    try:
        manifest = ReconstructionManifest.model_validate_json(manifest_payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else "Invalid manifest."
        raise ValidationApiError(f"Invalid manifest: {message}") from exc

    uploads_by_key: dict[str, UploadFile] = {}
    for key, upload in file_items:
        if key in uploads_by_key:
            raise ValidationApiError(f"Multipart file field '{key}' occurs more than once.")
        uploads_by_key[key] = upload

    expected = [view.upload_key for view in manifest.views]
    missing = [key for key in expected if key not in uploads_by_key]
    extra = [key for key in uploads_by_key if key not in set(expected)]
    if missing:
        raise ValidationApiError(f"Missing multipart file parts: {', '.join(missing)}.")
    if extra:
        raise ValidationApiError(f"Unexpected multipart file parts: {', '.join(extra)}.")

    scene = SceneInput(scene_id=manifest.scene_id, views=manifest.views)
    return ParsedMultipartRequest(
        scene=scene,
        uploads=[uploads_by_key[key] for key in expected],
        raw_options=manifest.options,
        client_request_id=client_request_id,
    )
