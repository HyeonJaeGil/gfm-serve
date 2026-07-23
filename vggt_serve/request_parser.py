from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from .contracts import SceneInput, ViewInput
from .errors import ValidationApiError


class ReconstructionManifest(SceneInput):
    model_config = ConfigDict(extra="forbid")

    options: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class ParsedMultipartRequest:
    scene: SceneInput
    uploads: list[UploadFile]
    upload_keys: list[str]
    raw_options: dict[str, Any]
    client_request_id: str | None
    compatibility_threshold: float | None
    legacy: bool


def _optional_string(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationApiError(f"{key} must be a text form field.")
    return value


def _parse_json_object(value: str | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationApiError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValidationApiError(f"{field_name} must be a JSON object.")
    return parsed


async def parse_reconstruction_multipart(request: Request) -> ParsedMultipartRequest:
    form = await request.form()
    file_items = [(key, value) for key, value in form.multi_items() if isinstance(value, UploadFile)]
    manifest_payload = _optional_string(form, "manifest")
    legacy_uploads = [(key, upload) for key, upload in file_items if key == "images"]

    if manifest_payload is not None and legacy_uploads:
        raise ValidationApiError("Do not mix manifest uploads with legacy repeated 'images' fields.")

    backend_options = _parse_json_object(_optional_string(form, "backend_options"), "backend_options")
    client_request_id = _optional_string(form, "client_request_id")
    threshold_text = _optional_string(form, "depth_conf_threshold")
    try:
        compatibility_threshold = float(threshold_text) if threshold_text is not None else None
    except ValueError as exc:
        raise ValidationApiError("depth_conf_threshold must be a number.") from exc

    if manifest_payload is None:
        non_legacy_files = [key for key, _ in file_items if key != "images"]
        if non_legacy_files:
            raise ValidationApiError("File parts other than 'images' require a manifest.")
        scene_id = _optional_string(form, "scene_id")
        views = [
            ViewInput(view_id=f"view-{index:03d}", upload_key=f"images-{index:03d}")
            for index, _ in enumerate(legacy_uploads)
        ]
        if not views:
            raise ValidationApiError("At least one image is required.")
        return ParsedMultipartRequest(
            scene=SceneInput(scene_id=scene_id, views=views),
            uploads=[upload for _, upload in legacy_uploads],
            upload_keys=[view.upload_key for view in views],
            raw_options=backend_options,
            client_request_id=client_request_id,
            compatibility_threshold=compatibility_threshold,
            legacy=True,
        )

    if _optional_string(form, "scene_id") is not None:
        raise ValidationApiError("scene_id belongs in the manifest when manifest is used.")
    if backend_options:
        raise ValidationApiError("Use manifest.options instead of backend_options with a manifest request.")

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
        upload_keys=expected,
        raw_options=manifest.options,
        client_request_id=client_request_id,
        compatibility_threshold=compatibility_threshold,
        legacy=False,
    )
