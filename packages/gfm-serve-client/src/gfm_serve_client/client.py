from __future__ import annotations

import json
import mimetypes
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx
import numpy as np

from .errors import (
    BackendMismatchError,
    GFMServeAPIError,
    InvalidRequestError,
)
from .models import (
    Artifact,
    BackendDescriptor,
    CameraParameters,
    DepthAnything3Options,
    ImagePath,
    Options,
    ReadyStatus,
    ReconstructionResult,
    VGGTOptions,
    options_payload,
)


class GFMServeClient:
    """Synchronous typed client for a GFM Serve HTTP process."""

    expected_backend: str | None = None

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float | httpx.Timeout = 600.0,
        transport: httpx.BaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers=headers,
        )
        self._verified_backend: str | None = None

    def __enter__(self) -> GFMServeClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        response = self._client.get("/healthz")
        self._raise_for_response(response)
        return response.json().get("status") == "ok"

    def ready(self) -> ReadyStatus:
        response = self._client.get("/readyz")
        payload = self._json_payload(response)
        if response.is_error and not (
            response.status_code == httpx.codes.SERVICE_UNAVAILABLE
            and payload.get("ready") is False
        ):
            self._raise_api_error(response, payload)
        status = ReadyStatus.model_validate(payload)
        self._verify_backend(status.backend)
        return status

    def model_descriptor(self, *, refresh: bool = False) -> BackendDescriptor:
        if refresh:
            self._verified_backend = None
        response = self._client.get("/v1/models/current")
        self._raise_for_response(response)
        descriptor = BackendDescriptor.model_validate(response.json())
        self._verify_backend(descriptor.backend)
        return descriptor

    def reconstruct(
        self,
        images: Sequence[ImagePath],
        *,
        cameras: Sequence[CameraParameters] | None = None,
        options: Options = None,
        scene_id: str | None = None,
        view_ids: Sequence[str] | None = None,
        client_request_id: str | None = None,
    ) -> ReconstructionResult:
        image_paths = self._validate_images(images)
        ids = self._view_ids(image_paths, view_ids)
        if cameras is not None and len(cameras) != len(image_paths):
            raise InvalidRequestError(
                f"Expected {len(image_paths)} cameras; got {len(cameras)}."
            )

        views: list[dict[str, Any]] = []
        for index, (path, view_id) in enumerate(zip(image_paths, ids, strict=True)):
            view: dict[str, Any] = {
                "view_id": view_id,
                "upload_key": f"image_{index:03d}",
            }
            if cameras is not None:
                camera = cameras[index]
                if not isinstance(camera, CameraParameters):
                    raise InvalidRequestError(
                        f"cameras[{index}] must be a CameraParameters instance."
                    )
                view["camera"] = camera.model_dump(mode="json")
            views.append(view)

        manifest = {
            "scene_id": scene_id,
            "views": views,
            "options": options_payload(options),
        }
        data = {"manifest": json.dumps(manifest, separators=(",", ":"))}
        if client_request_id is not None:
            data["client_request_id"] = client_request_id

        self._ensure_expected_backend()
        with ExitStack() as stack:
            files = {
                view["upload_key"]: (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
                for path, view in zip(image_paths, views, strict=True)
            }
            response = self._client.post("/v1/reconstructions", data=data, files=files)

        payload = self._json_payload(response)
        if response.is_error or payload.get("status") == "failed":
            self._raise_api_error(response, payload)
        result = ReconstructionResult.model_validate(payload)
        self._verify_backend(result.backend)
        return result

    def download_artifact(
        self,
        artifact: Artifact,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        destination_path = Path(destination).expanduser()
        if destination_path.exists() and destination_path.is_dir():
            destination_path = destination_path / artifact.name
        if destination_path.exists() and not overwrite:
            raise FileExistsError(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", artifact.url) as response:
            self._raise_for_response(response)
            with destination_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return destination_path

    def download_artifacts(
        self,
        result: ReconstructionResult,
        directory: str | Path,
        *,
        kinds: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        selected_kinds = set(kinds) if kinds is not None else None
        output_dir = Path(directory).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.download_artifact(
                artifact,
                output_dir / artifact.name,
                overwrite=overwrite,
            )
            for artifact in result.artifacts
            if selected_kinds is None or artifact.kind in selected_kinds
        ]

    def load_depth_archive(
        self,
        result: ReconstructionResult,
    ) -> dict[str, np.ndarray]:
        artifact = result.artifact(kind="depth_archive")
        response = self._client.get(artifact.url)
        self._raise_for_response(response)
        with np.load(BytesIO(response.content), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def _verify_backend(self, actual: str) -> None:
        if self.expected_backend is not None and actual != self.expected_backend:
            raise BackendMismatchError(expected=self.expected_backend, actual=actual)
        self._verified_backend = actual

    def _ensure_expected_backend(self) -> None:
        if self.expected_backend is not None and self._verified_backend is None:
            self.model_descriptor()

    @staticmethod
    def _validate_images(images: Sequence[ImagePath]) -> list[Path]:
        if not images:
            raise InvalidRequestError("At least one image is required.")
        paths = [Path(image).expanduser().resolve() for image in images]
        for path in paths:
            if not path.is_file():
                raise InvalidRequestError(f"Image does not exist or is not a file: {path}")
        return paths

    @staticmethod
    def _view_ids(paths: Sequence[Path], view_ids: Sequence[str] | None) -> list[str]:
        ids = (
            list(view_ids)
            if view_ids is not None
            else [f"view-{index:03d}" for index in range(len(paths))]
        )
        if len(ids) != len(paths):
            raise InvalidRequestError(f"Expected {len(paths)} view IDs; got {len(ids)}.")
        if any(not view_id for view_id in ids):
            raise InvalidRequestError("View IDs must not be empty.")
        if len(ids) != len(set(ids)):
            raise InvalidRequestError("View IDs must be unique.")
        return ids

    @staticmethod
    def _json_payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise GFMServeAPIError(
                status_code=response.status_code,
                code="invalid_response",
                message="The service returned a non-JSON response.",
            ) from exc
        if not isinstance(payload, dict):
            raise GFMServeAPIError(
                status_code=response.status_code,
                code="invalid_response",
                message="The service returned a JSON value that is not an object.",
            )
        return payload

    @classmethod
    def _raise_api_error(
        cls, response: httpx.Response, payload: dict[str, Any]
    ) -> None:
        error = payload.get("error")
        error_payload = error if isinstance(error, dict) else {}
        raise GFMServeAPIError(
            status_code=response.status_code,
            code=str(error_payload.get("code", "http_error")),
            message=str(error_payload.get("message", response.reason_phrase)),
            request_id=(
                str(payload["request_id"]) if payload.get("request_id") is not None else None
            ),
            response_payload=payload,
        )

    @classmethod
    def _raise_for_response(cls, response: httpx.Response) -> None:
        if response.is_error:
            cls._raise_api_error(response, cls._json_payload(response))


class VGGTClient(GFMServeClient):
    """Backend-specific client that rejects accidental connections to DA3."""

    expected_backend = "vggt"

    def reconstruct(
        self,
        images: Sequence[ImagePath],
        *,
        options: VGGTOptions | Mapping[str, Any] | None = None,
        depth_conf_threshold: float | None = None,
        scene_id: str | None = None,
        view_ids: Sequence[str] | None = None,
        client_request_id: str | None = None,
    ) -> ReconstructionResult:
        if options is not None and depth_conf_threshold is not None:
            raise InvalidRequestError(
                "Pass either options or depth_conf_threshold, not both."
            )
        selected_options: Options = (
            VGGTOptions(depth_conf_threshold=depth_conf_threshold)
            if depth_conf_threshold is not None
            else options
        )
        return super().reconstruct(
            images,
            options=selected_options,
            scene_id=scene_id,
            view_ids=view_ids,
            client_request_id=client_request_id,
        )


class DepthAnything3Client(GFMServeClient):
    """Backend-specific client with typed pose-conditioned DA3 input."""

    expected_backend = "depth-anything-3"

    def reconstruct(
        self,
        images: Sequence[ImagePath],
        *,
        cameras: Sequence[CameraParameters] | None = None,
        options: DepthAnything3Options | Mapping[str, Any] | None = None,
        scene_id: str | None = None,
        view_ids: Sequence[str] | None = None,
        client_request_id: str | None = None,
    ) -> ReconstructionResult:
        return super().reconstruct(
            images,
            cameras=cameras,
            options=options,
            scene_id=scene_id,
            view_ids=view_ids,
            client_request_id=client_request_id,
        )
