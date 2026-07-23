from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from .backends import BackendRunRequest, BackendRunResult, PreparedView, ReconstructionBackend
from .contracts import BackendDescriptor, ViewResult
from .config import Settings
from .errors import (
    ApiError,
    ArtifactNotFoundApiError,
    ExecutionFailedApiError,
    ServiceUnavailableApiError,
    UnsupportedMediaApiError,
    ValidationApiError,
)
from .schemas import (
    ArtifactInfo,
    ErrorInfo,
    HealthResponse,
    InputSummary,
    ReadyResponse,
    ReconstructionResponse,
    TimingStats,
)
from .request_parser import parse_reconstruction_multipart
from .storage import ArtifactDescriptor, PreparedImage, sanitize_filename, write_json


LOGGER = logging.getLogger(__name__)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
CONTENT_TYPE_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_backend(request: Request) -> ReconstructionBackend:
    return request.app.state.backend


async def _prepare_uploads(
    *,
    files: list[UploadFile] | None,
    run_dir: Path,
    settings: Settings,
) -> tuple[list[PreparedImage], int]:
    if not files:
        raise ValidationApiError("At least one image is required.")
    if len(files) > settings.max_images:
        raise ValidationApiError(f"At most {settings.max_images} images are allowed per request.")

    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    prepared_images: list[PreparedImage] = []
    total_bytes = 0

    for index, upload in enumerate(files, start=1):
        content_type = upload.content_type or ""
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise UnsupportedMediaApiError(
                f"Unsupported content type for '{upload.filename or f'image_{index}'}': {content_type or 'unknown'}."
            )

        payload = await upload.read()
        await upload.close()

        if not payload:
            raise ValidationApiError(f"Uploaded file '{upload.filename or f'image_{index}'}' is empty.")
        if len(payload) > settings.max_upload_bytes_per_file:
            raise ValidationApiError(
                f"Uploaded file '{upload.filename or f'image_{index}'}' exceeds the per-file size limit."
            )

        total_bytes += len(payload)
        if total_bytes > settings.max_upload_bytes_total:
            raise ValidationApiError("Combined upload size exceeds the total request size limit.")

        try:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                width, height = image.convert("RGB").size
        except (UnidentifiedImageError, OSError) as exc:
            raise ValidationApiError(f"Uploaded file '{upload.filename or f'image_{index}'}' is not a valid image.") from exc

        original_name = upload.filename or f"image_{index}"
        safe_name = sanitize_filename(original_name, f"image_{index}")
        if Path(safe_name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            safe_name += CONTENT_TYPE_EXTENSION[content_type]
        stored_name = f"{index:03d}_{safe_name}"
        output_path = input_dir / stored_name
        output_path.write_bytes(payload)

        prepared_images.append(
            PreparedImage(
                original_filename=original_name,
                stored_filename=stored_name,
                path=output_path,
                size_bytes=len(payload),
                width=width,
                height=height,
                content_type=content_type,
            )
        )

    return prepared_images, total_bytes


def _artifact_to_response(request: Request, request_id: str, artifact: ArtifactDescriptor) -> ArtifactInfo:
    return ArtifactInfo(
        name=artifact.name,
        kind=artifact.kind,
        url=str(request.url_for("download_artifact", request_id=request_id, name=artifact.name)),
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        metadata=artifact.metadata,
    )


def _build_response(
    *,
    backend_id: str,
    backend_descriptor: BackendDescriptor,
    request_id: str,
    client_request_id: str | None,
    status_value: str,
    input_summary: InputSummary | None,
    timings_ms: TimingStats,
    view_results: list[ViewResult] | None = None,
    artifacts: list[ArtifactInfo] | None = None,
    produced_outputs: list[str] | None = None,
    error: ErrorInfo | None = None,
    normalized_request: dict[str, object] | None = None,
    warnings: list[str] | None = None,
) -> ReconstructionResponse:
    return ReconstructionResponse(
        backend=backend_id,
        model=backend_descriptor,
        request_id=request_id,
        client_request_id=client_request_id,
        status=status_value,  # type: ignore[arg-type]
        input_summary=input_summary,
        timings_ms=timings_ms,
        view_results=view_results or [],
        camera_results=[result for result in (view_results or []) if result.camera is not None],
        artifacts=artifacts or [],
        produced_outputs=produced_outputs or [],
        normalized_request=normalized_request,
        warnings=warnings or [],
        error=error,
    )


def _parse_backend_options(
    *,
    backend: ReconstructionBackend,
    scene,
    raw_options: dict[str, object],
    depth_conf_threshold: float | None,
) -> BaseModel:
    raw_options = dict(raw_options)

    if depth_conf_threshold is not None:
        if backend.backend_id != "vggt":
            raise ValidationApiError(
                f"depth_conf_threshold is only supported by the 'vggt' backend. Use backend_options for '{backend.backend_id}'."
            )
        raw_options.setdefault("depth_conf_threshold", depth_conf_threshold)

    try:
        return backend.validate_request(scene, raw_options)
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            message = exc.errors()[0]["msg"] if exc.errors() else "Invalid backend_options."
        else:
            message = str(exc)
        raise ValidationApiError(f"Invalid backend_options for '{backend.backend_id}': {message}") from exc


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(request: Request) -> JSONResponse:
    backend = get_backend(request)
    ready = backend.is_ready()
    payload = ReadyResponse(
        status="ready" if ready else "not_ready",
        ready=ready,
        backend=backend.backend_id,
        capabilities=list(backend.capabilities),
        device=backend.device_description,
        error=backend.last_error,
    )
    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(payload.model_dump(mode="json"), status_code=status_code)


@router.get("/v1/models/current", response_model=BackendDescriptor)
async def current_model(request: Request) -> BackendDescriptor:
    return get_backend(request).descriptor


@router.get("/v1/artifacts/{request_id}/{name}", name="download_artifact")
async def download_artifact(request_id: str, name: str, request: Request) -> FileResponse:
    settings = get_settings(request)
    run_dir = (settings.data_root / request_id).resolve()
    candidate = (run_dir / name).resolve()

    if candidate.parent != run_dir or not candidate.exists() or not candidate.is_file():
        raise ArtifactNotFoundApiError()

    media_type = "application/octet-stream"
    if candidate.suffix == ".json":
        media_type = "application/json"
    return FileResponse(candidate, media_type=media_type, filename=candidate.name)


@router.post("/v1/reconstructions", response_model=ReconstructionResponse)
async def create_reconstruction(
    request: Request,
) -> JSONResponse:
    settings = get_settings(request)
    backend = get_backend(request)
    request_id = uuid4().hex
    run_dir = settings.data_root / request_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not backend.is_ready() and backend.last_error:
        error = ServiceUnavailableApiError(backend.last_error)
    else:
        error = None

    start = perf_counter()
    prepared_images: list[PreparedImage] = []
    input_summary: InputSummary | None = None
    validated_backend_options: BaseModel | None = None
    validation_ms = 0
    parsed_request = None
    scene_id: str | None = None
    client_request_id: str | None = None

    request_payload = {
        "request_id": request_id,
        "backend": backend.backend_id,
        "schema_version": "1.0",
        "scene_id": None,
        "client_request_id": None,
        "backend_options": None,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "filenames": [],
    }
    write_json(run_dir / "request.json", request_payload)

    try:
        if error is not None:
            raise error

        parsed_request = await parse_reconstruction_multipart(request)
        scene_id = parsed_request.scene.scene_id
        client_request_id = parsed_request.client_request_id
        request_payload["scene_id"] = scene_id
        request_payload["client_request_id"] = client_request_id
        request_payload["filenames"] = [upload.filename for upload in parsed_request.uploads]
        request_payload["normalized_manifest"] = parsed_request.scene.model_dump(mode="json")
        validated_backend_options = _parse_backend_options(
            backend=backend,
            scene=parsed_request.scene,
            raw_options=parsed_request.raw_options,
            depth_conf_threshold=parsed_request.compatibility_threshold,
        )
        prepared_images, total_bytes = await _prepare_uploads(
            files=parsed_request.uploads, run_dir=run_dir, settings=settings
        )
        validation_ms = int((perf_counter() - start) * 1000)
        input_summary = InputSummary(
            scene_id=scene_id,
            image_count=len(prepared_images),
            filenames=[image.original_filename for image in prepared_images],
            total_bytes=total_bytes,
        )
        prepared_views = [
            PreparedView(view_id=view.view_id, upload_key=view.upload_key, image=image)
            for view, image in zip(parsed_request.scene.views, prepared_images, strict=True)
        ]

        request_payload["files"] = [
            {
                "original_filename": image.original_filename,
                "stored_filename": image.stored_filename,
                "content_type": image.content_type,
                "size_bytes": image.size_bytes,
                "width": image.width,
                "height": image.height,
            }
            for image in prepared_images
        ]
        request_payload["backend_options"] = validated_backend_options.model_dump(mode="json")
        write_json(run_dir / "request.json", request_payload)

        result: BackendRunResult = backend.run(
            BackendRunRequest(
                request_id=request_id,
                run_dir=run_dir,
                scene=parsed_request.scene,
                views=prepared_views,
                backend_options=validated_backend_options,
            )
        )
        expected_view_ids = [view.view_id for view in parsed_request.scene.views]
        actual_view_ids = [view.view_id for view in result.view_results]
        if actual_view_ids != expected_view_ids:
            raise RuntimeError(
                f"Backend '{backend.backend_id}' returned view IDs {actual_view_ids}; "
                f"expected {expected_view_ids}."
            )

        result_json_artifact = run_dir / "result.json"
        artifact_descriptors = list(result.artifacts)
        artifact_descriptors.append(
            ArtifactDescriptor(
                name=result_json_artifact.name,
                path=result_json_artifact,
                kind="result_manifest",
                content_type="application/json",
                size_bytes=0,
            )
        )
        artifacts = [_artifact_to_response(request, request_id, artifact) for artifact in artifact_descriptors]

        response_payload = _build_response(
            backend_id=backend.backend_id,
            backend_descriptor=backend.descriptor,
            request_id=request_id,
            client_request_id=client_request_id,
            status_value="succeeded",
            input_summary=input_summary,
            timings_ms=TimingStats(
                validation=validation_ms,
                inference=result.timings_ms["inference"],
                postprocess=result.timings_ms["postprocess"],
                total=int((perf_counter() - start) * 1000),
            ),
            view_results=result.view_results,
            artifacts=artifacts,
            produced_outputs=result.produced_outputs,
            normalized_request={
                "scene": parsed_request.scene.model_dump(mode="json"),
                "options": validated_backend_options.model_dump(mode="json"),
            },
            warnings=result.warnings
            + (
                ["depth_conf_threshold was translated to VGGT backend options."]
                if parsed_request.compatibility_threshold is not None
                else []
            ),
        )
        write_json(result_json_artifact, response_payload.model_dump(mode="json"))

        result_json_artifact_descriptor = artifacts[-1].model_copy(
            update={"size_bytes": result_json_artifact.stat().st_size}
        )
        response_payload.artifacts[-1] = result_json_artifact_descriptor
        write_json(result_json_artifact, response_payload.model_dump(mode="json"))

        LOGGER.info(
            "Request succeeded",
            extra={
                "request_id": request_id,
                "backend": backend.backend_id,
                "scene_id": scene_id,
                "status": "succeeded",
                "image_count": len(prepared_images),
                "device": backend.device_description,
                "timings_ms": response_payload.timings_ms.model_dump(mode="json"),
            },
        )
        headers = {}
        if parsed_request.compatibility_threshold is not None:
            headers["Deprecation"] = "true"
            headers["Warning"] = '299 - "depth_conf_threshold is deprecated; use manifest.options"'
        return JSONResponse(
            response_payload.model_dump(mode="json"), status_code=status.HTTP_200_OK, headers=headers
        )

    except ApiError as exc:
        response_payload = _build_response(
            backend_id=backend.backend_id,
            backend_descriptor=backend.descriptor,
            request_id=request_id,
            client_request_id=client_request_id,
            status_value="failed",
            input_summary=input_summary,
            timings_ms=TimingStats(
                validation=validation_ms or int((perf_counter() - start) * 1000),
                total=int((perf_counter() - start) * 1000),
            ),
            error=ErrorInfo(code=exc.code, message=exc.message),
            normalized_request=(
                {"scene": parsed_request.scene.model_dump(mode="json")}
                if parsed_request is not None
                else None
            ),
        )
        write_json(run_dir / "result.json", response_payload.model_dump(mode="json"))
        LOGGER.warning(
            "Request failed",
            extra={
                "request_id": request_id,
                "backend": backend.backend_id,
                "scene_id": scene_id,
                "status": "failed",
                "code": exc.code,
                "image_count": len(prepared_images),
            },
        )
        return JSONResponse(response_payload.model_dump(mode="json"), status_code=exc.status_code)
    except Exception as exc:  # pragma: no cover - protection for runtime failures
        api_error = ExecutionFailedApiError(str(exc))
        response_payload = _build_response(
            backend_id=backend.backend_id,
            backend_descriptor=backend.descriptor,
            request_id=request_id,
            client_request_id=client_request_id,
            status_value="failed",
            input_summary=input_summary,
            timings_ms=TimingStats(
                validation=validation_ms or int((perf_counter() - start) * 1000),
                total=int((perf_counter() - start) * 1000),
            ),
            error=ErrorInfo(code=api_error.code, message=api_error.message),
        )
        write_json(run_dir / "result.json", response_payload.model_dump(mode="json"))
        LOGGER.exception(
            "Unhandled request failure",
            extra={
                "request_id": request_id,
                "backend": backend.backend_id,
                "scene_id": scene_id,
                "status": "failed",
                "code": api_error.code,
                "image_count": len(prepared_images),
            },
        )
        return JSONResponse(response_payload.model_dump(mode="json"), status_code=api_error.status_code)
