from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from vggt_serve.app import create_app
from vggt_serve.backends import BackendRunRequest, BackendRunResult, EmptyBackendOptions, ReconstructionBackend
from vggt_serve.config import Settings
from vggt_serve.errors import ServiceBusyApiError, ServiceUnavailableApiError
from vggt_serve.storage import ArtifactDescriptor, PreparedImage, write_depth_artifact, write_point_cloud_ply


def _image_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", size, color=color)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class StubVGGTOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depth_conf_threshold: float | None = Field(default=None, ge=0.0)


class StubBackend(ReconstructionBackend):
    def __init__(
        self,
        *,
        backend_id: str = "vggt",
        capabilities: tuple[str, ...] = ("camera_poses", "depth", "depth_confidence", "point_cloud"),
        ready: bool = True,
        last_error: str | None = None,
        busy: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self.display_name = f"{backend_id}-stub"
        self.capabilities = capabilities
        self._ready = ready
        self._last_error = last_error
        self._busy = busy

    def validate_options(self, payload: dict[str, object] | None) -> BaseModel:
        if self.backend_id == "vggt":
            return StubVGGTOptions.model_validate(payload or {})
        return EmptyBackendOptions.model_validate(payload or {})

    @property
    def device_description(self) -> str | None:
        return "stub-device" if self._ready else None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._last_error:
            raise RuntimeError(self._last_error)
        self._ready = True

    def run(self, request: BackendRunRequest) -> BackendRunResult:
        if self._last_error:
            raise ServiceUnavailableApiError(self._last_error)
        if self._busy:
            raise ServiceBusyApiError()

        artifacts: list[ArtifactDescriptor] = []
        if "depth" in self.capabilities:
            threshold = 5.0
            if self.backend_id == "vggt":
                options = StubVGGTOptions.model_validate(request.backend_options.model_dump(mode="python"))
                if options.depth_conf_threshold is not None:
                    threshold = options.depth_conf_threshold

            depth_maps = [np.ones((image.height, image.width), dtype=np.float32) for image in request.images]
            depth_conf = [
                np.full((image.height, image.width), threshold + 1.0, dtype=np.float32) for image in request.images
            ]
            depth_path = request.run_dir / "depth.npz"
            write_depth_artifact(
                depth_path,
                [image.original_filename for image in request.images],
                [(image.width, image.height) for image in request.images],
                depth_maps,
                depth_conf,
            )
            artifacts.append(
                ArtifactDescriptor(
                    name=depth_path.name,
                    path=depth_path,
                    kind="depth_archive",
                    content_type="application/octet-stream",
                    size_bytes=depth_path.stat().st_size,
                )
            )

        if "point_cloud" in self.capabilities:
            ply_path = request.run_dir / "point_cloud.ply"
            points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            write_point_cloud_ply(ply_path, points, colors)
            artifacts.append(
                ArtifactDescriptor(
                    name=ply_path.name,
                    path=ply_path,
                    kind="point_cloud",
                    content_type="application/octet-stream",
                    size_bytes=ply_path.stat().st_size,
                )
            )

        camera_results = []
        if "camera_poses" in self.capabilities:
            camera_results = [
                {
                    "filename": image.original_filename,
                    "original_size": {"width": image.width, "height": image.height},
                    "cam_from_world": np.eye(4, dtype=np.float32).tolist(),
                    "intrinsics": np.eye(3, dtype=np.float32).tolist(),
                }
                for image in request.images
            ]

        return BackendRunResult(
            camera_results=camera_results,
            artifacts=artifacts,
            produced_outputs=list(self.capabilities),
            timings_ms={"inference": 5, "postprocess": 2, "total": 7},
        )


def _make_client(tmp_path: Path, backend: StubBackend | None = None) -> TestClient:
    settings = Settings(data_root=tmp_path / "runs")
    app = create_app(settings=settings, backend=backend or StubBackend(), load_engine_on_startup=False)
    return TestClient(app)


def test_healthz(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_not_ready(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend=StubBackend(ready=False, last_error="backend load failed"))
    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "ready": False,
        "backend": "vggt",
        "capabilities": ["camera_poses", "depth", "depth_confidence", "point_cloud"],
        "device": None,
        "error": "backend load failed",
    }


def test_reconstruction_success(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[
            ("images", ("image_a.png", _image_bytes((12, 8), (255, 0, 0)), "image/png")),
            ("images", ("image_b.png", _image_bytes((8, 12), (0, 255, 0)), "image/png")),
        ],
        data={
            "scene_id": "scene-1",
            "client_request_id": "client-123",
            "depth_conf_threshold": "4.0",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "vggt"
    assert payload["status"] == "succeeded"
    assert payload["request_id"]
    assert payload["client_request_id"] == "client-123"
    assert payload["input_summary"]["image_count"] == 2
    assert len(payload["camera_results"]) == 2
    assert payload["produced_outputs"] == ["camera_poses", "depth", "depth_confidence", "point_cloud"]
    assert [(artifact["name"], artifact["kind"]) for artifact in payload["artifacts"]] == [
        ("depth.npz", "depth_archive"),
        ("point_cloud.ply", "point_cloud"),
        ("result.json", "result_manifest"),
    ]

    request_path = next((tmp_path / "runs").glob("*/request.json"))
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["backend"] == "vggt"
    assert request_payload["backend_options"] == {"depth_conf_threshold": 4.0}

    artifact_url = payload["artifacts"][1]["url"]
    artifact_path = urlparse(artifact_url).path
    artifact_response = client.get(artifact_path)
    assert artifact_response.status_code == 200
    assert artifact_response.content.startswith(b"ply\nformat binary_little_endian 1.0\n")


def test_reconstruction_rejects_corrupt_image(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("broken.png", b"not-a-real-image", "image/png"))],
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "validation_error"


def test_reconstruction_rejects_unsupported_media(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("notes.txt", b"text", "text/plain"))],
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media"


def test_reconstruction_returns_busy(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend=StubBackend(busy=True))
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("image.png", _image_bytes((10, 10), (0, 0, 255)), "image/png"))],
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_busy"


def test_non_vggt_backend_rejects_compat_threshold_field(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend=StubBackend(backend_id="depth-anything3", capabilities=("depth",)))
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("image.png", _image_bytes((10, 10), (0, 0, 255)), "image/png"))],
        data={"depth_conf_threshold": "2.0"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_depth_only_backend_returns_empty_camera_results(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend=StubBackend(backend_id="depth-anything3", capabilities=("depth",)))
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("image.png", _image_bytes((10, 10), (0, 0, 255)), "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "depth-anything3"
    assert payload["camera_results"] == []
    assert payload["produced_outputs"] == ["depth"]
    assert [(artifact["name"], artifact["kind"]) for artifact in payload["artifacts"]] == [
        ("depth.npz", "depth_archive"),
        ("result.json", "result_manifest"),
    ]


def test_backend_options_must_be_json_object(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("image.png", _image_bytes((10, 10), (0, 0, 255)), "image/png"))],
        data={"backend_options": '["not-an-object"]'},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"
