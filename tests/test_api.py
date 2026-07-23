from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from gfm_serve.app import create_app
from gfm_serve.backends import BackendRunRequest, BackendRunResult, EmptyBackendOptions, ReconstructionBackend
from gfm_serve.config import Settings
from gfm_serve.contracts import BackendDescriptor, CameraResult, ImageSize, SceneInput, ViewResult
from gfm_serve.errors import ServiceBusyApiError, ServiceUnavailableApiError
from gfm_serve.storage import ArtifactDescriptor, PreparedImage, write_depth_artifact, write_point_cloud_ply
from gfm_backend_vggt import VGGTBackend


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
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend=self.backend_id,
            model_id=f"{self.backend_id}/stub",
            inputs={
                "images": {"required": True},
                "camera.intrinsics": {"required": False},
                "camera.world_to_camera": {"required": False},
            },
            outputs=list(self.capabilities),
            options_schema=self.validate_options({}).__class__.model_json_schema(),
        )

    def validate_request(self, scene: SceneInput, options: dict[str, object] | None) -> BaseModel:
        return self.validate_options(options)

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

        view_results = [
            ViewResult(
                view_id=view.view_id,
                filename=view.image.original_filename,
                original_size=ImageSize(width=view.image.width, height=view.image.height),
            )
            for view in request.views
        ]
        if "camera_poses" in self.capabilities:
            view_results = [
                ViewResult(
                    view_id=view.view_id,
                    filename=view.image.original_filename,
                    original_size=ImageSize(width=view.image.width, height=view.image.height),
                    camera=CameraResult(
                        convention="opencv",
                        world_to_camera=np.eye(4, dtype=np.float32).tolist(),
                        intrinsics=np.eye(3, dtype=np.float32).tolist(),
                        source="predicted",
                    ),
                )
                for view in request.views
            ]

        return BackendRunResult(
            view_results=view_results,
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
        "model_descriptor_url": "/v1/models/current",
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
    assert payload["service_version"] == "0.2.0"
    assert payload["result_schema_version"] == "1.0"
    assert payload["model"]["backend"] == "vggt"
    assert payload["status"] == "succeeded"
    assert payload["request_id"]
    assert payload["client_request_id"] == "client-123"
    assert response.headers["deprecation"] == "true"
    assert payload["input_summary"]["image_count"] == 2
    assert payload["normalized_request"]["scene"]["scene_id"] == "scene-1"
    assert payload["warnings"] == [
        "depth_conf_threshold was translated to VGGT backend options."
    ]
    assert len(payload["camera_results"]) == 2
    assert set(payload["camera_results"][0]) == {
        "filename",
        "original_size",
        "cam_from_world",
        "intrinsics",
    }
    assert payload["camera_results"][0]["cam_from_world"] == np.eye(4).tolist()
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
    assert request_payload["schema_version"] == "1.0"
    assert [view["view_id"] for view in request_payload["normalized_manifest"]["views"]] == [
        "view-000",
        "view-001",
    ]

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
    assert len(payload["view_results"]) == 1
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


def test_manifest_request_associates_uploads_by_key(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend=StubBackend(backend_id="depth-anything3"))
    manifest = {
        "scene_id": "office",
        "views": [
            {
                "view_id": "right",
                "upload_key": "image_b",
                "camera": {
                    "convention": "opencv",
                    "world_to_camera": np.eye(4).tolist(),
                    "intrinsics": np.eye(3).tolist(),
                },
            },
            {"view_id": "left", "upload_key": "image_a"},
        ],
        "options": {},
    }

    response = client.post(
        "/v1/reconstructions",
        files=[
            ("image_a", ("left.png", _image_bytes((8, 6), (255, 0, 0)), "image/png")),
            ("image_b", ("right.png", _image_bytes((9, 7), (0, 255, 0)), "image/png")),
        ],
        data={"manifest": json.dumps(manifest)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["view_id"] for item in payload["view_results"]] == ["right", "left"]
    assert [item["filename"] for item in payload["view_results"]] == ["right.png", "left.png"]


def test_manifest_rejects_missing_and_extra_file_parts(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    manifest = {"views": [{"view_id": "one", "upload_key": "expected"}], "options": {}}

    missing = client.post("/v1/reconstructions", data={"manifest": json.dumps(manifest)})
    extra = client.post(
        "/v1/reconstructions",
        files=[
            ("expected", ("ok.png", _image_bytes((4, 4), (0, 0, 0)), "image/png")),
            ("extra", ("extra.png", _image_bytes((4, 4), (0, 0, 0)), "image/png")),
        ],
        data={"manifest": json.dumps(manifest)},
    )

    assert missing.status_code == 400
    assert "Missing multipart" in missing.json()["error"]["message"]
    assert extra.status_code == 400
    assert "Unexpected multipart" in extra.json()["error"]["message"]


def test_manifest_rejects_legacy_images_mixed_in(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    manifest = {"views": [{"view_id": "one", "upload_key": "expected"}], "options": {}}
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("legacy.png", _image_bytes((4, 4), (0, 0, 0)), "image/png"))],
        data={"manifest": json.dumps(manifest)},
    )

    assert response.status_code == 400
    assert "Do not mix" in response.json()["error"]["message"]


def test_current_model_exposes_descriptor_and_options_schema(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    response = client.get("/v1/models/current")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "vggt"
    assert payload["inputs"]["images"]["required"] is True
    assert payload["options_schema"]["additionalProperties"] is False


def test_manifest_rejects_invalid_camera_matrix(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend=StubBackend(backend_id="depth-anything3"))
    manifest = {
        "views": [
            {
                "view_id": "one",
                "upload_key": "image",
                "camera": {
                    "convention": "opencv",
                    "world_to_camera": [[1, 0], [0, 1]],
                },
            }
        ],
        "options": {},
    }
    response = client.post(
        "/v1/reconstructions",
        files=[("image", ("image.png", _image_bytes((4, 4), (0, 0, 0)), "image/png"))],
        data={"manifest": json.dumps(manifest)},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


def test_vggt_rejects_supplied_camera_input(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    manifest = {
        "views": [
            {
                "view_id": "one",
                "upload_key": "image",
                "camera": {
                    "convention": "opencv",
                    "world_to_camera": np.eye(4).tolist(),
                    "intrinsics": np.eye(3).tolist(),
                },
            }
        ],
        "options": {},
    }
    response = client.post(
        "/v1/reconstructions",
        files=[("image", ("image.png", _image_bytes((4, 4), (0, 0, 0)), "image/png"))],
        data={"manifest": json.dumps(manifest)},
    )

    assert response.status_code == 200  # Stub VGGT accepts cameras for transport isolation.
    real_backend = VGGTBackend(Settings(data_root=tmp_path / "real-runs"))
    with pytest.raises(ValueError, match="does not accept supplied camera"):
        real_backend.validate_request(SceneInput.model_validate({"views": manifest["views"]}), {})
