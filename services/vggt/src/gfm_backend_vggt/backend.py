from __future__ import annotations

import logging
import threading
from time import perf_counter

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from gfm_serve.config import Settings
from gfm_serve.contracts import BackendDescriptor, CameraResult, ImageSize, SceneInput, ViewResult
from gfm_serve.errors import ServiceBusyApiError, ServiceUnavailableApiError
from gfm_serve.storage import (
    ArtifactDescriptor,
    remap_square_tensor_to_original,
    rescale_intrinsics_to_original,
    sample_point_cloud,
    write_depth_artifact,
    write_point_cloud_ply,
)
from gfm_serve.backends.base import BackendRunRequest, BackendRunResult, ReconstructionBackend

from .config import VGGTBackendSettings


LOGGER = logging.getLogger(__name__)


class VGGTBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depth_conf_threshold: float | None = Field(default=None, ge=0.0)


class VGGTBackend(ReconstructionBackend):
    backend_id = "vggt"
    display_name = "VGGT"
    capabilities = ("camera_poses", "depth", "depth_confidence", "point_cloud")
    options_model = VGGTBackendOptions

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend_settings = VGGTBackendSettings()
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._model = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._last_error: str | None = None

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend=self.backend_id,
            model_id=self.backend_settings.model_id,
            inputs={"images": {"required": True}},
            outputs=list(self.capabilities),
            options_schema=self.options_model.model_json_schema(),
        )

    def validate_request(self, scene: SceneInput, options: dict[str, object] | None) -> BaseModel:
        if any(view.camera is not None for view in scene.views):
            raise ValueError("VGGT does not accept supplied camera inputs.")
        return self.validate_options(options)

    @property
    def device_description(self) -> str | None:
        if self._device is None:
            return None
        if self._device.type == "cuda":
            return f"cuda:{self._torch.cuda.get_device_name(0)}"
        return self._device.type

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            try:
                import torch
                from vggt.models.vggt import VGGT

                self._torch = torch
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self._device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
                    self._dtype = torch.bfloat16
                elif self._device.type == "cuda":
                    self._dtype = torch.float16
                else:
                    self._dtype = torch.float32

                torch.set_float32_matmul_precision("high")

                checkpoint_path = self.backend_settings.checkpoint_path
                if checkpoint_path is not None:
                    resolved = checkpoint_path.expanduser().resolve()
                    if resolved.is_dir():
                        model = VGGT.from_pretrained(str(resolved))
                    else:
                        model = VGGT()
                        state_dict = torch.load(resolved, map_location="cpu")
                        if isinstance(state_dict, dict) and "state_dict" in state_dict:
                            state_dict = state_dict["state_dict"]
                        model.load_state_dict(state_dict)
                else:
                    model = VGGT.from_pretrained(self.backend_settings.model_id)

                model.point_head = None
                model.track_head = None
                model.eval()
                model = model.to(self._device)

                self._model = model
                self._last_error = None
            except Exception as exc:  # pragma: no cover - startup/runtime protection
                self._last_error = str(exc)
                raise

    def run(self, request: BackendRunRequest) -> BackendRunResult:
        if self._model is None:
            try:
                self.load()
            except Exception as exc:  # pragma: no cover - startup/runtime protection
                raise ServiceUnavailableApiError(str(exc)) from exc

        if self._model is None or self._torch is None or self._device is None:
            raise ServiceUnavailableApiError(self._last_error or f"{self.display_name} backend is not ready.")

        if not self._run_lock.acquire(blocking=False):
            raise ServiceBusyApiError()

        try:
            options = VGGTBackendOptions.model_validate(request.backend_options.model_dump(mode="python"))
            depth_conf_threshold = options.depth_conf_threshold
            if depth_conf_threshold is None:
                depth_conf_threshold = self.backend_settings.default_depth_conf_threshold
            return self._run_locked(request=request, depth_conf_threshold=depth_conf_threshold)
        finally:
            self._run_lock.release()
            if self._device.type == "cuda":
                self._torch.cuda.empty_cache()

    def _run_locked(
        self,
        *,
        request: BackendRunRequest,
        depth_conf_threshold: float,
    ) -> BackendRunResult:
        try:
            from vggt.utils.geometry import unproject_depth_map_to_point_map
            from vggt.utils.load_fn import load_and_preprocess_images_square
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except Exception as exc:  # pragma: no cover - runtime protection
            raise ServiceUnavailableApiError(str(exc)) from exc

        start = perf_counter()
        preprocess_start = perf_counter()
        image_paths = [str(image.path) for image in request.images]
        tensor_images, _ = load_and_preprocess_images_square(
            image_paths,
            target_size=self.backend_settings.square_image_size,
        )
        tensor_images = tensor_images.to(self._device)

        with self._torch.inference_mode():
            with self._torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._device.type == "cuda",
                dtype=self._dtype,
            ):
                batched_images = tensor_images[None]
                aggregated_tokens_list, patch_start_idx = self._model.aggregator(batched_images)
                pose_enc = self._model.camera_head(aggregated_tokens_list)[-1]
                depth, depth_conf = self._model.depth_head(
                    aggregated_tokens_list,
                    batched_images,
                    patch_start_idx,
                )

            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched_images.shape[-2:])

        inference_ms = int((perf_counter() - preprocess_start) * 1000)

        extrinsic_np = extrinsic.squeeze(0).detach().to(self._torch.float32).cpu().numpy().astype(np.float32)
        intrinsic_np = intrinsic.squeeze(0).detach().to(self._torch.float32).cpu().numpy().astype(np.float32)
        depth_np = depth.squeeze(0).detach().to(self._torch.float32).cpu().numpy().astype(np.float32)
        depth_conf_np = depth_conf.squeeze(0).detach().to(self._torch.float32).cpu().numpy().astype(np.float32)

        postprocess_start = perf_counter()
        view_results: list[ViewResult] = []
        original_depth_maps: list[np.ndarray] = []
        original_depth_confidences: list[np.ndarray] = []
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []

        for index, image in enumerate(request.images):
            original_rgb = np.asarray(Image.open(image.path).convert("RGB"), dtype=np.uint8)

            depth_map = depth_np[index, ..., 0]
            depth_conf = depth_conf_np[index]
            depth_original = remap_square_tensor_to_original(depth_map, image.width, image.height).astype(np.float32)
            depth_conf_original = remap_square_tensor_to_original(depth_conf, image.width, image.height).astype(
                np.float32
            )
            depth_original = np.maximum(depth_original, 0.0)

            intrinsic_original = rescale_intrinsics_to_original(
                intrinsic_np[index],
                image.width,
                image.height,
                self.backend_settings.square_image_size,
            )

            cam_from_world = np.eye(4, dtype=np.float32)
            cam_from_world[:3, :4] = extrinsic_np[index]

            masked_depth = np.where(depth_conf_original >= depth_conf_threshold, depth_original, 0.0).astype(np.float32)
            world_points = unproject_depth_map_to_point_map(
                masked_depth[None, ..., None],
                extrinsic_np[index][None, ...],
                intrinsic_original[None, ...],
            )[0]

            valid_mask = (depth_conf_original >= depth_conf_threshold) & (masked_depth > 0.0)
            if np.any(valid_mask):
                all_points.append(world_points[valid_mask])
                all_colors.append(original_rgb[valid_mask])

            original_depth_maps.append(depth_original)
            original_depth_confidences.append(depth_conf_original)
            view_results.append(
                ViewResult(
                    view_id=request.views[index].view_id,
                    filename=image.original_filename,
                    original_size=ImageSize(width=image.width, height=image.height),
                    camera=CameraResult(
                        convention="opencv",
                        world_to_camera=cam_from_world.tolist(),
                        intrinsics=intrinsic_original.tolist(),
                        source="predicted",
                    ),
                )
            )

        if all_points:
            point_cloud = np.concatenate(all_points, axis=0)
            point_colors = np.concatenate(all_colors, axis=0)
            point_cloud, point_colors = sample_point_cloud(
                point_cloud,
                point_colors,
                self.backend_settings.max_point_cloud_points,
            )
        else:
            point_cloud = np.empty((0, 3), dtype=np.float32)
            point_colors = np.empty((0, 3), dtype=np.uint8)

        depth_path = request.run_dir / "depth.npz"
        write_depth_artifact(
            depth_path,
            [image.original_filename for image in request.images],
            [(image.width, image.height) for image in request.images],
            original_depth_maps,
            original_depth_confidences,
        )

        ply_path = request.run_dir / "point_cloud.ply"
        write_point_cloud_ply(ply_path, point_cloud, point_colors)

        postprocess_ms = int((perf_counter() - postprocess_start) * 1000)
        total_ms = int((perf_counter() - start) * 1000)

        artifacts = [
            ArtifactDescriptor(
                name=depth_path.name,
                path=depth_path,
                kind="depth_archive",
                content_type="application/octet-stream",
                size_bytes=depth_path.stat().st_size,
            ),
            ArtifactDescriptor(
                name=ply_path.name,
                path=ply_path,
                kind="point_cloud",
                content_type="application/octet-stream",
                size_bytes=ply_path.stat().st_size,
            ),
        ]

        LOGGER.info(
            "Reconstruction completed",
            extra={
                "request_id": request.request_id,
                "backend": self.backend_id,
                "image_count": len(request.images),
                "device": self.device_description,
                "timings_ms": {
                    "inference": inference_ms,
                    "postprocess": postprocess_ms,
                    "total": total_ms,
                },
            },
        )

        return BackendRunResult(
            view_results=view_results,
            artifacts=artifacts,
            produced_outputs=list(self.capabilities),
            timings_ms={
                "inference": inference_ms,
                "postprocess": postprocess_ms,
                "total": total_ms,
            },
        )
