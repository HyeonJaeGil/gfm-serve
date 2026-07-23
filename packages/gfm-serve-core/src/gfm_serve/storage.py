from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class PreparedImage:
    original_filename: str
    stored_filename: str
    path: Path
    size_bytes: int
    width: int
    height: int
    content_type: str


@dataclass(slots=True)
class ArtifactDescriptor:
    name: str
    path: Path
    kind: str
    content_type: str
    size_bytes: int
    metadata: dict[str, object] = field(default_factory=dict)


def sanitize_filename(filename: str | None, fallback: str) -> str:
    base = Path(filename or "").name or fallback
    safe = _FILENAME_SANITIZER.sub("_", base)
    return safe or fallback


def remap_square_tensor_to_original(square_map: np.ndarray, width: int, height: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2

    tensor = torch.as_tensor(square_map, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        squeeze_mode = "hw"
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        squeeze_mode = "hwc"
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported tensor shape for remapping: {tuple(tensor.shape)}")

    resized = F.interpolate(tensor, size=(max_dim, max_dim), mode="bilinear", align_corners=False)
    cropped = resized[..., top : top + height, left : left + width]

    if squeeze_mode == "hw":
        return cropped.squeeze(0).squeeze(0).cpu().numpy()

    return cropped.squeeze(0).permute(1, 2, 0).cpu().numpy()


def rescale_intrinsics_to_original(intrinsic: np.ndarray, width: int, height: int, square_size: int) -> np.ndarray:
    scale = max(width, height) / float(square_size)
    rescaled = intrinsic.astype(np.float32).copy()
    rescaled[0, 0] *= scale
    rescaled[1, 1] *= scale
    rescaled[0, 2] = width / 2.0
    rescaled[1, 2] = height / 2.0
    rescaled[0, 1] = 0.0
    rescaled[1, 0] = 0.0
    rescaled[2, 0] = 0.0
    rescaled[2, 1] = 0.0
    rescaled[2, 2] = 1.0
    return rescaled


def sample_point_cloud(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors

    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices], colors[indices]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_depth_artifact(
    path: Path,
    filenames: list[str],
    image_sizes: list[tuple[int, int]],
    depth_maps: list[np.ndarray],
    depth_confidences: list[np.ndarray],
) -> None:
    depth_array = np.empty(len(depth_maps), dtype=object)
    conf_array = np.empty(len(depth_confidences), dtype=object)
    for index, depth_map in enumerate(depth_maps):
        depth_array[index] = depth_map.astype(np.float32)
        conf_array[index] = depth_confidences[index].astype(np.float32)

    np.savez_compressed(
        path,
        filenames=np.asarray(filenames),
        image_sizes=np.asarray(image_sizes, dtype=np.int32),
        depth=depth_array,
        depth_conf=conf_array,
    )


def write_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3)")
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length")

    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    if len(points):
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)
