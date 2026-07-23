from __future__ import annotations

import numpy as np
import pytest

from gfm_serve.storage import remap_square_tensor_to_original, rescale_intrinsics_to_original, sample_point_cloud, write_point_cloud_ply


def test_rescale_intrinsics_to_original() -> None:
    intrinsic = np.array(
        [
            [250.0, 0.0, 259.0],
            [0.0, 240.0, 259.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rescaled = rescale_intrinsics_to_original(intrinsic, width=400, height=200, square_size=518)

    assert np.isclose(rescaled[0, 0], 250.0 * (400.0 / 518.0))
    assert np.isclose(rescaled[1, 1], 240.0 * (400.0 / 518.0))
    assert np.isclose(rescaled[0, 2], 200.0)
    assert np.isclose(rescaled[1, 2], 100.0)


def test_remap_square_tensor_to_original_shape() -> None:
    pytest.importorskip("torch")

    square = np.ones((518, 518), dtype=np.float32)
    remapped = remap_square_tensor_to_original(square, width=320, height=160)

    assert remapped.shape == (160, 320)


def test_sample_point_cloud_limits_vertices() -> None:
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.full((10, 3), 127, dtype=np.uint8)

    sampled_points, sampled_colors = sample_point_cloud(points, colors, max_points=4)

    assert sampled_points.shape == (4, 3)
    assert sampled_colors.shape == (4, 3)


def test_write_point_cloud_ply(tmp_path) -> None:
    path = tmp_path / "cloud.ply"
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

    write_point_cloud_ply(path, points, colors)

    payload = path.read_bytes()
    assert payload.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 2\n" in payload
