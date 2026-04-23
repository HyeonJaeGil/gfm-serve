from __future__ import annotations

import numpy as np
import pytest


geometry = pytest.importorskip("vggt.utils.geometry")
unproject_depth_map_to_point_map = geometry.unproject_depth_map_to_point_map


def test_unproject_depth_map_accepts_service_shape() -> None:
    depth = np.ones((1, 4, 5, 1), dtype=np.float32)
    extrinsic = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)[None]
    intrinsic = np.eye(3, dtype=np.float32)[None]

    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    assert points.shape == (1, 4, 5, 3)
