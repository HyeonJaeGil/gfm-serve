from __future__ import annotations

import pytest

from gfm_backend_depth_anything_3 import DA3Backend
from gfm_backend_vggt import VGGTBackend
from gfm_serve.config import Settings
from gfm_serve.testing import assert_backend_conformance


@pytest.mark.parametrize("backend_type", [VGGTBackend, DA3Backend])
def test_backend_conformance(backend_type, tmp_path) -> None:
    backend = backend_type(Settings(data_root=tmp_path / "runs"))

    assert_backend_conformance(backend)
