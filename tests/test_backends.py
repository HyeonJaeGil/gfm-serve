from __future__ import annotations

import pytest
from pydantic import ValidationError

from vggt_serve.backends import VGGTBackend, create_backend, list_backends
from vggt_serve.config import Settings


def test_create_backend_returns_vggt_backend(tmp_path) -> None:
    settings = Settings(data_root=tmp_path / "runs", backend="vggt")

    backend = create_backend(settings)

    assert isinstance(backend, VGGTBackend)
    assert backend.backend_id == "vggt"


def test_create_backend_rejects_unknown_backend(tmp_path) -> None:
    settings = Settings(data_root=tmp_path / "runs", backend="map-anything")

    with pytest.raises(ValueError, match="not implemented"):
        create_backend(settings)


def test_vggt_backend_options_reject_unknown_keys(tmp_path) -> None:
    backend = VGGTBackend(Settings(data_root=tmp_path / "runs"))

    with pytest.raises(ValidationError):
        backend.validate_options({"unexpected": 1})


def test_list_backends_includes_vggt() -> None:
    assert "vggt" in list_backends()


def test_vggt_default_depth_conf_threshold_is_low_enough_for_sparse_sequences(tmp_path) -> None:
    settings = Settings(data_root=tmp_path / "runs")

    assert settings.vggt_default_depth_conf_threshold == 1.0
