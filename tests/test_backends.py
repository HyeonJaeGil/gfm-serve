from __future__ import annotations

import pytest
from pydantic import ValidationError

from gfm_backend_vggt import VGGTBackend
from gfm_backend_vggt.config import VGGTBackendSettings
from gfm_serve.backends import create_backend, list_backends
from gfm_serve.backends import registry
from gfm_serve.config import Settings


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
    backend = VGGTBackend(settings)

    assert backend.backend_settings.default_depth_conf_threshold == 1.0


def test_default_max_images_is_32(tmp_path) -> None:
    settings = Settings(data_root=tmp_path / "runs")

    assert settings.max_images == 32


def test_gfm_environment_names_take_precedence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GFM_SERVE_DATA_ROOT", str(tmp_path / "gfm"))
    monkeypatch.setenv("RECON_SERVE_DATA_ROOT", str(tmp_path / "recon"))
    monkeypatch.setenv("GFM_SERVE_VGGT_MODEL_ID", "new/model")
    monkeypatch.setenv("RECON_SERVE_VGGT_MODEL_ID", "old/model")

    assert Settings().data_root == tmp_path / "gfm"
    assert VGGTBackendSettings().model_id == "new/model"


def test_legacy_environment_names_remain_supported(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RECON_SERVE_DATA_ROOT", str(tmp_path / "legacy"))
    monkeypatch.setenv("RECON_SERVE_VGGT_MODEL_ID", "legacy/model")

    assert Settings().data_root == tmp_path / "legacy"
    assert VGGTBackendSettings().model_id == "legacy/model"


def test_backend_selection_is_required_when_multiple_are_installed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        registry,
        "_discover_backends",
        lambda: {"one": lambda settings: object(), "two": lambda settings: object()},
    )

    with pytest.raises(ValueError, match="Exactly one backend"):
        create_backend(Settings(data_root=tmp_path / "runs", backend=None))
