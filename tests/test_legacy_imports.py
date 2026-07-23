from __future__ import annotations

import importlib
import sys

import pytest


def test_legacy_package_import_warns_and_forwards_version() -> None:
    sys.modules.pop("vggt_serve", None)

    with pytest.warns(DeprecationWarning, match="removed after 2026-12-31"):
        legacy = importlib.import_module("vggt_serve")

    assert legacy.__version__ == "0.2.0"


def test_legacy_config_import_forwards_common_settings() -> None:
    from gfm_serve.config import Settings as CurrentSettings
    from vggt_serve.config import Settings as LegacySettings

    assert LegacySettings is CurrentSettings
