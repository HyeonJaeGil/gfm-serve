"""Deprecated compatibility namespace for the former VGGT Serve package."""

from __future__ import annotations

import warnings

warnings.warn(
    "vggt_serve is deprecated; import gfm_serve instead. "
    "The compatibility namespace will be removed after 2026-12-31.",
    DeprecationWarning,
    stacklevel=2,
)

from gfm_serve import __version__

__all__ = ["__version__"]
