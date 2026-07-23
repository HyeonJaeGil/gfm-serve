from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VGGTBackendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True, frozen=True)

    model_id: str = Field(
        default="facebook/VGGT-1B",
        validation_alias="GFM_SERVE_VGGT_MODEL_ID",
    )
    model_revision: str = Field(
        default="main",
        validation_alias="GFM_SERVE_VGGT_MODEL_REVISION",
    )
    checkpoint_path: Path | None = Field(
        default=None,
        validation_alias="GFM_SERVE_VGGT_CHECKPOINT_PATH",
    )
    square_image_size: int = Field(
        default=518,
        validation_alias="GFM_SERVE_VGGT_SQUARE_IMAGE_SIZE",
    )
    default_depth_conf_threshold: float = Field(
        default=1.0,
        validation_alias="GFM_SERVE_VGGT_DEFAULT_DEPTH_CONF_THRESHOLD",
    )
    max_point_cloud_points: int = Field(
        default=500_000,
        validation_alias="GFM_SERVE_VGGT_MAX_POINT_CLOUD_POINTS",
    )
