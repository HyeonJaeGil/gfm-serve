from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VGGTBackendSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    checkpoint_path: Path | None
    square_image_size: int
    default_depth_conf_threshold: float
    max_point_cloud_points: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = Field(
        default="vggt-serve",
        validation_alias=AliasChoices("RECON_SERVE_SERVICE_NAME", "VGGT_SERVE_SERVICE_NAME"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("RECON_SERVE_HOST", "VGGT_SERVE_HOST"),
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("RECON_SERVE_PORT", "VGGT_SERVE_PORT"),
    )
    data_root: Path = Field(
        default_factory=lambda: Path("data/runs"),
        validation_alias=AliasChoices("RECON_SERVE_DATA_ROOT", "VGGT_SERVE_DATA_ROOT"),
    )
    backend: str = Field(
        default="vggt",
        validation_alias=AliasChoices("RECON_SERVE_BACKEND", "VGGT_SERVE_BACKEND"),
    )
    max_images: int = Field(
        default=16,
        validation_alias=AliasChoices("RECON_SERVE_MAX_IMAGES", "VGGT_SERVE_MAX_IMAGES"),
    )
    max_upload_bytes_per_file: int = Field(
        default=25 * 1024 * 1024,
        validation_alias=AliasChoices(
            "RECON_SERVE_MAX_UPLOAD_BYTES_PER_FILE",
            "VGGT_SERVE_MAX_UPLOAD_BYTES_PER_FILE",
        ),
    )
    max_upload_bytes_total: int = Field(
        default=250 * 1024 * 1024,
        validation_alias=AliasChoices("RECON_SERVE_MAX_UPLOAD_BYTES_TOTAL", "VGGT_SERVE_MAX_UPLOAD_BYTES_TOTAL"),
    )

    vggt_model_id: str = Field(
        default="facebook/VGGT-1B",
        validation_alias=AliasChoices("RECON_SERVE_VGGT_MODEL_ID", "VGGT_SERVE_MODEL_ID"),
    )
    vggt_checkpoint_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("RECON_SERVE_VGGT_CHECKPOINT_PATH", "VGGT_SERVE_CHECKPOINT_PATH"),
    )
    vggt_square_image_size: int = Field(
        default=518,
        validation_alias=AliasChoices("RECON_SERVE_VGGT_SQUARE_IMAGE_SIZE", "VGGT_SERVE_SQUARE_IMAGE_SIZE"),
    )
    vggt_default_depth_conf_threshold: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "RECON_SERVE_VGGT_DEFAULT_DEPTH_CONF_THRESHOLD",
            "VGGT_SERVE_DEFAULT_DEPTH_CONF_THRESHOLD",
        ),
    )
    vggt_max_point_cloud_points: int = Field(
        default=500_000,
        validation_alias=AliasChoices(
            "RECON_SERVE_VGGT_MAX_POINT_CLOUD_POINTS",
            "VGGT_SERVE_MAX_POINT_CLOUD_POINTS",
        ),
    )

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)

    @property
    def vggt_backend_settings(self) -> VGGTBackendSettings:
        return VGGTBackendSettings(
            model_id=self.vggt_model_id,
            checkpoint_path=self.vggt_checkpoint_path,
            square_image_size=self.vggt_square_image_size,
            default_depth_conf_threshold=self.vggt_default_depth_conf_threshold,
            max_point_cloud_points=self.vggt_max_point_cloud_points,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
