from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = Field(
        default="gfm-serve",
        validation_alias=AliasChoices(
            "GFM_SERVE_SERVICE_NAME", "RECON_SERVE_SERVICE_NAME", "VGGT_SERVE_SERVICE_NAME"
        ),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("GFM_SERVE_HOST", "RECON_SERVE_HOST", "VGGT_SERVE_HOST"),
    )
    port: int = Field(
        default=8000,
        validation_alias=AliasChoices("GFM_SERVE_PORT", "RECON_SERVE_PORT", "VGGT_SERVE_PORT"),
    )
    data_root: Path = Field(
        default_factory=lambda: Path("data/runs"),
        validation_alias=AliasChoices("GFM_SERVE_DATA_ROOT", "RECON_SERVE_DATA_ROOT", "VGGT_SERVE_DATA_ROOT"),
    )
    backend: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GFM_SERVE_BACKEND", "RECON_SERVE_BACKEND", "VGGT_SERVE_BACKEND"),
    )
    max_images: int = Field(
        default=32,
        validation_alias=AliasChoices("GFM_SERVE_MAX_IMAGES", "RECON_SERVE_MAX_IMAGES", "VGGT_SERVE_MAX_IMAGES"),
    )
    max_upload_bytes_per_file: int = Field(
        default=25 * 1024 * 1024,
        validation_alias=AliasChoices(
            "RECON_SERVE_MAX_UPLOAD_BYTES_PER_FILE",
            "GFM_SERVE_MAX_UPLOAD_BYTES_PER_FILE",
            "VGGT_SERVE_MAX_UPLOAD_BYTES_PER_FILE",
        ),
    )
    max_upload_bytes_total: int = Field(
        default=250 * 1024 * 1024,
        validation_alias=AliasChoices(
            "GFM_SERVE_MAX_UPLOAD_BYTES_TOTAL",
            "RECON_SERVE_MAX_UPLOAD_BYTES_TOTAL",
            "VGGT_SERVE_MAX_UPLOAD_BYTES_TOTAL",
        ),
    )

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
