from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DA3BackendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True, frozen=True)

    model_id: str = Field(
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        validation_alias="GFM_SERVE_DEPTH_ANYTHING_3_MODEL_ID",
    )
    model_revision: str = Field(
        default="main",
        validation_alias="GFM_SERVE_DEPTH_ANYTHING_3_MODEL_REVISION",
    )
    device: str = Field(
        default="auto",
        validation_alias="GFM_SERVE_DEPTH_ANYTHING_3_DEVICE",
    )
    max_point_cloud_points: int = Field(
        default=500_000,
        validation_alias="GFM_SERVE_DEPTH_ANYTHING_3_MAX_POINT_CLOUD_POINTS",
    )

    @property
    def supports_pose_input(self) -> bool:
        model_name = self.model_id.lower()
        return "mono" not in model_name and "metric" not in model_name

    @property
    def supports_gaussians(self) -> bool:
        model_name = self.model_id.lower()
        return "da3-giant" in model_name or "nested-giant-large" in model_name
