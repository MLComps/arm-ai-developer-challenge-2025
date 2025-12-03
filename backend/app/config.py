"""
Application Configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os


def get_project_root() -> Path:
    """Get project root from env var or default to parent of backend"""
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"])
    # Fallback: assume we're in backend/app/
    return Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Settings
    app_name: str = "Ambient Wildlife Monitoring API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths (relative to project root)
    classifier_model_path: str = "classifier-models/train-fox-background/weights/best.pt"
    video_assets_dir: str = "video-assets/normal"
    output_dir: str = "output"

    # VLM Settings
    vlm_endpoint: str = "http://localhost:11434/api/generate"
    vlm_model: str = "qwen3-vl:2b"
    vlm_num_samples: int = 5

    # CORS Settings
    cors_origins: list = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute using project root"""
        project_root = get_project_root()
        return str(project_root / relative_path)

    @property
    def absolute_classifier_model_path(self) -> str:
        return self.get_absolute_path(self.classifier_model_path)

    @property
    def absolute_video_assets_dir(self) -> str:
        return self.get_absolute_path(self.video_assets_dir)

    @property
    def absolute_output_dir(self) -> str:
        return self.get_absolute_path(self.output_dir)


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()
