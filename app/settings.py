from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Tuple

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
        env_prefix="MARSSEG_",
    )

    # --- модель ---
    model_name: str = "deeplabv3"
    model_weights: str = "models/deeplabv3.pth"

    # --- классы ---
    num_classes: int = 4
    class_labels: list[str] = ["soil", "bedrock", "sand", "big_rock"]

    # --- препроцессинг ---
    resize_to: int = 256
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    alpha: float = 1.5
    beta: float = 50

    # --- нормализация ---
    use_mean_std: bool = False

    # --- визуализация ---
    overlay_alpha: float = 0.5
    image_size: Tuple[int, int] = (256, 256)
    replicate_cv2_rgb_bug: bool = True

settings = Settings()