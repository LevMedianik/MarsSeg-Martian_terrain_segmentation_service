# app/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Tuple, List, Optional
import json

class Settings(BaseSettings):
    # Pydantic v2-конфиг (заменяет class Config)
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),  # снимает конфликт с полями model_*
        env_prefix="MARSSEG_",               # префикс переменных окружения
    )

    # --- модель ---
    model_name: str = "deeplabv3"
    model_weights: str = "models/deeplabv3.pth"
    weights_url: Optional[str] = None  # URL для автоскачивания весов

    # --- классы ---
    num_classes: int = 4
    class_labels: List[str] = ["soil", "bedrock", "sand", "big_rock"]

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
    replicate_cv2_rgb_bug: bool = True  # репликация шага RGB2LAB на BGR

    # --- валидация/приведение типов ---
    @field_validator("class_labels", mode="before")
    @classmethod
    def _parse_labels(cls, v):
        """
        Поддерживает:
          - JSON-строку из .env: '["soil","bedrock","sand","big_rock"]'
          - CSV-строку из .env:  'soil,bedrock,sand,big_rock'
          - уже список
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            # пробуем JSON
            if (s.startswith("[") and s.endswith("]")) or (s.startswith('"') and s.endswith('"')):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            # иначе — CSV
            return [x.strip() for x in s.split(",") if x.strip()]
        # дефолт
        return ["soil", "bedrock", "sand", "big_rock"]

settings = Settings()
