import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import torch
from torchvision import transforms

from .settings import settings

# --- Изображение ---

def apply_brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Коррекция яркости/контраста: new = alpha*img + beta
    img: np.uint8 [0..255]
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_clahe_rgb(img: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    CLAHE по каналу L. Если settings.replicate_cv2_rgb_bug=True,
    имитируем обучение: делаем RGB->BGR, затем применяем COLOR_RGB2LAB к BGR-массиву.
    """
    if settings.replicate_cv2_rgb_bug:
        # 1) получаем BGR из нашего RGB
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 2) намеренно используем RGB2LAB на BGR-массиве (как было при подготовке датасета)
        lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

from torchvision import transforms

def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    img = img.resize((settings.resize_to, settings.resize_to), Image.BILINEAR)
    rgb = np.array(img)

    # CLAHE + яркость/контраст
    proc = apply_clahe_rgb(rgb, settings.clahe_clip_limit, settings.clahe_tile_grid)
    proc = apply_brightness_contrast(proc, settings.alpha, settings.beta)

    # нормализация в [0,255]
    proc = cv2.normalize(proc, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    proc = proc.astype("uint8")

    # в тензор [0,1]
    tensor = transforms.ToTensor()(Image.fromarray(proc)).unsqueeze(0)

    return tensor, rgb

# --- Маска ---

def preprocess_mask(pil_mask: Image.Image) -> np.ndarray:
    """
    Подготовка маски (ground truth) из датасета для валидации/сравнения.
    Resize с сохранением классов.
    Возвращает np.uint8 (H,W), значения классов {0..C-1}.
    """
    mask = pil_mask.resize((settings.resize_to, settings.resize_to), Image.NEAREST)
    mask = np.array(mask).astype(np.uint8)
    return mask
