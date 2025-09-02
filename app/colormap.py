import numpy as np

# Легенда классов
CLASS_LABELS = ["soil", "bedrock", "sand", "big_rock"]

# Цвета RGB для каждой метки (uint8)
CLASS_COLORS = np.array([
    [128, 0, 0],     # soil     (красный)
    [0, 128, 0],     # bedrock  (зелёный)
    [128, 128, 0],   # sand     (желтый)
    [128, 0, 128],   # big_rock (фиолетовый)
], dtype=np.uint8)

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Превращает маску с индексами классов [H,W] -> цветное изображение [H,W,3]
    """
    h, w = mask.shape
    vis = CLASS_COLORS[mask.reshape(-1)].reshape(h, w, 3)
    return vis