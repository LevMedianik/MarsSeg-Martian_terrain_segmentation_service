import torch
import numpy as np
from PIL import Image
import os
from uuid import uuid4
from .settings import settings
from .loader import load_model
from .colormap import colorize_mask

os.makedirs("artifacts", exist_ok=True)

def predict_mask(tensor: torch.Tensor) -> np.ndarray:
    model, device = load_model()
    with torch.inference_mode():
        out = model(tensor.to(device))
        logits = out["out"] if isinstance(out, dict) and "out" in out else out
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return preds

def make_overlay(base_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float | None = None) -> np.ndarray:
    if alpha is None:
        alpha = settings.overlay_alpha
    overlay = alpha * mask_rgb.astype(np.float32) + (1 - alpha) * base_rgb.astype(np.float32)
    return overlay.clip(0, 255).astype(np.uint8)

def _webify(paths: list[str]) -> list[str]:
    out = []
    cwd = os.getcwd()
    for p in paths:
        if os.path.isabs(p):
            p = os.path.relpath(p, start=cwd)
        out.append(p.replace("\\", "/"))
    return out

def run_inference(tensor: torch.Tensor, base_rgb: np.ndarray, output: str = "overlay") -> list[str]:
    uid = str(uuid4())[:8]
    paths: list[str] = []

    mask = predict_mask(tensor)
    mask_rgb = colorize_mask(mask)

    if output in ("mask", "both"):
        mask_path = f"artifacts/{uid}_mask.png"
        Image.fromarray(mask_rgb).save(mask_path)
        paths.append(mask_path)

    if output in ("overlay", "both"):
        overlay = make_overlay(base_rgb, mask_rgb)
        overlay_path = f"artifacts/{uid}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        paths.append(overlay_path)

    return _webify(paths)