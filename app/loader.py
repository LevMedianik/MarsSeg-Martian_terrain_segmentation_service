import os
import torch
from .settings import settings
from .model_def import create_deeplabv3

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global _model
    if _model is not None:
        return _model, _device

    path = settings.model_weights
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")

    model = create_deeplabv3(num_classes=settings.num_classes, pretrained=True, dropout=0.2)
    sd = torch.load(path, map_location=_device)
    if isinstance(sd, torch.nn.Module):
        sd = sd.state_dict()

    # строгая загрузка — должны совпасть все ключи и формы
    missing, unexpected = model.load_state_dict(sd, strict=False)  # покажем отчёт
    print("[LOAD] missing:", missing)
    print("[LOAD] unexpected:", unexpected)
    if missing or unexpected:
        # попробуем strict=True, чтобы явно увидеть, где ломается
        model.load_state_dict(sd, strict=True)

    model.to(_device).eval()
    _model = model
    return _model, _device
