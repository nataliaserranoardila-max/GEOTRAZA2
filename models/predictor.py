"""
models/predictor.py
===================
Carga un archivo de pesos .pt (de cualquiera de los dos modelos) y devuelve
un objeto Predictor con el método `predict(imagen_rgb, mode)`.
"""
from __future__ import annotations

from pathlib import Path

import torch


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_predictor(path: str | Path, device: torch.device | None = None, infer_size=None):
    device = device or get_device()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    if arch == "mask2former":
        from models.mask2former_model import load_mask2former
        pred = load_mask2former(ckpt, device, infer_size)
    elif arch == "efficientps":
        from models.efficientps import load_efficientps
        pred = load_efficientps(ckpt, device, infer_size)
    else:
        raise ValueError(f"Arquitectura desconocida en {path}: {arch}")
    pred.meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "config")}
    return pred
