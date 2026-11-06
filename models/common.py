"""
models/common.py
================
Piezas compartidas por los dos modelos: preparación de la imagen para
inferencia, test-time augmentation (TTA) y la clase base `BasePredictor`.

Dos modos de inferencia panóptica:
  - "native": la cabeza panóptica propia de cada arquitectura
              (consultas de Mask2Former / fusión panóptica de EfficientPS).
  - "tta":    se promedian las probabilidades semánticas sobre las 8
              simetrías D4 de la imagen y las instancias se obtienen por
              componentes conexas (la misma definición del GT). Más lento
              (8 pasadas) pero normalmente más preciso.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config as C
from data.dataset import normalize, connected_instances
from models.postprocess import finalize


def prepare(img: np.ndarray, max_side: int, stride: int = 32):
    """Redimensiona (si excede max_side) y rellena a múltiplo de `stride`.
    Devuelve (tensor 1x3xHpxWp, (h, w) tamaño útil, (H0, W0) tamaño original)."""
    H0, W0 = img.shape[:2]
    f = min(1.0, max_side / max(H0, W0))
    h, w = int(round(H0 * f)), int(round(W0 * f))
    if (h, w) != (H0, W0):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    Hp, Wp = int(np.ceil(h / stride) * stride), int(np.ceil(w / stride) * stride)
    x = normalize(img)
    x = np.pad(x, ((0, 0), (0, Hp - h), (0, Wp - w)))
    return torch.from_numpy(x)[None], (h, w), (H0, W0)


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def d4_inv(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, -(k % 4), dims=(-2, -1))
    if k >= 4:
        x = x.flip(-1)
    return x


def resize_probs(p: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(p[None], size=size, mode="bilinear", align_corners=False)[0]


def smooth_probs(probs: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizado gaussiano de las probabilidades: elimina fragmentos y bordes ruidosos."""
    if not sigma:
        return probs
    p = np.stack([cv2.GaussianBlur(c.astype(np.float32), (0, 0), sigma) for c in probs])
    return p / np.maximum(p.sum(0, keepdims=True), 1e-6)


def class_bias(probs: np.ndarray, bias: dict | None) -> np.ndarray:
    """Multiplica la probabilidad de algunas clases (p. ej. Área Caminable, delgada y
    sub-detectada) y renormaliza. El factor se elige en validación."""
    if not bias:
        return probs
    p = probs.astype(np.float32).copy()
    for c, f in bias.items():
        p[int(c)] *= float(f)
    return p / np.maximum(p.sum(0, keepdims=True), 1e-6)


def panoptic_from_semantic(probs: np.ndarray, sigma: float = 0, bias: dict | None = None,
                           **kw) -> dict:
    """Panóptico "semántico primero": stuff = argmax; things = componentes conexas."""
    probs = class_bias(smooth_probs(probs, sigma), bias)
    return finalize(np.zeros(probs.shape[1:], np.int32), [], probs, **kw)


POST_FILE = C.OUTPUTS_DIR / "postprocess.json"
POST_DEFAULT = {"sigma": 0, "region_min_area": C.MIN_INSTANCE_AREA, "score_mode": "mean",
                "split_erode": 0, "bias": {}, "scales": [1.0]}


def load_post(model_name: str) -> dict:
    """Parámetros de post-procesamiento ajustados en VALIDACIÓN (tune_postprocess.py)."""
    out = {"native": dict(POST_DEFAULT), "tta": dict(POST_DEFAULT)}
    if POST_FILE.exists():
        try:
            d = C.read_json(POST_FILE).get(model_name, {})
            for m in out:
                out[m].update({k: v for k, v in d.get(m, {}).items() if k in POST_DEFAULT})
        except Exception:
            pass
    return out


class BasePredictor:
    """Interfaz común. Las subclases implementan `_sem_probs` y `_native`."""
    name = "base"

    def __init__(self, model, device, infer_size=1024):
        self.model = model.to(device).eval()
        self.device = device
        self.infer_size = infer_size
        self.post = load_post(self.name)

    def region_kw(self, mode="native") -> dict:
        p = self.post[mode]
        return {"region_min_area": p["region_min_area"], "score_mode": p["score_mode"],
                "split_erode": p["split_erode"]}

    def _d4_pass(self, x):
        acc = None
        for k in range(8):
            p = d4_inv(self._sem_probs(d4(x, k)), k)
            acc = p if acc is None else acc + p
        return acc

    @torch.no_grad()
    def tta_probs(self, img: np.ndarray, scales=(1.0,)) -> np.ndarray:
        """Probabilidades promediadas sobre 8 simetrías D4 y una o varias escalas."""
        H0, W0 = img.shape[:2]
        total = None
        for s in scales:
            im = img if s == 1.0 else cv2.resize(img, (int(W0 * s), int(H0 * s)),
                                                 interpolation=cv2.INTER_CUBIC)
            x, hw, _ = prepare(im, int(self.infer_size * s))
            try:
                acc = self._d4_pass(x.to(self.device))
            except torch.cuda.OutOfMemoryError:
                # GPU con poca memoria (p. ej. portátil): esta escala se calcula en CPU
                torch.cuda.empty_cache()
                self.model.to("cpu")
                try:
                    acc = self._d4_pass(x)
                finally:
                    self.model.to(self.device)
            p = resize_probs((acc / 8)[:, :hw[0], :hw[1]], (H0, W0)).float().to(self.device)
            total = p if total is None else total + p
        return (total / len(scales)).cpu().numpy()

    # --- a implementar ---
    def _sem_probs(self, x: torch.Tensor) -> torch.Tensor:
        """x: 1x3xHxW normalizado -> probabilidades (C, H, W) en ese tamaño."""
        raise NotImplementedError

    def _native(self, x, hw, orig) -> dict:
        raise NotImplementedError

    # --- API pública ---
    @torch.no_grad()
    def predict(self, img: np.ndarray, mode: str = "native") -> dict:
        t0 = time.perf_counter()
        x, hw, orig = prepare(img, self.infer_size)
        x = x.to(self.device)
        if mode == "tta":
            pt = self.post["tta"]
            probs = self.tta_probs(img, tuple(pt["scales"]))
            out = panoptic_from_semantic(probs, pt["sigma"], pt["bias"], **self.region_kw("tta"))
        else:
            out = self._native(x, hw, orig)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        out["time_ms"] = (time.perf_counter() - t0) * 1000
        out["model"] = self.name
        out["mode"] = mode
        return out
