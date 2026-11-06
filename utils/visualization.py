"""
utils/visualization.py
======================
Colorear mapas semánticos/panópticos y crear superposiciones. Lo usan el
notebook, el informe y la app web (mismos colores en todas partes).
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

import config as C

PALETTE = np.array([C.CLASS_COLORS[i] for i in range(C.NUM_CLASSES)], np.uint8)


def colorize_semantic(sem: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(sem, 0, C.NUM_CLASSES - 1)]


def colorize_panoptic(pan: np.ndarray, segments: list[dict], borders=True) -> np.ndarray:
    """Cada instancia "thing" recibe un tono distinto alrededor del color de su clase;
    los "stuff" conservan el color de clase. Bordes blancos entre segmentos."""
    out = np.zeros(pan.shape + (3,), np.uint8)
    rng = np.random.default_rng(7)
    for s in segments:
        base = np.array(C.CLASS_COLORS[s["category_id"]], np.float32) / 255
        if s.get("isthing"):
            h, l, sat = colorsys.rgb_to_hls(*base)
            h = (h + rng.uniform(-0.07, 0.07)) % 1
            l = np.clip(l + rng.uniform(-0.22, 0.18), 0.2, 0.85)
            col = np.array(colorsys.hls_to_rgb(h, l, sat)) * 255
        else:
            col = base * 255
        out[pan == s["id"]] = col.astype(np.uint8)
    if borders:
        edge = cv2.morphologyEx(pan.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3))) > 0
        out[edge] = (255, 255, 255)
    return out


def overlay(img: np.ndarray, color: np.ndarray, alpha=0.5, mask: np.ndarray | None = None):
    out = img.astype(np.float32).copy()
    m = np.ones(img.shape[:2], bool) if mask is None else mask
    out[m] = (1 - alpha) * out[m] + alpha * color.astype(np.float32)[m]
    return out.astype(np.uint8)


def class_mask_rgba(sem: np.ndarray, cls: int) -> np.ndarray:
    """Máscara RGBA de una sola clase (transparente fuera) para capas en la web."""
    rgba = np.zeros(sem.shape + (4,), np.uint8)
    m = sem == cls
    rgba[m, :3] = C.CLASS_COLORS[cls]
    rgba[m, 3] = 255
    return rgba


def disagreement_map(sem_a: np.ndarray, sem_b: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Resalta en magenta los píxeles donde los dos modelos no coinciden."""
    diff = sem_a != sem_b
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    gray = (gray * 0.45).astype(np.uint8)
    gray[diff] = (255, 40, 200)
    return gray


def error_map(gt: np.ndarray, pred: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Verde = acierto, rojo = error (respecto al ground truth)."""
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    out = (gray * 0.35).astype(np.float32)
    ok = gt == pred
    out[ok] = 0.5 * out[ok] + 0.5 * np.array((40, 200, 120))
    out[~ok] = 0.3 * out[~ok] + 0.7 * np.array((255, 60, 60))
    return out.astype(np.uint8)


def confidence_map(probs: np.ndarray) -> np.ndarray:
    """Mapa de confianza (prob. máxima) con colormap 'inferno'."""
    conf = (probs.astype(np.float32).max(0) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(conf, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
