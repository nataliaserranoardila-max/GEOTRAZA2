"""
data/dataset.py
===============
Carga, preprocesamiento y aumento de datos del dataset de Kaggle
"Reto 1 - Cartografía Automatizada".

Estructura esperada (después de descomprimir el zip en data/raw/):

    data/raw/Dataset segmentacion/recorte_XX.png
    data/raw/Dataset segmentacion/ground_truth_recorte_XX.npy   (H x W x 4)
    data/raw/Recortes adicionales/Recorte 1.XX.png               (sin etiqueta)

Pipeline de preprocesamiento
----------------------------
1. La imagen se lee en RGB (se descarta canal alfa si existe).
2. El GT (H x W x 4, multietiqueta) se binariza (> 0).
3. Los píxeles con más de una etiqueta se resuelven con `OVERLAP_PRIORITY`
   para obtener un mapa semántico de una sola etiqueta por píxel
   (0 = fondo, 1..4 = clases).
4. Para la parte panóptica se generan instancias:
     - "things" (Área Construida, Zona Verde): una instancia por cada
       componente conexa (manzanas / parques separados).
     - "stuff"  (Área Caminable, Carretera, Fondo): un único segmento por clase.
5. Aumento de datos en entrenamiento: escala aleatoria, recorte aleatorio,
   las 8 simetrías del cuadrado (D4: rotaciones de 90° + espejos, válidas
   porque una vista cenital no tiene "arriba"), y jitter de color/brillo.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

import config as C

# --------------------------------------------------------------------------
# Lectura de archivos
# --------------------------------------------------------------------------

def list_samples(seg_dir: Path = C.SEG_DIR) -> list[dict]:
    """Devuelve [{id, image, gt}] para cada imagen que tiene ground truth."""
    seg_dir = Path(seg_dir)
    samples = []
    for img_path in sorted(seg_dir.glob("recorte_*.png")):
        m = re.search(r"recorte_(\d+)", img_path.stem)
        if not m:
            continue
        sid = int(m.group(1))
        gt_path = seg_dir / f"ground_truth_recorte_{sid}.npy"
        if gt_path.exists():
            samples.append({"id": sid, "image": str(img_path), "gt": str(gt_path)})
    samples.sort(key=lambda s: s["id"])
    return samples


def list_extra_images(extra_dir: Path = C.EXTRA_DIR) -> list[str]:
    """Imágenes adicionales sin etiqueta (útiles para la demo de la app)."""
    extra_dir = Path(extra_dir)
    if not extra_dir.exists():
        return []
    return [str(p) for p in sorted(extra_dir.glob("*.png"))]


def load_image(path) -> np.ndarray:
    """Lee una imagen como RGB uint8 (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:  # PNG de 16 bits
        img = (img / img.max() * 255).astype(np.uint8)
    return img


def load_gt(path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Lee el GT multietiqueta (H, W, 4) como booleano. Ajusta tamaño si hace falta."""
    gt = np.load(path if hasattr(path, "read") else str(path))
    if gt.ndim == 2:  # por si viene ya como mapa de índices
        gt = np.stack([gt == k for k in range(1, 5)], axis=-1)
    if gt.shape[0] == 4 and gt.shape[-1] != 4:  # (4, H, W) -> (H, W, 4)
        gt = np.transpose(gt, (1, 2, 0))
    gt = gt > 0
    if shape is not None and gt.shape[:2] != tuple(shape):
        gt = np.stack(
            [cv2.resize(gt[..., k].astype(np.uint8), (shape[1], shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0 for k in range(4)], -1)
    return gt


# --------------------------------------------------------------------------
# Conversión de etiquetas
# --------------------------------------------------------------------------

def gt_to_semantic(gt: np.ndarray, priority=C.OVERLAP_PRIORITY) -> np.ndarray:
    """(H, W, 4) bool -> (H, W) uint8 con 0 = fondo y 1..4 = clases.

    Se pinta de menor a mayor prioridad para que la de mayor prioridad
    quede encima en los píxeles con solape."""
    sem = np.zeros(gt.shape[:2], np.uint8)
    for cls in reversed(priority):
        sem[gt[..., cls - 1]] = cls
    return sem


def overlap_stats(gt: np.ndarray) -> dict:
    """Porcentaje de píxeles con 0, 1, 2+ etiquetas (para el informe)."""
    n = gt.sum(-1)
    tot = n.size
    return {"sin_etiqueta": float((n == 0).sum() / tot),
            "una_etiqueta": float((n == 1).sum() / tot),
            "solapadas": float((n >= 2).sum() / tot)}


def connected_instances(binary: np.ndarray, min_area: int = C.MIN_INSTANCE_AREA):
    """Componentes conexas (8-vecindad) de una máscara binaria.
    Devuelve (mapa de etiquetas int32, lista de ids válidos)."""
    n, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    ids = []
    if n > 1:
        areas = np.bincount(lab.ravel(), minlength=n)
        ids = [i for i in range(1, n) if areas[i] >= min_area]
    return lab.astype(np.int32), ids


def semantic_to_panoptic(sem: np.ndarray, min_area: int = C.MIN_INSTANCE_AREA):
    """Mapa semántico -> (mapa panóptico int32, segments).

    segments: lista de dicts {id, category_id, isthing, area}. El id 0 se
    reserva para "vacío". Things: una instancia por componente conexa.
    Stuff: un segmento por clase."""
    pan = np.zeros(sem.shape, np.int32)
    segments = []
    next_id = 1
    for cls in C.EVAL_CLASSES + [0]:
        mask = sem == cls
        if not mask.any():
            continue
        if cls in C.THING_CLASSES:
            lab, ids = connected_instances(mask, min_area)
            for i in ids:
                m = lab == i
                pan[m] = next_id
                segments.append({"id": next_id, "category_id": cls, "isthing": True,
                                 "area": int(m.sum())})
                next_id += 1
        else:
            pan[mask] = next_id
            segments.append({"id": next_id, "category_id": cls, "isthing": False,
                             "area": int(mask.sum())})
            next_id += 1
    return pan, segments


def semantic_to_regions(sem: np.ndarray, classes=C.EVAL_CLASSES,
                        min_area: int = C.MIN_INSTANCE_AREA):
    """Regiones conexas de TODAS las clases evaluadas (para mAP por regiones).
    Devuelve (mapa int32, [(id, clase)])."""
    reg = np.zeros(sem.shape, np.int32)
    info = []
    nid = 1
    for cls in classes:
        lab, ids = connected_instances(sem == cls, min_area)
        for i in ids:
            reg[lab == i] = nid
            info.append((nid, cls))
            nid += 1
    return reg, info


# --------------------------------------------------------------------------
# Splits reproducibles
# --------------------------------------------------------------------------

def make_splits(samples: list[dict], ratios=C.SPLIT_RATIOS, seed=C.SEED,
                out_file: Path = C.SPLIT_FILE) -> dict:
    ids = [s["id"] for s in samples]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    splits = {"train": sorted(ids[:n_tr]),
              "val": sorted(ids[n_tr:n_tr + n_va]),
              "test": sorted(ids[n_tr + n_va:]),
              "seed": seed}
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def load_splits(samples: list[dict] | None = None, split_file: Path = C.SPLIT_FILE) -> dict:
    split_file = Path(split_file)
    if split_file.exists():
        return C.read_json(split_file)
    if samples is None:
        samples = list_samples()
    return make_splits(samples, out_file=split_file)


# --------------------------------------------------------------------------
# Aumento de datos
# --------------------------------------------------------------------------

def d4_transform(img: np.ndarray, sem: np.ndarray, k: int):
    """Aplica una de las 8 simetrías del cuadrado (k en 0..7)."""
    if k >= 4:
        img, sem = img[:, ::-1], sem[:, ::-1]
    r = k % 4
    if r:
        img, sem = np.rot90(img, r), np.rot90(sem, r)
    return np.ascontiguousarray(img), np.ascontiguousarray(sem)


def color_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brillo/contraste/saturación/tono suaves (cambios de iluminación y sensor)."""
    x = img.astype(np.float32)
    alpha = rng.uniform(0.8, 1.2)        # contraste
    beta = rng.uniform(-20, 20)          # brillo
    x = x * alpha + beta
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.8, 1.2)                      # saturación
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180    # tono
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.2:  # leve desenfoque (simula menor resolución)
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def random_scale_crop(img, sem, crop, rng, scale_range=(0.75, 1.5)):
    """Reescala aleatoriamente y toma un recorte de crop x crop (con padding si falta)."""
    s = rng.uniform(*scale_range)
    h, w = sem.shape
    nh, nw = max(int(h * s), 8), max(int(w * s), 8)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    sem = cv2.resize(sem, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = max(crop - nh, 0), max(crop - nw, 0)
    if ph or pw:  # padding: imagen en negro, etiqueta 255 (= ignorar)
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        sem = cv2.copyMakeBorder(sem, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
    h, w = sem.shape
    y = rng.integers(0, h - crop + 1)
    x = rng.integers(0, w - crop + 1)
    return img[y:y + crop, x:x + crop], sem[y:y + crop, x:x + crop]


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 RGB (H,W,3) -> float32 (3,H,W) normalizado con media/desv. de ImageNet."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(C.IMAGENET_MEAN, np.float32)) / np.array(C.IMAGENET_STD, np.float32)
    return x.transpose(2, 0, 1).copy()


# --------------------------------------------------------------------------
# Dataset de PyTorch
# --------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset

    class CartografiaDataset(Dataset):
        """Dataset genérico: devuelve (imagen normalizada, mapa semántico, id).

        - train=True : recortes aleatorios + aumento de datos; cada imagen
          aparece `crops_per_image` veces por época.
        - train=False: imagen completa (sin aumento), para validación/test.
        Las imágenes se cargan una sola vez en memoria (el dataset es pequeño).
        """

        def __init__(self, ids, seg_dir=C.SEG_DIR, train=True, crop_size=512,
                     crops_per_image=4, max_side=None, seed=C.SEED):
            all_samples = {s["id"]: s for s in list_samples(seg_dir)}
            self.ids = [i for i in ids if i in all_samples]
            self.train = train
            self.crop = crop_size
            self.cpi = crops_per_image if train else 1
            self.rng = np.random.default_rng(seed)
            self.cache = {}
            for i in self.ids:
                s = all_samples[i]
                img = load_image(s["image"])
                sem = gt_to_semantic(load_gt(s["gt"], img.shape[:2]))
                if max_side and max(img.shape[:2]) > max_side:  # limitar tamaño en val/test
                    f = max_side / max(img.shape[:2])
                    size = (int(img.shape[1] * f), int(img.shape[0] * f))
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    sem = cv2.resize(sem, size, interpolation=cv2.INTER_NEAREST)
                self.cache[i] = (img, sem)

        def __len__(self):
            return len(self.ids) * self.cpi

        def __getitem__(self, idx):
            sid = self.ids[idx % len(self.ids)]
            img, sem = self.cache[sid]
            if self.train:
                rng = np.random.default_rng(self.rng.integers(1 << 31) + idx)
                img, sem = random_scale_crop(img, sem, self.crop, rng)
                img, sem = d4_transform(img, sem, int(rng.integers(8)))
                img = color_jitter(img, rng)
            return torch.from_numpy(normalize(img)), torch.from_numpy(sem.astype(np.int64)), sid

except ImportError:  # permite usar las funciones de numpy sin torch instalado
    pass
