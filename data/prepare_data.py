"""
data/prepare_data.py
====================
Preparación de datos (ejecutar una vez, después de descargar el dataset):

  1. Verifica que cada imagen tenga su ground truth.
  2. Crea la división train / val / test reproducible (data/splits.json).
  3. Calcula estadísticas para el informe: tamaños, distribución de clases,
     píxeles solapados y número de instancias por clase.
  4. Guarda figuras de ejemplo (outputs/ejemplo_gt_*.png).

Uso:
    python data/prepare_data.py
    python data/prepare_data.py --download     # baja el zip de Kaggle (requiere kaggle.json)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config as C
from data.dataset import (gt_to_semantic, list_extra_images, list_samples, load_gt,
                          load_image, make_splits, overlap_stats, semantic_to_panoptic)


def download():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", C.KAGGLE_DATASET, "-p", str(C.RAW_DIR)], check=True)
    z = next(C.RAW_DIR.glob("*.zip"))
    with zipfile.ZipFile(z) as f:
        f.extractall(C.RAW_DIR)
    z.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--figures", type=int, default=3)
    args = ap.parse_args()
    if args.download:
        download()

    samples = list_samples()
    if not samples:
        sys.exit(f"No hay datos en {C.SEG_DIR}. Descomprime el zip de Kaggle en data/raw/.")
    print(f"Imágenes con GT: {len(samples)} | adicionales sin GT: {len(list_extra_images())}")
    splits = make_splits(samples)
    print({k: len(v) for k, v in splits.items() if k != "seed"})

    sizes, cls_px, ov, inst = [], np.zeros(C.NUM_CLASSES), [], {c: [] for c in C.THING_CLASSES}
    for s in samples:
        img = load_image(s["image"])
        gt = load_gt(s["gt"], img.shape[:2])
        sem = gt_to_semantic(gt)
        sizes.append(img.shape[:2])
        cls_px += np.bincount(sem.ravel(), minlength=C.NUM_CLASSES)
        ov.append(overlap_stats(gt))
        _, segs = semantic_to_panoptic(sem)
        for c in C.THING_CLASSES:
            inst[c].append(sum(1 for g in segs if g["category_id"] == c))

    stats = {
        "n_images": len(samples),
        "image_sizes": sorted({f"{h}x{w}" for h, w in sizes}),
        "class_pixel_fraction": {C.CLASS_NAMES[i]: float(v / cls_px.sum()) for i, v in enumerate(cls_px)},
        "overlap_mean": {k: float(np.mean([o[k] for o in ov])) for k in ov[0]},
        "instances_per_image_mean": {C.CLASS_NAMES[c]: float(np.mean(v)) for c, v in inst.items()},
        "splits": {k: len(v) for k, v in splits.items() if k != "seed"},
    }
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.write_json(C.OUTPUTS_DIR / "dataset_stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # figuras de ejemplo: imagen | GT semántico | GT panóptico (instancias)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.visualization import colorize_panoptic, colorize_semantic, overlay
    for s in samples[:args.figures]:
        img = load_image(s["image"])
        sem = gt_to_semantic(load_gt(s["gt"], img.shape[:2]))
        pan, segs = semantic_to_panoptic(sem)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
        for a, (im, t) in zip(ax, [(img, f"recorte_{s['id']}"),
                                   (overlay(img, colorize_semantic(sem), .55), "GT semántico (solapes resueltos)"),
                                   (colorize_panoptic(pan, segs), f"GT panóptico ({len(segs)} segmentos)")]):
            a.imshow(im); a.set_title(t); a.axis("off")
        fig.tight_layout()
        fig.savefig(C.OUTPUTS_DIR / f"ejemplo_gt_{s['id']}.png", dpi=110)
        plt.close(fig)
    print("Figuras y estadísticas en", C.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
