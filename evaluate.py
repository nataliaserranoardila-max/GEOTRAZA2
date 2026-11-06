"""
evaluate.py
===========
Reproduce las métricas de un modelo entrenado sobre el split de test (o val).

Ejemplos:
    python evaluate.py --weights models/weights/mask2former_best.pt
    python evaluate.py --weights models/weights/efficientps_best.pt --mode tta
    python evaluate.py --weights models/weights/mask2former_best.pt --split val

Guarda outputs/metrics_<modelo>_<split>_<modo>.json y muestra la tabla.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import config as C
from data.dataset import list_samples, load_splits, load_image, load_gt, gt_to_semantic
from utils.metrics import SegmentationEvaluator, meets_targets


def evaluate_predictor(predictor, ids, seg_dir=C.SEG_DIR, mode="native", progress=True):
    samples = {s["id"]: s for s in list_samples(seg_dir)}
    ev = SegmentationEvaluator()
    times = []
    for sid in tqdm(ids, disable=not progress, desc=f"eval {predictor.name} [{mode}]"):
        s = samples[sid]
        img = load_image(s["image"])
        gt = gt_to_semantic(load_gt(s["gt"], img.shape[:2]))
        pred = predictor.predict(img, mode=mode)
        ev.add(gt, pred)
        times.append(pred["time_ms"])
    res = ev.compute()
    res["mean_time_ms"] = float(np.mean(times)) if times else 0.0
    res["mode"] = mode
    res["model"] = predictor.name
    res["targets"] = meets_targets(res)
    return res


def print_table(res: dict):
    print(f"\n=== {res['model']} [{res['mode']}]  ({res['n_images']} imágenes) ===")
    print(f"mIoU {res['mIoU']:.4f} | mAP {res['mAP']:.4f} | mAP50 {res['mAP50']:.4f} | "
          f"mF1 {res['mF1']:.4f} | mPrec {res['mPrecision']:.4f} | PQ {res['PQ']:.4f} | "
          f"{res['mean_time_ms']:.0f} ms/img")
    print(f"{'Clase':<18}{'IoU':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AP':>8}{'AP50':>8}{'PQ':>8}")
    for name, m in res["per_class"].items():
        print(f"{name:<18}{m['IoU']:>8.4f}{m['Precision']:>8.4f}{m['Recall']:>8.4f}"
              f"{m['F1']:>8.4f}{m['AP']:>8.4f}{m['AP50']:>8.4f}{m['PQ']:>8.4f}")
    a = res.get("alt")
    if a:
        print(f"Protocolo principal: {res['protocol']} | protocolo {res['alt_protocol']}: "
              f"mIoU {a['mIoU']:.4f} mAP {a['mAP']:.4f} mF1 {a['mF1']:.4f} PQ {a['PQ']:.4f}")
    print("Mínimos del taller:", res["targets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--mode", default="native", choices=["native", "tta"])
    ap.add_argument("--data-dir", default=str(C.SEG_DIR))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from models.predictor import load_predictor, get_device
    pred = load_predictor(args.weights, get_device(args.device))
    splits = load_splits(list_samples(args.data_dir))
    res = evaluate_predictor(pred, splits[args.split], args.data_dir, args.mode)
    print_table(res)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.OUTPUTS_DIR / f"metrics_{pred.name.lower()}_{args.split}_{args.mode}.json"
    C.write_json(out, res)
    print("Guardado:", out)


if __name__ == "__main__":
    main()
