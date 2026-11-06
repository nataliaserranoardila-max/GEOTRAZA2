"""
tune_postprocess.py
===================
Ajusta el post-procesamiento de cada modelo usando SOLO el split de
VALIDACIÓN (el test no se toca, para no sesgar las métricas finales).
El ground truth nunca se modifica; solo cambia cómo se convierten las
salidas del modelo en mapas semánticos e instancias.

Protocolo: el de config.IGNORE_VOID (por defecto, píxeles sin anotar ignorados).

Etapa 1 — semántica (afecta IoU, F1, precisión; solo modo TTA):
  - scales  : TTA en una escala (1.0) o multiescala (1.0 + 1.5). Ampliar la
              imagen ayuda con estructuras delgadas como los andenes.
  - sigma   : suavizado gaussiano de las probabilidades.
  - bias    : factores que multiplican la probabilidad de Área Caminable
              (clase delgada que el modelo tiende a perder) y del Fondo
              (0 = nunca predecir "sin etiqueta").
  Criterio: (mIoU + mF1) / 2.

Etapa 2 — instancias (afecta mAP y PQ; ambos modos):
  - split_erode     : erosión (px) para separar manzanas unidas por puentes
                      delgados antes de contar instancias.
  - region_min_area : descarta regiones predichas diminutas.
  - score_mode      : puntaje de ranking de cada región para el mAP.
  Criterio: mAP.

Resultado: outputs/postprocess.json (lo usan evaluate.py, report_figures.py
y la app) y outputs/postprocess_busqueda_val.json (detalle para el informe).

Uso:
    python tune_postprocess.py
    python tune_postprocess.py --models mask2former
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from tqdm.auto import tqdm

import config as C
from data.dataset import gt_to_semantic, list_samples, load_gt, load_image, load_splits
from models.common import POST_FILE, class_bias, panoptic_from_semantic, smooth_probs
from models.postprocess import finalize
from utils.metrics import SegmentationEvaluator

SEM_GRID = {"scales": [[1.0], [1.0, 1.5]], "sigma": [0, 1.5, 3.0],
            "bias": [{}, {"1": 1.25}, {"1": 1.5}, {"1": 2.0},
                     # "0": factor del Fondo; 0 = el modelo nunca predice "sin etiqueta"
                     {"0": 0.5}, {"0": 0.5, "1": 1.5}, {"0": 0.0}, {"0": 0.0, "1": 1.25},
                     {"0": 0.0, "1": 1.5}]}
INST_GRID = {"split_erode": [0, 3, 6], "region_min_area": [64, 1024],
             "score_mode": ["mean", "mean_logarea", "mean_sqrtarea"]}


def semantic_scores(gts, probs_list):
    """mIoU / mF1 solo con matriz de confusión (rápido)."""
    nc = C.NUM_CLASSES
    cm = np.zeros((nc, nc), np.int64)
    for gt, p in zip(gts, probs_list):
        pr = p.argmax(0)
        v = (gt < nc) & (gt != 0) if C.IGNORE_VOID else gt < nc
        cm += np.bincount(gt[v].astype(np.int64) * nc + pr[v], minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    ec = C.EVAL_CLASSES
    iou = tp / np.maximum(tp + fp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(iou[ec].mean()), float(f1[ec].mean())


def full_eval(gts, outs):
    ev = SegmentationEvaluator()
    for gt, o in zip(gts, outs):
        ev.add(gt, o)
    return ev.compute()


def row(params, r):
    return {**params, "mIoU": r["mIoU"], "mF1": r["mF1"], "mAP": r["mAP"], "mAP50": r["mAP50"],
            "mAP_things": r["mAP_things"], "PQ": r["PQ"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mask2former", "efficientps"])
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    from models.predictor import load_predictor
    samples = {s["id"]: s for s in list_samples()}
    ids = load_splits()[args.split]
    imgs = [load_image(samples[i]["image"]) for i in ids]
    gts = [gt_to_semantic(load_gt(samples[i]["gt"], im.shape[:2])) for i, im in zip(ids, imgs)]

    result = C.read_json(POST_FILE) if POST_FILE.exists() else {}
    report = C.read_json(C.OUTPUTS_DIR / f"postprocess_busqueda_{args.split}.json") \
        if (C.OUTPUTS_DIR / f"postprocess_busqueda_{args.split}.json").exists() else {}
    for key in args.models:
        path = C.WEIGHTS_DIR / f"{key}_best.pt"
        if not path.exists():
            print("No existe", path, "— se omite"); continue
        pred = load_predictor(path)
        name = pred.name
        rep = report[name] = {}

        # ---------------- inferencia (una vez) ----------------
        nat, tta = [], {1: [], 2: []}
        for im in tqdm(imgs, desc=f"{name}: inferencia en {args.split}"):
            o = pred.predict(im, "native")
            nat.append((o["panoptic"], o["segments"], o["probs"].astype(np.float32)))
            p1 = pred.tta_probs(im, (1.0,))
            p15 = pred.tta_probs(im, (1.5,))
            tta[1].append(p1.astype(np.float16))
            tta[2].append(((p1 + p15) / 2).astype(np.float16))

        # ---------------- etapa 1: semántica (TTA) ----------------
        sem_rows = []
        for sc, sg, bi in itertools.product(SEM_GRID["scales"], SEM_GRID["sigma"], SEM_GRID["bias"]):
            plist = [class_bias(smooth_probs(p.astype(np.float32), sg), bi) for p in tta[len(sc)]]
            miou, mf1 = semantic_scores(gts, plist)
            sem_rows.append({"scales": sc, "sigma": sg, "bias": bi, "mIoU": miou, "mF1": mf1,
                             "score": (miou + mf1) / 2})
        sem_rows.sort(key=lambda r: -r["score"])
        bs = sem_rows[0]
        base_sem = next(r for r in sem_rows if r["scales"] == [1.0] and r["sigma"] == 0 and not r["bias"])
        print(f"\n{name} [tta] semántica  por defecto mIoU {base_sem['mIoU']:.4f} mF1 {base_sem['mF1']:.4f}"
              f"  ->  ajustada mIoU {bs['mIoU']:.4f} mF1 {bs['mF1']:.4f}  "
              f"(escalas {bs['scales']}, sigma {bs['sigma']}, sesgo {bs['bias']})")
        rep["tta_semantica"] = {"por_defecto": base_sem, "mejor": bs, "top5": sem_rows[:5]}
        tta_best = [class_bias(smooth_probs(p.astype(np.float32), bs["sigma"]), bs["bias"])
                    for p in tta[len(bs["scales"])]]

        # ---------------- etapa 2: instancias ----------------
        best = {}
        for mode in ("native", "tta"):
            rows = []
            for se, rmin, sm in tqdm(list(itertools.product(*INST_GRID.values())),
                                     desc=f"{name} [{mode}] instancias", leave=False):
                kw = dict(split_erode=se, region_min_area=rmin, score_mode=sm)
                if mode == "native":
                    outs = [finalize(pan, segs, p, **kw) for pan, segs, p in nat]
                else:
                    outs = [panoptic_from_semantic(p, 0, None, **kw) for p in tta_best]
                rows.append(row(kw, full_eval(gts, outs)))
            rows.sort(key=lambda r: (-r["mAP"], -r["PQ"]))
            b = rows[0]
            base = next(r for r in rows if r["split_erode"] == 0 and r["region_min_area"] == 64
                        and r["score_mode"] == "mean")
            print(f"{name} [{mode}] instancias por defecto mAP {base['mAP']:.4f} PQ {base['PQ']:.4f}"
                  f"  ->  ajustada mAP {b['mAP']:.4f} (things {b['mAP_things']:.4f}) PQ {b['PQ']:.4f}  "
                  f"(erosión {b['split_erode']}, área mín {b['region_min_area']}, puntaje {b['score_mode']})")
            rep[f"{mode}_instancias"] = {"por_defecto": base, "mejor": b, "top5": rows[:5]}
            best[mode] = {k: b[k] for k in INST_GRID}
        best["tta"].update({"scales": bs["scales"], "sigma": bs["sigma"], "bias": bs["bias"]})
        result[name] = best

        C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        C.write_json(POST_FILE, result)
        C.write_json(C.OUTPUTS_DIR / f"postprocess_busqueda_{args.split}.json", report)
    print("\nGuardado", POST_FILE)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
