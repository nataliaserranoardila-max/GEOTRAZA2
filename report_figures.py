"""
report_figures.py
=================
Genera todo el material para el informe en Word a partir de los resultados:

    outputs/tabla_comparativa.csv / .md      tabla de métricas de ambos modelos
    outputs/tabla_por_clase.csv              métricas por clase
    outputs/curvas_comparadas.png            pérdidas de ambos modelos
    outputs/comparacion_metricas.png         barras de métricas globales e IoU por clase
    outputs/confusion_<modelo>.png           matrices de confusión normalizadas
    outputs/cualitativo_<id>.png             imagen | GT | Mask2Former | EfficientPS | errores

Uso:
    python report_figures.py              # usa los pesos de models/weights y el dataset
    python report_figures.py --no-qual    # solo tablas y gráficas (sin inferencia)
"""
from __future__ import annotations

import argparse
import json

import config as C
from utils.plots import (plot_confusion, plot_history, plot_loss_comparison,
                         plot_model_comparison)

LABELS = {"mask2former": "Mask2Former", "efficientps": "EfficientPS"}


def load_results():
    res = {}
    for key, name in LABELS.items():
        for mode in ("native", "tta"):
            p = C.OUTPUTS_DIR / f"metrics_{key}_test_{mode}.json"
            if p.exists():
                res[f"{name} ({'TTA' if mode == 'tta' else 'nativo'})"] = C.read_json(p)
    return res


def tables(res):
    import pandas as pd
    rows = []
    for name, r in res.items():
        rows.append({"Modelo": name, "IoU (mIoU)": r["mIoU"], "mAP@[.5:.95]": r["mAP"], "mAP@0.5": r["mAP50"],
                     "mAP things": r.get("mAP_things", float("nan")), "mAP50 things": r.get("mAP50_things", float("nan")),
                     "F1 (mF1)": r["mF1"], "Precisión": r["mPrecision"], "Recall": r["mRecall"],
                     "PQ": r["PQ"], "ms/imagen": r.get("mean_time_ms", float("nan")),
                     "Protocolo": r.get("protocol", "strict"),
                     f"mIoU ({r.get('alt_protocol', '-')})": r.get("alt", {}).get("mIoU", float("nan")),
                     f"mF1 ({r.get('alt_protocol', '-')})": r.get("alt", {}).get("mF1", float("nan")),
                     f"mAP ({r.get('alt_protocol', '-')})": r.get("alt", {}).get("mAP", float("nan")),
                     "Cumple IoU≥0.7": r["mIoU"] >= .7, "Cumple mAP≥0.5": r["mAP"] >= .5,
                     "Cumple F1≥0.8": r["mF1"] >= .8})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(C.OUTPUTS_DIR / "tabla_comparativa.csv", index=False, encoding="utf-8-sig")
    (C.OUTPUTS_DIR / "tabla_comparativa.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    pc = []
    for name, r in res.items():
        for cls, m in r["per_class"].items():
            pc.append({"Modelo": name, "Clase": cls, **{k: round(v, 4) for k, v in m.items()}})
    pd.DataFrame(pc).to_csv(C.OUTPUTS_DIR / "tabla_por_clase.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    return df


def qualitative(n=4, mode="native"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.dataset import gt_to_semantic, list_samples, load_gt, load_image, load_splits
    from models.predictor import load_predictor
    from utils.visualization import colorize_panoptic, colorize_semantic, error_map, overlay

    preds = {}
    for key, name in LABELS.items():
        p = C.WEIGHTS_DIR / f"{key}_best.pt"
        if p.exists():
            preds[name] = load_predictor(p)
    samples = {s["id"]: s for s in list_samples()}
    ids = load_splits()["test"][:n]
    for sid in ids:
        img = load_image(samples[sid]["image"])
        gt = gt_to_semantic(load_gt(samples[sid]["gt"], img.shape[:2]))
        cols = [("Imagen", img), ("Ground truth", overlay(img, colorize_semantic(gt), .6))]
        for name, pr in preds.items():
            out = pr.predict(img, mode=mode)
            cols.append((f"{name} (semántica)", overlay(img, colorize_semantic(out["semantic"]), .6)))
            cols.append((f"{name} panóptica", colorize_panoptic(out["panoptic"], out["segments"])))
            cols.append((f"{name} errores vs GT", error_map(gt, out["semantic"], img)))
        k = len(cols)
        fig, ax = plt.subplots(1, k, figsize=(3.6 * k, 4))
        for a, (t, im) in zip(ax, cols):
            a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
        fig.tight_layout()
        fig.savefig(C.OUTPUTS_DIR / f"cualitativo_{sid}.png", dpi=110)
        plt.close(fig)
        print("cualitativo", sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-qual", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    C.OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

    hist = {}
    for key, name in LABELS.items():
        p = C.OUTPUTS_DIR / f"history_{key}.json"
        if p.exists():
            hist[name] = C.read_json(p)
            plot_history(hist[name], name, C.OUTPUTS_DIR / f"curvas_{key}.png")
    if hist:
        plot_loss_comparison(hist, C.OUTPUTS_DIR / "curvas_comparadas.png")

    res = load_results()
    if res:
        tables(res)
        plot_model_comparison(res, C.OUTPUTS_DIR / "comparacion_metricas.png")
        for name, r in res.items():
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plot_confusion(r, f"Matriz de confusión — {name}", C.OUTPUTS_DIR / f"confusion_{slug}.png")
    if not args.no_qual:
        qualitative(args.n)
    print("Material del informe en", C.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
