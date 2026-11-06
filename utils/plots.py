"""
utils/plots.py
==============
Gráficas para el informe: curvas de pérdida (train / val), evolución de
métricas de validación, comparación de modelos, matrices de confusión y
mosaicos cualitativos.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def plot_history(history: dict, name: str, out_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ep = history["epoch"]
    ax[0].plot(ep, history["train_loss"], label="Entrenamiento", lw=2)
    ax[0].plot(ep, history["val_loss"], label="Validación (prueba)", lw=2)
    ax[0].set(title=f"{name}: pérdida", xlabel="Época", ylabel="Pérdida"); ax[0].legend()

    comps = history.get("train_components", [])
    keys = [k for k in (comps[0].keys() if comps else []) if k != "loss"]
    for k in keys:
        ax[1].plot(ep, [c.get(k, np.nan) for c in comps], label=k, lw=1.5)
    ax[1].set(title="Componentes de la pérdida (train)", xlabel="Época")
    if keys:
        ax[1].legend(fontsize=7)
    else:
        ax[1].plot(ep, history["lr"], color="gray"); ax[1].set_title("Tasa de aprendizaje")

    ee = history.get("eval_epoch", [])
    for k, lab in [("val_mIoU", "mIoU"), ("val_mF1", "mF1"), ("val_mAP", "mAP"), ("val_PQ", "PQ")]:
        if history.get(k):
            ax[2].plot(ee, history[k], marker="o", label=lab, lw=2)
    ax[2].axhline(0.7, ls="--", c="C0", alpha=.5); ax[2].axhline(0.8, ls="--", c="C1", alpha=.5)
    ax[2].axhline(0.5, ls="--", c="C2", alpha=.5)
    ax[2].set(title="Métricas en validación", xlabel="Época", ylim=(0, 1)); ax[2].legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_loss_comparison(histories: dict, out_path=None):
    """Curvas de ambos modelos en una sola figura (para el informe)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in histories.items():
        tr = np.array(h["train_loss"]); va = np.array(h["val_loss"])
        ax[0].plot(h["epoch"], tr / tr[0], label=f"{name} train", lw=2)
        ax[0].plot(h["epoch"], va / va[0], "--", label=f"{name} val", lw=2)
        if h.get("val_mIoU"):
            ax[1].plot(h["eval_epoch"], h["val_mIoU"], marker="o", label=f"{name} mIoU")
    ax[0].set(title="Pérdida normalizada (÷ época 1)", xlabel="Época"); ax[0].legend()
    ax[1].set(title="mIoU en validación", xlabel="Época", ylim=(0, 1)); ax[1].legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_model_comparison(results: dict, out_path=None):
    """Barras agrupadas: métricas globales y IoU por clase para cada modelo."""
    metrics = ["mIoU", "mAP", "mAP50", "mF1", "mPrecision", "PQ"]
    names = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    w = 0.8 / len(names)
    x = np.arange(len(metrics))
    for i, n in enumerate(names):
        v = [results[n][m] for m in metrics]
        b = ax[0].bar(x + i * w, v, w, label=n)
        ax[0].bar_label(b, fmt="%.2f", fontsize=7)
    ax[0].set_xticks(x + w * (len(names) - 1) / 2, metrics)
    ax[0].set(ylim=(0, 1.05), title="Métricas globales (test)"); ax[0].legend()
    for thr, c in [(0.7, "C0"), (0.8, "C1"), (0.5, "C2")]:
        ax[0].axhline(thr, ls="--", c=c, alpha=.4)
    cls = [C.CLASS_NAMES[c] for c in C.EVAL_CLASSES]
    x = np.arange(len(cls))
    for i, n in enumerate(names):
        v = [results[n]["per_class"][c]["IoU"] for c in cls]
        b = ax[1].bar(x + i * w, v, w, label=n)
        ax[1].bar_label(b, fmt="%.2f", fontsize=7)
    ax[1].set_xticks(x + w * (len(names) - 1) / 2, cls)
    ax[1].set(ylim=(0, 1.05), title="IoU por clase (test)"); ax[1].legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_confusion(res: dict, title: str, out_path=None):
    cm = np.array(res.get("confusion_matrix_strict", res["confusion_matrix"]), np.float64)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.CLASS_NAMES)), C.CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(len(C.CLASS_NAMES)), C.CLASS_NAMES)
    for i in range(len(cmn)):
        for j in range(len(cmn)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set(xlabel="Predicción", ylabel="Ground truth", title=title)
    ax.grid(False)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def load_json(path):
    return C.read_json(path)
