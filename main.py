"""
main.py
=======
Aplicación web "GEOTRAZA · Estación de análisis orbital".

Servidor FastAPI que carga los dos modelos entrenados (Mask2Former y
EfficientPS), recibe una imagen satelital y devuelve la segmentación
panóptica de ambos, sus estadísticas y — si la imagen tiene ground truth —
las métricas IoU, mAP, F1, precisión y PQ calculadas en vivo.

Ejecutar:
    python main.py                      # http://127.0.0.1:8000
    python main.py --port 8080 --device cpu

Endpoints:
    GET  /                    interfaz web
    GET  /api/status          modelos cargados, dispositivo
    GET  /api/samples         imágenes del dataset (test con GT + adicionales)
    GET  /api/thumb/{kind}/{name}   miniatura
    POST /api/predict         inferencia (archivo o muestra del dataset)
    GET  /api/benchmark       métricas de test + historial de entrenamiento
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import config as C
from data.dataset import (gt_to_semantic, list_extra_images, list_samples, load_gt,
                          load_image, load_splits)
from utils.metrics import evaluate_single, meets_targets
from utils.visualization import colorize_panoptic, disagreement_map

ROOT = C.ROOT
app = FastAPI(title="GEOTRAZA · Cartografía Automatizada")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=C.OUTPUTS_DIR), name="outputs")  # figuras del entrenamiento

MODELS: dict = {}          # nombre -> predictor
MODEL_INFO: dict = {}      # nombre -> metadatos (época, métricas de validación)
LOCK = threading.Lock()    # una inferencia a la vez (memoria de GPU)
DEVICE = None
MAX_UPLOAD_SIDE = 4096


# --------------------------------------------------------------------------
# Carga de modelos
# --------------------------------------------------------------------------
def load_models(device=None, infer_size=None):
    global DEVICE
    from models.predictor import get_device, load_predictor
    DEVICE = get_device(device)
    for fname in ["mask2former_best.pt", "efficientps_best.pt"]:
        p = C.WEIGHTS_DIR / fname
        if not p.exists():
            print(f"[!] No se encontró {p} — ese modelo quedará desactivado.")
            continue
        t0 = time.time()
        pred = load_predictor(p, DEVICE, infer_size)
        MODELS[pred.name] = pred
        meta = getattr(pred, "meta", {})
        MODEL_INFO[pred.name] = {"epoch": meta.get("epoch"), "val_metrics": meta.get("val_metrics"),
                                 "file": fname, "size_mb": round(p.stat().st_size / 2**20, 1),
                                 "params_m": round(sum(q.numel() for q in pred.model.parameters()) / 1e6, 1),
                                 "train": {k: (meta.get("args") or {}).get(k) for k in
                                           ["epochs", "batch_size", "lr", "weight_decay", "crop",
                                            "crops_per_image", "checkpoint", "backbone", "class_weights",
                                            "infer_size", "backbone_lr_mult"]},
                                 "post": getattr(pred, "post", None)}
        print(f"[ok] {pred.name} cargado en {time.time() - t0:.1f}s ({DEVICE})")


# --------------------------------------------------------------------------
# Utilidades de codificación
# --------------------------------------------------------------------------
def png_b64(arr: np.ndarray) -> str:
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def jpg_b64(arr: np.ndarray, q=90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def segment_payload(pred: dict) -> tuple[np.ndarray, list]:
    """Mapa de ids de segmento codificado en RGB (id = R + 256·G) + lista de segmentos,
    para que el navegador sepa qué instancia hay bajo el cursor."""
    pan = pred["panoptic"]
    enc = np.zeros(pan.shape + (3,), np.uint8)
    enc[..., 0] = pan % 256
    enc[..., 1] = (pan // 256) % 256
    segs = [{"id": s["id"], "cls": s["category_id"], "thing": bool(s["isthing"]),
             "score": round(float(s.get("score", 0)), 3), "area": s["area"]}
            for s in pred["segments"]]
    return enc, segs


def download_image(url: str, limit=30 * 2**20) -> bytes:
    """Descarga una imagen desde un enlace (http/https) con límite de tamaño."""
    import urllib.request
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "El enlace debe empezar por http:// o https://")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (GEOTRAZA academic demo)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read(limit + 1)
    except Exception as e:
        raise HTTPException(400, f"No se pudo descargar la imagen del enlace ({type(e).__name__}). "
                                 "Prueba descargarla y subirla como archivo.")
    if len(data) > limit:
        raise HTTPException(400, "La imagen del enlace pesa más de 30 MB.")
    return data


def stats(pred: dict) -> dict:
    sem = pred["semantic"]
    tot = sem.size
    cov = {C.CLASS_NAMES[c]: float((sem == c).sum() / tot) for c in range(C.NUM_CLASSES)}
    inst = {C.CLASS_NAMES[c]: sum(1 for s in pred["segments"] if s["category_id"] == c)
            for c in C.THING_CLASSES}
    regions = {C.CLASS_NAMES[c]: sum(1 for r in pred["region_info"] if r[1] == c)
               for c in C.EVAL_CLASSES}
    conf = pred["probs"].astype(np.float32).max(0)
    return {"coverage": cov, "instances": inst, "regions": regions,
            "mean_confidence": float(conf.mean()),
            "low_conf_fraction": float((conf < 0.6).mean())}


def compact_metrics(res: dict) -> dict:
    keys = ["mIoU", "mAP", "mAP50", "mAP_things", "mAP50_things", "mF1", "mPrecision", "mRecall", "PQ", "PixelAccuracy"]
    keys = [k for k in keys if k in res]
    out = {k: res[k] for k in keys}
    out["per_class"] = res["per_class"]
    out["targets"] = meets_targets(res)
    out["protocol"] = res.get("protocol", "strict")
    out["alt_protocol"] = res.get("alt_protocol")
    out["alt"] = res.get("alt")
    return out


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return {"device": str(DEVICE), "models": MODEL_INFO,
            "classes": [{"id": i, "name": n, "color": C.CLASS_COLORS[i],
                         "thing": i in C.THING_CLASSES} for i, n in enumerate(C.CLASS_NAMES)],
            "dataset_available": C.SEG_DIR.exists()}


@lru_cache(maxsize=1)
def _sample_index():
    items = []
    if C.SEG_DIR.exists():
        samples = list_samples()
        splits = load_splits(samples)
        where = {i: k for k in ("train", "val", "test") for i in splits[k]}
        for s in samples:
            items.append({"kind": "seg", "name": Path(s["image"]).name, "id": s["id"],
                          "split": where.get(s["id"], "?"), "has_gt": True})
    for p in list_extra_images():
        items.append({"kind": "extra", "name": Path(p).name, "split": "sin etiqueta", "has_gt": False})
    if C.EXTERNAL_DIR.exists():  # pruebas de generalización con imágenes de otras fuentes
        for p in sorted(C.EXTERNAL_DIR.glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                items.append({"kind": "ext", "name": p.name, "split": "externa", "has_gt": False})
    return items


@app.get("/api/samples")
def samples():
    items = _sample_index()
    order = {"test": 0, "externa": 1, "val": 2, "sin etiqueta": 3, "train": 4}
    return sorted(items, key=lambda s: (order.get(s["split"], 9), s.get("id", 0)))


def _sample_path(kind: str, name: str) -> Path:
    base = {"seg": C.SEG_DIR, "extra": C.EXTRA_DIR, "ext": C.EXTERNAL_DIR}.get(kind)
    if base is None:
        raise HTTPException(404, "Muestra no encontrada")
    p = (base / name).resolve()
    if base.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "Muestra no encontrada")
    return p


@app.get("/api/thumb/{kind}/{name}")
def thumb(kind: str, name: str):
    img = load_image(_sample_path(kind, name))
    f = 220 / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


@app.post("/api/predict")
def predict(file: UploadFile | None = File(None), gt_file: UploadFile | None = File(None),
            sample_kind: str | None = Form(None), sample_name: str | None = Form(None),
            image_url: str | None = Form(None), scale: float = Form(1.0),
            mode: str = Form("native")):
    if not MODELS:
        raise HTTPException(503, "No hay modelos cargados. Copia los archivos .pt en models/weights/ "
                                 "y reinicia la aplicación.")
    if mode not in ("native", "tta"):
        raise HTTPException(400, "mode debe ser 'native' o 'tta'")

    # ---------- imagen de entrada ----------
    gt_sem, source = None, "subida"
    raw = None
    if file is not None:
        raw, name = file.file.read(), file.filename
    elif image_url:
        raw, name, source = download_image(image_url), image_url.split("?")[0].rsplit("/", 1)[-1] or "imagen", "enlace"
    if raw is not None:
        data = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(400, "No se pudo leer la imagen. Usa PNG, JPG o TIFF.")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype != np.uint8:
            img = (img / max(img.max(), 1) * 255).astype(np.uint8)
    elif sample_kind and sample_name:
        p = _sample_path(sample_kind, sample_name)
        img = load_image(p)
        name, source = sample_name, "dataset"
        if sample_kind == "seg":
            sid = p.stem.split("_")[-1]
            gt_path = C.SEG_DIR / f"ground_truth_recorte_{sid}.npy"
            if gt_path.exists():
                gt_sem = gt_to_semantic(load_gt(gt_path, img.shape[:2]))
    else:
        raise HTTPException(400, "Envía una imagen o elige una muestra del dataset.")

    # ajuste de escala: acerca/aleja la imagen para parecerse a la resolución del entrenamiento
    scale = float(np.clip(scale, 0.25, 3.0))
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (max(int(img.shape[1] * scale), 32), max(int(img.shape[0] * scale), 32)),
                         interpolation=interp)
        if gt_sem is not None:
            gt_sem = cv2.resize(gt_sem, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if max(img.shape[:2]) > MAX_UPLOAD_SIDE:
        f = MAX_UPLOAD_SIDE / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)

    if gt_file is not None and gt_sem is None:  # GT opcional subido por el usuario (.npy H x W x 4)
        try:
            gt_sem = gt_to_semantic(load_gt(io.BytesIO(gt_file.file.read()), img.shape[:2]))
        except Exception:
            raise HTTPException(400, "El ground truth debe ser un .npy de forma H x W x 4.")

    # ---------- inferencia ----------
    results = {}
    with LOCK:
        for mname, predictor in MODELS.items():
            pred = predictor.predict(img, mode=mode)
            enc, segs = segment_payload(pred)
            conf = (pred["probs"].astype(np.float32).max(0) * 255).astype(np.uint8)
            results[mname] = {
                "semantic": png_b64(pred["semantic"]),        # etiquetas 0..4 (escala de grises)
                "panoptic": png_b64(colorize_panoptic(pred["panoptic"], pred["segments"])),
                "segments_map": png_b64(enc),
                "confidence": png_b64(conf),
                "segments": segs,
                "stats": stats(pred),
                "time_ms": round(pred["time_ms"], 1),
                "metrics": compact_metrics(evaluate_single(gt_sem, pred)) if gt_sem is not None else None,
                "_sem": pred["semantic"],
            }

    comparison = None
    names = list(results)
    if len(names) == 2:
        a, b = results[names[0]]["_sem"], results[names[1]]["_sem"]
        agree = float((a == b).mean())
        per_cls = {}
        for c in C.EVAL_CLASSES:
            inter = ((a == c) & (b == c)).sum(); uni = ((a == c) | (b == c)).sum()
            per_cls[C.CLASS_NAMES[c]] = float(inter / uni) if uni else 1.0
        comparison = {"agreement": agree, "class_agreement_iou": per_cls,
                      "disagreement": png_b64(disagreement_map(a, b, img))}
    for r in results.values():
        r.pop("_sem")

    return {"name": name, "source": source, "mode": mode, "width": img.shape[1], "height": img.shape[0],
            "image": jpg_b64(img), "gt": png_b64(gt_sem) if gt_sem is not None else None,
            "results": results, "comparison": comparison}


@app.get("/api/benchmark")
def benchmark():
    out = {"test": {}, "history": {}, "figures": []}
    for key, label in [("mask2former", "Mask2Former"), ("efficientps", "EfficientPS")]:
        for mode in ("native", "tta"):
            p = C.OUTPUTS_DIR / f"metrics_{key}_test_{mode}.json"
            if p.exists():
                r = C.read_json(p)
                out["test"].setdefault(label, {})[mode] = compact_metrics(r) | {
                    "mean_time_ms": r.get("mean_time_ms"), "n_images": r.get("n_images")}
        h = C.OUTPUTS_DIR / f"history_{key}.json"
        if h.exists():
            hist = C.read_json(h)
            hist.pop("train_components", None)
            out["history"][label] = hist
    return out


@app.get("/api/hero")
def hero():
    """Escena de portada: una imagen del split de test y su ground truth superpuesto."""
    if not C.SEG_DIR.exists():
        return None
    from utils.visualization import colorize_semantic, overlay
    samples = {s["id"]: s for s in list_samples()}
    ids = [i for i in load_splits(list(samples.values()))["test"] if i in samples]
    if not ids:
        return None
    s = samples[ids[0]]
    img = load_image(s["image"])
    gt = gt_to_semantic(load_gt(s["gt"], img.shape[:2]))
    f = min(1.0, 1400 / max(img.shape[:2]))
    if f < 1:
        size = (int(img.shape[1] * f), int(img.shape[0] * f))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
    lab = overlay(img, colorize_semantic(gt), 0.62, mask=gt > 0)
    return {"name": Path(s["image"]).name, "image": jpg_b64(img, 88), "labeled": jpg_b64(lab, 88),
            "width": img.shape[1], "height": img.shape[0]}


@app.post("/api/capture")
def capture(name: str = Form(...), data: str = Form(...)):
    """Guarda una captura de la página (PNG en base64) en outputs/capturas/ para el informe."""
    import re
    if not re.fullmatch(r"[a-z0-9_\-]{1,48}", name):
        raise HTTPException(400, "Nombre de captura inválido")
    raw = base64.b64decode(data.split(",", 1)[-1])
    if len(raw) > 25 * 2**20 or not raw.startswith(b"\x89PNG"):
        raise HTTPException(400, "La captura debe ser un PNG de menos de 25 MB")
    d = C.OUTPUTS_DIR / "capturas"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.png").write_bytes(raw)
    return {"saved": f"outputs/capturas/{name}.png", "bytes": len(raw)}


@app.get("/api/mission")
def mission():
    """Todo lo que la presentación necesita: equipo, dataset, entrenamiento, ajuste y figuras."""
    def jload(name):
        p = C.OUTPUTS_DIR / name
        return C.read_json(p) if p.exists() else None

    figs = {}
    for key, pattern in [("ejemplos_gt", "ejemplo_gt_*.png"), ("aumento", "aumento_datos.png"),
                         ("curvas", "curvas_*.png"), ("comparacion", "comparacion_metricas.png"),
                         ("confusion", "confusion_*.png"), ("cualitativo", "cualitativo_*.png")]:
        figs[key] = [f"/outputs/{p.name}" for p in sorted(C.OUTPUTS_DIR.glob(pattern))]

    tuning = {}
    search = jload("postprocess_busqueda_val.json") or {}
    for name, d in search.items():
        t = {}
        for k, label in [("tta_semantica", "Semántica (TTA)"), ("native_instancias", "Instancias (nativo)"),
                         ("tta_instancias", "Instancias (TTA)")]:
            if k in d:
                t[label] = {"antes": d[k]["por_defecto"], "despues": d[k]["mejor"]}
        tuning[name] = t

    training = {}
    for key, label in [("mask2former", "Mask2Former"), ("efficientps", "EfficientPS")]:
        h = jload(f"history_{key}.json")
        if h:
            ts = [t for t in h.get("time_s", []) if t]
            training[label] = {"epochs_run": len(h.get("epoch", [])),
                               "mean_epoch_s": round(float(np.mean(ts)), 1) if ts else None,
                               "total_min": round(float(np.sum(ts)) / 60, 1) if ts else None,
                               "best_val_mIoU": max(h.get("val_mIoU") or [0])}
    return {"agency": getattr(C, "AGENCY", None), "mission": getattr(C, "MISSION", None),
            "team": C.TEAM, "course": C.COURSE, "professor": C.PROFESSOR, "university": C.UNIVERSITY,
            "dataset": jload("dataset_stats.json"), "figures": figs, "tuning": tuning,
            "training": training, "models": MODEL_INFO,
            "classes": [{"id": i, "name": n, "color": C.CLASS_COLORS[i]} for i, n in enumerate(C.CLASS_NAMES)],
            "protocol": "void" if C.IGNORE_VOID else "strict"}


@app.get("/favicon.ico")
def favicon():
    return FileResponse(ROOT / "static" / "favicon.svg", media_type="image/svg+xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (auto por defecto)")
    ap.add_argument("--infer-size", type=int, default=None,
                    help="lado mayor en inferencia (menor = más rápido en CPU)")
    args = ap.parse_args()
    load_models(args.device, args.infer_size)
    import uvicorn
    print(f"\n  GEOTRAZA listo en  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
