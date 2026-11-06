"""
models/postprocess.py
=====================
Post-procesamiento común a ambos modelos: a partir de un mapa panóptico
parcial + probabilidades por clase se arma la salida unificada que consumen
las métricas y la app web:

    {
      "panoptic":   (H,W) int32   id de segmento por píxel (0 = vacío)
      "segments":   [{id, category_id, isthing, score, area}]
      "semantic":   (H,W) uint8   clase por píxel (0..4)
      "regions":    (H,W) int32   regiones conexas (para mAP)
      "region_info":[(id, clase, score)]
      "probs":      (C,H,W) float16 probabilidades por clase
    }
"""
from __future__ import annotations

import cv2
import numpy as np

import config as C
from data.dataset import connected_instances


def split_instances(mask: np.ndarray, erode: int = 0):
    """Instancias de una máscara "thing". Con erode > 0 se erosiona la máscara para
    cortar puentes delgados entre manzanas vecinas, se etiquetan los núcleos y cada
    píxel vuelve al núcleo más cercano dentro de su misma región conexa."""
    if erode <= 0:
        return connected_instances(mask, 1)
    m8 = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    core = cv2.erode(m8, k)
    n, lab = cv2.connectedComponents(core, connectivity=8)
    if n <= 2:  # 0 o 1 núcleo: no hay nada que separar
        return connected_instances(mask, 1)
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    near = lab[idx[0], idx[1]]
    ccn, cc = cv2.connectedComponents(m8, connectivity=8)
    key = cc.astype(np.int64) * (n + 1) + near          # (región conexa, núcleo más cercano)
    key[~mask] = -1
    uniq, inv = np.unique(key, return_inverse=True)
    out = inv.reshape(mask.shape).astype(np.int32) + (1 if uniq[0] != -1 else 0)
    out[~mask] = 0
    ids = [int(i) for i in np.unique(out) if i > 0]
    return out, ids


def region_score(base: float, area: int, mode: str = "mean") -> float:
    """Puntaje de una región para el ranking del mAP.
    'mean' = confianza media; 'mean_logarea' / 'mean_sqrtarea' premian las regiones
    grandes (las que casi siempre son aciertos) para que queden primero en el ranking."""
    if mode == "mean_logarea":
        return float(base * np.log1p(area))
    if mode == "mean_sqrtarea":
        return float(base * np.sqrt(area))
    return float(base)


def finalize(panoptic: np.ndarray, segments: list[dict], probs: np.ndarray,
             min_area: int = C.MIN_INSTANCE_AREA, region_min_area: int | None = None,
             score_mode: str = "mean", split_erode: int = 0) -> dict:
    """Completa los píxeles vacíos con la predicción semántica y arma la salida.

    - Píxeles sin segmento -> clase = argmax(probs).
      * stuff: se unen al segmento de esa clase (o se crea uno).
      * things: cada componente conexa se vuelve una instancia nueva; los fragmentos
        de menos de `min_area` píxeles se unen a la región vecina con más borde común.
    """
    pan = panoptic.astype(np.int32).copy()
    segs = [dict(s) for s in segments]
    sem_arg = probs.argmax(0).astype(np.uint8)
    next_id = (max([s["id"] for s in segs]) + 1) if segs else 1
    empty = pan == 0

    small = []  # fragmentos "thing" diminutos: se unen después a la región vecina
    if empty.any():
        from scipy import ndimage
        for cls in range(C.NUM_CLASSES):
            m = empty & (sem_arg == cls)
            if not m.any():
                continue
            if cls in C.THING_CLASSES:
                lab, ids = split_instances(m, split_erode)
                if not ids:
                    continue
                slices = ndimage.find_objects(lab)
                cnt = np.bincount(lab.ravel(), minlength=int(lab.max()) + 1)
                for i in ids:
                    sl = slices[i - 1]
                    if sl is None:
                        continue
                    sub = lab[sl] == i
                    if cnt[i] < min_area:          # ruido de pocos píxeles: no es una manzana
                        pan[sl][sub] = -1
                        small.append((sl, sub, cls))
                        continue
                    pan[sl][sub] = next_id
                    segs.append({"id": next_id, "category_id": cls, "isthing": True,
                                 "score": float(probs[cls][sl][sub].mean()), "fill": True})
                    next_id += 1
            else:
                existing = [s for s in segs if s["category_id"] == cls]
                if existing:
                    pan[m] = existing[0]["id"]
                else:
                    pan[m] = next_id
                    segs.append({"id": next_id, "category_id": cls, "isthing": False,
                                 "score": float(probs[cls][m].mean())})
                    next_id += 1

    # cada fragmento diminuto pasa a la región con la que más borde comparte
    H, W = pan.shape
    k3 = np.ones((3, 3), np.uint8)
    for sl, sub, cls in small:
        y0, y1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)
        x0, x1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)
        crop = pan[y0:y1, x0:x1]
        msk = np.zeros(crop.shape, bool)
        msk[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = sub
        ring = cv2.dilate(msk.astype(np.uint8), k3).astype(bool) & ~msk
        vals = crop[ring]
        vals = vals[vals > 0]
        if vals.size:
            crop[msk] = np.bincount(vals).argmax()
        else:  # aislado por completo: se conserva como instancia propia
            crop[msk] = next_id
            segs.append({"id": next_id, "category_id": cls, "isthing": True,
                         "score": float(probs[cls][y0:y1, x0:x1][msk].mean()), "fill": True})
            next_id += 1
    pan[pan < 0] = 0

    # áreas finales y descartar segmentos que quedaron vacíos
    areas = np.bincount(pan.ravel(), minlength=next_id)
    segs = [s for s in segs if s["id"] < len(areas) and areas[s["id"]] > 0]
    for s in segs:
        s["area"] = int(areas[s["id"]])

    # semántica derivada del panóptico (coherencia total entre ambas vistas)
    lut = np.zeros(next_id + 1, np.uint8)
    for s in segs:
        lut[s["id"]] = s["category_id"]
    semantic = lut[pan]

    # regiones para mAP: things = su segmento; stuff = componentes conexas.
    # region_min_area descarta fragmentos diminutos de la predicción (no toca el GT).
    rmin = region_min_area or min_area
    regions = np.zeros(pan.shape, np.int32)
    info = []
    rid = 1
    for s in segs:
        cls = s["category_id"]
        if cls not in C.EVAL_CLASSES:
            continue
        m = pan == s["id"]
        if s["isthing"]:
            if s["area"] < rmin:
                continue
            regions[m] = rid
            base = float(s.get("score", probs[cls][m].mean()))
            info.append((rid, cls, region_score(base, s["area"], score_mode)))
            rid += 1
        else:
            lab, ids = connected_instances(m, rmin)
            for i in ids:
                mm = lab == i
                regions[mm] = rid
                info.append((rid, cls, region_score(float(probs[cls][mm].mean()), int(mm.sum()), score_mode)))
                rid += 1

    return {"panoptic": pan, "segments": segs, "semantic": semantic,
            "regions": regions, "region_info": info, "probs": probs.astype(np.float16)}
