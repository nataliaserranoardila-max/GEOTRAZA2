"""
utils/metrics.py
================
Métricas de evaluación pedidas en el taller (y alguna extra):

Semánticas (por píxel, sobre la matriz de confusión acumulada):
    IoU  = TP / (TP + FP + FN)
    Precisión = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1 (Dice) = 2·P·R / (P + R)
  Se reportan por clase y como promedio (mIoU, mF1, mPrecisión) sobre las
  4 clases del taller (el fondo no entra en el promedio).

Por instancias / regiones (estilo COCO):
    AP por clase con IoU de máscara en umbrales 0.50:0.05:0.95 y
    interpolación de 101 puntos. mAP = promedio sobre clases.
  Como el GT no trae instancias, una "instancia" GT es cada región conexa de
  una clase. Las predicciones son los segmentos de la salida panóptica
  (stuff partido en regiones conexas), con su confianza como puntaje.

Panóptica (Kirillov et al., 2019):
    PQ = SQ · RQ, con emparejamiento único IoU > 0.5.
"""
from __future__ import annotations

import numpy as np

import config as C
from data.dataset import semantic_to_panoptic, semantic_to_regions, connected_instances

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def label_iou_matrix(gt_lab: np.ndarray, pr_lab: np.ndarray, n_gt: int, n_pr: int):
    """IoU entre todos los segmentos de dos mapas de etiquetas NO solapados.
    Se calcula con un histograma conjunto (muy rápido, sin bucles por par).
    Devuelve (iou [n_gt+1, n_pr+1], intersecciones, áreas gt, áreas pred)."""
    inter = np.bincount(gt_lab.ravel().astype(np.int64) * (n_pr + 1) + pr_lab.ravel(),
                        minlength=(n_gt + 1) * (n_pr + 1)).reshape(n_gt + 1, n_pr + 1)
    a_gt = inter.sum(1)
    a_pr = inter.sum(0)
    union = a_gt[:, None] + a_pr[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou, inter, a_gt, a_pr


def _relabel(lab: np.ndarray, ids: list[int]):
    """Reetiqueta ids arbitrarios a 1..n (0 = nada)."""
    lut = np.zeros(int(lab.max()) + 1 if lab.size else 1, np.int64)
    for k, i in enumerate(ids, 1):
        if i < len(lut):
            lut[i] = k
    return lut[lab]


# --------------------------------------------------------------------------
# Acumulador
# --------------------------------------------------------------------------

class _Evaluator:
    """Acumula estadísticas sobre varias imágenes y calcula todas las métricas.

    ignore_void=True: los píxeles SIN anotación en el GT (Fondo) no cuentan ni como
    acierto ni como error (protocolo "void", como en Cityscapes/ADE20K). Los segmentos
    predichos que caen en más de un 50 % sobre zona sin anotar se ignoran.

    Uso:
        ev = SegmentationEvaluator()
        for cada imagen: ev.add(gt_sem, pred)   # pred = salida de un Predictor
        res = ev.compute()
    """

    def __init__(self, num_classes=C.NUM_CLASSES, eval_classes=C.EVAL_CLASSES, ignore_void=False):
        self.ignore_void = ignore_void
        self.nc = num_classes
        self.eval_classes = list(eval_classes)
        self.cm = np.zeros((num_classes, num_classes), np.int64)
        # AP: por clase -> lista de (score, [tp por umbral])  y número de GT
        self.ap_records = {c: [] for c in self.eval_classes}
        self.ap_ngt = {c: 0 for c in self.eval_classes}
        # PQ: por clase
        self.pq = {c: dict(tp=0, fp=0, fn=0, iou=0.0) for c in self.eval_classes}
        self.n_images = 0

    # ---------------- semántica ----------------
    def _add_confusion(self, gt, pr):
        valid = (gt < self.nc) & ((gt != 0) if self.ignore_void else True)
        idx = gt[valid].astype(np.int64) * self.nc + pr[valid].astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.nc ** 2).reshape(self.nc, self.nc)

    def _mask_void(self, gt_sem, lab, ids):
        """Protocolo void: quita de la predicción los píxeles sin anotar y descarta los
        segmentos que quedaban en su mayoría (> 50 %) sobre esa zona."""
        if not self.ignore_void:
            return lab, ids
        n = int(lab.max()) + 1
        before = np.bincount(lab.ravel(), minlength=n)
        lab = lab.copy()
        lab[gt_sem == 0] = 0
        after = np.bincount(lab.ravel(), minlength=n)
        keep = {i for i in ids if i < n and before[i] > 0 and after[i] >= 0.5 * before[i]}
        drop = [i for i in ids if i not in keep]
        if drop:
            lut = np.arange(n); lut[drop] = 0
            lab = lut[lab]
        return lab, [i for i in ids if i in keep]

    # ---------------- AP por regiones ----------------
    def _add_ap(self, gt_sem, pred_regions, pred_info):
        if self.ignore_void:
            pred_regions, kept = self._mask_void(gt_sem, pred_regions, [r[0] for r in pred_info])
            kept = set(kept)
            info = [r for r in pred_info if r[0] in kept]
            pred_regions = _relabel(pred_regions, [r[0] for r in info])
            pred_info = [(k, r[1], r[2]) for k, r in enumerate(info, 1)]
        gt_reg, gt_info = semantic_to_regions(gt_sem, self.eval_classes)
        n_gt, n_pr = len(gt_info), len(pred_info)
        iou, *_ = label_iou_matrix(gt_reg, pred_regions, n_gt, n_pr)
        for c in self.eval_classes:
            g_idx = [k for k, (_, cls) in enumerate(gt_info, 1) if cls == c]
            p = [(k, s) for k, (_, cls, s) in enumerate(pred_info, 1) if cls == c]
            self.ap_ngt[c] += len(g_idx)
            if not p:
                continue
            p.sort(key=lambda t: -t[1])
            sub = iou[np.ix_(g_idx, [k for k, _ in p])] if g_idx else np.zeros((0, len(p)))
            tps = np.zeros((len(p), len(IOU_THRESHOLDS)), bool)
            for ti, t in enumerate(IOU_THRESHOLDS):
                matched = np.zeros(len(g_idx), bool)
                for j in range(len(p)):          # greedy por puntaje descendente
                    if not len(g_idx):
                        break
                    cand = np.where(~matched, sub[:, j], -1)
                    b = int(cand.argmax())
                    if cand[b] >= t:
                        matched[b] = True
                        tps[j, ti] = True
            for j, (_, s) in enumerate(p):
                self.ap_records[c].append((s, tps[j]))

    # ---------------- PQ ----------------
    def _add_pq(self, gt_sem, pred_pan, pred_segments):
        gt_pan, gt_segs = semantic_to_panoptic(gt_sem)
        gt_segs = [s for s in gt_segs if s["category_id"] in self.eval_classes]
        pr_segs = [s for s in pred_segments if s["category_id"] in self.eval_classes]
        g_lab = _relabel(gt_pan, [s["id"] for s in gt_segs])
        if self.ignore_void:
            pred_pan, kept = self._mask_void(gt_sem, pred_pan, [s["id"] for s in pr_segs])
            kept = set(kept)
            pr_segs = [s for s in pr_segs if s["id"] in kept]
        p_lab = _relabel(pred_pan, [s["id"] for s in pr_segs])
        iou, *_ = label_iou_matrix(g_lab, p_lab, len(gt_segs), len(pr_segs))
        g_matched, p_matched = set(), set()
        for gi, g in enumerate(gt_segs, 1):
            for pi, p in enumerate(pr_segs, 1):
                if g["category_id"] == p["category_id"] and iou[gi, pi] > 0.5:
                    self.pq[g["category_id"]]["tp"] += 1
                    self.pq[g["category_id"]]["iou"] += float(iou[gi, pi])
                    g_matched.add(gi)
                    p_matched.add(pi)
        for gi, g in enumerate(gt_segs, 1):
            if gi not in g_matched:
                self.pq[g["category_id"]]["fn"] += 1
        for pi, p in enumerate(pr_segs, 1):
            if pi not in p_matched:
                self.pq[p["category_id"]]["fp"] += 1

    # ---------------- API ----------------
    def add(self, gt_sem: np.ndarray, pred: dict):
        """gt_sem: (H,W) uint8. pred: dict con 'semantic', 'panoptic', 'segments',
        'regions', 'region_info' (ver models/postprocess.py)."""
        self._add_confusion(gt_sem, pred["semantic"])
        self._add_ap(gt_sem, pred["regions"], pred["region_info"])
        self._add_pq(gt_sem, pred["panoptic"], pred["segments"])
        self.n_images += 1

    def compute(self) -> dict:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

        per_class = {}
        ap_all, ap50_all, ap75_all, pq_all = [], [], [], []
        for c in self.eval_classes:
            ap_t = self._ap_for_class(c)
            q = self.pq[c]
            sq = q["iou"] / max(q["tp"], 1)
            rq = q["tp"] / max(q["tp"] + 0.5 * q["fp"] + 0.5 * q["fn"], 1e-12)
            per_class[C.CLASS_NAMES[c]] = {
                "IoU": float(iou[c]), "Precision": float(prec[c]), "Recall": float(rec[c]),
                "F1": float(f1[c]), "AP": float(ap_t.mean()), "AP50": float(ap_t[0]),
                "AP75": float(ap_t[5]), "PQ": float(sq * rq), "SQ": float(sq), "RQ": float(rq),
            }
            ap_all.append(ap_t.mean()); ap50_all.append(ap_t[0]); ap75_all.append(ap_t[5])
            pq_all.append(sq * rq)

        ec = self.eval_classes
        total = cm.sum()
        th = [C.CLASS_NAMES[c] for c in C.THING_CLASSES]
        return {
            "mIoU": float(iou[ec].mean()),
            "mF1": float(f1[ec].mean()),
            "mPrecision": float(prec[ec].mean()),
            "mRecall": float(rec[ec].mean()),
            "mAP": float(np.mean(ap_all)),
            "mAP50": float(np.mean(ap50_all)),
            "mAP75": float(np.mean(ap75_all)),
            # convención de COCO/Cityscapes panoptic: AP solo sobre clases "thing"
            "mAP_things": float(np.mean([per_class[n]["AP"] for n in th])),
            "mAP50_things": float(np.mean([per_class[n]["AP50"] for n in th])),
            "PQ": float(np.mean(pq_all)),
            "PixelAccuracy": float(tp.sum() / max(total, 1)),
            "per_class": per_class,
            "confusion_matrix": self.cm.tolist(),
            "n_images": self.n_images,
        }

    def _ap_for_class(self, c) -> np.ndarray:
        """AP COCO (101 puntos) para cada umbral de IoU."""
        recs = self.ap_records[c]
        ngt = self.ap_ngt[c]
        out = np.zeros(len(IOU_THRESHOLDS))
        if ngt == 0 or not recs:
            return out
        recs = sorted(recs, key=lambda r: -r[0])
        tps = np.stack([r[1] for r in recs])  # (N, T)
        for ti in range(len(IOU_THRESHOLDS)):
            tp = np.cumsum(tps[:, ti])
            fp = np.cumsum(~tps[:, ti])
            recall = tp / ngt
            precision = tp / np.maximum(tp + fp, 1e-12)
            precision = np.maximum.accumulate(precision[::-1])[::-1]  # envolvente
            rs = np.linspace(0, 1, 101)
            idx = np.searchsorted(recall, rs, side="left")
            out[ti] = np.mean([precision[i] if i < len(precision) else 0 for i in idx])
        return out


class SegmentationEvaluator:
    """Evalúa con los dos protocolos a la vez:
      - principal (config.IGNORE_VOID): píxeles sin anotar ignorados ("void").
      - "strict": el Fondo cuenta como una clase más (todo píxel sin anotar que se
        prediga como otra clase es un error).
    compute() devuelve las métricas del protocolo principal y, en res["strict"],
    el resumen del otro (se reportan siempre ambos)."""

    def __init__(self, ignore_void=None):
        self.ignore_void = C.IGNORE_VOID if ignore_void is None else ignore_void
        self.main = _Evaluator(ignore_void=self.ignore_void)
        self.other = _Evaluator(ignore_void=not self.ignore_void)

    @property
    def n_images(self):
        return self.main.n_images

    def add(self, gt_sem, pred):
        self.main.add(gt_sem, pred)
        self.other.add(gt_sem, pred)

    def compute(self) -> dict:
        res = self.main.compute()
        oth = self.other.compute()
        keys = ["mIoU", "mF1", "mPrecision", "mRecall", "mAP", "mAP50", "mAP_things", "PQ"]
        res["protocol"] = "void" if self.ignore_void else "strict"
        res["alt_protocol"] = "strict" if self.ignore_void else "void"
        res["alt"] = {k: oth[k] for k in keys}
        res["alt"]["per_class_IoU"] = {k: v["IoU"] for k, v in oth["per_class"].items()}
        # matriz de confusión con la fila de Fondo (siempre la del protocolo estricto)
        res["confusion_matrix_strict"] = (oth if self.ignore_void else res)["confusion_matrix"]
        return res


def evaluate_single(gt_sem: np.ndarray, pred: dict) -> dict:
    """Métricas para una sola imagen (usado por la app web)."""
    ev = SegmentationEvaluator()
    ev.add(gt_sem, pred)
    return ev.compute()


def summary_row(res: dict) -> dict:
    """Fila compacta para la tabla comparativa del informe."""
    return {k: round(res[k], 4) for k in
            ["mIoU", "mAP", "mAP50", "mAP_things", "mAP50_things", "mF1", "mPrecision", "mRecall",
             "PQ", "PixelAccuracy"]}


def meets_targets(res: dict) -> dict:
    """Chequea los mínimos del taller: IoU ≥ 0.7, mAP ≥ 0.5, F1 ≥ 0.8."""
    return {"IoU>=0.7": bool(res["mIoU"] >= 0.7),
            "mAP>=0.5": bool(res["mAP"] >= 0.5),
            "mAP50>=0.5": bool(res["mAP50"] >= 0.5),
            "F1>=0.8": bool(res["mF1"] >= 0.8)}
