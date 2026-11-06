"""
models/efficientps.py
=====================
EfficientPS (Mohan & Valada, IJCV 2021) implementado en PyTorch puro.

Componentes (siguiendo el paper):
  1. Backbone EfficientNet (timm, preentrenado en ImageNet) -> C2..C5
     (stride 4, 8, 16, 32).
  2. FPN bidireccional ("2-way FPN"): un camino top-down y otro bottom-up
     que se suman en cada escala -> P4, P8, P16, P32 (256 canales).
  3. Cabeza semántica:
        - LSFE (Large Scale Feature Extractor) en P4 y P8
        - DPC  (Dense Prediction Cells) en P16 y P32
        - MC   (Mismatch Correction) para unir escalas gruesas con finas
        - concatenación a 1/4, conv 1x1 a clases y upsampling x4
  4. Cabeza de instancias tipo Mask R-CNN (RPN + RoIAlign + cajas + máscaras),
     usando los componentes de torchvision sobre las salidas del 2-way FPN.
  5. Fusión panóptica: combina los logits de máscara de cada instancia con
     los logits semánticos de su clase dentro de su caja, y resuelve
     solapes por puntaje; el resto de píxeles viene de la cabeza semántica.

Diferencias con el paper (para entrenar en Colab con un dataset pequeño):
  - GroupNorm en lugar de iABN sincronizado (batch pequeño).
  - BatchNorm del backbone congelado.
  - Pérdida semántica = CE con "bootstrapping" (25% de píxeles más difíciles,
    equivalente a la pérdida ponderada por píxel del paper) + Dice.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, paste_masks_in_image
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign

import config as C
from data.dataset import connected_instances
from models.common import BasePredictor
from models.postprocess import finalize

# things del dataset -> etiquetas de detección (0 = fondo de Mask R-CNN)
THING_TO_DET = {c: i + 1 for i, c in enumerate(C.THING_CLASSES)}
DET_TO_THING = {v: k for k, v in THING_TO_DET.items()}


def gn(ch):
    return nn.GroupNorm(min(32, ch // 4), ch)


class SepConv(nn.Sequential):
    """Convolución separable en profundidad 3x3 + GN + LeakyReLU (bloque base del paper)."""

    def __init__(self, cin, cout, dilation=(1, 1)):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=dilation, dilation=dilation, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False), gn(cout), nn.LeakyReLU(0.01, inplace=True))


class ConvGN(nn.Sequential):
    def __init__(self, cin, cout, k=1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), gn(cout),
                         nn.LeakyReLU(0.01, inplace=True))


# --------------------------------------------------------------------------
# 2-way FPN
# --------------------------------------------------------------------------
class TwoWayFPN(nn.Module):
    def __init__(self, in_chs, out_ch=256):
        super().__init__()
        self.lat_td = nn.ModuleList([ConvGN(c, out_ch) for c in in_chs])
        self.lat_bu = nn.ModuleList([ConvGN(c, out_ch) for c in in_chs])
        self.down = nn.ModuleList([SepConv(out_ch, out_ch) for _ in in_chs[:-1]])
        self.out = nn.ModuleList([SepConv(out_ch, out_ch) for _ in in_chs])

    def forward(self, feats):
        n = len(feats)
        td = [None] * n
        td[-1] = self.lat_td[-1](feats[-1])
        for i in range(n - 2, -1, -1):  # top-down
            td[i] = self.lat_td[i](feats[i]) + F.interpolate(
                td[i + 1], size=feats[i].shape[-2:], mode="nearest")
        bu = [None] * n
        bu[0] = self.lat_bu[0](feats[0])
        for i in range(1, n):            # bottom-up
            d = F.max_pool2d(self.down[i - 1](bu[i - 1]), 2, ceil_mode=True)
            bu[i] = self.lat_bu[i](feats[i]) + d[..., :feats[i].shape[-2], :feats[i].shape[-1]]
        return [self.out[i](td[i] + bu[i]) for i in range(n)]


# --------------------------------------------------------------------------
# Cabeza semántica
# --------------------------------------------------------------------------
class LSFE(nn.Sequential):
    def __init__(self, cin=256, cout=128):
        super().__init__(SepConv(cin, cout), SepConv(cout, cout))


class DPC(nn.Module):
    """Dense Prediction Cell con las tasas de dilatación usadas en EfficientPS."""

    def __init__(self, cin=256, cout=128, mid=256):
        super().__init__()
        self.c0 = SepConv(cin, mid, (1, 6))
        self.c1 = SepConv(mid, mid, (1, 1))
        self.c2 = SepConv(mid, mid, (6, 21))
        self.c3 = SepConv(mid, mid, (18, 15))
        self.c4 = SepConv(mid, mid, (6, 3))
        self.proj = ConvGN(mid * 5, cout)

    def forward(self, x):
        x0 = self.c0(x)
        x3 = self.c3(x0)
        return self.proj(torch.cat([x0, self.c1(x0), self.c2(x0), x3, self.c4(x3)], 1))


class MC(nn.Sequential):
    """Mismatch Correction: dos convs separables + upsampling x2."""

    def __init__(self, ch=128):
        super().__init__(SepConv(ch, ch), SepConv(ch, ch))

    def forward(self, x, size):
        return F.interpolate(super().forward(x), size=size, mode="bilinear", align_corners=False)


class SemanticHead(nn.Module):
    def __init__(self, num_classes, fpn_ch=256, ch=128):
        super().__init__()
        self.lsfe4, self.lsfe8 = LSFE(fpn_ch, ch), LSFE(fpn_ch, ch)
        self.dpc16, self.dpc32 = DPC(fpn_ch, ch), DPC(fpn_ch, ch)
        self.mc32, self.mc16, self.mc8 = MC(ch), MC(ch), MC(ch)
        self.cls = nn.Conv2d(ch * 4, num_classes, 1)

    def forward(self, P):
        p4, p8, p16, p32 = P
        s32 = self.dpc32(p32)
        s16 = self.dpc16(p16) + self.mc32(s32, p16.shape[-2:])
        s8 = self.lsfe8(p8) + self.mc16(s16, p8.shape[-2:])
        s4 = self.lsfe4(p4) + self.mc8(s8, p4.shape[-2:])
        size = p4.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return self.cls(torch.cat([s4, up(s8), up(s16), up(s32)], 1))


# --------------------------------------------------------------------------
# Modelo completo
# --------------------------------------------------------------------------
class EfficientPS(nn.Module):
    def __init__(self, backbone=C.EFFICIENTPS_DEFAULTS["backbone"], pretrained=True,
                 num_classes=C.NUM_CLASSES, n_things=len(C.THING_CLASSES)):
        super().__init__()
        import timm
        self.backbone_name = backbone
        self.class_weights = None  # se fija desde train.py (--class-weights)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(1, 2, 3, 4))
        chs = self.backbone.feature_info.channels()
        self.fpn = TwoWayFPN(chs, 256)
        self.sem_head = SemanticHead(num_classes)

        # ---- cabeza de instancias (Mask R-CNN) ----
        anchors = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                  aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 3), fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=1000, testing=500), nms_thresh=0.7)
        names = ["0", "1", "2", "3"]
        self.roi_heads = RoIHeads(
            box_roi_pool=MultiScaleRoIAlign(names, 7, 2),
            box_head=TwoMLPHead(256 * 7 * 7, 1024),
            box_predictor=FastRCNNPredictor(1024, n_things + 1),
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512,
            positive_fraction=0.25, bbox_reg_weights=None, score_thresh=0.05,
            nms_thresh=0.5, detections_per_img=100,
            mask_roi_pool=MultiScaleRoIAlign(names, 14, 2),
            mask_head=MaskRCNNHeads(256, (256, 256, 256, 256), 1),
            mask_predictor=MaskRCNNPredictor(256, 256, n_things + 1))

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():  # BN del backbone congelado
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def features(self, x):
        P = self.fpn(self.backbone(x))
        feats = OrderedDict([("0", P[0]), ("1", P[1]), ("2", P[2]), ("3", P[3]),
                             ("pool", F.max_pool2d(P[3], 1, 2))])
        return P, feats

    def semantic_logits(self, x):
        P, _ = self.features(x)
        return F.interpolate(self.sem_head(P), size=x.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x, sem_target=None, inst_targets=None):
        P, feats = self.features(x)
        sem = F.interpolate(self.sem_head(P), size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        il = ImageList(x, [tuple(x.shape[-2:])] * x.shape[0])
        if self.training:
            props, rpn_l = self.rpn(il, feats, inst_targets)
            _, roi_l = self.roi_heads(feats, props, il.image_sizes, inst_targets)
            losses = {**rpn_l, **roi_l}
            losses.update(semantic_loss(sem, sem_target, class_weights=self.class_weights))
            return losses
        props, _ = self.rpn(il, feats)
        dets, _ = self.roi_heads(feats, props, il.image_sizes)
        return sem, dets


# --------------------------------------------------------------------------
# Pérdidas y targets
# --------------------------------------------------------------------------
def semantic_loss(logits, target, top_k=0.25, ignore=255, class_weights=None):
    """CE con bootstrapping (píxeles más difíciles) + Dice multiclase.
    class_weights permite dar más peso a clases difíciles (Área Caminable)."""
    w = None if class_weights is None else torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
    ce = F.cross_entropy(logits, target, weight=w, ignore_index=ignore, reduction="none").flatten()
    valid = (target != ignore).flatten()
    ce = ce[valid]
    k = max(int(top_k * ce.numel()), 1)
    ce_loss = ce.topk(k).values.mean() if ce.numel() else logits.sum() * 0
    prob = logits.softmax(1)
    t = target.clone()
    t[t == ignore] = 0
    oh = F.one_hot(t, logits.shape[1]).permute(0, 3, 1, 2).float()
    m = (target != ignore).unsqueeze(1).float()
    inter = (prob * oh * m).sum((0, 2, 3))
    den = (prob * m).sum((0, 2, 3)) + (oh * m).sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return {"loss_sem_ce": ce_loss, "loss_sem_dice": dice[C.EVAL_CLASSES].mean()}


def make_instance_targets(sem: torch.Tensor, min_area=C.MIN_INSTANCE_AREA):
    """Mapa semántico (B,H,W) -> lista de dicts {boxes, labels, masks} para Mask R-CNN."""
    out = []
    for s in sem.cpu().numpy():
        boxes, labels, masks = [], [], []
        for cls in C.THING_CLASSES:
            lab, ids = connected_instances(s == cls, min_area)
            for i in ids:
                m = lab == i
                ys, xs = np.where(m)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(THING_TO_DET[cls])
                masks.append(m)
        h, w = s.shape
        out.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8) if masks
            else torch.zeros((0, h, w), dtype=torch.uint8)})
    return out


# --------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------
class EfficientPSPredictor(BasePredictor):
    name = "EfficientPS"

    def __init__(self, model, device, infer_size=1024, det_threshold=0.5):
        super().__init__(model, device, infer_size)
        self.dthr = det_threshold

    def _sem_probs(self, x):
        return self.model.semantic_logits(x)[0].float().softmax(0)

    def _native(self, x, hw, orig):
        sem_logits, dets = self.model(x)
        h, w = hw
        sem = F.interpolate(sem_logits[:, :, :h, :w].float(), size=orig, mode="bilinear",
                            align_corners=False)[0]
        probs = sem.softmax(0)
        d = dets[0]
        keep = d["scores"] > self.dthr
        boxes, labels, scores = d["boxes"][keep], d["labels"][keep], d["scores"][keep]
        pan = np.zeros(orig, np.int32)
        segments = []
        if len(boxes):
            # pasar máscaras 28x28 al tamaño original
            sy, sx = orig[0] / h, orig[1] / w
            b = boxes.clone()
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy
            mprob = paste_masks_in_image(d["masks"][keep].float(), b, orig)[:, 0]  # (N,H0,W0)
            taken = torch.zeros(orig, dtype=torch.bool, device=mprob.device)
            nid = 1
            for k in torch.argsort(scores, descending=True).tolist():
                cls = DET_TO_THING[int(labels[k])]
                x1, y1, x2, y2 = b[k].round().int().tolist()
                inbox = torch.zeros(orig, dtype=torch.bool, device=mprob.device)
                inbox[max(y1, 0):y2, max(x1, 0):x2] = True
                # fusión panóptica: promedio de prob. de máscara y prob. semántica
                fused = 0.5 * (mprob[k] + probs[cls] * inbox)
                m = (fused > 0.5) & ~taken
                full = int((fused > 0.5).sum())
                if full == 0 or int(m.sum()) < max(C.MIN_INSTANCE_AREA, 0.5 * full):
                    continue
                taken |= m
                pan[m.cpu().numpy()] = nid
                segments.append({"id": nid, "category_id": cls, "isthing": True,
                                 "score": float(scores[k])})
                nid += 1
        return finalize(pan, segments, probs.cpu().numpy(), **self.region_kw("native"))


def save_efficientps(model, path, extra=None, half=True):
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"arch": "efficientps", "backbone": model.backbone_name, "state_dict": sd,
                **(extra or {})}, path)


def load_efficientps(ckpt, device, infer_size=None):
    model = EfficientPS(ckpt["backbone"], pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in ckpt["state_dict"].items()})
    return EfficientPSPredictor(model, device,
                                infer_size or ckpt.get("infer_size", C.EFFICIENTPS_DEFAULTS["infer_size"]))
