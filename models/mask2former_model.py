"""
models/mask2former_model.py
===========================
Mask2Former (Cheng et al., CVPR 2022) para segmentación panóptica.

Arquitectura (implementación de Hugging Face `transformers`):
  backbone Swin Transformer -> pixel decoder (MSDeformAttn, multiescala)
  -> transformer decoder con "masked attention" y N consultas (queries).
  Cada consulta predice (clase, máscara). La misma arquitectura sirve para
  semántica, instancias y panóptica: solo cambia cómo se arman los targets.

Se parte de pesos preentrenados en COCO-panoptic y se reemplaza la cabeza
de clasificación por nuestras 5 clases (fine-tuning / transfer learning).

Targets panópticos por imagen:
  - things (Área Construida, Zona Verde): 1 máscara por componente conexa.
  - stuff  (Fondo, Caminable, Carretera): 1 máscara por clase presente.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from data.dataset import connected_instances
from models.common import BasePredictor
from models.postprocess import finalize

ID2LABEL = {i: n for i, n in enumerate(C.CLASS_NAMES)}
LABEL2ID = {n: i for i, n in ID2LABEL.items()}


def build_mask2former(checkpoint: str = C.MASK2FORMER_DEFAULTS["checkpoint"],
                      pretrained: bool = True, config_dict: dict | None = None):
    """Crea el modelo. Con `config_dict` se construye offline (para cargar un .pt)."""
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    if config_dict is not None:
        cfg = Mask2FormerConfig.from_dict(config_dict)
        return Mask2FormerForUniversalSegmentation(cfg)
    if checkpoint == "debug-tiny":  # modelo diminuto sin descargas (solo pruebas de código)
        from transformers import SwinConfig
        bb = SwinConfig(embed_dim=32, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                        out_features=["stage1", "stage2", "stage3", "stage4"])
        cfg = Mask2FormerConfig(backbone_config=bb, id2label=ID2LABEL, label2id=LABEL2ID,
                                hidden_dim=64, num_queries=20, decoder_layers=2,
                                encoder_layers=1, feature_size=64, mask_feature_size=64,
                                dim_feedforward=128, num_attention_heads=4, use_pretrained_backbone=False)
        return Mask2FormerForUniversalSegmentation(cfg)
    if pretrained:
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True)  # la cabeza de clases cambia de 134 a 5
    cfg = Mask2FormerConfig.from_pretrained(checkpoint, id2label=ID2LABEL, label2id=LABEL2ID)
    return Mask2FormerForUniversalSegmentation(cfg)


def make_targets(sem: torch.Tensor, min_area: int = C.MIN_INSTANCE_AREA):
    """Mapa semántico (B,H,W) -> listas `mask_labels` y `class_labels` de HF."""
    mask_labels, class_labels = [], []
    for s in sem.cpu().numpy():
        masks, labels = [], []
        for cls in range(C.NUM_CLASSES):
            m = s == cls
            if m.sum() < min_area:
                continue
            if cls in C.THING_CLASSES:
                lab, ids = connected_instances(m, min_area)
                for i in ids:
                    masks.append(lab == i)
                    labels.append(cls)
            else:
                masks.append(m)
                labels.append(cls)
        if not masks:  # recorte sin nada válido: máscara vacía de fondo
            masks, labels = [np.zeros_like(s, bool)], [0]
        mask_labels.append(torch.from_numpy(np.stack(masks)).float())
        class_labels.append(torch.tensor(labels, dtype=torch.long))
    return mask_labels, class_labels


def param_groups(model, lr, backbone_mult, wd):
    """Tasa de aprendizaje menor para el backbone preentrenado."""
    bb, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if "pixel_level_module.encoder" in n else rest).append(p)
    return [{"params": bb, "lr": lr * backbone_mult, "weight_decay": wd},
            {"params": rest, "lr": lr, "weight_decay": wd}]


class Mask2FormerPredictor(BasePredictor):
    name = "Mask2Former"

    def __init__(self, model, device, infer_size=1024, query_threshold=0.5,
                 overlap_threshold=0.8):
        super().__init__(model, device, infer_size)
        self.qthr = query_threshold
        self.othr = overlap_threshold

    def _forward(self, x):
        out = self.model(pixel_values=x)
        cls = out.class_queries_logits[0].float().softmax(-1)[:, :-1]   # (Q, C) sin "no-objeto"
        masks = out.masks_queries_logits[0].float()                      # (Q, h/4, w/4)
        return cls, masks

    def _sem_probs(self, x):
        cls, masks = self._forward(x)
        masks = F.interpolate(masks[None], size=x.shape[-2:], mode="bilinear",
                              align_corners=False)[0].sigmoid()
        sem = torch.einsum("qc,qhw->chw", cls, masks)
        return sem / sem.sum(0, keepdim=True).clamp_min(1e-6)

    def _native(self, x, hw, orig):
        cls, masks_lr = self._forward(x)
        H0, W0 = orig
        # probabilidades semánticas (para rellenar huecos y para la app)
        m_full = F.interpolate(masks_lr[None], size=x.shape[-2:], mode="bilinear",
                               align_corners=False)[0][:, :hw[0], :hw[1]]
        sem = torch.einsum("qc,qhw->chw", cls, m_full.sigmoid())
        sem = sem / sem.sum(0, keepdim=True).clamp_min(1e-6)
        sem = F.interpolate(sem[None], size=orig, mode="bilinear", align_corners=False)[0]

        # inferencia panóptica por consultas (como en el paper)
        scores, labels = cls.max(-1)
        keep = scores > self.qthr
        pan = np.zeros(orig, np.int32)
        segments = []
        if keep.any():
            mk = F.interpolate(m_full[keep][None], size=orig, mode="bilinear",
                               align_corners=False)[0].sigmoid()           # (K,H0,W0)
            sc, lb = scores[keep], labels[keep]
            assign = (sc[:, None, None] * mk).argmax(0)
            stuff_id = {}
            nid = 1
            for k in torch.argsort(sc, descending=True).tolist():
                c = int(lb[k])
                orig_mask = mk[k] >= 0.5
                mask = (assign == k) & orig_mask
                a, a0 = int(mask.sum()), int(orig_mask.sum())
                if a == 0 or a0 == 0 or a / a0 < self.othr:
                    continue
                m_np = mask.cpu().numpy()
                if c in C.THING_CLASSES:
                    if a < C.MIN_INSTANCE_AREA:
                        continue
                    pan[m_np & (pan == 0)] = nid
                    segments.append({"id": nid, "category_id": c, "isthing": True,
                                     "score": float(sc[k] * mk[k][mask].mean())})
                    nid += 1
                else:  # stuff: se fusionan todas las consultas de la misma clase
                    if c not in stuff_id:
                        stuff_id[c] = nid
                        segments.append({"id": nid, "category_id": c, "isthing": False,
                                         "score": float(sc[k])})
                        nid += 1
                    pan[m_np & (pan == 0)] = stuff_id[c]
        return finalize(pan, segments, sem.cpu().numpy(), **self.region_kw("native"))


def save_mask2former(model, path, extra: dict | None = None, half=True):
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"arch": "mask2former", "config": model.config.to_dict(),
                "state_dict": sd, **(extra or {})}, path)


def load_mask2former(ckpt: dict, device, infer_size=None):
    model = build_mask2former(config_dict=ckpt["config"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in ckpt["state_dict"].items()})
    return Mask2FormerPredictor(model, device,
                                infer_size or ckpt.get("infer_size", C.MASK2FORMER_DEFAULTS["infer_size"]))
