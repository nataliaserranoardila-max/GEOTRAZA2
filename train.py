"""
train.py
========
Entrena Mask2Former o EfficientPS sobre el dataset de Cartografía Automatizada.

Pensado para Google Colab (GPU), pero corre en cualquier máquina con PyTorch.

Ejemplos:
    python train.py --model mask2former
    python train.py --model efficientps --epochs 150
    python train.py --model mask2former --resume            # continúa tras desconexión
    python train.py --model mask2former --ckpt-dir /content/drive/MyDrive/carto_ckpt

Salidas:
    models/weights/<modelo>_best.pt        mejores pesos (según validación)
    <ckpt-dir>/<modelo>_last.pt            checkpoint completo para reanudar
    outputs/history_<modelo>.json          pérdidas y métricas por época
    outputs/metrics_<modelo>_test_*.json   métricas finales en test
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config as C
from data.dataset import CartografiaDataset, list_samples, load_splits
from evaluate import evaluate_predictor, print_table


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return x, y, [b[2] for b in batch]


# --------------------------------------------------------------------------
# Adaptadores por modelo: construir, calcular pérdida, predictor y guardado
# --------------------------------------------------------------------------
class Mask2FormerAdapter:
    name = "mask2former"
    defaults = C.MASK2FORMER_DEFAULTS

    def __init__(self, args, device):
        from models import mask2former_model as M
        self.M = M
        self.model = M.build_mask2former(args.checkpoint or self.defaults["checkpoint"],
                                         pretrained=not args.no_pretrained).to(device)
        self.device = device
        self.args = args

    def param_groups(self, lr, wd):
        return self.M.param_groups(self.model, lr, self.args.backbone_lr_mult, wd)

    def loss(self, x, y):
        ml, cl = self.M.make_targets(y)
        out = self.model(pixel_values=x, mask_labels=[m.to(self.device) for m in ml],
                         class_labels=[c.to(self.device) for c in cl])
        return out.loss, {"loss": float(out.loss.detach())}

    def predictor(self):
        return self.M.Mask2FormerPredictor(self.model, self.device, self.args.infer_size)

    def save(self, path, extra):
        self.M.save_mask2former(self.model, path, extra)


class EfficientPSAdapter:
    name = "efficientps"
    defaults = C.EFFICIENTPS_DEFAULTS

    def __init__(self, args, device):
        from models import efficientps as E
        self.E = E
        self.model = E.EfficientPS(args.backbone or self.defaults["backbone"],
                                   pretrained=not args.no_pretrained).to(device)
        if args.class_weights:
            self.model.class_weights = [float(v) for v in args.class_weights.split(",")]
            print("Pesos por clase en la CE semántica:", dict(zip(C.CLASS_NAMES, self.model.class_weights)))
        self.device = device
        self.args = args

    def param_groups(self, lr, wd):
        bb = list(self.model.backbone.parameters())
        ids = {id(p) for p in bb}
        rest = [p for p in self.model.parameters() if id(p) not in ids]
        return [{"params": bb, "lr": lr * self.args.backbone_lr_mult, "weight_decay": wd},
                {"params": rest, "lr": lr, "weight_decay": wd}]

    def loss(self, x, y):
        t = self.E.make_instance_targets(y)
        t = [{k: v.to(self.device) for k, v in d.items()} for d in t]
        losses = self.model(x, y.to(self.device), t)
        # pesos de cada término (semántica con más peso: define IoU/F1)
        w = {"loss_sem_ce": 1.0, "loss_sem_dice": 1.0, "loss_objectness": 1.0,
             "loss_rpn_box_reg": 1.0, "loss_classifier": 1.0, "loss_box_reg": 1.0,
             "loss_mask": 1.0}
        total = sum(w.get(k, 1.0) * v for k, v in losses.items())
        logs = {k: float(v.detach()) for k, v in losses.items()}
        logs["loss"] = float(total.detach())
        return total, logs

    def predictor(self):
        return self.E.EfficientPSPredictor(self.model, self.device, self.args.infer_size)

    def save(self, path, extra):
        self.E.save_efficientps(self.model, path, extra)


ADAPTERS = {"mask2former": Mask2FormerAdapter, "efficientps": EfficientPSAdapter}


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
def lr_lambda_factory(total_iters, warmup_iters, power=0.9):
    def f(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        p = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return max((1 - p) ** power, 0.01)
    return f


@torch.no_grad()
def validation_loss(adapter, val_ds, device, amp):
    """Pérdida sobre las imágenes de validación completas (con padding a /32)."""
    adapter.model.train()  # las cabezas solo calculan pérdida en modo train
    vals = []
    for i in range(len(val_ds)):
        x, y, _ = val_ds[i]
        H, W = y.shape
        Hp, Wp = math.ceil(H / 32) * 32, math.ceil(W / 32) * 32
        x = F.pad(x, (0, Wp - W, 0, Hp - H))[None].to(device)
        y = F.pad(y, (0, Wp - W, 0, Hp - H), value=255)[None]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss, _ = adapter.loss(x, y)
        if torch.isfinite(loss):
            vals.append(float(loss))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--data-dir", default=str(C.SEG_DIR))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--crop", type=int)
    ap.add_argument("--crops-per-image", type=int)
    ap.add_argument("--infer-size", type=int)
    ap.add_argument("--checkpoint", default=None, help="checkpoint HF de Mask2Former")
    ap.add_argument("--backbone", default=None, help="backbone timm de EfficientPS")
    ap.add_argument("--class-weights", default="1,2,1,1,1",
                    help="EfficientPS: pesos de la CE por clase (Fondo,Caminable,Construida,Verde,Carretera)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--warmup", type=float, default=0.05, help="fracción de iteraciones de warmup")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=1, help="acumulación de gradientes")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-dir", default=str(C.WEIGHTS_DIR))
    ap.add_argument("--limit", type=int, default=0, help="usar solo N imágenes (prueba rápida)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    Adapter = ADAPTERS[args.model]
    d = Adapter.defaults
    for k in ["epochs", "batch_size", "lr", "weight_decay", "crop", "crops_per_image", "infer_size"]:
        if getattr(args, k) is None:
            setattr(args, k, d["crop_size"] if k == "crop" else d[k])

    seed_all(args.seed)
    from models.predictor import get_device
    device = get_device(args.device)
    amp = (device.type == "cuda") and not args.no_amp
    print(f"Dispositivo: {device} | AMP: {amp}")

    # ---------------- datos ----------------
    samples = list_samples(args.data_dir)
    assert samples, f"No se encontraron imágenes en {args.data_dir}"
    splits = load_splits(samples)
    tr_ids, va_ids, te_ids = splits["train"], splits["val"], splits["test"]
    if args.limit:
        tr_ids, va_ids, te_ids = tr_ids[:args.limit], va_ids[:max(1, args.limit // 4)], te_ids[:max(1, args.limit // 4)]
    print(f"Imágenes -> train {len(tr_ids)} | val {len(va_ids)} | test {len(te_ids)}")
    train_ds = CartografiaDataset(tr_ids, args.data_dir, train=True, crop_size=args.crop,
                                  crops_per_image=args.crops_per_image, max_side=args.infer_size,
                                  seed=args.seed)
    val_ds = CartografiaDataset(va_ids, args.data_dir, train=False, max_side=args.infer_size)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda",
                    persistent_workers=args.workers > 0)

    # ---------------- modelo ----------------
    adapter = Adapter(args, device)
    n_params = sum(p.numel() for p in adapter.model.parameters()) / 1e6
    print(f"{args.model}: {n_params:.1f} M parámetros")
    opt = torch.optim.AdamW(adapter.param_groups(args.lr, args.weight_decay))
    total_iters = args.epochs * max(len(dl) // args.accum, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_iters, int(args.warmup * total_iters)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True); C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{args.model}_last.pt"
    best_path = C.WEIGHTS_DIR / f"{args.model}_best.pt"
    best_drive = ckpt_dir / f"{args.model}_best.pt"
    hist_path = C.OUTPUTS_DIR / f"history_{args.model}.json"

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": [], "time_s": [],
               "eval_epoch": [], "val_mIoU": [], "val_mF1": [], "val_mAP": [], "val_PQ": [],
               "train_components": []}
    start_epoch, best_score = 0, -1.0
    import shutil
    if args.resume and best_drive.exists() and not best_path.exists():
        shutil.copy(best_drive, best_path)          # sesión nueva de Colab: recuperar desde Drive
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        adapter.model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, start_epoch, best_score = ck["history"], ck["epoch"] + 1, ck["best_score"]
        print(f"Reanudando desde la época {start_epoch} (mejor score {best_score:.4f})")
        C.write_json(hist_path, history)

    # ---------------- bucle ----------------
    for epoch in range(start_epoch, args.epochs):
        adapter.model.train()
        t0 = time.time()
        losses, comps = [], {}
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl, desc=f"Época {epoch + 1}/{args.epochs}", leave=False)
        for it, (x, y, _) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, logs = adapter.loss(x, y)
            if not torch.isfinite(loss):
                print("  pérdida no finita, se omite el lote"); opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(logs["loss"])
            for k, v in logs.items():
                comps.setdefault(k, []).append(v)
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        va_loss = validation_loss(adapter, val_ds, device, amp)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["lr"].append(opt.param_groups[-1]["lr"])
        history["time_s"].append(time.time() - t0)
        history["train_components"].append({k: float(np.mean(v)) for k, v in comps.items()})
        msg = f"Época {epoch + 1:3d} | train {tr_loss:.4f} | val {va_loss:.4f} | {time.time() - t0:.0f}s"

        # ---- métricas de validación y mejor modelo ----
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            res = evaluate_predictor(adapter.predictor(), va_ids, args.data_dir, "native", progress=False)
            score = (res["mIoU"] + res["mF1"] + res["mAP"]) / 3
            history["eval_epoch"].append(epoch + 1)
            for k in ["mIoU", "mF1", "mAP", "PQ"]:
                history[f"val_{k}"].append(res[k])
            msg += f" | val mIoU {res['mIoU']:.4f} mF1 {res['mF1']:.4f} mAP {res['mAP']:.4f} PQ {res['PQ']:.4f}"
            if score > best_score:
                best_score = score
                extra = {"epoch": epoch + 1, "val_metrics": {k: res[k] for k in
                         ["mIoU", "mF1", "mAP", "mAP50", "mPrecision", "PQ"]},
                         "infer_size": args.infer_size, "args": vars(args)}
                adapter.save(best_path, extra)
                if best_drive != best_path:
                    adapter.save(best_drive, extra)
                msg += "  <- mejor"
        print(msg)

        C.write_json(hist_path, history)
        C.write_json(ckpt_dir / hist_path.name, history)   # copia en Drive
        torch.save({"model": adapter.model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "epoch": epoch, "best_score": best_score}, last_path)

    # ---------------- evaluación final en test ----------------
    from models.predictor import load_predictor
    pred = load_predictor(best_path, device)
    for mode in ["native", "tta"]:
        res = evaluate_predictor(pred, te_ids, args.data_dir, mode)
        print_table(res)
        for d in (C.OUTPUTS_DIR, ckpt_dir):
            C.write_json(d / f"metrics_{args.model}_test_{mode}.json", res)

    from utils.plots import plot_history
    plot_history(history, args.model, C.OUTPUTS_DIR / f"curvas_{args.model}.png")
    print("Listo. Mejores pesos en", best_path)


if __name__ == "__main__":
    main()
