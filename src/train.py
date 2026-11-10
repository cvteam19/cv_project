# src/train.py
"""
Train Faster R-CNN (ResNet-50 + FPN) baseline for fire/smoke detection.
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

# Torchvision model
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ---------- Dataset ----------
class FireSmokeDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_root: Path, split: str, flip_p: float = 0.0):
        self.root = Path(data_root)
        self.split = split
        self.flip_p = float(flip_p)
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing {self.images_dir}")
        self.image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        )
        self._to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(1 if int(cls) == 0 else 2)

        img_t = self._to_tensor(img)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.flip_p > 0.0 and torch.rand(1).item() < self.flip_p:
            img_t = torch.flip(img_t, dims=[-1])
            if boxes_t.numel() > 0:
                x1_old = boxes_t[:, 0].clone()
                x2_old = boxes_t[:, 2].clone()
                boxes_t[:, 0] = w - x2_old
                boxes_t[:, 2] = w - x1_old

        area = (
            (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------- Model ----------
def build_faster_rcnn(num_classes: int = 3, input_size: int = 640, pretrained: bool = True):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform.min_size = (input_size,)
    model.transform.max_size = input_size
    return model


# ---------- Train / eval loops ----------
def train_one_epoch(model, optimizer, loader, device, epoch, grad_clip=10.0):
    model.train()
    sums = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0,
            "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        sums["loss"] += loss.item()
        for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            if k in loss_dict:
                sums[k] += loss_dict[k].item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device):
    was_training = model.training
    model.train()
    sums = {"loss": 0.0}
    n = 0
    for images, targets in tqdm(loader, desc="Val loss", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        sums["loss"] += sum(v for v in loss_dict.values()).item()
        n += 1
    if not was_training:
        model.eval()
    return {"loss": sums["loss"] / max(n, 1)}


@torch.no_grad()
def evaluate_map(model, loader, device):
    was_training = model.training
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False).to(device)
    for images, targets in tqdm(loader, desc="Val mAP", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        preds_cpu = [
            {"boxes": p["boxes"].detach().cpu(),
             "scores": p["scores"].detach().cpu(),
             "labels": p["labels"].detach().cpu()} for p in preds
        ]
        tgt_cpu = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
        metric.update(preds_cpu, tgt_cpu)
    res = metric.compute()
    if was_training:
        model.train()
    return {"mAP": float(res["map"]), "mAP_50": float(res["map_50"])}


# ---------- Checkpointing ----------
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_metric, history, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "history": history,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    best_metric = ckpt.get("best_metric", float("-inf"))
    print(f"Resumed from {path} (epoch {ckpt['epoch']}, best_mAP={best_metric:.4f})")
    return ckpt["epoch"], best_metric, ckpt.get("history", [])


# ---------- Main ----------
def main(args):
    # seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Platform:", platform.platform())
    print("Python  :", sys.version.split()[0])
    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.version.cuda)
    print("Device  :", device)

    # --- helper to limit batches (smoke tests) ---
    from itertools import islice
    def _limit(loader, n):
        return loader if n is None else islice(loader, n)

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # argparse overrides config
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    lr = args.lr if args.lr is not None else cfg["training"]["lr"]

    data_root = Path(args.data_root)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDATA_ROOT = {data_root}")
    print(f"CKPT_DIR  = {ckpt_dir}")

    # dataset sanity
    for split in ("train", "val", "test"):
        img_dir = data_root / split / "images"
        if img_dir.is_dir():
            n_imgs = len([p for p in img_dir.iterdir() if p.suffix.lower() in FireSmokeDataset.IMG_EXTS])
            print(f"{split:>5}: {n_imgs} images")
        else:
            print(f"{split:>5}: DIR NOT FOUND ({img_dir})")

    # datasets
    train_ds = FireSmokeDataset(data_root, "train",
                                flip_p=cfg["data"]["augmentation"]["horizontal_flip_p"])
    val_ds = FireSmokeDataset(data_root, "val", flip_p=0.0)
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=cfg["data"]["num_workers"],
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=cfg["data"]["num_workers"],
                            collate_fn=collate_fn, pin_memory=True)

    # model
    model = build_faster_rcnn(
        cfg["model"]["num_classes"], cfg["model"]["input_size"], pretrained=True
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.2f} M")

    # optimizer / scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr,
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_ckpt = ckpt_dir / "faster_rcnn_best.pth"
    last_ckpt = ckpt_dir / "faster_rcnn_last.pth"
    history_path = ckpt_dir / "training_history.json"

    start_epoch = 1
    best_map = -1.0
    history = []
    if args.resume and last_ckpt.exists():
        start_epoch, best_map, history = load_checkpoint(last_ckpt, model, optimizer, scheduler, device)
        start_epoch += 1

    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, optimizer, _limit(train_loader, args.max_train_batches),
            device, epoch, grad_clip=cfg["training"]["grad_clip_max_norm"],
        )
        val_metrics = evaluate_loss(model, _limit(val_loader, args.max_val_batches), device)
        map_metrics = evaluate_map(model, _limit(val_loader, args.max_val_batches), device)
        scheduler.step()

        log = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_mAP_0.5": round(map_metrics["mAP_50"], 4),
            "val_mAP_0.5:0.95": round(map_metrics["mAP"], 4),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(time.time() - train_start, 1),
            "epoch_time_sec": round(time.time() - t0, 1),
        }
        history.append(log)
        print(json.dumps(log))

        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_map, history, cfg)
        if map_metrics["mAP"] > best_map:
            best_map = map_metrics["mAP"]
            epochs_no_improve = 0
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_map, history, cfg)
            print(f"  -> new best val mAP@0.5:0.95={best_map:.4f}, saved {best_ckpt.name}")
        else:
            epochs_no_improve += 1
            print(f"  -> no improvement ({epochs_no_improve}/{cfg['training']['early_stopping_patience']})")

        history_path.write_text(json.dumps(history, indent=2))

        if epochs_no_improve >= cfg["training"]["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nTraining finished. Best val mAP@0.5:0.95: {best_map:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"History file   : {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/data", help="Root containing train/val/test splits")
    parser.add_argument("--config", default="configs/baseline_config.yaml")
    parser.add_argument("--ckpt-dir", default="checkpoints/faster_rcnn")
    parser.add_argument("--epochs", type=int, default=None, help="Override config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config")
    parser.add_argument("--lr", type=float, default=None, help="Override config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if exists")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Limit train batches per epoch (smoke test)")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Limit val batches per epoch (smoke test)")
    args = parser.parse_args()
    main(args)