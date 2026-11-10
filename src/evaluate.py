"""
Unified evaluation script for Faster R-CNN and YOLOv8s baselines.
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms as T

# 0=bg, 1=smoke, 2=fire (torchvision convention inside the script)
CLASS_NAMES = {1: "smoke", 2: "fire"}


class FireSmokeDataset(torch.utils.data.Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, images_dir, labels_dir, transforms=None, return_pil=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.return_pil = return_pil
        self.files = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = [], []
        lbl = self.labels_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x1 = max(0.0, (xc - bw / 2) * w)
                y1 = max(0.0, (yc - bh / 2) * h)
                x2 = min(w, (xc + bw / 2) * w)
                y2 = min(h, (yc + bh / 2) * h)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(1 if int(cls) == 0 else 2)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if not self.return_pil and self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    if boxes.numel() else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return img, target, str(img_path)


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_map(preds, targets):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    p_remap = [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"] - 1} for p in preds]
    t_remap = [{"boxes": t["boxes"], "labels": t["labels"] - 1} for t in targets]
    metric.update(p_remap, t_remap)
    return metric.compute()


def per_class_prf(preds, targets, iou_thresh, score_thresh):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p, t in zip(preds, targets):
        for cls in [1, 2]:
            pm = (p["labels"] == cls) & (p["scores"] >= score_thresh)
            pb, tb = p["boxes"][pm], t["boxes"][t["labels"] == cls]
            if len(tb) == 0:
                stats[cls]["fp"] += len(pb)
                continue
            if len(pb) == 0:
                stats[cls]["fn"] += len(tb)
                continue
            ious, matched = box_iou(pb, tb), set()
            for i in range(len(pb)):
                bi, bj = 0.0, -1
                for j in range(len(tb)):
                    if j in matched:
                        continue
                    v = ious[i, j].item()
                    if v > bi:
                        bi, bj = v, j
                if bi >= iou_thresh and bj >= 0:
                    stats[cls]["tp"] += 1
                    matched.add(bj)
                else:
                    stats[cls]["fp"] += 1
            stats[cls]["fn"] += len(tb) - len(matched)

    out = {}
    for cls, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[CLASS_NAMES[cls]] = {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
    return out


def confusion_matrix_counts(preds, targets, iou_thresh, score_thresh):
    cm = np.zeros((3, 3), dtype=int)
    for p, t in zip(preds, targets):
        matched = set()
        for i in range(len(p["boxes"])):
            if p["scores"][i] < score_thresh:
                continue
            pc = int(p["labels"][i])
            bi, bj = 0.0, -1
            for j in range(len(t["boxes"])):
                if j in matched:
                    continue
                v = box_iou(p["boxes"][i:i+1], t["boxes"][j:j+1]).item()
                if v > bi:
                    bi, bj = v, j
            if bi >= iou_thresh and bj >= 0:
                cm[int(t["labels"][bj]), pc] += 1
                matched.add(bj)
            else:
                cm[0, pc] += 1
        for j in range(len(t["boxes"])):
            if j not in matched:
                cm[int(t["labels"][j]), 0] += 1
    return cm


def draw_predictions(img_path, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
                     score_thresh=0.30, out_path=None):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for box, lab in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1 + 3, y1 + 3), f"GT {CLASS_NAMES.get(int(lab), '?')}", fill="green", font=font)
    for box, sc, lab in zip(pred_boxes, pred_scores, pred_labels):
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 3, max(0, y1 - 20)),
                  f"{CLASS_NAMES.get(int(lab), '?')} {sc:.2f}", fill="red", font=font)
    if out_path:
        img.save(out_path, quality=90)
    return img


def find_failure_cases(records, score_thresh, iou_thresh=0.5):
    missed_fire, cloud_smoke_fp, bright_fire_fp = [], [], []
    for i, rec in enumerate(records):
        gt_b, gt_l = rec["gt_boxes"], rec["gt_labels"]
        keep = rec["scores"] >= score_thresh
        pb, ps, pl = rec["boxes"][keep], rec["scores"][keep], rec["labels"][keep]

        if len(gt_b) > 0:
            fire_gt = gt_b[gt_l == 2]
            if len(fire_gt) > 0:
                matched = 0
                for fb in fire_gt:
                    if len(pb) > 0:
                        ious = box_iou(torch.tensor(pb), torch.tensor(fb[None])).numpy().flatten()
                        if ious.max() >= iou_thresh:
                            matched += 1
                if matched == 0:
                    missed_fire.append(i)

        if len(gt_b) == 0 and len(pb) > 0:
            for k, (sc, lab) in enumerate(zip(ps, pl)):
                if lab == 1:
                    cloud_smoke_fp.append((i, k, float(sc)))
                elif lab == 2:
                    bright_fire_fp.append((i, k, float(sc)))

    cloud_smoke_fp.sort(key=lambda x: -x[2])
    bright_fire_fp.sort(key=lambda x: -x[2])
    return {
        "missed_fire": missed_fire[:20],
        "cloud_smoke_fp": cloud_smoke_fp[:20],
        "bright_fire_fp": bright_fire_fp[:20],
    }


@torch.no_grad()
def run_inference_faster_rcnn(model, loader, device, score_thresh):
    preds, targets, records = [], [], []
    n_img = 0
    t0 = time.time()
    for images, tgts, paths in tqdm(loader, desc="Inference (Faster R-CNN)"):
        images = [im.to(device, non_blocking=True) for im in images]
        outputs = model(images)
        for out, tgt, p in zip(outputs, tgts, paths):
            keep = out["scores"] >= score_thresh
            preds.append({"boxes": out["boxes"][keep].cpu(),
                          "scores": out["scores"][keep].cpu(),
                          "labels": out["labels"][keep].cpu()})
            targets.append({"boxes": tgt["boxes"].cpu(), "labels": tgt["labels"].cpu()})
            records.append({
                "path": p,
                "boxes": out["boxes"][keep].cpu().numpy(),
                "scores": out["scores"][keep].cpu().numpy(),
                "labels": out["labels"][keep].cpu().numpy(),
                "gt_boxes": tgt["boxes"].cpu().numpy(),
                "gt_labels": tgt["labels"].cpu().numpy(),
            })
        n_img += len(images)
    return preds, targets, records, n_img / max(time.time() - t0, 1e-6)


@torch.no_grad()
def run_inference_yolo(core, loader, device, score_floor, nms_iou, max_det, protocol_dir):
    sys.path.insert(0, protocol_dir)
    from final_yolo_protocol import predict_with_baseline_input

    preds, targets, records = [], [], []
    n_img = 0
    t0 = time.time()
    for images, tgts, paths in tqdm(loader, desc="Inference (YOLO)"):
        out = predict_with_baseline_input(core, list(images), device=device,
                                          score_floor=score_floor,
                                          nms_iou=nms_iou, max_det=max_det)
        for o, tgt, p in zip(out, tgts, paths):
            boxes = o["boxes"].cpu()
            scores = o["scores"].cpu()
            labels = (o["labels"].cpu() + 1).long()
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
            targets.append({"boxes": tgt["boxes"].cpu(), "labels": tgt["labels"].cpu()})
            records.append({
                "path": p,
                "boxes": boxes.numpy(),
                "scores": scores.numpy(),
                "labels": labels.numpy(),
                "gt_boxes": tgt["boxes"].numpy(),
                "gt_labels": tgt["labels"].numpy(),
            })
        n_img += len(images)
    return preds, targets, records, n_img / max(time.time() - t0, 1e-6)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_images_dir = Path(args.data_root) / args.split / "images"
    test_labels_dir = Path(args.data_root) / args.split / "labels"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_yolo = args.model_type == "yolo"

    test_ds = FireSmokeDataset(
        test_images_dir, test_labels_dir,
        transforms=None if is_yolo else T.ToTensor(),
        return_pil=is_yolo,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Test dataset: {len(test_ds)} images")

    # --- Load model ---
    model_name = "Unknown"
    ckpt_epoch = "N/A"

    if args.model_type == "faster_rcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        model = fasterrcnn_resnet50_fpn(weights=None)
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, args.num_classes)
        model.transform.min_size = (args.input_size,)
        model.transform.max_size = args.input_size

        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        ckpt_epoch = ckpt.get("epoch", "?")
        model_name = "Faster R-CNN ResNet-50 FPN"

        preds, targets, records, fps = run_inference_faster_rcnn(
            model, test_loader, device, args.score_low
        )

    elif args.model_type == "yolo":
        from ultralytics import YOLO
        yolo_wrapper = YOLO(args.ckpt)
        core = yolo_wrapper.model.to(device).float().eval()
        model_name = "YOLOv8s (baseline-input adapter)"
        ckpt_epoch = args.yolo_ckpt_epoch

        preds, targets, records, fps = run_inference_yolo(
            core, test_loader, device,
            score_floor=args.score_low,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
            protocol_dir=args.protocol_dir,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Inference FPS: {fps:.2f}")

    # --- Metrics ---
    m = compute_map(preds, targets)
    mAP50 = float(m["map_50"])
    mAP5095 = float(m["map"])
    per_cls = [float(x) for x in m["map_per_class"]]
    print(f"\nmAP@0.5      = {mAP50:.4f}")
    print(f"mAP@0.5:0.95 = {mAP5095:.4f}")

    prf = per_class_prf(preds, targets, args.iou_thresh, args.score_prf)
    print("\nPer-class P/R/F1:")
    print(json.dumps(prf, indent=2))

    cm = confusion_matrix_counts(preds, targets, args.iou_thresh, args.score_prf)
    print("\nConfusion matrix:")
    print(cm)

    # --- Threshold sweep ---
    empty_idx = [i for i, r in enumerate(records) if len(r["gt_boxes"]) == 0]
    print(f"Empty test images: {len(empty_idx)} / {len(records)}")
    for thresh in [0.10, 0.30, 0.50, 0.55, 0.70]:
        fp_empty = sum(int((records[i]["scores"] >= thresh).sum()) for i in empty_idx)
        total_fp = 0
        for rec in records:
            for i in range(len(rec["boxes"])):
                if rec["scores"][i] < thresh:
                    continue
                if len(rec["gt_boxes"]) == 0:
                    total_fp += 1
                    continue
                ious = box_iou(torch.tensor(rec["boxes"][i:i+1]),
                               torch.tensor(rec["gt_boxes"]))
                if ious.max().item() < args.iou_thresh:
                    total_fp += 1
        print(f"  conf>={thresh:.2f}: FP(all)={total_fp}, "
              f"FP/empty={fp_empty / len(empty_idx):.3f}, "
              f"FP/img={total_fp / len(records):.3f}")

    # --- Figures ---
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["bg", "smoke", "fire"],
                yticklabels=["bg", "smoke", "fire"])
    plt.xlabel("Predicted"); plt.ylabel("Ground truth")
    plt.title(f"Confusion matrix @ IoU={args.iou_thresh}, score={args.score_prf}")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    cls_names = list(prf.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(cls_names)); w = 0.25
    for i, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (i - 1) * w, [prf[c][metric] for c in cls_names], w, label=metric)
    ax.set_xticks(x); ax.set_xticklabels(cls_names); ax.set_ylim(0, 1)
    ax.set_ylabel("Score"); ax.legend()
    ax.set_title(f"Per-class P/R/F1 @ IoU={args.iou_thresh}, score={args.score_prf}")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_prf.png", dpi=150)
    plt.close()
    print(f"\nFigures → {fig_dir}")

    # --- Failure cases ---
    failures = find_failure_cases(records, score_thresh=args.score_prf)
    fail_dir = fig_dir / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    for cat, idxs in failures.items():
        for rank, item in enumerate(idxs[:5]):
            i = item if isinstance(item, int) else item[0]
            rec = records[i]
            draw_predictions(
                img_path=rec["path"],
                gt_boxes=rec["gt_boxes"], gt_labels=rec["gt_labels"],
                pred_boxes=rec["boxes"], pred_scores=rec["scores"], pred_labels=rec["labels"],
                score_thresh=args.score_prf,
                out_path=str(fail_dir / f"{cat}_{rank}.jpg"),
            )
    print(f"Failure examples → {fail_dir}")

    # --- Save results ---
    results = {
        "model": model_name,
        "model_type": args.model_type,
        "checkpoint": args.ckpt,
        "checkpoint_epoch": ckpt_epoch,
        "split": args.split,
        "num_images": len(test_ds),
        "mAP@0.5": mAP50,
        "mAP@0.5:0.95": mAP5095,
        "per_class_mAP@0.5:0.95": {
            "smoke": per_cls[0] if len(per_cls) > 0 else None,
            "fire": per_cls[1] if len(per_cls) > 1 else None,
        },
        "per_class_PRF@0.5": prf,
        "confusion_matrix": cm.tolist(),
        "fps": fps,
        "iou_thresh": args.iou_thresh,
        "score_thresh_prf": args.score_prf,
        "score_thresh_inference": args.score_low,
    }
    res_path = out_dir / "results" / f"{args.model_type}_test_thr{args.score_prf:.2f}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics JSON → {res_path}")
    print(f"\n✅ Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["faster_rcnn", "yolo"], required=True)
    parser.add_argument("--ckpt", required=True, help="Path to .pth or .pt")
    parser.add_argument("--data-root", default="data/data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default="evaluation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--score-low", type=float, default=0.001)
    parser.add_argument("--score-prf", type=float, default=0.55,
                        help="Operating threshold for P/R/F1 & confusion matrix")
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70, help="YOLO only")
    parser.add_argument("--max-det", type=int, default=100, help="YOLO only")
    parser.add_argument("--yolo-ckpt-epoch", default="N/A")
    parser.add_argument("--protocol-dir", default="yolo",
                        help="Directory with final_yolo_protocol.py & budget_protocol.py")
    args = parser.parse_args()
    main(args)