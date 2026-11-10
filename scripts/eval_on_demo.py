"""
Evaluate Faster R-CNN and YOLOv8s on a labeled subset
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import ToTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = {1: "smoke", 2: "fire"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_labels(lbl_path: Path, w: int, h: int):
    boxes, labels = [], []
    if not lbl_path.exists():
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5: continue
        cls, xc, yc, bw, bh = map(float, parts)
        x1 = max(0.0, (xc - bw / 2) * w)
        y1 = max(0.0, (yc - bh / 2) * h)
        x2 = min(w, (xc + bw / 2) * w)
        y2 = min(h, (yc + bh / 2) * h)
        if x2 <= x1 or y2 <= y1: continue
        boxes.append([x1, y1, x2, y2])
        labels.append(1 if int(cls) == 0 else 2)
    return (
        torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.as_tensor(labels, dtype=torch.int64),
    )


def ensure_frcnn_weights():
    ckpt = REPO_ROOT / "checkpoints" / "faster_rcnn" / "faster_rcnn_best.pth"
    if ckpt.exists(): return ckpt
    raise FileNotFoundError(f"{ckpt} not found.")

def ensure_yolo_weights():
    ckpt = REPO_ROOT / "yolo" / "selected_best.pt"
    if ckpt.exists(): return ckpt
    raise FileNotFoundError(f"{ckpt} not found.")


def load_frcnn(ckpt_path, device, num_classes=3, input_size=640):
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    model.transform.min_size = (input_size,)
    model.transform.max_size = input_size
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.inference_mode()
def run_frcnn(model, items, device, score_thr=0.001):
    preds, targets = [], []
    for img_path, boxes, labels in items:
        img = Image.open(img_path).convert("RGB")
        img_t = ToTensor()(img).to(device)
        out = model([img_t])[0]
        keep = out["scores"] >= score_thr
        preds.append({"boxes": out["boxes"][keep].cpu(), "scores": out["scores"][keep].cpu(), "labels": out["labels"][keep].cpu()})
        targets.append({"boxes": boxes, "labels": labels})
    return preds, targets


@torch.inference_mode()
def run_yolo(items, device, score_thr=0.001, nms_iou=0.7, max_det=100):
    proto = str((REPO_ROOT / "yolo").resolve())
    if proto not in sys.path: sys.path.insert(0, proto)
    from final_yolo_protocol import predict_with_baseline_input
    from ultralytics import YOLO

    core = YOLO(str(ensure_yolo_weights())).model.to(device).float().eval()
    preds, targets = [], []
    for img_path, boxes, labels in items:
        img = Image.open(img_path).convert("RGB")
        out = predict_with_baseline_input(core, [img], device=str(device), score_floor=score_thr, nms_iou=nms_iou, max_det=max_det)[0]
        preds.append({"boxes": out["boxes"].cpu(), "scores": out["scores"].cpu(), "labels": (out["labels"].cpu() + 1).long()})
        targets.append({"boxes": boxes, "labels": labels})
    return preds, targets


def compute_map(preds, targets):
    metric = MeanAveragePrecision(
        box_format="xyxy", 
        iou_type="bbox", 
        class_metrics=True,
        backend="faster_coco_eval"
    )
    p_remap = [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"] - 1} for p in preds]
    t_remap = [{"boxes": t["boxes"], "labels": t["labels"] - 1} for t in targets]
    metric.update(p_remap, t_remap)
    return metric.compute()


def per_class_prf(preds, targets, iou_thresh=0.5, score_thresh=0.3):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p, t in zip(preds, targets):
        for cls in [1, 2]:
            pm = (p["labels"] == cls) & (p["scores"] >= score_thresh)
            pb, tb = p["boxes"][pm], t["boxes"][t["labels"] == cls]
            if len(tb) == 0: stats[cls]["fp"] += len(pb); continue
            if len(pb) == 0: stats[cls]["fn"] += len(tb); continue
            ious, matched = box_iou(pb, tb), set()
            for i in range(len(pb)):
                bi, bj = 0.0, -1
                for j in range(len(tb)):
                    if j in matched: continue
                    v = ious[i, j].item()
                    if v > bi: bi, bj = v, j
                if bi >= iou_thresh and bj >= 0: stats[cls]["tp"] += 1; matched.add(bj)
                else: stats[cls]["fp"] += 1
            stats[cls]["fn"] += len(tb) - len(matched)
    out = {}
    for cls, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[CLASS_NAMES[cls]] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    data_dir = Path(args.data)
    image_paths = sorted(p for p in (data_dir / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_paths: raise FileNotFoundError(f"No images in {data_dir / 'images'}")
    print(f"[images] {len(image_paths)}")

    items = []
    for p in image_paths:
        with Image.open(p) as im: w, h = im.size
        boxes, labels = read_yolo_labels(data_dir / "labels" / (p.stem + ".txt"), w, h)
        items.append((p, boxes, labels))
    print(f"[gt] total ground-truth boxes: {sum(len(t[1]) for t in items)}")

    results = {}
    print("\n=== Faster R-CNN ===")
    p1, t1 = run_frcnn(load_frcnn(ensure_frcnn_weights(), device), items, device)
    m1 = compute_map(p1, t1)
    prf1 = per_class_prf(p1, t1, iou_thresh=args.iou, score_thresh=args.score)
    results["faster_rcnn"] = {"mAP@0.5": float(m1["map_50"]), "mAP@0.5:0.95": float(m1["map"]), "PRF": prf1}

    print("\n=== YOLOv8s ===")
    p2, t2 = run_yolo(items, device)
    m2 = compute_map(p2, t2)
    prf2 = per_class_prf(p2, t2, iou_thresh=args.iou, score_thresh=args.score)
    results["yolo"] = {"mAP@0.5": float(m2["map_50"]), "mAP@0.5:0.95": float(m2["map"]), "PRF": prf2}

    print("\n" + "=" * 68)
    print(f"METRICS on {len(image_paths)} labeled images (IoU={args.iou}, score>={args.score})")
    print("=" * 68)
    print(f"{'metric':<28} {'Faster R-CNN':>14} {'YOLOv8s':>14}")
    print("-" * 68)
    for key in ["mAP@0.5", "mAP@0.5:0.95"]:
        v1, v2 = results["faster_rcnn"][key], results["yolo"][key]
        print(f"{key:<28} {v1:>14.4f} {v2:>14.4f}")
    for cls in ["smoke", "fire"]:
        for metric in ["precision", "recall", "f1"]:
            v1 = results["faster_rcnn"]["PRF"][cls][metric]
            v2 = results["yolo"]["PRF"][cls][metric]
            print(f"{cls + ' ' + metric:<28} {v1:>14.4f} {v2:>14.4f}")
    print("=" * 68)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "demo_metrics.json").write_text(__import__("json").dumps(results, indent=2))
    print(f"\nSaved → {out / 'demo_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="demo_test")
    parser.add_argument("--out", default="results")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.3)
    args = parser.parse_args()
    main(args)