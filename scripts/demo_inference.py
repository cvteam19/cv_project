"""
Quick demo: run Faster R-CNN and YOLOv8s on a folder of images,
save detections as JSON and annotated PNGs.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import argparse
import json
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor

CLASS_NAMES = {1: "smoke", 2: "fire"}  # torchvision convention: 0=bg,1=smoke,2=fire
YOLO_NAMES = {0: "smoke", 1: "fire"}   # YOLO convention (from data.yaml)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------- weights ----------
def ensure_faster_rcnn_weights(ckpt_dir: Path) -> Path:
    ckpt = ckpt_dir / "faster_rcnn_best.pth"
    if ckpt.exists():
        return ckpt
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        print(f"[download] {ckpt.name} from HuggingFace...")
        path = hf_hub_download(
            repo_id="MariKarrp/cvteam19-fire-smoke-detection",
            filename="faster_rcnn_best.pth",
            cache_dir=str(ckpt_dir),
        )
        import shutil
        shutil.copy(path, ckpt)
        return ckpt
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {ckpt.name}: {e}\n"
            f"Either place it manually at {ckpt} or install huggingface_hub."
        )


def ensure_yolo_weights(ckpt_dir: Path) -> Path:
    ckpt = ckpt_dir / "selected_best.pt"
    if ckpt.exists():
        return ckpt
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        print(f"[download] {ckpt.name} from HuggingFace...")
        path = hf_hub_download(
            repo_id="MariKarrp/cvteam19-fire-smoke-detection",
            filename="selected_best.pt",
            cache_dir=str(ckpt_dir),
        )
        import shutil
        shutil.copy(path, ckpt)
        return ckpt
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {ckpt.name}: {e}\n"
            f"Either place it manually at {ckpt} or install huggingface_hub."
        )


# ---------- models ----------
def load_faster_rcnn(ckpt_path: Path, device, num_classes=3, input_size=640):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform.min_size = (input_size,)
    model.transform.max_size = input_size

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_yolo(ckpt_path: Path, device):
    from ultralytics import YOLO
    wrapper = YOLO(str(ckpt_path))
    core = wrapper.model.to(device).float().eval()
    return core  

# ---------- drawing ----------
def get_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(img: Image.Image, dets, names, thr=0.30):
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    for d in dets:
        if d["score"] < thr:
            continue
        x1, y1, x2, y2 = d["box"]
        color = (255, 0, 0) if names.get(d["label"]) == "fire" else (255, 140, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{names.get(d['label'], '?')} {d['score']:.2f}"
        draw.text((x1 + 3, max(0, y1 - 20)), label, fill=color, font=font)
    return img


# ---------- inference ----------
@torch.no_grad()
def run_faster_rcnn(model, images_paths, device, score_thr=0.05):
    results = []
    for p in images_paths:
        img = Image.open(p).convert("RGB")
        img_t = ToTensor()(img).to(device)
        out = model([img_t])[0]
        dets = []
        for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
            s = float(score.item())
            if s < score_thr:
                continue
            dets.append({
                "box": [float(x) for x in box.tolist()],
                "score": s,
                "label": int(label.item()),
            })
        results.append({"image": str(p), "detections": dets})
    return results


@torch.inference_mode()
def run_yolo(core_model, image_paths, device, score_thr=0.05, nms_iou=0.7, max_det=100):
    import sys
    proto_dir = str((REPO_ROOT / "yolo").resolve())
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)
    from final_yolo_protocol import predict_with_baseline_input

    results = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        out = predict_with_baseline_input(
            core_model, [img], device=str(device),
            score_floor=score_thr, nms_iou=nms_iou, max_det=max_det,
        )[0]
        
        dets = []
        for box, conf, cls in zip(out["boxes"], out["scores"], out["labels"]):
            dets.append({
                "box": [float(x) for x in box.tolist()],
                "score": float(conf.item()),
                "label": int(cls.item()) + 1,
            })
        results.append({"image": str(p), "detections": dets})
    return results

# ---------- main ----------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"--images dir not found: {images_dir}")
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images in {images_dir}")
    print(f"[images] {len(image_paths)} from {images_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"device": str(device), "num_images": len(image_paths), "models": {}}

    # ---- Faster R-CNN ----
    print("\n=== Faster R-CNN ===")
    frcnn_ckpt = ensure_faster_rcnn_weights(Path("checkpoints/faster_rcnn"))
    print(f"[ckpt] {frcnn_ckpt}")
    frcnn = load_faster_rcnn(frcnn_ckpt, device)

    t0 = time.time()
    frcnn_results = run_faster_rcnn(frcnn, image_paths, device, score_thr=args.score_thr)
    frcnn_time = time.time() - t0
    frcnn_fps = len(image_paths) / max(frcnn_time, 1e-6)
    print(f"[done] {len(image_paths)} images in {frcnn_time:.1f}s ({frcnn_fps:.2f} img/s)")

    (out_dir / "demo_faster_rcnn.json").write_text(json.dumps(frcnn_results, indent=2))
    for r in frcnn_results:
        img = Image.open(r["image"])
        draw_detections(img, r["detections"], CLASS_NAMES, thr=args.score_thr).save(
            out_dir / f"frcnn_{Path(r['image']).stem}.png"
        )
    summary["models"]["faster_rcnn"] = {
        "checkpoint": str(frcnn_ckpt),
        "fps": round(frcnn_fps, 2),
        "total_time_sec": round(frcnn_time, 1),
        "total_detections": sum(len(r["detections"]) for r in frcnn_results),
    }

    # ---- YOLO ----
    print("\n=== YOLOv8s ===")
    yolo_ckpt = ensure_yolo_weights(Path("yolo"))
    print(f"[ckpt] {yolo_ckpt}")
    yolo = load_yolo(yolo_ckpt, device)

    t0 = time.time()
    yolo_results = run_yolo(yolo, image_paths, device, score_thr=args.score_thr)
    yolo_time = time.time() - t0
    yolo_fps = len(image_paths) / max(yolo_time, 1e-6)
    print(f"[done] {len(image_paths)} images in {yolo_time:.1f}s ({yolo_fps:.2f} img/s)")

    (out_dir / "demo_yolo.json").write_text(json.dumps(yolo_results, indent=2))
    for r in yolo_results:
        img = Image.open(r["image"])
        draw_detections(img, r["detections"], CLASS_NAMES, thr=args.score_thr).save(
            out_dir / f"yolo_{Path(r['image']).stem}.png"
        )
    summary["models"]["yolo"] = {
        "checkpoint": str(yolo_ckpt),
        "fps": round(yolo_fps, 2),
        "total_time_sec": round(yolo_time, 1),
        "total_detections": sum(len(r["detections"]) for r in yolo_results),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- print table ----
    print("\n" + "=" * 60)
    print(f"{'model':<15} {'images':>8} {'fps':>8} {'detections':>12} {'time, s':>10}")
    print("-" * 60)
    for name, s in summary["models"].items():
        print(f"{name:<15} {len(image_paths):>8} {s['fps']:>8.2f} "
              f"{s['total_detections']:>12} {s['total_time_sec']:>10.1f}")
    print("=" * 60)
    print(f"\nAll results saved to: {out_dir.resolve()}")
    print(f"  - demo_faster_rcnn.json / demo_yolo.json  (raw detections)")
    print(f"  - frcnn_*.png / yolo_*.png                (annotated images)")
    print(f"  - summary.json                            (fps, totals)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="demo_test/images",
                        help="Folder with input images")
    parser.add_argument("--out", default="results", help="Output folder")
    parser.add_argument("--score-thr", type=float, default=0.30,
                        help="Confidence threshold for saving detections")
    args = parser.parse_args()
    main(args)