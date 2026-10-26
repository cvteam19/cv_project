# Fire and Smoke Detection — Evaluation & Analysis

**Owner:** Ekaterina Kuznetsova (evaluation lead)

---

## Contents

- `evaluation.ipynb` — unified evaluation pipeline for **both** models.
  Computes mAP@0.5, mAP@0.5:0.95, per-class P/R/F1, confusion matrix,
  FPS, FP-vs-threshold sweep, and extracts top-K false positives.
- `speed_benchmark.ipynb` — controlled speed benchmark for both models
  (warmup, CUDA sync, GPU-only measurement).
- `results/faster_rcnn_test_thr0.70.json` — FRCN test metrics at score=0.70.
- `results/yolo_test_thr0.30.json` — YOLOv8s test metrics at score=0.30.
- `results/speed_benchmark.json` — controlled speed benchmark output.
- `figures/confusion_matrix_*.png` — one per model per threshold.
- `figures/per_class_prf_*.png` — one per model per threshold.
- `figures/fp_examples_*.png` — representative false positives by category (pending).
- `fp_analysis/<model>/fp_candidates.csv` — top-500 FPs per model, sorted by
  confidence, with a manual `category` column (C/F/L/B/S/O).
- `fp_analysis/<model>/crops/` — crops of FP boxes (regenerable from the notebook).

---

## How to run

### Prerequisites

1. Kaggle notebook, GPU T4, Internet ON.
2. Add inputs:
   - **Dataset** — `linxel/dfire-prepared`.
   - **Checkpoints** — a dataset folder with:
     - `yolo_selected_best.pt` (YOLOv8s)
     - `faster_rcnn_best.pth` (Faster R-CNN)
     - `final_yolo_protocol.py` and `budget_protocol.py`
       (**both** are required. stored in **yolo** folder).

### Run

1. Open `evaluation.ipynb`.
2. In `CFG`, set:
   - `model_type` — `"yolo"` or `"faster_rcnn"`.
   - `ckpt` — path to the checkpoint.
   - `protocol_dir` — path to the **folder** containing the two `.py` files.
   - `score_prf` — the operating confidence threshold for this run
     (0.30 for YOLOv8s; 0.70 for Faster R-CNN).
3. **Run All**. Results land in `/kaggle/working/evaluation/`.

Runtime: ~5 min inference + metrics for one model on 4,306 images on T4.

### Note on YOLOv8s

Do **not** load YOLOv8s via `YOLO(...).predict()`. That applies Ultralytics'
default letterbox preprocessing, which is **incompatible** with this checkpoint.
The notebook uses `predict_with_baseline_input(core, pil_images, ...)` from
`final_yolo_protocol.py`, which applies Faster R-CNN's `GeneralizedRCNNTransform`.
See `SECOND_MODEL_REPORT.md` for details.

---

## Metric definitions

- **mAP@0.5** — COCO-style via `torchmetrics.MeanAveragePrecision`, score floor 0.001, NMS IoU 0.7, max_det 100.
  Threshold-independent.
- **mAP@0.5:0.95** — averaged over 10 IoU thresholds. Threshold-independent.
- **P/R/F1** — manual greedy matching, IoU=0.5, **score=`score_prf`**.
  Threshold-dependent — depends on the operating confidence threshold.
- **Confusion matrix** — rows = GT (bg/smoke/fire), cols = pred.
  Column `pred bg` is always 0: detectors do not predict "background"; a 0 here is a false negative.
- **FPS / latency (controlled benchmark)** — GPU-only time (preprocessing + forward + NMS),
  images preloaded in RAM, `.cpu().numpy()` excluded, warmup and `torch.cuda.synchronize()`
  around each iteration. See `results/speed_benchmark.json`.
- **FPS in `results/*_test_thr*.json`** — full pipeline including PIL decode and
  `.cpu().numpy()`. **Not comparable across sessions** due to Kaggle free-tier GPU contention.
- **FP/empty** — average false positives on the **2,008 completely empty test images**
  (no GT boxes). This is the **false-alarm rate** — the operationally relevant metric.
- **FP/img** — average FP across all 4,306 test images (mixes false alarms and localization errors).

---

## Results summary

### Accuracy

| Model | Threshold | mAP@0.5 | mAP@0.5:0.95 | smoke F1 | fire F1 | FP/img | FP/empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN R50-FPN | 0.30 | **0.7687** | **0.4112** | 0.676 | 0.549 | 1.103 | 0.128 |
| Faster R-CNN R50-FPN | 0.70 | **0.7687** | **0.4112** | **0.789** | **0.666** | 0.420 | 0.044 |
| YOLOv8s (SGD, epoch 17) | 0.30 | 0.6348 | 0.3423 | 0.697 | 0.556 | **0.223** | **0.033** |

mAP is threshold-independent; P/R/F1/FP vary with the operating confidence threshold.
The two FRCN rows show the same model at two operating points (see `EVALUATION_REPORT.md` §4).

### Speed (controlled benchmark, T4, same session)

| Model | Latency (ms/img, b=4) | Throughput (img/s, b=4) | Speedup |
|---|---:|---:|---:|
| Faster R-CNN R50-FPN | 42.9 ± 0.7 | 23.3 | 1.0× |
| YOLOv8s (adapter) | **13.1 ± 0.2** | **76.6** | **3.3×** |

**Summary of comparison:**

- **Faster R-CNN** — higher mAP (0.769 vs 0.635) and higher F1 at its tuned threshold,
  but ~3.3× slower and with more false alarms at equivalent recall.
- **YOLOv8s** — lower mAP and F1, but 3.3× faster and with ~2–4× fewer false alarms.
- Both models pass the project's FP budget (< 0.5 FP/img) when operating at an appropriate
  threshold. The two detectors represent different precision/recall/latency trade-offs;
  neither dominates on all axes.

---

## Known limitations

1. **Training budget is time-matched, not compute-matched.** LR schedules differ
   (Faster R-CNN epoch-based, YOLOv8s wall-clock). The comparison is a controlled
   budget-matched study, not an isolated architecture effect.
2. **Threshold 0.30 was locked from YOLOv8s validation** and applied to Faster R-CNN
   for a unified comparison. FRCN at 0.30 is recall-biased (P=0.55, R=0.89); at 0.70
   it is F1-biased (P=0.76, R=0.82). A **validation-side threshold sweep for FRCN**
   is planned to identify its F1-maximizing operating point — this is remaining work.
3. **FPS across sessions varies 2–3×** due to Kaggle free-tier GPU contention.
   Use the **controlled benchmark** (`speed_benchmark.json`), not the per-run FPS in
   `results/*_test_thr*.json`, for any cross-model comparison.
4. **FP categorization** (top-500 per model) is manual labelling — **in progress**.