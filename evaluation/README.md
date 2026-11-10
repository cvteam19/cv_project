# Fire and Smoke Detection — Evaluation & Analysis

**Owner:** Ekaterina Kuznetsova (evaluation lead)

---

## Contents

- `evaluation.ipynb` — unified evaluation pipeline for **both** models.
  Computes mAP@0.5, mAP@0.5:0.95, per-class P/R/F1, confusion matrix,
  FPS, FP-vs-threshold sweep, and extracts top-K false positives.
- `speed_benchmark.ipynb` — controlled speed benchmark for both models
  (warmup, CUDA sync, GPU-only measurement).
- `results/faster_rcnn_test_thr0.30.json`, `_thr0.55.json`, `_thr0.70.json` — FRCN
  test metrics at three operating points.
- `results/yolo_test_thr0.30.json` — YOLOv8s test metrics at score=0.30.
- `results/speed_benchmark.json` — controlled speed benchmark output.
- `figures/confusion_matrix_*.png` — one per model per threshold.
- `figures/per_class_prf_*.png` — one per model per threshold.
- `figures/yolo_failure_cases/*.jpg` — representative YOLOv8s failure cases (by category).
- `figures/faster_rcnn_failure_cases/*.jpg` — representative FRCN failure cases (by category).
- `fp_analysis/<model>/fp_candidates.csv` — top FPs per model, sorted by
  confidence (regenerable from the notebook).
- `fp_analysis/<model>/crops/` — crops of FP boxes (regenerable from the notebook).

---

## How to run

### Prerequisites

1. Kaggle notebook, GPU T4, Internet ON.
2. Add inputs:
   - **Dataset** — `linxel/dfire-prepared`.
   - **Checkpoints** — a dataset folder (`/kaggle/input/datasets/c3m3nt/cv-project-checkpoints`) with:
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
     (0.30 for YOLOv8s; 0.55 for Faster R-CNN).
3. **Run All**. Results land in `/kaggle/working/evaluation/`.

Runtime: ~7 min inference + metrics for one model on 4,306 images on T4.

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
- **FP/img** — average **class-aware** FP across all 4,306 test images
  (`(smoke_FP + fire_FP) / 4306`). A prediction is FP iff its best IoU with a GT
  of the same class is below 0.5. Includes false alarms and localization errors.

---

## Results summary

### Accuracy (threshold-independent)

| Model | mAP@0.5 | mAP@0.5:0.95 | smoke AP | fire AP |
|---|---:|---:|---:|---:|
| **Faster R-CNN R50-FPN** | **0.7687** | **0.4112** | **0.4700** | **0.3523** |
| YOLOv8s | 0.6348 | 0.3423 | 0.3972 | 0.2874 |

### Operating points (validation-tuned thresholds)

| Model | Threshold | smoke F1 | fire F1 | macro F1 | FP/img | FP/empty |
|---|---:|---:|---:|---:|---:|---:|
| **Faster R-CNN @ 0.55** | 0.55 | **0.756** | **0.633** | **0.694** | 0.654 | 0.078 |
| **YOLOv8s @ 0.30** | 0.30 | 0.697 | 0.556 | 0.626 | **0.284** | **0.033** |

FRCN at 0.55 yields FP/img=0.654 — above the project's
0.5 budget. At 0.70, FRCN meets the budget (FP/img=0.439)  (see `EVALUATION_REPORT.md` §2.3).

### Speed (controlled benchmark, T4, same session)

| Model | Latency (ms/img, b=4) | Throughput (img/s, b=4) | Speedup |
|---|---:|---:|---:|
| Faster R-CNN R50-FPN | 42.9 ± 0.7 | 23.3 | 1.0× |
| YOLOv8s (adapter) | **13.1 ± 0.2** | **76.6** | **3.3×** |

**Summary of comparison:**

- **Faster R-CNN** — higher mAP (0.769 vs 0.635) and higher F1 at appropriate threshold,
  but ~3.3× slower and with more false alarms at equivalent recall.
- **YOLOv8s** — lower mAP and F1, but 3.3× faster and with ~2–4× fewer false alarms.
- Both models pass the project's FP budget (< 0.5 FP/img) when operating at an appropriate
  threshold. The two detectors represent different precision/recall/latency trade-offs;
  neither dominates on all axes.

---

## Known limitations

1. **Training budget is time-matched, not compute-matched.** LR schedules differ
   (Faster R-CNN epoch-based cosine, YOLOv8s wall-clock cosine). The comparison is
   a **controlled budget-matched study**, not an isolated architecture effect.
2. **Failure cases are manually selected** from the automatically extracted
   candidate pool (`find_failure_cases` in `evaluation.ipynb`), and the choice
   of representative examples is subjective. Six cases across two
   models cover the three main failure categories; more granular distributions
   would require manual labelling of all 500+ FP candidates per model, which
   is out of scope for this submission.
3. **FPS in per-run JSONs** varies 2–3× across sessions; only the controlled
   benchmark is used for cross-model comparison.
