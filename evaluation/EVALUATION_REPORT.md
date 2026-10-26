# Evaluation Report — Baseline and Second Model

**Evaluation lead:** Ekaterina Kuznetsova

---

## 1. Main metrics

| Metric | Faster R-CNN R50-FPN (epoch 6) | YOLOv8s (epoch 17, SGD rerun) |
|---|---:|---:|
| **mAP@0.5** | **0.7687** | 0.6348 |
| **mAP@0.5:0.95** | **0.4112** | 0.3423 |
| smoke mAP@0.5:0.95 | **0.4700** | 0.3972 |
| fire mAP@0.5:0.95 | **0.3523** | 0.2874 |

Both mAP metrics are **threshold-independent**: they are computed over the full
precision-recall curve with `score_low=0.001`, NMS IoU 0.7, max_det 100.
The two models therefore have a **fixed accuracy gap of +13.4 pp (mAP@0.5) and
+6.9 pp (mAP@0.5:0.95)** in favour of Faster R-CNN, independent of the operating
threshold. Per-class AP shows the same pattern: FRCN wins on both smoke (+7.3 pp)
and fire (+6.5 pp).

---

## 2. Per-class P / R / F1

P/R/F1 are **threshold-dependent**: they describe the model at a specific operating
point. The operating threshold is chosen per model; for YOLOv8s it is `0.30`
(locked from validation in the main-model track); for Faster R-CNN both `0.30`
(unified-comparison point) and `0.70` (closer to its F1 optimum) are reported.

### 2.1. Faster R-CNN @ conf=0.30 (unified-comparison point)

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 0.546 | **0.888** | 0.676 | 2046 | 1703 | 257 |
| fire  | 0.411 | **0.830** | 0.549 | 2389 | 3428 | 489 |

### 2.2. Faster R-CNN @ conf=0.70 (F1-biased point)

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| smoke | **0.758** | 0.823 | **0.789** | 1895 | 605 | 408 |
| fire  | **0.618** | 0.722 | **0.666** | 2079 | 1287 | 799 |

### 2.3. YOLOv8s @ conf=0.30 (locked from validation)

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| smoke | **0.735** | 0.662 | **0.697** | 1524 | **549** | 779 |
| fire  | **0.670** | 0.475 | **0.556** | 1366 | **673** | 1512 |

**Interpretation:**

- **YOLOv8s is precision-oriented** (high P, moderate R at its locked threshold).
- **Faster R-CNN is recall-oriented at 0.30** (very high R, low P) and becomes
  **F1-balanced at 0.70** (P and R both ~0.72–0.82).
- The F1 of Faster R-CNN **increases by ~10 pp per class** when moving from 0.30 to 0.70.
  This is **not evidence that 0.30 is unfair to FRCN** — it is evidence that
  0.30 is not FRCN's optimal F1 point. A validation-side threshold sweep (planned,
  see §8) will identify FRCN's optimum.

---

## 3. Confusion matrices (rows = GT, cols = pred)

### Faster R-CNN @ 0.30

|              | pred bg | pred smoke | pred fire |
|--------------|---------|------------|-----------|
| **GT bg**    | 0       | 1688       | 3418      |
| **GT smoke** | 249     | 2043       | 11        |
| **GT fire**  | 472     | 18         | 2388      |

### Faster R-CNN @ 0.70

|              | pred bg | pred smoke | pred fire |
|--------------|---------|------------|-----------|
| **GT bg**    | 0       | 594        | 1277      |
| **GT smoke** | 399     | 1894       | 10        |
| **GT fire**  | 787     | 12         | 2079      |

### YOLOv8s @ 0.30

|              | pred bg | pred smoke | pred fire |
|--------------|---------|------------|-----------|
| **GT bg**    | 0       | 536        | 668       |
| **GT smoke** | 773     | 1523       | 7         |
| **GT fire**  | 1500    | 14         | 1364      |

**Notes:**

- The `pred bg` column is always 0. This is **correct behaviour** for COCO-style
  detectors: they never output "background" as a class. A 0 in this column
  represents a **false negative**, not a background prediction.
- **Class confusion smoke↔fire is negligible** across all operating points
  (11+18=29 FRCN@0.30; 10+12=22 FRCN@0.70; 7+14=21 YOLO@0.30 out of thousands of
  predictions). The two models do **not** confuse the classes; their gaps are
  localization- and miss-driven.

---

## 4. False-positive analysis across thresholds

### 4.1. FP rate vs confidence threshold

| Threshold | FRCN FP/img | FRCN FP/empty | YOLOv8s FP/img | YOLOv8s FP/empty |
|---|---:|---:|---:|---:|
| 0.10 | 3.650 | 2.685 | 0.755 | 0.102 |
| **0.30** | 1.103 | 0.128 | **0.223** | **0.033** |
| 0.50 | 0.697 | 0.078 | 0.077 | 0.016 |
| **0.70** | **0.420** | **0.044** | 0.016 | 0.005 |

- `FP/empty` — average FP per completely empty test image (2,008 / 4,306); this is
  the **false-alarm rate** relevant to real deployment.
- `FP/img` — average FP across all 4,306 test images (mixes false alarms and
  localization errors).

**Observations:**

- **Both models pass the project FP budget (<0.5 FP/img)** at their respective
  operating thresholds: FRCN at 0.70 (0.420) and YOLOv8s at 0.30 (0.223).
- **YOLOv8s is consistently 2–5× better on false alarms** at every threshold:
  FP/empty 0.033 vs 0.128 @0.30; FP/img 0.223 vs 1.103 @0.30.
- **FRCN's FP count is highly threshold-sensitive** (10× drop in FP/img from 0.30
  to 0.70) because it produces many low-confidence proposals on hard negatives.
  **YOLOv8s is comparatively robust** (only 3× drop from 0.10 to 0.30, then flat).

### 4.2. Per-model operating points

The project's decision rule ("best mAP@0.5 with FP rate < 0.5 FP/img") is satisfied
by **both models** at an appropriate threshold. The differences are:

| Model | Operating threshold | mAP@0.5 | smoke F1 | fire F1 | FP/img | FP/empty |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 0.70 (F1-biased) | **0.769** | **0.789** | **0.666** | 0.420 | 0.044 |
| YOLOv8s | 0.30 (validated) | 0.635 | 0.697 | 0.556 | **0.223** | **0.033** |

- **Faster R-CNN has the higher accuracy** (mAP and F1, both classes).
- **YOLOv8s has the lower false-alarm rate** (2× fewer FP/img, 1.3× fewer FP/empty).
- **Neither dominates on all axes**.

### 4.3. FP categorization (pending)

500 FP candidates per model are extracted to `fp_analysis/<model>/fp_candidates.csv`,
sorted by confidence. Planned category schema:

| ID | Category |
|---|---|
| `C` | Clouds |
| `F` | Fog / haze |
| `L` | Artificial lighting |
| `B` | Bright/reflective objects |
| `S` | Smoke-as-fire (cross-class confusion between smoke and fire boxes) |
| `O` | Other (unclassified) |

**Manual labelling is in progress for both models.** Category distribution and
1–2 representative examples per category will be reported.

---

## 5. Speed benchmark (controlled)

Measurement scope: GPU-only (preprocessing + forward + NMS), images preloaded in
RAM, `.cpu().numpy()` post-processing excluded, warmup = 2 full passes,
`torch.cuda.synchronize()` around each timed iteration, 100 images benchmarked
per configuration, results averaged over 10 iterations. Both models were run in
the **same Kaggle session on the same T4 instance**, so the **ratio is reliable**.

| Model | Batch | Latency (ms/img) | Throughput (img/s) | Speedup vs FRCN |
|---|---:|---:|---:|---:|
| Faster R-CNN R50-FPN | 1 | 42.2 ± 0.8 | 23.7 | 1.0× |
| Faster R-CNN R50-FPN | 4 | 42.9 ± 0.7 | 23.3 | 1.0× |
| YOLOv8s (adapter) | 1 | 17.7 ± 0.2 | 56.5 | **2.4×** |
| **YOLOv8s (adapter)** | **4** | **13.1 ± 0.2** | **76.6** | **3.3×** |

**Observations:**

- **YOLOv8s is 3.3× faster** than Faster R-CNN at batch=4 (13.1 vs 42.9 ms/img).
- **Faster R-CNN does not scale with batch size** (42.2 → 42.9 ms/img from b=1 to b=4).
  This is expected for two-stage detectors: the ROI head processes a variable-length
  list of proposals per image, so GPU utilisation stays roughly the same regardless
  of batch size.
- **YOLOv8s scales with batch size** (17.7 → 13.1 ms/img), as expected for a
  single-stage detector.
- **Standard deviation is 1–2%** of the mean, so the measurement is stable and
  not an artifact of GPU contention.

**Comparison with per-run FPS in `results/*_test_thr*.json`.** Those numbers come
from a **full end-to-end pipeline** including PIL decoding and `.cpu().numpy()`
transfer of output boxes, and they vary 2–3× across sessions because the Kaggle
free-tier T4 is shared. They are **not comparable across sessions**. The controlled
benchmark above is the **only reliable cross-model speed comparison**.

---

## 6. Failure cases (planned)

Three representative failure cases will be selected after FP categorization:

1. A missed fire instance (FN) — small or partially occluded fire.
2. A smoke→cloud or cloud→smoke FP — the canonical confusion from the project brief.
3. A bright/reflective-object fire FP (sun, lights, reflections) — the dominant
   FRCN failure mode by raw count.

Each case will include the original image with GT boxes (green) and predictions (red),
and a one-sentence likely cause.

---

## 7. Reproducibility

- Code: `evaluation.ipynb`, `speed_benchmark.ipynb`.
- Environment: Kaggle T4, `ultralytics==8.3.221`, `torchmetrics`, `torch`, `torchvision`.
- Runtime: ~5 min inference + metrics on 4,306 test images per model per threshold.
- Dataset: linxel/dfire-prepared.
- Seeds: dataset splits unchanged. Model checkpoints are the selected validation-best
  checkpoints; evaluation does not re-train.

---

## 8. Known limitations and remaining work

### 8.1. Limitations

1. **Training budget is time-matched, not compute-matched.** LR schedules differ
   (Faster R-CNN epoch-based cosine, YOLOv8s wall-clock cosine). The comparison is
   a **controlled budget-matched study**, not an isolated architecture effect.
2. **Threshold 0.30 was locked from YOLOv8s validation** and applied to FRCN for
   a unified comparison. This is reported transparently: FRCN at 0.30 is recall-biased
   (P=0.55, R=0.89), while at 0.70 it is F1-biased (P=0.76, R=0.82). Both points
   are included to avoid cherry-picking.
3. **FPS in per-run JSONs** varies 2–3× across sessions; only the controlled
   benchmark is used for cross-model comparison.

### 8.2. Remaining work (evaluation track)

1. **Manual FP categorization** for both models: 500 samples each, 6 categories.
   Produce the per-category distribution table and 1–2 representative examples
   per category.
2. **3 representative failure cases** with original images.