# Evaluation & Analysis

## Contents

- `evaluation.ipynb` — unified evaluation pipeline.
  Computes: mAP@0.5, mAP@0.5:0.95, per-class P/R/F1, confusion matrix, FPS,
  and extracts top-K false positives for manual categorization.
- `results/faster_rcnn_test.json` — all metrics for the baseline on the test split.
- `figures/` — confusion matrix, per-class PRF.
- `fp_analysis/fp_candidates.csv` — top-100 FPs with manual `category` column.
- `fp_analysis/crops/` — **not committed**, regenerable from the notebook.

## How to run

1. Open `evaluation.ipynb` in Kaggle.
2. Add inputs (right panel):
   - Dataset: `sayedgamal99/smoke-fire-detection-yolo`
   - Your checkpoints dataset
3. Set `Accelerator: GPU T4`, `Internet: On`.
4. Set the three paths in the `CFG` dataclass at the top of the cell.
5. **Run All**. Results appear under `/kaggle/working/evaluation/`.

Runtime: ~5 min inference + metrics on T4 for 4 306 images.

## Metric definitions

- **mAP@0.5** — COCO-style, via `torchmetrics.MeanAveragePrecision`.
- **mAP@0.5:0.95** — averaged over 10 IoU thresholds.
- **P/R/F1** — manual greedy matching, IoU=0.5, score=0.5.
- **Confusion matrix** — rows=GT (bg/smoke/fire), cols=pred.
- **FPS** — inference only, batch=2, T4.
- **FP rate** — FPs with score ≥ 0.5 and no GT match (IoU ≥ 0.5) per image.