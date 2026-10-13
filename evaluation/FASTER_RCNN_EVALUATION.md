# Evaluation Report — First Working Model

## Results

### 1. Main metrics

| Metric                  | Value     |
|-------------------------|-----------|
| **mAP@0.5**             | **0.6965** |
| **mAP@0.5:0.95**        | **0.3551** |
| FPS (Tesla T4, batch=2) | **20.8**  |

Per-class mAP@0.5:0.95 (secondary): smoke **0.392**, fire **0.318**.

### 2. Per-class P / R / F1 @ IoU=0.5, score=0.5

| Class | Precision | Recall | F1    | TP    | FP    | FN   |
|-------|-----------|--------|-------|-------|-------|------|
| smoke | 0.5139    | 0.8037 | 0.6269 | 1 851 | 1 751 | 452  |
| fire  | 0.5964    | 0.6855 | 0.6379 | 1 973 | 1 335 | 905  |

### 3. Confusion matrix (rows = GT, cols = pred)

|              | pred bg | pred smoke | pred fire |
|--------------|---------|------------|-----------|
| **GT bg**    | 0       | 1 741      | 1 325     |
| **GT smoke** | 443     | 1 850      | 10        |
| **GT fire**  | 894     | 11         | 1 973     |

### 4. False positives

- **3 066** FPs on background regions (1 741 smoke + 1 325 fire).
- FP rate on the full test set: **3 066 / 4 306 ≈ 0.712 FP/image** —
  above the decision-rule threshold of 0.5 FP/image from the project plan.

### 5. Figures

- `figures/confusion_matrix.png`
- `figures/per_class_prf.png`

