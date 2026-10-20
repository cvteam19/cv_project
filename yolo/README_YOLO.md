# Second model: YOLOv8s

Main model lead: **Daria Potapova** (`dariapotapova`).

This folder contains the selected **new SGD run**. Earlier YOLO/AdamW experiments are
not included. The results below are validation results, not a final test comparison.

## Model and data

- COCO-pretrained `yolov8s.pt`, fine-tuned for smoke and fire.
- Dataset: [linxel/dfire-prepared](https://www.kaggle.com/datasets/linxel/dfire-prepared).
- Original splits: train **14,122**, validation **3,099**, test **4,306** images.
- Labels: **0=smoke, 1=fire**. Background has no YOLO class ID.
- Input: torchvision `GeneralizedRCNNTransform(min_size=640, max_size=640)`;
  aspect ratio preserved, ImageNet mean/std, zero padding after normalization at
  the bottom/right to the batch maximum dimensions rounded up to a multiple of 32.
  This replaces YOLO's ordinary letterbox and normalization; inputs are not always square.

## Actual training configuration

| Setting | Value |
|---|---|
| GPU | Tesla T4, Kaggle |
| Optimizer | SGD, LR 0.005, momentum 0.9, weight decay 0.0005 |
| Decay scope / Nesterov | All trainable parameters / disabled |
| Batch / gradient accumulation | 4 / none |
| Precision | FP32, AMP disabled |
| Augmentation | Horizontal flip p=0.5, jointly updating image and boxes |
| Mosaic, HSV, scaling, translation | Disabled |
| Warmup / EMA averaging | Disabled / disabled |
| Gradient clipping / seed | Norm 10 / 42 |
| Learning rate | Cosine by elapsed train+validation time, 0.005 to 0.000001 over 3 hours |
| Best checkpoint | Maximum validation torchmetrics COCO mAP@0.5:0.95 |
| Early stopping | Patience 3 on that metric, or time limit |

The training call took **10,970.24 seconds** (about **3 h 3 min**, including its concluding
validation). There are **16 logged epochs**, with the final epoch incomplete. The selected
checkpoint is from **epoch 14**. The 1000-epoch setting was a technical upper guard, not
a claim that 1000 epochs were completed. The native trainer adapts its epoch estimate
when `time=3` is supplied.

`cos_lr=false` in args does not mean cosine decay was absent: `optimizer_step()` applies
the explicit time-based schedule from `budget_protocol.py`. Classification-only defaults
such as `auto_augment` and `erasing` in Ultralytics' general args are not applied by the
custom detection dataset. The only stochastic augmentation is the joint horizontal flip.

## Recorded validation results

The saved best checkpoint was reevaluated on 3,099 validation images, containing
3,931 annotated instances and 1,375 completely empty images.

| Metric | Value |
|---|---:|
| mAP@0.5 | 0.6378518 |
| mAP@0.5:0.95 | 0.3411905 |
| Smoke AP@0.5:0.95 | 0.3901825 |
| Fire AP@0.5:0.95 | 0.2921985 |
| Locked confidence | 0.30 |
| Precision / recall / F1, micro | 0.6649 / 0.5785 / 0.6187 |
| TP / FP / FN | 2,274 / 1,146 / 1,657 |
| FP per empty image | 53 / 1,375 = 0.0385455 |
| Empty images with any detection | 44 / 1,375 = 3.2% |
| Recorded throughput / latency, batch 4 | 64.09 images/s / 15.60 ms per image |

Confidence was selected from the validation sweep by maximum micro-F1, with IoU >= 0.5
class-aware matching. mAP uses score floor 0.001, NMS IoU 0.7, max detections 100.
The FP/image target <0.5 is met **on validation**; this does not establish test performance.

Timing includes RGB PIL conversion/preprocessing, H2D transfer, inference and postprocessing;
it excludes file decoding, output D2H copies and metrics. Three warmup batches and CUDA
synchronization were used. This is one recorded validation pass, not a definitive speed
comparison with Faster R-CNN. Values are stored in `yolov8s_val_metrics.json`.

The epoch-14 training log has mAP@0.5:0.95 **0.34107**, whereas reevaluation of the serialized
checkpoint gives **0.3411905**. These are separate recorded measurements; the reevaluation
value is used above. The checkpoint serializer stores model weights in FP16; they are
loaded as FP32 for inference. Do not replace either recorded value with an invented match.

## Files

- `main.ipynb`: standalone Kaggle reproduction notebook, with outputs cleared.
- `selected_best.pt`: actual selected inference checkpoint (about 22.5 MB).
- `checkpoint_manifest.json`: checkpoint hash, size and provenance.
- `training_history.json`, `training_history.csv`, `training_curves.png`: convenient
  training exports. JSON is derived from the original CSV, preserving its recorded values.
- `train/`: original Ultralytics logs, args, plots and sample batches.
- `training_summary.json`, `requested_training_config.json`, `verified_runtime.json`:
  requested and effective training settings and runtime checks.
- `environment.json`, `pip_freeze.txt`: actual run environment and packages.
- `data.yaml`, `dataset_source.json`: original dataset configuration/provenance.
  Their absolute Kaggle paths describe the run; `main.ipynb` generates paths for a new run.
- `final_yolo_protocol.py`, `budget_protocol.py`, `main_model.py`, `final_evaluation.py`:
  exact helper files included in the completed run's archive.
- `yolov8s_val_*.json/csv/png/jpg`, `yolov8s_validation_*`, `locked_selection.json`:
  validation metrics, predictions, operating point, sensitivity and error candidates.
- `SECOND_MODEL_REPORT.md`: short second-approach report with verified initial results.

The archive only included the selected checkpoint, not `last.pt`. `selected_best.pt`
is an inference checkpoint, not a resumable optimizer checkpoint. This repository
does not claim to provide an executed notebook or a resumable last checkpoint.

## Reproduce training on Kaggle

1. Upload `main.ipynb` to Kaggle. Enable a GPU and Internet; add `linxel/dfire-prepared`.
2. Run cells in order. The input/box, flip and optimizer checks run before training.
3. Leave `RUN_FINAL_TEST=False` while choosing settings. A fresh run uses COCO weights,
   copies/validates data and trains under the specified budget. Results can vary by environment.
4. Save a Kaggle Version with outputs. The notebook exports an artifact ZIP.

The optional baseline-evaluation block does not train Faster R-CNN. Only supply the
corrected baseline weights if that shared evaluation is wanted.

## Load the saved model: important input adapter

Do **not** use ordinary `YOLO.predict()` or ordinary `.val()` preprocessing with these weights.
The checkpoint was trained with the shared Faster R-CNN input transform. Run from this
folder (or put it on `sys.path`) and use the adapter:

```python
import torch
from PIL import Image
from ultralytics import YOLO
from final_yolo_protocol import predict_with_baseline_input

core = YOLO('selected_best.pt').model.to('cuda').float().eval()
with Image.open('example.jpg') as image:
    batch = [image.convert('RGB')]
with torch.inference_mode():
    predictions = predict_with_baseline_input(core, batch, device='cuda',
        score_floor=0.001, nms_iou=0.7, max_det=100)
# boxes: original-image pixel xyxy; labels: 0=smoke, 1=fire.
# For operating P/R/F1 or visualization, apply scores >= 0.30 afterward.
# Keep the low score floor when calculating mAP.
```

To run validation through Ultralytics, supply `validator=BaselineInputValidator` from
`final_yolo_protocol.py`, `half=False`, and a data YAML with valid paths. The standalone
notebook demonstrates the complete evaluation workflow.

Verify weights: SHA256
`054780cd560b52a2243954afa18494fbd993a2222a04d5860cd08d32263847ec`.

## Comparison limits and handoff

The corrected baseline currently records validation mAP50 **0.7675** and mAP50:95
**0.4204** after six epochs (about 4 h 1 min), using an epoch-based cosine schedule.
YOLO used about three hours and a time-based schedule. Inference filtering also needs
to be unified. The existing scores are preliminary, not an equal-budget architecture ablation.

The evaluation lead should use this **new SGD checkpoint** and its adapter for the final
common test evaluation, choose baseline confidence on validation, and report common
metrics, timing and error categories. The candidate images are automatically selected
examples with GT in green and predictions in red; they are not a completed manual
cloud/fog/lighting/bright-object classification. Hard-negative fine-tuning, resolution
experiments and RT-DETR training were not completed in this run and are not claimed here.
