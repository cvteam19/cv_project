# Second Working Model: YOLOv8s

## New approach

Our second approach is YOLOv8s, a single-stage detector fine-tuned from COCO-pretrained
weights for smoke and fire detection. It is compared with the updated Faster R-CNN
ResNet-50-FPN baseline, a two-stage detector.

## Motivation

YOLOv8s was selected as a compact alternative with a potential inference-speed advantage.
We investigate the trade-off between detection quality, speed and false positives;
improved accuracy over Faster R-CNN was not assumed in advance.

## Implementation and revision of the plan

Both models use `linxel/dfire-prepared` with 14,122 training, 3,099 validation and
4,306 test images. We revised the original recipes: YOLO's AdamW and multiple
augmentations were replaced with SGD and horizontal flips only. The baseline's
image-only flip was corrected to update bounding boxes jointly.

The updated models share SGD (initial LR 0.005, momentum 0.9, weight decay 0.0005,
no Nesterov), batch 4, FP32 without AMP, horizontal flips p=0.5, and checkpoint selection
by maximum validation mAP@0.5:0.95. YOLO uses the baseline's resize/normalization/padding:
aspect ratio preserved, max dimension 640, ImageNet normalization and batch padding.

To bring training durations closer, YOLO was retrained from the original COCO weights
with a time budget of 14,488.2 seconds, taken from the baseline log. Both runs used
Tesla T4 GPUs. YOLO retains time-based cosine LR decay; Faster R-CNN retains cosine
decay over six epochs.

| Setting | Updated Faster R-CNN | YOLOv8s rerun |
|---|---:|---:|
| Logged epochs | 6 | 20 |
| Selected epoch | 6 | 17 |
| Recorded duration | 14,488.2 s (about 4 h 1 min) | 14,419.82 s (about 4 h) |
| LR schedule | Cosine over 6 epochs | Cosine over the configured time budget |

YOLO's three epochs after epoch 17 did not improve validation mAP, consistent with
the configured early-stopping patience of three. Its recorded training-call duration
was approximately 68 seconds shorter than the baseline duration.

## Initial results

| Recorded validation metric | Updated Faster R-CNN | YOLOv8s rerun |
|---|---:|---:|
| mAP@0.5 | 0.7675 | 0.6484 |
| mAP@0.5:0.95 | 0.4204 | 0.3452 |

The recorded validation scores favour Faster R-CNN. Training durations are now close,
but LR schedules and inference-filtering settings still differ. These results therefore
do not isolate architecture effects; a common evaluation of the selected checkpoints
is still required.

For YOLO, confidence 0.30 maximized validation micro-F1. At that threshold, micro
precision/recall/F1 were 0.7114/0.5800/0.6390, with 2,280 TP, 925 FP and 1,651 FN.
There were 39 false-positive boxes across 1,375 empty validation images:
0.0284 FP/image, below the project target of 0.5. Fire recall was 0.5041, making
missed fire instances a priority for inspection.

This folder contains validation results, not final YOLO test results. The baseline-time
rerun replaces the earlier three-hour run as the active second-model checkpoint.
Unified test evaluation, speed comparison and detailed error analysis are assigned
to the evaluation lead. No completed hard-negative or RT-DETR experiment is claimed.

Sources: [YOLO metrics](yolov8s_val_metrics.json), [training log](train/results.csv),
[run summary](training_summary.json), [baseline duration](baseline_budget_reference.json),
[baseline history](../baseline/training_history.json), [verification](validation_audit.json).
