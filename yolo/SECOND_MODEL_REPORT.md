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

The completed runs did not fully match the intended common time budget. YOLO used
cosine LR decay by elapsed time over three hours; the baseline retained six-epoch
cosine decay. Both used Tesla T4 GPUs.

| Setting | Updated Faster R-CNN | YOLOv8s |
|---|---:|---:|
| Logged epochs | 6 | 16; final epoch incomplete |
| Selected epoch | 6 | 14 |
| Recorded duration | About 4 h 1 min | About 3 h 3 min |
| LR schedule | Cosine over 6 epochs | Cosine over 3 hours |

## Initial results

| Recorded validation metric | Updated Faster R-CNN | YOLOv8s |
|---|---:|---:|
| mAP@0.5 | 0.7675 | 0.6379 |
| mAP@0.5:0.95 | 0.4204 | 0.3412 |

The recorded validation scores favour Faster R-CNN. Different training durations,
LR schedules and inference filtering prevent interpreting this as an isolated
architecture effect. A unified evaluation of the completed checkpoints is still needed.

For YOLO, confidence 0.30 maximized validation micro-F1. At that threshold, micro
precision/recall/F1 were 0.6649/0.5785/0.6187. There were 53 false-positive boxes across
1,375 empty validation images: 0.0385 FP/image, below the project target of 0.5.
Fire recall was 0.5005, making missed fire instances a priority for further inspection.

This folder contains validation results, not final YOLO test results. The new SGD
checkpoint is retained as the project's second working model. Unified test evaluation,
speed comparison and detailed error analysis are assigned to the evaluation lead.

Sources: [YOLO metrics](yolov8s_val_metrics.json), [training log](train/results.csv),
[run summary](training_summary.json), [baseline history](../baseline/training_history.json).
