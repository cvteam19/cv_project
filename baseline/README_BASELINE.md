# Baseline: Faster R-CNN (ResNet-50 + FPN)

## Model configuration
- Architecture: Faster R-CNN with ResNet-50 + FPN backbone
- Pretrained weights: COCO (`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`)
- Detection head: replaced with 3-way classifier (background, smoke, fire)
- Input resolution: 640x640

## Training setup
- Optimizer: SGD, lr=0.005, momentum=0.9, weight_decay=0.0005
- Scheduler: CosineAnnealingLR over 6 epochs (eta_min=1e-6)
- Batch size: 4
- Augmentation: ['horizontal_flip_p0.5']
- Gradient clipping: max-norm 10.0
- Early stopping: patience 3 on validation mAP
- Random seed: 42

## Dataset
- D-Fire, splits: train=14122, val=3099
- Classes: 0=smoke, 1=fire (mapped to torchvision 1=smoke, 2=fire)

## Artifacts produced
- `faster_rcnn_best.pth` — best checkpoint by mAP
- `faster_rcnn_last.pth` — last epoch (for resume)
- `training_history.json` — per-epoch metrics
- `training_curves.png` — loss curves
- `baseline_predictions.png` and `baseline_predictions_positive.png` — sample predictions (score >= 0.5)

## Limitations
- Training was performed on Kaggle free tier (Tesla T4, 16 GB).
  Google Colab enforces an idle-session timeout (~2–3 h) and a session limit (~9 h).
  A full 50-epoch run of Faster R-CNN at 640x640 does not fit in a single session,
  and hard session kills may erase checkpoints.
- As a result, the baseline was trained for a reduced number of epochs
  (6) with early stopping (patience 3).
  Validation mAP is used as the model-selection signal.
- Further training is possible via the RESUME mechanism (see `faster_rcnn_last.pth`),
  but remains bounded by the platform session limits.
- This is a platform limitation, not a method limitation

## Notes
- Validation mAP is used as the checkpointing signal inside this notebook
