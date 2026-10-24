# YOLOv8s: retraining with the baseline time budget

Use YOLOv8s_BASELINE_TIME_KAGGLE.ipynb for the new run.
The previous three-hour experiment and its results remain separate.

## Changes

- The baseline history records elapsed_sec=14488.2: 4 h 1 min 28.2 s for training
  and validation. Reference: https://github.com/cvteam19/cv_project/blob/main/baseline/training_history.json
- This YOLO run sets time=14488.2/3600 hours. Cosine LR decay spans that budget,
  retaining initial LR 0.005 and minimum LR 0.000001.
- Outputs use /kaggle/working/yolov8s_baseline_time and a separate artifact ZIP.
- Start fresh from COCO-pretrained yolov8s.pt. Do not resume selected_best.pt:
  the previous optimizer followed a different learning-rate schedule.
- Use one Tesla T4. Equal elapsed time on different GPUs does not provide the
  same computational budget.

All remaining settings are unchanged: SGD momentum=0.9, weight_decay=0.0005,
Nesterov=False, batch=4, FP32 without AMP or gradient accumulation, clipping=10,
joint horizontal flips p=0.5, baseline input preprocessing, seed=42, no warmup
and no EMA averaging. Select the checkpoint by validation COCO mAP@0.5:0.95.

Patience remains 3. This matches the maximum available time: early stopping may
finish YOLO sooner. Report the actual duration from the new log rather than
presenting the configured limit as the observed duration. The baseline completed
six epochs without triggering early stopping.

The final in-loop validation and concluding best-checkpoint evaluation may slightly
exceed the limit. Dataset preparation and separate evaluation are additional.
Matching the time budget does not align every condition: the baseline retains
cosine decay over six epochs; YOLO retains time-based cosine decay.

## Run and collect outputs

1. Upload the notebook to Kaggle. Enable Internet and a Tesla T4; attach
   linxel/dfire-prepared.
2. Run the cells in order. Allow more than four hours of GPU quota for the
   training loop plus concluding checks.
3. Keep RUN_FINAL_TEST=False. Test results are not needed to assess validation
   changes in this rerun.
4. Save a Kaggle Version with outputs and yolov8s_baseline_time_artifacts.zip.
5. Inspect the new archive for actual duration, selected epoch, validation mAP
   and confidence threshold. Do not claim updated results or improvement before
   the run completes.

Provide the evaluation lead with the new checkpoint and final_yolo_protocol.py
for the common final evaluation. Preserve the old checkpoint as experiment
history. Do not replace published results with unverified numbers.

Status: notebook prepared; syntax and configuration checks can run locally.
The new GPU training run has not been executed. Additional time does not
guarantee better detection quality.
