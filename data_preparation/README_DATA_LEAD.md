# Fire and Smoke Detection — Data Preparation

## Dataset
D-Fire YOLO-ready dataset:
https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo

Prepared dataset path:
`/kaggle/working/dfire_prepared/data`

Ultralytics configuration:
`/kaggle/working/dfire_prepared/data.yaml`

## Fixed dataset splits

| Split | Images | Empty labels | Boxes | Smoke boxes | Fire boxes |
|---|---:|---:|---:|---:|---:|
| Train | 14122 | 6458 | 17419 | 7788 | 9631 |
| Validation | 3099 | 1375 | 3931 | 1755 | 2176 |
| Test | 4306 | 2008 | 5181 | 2303 | 2878 |
| Total | 21527 | 9841 | 26531 | 11846 | 14685 |

Class IDs in YOLO labels:
- `0`: smoke
- `1`: fire

## Label preparation

- The original dataset was not modified.
- A prepared working copy was created under `dfire_prepared/data`.
- 26 invalid annotation rows were removed.
- 371 boxes crossing image boundaries were clipped to the visible image area.
- The prepared labels were checked for valid coordinates and class IDs.
- Empty label files were retained as images without annotated objects.

## Negative samples

A CSV of train images with empty prepared labels is available at:

`/kaggle/working/dfire_prepared/negative_samples_train.csv`

Number of candidates: 6458

These are treated as negative examples based on the dataset annotations.
A sample of images was visually inspected; not every image was checked
individually. The CSV does not establish that every image is a hard negative.
Hard negatives specifically contain confusing backgrounds such as clouds,
fog, bright objects, or artificial lighting.

## Model data loading

- YOLOv8s: use the Ultralytics configuration in `data.yaml`.
- RT-DETR: use the same Ultralytics configuration in `data.yaml`.
- Faster R-CNN: use the PyTorch `FireSmokeDataset` and detection collate
  function from the Data Lead notebook. Torchvision class IDs are shifted:
  smoke = 1, fire = 2, background = 0.
- Images may have different original dimensions. The Faster R-CNN dataset
  returns images in their original resolution; the torchvision detection
  model applies its own resizing pipeline.

## Split policy

- Train is used for model fitting.
- Validation is used for model selection and tuning.
- Test is reserved for final evaluation and must not be used for training
  or tuning.

## Important limitations

- A successful data audit does not prove model quality.
- The negative candidate list was generated from empty labels, not from
  a manual review of every image.
- A loader smoke test is not a substitute for full training or evaluation.
- Record mAP, precision, recall, and false-positive categories during
  the model evaluation stage.
