# Fire and Smoke Detection — Data Preparation

## Dataset

**Original D-Fire YOLO-ready dataset:**  
https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo

**Prepared dataset (published on Kaggle):**  
https://www.kaggle.com/datasets/linxel/dfire-prepared

**Data preparation notebook and outputs:**  
https://www.kaggle.com/code/linxel/data-preparation/output

Prepared dataset path in the original Kaggle session:  
`/kaggle/working/dfire_prepared/data`

Ultralytics configuration in the original Kaggle session:  
`/kaggle/working/dfire_prepared/data.yaml`

**Important:** The prepared dataset is published separately on Kaggle so it can be added as an input to other notebooks. The paths under `/kaggle/working/` are specific to the original Kaggle runtime and are not automatically available in another notebook. When using the published dataset, check its directory structure and update the dataset configuration path if necessary.

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

A CSV of train images with empty prepared labels is available in the data preparation outputs:

`negative_samples_train.csv`

Original Kaggle working path:  
`/kaggle/working/dfire_prepared/negative_samples_train.csv`

Number of candidates: 6458.

These are treated as negative-sample candidates based on the dataset annotations. A sample of images was visually inspected; not every image was checked individually. The CSV does not establish that every image is a hard negative.

## Model data loading

- **YOLOv8s:** use the Ultralytics configuration in `data.yaml`.
- **RT-DETR:** use the same Ultralytics configuration in `data.yaml`.
- **Faster R-CNN:** use the PyTorch `FireSmokeDataset` and detection collate function from the Data Lead notebook. Torchvision class IDs are shifted: smoke = 1, fire = 2, background = 0.
- Images may have different original dimensions. The Faster R-CNN dataset returns images in their original resolution; the Torchvision detection model applies its own resizing pipeline.

## Split policy

- **Train** is used for model fitting.
- **Validation** is used for model selection and tuning.
- **Test** is reserved for final evaluation and must not be used for training or tuning.

## Important limitations

- A successful data audit
