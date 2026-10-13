# Data Preparation and Pipeline Verification Report

## Fire and Smoke Detection in Images

## 1. Objective

The Data Lead was responsible for preparing the dataset for fire and smoke object detection, verifying dataset splits and annotation integrity, creating a list of negative-sample candidates, packaging the prepared dataset for reuse, and providing data-loading support for YOLOv8s, Faster R-CNN, and RT-DETR.

## 2. Dataset

The project uses the D-Fire dataset in YOLO annotation format, obtained from Kaggle:

https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo

**Data preparation notebook and outputs:**

https://www.kaggle.com/code/linxel/data-preparation/output

The dataset contains two classes:

- Class 0 — smoke
- Class 1 — fire

The dataset contains 21,527 images across the predefined training, validation, and test splits.

The source dataset description reports 26,557 bounding boxes: 14,692 fire boxes and 11,865 smoke boxes. During preparation, 26 invalid annotation rows were removed: 19 smoke rows and 7 fire rows.

The prepared dataset contains 26,531 valid bounding boxes:

- 14,685 fire boxes
- 11,846 smoke boxes

The original dataset was not modified. The prepared annotations were stored separately, and the original images were paired with their corresponding prepared labels.

A portable archive, `dfire_prepared_dataset.zip`, was created to package the images, prepared labels, and `data.yaml` configuration in one file.

The archive uses the following structure:

```text
dfire_prepared_dataset.zip
├── data.yaml
└── data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

The archive contains 21,527 images and 21,527 corresponding prepared label files. Its recorded size was approximately 2,906.51 MB. The ZIP integrity check completed successfully.

**Important:** The archive is stored in the Kaggle working environment until downloaded or saved through a supported persistence mechanism. A file in `/kaggle/working` is not automatically a permanent, independently accessible dataset.

## 3. Dataset Audit

The audit checked image counts, label-file correspondence, annotation formats, class IDs, normalized coordinates, and bounding boxes crossing image boundaries.

### Prepared Dataset Statistics

| Split | Images | Empty labels | Valid boxes | Smoke boxes | Fire boxes |
|---|---:|---:|---:|---:|---:|
| Train | 14,122 | 6,458 | 17,419 | 7,788 | 9,631 |
| Validation | 3,099 | 1,375 | 3,931 | 1,755 | 2,176 |
| Test | 4,306 | 2,008 | 5,181 | 2,303 | 2,878 |
| **Total** | **21,527** | **9,841** | **26,531** | **11,846** | **14,685** |

The audit and preparation identified the following:

- 26 invalid annotation rows were removed from the prepared copy.
- 371 bounding boxes crossing image boundaries were clipped to the visible image area.
- The audit reported no missing label files, orphan label files, or unreadable images.
- No exact image duplicates were found across splits in the reported MD5 comparison.
- A subsequent validation of the prepared labels found zero malformed rows, invalid class IDs, non-finite coordinate values, or coordinates outside the accepted normalized range.

Selected problematic annotations were visually inspected. The reviewed examples showed visible smoke or fire, including objects extending beyond image boundaries. Clipping retained the portion of each affected bounding box inside the image.

The original split assignments were preserved. Annotation changes were made only in the separate prepared copy.

## 4. Annotation Preparation

The prepared dataset follows the YOLO annotation format and retains the original class mapping:

- `0` — smoke
- `1` — fire

The preparation procedure:

- Removed 26 invalid annotation rows.
- Clipped 371 boundary-crossing bounding boxes to the visible image area.
- Preserved valid class IDs and normalized YOLO coordinates.
- Retained empty label files.
- Preserved the original dataset and its train, validation, and test assignments.
- Packaged the original images together with their corresponding prepared label files.

The prepared annotations were checked programmatically for the expected five-field YOLO format, valid class IDs, finite numeric values, normalized coordinates, and positive bounding-box width and height.

These checks establish structural validity, but do not guarantee that every bounding box perfectly matches the visible object.

Empty label files are treated as images without annotated objects. This interpretation is based on the annotations and does not independently prove that every such image is free of visible smoke or fire.

## 5. Negative-Sample Candidates

The file `negative_samples_train.csv` lists 6,458 training images with empty prepared labels. All candidate image paths were found successfully.

A random sample of 12 images was visually inspected, and no visible fire or smoke was identified in those examples.

These images are negative-sample candidates based on their annotations and the limited visual inspection. The entire list has not been manually reviewed, so it should not be described as a fully verified hard-negative dataset.

Hard negatives are challenging negative examples containing confusing backgrounds or objects, such as clouds, fog, artificial lighting, and bright or reflective objects that could be mistaken for fire or smoke.

Only the training split was used to generate the candidate list.

## 6. Data Loading and Pipeline Verification

### YOLOv8s

The Ultralytics dataset configuration was created at:

`/kaggle/working/dfire_prepared/data.yaml`

A short one-epoch smoke test using a small fraction of the training data completed successfully.

**Result:** PASSED — tiny smoke test completed.

### RT-DETR

RT-DETR was tested using the Ultralytics pipeline and dataset configuration. A short one-epoch smoke test completed successfully.

**Result:** PASSED — tiny smoke test completed.

### Faster R-CNN

A PyTorch `FireSmokeDataset` and `DataLoader` were implemented for Faster R-CNN.

The loader returns:

- RGB image tensors.
- Bounding boxes in absolute `xyxy` pixel coordinates.
- Class labels following the Torchvision detection convention: `0` for background, `1` for smoke, and `2` for fire.
- Image IDs, box areas, and crowd flags.

The dataset returns images at their original dimensions. The Torchvision detection model handles resizing internally.

In the recorded DataLoader test, a batch of four images was loaded successfully. The first image had tensor shape `(3, 610, 1086)`, and its target contained zero bounding boxes.

An empty target is valid for an image with no annotated objects. However, the first image alone does not establish that all dataset targets are correct.

**Result:** PASSED — dataset and batch checks completed.

The YOLOv8s and RT-DETR smoke tests verified basic pipeline execution, not final model performance. The Faster R-CNN test verified data loading and batch compatibility, not successful model training.

## 7. Data Configuration, Packaging, and Documentation

The preparation workflow produced the following artifacts:

| Artifact | Purpose |
|---|---|
| `data.yaml` | Ultralytics dataset configuration |
| `dataset_summary.json` | Prepared dataset statistics |
| `data_lead_final_summary.json` | Summary of preparation results and pipeline checks |
| `removed_label_rows.csv` | Invalid annotation rows removed during preparation |
| `negative_samples_train.csv` | Training images with empty prepared labels |
| `README_DATA_LEAD.md` | Dataset preparation and pipeline documentation |
| `data-preparation.ipynb` | Notebook containing the preparation workflow and data-loading code |
| `dfire_prepared_dataset.zip` | Portable archive containing images, prepared labels, and `data.yaml` |

The final archive was created at:

`/kaggle/working/dfire_prepared_dataset.zip`

The archive verification reported:

- Images: 21,527
- Prepared labels: 21,527
- ZIP integrity check: passed (`testzip()` returned `None`)

The archive's `data.yaml` uses relative paths:

- `data/train/images`
- `data/val/images`
- `data/test/images`

This configuration is intended to be used from the extracted archive root. After extraction, the dataset can be connected to a compatible training notebook without relying on the original `/kaggle/working/dfire_prepared/data` path.

The Faster R-CNN `FireSmokeDataset` and its detection collate function are available in the Data Lead notebook.

## 8. Split and Evaluation Policy

The dataset splits are used as follows:

- **Train:** model fitting.
-
