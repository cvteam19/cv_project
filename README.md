# Fire and Smoke Detection: Faster R-CNN vs YOLOv8s

This repository provides a complete, reproducible pipeline for detecting visible fire and smoke in images. It compares a two-stage detector, **Faster R-CNN (ResNet-50 + FPN)**, and a one-stage detector, **YOLOv8s**, using the prepared D-Fire dataset.

---

### Quick Start (Docker)

```bash
#Powershell

# 1. Build the image (takes ~2-3 minutes)
docker build -t fire-detection .

# 2. Run the demo (inference + metrics)
docker run --rm -v ${PWD}/results:/app/results fire-detection
```

**What it does:**
The container runs `demo_inference.py` (draws bounding boxes on test images) and `eval_on_demo.py` (computes mAP, Precision, Recall). The results (images + JSON) will appear in the results/ folder on your host machine.


### Alternative: Local Python 
If you prefer to run locally, you need Python 3.11+
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/demo_inference.py --images demo_test/images --out results
python scripts/eval_on_demo.py --data demo_test --out results
```

---



### Training & Full Evaluation

To fully reproduce the training process, a GPU is required (a Tesla T4 was used originally). Training took approximately 4-5 hours for each model.

**Faster R-CNN:**
```bash
python -m src.train --data-root data/data --config configs/baseline_config.yaml
```

**YOLOv8s:**
```bash
python yolo/main_model.py --data data/data/data.yaml --epochs 50
```

**Full Test Set Evaluation:**
To recalculate the metrics on the full 4,306 test images (requires downloading the ~2GB dataset):
```bash
python scripts/setup_data.py --out data
python src/evaluate.py --model-type faster_rcnn --ckpt checkpoints/faster_rcnn/faster_rcnn_best.pth --data-root data/data --split test
```

---

### Live demo notebook: `run_demo.ipynb`

Demonstration notebook that runs **both** models (YOLOv8s and Faster R-CNN)
on 6 representative D-Fire test images and produces side-by-side visualizations
(Ground truth | YOLOv8s | Faster R-CNN).

**What it does:**

- Picks 6 test images by category: fire+smoke, smoke-only, fire-only, small fire,
  bright scene, cloud scene.
- Runs both models on each image.
- Saves 3-panel visualizations to /kaggle/working/demo/.
- Writes demo_summary.json with per-image inference times and box counts.

**How to run:**

1. Open run_demo.ipynb in a **Kaggle Notebook** (GPU T4, Internet ON).
2. Add **two inputs**:

    - **Dataset** — linxel/dfire-prepared
  
 
    - **Checkpoints dataset** (/kaggle/input/datasets/c3m3nt/cv-project-checkpoints) containing:
    
    - yolo_selected_best.pt
     
    - faster_rcnn_best.pth
     
    - final_yolo_protocol.py
     
    - budget_protocol.py
3. If your checkpoint dataset is **not** at `/kaggle/input/datasets/c3m3nt/cv-project-checkpoints`, edit CKPT_DIR in cell 2.
4. **Run All**. Results land in /kaggle/working/demo/.

**Runtime**: ~3 min on T4.



---


### Results 

**Dataset:**

Official evaluation was conducted on the full test set of **4,306 images** on a Tesla T4 GPU. Detailed metrics are available in `evaluation/results/*.json`.

<div align="center">

| Metric | Faster R-CNN | YOLOv8s |
| :---: | :---: | :---: |
| mAP@0.5 | 0.7687 | 0.6348 |
| mAP@0.5:0.95 | 0.4112 | 0.3423 |
| Smoke F1 | 0.756 | 0.697 |
| Fire F1 | 0.633 | 0.556 |
| FP / image | 0.697 | 0.223 |
| Speed (T4 GPU) | 23.3 img/s | 76.6 img/s |

</div>

**False Positives Analysis:** 
The dataset includes 9,841 empty-label images (category `AoF` - Absence of Fire) to measure false positives caused by clouds, fog, lighting, and bright objects. At their selected operating thresholds, Faster R-CNN produced more false positives (0.697 FP/img), exceeding the target of 0.5. YOLOv8s was more conservative (0.223 FP/img), meeting the requirement. Representative failure cases (clouds mistaken for smoke, sunsets mistaken for fire) are saved in `evaluation/figures/`.

---

**Demo_test sample:**

To prove that the pipeline works end-to-end, the Docker container can run a quick evaluation on the 23 labeled images from `demo_test/` (50 ground-truth boxes). Despite the small sample size, the results closely match the official test set metrics:

<div align="center">

| Metric | Faster R-CNN | YOLOv8s |
| :---: | :---: | :---: |
| mAP@0.5 | 0.7554 | 0.6648 |
| mAP@0.5:0.95 | 0.4242 | 0.3966 |
| Smoke F1 | 0.7000 | 0.7857 |
| Fire F1 | 0.5490 | 0.5263 |

</div>

*Note: Small subsets naturally have higher variance than the full 4,306-image test set. These results confirm the correctness of the code, preprocessing, and evaluation pipeline.*

---

### Team & Contributions

Team 19, Innopolis University.

<div align="center">

| Member | Role & Contribution |
|---|---|
| Daria Potapova | Team leader & Main model lead — YOLOv8-s training and tuning, hard-negative fine-tuning ablation, resolution and threshold sensitivity experiments |
| Aksiniia Miasoedova | Data lead — dataset download and split verification, dataloader setup for all models, curation of the hard-negative subset |
| Maria Karpova | Baseline lead — Faster R-CNN (ResNet-50 + FPN) implementation and training, head replacement for 2 classes, baseline metrics on validation/test, environment and runtime logging |
| Ekaterina Kuznetsova | Evaluation and analysis lead — unified evaluation scripts, false-positive categorization, false-negative inspection, all tables and figures |

</div>

---


### Environment & Dependencies

- Python 3.11+
- PyTorch 2.15 (CPU version is sufficient for the demo)
- Ultralytics 8.3.221 (strictly pinned, as the custom YOLO preprocessing pipeline relies on this version)
- All dependencies are pinned in `requirements.txt`.