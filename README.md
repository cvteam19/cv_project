# cv_project

---

---

## Live demo notebook (`run_demo.ipynb`)

Demonstration notebook that runs **both** models (YOLOv8s and Faster R-CNN)
on 6 representative D-Fire test images and produces side-by-side visualizations
(Ground truth | YOLOv8s | Faster R-CNN).

### What it does

- Picks 6 test images by category: fire+smoke, smoke-only, fire-only, small fire,
  bright scene, cloud scene.
- Runs both models on each image.
- Saves 3-panel visualizations to `/kaggle/working/demo/`.
- Writes `demo_summary.json` with per-image inference times and box counts.

### How to run

1. Open `run_demo.ipynb` in a **Kaggle Notebook** (GPU T4, Internet ON).
2. Add **two inputs**:
   - **Dataset** — `linxel/dfire-prepared`
   - **Checkpoints dataset** (`/kaggle/input/datasets/c3m3nt/cv-project-checkpoints`) containing:
     - `yolo_selected_best.pt`
     - `faster_rcnn_best.pth`
     - `final_yolo_protocol.py`
     - `budget_protocol.py`
3. If your checkpoint dataset is **not** at
   `/kaggle/input/datasets/c3m3nt/cv-project-checkpoints`, edit `CKPT_DIR` in cell 2.
4. **Run All**. Results land in `/kaggle/working/demo/`.

**Runtime**: ~3 min on T4.

