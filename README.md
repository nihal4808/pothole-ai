# 🕳️ Pothole AI — Hybrid Multi-Task Deep Learning Framework

**Real-Time Pothole Detection, Segmentation, Depth Estimation & Severity Classification**

A production-ready AI system combining YOLOv8 detection, YOLOv8-Seg segmentation, and MiDaS depth estimation into a unified pipeline with a Streamlit dashboard.

---

## Architecture

```
Input Image/Video
        │
        ▼
┌─────────────────┐
│  Preprocessing   │  Resize, Denoise, CLAHE, Gamma
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│ YOLOv8 │ │YOLOv8  │ │ MiDaS  │
│  Det   │ │  Seg   │ │ Depth  │
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    └────┬─────┴──────────┘
         ▼
┌─────────────────┐
│ Severity Estimator │  area × depth → LOW/MEDIUM/HIGH
└────────┬────────┘
         ▼
   Visualization Output
```

## Quick Start

### 1. Install Dependencies

```bash
cd pothole-ai
pip install -r requirements.txt
```

### 2. Run Inference

```bash
# Single image
python inference.py --source road_image.jpg --save-output

# Video file
python inference.py --source road_video.mp4 --save-output

# Webcam (live)
python inference.py --source 0

# With custom models
python inference.py --source image.jpg \
    --det-model weights/best_det.pt \
    --seg-model weights/best_seg.pt \
    --depth-model DPT_Hybrid

# CPU-only mode
python inference.py --source image.jpg --device cpu

# Detection only (faster)
python inference.py --source image.jpg --no-segment --no-depth
```

### 3. Launch Streamlit Dashboard

```bash
streamlit run ui/streamlit_app.py
```

---

## Training

### Prepare Dataset

Organize your dataset in YOLO format:

```
datasets/pothole/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Generate `data.yaml`:

```python
from datasets.dataset_loader import PotholeDatasetManager
manager = PotholeDatasetManager("datasets/pothole", class_names=["pothole"])
manager.generate_yaml()
```

### Train Detection Model

```bash
python train.py --task detect --data datasets/pothole/data.yaml \
    --epochs 100 --batch 16 --imgsz 640
```

### Train Segmentation Model

```bash
python train.py --task segment --data datasets/pothole/data.yaml \
    --epochs 100 --batch 16 --imgsz 640
```

### Resume Training

```bash
python train.py --task detect --resume \
    --weights runs/pothole_train/weights/last.pt \
    --data datasets/pothole/data.yaml
```

### RDD2022 Dataset

```python
from datasets.dataset_loader import prepare_rdd2022
yaml_path = prepare_rdd2022("path/to/RDD2022", "datasets/rdd2022_yolo")
```

---

## ONNX Export (Edge Deployment)

```bash
# Export all models
python export_onnx.py --all

# Export specific model
python export_onnx.py --det --det-model weights/best_det.pt
python export_onnx.py --depth --depth-model MiDaS_small

# With FP16 + simplification
python export_onnx.py --all --fp16 --simplify
```

---

## Project Structure

```
pothole-ai/
├── models/
│   ├── detection.py          # YOLOv8 detection wrapper
│   ├── segmentation.py       # YOLOv8-Seg segmentation wrapper
│   └── depth.py              # MiDaS depth estimation wrapper
├── pipeline/
│   ├── preprocess.py          # Image preprocessing (CLAHE, denoise)
│   ├── multitask_inference.py # Unified multi-task pipeline
│   └── severity_estimator.py  # Severity classification (LOW/MED/HIGH)
├── ui/
│   └── streamlit_app.py       # Streamlit dashboard
├── utils/
│   ├── visualization.py       # Overlay rendering utilities
│   └── metrics.py             # Evaluation metrics (mAP, Dice, IoU)
├── datasets/
│   └── dataset_loader.py      # Dataset loading + augmentation
├── weights/                    # Model weights directory
├── train.py                    # Training script
├── inference.py                # CLI inference script
├── export_onnx.py              # ONNX export pipeline
└── requirements.txt
```

## Performance Tips

| Optimization | Impact | How |
|---|---|---|
| Use `yolov8n` models | ↑ 2-3x FPS | Default — smallest models |
| Disable depth | ↑ 40% FPS | `--no-depth` flag |
| Use `MiDaS_small` | ↑ 2x depth speed | Default depth model |
| ONNX export | ↑ 20-50% | `python export_onnx.py --all` |
| FP16 inference | ↑ 30% on GPU | `--fp16` during export |
| Reduce `--imgsz` | ↑ FPS | `--imgsz 416` or `--imgsz 320` |

## Tech Stack

- **PyTorch** — Deep learning framework
- **Ultralytics YOLOv8** — Detection & segmentation
- **MiDaS** — Monocular depth estimation
- **OpenCV** — Image/video processing
- **Albumentations** — Training augmentation
- **Streamlit** — Web dashboard
- **ONNX / ONNX Runtime** — Edge deployment
