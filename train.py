"""
train.py
Training Script for Pothole Detection & Segmentation.

Trains YOLOv8 detection or YOLOv8-Seg segmentation models using
Ultralytics training API with configurable hyperparameters.

Usage:
    # Train detection model
    python train.py --task detect --data datasets/pothole/data.yaml --epochs 100

    # Train segmentation model
    python train.py --task segment --data datasets/pothole/data.yaml --epochs 100

    # Resume training
    python train.py --task detect --resume --weights runs/detect/train/weights/last.pt
"""

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Default Hyperparameters ────────────────────────────────────────

DEFAULTS = {
    "detect": {
        "model": "yolov8n.pt",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "optimizer": "auto",
        "augment": True,
    },
    "segment": {
        "model": "yolov8n-seg.pt",
        "epochs": 100,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "optimizer": "auto",
        "augment": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 models for pothole detection/segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--task", type=str, choices=["detect", "segment"], default="detect",
        help="Training task: 'detect' for bounding boxes, 'segment' for masks.",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML file (Ultralytics format).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Base model weights. Default: yolov8n.pt (detect) / yolov8n-seg.pt (segment).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--lr0", type=float, default=None,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="auto",
        help="Optimizer: SGD, Adam, AdamW, auto (default: auto).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', '0', '0,1', or 'auto'.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of data loading workers (default: 8).",
    )
    parser.add_argument(
        "--project", type=str, default="runs",
        help="Project directory for saving results.",
    )
    parser.add_argument(
        "--name", type=str, default="pothole_train",
        help="Experiment name.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to specific weights for resuming (used with --resume).",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--save-period", type=int, default=10,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable YOLOv8 built-in augmentation.",
    )

    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Execute the training pipeline."""

    # Resolve defaults
    defaults = DEFAULTS[args.task]
    model_path = args.model or defaults["model"]
    epochs = args.epochs or defaults["epochs"]
    batch = args.batch or defaults["batch"]
    lr0 = args.lr0 or defaults["lr0"]

    # Validate dataset
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset YAML not found: {data_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"  Pothole AI Training")
    logger.info(f"  Task:      {args.task}")
    logger.info(f"  Model:     {model_path}")
    logger.info(f"  Dataset:   {data_path}")
    logger.info(f"  Epochs:    {epochs}")
    logger.info(f"  Batch:     {batch}")
    logger.info(f"  Image Size: {args.imgsz}")
    logger.info(f"  Device:    {args.device}")
    logger.info("=" * 60)

    # Load model
    if args.resume and args.weights:
        logger.info(f"Resuming from: {args.weights}")
        model = YOLO(args.weights)
    else:
        model = YOLO(model_path)

    # Train
    training_args = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": args.imgsz,
        "lr0": lr0,
        "optimizer": args.optimizer,
        "device": args.device if args.device != "auto" else None,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "save_period": args.save_period,
        "resume": args.resume,
        "augment": not args.no_augment,
        "verbose": True,
        "exist_ok": True,
    }

    # Filter None values
    training_args = {k: v for k, v in training_args.items() if v is not None}

    results = model.train(**training_args)

    # Log results
    logger.info("=" * 60)
    logger.info("  Training Complete!")
    logger.info(f"  Results saved to: {args.project}/{args.name}")
    logger.info("=" * 60)

    # Validate final model
    logger.info("Running validation on best model...")
    val_results = model.val()
    logger.info(f"  mAP50:    {val_results.box.map50:.4f}")
    logger.info(f"  mAP50-95: {val_results.box.map:.4f}")

    if args.task == "segment" and hasattr(val_results, "seg"):
        logger.info(f"  Seg mAP50: {val_results.seg.map50:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
