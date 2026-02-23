"""
models/detection.py
YOLOv8-based Pothole Detection Module.

Wraps Ultralytics YOLOv8 for bounding-box detection of potholes.
Supports GPU/CPU auto-selection and configurable thresholds.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class Detection:
    """Single pothole detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    class_name: str = "pothole"

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Bounding box center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class DetectionResult:
    """Aggregated detection results for a single frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


# ─── Detector Class ─────────────────────────────────────────────────

class PotholeDetector:
    """
    YOLOv8-based pothole detector.

    Args:
        model_path: Path to YOLOv8 weights (.pt file).
                    Defaults to 'yolov8n.pt' (nano, fastest).
        conf_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        device: Device string ('cuda', 'cpu', or 'auto').
        img_size: Input image size for inference.
    """

    # Supported YOLOv8 model variants (detection)
    VARIANTS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        img_size: int = 640,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = self._resolve_device(device)
        self.model: Optional[YOLO] = None

    # ── Device Resolution ────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to best available device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Model Loading ────────────────────────────────────────────

    def load_model(self) -> "PotholeDetector":
        """
        Load YOLOv8 model weights.
        Downloads pretrained weights automatically if not found locally.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading detection model: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        logger.info("Detection model loaded successfully.")
        return self

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    # ── Inference ────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run pothole detection on an image.

        Args:
            image: Input image as BGR numpy array (OpenCV format).

        Returns:
            DetectionResult with list of Detection objects.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run YOLOv8 inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        # Parse results
        detections = []
        inference_time_ms = 0.0

        if results and len(results) > 0:
            result = results[0]
            # Inference timing (preprocess + inference + postprocess)
            if hasattr(result, "speed"):
                inference_time_ms = sum(result.speed.values())

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Extract bounding box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Get class name from model
                    cls_name = (
                        self.model.names.get(cls_id, "pothole")
                        if self.model.names
                        else "pothole"
                    )

                    detections.append(
                        Detection(
                            bbox=tuple(xyxy),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                    )

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
        )

    # ── Batch Inference ──────────────────────────────────────────

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        Run detection on a batch of images.

        Args:
            images: List of BGR numpy arrays.

        Returns:
            List of DetectionResult, one per image.
        """
        return [self.detect(img) for img in images]

    # ── Configuration ────────────────────────────────────────────

    def set_confidence(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.conf_threshold = max(0.0, min(1.0, threshold))

    def set_iou(self, threshold: float) -> None:
        """Update IoU threshold for NMS."""
        self.iou_threshold = max(0.0, min(1.0, threshold))

    def __repr__(self) -> str:
        return (
            f"PotholeDetector(model={self.model_path}, "
            f"device={self.device}, conf={self.conf_threshold})"
        )
