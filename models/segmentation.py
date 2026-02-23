"""
models/segmentation.py
YOLOv8-Seg Pothole Segmentation Module.

Wraps Ultralytics YOLOv8-Seg for pixel-level pothole mask generation.
Includes mask post-processing with morphological operations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class SegmentationMask:
    """Single pothole segmentation result."""
    mask: np.ndarray           # Binary mask (H, W), dtype=uint8, values 0 or 255
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    class_name: str = "pothole"

    @property
    def area_pixels(self) -> int:
        """Number of pixels in the segmented region."""
        return int(np.sum(self.mask > 0))

    @property
    def contour(self) -> Optional[np.ndarray]:
        """Largest contour of the mask."""
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            return max(contours, key=cv2.contourArea)
        return None


@dataclass
class SegmentationResult:
    """Aggregated segmentation results for a single frame."""
    masks: List[SegmentationMask] = field(default_factory=list)
    combined_mask: Optional[np.ndarray] = None  # Union of all masks
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.masks)


# ─── Segmentor Class ────────────────────────────────────────────────

class PotholeSegmentor:
    """
    YOLOv8-Seg based pothole segmentor.

    Args:
        model_path: Path to YOLOv8-Seg weights (.pt file).
                    Defaults to 'yolov8n-seg.pt' (nano-seg, fastest).
        conf_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        device: Device string ('cuda', 'cpu', or 'auto').
        img_size: Input image size for inference.
        morphology_kernel: Kernel size for mask cleanup.
    """

    VARIANTS = [
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
        "yolov8l-seg.pt", "yolov8x-seg.pt",
    ]

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        img_size: int = 640,
        morphology_kernel: int = 5,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = self._resolve_device(device)
        self.morphology_kernel = morphology_kernel
        self.model: Optional[YOLO] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to best available device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Model Loading ────────────────────────────────────────────

    def load_model(self) -> "PotholeSegmentor":
        """
        Load YOLOv8-Seg model weights.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading segmentation model: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        logger.info("Segmentation model loaded successfully.")
        return self

    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Mask Post-processing ─────────────────────────────────────

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the segmentation mask.
        Removes small noise and fills small holes.

        Args:
            mask: Binary mask (H, W), uint8.

        Returns:
            Cleaned binary mask.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morphology_kernel, self.morphology_kernel)
        )
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    # ── Inference ────────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Run pothole segmentation on an image.

        Args:
            image: Input image as BGR numpy array (OpenCV format).

        Returns:
            SegmentationResult with per-instance masks and combined mask.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w = image.shape[:2]

        # Run YOLOv8-Seg inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        seg_masks = []
        combined = np.zeros((h, w), dtype=np.uint8)
        inference_time_ms = 0.0

        if results and len(results) > 0:
            result = results[0]

            if hasattr(result, "speed"):
                inference_time_ms = sum(result.speed.values())

            # Check if masks are available
            if result.masks is not None and result.boxes is not None:
                masks_data = result.masks.data.cpu().numpy()  # (N, H_mask, W_mask)
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get bounding box
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = (
                        self.model.names.get(cls_id, "pothole")
                        if self.model.names
                        else "pothole"
                    )

                    # Resize mask to original image dimensions
                    raw_mask = masks_data[i]
                    mask_resized = cv2.resize(
                        raw_mask, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                    # Binarize
                    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                    # Clean up
                    binary_mask = self._clean_mask(binary_mask)

                    seg_masks.append(
                        SegmentationMask(
                            mask=binary_mask,
                            bbox=tuple(xyxy),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                    )

                    # Accumulate into combined mask
                    combined = cv2.bitwise_or(combined, binary_mask)

        return SegmentationResult(
            masks=seg_masks,
            combined_mask=combined,
            inference_time_ms=inference_time_ms,
        )

    # ── Configuration ────────────────────────────────────────────

    def set_confidence(self, threshold: float) -> None:
        self.conf_threshold = max(0.0, min(1.0, threshold))

    def set_iou(self, threshold: float) -> None:
        self.iou_threshold = max(0.0, min(1.0, threshold))

    def __repr__(self) -> str:
        return (
            f"PotholeSegmentor(model={self.model_path}, "
            f"device={self.device}, conf={self.conf_threshold})"
        )
