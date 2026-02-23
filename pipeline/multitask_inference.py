"""
pipeline/multitask_inference.py
Unified Multi-Task Inference Pipeline.

Orchestrates detection, segmentation, and depth estimation into a
single inference call with timing and result aggregation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.detection import Detection, DetectionResult, PotholeDetector
from models.segmentation import PotholeSegmentor, SegmentationResult
from models.depth import DepthEstimator, DepthResult
from pipeline.preprocess import ImagePreprocessor
from pipeline.severity_estimator import SeverityEstimator, SeverityLabel

logger = logging.getLogger(__name__)


# ─── Pipeline Result ────────────────────────────────────────────────

@dataclass
class PotholeInfo:
    """Combined information about a single detected pothole."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    mask: Optional[np.ndarray] = None       # Binary mask for this pothole
    mask_area_pixels: int = 0
    mean_depth: float = 0.0
    severity: str = "UNKNOWN"
    severity_score: float = 0.0


@dataclass
class PipelineResult:
    """Complete result from the multi-task pipeline."""
    # Per-pothole results
    potholes: List[PotholeInfo] = field(default_factory=list)

    # Raw results from each head
    detection_result: Optional[DetectionResult] = None
    segmentation_result: Optional[SegmentationResult] = None
    depth_result: Optional[DepthResult] = None

    # Timing
    total_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    segmentation_time_ms: float = 0.0
    depth_time_ms: float = 0.0

    @property
    def fps(self) -> float:
        """Frames per second based on total pipeline time."""
        if self.total_time_ms > 0:
            return 1000.0 / self.total_time_ms
        return 0.0

    @property
    def count(self) -> int:
        return len(self.potholes)

    @property
    def timing_breakdown(self) -> Dict[str, float]:
        return {
            "preprocess_ms": self.preprocess_time_ms,
            "detection_ms": self.detection_time_ms,
            "segmentation_ms": self.segmentation_time_ms,
            "depth_ms": self.depth_time_ms,
            "total_ms": self.total_time_ms,
        }


# ─── Multi-Task Pipeline ────────────────────────────────────────────

class MultiTaskPipeline:
    """
    Unified multi-task pipeline for pothole detection, segmentation,
    depth estimation, and severity classification.

    Orchestrates all model heads and produces a combined PipelineResult.

    Args:
        detector: PotholeDetector instance (or None to skip detection).
        segmentor: PotholeSegmentor instance (or None to skip segmentation).
        depth_estimator: DepthEstimator instance (or None to skip depth).
        preprocessor: ImagePreprocessor instance (or None for no preprocessing).
        severity_estimator: SeverityEstimator instance (or None to skip severity).
        enable_detection: Whether to run the detection head.
        enable_segmentation: Whether to run the segmentation head.
        enable_depth: Whether to run the depth estimation head.
    """

    def __init__(
        self,
        detector: Optional[PotholeDetector] = None,
        segmentor: Optional[PotholeSegmentor] = None,
        depth_estimator: Optional[DepthEstimator] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        severity_estimator: Optional[SeverityEstimator] = None,
        enable_detection: bool = True,
        enable_segmentation: bool = True,
        enable_depth: bool = True,
    ):
        self.detector = detector
        self.segmentor = segmentor
        self.depth_estimator = depth_estimator
        self.preprocessor = preprocessor or ImagePreprocessor.default()
        self.severity_estimator = severity_estimator or SeverityEstimator()

        self.enable_detection = enable_detection
        self.enable_segmentation = enable_segmentation
        self.enable_depth = enable_depth

    # ── Initialization ───────────────────────────────────────────

    def load_all_models(self) -> "MultiTaskPipeline":
        """Load all enabled model heads."""
        if self.enable_detection and self.detector:
            self.detector.load_model()
        if self.enable_segmentation and self.segmentor:
            self.segmentor.load_model()
        if self.enable_depth and self.depth_estimator:
            self.depth_estimator.load_model()
        logger.info("All pipeline models loaded.")
        return self

    @classmethod
    def create_default(
        cls,
        device: str = "auto",
        det_model: str = "yolov8n.pt",
        seg_model: str = "yolov8n-seg.pt",
        depth_model: str = "MiDaS_small",
        conf_threshold: float = 0.25,
    ) -> "MultiTaskPipeline":
        """
        Factory method to create pipeline with default configurations.

        Args:
            device: 'cuda', 'cpu', or 'auto'.
            det_model: YOLOv8 detection weights.
            seg_model: YOLOv8-Seg segmentation weights.
            depth_model: MiDaS variant name.
            conf_threshold: Confidence threshold for det/seg heads.

        Returns:
            Configured MultiTaskPipeline (models not yet loaded).
        """
        return cls(
            detector=PotholeDetector(
                model_path=det_model,
                conf_threshold=conf_threshold,
                device=device,
            ),
            segmentor=PotholeSegmentor(
                model_path=seg_model,
                conf_threshold=conf_threshold,
                device=device,
            ),
            depth_estimator=DepthEstimator(
                model_type=depth_model,
                device=device,
            ),
            preprocessor=ImagePreprocessor.default(),
            severity_estimator=SeverityEstimator(),
        )

    # ── Core Inference ───────────────────────────────────────────

    def run(self, image: np.ndarray) -> PipelineResult:
        """
        Run the full multi-task inference pipeline.

        Args:
            image: Input BGR image (numpy array, uint8).

        Returns:
            PipelineResult with detections, masks, depth, and severity.
        """
        pipeline_start = time.perf_counter()
        result = PipelineResult()

        # ── Step 1: Preprocess ───────────────────────────────────
        t0 = time.perf_counter()
        processed = self.preprocessor.process(image)
        result.preprocess_time_ms = (time.perf_counter() - t0) * 1000

        # ── Step 2: Detection ────────────────────────────────────
        det_result = None
        if self.enable_detection and self.detector and self.detector.is_loaded():
            det_result = self.detector.detect(processed)
            result.detection_result = det_result
            result.detection_time_ms = det_result.inference_time_ms

        # ── Step 3: Segmentation ─────────────────────────────────
        seg_result = None
        if self.enable_segmentation and self.segmentor and self.segmentor.is_loaded():
            seg_result = self.segmentor.segment(processed)
            result.segmentation_result = seg_result
            result.segmentation_time_ms = seg_result.inference_time_ms

        # ── Step 4: Depth Estimation ─────────────────────────────
        depth_result = None
        if self.enable_depth and self.depth_estimator and self.depth_estimator.is_loaded():
            depth_result = self.depth_estimator.estimate_depth(processed)
            result.depth_result = depth_result
            result.depth_time_ms = depth_result.inference_time_ms

        # ── Step 5: Merge & Severity ─────────────────────────────
        result.potholes = self._merge_results(det_result, seg_result, depth_result)

        result.total_time_ms = (time.perf_counter() - pipeline_start) * 1000
        return result

    # ── Result Merging ───────────────────────────────────────────

    def _merge_results(
        self,
        det_result: Optional[DetectionResult],
        seg_result: Optional[SegmentationResult],
        depth_result: Optional[DepthResult],
    ) -> List[PotholeInfo]:
        """
        Merge detection, segmentation, and depth results into
        unified PotholeInfo objects. Associates masks to detections
        by IoU of bounding boxes.
        """
        potholes: List[PotholeInfo] = []

        if det_result is None or det_result.count == 0:
            # If no detections but we have segmentation masks, use those
            if seg_result and seg_result.count > 0:
                for seg_mask in seg_result.masks:
                    mean_depth = 0.0
                    if depth_result:
                        mean_depth = depth_result.get_mean_depth_in_mask(seg_mask.mask)

                    severity, score = self.severity_estimator.estimate(
                        mask_area=seg_mask.area_pixels,
                        mean_depth=mean_depth,
                    )

                    potholes.append(PotholeInfo(
                        bbox=seg_mask.bbox,
                        confidence=seg_mask.confidence,
                        mask=seg_mask.mask,
                        mask_area_pixels=seg_mask.area_pixels,
                        mean_depth=mean_depth,
                        severity=severity.value,
                        severity_score=score,
                    ))
            return potholes

        # Match each detection with the best-overlapping segmentation mask
        for det in det_result.detections:
            best_mask = None
            best_iou = 0.0

            if seg_result and seg_result.count > 0:
                for seg_mask in seg_result.masks:
                    iou = self._bbox_iou(det.bbox, seg_mask.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = seg_mask

            mask_area = best_mask.area_pixels if best_mask else 0
            mask_data = best_mask.mask if best_mask else None

            mean_depth = 0.0
            if depth_result and mask_data is not None:
                mean_depth = depth_result.get_mean_depth_in_mask(mask_data)
            elif depth_result:
                # Fallback: use depth in bounding box region
                x1, y1, x2, y2 = det.bbox
                dm = depth_result.depth_map
                h, w = dm.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                if x2c > x1c and y2c > y1c:
                    mean_depth = float(np.mean(dm[y1c:y2c, x1c:x2c]))

            severity, score = self.severity_estimator.estimate(
                mask_area=mask_area,
                mean_depth=mean_depth,
            )

            potholes.append(PotholeInfo(
                bbox=det.bbox,
                confidence=det.confidence,
                mask=mask_data,
                mask_area_pixels=mask_area,
                mean_depth=mean_depth,
                severity=severity.value,
                severity_score=score,
            ))

        return potholes

    @staticmethod
    def _bbox_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def __repr__(self) -> str:
        heads = []
        if self.enable_detection:
            heads.append("det")
        if self.enable_segmentation:
            heads.append("seg")
        if self.enable_depth:
            heads.append("depth")
        return f"MultiTaskPipeline(heads=[{', '.join(heads)}])"
