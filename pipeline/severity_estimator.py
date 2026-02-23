"""
pipeline/severity_estimator.py
Pothole Severity Estimation Module.

Classifies pothole severity (LOW / MEDIUM / HIGH) based on
segmented mask area and mean depth from MiDaS.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class SeverityLabel(Enum):
    """Pothole severity classification."""
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class SeverityThresholds:
    """
    Configurable thresholds for severity classification.

    The severity score is computed as:
        score = area_weight * norm_area + depth_weight * norm_depth

    Where:
        norm_area  = clamp(area / area_max, 0, 1)
        norm_depth = clamp(mean_depth, 0, 1)  (already 0-1 from MiDaS)

    Thresholds on the composite score:
        score < low_threshold   → LOW
        score < high_threshold  → MEDIUM
        score >= high_threshold → HIGH
    """
    # Normalization
    area_max: int = 50000          # Max expected pothole area in pixels (for normalization)

    # Weights
    area_weight: float = 0.4
    depth_weight: float = 0.6

    # Score thresholds
    low_threshold: float = 0.3
    high_threshold: float = 0.6


class SeverityEstimator:
    """
    Estimates pothole severity from mask area and depth.

    Combines normalized area (from segmentation) with mean depth
    (from MiDaS) using a weighted score, then classifies into
    LOW / MEDIUM / HIGH categories.

    Args:
        thresholds: SeverityThresholds for tuning classification.
    """

    def __init__(self, thresholds: SeverityThresholds = None):
        self.thresholds = thresholds or SeverityThresholds()

    def estimate(
        self,
        mask_area: int,
        mean_depth: float,
    ) -> Tuple[SeverityLabel, float]:
        """
        Estimate severity for a single pothole.

        Args:
            mask_area: Number of pixels in the pothole segmentation mask.
            mean_depth: Mean normalized depth (0-1) in the pothole region.

        Returns:
            Tuple of (SeverityLabel, composite_score).
        """
        if mask_area <= 0 and mean_depth <= 0:
            return SeverityLabel.UNKNOWN, 0.0

        # Normalize area to [0, 1]
        norm_area = min(1.0, mask_area / max(1, self.thresholds.area_max))

        # Depth is already normalized [0, 1] from MiDaS
        norm_depth = max(0.0, min(1.0, mean_depth))

        # Composite severity score
        score = (
            self.thresholds.area_weight * norm_area
            + self.thresholds.depth_weight * norm_depth
        )
        score = max(0.0, min(1.0, score))

        # Classify
        if score < self.thresholds.low_threshold:
            label = SeverityLabel.LOW
        elif score < self.thresholds.high_threshold:
            label = SeverityLabel.MEDIUM
        else:
            label = SeverityLabel.HIGH

        return label, score

    def estimate_batch(
        self,
        areas: list,
        depths: list,
    ) -> list:
        """
        Estimate severity for multiple potholes.

        Args:
            areas: List of mask areas (int).
            depths: List of mean depths (float).

        Returns:
            List of (SeverityLabel, score) tuples.
        """
        return [
            self.estimate(area, depth)
            for area, depth in zip(areas, depths)
        ]

    def __repr__(self) -> str:
        t = self.thresholds
        return (
            f"SeverityEstimator("
            f"low<{t.low_threshold}, high>={t.high_threshold}, "
            f"area_w={t.area_weight}, depth_w={t.depth_weight})"
        )
