"""
models/depth.py
MiDaS Monocular Depth Estimation Module.

Wraps Intel MiDaS for relative depth map generation.
Supports multiple model variants from full-size DPT to lightweight MiDaS_small.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class DepthResult:
    """Depth estimation result for a single frame."""
    depth_map: np.ndarray        # (H, W) float32, normalized 0-1
    raw_depth_map: np.ndarray    # (H, W) float32, raw MiDaS output
    inference_time_ms: float = 0.0

    @property
    def shape(self):
        return self.depth_map.shape

    def get_depth_at(self, x: int, y: int) -> float:
        """Get normalized depth value at pixel (x, y)."""
        if 0 <= y < self.depth_map.shape[0] and 0 <= x < self.depth_map.shape[1]:
            return float(self.depth_map[y, x])
        return 0.0

    def get_mean_depth_in_mask(self, mask: np.ndarray) -> float:
        """
        Compute mean depth value within a binary mask region.

        Args:
            mask: Binary mask (H, W), non-zero pixels define the region.

        Returns:
            Mean normalized depth in the masked region.
        """
        if mask.shape != self.depth_map.shape:
            mask = cv2.resize(
                mask, (self.depth_map.shape[1], self.depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        region = self.depth_map[mask > 0]
        if len(region) == 0:
            return 0.0
        return float(np.mean(region))


# ─── Depth Estimator Class ──────────────────────────────────────────

class DepthEstimator:
    """
    MiDaS-based monocular depth estimator.

    Args:
        model_type: MiDaS model variant.
            - 'DPT_Large'   : Highest accuracy, slowest.
            - 'DPT_Hybrid'  : Balanced accuracy/speed.
            - 'MiDaS_small' : Fastest, suitable for edge devices.
        device: Device string ('cuda', 'cpu', or 'auto').
    """

    VARIANTS = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "auto",
    ):
        if model_type not in self.VARIANTS:
            logger.warning(
                f"Unknown model_type '{model_type}', falling back to 'MiDaS_small'."
            )
            model_type = "MiDaS_small"

        self.model_type = model_type
        self.device = self._resolve_device(device)
        self.model: Optional[torch.nn.Module] = None
        self.transform = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Model Loading ────────────────────────────────────────────

    def load_model(self) -> "DepthEstimator":
        """
        Load MiDaS model and transforms from torch.hub.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading depth model: {self.model_type} on {self.device}")

        # Load MiDaS model from Intel's torch hub
        self.model = torch.hub.load(
            "intel-isl/MiDaS", self.model_type, trust_repo=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Load corresponding transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )

        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        logger.info("Depth model loaded successfully.")
        return self

    def is_loaded(self) -> bool:
        return self.model is not None and self.transform is not None

    # ── Inference ────────────────────────────────────────────────

    @torch.no_grad()
    def estimate_depth(self, image: np.ndarray) -> DepthResult:
        """
        Estimate relative depth map from an image.

        Args:
            image: Input image as BGR numpy array (OpenCV format).

        Returns:
            DepthResult with normalized and raw depth maps.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w = image.shape[:2]

        # Convert BGR → RGB for MiDaS
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transforms
        input_batch = self.transform(image_rgb).to(self.device)

        # Timing
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None

        import time
        t0 = time.perf_counter()

        if start is not None:
            start.record()

        # Forward pass
        prediction = self.model(input_batch)

        if end is not None:
            end.record()
            torch.cuda.synchronize()
            inference_time_ms = start.elapsed_time(end)
        else:
            inference_time_ms = (time.perf_counter() - t0) * 1000.0

        # Resize prediction to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        raw_depth = prediction.cpu().numpy()

        # Normalize to 0-1 range
        depth_min = raw_depth.min()
        depth_max = raw_depth.max()
        if depth_max - depth_min > 0:
            normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min)
        else:
            normalized_depth = np.zeros_like(raw_depth)

        return DepthResult(
            depth_map=normalized_depth.astype(np.float32),
            raw_depth_map=raw_depth.astype(np.float32),
            inference_time_ms=inference_time_ms,
        )

    # ── Utilities ────────────────────────────────────────────────

    def to_heatmap(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
        """
        Convert normalized depth map to a colored heatmap.

        Args:
            depth_map: Normalized depth map (0-1), shape (H, W).
            colormap: OpenCV colormap constant.

        Returns:
            BGR heatmap image, shape (H, W, 3), uint8.
        """
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, colormap)

    def __repr__(self) -> str:
        return f"DepthEstimator(model={self.model_type}, device={self.device})"
