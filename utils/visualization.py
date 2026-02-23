"""
utils/visualization.py
Visualization Utilities for Pothole Detection.

Provides overlay rendering functions for bounding boxes, segmentation masks,
depth heatmaps, and severity labels. All overlays are toggleable.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Color Palette ──────────────────────────────────────────────────

SEVERITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "LOW":     (0, 200, 0),      # Green
    "MEDIUM":  (0, 165, 255),    # Orange
    "HIGH":    (0, 0, 255),      # Red
    "UNKNOWN": (180, 180, 180),  # Gray
}

MASK_COLORS = [
    (255, 50, 50),    # Blue-ish
    (50, 255, 50),    # Green
    (50, 50, 255),    # Red
    (255, 255, 50),   # Cyan
    (255, 50, 255),   # Magenta
    (50, 255, 255),   # Yellow
]

BBOX_COLOR = (0, 255, 0)       # Default green bounding box
BBOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


# ─── Drawing Functions ──────────────────────────────────────────────

def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    confidences: List[float],
    severities: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes with confidence and optional severity labels.

    Args:
        image: BGR image (will be modified in-place).
        bboxes: List of (x1, y1, x2, y2) bounding boxes.
        confidences: Confidence scores per box.
        severities: Optional severity labels per box.
        color: Override color for all boxes (uses severity colors if None).

    Returns:
        Annotated image.
    """
    result = image.copy()

    for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        severity = severities[i] if severities else None

        # Choose color based on severity
        box_color = color or SEVERITY_COLORS.get(severity, BBOX_COLOR)

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), box_color, BBOX_THICKNESS)

        # Build label
        label_parts = [f"{conf:.0%}"]
        if severity:
            label_parts.insert(0, severity)
        label = " | ".join(label_parts)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(
            result, label, (x1 + 3, y1 - 5),
            FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
        )

    return result


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 100, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a semi-transparent segmentation mask on the image.

    Args:
        image: BGR image.
        mask: Binary mask (H, W), non-zero pixels are the region.
        color: BGR color for the overlay.
        alpha: Transparency (0 = fully transparent, 1 = fully opaque).

    Returns:
        Image with mask overlay.
    """
    result = image.copy()
    overlay = result.copy()

    # Create colored overlay
    mask_bool = mask > 0
    overlay[mask_bool] = color

    # Blend
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    # Draw contour outline
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def overlay_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay multiple masks with different colors."""
    result = image.copy()
    for i, mask in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        result = overlay_mask(result, mask, color=color, alpha=alpha)
    return result


def draw_depth_heatmap(
    image: np.ndarray,
    depth_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_MAGMA,
) -> np.ndarray:
    """
    Overlay a depth heatmap on the image.

    Args:
        image: BGR image.
        depth_map: Normalized depth map (0-1), shape (H, W).
        alpha: Blend factor for the heatmap.
        colormap: OpenCV colormap to use.

    Returns:
        Image with depth heatmap overlay.
    """
    h, w = image.shape[:2]

    # Resize depth map to match image if needed
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to colored heatmap
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, colormap)

    # Blend
    return cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)


def draw_severity_labels(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    severities: List[str],
    scores: List[float],
) -> np.ndarray:
    """
    Draw severity labels below bounding boxes.

    Args:
        image: BGR image.
        bboxes: List of (x1, y1, x2, y2).
        severities: List of severity strings.
        scores: List of severity scores (0-1).

    Returns:
        Annotated image.
    """
    result = image.copy()

    for bbox, severity, score in zip(bboxes, severities, scores):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        color = SEVERITY_COLORS.get(severity, (200, 200, 200))

        label = f"Severity: {severity} ({score:.0%})"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)

        # Position below bounding box
        ly = y2 + th + 8
        cv2.rectangle(result, (x1, y2 + 2), (x1 + tw + 6, ly + 4), color, -1)
        cv2.putText(result, label, (x1 + 3, ly), FONT, 0.5, (255, 255, 255), 1)

    return result


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (15, 35),
) -> np.ndarray:
    """Draw FPS counter on the image."""
    result = image.copy()
    label = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
    x, y = position
    cv2.rectangle(result, (x - 5, y - th - 8), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(result, label, (x, y), FONT, 0.8, (0, 255, 0), 2)
    return result


# ─── Composite Visualization ────────────────────────────────────────

def compose_visualization(
    image: np.ndarray,
    potholes: list,
    depth_map: Optional[np.ndarray] = None,
    fps: float = 0.0,
    show_bbox: bool = True,
    show_mask: bool = True,
    show_depth: bool = True,
    show_severity: bool = True,
    show_fps: bool = True,
) -> np.ndarray:
    """
    Compose the final visualization with all toggleable overlays.

    Args:
        image: Original BGR image.
        potholes: List of PotholeInfo objects from PipelineResult.
        depth_map: Normalized depth map (optional).
        fps: Current FPS value.
        show_bbox: Toggle bounding box overlay.
        show_mask: Toggle segmentation mask overlay.
        show_depth: Toggle depth heatmap overlay.
        show_severity: Toggle severity labels.
        show_fps: Toggle FPS counter.

    Returns:
        Fully annotated visualization image.
    """
    result = image.copy()

    # Depth heatmap (drawn first, as background layer)
    if show_depth and depth_map is not None:
        result = draw_depth_heatmap(result, depth_map, alpha=0.35)

    # Segmentation masks
    if show_mask:
        masks = [p.mask for p in potholes if p.mask is not None]
        if masks:
            result = overlay_masks(result, masks, alpha=0.4)

    # Bounding boxes
    if show_bbox and potholes:
        bboxes = [p.bbox for p in potholes]
        confs = [p.confidence for p in potholes]
        sevs = [p.severity for p in potholes] if show_severity else None
        result = draw_bboxes(result, bboxes, confs, severities=sevs)

    # Severity labels (if not already shown on bbox)
    if show_severity and not show_bbox and potholes:
        bboxes = [p.bbox for p in potholes]
        sevs = [p.severity for p in potholes]
        scores = [p.severity_score for p in potholes]
        result = draw_severity_labels(result, bboxes, sevs, scores)

    # FPS counter
    if show_fps:
        result = draw_fps(result, fps)

    return result
