"""
utils/metrics.py
Evaluation Metrics for Pothole Detection and Segmentation.

Provides IoU, mAP, Dice coefficient, pixel accuracy, and an evaluation
runner for validating model performance on test datasets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Bounding Box Metrics ───────────────────────────────────────────

def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) format.
        box2: (x1, y1, x2, y2) format.

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Compute IoU matrix between predicted and ground-truth boxes.

    Returns:
        Matrix of shape (num_pred, num_gt) with IoU values.
    """
    matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            matrix[i, j] = compute_iou(pb, gb)
    return matrix


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """
    Compute Average Precision using 11-point interpolation (PASCAL VOC style).

    Args:
        precisions: Array of precision values.
        recalls: Array of recall values.

    Returns:
        AP value.
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if mask.any():
            ap += np.max(precisions[mask])
    return ap / 11.0


def compute_map(
    pred_boxes_list: List[List[Tuple[int, int, int, int]]],
    pred_scores_list: List[List[float]],
    gt_boxes_list: List[List[Tuple[int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) across a dataset.

    Args:
        pred_boxes_list: Predicted boxes per image.
        pred_scores_list: Confidence scores per image.
        gt_boxes_list: Ground-truth boxes per image.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dictionary with 'mAP', 'precision', 'recall'.
    """
    all_preds = []  # (score, is_tp, image_idx)
    total_gt = 0

    for img_idx, (preds, scores, gts) in enumerate(
        zip(pred_boxes_list, pred_scores_list, gt_boxes_list)
    ):
        total_gt += len(gts)
        matched_gt = set()

        # Sort predictions by confidence
        sorted_indices = np.argsort(scores)[::-1]

        for pred_idx in sorted_indices:
            pred_box = preds[pred_idx]
            score = scores[pred_idx]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
            if is_tp:
                matched_gt.add(best_gt_idx)

            all_preds.append((score, is_tp))

    if not all_preds or total_gt == 0:
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0}

    # Sort all predictions by score
    all_preds.sort(key=lambda x: x[0], reverse=True)

    tp_cumsum = np.cumsum([int(p[1]) for p in all_preds])
    fp_cumsum = np.cumsum([int(not p[1]) for p in all_preds])

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt

    ap = compute_ap(precisions, recalls)

    return {
        "mAP": float(ap),
        "precision": float(precisions[-1]) if len(precisions) > 0 else 0.0,
        "recall": float(recalls[-1]) if len(recalls) > 0 else 0.0,
    }


# ─── Segmentation Metrics ──────────────────────────────────────────

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient (F1) between predicted and ground-truth masks.

    Args:
        pred_mask: Binary prediction mask.
        gt_mask: Binary ground-truth mask.

    Returns:
        Dice coefficient in [0, 1].
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()

    if total == 0:
        return 1.0  # Both empty = perfect match
    return 2.0 * intersection / total


def compute_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute pixel-level accuracy between predicted and ground-truth masks.

    Args:
        pred_mask: Binary prediction mask.
        gt_mask: Binary ground-truth mask.

    Returns:
        Accuracy in [0, 1].
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    correct = np.equal(pred_bool, gt_bool).sum()
    total = pred_mask.size

    return correct / total if total > 0 else 0.0


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        pred_mask: Binary prediction mask.
        gt_mask: Binary ground-truth mask.

    Returns:
        IoU in [0, 1].
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    return intersection / union if union > 0 else 0.0


# ─── Evaluation Runner ─────────────────────────────────────────────

def evaluate_detection(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate detection performance on a set of results.

    Args:
        predictions: List of dicts with 'boxes' and 'scores' keys.
        ground_truths: List of dicts with 'boxes' key.
        iou_threshold: IoU threshold for TP matching.

    Returns:
        Dictionary of metrics including mAP, precision, recall.
    """
    pred_boxes = [p.get("boxes", []) for p in predictions]
    pred_scores = [p.get("scores", []) for p in predictions]
    gt_boxes = [g.get("boxes", []) for g in ground_truths]

    return compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold)


def evaluate_segmentation(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate segmentation performance.

    Args:
        pred_masks: List of predicted binary masks.
        gt_masks: List of ground-truth binary masks.

    Returns:
        Dictionary with mean Dice, IoU, and pixel accuracy.
    """
    dices = []
    ious = []
    accs = []

    for pred, gt in zip(pred_masks, gt_masks):
        dices.append(compute_dice(pred, gt))
        ious.append(compute_mask_iou(pred, gt))
        accs.append(compute_pixel_accuracy(pred, gt))

    return {
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_pixel_accuracy": float(np.mean(accs)) if accs else 0.0,
    }
