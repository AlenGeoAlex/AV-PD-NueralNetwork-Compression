"""
All metric computation for YOLO model evaluation.

Computes:
  - mAP50 / mAP50-95 (global and per-class)
  - Precision / Recall / F1
  - KITTI difficulty-stratified AP (Easy / Moderate / Hard)
  - Class-specific AP with KITTI IoU thresholds (0.7 for car/large_vehicle, 0.5 for pedestrian/cyclist)
  - Inference latency (ms/image) and FPS
  - Model size (MB) and parameter count
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["pedestrian", "cyclist", "car", "large_vehicle"]
CLASS_IOU_THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.70, 3: 0.70}   # KITTI per-class IoU
DIFFICULTIES = ["Easy", "Moderate", "Hard"]
IOU_RANGE = np.linspace(0.50, 0.95, 10)   # for mAP50-95


# ─── IoU helpers ──────────────────────────────────────────────────────────────

def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes in [x1, y1, x2, y2] pixel format.
    box1: (N, 4), box2: (M, 4) → returns (N, M)
    """
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[None, :, 0]) * (box2[:, 3] - box2[None, :, 1])
    # fix: straightforward area
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> list[float]:
    """Convert normalised YOLO cx/cy/w/h to pixel x1/y1/x2/y2."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


# ─── Average Precision ────────────────────────────────────────────────────────

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute AP using the 11-point interpolation method (VOC 2007 style).
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precisions[recalls >= thr]
        ap += (np.max(prec_at_rec) if prec_at_rec.size > 0 else 0.0)
    return ap / 11.0


def compute_ap_from_matches(
    detections: list[dict],          # [{"score": float, "matched": bool}]
    n_gt: int,
    iou_threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Given a flat sorted list of detection dicts (across all images) and
    total GT count, return (AP, recall_curve, precision_curve).
    """
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for i, det in enumerate(detections):
        tp[i] = 1.0 if det["matched"] else 0.0
        fp[i] = 0.0 if det["matched"] else 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_ap(recall, precision)
    return ap, recall, precision


# ─── Per-image matching ───────────────────────────────────────────────────────

def match_predictions_to_gt(
    pred_boxes: np.ndarray,     # (N, 4) xyxy pixels
    pred_scores: np.ndarray,    # (N,)
    pred_classes: np.ndarray,   # (N,) int
    gt_boxes: np.ndarray,       # (M, 4) xyxy pixels
    gt_classes: np.ndarray,     # (M,) int
    iou_threshold: float,
    class_id: int,
) -> list[dict]:
    """
    Match predictions of a single class in a single image against GT boxes.
    Returns list of {"score": float, "matched": bool} dicts.
    """
    mask_pred = pred_classes == class_id
    mask_gt   = gt_classes   == class_id

    p_boxes   = pred_boxes[mask_pred]
    p_scores  = pred_scores[mask_pred]
    g_boxes   = gt_boxes[mask_gt]

    results: list[dict] = []

    if len(p_boxes) == 0:
        return results

    if len(g_boxes) == 0:
        return [{"score": s, "matched": False} for s in p_scores]

    iou_mat = box_iou_xyxy(p_boxes, g_boxes)   # (N_pred, N_gt)
    matched_gt = set()

    # Sort predictions by score descending
    order = np.argsort(p_scores)[::-1]
    for idx in order:
        best_iou = 0.0
        best_j   = -1
        for j in range(len(g_boxes)):
            if j in matched_gt:
                continue
            if iou_mat[idx, j] > best_iou:
                best_iou = iou_mat[idx, j]
                best_j   = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            results.append({"score": float(p_scores[idx]), "matched": True})
        else:
            results.append({"score": float(p_scores[idx]), "matched": False})

    return results


# ─── Model size / param count ─────────────────────────────────────────────────

def get_model_info(model_path: str) -> dict[str, Any]:
    """Return file size in MB and parameter count from a .pt weights file."""
    path = Path(model_path)
    size_mb = path.stat().st_size / (1024 ** 2)

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            model_obj = ckpt.get("model", None) or ckpt.get("ema", None)
        else:
            model_obj = ckpt

        if model_obj is not None and hasattr(model_obj, "parameters"):
            params_m = sum(p.numel() for p in model_obj.parameters()) / 1e6
        else:
            params_m = None
    except Exception:
        params_m = None

    return {"size_mb": round(size_mb, 2), "params_m": round(params_m, 2) if params_m else None}


# ─── Inference latency ────────────────────────────────────────────────────────

def measure_latency(model, img_size: int = 640, n_warmup: int = 50, n_runs: int = 200,
                    device: str = "cpu") -> dict[str, float]:
    """
    Measure mean/std inference latency (ms/image) and FPS.
    Uses a dummy tensor — no image loading overhead.
    """
    dummy = torch.zeros(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(dummy)

    latencies = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(latencies))
    std_ms  = float(np.std(latencies))
    fps     = 1000.0 / mean_ms

    return {
        "latency_mean_ms": round(mean_ms, 3),
        "latency_std_ms":  round(std_ms,  3),
        "fps":             round(fps, 2),
    }