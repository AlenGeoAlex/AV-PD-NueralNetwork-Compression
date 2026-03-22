"""
meta_loader.py — Load and index per-image metadata sidecars.

The merged dataset stores a JSON sidecar next to every image in
  <dataset_root>/meta/test/<stem>.json

This module reads those files and returns structured GT annotations
with difficulty labels and per-class KITTI IoU thresholds baked in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DIFFICULTIES = ["Easy", "Moderate", "Hard", "Ignore"]

# KITTI-style IoU threshold per class (0=pedestrian, 1=cyclist, 2=car, 3=large_vehicle)
CLASS_IOU_THRESHOLDS = {0: 0.50, 1: 0.50, 2: 0.70, 3: 0.70}

# Fallback IoU for classes not in the KITTI map (EuroCity-only classes if any)
DEFAULT_IOU = 0.50


def load_meta_sidecar(meta_path: Path) -> dict[str, Any]:
    """Load a single JSON sidecar. Returns {} if file is missing or malformed."""
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def get_gt_for_image(
    meta: dict[str, Any],
    img_w: int,
    img_h: int,
) -> dict[str, Any]:
    """
    Parse a meta sidecar into structured GT data for evaluation.

    Returns:
    {
        "boxes_xyxy":   np.ndarray  (N, 4)  — pixel coordinates
        "classes":      np.ndarray  (N,)    — int class IDs
        "difficulties": list[str]   (N,)    — "Easy"/"Moderate"/"Hard"/"Ignore"
        "iou_thresholds": np.ndarray (N,)   — per-annotation IoU threshold
        "sources":      list[str]   (N,)    — "kitti" or "eurocity"
    }
    """
    boxes, classes, difficulties, iou_thresholds, sources = [], [], [], [], []

    for ann in meta.get("annotations", []):
        # Skip DontCare and null-class annotations
        if ann.get("is_dontcare", False):
            continue
        class_id = ann.get("class_id")
        if class_id is None:
            continue
        # Skip Ignore difficulty (these are excluded from GT counts in KITTI eval)
        difficulty = ann.get("difficulty", "Ignore")

        bbox_yolo = ann.get("bbox_yolo")
        if bbox_yolo is None:
            continue

        cx, cy, w, h = bbox_yolo
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        boxes.append([x1, y1, x2, y2])
        classes.append(int(class_id))
        difficulties.append(difficulty)

        # Use meta-stored iou_threshold if available (KITTI), else class default
        iou_thr = ann.get("iou_threshold") or CLASS_IOU_THRESHOLDS.get(int(class_id), DEFAULT_IOU)
        iou_thresholds.append(float(iou_thr))
        sources.append(meta.get("source", "unknown"))

    return {
        "boxes_xyxy":     np.array(boxes,       dtype=np.float32).reshape(-1, 4),
        "classes":        np.array(classes,      dtype=np.int32),
        "difficulties":   difficulties,
        "iou_thresholds": np.array(iou_thresholds, dtype=np.float32),
        "sources":        sources,
    }


def build_meta_index(dataset_root: str, split: str = "test") -> dict[str, Path]:
    """
    Scan <dataset_root>/meta/<split>/ and return {stem: path} mapping.
    Handles both .json extensions.
    """
    meta_dir = Path(dataset_root) / "meta" / split
    if not meta_dir.exists():
        return {}
    return {p.stem: p for p in sorted(meta_dir.glob("*.json"))}