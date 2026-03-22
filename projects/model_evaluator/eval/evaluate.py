"""
Core evaluation engine.

Entry point:
    run_evaluation(model_path, test_set_path, output_dir, ...)

Produces:
    <output_dir>/
        metrics_summary.csv        full scalar metrics
        per_class_ap.csv           per-class AP at KITTI IoU thresholds
        difficulty_ap.csv          AP by difficulty × class
        comparison.json            machine-readable results for benchmarking_suite
        per_class_ap_bar.png
        difficulty_ap_bar.png
        precision_recall.png
        metrics_radar.png
        latency_histogram.png
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .meta_loader import build_meta_index, get_gt_for_image, load_meta_sidecar
from .metrics import (
    CLASS_IOU_THRESHOLDS,
    CLASS_NAMES,
    DIFFICULTIES,
    IOU_RANGE,
    compute_ap,
    compute_ap_from_matches,
    get_model_info,
    match_predictions_to_gt,
    measure_latency,
)
from .plots import generate_all_plots


# ─── YOLO inference wrapper ───────────────────────────────────────────────────

def _load_yolo(model_path: str):
    """Load any YOLO model via Ultralytics."""
    from ultralytics import YOLO
    return YOLO(model_path)


def _run_inference(yolo_model, image_path: Path, conf: float = 0.001, iou: float = 0.6):
    """
    Run inference on a single image. Returns (pred_boxes_xyxy, pred_scores, pred_classes).
    conf very low so we get the full curve for AP computation.
    """
    results = yolo_model.predict(
        str(image_path), conf=conf, iou=iou, verbose=False, save=False
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    boxes   = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores  = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    return boxes, scores, classes


def _get_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) without loading full image into memory."""
    try:
        from PIL import Image
        with Image.open(str(image_path)) as img:
            return img.size   # (W, H)
    except Exception:
        return 1242, 375   # KITTI default fallback


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    model_path: str,
    test_set_path: str,
    output_dir: str,
    model_name: str | None = None,
    conf_threshold: float = 0.001,
    iou_nms: float = 0.6,
    latency_warmup: int = 50,
    latency_runs: int = 200,
    img_size: int = 640,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a YOLO model on a test set.

    Parameters
    ----------
    model_path      : path to .pt weights file (any YOLO variant)
    test_set_path   : root of merged dataset, e.g. /path/to/Dataset/merged_dataset
                      Expected layout: images/test/, labels/test/, meta/test/
    output_dir      : where to write results (created if needed)
    model_name      : display name — defaults to model filename stem
    conf_threshold  : low conf to preserve full PR curve (default 0.001)
    iou_nms         : NMS IoU threshold
    latency_warmup  : warmup iterations for latency measurement
    latency_runs    : timed iterations for latency measurement
    img_size        : inference image size
    device          : "cpu" / "cuda" / None (auto)
    verbose         : print progress

    Returns
    -------
    Full results dict (also written to comparison.json)
    """

    # ── Setup ────────────────────────────────────────────────────────────────
    model_path   = str(model_path)
    test_set_path = str(test_set_path)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = Path(model_path).stem

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"  Model:      {model_path}")
        print(f"  Test set:   {test_set_path}")
        print(f"  Output:     {output_dir}")
        print(f"  Device:     {device}")
        print(f"{'='*60}\n")

    # ── Load model ───────────────────────────────────────────────────────────
    yolo = _load_yolo(model_path)
    yolo.to(device)

    # ── Discover test images ─────────────────────────────────────────────────
    images_dir = Path(test_set_path) / "images" / "test"
    if not images_dir.exists():
        raise FileNotFoundError(f"Test images not found at: {images_dir}")

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    if verbose:
        print(f"  Found {len(image_paths)} test images")

    # ── Build meta index ─────────────────────────────────────────────────────
    meta_index = build_meta_index(test_set_path, split="test")
    has_meta   = len(meta_index) > 0
    if verbose:
        print(f"  Meta sidecars: {len(meta_index)} ({'loaded' if has_meta else 'NOT FOUND — difficulty breakdown disabled'})")

    # ── Per-class accumulators ────────────────────────────────────────────────
    # detections_by_class[class_id][iou_thr] = list of {score, matched}
    # n_gt_by_class[class_id] = int

    class_ids = list(range(len(CLASS_NAMES)))

    # For mAP50 (standard IoU 0.5)
    dets_map50:   dict[int, list] = {c: [] for c in class_ids}
    n_gt_map50:   dict[int, int]  = {c: 0  for c in class_ids}

    # For mAP50-95 (averaged over 10 IoU thresholds)
    dets_map5095: dict[int, dict[float, list]] = {
        c: {float(t): [] for t in IOU_RANGE} for c in class_ids
    }
    n_gt_5095:    dict[int, int] = {c: 0 for c in class_ids}

    # For KITTI IoU thresholds (class-specific)
    dets_kitti:   dict[int, list] = {c: [] for c in class_ids}
    n_gt_kitti:   dict[int, int]  = {c: 0  for c in class_ids}

    # For difficulty-stratified AP
    dets_diff: dict[str, dict[int, list]] = {
        d: {c: [] for c in class_ids} for d in DIFFICULTIES
    }
    n_gt_diff: dict[str, dict[int, int]] = {
        d: {c: 0 for c in class_ids} for d in DIFFICULTIES
    }

    # PR curve data (IoU@0.5)
    pr_accumulator: dict[int, list] = {c: [] for c in class_ids}

    # ── Inference loop ────────────────────────────────────────────────────────
    t_start = time.time()

    for idx, img_path in enumerate(image_paths):
        stem = img_path.stem
        img_w, img_h = _get_image_size(img_path)

        # --- Predictions ---
        pred_boxes, pred_scores, pred_classes = _run_inference(
            yolo, img_path, conf=conf_threshold, iou=iou_nms
        )

        # --- Ground truth ---
        meta = {}
        if has_meta and stem in meta_index:
            meta = load_meta_sidecar(meta_index[stem])

        gt = get_gt_for_image(meta, img_w, img_h)
        gt_boxes       = gt["boxes_xyxy"]
        gt_classes     = gt["classes"]
        gt_difficulties = gt["difficulties"]

        if len(gt_boxes) == 0 and len(meta) == 0:
            # Fallback: parse YOLO label file
            gt_boxes, gt_classes, gt_difficulties = _load_gt_from_label(
                test_set_path, stem, img_w, img_h
            )

        # --- Accumulate matches ---
        for class_id in class_ids:
            cls_iou_kitti = CLASS_IOU_THRESHOLDS[class_id]

            # Subset GT to non-Ignore for this image
            gt_mask = gt_classes == class_id
            gt_non_ignore = [
                i for i, (g, d) in enumerate(zip(gt_classes, gt_difficulties))
                if g == class_id and d != "Ignore"
            ]
            n_gt_this_class = len(gt_non_ignore)

            if len(gt_boxes) > 0 and n_gt_this_class > 0:
                gt_boxes_cls = gt_boxes[gt_non_ignore]
                gt_cls_arr   = np.full(len(gt_non_ignore), class_id, dtype=np.int32)
            else:
                gt_boxes_cls = np.zeros((0, 4), dtype=np.float32)
                gt_cls_arr   = np.zeros(0, dtype=np.int32)

            # mAP@0.5
            matches_05 = match_predictions_to_gt(
                pred_boxes, pred_scores, pred_classes,
                gt_boxes_cls, gt_cls_arr, iou_threshold=0.5, class_id=class_id
            )
            dets_map50[class_id].extend(matches_05)
            n_gt_map50[class_id] += n_gt_this_class
            pr_accumulator[class_id].extend(matches_05)

            # KITTI IoU
            if cls_iou_kitti != 0.5:
                matches_kitti = match_predictions_to_gt(
                    pred_boxes, pred_scores, pred_classes,
                    gt_boxes_cls, gt_cls_arr, iou_threshold=cls_iou_kitti, class_id=class_id
                )
            else:
                matches_kitti = matches_05
            dets_kitti[class_id].extend(matches_kitti)
            n_gt_kitti[class_id] += n_gt_this_class

            # mAP50-95
            n_gt_5095[class_id] += n_gt_this_class
            for thr in IOU_RANGE:
                thr_f = float(thr)
                if abs(thr_f - 0.5) < 1e-6:
                    dets_map5095[class_id][thr_f].extend(matches_05)
                else:
                    m = match_predictions_to_gt(
                        pred_boxes, pred_scores, pred_classes,
                        gt_boxes_cls, gt_cls_arr, iou_threshold=thr_f, class_id=class_id
                    )
                    dets_map5095[class_id][thr_f].extend(m)

            # Difficulty-stratified (IoU@0.5 for both thresholds — KITTI standard)
            for diff in DIFFICULTIES:
                diff_indices = [
                    i for i, (g, d) in enumerate(zip(gt_classes, gt_difficulties))
                    if g == class_id and d == diff
                ]
                n_gt_diff[diff][class_id] += len(diff_indices)

                if len(diff_indices) > 0:
                    gt_diff_boxes = gt_boxes[diff_indices]
                    gt_diff_cls   = np.full(len(diff_indices), class_id, dtype=np.int32)
                    m_diff = match_predictions_to_gt(
                        pred_boxes, pred_scores, pred_classes,
                        gt_diff_boxes, gt_diff_cls, iou_threshold=0.5, class_id=class_id
                    )
                else:
                    m_diff = []
                dets_diff[diff][class_id].extend(m_diff)

        if verbose and (idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(image_paths)}] {elapsed:.1f}s elapsed")

    if verbose:
        print(f"\n  Inference complete in {time.time() - t_start:.1f}s")
        print("  Computing metrics...")

    # ── Compute AP scores ─────────────────────────────────────────────────────

    per_class_ap_50    = {}
    per_class_ap_kitti = {}
    pr_curves          = {}

    for class_id, cls_name in enumerate(CLASS_NAMES):
        ap50, rec, prec = compute_ap_from_matches(dets_map50[class_id], n_gt_map50[class_id])
        per_class_ap_50[cls_name] = float(ap50)
        pr_curves[cls_name] = {
            "recall":    rec.tolist(),
            "precision": prec.tolist(),
            "ap":        float(ap50),
        }

        ap_kitti, _, _ = compute_ap_from_matches(
            dets_kitti[class_id], n_gt_kitti[class_id],
            iou_threshold=CLASS_IOU_THRESHOLDS[class_id]
        )
        per_class_ap_kitti[cls_name] = float(ap_kitti)

    map50 = float(np.mean(list(per_class_ap_50.values())))

    # mAP50-95
    ap_5095_per_class = {}
    for class_id, cls_name in enumerate(CLASS_NAMES):
        aps = []
        for thr in IOU_RANGE:
            ap_t, _, _ = compute_ap_from_matches(
                dets_map5095[class_id][float(thr)], n_gt_5095[class_id]
            )
            aps.append(ap_t)
        ap_5095_per_class[cls_name] = float(np.mean(aps))
    map50_95 = float(np.mean(list(ap_5095_per_class.values())))

    # Difficulty AP (IoU@0.5)
    difficulty_ap: dict[str, dict[str, float]] = {}
    for diff in DIFFICULTIES:
        difficulty_ap[diff] = {}
        for class_id, cls_name in enumerate(CLASS_NAMES):
            ap_d, _, _ = compute_ap_from_matches(
                dets_diff[diff][class_id], n_gt_diff[diff][class_id]
            )
            difficulty_ap[diff][cls_name] = float(ap_d)

    # Global precision / recall at conf=0.25
    global_tp = global_fp = global_fn = 0
    for class_id in class_ids:
        dets_sorted = sorted(dets_map50[class_id], key=lambda d: d["score"], reverse=True)
        # Use predictions above a reasonable threshold (conf 0.25)
        for d in dets_sorted:
            if d["score"] >= 0.25:
                if d["matched"]:
                    global_tp += 1
                else:
                    global_fp += 1
        global_fn += max(0, n_gt_map50[class_id] - sum(
            1 for d in dets_sorted if d["score"] >= 0.25 and d["matched"]
        ))

    precision = global_tp / (global_tp + global_fp + 1e-9)
    recall    = global_tp / (global_tp + global_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # ── Model info & latency ──────────────────────────────────────────────────
    model_info  = get_model_info(model_path)
    latency_info = measure_latency(
        yolo.model, img_size=img_size,
        n_warmup=latency_warmup, n_runs=latency_runs, device=device
    )

    if verbose:
        print(f"\n  mAP50:     {map50*100:.2f}%")
        print(f"  mAP50-95:  {map50_95*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1:        {f1*100:.2f}%")
        print(f"  FPS:       {latency_info['fps']}")
        print(f"  Size (MB): {model_info['size_mb']}")

    # ── Assemble results dict ─────────────────────────────────────────────────
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "summary": {
            "map50":      round(map50,     4),
            "map50_95":   round(map50_95,  4),
            "precision":  round(precision, 4),
            "recall":     round(recall,    4),
            "f1":         round(f1,        4),
        },
        "per_class_ap":       {k: round(v, 4) for k, v in per_class_ap_50.items()},
        "per_class_ap_kitti": {k: round(v, 4) for k, v in per_class_ap_kitti.items()},
        "map50_95_per_class": {k: round(v, 4) for k, v in ap_5095_per_class.items()},
        "difficulty_ap":      {
            d: {k: round(v, 4) for k, v in vals.items()}
            for d, vals in difficulty_ap.items()
        },
        "pr_curves":  pr_curves,
        "latency":    latency_info,
        "model_info": model_info,
        "eval_config": {
            "conf_threshold": conf_threshold,
            "iou_nms":        iou_nms,
            "img_size":       img_size,
            "device":         device,
            "n_images":       len(image_paths),
        },
    }

    # ── Write CSV outputs ─────────────────────────────────────────────────────
    _write_summary_csv(results, output_dir_p)
    _write_per_class_csv(results, output_dir_p)
    _write_difficulty_csv(results, output_dir_p)

    # ── Write comparison.json ─────────────────────────────────────────────────
    comparison_json = output_dir_p / "comparison.json"
    with open(comparison_json, "w") as f:
        # Exclude raw PR curve data (large, not needed for benchmarking suite)
        export = {k: v for k, v in results.items() if k != "pr_curves"}
        json.dump(export, f, indent=2)

    if verbose:
        print(f"\n  Saved comparison.json: {comparison_json}")

    # ── Generate plots ────────────────────────────────────────────────────────
    plots_dir = output_dir_p / "plots"
    generate_all_plots(results, plots_dir, model_name)

    if verbose:
        print(f"  Plots saved to: {plots_dir}")
        print(f"\n{'='*60}\n")

    return results


# ─── CSV writers ──────────────────────────────────────────────────────────────

def _write_summary_csv(results: dict, output_dir: Path) -> None:
    path = output_dir / "metrics_summary.csv"
    s    = results["summary"]
    li   = results["latency"]
    mi   = results["model_info"]

    rows = [
        ["metric", "value"],
        ["model_name",       results["model_name"]],
        ["map50",            f"{s['map50']:.4f}"],
        ["map50_95",         f"{s['map50_95']:.4f}"],
        ["precision",        f"{s['precision']:.4f}"],
        ["recall",           f"{s['recall']:.4f}"],
        ["f1",               f"{s['f1']:.4f}"],
        ["latency_mean_ms",  f"{li['latency_mean_ms']:.3f}"],
        ["latency_std_ms",   f"{li['latency_std_ms']:.3f}"],
        ["fps",              f"{li['fps']:.2f}"],
        ["size_mb",          f"{mi['size_mb']:.2f}"],
        ["params_m",         str(mi.get("params_m", "N/A"))],
        ["n_images_evaluated", str(results["eval_config"]["n_images"])],
        ["device",           results["eval_config"]["device"]],
    ]

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _write_per_class_csv(results: dict, output_dir: Path) -> None:
    path = output_dir / "per_class_ap.csv"
    rows = [["class", "ap50_iou0.5", "ap_kitti_iou", "kitti_iou_threshold", "ap50_95"]]

    kitti_ious = {"pedestrian": 0.5, "cyclist": 0.5, "car": 0.7, "large_vehicle": 0.7}

    for cls in CLASS_NAMES:
        rows.append([
            cls,
            f"{results['per_class_ap'].get(cls, 0):.4f}",
            f"{results['per_class_ap_kitti'].get(cls, 0):.4f}",
            str(kitti_ious.get(cls, 0.5)),
            f"{results['map50_95_per_class'].get(cls, 0):.4f}",
        ])

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _write_difficulty_csv(results: dict, output_dir: Path) -> None:
    path  = output_dir / "difficulty_ap.csv"
    diffs = ["Easy", "Moderate", "Hard"]

    header = ["class"] + diffs
    rows   = [header]

    for cls in CLASS_NAMES:
        row = [cls] + [
            f"{results['difficulty_ap'].get(d, {}).get(cls, 0):.4f}"
            for d in diffs
        ]
        rows.append(row)

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ─── GT fallback from label files ────────────────────────────────────────────

def _load_gt_from_label(
    dataset_root: str, stem: str, img_w: int, img_h: int
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Fallback: load GT from YOLO .txt label file when no meta sidecar exists.
    All boxes get difficulty "Moderate" by default.
    """
    label_path = Path(dataset_root) / "labels" / "test" / f"{stem}.txt"
    boxes, classes, diffs = [], [], []

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)
            diffs.append("Moderate")

    return (
        np.array(boxes, dtype=np.float32).reshape(-1, 4),
        np.array(classes, dtype=np.int32),
        diffs,
    )