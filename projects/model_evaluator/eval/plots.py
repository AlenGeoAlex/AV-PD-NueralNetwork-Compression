"""
plots.py — Generate and save all evaluation plots for a single model run.

Plots produced:
  - per_class_ap_bar.png      : AP per class (with KITTI IoU thresholds noted)
  - difficulty_ap_bar.png     : mAP50 per difficulty level (Easy/Moderate/Hard)
  - precision_recall.png      : PR curves per class
  - metrics_radar.png         : Radar chart: mAP50, mAP50-95, Precision, Recall, F1
  - latency_histogram.png     : FPS / latency summary bar
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CLASS_NAMES   = ["pedestrian", "cyclist", "car", "large_vehicle"]
CLASS_COLORS  = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"]
DIFF_COLORS   = {"Easy": "#57CC99", "Moderate": "#F4A261", "Hard": "#EF233C"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Per-class AP bar ─────────────────────────────────────────────────────────

def plot_per_class_ap(per_class_ap: dict[str, float], output_dir: Path, model_name: str) -> Path:
    classes = list(per_class_ap.keys())
    values  = [per_class_ap[c] * 100 for c in classes]
    iou_notes = {"pedestrian": "IoU@0.5", "cyclist": "IoU@0.5",
                 "car": "IoU@0.7", "large_vehicle": "IoU@0.7"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, values, color=CLASS_COLORS[:len(classes)], width=0.55, edgecolor="white")

    for bar, val, cls in zip(bars, values, classes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%\n({iou_notes.get(cls, '')})",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Average Precision (%)")
    ax.set_title(f"{model_name} — Per-Class AP (KITTI IoU thresholds)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Class")

    out = output_dir / "per_class_ap_bar.png"
    _save(fig, out)
    return out


# ─── Difficulty-stratified AP bar ─────────────────────────────────────────────

def plot_difficulty_ap(
    difficulty_map: dict[str, dict[str, float]],  # {difficulty: {class: ap}}
    output_dir: Path,
    model_name: str,
) -> Path:
    difficulties = [d for d in ["Easy", "Moderate", "Hard"] if d in difficulty_map]
    classes      = CLASS_NAMES

    x     = np.arange(len(difficulties))
    width = 0.18
    offsets = np.linspace(-(len(classes)-1)/2, (len(classes)-1)/2, len(classes)) * width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (cls, color) in enumerate(zip(classes, CLASS_COLORS)):
        vals = [difficulty_map[d].get(cls, 0.0) * 100 for d in difficulties]
        bars = ax.bar(x + offsets[i], vals, width, label=cls, color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylim(0, 115)
    ax.set_ylabel("AP (%)")
    ax.set_title(f"{model_name} — AP by Difficulty Level (KITTI criteria)")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    out = output_dir / "difficulty_ap_bar.png"
    _save(fig, out)
    return out


# ─── Precision-Recall curves ──────────────────────────────────────────────────

def plot_pr_curves(
    pr_data: dict[str, dict],   # {class_name: {"recall": [...], "precision": [...]}}
    output_dir: Path,
    model_name: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
        if cls not in pr_data:
            continue
        rec  = pr_data[cls]["recall"]
        prec = pr_data[cls]["precision"]
        if len(rec) == 0:
            continue
        ap = pr_data[cls].get("ap", 0.0)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{cls} (AP={ap*100:.1f}%)")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Precision-Recall Curves (IoU@0.5)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    out = output_dir / "precision_recall.png"
    _save(fig, out)
    return out


# ─── Radar / spider chart ─────────────────────────────────────────────────────

def plot_metrics_radar(summary: dict[str, float], output_dir: Path, model_name: str) -> Path:
    labels = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]
    keys   = ["map50", "map50_95", "precision", "recall", "f1"]

    values = [summary.get(k, 0.0) for k in keys]
    # Normalise: all metrics are already 0-1

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles      = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values_plot, color="#4C9BE8", linewidth=2)
    ax.fill(angles, values_plot, color="#4C9BE8", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax.set_title(f"{model_name} — Performance Radar", y=1.08)
    ax.grid(color="grey", alpha=0.3)

    out = output_dir / "metrics_radar.png"
    _save(fig, out)
    return out


# ─── Latency / throughput bar ─────────────────────────────────────────────────

def plot_latency(latency_info: dict[str, float], output_dir: Path, model_name: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: latency bar with error
    ax = axes[0]
    mean_ms = latency_info.get("latency_mean_ms", 0)
    std_ms  = latency_info.get("latency_std_ms", 0)
    ax.bar(["Inference"], [mean_ms], yerr=[std_ms], capsize=8, color="#4C9BE8",
           edgecolor="white", width=0.4)
    ax.set_ylabel("Latency (ms/image)")
    ax.set_title("Inference Latency")
    ax.text(0, mean_ms + std_ms + 0.3, f"{mean_ms:.1f} ± {std_ms:.1f} ms",
            ha="center", va="bottom", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, (mean_ms + std_ms) * 1.5 + 1)

    # Right: FPS bar
    ax2 = axes[1]
    fps = latency_info.get("fps", 0)
    ax2.bar(["FPS"], [fps], color="#2A9D8F", edgecolor="white", width=0.4)
    ax2.set_ylabel("Frames Per Second")
    ax2.set_title("Throughput (FPS)")
    ax2.text(0, fps + 0.5, f"{fps:.1f} FPS", ha="center", va="bottom", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, fps * 1.5 + 1)

    fig.suptitle(f"{model_name} — Speed Metrics", fontsize=13)
    fig.tight_layout()

    out = output_dir / "latency_histogram.png"
    _save(fig, out)
    return out


# ─── Orchestrate all plots ────────────────────────────────────────────────────

def generate_all_plots(results: dict[str, Any], output_dir: Path, model_name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    if "per_class_ap" in results:
        generated.append(plot_per_class_ap(results["per_class_ap"], output_dir, model_name))

    if "difficulty_ap" in results:
        generated.append(plot_difficulty_ap(results["difficulty_ap"], output_dir, model_name))

    if "pr_curves" in results:
        generated.append(plot_pr_curves(results["pr_curves"], output_dir, model_name))

    summary = results.get("summary", {})
    if summary:
        generated.append(plot_metrics_radar(summary, output_dir, model_name))

    if "latency" in results:
        generated.append(plot_latency(results["latency"], output_dir, model_name))

    return generated