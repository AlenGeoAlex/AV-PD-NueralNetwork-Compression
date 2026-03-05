"""
Common fine-tuning script for all compression stages.

Usage (standalone):
    python finetune.py --weights yolo11s.pt --name yolo11s_baseline

Usage (imported) — plain kwargs:
    from finetune import finetune
    best_weights = finetune(weights="yolo11s.pt", name="yolo11s_baseline")

Usage (imported) — with FinetuneConfig:
    from finetune import finetune, FinetuneConfig
    cfg = FinetuneConfig(weights="yolo11s.pt", name="yolo11s_baseline", epochs=50)
    best_weights = finetune(cfg)

Usage (imported) — with preset configs:
    from finetune import finetune, BASELINE, POST_PRUNING_20, POST_PRUNING_40, POST_PRUNING_60
    cfg = BASELINE(weights="yolo11s.pt", name="yolo11s_baseline")
    best_weights = finetune(cfg)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """
    Configuration for a fine-tuning run.

    Required:
        weights:  Input .pt file path.
        name:     Run name — used for output folder, make it descriptive.
                  e.g. "yolo11s_baseline", "yolo11s_pruned_40", "yolo11s_post_kd"

    Optional (all have sensible defaults):
        data, epochs, imgsz, batch, lr0, lrf, optimizer, project, device
    """
    weights:   str
    name:      str
    data:      str           = "pedestrian.yaml"
    epochs:    int           = 50
    imgsz:     int           = 640
    batch:     int           = 32
    lr0:       float         = 0.001
    lrf:       float         = 0.01
    optimizer: str           = "SGD"
    project:   str           = "runs/finetune"
    workers:   int           = 1
    device:    Optional[str] = None

    @property
    def best_weights(self) -> str:
        """Expected output path — available before the run for planning/chaining."""
        return str(Path(self.project) / self.name / "weights" / "best.pt")


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

def BASELINE(weights: str, name: str, **kwargs) -> FinetuneConfig:
    """Fine-tune from COCO pretrained weights. Full 50 epochs."""
    return FinetuneConfig(weights=weights, name=name, epochs=50, **kwargs)

def POST_PRUNING_20(weights: str, name: str, **kwargs) -> FinetuneConfig:
    """Recovery after 20% pruning. Light damage — 20 epochs."""
    return FinetuneConfig(weights=weights, name=name, epochs=20, **kwargs)

def POST_PRUNING_40(weights: str, name: str, **kwargs) -> FinetuneConfig:
    """Recovery after 40% pruning. Moderate damage — 30 epochs."""
    return FinetuneConfig(weights=weights, name=name, epochs=30, **kwargs)

def POST_PRUNING_60(weights: str, name: str, **kwargs) -> FinetuneConfig:
    """Recovery after 60% pruning. Heavy damage — 40 epochs."""
    return FinetuneConfig(weights=weights, name=name, epochs=40, **kwargs)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def finetune(
    cfg:       Optional[FinetuneConfig] = None,
    *,
    weights:   Optional[str]  = None,
    name:      Optional[str]  = None,
    data:      str            = "pedestrian.yaml",
    epochs:    int            = 50,
    imgsz:     int            = 640,
    batch:     int            = 32,
    lr0:       float          = 0.001,
    lrf:       float          = 0.01,
    optimizer: str            = "SGD",
    project:   str            = "runs/finetune",
    workers                   = 1,
    device:    Optional[str]  = None,
) -> str:
    """
    Fine-tunes a YOLO model on the given dataset.

    Works for any compression stage:
        - Baseline fine-tuning from COCO weights
        - Post-pruning recovery
        - Post-quantization recovery
        - Any other recovery step

    Can be called two ways:

        # 1. Pass a FinetuneConfig (or preset):
        cfg = BASELINE(weights="yolo11s.pt", name="yolo11s_baseline")
        best_weights = finetune(cfg)

        # 2. Plain kwargs (no config needed):
        best_weights = finetune(weights="yolo11s.pt", name="yolo11s_baseline")

    Args:
        cfg:       FinetuneConfig instance. If provided, all other args are ignored.
        weights:   Path to input .pt file (pretrained, pruned, etc.)
        name:      Run name for the output folder — make it descriptive.
        data:      Path to dataset yaml config.
        epochs:    Training epochs.
                       50  — baseline fine-tune from COCO
                       20  — post-pruning recovery at 20%
                       30  — post-pruning recovery at 40%
                       40  — post-pruning recovery at 60%
        imgsz:     Input image size.
        batch:     Batch size. 32 fits comfortably on 24GB VRAM.
        lr0:       Initial learning rate.
        lrf:       Final LR = lr0 * lrf (cosine decay floor).
        optimizer: Optimizer — SGD or Adam.
        project:   Root output directory.
        device:    e.g. "0", "0,1", "cpu". None = auto-detect.

    Returns:
        best_weights (str): Path to best.pt — feed directly into the next pipeline stage.
    """
    # cfg takes full priority — unpack it into local vars
    if cfg is not None:
        weights   = cfg.weights
        name      = cfg.name
        data      = cfg.data
        epochs    = cfg.epochs
        imgsz     = cfg.imgsz
        batch     = cfg.batch
        lr0       = cfg.lr0
        lrf       = cfg.lrf
        optimizer = cfg.optimizer
        project   = cfg.project
        device    = cfg.device

    if not weights or not name:
        raise ValueError(
            "weights and name are required. "
            "Pass a FinetuneConfig or provide them as kwargs."
        )

    model = YOLO(weights)

    train_args = dict(
        data      = data,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        lr0       = lr0,
        lrf       = lrf,
        optimizer = optimizer,
        project   = project,
        name      = name,
        workers = workers
    )
    if device is not None:
        train_args["device"] = device

    model.train(**train_args)

    best_weights = str(Path(project) / name / "weights" / "best.pt")
    print(f"Fine-tuning complete.")
    print(f"   Export path: {best_weights}\n")

    return best_weights
