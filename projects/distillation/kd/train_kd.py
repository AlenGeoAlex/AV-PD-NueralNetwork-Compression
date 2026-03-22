"""
What happens per run:
    1. Sanity checks verify all paths and hyperparameter bounds
    2. Run name is built from the active hyperparameters
    3. DistillationTrainer is created with teacher frozen
    4. Training loop runs for `epochs` iterations
       Each iteration:
           a. Forward pass through teacher (no grad)
           b. Forward pass through student (with grad)
           c. Hard loss computed against GT labels
           d. Soft loss computed between teacher and student predictions
           e. Total loss = alpha * hard + (1-alpha) * soft
           f. Backprop through student only
           g. Optimizer step
    5. Best and last weights saved to {project_dir}/{run_name}/weights/
    6. Results logged to {project_dir}/{run_name}/results.csv
"""

import torch
from pathlib import Path
from .distillation_trainer import DistillationTrainer


# =============================================================================
# HELPERS
# =============================================================================

def make_run_name(
    teacher_weights: str,
    student_weights: str,
    alpha: float,
    temperature: float,
    batch_size: int,
    epochs: int,
    scale_weights: list[float],
) -> str:
    """
    Build a human-readable, unique run name from the active hyperparameters.
    Example: yolo11x_to_yolo11s__a0.7_t3.0_bs16_e50_scales1.0-1.0-1.0
    """
    teacher_tag = Path(teacher_weights).stem
    student_tag = Path(student_weights).stem
    scales_tag  = "-".join(str(w) for w in scale_weights)

    return (
        f"{teacher_tag}_to_{student_tag}"
        f"__a{alpha}"
        f"_t{temperature}"
        f"_bs{batch_size}"
        f"_e{epochs}"
        f"_scales{scales_tag}"
    )


# =============================================================================
# SANITY CHECKS
# =============================================================================

def sanity_checks(
    teacher_weights: str,
    student_weights: str,
    data_config: str,
    alpha: float,
    temperature: float,
    scale_weights: list[float],
    device: str,
) -> None:
    """
    Verify all required files and hyperparameter values before training starts.
    Raises SystemExit(1) if any check fails so training never starts with a
    bad config.

    Parameters
    ----------
    teacher_weights : path to teacher .pt file
    student_weights : path to student .pt file
    data_config     : path to dataset .yaml file
    alpha           : hard/soft loss balance (must be in [0, 1])
    temperature     : distillation temperature (must be >= 1.0)
    scale_weights   : per-scale weights list (must have exactly 3 values)
    device          : torch device string
    """
    print("\n" + "=" * 60)
    print("Running pre-training sanity checks...")
    print("=" * 60)

    errors = []

    # ── File existence ────────────────────────────────────────────────────────
    if not Path(teacher_weights).exists():
        errors.append(
            f"  ❌ Teacher weights not found : {teacher_weights}\n"
            f"     Fix: from ultralytics import YOLO; YOLO('{teacher_weights}')"
        )
    else:
        print(f"  ✅ Teacher weights found     : {teacher_weights}")

    if not Path(student_weights).exists():
        errors.append(
            f"  ❌ Student weights not found : {student_weights}\n"
            f"     Fix: from ultralytics import YOLO; YOLO('{student_weights}')"
        )
    else:
        print(f"  ✅ Student weights found     : {student_weights}")

    if not Path(data_config).exists():
        errors.append(
            f"  ❌ Dataset config not found  : {data_config}\n"
            f"     Fix: create a .yaml file pointing to your train/val folders"
        )
    else:
        print(f"  ✅ Dataset config found      : {data_config}")

    # ── Hyperparameter bounds ─────────────────────────────────────────────────
    if not 0.0 <= alpha <= 1.0:
        errors.append(f"  ❌ alpha must be in [0.0, 1.0], got {alpha}")
    else:
        print(f"  ✅ Alpha                     : {alpha}  (hard={alpha}, soft={round(1 - alpha, 4)})")

    if temperature < 1.0:
        errors.append(f"  ❌ temperature must be >= 1.0, got {temperature}")
    else:
        print(f"  ✅ Temperature               : {temperature}")

    if len(scale_weights) != 3:
        errors.append(
            f"  ❌ scale_weights must have exactly 3 values [P3, P4, P5], "
            f"got {len(scale_weights)}"
        )
    else:
        print(f"  ✅ Scale weights             : {scale_weights}  [P3, P4, P5]")

    # ── Hardware ──────────────────────────────────────────────────────────────
    if device != "cpu":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU found                 : {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  ⚠️  No GPU found — falling back to CPU (training will be slow)")
    else:
        print(f"  ✅ Device                    : cpu")

    # ── Result ────────────────────────────────────────────────────────────────
    if errors:
        print("\n❌ Sanity checks FAILED. Fix these before training:\n")
        for e in errors:
            print(e)
        raise SystemExit(1)

    print("\n✅ All sanity checks passed — starting training\n")



def main(
    teacher_weights: str,
    student_weights: str,
    data_config:     str,
    project_dir:     str         = "runs/kd",
    alpha:           float       = 0.7,
    temperature:     float       = 3.0,
    scale_weights:   list[float] = None,
    epochs:          int         = 50,
    imgsz:           int         = 640,
    batch_size:      int         = 16,
    workers:         int         = 1,
    lr0:             float       = 0.001,
    lrf:             float       = 0.01,
    optimizer:       str         = "SGD",
    device:          str         = None,
) -> None:
    """
    Launch one KD training run with the given hyperparameters.
    """
    if scale_weights is None:
        scale_weights = [1.0, 1.0, 1.0]
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    sanity_checks(
        teacher_weights = teacher_weights,
        student_weights = student_weights,
        data_config     = data_config,
        alpha           = alpha,
        temperature     = temperature,
        scale_weights   = scale_weights,
        device          = device,
    )

    run_name = make_run_name(
        teacher_weights = teacher_weights,
        student_weights = student_weights,
        alpha           = alpha,
        temperature     = temperature,
        batch_size      = batch_size,
        epochs          = epochs,
        scale_weights   = scale_weights,
    )

    print("=" * 60)
    print("Knowledge Distillation Training Summary")
    print("=" * 60)
    print(f"  Teacher        : {teacher_weights}")
    print(f"  Student        : {student_weights}")
    print(f"  Dataset        : {data_config}")
    print(f"  Epochs         : {epochs}")
    print(f"  Image size     : {imgsz}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Device         : {device}")
    print(f"  Alpha          : {alpha}  (hard={alpha}, soft={round(1 - alpha, 4)})")
    print(f"  Temperature    : {temperature}")
    print(f"  Scale weights  : {scale_weights}  [P3, P4, P5]")
    print(f"  LR0 / LRF      : {lr0} / {lrf}")
    print(f"  Optimizer      : {optimizer}")
    print(f"  Output dir     : {project_dir}/{run_name}")
    print("=" * 60 + "\n")

    trainer = DistillationTrainer(
        teacher_weights       = teacher_weights,
        alpha                 = alpha,
        temperature           = temperature,
        distill_scale_weights = scale_weights,
        overrides={
            "model"    : student_weights,
            "data"     : data_config,
            "imgsz"    : imgsz,
            "epochs"   : epochs,
            "batch"    : batch_size,
            "workers"  : workers,
            "device"   : device,
            "optimizer": optimizer,
            "lr0"      : lr0,
            "lrf"      : lrf,
            "project"  : project_dir,
            "name"     : run_name,
            "save"     : True,
            "verbose"  : True,
        }
    )

    print("Starting KD training...")
    trainer.train()

    # ── Post-training summary ─────────────────────────────────────────────────
    best_weights = Path(project_dir) / run_name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Best weights : {best_weights}")
    print(f"  Results CSV  : {project_dir}/{run_name}/results.csv")
    print("=" * 60)