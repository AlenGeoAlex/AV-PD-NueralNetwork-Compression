"""
distillation_trainer.py
-----------------------
Subclasses Ultralytics DetectionTrainer to inject Knowledge Distillation
into the standard YOLO11/YOLO26 training loop (Ultralytics 8.4.21+).

How the training loop works in Ultralytics 8.4.21+
---------------------------------------------------
_do_train() calls:
    loss, loss_items = self.model(batch)

DetectionModel.forward(batch) handles this:
    if isinstance(x, dict):         # batch is a dict → training mode
        return self.loss(x)

DetectionModel.loss(batch) does:
    if preds is None:
        preds = self.forward(batch["img"])   # run forward pass
    return self.criterion(preds, batch)      # compute loss

So the call chain is:
    model(batch)
        → model.loss(batch)
            → preds = model.forward(img)     # get prediction dict
            → model.criterion(preds, batch)  # compute loss  ← OUR HOOK

Why we wrap model.criterion and not DetectionTrainer.criterion
--------------------------------------------------------------
DetectionTrainer.criterion does NOT exist in 8.4.21. The loss is
computed inside the model itself via model.criterion (v8DetectionLoss).
Wrapping model.criterion is the correct interception point — we receive
the already-computed preds dict and add soft loss on top without
running an extra forward pass.

Why we patch in _setup_train and not __init__
---------------------------------------------
model.criterion is None until _setup_train() initialises it via
model.init_criterion(). self.device is also not resolved until
_setup_train() runs. Loading the teacher or wrapping criterion in
__init__ would fail.

Works for both YOLO11 and YOLO26
---------------------------------
YOLO11 preds: flat dict   {'boxes': ..., 'scores': ..., 'feats': ...}
YOLO26 preds: nested dict {'one2many': {...}, 'one2one': {...}}

DistillationLoss._extract() handles the format difference automatically.
The only constraint: teacher and student must be the same model family
(both YOLO11 or both YOLO26) since box channel counts differ.
"""

import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .distillation_loss import DistillationLoss


class DistillationTrainer(DetectionTrainer):
    """
    YOLO11/YOLO26 trainer with Knowledge Distillation (Ultralytics 8.4.21+).

    Args:
        teacher_weights (str):
            Path to teacher .pt file. e.g. "yolo11x.pt" or "yolo26x.pt"
            Must be the same model family as the student.

        alpha (float):
            Hard loss weight. (1-alpha) = soft loss weight.
            Range 0.0-1.0. Recommended: 0.7
            - alpha=1.0 → pure hard loss, no distillation
            - alpha=0.0 → pure soft loss, ignores GT labels
            - alpha=0.7 → 70% GT, 30% teacher signal

        temperature (float):
            Softens class distributions for KL divergence.
            Range 1.0-5.0. Recommended: 3.0
            Higher = softer distributions = more inter-class structure

        distill_scale_weights (list[float]):
            Kept for API compatibility with train_kd.py.
            Not used in 8.4.21+ — scores/boxes are flattened across scales.

        overrides (dict):
            Standard Ultralytics trainer overrides
            (model, data, epochs, batch, lr0, etc.)
    """

    def __init__(
        self,
        teacher_weights:       str,
        alpha:                 float = 0.7,
        temperature:           float = 3.0,
        distill_scale_weights: list  = None,
        cfg                          = None,
        overrides                    = None,
    ):
        super().__init__(cfg=DEFAULT_CFG, overrides=overrides)

        self.alpha             = alpha
        self.temperature       = temperature
        self.teacher_weights   = teacher_weights
        self.distill_scale_weights = distill_scale_weights or [1.0, 1.0, 1.0]

        self.distillation_loss_fn = DistillationLoss(
            temperature   = temperature,
            scale_weights = self.distill_scale_weights,
        )

        print(f"\n[KD] DistillationTrainer initialised")
        print(f"[KD] Alpha={self.alpha}  Temperature={self.temperature}\n")

    # TEACHER LOADING
    def _load_teacher(self, weights: str) -> torch.nn.Module:
        """
        Load and freeze teacher model via YOLO() wrapper.

        Why YOLO() and not torch.load()?
            torch.load() on a .pt file returns a raw state dict or a partially
            initialised object — the forward() method does not return the
            prediction dict we need. YOLO() initialises the full model
            correctly so forward() returns {'boxes', 'scores', 'feats'}
            (YOLO11) or {'one2many', 'one2one'} (YOLO26).

        Why train() mode and not eval()?
            In Ultralytics 8.4.21+, eval() changes the output format —
            the model fuses layers and returns processed detections instead
            of the raw prediction dict. train() mode preserves the dict
            format we need for distillation.
            requires_grad=False + torch.no_grad() are sufficient to ensure
            no weight updates ever happen.
        """
        from ultralytics import YOLO

        print(f"[KD] Loading teacher from: {weights}")
        teacher = YOLO(weights).model
        teacher = teacher.to(self.device)

        # Freeze all parameters — no gradient computation ever
        for param in teacher.parameters():
            param.requires_grad = False

        # train() mode — preserves prediction dict output format
        teacher.train()

        print(f"[KD] Teacher loaded and frozen ✅")
        return teacher

    def _setup_train(self, *args, **kwargs):
        """
        After parent setup (device, model, optimizer all resolved),
        load the teacher and wrap model.criterion with the KD version.

        The wrapped criterion:
            1. Calls original criterion  → hard loss against GT labels
            2. Runs teacher forward pass → teacher prediction dict
            3. Calls DistillationLoss    → soft loss (KL + MSE)
            4. Returns alpha*hard + (1-alpha)*soft

        loss_items is returned unchanged from the original criterion so
        box_loss / cls_loss / dfl_loss columns in results.csv stay correct.
        """
        super()._setup_train(*args, **kwargs)

        # Load teacher now that self.device is fully resolved
        self.teacher = self._load_teacher(self.teacher_weights)

        # Unwrap DataParallel if present
        model = self.model.module if hasattr(self.model, "module") else self.model

        # Initialise criterion if not already done
        # Use getattr to safely handle both None and missing attribute
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()

        # Capture in local variables — avoids closure issues with self
        original_criterion = model.criterion
        teacher            = self.teacher
        alpha              = self.alpha
        loss_fn            = self.distillation_loss_fn

        def kd_criterion(preds, batch):
            """
            Wrapped criterion that adds soft distillation loss.

            Called by model.loss(batch) with already-computed preds dict.
            preds is never None here — model.loss() always runs the forward
            pass before calling criterion.

            YOLO11 preds: {'boxes': [B,64,8400], 'scores': [B,nc,8400], ...}
            YOLO26 preds: {'one2many': {'boxes': [B,4,8400], ...}, 'one2one': ...}

            Returns:
                total_loss : alpha * hard + (1-alpha) * soft  (scalar)
                loss_items : original hard loss breakdown      (unchanged)
            """
            # ── Hard loss ─────────────────────────────────────────────────────
            # Standard YOLO detection loss against ground truth labels.
            # Returns (loss_tensor, loss_items_tensor).
            # loss_tensor may be shape [3] (box, cls, dfl components) —
            # sum() reduces to scalar before combining with soft loss.
            hard_loss, loss_items = original_criterion(preds, batch)
            hard_loss = hard_loss.sum()   # ensure scalar

            # ── Teacher forward pass ──────────────────────────────────────────
            with torch.no_grad():
                teacher_preds = teacher(batch["img"])

            # nc from student predictions
            # Read at runtime — works for any nc (2, 4, 80, etc.)
            # _extract() handles both YOLO11 flat and YOLO26 nested format
            student_flat = DistillationLoss._extract(preds)
            nc = student_flat["scores"].shape[1]

            # Soft loss
            soft_loss = loss_fn(
                student_preds = preds,
                teacher_preds = teacher_preds,
                nc            = nc,
            )

            total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

            # loss_items unchanged — keeps CSV logging columns correct
            return total_loss, loss_items

        model.criterion = kd_criterion
        print(f"[KD] model.criterion wrapped with KD ✅\n")