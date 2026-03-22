"""
distillation_loss.py
--------------------
Response-Based Knowledge Distillation loss for YOLO11 and YOLO26
(Ultralytics 8.4.21+).

Teacher : YOLO11x or YOLO26x  (frozen, no gradients)
Student : YOLO11s or YOLO26s  (being trained)

The student is trained with a combined loss:
    Total Loss = alpha * Hard Loss  +  (1 - alpha) * Soft Loss

Hard Loss  : standard YOLO detection loss against ground-truth labels
Soft Loss  : this file — measures how closely the student's predictions
             match the teacher's predictions.

Prediction format differences between models
--------------------------------------------
YOLO11 returns a flat dict:
    preds = {
        'boxes':  [B, 64, 8400]   4 * 16 DFL bins, all anchors flattened
        'scores': [B, nc, 8400]   class logits
        'feats':  list of 3       raw neck features (not used)
    }

YOLO26 returns a nested dict (two detection heads):
    preds = {
        'one2many': {             used during training
            'boxes':  [B, 4, 8400]   direct cx/cy/w/h — DFL removed
            'scores': [B, nc, 8400]
            'feats':  list of 3
        },
        'one2one': {              used during inference (replaces NMS)
            'boxes':  [B, 4, 8400]
            'scores': [B, nc, 8400]
            'feats':  list of 3
        }
    }

Key differences:
    - YOLO26 boxes: [B, 4, 8400]  vs YOLO11 boxes: [B, 64, 8400]
      YOLO26 removed DFL — boxes are direct (cx, cy, w, h), not bin distributions
    - YOLO26 uses one2many head for training, one2one for inference
    - Both use the same scores format so KL divergence is identical

8400 = 80*80 + 40*40 + 20*20  (all scales concatenated, 640px input)

Soft Loss components:
    Classification : KL Divergence on scores with temperature scaling
    Box regression : MSE on boxes (continuous values for both models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Computes the soft (distillation) loss between teacher and student
    predictions. Handles both YOLO11 (flat dict) and YOLO26 (nested dict)
    output formats automatically.

    Args:
        temperature (float):
            Controls how soft the probability distributions are.
            - T=1  → standard softmax (sharp, confident)
            - T>1  → softer, exposes more inter-class structure
            Recommended: 3.0

        cls_loss_weight (float):
            Weight applied to the classification distillation loss.
            Default: 1.0

        box_loss_weight (float):
            Weight applied to the box regression distillation loss.
            Default: 1.0

        scale_weights (list[float]):
            Kept for API compatibility with train_kd.py.
            Not used — scores/boxes are flattened across all scales
            in Ultralytics 8.4.21+.
    """

    def __init__(
        self,
        temperature:      float = 3.0,
        cls_loss_weight:  float = 1.0,
        box_loss_weight:  float = 1.0,
        scale_weights:    list  = None,
    ):
        super().__init__()

        self.T               = temperature
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight

        if scale_weights is not None and scale_weights != [1.0, 1.0, 1.0]:
            print(
                "[KD] Warning: scale_weights is ignored in Ultralytics 8.4.21+. "
                "scores/boxes are flattened across all scales."
            )

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    # ─────────────────────────────────────────────────────────────────────────
    # FORMAT HANDLER
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract(preds: dict) -> dict:
        """
        Normalise prediction dict to flat format with 'boxes' and 'scores'.

        YOLO11 returns a flat dict   → return as-is
        YOLO26 returns a nested dict → extract 'one2many' (training head)

        Args:
            preds: raw model output dict

        Returns:
            flat dict with at least 'boxes' and 'scores' keys
        """
        if "one2many" in preds:
            return preds["one2many"]   # YOLO26 — use training head
        return preds                   # YOLO11 — already flat

    # ─────────────────────────────────────────────────────────────────────────
    # LOSS COMPONENTS
    # ─────────────────────────────────────────────────────────────────────────

    def _cls_distill_loss(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL Divergence between student and teacher class logits
        with temperature scaling.

        Both tensors shape: [B, nc, 8400]

        Steps:
            1. Permute to [B*8400, nc] — softmax operates over class dim
            2. Scale logits by T
            3. Teacher → softmax probs    (target distribution)
            4. Student → log-softmax      (KLDiv input)
            5. KL divergence * T²         (restores gradient magnitude)

        Why T²?
            Dividing logits by T shrinks gradients by 1/T².
            Multiplying loss by T² cancels this out so gradient scale
            stays consistent regardless of temperature chosen.
            Reference: Hinton et al., "Distilling the Knowledge in a
            Neural Network" (2015)
        """
        B, nc, A = student_scores.shape

        # [B, nc, 8400] → [B*8400, nc]
        s = student_scores.permute(0, 2, 1).reshape(-1, nc)
        t = teacher_scores.permute(0, 2, 1).reshape(-1, nc)

        s_soft = F.log_softmax(s / self.T, dim=1)
        t_soft = F.softmax(t / self.T, dim=1)

        return self.kl_loss(s_soft, t_soft) * (self.T ** 2)

    def _box_distill_loss(
        self,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE between student and teacher box predictions.

        YOLO11: [B, 64, 8400] — 4 * 16 DFL bins
        YOLO26: [B,  4, 8400] — direct (cx, cy, w, h), DFL removed
        """
        return F.mse_loss(student_boxes, teacher_boxes)

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        student_preds: dict,
        teacher_preds: dict,
        nc:            int,
    ) -> torch.Tensor:
        """
        Compute total soft distillation loss.

        Handles both YOLO11 and YOLO26 output formats automatically
        via _extract().

        Args:
            student_preds : raw model output dict (YOLO11 flat or YOLO26 nested)
            teacher_preds : same structure, gradients detached in trainer
            nc            : number of classes (e.g. 4 for your dataset)

        Returns:
            total_soft_loss : scalar, differentiable w.r.t. student params only
        """
        # Normalise to flat format — handles YOLO11 and YOLO26
        student = self._extract(student_preds)
        teacher = self._extract(teacher_preds)

        student_scores = student["scores"]             # [B, nc, 8400]
        student_boxes  = student["boxes"]              # [B, 64 or 4, 8400]
        teacher_scores = teacher["scores"].detach()
        teacher_boxes  = teacher["boxes"].detach()

        assert student_scores.shape[1] == nc, (
            f"Student scores nc mismatch: expected {nc}, "
            f"got {student_scores.shape[1]}"
        )

        # Teacher may have different nc (e.g. COCO=80 vs your dataset=4)
        # Slice teacher scores down to student nc so KL shapes align
        if teacher_scores.shape[1] != nc:
            teacher_scores = teacher_scores[:, :nc, :]

        assert student_boxes.shape[1] == teacher_boxes.shape[1], (
            f"Box channel mismatch: student={student_boxes.shape[1]}, "
            f"teacher={teacher_boxes.shape[1]}. "
            f"Cannot mix YOLO11 and YOLO26 as teacher/student."
        )

        cls_loss = self._cls_distill_loss(student_scores, teacher_scores)
        box_loss = self._box_distill_loss(student_boxes, teacher_boxes)

        return self.cls_loss_weight * cls_loss + self.box_loss_weight * box_loss