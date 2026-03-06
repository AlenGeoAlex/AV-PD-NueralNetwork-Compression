# Knowledge Distillation — Ultralytics 8.4.21 Bug Investigation

## Summary

This document records the full debugging journey of getting KD working
on Ultralytics 8.4.21+. The original implementation was written against
an older Ultralytics API. Every breaking change and fix is documented here
so this can be used as a reference for future model versions.

---

## The Core Problem

**Ultralytics breaks internal APIs between major versions.**

KD requires hooking into the loss computation to inject soft loss.
Ultralytics does not expose a stable API for this — every version moves
where the loss is computed. The original code was correct for older
versions but silently failed on 8.4.21.

---

## Timeline of Bugs Found and Fixed

---

### Bug 1 — Wrong interception point: `criterion()` override

**Original approach:**
```python
class DistillationTrainer(DetectionTrainer):
    def criterion(self, preds, batch):
        hard_loss = super().criterion(preds, batch)
        soft_loss = ...
        return alpha * hard + (1-alpha) * soft
```

**What happened:**
`DetectionTrainer.criterion` does not exist in 8.4.21.
The method was never called. Training ran normally with zero distillation.

**How we discovered it:**
All 9 grid search runs (alpha 0.5/0.7/0.9 × temp 2.0/3.0/5.0) produced
identical results to 5 decimal places. Statistically impossible if
hyperparameters were actually different.

**Confirmed by:**
```python
for name, method in inspect.getmembers(DetectionTrainer, predicate=inspect.isfunction):
    if "loss" in name.lower() or "criterion" in name.lower():
        print(name)
# Output: only 'label_loss_items' — no 'criterion'
```

---

### Bug 2 — Wrong second interception point: `model.loss()` patch

**Second approach:**
```python
def _setup_train(self, *args, **kwargs):
    original_loss = self.model.loss

    def kd_loss(batch, preds=None):
        if preds is None:
            return original_loss(batch, preds)
        ...
    self.model.loss = kd_loss
```

**What happened:**
`_do_train()` calls `self.model(batch)`.
`DetectionModel.forward()` checks `isinstance(x, dict)` and calls
`self.loss(x)` with no preds argument.
So `preds` was always `None`, always falling back to original loss.
KD never fired.

**How we discovered it:**
Added debug print inside `kd_loss`:
```python
print(f"[KD] kd_loss called — preds is {'None' if preds is None else 'populated'}")
# Output: always None
```

**Root cause:**
```python
# DetectionModel.forward():
if isinstance(x, dict):
    return self.loss(x)   # ← calls loss with NO preds argument

# DetectionModel.loss():
def loss(self, batch, preds=None):
    if preds is None:
        preds = self.forward(batch["img"])   # ← computes preds internally
    return self.criterion(preds, batch)      # ← criterion is the real hook
```

---

### Bug 3 — Wrong teacher loader: `torch.load()`

**Original approach:**
```python
teacher = torch.load(weights, map_location="cpu", weights_only=False)
if isinstance(teacher, dict):
    teacher = teacher.get("model", ...)
```

**What happened:**
`torch.load()` on a .pt file returns a partially initialised Ultralytics
object. Its `forward()` method does not return the prediction dict
format (`boxes`, `scores`, `feats`). It returns processed detections.
`teacher_output[1]` was indexing into the wrong structure, producing
nonsense soft loss values near zero.

**Fix:**
```python
from ultralytics import YOLO
teacher = YOLO(weights).model   # correct initialisation
```

---

### Bug 4 — Teacher in `eval()` mode changes output format

**What happened:**
Setting `teacher.eval()` causes Ultralytics to fuse model layers and
return processed detections instead of the raw prediction dict.
`preds["scores"]` and `preds["boxes"]` were not available in eval mode.

**Fix:**
```python
teacher.train()   # keeps prediction dict format
# requires_grad=False + torch.no_grad() still prevent any updates
```

---

### Bug 5 — `model.criterion` attribute error

**Code:**
```python
if model.criterion is None:
    model.criterion = model.init_criterion()
```

**What happened:**
`nn.Module` raises `AttributeError` for missing attributes instead of
returning `None`. `model.criterion` does not exist as an attribute until
first use — it is set lazily inside `model.loss()`.

**Fix:**
```python
if getattr(model, "criterion", None) is None:
    model.criterion = model.init_criterion()
```

---

### Bug 6 — `hard_loss` is not a scalar

**Code:**
```python
total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
```

**Error:**
```
RuntimeError: a Tensor with 3 elements cannot be converted to Scalar
```

**What happened:**
`original_criterion()` returns `(loss_tensor, loss_items)` where
`loss_tensor` is shape `[3]` — the three individual components
(box_loss, cls_loss, dfl_loss) — not a scalar sum.

**Fix:**
```python
hard_loss, loss_items = original_criterion(preds, batch)
hard_loss = hard_loss.sum()   # reduce [3] → scalar
```

---

### Bug 7 — YOLO26 nested output format

**What happened:**
YOLO26 returns a nested dict instead of a flat dict:
```python
# YOLO11
preds = {'boxes': ..., 'scores': ..., 'feats': ...}

# YOLO26
preds = {
    'one2many': {'boxes': ..., 'scores': ..., 'feats': ...},
    'one2one':  {'boxes': ..., 'scores': ..., 'feats': ...},
}
```

`preds["scores"]` KeyError for YOLO26.

**Fix:**
```python
@staticmethod
def _extract(preds):
    if "one2many" in preds:
        return preds["one2many"]   # YOLO26 training head
    return preds                   # YOLO11
```

---

### Bug 8 — YOLO26 box channel count differs

**YOLO11 boxes:** `[B, 64, 8400]` — 4 × 16 DFL distribution bins  
**YOLO26 boxes:** `[B,  4, 8400]` — direct (cx, cy, w, h), DFL removed

MSE still works for both since both are continuous regression values.
But teacher and student must be the same model family — mixing
YOLO11 teacher with YOLO26 student would cause shape mismatch.

**Added assertion:**
```python
assert student_boxes.shape[1] == teacher_boxes.shape[1], (
    "Cannot mix YOLO11 and YOLO26 as teacher/student."
)
```

---

## Final Working Architecture

```
_setup_train()
    ↓
loads teacher via YOLO(weights).model
sets teacher.train() — preserves dict output format
sets requires_grad=False — no weight updates
    ↓
getattr(model, "criterion", None) is None
    → model.criterion = model.init_criterion()  (v8DetectionLoss)
    ↓
wraps model.criterion with kd_criterion()

─── per batch ────────────────────────────────────

model(batch)
    → model.forward(batch)         isinstance(x, dict) → training
    → model.loss(batch)
    → preds = model.forward(img)   student forward pass
    → kd_criterion(preds, batch)   OUR HOOK

        hard_loss, loss_items = original_criterion(preds, batch)
        hard_loss = hard_loss.sum()

        with torch.no_grad():
            teacher_preds = teacher(batch["img"])

        student_flat = _extract(preds)          handles YOLO11 + YOLO26
        teacher_flat = _extract(teacher_preds)

        soft_loss = KLDiv(scores) + MSE(boxes)

        total = alpha * hard + (1-alpha) * soft
        return total, loss_items                loss_items unchanged
```

---

## How to Verify KD is Working

Add this debug print to `kd_criterion` temporarily:

```python
print(
    f"[KD DEBUG] alpha={alpha}  T={loss_fn.T}"
    f"  hard={hard_loss.item():.4f}"
    f"  soft={soft_loss.item():.6f}"
)
```

Run two extreme configs for 1 epoch:
```python
for alpha, temp in [(0.5, 2.0), (0.9, 5.0)]:
    main(alpha=alpha, temperature=temp, epochs=1, ...)
```

**Expected output:**
```
alpha=0.5  T=2.0  hard=X.XXXX  soft=0.XXXXXX   ← soft non-zero
alpha=0.9  T=5.0  hard=X.XXXX  soft=0.XXXXXX   ← soft differs from above
```

- `soft` non-zero → teacher output format is correct
- `soft` differs between configs → temperature is wiring through
- `total` differs between configs → alpha is wiring through

---

## Version Compatibility Notes

| Ultralytics version | Interception point | preds format | Status |
|---------------------|-------------------|--------------|--------|
| < 8.0 | `DetectionTrainer.criterion()` override | `preds[1]` tuple `[P3, P4, P5]` | Not supported |
| 8.0 - 8.2 | `DetectionTrainer.criterion()` override | `preds[1]` tuple `[P3, P4, P5]` | Works with original code |
| 8.3 - 8.4.21 | `model.criterion` wrap | flat dict `{boxes, scores, feats}` | Current implementation |
| > 8.4.21 | Unknown — re-inspect `_do_train` | Unknown | Re-run inspection chain |

**If Ultralytics is upgraded**, run this inspection chain first:
```python
import inspect
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

# Find loss-related methods
for name, method in inspect.getmembers(DetectionTrainer, predicate=inspect.isfunction):
    if "loss" in name.lower() or "criterion" in name.lower():
        print(name)

# Trace the call chain
print(inspect.getsource(DetectionModel.forward))
print(inspect.getsource(DetectionModel.loss))
```

Then verify teacher output format:
```python
from ultralytics import YOLO
import torch
m = YOLO("yolo11x.pt").model
m.train()
out = m(torch.zeros(2, 3, 640, 640))
print(type(out))
print(out.keys() if isinstance(out, dict) else [x.shape for x in out])
```
