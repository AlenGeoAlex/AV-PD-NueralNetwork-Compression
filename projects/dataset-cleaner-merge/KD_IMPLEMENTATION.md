# Knowledge Distillation — Implementation Reference

## What This Is

Response-based Knowledge Distillation pipeline for YOLO11 and YOLO26,
built on top of Ultralytics 8.4.21+.

**Teacher:** YOLO11x or YOLO26x (large, frozen)  
**Student:** YOLO11s or YOLO26s (small, being trained)  
**Dataset:** KITTI + EuroCity (4 classes: pedestrian, cyclist, car, large_vehicle)

---

## How KD Works

### Loss Formula

```
Total Loss = alpha * Hard Loss + (1 - alpha) * Soft Loss
```

| Component  | What it is | Default weight |
|------------|-----------|----------------|
| Hard Loss  | Standard YOLO detection loss against GT labels | alpha = 0.7 |
| Soft Loss  | How closely student matches teacher predictions | 1 - alpha = 0.3 |

### Soft Loss Components

**Classification — KL Divergence with temperature scaling**
```
student_scores: [B, nc, 8400]  →  log_softmax(scores / T)
teacher_scores: [B, nc, 8400]  →  softmax(scores / T)
loss = KLDiv(student, teacher) * T²
```
Multiply by T² to restore gradient magnitude (Hinton 2015).

**Box regression — MSE**
```
student_boxes: [B, 64, 8400]  (YOLO11 — DFL bins)
teacher_boxes: [B,  4, 8400]  (YOLO26 — direct cx/cy/w/h)
loss = MSE(student_boxes, teacher_boxes)
```

### Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| alpha | 0.7 | Higher = trust GT more, less distillation |
| temperature | 3.0 | Higher = softer distributions, more dark knowledge |
| lr0 | 0.001 | Lower than standard (0.01) — fine-tuning, not training from scratch |

---

## File Structure

```
kd/
    __init__.py              exports: main, DistillationTrainer, DistillationLoss
    distillation_loss.py     soft loss calculator — KLDiv + MSE
    distillation_trainer.py  subclasses DetectionTrainer, wraps model.criterion
    train_kd.py              entry point — sanity checks + launches training
ft/
    __init__.py
    finetune.py              common fine-tuning script for all stages
```

---

## Training Flow (per batch)

```
1. model(batch)
       ↓
2. model.loss(batch)
       ↓
3. preds = model.forward(batch["img"])    ← student forward pass
       ↓
4. model.criterion(preds, batch)          ← OUR HOOK (kd_criterion)
       ↓
5a. hard_loss = original_criterion(preds, batch)   ← GT labels
5b. teacher_preds = teacher(batch["img"])          ← teacher forward (no grad)
5c. soft_loss = DistillationLoss(student, teacher) ← KL + MSE
5d. total = alpha * hard + (1-alpha) * soft
       ↓
6. total.backward()                       ← student only, teacher frozen
```

---

## Model Output Format Differences

### YOLO11 — flat dict
```python
preds = {
    'boxes':  [B, 64, 8400],   # 4 * 16 DFL bins
    'scores': [B, nc, 8400],   # class logits
    'feats':  [list of 3],     # raw neck features (not used)
}
```

### YOLO26 — nested dict
```python
preds = {
    'one2many': {              # training head
        'boxes':  [B, 4, 8400],    # direct cx/cy/w/h — DFL removed
        'scores': [B, nc, 8400],
        'feats':  [list of 3],
    },
    'one2one': {               # inference head (replaces NMS)
        'boxes':  [B, 4, 8400],
        'scores': [B, nc, 8400],
        'feats':  [list of 3],
    }
}
```

`DistillationLoss._extract()` handles both formats automatically:
```python
if "one2many" in preds:
    return preds["one2many"]   # YOLO26
return preds                   # YOLO11
```

---

## Usage

### Single run
```python
from kd.train_kd import main

main(
    teacher_weights = "yolo11x.pt",
    student_weights = "runs/finetune/yolo11s/weights/best.pt",
    data_config     = "pedestrian.yaml",
    alpha           = 0.7,
    temperature     = 3.0,
    epochs          = 50,
    batch_size      = 32,
)
```

### Hyperparameter grid search
```python
for alpha in [0.5, 0.7, 0.9]:
    for temp in [2.0, 3.0, 5.0]:
        main(
            teacher_weights = "yolo11x.pt",
            student_weights = student_weights,
            data_config     = "pedestrian.yaml",
            alpha           = alpha,
            temperature     = temp,
            epochs          = 20,     # reduced for grid search
            batch_size      = 32,
        )
```

### Fine-tuning (before KD)
```python
from ft.finetune import finetune, BASELINE

student_weights = finetune(
    BASELINE(weights="yolo11s.pt", name="yolo11s_baseline")
)
```

---

## Baseline Comparison Plan

| Model | Description | Expected |
|-------|-------------|---------|
| YOLO11n pretrained | No fine-tuning | Lower bound |
| YOLO11s no KD | Fine-tuned, no distillation | Baseline |
| YOLO11s + KD | Fine-tuned student + teacher guidance | Target |
| YOLO11x teacher | Upper bound | Upper bound |

**KD succeeds if:** `mAP(YOLO11s+KD) > mAP(YOLO11s no KD)` while FPS and model size stay the same.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| mAP50 | IoU threshold 0.5 — standard detection metric |
| mAP50-95 | IoU 0.5→0.95 — stricter, rewards box precision |
| Precision | Correct detections / total detections |
| Recall | Found objects / total real objects |
| FPS | 1000 / mean_latency_ms (200 runs, 50 warmup) |
| Size (MB) | Weights file size |
| Params (M) | Total parameter count |
