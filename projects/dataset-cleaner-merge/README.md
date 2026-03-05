# Dataset Structure Reference

This document describes the directory layout of the merged dataset produced by the orchestrator, and the format of every file type it generates. It covers `images/`, `labels/`, and `meta/` in full detail.

---

## Top-Level Layout

```
Dataset/merged_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── meta/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset.yaml
└── split_manifest.csv
```

The three splits — `train`, `val`, `test` — are produced by stratified splitting per source dataset (KITTI and EuroCity are each split independently, then merged). Default ratios are 70 / 15 / 15.

---

## Naming Convention

Every file across all three directories (`images/`, `labels/`, `meta/`) shares the same stem, prefixed with the source dataset name to avoid collisions:

```
<source>_<original_stem>.<ext>

kitti_000001.png        ← image
kitti_000001.txt        ← YOLO label
kitti_000001.json       ← metadata sidecar

eurocity_aachen_000001_000019.png
eurocity_aachen_000001_000019.txt
eurocity_aachen_000001_000019.json
```

This means for any given stem you can always find all three files by just swapping the directory and extension.

---

## images/

Contains the actual image files (`.png`). By default these are **symlinks** pointing to the original source location to avoid duplicating disk space. If `copy_images=True` was passed to the orchestrator, they are physical copies instead.

```
images/
└── train/
    ├── kitti_000001.png        → /home/aga2/Dataset/KITTI/images/training/image_2/000001.png
    ├── kitti_000002.png        → ...
    ├── eurocity_aachen_000001_000019.png  → /home/aga2/Dataset/EuroCity/...
    └── ...
```

Nothing in this directory is ever read by the pipeline itself — it exists purely for YOLO training, which needs the image path to load pixels.

---

## labels/

Contains YOLO-format `.txt` label files. One file per image, same stem as the image.

### Format

Each line in the file is one bounding box:

```
<class_id> <cx> <cy> <w> <h>
```

All values are **space-separated**. `cx`, `cy`, `w`, `h` are **normalised** to `[0, 1]` relative to image width and height. `class_id` is an integer.

### Class IDs

| ID | Class          | KITTI sources                        | EuroCity sources                          |
|----|----------------|--------------------------------------|-------------------------------------------|
| 0  | pedestrian     | `Pedestrian`, `Person_sitting`       | `pedestrian`                              |
| 1  | cyclist        | `Cyclist`                            | `rider`, `motocyclist`, `scooter`, `motorbike` |
| 2  | car            | `Car`, `Van`                         | `car`, `van`, `taxi`, `pickup`, `vehicle` |
| 3  | large_vehicle  | `Truck`, `Tram`, `Bus`               | `truck`, `bus`, `tram`, `trailer`, `large-vehicle` |

### What is dropped / excluded from labels

These appear in the raw source annotations but are **never written** to `.txt` label files:

- **KITTI `DontCare`** — kept in metadata only (see below)
- **KITTI `Misc`** — dropped entirely
- **EuroCity group annotations** (`bicycle-group`, `scooter-group`, etc.) — riderless vehicle groups, not individual detectable instances
- **EuroCity `depiction` / `reflection` tag** — signs, posters, mirrors; dropped entirely
- **Zero-area boxes** — boxes where clamping to image bounds produces `w=0` or `h=0`

### Example label file

```
# kitti_000001.txt
0 0.757883 0.448438 0.102148 0.256771   ← pedestrian
2 0.573958 0.357454 0.025000 0.044010   ← car
2 0.535556 0.349753 0.072396 0.063182   ← car
2 0.289670 0.401563 0.184201 0.095573   ← car
```

Empty `.txt` files are valid — they represent **negative samples** (images with no annotated objects of the 4 target classes).

### Cache files

When YOLO first scans the labels directory it writes a `.cache` file (e.g. `labels/train.cache`). This file records class counts and is tied to `nc` in `dataset.yaml`. **If you change `nc` or regenerate the dataset, always delete the cache files before retraining:**

```bash
rm Dataset/merged_dataset/labels/train.cache
rm Dataset/merged_dataset/labels/val.cache
rm Dataset/merged_dataset/labels/test.cache
```

---

## meta/

Contains JSON metadata sidecars. One file per image, same stem as the image. These are **not used by YOLO training** — they exist for evaluation (difficulty-stratified AP, per-class IoU thresholds) and for debugging the preprocessing pipeline.

### Top-level structure

```json
{
  "image":       "relative/path/to/original/frame.png",
  "source":      "kitti",
  "n_boxes":     4,
  "annotations": [ ... ]
}
```

### Per-annotation entry

Each object in `annotations` corresponds to one raw annotation from the source dataset. This includes objects that were **dropped from the YOLO label** (e.g. `DontCare`) so the full annotation picture is preserved.

#### KITTI annotation entry

```json
{
  "class_id":      2,
  "identity":      "Car",
  "bbox_px":       [548.0, 171.33, 572.40, 194.42],
  "bbox_yolo":     [0.5736, 0.3574, 0.0250, 0.0440],
  "bbox_h_px":     23.09,
  "occlusion":     2,
  "truncation":    0.0,
  "difficulty":    "Ignore",
  "iou_threshold": 0.70,
  "is_dontcare":   false
}
```

#### EuroCity annotation entry

```json
{
  "class_id":   0,
  "identity":   "pedestrian",
  "bbox_px":    [100.0, 200.0, 160.0, 350.0],
  "bbox_yolo":  [0.0677, 0.2612, 0.0313, 0.1465],
  "bbox_h_px":  150.0,
  "occlusion":  0,
  "truncation": 0.0,
  "tags":       [],
  "difficulty": "Easy"
}
```

### Field reference

| Field           | Type          | Description |
|-----------------|---------------|-------------|
| `class_id`      | `int \| null` | YOLO class ID (0–3). `null` for KITTI `DontCare`. |
| `identity`      | `str`         | Original class string from source annotation. |
| `bbox_px`       | `[x0,y0,x1,y1]` | Bounding box in pixel coordinates, clamped to image bounds. |
| `bbox_yolo`     | `[cx,cy,w,h]` | Normalised YOLO box. `null` for `DontCare` or zero-area boxes. |
| `bbox_h_px`     | `float`       | Bounding box height in pixels **before** clamping. Used for difficulty. |
| `occlusion`     | `int`         | 0 = visible, 1 = partly occluded, 2 = heavily occluded. |
| `truncation`    | `float`       | 0.0–1.0. Fraction of box outside image boundary. |
| `difficulty`    | `str`         | `Easy`, `Moderate`, `Hard`, or `Ignore`. See thresholds below. |
| `iou_threshold` | `float \| null` | **KITTI only.** Per-class eval IoU: 0.70 for car/large_vehicle, 0.50 for pedestrian/cyclist. |
| `is_dontcare`   | `bool`        | **KITTI only.** True for `DontCare` regions. |
| `tags`          | `list[str]`   | **EuroCity only.** Raw tags from the source JSON. |

### Difficulty thresholds

Difficulty is computed per annotation using KITTI's official criteria, applied consistently to both datasets:

| Level    | Min bbox height | Max occlusion | Max truncation |
|----------|-----------------|---------------|----------------|
| Easy     | 40 px           | 0             | 0.15           |
| Moderate | 25 px           | 1             | 0.30           |
| Hard     | 25 px           | 2             | 0.50           |
| Ignore   | does not meet Hard | —          | —              |

For EuroCity, occlusion is inferred from object tags (`occluded` → 1, `heavy-occlusion` → 2). Truncation is approximated geometrically from how far the bounding box extends beyond the image border.

### How to use metadata at evaluation time

Filter your predictions by difficulty before computing AP:

```python
import json
from pathlib import Path

def load_meta(meta_dir, split, stem):
    path = Path(meta_dir) / "meta" / split / f"{stem}.json"
    return json.loads(path.read_text())

def get_annotations_by_difficulty(meta, difficulty):
    return [a for a in meta["annotations"] if a["difficulty"] == difficulty]

# Example: get all Easy pedestrian GT boxes for an image
meta = load_meta("Dataset/merged_dataset", "val", "kitti_000001")
easy_peds = [
    a for a in meta["annotations"]
    if a["difficulty"] == "Easy" and a["class_id"] == 0
]
```

---

## dataset.yaml

Read by Ultralytics YOLO at training and validation time. Auto-generated by the orchestrator.

```yaml
path:  /home/aga2/Dataset/merged_dataset
train: images/train
val:   images/val
test:  images/test
nc:    4
names:
  - pedestrian
  - cyclist
  - car
  - large_vehicle
```

---

## split_manifest.csv

A full audit trail of every sample in the dataset. One row per image.

| Column      | Description |
|-------------|-------------|
| `split`     | `train`, `val`, or `test` |
| `source`    | `kitti` or `eurocity` |
| `original`  | Absolute path to the source image |
| `image_out` | Path to the image in the merged dataset |
| `label_out` | Path to the corresponding `.txt` label file |
| `meta_out`  | Path to the corresponding `.json` metadata sidecar |
| `n_boxes`   | Number of YOLO boxes written to the label file |

Useful for debugging class imbalance, verifying split ratios, and tracing a specific image back to its source.
