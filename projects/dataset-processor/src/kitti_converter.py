"""
kitti_converter.py
------------------
Converts the KITTI Object Detection dataset to YOLO format.

KITTI does not ship an official train/val split — the test set labels are
withheld on the evaluation server. The community convention is to split the
7,481 labelled training images by filename order:

    train: first 6,000 images  (indices 000000 – 005999)
    val:   remaining 1,481     (indices 006000 – 007480)

This is deterministic and avoids temporal data leakage, because KITTI images
are sequential frames from a continuous drive. Randomly splitting would place
near-identical frames in both train and val.

Source layout expected:
    <images_dir>/           e.g.  kitti/training/image_2/
        000000.png
        000001.png
        ...
    <labels_dir>/           e.g.  kitti/training/label_2/
        000000.txt
        000001.txt
        ...

KITTI label format (space-separated columns per row):
    Col  Field        Type    Description
    ---  -----------  ------  ----------------------------------------
     0   type         str     Object class  e.g. "Pedestrian", "Car"
     1   truncated    float   0.0–1.0  fraction cut off by image border
     2   occluded     int     0=visible, 1=partly, 2=largely, 3=unknown
     3   alpha        float   Observation angle [-π, π]
     4   bbox_x1      float   2-D bbox left   (pixels)
     5   bbox_y1      float   2-D bbox top    (pixels)
     6   bbox_x2      float   2-D bbox right  (pixels)
     7   bbox_y2      float   2-D bbox bottom (pixels)
     8–10 dimensions  float   3-D height/width/length (metres) — ignored
    11–13 location    float   3-D x/y/z (metres)              — ignored
    14   rotation_y   float   3-D rotation                     — ignored

Filtering applied:
    - type must be "Pedestrian"  (case-sensitive, matches KITTI convention)
    - occluded must NOT be 3     (fully occluded / unknown — unreliable label)

Usage:
    from dataset_converter import KITTIConverter

    for split in ("train", "val"):
        KITTIConverter(
            images_dir="kitti/training/image_2",
            labels_dir="kitti/training/label_2",
            output_dir="dataset",
            split=split,
        ).convert()
"""

from __future__ import annotations

import logging
from pathlib import Path

from base_converter import DatasetConverter
from bbox import Annotation, BoundingBox

logger = logging.getLogger(__name__)


class KITTIConverter(DatasetConverter):
    """
    Converts KITTI Object Detection annotations to YOLO format.

    Args:
        images_dir:  Directory containing KITTI training images (.png).
        labels_dir:  Directory containing KITTI label files (.txt).
        output_dir:  Root output directory for the YOLO dataset.
        split:       "train" or "val" — controls which images are processed
                     and which output subfolder is written to.
        train_count: Number of images assigned to the train split.
                     Default is 6000, matching community convention.
                     The remainder go to val.
    """

    # KITTI .txt column indices
    _COL_TYPE      = 0
    _COL_OCCLUDED  = 2
    _COL_X1        = 4
    _COL_Y1        = 5
    _COL_X2        = 6
    _COL_Y2        = 7

    # Labels treated as "pedestrian" for class_id = 0
    _PEDESTRIAN_CLASSES: frozenset[str] = frozenset({"Pedestrian"})

    # Occluded value to reject (fully occluded / unknown)
    _OCCLUDED_REJECT = 3

    # Community-standard split boundary
    DEFAULT_TRAIN_COUNT = 6000

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        output_dir: str | Path,
        split: str = "train",
        train_count: int = DEFAULT_TRAIN_COUNT,
    ):
        super().__init__(output_dir, split)
        self.images_dir  = Path(images_dir)
        self.labels_dir  = Path(labels_dir)
        self.train_count = train_count

    # ------------------------------------------------------------------
    # iter_samples — deterministic split by filename order
    # ------------------------------------------------------------------

    def iter_samples(self):
        """
        Yield (image_path, label_path) for the requested split.

        All images are sorted by filename (numeric order) first, then
        split at index train_count:
            train → indices [0, train_count)
            val   → indices [train_count, end)

        Only images that have a matching label file are included.
        """
        # Collect all images that have a corresponding label file
        all_samples = sorted(
            (img, self.labels_dir / (img.stem + ".txt"))
            for img in self.images_dir.glob("*.png")
            if (self.labels_dir / (img.stem + ".txt")).exists()
        )

        if not all_samples:
            logger.warning("No paired image/label files found in %s", self.images_dir)
            return

        if self.split == "train":
            selected = all_samples[: self.train_count]
        else:  # val
            selected = all_samples[self.train_count :]

        logger.info(
            "KITTI split=%s | total_paired=%d | selected=%d",
            self.split,
            len(all_samples),
            len(selected),
        )

        yield from selected

    # ------------------------------------------------------------------
    # parse_annotations — KITTI .txt format
    # ------------------------------------------------------------------

    def parse_annotations(
        self,
        label_path: Path,
        image_path: Path,  # unused, part of interface
    ) -> list[Annotation]:
        """
        Parse one KITTI label file and return pedestrian annotations.

        Rows are dropped if:
            - type is not in _PEDESTRIAN_CLASSES
            - occluded == _OCCLUDED_REJECT (3)
            - resulting BoundingBox fails is_valid()
        """
        annotations: list[Annotation] = []

        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Guard against malformed lines
            if len(parts) < 8:
                logger.debug(
                    "%s line %d: expected ≥8 columns, got %d — skipping",
                    label_path.name, line_no, len(parts),
                )
                continue

            obj_type = parts[self._COL_TYPE]
            occluded = int(parts[self._COL_OCCLUDED])

            # Filter 1: pedestrian class only
            if obj_type not in self._PEDESTRIAN_CLASSES:
                continue

            # Filter 2: drop fully occluded / unknown
            if occluded == self._OCCLUDED_REJECT:
                continue

            bbox = BoundingBox(
                x1=float(parts[self._COL_X1]),
                y1=float(parts[self._COL_Y1]),
                x2=float(parts[self._COL_X2]),
                y2=float(parts[self._COL_Y2]),
            )

            if not bbox.is_valid():
                logger.debug(
                    "%s line %d: invalid bbox %s — skipping",
                    label_path.name, line_no, bbox,
                )
                continue

            annotations.append(Annotation(bbox=bbox, class_id=self.CLASS_ID))

        return annotations