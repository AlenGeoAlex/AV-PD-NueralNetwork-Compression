"""
citypersons_converter.py
------------------------
Converts the CityPersons dataset (Cityscapes-based) to YOLO format.

CityPersons (Zhang et al., CVPR 2017) is built on top of Cityscapes imagery.
It ships with an OFFICIAL train/val split organised as city subdirectories:

    leftImg8bit/
        train/   ← 18 cities
        val/     ←  3 cities  (frankfurt, lindau, munster)

    gtBboxCityPersons/
        train/
        val/

We respect this official split directly. Do NOT randomly re-split, because:
    - All frames within a city share the same scene geometry and appearance.
    - Random splitting would cause scene leakage across train and val.

Source layout expected:
    <images_dir>/                   e.g.  cityscapes/leftImg8bit/train/
        <city>/
            <city>_<seq>_<frame>_leftImg8bit.png

    <annotations_dir>/              e.g.  cityscapes/gtBboxCityPersons/train/
        <city>/
            <city>_<seq>_<frame>_gtBboxCityPersons.json

CityPersons JSON format (one file per image):
    {
        "imgWidth":  int,
        "imgHeight": int,
        "objects": [
            {
                "label":   str,          e.g. "pedestrian", "rider", "ignore"
                "bbox":    [x, y, w, h], full bounding box (pixels, top-left origin)
                "bboxVis": [x, y, w, h]  visible portion of the bounding box
            },
            ...
        ]
    }

Filtering applied:
    - label must be in _PEDESTRIAN_LABELS
    - bboxVis width and height must both be > 0  (zero = fully occluded)
    - resulting BoundingBox must pass is_valid()

We use bboxVis (visible portion) rather than the full bbox, so that
occluded regions do not inflate the ground-truth box during training.

Usage:
    from dataset_converter import CityPersonsConverter

    for split in ("train", "val"):
        CityPersonsConverter(
            images_dir=f"cityscapes/leftImg8bit/{split}",
            annotations_dir=f"cityscapes/gtBboxCityPersons/{split}",
            output_dir="dataset",
            split=split,
        ).convert()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from base_converter import DatasetConverter
from bbox import Annotation, BoundingBox

logger = logging.getLogger(__name__)


class CityPersonsConverter(DatasetConverter):
    """
    Converts CityPersons (Cityscapes) annotations to YOLO format.

    Args:
        images_dir:       Directory containing leftImg8bit images,
                          organised as <city>/<image>.png sub-folders.
        annotations_dir:  Directory containing gtBboxCityPersons JSON files,
                          organised as <city>/<annotation>.json sub-folders.
        output_dir:       Root output directory for the YOLO dataset.
        split:            "train" or "val" — must match the source directories
                          passed in (no internal re-splitting is done here;
                          the official Cityscapes split is used as-is).
    """

    # CityPersons object labels to treat as pedestrian (class_id = 0)
    # "rider" is excluded — they are cyclists/motorbike riders, not pedestrians
    # "ignore" regions are excluded by default
    _PEDESTRIAN_LABELS: frozenset[str] = frozenset({
        "pedestrian",
        "sitting person",
        "person (other)",
    })

    # Annotation filename suffix
    _ANN_SUFFIX  = "_gtBboxCityPersons.json"

    # Image filename suffix
    _IMG_SUFFIX  = "_leftImg8bit.png"

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        output_dir: str | Path,
        split: str = "train",
    ):
        super().__init__(output_dir, split)
        self.images_dir      = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)

    # ------------------------------------------------------------------
    # iter_samples — walk city subdirectories, match image ↔ annotation
    # ------------------------------------------------------------------

    def iter_samples(self):
        """
        Walk all city subdirectories under annotations_dir and yield
        (image_path, annotation_path) pairs where both files exist.

        Naming convention:
            Annotation: <city>_<seq>_<frame>_gtBboxCityPersons.json
            Image:      <city>_<seq>_<frame>_leftImg8bit.png

        Both live under a <city>/ subdirectory inside their respective roots.
        """
        ann_paths = sorted(self.annotations_dir.rglob(f"*{self._ANN_SUFFIX}"))

        if not ann_paths:
            logger.warning(
                "No annotation files found under %s", self.annotations_dir
            )
            return

        found = 0
        for ann_path in ann_paths:
            # Derive matching image path
            # e.g.  aachen_000000_000019_gtBboxCityPersons.json
            #    →  aachen_000000_000019_leftImg8bit.png
            stem = ann_path.name.replace(self._ANN_SUFFIX, "")
            city = ann_path.parent.name

            image_path = self.images_dir / city / f"{stem}{self._IMG_SUFFIX}"

            if image_path.exists():
                found += 1
                yield image_path, ann_path
            else:
                logger.debug(
                    "Image not found for annotation %s (expected %s)",
                    ann_path.name,
                    image_path,
                )

        logger.info(
            "CityPersons split=%s | annotations=%d | matched=%d",
            self.split,
            len(ann_paths),
            found,
        )

    # ------------------------------------------------------------------
    # parse_annotations — CityPersons JSON format
    # ------------------------------------------------------------------

    def parse_annotations(
        self,
        label_path: Path,
        image_path: Path,  # unused, part of interface
    ) -> list[Annotation]:
        """
        Parse one CityPersons JSON annotation file.

        Uses bboxVis (visible bounding box portion).
        Falls back to full bbox only if bboxVis is absent from the file.

        Objects are dropped if:
            - label not in _PEDESTRIAN_LABELS
            - bboxVis (or bbox) has zero or negative width/height
            - resulting BoundingBox fails is_valid()
        """
        data = json.loads(label_path.read_text())
        annotations: list[Annotation] = []

        for obj in data.get("objects", []):
            label = obj.get("label", "").strip().lower()

            # Filter: pedestrian labels only
            if label not in self._PEDESTRIAN_LABELS:
                continue

            # Prefer bboxVis (visible portion); fall back to full bbox
            raw_box = obj.get("bboxVis") or obj.get("bbox")
            if raw_box is None:
                logger.debug("%s: object '%s' has no bbox — skipping", label_path.name, label)
                continue

            x, y, w, h = raw_box

            # Drop if the visible area is zero (fully occluded)
            if w <= 0 or h <= 0:
                continue

            bbox = BoundingBox.from_xywh(x, y, w, h)

            if not bbox.is_valid():
                logger.debug(
                    "%s: invalid bbox after conversion %s — skipping",
                    label_path.name, bbox,
                )
                continue

            annotations.append(Annotation(bbox=bbox, class_id=self.CLASS_ID))

        return annotations