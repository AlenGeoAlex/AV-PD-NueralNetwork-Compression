"""
Reader and annotation converter for the KITTI Object Detection dataset.

Expected directory layout:
    <kitti_root>/
        training/
            image_2/
                <frame_id>.png
            label_2/
                <frame_id>.txt

KITTI label format (space-separated, one object per line):
    Field  0  : class string
    Field  1  : truncation  (0.0–1.0; -1 for DontCare)
    Field  2  : occlusion   (0=visible,1=partly,2=fully,3=unknown; -1 for DontCare)
    Field  3  : alpha       (observation angle, not used here)
    Fields 4-7: 2-D bbox    left top right bottom  (pixels, 0-indexed)
    Fields 8-10: 3-D size   height width length     (metres)
    Fields 11-13: 3-D loc   x y z                   (metres, camera coords)
    Field 14  : rotation_y  (not used here)

Class mapping
-------------
    Pedestrian / Person_sitting  → 0  (pedestrian)
    Cyclist                      → 1  (cyclist)
    Car / Van                    → 2  (car)
    Truck / Tram / Bus           → 3  (large vehicle)
    DontCare                     → metadata only (no YOLO label emitted)
    Misc / everything else       → DROPPED

Difficulty (official KITTI thresholds)
---------------------------------------
    Easy     : bbox_h >= 40px, occlusion = 0,   truncation <= 0.15
    Moderate : bbox_h >= 25px, occlusion <= 1,  truncation <= 0.30
    Hard     : bbox_h >= 25px, occlusion <= 2,  truncation <= 0.50
    Ignore   : does not meet Hard criteria  (also used for DontCare)

IoU thresholds (for evaluation — stored in metadata)
------------------------------------------------------
    car        : 0.70
    pedestrian : 0.50
    cyclist    : 0.50

Metadata sidecar format (JSON, one file per image, mirrors label_2/ tree)
--------------------------------------------------------------------------
    {
      "image": "000001.png",
      "source": "kitti",
      "imagewidth": int,
      "imageheight": int,
      "annotations": [
        {
          "class_id":   int | null,       ← null for DontCare
          "identity":   str,              ← original KITTI class string
          "bbox_px":    [x1, y1, x2, y2],
          "bbox_yolo":  [cx, cy, w, h],   ← null for DontCare
          "bbox_h_px":  float,
          "occlusion":  int,              ← 0 / 1 / 2 / 3
          "truncation": float,
          "difficulty": str,
          "iou_threshold": float | null,  ← per-class eval IoU
          "is_dontcare": bool
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

from .base import AnnotationConverter, BoundingBox, DatasetReader, Sample

logger = logging.getLogger(__name__)


CLASS_PEDESTRIAN    = 0
CLASS_CYCLIST       = 1
CLASS_CAR           = 2
CLASS_LARGE_VEHICLE = 3

_CLASS_MAP: dict[str, tuple[int, float]] = {
    "pedestrian":     (CLASS_PEDESTRIAN,    0.50),
    "person_sitting": (CLASS_PEDESTRIAN,    0.50),
    "cyclist":        (CLASS_CYCLIST,       0.50),
    "car":            (CLASS_CAR,           0.70),
    "van":            (CLASS_CAR,           0.70),
    "truck":          (CLASS_LARGE_VEHICLE, 0.70),
    "tram":           (CLASS_LARGE_VEHICLE, 0.70),
    "bus":            (CLASS_LARGE_VEHICLE, 0.70),
}

_DONTCARE = "dontcare"
_DROPPED  = {"misc"}

_DIFF_THRESHOLDS = [
    ("Easy",     40, 0, 0.15),
    ("Moderate", 25, 1, 0.30),
    ("Hard",     25, 2, 0.50),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_difficulty(bbox_h_px: float, occlusion: int,
                        truncation: float) -> str:
    for label, min_h, max_occ, max_trunc in _DIFF_THRESHOLDS:
        if bbox_h_px >= min_h and occlusion <= max_occ and truncation <= max_trunc:
            return label
    return "Ignore"


def _parse_line(line: str) -> dict | None:
    """
    Parse one KITTI label line.
    Returns a raw dict or None if the line is malformed / should be skipped.
    """
    parts = line.strip().split()
    if len(parts) < 15:
        return None

    cls_str    = parts[0]
    cls_lower  = cls_str.lower()

    try:
        truncation = float(parts[1])
        occlusion  = int(parts[2])
        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])
    except (ValueError, IndexError):
        return None

    return {
        "cls_str":    cls_str,
        "cls_lower":  cls_lower,
        "truncation": truncation,
        "occlusion":  occlusion,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    }


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class KITTIAnnotationConverter(AnnotationConverter):
    """
    Converts a single KITTI .txt label file into:
      - a list of BoundingBox  (YOLO training labels; DontCare excluded)
      - a list of metadata dicts  (all annotations including DontCare)
    """

    def convert(
        self,
        annotation_path: Path,
        image_w: int,
        image_h: int,
    ) -> tuple[list[BoundingBox], list[dict]]:

        boxes:    list[BoundingBox] = []
        metadata: list[dict]        = []

        if not annotation_path.exists():
            logger.warning("Annotation not found: %s", annotation_path)
            return boxes, metadata

        lines = annotation_path.read_text().splitlines()

        for line in lines:
            if not line.strip():
                continue

            raw = _parse_line(line)
            if raw is None:
                logger.debug("Malformed line in %s: %r", annotation_path.name, line)
                continue

            cls_lower  = raw["cls_lower"]
            cls_str    = raw["cls_str"]
            truncation = max(0.0, raw["truncation"])   # DontCare has -1
            occlusion  = max(0,   raw["occlusion"])    # DontCare has -1
            x1, y1, x2, y2 = raw["x1"], raw["y1"], raw["x2"], raw["y2"]

            is_dontcare = cls_lower == _DONTCARE

            # Silently skip misc / unknown classes
            if cls_lower in _DROPPED:
                continue
            if not is_dontcare and cls_lower not in _CLASS_MAP:
                logger.debug("Unknown class %r in %s — dropped", cls_str, annotation_path.name)
                continue

            bbox_h_px  = y2 - y1
            difficulty = _compute_difficulty(bbox_h_px, occlusion, truncation)

            if is_dontcare:
                class_id      = None
                iou_threshold = None
            else:
                class_id, iou_threshold = _CLASS_MAP[cls_lower]

            # Clamp to image bounds
            x1c = max(0.0, min(x1, image_w))
            y1c = max(0.0, min(y1, image_h))
            x2c = max(0.0, min(x2, image_w))
            y2c = max(0.0, min(y2, image_h))

            bw = x2c - x1c
            bh = y2c - y1c

            # YOLO coords (only for real classes, not DontCare)
            if not is_dontcare and bw > 0 and bh > 0:
                cx = (x1c + bw / 2) / image_w
                cy = (y1c + bh / 2) / image_h
                w  = bw / image_w
                h  = bh / image_h

                boxes.append(BoundingBox(
                    class_id=class_id,
                    cx=cx, cy=cy, w=w, h=h,
                ))
                bbox_yolo = [cx, cy, w, h]
            else:
                bbox_yolo = None

            metadata.append({
                "class_id":      class_id,
                "identity":      cls_str,
                "bbox_px":       [x1c, y1c, x2c, y2c],
                "bbox_yolo":     bbox_yolo,
                "bbox_h_px":     bbox_h_px,
                "occlusion":     occlusion,
                "truncation":    round(truncation, 4),
                "difficulty":    difficulty,
                "iou_threshold": iou_threshold,
                "is_dontcare":   is_dontcare,
            })

        return boxes, metadata


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class KITTIReader(DatasetReader):
    """
    Walks a KITTI split and yields Sample objects, optionally writing
    JSON metadata sidecars that mirror the label_2/ tree.

    Parameters
    ----------
    kitti_root : Path
        Root directory containing training/ or testing/ subdirectories.
    meta_dir : Path | None
        Root under which metadata sidecars are written.
        Structure: <meta_dir>/training/meta/<frame>.json
        If None, no sidecars are written (metadata still on Sample).
    split : str
        "training" or "testing". Default "training".
    image_ext : str
        Image extension. Default ".png".
    include_no_label_images : bool
        Yield test images that have no label file. Default False.
    write_meta : bool
        Write JSON sidecar files. Default True.
    """

    def __init__(
        self,
        kitti_root: str | Path,
        meta_dir: str | Path | None = None,
        split: str = "training",
        image_ext: str = ".png",
        include_no_label_images: bool = False,
        write_meta: bool = True,
    ) -> None:
        self._root              = Path(kitti_root)
        self._meta_root         = Path(meta_dir) if meta_dir else None
        self._split             = split
        self._image_ext         = image_ext
        self._include_no_label  = include_no_label_images
        self._write_meta        = write_meta and (self._meta_root is not None)
        self._converter         = KITTIAnnotationConverter()

        self._image_dir = self._root / "images" / split / "image_2"
        self._label_dir = self._root / "labels" / split / "label_2"

        if not self._image_dir.exists():
            raise FileNotFoundError(
                f"KITTI image directory not found: {self._image_dir}"
            )
        if not self._label_dir.exists() and not include_no_label_images:
            raise FileNotFoundError(
                f"KITTI label directory not found: {self._label_dir}"
            )

        if self._write_meta:
            self._meta_dir = self._meta_root / split / "meta"
            self._meta_dir.mkdir(parents=True, exist_ok=True)
            logger.info("KITTI metadata will be written to: %s", self._meta_dir)
        else:
            self._meta_dir = None

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "kitti"

    # ------------------------------------------------------------------

    def _write_sidecar(
        self,
        img_path: Path,
        image_w: int,
        image_h: int,
        annotations: list[dict],
    ) -> None:
        sidecar = self._meta_dir / img_path.with_suffix(".json").name
        payload = {
            "image":       img_path.name,
            "source":      self.name,
            "imagewidth":  image_w,
            "imageheight": image_h,
            "annotations": annotations,
        }
        try:
            with sidecar.open("w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            logger.warning("Could not write sidecar %s: %s", sidecar, exc)

    # ------------------------------------------------------------------

    def read(self) -> Iterator[Sample]:
        image_paths = sorted(self._image_dir.glob(f"*{self._image_ext}"))

        if not image_paths:
            logger.warning("No images found under %s", self._image_dir)
            return

        logger.info(
            "KITTI: found %d images under %s", len(image_paths), self._image_dir
        )

        for img_path in image_paths:
            label_path = self._label_dir / img_path.with_suffix(".txt").name

            if not label_path.exists():
                if self._include_no_label:
                    yield Sample(
                        image_path=img_path,
                        boxes=[],
                        source=self.name,
                        split_hint=self._split,
                    )
                else:
                    logger.debug("No label for %s — skipping", img_path.name)
                continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception as exc:
                logger.warning("Cannot open image %s: %s", img_path, exc)
                continue

            boxes, annotations = self._converter.convert(label_path, w, h)

            if self._write_meta:
                self._write_sidecar(img_path, w, h, annotations)

            yield Sample(
                image_path=img_path,
                boxes=boxes,
                source=self.name,
                split_hint=self._split,
                meta=annotations,
            )