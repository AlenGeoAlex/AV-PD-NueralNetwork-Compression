"""
Reader and annotation converter for the EuroCity Persons dataset.

Expected directory layout:
    <eurocity_root>/
        ECP/
            day/
                img/
                    val/
                        <city>/
                            <frame>.png
                labels/
                    val/
                        <city>/
                            <frame>.json

EuroCity annotation JSON format (per image):
    {
      "imagewidth": int,
      "imageheight": int,
      "tags": [...],
      "children": [
        {
          "identity": "pedestrian" | "rider" | "motocyclist" | "bicycle-group" | ...,
          "x0": float, "y0": float,   ← top-left corner (pixels)
          "x1": float, "y1": float,   ← bottom-right corner (pixels)
          "tags": [...],              ← e.g. ["depiction", "occluded", ...]
          "children": [...]           ← nested (e.g. rider + vehicle pair)
        },
        ...
      ]
    }

Class mapping
-------------
    pedestrian                        → 0
    rider / motocyclist               → 1  (cyclist)
    scooter / motorbike               → 1  (cyclist, ridden)
    car / vehicle / taxi / van        → 2  (car)
    truck / bus / large-vehicle       → 3  (large vehicle)
    everything else                   → DROPPED

Difficulty (KITTI-style thresholds applied to EuroCity geometry)
----------------------------------------------------------------
    Easy     : bbox_h >= 40px, occlusion <= 0, truncation <= 0.15
    Moderate : bbox_h >= 25px, occlusion <= 1, truncation <= 0.30
    Hard     : bbox_h >= 25px, occlusion <= 2, truncation <= 0.50
    Ignore   : does not meet Hard criteria

Occlusion is inferred from object tags:
    no occlusion tag  → 0
    "occluded"        → 1
    "heavy-occlusion" → 2
    "depiction"       → treated as ignore (dropped from training labels)

Truncation is approximated from how close the bounding box edge is to the
image border, expressed as the fraction of the box that lies outside:
    trunc = max(0, overlap_pixels) / box_dimension

Metadata sidecar format (JSON, one file per image, mirrors labels/ tree)
------------------------------------------------------------------------
    {
      "image": "relative/path/to/frame.png",
      "source": "eurocity",
      "imagewidth": 1920,
      "imageheight": 1024,
      "annotations": [
        {
          "class_id": 0,
          "identity": "pedestrian",
          "bbox_px": [x0, y0, x1, y1],   ← original pixel coords (clamped)
          "bbox_yolo": [cx, cy, w, h],    ← normalised YOLO format
          "bbox_h_px": float,             ← pixel height (for difficulty)
          "occlusion": int,               ← 0 / 1 / 2
          "truncation": float,            ← 0.0 – 1.0 approximation
          "tags": [...],                  ← raw tags from JSON
          "difficulty": "Easy" | "Moderate" | "Hard" | "Ignore"
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# YOLO class IDs
CLASS_PEDESTRIAN    = 0
CLASS_CYCLIST       = 1
CLASS_CAR           = 2
CLASS_LARGE_VEHICLE = 3

# Identities that map to each class
_PEDESTRIAN_IDS = {"pedestrian"}

_CYCLIST_IDS = {
    "rider",
    "motocyclist",
    "scooter",          # ridden scooter → treat as cyclist
    "motorbike",        # ridden motorbike
}

_CAR_IDS = {
    "car",
    "vehicle",          # generic vehicle tag used in some ECP versions
    "taxi",
    "van",
    "pickup",
}

_LARGE_VEHICLE_IDS = {
    "truck",
    "bus",
    "large-vehicle",
    "trailer",
    "articulated-bus",
    "tram",
}

LABEL_MAP: dict[str, int] = (
    {k: CLASS_PEDESTRIAN    for k in _PEDESTRIAN_IDS}
    | {k: CLASS_CYCLIST       for k in _CYCLIST_IDS}
    | {k: CLASS_CAR           for k in _CAR_IDS}
    | {k: CLASS_LARGE_VEHICLE for k in _LARGE_VEHICLE_IDS}
)

# Tags that mark an annotation as a depiction / reflection / irrelevant —
# these are silently dropped (no YOLO label, no metadata entry).
_IGNORE_TAGS = {"depiction", "reflection", "ignore"}

# Tags that encode occlusion level
_OCCLUSION_TAG_MAP: dict[str, int] = {
    "occluded":       1,
    "heavy-occlusion": 2,
    "partial-occlusion": 1,
}

# Difficulty thresholds (mirrors KITTI Easy / Moderate / Hard)
_DIFF_THRESHOLDS = [
    # (label,      min_h, max_occ, max_trunc)
    ("Easy",       40,    0,       0.15),
    ("Moderate",   25,    1,       0.30),
    ("Hard",       25,    2,       0.50),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_occlusion(tags: list[str]) -> int:
    """Return occlusion level (0 / 1 / 2) from an object's tag list."""
    level = 0
    for tag in tags:
        level = max(level, _OCCLUSION_TAG_MAP.get(tag.lower(), 0))
    return level


def _infer_truncation(x0: float, y0: float, x1: float, y1: float,
                      image_w: int, image_h: int) -> float:
    """
    Approximate truncation as the fraction of the box area that lies outside
    the image boundary (0.0 = fully inside, 1.0 = fully outside).

    We use the simpler 1-D max-axis heuristic used by the KITTI devkit:
        trunc = max(clip_w / bw, clip_h / bh)   where clip_* >= 0
    """
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return 1.0

    clip_left   = max(0.0, -x0)
    clip_right  = max(0.0, x1 - image_w)
    clip_top    = max(0.0, -y0)
    clip_bottom = max(0.0, y1 - image_h)

    trunc_w = (clip_left + clip_right) / bw
    trunc_h = (clip_top + clip_bottom) / bh
    return min(1.0, max(trunc_w, trunc_h))


def _compute_difficulty(bbox_h_px: float, occlusion: int,
                        truncation: float) -> str:
    """Return KITTI-style difficulty label for a single annotation."""
    for label, min_h, max_occ, max_trunc in _DIFF_THRESHOLDS:
        if bbox_h_px >= min_h and occlusion <= max_occ and truncation <= max_trunc:
            return label
    return "Ignore"


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class EuroCityAnnotationConverter(AnnotationConverter):
    """
    Converts a single EuroCity .json label file into:
      - a list of BoundingBox objects  (for YOLO training labels)
      - a list of metadata dicts       (for the sidecar JSON)
    """

    def convert(
        self,
        annotation_path: Path,
        image_w: int,
        image_h: int,
    ) -> tuple[list[BoundingBox], list[dict]]:
        """
        Returns
        -------
        boxes    : normalised YOLO BoundingBox list  (dropped = ignored tags / unknown class)
        metadata : per-annotation metadata list       (same objects, richer fields)
        """
        boxes:    list[BoundingBox] = []
        metadata: list[dict]        = []

        if not annotation_path.exists():
            logger.warning("Annotation not found: %s", annotation_path)
            return boxes, metadata

        try:
            with annotation_path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in %s: %s", annotation_path, exc)
            return boxes, metadata

        children = data.get("children", [])
        if not children:
            return boxes, metadata  # valid empty image

        for obj in children:
            identity = obj.get("identity", "").lower()
            tags     = [t.lower() for t in obj.get("tags", [])]

            # Drop depictions / reflections / explicit ignore annotations
            if any(t in _IGNORE_TAGS for t in tags):
                logger.debug("Ignored tag on %s in %s", identity, annotation_path.name)
                continue

            # Drop classes we don't want
            if identity not in LABEL_MAP:
                continue

            class_id = LABEL_MAP[identity]

            try:
                x0 = float(obj["x0"])
                y0 = float(obj["y0"])
                x1 = float(obj["x1"])
                y1 = float(obj["y1"])
            except (KeyError, ValueError, TypeError):
                logger.debug("Malformed bbox in %s: %s", annotation_path, obj)
                continue

            # Compute truncation BEFORE clamping (needs original coords)
            truncation = _infer_truncation(x0, y0, x1, y1, image_w, image_h)
            occlusion  = _infer_occlusion(tags)

            # Clamp to image bounds for YOLO label
            x0c = max(0.0, min(x0, image_w))
            y0c = max(0.0, min(y0, image_h))
            x1c = max(0.0, min(x1, image_w))
            y1c = max(0.0, min(y1, image_h))

            bw = x1c - x0c
            bh = y1c - y0c
            if bw <= 0 or bh <= 0:
                logger.debug("Zero-area box skipped in %s", annotation_path)
                continue

            # Use the UNCLAMPED height for difficulty (matches KITTI devkit intent)
            bbox_h_px  = y1 - y0
            difficulty = _compute_difficulty(bbox_h_px, occlusion, truncation)

            # YOLO normalised box
            cx = (x0c + bw / 2) / image_w
            cy = (y0c + bh / 2) / image_h
            w  = bw / image_w
            h  = bh / image_h

            boxes.append(BoundingBox(
                class_id=class_id,
                cx=cx, cy=cy, w=w, h=h,
            ))

            metadata.append({
                "class_id":   class_id,
                "identity":   identity,
                "bbox_px":    [x0c, y0c, x1c, y1c],
                "bbox_yolo":  [cx, cy, w, h],
                "bbox_h_px":  bbox_h_px,
                "occlusion":  occlusion,
                "truncation": round(truncation, 4),
                "tags":       tags,
                "difficulty": difficulty,
            })

        return boxes, metadata


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class EuroCityReader(DatasetReader):
    """
    Walks one EuroCity split (e.g. day/val) and yields Sample objects,
    while writing JSON metadata sidecars that mirror the labels/ tree.

    Directory layout produced
    -------------------------
        <output_root>/
            images/  <split>/  <city>/  <frame>.png   (symlink or copy)
            labels/  <split>/  <city>/  <frame>.txt   (YOLO format)
            meta/    <split>/  <city>/  <frame>.json  (sidecar metadata)

    Parameters
    ----------
    eurocity_root : Path
        Root directory containing the ECP/ subdirectory structure.
    eurocity_label_root : Path | None
        Override for the label root (defaults to eurocity_root).
    meta_dir : Path | None
        Root directory under which metadata sidecars are written.
        Structure mirrors labels/: meta/<split>/<city>/<frame>.json
        If None, no sidecar files are written (metadata is still attached
        to each Sample for in-memory use).
    time_of_day : str
        "day" or "night". Default "day".
    split : str
        Dataset split: "val", "train", or "test". Default "val".
    image_ext : str
        Image file extension. Default ".png".
    include_no_label_images : bool
        Yield images even when no matching JSON label file exists.
    write_meta : bool
        Write JSON sidecar files to meta_dir. Default True.
    """

    def __init__(
        self,
        eurocity_root: str | Path,
        eurocity_label_root: str | Path | None = None,
        meta_dir: str | Path | None = None,
        time_of_day: str = "day",
        split: str = "val",
        image_ext: str = ".png",
        include_no_label_images: bool = False,
        write_meta: bool = True,
    ) -> None:
        self._root       = Path(eurocity_root)
        label_root       = Path(eurocity_label_root) if eurocity_label_root else self._root
        self._meta_root  = Path(meta_dir) if meta_dir else None
        self._time_of_day     = time_of_day
        self._split           = split
        self._image_ext       = image_ext
        self._include_no_label = include_no_label_images
        self._write_meta      = write_meta and (self._meta_root is not None)
        self._converter       = EuroCityAnnotationConverter()

        self._image_dir = self._root  / "ECP" / time_of_day / "img"    / split
        self._label_dir = label_root  / "ECP" / time_of_day / "labels" / split

        if not self._image_dir.exists():
            raise FileNotFoundError(
                f"EuroCity image directory not found: {self._image_dir}"
            )
        if not self._label_dir.exists():
            raise FileNotFoundError(
                f"EuroCity label directory not found: {self._label_dir}"
            )

        if self._write_meta:
            # meta dir mirrors the label dir structure
            self._meta_dir = (
                self._meta_root / "ECP" / time_of_day / "meta" / split
            )
            self._meta_dir.mkdir(parents=True, exist_ok=True)
            logger.info("EuroCity metadata will be written to: %s", self._meta_dir)
        else:
            self._meta_dir = None

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "eurocity"

    # ------------------------------------------------------------------

    def _write_sidecar(
        self,
        img_path: Path,
        image_w: int,
        image_h: int,
        annotations: list[dict],
    ) -> None:
        """Write a JSON metadata sidecar that mirrors the labels/ tree."""
        rel        = img_path.relative_to(self._image_dir)
        sidecar    = self._meta_dir / rel.with_suffix(".json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "image":       str(rel),
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
        image_paths = sorted(self._image_dir.rglob(f"*{self._image_ext}"))

        if not image_paths:
            logger.warning("No images found under %s", self._image_dir)
            return

        logger.info(
            "EuroCity: found %d images under %s", len(image_paths), self._image_dir
        )

        for img_path in image_paths:
            rel        = img_path.relative_to(self._image_dir)
            label_path = self._label_dir / rel.with_suffix(".json")

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
                # Pass metadata through for any downstream consumer
                # (base.Sample will need a `meta` field — see note below)
                meta=annotations,
            )