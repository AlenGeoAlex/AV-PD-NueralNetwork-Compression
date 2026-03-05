"""
Abstract interfaces for dataset processors.
All dataset-specific converters must implement these interfaces.
The orchestrator depends only on these abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    A single bounding box in YOLO normalised format.

    class_id : int   — 0=pedestrian, 1=cyclist, 2=car, 3=large vehicle
    cx, cy   : float — centre x/y relative to image width/height  [0..1]
    w, h     : float — box width/height relative to image size     [0..1]
    """
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Sample:
    """
    One image + its annotations, source-agnostic.

    image_path : absolute path to the image file on disk
    boxes      : list of BoundingBox (may be empty for negative samples)
    source     : dataset name tag used for stratified splitting
                 ("kitti" / "eurocity")
    split_hint : original split name from the source dataset
                 ("train" / "val" / "test") — used only as metadata;
                 the orchestrator decides the final split.
    meta       : per-annotation metadata list produced by the converter.
                 Each entry is a dict with keys: class_id, identity,
                 bbox_px, bbox_yolo, bbox_h_px, occlusion, truncation,
                 difficulty, tags (EuroCity) / iou_threshold / is_dontcare
                 (KITTI).  Empty list when metadata was not requested.
    """
    image_path: Path
    boxes:      list[BoundingBox]
    source:     str
    split_hint: str        = "train"
    meta:       list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class DatasetReader(ABC):
    """
    Knows how to FIND images and labels for one dataset.
    Yields Sample objects — does NOT write anything to disk
    (except optional metadata sidecars handled internally).
    """

    @abstractmethod
    def read(self) -> Iterator[Sample]:
        """
        Yield every Sample in the dataset.
        Implementations must be lazy (generator) to handle large datasets.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier e.g. 'kitti', 'eurocity'."""
        ...


class AnnotationConverter(ABC):
    """
    Knows how to READ one annotation format and return both YOLO boxes
    and rich per-annotation metadata.

    One converter per source format (KITTI txt, EuroCity JSON, …).
    """

    # Mapping from raw source label string -> YOLO class id.
    # Subclasses must define this (or manage mapping internally).
    LABEL_MAP: dict[str, int] = {}

    @abstractmethod
    def convert(
        self,
        annotation_path: Path,
        image_w: int,
        image_h: int,
    ) -> tuple[list[BoundingBox], list[dict]]:
        """
        Parse annotation_path and return:
            boxes    : normalised BoundingBox list  (DontCare / ignored excluded)
            metadata : list of per-annotation dicts (all annotations, richer fields)

        Boxes whose labels are not in LABEL_MAP are silently dropped.
        """
        ...