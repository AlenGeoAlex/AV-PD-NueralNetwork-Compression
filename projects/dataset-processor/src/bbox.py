from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """
    A bounding box in absolute pixel coordinates.

    Convention: top-left (x1, y1) → bottom-right (x2, y2)

    This is the internal representation used throughout the pipeline.
    All dataset-specific formats (KITTI x1y1x2y2, CityPersons xywh)
    are converted into this form before any further processing.
    """

    x1: float  # left edge   (pixels)
    y1: float  # top edge    (pixels)
    x2: float  # right edge  (pixels)
    y2: float  # bottom edge (pixels)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """
        Return True if the box has strictly positive area.
        Boxes with zero width or height are discarded during conversion.
        """
        return self.x2 > self.x1 and self.y2 > self.y1

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """
        Create a BoundingBox from top-left + width/height format.
        Used for CityPersons annotations.

        Args:
            x, y: Top-left corner in pixels.
            w, h: Width and height in pixels.
        """
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    def to_yolo(self, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """
        Convert pixel (x1, y1, x2, y2) → YOLO normalised (cx, cy, w, h).

        All values are normalised by image dimensions and clipped to [0, 1].

        Args:
            img_width:  Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            (cx, cy, w, h) — all in [0, 1].
        """
        cx = (self.x1 + self.x2) / 2.0 / img_width
        cy = (self.y1 + self.y2) / 2.0 / img_height
        w  = (self.x2 - self.x1) / img_width
        h  = (self.y2 - self.y1) / img_height

        # Clip to valid range — guards against boxes that slightly exceed borders
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        return cx, cy, w, h


@dataclass
class Annotation:
    """
    A single labelled object in an image.

    class_id is always 0 in this project (pedestrian-only detection).
    It is kept as a field rather than a constant so the converter
    remains extensible if multi-class support is added later.
    """

    bbox: BoundingBox
    class_id: int = 0  # 0 = pedestrian