"""
base_converter.py
-----------------
Abstract base class that owns the main conversion loop,
YOLO label writing, and image copying.

Subclasses only need to implement:
    - iter_samples()       → yields (image_path, label_path)
    - parse_annotations()  → returns list[Annotation] for one sample
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from bbox import Annotation

logger = logging.getLogger(__name__)


class DatasetConverter(ABC):
    """
    Abstract base for all dataset converters.

    Output structure produced:
        <output_dir>/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

    Args:
        output_dir: Root output directory. Created if it does not exist.
        split:      "train" or "val" — controls which subfolder is written to.
    """

    CLASS_ID = 0  # pedestrian

    def __init__(self, output_dir: str | Path, split: str = "train"):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.output_dir = Path(output_dir)
        self.split = split

        self._images_out = self.output_dir / "images" / split
        self._labels_out = self.output_dir / "labels" / split

        self._images_out.mkdir(parents=True, exist_ok=True)
        self._labels_out.mkdir(parents=True, exist_ok=True)

        self.stats: dict[str, int] = {
            "total":            0,  # total images processed
            "with_pedestrians": 0,  # images that had ≥1 pedestrian box
            "empty":            0,  # images with no pedestrians (label file still written)
            "skipped":          0,  # images that raised an exception
        }

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these two methods
    # ------------------------------------------------------------------

    @abstractmethod
    def iter_samples(self):
        """
        Yield (image_path, label_path) pairs over the dataset split.

        Both paths must exist at the time of yielding.
        The generator is responsible for matching images to their annotations.
        """

    @abstractmethod
    def parse_annotations(
        self,
        label_path: Path,
        image_path: Path,
    ) -> list[Annotation]:
        """
        Parse one dataset-specific annotation file.

        Args:
            label_path: Path to the annotation file for this image.
            image_path: Path to the image (available for context/validation).

        Returns:
            List of Annotation objects. Return an empty list if no
            pedestrians are present — do NOT raise an exception.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_image_size(image_path: Path) -> tuple[int, int]:
        """
        Return (width, height) of an image without decoding all pixel data.
        Uses PIL's lazy-loading header read.
        """
        with Image.open(image_path) as img:
            return img.size  # (width, height)

    def _write_yolo_label(
        self,
        annotations: list[Annotation],
        img_width: int,
        img_height: int,
        out_path: Path,
    ) -> int:
        """
        Write a YOLO-format label file.

        Each line:
            class_id cx cy w h
        All values are space-separated; cx/cy/w/h are written to 6 decimal places.

        An empty file is written if there are no valid annotations —
        YOLO training expects a label file for every image in the dataset.

        Returns:
            Number of boxes written to the file.
        """
        lines: list[str] = []

        for ann in annotations:
            if not ann.bbox.is_valid():
                logger.debug("Skipping invalid bbox: %s", ann.bbox)
                continue
            cx, cy, w, h = ann.bbox.to_yolo(img_width, img_height)
            lines.append(f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        out_path.write_text("\n".join(lines))
        return len(lines)

    # ------------------------------------------------------------------
    # Main conversion loop
    # ------------------------------------------------------------------

    def convert(self) -> dict[str, int]:
        """
        Run the full conversion pipeline for this split.

        For each (image, label) pair:
            1. Parse annotations via parse_annotations()
            2. Read image dimensions
            3. Copy image to output directory
            4. Write YOLO label file to output directory

        Returns:
            Stats dict: total, with_pedestrians, empty, skipped.
        """
        logger.info(
            "Starting %s | split=%s → %s",
            self.__class__.__name__,
            self.split,
            self.output_dir,
        )

        for image_path, label_path in self.iter_samples():
            self.stats["total"] += 1

            try:
                annotations = self.parse_annotations(label_path, image_path)
                img_w, img_h = self.get_image_size(image_path)

                dest_image = self._images_out / image_path.name
                dest_label = self._labels_out / (image_path.stem + ".txt")

                shutil.copy2(image_path, dest_image)
                n_boxes = self._write_yolo_label(annotations, img_w, img_h, dest_label)

                if n_boxes > 0:
                    self.stats["with_pedestrians"] += 1
                else:
                    self.stats["empty"] += 1
                    logger.debug("No pedestrians in %s", image_path.name)

            except Exception as exc:  # noqa: BLE001
                self.stats["skipped"] += 1
                logger.warning("Skipped %s — %s: %s", image_path.name, type(exc).__name__, exc)

        self._log_summary()
        return self.stats

    def _log_summary(self) -> None:
        s = self.stats
        logger.info(
            "Finished | total=%d | with_pedestrians=%d | empty=%d | skipped=%d",
            s["total"], s["with_pedestrians"], s["empty"], s["skipped"],
        )