"""
dataset_converter
=================
Preprocessing pipeline that converts KITTI and CityPersons pedestrian
detection datasets into YOLO format.

Public API
----------
    KITTIConverter          — converts KITTI Object Detection labels
    CityPersonsConverter    — converts CityPersons (Cityscapes) labels
    write_dataset_yaml      — writes dataset.yaml for Ultralytics YOLO
    print_split_summary     — prints a train/val stats table

Quick start
-----------
    from dataset_converter import (
        KITTIConverter,
        CityPersonsConverter,
        write_dataset_yaml,
        print_split_summary,
    )

    # ── KITTI ────────────────────────────────────────────────────────
    kitti_stats = {}
    for split in ("train", "val"):
        conv = KITTIConverter(
            images_dir="kitti/training/image_2",
            labels_dir="kitti/training/label_2",
            output_dir="dataset",
            split=split,
        )
        kitti_stats[split] = conv.convert()

    # ── CityPersons ──────────────────────────────────────────────────
    cp_stats = {}
    for split in ("train", "val"):
        conv = CityPersonsConverter(
            images_dir=f"cityscapes/leftImg8bit/{split}",
            annotations_dir=f"cityscapes/gtBboxCityPersons/{split}",
            output_dir="dataset",
            split=split,
        )
        cp_stats[split] = conv.convert()

    # ── Finalise ─────────────────────────────────────────────────────
    write_dataset_yaml("dataset")
    print_split_summary(kitti_stats["train"], kitti_stats["val"], "KITTI")
    print_split_summary(cp_stats["train"],    cp_stats["val"],    "CityPersons")

Split strategy
--------------
    KITTI
        No official val split exists — test labels are withheld.
        We use the community convention: first 6,000 images → train,
        remaining 1,481 → val (sorted by filename / sequence order).
        This avoids temporal leakage from sequential video frames.

    CityPersons
        Uses the OFFICIAL Cityscapes train/val city split as-is.
        Do not randomly re-split — all frames within a city share
        the same scene, so random splitting causes scene leakage.
"""

from bbox import Annotation, BoundingBox
from base_converter import DatasetConverter
from kitti_converter import KITTIConverter
from citypersons_converter import CityPersonsConverter
from utils import write_dataset_yaml, print_split_summary

__all__ = [
    "BoundingBox",
    "Annotation",
    "DatasetConverter",
    "KITTIConverter",
    "CityPersonsConverter",
    "write_dataset_yaml",
    "print_split_summary",
]