"""
Merges multiple datasets, stratifies, splits, and writes
output in standard YOLO directory structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from .base import DatasetReader, Sample
from .kitti import KITTIReader
from .eurocity import EuroCityReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------

CLASS_NAMES = ["pedestrian", "cyclist", "car", "large_vehicle"]


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class StratifiedSplitter:
    """
    Splits a list of Sample objects into train/val/test while ensuring that
    each source dataset contributes proportionally to every split.

    Parameters
    ----------
    train_ratio : float   default 0.70
    val_ratio   : float   default 0.15
    test_ratio  : float   inferred as 1 - train - val
    seed        : int     for reproducibility
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        seed:        int   = 42,
    ) -> None:
        if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
            raise ValueError("train_ratio and val_ratio must be in (0, 1)")
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")

        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed

    def split(
        self, samples: list[Sample]
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """Returns (train_samples, val_samples, test_samples)."""
        rng = random.Random(self.seed)

        by_source: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_source[s.source].append(s)

        train, val, test = [], [], []

        for source, group in by_source.items():
            rng.shuffle(group)
            n       = len(group)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)

            train.extend(group[:n_train])
            val.extend(group[n_train : n_train + n_val])
            test.extend(group[n_train + n_val :])

            logger.info(
                "  %-12s  total=%d  train=%d  val=%d  test=%d",
                source, n,
                len(group[:n_train]),
                len(group[n_train : n_train + n_val]),
                len(group[n_train + n_val :]),
            )

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        return train, val, test


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class DatasetWriter:
    """
    Writes samples to YOLO output directory structure, creates
    dataset.yaml, and optional metadata sidecars.

    Parameters
    ----------
    output_dir  : Path
        Root of the output dataset.
    meta_dir    : Path | None
        Root under which metadata sidecars are written.
        Mirrors the labels/ tree: meta/<split>/<stem>.json
        If None, metadata sidecars are not written.
    copy_images : bool
        True  → physically copy images (safe, uses disk space).
        False → create symbolic links (fast, default).
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        output_dir:  str | Path,
        meta_dir:    str | Path | None = None,
        copy_images: bool = False,
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.meta_dir    = Path(meta_dir) if meta_dir else None
        self.copy_images = copy_images

    def write(
        self,
        train: list[Sample],
        val:   list[Sample],
        test:  list[Sample],
    ) -> None:
        split_map = {"train": train, "val": val, "test": test}

        # Create directory structure
        for split in self.SPLITS:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            if self.meta_dir:
                (self.meta_dir / "meta" / split).mkdir(parents=True, exist_ok=True)

        manifest_rows = []

        for split, samples in split_map.items():
            logger.info("Writing %s split (%d samples)…", split, len(samples))
            for sample in samples:
                img_stem = self._unique_stem(sample)
                img_ext  = sample.image_path.suffix

                dst_img   = self.output_dir / "images" / split / (img_stem + img_ext)
                dst_label = self.output_dir / "labels" / split / (img_stem + ".txt")

                self._write_image(sample.image_path, dst_img)
                self._write_label(sample, dst_label)

                dst_meta = None
                if self.meta_dir and sample.meta:
                    dst_meta = self.meta_dir / "meta" / split / (img_stem + ".json")
                    self._write_meta(sample, dst_meta)

                manifest_rows.append({
                    "split":      split,
                    "source":     sample.source,
                    "original":   str(sample.image_path),
                    "image_out":  str(dst_img),
                    "label_out":  str(dst_label),
                    "meta_out":   str(dst_meta) if dst_meta else "",
                    "n_boxes":    len(sample.boxes),
                })

        self._write_yaml()
        self._write_manifest(manifest_rows)
        logger.info("Dataset written to %s", self.output_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unique_stem(self, sample: Sample) -> str:
        """Collision-free stem: <source>_<original_stem>"""
        return f"{sample.source}_{sample.image_path.stem}"

    def _write_image(self, src: Path, dst: Path) -> None:
        if dst.exists():
            return
        if self.copy_images:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())

    def _write_label(self, sample: Sample, dst: Path) -> None:
        with dst.open("w") as f:
            for box in sample.boxes:
                f.write(box.to_yolo_line() + "\n")
            # Empty file is valid for YOLO negative samples

    def _write_meta(self, sample: Sample, dst: Path) -> None:
        """Write per-image metadata sidecar JSON."""
        payload = {
            "image":    str(sample.image_path),
            "source":   sample.source,
            "n_boxes":  len(sample.boxes),
            "annotations": sample.meta,
        }
        try:
            with dst.open("w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            logger.warning("Could not write metadata sidecar %s: %s", dst, exc)

    def _write_yaml(self) -> None:
        yaml_path = self.output_dir / "dataset.yaml"
        config = {
            "path":  str(self.output_dir.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc":    len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
        with yaml_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info("Dataset YAML written: %s", yaml_path)

    def _write_manifest(self, rows: list[dict]) -> None:
        manifest_path = self.output_dir / "split_manifest.csv"
        fieldnames = [
            "split", "source", "original",
            "image_out", "label_out", "meta_out", "n_boxes",
        ]
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Manifest written: %s", manifest_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DatasetOrchestrator:
    """
    Top-level coordinator. Accepts one or more DatasetReader instances,
    collects all samples, stratifies, splits, and writes the output.

    Parameters
    ----------
    readers      : list[DatasetReader]  — at least one required
    output_dir   : Path
    meta_dir     : Path | None  — root for metadata sidecars; if None,
                                  sidecars are not written
    train_ratio  : float   default 0.70
    val_ratio    : float   default 0.15
    seed         : int     default 42
    copy_images  : bool    default False (symlinks)
    """

    def __init__(
        self,
        readers:     list[DatasetReader],
        output_dir:  str | Path,
        meta_dir:    str | Path | None = None,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        seed:        int   = 42,
        copy_images: bool  = False,
    ) -> None:
        if not readers:
            raise ValueError("At least one DatasetReader must be provided.")
        self.readers  = readers
        self.splitter = StratifiedSplitter(train_ratio, val_ratio, seed)
        self.writer   = DatasetWriter(output_dir, meta_dir, copy_images)

    def run(self) -> None:
        # 1. Collect all samples
        all_samples: list[Sample] = []
        for reader in self.readers:
            logger.info("Reading dataset: %s", reader.name)
            samples = list(reader.read())
            logger.info("  → %d samples loaded", len(samples))
            all_samples.extend(samples)

        if not all_samples:
            raise RuntimeError("No samples collected — check your input paths.")

        logger.info("Total samples: %d", len(all_samples))

        # 2. Stratified split
        logger.info("Splitting (stratified by source)…")
        train, val, test = self.splitter.split(all_samples)
        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train), len(val), len(test),
        )

        # 3. Write output
        self.writer.write(train, val, test)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge KITTI and/or EuroCity Persons into a unified YOLO dataset "
            "with optional per-image metadata sidecars."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- KITTI ---
    kitti = p.add_argument_group("KITTI")
    kitti.add_argument("--kitti-root",    type=Path, default=None,
                       help="Root of the KITTI dataset. Omit to skip.")
    kitti.add_argument("--kitti-split",   default="training",
                       help="KITTI internal split name.")
    kitti.add_argument("--kitti-image-ext", default=".png")
    kitti.add_argument("--kitti-include-no-label", action="store_true")

    # --- EuroCity ---
    euro = p.add_argument_group("EuroCity Persons")
    euro.add_argument("--eurocity-root",       type=Path, default=None,
                      help="Root of EuroCity (contains ECP/). Omit to skip.")
    euro.add_argument("--eurocity-label-root", type=Path, default=None,
                      help="Override label root (defaults to --eurocity-root).")
    euro.add_argument("--eurocity-split",      default="val",
                      choices=["train", "val", "test"])
    euro.add_argument("--eurocity-time",       default="day",
                      choices=["day", "night"])
    euro.add_argument("--eurocity-image-ext",  default=".png")
    euro.add_argument("--eurocity-include-no-label", action="store_true")

    # --- Output ---
    out = p.add_argument_group("Output")
    out.add_argument("--output-dir",  type=Path, required=True,
                     help="Root directory for the merged dataset.")
    out.add_argument("--meta-dir",    type=Path, default=None,
                     help=(
                         "Root directory for JSON metadata sidecars "
                         "(mirrors labels/ structure under meta/). "
                         "Defaults to --output-dir when flag is present. "
                         "Omit to skip metadata writing."
                     ))
    out.add_argument("--train-ratio", type=float, default=0.70)
    out.add_argument("--val-ratio",   type=float, default=0.15)
    out.add_argument("--seed",        type=int,   default=42)
    out.add_argument("--copy-images", action="store_true",
                     help="Copy images instead of symlinking.")
    out.add_argument("--log-level",   default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve meta_dir: if flag given without value it stays as provided path,
    # if not given at all it's None (no sidecars written).
    meta_dir = args.meta_dir  # None → no sidecars

    readers: list[DatasetReader] = []

    if args.kitti_root:
        readers.append(KITTIReader(
            kitti_root=args.kitti_root,
            meta_dir=meta_dir,
            split=args.kitti_split,
            image_ext=args.kitti_image_ext,
            include_no_label_images=args.kitti_include_no_label,
            write_meta=(meta_dir is not None),
        ))
    else:
        logger.info("--kitti-root not provided, skipping KITTI.")

    if args.eurocity_root:
        readers.append(EuroCityReader(
            eurocity_root=args.eurocity_root,
            eurocity_label_root=args.eurocity_label_root,
            meta_dir=meta_dir,
            time_of_day=args.eurocity_time,
            split=args.eurocity_split,
            image_ext=args.eurocity_image_ext,
            include_no_label_images=args.eurocity_include_no_label,
            write_meta=(meta_dir is not None),
        ))
    else:
        logger.info("--eurocity-root not provided, skipping EuroCity.")

    if not readers:
        parser.error(
            "At least one of --kitti-root or --eurocity-root must be provided."
        )

    DatasetOrchestrator(
        readers=readers,
        output_dir=args.output_dir,
        meta_dir=meta_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_images=args.copy_images,
    ).run()


if __name__ == "__main__":
    main()