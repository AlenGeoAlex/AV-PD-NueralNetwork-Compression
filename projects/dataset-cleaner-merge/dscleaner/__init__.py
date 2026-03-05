"""
dataset_processor — Convert, merge, and split KITTI and EuroCity Persons
into a unified YOLO-format dataset.

Public API
----------
from dataset_processor import DatasetOrchestrator, KITTIReader, EuroCityReader
from dataset_processor.base import Sample, BoundingBox
"""

from .base import AnnotationConverter, BoundingBox, DatasetReader, Sample
from .kitti import KITTIAnnotationConverter, KITTIReader
from .eurocity import EuroCityAnnotationConverter, EuroCityReader
from .orchestrator import DatasetOrchestrator, StratifiedSplitter, DatasetWriter

__all__ = [
    "AnnotationConverter",
    "BoundingBox",
    "DatasetReader",
    "Sample",
    "KITTIAnnotationConverter",
    "KITTIReader",
    "EuroCityAnnotationConverter",
    "EuroCityReader",
    "DatasetOrchestrator",
    "StratifiedSplitter",
    "DatasetWriter",
]