"""
models/__init__.py
Package initializer for model wrappers.
"""

from models.detection import PotholeDetector
from models.segmentation import PotholeSegmentor
from models.depth import DepthEstimator

__all__ = ["PotholeDetector", "PotholeSegmentor", "DepthEstimator"]
