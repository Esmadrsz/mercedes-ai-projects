"""
defect_detection — Synthetic image generation, feature extraction,
                   classification and visualization.
"""
from .generator  import build_dataset, CLASS_NAMES
from .features   import extract_all
from .classifier import train
from .visualizer import render

__all__ = ["build_dataset", "CLASS_NAMES", "extract_all", "train", "render"]
