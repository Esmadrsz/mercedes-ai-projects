"""
digital_twin — Vehicle sensor simulation and anomaly detection.
"""
from .simulator import simulate
from .detector  import detect
from .dashboard import render

__all__ = ["simulate", "detect", "render"]
