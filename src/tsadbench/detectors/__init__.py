"""Detector registry. Importing this package registers the built-in detectors."""
from tsadbench.detectors.base import Detector, DetectorResult, REGISTRY, get_detector, register
from tsadbench.detectors.smed import detector as _smed  # noqa: F401  (registers "smed")
from tsadbench.detectors.damp import detector as _damp  # noqa: F401  (registers "damp")

__all__ = ["Detector", "DetectorResult", "REGISTRY", "get_detector", "register"]
