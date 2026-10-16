"""Detector interface + registry.

Every method (SMED, DAMP, ingested baselines) implements the same small protocol so
the runner, evaluation, postprocessing, and leaderboard treat them uniformly.

    @register("mymethod")
    class MyDetector:
        def fit(self, split): ...          # train / calibrate on the split's train window
        def score(self, split) -> DetectorResult: ...   # causal scoring over the test window

    det = get_detector("mymethod", config={...})
    det.fit(split); result = det.score(split)

`DetectorResult.labels` (channel -> int8 array) is the bit-exact invariant the golden
gate checks; `scores`/`thresholds` are secondary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass
class DetectorResult:
    labels: dict[str, np.ndarray]                       # channel -> int8 labels (the invariant)
    scores: dict[str, np.ndarray] | None = None         # channel -> float32 anomaly scores
    thresholds: dict[str, float] | None = None          # channel -> decision threshold
    timestamps: dict[str, list] | None = None           # channel -> per-window timestamps (for persistence)
    meta: dict = field(default_factory=dict)            # method name, hyperparams, timings, env


@runtime_checkable
class Detector(Protocol):
    name: str

    def fit(self, split) -> None: ...

    def score(self, split) -> "DetectorResult": ...


REGISTRY: dict[str, Callable[..., Detector]] = {}


def register(name: str):
    """Class decorator: register a detector factory under `name`."""
    def deco(cls):
        if name in REGISTRY:
            raise ValueError(f"detector {name!r} already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def get_detector(name: str, *args, **kwargs) -> Detector:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](*args, **kwargs)
