"""DAMP (X-Lag-Amnesic) detector.

DAMP shares SMED's bounded-lookback left-matrix-profile structure; the only difference is
the distance metric. We therefore reuse the exact SMED pipeline (windowing + IPR threshold)
and swap the per-window distance:

    euclidean="standard"  ->  raw Euclidean (this is SMED)
    euclidean="znorm"     ->  z-normalized Euclidean (this is DAMP)

This gives a controlled comparison (standard vs z-normalized distance), holding
everything else fixed. `znorm` is the default (true DAMP); `standard` exists to verify that the
shared pipeline reproduces SMED.
"""
from __future__ import annotations

from tsadbench.detectors.base import DetectorResult, register
from tsadbench.detectors.smed.core import eval_channel
from tsadbench.detectors.smed.distance import best_match
from tsadbench.detectors.damp.distance import best_match_znorm

_DIST = {"standard": best_match, "znorm": best_match_znorm}


@register("damp")
class DAMPDetector:
    def __init__(self, config: dict | None = None, euclidean: str = "znorm"):
        if euclidean not in _DIST:
            raise ValueError(f"euclidean must be one of {sorted(_DIST)}, got {euclidean!r}")
        self.config = dict(config or {})
        self.euclidean = euclidean

    def fit(self, split) -> None:
        return None

    def score(self, split) -> DetectorResult:
        cfg = {**split.run_config_dict, **self.config}
        dist_fn = _DIST[self.euclidean]
        labels, scores, thresholds, timestamps = {}, {}, {}, {}
        for ch in split.channels:
            _, ts, lab, dist, thr, _, _ = eval_channel(ch, cfg, dist_fn=dist_fn)
            labels[ch] = lab
            scores[ch] = dist
            thresholds[ch] = float(thr[0]) if len(thr) else float("nan")
            timestamps[ch] = ts
        return DetectorResult(
            labels=labels, scores=scores, thresholds=thresholds, timestamps=timestamps,
            meta={"name": "damp", "euclidean": self.euclidean,
                  "algo_name": f"damp_{self.euclidean}",
                  "mission": cfg["mission"], "phase": cfg["phase"],
                  "channels_mode": cfg["channels_mode"], "signal_mode": cfg["signal_mode"]},
        )
