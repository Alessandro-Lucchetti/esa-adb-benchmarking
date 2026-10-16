"""SMED as a Detector (adapter over the frozen numeric core).

`score` calls the bit-exact-frozen `eval_channel` per channel and assembles a
DetectorResult, so SMED plugs into the same runner/eval/leaderboard path as the
baselines without touching the numeric core. Channels run sequentially here; the
multiprocessing path lives in the SMED-specific `run_smed` driver.
"""
from __future__ import annotations

from tsadbench.detectors.base import DetectorResult, register
from tsadbench.detectors.smed.core import eval_channel


@register("smed")
class SMEDDetector:
    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})

    def fit(self, split) -> None:
        # SMED has no separate training step: warm-up + IPR threshold are computed
        # inside eval_channel/warm_up at score time.
        return None

    def score(self, split) -> DetectorResult:
        cfg = {**split.run_config_dict, **self.config}
        labels, scores, thresholds, timestamps = {}, {}, {}, {}
        for ch in split.channels:
            _, ts, lab, dist, thr, _, _ = eval_channel(ch, cfg)
            labels[ch] = lab
            scores[ch] = dist
            thresholds[ch] = float(thr[0]) if len(thr) else float("nan")
            timestamps[ch] = ts
        return DetectorResult(
            labels=labels, scores=scores, thresholds=thresholds, timestamps=timestamps,
            meta={"name": "smed", "mission": cfg["mission"], "phase": cfg["phase"],
                  "channels_mode": cfg["channels_mode"], "signal_mode": cfg["signal_mode"]},
        )
