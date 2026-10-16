"""Run any Detector on a Split, persist predictions, and score them.

Detector-agnostic counterpart to the SMED-specific run_smed: it takes a DetectorResult
(labels/scores/thresholds/timestamps per channel), writes the standard on-disk layout
(labels/, scores/, config.json), and computes metrics via the shared eval pipeline so every
method (SMED, DAMP, ingested baselines) is scored identically and lands in the leaderboard.

Outputs go to  runs/<algo_name>/<run_name>/  (gitignored). SMED has its own optimized,
parallel run_smed; this path runs the detector's score() (typically sequential).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsadbench.eval.metrics.utils import label_changes
from tsadbench.eval.orchestrator import metrics_light
from tsadbench.io.misc import save_json, project_root
from tsadbench.runner.naming import build_run_name


def run_detector(detector, split, *, ew_only: bool = True, runs_root: Path | None = None) -> dict:
    result = detector.score(split)
    channels = list(result.labels)

    # Effective config = split defaults overridden by the detector's own config (p, k, ...),
    # so the run name/folder reflects what was actually run (distinct per swept hyperparameter).
    overrides = dict(getattr(detector, "config", {}) or {})
    cfg = {**split.run_config_dict, **overrides, **result.meta}
    cfg["algo_name"] = result.meta.get("algo_name", detector.name)
    cfg["channels"] = channels

    run_name = build_run_name(cfg, channels)
    runs_root = runs_root or (project_root() / "runs")
    pred_path = runs_root / cfg["algo_name"] / run_name
    (pred_path / "labels").mkdir(parents=True, exist_ok=True)
    (pred_path / "scores").mkdir(parents=True, exist_ok=True)

    predictions_dict = {}
    for ch in channels:
        ts = result.timestamps[ch]
        lab = np.asarray(result.labels[ch])
        sc = np.asarray(result.scores[ch]) if result.scores else np.zeros(len(lab), np.float32)
        thr = result.thresholds[ch] if result.thresholds else float("nan")
        pd.DataFrame({"timestamp": ts, "label": lab}).to_parquet(pred_path / "labels" / f"{ch}.parquet")
        pd.DataFrame({"timestamp": ts, "score": sc,
                      "threshold": np.full(len(lab), thr, np.float32)}).to_parquet(pred_path / "scores" / f"{ch}.parquet")
        predictions_dict[ch] = label_changes(ts, lab)

    # metrics_light reuses an existing event.parquet if present; drop it so metrics
    # are always recomputed from the freshly written predictions.
    stale = pred_path / "labels" / "event.parquet"
    if stale.exists():
        stale.unlink()

    save_json(cfg, pred_path / "config")
    cfg = metrics_light(pred_path=pred_path, config=cfg, predictions_dict=predictions_dict,
                        ew_only=ew_only, t0=None)
    save_json(cfg, pred_path / "config")
    return cfg
