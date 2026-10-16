"""Robust out-of-limit (OOL) baseline, frozen copy of scripts/limits_psweep.py.

Per channel:
  training    envelope [Q_p - k*IPR, Q_{100-p} + k*IPR] on the nominal samples of the
              warm-up window (first `train_durations[phase-1]` months of the mission,
              ground-truth event intervals masked out)
  evaluation  every query window of the SMED grid (start = r + j*s, length q) over the
              test period is flagged when its max exceeds the upper limit or its min
              falls below the lower limit

Labels are int8 per window, one per query start, identical to the run that produced
reference_checksums.json.
"""
import time
import gc
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp
from numpy.typing import NDArray

from ool_core.helpers import load_json, load_gt_events, ROOT


def fit_envelope(
    train_values: NDArray[np.float32],
    train_times: NDArray[np.datetime64],
    starts: NDArray[np.datetime64],
    ends: NDArray[np.datetime64],
    p: float,
    k: float,
) -> Tuple[float, float, int]:
    """Two-sided Tukey fence on the event-masked training samples. Returns (lo, hi, n_nominal)."""
    inside = (np.searchsorted(starts, train_times, side="right")
              - np.searchsorted(ends, train_times, side="left")) > 0
    nominal = train_values[~inside]
    if len(nominal) == 0:
        raise ValueError("no nominal training samples left after event masking")
    p_l = float(np.percentile(nominal, p))
    p_h = float(np.percentile(nominal, 100 - p))
    ipr = p_h - p_l
    return p_l - k * ipr, p_h + k * ipr, int(len(nominal))


def check_limits(
    values: NDArray[np.float32],
    lo: float,
    hi: float,
    query_length: int,
    reference_length: int,
    slide_length: int,
) -> Tuple[NDArray[np.int8], NDArray[np.int64]]:
    """Window-wise limit check on the SMED query grid. Returns (labels, query_starts)."""
    n = len(values)
    num_windows = (n - query_length - reference_length - 1) // slide_length + 1
    if num_windows <= 0:
        raise ValueError(
            f"channel too short: need > reference_length+query_length+1 "
            f"({reference_length + query_length + 1}) samples, got {n}"
        )
    qs = reference_length + np.arange(num_windows, dtype=np.int64) * slide_length

    windows = np.lib.stride_tricks.sliding_window_view(values, query_length)
    wmax = windows.max(axis=1)[qs]
    wmin = windows.min(axis=1)[qs]
    labels = ((wmax > hi) | (wmin < lo)).astype(np.int8)
    return labels, qs


def eval_channel(
    channel: str,
    config: Dict[str, Any],
) -> Tuple[
    str,                    # channel name
    NDArray[np.int8],       # labels, one per query window
    Tuple[float, float],    # envelope (lo, hi)
    float,                  # training time (envelope estimation)
    float,                  # evaluation time (window extrema + limit check)
]:
    mission = config["mission"]
    metadata = load_json(ROOT / "data" / "metadata.json")
    duration = pd.DateOffset(months=metadata[mission]["train_durations"][int(config["phase"]) - 1])
    test_start = Timestamp(metadata[mission]["test_period"][0])

    # ---- I/O (not timed): one parquet read, split into warm-up window and test period ----
    df = pd.read_parquet(ROOT / "data" / mission / config["signal_mode"] / f"{channel}.parquet")

    train_df = df[df["timestamp"] <= test_start]
    t0 = train_df["timestamp"].min()
    train_df = train_df[train_df["timestamp"] <= t0 + duration].reset_index(drop=True)
    train_values = np.ascontiguousarray(train_df[channel].to_numpy(), dtype=np.float32)
    train_times = train_df["timestamp"].to_numpy("datetime64[ns]")

    test_df = df[df["timestamp"] >= test_start].reset_index(drop=True)
    values = np.ascontiguousarray(test_df[channel].to_numpy(), dtype=np.float32)

    gt_df = load_gt_events(ROOT / "data" / mission / "labels.csv", [channel]).loc[:, ["StartTime", "EndTime"]]
    starts = np.sort(gt_df.StartTime.to_numpy("datetime64[ns]"))
    ends = np.sort(gt_df.EndTime.to_numpy("datetime64[ns]"))

    del df, train_df, test_df
    gc.collect()

    # ---- training: event masking + percentiles + fence ----
    # perf_counter (monotonic): immune to host clock steps, unlike time.time()
    train_start = time.perf_counter()
    lo, hi, _ = fit_envelope(train_values, train_times, starts, ends, config["p"], config["k"])
    train_time = time.perf_counter() - train_start

    # ---- evaluation: window extrema + limit check on the SMED grid ----
    eval_start = time.perf_counter()
    labels, _ = check_limits(values, lo, hi, config["query_length"],
                             config["reference_length"], config["slide_length"])
    eval_time = time.perf_counter() - eval_start

    return channel, labels, (lo, hi), train_time, eval_time
