"""Robust out-of-limit (OOL) baseline percentile sweep.

The operational status-quo detector: a per-channel robust envelope (two-sided Tukey
fence on percentiles of nominal training values), with no similarity search. Windows
are SMED's exact query grid (q100/r750/s5) and a window is flagged when any of its
samples leaves the envelope, so the labels drop into the shared event-wise pipeline
next to SMED/DAMP/Telemanom. Training samples inside ground-truth events are masked
(the sample-level analog of SMED's warm-up window masking).

The envelope percentiles and the per-window max/min do the expensive work once per
channel; each p then re-derives the fence and relabels cheaply (same pattern as
damp_psweep.py). p=0.1 mirrors SMED's own threshold setting; larger p values give
the baseline its best-case operating point.

Usage:
    python scripts/limits_psweep.py <channels_mode> [--phase N] [--full-metrics] [--prune] [p1 p2 ...]
        channels_mode   light | full
        --phase         ESA-ADB training phase 1-5 (3/10/21/42/84 months of nominal
                        data for the envelope); default 5, the paper's main setting
        --full-metrics  complete ESA-ADB suite (event-wise, subsystem/channel-aware,
                        alarming precision, ADTQC, affiliation) instead of event-wise only;
                        same code path as the SMED table runs
        --prune         delete per-channel labels/scores after scoring (keeps config.json
                        and event.parquet); for sensitivity sweeps over many p values
"""
import argparse
import shutil
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsadbench.data.splits import Split
from tsadbench.runner.config import RunConfig
from tsadbench.runner.naming import build_run_name
from tsadbench.eval.orchestrator import metrics_light
from tsadbench.io.misc import save_json, project_root, load_metadata, load_gt_events

ALGO = "oolimit"
DEFAULT_PS = [0.1, 1.0, 3.0, 5.0, 10.0, 25.0]


def nominal_train_values(split: Split, channel: str, metadata: dict) -> np.ndarray:
    """Training-window samples with ground-truth event intervals masked out.

    Window selection mirrors smed.core.warm_up (forward mode from the start of the
    mission, phase-dependent duration); masking is per sample rather than per window
    because the envelope is estimated on values, not on windowed distances.
    """
    cfg = split.config
    duration = pd.DateOffset(months=metadata[cfg.mission]["train_durations"][int(cfg.phase) - 1])
    train_end = pd.Timestamp(metadata[cfg.mission]["test_period"][0])

    df = split.load_channel(channel)
    df = df[df["timestamp"] <= train_end]
    t0 = df["timestamp"].min()
    df = df[df["timestamp"] <= t0 + duration].reset_index(drop=True)

    gt_df = load_gt_events(split.data_dir / "labels.csv", [channel]).loc[:, ["StartTime", "EndTime"]]
    starts = np.sort(gt_df.StartTime.to_numpy("datetime64[ns]"))
    ends = np.sort(gt_df.EndTime.to_numpy("datetime64[ns]"))
    t = df["timestamp"].to_numpy("datetime64[ns]")
    inside = (np.searchsorted(starts, t, side="right") - np.searchsorted(ends, t, side="left")) > 0

    values = np.ascontiguousarray(df[channel].to_numpy(), dtype=np.float32)[~inside]
    if len(values) == 0:
        raise ValueError(f"{channel}: no nominal training samples left after event masking")
    return values


def sweep(channels_mode: str, ps: list[float], phase: str = "5",
          full_metrics: bool = False, prune: bool = False) -> None:
    t0 = time.time()
    cfg = RunConfig(algo_name=ALGO, mission="ESA-Mission1", channels_mode=channels_mode,
                    signal_mode="raw", phase=str(phase), warm_up_mode="forward",
                    query_length=100, reference_length=750, slide_length=5, k=1.5)
    split = Split(cfg)
    base = cfg.to_run_dict()
    channels = split.channels
    metadata = load_metadata()
    test_start = pd.Timestamp(metadata["ESA-Mission1"]["test_period"][0])
    q, r, s, k = base["query_length"], base["reference_length"], base["slide_length"], base["k"]

    runs_root = project_root() / "runs" / (f"{ALGO}_full" if channels_mode == "full" else ALGO)

    # ---- expensive pass: train percentiles + per-window test extrema, once per channel ----
    perc = {}          # channel -> {p: (p_l, p_h)} on nominal training values
    test = {}          # channel -> (window-start timestamps, window max, window min)
    for i, ch in enumerate(channels):
        train_vals = nominal_train_values(split, ch, metadata)
        perc[ch] = {p: (float(np.percentile(train_vals, p)),
                        float(np.percentile(train_vals, 100 - p))) for p in ps}

        test_df = split.load_channel(ch)
        test_df = test_df[test_df["timestamp"] >= test_start].reset_index(drop=True)
        values = np.ascontiguousarray(test_df[ch].to_numpy(), dtype=np.float32)
        n = len(values)
        num_windows = (n - q - r - 1) // s + 1
        if num_windows <= 0:
            raise ValueError(f"{ch}: test period too short for the q{q}/r{r} window grid")
        qs = r + np.arange(num_windows, dtype=np.int64) * s

        windows = np.lib.stride_tricks.sliding_window_view(values, q)
        wmax = windows.max(axis=1)[qs]
        wmin = windows.min(axis=1)[qs]
        ts = test_df["timestamp"].to_numpy("datetime64[ns]")[qs]
        test[ch] = (ts, wmax, wmin)
        print(f"[{time.time()-t0:6.1f}s] extrema done {i+1}/{len(channels)} {ch} "
              f"(train={len(train_vals)}, test windows={num_windows})")
    detector_runtime = time.time() - t0   # shared, p-independent part (includes parquet loading)

    # ---- cheap pass: envelope + label + score for each p ----
    for p in ps:
        t_p = time.time()
        run_cfg = {**base, "p": p, "channels": channels}
        run_name = build_run_name(run_cfg, channels)
        pred_path = runs_root / run_name
        (pred_path / "labels").mkdir(parents=True, exist_ok=True)
        (pred_path / "scores").mkdir(parents=True, exist_ok=True)

        envelope = {}
        for ch in channels:
            p_l, p_h = perc[ch][p]
            ipr = p_h - p_l
            lo, hi = p_l - k * ipr, p_h + k * ipr
            envelope[ch] = [lo, hi]
            ts, wmax, wmin = test[ch]
            lab = ((wmax > hi) | (wmin < lo)).astype(np.int8)
            exceed = np.maximum(wmax - hi, lo - wmin).astype(np.float32)  # >0 iff out of limits
            pd.DataFrame({"timestamp": ts, "label": lab}).to_parquet(
                pred_path / "labels" / f"{ch}.parquet")
            pd.DataFrame({"timestamp": ts, "score": exceed,
                          "threshold": np.zeros(len(exceed), np.float32)}).to_parquet(
                pred_path / "scores" / f"{ch}.parquet")

        stale = pred_path / "labels" / "event.parquet"
        if stale.exists():
            stale.unlink()

        run_cfg["envelope"] = envelope
        run_cfg["detector_runtime"] = detector_runtime
        run_cfg["threshold_runtime"] = time.time() - t_p
        save_json(run_cfg, pred_path / "config")
        modes = ["all_events", "anomalies_alone"]
        if full_metrics:
            # predictions_dict=None: per-channel labels reloaded from disk, as for the SMED runs
            run_cfg = metrics_light(pred_path=pred_path, config=run_cfg, modes=modes,
                                    ew_only=False, t0=t0)
        else:
            # ew_only never touches predictions_dict; pass a sentinel to skip the disk reload
            run_cfg = metrics_light(pred_path=pred_path, config=run_cfg, predictions_dict={},
                                    modes=modes, ew_only=True, t0=t0)
        save_json(run_cfg, pred_path / "config")
        ew = run_cfg["metrics_all_events"]["event_wise"]
        aa = run_cfg["metrics_anomalies_alone"]["event_wise"]
        print(f"[{time.time()-t0:6.1f}s] p={p}: all_events P={ew['precision']} "
              f"R={ew['recall']} F0.5={ew['f0.5']} | anomalies_alone F0.5={aa['f0.5']}")

        if prune:
            for f in (pred_path / "labels").glob("channel_*.parquet"):
                f.unlink()
            shutil.rmtree(pred_path / "scores", ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channels_mode", choices=["light", "full"])
    ap.add_argument("ps", nargs="*", type=float, default=DEFAULT_PS)
    ap.add_argument("--phase", default="5", choices=["1", "2", "3", "4", "5"])
    ap.add_argument("--full-metrics", action="store_true")
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_intermixed_args()   # flags may precede the positional p values
    sweep(args.channels_mode, args.ps or DEFAULT_PS, phase=args.phase,
          full_metrics=args.full_metrics, prune=args.prune)
