"""Efficient DAMP (z-norm) percentile sweep for the SMED vs DAMP distance ablation.

The IPR/Tukey threshold and the resulting labels depend on p, but the underlying
warm-up and test *distances* do not. So we compute each channel's warm-up distance
set and test distance profile exactly once (the expensive part), then re-threshold
across every p cheaply.

For each p we write a standard run folder (labels/ + scores/ + config.json) under
runs/<root>/ and score it with the shared event-wise pipeline, so the DAMP rows sit
next to SMED/Telemanom identically. p=0.1 is the strict controlled ablation (SMED's
exact config, only the distance metric differs); the larger p values steelman DAMP
with its own best threshold.

The sweep itself is distance-agnostic: scripts/smed_psweep.py reuses it with SMED's raw
Euclidean kernel for the p sensitivity of SMED at the nominal window sizes.

Usage:
    python scripts/damp_psweep.py <channels_mode> [--phase N] [p1 p2 ...]
        channels_mode: light | full
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsadbench.detectors.damp.distance import best_match_znorm
from tsadbench.detectors.smed.core import warm_up, run_algorithm
from tsadbench.data.splits import Split
from tsadbench.runner.config import RunConfig
from tsadbench.runner.naming import build_run_name
from tsadbench.runner.workers import worker_count
from tsadbench.eval.metrics.utils import label_changes
from tsadbench.eval.orchestrator import metrics_light
from tsadbench.io.misc import save_json, project_root, load_metadata

ALGO = "damp_znorm"
DEFAULT_PS = [0.1, 1.0, 3.0, 5.0, 10.0, 25.0]


def channel_distances(ch: str, base: dict, path: Path, test_start: pd.Timestamp, dist_fn):
    """Warm-up distances (no-overlap masked), window-start timestamps and test distances."""
    wd, _ = warm_up(ch, base, dist_fn=dist_fn)
    test_df = pd.read_parquet(path)
    test_df = test_df[test_df["timestamp"] >= test_start].reset_index(drop=True)
    values = np.ascontiguousarray(test_df[ch].to_numpy(), dtype=np.float32)
    _, qs, dist = run_algorithm(values=values, slide_length=base["slide_length"],
                                query_length=base["query_length"],
                                reference_length=base["reference_length"],
                                threshold=np.inf, dist_fn=dist_fn)
    return wd, test_df["timestamp"].iloc[qs].reset_index(drop=True), dist


def sweep(channels_mode: str, ps: list[float], phase: str = "5",
          dist_fn=best_match_znorm, algo: str = ALGO, euclidean: str = "znorm") -> None:
    t0 = time.time()
    cfg = RunConfig(algo_name=algo, mission="ESA-Mission1", channels_mode=channels_mode,
                    signal_mode="raw", phase=str(phase), warm_up_mode="forward",
                    query_length=100, reference_length=750, slide_length=5, k=1.5)
    split = Split(cfg)
    base = cfg.to_run_dict()
    channels = split.channels
    metadata = load_metadata()
    test_start = pd.Timestamp(metadata["ESA-Mission1"]["test_period"][0])

    runs_root = project_root() / "runs" / (f"{algo}_full" if channels_mode == "full" else algo)

    # ---- expensive pass: distances computed once per channel, channels in parallel ----
    warm = {}          # channel -> warm-up distances (no-overlap masked)
    test = {}          # channel -> (timestamps of window starts, test distances)
    workers = worker_count(len(channels))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = {ch: ex.submit(channel_distances, ch, base, split.channel_path(ch), test_start, dist_fn)
                for ch in channels}
        for i, ch in enumerate(channels):
            wd, ts, dist = jobs[ch].result()
            warm[ch] = wd
            test[ch] = (ts, dist)
            print(f"[{time.time()-t0:6.1f}s] distances done {i+1}/{len(channels)} {ch} "
                  f"(warm={len(wd)}, test={len(dist)})")

    # ---- cheap pass: threshold + label + score for each p ----
    for p in ps:
        run_cfg = {**base, "p": p, "channels": channels}
        run_name = build_run_name(run_cfg, channels)
        pred_path = runs_root / run_name
        (pred_path / "labels").mkdir(parents=True, exist_ok=True)
        (pred_path / "scores").mkdir(parents=True, exist_ok=True)

        predictions_dict = {}
        for ch in channels:
            wd = warm[ch]
            p_l = float(np.percentile(wd, p))
            p_h = float(np.percentile(wd, 100 - p))
            thr = p_h + base["k"] * (p_h - p_l)
            ts, dist = test[ch]
            lab = (dist > thr).astype(np.int8)
            pd.DataFrame({"timestamp": ts, "label": lab}).to_parquet(
                pred_path / "labels" / f"{ch}.parquet")
            pd.DataFrame({"timestamp": ts, "score": dist,
                          "threshold": np.full(len(dist), thr, np.float32)}).to_parquet(
                pred_path / "scores" / f"{ch}.parquet")
            predictions_dict[ch] = label_changes(ts, lab)

        stale = pred_path / "labels" / "event.parquet"
        if stale.exists():
            stale.unlink()

        run_cfg["euclidean"] = euclidean
        save_json(run_cfg, pred_path / "config")
        run_cfg = metrics_light(pred_path=pred_path, config=run_cfg,
                                predictions_dict=predictions_dict,
                                modes=["all_events", "anomalies_alone"], ew_only=True, t0=t0)
        save_json(run_cfg, pred_path / "config")
        ew = run_cfg["metrics_all_events"]["event_wise"]
        aa = run_cfg["metrics_anomalies_alone"]["event_wise"]
        print(f"[{time.time()-t0:6.1f}s] p={p}: all_events P={ew['precision']} "
              f"R={ew['recall']} F0.5={ew['f0.5']} | anomalies_alone F0.5={aa['f0.5']}")


def cli(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("channels_mode", choices=["light", "full"])
    ap.add_argument("ps", nargs="*", type=float, default=DEFAULT_PS)
    ap.add_argument("--phase", default="5", help="ESA-ADB training phase 1-5 (warm-up length)")
    return ap.parse_intermixed_args()   # flags may precede the positional p values


if __name__ == "__main__":
    args = cli(__doc__)
    sweep(args.channels_mode, args.ps or DEFAULT_PS, phase=args.phase)
