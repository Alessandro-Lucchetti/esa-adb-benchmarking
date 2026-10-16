"""Score prediction runs already on disk with the standard ESA-ADB metric suite.

Takes run folders in the usual layout (``config.json`` plus ``labels/channel_*.parquet``),
computes event-wise, channel/subsystem-aware, alarming precision, ADTQC and
affiliation-based metrics in both ``all_events`` and ``anomalies_alone`` modes, writes
them back into each ``config.json`` and collects everything in one summary CSV.

A path that has no ``config.json`` is treated as a parent folder and every run inside it
is scored.

Usage:
    python scripts/score_runs.py runs/telemanom
    python scripts/score_runs.py runs/telemanom/<run_name> --out summary.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsadbench.eval.orchestrator import metrics_light
from tsadbench.io.misc import save_json, load_json

MODES = ["all_events", "anomalies_alone"]
FLAT_KEYS = [
    ("event_wise", "precision"), ("event_wise", "recall"), ("event_wise", "f0.5"),
    ("subsystem_aware", "precision"), ("subsystem_aware", "recall"), ("subsystem_aware", "f0.5"),
    ("channel_aware", "precision"), ("channel_aware", "recall"), ("channel_aware", "f0.5"),
    ("alarming_precision", None),
    ("adtqc", "after_rate"), ("adtqc", "total_score"),
    ("affiliation_based", "precision"), ("affiliation_based", "recall"), ("affiliation_based", "f0.5"),
]


def collect_runs(paths: list[str]) -> list[Path]:
    runs = []
    for p in paths:
        path = Path(p)
        if (path / "config.json").exists():
            runs.append(path)
        elif path.is_dir():
            runs += sorted(c for c in path.iterdir() if (c / "config.json").exists())
        else:
            raise FileNotFoundError(f"not a run folder or a folder of runs: {path}")
    if not runs:
        raise FileNotFoundError(f"no runs found under {paths}")
    return runs


def flatten(run: Path, config: dict) -> list[dict]:
    rows = []
    for mode in MODES:
        metrics = config.get(f"metrics_{mode}", {})
        row = {
            "run": run.name,
            "algo": config.get("algo_name", ""),
            "channels_mode": config.get("channels_mode", ""),
            "n_channels": len(config.get("channels", [])),
            "mode": mode,
        }
        for group, key in FLAT_KEYS:
            name = group if key is None else f"{group}_{key}"
            value = metrics.get(group)
            row[name] = value if key is None else (value or {}).get(key)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folders, or folders containing runs")
    ap.add_argument("--out", default=None, help="summary CSV (default: <first path>/metrics_summary.csv)")
    ap.add_argument("--ew-only", action="store_true", help="event-wise metrics only (much faster)")
    args = ap.parse_args()

    runs = collect_runs(args.paths)
    print(f"scoring {len(runs)} runs\n")

    t0 = time.time()
    rows = []
    for i, run in enumerate(runs, 1):
        print(f"{'#' * 70}\n# [{i}/{len(runs)}] {run.name}\n{'#' * 70}", flush=True)
        config = load_json(run / "config.json")
        # metrics_light reuses an existing event.parquet; drop it so the predicted
        # events are always rebuilt from the labels currently on disk
        stale = run / "labels" / "event.parquet"
        if stale.exists():
            stale.unlink()
        config = metrics_light(pred_path=run, config=config, modes=MODES,
                               ew_only=args.ew_only, t0=t0)
        save_json(config, run / "config")
        rows.append(flatten(run, config))

    df = pd.DataFrame([r for group in rows for r in group])
    out = Path(args.out) if args.out else Path(args.paths[0]) / "metrics_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nsummary written to {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
