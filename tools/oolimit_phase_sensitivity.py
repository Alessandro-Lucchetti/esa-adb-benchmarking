"""Training-length sensitivity of the robust out-of-limit (OOL) baseline.

Collects the `runs/oolimit[_full]/oolimit_*_phase_<n>_*_p<p>/config.json` written by
    python scripts/limits_psweep.py <set> --phase <n> <p>
for the five ESA-ADB training phases of Mission 1 (3 / 10 / 21 / 42 / 84 months of
nominal history for the envelope) and tabulates the corrected event-wise F0.5 (all
events), in the layout of the paper's training-length table (tab:phase_stats_m1).
The published SMED / Q-GPR / Telemanom-ESA-Pruned values for the lightweight set
are carried along for reference (they are not recomputed here).

Usage:
    python tools/oolimit_phase_sensitivity.py [--p 0.1] [--k 1.5]
"""
import argparse
import json

import numpy as np
import pandas as pd

from tsadbench.io.misc import project_root, load_metadata

ROOTS = {"light": "oolimit", "full": "oolimit_full"}

# paper, tab:phase_stats_m1 (lightweight set, corrected event-wise F0.5, all events)
PAPER_LIGHT = {
    "training_length": ["9 weeks", "8 months", "18 months", "39 months", "81 months"],
    "validation_length": ["3 weeks", "2 months", "3 months", "3 months", "3 months"],
    "smed": [0.724, 0.786, 0.786, 0.803, 0.811],
    "qgpr": [0.302, 0.767, 0.856, 0.883, 0.898],
    "telemanom_esa_pruned": [0.227, 0.311, 0.776, 0.776, 0.786],
}


def collect(p: float = 0.1, k: float = 1.5) -> pd.DataFrame:
    months = load_metadata()["ESA-Mission1"]["train_durations"]
    rows = []
    for set_name, folder in ROOTS.items():
        for cfg_file in sorted((project_root() / "runs" / folder).glob("oolimit_*/config.json")):
            cfg = json.loads(cfg_file.read_text())
            if float(cfg["p"]) != p or float(cfg["k"]) != k or "metrics_all_events" not in cfg:
                continue
            ew = cfg["metrics_all_events"]["event_wise"]
            ev_file = cfg_file.parent / "labels" / "event.parquet"
            phase = int(cfg["phase"])
            rows.append({"set": set_name, "phase": phase, "warm_up_months": months[phase - 1],
                         "precision": float(ew["precision"]), "recall": float(ew["recall"]),
                         "f0.5": float(ew["f0.5"]),
                         "n_predicted_events": len(pd.read_parquet(ev_file)) if ev_file.exists() else np.nan})
    df = pd.DataFrame(rows).sort_values(["set", "phase"], ascending=[False, True])
    return df.reset_index(drop=True)


def paper_table(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight-set table in the paper's column order, OOL appended."""
    light = df[df["set"] == "light"].set_index("phase")["f0.5"]
    full = df[df["set"] == "full"].set_index("phase")["f0.5"]
    out = pd.DataFrame({"phase": [f"Phase {i}" for i in range(1, 6)], **PAPER_LIGHT})
    out["ool_light"] = [light.get(i, np.nan) for i in range(1, 6)]
    out["ool_full"] = [full.get(i, np.nan) for i in range(1, 6)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.1, help="envelope percentile of the runs to collect")
    ap.add_argument("--k", type=float, default=1.5)
    args = ap.parse_args()

    runs = collect(args.p, args.k)
    table = paper_table(runs)
    out = project_root() / "figures" / "oolimit_sensitivity_phase_ESA-Mission1"
    out.parent.mkdir(exist_ok=True)
    runs.to_csv(f"{out}_runs.csv", index=False)
    table.to_csv(f"{out}.csv", index=False)
    print(runs.to_string(index=False))
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {out}.csv / {out}_runs.csv")
