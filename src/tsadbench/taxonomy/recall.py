"""Recall of detector runs per archetype.

A window counts as detected when the run's binary output, held constant between its own
samples (sample-and-hold), is 1 at any instant of the labelled interval [t0, t1]; the rule
is well defined for zero-duration labels too (was an alarm active at that instant?).
Windows are restricted to those starting inside the mission's test period. Runs use the
standard layout, ``<run>/labels/channel_NN.parquet`` with columns (timestamp, label).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tsadbench.io.misc import load_metadata

ARCHETYPE_ORDER = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "U0")


def in_test_period(windows: pd.DataFrame, mission: str = "ESA-Mission1") -> pd.DataFrame:
    start, end = pd.to_datetime(load_metadata()[mission]["test_period"])
    return windows[(windows.t0 >= start) & (windows.t0 <= end)].reset_index(drop=True)


def window_hits(windows: pd.DataFrame, run_dir: Path) -> np.ndarray:
    """1.0 / 0.0 per window: is the run's output active at some instant of the window?"""
    missing = [int(ch) for ch in sorted(windows.channel.unique())
               if not (run_dir / "labels" / f"channel_{ch}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name}: no labels for channels {missing}")
    hits = np.full(len(windows), np.nan)
    for ch, w in windows.groupby("channel"):
        df = pd.read_parquet(run_dir / "labels" / f"channel_{ch}.parquet")
        ts = df["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
        lab = df["label"].to_numpy(np.int8)
        o = np.argsort(ts)
        ts, lab = ts[o], lab[o]
        t0 = w.t0.to_numpy("datetime64[ns]").astype(np.int64)
        t1 = w.t1.to_numpy("datetime64[ns]").astype(np.int64)
        i0 = np.searchsorted(ts, t0, "left")
        i1 = np.searchsorted(ts, t1, "right")
        cum = np.concatenate([[0], np.cumsum(lab)])
        held = (i0 > 0) & (lab[np.maximum(i0 - 1, 0)] == 1)  # last output before t0
        hits[w.index] = (((cum[i1] - cum[i0]) > 0) | held).astype(float)
    return hits


def archetype_recall(windows: pd.DataFrame, runs: dict[str, Path],
                     mission: str = "ESA-Mission1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-window hits and per-archetype recall of each run in ``runs`` (name -> run folder).

    ``windows`` is the taxonomy window table, already restricted to the channels of
    interest. Returns (per-window table with one hit column per run, summary with one
    row per archetype, U0 = unassigned, and a final A1-A8 row).
    """
    W = in_test_period(windows, mission)
    for name, run_dir in runs.items():
        W[name] = window_hits(W, Path(run_dir))

    def row(label, S):
        r = dict(archetype=label, n_windows=len(S), n_events=S.event_id.nunique())
        for name in runs:
            r[f"{name}_hits"] = int(S[name].sum()) if len(S) else 0
            r[f"{name}_recall"] = round(S[name].mean(), 4) if len(S) else np.nan
        return r

    rows = [row(a, W[W.archetype_id == a]) for a in ARCHETYPE_ORDER]
    rows.append(row("A1-A8", W[W.archetype_id != "U0"]))
    return W, pd.DataFrame(rows)
