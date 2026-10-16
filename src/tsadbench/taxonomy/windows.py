"""Labelled windows, the unit of analysis of the event taxonomy.

A window is one row of ``labels.csv`` on a target channel: one labelled (event, channel)
interval. A pair with several disjoint intervals gives one window per interval, numbered
in time order (``interval``). Each window is read together with a context on both sides,

    pre  = clip(5 x duration, 1 d, 30 d)    before the start
    post = clip(duration, 1 d, 7 d)         after the end

Event samples are always kept at full resolution. A context side longer than ``CTX_CAP``
samples is min-max decimated on uniform time bins (original timestamps kept), and its
rows are flagged ``dec``; samples of the context that fall inside another labelled
interval of the channel are flagged ``other_ev``.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tsadbench.io.misc import load_metadata, project_root

DAY_S = 86_400
CTX_CAP = 100_000
PRE, EVENT, POST = 0, 1, 2


def data_dir(mission: str) -> Path:
    return project_root() / "data" / mission


def load_windows(mission: str = "ESA-Mission1") -> pd.DataFrame:
    """Every labelled interval on a target channel, ordered by channel and start time.

    Columns: ``event_id, channel, interval, t0, t1, group, esa_category`` (times tz-naive
    UTC, ``group`` from ``channels.csv``, ``esa_category`` from ``anomaly_types.csv``).
    """
    d = data_dir(mission)
    targets = load_metadata()[mission]["full"]
    lab = pd.read_csv(d / "labels.csv")
    lab = lab[lab["Channel"].isin(targets)].copy()
    lab["t0"] = pd.to_datetime(lab["StartTime"]).dt.tz_convert(None)
    lab["t1"] = pd.to_datetime(lab["EndTime"]).dt.tz_convert(None)
    lab["channel"] = lab["Channel"].str.removeprefix("channel_").astype(int)
    lab = lab.sort_values(["channel", "t0"], kind="stable").reset_index(drop=True)
    lab["interval"] = lab.groupby(["ID", "channel"]).cumcount()

    groups = pd.read_csv(d / "channels.csv").set_index("Channel")["Group"]
    categories = pd.read_csv(d / "anomaly_types.csv").set_index("ID")["Category"]
    return pd.DataFrame({
        "event_id": lab["ID"],
        "channel": lab["channel"],
        "interval": lab["interval"],
        "t0": lab["t0"],
        "t1": lab["t1"],
        "group": lab["Channel"].map(groups).astype(int),
        "esa_category": lab["ID"].map(categories),
    })


def load_channel(mission: str, channel: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw samples of one channel as (datetime64[ns], float64), sorted, without NaN and
    keeping the first value on duplicated timestamps."""
    col = f"channel_{channel}"
    df = pd.read_parquet(data_dir(mission) / "raw" / f"{col}.parquet")
    df = df.dropna(subset=[col]).sort_values("timestamp", kind="stable")
    df = df.drop_duplicates(subset="timestamp", keep="first")
    return df["timestamp"].to_numpy("datetime64[ns]"), df[col].to_numpy(np.float64)


def context_seconds(t0: pd.Timestamp, t1: pd.Timestamp) -> tuple[int, int]:
    """Length in seconds of the (pre, post) context of the window [t0, t1)."""
    dur_h = (t1 - t0).total_seconds() / 3600
    pre = int(min(max(5 * dur_h * 3600, DAY_S), 30 * DAY_S))
    post = int(min(max(dur_h * 3600, DAY_S), 7 * DAY_S))
    return pre, post


def in_intervals(ts: np.ndarray, intervals: list[tuple]) -> np.ndarray:
    """True where ``ts`` falls inside any [t0, t1) of ``intervals``."""
    inside = np.zeros(len(ts), dtype=bool)
    for t0, t1 in intervals:
        inside[np.searchsorted(ts, np.datetime64(t0)):np.searchsorted(ts, np.datetime64(t1))] = True
    return inside


def minmax_decimate(ts: np.ndarray, v: np.ndarray, cap: int) -> np.ndarray:
    """Indices of a min-max decimation of (ts, v) to about ``cap`` samples: per uniform time
    bin the argmin and the argmax, plus the first and last sample."""
    n = len(ts)
    if n <= cap:
        return np.arange(n)
    n_bins = max(cap // 2, 1)
    span = (ts[-1] - ts[0]).astype("timedelta64[ns]").astype(np.int64)
    if span <= 0:
        return np.unique(np.linspace(0, n - 1, cap).astype(np.int64))
    rel = (ts - ts[0]).astype("timedelta64[ns]").astype(np.int64)
    bins = np.minimum((rel * n_bins) // (span + 1), n_bins - 1)
    keep = {0, n - 1}
    order = np.argsort(bins, kind="stable")
    b_sorted = bins[order]
    starts = np.searchsorted(b_sorted, np.arange(n_bins), side="left")
    ends = np.searchsorted(b_sorted, np.arange(n_bins), side="right")
    for s, e in zip(starts, ends):
        if e > s:
            idx = order[s:e]
            keep.add(int(idx[np.argmin(v[idx])]))
            keep.add(int(idx[np.argmax(v[idx])]))
    return np.array(sorted(keep), dtype=np.int64)


def segment(ts: np.ndarray, v: np.ndarray, t0: pd.Timestamp, t1: pd.Timestamp,
            others: list[tuple], ctx_cap: int = CTX_CAP) -> pd.DataFrame | None:
    """Samples of one window and its context, or None when there are none at all.

    Columns: ``timestamp, value, seg`` (PRE / EVENT / POST), ``other_ev``, ``dec``.
    ``others`` are the channel's other labelled intervals.
    """
    pre_s, post_s = context_seconds(t0, t1)
    i0 = np.searchsorted(ts, np.datetime64(t0))
    i1 = np.searchsorted(ts, np.datetime64(t1))
    j0 = np.searchsorted(ts, np.datetime64(t0) - np.timedelta64(pre_s, "s"))
    k1 = np.searchsorted(ts, np.datetime64(t1) + np.timedelta64(post_s, "s"))

    parts = []
    for lo, hi, code in ((j0, i0, PRE), (i0, i1, EVENT), (i1, k1, POST)):
        if hi <= lo:
            continue
        sts, sv = ts[lo:hi], v[lo:hi]
        dec = code != EVENT and len(sts) > ctx_cap
        if dec:
            idx = minmax_decimate(sts, sv, ctx_cap)
            sts, sv = sts[idx], sv[idx]
        parts.append(pd.DataFrame({
            "timestamp": sts, "value": sv,
            "seg": np.full(len(sts), code, dtype=np.int8),
            "other_ev": in_intervals(sts, others) if others else np.zeros(len(sts), dtype=bool),
            "dec": np.full(len(sts), dec, dtype=bool),
        }))
    return pd.concat(parts, ignore_index=True) if parts else None
