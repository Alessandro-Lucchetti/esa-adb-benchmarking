"""ESA-ADB published Telemanom-ESA predictions in the standard run layout.

The ESA-ADB benchmark distributes TimeEval run folders containing ``anomaly_scores.ts``:
a headerless CSV with one row per test timestamp and one column per target channel,
already thresholded to binary (``threshold_scores=1`` in ``hyper_params.json``). Column
order follows ``hyper_params.json`` ``target_channels``. Two conventions differ from this
repository:

- channel names are 0-based in ESA-ADB (channel_0..channel_75 for Mission 1), ours are
  1-based, so channel_N maps to channel_{N+1};
- rows carry no timestamps: they are a regular 30 s grid over the test period, both ends
  included (Mission 1: 2007-01-01..2014-01-01 = 7,364,161 rows).

Scoring the ingested labels reproduces the run's own metrics.csv (event-wise P / R / F0.5
0.049 / 0.868 / 0.061 against the published 0.0499 / 0.8696 / 0.0614).
"""
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from tsadbench.io.misc import load_metadata

# the full-set Telemanom-ESA-Pruned run of Mission 1 reported by ESA-ADB (Phase 5)
FULL_SET_RUN = "Full-set/Telemanom-ESA/f81ddbb7f1c4198051347bcf01e3f3ac/ESA-Mission1/84_months/1/"
ROWS_PER_BLOCK = 100_000


def read_binary_scores(f: BinaryIO, n_rows: int, n_cols: int) -> np.ndarray:
    """Parse ``anomaly_scores.ts`` into an (n_rows, n_cols) int8 array.

    The binary scores are written as fixed-width floats (``0.000000000000000000e+00`` or
    ``1.000000000000000000e+00``), so each block of rows is decoded as a byte matrix and
    checked against the expected layout; anything else raises.
    """
    first = f.readline()
    width = len(first)
    fields = first.rstrip(b"\r\n").split(b",")
    if len(fields) != n_cols or len({len(x) for x in fields}) != 1:
        raise ValueError(f"unexpected first row: {len(fields)} fields, expected {n_cols} of equal width")
    step = len(fields[0]) + 1
    digit = np.arange(n_cols) * step
    template = np.frombuffer(first, np.uint8).copy()
    template[digit] = ord("0")
    rest = np.ones(width, bool)
    rest[digit] = False

    out = np.empty((n_rows, n_cols), np.int8)
    buf, row = first, 0
    while row < n_rows:
        k = min(ROWS_PER_BLOCK, n_rows - row)
        need = k * width - len(buf)
        data = buf + (f.read(need) if need > 0 else b"")
        buf = b""
        if len(data) != k * width:
            raise ValueError(f"file ends after {row + len(data) // width} rows, expected {n_rows}")
        B = np.frombuffer(data, np.uint8).reshape(k, width)
        d = B[:, digit]
        if not ((B[:, rest] == template[rest]).all() and np.isin(d, (ord("0"), ord("1"))).all()):
            raise ValueError(f"rows {row}..{row + k}: not binary fixed-width scores")
        out[row:row + k] = d - ord("0")
        row += k
    if f.read(1):
        raise ValueError(f"more than {n_rows} rows")
    return out


def ingest(scores: BinaryIO, hyper_params: dict, out: Path, mission: str = "ESA-Mission1",
           source: str = "") -> dict:
    """Write ``labels/channel_NN.parquet`` and ``config.json`` under ``out``."""
    src_channels = hyper_params["target_channels"]  # 0-based names, column order
    out_channels = [f"channel_{int(c.split('_')[1]) + 1}" for c in src_channels]

    metadata = load_metadata()
    start, end = metadata[mission]["test_period"]
    timestamps = pd.date_range(start, end, freq="30s")
    print(f"{len(src_channels)} columns; grid {start}..{end} = {len(timestamps)} rows")
    labels = read_binary_scores(scores, len(timestamps), len(src_channels))

    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for j, ch in enumerate(out_channels):
        pd.DataFrame({"timestamp": timestamps, "label": labels[:, j]}).to_parquet(
            labels_dir / f"{ch}.parquet")
        stats[ch] = int(labels[:, j].sum())

    full = set(metadata[mission]["full"])
    config = {
        "algo_name": "Telemanom-ESA",
        "mission": mission,
        "channels_mode": "full" if set(out_channels) == full else "light",
        "signal_mode": "esa-adb-30s-grid",
        "channels": out_channels,
        "source": source,
        "provenance": ("ESA-ADB published benchmark predictions (TimeEval "
                       "anomaly_scores.ts, threshold_scores=1 -> binary per-channel "
                       "labels); ESA-ADB 0-based channel_N mapped to 1-based "
                       "channel_N+1; timestamps = 30s grid over the test period"),
        "positives_per_channel": stats,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(out_channels)} label parquets + config.json under {out}")
    return config
