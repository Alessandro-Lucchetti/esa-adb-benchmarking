import time
import json
import numpy as np
import pandas as pd
import re
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.axes as m_axes
from tqdm import tqdm
import portion as P

from tsadbench.eval.metrics.utils import convert_time_series_to_events, label_changes


def save_json(data: dict, out_file: str | Path) -> None:
    """
    Serialize *data* to ``<out_file>.json`` with 2-space indentation.
    """
    out_file = Path(out_file).with_suffix(".json")
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

def load_json(in_file: str | Path) -> dict:
    """
    Load and return an object from a JSON file created by :func:`save_json`.
    """
    in_file = Path(in_file).with_suffix(".json")
    with open(in_file, "r") as f:
        return json.load(f)

def load_metadata() -> dict:
    """
    Load and return the metadata.json file.
    """
    return load_json(project_root() / "data" / "metadata")

def parse_channels_mode(
    config: dict,
    dev: str = "channel_41",
    ) -> list[str]:
    """
    Determine which telemetry channels to use based on ``config``.
    """
    channels_mode = config["channels_mode"]
    mission = config["mission"]
    metadata = load_json(project_root() / "data" / "metadata")

    if channels_mode == "full":
        channels = metadata[mission]["full"]

    elif channels_mode == "light":
        channels = metadata[mission]["light"]

    elif channels_mode == "manual":
        raw_input = input("Input channel numbers, separated by commas: ").split(",")
        channels = [f"channel_{c.strip()}" for c in raw_input]

    elif channels_mode == "dev":
        channels = [dev]

    else:
        raise ValueError(f"Invalid channels mode choice: {channels_mode}")

    return channels

def load_gt_events(
    label_path: str | Path,
    channels: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    mode: str = "all_events",
    ) -> pd.DataFrame:
    """
    Load and filter ground-truth event intervals.
    """
    df = pd.read_csv(label_path)
    df["StartTime"] = pd.to_datetime(df["StartTime"]).dt.tz_convert(None)
    df["EndTime"] = pd.to_datetime(df["EndTime"]).dt.tz_convert(None)

    if mode != "all_events":
        anomaly_types_path = Path(label_path).parent / "anomaly_types.csv"
        types_df = pd.read_csv(anomaly_types_path)
        types_df = types_df[types_df["Category"] == "Anomaly"]
        df = df[df["ID"].isin(types_df["ID"])]

    if channels is not None:
        df = df[df["Channel"].isin(channels)]
    if start is not None and end is not None:
        df = filter_time(df, start, end)
    return df


def filter_time(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ) -> pd.DataFrame:
    """
    Clip each interval in *df* to ``[start, end]`` (inclusive).
    """
    df = df[(df["EndTime"] >= start) & (df["StartTime"] <= end)].copy()
    df["StartTime"] = df["StartTime"].apply(lambda t: max(t, start))
    df["EndTime"]   = df["EndTime"].apply(lambda t: min(t, end))
    return df

def load_predictions_dict(
    path: str | Path,
    start: pd.Timestamp = None,
    end: pd.Timestamp = None,
    t0: float = None,
    changes_only: bool = False,
    ) -> dict[str, list[tuple[pd.Timestamp, int]]]:
    """
    Read all per channel Pandas-backed Parquet file in the folder indicated by path.
    ``changes_only`` keeps only the rows the metrics read (``label_changes``).
    """
    if t0 is None:
        t0 = time.time()
    print(f"[{(time.time() - t0):6.2f}s] Loading predictions dict from {path}")
    predictions_folder = Path(path) / "labels"
    predictions_dict = {}

    pattern = re.compile(r"^channel_\d+\.parquet$")

    for parquet_file in predictions_folder.glob("channel_*.parquet"):
        if pattern.match(parquet_file.name):
            ch = parquet_file.stem
            df= pd.read_parquet(parquet_file)
            if start is not None and end is not None:
                df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
            if changes_only:
                predictions_dict[ch] = label_changes(df["timestamp"], df["label"].values)
            else:
                predictions_dict[ch] = list(zip(df["timestamp"], df["label"].values))

    config = load_json(Path(path) / "config.json")
    channels = parse_channels_mode(config)
    for ch in channels:
        if ch not in predictions_dict:
            predictions_dict[ch] = [[start, 0], [end, 0]]

    return predictions_dict

def load_predictions_event(
    path: str | Path,
    start: pd.Timestamp = None,
    end: pd.Timestamp = None,
    group: dict[str, list[str]] = None,
    t0: float = None,
    ) -> P.Interval:
    """
    Read (or create, if not presente) event.parquet and event DataFrame.
    """
    if t0 is None:
        t0 = time.time()

    predictions_folder = Path(path)
    if group is not None:
        assert len(group) == 1, "Only one group at a time is supported"
        group, channels = next(iter(group.items())) 
        predictions_event_file = predictions_folder / "labels" / f"event-group_{group}.parquet"
    else:
        predictions_event_file = predictions_folder / "labels" / "event.parquet"

    if not predictions_event_file.exists():
        print(f"[{(time.time() - t0):6.2f}s] Predictions event file not found at {predictions_event_file}.")
        if group is None:
            predictions_event = build_predictions_event(predictions_folder, predictions_event_file, t0=t0)
        else:
            predictions_event = build_predictions_event(predictions_folder, predictions_event_file, channels, t0=t0)
    else:
        print(f"[{(time.time() - t0):6.2f}s] Loading predictions event from {predictions_event_file}")
        event_predictions_df = pd.read_parquet(predictions_event_file)
        intervals = []
        for event_start, event_end in event_predictions_df.itertuples(index=False, name=None):
            intervals.append(P.closedopen(event_start, event_end))
        predictions_event = P.Interval(*intervals)

    if start is not None and end is not None:
        predictions_event = P.closed(start, end) & predictions_event

    return predictions_event

def project_root(markers=('.git', 'pyproject.toml')) -> Path:
    """Closest ancestor of this file holding any of *markers*.

    ``pyproject.toml`` covers checkouts without ``.git`` (release zips, ``git archive``).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    raise RuntimeError("Project root not found")

def select_prediction():
    # imported here so headless installs without Tk (e.g. Debian python3 without python3-tk)
    # can still import the package
    import tkinter as tk
    from tkinter import filedialog

    tk_root = tk.Tk()
    tk_root.withdraw()
    tk_root.attributes("-topmost", True)
    pred_folder = filedialog.askdirectory(title="Pick a predictions folder")
    tk_root.destroy()
    if not pred_folder:
        raise ValueError("No folder selected – aborting.")
    return Path(pred_folder).resolve()


def add_event_bands(
        ax: m_axes.Axes,
        events_df: pd.DataFrame,
        channel: str,
        facecolor: str,
        edgecolor: str = "none",
        *,
        y_min: float = -1_000.0,
        y_height: float = 3_000.0,
        edge_alpha: float = 0.3,
) -> None:
    rows = events_df[events_df["Channel"] == channel]
    for _, row in rows.iterrows():
        ax.add_artist(
            Rectangle(
                (row["StartTime"], y_min),
                row["EndTime"] - row["StartTime"],
                y_height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=edge_alpha,
            )
        )

def build_union_labels(
    pred_path: str | Path,
    channels: list[str] = None,
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    t0: float = None,
) -> list[tuple[pd.Timestamp, int]]:
    """
    Union of the channels' labels (1 where any channel is anomalous) as the
    ``(timestamp, label)`` rows that ``convert_time_series_to_events`` turns into events.

    Each channel holds its label from one of its timestamps to the next (0 before its
    first one), so on the union of all timestamps the union label can only change where a
    channel starts, changes label or ends: it is evaluated on those timestamps only, which
    gives the same events as the whole union grid with memory proportional to the label
    changes.
    """
    if t0 is None:
        t0 = time.time()

    pred_path = Path(pred_path) / "labels"
    if channels is not None:
        print(f"[{(time.time() - t0):6.2f}s] Building union labels from channels: {channels}")
        files = [pred_path / f"{ch}.parquet" for ch in channels]
    else:
        print(f"[{(time.time() - t0):6.2f}s] Building union labels from all available channels under {pred_path}")
        files = sorted(pred_path.glob("channel_*.parquet"))
        pattern = re.compile(r"^channel_\d+\.parquet$")
        files = [f for f in files if pattern.match(f.name)]
    if not files:
        raise FileNotFoundError(
            f'[{(time.time() - t0):6.2f}s] No parquet files matching "^channel_\\d+\\.parquet$" under {pred_path}'
        )

    # per channel: its first and last timestamps and those where its label changes
    steps = []
    for f in tqdm(files, desc=f"[{(time.time() - t0):6.2f}s] Loading parquet files", unit="file"):
        df = pd.read_parquet(f, columns=[timestamp_col, label_col])
        t = df[timestamp_col].to_numpy("datetime64[ns]").view("int64")
        lab = df[label_col].to_numpy().astype("uint8")
        order = np.argsort(t, kind="stable")
        t, lab = t[order], lab[order]
        if len(t) == 0:
            continue
        keep = np.ones(len(t), bool)
        np.not_equal(lab[1:-1], lab[:-2], out=keep[1:-1])
        steps.append((t[keep], lab[keep]))

    grid = np.unique(np.concatenate([t for t, _ in steps]))
    any_active = np.zeros(len(grid), "uint8")
    for t, lab in steps:
        i = np.searchsorted(t, grid, side="right") - 1      # last own timestamp <= grid point
        np.maximum(any_active, np.where(i >= 0, lab[np.maximum(i, 0)], 0).astype("uint8"), out=any_active)

    return list(zip(pd.DatetimeIndex(grid.view("datetime64[ns]")), any_active))


def build_predictions_event(
    pred_path: str | Path,
    predictions_event_file: str | Path,
    channels: list[str] = None,
    t0: float = None,
) -> P.Interval:
    """
    """
    if t0 is None:
        t0 = time.time()
    print(f"[{(time.time() - t0):6.2f}s] Building predictions event from {pred_path}")
    pred_path = Path(pred_path)
    aggregated = build_union_labels(pred_path, channels)
    predictions_event = convert_time_series_to_events(aggregated)
    print(f"[{(time.time() - t0):6.2f}s] Converting Portion intervals to list of tuples")
    predictions_event_list = [(iv.lower, iv.upper) for iv in predictions_event]
    print(f"[{(time.time() - t0):6.2f}s] Converting list of tuples to DataFrame and saving to {predictions_event_file}")
    predictions_event_df = pd.DataFrame(predictions_event_list, columns=["StartTime", "EndTime"])
    predictions_event_df.to_parquet(predictions_event_file, index=False)

    return predictions_event