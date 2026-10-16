import time
import gc
import numpy as np
import pandas as pd
from pandas import Timestamp
from typing import Any, Dict, Tuple, List
from numpy.typing import NDArray

from smed_core.distance import best_match
from smed_core.helpers import load_json, load_gt_events, ROOT


def run_algorithm(
    values: NDArray[np.float32],
    slide_length: int = 5,
    query_length: int = 90,
    reference_length: int = 720,
    threshold: float = np.inf
) -> tuple[
    NDArray[np.int8],
    NDArray[np.int32],
    NDArray[np.float32],
]:
    n = len(values)
    num_windows = (n - query_length - reference_length - 1) // slide_length + 1

    labels = np.zeros(num_windows, dtype=np.int8)
    query_starts = np.empty(num_windows, dtype=np.int32)
    distances = np.empty(num_windows, dtype=np.float32)

    for j in range(num_windows):
        i = reference_length + j * slide_length
        query = values[i : i + query_length]
        query_starts[j] = i

        ref_start = i - reference_length
        reference = values[ref_start : i]

        distance, _ = best_match(query, reference)
        distances[j] = distance

        labels[j] = 1 if distance > threshold else 0

    return labels, query_starts, distances


def warm_up(
    channel: str,
    config: Dict[str, Any],
) -> Tuple[NDArray[np.float32], float]:

    phase = config["phase"]
    mission = config["mission"]
    metadata = load_json(ROOT / "data" / "metadata.json")
    duration = pd.DateOffset(months=metadata[mission]["train_durations"][int(phase) - 1])
    train_end = Timestamp(metadata[mission]["test_period"][0])
    mode = config["warm_up_mode"]

    train_file = ROOT / "data" / mission / config["signal_mode"] / f"{channel}.parquet"
    train_df = pd.read_parquet(train_file)

    train_df = train_df[train_df["timestamp"] <= train_end]
    train_start = train_df["timestamp"].min()

    if mode == "forward":
        t0 = train_start
        t1 = t0 + duration
        train_df = train_df[(train_df["timestamp"] >= t0) & (train_df["timestamp"] <= t1)]
    elif mode == "backward":
        t1 = train_end
        t0 = t1 - duration
        train_df = train_df[(train_df["timestamp"] >= t0) & (train_df["timestamp"] <= t1)]

    train_df = train_df.reset_index(drop=True)
    values_train = np.ascontiguousarray(train_df[channel].to_numpy(), dtype=np.float32)
    times_train = train_df["timestamp"]

    start_time = time.time()

    _, query_starts, warm_up_distances = run_algorithm(
        values=values_train,
        slide_length=config["slide_length"],
        query_length=config["query_length"],
        reference_length=config["reference_length"],
    )

    warmup_time = time.time() - start_time

    del train_df
    gc.collect()

    window_starts = times_train[query_starts].reset_index(drop=True)
    window_ends = times_train[query_starts + config["query_length"]].reset_index(drop=True)

    labels_path = ROOT / "data" / mission / "labels.csv"

    gt_df = (
        load_gt_events(labels_path, [channel])
        .loc[:, ["StartTime", "EndTime"]]
    )
    starts = np.sort(gt_df.StartTime.to_numpy("datetime64[ns]"))
    ends = np.sort(gt_df.EndTime.to_numpy("datetime64[ns]"))

    n_started_by_end = np.searchsorted(starts, window_ends, side="right")
    n_ended_before_start = np.searchsorted(ends, window_starts, side="left")
    mask_no_overlap = (n_started_by_end - n_ended_before_start) == 0

    warm_up_distances = warm_up_distances[mask_no_overlap]

    return warm_up_distances, warmup_time


def eval_channel(
    channel: str,
    config: Dict[str, Any],
) -> Tuple[
    str,
    List[pd.Timestamp],
    NDArray[np.int8],
    NDArray[np.float32],
    NDArray[np.float32],
    float,
    float,
]:
    warm_up_distances, warmup_time = warm_up(channel, config)

    mission = config["mission"]
    metadata = load_json(ROOT / "data" / "metadata.json")

    test_start = Timestamp(metadata[mission]["test_period"][0])
    test_file = ROOT / "data" / mission / config["signal_mode"] / f"{channel}.parquet"
    test_df = pd.read_parquet(test_file)
    test_df = test_df[test_df["timestamp"] >= test_start].reset_index(drop=True)

    values = np.ascontiguousarray(test_df[channel].to_numpy(), dtype=np.float32)
    times = test_df["timestamp"]

    del test_df
    gc.collect()

    p_l = float(np.percentile(warm_up_distances, config["p"]))
    p_h = float(np.percentile(warm_up_distances, (100 - config["p"])))
    ipr = p_h - p_l
    threshold = p_h + (config["k"] * ipr)

    eval_start = time.time()

    labels, query_starts, distances = run_algorithm(
        values=values,
        slide_length=config["slide_length"],
        query_length=config["query_length"],
        reference_length=config["reference_length"],
        threshold=threshold,
    )

    eval_time = time.time() - eval_start

    thresholds = np.full_like(distances, threshold, dtype=np.float32)

    return channel, times[query_starts].tolist(), labels, distances, thresholds, warmup_time, eval_time
