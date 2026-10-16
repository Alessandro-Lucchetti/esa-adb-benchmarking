# Import required python libraries:
import time
from math import inf
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from pandas import Timestamp
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from typing import Any, Dict, Tuple, List, Optional
from numpy.typing import NDArray

from tsadbench.eval.metrics.utils import label_changes
from tsadbench.eval.orchestrator import metrics_light
from tsadbench.io.misc import parse_channels_mode, save_json, load_json, project_root, load_gt_events
from tsadbench.detectors.smed.distance import best_match
from tsadbench.runner.naming import build_run_name
from tsadbench.runner.workers import worker_count


# %% Functions definitions:

def run_algorithm(
    values: NDArray[np.float32],
    slide_length: int = 5,
    query_length: int = 90,
    reference_length: int = 720,
    threshold: float = np.inf,
    dist_fn=best_match,
) -> tuple[
    NDArray[np.int8],     # labels
    NDArray[np.int32],    # query_starts
    NDArray[np.float32],  # distances
]:

    # --- allocate output arrays ---
    n              = len(values)
    num_windows    = (n - query_length - reference_length - 1) // slide_length + 1

    if num_windows <= 0:
        raise ValueError(
            f"channel too short for SMED: need > reference_length+query_length+1 "
            f"({reference_length + query_length + 1}) samples, got {n}"
        )

    labels         = np.zeros(num_windows, dtype=np.int8)
    query_starts   = np.empty(num_windows, dtype=np.int32)
    distances      = np.empty(num_windows, dtype=np.float32)

    for j in range(num_windows):
        # Define query:
        i = reference_length + j * slide_length
        query = values[i : i + query_length]
        query_starts[j] = i

        # Define reference:
        ref_start = i - reference_length
        reference = values[ref_start : i]

        # Compute minimum (Euclidean or z-normalized) distance:
        distance, _ = dist_fn(query, reference)
        distances[j] = distance

        # Predict anomaly only when distance > threshold
        labels[j] = 1 if distance > threshold else 0

    return labels, query_starts, distances


def warm_up(
    channel: str,
    config: Dict[str, Any],
    dist_fn=best_match,
    ) -> Tuple[NDArray[np.float32], float]:  # Return distances and warmup time

    phase    = config["phase"]
    mission  = config["mission"]
    metadata = load_json(project_root() / "data" / "metadata.json")
    duration = pd.DateOffset(months=metadata[mission]["train_durations"][int(phase)-1])
    train_end = Timestamp(metadata[mission]["test_period"][0])
    mode     = config["warm_up_mode"]

    train_file = project_root() / "data" / mission / config["signal_mode"] / f"{channel}.parquet"
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

    _, query_starts, warm_up_distances = run_algorithm(values=values_train,
                                        slide_length=config["slide_length"],
                                        query_length=config["query_length"],
                                        reference_length=config["reference_length"],
                                        dist_fn=dist_fn)

    warmup_time = time.time() - start_time

    del train_df
    gc.collect()

    window_starts = times_train[query_starts].reset_index(drop=True)
    window_ends = times_train[query_starts + config["query_length"]].reset_index(drop=True)

    labels_path = project_root() / "data" / mission / "labels.csv"

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
    dist_fn=best_match,
    ) -> Tuple[
        str,                           # channel name
        List[pd.Timestamp],            # timestamps
        NDArray[np.int8],              # labels
        NDArray[np.float32],           # distances
        NDArray[np.float32],           # thresholds
        float,                         # warmup time
        float                          # evaluation time
    ]:

    warm_up_distances, warmup_time = warm_up(channel, config, dist_fn=dist_fn)

    mission   = config["mission"]
    test_file = project_root() / "data" / mission / config["signal_mode"] / f"{channel}.parquet"
    metadata = load_json(project_root() / "data" / "metadata.json")

    test_start = Timestamp(metadata[mission]["test_period"][0])
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

    labels, query_starts, distances = run_algorithm(values=values,
                                                    slide_length=config["slide_length"],
                                                    query_length=config["query_length"],
                                                    reference_length=config["reference_length"],
                                                    threshold=threshold,
                                                    dist_fn=dist_fn)

    eval_time = time.time() - eval_start

    thresholds = np.full_like(distances, threshold, dtype=np.float32)

    return channel, times[query_starts].tolist(), labels, distances, thresholds, warmup_time, eval_time


def run_smed(config: Dict[str, Any], t0: float | None = None):
    t0 = t0 or time.time()

    if config["experimental_mode"]:
        config["plot"] = False

    if config["mission"] == "ESA-Mission2" and config["phase"] == "5":
        print(
            f"[{(time.time() - t0):6.2f}s] WARNING: Phase 5 is available only for ESA-Mission1. Switching to Phase 4.")
        config["phase"] = "4"

    algo_name = config["algo_name"]
    mission = config["mission"]
    signal_mode = config["signal_mode"]
    query_length = config["query_length"]
    reference_length = config["reference_length"]
    slide_length = config["slide_length"]
    k = config["k"]
    p = config["p"]
    phase = config["phase"]

    print(f"[{(time.time() - t0):6.2f}s] Running {algo_name} with config:")
    print(json.dumps(config, indent=2))

    channels = parse_channels_mode(config, "channel_41")
    config["channels"] = channels

    if config["experimental_mode"]:
        pred_folder = project_root() / algo_name / "experiments" / config["exp_id"]
    else:
        pred_folder = project_root() / algo_name / "predictions"

    out_name = build_run_name(config, channels)
    pred_path = pred_folder / out_name


    labels_folder = pred_path / "labels"
    scores_folder = pred_path / "scores"
    pred_path.mkdir(exist_ok=True, parents=True)
    labels_folder.mkdir(exist_ok=True)
    scores_folder.mkdir(exist_ok=True)

    config.update({"predictions_name": str(out_name)})

    start_time = time.time()
    predictions_dict = {}
    scores_thresholds_dict = {}

    max_warmup_time = 0.0
    max_eval_time = 0.0
    avg_warmup_time = 0.0
    avg_eval_time = 0.0
    num_channels = 0

    # per channel while the workers run: window starts as a compact index and the labels;
    # the (timestamp, label) lists of the metrics are built once the pool is done
    compact = {}
    max_workers = worker_count(len(channels), cap=30)
    print(f"[{(time.time() - t0):6.2f}s] Running in parallel with {max_workers}/{cpu_count()} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(eval_channel, ch, config): ch for ch in channels}
        with tqdm(total=len(futures), desc=f"[{(time.time() - t0):6.2f}s] Evaluating channels", unit="ch") as pbar:
            for future in as_completed(futures):
                channel, timestamps, labels, scores, thresholds, warmup_time, eval_time = future.result()
                del futures[future]   # the future holds the result until it is dropped

                max_warmup_time = max(max_warmup_time, warmup_time)
                max_eval_time = max(max_eval_time, eval_time)
                avg_warmup_time += warmup_time
                avg_eval_time += eval_time
                num_channels += 1

                # Save labels
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "label": labels
                })
                df.to_parquet(pred_path / "labels" / f"{channel}.parquet")

                # Save scores and thresholds
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "score": scores,
                    "threshold": thresholds
                })
                df.to_parquet(pred_path / "scores" / f"{channel}.parquet")

                compact[channel] = (pd.DatetimeIndex(timestamps), labels)

                # scores and thresholds are kept only for the plots
                if config["plot"]:
                    scores_thresholds: List[Tuple[pd.Timestamp, np.float32, np.float32]] = [
                        (t, s, th)
                        for t, s, th in zip(
                            timestamps,
                            scores,
                            thresholds
                        )
                    ]
                    scores_thresholds_dict[channel] = scores_thresholds
                pbar.update(1)

    for channel in channels:
        index, labels = compact.pop(channel)
        if config["plot"]:
            predictions_dict[channel] = [(t, l) for t, l in zip(index, labels)]
        else:   # the metrics read only the label changes
            predictions_dict[channel] = label_changes(index, labels)

    avg_warmup_time /= num_channels
    avg_eval_time /= num_channels

    print(f"[{(time.time() - t0):6.2f}s] Evaluation completed in {time.time() - start_time:6.2f}s")
    print(f"[{(time.time() - t0):6.2f}s]   - Avg warmup time per channel: {avg_warmup_time:6.2f}s (max: {max_warmup_time:6.2f}s)")
    print(f"[{(time.time() - t0):6.2f}s]   - Avg evaluation time per channel: {avg_eval_time:6.2f}s (max: {max_eval_time:6.2f}s)")

    config["algorithm_runtime"] = str(time.time() - start_time)
    config["warmup_runtime_avg"] = str(avg_warmup_time)
    config["warmup_runtime_max"] = str(max_warmup_time)
    config["evaluation_runtime_avg"] = str(avg_eval_time)
    config["evaluation_runtime_max"] = str(max_eval_time)

    save_json(config, pred_path / "config")

    config = metrics_light(pred_path=pred_path,
                           config=config,
                           predictions_dict=predictions_dict,
                           ew_only=config["experimental_mode"],
                           t0=t0)

    save_json(config, pred_path / "config")

    if config["plot"]:
        from tsadbench.viz.plots import plot_results  # lazy import: avoid Tk/display deps on headless runs
        plot_results(config, predictions_dict, scores_thresholds_dict, t0)

    if config["experimental_mode"]:
        print(f"[{(time.time() - t0):6.2f}s] Experimental mode enabled. Returning metrics.")
        return config

    return None


