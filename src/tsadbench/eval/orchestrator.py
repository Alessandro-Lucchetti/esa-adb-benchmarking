import time
from pathlib import Path

import pandas as pd
from tsadbench.io.misc import load_gt_events, project_root, select_prediction, save_json, load_json, \
    load_predictions_dict, parse_channels_mode, load_metadata, load_predictions_event
import portion as P
import copy
from tsadbench.eval.metrics.ESA_ADB_metrics import ESAScores
from tsadbench.eval.metrics.ranking_metrics import ChannelAwareFScore
from tsadbench.eval.metrics.latency_metrics import ADTQC

def metrics_light(
        pred_path: Path | str = None,
        config: dict = None,
        predictions_event: P.Interval | None = None,
        predictions_dict: dict = None,
        modes: list[str] = None,
        ew_only: bool = False,
        t0: float | None = None
) -> dict:
    """
    Compute evaluation metrics for an evaluation run and persist them in ``config``.
    """
    start_time = time.time()
    if t0 is None:
        t0 = start_time

    if pred_path is None:
        if config is None or predictions_dict is None or predictions_event is None:
            raise ValueError("Must provide predictions path or data")
    else:
        pred_path = Path(pred_path)

    print(f"[{(time.time() - t0):6.2f}s] Starting metrics computation")

    if config is None:
        print(f"[{(time.time() - t0):6.2f}s] Loading config file")
        config = load_json(pred_path / "config")

    mission = config["mission"]
    channels = parse_channels_mode(config)
    data_path = project_root() / "data"
    metadata = load_metadata()

    label_path = data_path / mission / "labels.csv"
    channels_path = data_path / mission / "channels.csv"
    start, end = pd.to_datetime(metadata[mission]["test_period"])

    if pred_path is not None:
        if predictions_dict is None:
            predictions_dict = load_predictions_dict(pred_path, start, end, changes_only=True)
        if predictions_event is None:
            predictions_event = load_predictions_event(pred_path, start, end, t0=t0)
    else:
        predictions_dict = copy.copy(predictions_dict)

    if modes is None:
        modes = ["all_events", "anomalies_alone"]

    for mode in modes:
        print(f"[{(time.time() - t0):6.2f}s] Computing metrics for {mode.capitalize().replace('_', ' ')}")

        print(f"[{(time.time() - t0):6.2f}s] Loading ground truth labels")
        gt_df = load_gt_events(label_path, channels, start, end, mode)
        print(f"[{(time.time() - t0):6.2f}s] Loaded ground truth labels")

        print(f"[{(time.time() - t0):6.2f}s] Computing EW, Aff, and AP")
        esa_metric = ESAScores(betas=0.5, full_range=(start, end))
        esa_df = pd.concat([merge_with_portion(g) for _, g in gt_df.groupby("ID")], ignore_index=True)
        ew_ap_aff_score = esa_metric.score(esa_df, predictions_event, ew_only=ew_only)
        print(f"[{(time.time() - t0):6.2f}s] EW, Aff, and AP computed")
        if ew_only:
            print(f"[{(time.time() - t0):6.2f}s] Printing metrics")
            print_metrics_table(ew_ap_aff_score, mode=mode)
            print(f"[{(time.time() - t0):6.2f}s] Saving metrics to config")
            config = save_metrics_to_config(config, mode, ew_ap_aff_score)
            continue

        print(f"[{(time.time() - t0):6.2f}s] Computing ADTQC")
        adtqc_metric = ADTQC(full_range=(start, end))
        adtqc_score = adtqc_metric.score(gt_df, predictions_dict)
        print(f"[{(time.time() - t0):6.2f}s] ADTQC computed")

        if len(channels) > 1:
            channel_df = pd.read_csv(channels_path)
            grouped = channel_df.groupby("Subsystem")["Channel"].apply(list)
            subsystems_mapping = {
                f"s{i + 1}": [ch for ch in chs if ch in channels]
                for i, chs in enumerate(grouped.values)
                if any(ch in channels for ch in chs)
            }
            ranking_metric = ChannelAwareFScore()
            if len(subsystems_mapping) > 1:
                print(f"[{(time.time() - t0):6.2f}s] Computing Channel and Sub-system aware scores")
                ch_sub_score = ranking_metric.score(gt_df, predictions_dict, subsystems_mapping)
                print(f"[{(time.time() - t0):6.2f}s] Channel and Sub-system aware scores computed")

            else:
                print(f"[{(time.time() - t0):6.2f}s] Computing Channel aware scores")
                ch_sub_score = ranking_metric.score(gt_df, predictions_dict)
                print(f"[{(time.time() - t0):6.2f}s] Channel aware scores computed")

            print(f"[{(time.time() - t0):6.2f}s] Printing metrics")
            print_metrics_table(ew_ap_aff_score, adtqc_score, ch_sub_score, mode=mode)
            print(f"[{(time.time() - t0):6.2f}s] Saving metrics to config")
            config = save_metrics_to_config(config, mode, ew_ap_aff_score, adtqc_score, ch_sub_score)
        else:
            print(f"[{(time.time() - t0):6.2f}s] Printing metrics")
            print_metrics_table(ew_ap_aff_score, adtqc_score, mode=mode)
            print(f"[{(time.time() - t0):6.2f}s] Saving metrics to config")
            config = save_metrics_to_config(config, mode, ew_ap_aff_score, adtqc_score)

    if ew_only:
        print(f"[{(time.time() - t0):6.2f}s] EW metrics computed in {time.time() - start_time:.2f}s\n")
        config["ew_metrics_runtime"] = time.time() - start_time
    else:
        print(f"[{(time.time() - t0):6.2f}s] Metrics computed in {time.time() - start_time:.2f}s\n")
        config["metrics_runtime"] = time.time() - start_time
    return config

def save_metrics_to_config(
    config: dict,
    mode: str,
    ew_ap_aff_score: dict,
    adtqc_score: dict = None,
    ch_sub_score: dict | None = None,
    ) -> dict:
    """
    Insert a block of pre-computed metric values into ``config`` under
    ``"metrics_<mode>"`` and return the mutated dictionary.
    """

    metrics = {
        "event_wise": {
            "precision": round3(ew_ap_aff_score["EW_precision"]),
            "recall": round3(ew_ap_aff_score["EW_recall"]),
            "f0.5": round3(ew_ap_aff_score["EW_F_0.50"]),
        }
    }

    if ch_sub_score is not None:
        if "subsystem_precision" in ch_sub_score:
            metrics["subsystem_aware"] = {
                "precision": round3(ch_sub_score["subsystem_precision"]),
                "recall": round3(ch_sub_score["subsystem_recall"]),
                "f0.5": round3(ch_sub_score["subsystem_F0.50"]),
            }
        metrics["channel_aware"] = {
            "precision": round3(ch_sub_score["channel_precision"]),
            "recall": round3(ch_sub_score["channel_recall"]),
            "f0.5": round3(ch_sub_score["channel_F0.50"]),
        }

    if "alarming_precision" in ew_ap_aff_score:
        metrics["alarming_precision"] = round3(ew_ap_aff_score["alarming_precision"])

    if adtqc_score is not None:
        metrics["adtqc"] = {
            "after_rate": round3(adtqc_score["AfterRate"]),
            "total_score": round3(adtqc_score["Total"]),
        }
    if "AFF_precision" in ew_ap_aff_score:
        metrics["affiliation_based"] = {
            "precision": round3(ew_ap_aff_score["AFF_precision"]),
            "recall": round3(ew_ap_aff_score["AFF_recall"]),
            "f0.5": round3(ew_ap_aff_score["AFF_F_0.50"]),
        }


    config[f"metrics_{mode}"] = metrics
    return config

def print_metrics_table(
    ew_ap_aff_score: dict[str, float],
    adtqc_score: dict[str, float] = None,
    ch_sub_score: dict[str, float] | None = None,
    mode: str | None = None,
    ) -> None:

    ew_precision = ew_ap_aff_score["EW_precision"]
    ew_recall = ew_ap_aff_score["EW_recall"]
    ew_f = ew_ap_aff_score["EW_F_0.50"]

    subsystem_precision: float | None = None
    subsystem_recall: float | None = None
    subsystem_f: float | None = None

    channel_precision: float | None = None
    channel_recall: float | None = None
    channel_f: float | None = None

    if ch_sub_score is not None:
        if "subsystem_precision" in ch_sub_score.keys():
            subsystem_precision = ch_sub_score["subsystem_precision"]
            subsystem_recall = ch_sub_score["subsystem_recall"]
            subsystem_f = ch_sub_score["subsystem_F0.50"]

        channel_precision = ch_sub_score["channel_precision"]
        channel_recall = ch_sub_score["channel_recall"]
        channel_f = ch_sub_score["channel_F0.50"]

    alarming_precision = ew_ap_aff_score["alarming_precision"]

    if "AFF_precision" in ew_ap_aff_score.keys():
        aff_precision = ew_ap_aff_score["AFF_precision"]
        aff_recall = ew_ap_aff_score["AFF_recall"]
        aff_f = ew_ap_aff_score["AFF_F_0.50"]
    else:
        aff_precision = None
        aff_recall = None
        aff_f = None

    print("=" * 40)
    print(mode.upper().replace("_", " "))
    print("=" * 40)

    print(f"                    Precision    {ew_precision:.3f}")
    print(f"Event-wise          Recall       {ew_recall:.3f}")
    print(f"                    F0.5         {ew_f:.3f}")

    print("=" * 40)


    if ch_sub_score is not None:
        if "subsystem_precision" in ch_sub_score.keys():
            print(f"                    Precision    {subsystem_precision:.3f}")
            print(f"Subsystem-aware     Recall       {subsystem_recall:.3f}")
            print(f"                    F0.5         {subsystem_f:.3f}")

            print("=" * 40)

        print(f"                    Precision    {channel_precision:.3f}")
        print(f"Channel-aware       Recall       {channel_recall:.3f}")
        print(f"                    F0.5         {channel_f:.3f}")

        print("=" * 40)

    print(f"Alarming precision               {alarming_precision:.3f}")

    print("=" * 40)

    if adtqc_score is not None:
        after_rate = adtqc_score["AfterRate"]
        score = adtqc_score["Total"]
        print(f"ADTQC               After ratio  {after_rate:.3f}")
        print(f"                    Score        {score:.3f}")

        print("=" * 40)

    if aff_precision is not None:
        print(f"                    Precision    {aff_precision:.3f}")
        print(f"Affiliation-based   Recall       {aff_recall:.3f}")
        print(f"                    F0.5         {aff_f:.3f}")

        print("=" * 40)

def merge_with_portion(group: pd.DataFrame) -> pd.DataFrame:
    interval = P.empty()
    for _, row in group.iterrows():
        start = pd.to_datetime(row["StartTime"])
        end = pd.to_datetime(row["EndTime"])
        interval |= P.closed(start, end)

    merged_rows = []
    for i in interval:
        merged_rows.append([
            group["ID"].iloc[0],
            pd.to_datetime(i.lower),
            pd.to_datetime(i.upper)
        ])
    return pd.DataFrame(merged_rows, columns=["ID", "StartTime", "EndTime"])

def round3(x):
    return format3(round(x, 3)) if isinstance(x, float) else format3(x)

def format3(x):
    return f"{x:.3f}" if isinstance(x, float) else x

def main() -> None:
    pred_path = select_prediction()
    config = metrics_light(pred_path, ew_only=False, modes=["all_events"])
    save_json(config, pred_path / "config")

if __name__ == "__main__":
    main()
