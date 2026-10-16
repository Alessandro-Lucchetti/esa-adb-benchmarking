import sys
import time
from pathlib import Path

import pandas as pd
import portion as P

from tsadbench.eval.orchestrator import merge_with_portion
from tsadbench.io.misc import project_root, select_prediction, save_json, filter_time, load_json, load_metadata, load_predictions_event


def per_class_recall(pred_path, t0: float = None) -> dict:
    start_time = time.time()
    config = load_json(pred_path / "config.json")

    if not t0:
        t0 = start_time

    config["per_class"] = {}
    mission = config["mission"]
    data_path = project_root() / "data"
    metadata = load_metadata()
    label_path = data_path / mission / "labels.csv"
    anomaly_path = data_path / mission / "anomaly_types.csv"
    start, end = pd.to_datetime(metadata[mission]["test_period"])

    channels = metadata[mission]["full"]
    df = pd.read_csv(label_path)
    df["StartTime"] = pd.to_datetime(df["StartTime"]).dt.tz_convert(None)
    df["EndTime"] = pd.to_datetime(df["EndTime"]).dt.tz_convert(None)
    df = df[df["Channel"].isin(channels)]
    if start is not None and end is not None:
        df = filter_time(df, start, end)
    print(f"[{(time.time() - t0):6.2f}s] Loaded ground truth labels")
    class_df = pd.read_csv(anomaly_path)
    classes = class_df["Class"].unique().tolist()

    total_id = 0
    total_tp = 0

    events_pred = load_predictions_event(pred_path, start, end)

    for cls in classes:
        filtered_ids = class_df[class_df["Class"] == cls]["ID"].tolist()
        valid_ids = [f_id for f_id in filtered_ids if f_id in df["ID"].values]
        gt_df = df[df["ID"].isin(valid_ids)]
        if gt_df.empty:
            print(f"[{(time.time() - t0):6.2f}s] No events for class {cls}, skipping")
            continue
        else:
            print(f"[{(time.time() - t0):6.2f}s] Computing recall for class {cls}")
            total_id += len(valid_ids)

        ew_df = pd.concat(
            [merge_with_portion(g) for _, g in gt_df.groupby("ID")],
            ignore_index=True,
        )

        true_positives = 0
        for aid in valid_ids:
            gt = ew_df[ew_df["ID"] == aid]

            gt_intervals = []
            for _, row in gt[["StartTime", "EndTime"]].iterrows():
                gt_intervals.append(P.closed(*row))
            gt_intervals = P.Interval(*gt_intervals)

            at_least_one_detected = False
            for event in events_pred:
                if event.upper < gt_intervals.lower or event.lower > gt_intervals.upper:
                    continue
                intersections = [not (event & g).empty for g in gt_intervals]
                if not any(intersections):
                    continue
                if not at_least_one_detected:
                    true_positives += 1
                    total_tp += 1
                    at_least_one_detected = True


        config["per_class"][cls] = {
            "recall": true_positives / len(valid_ids),
            "true_positives": true_positives,
            "divider": len(valid_ids)
        }

        print(f"[{(time.time() - t0):6.2f}s] Class {cls} recall: {(true_positives / len(valid_ids)):.4f} ({true_positives}/{len(valid_ids)})")

    print(f"[{(time.time() - t0):6.2f}s] Per class computed in {time.time() - start_time:.2f}s\n")
    print(f"Overall recall: {(total_tp / total_id):.4f} ({total_tp}/{total_id})")
    return config

def main() -> None:
    """Usage: python -m tsadbench.analysis.per_class [run_dir]  (folder picker without an argument)"""
    pred_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else select_prediction()
    config = per_class_recall(pred_path)
    save_json(config, pred_path / "config")

if __name__ == "__main__":
    main()