import json
import pandas as pd
from pathlib import Path

# Root of the benchmark package (smed_benchmark/)
ROOT = Path(__file__).resolve().parent.parent


def load_json(in_file: str | Path) -> dict:
    in_file = Path(in_file).with_suffix(".json")
    with open(in_file, "r") as f:
        return json.load(f)


def parse_channels_mode(config: dict) -> list[str]:
    channels_mode = config["channels_mode"]
    mission = config["mission"]
    metadata = load_json(ROOT / "data" / "metadata")

    if channels_mode == "full":
        return metadata[mission]["full"]
    elif channels_mode == "light":
        return metadata[mission]["light"]
    else:
        raise ValueError(f"Invalid channels mode: {channels_mode!r}")


def load_gt_events(
    label_path: str | Path,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(label_path)
    df["StartTime"] = pd.to_datetime(df["StartTime"]).dt.tz_convert(None)
    df["EndTime"] = pd.to_datetime(df["EndTime"]).dt.tz_convert(None)
    if channels is not None:
        df = df[df["Channel"].isin(channels)]
    return df
