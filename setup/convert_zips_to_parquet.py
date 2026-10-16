"""Convert ESA-ADB raw channel zips (pickled DataFrames) to per-channel parquet, float32.

Each ESA zip holds a pickled pandas DataFrame with a datetime index and one float32 column
(channel_NN), at the raw irregular sampling. This writes data/<mission>/raw/channel_NN.parquet
with columns [timestamp, channel_NN], preserving float32 (Windows-converted parquets upcast to
float64 for no benefit, since SMED casts back to float32).

Usage:
    python setup/convert_zips_to_parquet.py \
        --mission ESA-Mission2 \
        --zip-dir ~/projects/telemetry-xai/esa-adb/data/ESA-Mission2/channels \
        --channels full        # full | light | all
"""
import argparse
import json
import pickle
import warnings
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tsadbench.io.misc import project_root


def convert_channel(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pickle.load(f)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})  # datetime index -> timestamp
    col = [c for c in df.columns if c != "timestamp"][0]
    df[col] = df[col].astype("float32")
    return df


def selected_channels(mission: str, mode: str, zip_dir: Path) -> list[str]:
    if mode == "all":
        return sorted(p.stem for p in zip_dir.glob("channel_*.zip"))
    meta = json.loads((project_root() / "data" / "metadata.json").read_text())
    return meta[mission][mode]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--zip-dir", required=True, type=Path)
    ap.add_argument("--channels", default="full", choices=["full", "light", "all"])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    zip_dir = args.zip_dir.expanduser()
    out_dir = args.out_dir or (project_root() / "data" / args.mission / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = selected_channels(args.mission, args.channels, zip_dir)

    print(f"{args.mission}: converting {len(channels)} raw channel zips -> {out_dir}")
    for ch in tqdm(channels, unit="ch"):
        df = convert_channel(zip_dir / f"{ch}.zip")
        df.to_parquet(out_dir / f"{ch}.parquet", index=False, compression="zstd")

    sample = pd.read_parquet(out_dir / f"{channels[-1]}.parquet")
    col = [c for c in sample.columns if c != "timestamp"][0]
    assert sample[col].dtype == "float32", f"expected float32, got {sample[col].dtype}"
    print(f"done: {len(channels)} channels, sample {channels[-1]} dtype={sample[col].dtype} rows={len(sample)}")


if __name__ == "__main__":
    main()
