"""ESA-ADB Mission 1 for the reproduction: download, conversion, content checksums.

The Zenodo archive holds the metadata CSVs and one zip per channel (a pickled DataFrame);
the target channels are converted to ``data/ESA-Mission1/raw/channel_NN.parquet`` by
``setup/convert_zips_to_parquet.py``. ``reproduce/data_checksums.json`` stores a SHA-256 of
every converted channel (timestamps as int64 nanoseconds, then values as float32) and of
the CSVs, so the result is checked independently of the parquet encoding.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tsadbench.io.misc import load_metadata, project_root
from tsadbench.reproduce import ESA_ADB_RECORD, MISSION
from tsadbench.reproduce.download import download, extract_members, zenodo_file

CSVS = ("labels.csv", "channels.csv", "anomaly_types.csv")


def checksums_path() -> Path:
    return project_root() / "reproduce" / "data_checksums.json"


def channel_digest(path: Path) -> str:
    df = pd.read_parquet(path)
    col = [c for c in df.columns if c != "timestamp"][0]
    h = hashlib.sha256(df["timestamp"].to_numpy("datetime64[ns]").view(np.int64).tobytes())
    h.update(df[col].to_numpy(np.float32).tobytes())
    return h.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(mission: str = MISSION) -> dict:
    d = project_root() / "data" / mission
    out = {name: file_digest(d / name) for name in CSVS}
    for ch in load_metadata()[mission]["full"]:
        out[ch] = channel_digest(d / "raw" / f"{ch}.parquet")
    return out


def verify(mission: str = MISSION) -> list[str]:
    """Names whose content differs from ``reproduce/data_checksums.json`` (empty when all match)."""
    expected = json.loads(checksums_path().read_text())[mission]
    d = project_root() / "data" / mission
    bad = []
    for name, digest in expected.items():
        path = d / name if name in CSVS else d / "raw" / f"{name}.parquet"
        got = (file_digest if name in CSVS else channel_digest)(path) if path.exists() else None
        if got != digest:
            bad.append(name)
    return bad


def prepare(zip_path: Path | None = None, mission: str = MISSION) -> None:
    """Download (unless ``zip_path`` is given), convert the target channels, verify."""
    root = project_root()
    data_dir = root / "data" / mission
    if zip_path is None:
        url, md5, size = zenodo_file(ESA_ADB_RECORD, f"{mission}.zip")
        print(f"downloading {mission}.zip ({size / 2**30:.1f} GB) from Zenodo record {ESA_ADB_RECORD}")
        zip_path = download(url, root / "data" / "downloads" / f"{mission}.zip", md5=md5, size=size)

    channels = load_metadata()[mission]["full"]
    tmp = root / "data" / "downloads" / f"{mission}-channels"
    with zipfile.ZipFile(zip_path) as z:
        members = {f"{mission}/{name}": data_dir / name for name in CSVS}
        members.update({f"{mission}/channels/{ch}.zip": tmp / f"{ch}.zip" for ch in channels})
        print(f"extracting {len(CSVS)} tables and {len(channels)} channel archives")
        extract_members(z, members)
    subprocess.run([sys.executable, str(root / "setup" / "convert_zips_to_parquet.py"), "--mission", mission,
                    "--zip-dir", str(tmp), "--channels", "full"], check=True)
    shutil.rmtree(tmp)
    bad = verify(mission)
    if bad:
        raise RuntimeError(f"converted data differ from reproduce/data_checksums.json: {bad}")
    print(f"{mission}: {len(channels)} channels and {len(CSVS)} tables match the reference checksums")
