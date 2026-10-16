"""Ingest ESA-ADB published Telemanom-ESA predictions into the standard run layout.

Usage:
    python scripts/ingest_telemanom.py <timeeval_run_dir> <output_run_dir> [mission]
    python scripts/ingest_telemanom.py Telemanom-ESA-full-sets.zip <output_run_dir>

A run folder must contain anomaly_scores.ts and hyper_params.json. From the full-set zip
published with ESA-ADB (results/README.md of https://github.com/kplabs-pl/ESA-ADB) the
Mission 1 run reported in the paper is read without unpacking. See tsadbench.io.telemanom
for the conventions.
"""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsadbench.io.telemanom import FULL_SET_RUN, ingest


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    mission = sys.argv[3] if len(sys.argv) > 3 else "ESA-Mission1"
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z, z.open(FULL_SET_RUN + "anomaly_scores.ts") as scores:
            hp = json.loads(z.read(FULL_SET_RUN + "hyper_params.json"))
            ingest(scores, hp, out, mission, source=f"{src.name}:{FULL_SET_RUN}")
    else:
        hp = json.loads((src / "hyper_params.json").read_text())
        with open(src / "anomaly_scores.ts", "rb") as scores:
            ingest(scores, hp, out, mission, source=str(src))


if __name__ == "__main__":
    main()
