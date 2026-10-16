"""Reproduction of the paper numbers, from the dataset download to a PASS/DIFF report.

Entry point: ``scripts/reproduce.py`` (also the Docker image in ``reproduce/``). The paper
values and where each is recomputed are listed in ``reproduce/paper_values.csv``.

    download   resumable downloads, remote zip members (Zenodo, ESA-ADB SharePoint)
    data       ESA-ADB Mission 1 download, conversion to parquet and content checksums
    report     resolution of each paper value against the runs on disk, report writers

Every run is written where the rest of the repository expects it (``smed/predictions/``,
``runs/``), so the usual scripts can inspect or re-score it.
"""
from pathlib import Path

from tsadbench.io.misc import project_root

MISSION = "ESA-Mission1"
ESA_ADB_RECORD = 12528696          # ESA Anomalies Dataset, Zenodo (CC BY 3.0 IGO)
# Telemanom-ESA-Pruned predictions published by ESA-ADB (results/README.md of
# https://github.com/kplabs-pl/ESA-ADB): full sets
TELEMANOM_FULL_SETS = ("https://kplabs-my.sharepoint.com/:u:/g/personal/kkotowski_kplabs_pl/"
                       "IQBttDSD1aKVSZYQXcP5jLl1AX1uPC_U_noi-PPyi9kn0jk?e=BkUhNl")
# Q-GPR predictions of the paper (its code is not released), built by
# tools/release/build_qgpr_archive.py; path relative to the repository root
QGPR_ARCHIVE = "results/qgpr_predictions.zip"
QGPR_MD5 = "4ab389d34c784f38f93a5565b1698a55"

SMED_PHASES = (1, 2, 3, 4, 5)
ZNORM_PS = (0.1, 1.0, 3.0, 5.0, 10.0, 25.0)   # percent, as in run names (paper: fraction)
QGPR_CONFIGS = ("qgpr_only", "max_only", "qgpr_max", "qgpr_max_cusum")
QGPR_SEEDS = (0, 1, 2, 3, 5, 7, 13, 21, 42, 55, 77, 100, 123, 777, 2026)


def _name(algo: str, phase: int, channels: str, p: float = 0.1) -> str:
    return f"{algo}_{MISSION}_phase_{phase}_raw_{channels}_q100_r750_s5_k1.5_p{p}"


def runs() -> dict[str, Path]:
    """Run folders behind the paper values, by key (relative to the repository root)."""
    r = {}
    for ph in SMED_PHASES:
        r[f"smed_light_{ph}"] = Path("smed/predictions") / _name("smed", ph, "light")
        r[f"ool_light_{ph}"] = Path("runs/oolimit") / _name("oolimit", ph, "light")
    r["smed_full_5"] = Path("smed/predictions") / _name("smed", 5, "full")
    r["ool_full_5"] = Path("runs/oolimit_full") / _name("oolimit", 5, "full")
    for p in ZNORM_PS:
        r[f"znorm_light_{p}"] = Path("runs/damp_znorm") / _name("damp_znorm", 5, "light", p)
        r[f"znorm_full_{p}"] = Path("runs/damp_znorm_full") / _name("damp_znorm", 5, "full", p)
    for c in QGPR_CONFIGS:
        for s in ("light", "full"):
            r[f"{c}_{s}"] = Path("runs/qgpr") / f"{c}_{s}"
    for ph in SMED_PHASES:
        r[f"qgpr_phase_{ph}"] = Path("runs/qgpr_phase") / f"phase{ph}_light"
    for seed in QGPR_SEEDS:
        r[f"qgpr_seed_{seed}"] = Path("runs/qgpr_seeds") / f"seed_{seed}_light"
    r["telemanom_full"] = Path("runs/telemanom") / f"telemanom_{MISSION}_full"
    return r


# runs scored per archetype (paper Tables 8 and 11), by column name of the recall table
RECALL_RUNS = {"smed": "smed_full_5", "qgpr": "qgpr_max_cusum_full", "telemanom": "telemanom_full",
               "ool": "ool_full_5", "znorm": "znorm_full_1.0"}


def output_dir() -> Path:
    return project_root() / "runs" / "reproduce"
