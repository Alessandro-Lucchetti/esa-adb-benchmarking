"""Quantitative characterization of the ESA-ADB Mission 1 telemetry and a label-free
taxonomy of its annotated windows.

    sampling    per-channel sampling statistics of the raw streams, eight sampling families
    windows     labelled windows (one per labels.csv interval on a target channel) and context
    features    morphological features of each window
    classify    common signature (PCA) and 113 density + 4 rule-based classes
    archetypes  12 traits per class, Ward hierarchy, 8 rule-defined archetypes, tables
    recall      per-archetype recall of detector runs

Command line: ``scripts/taxonomy.py``; reference tables in ``data/ESA-Mission1/taxonomy/``.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsadbench.io.misc import project_root

TABLES = ("sampling_stats", "sampling_families", "window_features", "units", "windows",
          "archetypes", "dendrogram")
TIME_COLUMNS = ("t0", "t1")


def reference_dir(mission: str = "ESA-Mission1") -> Path:
    return project_root() / "data" / mission / "taxonomy"


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in TIME_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def write_taxonomy(tax: dict, signature_info: dict, out: Path) -> None:
    """Write the tables returned by ``archetypes.build_taxonomy`` to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    tax["units"].to_csv(out / "units.csv", index=False)
    tax["windows"].to_csv(out / "windows.csv", index=False)
    tax["archetypes"].to_csv(out / "archetypes.csv", index=False)
    L = tax["linkage"]
    pd.DataFrame({"left": L[:, 0].astype(int), "right": L[:, 1].astype(int), "height": L[:, 2],
                  "size": L[:, 3].astype(int)}).to_csv(out / "dendrogram.csv", index=False)
    signature = {k: v for k, v in signature_info.items() if k != "flags"}
    signature["n_flags"] = len(signature_info["flags"])
    meta = dict(signature=signature, **tax["meta"])
    (out / "taxonomy.json").write_text(json.dumps(meta, indent=2) + "\n", newline="\n")


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-9) -> list[str]:
    """Differences between two tables: shape, columns, then column by column (floats
    within ``rtol``, NaN equal to NaN). Empty when they match."""
    if list(a.columns) != list(b.columns):
        return [f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"]
    if len(a) != len(b):
        return [f"{len(a)} rows against {len(b)}"]
    diffs = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
            ok = np.isclose(x.astype(float), y.astype(float), rtol=rtol, atol=0, equal_nan=True)
        else:
            ok = (x.astype(str).values == y.astype(str).values)
        if not ok.all():
            diffs.append(f"{c}: {int((~ok).sum())} rows differ")
    return diffs
